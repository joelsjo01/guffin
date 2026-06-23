#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

constexpr float gemma2_rope_frequency_base = 10000.0F;

[[nodiscard]] std::uint32_t checked_prompt_token_count(std::span<const std::uint32_t> prompt_token_ids) {
    if (prompt_token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Gemma2 Metal runtime shell prompt token count exceeds uint32");
    }

    return static_cast<std::uint32_t>(prompt_token_ids.size());
}

void require_prefill_hidden_states_match_schedule(
    const Gemma2MetalSchedule& prefill,
    const tensor::BackendBuffer& hidden_states
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal layer input primitive requires a prefill schedule");
    }
    if (hidden_states.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill hidden states must use the prefill schedule backend");
    }
    if (hidden_states.descriptor() != prefill.scratch().normalized_hidden) {
        throw std::invalid_argument("Gemma2 Metal prefill hidden states must match scheduled normalized hidden shape");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal layer input primitive requires at least one layer");
    }
}

void require_prefill_layer_input_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalLayerInputActivations& layer_input
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal attention projection primitive requires a prefill schedule");
    }
    if (layer_input.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal attention projection input phase must match schedule");
    }
    if (layer_input.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument("Gemma2 Metal attention projection input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal attention projection primitive requires at least one layer");
    }
    if (layer_input.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal attention projection input layer must match schedule");
    }
    if (layer_input.normalized_attention_input.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal attention projection input must use the schedule backend");
    }
    if (layer_input.normalized_attention_input.descriptor() != prefill.scratch().normalized_hidden) {
        throw std::invalid_argument("Gemma2 Metal attention projection input must match scheduled normalized hidden shape");
    }
}

void require_attention_projection_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected,
    const char* name
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(std::string("Gemma2 Metal attention projection ") + name
            + " output must match scheduled scratch shape");
    }
}

void require_prefill_attention_projection_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionProjectionActivations& projection
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal attention position primitive requires a prefill schedule");
    }
    if (projection.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal attention position input phase must match schedule");
    }
    if (projection.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument("Gemma2 Metal attention position input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal attention position primitive requires at least one layer");
    }
    if (projection.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal attention position input layer must match schedule");
    }
    if (projection.query.backend() != prefill.backend() || projection.key.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal attention position inputs must use the schedule backend");
    }
    if (projection.query.descriptor() != prefill.scratch().query) {
        throw std::invalid_argument("Gemma2 Metal attention position query must match scheduled query shape");
    }
    if (projection.key.descriptor() != prefill.scratch().key) {
        throw std::invalid_argument("Gemma2 Metal attention position key must match scheduled key shape");
    }
}

[[nodiscard]] Gemma2MetalLayerInputActivations create_gemma2_metal_prefill_first_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalInputActivations& prefill_input
) {
    require_prefill_hidden_states_match_schedule(prefill, prefill_input.hidden_states);
    const auto& first_layer = prefill.layers().front();
    auto normalized_attention_input = tensor::ops::rms_norm_f32_to_backend(
        prefill_input.hidden_states,
        plan.weights().blocks.at(first_layer.layer_index).attention.attention_norm_weight,
        plan.config().rms_norm_epsilon
    );

    return Gemma2MetalLayerInputActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .normalized_attention_input = std::move(normalized_attention_input),
    };
}

[[nodiscard]] Gemma2MetalAttentionProjectionActivations create_gemma2_metal_prefill_first_layer_attention_projection(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalLayerInputActivations& layer_input
) {
    require_prefill_layer_input_matches_schedule(prefill, layer_input);
    const auto& first_layer = prefill.layers().front();
    const auto& attention = plan.weights().blocks.at(first_layer.layer_index).attention;

    auto query = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        layer_input.normalized_attention_input,
        attention.query_weight
    );
    auto key = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        layer_input.normalized_attention_input,
        attention.key_weight
    );
    auto value = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        layer_input.normalized_attention_input,
        attention.value_weight
    );

    require_attention_projection_output_shape(query, prefill.scratch().query, "query");
    require_attention_projection_output_shape(key, prefill.scratch().key, "key");
    require_attention_projection_output_shape(value, prefill.scratch().value, "value");

    return Gemma2MetalAttentionProjectionActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .query = std::move(query),
        .key = std::move(key),
        .value = std::move(value),
    };
}

[[nodiscard]] Gemma2MetalAttentionPositionActivations create_gemma2_metal_prefill_first_layer_attention_position(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionProjectionActivations& projection
) {
    require_prefill_attention_projection_matches_schedule(prefill, projection);
    const auto& first_layer = prefill.layers().front();
    const auto head_dimension = plan.config().attention_key_length;
    const auto rope_dimension_count = head_dimension;

    auto query = tensor::ops::rope_f32_by_heads_with_position_start_to_backend(
        projection.query,
        prefill.cache_position(),
        plan.config().attention_head_count,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );
    auto key = tensor::ops::rope_f32_by_heads_with_position_start_to_backend(
        projection.key,
        prefill.cache_position(),
        plan.config().attention_head_count_kv,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );

    require_attention_projection_output_shape(query, prefill.scratch().query, "positioned query");
    require_attention_projection_output_shape(key, prefill.scratch().key, "positioned key");

    return Gemma2MetalAttentionPositionActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .position_start = prefill.cache_position(),
        .query_head_count = plan.config().attention_head_count,
        .key_head_count = plan.config().attention_head_count_kv,
        .head_dimension = head_dimension,
        .rope_dimension_count = rope_dimension_count,
        .rope_frequency_base = gemma2_rope_frequency_base,
        .query = std::move(query),
        .key = std::move(key),
    };
}

}  // namespace

Gemma2MetalGenerationRuntimeShell create_gemma2_metal_generation_runtime_shell(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids
) {
    auto plan = create_gemma2_metal_model_plan(model);
    const auto prompt_token_count = checked_prompt_token_count(prompt_token_ids);
    auto prefill = create_gemma2_metal_prefill_schedule(plan, prompt_token_count);
    auto first_decode = create_gemma2_metal_decode_schedule(plan, prompt_token_count);
    const auto embedding_scale = gemma2_input_embedding_scale(plan.config().embedding_length);
    auto prefill_input = Gemma2MetalInputActivations {
        .token_ids = std::vector<std::uint32_t>(prompt_token_ids.begin(), prompt_token_ids.end()),
        .hidden_states = tensor::ops::embed_f16_gguf_weight_token_ids_to_backend(
            prompt_token_ids,
            plan.weights().token_embedding_weight,
            embedding_scale
        ),
        .embedding_scale = embedding_scale,
    };
    auto prefill_first_layer_input = create_gemma2_metal_prefill_first_layer_input(
        plan,
        prefill,
        prefill_input
    );
    auto prefill_first_layer_attention_projection =
        create_gemma2_metal_prefill_first_layer_attention_projection(
            plan,
            prefill,
            prefill_first_layer_input
        );
    auto prefill_first_layer_attention_position =
        create_gemma2_metal_prefill_first_layer_attention_position(
            plan,
            prefill,
            prefill_first_layer_attention_projection
        );

    return Gemma2MetalGenerationRuntimeShell {
        .plan = std::move(plan),
        .prefill = std::move(prefill),
        .first_decode = std::move(first_decode),
        .prefill_input = std::move(prefill_input),
        .prefill_first_layer_input = std::move(prefill_first_layer_input),
        .prefill_first_layer_attention_projection = std::move(prefill_first_layer_attention_projection),
        .prefill_first_layer_attention_position = std::move(prefill_first_layer_attention_position),
    };
}

}  // namespace local_gguf_engine::engine
