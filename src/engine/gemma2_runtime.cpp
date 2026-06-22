#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

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

    return Gemma2MetalGenerationRuntimeShell {
        .plan = std::move(plan),
        .prefill = std::move(prefill),
        .first_decode = std::move(first_decode),
        .prefill_input = std::move(prefill_input),
        .prefill_first_layer_input = std::move(prefill_first_layer_input),
    };
}

}  // namespace local_gguf_engine::engine
