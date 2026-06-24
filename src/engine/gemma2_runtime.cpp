#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"

#include <cmath>
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

void require_prefill_attention_position_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionPositionActivations& position
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal attention score primitive requires a prefill schedule");
    }
    if (position.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal attention score input phase must match schedule");
    }
    if (position.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument("Gemma2 Metal attention score input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal attention score primitive requires at least one layer");
    }
    if (position.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal attention score input layer must match schedule");
    }
    if (position.query.backend() != prefill.backend() || position.key.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal attention score inputs must use the schedule backend");
    }
    if (position.query.descriptor() != prefill.scratch().query) {
        throw std::invalid_argument("Gemma2 Metal attention score query must match scheduled query shape");
    }
    if (position.key.descriptor() != prefill.scratch().key) {
        throw std::invalid_argument("Gemma2 Metal attention score key must match scheduled key shape");
    }
}

void require_attention_score_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument("Gemma2 Metal attention score output must match scheduled scratch shape");
    }
}

void require_prefill_attention_scores_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionScoreActivations& scores
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal attention probability primitive requires a prefill schedule");
    }
    if (scores.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal attention probability input phase must match schedule");
    }
    if (scores.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument("Gemma2 Metal attention probability input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal attention probability primitive requires at least one layer");
    }
    if (scores.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal attention probability input layer must match schedule");
    }
    if (scores.scores.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal attention probability input must use the schedule backend");
    }
    if (scores.scores.descriptor() != prefill.scratch().attention_scores) {
        throw std::invalid_argument("Gemma2 Metal attention probability input must match scheduled score shape");
    }
}

void require_attention_probability_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument("Gemma2 Metal attention probability output must match scheduled scratch shape");
    }
}

void require_prefill_attention_probabilities_match_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionProjectionActivations& projection,
    const Gemma2MetalAttentionProbabilityActivations& probabilities
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal attention value primitive requires a prefill schedule");
    }
    if (probabilities.phase != prefill.phase() || projection.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal attention value input phase must match schedule");
    }
    if (
        probabilities.scratch_ownership != prefill.scratch_ownership()
        || projection.scratch_ownership != prefill.scratch_ownership()
    ) {
        throw std::invalid_argument("Gemma2 Metal attention value input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal attention value primitive requires at least one layer");
    }
    if (
        probabilities.layer_index != prefill.layers().front().layer_index
        || projection.layer_index != prefill.layers().front().layer_index
    ) {
        throw std::invalid_argument("Gemma2 Metal attention value input layer must match schedule");
    }
    if (probabilities.probabilities.backend() != prefill.backend() || projection.value.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal attention value inputs must use the schedule backend");
    }
    if (probabilities.probabilities.descriptor() != prefill.scratch().attention_probabilities) {
        throw std::invalid_argument("Gemma2 Metal attention value probabilities must match scheduled scratch shape");
    }
    if (projection.value.descriptor() != prefill.scratch().value) {
        throw std::invalid_argument("Gemma2 Metal attention value input must match scheduled value shape");
    }
}

void require_attention_value_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument("Gemma2 Metal attention value output must match scheduled scratch shape");
    }
}

void require_prefill_attention_values_match_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionValueActivations& values
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal attention output projection primitive requires a prefill schedule");
    }
    if (values.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal attention output projection input phase must match schedule");
    }
    if (values.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal attention output projection input scratch ownership must match schedule"
        );
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal attention output projection primitive requires at least one layer");
    }
    if (values.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal attention output projection input layer must match schedule");
    }
    if (values.attention_output.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal attention output projection input must use the schedule backend");
    }
    if (values.attention_output.descriptor() != prefill.scratch().attention_output) {
        throw std::invalid_argument(
            "Gemma2 Metal attention output projection input must match scheduled attention output shape"
        );
    }
}

void require_projected_attention_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            "Gemma2 Metal projected attention output must match scheduled scratch shape"
        );
    }
}

void require_prefill_attention_output_projection_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalInputActivations& prefill_input,
    const Gemma2MetalAttentionOutputProjectionActivations& output_projection
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal post-attention residual primitive requires a prefill schedule");
    }
    if (output_projection.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal post-attention residual input phase must match schedule");
    }
    if (output_projection.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal post-attention residual input scratch ownership must match schedule"
        );
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal post-attention residual primitive requires at least one layer");
    }
    if (output_projection.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal post-attention residual input layer must match schedule");
    }
    if (
        output_projection.projected_attention_output.backend() != prefill.backend()
        || prefill_input.hidden_states.backend() != prefill.backend()
    ) {
        throw std::invalid_argument("Gemma2 Metal post-attention residual inputs must use the schedule backend");
    }
    if (output_projection.projected_attention_output.descriptor() != prefill.scratch().projected_attention_output) {
        throw std::invalid_argument(
            "Gemma2 Metal post-attention residual input must match scheduled projected attention shape"
        );
    }
    if (prefill_input.hidden_states.descriptor() != prefill.scratch().normalized_hidden) {
        throw std::invalid_argument(
            "Gemma2 Metal post-attention residual input must match scheduled hidden state shape"
        );
    }
}

void require_post_attention_residual_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            "Gemma2 Metal post-attention residual output must match scheduled scratch shape"
        );
    }
}

void require_prefill_post_attention_residual_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPostAttentionResidualActivations& post_attention_residual
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up primitive requires a prefill schedule");
    }
    if (post_attention_residual.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up input phase must match schedule");
    }
    if (post_attention_residual.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up primitive requires at least one layer");
    }
    if (post_attention_residual.layer_index != prefill.layers().front().layer_index) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up input layer must match schedule");
    }
    if (post_attention_residual.post_attention_residual.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up input must use the schedule backend");
    }
    if (post_attention_residual.post_attention_residual.descriptor() != prefill.scratch().post_attention_residual) {
        throw std::invalid_argument("Gemma2 Metal FFN gate/up input must match scheduled residual shape");
    }
}

void require_feed_forward_gate_up_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected,
    const char* name
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(std::string("Gemma2 Metal FFN ") + name
            + " output must match scheduled scratch shape");
    }
}

void require_prefill_feed_forward_gate_up_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPostAttentionResidualActivations& post_attention_residual,
    const Gemma2MetalFeedForwardGateUpActivations& gate_up
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal FFN output primitive requires a prefill schedule");
    }
    if (post_attention_residual.phase != prefill.phase() || gate_up.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal FFN output input phase must match schedule");
    }
    if (
        post_attention_residual.scratch_ownership != prefill.scratch_ownership()
        || gate_up.scratch_ownership != prefill.scratch_ownership()
    ) {
        throw std::invalid_argument("Gemma2 Metal FFN output input scratch ownership must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal FFN output primitive requires at least one layer");
    }
    if (
        post_attention_residual.layer_index != prefill.layers().front().layer_index
        || gate_up.layer_index != prefill.layers().front().layer_index
    ) {
        throw std::invalid_argument("Gemma2 Metal FFN output input layer must match schedule");
    }
    if (
        post_attention_residual.post_attention_residual.backend() != prefill.backend()
        || gate_up.normalized_feed_forward_input.backend() != prefill.backend()
        || gate_up.gate.backend() != prefill.backend()
        || gate_up.up.backend() != prefill.backend()
    ) {
        throw std::invalid_argument("Gemma2 Metal FFN output inputs must use the schedule backend");
    }
    if (post_attention_residual.post_attention_residual.descriptor() != prefill.scratch().post_attention_residual) {
        throw std::invalid_argument("Gemma2 Metal FFN output residual input must match scheduled scratch shape");
    }
    if (gate_up.normalized_feed_forward_input.descriptor() != prefill.scratch().normalized_feed_forward_input) {
        throw std::invalid_argument("Gemma2 Metal FFN output normalized input must match scheduled scratch shape");
    }
    if (gate_up.gate.descriptor() != prefill.scratch().feed_forward_gate) {
        throw std::invalid_argument("Gemma2 Metal FFN output gate input must match scheduled scratch shape");
    }
    if (gate_up.up.descriptor() != prefill.scratch().feed_forward_up) {
        throw std::invalid_argument("Gemma2 Metal FFN output up input must match scheduled scratch shape");
    }
}

void require_feed_forward_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected,
    const char* name
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(std::string("Gemma2 Metal FFN ") + name
            + " output must match scheduled scratch shape");
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

[[nodiscard]] Gemma2MetalAttentionScoreActivations create_gemma2_metal_prefill_first_layer_attention_scores(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionPositionActivations& position
) {
    require_prefill_attention_position_matches_schedule(prefill, position);
    const auto& first_layer = prefill.layers().front();
    const auto key_length = plan.config().attention_key_length;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(key_length));
    const auto attention_token_count = first_layer.attention_token_count;
    auto scores = tensor::ops::causal_grouped_attention_scores_f32_to_backend(
        position.query,
        position.key,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        key_length,
        attention_token_count,
        scale,
        plan.config().attention_logit_softcap
    );

    require_attention_score_output_shape(scores, prefill.scratch().attention_scores);

    return Gemma2MetalAttentionScoreActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .query_position_start = prefill.cache_position(),
        .attention_token_count = attention_token_count,
        .query_head_count = plan.config().attention_head_count,
        .key_head_count = plan.config().attention_head_count_kv,
        .key_length = key_length,
        .scale = scale,
        .logit_softcap = plan.config().attention_logit_softcap,
        .scores = std::move(scores),
    };
}

[[nodiscard]] Gemma2MetalAttentionProbabilityActivations create_gemma2_metal_prefill_first_layer_attention_probabilities(
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionScoreActivations& scores
) {
    require_prefill_attention_scores_matches_schedule(prefill, scores);
    const auto& first_layer = prefill.layers().front();
    auto probabilities = tensor::ops::softmax_f32_to_backend(scores.scores);
    require_attention_probability_output_shape(probabilities, prefill.scratch().attention_probabilities);

    return Gemma2MetalAttentionProbabilityActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .attention_token_count = first_layer.attention_token_count,
        .query_head_count = scores.query_head_count,
        .probabilities = std::move(probabilities),
    };
}

[[nodiscard]] Gemma2MetalAttentionValueActivations create_gemma2_metal_prefill_first_layer_attention_values(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionProjectionActivations& projection,
    const Gemma2MetalAttentionProbabilityActivations& probabilities
) {
    require_prefill_attention_probabilities_match_schedule(prefill, projection, probabilities);
    const auto& first_layer = prefill.layers().front();
    const auto attention_token_count = first_layer.attention_token_count;
    const auto value_length = plan.config().attention_value_length;
    auto attention_output = tensor::ops::causal_grouped_attention_values_f32_to_backend(
        probabilities.probabilities,
        projection.value,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        value_length,
        attention_token_count
    );

    require_attention_value_output_shape(attention_output, prefill.scratch().attention_output);

    return Gemma2MetalAttentionValueActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .query_position_start = prefill.cache_position(),
        .attention_token_count = attention_token_count,
        .query_head_count = plan.config().attention_head_count,
        .value_head_count = plan.config().attention_head_count_kv,
        .value_length = value_length,
        .attention_output = std::move(attention_output),
    };
}

[[nodiscard]] Gemma2MetalAttentionOutputProjectionActivations
create_gemma2_metal_prefill_first_layer_attention_output_projection(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalAttentionValueActivations& values
) {
    require_prefill_attention_values_match_schedule(prefill, values);
    const auto& first_layer = prefill.layers().front();
    const auto& attention = plan.weights().blocks.at(first_layer.layer_index).attention;
    auto projected_attention_output = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        values.attention_output,
        attention.output_weight
    );

    require_projected_attention_output_shape(
        projected_attention_output,
        prefill.scratch().projected_attention_output
    );

    return Gemma2MetalAttentionOutputProjectionActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .attention_output_width = values.query_head_count * values.value_length,
        .hidden_width = plan.config().embedding_length,
        .projected_attention_output = std::move(projected_attention_output),
    };
}

[[nodiscard]] Gemma2MetalPostAttentionResidualActivations
create_gemma2_metal_prefill_first_layer_post_attention_residual(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalInputActivations& prefill_input,
    const Gemma2MetalAttentionOutputProjectionActivations& output_projection
) {
    require_prefill_attention_output_projection_matches_schedule(prefill, prefill_input, output_projection);
    const auto& first_layer = prefill.layers().front();
    const auto& attention = plan.weights().blocks.at(first_layer.layer_index).attention;
    auto normalized_attention_output = tensor::ops::rms_norm_f32_to_backend(
        output_projection.projected_attention_output,
        attention.post_attention_norm_weight,
        plan.config().rms_norm_epsilon
    );
    auto post_attention_residual = tensor::ops::add_f32_to_backend(
        prefill_input.hidden_states,
        normalized_attention_output
    );

    require_post_attention_residual_output_shape(
        post_attention_residual,
        prefill.scratch().post_attention_residual
    );

    return Gemma2MetalPostAttentionResidualActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .post_attention_residual = std::move(post_attention_residual),
    };
}

[[nodiscard]] Gemma2MetalFeedForwardGateUpActivations create_gemma2_metal_prefill_first_layer_feed_forward_gate_up(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPostAttentionResidualActivations& post_attention_residual
) {
    require_prefill_post_attention_residual_matches_schedule(prefill, post_attention_residual);
    const auto& first_layer = prefill.layers().front();
    const auto& feed_forward = plan.weights().blocks.at(first_layer.layer_index).feed_forward;
    auto normalized_feed_forward_input = tensor::ops::rms_norm_f32_to_backend(
        post_attention_residual.post_attention_residual,
        feed_forward.feed_forward_norm_weight,
        plan.config().rms_norm_epsilon
    );
    auto gate = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_feed_forward_input,
        feed_forward.gate_weight
    );
    auto up = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_feed_forward_input,
        feed_forward.up_weight
    );

    require_feed_forward_gate_up_output_shape(
        normalized_feed_forward_input,
        prefill.scratch().normalized_feed_forward_input,
        "normalized input"
    );
    require_feed_forward_gate_up_output_shape(gate, prefill.scratch().feed_forward_gate, "gate projection");
    require_feed_forward_gate_up_output_shape(up, prefill.scratch().feed_forward_up, "up projection");

    return Gemma2MetalFeedForwardGateUpActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .feed_forward_width = plan.config().feed_forward_length,
        .normalized_feed_forward_input = std::move(normalized_feed_forward_input),
        .gate = std::move(gate),
        .up = std::move(up),
    };
}

[[nodiscard]] Gemma2MetalFeedForwardOutputActivations create_gemma2_metal_prefill_first_layer_feed_forward_output(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPostAttentionResidualActivations& post_attention_residual,
    const Gemma2MetalFeedForwardGateUpActivations& gate_up
) {
    require_prefill_feed_forward_gate_up_matches_schedule(prefill, post_attention_residual, gate_up);
    const auto& first_layer = prefill.layers().front();
    const auto& feed_forward = plan.weights().blocks.at(first_layer.layer_index).feed_forward;
    auto feed_forward_gated = tensor::ops::silu_multiply_f32_to_backend(gate_up.gate, gate_up.up);
    auto projected_feed_forward_output = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        feed_forward_gated,
        feed_forward.down_weight
    );
    auto normalized_feed_forward_output = tensor::ops::rms_norm_f32_to_backend(
        projected_feed_forward_output,
        feed_forward.post_feed_forward_norm_weight,
        plan.config().rms_norm_epsilon
    );
    auto feed_forward_residual = tensor::ops::add_f32_to_backend(
        post_attention_residual.post_attention_residual,
        normalized_feed_forward_output
    );

    require_feed_forward_output_shape(feed_forward_gated, prefill.scratch().feed_forward_gated, "gated");
    require_feed_forward_output_shape(
        projected_feed_forward_output,
        prefill.scratch().projected_feed_forward_output,
        "down projection"
    );
    require_feed_forward_output_shape(
        feed_forward_residual,
        prefill.scratch().feed_forward_residual,
        "residual"
    );

    return Gemma2MetalFeedForwardOutputActivations {
        .layer_index = first_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .feed_forward_width = plan.config().feed_forward_length,
        .feed_forward_gated = std::move(feed_forward_gated),
        .projected_feed_forward_output = std::move(projected_feed_forward_output),
        .feed_forward_residual = std::move(feed_forward_residual),
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
    auto prefill_first_layer_attention_scores =
        create_gemma2_metal_prefill_first_layer_attention_scores(
            plan,
            prefill,
            prefill_first_layer_attention_position
        );
    auto prefill_first_layer_attention_probabilities =
        create_gemma2_metal_prefill_first_layer_attention_probabilities(
            prefill,
            prefill_first_layer_attention_scores
        );
    auto prefill_first_layer_attention_values =
        create_gemma2_metal_prefill_first_layer_attention_values(
            plan,
            prefill,
            prefill_first_layer_attention_projection,
            prefill_first_layer_attention_probabilities
        );
    auto prefill_first_layer_attention_output_projection =
        create_gemma2_metal_prefill_first_layer_attention_output_projection(
            plan,
            prefill,
            prefill_first_layer_attention_values
        );
    auto prefill_first_layer_post_attention_residual =
        create_gemma2_metal_prefill_first_layer_post_attention_residual(
            plan,
            prefill,
            prefill_input,
            prefill_first_layer_attention_output_projection
        );
    auto prefill_first_layer_feed_forward_gate_up =
        create_gemma2_metal_prefill_first_layer_feed_forward_gate_up(
            plan,
            prefill,
            prefill_first_layer_post_attention_residual
        );
    auto prefill_first_layer_feed_forward_output =
        create_gemma2_metal_prefill_first_layer_feed_forward_output(
            plan,
            prefill,
            prefill_first_layer_post_attention_residual,
            prefill_first_layer_feed_forward_gate_up
        );

    return Gemma2MetalGenerationRuntimeShell {
        .plan = std::move(plan),
        .prefill = std::move(prefill),
        .first_decode = std::move(first_decode),
        .prefill_input = std::move(prefill_input),
        .prefill_first_layer_input = std::move(prefill_first_layer_input),
        .prefill_first_layer_attention_projection = std::move(prefill_first_layer_attention_projection),
        .prefill_first_layer_attention_position = std::move(prefill_first_layer_attention_position),
        .prefill_first_layer_attention_scores = std::move(prefill_first_layer_attention_scores),
        .prefill_first_layer_attention_probabilities = std::move(prefill_first_layer_attention_probabilities),
        .prefill_first_layer_attention_values = std::move(prefill_first_layer_attention_values),
        .prefill_first_layer_attention_output_projection =
            std::move(prefill_first_layer_attention_output_projection),
        .prefill_first_layer_post_attention_residual =
            std::move(prefill_first_layer_post_attention_residual),
        .prefill_first_layer_feed_forward_gate_up =
            std::move(prefill_first_layer_feed_forward_gate_up),
        .prefill_first_layer_feed_forward_output =
            std::move(prefill_first_layer_feed_forward_output),
    };
}

}  // namespace local_gguf_engine::engine
