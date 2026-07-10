#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/dtype.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace local_gguf_engine::engine {

namespace {

constexpr float gemma2_rope_frequency_base = 10000.0F;

[[nodiscard]] std::uint32_t checked_prompt_token_count(std::span<const std::uint32_t> prompt_token_ids) {
    if (prompt_token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Gemma2 Metal runtime shell prompt token count exceeds uint32");
    }

    return static_cast<std::uint32_t>(prompt_token_ids.size());
}

void require_gemma2_metal_generation_inputs(
    const Gemma2ModelConfig& config,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens
) {
    if (max_new_tokens == 0U) {
        throw std::invalid_argument("Gemma2 Metal generation max_new_tokens must be positive");
    }
    if (prompt_token_ids.empty()) {
        throw std::invalid_argument("Gemma2 Metal generation prompt token ids must not be empty");
    }
    if (prompt_token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Gemma2 Metal generation prompt token count exceeds uint32");
    }
    if (max_new_tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Gemma2 Metal generation max_new_tokens exceeds uint32");
    }
    if (prompt_token_ids.size() > std::numeric_limits<std::size_t>::max() - max_new_tokens) {
        throw std::invalid_argument("Gemma2 Metal generation token count overflows");
    }
    if (prompt_token_ids.size() + max_new_tokens > static_cast<std::size_t>(config.context_length)) {
        throw std::invalid_argument("Gemma2 Metal generation token count exceeds context length");
    }
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

void require_decode_hidden_states_match_schedule(
    const Gemma2MetalSchedule& decode,
    const tensor::BackendBuffer& hidden_states
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode layer input primitive requires a decode schedule");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode layer input primitive requires one scheduled token");
    }
    if (hidden_states.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode hidden states must use the decode schedule backend");
    }
    if (hidden_states.descriptor() != decode.scratch().normalized_hidden) {
        throw std::invalid_argument("Gemma2 Metal decode hidden states must match scheduled normalized hidden shape");
    }
    if (decode.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal decode layer input primitive requires at least one layer");
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

void require_prefill_kv_cache_layer_shape(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalKVCacheLayer& cache_layer
) {
    const auto& config = plan.config();
    if (cache_layer.key.backend() != plan.backend() || cache_layer.value.backend() != plan.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write target must use the plan backend");
    }
    if (cache_layer.key.descriptor().dtype != tensor::DType::f32) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache key target must be F32");
    }
    if (cache_layer.value.descriptor().dtype != tensor::DType::f32) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache value target must be F32");
    }
    if (
        cache_layer.key.descriptor().shape.rank() != 3U
        || cache_layer.key.descriptor().shape.dimension(0) != config.context_length
        || cache_layer.key.descriptor().shape.dimension(1) != config.attention_head_count_kv
        || cache_layer.key.descriptor().shape.dimension(2) != config.attention_key_length
    ) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache key target shape mismatch");
    }
    if (
        cache_layer.value.descriptor().shape.rank() != 3U
        || cache_layer.value.descriptor().shape.dimension(0) != config.context_length
        || cache_layer.value.descriptor().shape.dimension(1) != config.attention_head_count_kv
        || cache_layer.value.descriptor().shape.dimension(2) != config.attention_value_length
    ) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache value target shape mismatch");
    }
}

void require_prefill_kv_cache_write_matches_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalAttentionPositionActivations& position,
    const Gemma2MetalAttentionProjectionActivations& projection
) {
    if (plan.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write plan backend must match schedule");
    }
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write requires a prefill schedule");
    }
    if (static_cast<std::size_t>(schedule_layer_offset) >= prefill.layers().size()) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write layer offset is not scheduled");
    }

    const auto& scheduled_layer = prefill.layers()[schedule_layer_offset];
    if (position.layer_index != scheduled_layer.layer_index || projection.layer_index != scheduled_layer.layer_index) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write layer must match schedule");
    }
    if (position.phase != prefill.phase() || projection.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write phase must match schedule");
    }
    if (
        position.scratch_ownership != prefill.scratch_ownership()
        || projection.scratch_ownership != prefill.scratch_ownership()
    ) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write scratch ownership must match schedule");
    }
    if (position.key.backend() != prefill.backend() || projection.value.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write inputs must use the schedule backend");
    }
    if (position.key.descriptor() != prefill.scratch().key) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write key must match scheduled key shape");
    }
    if (projection.value.descriptor() != prefill.scratch().value) {
        throw std::invalid_argument("Gemma2 Metal prefill KV cache write value must match scheduled value shape");
    }
    require_prefill_kv_cache_layer_shape(plan, plan.kv_cache().at(scheduled_layer.layer_index));
}

void require_decode_kv_cache_write_matches_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalAttentionPositionActivations& position,
    const Gemma2MetalAttentionProjectionActivations& projection
) {
    if (plan.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write plan backend must match schedule");
    }
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write requires a decode schedule");
    }
    if (decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write requires resident decode scratch");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write requires one scheduled token");
    }
    if (static_cast<std::size_t>(schedule_layer_offset) >= decode.layers().size()) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write layer offset is not scheduled");
    }

    const auto& scheduled_layer = decode.layers()[schedule_layer_offset];
    if (position.layer_index != scheduled_layer.layer_index || projection.layer_index != scheduled_layer.layer_index) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write layer must match schedule");
    }
    if (position.phase != decode.phase() || projection.phase != decode.phase()) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write phase must match schedule");
    }
    if (
        position.scratch_ownership != decode.scratch_ownership()
        || projection.scratch_ownership != decode.scratch_ownership()
    ) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write scratch ownership must match schedule");
    }
    if (position.position_start != decode.cache_position()) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write position must match schedule");
    }
    if (position.key.backend() != decode.backend() || projection.value.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write inputs must use the schedule backend");
    }
    if (position.key.descriptor() != decode.scratch().key) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write key must match scheduled key shape");
    }
    if (projection.value.descriptor() != decode.scratch().value) {
        throw std::invalid_argument("Gemma2 Metal decode KV cache write value must match scheduled value shape");
    }
    require_prefill_kv_cache_layer_shape(plan, plan.kv_cache().at(scheduled_layer.layer_index));
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

void require_layer_handoff_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            "Gemma2 Metal layer handoff output must match scheduled scratch shape"
        );
    }
}

void require_prefill_output_norm_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill output norm must match scheduled normalized hidden shape"
        );
    }
}

void require_decode_output_norm_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            "Gemma2 Metal decode output norm must match scheduled normalized hidden shape"
        );
    }
}

void require_prefill_logits_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected,
    const char* name
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            std::string("Gemma2 Metal prefill logits ") + name + " must match scheduled logits shape"
        );
    }
}

void require_decode_logits_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected,
    const char* name
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument(
            std::string("Gemma2 Metal decode logits ") + name + " must match scheduled logits shape"
        );
    }
}

void require_scheduled_prefill_block_inputs_match_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const tensor::BackendBuffer& normalized_attention_input,
    const tensor::BackendBuffer& residual_input
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal prefill block runner requires a prefill schedule");
    }
    if (plan.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill block runner plan backend must match schedule");
    }
    if (static_cast<std::size_t>(schedule_layer_offset) >= prefill.layers().size()) {
        throw std::invalid_argument("Gemma2 Metal prefill block runner layer offset is not scheduled");
    }
    if (
        normalized_attention_input.backend() != prefill.backend()
        || residual_input.backend() != prefill.backend()
    ) {
        throw std::invalid_argument("Gemma2 Metal prefill block runner inputs must use the schedule backend");
    }

    const auto& expected_normalized_input = schedule_layer_offset == 0U
        ? prefill.scratch().normalized_hidden
        : prefill.scratch().layer_handoff_normalized_attention_input;
    const auto& expected_residual_input = schedule_layer_offset == 0U
        ? prefill.scratch().normalized_hidden
        : prefill.scratch().feed_forward_residual;
    if (normalized_attention_input.descriptor() != expected_normalized_input) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill block normalized attention input must match scheduled input shape"
        );
    }
    if (residual_input.descriptor() != expected_residual_input) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill block residual input must match scheduled residual shape"
        );
    }
}

void require_scheduled_decode_block_inputs_match_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const tensor::BackendBuffer& normalized_attention_input,
    const tensor::BackendBuffer& residual_input
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode block runner requires a decode schedule");
    }
    if (decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode) {
        throw std::invalid_argument("Gemma2 Metal decode block runner requires resident decode scratch");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode block runner requires one scheduled token");
    }
    if (plan.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode block runner plan backend must match schedule");
    }
    if (static_cast<std::size_t>(schedule_layer_offset) >= decode.layers().size()) {
        throw std::invalid_argument("Gemma2 Metal decode block runner layer offset is not scheduled");
    }
    if (
        normalized_attention_input.backend() != decode.backend()
        || residual_input.backend() != decode.backend()
    ) {
        throw std::invalid_argument("Gemma2 Metal decode block runner inputs must use the schedule backend");
    }

    const auto& expected_normalized_input = schedule_layer_offset == 0U
        ? decode.scratch().normalized_hidden
        : decode.scratch().layer_handoff_normalized_attention_input;
    const auto& expected_residual_input = schedule_layer_offset == 0U
        ? decode.scratch().normalized_hidden
        : decode.scratch().feed_forward_residual;
    if (normalized_attention_input.descriptor() != expected_normalized_input) {
        throw std::invalid_argument(
            "Gemma2 Metal decode block normalized attention input must match scheduled input shape"
        );
    }
    if (residual_input.descriptor() != expected_residual_input) {
        throw std::invalid_argument("Gemma2 Metal decode block residual input must match scheduled residual shape");
    }
}

void require_scheduled_prefill_layer_input_residual_matches_schedule(
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input requires a prefill schedule");
    }
    if (schedule_layer_offset == 0U) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input requires a nonzero layer offset");
    }
    if (previous_feed_forward_output.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input residual phase must match schedule");
    }
    if (previous_feed_forward_output.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input residual scratch ownership must match schedule");
    }
    if (static_cast<std::size_t>(schedule_layer_offset) > prefill.layers().size()) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input offset exceeds scheduled layers");
    }
    if (previous_feed_forward_output.layer_index != prefill.layers()[schedule_layer_offset - 1U].layer_index) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input residual layer must precede target layer");
    }
    if (previous_feed_forward_output.feed_forward_residual.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input residual must use the schedule backend");
    }
    if (previous_feed_forward_output.feed_forward_residual.descriptor() != prefill.scratch().feed_forward_residual) {
        throw std::invalid_argument(
            "Gemma2 Metal reusable layer input residual must match scheduled FFN residual shape"
        );
    }
}

void require_scheduled_decode_layer_input_residual_matches_schedule(
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input requires a decode schedule");
    }
    if (decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input requires resident decode scratch");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input requires one scheduled token");
    }
    if (schedule_layer_offset == 0U) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input requires a nonzero layer offset");
    }
    if (previous_feed_forward_output.phase != decode.phase()) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input residual phase must match schedule");
    }
    if (previous_feed_forward_output.scratch_ownership != decode.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal reusable decode layer input residual scratch ownership must match schedule"
        );
    }
    if (static_cast<std::size_t>(schedule_layer_offset) > decode.layers().size()) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input offset exceeds scheduled layers");
    }
    if (previous_feed_forward_output.layer_index != decode.layers()[schedule_layer_offset - 1U].layer_index) {
        throw std::invalid_argument(
            "Gemma2 Metal reusable decode layer input residual layer must precede target layer"
        );
    }
    if (previous_feed_forward_output.feed_forward_residual.backend() != decode.backend()) {
        throw std::invalid_argument(
            "Gemma2 Metal reusable decode layer input residual must use the schedule backend"
        );
    }
    if (previous_feed_forward_output.feed_forward_residual.descriptor() != decode.scratch().feed_forward_residual) {
        throw std::invalid_argument(
            "Gemma2 Metal reusable decode layer input residual must match scheduled FFN residual shape"
        );
    }
}

void require_prefill_terminal_feed_forward_output_matches_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalFeedForwardOutputActivations& feed_forward_output
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal prefill output norm requires a prefill schedule");
    }
    if (plan.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill output norm plan backend must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal prefill output norm requires at least one layer");
    }
    if (feed_forward_output.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal prefill output norm residual phase must match schedule");
    }
    if (feed_forward_output.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill output norm residual scratch ownership must match schedule"
        );
    }
    if (feed_forward_output.layer_index != prefill.layers().back().layer_index) {
        throw std::invalid_argument("Gemma2 Metal prefill output norm residual layer must be terminal");
    }
    if (feed_forward_output.feed_forward_residual.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill output norm residual must use the schedule backend");
    }
    if (feed_forward_output.feed_forward_residual.descriptor() != prefill.scratch().feed_forward_residual) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill output norm residual must match scheduled FFN residual shape"
        );
    }
}

void require_decode_terminal_feed_forward_output_matches_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalFeedForwardOutputActivations& feed_forward_output
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode output norm requires a decode schedule");
    }
    if (decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode) {
        throw std::invalid_argument("Gemma2 Metal decode output norm requires resident decode scratch");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode output norm requires one scheduled token");
    }
    if (plan.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode output norm plan backend must match schedule");
    }
    if (decode.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal decode output norm requires at least one layer");
    }
    if (feed_forward_output.phase != decode.phase()) {
        throw std::invalid_argument("Gemma2 Metal decode output norm residual phase must match schedule");
    }
    if (feed_forward_output.scratch_ownership != decode.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal decode output norm residual scratch ownership must match schedule"
        );
    }
    if (feed_forward_output.layer_index != decode.layers().back().layer_index) {
        throw std::invalid_argument("Gemma2 Metal decode output norm residual layer must be terminal");
    }
    if (feed_forward_output.feed_forward_residual.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode output norm residual must use the schedule backend");
    }
    if (feed_forward_output.feed_forward_residual.descriptor() != decode.scratch().feed_forward_residual) {
        throw std::invalid_argument(
            "Gemma2 Metal decode output norm residual must match scheduled FFN residual shape"
        );
    }
}

void require_prefill_output_norm_matches_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPrefillOutputNormActivations& output_norm
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection requires a prefill schedule");
    }
    if (plan.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection plan backend must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection requires at least one layer");
    }
    if (output_norm.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection input phase must match schedule");
    }
    if (output_norm.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill logits projection input scratch ownership must match schedule"
        );
    }
    if (output_norm.schedule_layer_offset != prefill.layers().size() - 1U) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection input layer offset must be terminal");
    }
    if (output_norm.layer_index != prefill.layers().back().layer_index) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection input layer must be terminal");
    }
    if (output_norm.hidden_width != plan.config().embedding_length) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection input hidden width must match config");
    }
    if (output_norm.normalized_hidden_states.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill logits projection input must use the schedule backend");
    }
    if (output_norm.normalized_hidden_states.descriptor() != prefill.scratch().normalized_hidden) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill logits projection input must match scheduled normalized hidden shape"
        );
    }
}

void require_decode_output_norm_matches_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalDecodeOutputNormActivations& output_norm
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection requires a decode schedule");
    }
    if (decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection requires resident decode scratch");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection requires one scheduled token");
    }
    if (plan.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection plan backend must match schedule");
    }
    if (decode.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection requires at least one layer");
    }
    if (output_norm.phase != decode.phase()) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection input phase must match schedule");
    }
    if (output_norm.scratch_ownership != decode.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal decode logits projection input scratch ownership must match schedule"
        );
    }
    if (output_norm.schedule_layer_offset != decode.layers().size() - 1U) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection input layer offset must be terminal");
    }
    if (output_norm.layer_index != decode.layers().back().layer_index) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection input layer must be terminal");
    }
    if (output_norm.hidden_width != plan.config().embedding_length) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection input hidden width must match config");
    }
    if (output_norm.normalized_hidden_states.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode logits projection input must use the schedule backend");
    }
    if (output_norm.normalized_hidden_states.descriptor() != decode.scratch().normalized_hidden) {
        throw std::invalid_argument(
            "Gemma2 Metal decode logits projection input must match scheduled normalized hidden shape"
        );
    }
}

void require_prefill_logits_match_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPrefillLogitsActivations& logits
) {
    if (prefill.phase() != Gemma2MetalSchedulePhase::prefill) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection requires a prefill schedule");
    }
    if (plan.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection plan backend must match schedule");
    }
    if (prefill.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection requires at least one layer");
    }
    if (prefill.token_count() == 0U) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection requires at least one token");
    }
    if (logits.phase != prefill.phase()) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection input phase must match schedule");
    }
    if (logits.scratch_ownership != prefill.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal prefill next-token selection input scratch ownership must match schedule"
        );
    }
    if (logits.schedule_layer_offset != prefill.layers().size() - 1U) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection input layer offset must be terminal");
    }
    if (logits.layer_index != prefill.layers().back().layer_index) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection input layer must be terminal");
    }
    if (logits.hidden_width != plan.config().embedding_length || logits.vocab_size != plan.config().vocab_size) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection input dimensions must match config");
    }
    if (logits.logits.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection logits must use the schedule backend");
    }
    if (logits.logits.descriptor() != prefill.scratch().logits) {
        throw std::invalid_argument("Gemma2 Metal prefill next-token selection logits must match scheduled shape");
    }
}

void require_decode_logits_match_schedule(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalDecodeLogitsActivations& logits
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection requires a decode schedule");
    }
    if (decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection requires resident decode scratch");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection requires one scheduled token");
    }
    if (plan.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection plan backend must match schedule");
    }
    if (decode.layers().empty()) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection requires at least one layer");
    }
    if (logits.phase != decode.phase()) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection input phase must match schedule");
    }
    if (logits.scratch_ownership != decode.scratch_ownership()) {
        throw std::invalid_argument(
            "Gemma2 Metal decode next-token selection input scratch ownership must match schedule"
        );
    }
    if (logits.schedule_layer_offset != decode.layers().size() - 1U) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection input layer offset must be terminal");
    }
    if (logits.layer_index != decode.layers().back().layer_index) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection input layer must be terminal");
    }
    if (logits.hidden_width != plan.config().embedding_length || logits.vocab_size != plan.config().vocab_size) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection input dimensions must match config");
    }
    if (logits.logits.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection logits must use the schedule backend");
    }
    if (logits.logits.descriptor() != decode.scratch().logits) {
        throw std::invalid_argument("Gemma2 Metal decode next-token selection logits must match scheduled shape");
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

[[nodiscard]] Gemma2MetalPrefillOutputNormActivations create_gemma2_metal_prefill_output_norm(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalFeedForwardOutputActivations& terminal_feed_forward_output
) {
    require_prefill_terminal_feed_forward_output_matches_schedule(
        plan,
        prefill,
        terminal_feed_forward_output
    );

    auto normalized_hidden_states = tensor::ops::rms_norm_f32_to_backend(
        terminal_feed_forward_output.feed_forward_residual,
        plan.weights().output_norm_weight,
        plan.config().rms_norm_epsilon
    );

    require_prefill_output_norm_output_shape(
        normalized_hidden_states,
        prefill.scratch().normalized_hidden
    );

    return Gemma2MetalPrefillOutputNormActivations {
        .schedule_layer_offset = static_cast<std::uint32_t>(prefill.layers().size() - 1U),
        .layer_index = terminal_feed_forward_output.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .normalized_hidden_states = std::move(normalized_hidden_states),
    };
}

[[nodiscard]] tensor::BackendBuffer project_gemma2_logits_to_backend(
    const tensor::BackendBuffer& normalized_hidden_states,
    const tensor::BackendBuffer& output_weight
) {
    switch (output_weight.descriptor().dtype) {
        case tensor::DType::f16:
            return tensor::ops::matmul_f32_f16_gguf_weight_to_backend(
                normalized_hidden_states,
                output_weight
            );
        case tensor::DType::q4_k:
        case tensor::DType::q6_k:
            return tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
                normalized_hidden_states,
                output_weight
            );
        case tensor::DType::f32:
        case tensor::DType::q8_0:
            break;
    }

    throw std::invalid_argument("Gemma2 Metal logits output weight dtype is not supported");
}

[[nodiscard]] Gemma2MetalDecodeOutputNormActivations create_gemma2_metal_decode_output_norm(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalFeedForwardOutputActivations& terminal_feed_forward_output
) {
    require_decode_terminal_feed_forward_output_matches_schedule(
        plan,
        decode,
        terminal_feed_forward_output
    );

    auto normalized_hidden_states = tensor::ops::rms_norm_f32_to_backend(
        terminal_feed_forward_output.feed_forward_residual,
        plan.weights().output_norm_weight,
        plan.config().rms_norm_epsilon
    );

    require_decode_output_norm_output_shape(
        normalized_hidden_states,
        decode.scratch().normalized_hidden
    );

    return Gemma2MetalDecodeOutputNormActivations {
        .schedule_layer_offset = static_cast<std::uint32_t>(decode.layers().size() - 1U),
        .layer_index = terminal_feed_forward_output.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .normalized_hidden_states = std::move(normalized_hidden_states),
    };
}

[[nodiscard]] Gemma2MetalDecodeLogitsActivations create_gemma2_metal_decode_logits(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalDecodeOutputNormActivations& output_norm
) {
    require_decode_output_norm_matches_schedule(plan, decode, output_norm);

    auto raw_logits = project_gemma2_logits_to_backend(
        output_norm.normalized_hidden_states,
        plan.weights().output_weight
    );
    require_decode_logits_output_shape(raw_logits, decode.scratch().logits, "raw output");

    auto logits = tensor::ops::softcap_f32_to_backend(
        raw_logits,
        plan.config().final_logit_softcap
    );
    require_decode_logits_output_shape(logits, decode.scratch().logits, "softcapped output");

    return Gemma2MetalDecodeLogitsActivations {
        .schedule_layer_offset = output_norm.schedule_layer_offset,
        .layer_index = output_norm.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .vocab_size = plan.config().vocab_size,
        .final_logit_softcap = plan.config().final_logit_softcap,
        .raw_logits = std::move(raw_logits),
        .logits = std::move(logits),
    };
}

[[nodiscard]] Gemma2MetalPrefillLogitsActivations create_gemma2_metal_prefill_logits(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPrefillOutputNormActivations& output_norm
) {
    require_prefill_output_norm_matches_schedule(plan, prefill, output_norm);

    auto raw_logits = project_gemma2_logits_to_backend(
        output_norm.normalized_hidden_states,
        plan.weights().output_weight
    );
    require_prefill_logits_output_shape(raw_logits, prefill.scratch().logits, "raw output");

    auto logits = tensor::ops::softcap_f32_to_backend(
        raw_logits,
        plan.config().final_logit_softcap
    );
    require_prefill_logits_output_shape(logits, prefill.scratch().logits, "softcapped output");

    return Gemma2MetalPrefillLogitsActivations {
        .schedule_layer_offset = output_norm.schedule_layer_offset,
        .layer_index = output_norm.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .vocab_size = plan.config().vocab_size,
        .final_logit_softcap = plan.config().final_logit_softcap,
        .raw_logits = std::move(raw_logits),
        .logits = std::move(logits),
    };
}

[[nodiscard]] Gemma2MetalPrefillNextTokenSelection create_gemma2_metal_prefill_next_token_selection(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    const Gemma2MetalPrefillLogitsActivations& logits
) {
    require_prefill_logits_match_schedule(plan, prefill, logits);

    const auto selected = tensor::ops::top_logit_f32_from_backend(logits.logits);

    return Gemma2MetalPrefillNextTokenSelection {
        .schedule_layer_offset = logits.schedule_layer_offset,
        .layer_index = logits.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .logits_row_index = prefill.token_count() - 1U,
        .vocab_size = plan.config().vocab_size,
        .token_id = selected.token,
        .logit = selected.logit,
    };
}

[[nodiscard]] Gemma2MetalDecodeNextTokenSelection create_gemma2_metal_decode_next_token_selection(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalDecodeLogitsActivations& logits
) {
    require_decode_logits_match_schedule(plan, decode, logits);

    const auto selected = tensor::ops::top_logit_f32_from_backend(logits.logits);

    return Gemma2MetalDecodeNextTokenSelection {
        .schedule_layer_offset = logits.schedule_layer_offset,
        .layer_index = logits.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .logits_row_index = 0U,
        .vocab_size = plan.config().vocab_size,
        .token_id = selected.token,
        .logit = selected.logit,
    };
}

[[nodiscard]] Gemma2MetalPrefillKVCacheWriteActivations write_gemma2_metal_prefill_kv_cache(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalAttentionPositionActivations& position,
    const Gemma2MetalAttentionProjectionActivations& projection
) {
    require_prefill_kv_cache_write_matches_schedule(
        plan,
        prefill,
        schedule_layer_offset,
        position,
        projection
    );

    const auto& scheduled_layer = prefill.layers()[schedule_layer_offset];
    auto& cache_layer = plan.kv_cache().at(scheduled_layer.layer_index);
    tensor::ops::write_prefill_kv_cache_f32_to_backend(
        position.key,
        projection.value,
        cache_layer.key,
        cache_layer.value,
        prefill.cache_position(),
        plan.config().attention_head_count_kv,
        plan.config().attention_key_length,
        plan.config().attention_value_length
    );

    return Gemma2MetalPrefillKVCacheWriteActivations {
        .schedule_layer_offset = schedule_layer_offset,
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .cache_position = prefill.cache_position(),
        .token_count = prefill.token_count(),
        .key_head_count = plan.config().attention_head_count_kv,
        .key_length = plan.config().attention_key_length,
        .value_head_count = plan.config().attention_head_count_kv,
        .value_length = plan.config().attention_value_length,
        .key_cache = cache_layer.key,
        .value_cache = cache_layer.value,
    };
}

[[nodiscard]] Gemma2MetalInputActivations create_gemma2_metal_decode_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode) {
        throw std::invalid_argument("Gemma2 Metal decode input requires a decode schedule");
    }
    if (decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 Metal decode input requires one scheduled token");
    }
    if (token_id >= plan.config().vocab_size) {
        throw std::invalid_argument("Gemma2 Metal decode input token must be within vocab");
    }

    std::vector<std::uint32_t> token_ids {token_id};
    auto hidden_states = tensor::ops::embed_f16_gguf_weight_token_ids_to_backend(
        std::span<const std::uint32_t>(token_ids),
        plan.weights().token_embedding_weight,
        embedding_scale
    );
    require_decode_hidden_states_match_schedule(decode, hidden_states);

    return Gemma2MetalInputActivations {
        .token_ids = std::move(token_ids),
        .hidden_states = std::move(hidden_states),
        .embedding_scale = embedding_scale,
    };
}

[[nodiscard]] Gemma2MetalLayerInputActivations create_gemma2_metal_first_decode_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    const Gemma2MetalInputActivations& decode_input
) {
    require_decode_hidden_states_match_schedule(decode, decode_input.hidden_states);
    const auto& first_layer = decode.layers().front();
    auto normalized_attention_input = tensor::ops::rms_norm_f32_to_backend(
        decode_input.hidden_states,
        plan.weights().blocks.at(first_layer.layer_index).attention.attention_norm_weight,
        plan.config().rms_norm_epsilon
    );
    if (normalized_attention_input.descriptor() != decode.scratch().normalized_hidden) {
        throw std::invalid_argument("Gemma2 Metal decode layer input must match scheduled normalized hidden shape");
    }

    return Gemma2MetalLayerInputActivations {
        .layer_index = first_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .normalized_attention_input = std::move(normalized_attention_input),
    };
}

[[nodiscard]] Gemma2MetalDecodeKVCacheWriteActivations write_gemma2_metal_decode_kv_cache(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalAttentionPositionActivations& position,
    const Gemma2MetalAttentionProjectionActivations& projection
) {
    require_decode_kv_cache_write_matches_schedule(
        plan,
        decode,
        schedule_layer_offset,
        position,
        projection
    );

    const auto& scheduled_layer = decode.layers()[schedule_layer_offset];
    auto& cache_layer = plan.kv_cache().at(scheduled_layer.layer_index);
    tensor::ops::write_kv_cache_f32_to_backend(
        position.key,
        projection.value,
        cache_layer.key,
        cache_layer.value,
        decode.cache_position(),
        plan.config().attention_head_count_kv,
        plan.config().attention_key_length,
        plan.config().attention_value_length
    );

    return Gemma2MetalDecodeKVCacheWriteActivations {
        .schedule_layer_offset = schedule_layer_offset,
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .cache_position = decode.cache_position(),
        .token_count = decode.token_count(),
        .total_kv_token_count = decode.total_kv_token_count(),
        .key_head_count = plan.config().attention_head_count_kv,
        .key_length = plan.config().attention_key_length,
        .value_head_count = plan.config().attention_head_count_kv,
        .value_length = plan.config().attention_value_length,
        .key_cache = cache_layer.key,
        .value_cache = cache_layer.value,
    };
}

}  // namespace

std::optional<Gemma2MetalLayerHandoffActivations> create_gemma2_metal_scheduled_prefill_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
) {
    if (plan.backend() != prefill.backend()) {
        throw std::invalid_argument("Gemma2 Metal reusable layer input plan backend must match schedule");
    }
    require_scheduled_prefill_layer_input_residual_matches_schedule(
        prefill,
        schedule_layer_offset,
        previous_feed_forward_output
    );
    if (static_cast<std::size_t>(schedule_layer_offset) == prefill.layers().size()) {
        return std::nullopt;
    }

    const auto& scheduled_layer = prefill.layers()[schedule_layer_offset];
    const auto& attention = plan.weights().blocks.at(scheduled_layer.layer_index).attention;
    auto normalized_attention_input = tensor::ops::rms_norm_f32_to_backend(
        previous_feed_forward_output.feed_forward_residual,
        attention.attention_norm_weight,
        plan.config().rms_norm_epsilon
    );

    require_layer_handoff_output_shape(
        normalized_attention_input,
        prefill.scratch().layer_handoff_normalized_attention_input
    );

    return Gemma2MetalLayerHandoffActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .normalized_attention_input = std::move(normalized_attention_input),
    };
}

std::optional<Gemma2MetalLayerHandoffActivations> create_gemma2_metal_scheduled_decode_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
) {
    if (plan.backend() != decode.backend()) {
        throw std::invalid_argument("Gemma2 Metal reusable decode layer input plan backend must match schedule");
    }
    require_scheduled_decode_layer_input_residual_matches_schedule(
        decode,
        schedule_layer_offset,
        previous_feed_forward_output
    );
    if (static_cast<std::size_t>(schedule_layer_offset) == decode.layers().size()) {
        return std::nullopt;
    }

    const auto& scheduled_layer = decode.layers()[schedule_layer_offset];
    const auto& attention = plan.weights().blocks.at(scheduled_layer.layer_index).attention;
    auto normalized_attention_input = tensor::ops::rms_norm_f32_to_backend(
        previous_feed_forward_output.feed_forward_residual,
        attention.attention_norm_weight,
        plan.config().rms_norm_epsilon
    );

    require_layer_handoff_output_shape(
        normalized_attention_input,
        decode.scratch().layer_handoff_normalized_attention_input
    );

    return Gemma2MetalLayerHandoffActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .normalized_attention_input = std::move(normalized_attention_input),
    };
}

Gemma2MetalPrefillBlockActivations run_gemma2_metal_scheduled_prefill_block(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const tensor::BackendBuffer& normalized_attention_input,
    const tensor::BackendBuffer& residual_input
) {
    require_scheduled_prefill_block_inputs_match_schedule(
        plan,
        prefill,
        schedule_layer_offset,
        normalized_attention_input,
        residual_input
    );

    const auto& scheduled_layer = prefill.layers().at(schedule_layer_offset);
    const auto& block_weights = plan.weights().blocks.at(scheduled_layer.layer_index);
    const auto& attention = block_weights.attention;
    auto query = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_attention_input,
        attention.query_weight
    );
    auto key = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_attention_input,
        attention.key_weight
    );
    auto value = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_attention_input,
        attention.value_weight
    );

    require_attention_projection_output_shape(query, prefill.scratch().query, "scheduled block query");
    require_attention_projection_output_shape(key, prefill.scratch().key, "scheduled block key");
    require_attention_projection_output_shape(value, prefill.scratch().value, "scheduled block value");

    auto attention_projection = Gemma2MetalAttentionProjectionActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .query = std::move(query),
        .key = std::move(key),
        .value = std::move(value),
    };

    const auto head_dimension = plan.config().attention_key_length;
    const auto rope_dimension_count = head_dimension;
    auto positioned_query = tensor::ops::rope_f32_by_heads_with_position_start_to_backend(
        attention_projection.query,
        prefill.cache_position(),
        plan.config().attention_head_count,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );
    auto positioned_key = tensor::ops::rope_f32_by_heads_with_position_start_to_backend(
        attention_projection.key,
        prefill.cache_position(),
        plan.config().attention_head_count_kv,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );

    require_attention_projection_output_shape(
        positioned_query,
        prefill.scratch().query,
        "scheduled block positioned query"
    );
    require_attention_projection_output_shape(
        positioned_key,
        prefill.scratch().key,
        "scheduled block positioned key"
    );

    auto attention_position = Gemma2MetalAttentionPositionActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .position_start = prefill.cache_position(),
        .query_head_count = plan.config().attention_head_count,
        .key_head_count = plan.config().attention_head_count_kv,
        .head_dimension = head_dimension,
        .rope_dimension_count = rope_dimension_count,
        .rope_frequency_base = gemma2_rope_frequency_base,
        .query = std::move(positioned_query),
        .key = std::move(positioned_key),
    };

    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    const auto attention_token_count = scheduled_layer.attention_token_count;
    auto scores = tensor::ops::causal_grouped_attention_scores_f32_to_backend(
        attention_position.query,
        attention_position.key,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        head_dimension,
        attention_token_count,
        scale,
        plan.config().attention_logit_softcap
    );

    require_attention_score_output_shape(scores, prefill.scratch().attention_scores);

    auto attention_scores = Gemma2MetalAttentionScoreActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .query_position_start = prefill.cache_position(),
        .attention_token_count = attention_token_count,
        .query_head_count = plan.config().attention_head_count,
        .key_head_count = plan.config().attention_head_count_kv,
        .key_length = head_dimension,
        .scale = scale,
        .logit_softcap = plan.config().attention_logit_softcap,
        .scores = std::move(scores),
    };

    auto probabilities = tensor::ops::softmax_f32_to_backend(attention_scores.scores);
    require_attention_probability_output_shape(probabilities, prefill.scratch().attention_probabilities);

    auto attention_probabilities = Gemma2MetalAttentionProbabilityActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .attention_token_count = attention_token_count,
        .query_head_count = attention_scores.query_head_count,
        .probabilities = std::move(probabilities),
    };

    const auto value_length = plan.config().attention_value_length;
    auto attention_output = tensor::ops::causal_grouped_attention_values_f32_to_backend(
        attention_probabilities.probabilities,
        attention_projection.value,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        value_length,
        attention_token_count
    );

    require_attention_value_output_shape(attention_output, prefill.scratch().attention_output);

    auto attention_values = Gemma2MetalAttentionValueActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .query_position_start = prefill.cache_position(),
        .attention_token_count = attention_token_count,
        .query_head_count = plan.config().attention_head_count,
        .value_head_count = plan.config().attention_head_count_kv,
        .value_length = value_length,
        .attention_output = std::move(attention_output),
    };

    auto projected_attention_output = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        attention_values.attention_output,
        attention.output_weight
    );

    require_projected_attention_output_shape(
        projected_attention_output,
        prefill.scratch().projected_attention_output
    );

    auto attention_output_projection = Gemma2MetalAttentionOutputProjectionActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .attention_output_width = attention_values.query_head_count * attention_values.value_length,
        .hidden_width = plan.config().embedding_length,
        .projected_attention_output = std::move(projected_attention_output),
    };

    auto normalized_attention_output = tensor::ops::rms_norm_f32_to_backend(
        attention_output_projection.projected_attention_output,
        attention.post_attention_norm_weight,
        plan.config().rms_norm_epsilon
    );
    auto post_attention_residual_output = tensor::ops::add_f32_to_backend(
        residual_input,
        normalized_attention_output
    );

    require_post_attention_residual_output_shape(
        post_attention_residual_output,
        prefill.scratch().post_attention_residual
    );

    auto post_attention_residual = Gemma2MetalPostAttentionResidualActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .post_attention_residual = std::move(post_attention_residual_output),
    };

    const auto& feed_forward = block_weights.feed_forward;
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
        "scheduled block normalized input"
    );
    require_feed_forward_gate_up_output_shape(
        gate,
        prefill.scratch().feed_forward_gate,
        "scheduled block gate projection"
    );
    require_feed_forward_gate_up_output_shape(
        up,
        prefill.scratch().feed_forward_up,
        "scheduled block up projection"
    );

    auto feed_forward_gate_up = Gemma2MetalFeedForwardGateUpActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .feed_forward_width = plan.config().feed_forward_length,
        .normalized_feed_forward_input = std::move(normalized_feed_forward_input),
        .gate = std::move(gate),
        .up = std::move(up),
    };

    auto feed_forward_gated = tensor::ops::silu_multiply_f32_to_backend(
        feed_forward_gate_up.gate,
        feed_forward_gate_up.up
    );
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

    require_feed_forward_output_shape(
        feed_forward_gated,
        prefill.scratch().feed_forward_gated,
        "scheduled block gated"
    );
    require_feed_forward_output_shape(
        projected_feed_forward_output,
        prefill.scratch().projected_feed_forward_output,
        "scheduled block down projection"
    );
    require_feed_forward_output_shape(
        feed_forward_residual,
        prefill.scratch().feed_forward_residual,
        "scheduled block residual"
    );

    auto feed_forward_output = Gemma2MetalFeedForwardOutputActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .feed_forward_width = plan.config().feed_forward_length,
        .feed_forward_gated = std::move(feed_forward_gated),
        .projected_feed_forward_output = std::move(projected_feed_forward_output),
        .feed_forward_residual = std::move(feed_forward_residual),
    };

    return Gemma2MetalPrefillBlockActivations {
        .schedule_layer_offset = schedule_layer_offset,
        .layer_index = scheduled_layer.layer_index,
        .phase = prefill.phase(),
        .scratch_ownership = prefill.scratch_ownership(),
        .attention_projection = std::move(attention_projection),
        .attention_position = std::move(attention_position),
        .attention_scores = std::move(attention_scores),
        .attention_probabilities = std::move(attention_probabilities),
        .attention_values = std::move(attention_values),
        .attention_output_projection = std::move(attention_output_projection),
        .post_attention_residual = std::move(post_attention_residual),
        .feed_forward_gate_up = std::move(feed_forward_gate_up),
        .feed_forward_output = std::move(feed_forward_output),
    };
}

Gemma2MetalDecodeBlockActivations run_gemma2_metal_scheduled_decode_block(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const tensor::BackendBuffer& normalized_attention_input,
    const tensor::BackendBuffer& residual_input
) {
    require_scheduled_decode_block_inputs_match_schedule(
        plan,
        decode,
        schedule_layer_offset,
        normalized_attention_input,
        residual_input
    );

    const auto& scheduled_layer = decode.layers().at(schedule_layer_offset);
    const auto& block_weights = plan.weights().blocks.at(scheduled_layer.layer_index);
    const auto& attention = block_weights.attention;
    auto query = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_attention_input,
        attention.query_weight
    );
    auto key = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_attention_input,
        attention.key_weight
    );
    auto value = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        normalized_attention_input,
        attention.value_weight
    );

    require_attention_projection_output_shape(query, decode.scratch().query, "decode block query");
    require_attention_projection_output_shape(key, decode.scratch().key, "decode block key");
    require_attention_projection_output_shape(value, decode.scratch().value, "decode block value");

    auto attention_projection = Gemma2MetalAttentionProjectionActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .query = std::move(query),
        .key = std::move(key),
        .value = std::move(value),
    };

    const auto head_dimension = plan.config().attention_key_length;
    const auto rope_dimension_count = head_dimension;
    auto positioned_query = tensor::ops::rope_f32_by_heads_with_position_start_to_backend(
        attention_projection.query,
        decode.cache_position(),
        plan.config().attention_head_count,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );
    auto positioned_key = tensor::ops::rope_f32_by_heads_with_position_start_to_backend(
        attention_projection.key,
        decode.cache_position(),
        plan.config().attention_head_count_kv,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );

    require_attention_projection_output_shape(positioned_query, decode.scratch().query, "decode block positioned query");
    require_attention_projection_output_shape(positioned_key, decode.scratch().key, "decode block positioned key");

    auto attention_position = Gemma2MetalAttentionPositionActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .position_start = decode.cache_position(),
        .query_head_count = plan.config().attention_head_count,
        .key_head_count = plan.config().attention_head_count_kv,
        .head_dimension = head_dimension,
        .rope_dimension_count = rope_dimension_count,
        .rope_frequency_base = gemma2_rope_frequency_base,
        .query = std::move(positioned_query),
        .key = std::move(positioned_key),
    };

    auto kv_cache_write = write_gemma2_metal_decode_kv_cache(
        plan,
        decode,
        schedule_layer_offset,
        attention_position,
        attention_projection
    );
    const auto& cache_layer = plan.kv_cache().at(scheduled_layer.layer_index);

    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    const auto attention_token_count = scheduled_layer.attention_token_count;
    auto scores = tensor::ops::causal_grouped_attention_scores_f32_with_kv_cache_to_backend(
        attention_position.query,
        cache_layer.key,
        decode.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        head_dimension,
        attention_token_count,
        scale,
        plan.config().attention_logit_softcap
    );

    const auto exact_attention_scores_shape = tensor::TensorDescriptor::from(
        tensor::Shape::from({decode.token_count(), plan.config().attention_head_count, attention_token_count}),
        tensor::DType::f32
    );
    require_attention_score_output_shape(scores, exact_attention_scores_shape);

    auto attention_scores = Gemma2MetalAttentionScoreActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .query_position_start = decode.cache_position(),
        .attention_token_count = attention_token_count,
        .query_head_count = plan.config().attention_head_count,
        .key_head_count = plan.config().attention_head_count_kv,
        .key_length = head_dimension,
        .scale = scale,
        .logit_softcap = plan.config().attention_logit_softcap,
        .scores = std::move(scores),
    };

    auto probabilities = tensor::ops::softmax_f32_to_backend(attention_scores.scores);
    require_attention_probability_output_shape(probabilities, exact_attention_scores_shape);

    auto attention_probabilities = Gemma2MetalAttentionProbabilityActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .attention_token_count = attention_token_count,
        .query_head_count = attention_scores.query_head_count,
        .probabilities = std::move(probabilities),
    };

    const auto value_length = plan.config().attention_value_length;
    auto attention_output = tensor::ops::causal_grouped_attention_values_f32_with_kv_cache_to_backend(
        attention_probabilities.probabilities,
        cache_layer.value,
        decode.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        value_length,
        attention_token_count
    );

    require_attention_value_output_shape(attention_output, decode.scratch().attention_output);

    auto attention_values = Gemma2MetalAttentionValueActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .query_position_start = decode.cache_position(),
        .attention_token_count = attention_token_count,
        .query_head_count = plan.config().attention_head_count,
        .value_head_count = plan.config().attention_head_count_kv,
        .value_length = value_length,
        .attention_output = std::move(attention_output),
    };

    auto projected_attention_output = tensor::ops::matmul_f32_quantized_gguf_weight_to_backend(
        attention_values.attention_output,
        attention.output_weight
    );

    require_projected_attention_output_shape(projected_attention_output, decode.scratch().projected_attention_output);

    auto attention_output_projection = Gemma2MetalAttentionOutputProjectionActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .attention_output_width = attention_values.query_head_count * attention_values.value_length,
        .hidden_width = plan.config().embedding_length,
        .projected_attention_output = std::move(projected_attention_output),
    };

    auto normalized_attention_output = tensor::ops::rms_norm_f32_to_backend(
        attention_output_projection.projected_attention_output,
        attention.post_attention_norm_weight,
        plan.config().rms_norm_epsilon
    );
    auto post_attention_residual_output = tensor::ops::add_f32_to_backend(
        residual_input,
        normalized_attention_output
    );

    require_post_attention_residual_output_shape(post_attention_residual_output, decode.scratch().post_attention_residual);

    auto post_attention_residual = Gemma2MetalPostAttentionResidualActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .post_attention_residual = std::move(post_attention_residual_output),
    };

    const auto& feed_forward = block_weights.feed_forward;
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
        decode.scratch().normalized_feed_forward_input,
        "decode block normalized input"
    );
    require_feed_forward_gate_up_output_shape(gate, decode.scratch().feed_forward_gate, "decode block gate projection");
    require_feed_forward_gate_up_output_shape(up, decode.scratch().feed_forward_up, "decode block up projection");

    auto feed_forward_gate_up = Gemma2MetalFeedForwardGateUpActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .feed_forward_width = plan.config().feed_forward_length,
        .normalized_feed_forward_input = std::move(normalized_feed_forward_input),
        .gate = std::move(gate),
        .up = std::move(up),
    };

    auto feed_forward_gated = tensor::ops::silu_multiply_f32_to_backend(
        feed_forward_gate_up.gate,
        feed_forward_gate_up.up
    );
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

    require_feed_forward_output_shape(feed_forward_gated, decode.scratch().feed_forward_gated, "decode block gated");
    require_feed_forward_output_shape(
        projected_feed_forward_output,
        decode.scratch().projected_feed_forward_output,
        "decode block down projection"
    );
    require_feed_forward_output_shape(feed_forward_residual, decode.scratch().feed_forward_residual, "decode block residual");

    auto feed_forward_output = Gemma2MetalFeedForwardOutputActivations {
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .hidden_width = plan.config().embedding_length,
        .feed_forward_width = plan.config().feed_forward_length,
        .feed_forward_gated = std::move(feed_forward_gated),
        .projected_feed_forward_output = std::move(projected_feed_forward_output),
        .feed_forward_residual = std::move(feed_forward_residual),
    };

    return Gemma2MetalDecodeBlockActivations {
        .schedule_layer_offset = schedule_layer_offset,
        .layer_index = scheduled_layer.layer_index,
        .phase = decode.phase(),
        .scratch_ownership = decode.scratch_ownership(),
        .attention_projection = std::move(attention_projection),
        .attention_position = std::move(attention_position),
        .kv_cache_write = std::move(kv_cache_write),
        .attention_scores = std::move(attention_scores),
        .attention_probabilities = std::move(attention_probabilities),
        .attention_values = std::move(attention_values),
        .attention_output_projection = std::move(attention_output_projection),
        .post_attention_residual = std::move(post_attention_residual),
        .feed_forward_gate_up = std::move(feed_forward_gate_up),
        .feed_forward_output = std::move(feed_forward_output),
    };
}

Gemma2MetalDecodeStepActivations run_gemma2_metal_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale
) {
    auto input = create_gemma2_metal_decode_input(
        plan,
        decode,
        token_id,
        embedding_scale
    );
    auto first_layer_input = create_gemma2_metal_first_decode_layer_input(
        plan,
        decode,
        input
    );
    auto first_block = run_gemma2_metal_scheduled_decode_block(
        plan,
        decode,
        0,
        first_layer_input.normalized_attention_input,
        input.hidden_states
    );

    std::vector<Gemma2MetalLayerHandoffActivations> reusable_layer_inputs;
    std::vector<Gemma2MetalDecodeBlockActivations> reusable_blocks;
    const auto scheduled_decode_layer_count = decode.layers().size();
    if (scheduled_decode_layer_count > 1U) {
        reusable_layer_inputs.reserve(scheduled_decode_layer_count - 1U);
        reusable_blocks.reserve(scheduled_decode_layer_count - 1U);
    }

    const Gemma2MetalFeedForwardOutputActivations* terminal_feed_forward_output =
        &first_block.feed_forward_output;
    for (
        std::uint32_t schedule_layer_offset = 1U;
        static_cast<std::size_t>(schedule_layer_offset) < scheduled_decode_layer_count;
        ++schedule_layer_offset
    ) {
        auto next_layer_input = create_gemma2_metal_scheduled_decode_layer_input(
            plan,
            decode,
            schedule_layer_offset,
            *terminal_feed_forward_output
        );
        if (!next_layer_input.has_value()) {
            break;
        }

        reusable_layer_inputs.push_back(std::move(*next_layer_input));
        reusable_blocks.push_back(run_gemma2_metal_scheduled_decode_block(
            plan,
            decode,
            schedule_layer_offset,
            reusable_layer_inputs.back().normalized_attention_input,
            terminal_feed_forward_output->feed_forward_residual
        ));
        terminal_feed_forward_output = &reusable_blocks.back().feed_forward_output;
    }

    auto terminal_output = *terminal_feed_forward_output;
    auto output_norm = create_gemma2_metal_decode_output_norm(
        plan,
        decode,
        terminal_output
    );
    auto logits = create_gemma2_metal_decode_logits(
        plan,
        decode,
        output_norm
    );
    auto next_token_selection = create_gemma2_metal_decode_next_token_selection(
        plan,
        decode,
        logits
    );

    return Gemma2MetalDecodeStepActivations {
        .decode = decode,
        .input = std::move(input),
        .first_layer_input = std::move(first_layer_input),
        .first_block = std::move(first_block),
        .reusable_layer_inputs = std::move(reusable_layer_inputs),
        .reusable_blocks = std::move(reusable_blocks),
        .terminal_feed_forward_output = std::move(terminal_output),
        .output_norm = std::move(output_norm),
        .logits = std::move(logits),
        .next_token_selection = next_token_selection,
    };
}

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
    std::vector<Gemma2MetalLayerHandoffActivations> prefill_reusable_layer_inputs;
    std::vector<Gemma2MetalPrefillBlockActivations> prefill_reusable_blocks;
    const auto scheduled_prefill_layer_count = prefill.layers().size();
    if (scheduled_prefill_layer_count > 1U) {
        prefill_reusable_layer_inputs.reserve(scheduled_prefill_layer_count - 1U);
        prefill_reusable_blocks.reserve(scheduled_prefill_layer_count - 1U);
    }

    const Gemma2MetalFeedForwardOutputActivations* terminal_feed_forward_output =
        &prefill_first_layer_feed_forward_output;
    for (
        std::uint32_t schedule_layer_offset = 1U;
        static_cast<std::size_t>(schedule_layer_offset) < scheduled_prefill_layer_count;
        ++schedule_layer_offset
    ) {
        auto next_layer_input = create_gemma2_metal_scheduled_prefill_layer_input(
            plan,
            prefill,
            schedule_layer_offset,
            *terminal_feed_forward_output
        );
        if (!next_layer_input.has_value()) {
            break;
        }

        prefill_reusable_layer_inputs.push_back(std::move(*next_layer_input));
        prefill_reusable_blocks.push_back(run_gemma2_metal_scheduled_prefill_block(
            plan,
            prefill,
            schedule_layer_offset,
            prefill_reusable_layer_inputs.back().normalized_attention_input,
            terminal_feed_forward_output->feed_forward_residual
        ));
        terminal_feed_forward_output = &prefill_reusable_blocks.back().feed_forward_output;
    }
    std::vector<Gemma2MetalPrefillKVCacheWriteActivations> prefill_kv_cache_writes;
    prefill_kv_cache_writes.reserve(scheduled_prefill_layer_count);
    prefill_kv_cache_writes.push_back(write_gemma2_metal_prefill_kv_cache(
        plan,
        prefill,
        0,
        prefill_first_layer_attention_position,
        prefill_first_layer_attention_projection
    ));
    for (const auto& block : prefill_reusable_blocks) {
        prefill_kv_cache_writes.push_back(write_gemma2_metal_prefill_kv_cache(
            plan,
            prefill,
            block.schedule_layer_offset,
            block.attention_position,
            block.attention_projection
        ));
    }

    auto prefill_output_norm = create_gemma2_metal_prefill_output_norm(
        plan,
        prefill,
        *terminal_feed_forward_output
    );
    auto prefill_logits = create_gemma2_metal_prefill_logits(
        plan,
        prefill,
        prefill_output_norm
    );
    auto prefill_next_token_selection = create_gemma2_metal_prefill_next_token_selection(
        plan,
        prefill,
        prefill_logits
    );
    auto first_decode_step = run_gemma2_metal_decode_step(
        plan,
        first_decode,
        prefill_next_token_selection.token_id,
        embedding_scale
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
        .prefill_reusable_layer_inputs = std::move(prefill_reusable_layer_inputs),
        .prefill_reusable_blocks = std::move(prefill_reusable_blocks),
        .prefill_kv_cache_writes = std::move(prefill_kv_cache_writes),
        .prefill_output_norm = std::move(prefill_output_norm),
        .prefill_logits = std::move(prefill_logits),
        .prefill_next_token_selection = prefill_next_token_selection,
        .first_decode_input = std::move(first_decode_step.input),
        .first_decode_first_layer_input = std::move(first_decode_step.first_layer_input),
        .first_decode_block = std::move(first_decode_step.first_block),
        .first_decode_reusable_layer_inputs = std::move(first_decode_step.reusable_layer_inputs),
        .first_decode_reusable_blocks = std::move(first_decode_step.reusable_blocks),
        .first_decode_terminal_feed_forward_output =
            std::move(first_decode_step.terminal_feed_forward_output),
        .first_decode_output_norm = std::move(first_decode_step.output_norm),
        .first_decode_logits = std::move(first_decode_step.logits),
        .first_decode_next_token_selection = first_decode_step.next_token_selection,
    };
}

Gemma2MetalGeneratedTokenLoopActivations create_gemma2_metal_generated_token_loop(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    std::optional<std::uint32_t> stop_token_id
) {
    require_gemma2_metal_generation_inputs(
        model.gemma2_config(),
        prompt_token_ids,
        max_new_tokens
    );

    auto runtime_shell = create_gemma2_metal_generation_runtime_shell(model, prompt_token_ids);
    const auto embedding_scale = gemma2_input_embedding_scale(
        runtime_shell.plan.config().embedding_length
    );

    std::vector<std::uint32_t> generated_token_ids;
    generated_token_ids.reserve(max_new_tokens);
    std::vector<Gemma2MetalDecodeStepActivations> decode_steps;
    decode_steps.reserve(max_new_tokens);

    generated_token_ids.push_back(runtime_shell.prefill_next_token_selection.token_id);
    decode_steps.push_back(Gemma2MetalDecodeStepActivations {
        .decode = runtime_shell.first_decode,
        .input = runtime_shell.first_decode_input,
        .first_layer_input = runtime_shell.first_decode_first_layer_input,
        .first_block = runtime_shell.first_decode_block,
        .reusable_layer_inputs = runtime_shell.first_decode_reusable_layer_inputs,
        .reusable_blocks = runtime_shell.first_decode_reusable_blocks,
        .terminal_feed_forward_output = runtime_shell.first_decode_terminal_feed_forward_output,
        .output_norm = runtime_shell.first_decode_output_norm,
        .logits = runtime_shell.first_decode_logits,
        .next_token_selection = runtime_shell.first_decode_next_token_selection,
    });
    if (stop_token_id == generated_token_ids.back() || generated_token_ids.size() == max_new_tokens) {
        return Gemma2MetalGeneratedTokenLoopActivations {
            .runtime_shell = std::move(runtime_shell),
            .generated_token_ids = std::move(generated_token_ids),
            .decode_steps = std::move(decode_steps),
        };
    }

    generated_token_ids.push_back(runtime_shell.first_decode_next_token_selection.token_id);
    if (stop_token_id == generated_token_ids.back() || generated_token_ids.size() == max_new_tokens) {
        return Gemma2MetalGeneratedTokenLoopActivations {
            .runtime_shell = std::move(runtime_shell),
            .generated_token_ids = std::move(generated_token_ids),
            .decode_steps = std::move(decode_steps),
        };
    }

    const auto prompt_token_count = checked_prompt_token_count(prompt_token_ids);
    while (generated_token_ids.size() < max_new_tokens) {
        const auto cached_token_count = static_cast<std::uint32_t>(
            prompt_token_count + generated_token_ids.size() - 1U
        );
        auto decode = create_gemma2_metal_decode_schedule(
            runtime_shell.plan,
            cached_token_count
        );
        decode_steps.push_back(run_gemma2_metal_decode_step(
            runtime_shell.plan,
            decode,
            generated_token_ids.back(),
            embedding_scale
        ));
        generated_token_ids.push_back(decode_steps.back().next_token_selection.token_id);
        if (stop_token_id == generated_token_ids.back()) {
            break;
        }
    }

    return Gemma2MetalGeneratedTokenLoopActivations {
        .runtime_shell = std::move(runtime_shell),
        .generated_token_ids = std::move(generated_token_ids),
        .decode_steps = std::move(decode_steps),
    };
}

std::vector<std::uint32_t> generate_gemma2_metal_argmax_tokens(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    std::optional<std::uint32_t> stop_token_id
) {
    auto loop = create_gemma2_metal_generated_token_loop(
        model,
        prompt_token_ids,
        max_new_tokens,
        stop_token_id
    );

    std::vector<std::uint32_t> token_ids(prompt_token_ids.begin(), prompt_token_ids.end());
    token_ids.reserve(prompt_token_ids.size() + loop.generated_token_ids.size());
    token_ids.insert(
        token_ids.end(),
        loop.generated_token_ids.begin(),
        loop.generated_token_ids.end()
    );
    return token_ids;
}

}  // namespace local_gguf_engine::engine
