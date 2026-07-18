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

#include "../tensor/backends/metal/metal_command_context.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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

[[nodiscard]] std::uint64_t monotonic_now_ns(const Gemma2MonotonicClock& clock) {
    if (clock) {
        return clock();
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count());
}

[[nodiscard]] std::uint64_t elapsed_ns(std::uint64_t start, std::uint64_t end) {
    if (end < start) {
        throw std::runtime_error("Gemma2 measurement clock moved backwards");
    }
    return end - start;
}

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

void require_gemma2_metal_plan_generation_inputs(
    const Gemma2ModelConfig& config,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
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
    const auto context_length = static_cast<std::size_t>(config.context_length);
    if (cache_position > config.context_length) {
        throw std::invalid_argument("Gemma2 Metal generation cache position exceeds context length");
    }
    const auto used_context = static_cast<std::size_t>(cache_position);
    if (prompt_token_ids.size() > context_length - used_context) {
        throw std::invalid_argument("Gemma2 Metal generation token count exceeds context length");
    }
    if (prompt_token_ids.size() + max_new_tokens > context_length - used_context) {
        throw std::invalid_argument("Gemma2 Metal generation token count exceeds context length");
    }
}

[[nodiscard]] bool token_is_stop(
    std::uint32_t token_id,
    std::span<const std::uint32_t> stop_token_ids
) noexcept {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), token_id) != stop_token_ids.end();
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



void require_kv_cache_layer_shape(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalKVCacheLayer& cache_layer
) {
    const auto& config = plan.config();
    if (cache_layer.key.backend() != plan.backend() || cache_layer.value.backend() != plan.backend()) {
        throw std::invalid_argument("Gemma2 Metal KV cache write target must use the plan backend");
    }
    if (cache_layer.key.descriptor().dtype != tensor::DType::f32) {
        throw std::invalid_argument("Gemma2 Metal KV cache key target must be F32");
    }
    if (cache_layer.value.descriptor().dtype != tensor::DType::f32) {
        throw std::invalid_argument("Gemma2 Metal KV cache value target must be F32");
    }
    if (
        cache_layer.key.descriptor().shape.rank() != 3U
        || cache_layer.key.descriptor().shape.dimension(0) != config.context_length
        || cache_layer.key.descriptor().shape.dimension(1) != config.attention_head_count_kv
        || cache_layer.key.descriptor().shape.dimension(2) != config.attention_key_length
    ) {
        throw std::invalid_argument("Gemma2 Metal KV cache key target shape mismatch");
    }
    if (
        cache_layer.value.descriptor().shape.rank() != 3U
        || cache_layer.value.descriptor().shape.dimension(0) != config.context_length
        || cache_layer.value.descriptor().shape.dimension(1) != config.attention_head_count_kv
        || cache_layer.value.descriptor().shape.dimension(2) != config.attention_value_length
    ) {
        throw std::invalid_argument("Gemma2 Metal KV cache value target shape mismatch");
    }
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
    require_kv_cache_layer_shape(plan, plan.kv_cache().at(scheduled_layer.layer_index));
}

void require_attention_score_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument("Gemma2 Metal attention score output must match scheduled scratch shape");
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


void require_attention_value_output_shape(
    const tensor::BackendBuffer& output,
    const tensor::TensorDescriptor& expected
) {
    if (output.descriptor() != expected) {
        throw std::invalid_argument("Gemma2 Metal attention value output must match scheduled scratch shape");
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
    throw std::invalid_argument("Gemma2 Metal output projection weight dtype is unsupported");
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

[[nodiscard]] Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_tokens_with_plan_impl(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    const Gemma2MetalTokenSelector& select_token,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock,
    bool use_resident_top_logit
);

}  // namespace

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
    auto positioned_query = tensor::ops::rope_neox_f32_by_heads_with_position_start_to_backend(
        attention_projection.query,
        decode.cache_position(),
        plan.config().attention_head_count,
        head_dimension,
        rope_dimension_count,
        gemma2_rope_frequency_base
    );
    auto positioned_key = tensor::ops::rope_neox_f32_by_heads_with_position_start_to_backend(
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

    auto feed_forward_gated = tensor::ops::gelu_tanh_multiply_f32_to_backend(
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

Gemma2MetalResidentDecodeResult run_gemma2_metal_resident_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale,
    bool select_top_logit
) {
    if (decode.phase() != Gemma2MetalSchedulePhase::decode
        || decode.scratch_ownership() != Gemma2MetalScratchOwnership::resident_decode
        || decode.token_count() != 1U) {
        throw std::invalid_argument("Gemma2 resident Metal decode requires a one-token decode schedule");
    }
    if (plan.backend() != decode.backend() || decode.layers().size() != plan.weights().blocks.size()) {
        throw std::invalid_argument("Gemma2 resident Metal decode plan and schedule do not match");
    }
    if (token_id >= plan.config().vocab_size) {
        throw std::invalid_argument("Gemma2 resident Metal decode token is outside vocabulary");
    }

    namespace metal = tensor::backends::metal;
    metal::MetalCommandContext command("Gemma2 resident decode");
    const auto& config = plan.config();
    const auto& scratch = plan.decode_scratch();

    command.write_u32(scratch.token_id_staging, token_id);
    // feed_forward_residual is the layer-to-layer residual ping-pong endpoint;
    // before layer zero it safely serves as the embedded hidden state.
    command.encode_embedding(
        scratch.token_id_staging,
        plan.weights().token_embedding_weight,
        scratch.feed_forward_residual,
        embedding_scale
    );

    const auto head_dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    for (std::size_t offset = 0; offset < decode.layers().size(); ++offset) {
        const auto& scheduled_layer = decode.layers()[offset];
        const auto& block = plan.weights().blocks[scheduled_layer.layer_index];
        const auto& attention = block.attention;
        const auto& feed_forward = block.feed_forward;
        const auto& cache = plan.kv_cache()[scheduled_layer.layer_index];

        command.encode_rms_norm(
            scratch.feed_forward_residual,
            attention.attention_norm_weight,
            scratch.normalized_hidden,
            config.rms_norm_epsilon
        );
        command.encode_matmul(scratch.normalized_hidden, attention.query_weight, scratch.raw_query);
        command.encode_matmul(scratch.normalized_hidden, attention.key_weight, scratch.raw_key);
        command.encode_matmul(scratch.normalized_hidden, attention.value_weight, scratch.value);
        command.encode_rope(
            scratch.raw_query,
            scratch.query,
            decode.cache_position(),
            config.attention_head_count,
            head_dimension,
            head_dimension,
            gemma2_rope_frequency_base
        );
        command.encode_rope(
            scratch.raw_key,
            scratch.key,
            decode.cache_position(),
            config.attention_head_count_kv,
            head_dimension,
            head_dimension,
            gemma2_rope_frequency_base
        );
        command.encode_kv_cache_write(
            scratch.key,
            scratch.value,
            cache.key,
            cache.value,
            decode.cache_position()
        );
        command.encode_attention_scores(
            scratch.query,
            cache.key,
            scratch.attention_scores,
            decode.cache_position(),
            config.attention_head_count,
            config.attention_head_count_kv,
            head_dimension,
            scheduled_layer.attention_token_count,
            attention_scale,
            config.attention_logit_softcap
        );
        command.encode_softmax(
            scratch.attention_scores,
            scratch.attention_probabilities,
            config.attention_head_count,
            scheduled_layer.attention_token_count
        );
        command.encode_attention_values(
            scratch.attention_probabilities,
            cache.value,
            scratch.attention_output,
            decode.cache_position(),
            config.attention_head_count,
            config.attention_head_count_kv,
            config.attention_value_length,
            scheduled_layer.attention_token_count
        );
        command.encode_matmul(
            scratch.attention_output,
            attention.output_weight,
            scratch.projected_attention_output
        );
        // RMSNorm is safe in place: the current kernel finishes its reduction
        // before overwriting any element in the row.
        command.encode_rms_norm(
            scratch.projected_attention_output,
            attention.post_attention_norm_weight,
            scratch.projected_attention_output,
            config.rms_norm_epsilon
        );
        command.encode_add(
            scratch.feed_forward_residual,
            scratch.projected_attention_output,
            scratch.post_attention_residual
        );
        command.encode_rms_norm(
            scratch.post_attention_residual,
            feed_forward.feed_forward_norm_weight,
            scratch.normalized_feed_forward_input,
            config.rms_norm_epsilon
        );
        command.encode_matmul(
            scratch.normalized_feed_forward_input,
            feed_forward.gate_weight,
            scratch.feed_forward_gate
        );
        command.encode_matmul(
            scratch.normalized_feed_forward_input,
            feed_forward.up_weight,
            scratch.feed_forward_up
        );
        command.encode_gelu_tanh_multiply(
            scratch.feed_forward_gate,
            scratch.feed_forward_up,
            scratch.feed_forward_gated
        );
        command.encode_matmul(
            scratch.feed_forward_gated,
            feed_forward.down_weight,
            scratch.projected_feed_forward_output
        );
        command.encode_rms_norm(
            scratch.projected_feed_forward_output,
            feed_forward.post_feed_forward_norm_weight,
            scratch.projected_feed_forward_output,
            config.rms_norm_epsilon
        );
        command.encode_add(
            scratch.post_attention_residual,
            scratch.projected_feed_forward_output,
            scratch.feed_forward_residual
        );
    }

    command.encode_rms_norm(
        scratch.feed_forward_residual,
        plan.weights().output_norm_weight,
        scratch.normalized_hidden,
        config.rms_norm_epsilon
    );
    command.encode_matmul(scratch.normalized_hidden, plan.weights().output_weight, scratch.logits);
    command.encode_softcap(scratch.logits, scratch.logits, config.final_logit_softcap);
    if (select_top_logit) {
        command.encode_top_logit(scratch.logits, scratch.top_logit_staging);
    }

    const auto metal_stats = command.commit_and_wait();
    std::optional<Gemma2MetalDecodeNextTokenSelection> selection;
    if (select_top_logit) {
        const auto top = command.read_top_logit(scratch.top_logit_staging);
        selection = Gemma2MetalDecodeNextTokenSelection {
            .schedule_layer_offset = static_cast<std::uint32_t>(decode.layers().size() - 1U),
            .layer_index = decode.layers().back().layer_index,
            .phase = decode.phase(),
            .scratch_ownership = decode.scratch_ownership(),
            .logits_row_index = 0,
            .vocab_size = config.vocab_size,
            .token_id = top.token,
            .logit = top.logit,
        };
    }

    return Gemma2MetalResidentDecodeResult {
        .logits = scratch.logits,
        .next_token_selection = std::move(selection),
        .execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = metal_stats.command_buffer_count,
            .commit_count = metal_stats.commit_count,
            .synchronous_wait_count = metal_stats.synchronous_wait_count,
            .compute_encoder_count = metal_stats.compute_encoder_count,
            .blit_encoder_count = metal_stats.blit_encoder_count,
            .transient_buffer_allocation_count = metal_stats.transient_buffer_allocation_count,
            .gpu_duration_ns = metal_stats.gpu_duration_ns,
        },
    };
}

Gemma2MetalResidentPrefillResult run_gemma2_metal_resident_prefill(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    float embedding_scale,
    bool select_top_logit
) {
    if (prompt_token_ids.empty()) {
        throw std::invalid_argument("Gemma2 resident Metal prefill token ids must not be empty");
    }
    if (cache_position > plan.config().context_length
        || prompt_token_ids.size() > plan.config().context_length - cache_position) {
        throw std::invalid_argument("Gemma2 resident Metal prefill range exceeds context length");
    }
    for (const auto token_id : prompt_token_ids) {
        if (token_id >= plan.config().vocab_size) {
            throw std::invalid_argument("Gemma2 resident Metal prefill token is outside vocabulary");
        }
    }

    namespace metal = tensor::backends::metal;
    const auto& config = plan.config();
    const auto& scratch = plan.prefill_scratch();
    const auto head_dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    const auto hidden_elements_per_token = static_cast<std::uint64_t>(config.embedding_length);
    const auto feed_forward_elements_per_token =
        static_cast<std::uint64_t>(config.feed_forward_length);
    const auto chunk_count = static_cast<std::uint32_t>(
        (prompt_token_ids.size() + gemma2_metal_prefill_chunk_size - 1U)
        / gemma2_metal_prefill_chunk_size
    );

    std::vector<metal::MetalCommandContext> commands;
    commands.reserve(chunk_count);
    std::size_t token_offset = 0;
    while (token_offset < prompt_token_ids.size()) {
        const auto chunk_token_count = static_cast<std::uint32_t>(std::min<std::size_t>(
            gemma2_metal_prefill_chunk_size,
            prompt_token_ids.size() - token_offset
        ));
        const auto chunk_cache_position =
            cache_position + static_cast<std::uint32_t>(token_offset);
        const auto chunk_tokens = prompt_token_ids.subspan(token_offset, chunk_token_count);
        const auto schedule = create_gemma2_metal_prefill_schedule(
            plan,
            chunk_token_count,
            chunk_cache_position
        );
        const bool final_chunk = token_offset + chunk_token_count == prompt_token_ids.size();

        metal::MetalCommandContext command("Gemma2 resident prefill chunk");
        command.write_u32s(scratch.token_id_staging, chunk_cache_position, chunk_tokens);
        command.encode_embedding(
            scratch.token_id_staging,
            plan.weights().token_embedding_weight,
            scratch.feed_forward_residual,
            embedding_scale,
            chunk_cache_position,
            chunk_token_count
        );

        for (std::size_t offset = 0; offset < schedule.layers().size(); ++offset) {
            const auto& scheduled_layer = schedule.layers()[offset];
            const auto& block = plan.weights().blocks[scheduled_layer.layer_index];
            const auto& attention = block.attention;
            const auto& feed_forward = block.feed_forward;
            const auto& cache = plan.kv_cache()[scheduled_layer.layer_index];

            command.encode_rms_norm(
                scratch.feed_forward_residual,
                attention.attention_norm_weight,
                scratch.normalized_hidden,
                config.rms_norm_epsilon,
                chunk_token_count
            );
            command.encode_matmul(
                scratch.normalized_hidden,
                attention.query_weight,
                scratch.raw_query,
                chunk_token_count
            );
            command.encode_matmul(
                scratch.normalized_hidden,
                attention.key_weight,
                scratch.raw_key,
                chunk_token_count
            );
            command.encode_matmul(
                scratch.normalized_hidden,
                attention.value_weight,
                scratch.value,
                chunk_token_count
            );
            command.encode_rope(
                scratch.raw_query,
                scratch.query,
                chunk_cache_position,
                config.attention_head_count,
                head_dimension,
                head_dimension,
                gemma2_rope_frequency_base,
                chunk_token_count
            );
            command.encode_rope(
                scratch.raw_key,
                scratch.key,
                chunk_cache_position,
                config.attention_head_count_kv,
                head_dimension,
                head_dimension,
                gemma2_rope_frequency_base,
                chunk_token_count
            );
            command.encode_kv_cache_write(
                scratch.key,
                scratch.value,
                cache.key,
                cache.value,
                chunk_cache_position,
                chunk_token_count
            );
            command.encode_attention_scores(
                scratch.query,
                cache.key,
                scratch.attention_scores,
                chunk_cache_position,
                config.attention_head_count,
                config.attention_head_count_kv,
                head_dimension,
                scheduled_layer.attention_token_count,
                attention_scale,
                config.attention_logit_softcap,
                chunk_token_count
            );
            command.encode_softmax(
                scratch.attention_scores,
                scratch.attention_probabilities,
                chunk_token_count * config.attention_head_count,
                scheduled_layer.attention_token_count
            );
            command.encode_attention_values(
                scratch.attention_probabilities,
                cache.value,
                scratch.attention_output,
                chunk_cache_position,
                config.attention_head_count,
                config.attention_head_count_kv,
                config.attention_value_length,
                scheduled_layer.attention_token_count,
                chunk_token_count
            );
            command.encode_matmul(
                scratch.attention_output,
                attention.output_weight,
                scratch.projected_attention_output,
                chunk_token_count
            );
            command.encode_rms_norm(
                scratch.projected_attention_output,
                attention.post_attention_norm_weight,
                scratch.projected_attention_output,
                config.rms_norm_epsilon,
                chunk_token_count
            );
            command.encode_add(
                scratch.feed_forward_residual,
                scratch.projected_attention_output,
                scratch.post_attention_residual,
                chunk_token_count * hidden_elements_per_token
            );
            command.encode_rms_norm(
                scratch.post_attention_residual,
                feed_forward.feed_forward_norm_weight,
                scratch.normalized_feed_forward_input,
                config.rms_norm_epsilon,
                chunk_token_count
            );
            command.encode_matmul(
                scratch.normalized_feed_forward_input,
                feed_forward.gate_weight,
                scratch.feed_forward_gate,
                chunk_token_count
            );
            command.encode_matmul(
                scratch.normalized_feed_forward_input,
                feed_forward.up_weight,
                scratch.feed_forward_up,
                chunk_token_count
            );
            command.encode_gelu_tanh_multiply(
                scratch.feed_forward_gate,
                scratch.feed_forward_up,
                scratch.feed_forward_gated,
                chunk_token_count * feed_forward_elements_per_token
            );
            command.encode_matmul(
                scratch.feed_forward_gated,
                feed_forward.down_weight,
                scratch.projected_feed_forward_output,
                chunk_token_count
            );
            command.encode_rms_norm(
                scratch.projected_feed_forward_output,
                feed_forward.post_feed_forward_norm_weight,
                scratch.projected_feed_forward_output,
                config.rms_norm_epsilon,
                chunk_token_count
            );
            command.encode_add(
                scratch.post_attention_residual,
                scratch.projected_feed_forward_output,
                scratch.feed_forward_residual,
                chunk_token_count * hidden_elements_per_token
            );
        }

        if (final_chunk) {
            command.encode_rms_norm(
                scratch.feed_forward_residual,
                plan.weights().output_norm_weight,
                scratch.final_normalized_hidden,
                config.rms_norm_epsilon,
                1U,
                chunk_token_count - 1U
            );
            command.encode_matmul(
                scratch.final_normalized_hidden,
                plan.weights().output_weight,
                scratch.logits,
                1U
            );
            command.encode_softcap(
                scratch.logits,
                scratch.logits,
                config.final_logit_softcap,
                config.vocab_size
            );
            if (select_top_logit) {
                command.encode_top_logit(scratch.logits, scratch.top_logit_staging);
            }
        }

        static_cast<void>(command.commit());
        commands.push_back(std::move(command));
        token_offset += chunk_token_count;
    }

    static_cast<void>(commands.back().wait_until_completed());
    Gemma2MetalPrefillExecutionStats execution {
        .chunk_count = chunk_count,
        .scheduled_layer_execution_count =
            static_cast<std::uint32_t>(plan.layers().size()) * chunk_count,
        .kv_write_count = static_cast<std::uint32_t>(plan.layers().size()) * chunk_count,
    };
    for (const auto& command : commands) {
        const auto stats = command.completion_stats();
        execution.command_buffer_count += stats.command_buffer_count;
        execution.commit_count += stats.commit_count;
        execution.synchronous_wait_count += stats.synchronous_wait_count;
        execution.compute_encoder_count += stats.compute_encoder_count;
        execution.blit_encoder_count += stats.blit_encoder_count;
        execution.transient_buffer_allocation_count += stats.transient_buffer_allocation_count;
        execution.gpu_duration_ns += stats.gpu_duration_ns;
    }

    std::optional<Gemma2MetalPrefillNextTokenSelection> selection;
    if (select_top_logit) {
        const auto top = commands.back().read_top_logit(scratch.top_logit_staging);
        selection = Gemma2MetalPrefillNextTokenSelection {
            .schedule_layer_offset = static_cast<std::uint32_t>(plan.layers().size() - 1U),
            .layer_index = plan.layers().back().layer_index,
            .phase = Gemma2MetalSchedulePhase::prefill,
            .scratch_ownership = Gemma2MetalScratchOwnership::resident_prefill,
            .logits_row_index = 0,
            .vocab_size = config.vocab_size,
            .token_id = top.token,
            .logit = top.logit,
        };
    }

    return Gemma2MetalResidentPrefillResult {
        .logits = scratch.logits,
        .next_token_selection = std::move(selection),
        .execution = execution,
    };
}

Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_argmax_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock
) {
    const auto select_token = [](const tensor::BackendBuffer& logits) {
        return tensor::ops::top_logit_f32_from_backend(logits).token;
    };
    return generate_gemma2_metal_tokens_with_plan_impl(
        plan,
        prompt_token_ids,
        cache_position,
        max_new_tokens,
        stop_token_ids,
        select_token,
        on_token,
        clock,
        true
    );
}

namespace {

Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_tokens_with_plan_impl(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    const Gemma2MetalTokenSelector& select_token,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock,
    bool use_resident_top_logit
) {
    if (!select_token) {
        throw std::invalid_argument("Gemma2 Metal sampled generation requires a token selector");
    }
    require_gemma2_metal_plan_generation_inputs(
        plan.config(),
        prompt_token_ids,
        cache_position,
        max_new_tokens
    );

    const auto prompt_token_count = checked_prompt_token_count(prompt_token_ids);
    Gemma2GenerationMetrics metrics {
        .prefill_started_ns = monotonic_now_ns(clock),
        .newly_evaluated_prompt_token_count = prompt_token_count,
    };
    const auto embedding_scale = gemma2_input_embedding_scale(plan.config().embedding_length);
    auto prefill_result = run_gemma2_metal_resident_prefill(
        plan,
        prompt_token_ids,
        cache_position,
        embedding_scale,
        use_resident_top_logit
    );
    metrics.prefill_completed_ns = monotonic_now_ns(clock);
    auto next_logits = prefill_result.logits;

    std::vector<std::uint32_t> generated_token_ids;
    generated_token_ids.reserve(max_new_tokens);
    auto cached_token_count = cache_position + prompt_token_count;
    std::optional<std::uint32_t> stop_token_id;
    std::optional<std::uint32_t> resident_selected_token =
        prefill_result.next_token_selection.has_value()
        ? std::optional<std::uint32_t>(prefill_result.next_token_selection->token_id)
        : std::nullopt;
    bool selected_first_token = false;

    for (std::size_t sampled_count = 0; sampled_count < max_new_tokens; ++sampled_count) {
        const auto selection_started_ns = monotonic_now_ns(clock);
        const auto selected_token_id = resident_selected_token.has_value()
            ? *resident_selected_token
            : select_token(next_logits);
        const auto selection_completed_ns = monotonic_now_ns(clock);
        if (selected_token_id >= plan.config().vocab_size) {
            throw std::invalid_argument("Gemma2 Metal token selector returned token outside vocabulary");
        }
        if (!selected_first_token) {
            selected_first_token = true;
            metrics.first_token_selected_ns = selection_completed_ns;
            metrics.prefill_duration_ns = elapsed_ns(metrics.prefill_started_ns, selection_completed_ns);
            metrics.runtime_ttft_ns = metrics.prefill_duration_ns;
        } else {
            metrics.decode_duration_ns += elapsed_ns(selection_started_ns, selection_completed_ns);
        }
        const auto decode = create_gemma2_metal_decode_schedule(plan, cached_token_count);
        const auto decode_started_ns = monotonic_now_ns(clock);
        if (metrics.decode_token_count == 0U) {
            metrics.decode_started_ns = decode_started_ns;
        }
        auto decode_step = run_gemma2_metal_resident_decode_step(
            plan,
            decode,
            selected_token_id,
            embedding_scale,
            use_resident_top_logit
        );
        const auto decode_completed_ns = monotonic_now_ns(clock);
        metrics.decode_completed_ns = decode_completed_ns;
        metrics.decode_duration_ns += elapsed_ns(decode_started_ns, decode_completed_ns);
        ++metrics.decode_token_count;
        cached_token_count = decode.total_kv_token_count();

        if (token_is_stop(selected_token_id, stop_token_ids)) {
            stop_token_id = selected_token_id;
            metrics.stop_reason = Gemma2GenerationStopReason::stop_token;
            break;
        }

        generated_token_ids.push_back(selected_token_id);
        if (on_token) {
            const auto decision = on_token(Gemma2MetalGeneratedTokenEvent {
                .token_id = selected_token_id,
                .generated_token_count = generated_token_ids.size(),
                .cached_token_count = cached_token_count,
            });
            if (decision == Gemma2MetalTokenStreamDecision::stop_generation) {
                metrics.stop_reason = Gemma2GenerationStopReason::callback_stop;
                break;
            }
        }
        if (sampled_count + 1U == max_new_tokens) {
            break;
        }

        next_logits = decode_step.logits;
        resident_selected_token = decode_step.next_token_selection.has_value()
            ? std::optional<std::uint32_t>(decode_step.next_token_selection->token_id)
            : std::nullopt;
    }

    metrics.generation_completed_ns = monotonic_now_ns(clock);
    metrics.generated_token_count = checked_prompt_token_count(
        std::span<const std::uint32_t>(generated_token_ids)
    );

    return Gemma2MetalPlanGeneratedTokens {
        .generated_token_ids = std::move(generated_token_ids),
        .stop_token_id = stop_token_id,
        .prefill_cache_position = cache_position,
        .prefill_token_count = prompt_token_count,
        .cached_token_count = cached_token_count,
        .metrics = metrics,
    };
}

}  // namespace

Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_sampled_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    const Gemma2MetalTokenSelector& select_token,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock
) {
    return generate_gemma2_metal_tokens_with_plan_impl(
        plan,
        prompt_token_ids,
        cache_position,
        max_new_tokens,
        stop_token_ids,
        select_token,
        on_token,
        clock,
        false
    );
}

std::vector<std::uint32_t> generate_gemma2_metal_argmax_tokens(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    std::optional<std::uint32_t> stop_token_id
) {
    require_gemma2_metal_generation_inputs(model.gemma2_config(), prompt_token_ids, max_new_tokens);
    auto plan = create_gemma2_metal_model_plan(model);
    const std::array<std::uint32_t, 1> stop_storage {stop_token_id.value_or(0U)};
    const auto stop_tokens = stop_token_id.has_value()
        ? std::span<const std::uint32_t>(stop_storage)
        : std::span<const std::uint32_t>();
    auto generated = generate_gemma2_metal_argmax_tokens_with_plan(
        plan,
        prompt_token_ids,
        0,
        max_new_tokens,
        stop_tokens
    );

    std::vector<std::uint32_t> token_ids(prompt_token_ids.begin(), prompt_token_ids.end());
    token_ids.reserve(prompt_token_ids.size() + generated.generated_token_ids.size());
    token_ids.insert(
        token_ids.end(),
        generated.generated_token_ids.begin(),
        generated.generated_token_ids.end()
    );
    return token_ids;
}

}  // namespace local_gguf_engine::engine
