#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"

#include "../tensor/backends/metal/metal_command_context.hpp"
#include "../tensor/backends/metal_backend.hpp"

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

[[nodiscard]] Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_tokens_with_plan_impl(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock
);

}  // namespace

Gemma2MetalResidentDecodeResult run_gemma2_metal_resident_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request,
    Gemma2MetalDecodeOutputMode output_mode
) {
    if (plan.has_pending_verification()) {
        throw std::logic_error("Gemma2 resident decode cannot run with pending verification");
    }
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
    if (output_mode == Gemma2MetalDecodeOutputMode::kv_only && selection_request.has_value()) {
        throw std::invalid_argument("Gemma2 KV-only decode cannot request token selection");
    }

    namespace metal = tensor::backends::metal;
    metal::MetalCommandContext command("Gemma2 resident decode", true);
    const auto& config = plan.config();
    const auto& scratch = plan.decode_scratch();

    command.write_u32(scratch.token_id_staging, token_id);
    // feed_forward_residual is the layer-to-layer residual ping-pong endpoint;
    // before layer zero it safely serves as the embedded hidden state.
    command.push_debug_group("Embedding");
    command.encode_embedding(
        scratch.token_id_staging,
        plan.weights().token_embedding_weight,
        scratch.feed_forward_residual,
        embedding_scale
    );
    command.encode_memory_barrier();
    command.pop_debug_group();

    const auto head_dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    for (std::size_t offset = 0; offset < decode.layers().size(); ++offset) {
        const auto& scheduled_layer = decode.layers()[offset];
        const auto& block = plan.weights().blocks[scheduled_layer.layer_index];
        const auto& attention = block.attention;
        const auto& feed_forward = block.feed_forward;
        const auto& cache = plan.kv_cache()[scheduled_layer.layer_index];
        const auto layer_label = "Layer " + std::to_string(scheduled_layer.layer_index)
            + " attention projections";
        command.push_debug_group(layer_label);
        command.encode_rms_norm(
            scratch.feed_forward_residual,
            attention.attention_norm_weight,
            scratch.normalized_hidden,
            config.rms_norm_epsilon
        );
        command.encode_memory_barrier();
        command.encode_matmul(scratch.normalized_hidden, attention.query_weight, scratch.raw_query);
        command.encode_matmul(scratch.normalized_hidden, attention.key_weight, scratch.raw_key);
        command.encode_matmul(
            scratch.normalized_hidden,
            attention.value_weight,
            scratch.raw_value
        );
        command.encode_memory_barrier();
        command.pop_debug_group();
        command.push_debug_group("Attention and KV");
        command.encode_rope(
            scratch.raw_query,
            scratch.query,
            decode.cache_position(),
            config.attention_head_count,
            head_dimension,
            head_dimension,
            gemma2_rope_frequency_base
        );
        command.encode_gemma2_kv_cache_store(
            scratch.raw_key,
            scratch.raw_value,
            cache.key,
            cache.value,
            decode.cache_position(),
            config.attention_head_count_kv,
            head_dimension,
            head_dimension,
            gemma2_rope_frequency_base,
            cache.row_mask
        );
        command.encode_memory_barrier();
        command.encode_gemma2_decode_attention(
            scratch.query,
            cache.key,
            cache.value,
            scratch.attention_output,
            scratch.attention_partial_output,
            scratch.attention_partial_maxima,
            scratch.attention_partial_denominators,
            decode.cache_position(),
            scheduled_layer.attention_token_count,
            attention_scale,
            config.attention_logit_softcap,
            cache.row_mask
        );
        command.encode_memory_barrier();
        command.pop_debug_group();
        command.push_debug_group("Attention output projection");
        command.encode_matmul(
            scratch.attention_output,
            attention.output_weight,
            scratch.projected_attention_output
        );
        command.encode_memory_barrier();
        command.pop_debug_group();
        command.push_debug_group("FFN gate up and GELU");
        command.encode_rms_norm_add(
            scratch.projected_attention_output,
            attention.post_attention_norm_weight,
            scratch.feed_forward_residual,
            scratch.post_attention_residual,
            config.rms_norm_epsilon
        );
        command.encode_memory_barrier();
        command.encode_rms_norm(
            scratch.post_attention_residual,
            feed_forward.feed_forward_norm_weight,
            scratch.normalized_feed_forward_input,
            config.rms_norm_epsilon
        );
        command.encode_memory_barrier();
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
        command.encode_memory_barrier();
        command.encode_gelu_tanh_multiply(
            scratch.feed_forward_gate,
            scratch.feed_forward_up,
            scratch.feed_forward_gated
        );
        command.encode_memory_barrier();
        command.pop_debug_group();
        command.push_debug_group("FFN down projection");
        command.encode_matmul(
            scratch.feed_forward_gated,
            feed_forward.down_weight,
            scratch.projected_feed_forward_output
        );
        command.encode_memory_barrier();
        command.encode_rms_norm_add(
            scratch.projected_feed_forward_output,
            feed_forward.post_feed_forward_norm_weight,
            scratch.post_attention_residual,
            scratch.feed_forward_residual,
            config.rms_norm_epsilon
        );
        command.encode_memory_barrier();
        command.pop_debug_group();
    }

    if (output_mode == Gemma2MetalDecodeOutputMode::logits) {
        command.push_debug_group("Final norm and output head");
        command.encode_rms_norm(
            scratch.feed_forward_residual,
            plan.weights().output_norm_weight,
            scratch.normalized_hidden,
            config.rms_norm_epsilon
        );
        command.encode_memory_barrier();
        command.encode_matmul(scratch.normalized_hidden, plan.weights().output_weight, scratch.logits);
        command.encode_memory_barrier();
        command.pop_debug_group();
    }
    std::uint32_t selector_dispatch_count = 0;
    if (selection_request.has_value()) {
        command.push_debug_group("Token selection");
        selector_dispatch_count = command.encode_token_candidates(
            scratch.logits, plan.selection_workspace(), *selection_request, config.final_logit_softcap
        );
        command.pop_debug_group();
    }

    const auto metal_stats = command.commit_and_wait();
    auto candidates = selection_request.has_value()
        ? std::optional<CompactTokenCandidates>(command.read_token_candidates(plan.selection_workspace()))
        : std::nullopt;

    return Gemma2MetalResidentDecodeResult {
        .logits = scratch.logits,
        .produced_logits = output_mode == Gemma2MetalDecodeOutputMode::logits,
        .candidates = candidates,
        .execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = metal_stats.command_buffer_count,
            .commit_count = metal_stats.commit_count,
            .synchronous_wait_count = metal_stats.synchronous_wait_count,
            .compute_encoder_count = metal_stats.compute_encoder_count,
            .compute_dispatch_count = metal_stats.compute_dispatch_count,
            .blit_encoder_count = metal_stats.blit_encoder_count,
            .memory_barrier_count = metal_stats.memory_barrier_count,
            .transient_buffer_allocation_count = metal_stats.transient_buffer_allocation_count,
            .selector_dispatch_count = selector_dispatch_count,
            .compact_readback_bytes = candidates.has_value() ? candidates->readback_bytes : 0U,
            .candidate_count = candidates.has_value() ? candidates->count : 0U,
            .gpu_duration_ns = metal_stats.gpu_duration_ns,
            .q4_decode_projection_count = metal_stats.q4_decode_projection_count,
            .q8_decode_projection_count = metal_stats.q8_decode_projection_count,
            .q8_width_two_projection_count = metal_stats.q8_width_two_projection_count,
            .q4_small_batch_projection_count = metal_stats.q4_small_batch_projection_count,
            .q4_width_two_projection_count = metal_stats.q4_width_two_projection_count,
            .q4_exact_16_projection_count = metal_stats.q4_exact_16_projection_count,
            .q4_generic_tiled_projection_count = metal_stats.q4_generic_tiled_projection_count,
            .q6_decode_projection_count = metal_stats.q6_decode_projection_count,
            .q6_small_batch_projection_count = metal_stats.q6_small_batch_projection_count,
            .q6_width_two_projection_count = metal_stats.q6_width_two_projection_count,
            .q6_exact_16_projection_count = metal_stats.q6_exact_16_projection_count,
            .q6_generic_tiled_projection_count = metal_stats.q6_generic_tiled_projection_count,
            .f16_projection_count = metal_stats.f16_projection_count,
            .output_head_elision_count = output_mode == Gemma2MetalDecodeOutputMode::kv_only ? 1U : 0U,
            .output_elided_weight_bytes = output_mode == Gemma2MetalDecodeOutputMode::kv_only
                ? plan.weights().output_norm_weight.descriptor().byte_count
                    + plan.weights().output_weight.descriptor().byte_count
                : 0U,
            .decode_attention_dispatch_count = metal_stats.decode_attention_dispatch_count,
            .decode_attention_partition_count = metal_stats.decode_attention_partition_count,
            .decode_attention_token_count = metal_stats.decode_attention_token_count,
        },
    };
}

Gemma2MetalVerifierResult run_gemma2_metal_width_two_verifier(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& verifier,
    std::span<const std::uint32_t, 2> token_ids,
    float embedding_scale
) {
    if (plan.has_pending_verification()) {
        throw std::logic_error("Gemma2 width-two verifier already has a pending transaction");
    }
    if (verifier.phase() != Gemma2MetalSchedulePhase::verify
        || verifier.scratch_ownership() != Gemma2MetalScratchOwnership::resident_verifier
        || verifier.token_count() != 2U) {
        throw std::invalid_argument("Gemma2 width-two verifier requires a two-token verifier schedule");
    }
    if (plan.backend() != verifier.backend()
        || verifier.layers().size() != plan.weights().blocks.size()) {
        throw std::invalid_argument("Gemma2 width-two verifier plan and schedule do not match");
    }
    for (const auto token_id : token_ids) {
        if (token_id >= plan.config().vocab_size) {
            throw std::invalid_argument("Gemma2 width-two verifier token is outside vocabulary");
        }
    }

    namespace metal = tensor::backends::metal;
    metal::MetalCommandContext command("Gemma2 width-two verifier", true);
    const auto& config = plan.config();
    const auto& scratch = plan.verifier_scratch();
    command.write_u32s(scratch.token_id_staging, 0U, token_ids);
    command.push_debug_group("Width-two embedding");
    command.encode_embedding(
        scratch.token_id_staging,
        plan.weights().token_embedding_weight,
        scratch.feed_forward_residual,
        embedding_scale,
        0U,
        2U
    );
    command.encode_memory_barrier();
    command.pop_debug_group();

    const auto head_dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    for (const auto& scheduled_layer : verifier.layers()) {
        const auto& block = plan.weights().blocks[scheduled_layer.layer_index];
        const auto& attention = block.attention;
        const auto& feed_forward = block.feed_forward;
        const auto& cache = plan.kv_cache()[scheduled_layer.layer_index];
        const auto& tentative = scratch.tentative_kv[scheduled_layer.layer_index];

        command.push_debug_group("Width-two attention projections");
        command.encode_rms_norm(
            scratch.feed_forward_residual,
            attention.attention_norm_weight,
            scratch.normalized_hidden,
            config.rms_norm_epsilon,
            2U
        );
        command.encode_memory_barrier();
        command.encode_matmul(scratch.normalized_hidden, attention.query_weight, scratch.raw_query, 2U);
        command.encode_matmul(scratch.normalized_hidden, attention.key_weight, scratch.raw_key, 2U);
        command.encode_matmul(scratch.normalized_hidden, attention.value_weight, scratch.raw_value, 2U);
        command.encode_memory_barrier();
        command.encode_rope(
            scratch.raw_query,
            scratch.query,
            verifier.cache_position(),
            config.attention_head_count,
            head_dimension,
            head_dimension,
            gemma2_rope_frequency_base,
            2U
        );
        command.encode_gemma2_kv_cache_copy(
            cache.key, cache.value, scratch.backup_key, scratch.backup_value,
            verifier.cache_position(), 2U, cache.row_mask, 1U
        );
        command.encode_memory_barrier();
        // Before the sliding ring is full, row one occupies unused storage and
        // both proposed rows can be staged together. At full-window pressure,
        // however, row one's slot still holds the oldest key/value needed by
        // row-zero attention, so that store must wait until row zero finishes.
        const bool second_store_aliases_first_attention =
            scheduled_layer.attention_token_count == cache.row_capacity;
        command.encode_gemma2_kv_cache_store(
            scratch.raw_key, scratch.raw_value, cache.key, cache.value,
            verifier.cache_position(), config.attention_head_count_kv,
            head_dimension, head_dimension, gemma2_rope_frequency_base,
            cache.row_mask, second_store_aliases_first_attention ? 1U : 2U, 0U
        );
        command.encode_memory_barrier();
        command.encode_gemma2_decode_attention(
            scratch.query, cache.key, cache.value, scratch.attention_output,
            scratch.attention_partial_output, scratch.attention_partial_maxima,
            scratch.attention_partial_denominators, verifier.cache_position(),
            scheduled_layer.attention_token_count, attention_scale,
            config.attention_logit_softcap, cache.row_mask, 0U, 0U
        );
        command.encode_memory_barrier();
        if (second_store_aliases_first_attention) {
            command.encode_gemma2_kv_cache_store(
                scratch.raw_key, scratch.raw_value, cache.key, cache.value,
                verifier.cache_position() + 1U, config.attention_head_count_kv,
                head_dimension, head_dimension, gemma2_rope_frequency_base,
                cache.row_mask, 1U, 1U
            );
            command.encode_memory_barrier();
        }
        command.encode_gemma2_decode_attention(
            scratch.query, cache.key, cache.value, scratch.attention_output,
            scratch.attention_partial_output, scratch.attention_partial_maxima,
            scratch.attention_partial_denominators, verifier.cache_position() + 1U,
            scheduled_layer.second_attention_token_count, attention_scale,
            config.attention_logit_softcap, cache.row_mask, 1U, 1U
        );
        command.encode_memory_barrier();
        command.encode_gemma2_kv_cache_copy(
            cache.key, cache.value, tentative.key, tentative.value,
            verifier.cache_position(), 2U, cache.row_mask, 1U
        );
        command.encode_memory_barrier();
        command.encode_gemma2_kv_cache_copy(
            scratch.backup_key, scratch.backup_value, cache.key, cache.value,
            verifier.cache_position(), 2U, 1U, cache.row_mask
        );
        command.encode_memory_barrier();
        command.pop_debug_group();

        command.encode_matmul(
            scratch.attention_output, attention.output_weight,
            scratch.projected_attention_output, 2U
        );
        command.encode_memory_barrier();
        command.encode_rms_norm_add(
            scratch.projected_attention_output,
            attention.post_attention_norm_weight,
            scratch.feed_forward_residual,
            scratch.post_attention_residual,
            config.rms_norm_epsilon,
            2U
        );
        command.encode_memory_barrier();
        command.encode_rms_norm(
            scratch.post_attention_residual,
            feed_forward.feed_forward_norm_weight,
            scratch.normalized_feed_forward_input,
            config.rms_norm_epsilon,
            2U
        );
        command.encode_memory_barrier();
        command.encode_matmul(
            scratch.normalized_feed_forward_input, feed_forward.gate_weight,
            scratch.feed_forward_gate, 2U
        );
        command.encode_matmul(
            scratch.normalized_feed_forward_input, feed_forward.up_weight,
            scratch.feed_forward_up, 2U
        );
        command.encode_memory_barrier();
        command.encode_gelu_tanh_multiply(
            scratch.feed_forward_gate, scratch.feed_forward_up,
            scratch.feed_forward_gated,
            static_cast<std::uint64_t>(2U) * config.feed_forward_length
        );
        command.encode_memory_barrier();
        command.encode_matmul(
            scratch.feed_forward_gated, feed_forward.down_weight,
            scratch.projected_feed_forward_output, 2U
        );
        command.encode_memory_barrier();
        command.encode_rms_norm_add(
            scratch.projected_feed_forward_output,
            feed_forward.post_feed_forward_norm_weight,
            scratch.post_attention_residual,
            scratch.feed_forward_residual,
            config.rms_norm_epsilon,
            2U
        );
        command.encode_memory_barrier();
    }

    command.push_debug_group("Width-two final norm and output head");
    command.encode_rms_norm(
        scratch.feed_forward_residual, plan.weights().output_norm_weight,
        scratch.normalized_hidden, config.rms_norm_epsilon, 2U
    );
    command.encode_memory_barrier();
    command.encode_matmul(
        scratch.normalized_hidden, plan.weights().output_weight, scratch.logits, 2U
    );
    command.encode_memory_barrier();
    command.pop_debug_group();

    const auto stats = command.commit_and_wait();
    plan.set_pending_verification(verifier.cache_position());
    return Gemma2MetalVerifierResult {
        .logits = scratch.logits,
        .cache_position = verifier.cache_position(),
        .execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = stats.command_buffer_count,
            .commit_count = stats.commit_count,
            .synchronous_wait_count = stats.synchronous_wait_count,
            .compute_encoder_count = stats.compute_encoder_count,
            .compute_dispatch_count = stats.compute_dispatch_count,
            .blit_encoder_count = stats.blit_encoder_count,
            .memory_barrier_count = stats.memory_barrier_count,
            .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
            .gpu_duration_ns = stats.gpu_duration_ns,
            .q4_decode_projection_count = stats.q4_decode_projection_count,
            .q8_decode_projection_count = stats.q8_decode_projection_count,
            .q8_width_two_projection_count = stats.q8_width_two_projection_count,
            .q4_small_batch_projection_count = stats.q4_small_batch_projection_count,
            .q4_width_two_projection_count = stats.q4_width_two_projection_count,
            .q4_exact_16_projection_count = stats.q4_exact_16_projection_count,
            .q4_generic_tiled_projection_count = stats.q4_generic_tiled_projection_count,
            .q6_decode_projection_count = stats.q6_decode_projection_count,
            .q6_small_batch_projection_count = stats.q6_small_batch_projection_count,
            .q6_width_two_projection_count = stats.q6_width_two_projection_count,
            .q6_exact_16_projection_count = stats.q6_exact_16_projection_count,
            .q6_generic_tiled_projection_count = stats.q6_generic_tiled_projection_count,
            .f16_projection_count = stats.f16_projection_count,
            .decode_attention_dispatch_count = stats.decode_attention_dispatch_count,
            .decode_attention_partition_count = stats.decode_attention_partition_count,
            .decode_attention_token_count = stats.decode_attention_token_count,
        },
    };
}

Gemma2MetalVerifierCommitResult commit_gemma2_metal_width_two_verification(
    Gemma2MetalModelPlan& plan,
    std::uint32_t accepted_token_count
) {
    if (!plan.has_pending_verification()) {
        throw std::logic_error("Gemma2 width-two commit requires pending verification");
    }
    if (accepted_token_count > 2U) {
        throw std::invalid_argument("Gemma2 width-two commit accepts at most two tokens");
    }
    if (accepted_token_count == 0U) {
        plan.clear_pending_verification();
        return Gemma2MetalVerifierCommitResult {};
    }

    namespace metal = tensor::backends::metal;
    metal::MetalCommandContext command("Gemma2 width-two KV commit", true);
    const auto position = plan.pending_verification_position();
    const auto& scratch = plan.verifier_scratch();
    for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
        const auto& tentative = scratch.tentative_kv[layer];
        const auto& cache = plan.kv_cache()[layer];
        command.encode_gemma2_kv_cache_copy(
            tentative.key, tentative.value, cache.key, cache.value,
            position, accepted_token_count, 1U, cache.row_mask
        );
    }
    const auto stats = command.commit_and_wait();
    plan.clear_pending_verification();
    return Gemma2MetalVerifierCommitResult {
        .accepted_token_count = accepted_token_count,
        .execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = stats.command_buffer_count,
            .commit_count = stats.commit_count,
            .synchronous_wait_count = stats.synchronous_wait_count,
            .compute_encoder_count = stats.compute_encoder_count,
            .compute_dispatch_count = stats.compute_dispatch_count,
            .blit_encoder_count = stats.blit_encoder_count,
            .memory_barrier_count = stats.memory_barrier_count,
            .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
            .gpu_duration_ns = stats.gpu_duration_ns,
        },
    };
}

void discard_gemma2_metal_width_two_verification(Gemma2MetalModelPlan& plan) noexcept {
    plan.clear_pending_verification();
}

Gemma2MetalResidentPrefillResult run_gemma2_metal_resident_prefill(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request
) {
    if (plan.has_pending_verification()) {
        throw std::logic_error("Gemma2 resident prefill cannot run with pending verification");
    }
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
    std::vector<metal::MetalCommandContext> commands;
    commands.reserve(
        (prompt_token_ids.size() + gemma2_metal_prefill_chunk_size - 1U)
        / gemma2_metal_prefill_chunk_size + 1U
    );
    std::size_t token_offset = 0;
    std::uint32_t selector_dispatch_count = 0;
    while (token_offset < prompt_token_ids.size()) {
        const auto chunk_cache_position =
            cache_position + static_cast<std::uint32_t>(token_offset);
        // A submitted chunk may not wrap the exact sliding ring: later writes
        // would otherwise overwrite old rows still needed by its early queries.
        const auto rows_until_ring_boundary =
            config.sliding_window - (chunk_cache_position & (config.sliding_window - 1U));
        const auto chunk_token_count = static_cast<std::uint32_t>(std::min<std::size_t>({
            gemma2_metal_prefill_chunk_size,
            prompt_token_ids.size() - token_offset,
            rows_until_ring_boundary,
        }));
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
            command.encode_gemma2_kv_cache_store(
                scratch.raw_key,
                scratch.value,
                cache.key,
                cache.value,
                chunk_cache_position,
                config.attention_head_count_kv,
                head_dimension,
                head_dimension,
                gemma2_rope_frequency_base,
                cache.row_mask,
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
                cache.row_mask,
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
                cache.row_mask,
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
            if (selection_request.has_value()) {
                selector_dispatch_count = command.encode_token_candidates(
                    scratch.logits,
                    plan.selection_workspace(),
                    *selection_request,
                    config.final_logit_softcap
                );
            }
        }

        static_cast<void>(command.commit());
        commands.push_back(std::move(command));
        token_offset += chunk_token_count;
    }

    static_cast<void>(commands.back().wait_until_completed());
    const auto chunk_count = static_cast<std::uint32_t>(commands.size());
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
        execution.compute_dispatch_count += stats.compute_dispatch_count;
        execution.blit_encoder_count += stats.blit_encoder_count;
        execution.memory_barrier_count += stats.memory_barrier_count;
        execution.transient_buffer_allocation_count += stats.transient_buffer_allocation_count;
        execution.gpu_duration_ns += stats.gpu_duration_ns;
        execution.q4_decode_projection_count += stats.q4_decode_projection_count;
        execution.q8_decode_projection_count += stats.q8_decode_projection_count;
        execution.q4_small_batch_projection_count += stats.q4_small_batch_projection_count;
        execution.q4_exact_16_projection_count += stats.q4_exact_16_projection_count;
        execution.q4_generic_tiled_projection_count += stats.q4_generic_tiled_projection_count;
        execution.q6_decode_projection_count += stats.q6_decode_projection_count;
        execution.q6_small_batch_projection_count += stats.q6_small_batch_projection_count;
        execution.q6_exact_16_projection_count += stats.q6_exact_16_projection_count;
        execution.q6_generic_tiled_projection_count += stats.q6_generic_tiled_projection_count;
        execution.f16_projection_count += stats.f16_projection_count;
        execution.decode_attention_dispatch_count += stats.decode_attention_dispatch_count;
        execution.decode_attention_partition_count += stats.decode_attention_partition_count;
        execution.decode_attention_token_count += stats.decode_attention_token_count;
    }

    auto candidates = selection_request.has_value()
        ? std::optional<CompactTokenCandidates>(commands.back().read_token_candidates(plan.selection_workspace()))
        : std::nullopt;
    execution.selector_dispatch_count = selector_dispatch_count;
    execution.compact_readback_bytes = candidates.has_value() ? candidates->readback_bytes : 0U;
    execution.candidate_count = candidates.has_value() ? candidates->count : 0U;

    return Gemma2MetalResidentPrefillResult {
        .logits = scratch.logits,
        .candidates = candidates,
        .execution = execution,
    };
}

Gemma2MetalWarmupStats warm_up_gemma2_metal_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t, 16> token_ids,
    TokenSampler sampler_copy
) {
    if (plan.config().context_length <= token_ids.size()) {
        throw std::invalid_argument("Gemma2 Metal warm-up requires at least 17 context positions");
    }

    const auto allocation_identities = plan.allocation_identities();
    const auto started = std::chrono::steady_clock::now();
    const auto embedding_scale = gemma2_input_embedding_scale(plan.config().embedding_length);
    const auto prefill = run_gemma2_metal_resident_prefill(
        plan,
        token_ids,
        0U,
        embedding_scale,
        sampler_copy.selection_request()
    );
    if (!prefill.candidates.has_value()) {
        throw std::runtime_error("Gemma2 Metal warm-up prefill did not return token candidates");
    }
    const auto selected_token = sampler_copy.sample(std::span<const TokenLogit>(
        prefill.candidates->values.data(), prefill.candidates->count
    ));
    sampler_copy.observe_tokens(std::span<const std::uint32_t>(&selected_token, 1U));
    const auto decode_schedule = create_gemma2_metal_decode_schedule(
        plan, static_cast<std::uint32_t>(token_ids.size())
    );
    const auto decode = run_gemma2_metal_resident_decode_step(
        plan,
        decode_schedule,
        selected_token,
        embedding_scale,
        sampler_copy.selection_request()
    );
    if (!decode.candidates.has_value()) {
        throw std::runtime_error("Gemma2 Metal warm-up decode did not return token candidates");
    }
    if (plan.allocation_identities() != allocation_identities) {
        throw std::runtime_error("Gemma2 Metal warm-up replaced a resident allocation");
    }
    const auto completed = std::chrono::steady_clock::now();
    return Gemma2MetalWarmupStats {
        .wall_duration_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(completed - started).count()
        ),
        .prefill_token_count = static_cast<std::uint32_t>(token_ids.size()),
        .kv_clear_pass_count = 0U,
        .host_cleared_kv_bytes = 0U,
        .prefill = prefill.execution,
        .decode = decode.execution,
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
    TokenSampler sampler(TokenSamplerOptions {.temperature = 0.0F});
    return generate_gemma2_metal_tokens_with_plan_impl(
        plan,
        prompt_token_ids,
        cache_position,
        max_new_tokens,
        stop_token_ids,
        sampler,
        on_token,
        clock
    );
}

namespace {

Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_tokens_with_plan_impl(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock
) {
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
        sampler.selection_request()
    );
    metrics.prefill_completed_ns = monotonic_now_ns(clock);
    metrics.prompt_evaluation_duration_ns = elapsed_ns(
        metrics.prefill_started_ns, metrics.prefill_completed_ns
    );
    if (!prefill_result.candidates.has_value()) {
        throw std::runtime_error("Gemma2 Metal prefill omitted compact token candidates");
    }
    auto candidates = *prefill_result.candidates;
    metrics.compact_readback_bytes += prefill_result.execution.compact_readback_bytes;
    metrics.selector_dispatch_count += prefill_result.execution.selector_dispatch_count;
    metrics.max_candidate_count = prefill_result.execution.candidate_count;

    std::vector<std::uint32_t> generated_token_ids;
    generated_token_ids.reserve(max_new_tokens);
    std::vector<std::uint32_t> selected_token_ids;
    selected_token_ids.reserve(max_new_tokens);
    Gemma2MetalDecodeExecutionStats decode_execution;
    auto cached_token_count = cache_position + prompt_token_count;
    std::optional<std::uint32_t> stop_token_id;
    bool selected_first_token = false;

    for (std::size_t sampled_count = 0; sampled_count < max_new_tokens; ++sampled_count) {
        const auto selection_started_ns = monotonic_now_ns(clock);
        const auto selected_token_id = sampler.sample(
            std::span<const TokenLogit>(candidates.values).first(candidates.count)
        );
        const auto selection_completed_ns = monotonic_now_ns(clock);
        metrics.selection_duration_ns += elapsed_ns(selection_started_ns, selection_completed_ns);
        metrics.last_token_selected_ns = selection_completed_ns;
        if (selected_token_id >= plan.config().vocab_size) {
            throw std::invalid_argument("Gemma2 Metal token selector returned token outside vocabulary");
        }
        if (!selected_first_token) {
            selected_first_token = true;
            metrics.first_token_selected_ns = selection_completed_ns;
            metrics.runtime_first_token_selected_ns = elapsed_ns(
                metrics.prefill_started_ns, selection_completed_ns
            );
        }
        selected_token_ids.push_back(selected_token_id);
        sampler.observe_tokens(std::span<const std::uint32_t>(&selected_token_id, 1U));
        const auto decode = create_gemma2_metal_decode_schedule(plan, cached_token_count);
        const auto decode_started_ns = monotonic_now_ns(clock);
        if (metrics.decode_token_count == 0U) {
            metrics.decode_started_ns = decode_started_ns;
        }
        const auto needs_next_candidate = sampled_count + 1U < max_new_tokens
            && !token_is_stop(selected_token_id, stop_token_ids);
        auto decode_step = run_gemma2_metal_resident_decode_step(
            plan,
            decode,
            selected_token_id,
            embedding_scale,
            needs_next_candidate
                ? std::optional<TokenSelectionRequest>(sampler.selection_request())
                : std::nullopt,
            needs_next_candidate
                ? Gemma2MetalDecodeOutputMode::logits
                : Gemma2MetalDecodeOutputMode::kv_only
        );
        const auto decode_completed_ns = monotonic_now_ns(clock);
        metrics.decode_completed_ns = decode_completed_ns;
        metrics.decode_duration_ns += elapsed_ns(decode_started_ns, decode_completed_ns);
        ++metrics.decode_token_count;
        decode_execution.command_buffer_count += decode_step.execution.command_buffer_count;
        decode_execution.commit_count += decode_step.execution.commit_count;
        decode_execution.synchronous_wait_count += decode_step.execution.synchronous_wait_count;
        decode_execution.compute_encoder_count += decode_step.execution.compute_encoder_count;
        decode_execution.compute_dispatch_count += decode_step.execution.compute_dispatch_count;
        decode_execution.blit_encoder_count += decode_step.execution.blit_encoder_count;
        decode_execution.memory_barrier_count += decode_step.execution.memory_barrier_count;
        decode_execution.transient_buffer_allocation_count +=
            decode_step.execution.transient_buffer_allocation_count;
        decode_execution.selector_dispatch_count += decode_step.execution.selector_dispatch_count;
        decode_execution.compact_readback_bytes += decode_step.execution.compact_readback_bytes;
        decode_execution.candidate_count = std::max(
            decode_execution.candidate_count, decode_step.execution.candidate_count
        );
        decode_execution.gpu_duration_ns += decode_step.execution.gpu_duration_ns;
        decode_execution.q4_decode_projection_count += decode_step.execution.q4_decode_projection_count;
        decode_execution.q8_decode_projection_count += decode_step.execution.q8_decode_projection_count;
        decode_execution.q4_small_batch_projection_count += decode_step.execution.q4_small_batch_projection_count;
        decode_execution.q4_exact_16_projection_count += decode_step.execution.q4_exact_16_projection_count;
        decode_execution.q4_generic_tiled_projection_count += decode_step.execution.q4_generic_tiled_projection_count;
        decode_execution.q6_decode_projection_count += decode_step.execution.q6_decode_projection_count;
        decode_execution.q6_small_batch_projection_count += decode_step.execution.q6_small_batch_projection_count;
        decode_execution.q6_exact_16_projection_count += decode_step.execution.q6_exact_16_projection_count;
        decode_execution.q6_generic_tiled_projection_count += decode_step.execution.q6_generic_tiled_projection_count;
        decode_execution.f16_projection_count += decode_step.execution.f16_projection_count;
        decode_execution.output_head_elision_count += decode_step.execution.output_head_elision_count;
        decode_execution.output_elided_weight_bytes += decode_step.execution.output_elided_weight_bytes;
        decode_execution.decode_attention_dispatch_count += decode_step.execution.decode_attention_dispatch_count;
        decode_execution.decode_attention_partition_count += decode_step.execution.decode_attention_partition_count;
        if (decode_step.execution.decode_attention_token_count
                > std::numeric_limits<std::uint64_t>::max()
                    - decode_execution.decode_attention_token_count) {
            throw std::overflow_error("Gemma2 decode attention token accounting overflow");
        }
        decode_execution.decode_attention_token_count += decode_step.execution.decode_attention_token_count;
        cached_token_count = decode.total_kv_token_count();
        if (decode_step.candidates.has_value()) {
            candidates = *decode_step.candidates;
            metrics.compact_readback_bytes += decode_step.execution.compact_readback_bytes;
            metrics.selector_dispatch_count += decode_step.execution.selector_dispatch_count;
            metrics.max_candidate_count = std::max(
                metrics.max_candidate_count, decode_step.execution.candidate_count
            );
        } else if (needs_next_candidate) {
            throw std::runtime_error("Gemma2 Metal decode omitted compact token candidates");
        }

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

    }

    metrics.generation_completed_ns = monotonic_now_ns(clock);
    metrics.generated_token_count = checked_prompt_token_count(
        std::span<const std::uint32_t>(generated_token_ids)
    );
    metrics.selected_token_count = checked_prompt_token_count(
        std::span<const std::uint32_t>(selected_token_ids)
    );

    return Gemma2MetalPlanGeneratedTokens {
        .generated_token_ids = std::move(generated_token_ids),
        .selected_token_ids = std::move(selected_token_ids),
        .stop_token_id = stop_token_id,
        .prefill_cache_position = cache_position,
        .prefill_token_count = prompt_token_count,
        .cached_token_count = cached_token_count,
        .metrics = metrics,
        .prefill_execution = prefill_result.execution,
        .decode_execution = decode_execution,
    };
}

}  // namespace

Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_sampled_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock
) {
    return generate_gemma2_metal_tokens_with_plan_impl(
        plan,
        prompt_token_ids,
        cache_position,
        max_new_tokens,
        stop_token_ids,
        sampler,
        on_token,
        clock
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
