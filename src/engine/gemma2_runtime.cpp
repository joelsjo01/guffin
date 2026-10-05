#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/standalone_chat.hpp"

#include "gemma2_verification_state_access.hpp"
#include "gemma_target_runtime.hpp"

#include "../tensor/backends/metal/metal_command_context.hpp"
#include "../tensor/backends/metal_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace local_gguf_engine::engine {

namespace {

[[nodiscard]] std::size_t proposal_bucket_index(const RetrievalProposal& proposal) noexcept {
    const auto length_bucket = proposal.match_length <= 2U ? 0U
        : proposal.match_length <= 4U ? 1U
        : proposal.match_length <= 8U ? 2U : 3U;
    const auto support_bucket = proposal.support_count <= 2U ? 0U
        : proposal.support_count == 3U ? 1U : 2U;
    const auto confidence_bucket = proposal.confidence < (2.0 / 3.0) ? 0U
        : proposal.confidence < 1.0 ? 1U : 2U;
    return (length_bucket * gemma2_proposal_support_bucket_count + support_bucket)
        * gemma2_proposal_confidence_bucket_count + confidence_bucket;
}

[[nodiscard]] std::uint64_t selected_token_sequence_hash(
    std::span<const std::uint32_t> token_ids
) noexcept {
    auto hash = std::uint64_t {14695981039346656037ULL};
    for (const auto token : token_ids) {
        for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
            hash ^= (token >> shift) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

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
    const GemmaModelConfig& config,
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
    const GemmaModelConfig& config,
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

void accumulate_decode_stats(
    Gemma2MetalDecodeExecutionStats& total,
    const Gemma2MetalDecodeExecutionStats& value
) {
    total.command_buffer_count += value.command_buffer_count;
    total.commit_count += value.commit_count;
    total.synchronous_wait_count += value.synchronous_wait_count;
    total.compute_encoder_count += value.compute_encoder_count;
    total.compute_dispatch_count += value.compute_dispatch_count;
    total.blit_encoder_count += value.blit_encoder_count;
    total.memory_barrier_count += value.memory_barrier_count;
    total.transient_buffer_allocation_count += value.transient_buffer_allocation_count;
    total.selector_dispatch_count += value.selector_dispatch_count;
    total.compact_readback_bytes += value.compact_readback_bytes;
    total.candidate_count = std::max(total.candidate_count, value.candidate_count);
    total.gpu_duration_ns += value.gpu_duration_ns;
    total.q4_decode_projection_count += value.q4_decode_projection_count;
    total.q8_decode_projection_count += value.q8_decode_projection_count;
    total.q8_width_two_projection_count += value.q8_width_two_projection_count;
    total.q4_small_batch_projection_count += value.q4_small_batch_projection_count;
    total.q4_width_two_projection_count += value.q4_width_two_projection_count;
    total.q4_exact_16_projection_count += value.q4_exact_16_projection_count;
    total.q4_generic_tiled_projection_count += value.q4_generic_tiled_projection_count;
    total.q6_decode_projection_count += value.q6_decode_projection_count;
    total.q6_small_batch_projection_count += value.q6_small_batch_projection_count;
    total.q6_width_two_projection_count += value.q6_width_two_projection_count;
    total.q6_exact_16_projection_count += value.q6_exact_16_projection_count;
    total.q6_generic_tiled_projection_count += value.q6_generic_tiled_projection_count;
    total.f16_projection_count += value.f16_projection_count;
    total.output_head_elision_count += value.output_head_elision_count;
    total.output_elided_weight_bytes += value.output_elided_weight_bytes;
    total.decode_attention_dispatch_count += value.decode_attention_dispatch_count;
    total.decode_attention_partition_count += value.decode_attention_partition_count;
    if (value.decode_attention_token_count
        > std::numeric_limits<std::uint64_t>::max() - total.decode_attention_token_count) {
        throw std::overflow_error("Gemma2 decode attention token accounting overflow");
    }
    total.decode_attention_token_count += value.decode_attention_token_count;
}

void accumulate_verifier_stats(Gemma2MetalDecodeExecutionStats& total,
                              const TargetVerifierExecutionStats& value) {
#define ADD(field) total.field += value.field
    ADD(command_buffer_count); ADD(commit_count); ADD(synchronous_wait_count);
    ADD(compute_encoder_count); ADD(compute_dispatch_count); ADD(blit_encoder_count);
    ADD(memory_barrier_count); ADD(transient_buffer_allocation_count);
    ADD(selector_dispatch_count); ADD(compact_readback_bytes); ADD(gpu_duration_ns);
    ADD(q4_decode_projection_count); ADD(q8_decode_projection_count);
    ADD(q8_width_two_projection_count); ADD(q4_small_batch_projection_count);
    ADD(q4_width_two_projection_count); ADD(q4_exact_16_projection_count);
    ADD(q4_generic_tiled_projection_count); ADD(q6_decode_projection_count);
    ADD(q6_small_batch_projection_count); ADD(q6_width_two_projection_count);
    ADD(q6_exact_16_projection_count); ADD(q6_generic_tiled_projection_count);
    ADD(f16_projection_count);
#undef ADD
    total.decode_attention_dispatch_count += value.attention_dispatch_count;
    total.decode_attention_partition_count += value.attention_partition_count;
    total.decode_attention_token_count += value.attention_token_count;
}

[[nodiscard]] Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_tokens_with_plan_impl(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock,
    RetrievalProposer* proposer,
    Gemma2GenerationCommitJournal* journal,
    StandaloneChatPlan* standalone,
    std::span<const std::uint32_t> standalone_context
);

}  // namespace

Gemma2MetalResidentDecodeResult run_gemma2_metal_resident_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request,
    Gemma2MetalDecodeOutputMode output_mode,
    Gemma2ExecutionReceipt* completion
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

    if (plan.config().architecture == ModelArchitecture::gemma3) {
        return run_gemma3_resident_decode(plan, decode.cache_position(), token_id,
            embedding_scale, selection_request, output_mode, completion);
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
        command.encode_matmul(
            scratch.normalized_hidden, plan.weights().output_weight, scratch.logits
        );
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

    static_cast<void>(command.commit());
    if (completion) completion->completion_unknown = true;
    const auto metal_stats = command.wait_until_completed();
    if (completion) completion->completion_unknown = false;
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
    float embedding_scale,
    std::optional<std::array<TokenSelectionRequest, 2>> selection_requests
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

    if (gemma_target_uses_shared_retrieval(plan)) {
        if (plan.config().sliding_window < 2U)
            throw std::invalid_argument("width-two verification requires two local cache positions");
        return run_gemma_target_width_two_verifier(plan, verifier.cache_position(), token_ids,
            embedding_scale, selection_requests);
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
        const auto& backup = scratch.kv_backups[scheduled_layer.layer_index];

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
            cache.key, cache.value, backup.backup_key, backup.backup_value,
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

    auto selector_dispatch_count = std::uint32_t {0};
    if (selection_requests.has_value()) {
        selector_dispatch_count = command.encode_token_candidates_width_two(
            scratch.logits,
            plan.selection_workspace(),
            *selection_requests,
            config.final_logit_softcap
        );
    }

    static_cast<void>(command.commit());
    Gemma2MetalVerificationStateAccess::set_pending(plan, verifier.cache_position());
    auto completed = false;
    metal::MetalCommandStats stats;
    auto selected_candidates = std::optional<std::array<CompactTokenCandidates, 2>> {};
    try {
        stats = command.wait_until_completed();
        Gemma2MetalVerificationStateAccess::mark_completed(plan);
        completed = true;
        if (selection_requests.has_value()) {
            selected_candidates = command.read_token_candidates_width_two(
                plan.selection_workspace()
            );
        }
    } catch (...) {
        if (completed) {
            try {
                (void)commit_gemma2_metal_width_two_verification(plan, 0U);
            } catch (...) {
                // Keep the transaction pending if restoration fails so every
                // later product operation fails closed on the uncertain K/V.
            }
        }
        // A wait failure also leaves the submitted transaction pending: the
        // command's final K/V state is not safe to consume or overwrite.
        throw;
    }
    return Gemma2MetalVerifierResult {
        .logits = scratch.logits,
        .candidates = selected_candidates,
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
            .selector_dispatch_count = selector_dispatch_count,
            .compact_readback_bytes = selected_candidates.has_value()
                ? selected_candidates->at(0).readback_bytes
                    + selected_candidates->at(1).readback_bytes
                : 0U,
            .candidate_count = selected_candidates.has_value()
                ? std::max(
                    selected_candidates->at(0).count,
                    selected_candidates->at(1).count
                ) : 0U,
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
    if (!plan.pending_verification_completed()) {
        throw std::logic_error("Gemma2 width-two commit requires completed verification");
    }
    if (plan.provisionally_committed_token_count() != 0U) {
        throw std::logic_error("Gemma2 width-two commit cannot replace a provisional commit");
    }
    if (accepted_token_count > 2U) {
        throw std::invalid_argument("Gemma2 width-two commit accepts at most two tokens");
    }
    Gemma2MetalDecodeExecutionStats execution;
    if (accepted_token_count < 2U) {
        namespace metal = tensor::backends::metal;
        metal::MetalCommandContext command("Gemma2 width-two KV suffix rollback", true,
            plan.config().architecture == ModelArchitecture::gemma3);
        const auto position = plan.pending_verification_position() + accepted_token_count;
        const auto rollback_count = 2U - accepted_token_count;
        const auto& scratch = plan.verifier_scratch();
        for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
            const auto& backup = scratch.kv_backups[layer];
            const auto& cache = plan.kv_cache()[layer];
            command.encode_gemma2_kv_cache_copy(
                backup.backup_key, backup.backup_value, cache.key, cache.value,
                position, rollback_count, 1U, cache.row_mask
            );
        }
        const auto stats = command.commit_and_wait();
        execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = stats.command_buffer_count,
            .commit_count = stats.commit_count,
            .synchronous_wait_count = stats.synchronous_wait_count,
            .compute_encoder_count = stats.compute_encoder_count,
            .compute_dispatch_count = stats.compute_dispatch_count,
            .blit_encoder_count = stats.blit_encoder_count,
            .memory_barrier_count = stats.memory_barrier_count,
            .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
            .gpu_duration_ns = stats.gpu_duration_ns,
        };
    }
    Gemma2MetalVerificationStateAccess::clear(plan);
    return Gemma2MetalVerifierCommitResult {
        .accepted_token_count = accepted_token_count,
        .execution = execution,
    };
}

Gemma2MetalVerifierCommitResult provisionally_commit_gemma2_metal_width_two_verification(
    Gemma2MetalModelPlan& plan,
    std::uint32_t accepted_token_count
) {
    if (!plan.pending_verification_completed()
        || plan.provisionally_committed_token_count() != 0U) {
        throw std::logic_error("Gemma2 provisional width-two commit requires an uncommitted verification");
    }
    if (accepted_token_count == 0U || accepted_token_count > 2U) {
        throw std::invalid_argument("Gemma2 provisional width-two commit accepts one or two tokens");
    }
    Gemma2MetalDecodeExecutionStats execution;
    if (accepted_token_count < 2U) {
        namespace metal = tensor::backends::metal;
        metal::MetalCommandContext command("Gemma2 provisional width-two KV suffix rollback", true,
            plan.config().architecture == ModelArchitecture::gemma3);
        const auto position = plan.pending_verification_position() + accepted_token_count;
        const auto& scratch = plan.verifier_scratch();
        for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
            const auto& backup = scratch.kv_backups[layer];
            const auto& cache = plan.kv_cache()[layer];
            command.encode_gemma2_kv_cache_copy(
                backup.backup_key, backup.backup_value, cache.key, cache.value,
                position, 2U - accepted_token_count, 1U, cache.row_mask
            );
        }
        const auto stats = command.commit_and_wait();
        execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = stats.command_buffer_count,
            .commit_count = stats.commit_count,
            .synchronous_wait_count = stats.synchronous_wait_count,
            .compute_encoder_count = stats.compute_encoder_count,
            .compute_dispatch_count = stats.compute_dispatch_count,
            .blit_encoder_count = stats.blit_encoder_count,
            .memory_barrier_count = stats.memory_barrier_count,
            .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
            .gpu_duration_ns = stats.gpu_duration_ns,
        };
    }
    Gemma2MetalVerificationStateAccess::set_provisional(plan, accepted_token_count);
    return Gemma2MetalVerifierCommitResult {
        .accepted_token_count = accepted_token_count,
        .execution = execution,
    };
}

Gemma2MetalVerifierCommitResult finalize_gemma2_metal_width_two_verification(
    Gemma2MetalModelPlan& plan,
    std::uint32_t retained_token_count
) {
    if (!plan.pending_verification_completed()
        || plan.provisionally_committed_token_count() == 0U) {
        throw std::logic_error("Gemma2 width-two finalize requires a provisional commit");
    }
    const auto provisional = plan.provisionally_committed_token_count();
    if (retained_token_count > provisional) {
        throw std::invalid_argument("Gemma2 width-two finalize cannot retain an uncommitted token");
    }
    Gemma2MetalDecodeExecutionStats execution;
    if (retained_token_count < provisional) {
        namespace metal = tensor::backends::metal;
        metal::MetalCommandContext command("Gemma2 provisional width-two KV rollback", true,
            plan.config().architecture == ModelArchitecture::gemma3);
        const auto position = plan.pending_verification_position() + retained_token_count;
        const auto rollback_count = provisional - retained_token_count;
        const auto& scratch = plan.verifier_scratch();
        for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
            const auto& cache = plan.kv_cache()[layer];
            const auto& backup = scratch.kv_backups[layer];
            command.encode_gemma2_kv_cache_copy(
                backup.backup_key, backup.backup_value, cache.key, cache.value,
                position, rollback_count, 1U, cache.row_mask
            );
        }
        const auto stats = command.commit_and_wait();
        execution = Gemma2MetalDecodeExecutionStats {
            .command_buffer_count = stats.command_buffer_count,
            .commit_count = stats.commit_count,
            .synchronous_wait_count = stats.synchronous_wait_count,
            .compute_encoder_count = stats.compute_encoder_count,
            .compute_dispatch_count = stats.compute_dispatch_count,
            .blit_encoder_count = stats.blit_encoder_count,
            .memory_barrier_count = stats.memory_barrier_count,
            .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
            .gpu_duration_ns = stats.gpu_duration_ns,
        };
    }
    Gemma2MetalVerificationStateAccess::clear(plan);
    return Gemma2MetalVerifierCommitResult {
        .accepted_token_count = retained_token_count,
        .execution = execution,
    };
}

void discard_gemma2_metal_width_two_verification(Gemma2MetalModelPlan& plan) {
    // The verifier leaves computed rows resident, so every discarded suffix
    // must be restored from the per-layer backups before clearing the state.
    if (!plan.has_pending_verification()) return;
    if (plan.provisionally_committed_token_count() == 0U) {
        (void)commit_gemma2_metal_width_two_verification(plan, 0U);
    } else {
        (void)finalize_gemma2_metal_width_two_verification(plan, 0U);
    }
}

Gemma2MetalResidentPrefillResult run_gemma2_metal_resident_prefill(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request,
    Gemma2ExecutionReceipt* completion
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

    if (plan.config().architecture == ModelArchitecture::gemma3) {
        return run_gemma3_resident_prefill(plan, prompt_token_ids, cache_position,
            embedding_scale, selection_request, completion);
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
    auto staged_kv_copy_dispatch_count = std::uint32_t {0U};
    auto staged_kv_copy_logical_bytes = std::uint64_t {0U};
    while (token_offset < prompt_token_ids.size()) {
        const auto chunk_cache_position =
            cache_position + static_cast<std::uint32_t>(token_offset);
        const auto chunk_token_count = static_cast<std::uint32_t>(std::min<std::size_t>(
            gemma2_metal_prefill_chunk_size,
            prompt_token_ids.size() - token_offset
        ));
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
            const auto wraps_sliding_ring = scheduled_layer.uses_sliding_window
                && chunk_cache_position + chunk_token_count > config.sliding_window;

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
            if (wraps_sliding_ring) {
                command.encode_gemma2_kv_cache_store(
                    scratch.raw_key,
                    scratch.value,
                    scratch.current_chunk_key,
                    scratch.current_chunk_value,
                    chunk_cache_position,
                    config.attention_head_count_kv,
                    head_dimension,
                    head_dimension,
                    gemma2_rope_frequency_base,
                    gemma2_metal_prefill_chunk_size - 1U,
                    chunk_token_count
                );
                command.encode_wrapped_prefill_attention_scores(
                    scratch.query,
                    cache.key,
                    scratch.current_chunk_key,
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
            } else {
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
            }
            command.encode_softmax(
                scratch.attention_scores,
                scratch.attention_probabilities,
                chunk_token_count * config.attention_head_count,
                scheduled_layer.attention_token_count
            );
            if (wraps_sliding_ring) {
                command.encode_wrapped_prefill_attention_values(
                    scratch.attention_probabilities,
                    cache.value,
                    scratch.current_chunk_value,
                    scratch.attention_output,
                    chunk_cache_position,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.attention_value_length,
                    scheduled_layer.attention_token_count,
                    cache.row_mask,
                    chunk_token_count
                );
                // Tiny test rings may be narrower than a prefill chunk. Only
                // its final live window can be copied without duplicate
                // physical destinations in one dispatch.
                const auto copy_token_count = std::min(
                    chunk_token_count, cache.row_capacity
                );
                const auto copy_position_start = chunk_cache_position
                    + chunk_token_count - copy_token_count;
                command.encode_gemma2_kv_cache_copy(
                    scratch.current_chunk_key,
                    scratch.current_chunk_value,
                    cache.key,
                    cache.value,
                    copy_position_start,
                    copy_token_count,
                    gemma2_metal_prefill_chunk_size - 1U,
                    cache.row_mask
                );
                ++staged_kv_copy_dispatch_count;
                staged_kv_copy_logical_bytes +=
                    static_cast<std::uint64_t>(copy_token_count)
                    * config.attention_head_count_kv
                    * (config.attention_key_length + config.attention_value_length)
                    * sizeof(std::uint16_t) * 2U;
            } else {
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
            }
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
        if (completion) completion->completion_unknown = true;
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
        .staged_kv_copy_dispatch_count = staged_kv_copy_dispatch_count,
        .staged_kv_copy_logical_bytes = staged_kv_copy_logical_bytes,
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

    if (completion) completion->completion_unknown = false;
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
    const auto embedding_scale = gemma_input_embedding_scale(plan.config().embedding_length);
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
    Gemma2MetalDecodeExecutionStats verifier_execution;
    if (plan.config().context_length >= token_ids.size() + 3U) {
        const auto guaranteed = sampler_copy.sample(std::span<const TokenLogit>(
            decode.candidates->values.data(), decode.candidates->count
        ));
        sampler_copy.observe_tokens(std::span<const std::uint32_t>(&guaranteed, 1U));
        auto hypothetical = sampler_copy;
        hypothetical.observe_tokens(std::span<const std::uint32_t>(&guaranteed, 1U));
        const std::array<TokenSelectionRequest, 2> requests {
            sampler_copy.selection_request(), hypothetical.selection_request(),
        };
        const std::array<std::uint32_t, 2> verifier_tokens {guaranteed, guaranteed};
        const auto verifier = run_gemma2_metal_width_two_verifier(
            plan,
            create_gemma2_metal_verifier_schedule(
                plan, static_cast<std::uint32_t>(token_ids.size()) + 1U
            ),
            std::span<const std::uint32_t, 2>(verifier_tokens),
            embedding_scale,
            requests
        );
        if (!verifier.candidates.has_value()) {
            discard_gemma2_metal_width_two_verification(plan);
            throw std::runtime_error("Gemma2 Metal warm-up verifier did not return token candidates");
        }
        verifier_execution = verifier.execution;
        discard_gemma2_metal_width_two_verification(plan);
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
        .verifier = verifier_execution,
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
        clock,
        nullptr,
        nullptr,
        nullptr,
        {}
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
    const Gemma2MonotonicClock& clock,
    RetrievalProposer* proposer,
    Gemma2GenerationCommitJournal* journal,
    StandaloneChatPlan* standalone,
    std::span<const std::uint32_t> standalone_context
) {
    require_gemma2_metal_plan_generation_inputs(
        plan.config(),
        prompt_token_ids,
        cache_position,
        max_new_tokens
    );
    if (standalone && (standalone_context.size() != cache_position + prompt_token_ids.size()
        || !std::equal(prompt_token_ids.begin(), prompt_token_ids.end(),
            standalone_context.begin() + cache_position)))
        throw std::invalid_argument("standalone draft context does not match target prefill");
    const auto draft_before = standalone ? standalone->execution_stats() : StandaloneDraftExecutionStats{};
    const auto rebuild_before = standalone ? standalone->rebuild_duration_ns() : 0U;
    // Publish the first ordinary target token before spending time on a draft
    // whose context is not needed to select that token. The deferred owner is
    // synchronized before returning or beginning any speculative cycle.
    bool draft_context_deferred = standalone && standalone->direct_ids();
    if (draft_context_deferred) standalone->validate_direct_context(standalone_context);

    const auto prompt_token_count = checked_prompt_token_count(prompt_token_ids);
    Gemma2GenerationMetrics metrics {
        .prefill_started_ns = monotonic_now_ns(clock),
        .newly_evaluated_prompt_token_count = prompt_token_count,
    };
    if (standalone && !draft_context_deferred) {
        const auto started = monotonic_now_ns(clock);
        // A fresh/rebased target context is a new prompt interpretation, not
        // a continuation of any coincidentally matching old generated prefix.
        // Native emission receipts are reused only with a retained target K/V
        // prefix. Regeneration of a first turn deterministically re-encodes its
        // original prompt; its saved checkpoint still owns failure recovery.
        standalone->synchronize_context(standalone_context, cache_position == 0);
        metrics.standalone_prefill_ns = elapsed_ns(started, monotonic_now_ns(clock));
        metrics.standalone_draft_wall_ns = metrics.standalone_prefill_ns;
    }
    const auto draft_after_prefill = standalone ? standalone->execution_stats() : StandaloneDraftExecutionStats{};
    const auto embedding_scale = gemma_input_embedding_scale(plan.config().embedding_length);
    auto prefill_result = run_gemma2_metal_resident_prefill(
        plan,
        prompt_token_ids,
        cache_position,
        embedding_scale,
        sampler.selection_request(),
        journal ? &journal->completion : nullptr
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
    // Reuse one fixed-capacity sampler copy for target preview. Copy assignment
    // preserves its reserved history storage, so speculative cycles allocate
    // no host memory.
    auto preview_sampler = sampler;

    std::optional<std::uint32_t> pending_selected_token;
    bool inside_standalone_cycle = false;
    const auto record_commit = [&](std::uint32_t token, std::uint32_t cached) {
        if (journal) journal->record(token, cached, sampler);
        if (standalone && !draft_context_deferred) {
            const auto started = monotonic_now_ns(clock);
            standalone->observe_committed(token, cached - 1U);
            if (!inside_standalone_cycle) {
                const auto completed = monotonic_now_ns(clock);
                const auto duration = elapsed_ns(started, completed);
                metrics.standalone_draft_wall_ns += duration;
                metrics.decode_duration_ns += duration;
                metrics.decode_completed_ns = completed;
            }
        }
    };
    const auto preflight_draft_commit = [&](std::span<const std::uint32_t> tokens, bool inside_decode) {
        if (!standalone || standalone->kind() != StandaloneDraftKind::gemma3_270m_q8) return;
        const auto started = monotonic_now_ns(clock);
        if (draft_context_deferred) standalone->validate_direct_context(standalone_context, tokens);
        else standalone->validate_committed_prefix(tokens);
        const auto completed = monotonic_now_ns(clock);
        const auto duration = elapsed_ns(started, completed);
        metrics.standalone_draft_wall_ns += duration;
        if (!inside_decode) metrics.decode_duration_ns += duration;
    };

    while (selected_token_ids.size() < max_new_tokens) {
        const bool using_pending_token = pending_selected_token.has_value();
        const auto selection_started_ns = monotonic_now_ns(clock);
        const auto selected_token_id = using_pending_token
            ? *pending_selected_token
            : sampler.sample(std::span<const TokenLogit>(candidates.values).first(candidates.count));
        pending_selected_token.reset();
        const auto selection_completed_ns = monotonic_now_ns(clock);
        if (!using_pending_token) {
            metrics.selection_duration_ns += elapsed_ns(selection_started_ns, selection_completed_ns);
        }
        metrics.last_token_selected_ns = selection_completed_ns;
        if (selected_token_id >= plan.config().vocab_size) {
            throw std::invalid_argument("Gemma2 Metal token selector returned token outside vocabulary");
        }
        preflight_draft_commit(std::span(&selected_token_id, 1), false);
        if (!selected_first_token) {
            selected_first_token = true;
            metrics.first_token_selected_ns = selection_completed_ns;
            metrics.runtime_first_token_selected_ns = elapsed_ns(
                metrics.prefill_started_ns, selection_completed_ns
            );
        }
        selected_token_ids.push_back(selected_token_id);
        if (!using_pending_token) {
            sampler.observe_tokens(std::span<const std::uint32_t>(&selected_token_id, 1U));
        }
        if (proposer != nullptr) {
            proposer->observe(selected_token_id);
        }

        const auto remaining = max_new_tokens - selected_token_ids.size();
        const bool verifier_can_fit = !draft_context_deferred && remaining > 0U
            && cached_token_count + 2U <= plan.config().context_length
            && !token_is_stop(selected_token_id, stop_token_ids);
        RetrievalProposal proposal;
        if (proposer != nullptr && verifier_can_fit) {
            const auto proposal_started_ns = monotonic_now_ns(clock);
            proposal = proposer->propose();
            const auto proposal_completed_ns = monotonic_now_ns(clock);
            metrics.proposer_duration_ns += elapsed_ns(proposal_started_ns, proposal_completed_ns);
        }
        if (proposer != nullptr && verifier_can_fit && retrieval_proposal_is_profitable(proposal)) {
            const std::array<TokenSelectionRequest, 2> requests {
                sampler.selection_request(),
                sampler.selection_request_after_observing(proposal.token_id),
            };
            const std::array<std::uint32_t, 2> verifier_tokens {
                selected_token_id, proposal.token_id,
            };
            const auto decode_started_ns = monotonic_now_ns(clock);
            if (metrics.decode_token_count == 0U) metrics.decode_started_ns = decode_started_ns;
            auto verified = run_gemma2_metal_width_two_verifier(
                plan,
                create_gemma2_metal_verifier_schedule(plan, cached_token_count),
                std::span<const std::uint32_t, 2>(verifier_tokens),
                embedding_scale,
                requests
            );
            if (!verified.candidates.has_value()) {
                discard_gemma2_metal_width_two_verification(plan);
                throw std::runtime_error("Gemma2 width-two verifier omitted compact candidates");
            }
            preview_sampler = sampler;
            const auto preview_started_ns = monotonic_now_ns(clock);
            const auto next_target_token = preview_sampler.sample(
                std::span<const TokenLogit>(verified.candidates->at(0).values)
                    .first(verified.candidates->at(0).count)
            );
            const auto preview_completed_ns = monotonic_now_ns(clock);
            metrics.selection_duration_ns += elapsed_ns(preview_started_ns, preview_completed_ns);
            metrics.last_token_selected_ns = preview_completed_ns;
            preview_sampler.observe_tokens(std::span<const std::uint32_t>(&next_target_token, 1U));
            const bool matched = next_target_token == proposal.token_id;
            if (matched) {
                try { preflight_draft_commit(verifier_tokens, true); }
                catch (...) { discard_gemma2_metal_width_two_verification(plan); throw; }
            }
            ++metrics.speculative_cycle_count;
            if (matched) ++metrics.speculative_second_token_match_count;
            else ++metrics.speculative_second_token_miss_count;
            auto& proposal_bucket = metrics.speculative_proposal_buckets[
                proposal_bucket_index(proposal)
            ];
            ++proposal_bucket.cycle_count;
            if (matched) ++proposal_bucket.match_count;
            accumulate_decode_stats(decode_execution, verified.execution);
            metrics.compact_readback_bytes += verified.execution.compact_readback_bytes;
            metrics.selector_dispatch_count += verified.execution.selector_dispatch_count;
            metrics.max_candidate_count = std::max(
                metrics.max_candidate_count, verified.execution.candidate_count
            );

            if (matched) {
                const auto provisional = provisionally_commit_gemma2_metal_width_two_verification(plan, 2U);
                accumulate_decode_stats(decode_execution, provisional.execution);
                const auto primary_decode_completed_ns = monotonic_now_ns(clock);
                metrics.decode_completed_ns = primary_decode_completed_ns;
                metrics.decode_duration_ns += elapsed_ns(
                    decode_started_ns, primary_decode_completed_ns
                );
                cached_token_count += 2U;
                generated_token_ids.push_back(selected_token_id);
                record_commit(selected_token_id, cached_token_count - 1U);
                bool stop_after_first = false;
                try {
                    if (on_token) {
                        stop_after_first = on_token(Gemma2MetalGeneratedTokenEvent {
                            .token_id = selected_token_id,
                            .generated_token_count = generated_token_ids.size(),
                            .cached_token_count = cached_token_count - 1U,
                        }) == Gemma2MetalTokenStreamDecision::stop_generation;
                    }
                } catch (...) {
                    const auto rollback = finalize_gemma2_metal_width_two_verification(plan, 1U);
                    accumulate_decode_stats(decode_execution, rollback.execution);
                    throw;
                }
                if (stop_after_first) {
                    const auto rollback_started_ns = monotonic_now_ns(clock);
                    const auto rollback = finalize_gemma2_metal_width_two_verification(plan, 1U);
                    const auto rollback_completed_ns = monotonic_now_ns(clock);
                    accumulate_decode_stats(decode_execution, rollback.execution);
                    metrics.decode_completed_ns = rollback_completed_ns;
                    metrics.decode_duration_ns += elapsed_ns(
                        rollback_started_ns, rollback_completed_ns
                    );
                    ++metrics.speculative_rollback_count;
                    --cached_token_count;
                    metrics.last_token_selected_ns = selection_completed_ns;
                    metrics.stop_reason = Gemma2GenerationStopReason::callback_stop;
                    ++metrics.decode_token_count;
                    break;
                }

                sampler = preview_sampler;
                selected_token_ids.push_back(next_target_token);
                proposer->observe(next_target_token);
                candidates = verified.candidates->at(1);
                record_commit(next_target_token, cached_token_count);
                if (token_is_stop(next_target_token, stop_token_ids)) {
                    stop_token_id = next_target_token;
                    metrics.stop_reason = Gemma2GenerationStopReason::stop_token;
                } else {
                    generated_token_ids.push_back(next_target_token);
                    try {
                        if (on_token && on_token(Gemma2MetalGeneratedTokenEvent {
                                .token_id = next_target_token,
                                .generated_token_count = generated_token_ids.size(),
                                .cached_token_count = cached_token_count,
                            }) == Gemma2MetalTokenStreamDecision::stop_generation) {
                            metrics.stop_reason = Gemma2GenerationStopReason::callback_stop;
                        }
                    } catch (...) {
                        const auto finalized = finalize_gemma2_metal_width_two_verification(plan, 2U);
                        accumulate_decode_stats(decode_execution, finalized.execution);
                        throw;
                    }
                }
                const auto finalized = finalize_gemma2_metal_width_two_verification(plan, 2U);
                accumulate_decode_stats(decode_execution, finalized.execution);
                metrics.decode_token_count += 2U;
                if (stop_token_id.has_value()
                    || metrics.stop_reason == Gemma2GenerationStopReason::callback_stop) break;
                continue;
            }

            const auto committed = commit_gemma2_metal_width_two_verification(plan, 1U);
            accumulate_decode_stats(decode_execution, committed.execution);
            ++cached_token_count;
            const auto decode_completed_ns = monotonic_now_ns(clock);
            metrics.decode_completed_ns = decode_completed_ns;
            metrics.decode_duration_ns += elapsed_ns(decode_started_ns, decode_completed_ns);
            ++metrics.decode_token_count;
            generated_token_ids.push_back(selected_token_id);
            record_commit(selected_token_id, cached_token_count);
            if (on_token && on_token(Gemma2MetalGeneratedTokenEvent {
                    .token_id = selected_token_id,
                    .generated_token_count = generated_token_ids.size(),
                    .cached_token_count = cached_token_count,
                }) == Gemma2MetalTokenStreamDecision::stop_generation) {
                metrics.last_token_selected_ns = selection_completed_ns;
                metrics.stop_reason = Gemma2GenerationStopReason::callback_stop;
                break;
            }
            sampler = preview_sampler;
            pending_selected_token = next_target_token;
            continue;
        }
        if (proposer != nullptr && verifier_can_fit) {
            ++metrics.speculative_fallback_count;
        }

        // The standalone draft runs after the frozen retrieval gate. The
        // controller owns the accepted prefix and any already-selected carry.
        if (standalone != nullptr && !draft_context_deferred && remaining > 0U
            && !token_is_stop(selected_token_id, stop_token_ids)
            && standalone->cycle_horizon(cached_token_count, remaining + 1U) != 0) {
            const auto begin = monotonic_now_ns(clock);
            if (metrics.decode_token_count == 0U) metrics.decode_started_ns = begin;
            const auto on_committed = [&](const ChainCommittedToken& token) {
                if (token.depth != 0U) {
                    selected_token_ids.push_back(token.token_id);
                    if (proposer != nullptr) proposer->observe(token.token_id);
                    metrics.last_token_selected_ns = monotonic_now_ns(clock);
                }
                record_commit(token.token_id, token.cached_token_count);
                if (token.stop_token) return Gemma2MetalTokenStreamDecision::continue_generation;
                generated_token_ids.push_back(token.token_id);
                return on_token ? on_token(Gemma2MetalGeneratedTokenEvent {
                    .token_id = token.token_id,
                    .generated_token_count = generated_token_ids.size(),
                    .cached_token_count = token.cached_token_count,
                }) : Gemma2MetalTokenStreamDecision::continue_generation;
            };
            inside_standalone_cycle = true;
            const auto cycle = standalone->run(plan, sampler, selected_token_id,
                cached_token_count, remaining + 1U, stop_token_ids, on_committed);
            inside_standalone_cycle = false;
            metrics.standalone_draft_wall_ns += cycle.draft.wall_duration_ns;
            const auto end = monotonic_now_ns(clock);
            metrics.decode_completed_ns = end;
            const auto duration = elapsed_ns(begin, end);
            metrics.learned_cycle_duration_ns += duration;
            metrics.decode_duration_ns += duration;
            // Draft GPU work is accumulated once from the complete ledger
            // below, including out-of-cycle committed-token catch-up.
            accumulate_verifier_stats(decode_execution, cycle.target);
            accumulate_decode_stats(decode_execution, cycle.endpoint);
            metrics.learned_draft_duration_ns += cycle.draft.wall_duration_ns;
            metrics.learned_draft_gpu_ns += cycle.draft.gpu_duration_ns;
            ++metrics.learned_cycle_count;
            metrics.learned_accepted_edge_count += cycle.accepted_edges;
            const auto horizon = cycle.proposal_count;
            if (cycle.accepted_edges > horizon || horizon > external_speculation_max_horizon
                || horizon == 0 || (cycle.rejected && cycle.accepted_edges == horizon))
                throw std::logic_error("invalid committed speculative acceptance receipt");
            metrics.learned_proposed_edge_count += horizon;
            ++metrics.learned_horizon_histogram[horizon];
            const auto reached = cycle.accepted_edges + (cycle.rejected ? 1U : 0U);
            for (std::uint32_t depth = 0; depth < reached; ++depth)
                ++metrics.learned_reached_depth_histogram[depth];
            ++metrics.learned_accepted_prefix_histogram[cycle.accepted_edges];
            if (cycle.rejected) ++metrics.learned_rejection_depth_histogram[cycle.accepted_edges];
            else if (cycle.accepted_edges == horizon) ++metrics.learned_full_accept_count;
            else ++metrics.learned_censored_cycle_count;
            if (cycle.has_next_candidates) ++metrics.learned_horizon_limited_cycle_count;
            metrics.learned_target_gpu_ns += cycle.target.gpu_duration_ns;
            metrics.compact_readback_bytes += cycle.target.compact_readback_bytes;
            metrics.selector_dispatch_count += cycle.target.selector_dispatch_count;
            metrics.decode_token_count += cycle.retained_rows;
            cached_token_count += cycle.retained_rows;
            if (cycle.carry) pending_selected_token = cycle.carry;
            if (cycle.has_next_candidates) candidates = cycle.next_candidates;
            if (cycle.stop_token) {
                stop_token_id = cycle.stop_token;
                metrics.stop_reason = Gemma2GenerationStopReason::stop_token;
                break;
            }
            if (cycle.callback_stopped) {
                metrics.stop_reason = Gemma2GenerationStopReason::callback_stop;
                break;
            }
            continue;
        }

        const auto decode = create_gemma2_metal_decode_schedule(plan, cached_token_count);
        const auto decode_started_ns = monotonic_now_ns(clock);
        if (metrics.decode_token_count == 0U) {
            metrics.decode_started_ns = decode_started_ns;
        }
        const auto needs_next_candidate = selected_token_ids.size() < max_new_tokens
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
                : Gemma2MetalDecodeOutputMode::kv_only,
            journal ? &journal->completion : nullptr
        );
        const auto decode_completed_ns = monotonic_now_ns(clock);
        metrics.decode_completed_ns = decode_completed_ns;
        metrics.decode_duration_ns += elapsed_ns(decode_started_ns, decode_completed_ns);
        ++metrics.decode_token_count;
        accumulate_decode_stats(decode_execution, decode_step.execution);
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

        record_commit(selected_token_id, cached_token_count);
        const auto synchronize_deferred_draft = [&] {
            if (!draft_context_deferred) return;
            const auto started = monotonic_now_ns(clock);
            standalone->synchronize_context(standalone_context, cache_position == 0);
            const auto synchronized = monotonic_now_ns(clock);
            standalone->observe_committed_pending(selected_token_id, cached_token_count - 1U);
            const auto completed = monotonic_now_ns(clock);
            metrics.standalone_prefill_ns += elapsed_ns(started, synchronized);
            const auto duration = elapsed_ns(started, completed);
            metrics.standalone_draft_wall_ns += duration;
            metrics.decode_duration_ns += duration;
            metrics.decode_completed_ns = completed;
            draft_context_deferred = false;
        };
        if (token_is_stop(selected_token_id, stop_token_ids)) {
            synchronize_deferred_draft();
            stop_token_id = selected_token_id;
            metrics.stop_reason = Gemma2GenerationStopReason::stop_token;
            break;
        }

        generated_token_ids.push_back(selected_token_id);
        auto decision = Gemma2MetalTokenStreamDecision::continue_generation;
        if (on_token) {
            try {
                decision = on_token(Gemma2MetalGeneratedTokenEvent {
                    .token_id = selected_token_id,
                    .generated_token_count = generated_token_ids.size(),
                    .cached_token_count = cached_token_count,
                });
            } catch (...) {
                // The target/journal already own this delivered prefix. Finish
                // draft recovery even when the callback throws, then preserve
                // that original exception. Draft failure marks its own owner
                // poisoned, so the session cannot silently reuse stale state.
                const auto original = std::current_exception();
                try { synchronize_deferred_draft(); } catch (...) {}
                std::rethrow_exception(original);
            }
        }
        synchronize_deferred_draft();
        if (decision == Gemma2MetalTokenStreamDecision::stop_generation) {
            metrics.stop_reason = Gemma2GenerationStopReason::callback_stop;
            break;
        }
        if (selected_token_ids.size() == max_new_tokens) {
            break;
        }

    }

    if (standalone) {
        const auto after = standalone->execution_stats();
        metrics.standalone_rebuild_ns = standalone->rebuild_duration_ns() - rebuild_before;
        metrics.standalone_draft_uploaded_bytes = after.uploaded_bytes - draft_before.uploaded_bytes;
        metrics.standalone_draft_readback_bytes = after.readback_bytes - draft_before.readback_bytes;
        metrics.standalone_draft_gpu_ns = after.gpu_duration_ns - draft_before.gpu_duration_ns;
        metrics.standalone_draft_command_buffer_count = after.command_buffer_count - draft_before.command_buffer_count;
        metrics.standalone_draft_compute_dispatch_count = after.compute_dispatch_count - draft_before.compute_dispatch_count;
        const auto commands = after.command_buffer_count - draft_after_prefill.command_buffer_count;
        decode_execution.command_buffer_count += commands;
        decode_execution.commit_count += commands;
        decode_execution.synchronous_wait_count += commands;
        decode_execution.compute_encoder_count += commands;
        decode_execution.compute_dispatch_count += after.compute_dispatch_count - draft_after_prefill.compute_dispatch_count;
        decode_execution.memory_barrier_count += after.memory_barrier_count - draft_after_prefill.memory_barrier_count;
        decode_execution.gpu_duration_ns += after.gpu_duration_ns - draft_after_prefill.gpu_duration_ns;
    }
    metrics.generation_completed_ns = monotonic_now_ns(clock);
    metrics.generated_token_count = checked_prompt_token_count(
        std::span<const std::uint32_t>(generated_token_ids)
    );
    metrics.selected_token_count = checked_prompt_token_count(
        std::span<const std::uint32_t>(selected_token_ids)
    );
    metrics.selected_token_hash = selected_token_sequence_hash(selected_token_ids);

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

void Gemma2GenerationCommitJournal::record(
    std::uint32_t token, std::uint32_t cached, const TokenSampler& authoritative
) {
    if (count >= token_storage.size() || (count && cached != cached_token_count + 1U)) {
        throw std::logic_error("generation commit journal capacity or cache discontinuity");
    }
    token_storage[count++] = token;
    cached_token_count = cached;
    sampler = authoritative.checkpoint();
}

Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_sampled_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock,
    RetrievalProposer* proposer,
    Gemma2GenerationCommitJournal* journal,
    StandaloneChatPlan* standalone,
    std::span<const std::uint32_t> standalone_context
) {
    return generate_gemma2_metal_tokens_with_plan_impl(
        plan,
        prompt_token_ids,
        cache_position,
        max_new_tokens,
        stop_token_ids,
        sampler,
        on_token,
        clock,
        proposer,
        journal,
        standalone,
        standalone_context
    );
}

std::vector<std::uint32_t> generate_gemma2_metal_argmax_tokens(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    std::optional<std::uint32_t> stop_token_id
) {
    require_gemma2_metal_generation_inputs(model.gemma_config(), prompt_token_ids, max_new_tokens);
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
