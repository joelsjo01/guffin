#include "gemma_target_runtime.hpp"
#include "gemma2_verification_state_access.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace local_gguf_engine::engine {
namespace metal = tensor::backends::metal;

void encode_gemma_target_blocks(
    metal::MetalCommandContext& encoder, Gemma2MetalModelPlan& target,
    const Gemma2MetalTargetVerifierScratch& scratch,
    std::uint32_t row_capacity, std::uint32_t row_count,
    const tensor::BackendBuffer& token_source, std::uint32_t token_offset,
    float embedding_scale, bool produce_final_hidden,
    std::optional<std::uint32_t> contiguous_prompt_position
) {
    const auto& config = target.config();
    if (contiguous_prompt_position
        && (row_count == 0U || *contiguous_prompt_position > config.context_length
            || row_count > config.context_length - *contiguous_prompt_position)) {
        throw std::invalid_argument("Gemma contiguous prompt block exceeds the bound target context");
    }
    auto* command = &encoder;
    command->encode_embedding(
        token_source,
        target.weights().token_embedding_weight,
        scratch.feed_forward_residual,
        embedding_scale,
        token_offset,
        row_count
    );
    command->encode_memory_barrier();

    const bool prefill_projection = contiguous_prompt_position.has_value();
    // Small query batches retain the measured vector path; 64+ useful rows
    // amortize the shared matrix tile, including its partially filled key tiles.
    const bool bulk_attention = prefill_projection && row_count >= 64U
        && config.architecture == ModelArchitecture::gemma3
        && gemma_kv_dtype(config) == tensor::DType::f16;
    const auto* kv_numerical_status = config.architecture == ModelArchitecture::gemma3
        && gemma_kv_dtype(config) == tensor::DType::f16 ? &scratch.input_status : nullptr;
    const auto* reusable_sums = !prefill_projection && row_count <= 2U && scratch.q4_input_sums.has_value()
        ? &*scratch.q4_input_sums : nullptr;
    const auto sums_for = [&](const tensor::BackendBuffer& weight) {
        return weight.descriptor().dtype == tensor::DType::q4_k
            && weight.descriptor().packing == tensor::TensorPacking::kquant_columns8
            ? reusable_sums : nullptr;
    };
    const auto attention_scale = 1.0F
        / std::sqrt(static_cast<float>(config.attention_key_length));
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto& block = target.weights().blocks[layer];
        const auto& cache = target.kv_cache()[layer];
        const auto& private_cache = scratch.kv_layers[layer];
        const bool terminal_kv_only = prefill_projection
            && config.architecture == ModelArchitecture::gemma3
            && !produce_final_hidden && layer + 1U == config.block_count;
        const bool qkv_sums = sums_for(block.attention.query_weight)
            || sums_for(block.attention.key_weight) || sums_for(block.attention.value_weight);
        const bool feed_forward_sums = sums_for(block.feed_forward.gate_weight)
            || sums_for(block.feed_forward.up_weight);
        command->encode_rms_norm(
            scratch.feed_forward_residual,
            block.attention.attention_norm_weight,
            scratch.normalized_hidden,
            config.rms_norm_epsilon,
            row_count, 0U, qkv_sums ? reusable_sums : nullptr
        );
        command->encode_memory_barrier();
        if (!terminal_kv_only) {
            command->encode_verifier_matmul(
                scratch.normalized_hidden, block.attention.query_weight,
                scratch.raw_query, row_count, prefill_projection, sums_for(block.attention.query_weight)
            );
        }
        command->encode_verifier_matmul(
            scratch.normalized_hidden, block.attention.key_weight,
            scratch.raw_key, row_count, prefill_projection, sums_for(block.attention.key_weight)
        );
        command->encode_verifier_matmul(
            scratch.normalized_hidden, block.attention.value_weight,
            scratch.raw_value, row_count, prefill_projection, sums_for(block.attention.value_weight)
        );
        command->encode_memory_barrier();
        if (config.uses_qk_norm) {
            if (!terminal_kv_only) {
                command->encode_head_rms_norm(
                    scratch.raw_query, block.attention.query_norm_weight.value(),
                    scratch.raw_query, config.rms_norm_epsilon, row_count,
                    config.attention_head_count, config.attention_key_length
                );
            }
            command->encode_head_rms_norm(
                scratch.raw_key, block.attention.key_norm_weight.value(),
                scratch.raw_key, config.rms_norm_epsilon, row_count,
                config.attention_head_count_kv, config.attention_key_length
            );
            command->encode_memory_barrier();
        }
        if (!terminal_kv_only) {
            command->encode_verifier_rope(
                scratch.raw_query, scratch.absolute_position_staging, scratch.query,
                row_count, config.attention_head_count,
                config.attention_key_length, config.attention_key_length,
                gemma_layer_rope_freq_base(config, layer),
                gemma_layer_rope_freq_scale(config, layer)
            );
        }
        command->encode_verifier_kv_store(
            scratch.raw_key, scratch.raw_value, scratch.absolute_position_staging,
            private_cache.private_key, private_cache.private_value,
            row_count, config.attention_head_count_kv,
            config.attention_key_length, config.attention_key_length,
            gemma_layer_rope_freq_base(config, layer),
            gemma_layer_rope_freq_scale(config, layer), kv_numerical_status
        );
        command->encode_memory_barrier();
        // Future attention reads final-layer K/V formed from this block's
        // input, independent of its Q/attention/MLP outputs. A non-final prompt
        // chunk has no logits and no later block consuming those outputs.
        if (terminal_kv_only) continue;
        if (bulk_attention) {
            const auto position = *contiguous_prompt_position;
            const auto end = position + row_count;
            const auto attention_tokens = gemma_layer_uses_sliding_window(config, layer)
                ? std::min(end, config.sliding_window) : end;
            // Private rows start at slot zero even for an unaligned follow-up.
            // Keep the durable ring unchanged until every query has consumed
            // its own causal window; publication remains after all blocks.
            command->encode_f16_prefill_attention(
                scratch.query, cache.key, cache.value,
                private_cache.private_key, private_cache.private_value,
                scratch.attention_output, position,
                config.attention_head_count, config.attention_head_count_kv,
                config.attention_key_length, config.attention_value_length,
                attention_tokens, attention_scale, config.attention_logit_softcap,
                cache.row_mask, row_count, true, position
            );
        } else {
            const auto schedule_offset = static_cast<std::uint64_t>(layer)
                * row_capacity * 4U * sizeof(std::uint32_t);
            command->encode_verifier_attention(
                scratch.query, cache.key, cache.value,
                private_cache.private_key, private_cache.private_value,
                scratch.attention_schedule_staging, schedule_offset,
                scratch.ancestor_index_staging,
                scratch.attention_output, scratch.attention_partial_output,
                scratch.attention_partial_maxima, scratch.attention_partial_denominators,
                row_count, cache.row_mask,
                config.attention_head_count, config.attention_head_count_kv,
                config.attention_key_length, config.attention_value_length,
                attention_scale, config.attention_logit_softcap, sums_for(block.attention.output_weight)
            );
        }
        command->encode_memory_barrier();
        command->encode_verifier_matmul(
            scratch.attention_output, block.attention.output_weight,
            scratch.projected_attention_output, row_count, prefill_projection, sums_for(block.attention.output_weight)
        );
        command->encode_memory_barrier();
        command->encode_rms_norm_add(
            scratch.projected_attention_output,
            block.attention.post_attention_norm_weight,
            scratch.feed_forward_residual,
            scratch.post_attention_residual,
            config.rms_norm_epsilon,
            row_count
        );
        command->encode_memory_barrier();
        command->encode_rms_norm(
            scratch.post_attention_residual,
            block.feed_forward.feed_forward_norm_weight,
            scratch.normalized_feed_forward_input,
            config.rms_norm_epsilon,
            row_count, 0U, feed_forward_sums ? reusable_sums : nullptr
        );
        command->encode_memory_barrier();
        command->encode_verifier_matmul(
            scratch.normalized_feed_forward_input, block.feed_forward.gate_weight,
            scratch.feed_forward_gate, row_count, prefill_projection, sums_for(block.feed_forward.gate_weight)
        );
        command->encode_verifier_matmul(
            scratch.normalized_feed_forward_input, block.feed_forward.up_weight,
            scratch.feed_forward_up, row_count, prefill_projection, sums_for(block.feed_forward.up_weight)
        );
        command->encode_memory_barrier();
        command->encode_gelu_tanh_multiply(
            scratch.feed_forward_gate, scratch.feed_forward_up,
            scratch.feed_forward_gated,
            static_cast<std::uint64_t>(row_count)
                * config.feed_forward_length, sums_for(block.feed_forward.down_weight)
        );
        command->encode_memory_barrier();
        command->encode_verifier_matmul(
            scratch.feed_forward_gated, block.feed_forward.down_weight,
            scratch.projected_feed_forward_output, row_count, prefill_projection, sums_for(block.feed_forward.down_weight)
        );
        command->encode_memory_barrier();
        command->encode_rms_norm_add(
            scratch.projected_feed_forward_output,
            block.feed_forward.post_feed_forward_norm_weight,
            scratch.post_attention_residual,
            scratch.feed_forward_residual,
            config.rms_norm_epsilon,
            row_count
        );
        command->encode_memory_barrier();
    }

    if (!produce_final_hidden) return;
    command->encode_rms_norm(
        scratch.feed_forward_residual, target.weights().output_norm_weight,
        scratch.normalized_hidden, config.rms_norm_epsilon,
        row_count
    );
    command->encode_memory_barrier();
}

namespace {

Gemma2MetalTargetVerifierPlan& ordinary_workspace(Gemma2MetalModelPlan& target) {
    auto& workspace = target.target_execution_workspace();
    if (!workspace) {
        const auto rows = target.config().architecture == ModelArchitecture::gemma3
            ? gemma2_metal_prefill_chunk_size : 2U;
        workspace = std::make_shared<Gemma2MetalTargetVerifierPlan>(
            create_gemma2_metal_target_verifier_plan(target, rows));
    }
    return *workspace;
}

// Private rows keep the old local ring intact until every query in the chunk
// has consumed its own causal window. Logical positions never use ring slots.
void stage_contiguous_rows(metal::MetalCommandContext& command,
    const Gemma2MetalModelPlan& target, const Gemma2MetalTargetVerifierPlan& workspace,
    std::span<const std::uint32_t> tokens, std::uint32_t position) {
    const auto count = static_cast<std::uint32_t>(tokens.size());
    const auto& scratch = workspace.scratch();
    if (gemma_kv_dtype(target.config()) == tensor::DType::f16)
        command.write_u32(scratch.input_status, 0U);
    std::array<std::uint32_t, gemma2_metal_prefill_chunk_size> positions {}, ancestors {};
    for (std::uint32_t row = 0; row < count; ++row) {
        positions[row] = position + row;
        ancestors[row] = row;
    }
    command.write_u32s(scratch.token_id_staging, 0U, tokens);
    command.write_u32s(scratch.absolute_position_staging, 0U,
        std::span<const std::uint32_t>(positions).first(count));
    command.write_u32s(scratch.ancestor_index_staging, 0U,
        std::span<const std::uint32_t>(ancestors).first(count));
    command.write_u32s(scratch.selected_path_staging, 0U,
        std::span<const std::uint32_t>(ancestors).first(count));
    for (std::uint32_t layer = 0; layer < target.config().block_count; ++layer) {
        const auto local = gemma_layer_uses_sliding_window(target.config(), layer);
        for (std::uint32_t row = 0; row < count; ++row) {
            const auto window_start = local && positions[row] >= target.config().sliding_window
                ? positions[row] + 1U - target.config().sliding_window : 0U;
            const auto base_start = std::min(position, window_start);
            const auto private_start = window_start > position ? window_start - position : 0U;
            const std::array<std::uint32_t, 4> schedule {
                base_start, position - base_start, private_start, row + 1U - private_start,
            };
            command.write_u32s(scratch.attention_schedule_staging,
                (layer * workspace.row_capacity() + row) * 4U, schedule);
        }
    }
}

void publish_contiguous_rows(metal::MetalCommandContext& command,
    Gemma2MetalModelPlan& target, const Gemma2MetalTargetVerifierScratch& scratch,
    std::uint32_t position, std::uint32_t count) {
    const auto width = target.config().attention_head_count_kv * target.config().attention_key_length;
    for (std::uint32_t layer = 0; layer < target.config().block_count; ++layer) {
        const auto& cache = target.kv_cache()[layer];
        const auto& staged = scratch.kv_layers[layer];
        command.encode_verifier_kv_scatter(staged.private_key, staged.private_value,
            scratch.selected_path_staging, cache.key, cache.value,
            position, count, width, cache.row_mask);
    }
}

template <class Stats>
void account_contiguous_attention(Stats& stats, const GemmaModelConfig& config,
    std::uint32_t position, std::uint32_t count, bool final_prompt_chunk = true) {
    // Count evaluated queries independently of partition/reduction launches.
    // Non-final Gemma 3 prompt chunks materialize terminal K/V without queries.
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        if (!final_prompt_chunk && config.architecture == ModelArchitecture::gemma3
            && layer + 1U == config.block_count) continue;
        for (std::uint32_t row = 0; row < count; ++row) {
            const auto tokens = gemma_layer_uses_sliding_window(config, layer)
                ? std::min(position + row + 1U, config.sliding_window) : position + row + 1U;
            ++stats.decode_attention_dispatch_count;
            stats.decode_attention_partition_count += metal::gemma2_decode_attention_partition_count(tokens);
            stats.decode_attention_token_count += tokens;
        }
    }
}

Gemma2MetalDecodeExecutionStats decode_stats(const metal::MetalCommandStats& stats) {
    Gemma2MetalDecodeExecutionStats output;
#define COPY(field) output.field = stats.field
    COPY(command_buffer_count); COPY(commit_count); COPY(synchronous_wait_count);
    COPY(compute_encoder_count); COPY(compute_dispatch_count); COPY(blit_encoder_count);
    COPY(memory_barrier_count); COPY(transient_buffer_allocation_count); COPY(gpu_duration_ns);
    COPY(q4_decode_projection_count); COPY(q8_decode_projection_count);
    COPY(q8_width_two_projection_count); COPY(q4_small_batch_projection_count);
    COPY(q4_width_two_projection_count); COPY(q4_exact_16_projection_count);
    COPY(q4_generic_tiled_projection_count); COPY(q6_decode_projection_count);
    COPY(q6_small_batch_projection_count); COPY(q6_width_two_projection_count);
    COPY(q6_exact_16_projection_count); COPY(q6_generic_tiled_projection_count);
    COPY(f16_projection_count); COPY(decode_attention_dispatch_count);
    COPY(decode_attention_partition_count); COPY(decode_attention_token_count);
#undef COPY
    return output;
}

void add_prefill_stats(Gemma2MetalPrefillExecutionStats& output,
    const metal::MetalCommandStats& stats) {
#define ADD(field) output.field += stats.field
    ADD(command_buffer_count); ADD(commit_count); ADD(synchronous_wait_count);
    ADD(compute_encoder_count); ADD(compute_dispatch_count); ADD(blit_encoder_count);
    ADD(memory_barrier_count); ADD(transient_buffer_allocation_count); ADD(gpu_duration_ns);
    ADD(q4_decode_projection_count); ADD(q8_decode_projection_count);
    ADD(q4_small_batch_projection_count); ADD(q4_exact_16_projection_count);
    ADD(q4_generic_tiled_projection_count); ADD(q6_decode_projection_count);
    ADD(q6_small_batch_projection_count); ADD(q6_exact_16_projection_count);
    ADD(q6_generic_tiled_projection_count); ADD(f16_projection_count);
    ADD(decode_attention_dispatch_count); ADD(decode_attention_partition_count);
    ADD(decode_attention_token_count);
#undef ADD
}

metal::MetalCommandStats execute_contiguous_chunk(
    Gemma2MetalModelPlan& target, Gemma2MetalTargetVerifierPlan& workspace,
    std::span<const std::uint32_t> tokens, std::uint32_t position, float embedding_scale,
    const tensor::BackendBuffer& logits, bool produce_logits,
    const std::optional<TokenSelectionRequest>& request,
    Gemma2ExecutionReceipt* completion, std::uint32_t& selector_dispatches,
    bool prompt_prefill = false) {
    if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0F)
        throw std::invalid_argument("Gemma target embedding scale must be finite and positive");
    if (position > target.config().context_length
        || tokens.empty() || tokens.size() > target.config().context_length - position)
        throw std::invalid_argument("Gemma 3 execution exceeds the bound target context");
    auto& command = target_workspace_command(workspace);
    const auto& scratch = workspace.scratch();
    const auto count = static_cast<std::uint32_t>(tokens.size());
    bool submitted = false;
    try {
        stage_contiguous_rows(command, target, workspace, tokens, position);
        encode_gemma_target_blocks(command, target, scratch, workspace.row_capacity(),
            count, scratch.token_id_staging, 0U, embedding_scale, produce_logits,
            prompt_prefill ? std::optional<std::uint32_t>(position) : std::nullopt);
        if (produce_logits) {
            command.encode_target_head_rows(scratch.normalized_hidden, count - 1U,
                target.weights().output_weight, logits, 1U);
            command.encode_memory_barrier();
            if (request) selector_dispatches = command.encode_token_candidates(
                logits, target.selection_workspace(), *request, target.config().final_logit_softcap);
        }
        publish_contiguous_rows(command, target, scratch, position, count);
        submitted = true;
        static_cast<void>(command.commit());
        if (completion) completion->completion_unknown = true;
        const auto stats = command.wait_until_completed();
        if (completion) completion->completion_unknown = false;
        if (target.config().architecture == ModelArchitecture::gemma3
            && gemma_kv_dtype(target.config()) == tensor::DType::f16
            && (command.read_verifier_status(scratch.input_status) & metal::verifier_kv_nonfinite_status)) {
            // Publication is in this command: a successful GPU wait does not
            // make a numerically invalid durable cache safe to reuse.
            Gemma2MetalVerificationStateAccess::poison(target);
            throw std::overflow_error("Gemma 3 F16 KV storage produced nonfinite values");
        }
        return stats;
    } catch (...) {
        if (submitted && (!completion || completion->completion_unknown))
            Gemma2MetalVerificationStateAccess::poison(target);
        throw;
    }
}

}  // namespace

Gemma2MetalResidentDecodeResult run_gemma3_resident_decode(
    Gemma2MetalModelPlan& target, std::uint32_t position, std::uint32_t token,
    float embedding_scale, std::optional<TokenSelectionRequest> request,
    Gemma2MetalDecodeOutputMode mode,
    Gemma2ExecutionReceipt* completion) {
    auto& workspace = ordinary_workspace(target);
    const auto& logits = target.decode_scratch().logits;
    std::uint32_t selector_dispatches = 0;
    const std::array<std::uint32_t, 1> tokens {token};
    const auto produced = mode == Gemma2MetalDecodeOutputMode::logits;
    const auto stats = execute_contiguous_chunk(target, workspace, tokens, position,
        embedding_scale, logits, produced, request, completion, selector_dispatches);
    auto result = Gemma2MetalResidentDecodeResult { .logits = logits, .produced_logits = produced };
    result.execution = decode_stats(stats);
    account_contiguous_attention(result.execution, target.config(), position, 1U);
    result.execution.selector_dispatch_count = selector_dispatches;
    if (request) {
        // The finished context must remain intact through compact readback.
        result.candidates = target_workspace_completed_candidates(workspace, target.selection_workspace());
        result.execution.compact_readback_bytes = result.candidates->readback_bytes;
        result.execution.candidate_count = result.candidates->count;
    }
    if (!produced) {
        result.execution.output_head_elision_count = 1U;
        result.execution.output_elided_weight_bytes = target.weights().output_weight.descriptor().byte_count
            + target.weights().output_norm_weight.descriptor().byte_count;
    }
    return result;
}

Gemma2MetalResidentPrefillResult run_gemma3_resident_prefill(
    Gemma2MetalModelPlan& target, std::span<const std::uint32_t> tokens, std::uint32_t position,
    float embedding_scale, std::optional<TokenSelectionRequest> request,
    Gemma2ExecutionReceipt* completion) {
    auto& workspace = ordinary_workspace(target);
    const auto& logits = target.prefill_scratch().logits;
    Gemma2MetalResidentPrefillResult result { .logits = logits };
    for (std::size_t offset = 0; offset < tokens.size();) {
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(
            workspace.row_capacity(), tokens.size() - offset));
        const auto last = offset + count == tokens.size();
        std::uint32_t selector_dispatches = 0;
        const auto stats = execute_contiguous_chunk(target, workspace, tokens.subspan(offset, count),
            position + static_cast<std::uint32_t>(offset), embedding_scale, logits, last,
            last ? request : std::nullopt, completion, selector_dispatches, true);
        add_prefill_stats(result.execution, stats);
        account_contiguous_attention(result.execution, target.config(),
            position + static_cast<std::uint32_t>(offset), count, last);
        ++result.execution.chunk_count;
        result.execution.scheduled_layer_execution_count += target.config().block_count;
        result.execution.kv_write_count += target.config().block_count;
        result.execution.staged_kv_copy_dispatch_count += target.config().block_count;
        result.execution.staged_kv_copy_logical_bytes += static_cast<std::uint64_t>(count)
            * target.config().block_count * target.config().attention_head_count_kv
            * target.config().attention_key_length * 2U
            * (gemma_kv_dtype(target.config()) == tensor::DType::f16 ? sizeof(std::uint16_t) : sizeof(float)) * 2U;
        result.execution.selector_dispatch_count += selector_dispatches;
        offset += count;
    }
    if (request) {
        result.candidates = target_workspace_completed_candidates(workspace, target.selection_workspace());
        result.execution.compact_readback_bytes = result.candidates->readback_bytes;
        result.execution.candidate_count = result.candidates->count;
    }
    return result;
}

Gemma2MetalVerifierResult run_gemma_target_width_two_verifier(
    Gemma2MetalModelPlan& target, std::uint32_t position,
    std::span<const std::uint32_t, 2> tokens, float embedding_scale,
    std::optional<std::array<TokenSelectionRequest, 2>> requests) {
    if (position > target.config().context_length
        || tokens.size() > target.config().context_length - position)
        throw std::invalid_argument("Gemma 3 verifier exceeds the bound target context");
    if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0F)
        throw std::invalid_argument("Gemma target embedding scale must be finite and positive");
    auto& workspace = ordinary_workspace(target);
    auto& command = target_workspace_command(workspace);
    const auto& scratch = workspace.scratch();
    const auto& logits = target.verifier_scratch().logits;
    stage_contiguous_rows(command, target, workspace, tokens, position);
    // These reads are independent of embedding. Its existing initial barrier
    // completes every backup before the verifier can publish any durable row.
    const auto& backups = target.verifier_scratch().kv_backups;
    for (std::size_t layer = 0; layer < target.kv_cache().size(); ++layer) {
        const auto& cache = target.kv_cache()[layer];
        const auto& backup = backups[layer];
        command.encode_gemma2_kv_cache_copy(cache.key, cache.value,
            backup.backup_key, backup.backup_value, position, 2U, cache.row_mask, 1U);
    }
    encode_gemma_target_blocks(command, target, scratch, workspace.row_capacity(),
        2U, scratch.token_id_staging, 0U, embedding_scale);
    command.encode_target_head_rows(scratch.normalized_hidden, 0U,
        target.weights().output_weight, logits, 2U);
    command.encode_memory_barrier();
    std::uint32_t selector_dispatches = 0;
    if (requests) selector_dispatches = command.encode_token_candidates_width_two(
        logits, target.selection_workspace(), *requests, target.config().final_logit_softcap);
    publish_contiguous_rows(command, target, scratch, position, 2U);
    static_cast<void>(command.commit());
    Gemma2MetalVerificationStateAccess::set_pending(target, position);
    bool completed = false;
    Gemma2MetalVerifierResult result { .logits = logits, .cache_position = position };
    try {
        const auto stats = command.wait_until_completed();
        Gemma2MetalVerificationStateAccess::mark_completed(target);
        completed = true;
        if (target.config().architecture == ModelArchitecture::gemma3
            && gemma_kv_dtype(target.config()) == tensor::DType::f16
            && (command.read_verifier_status(scratch.input_status) & metal::verifier_kv_nonfinite_status))
            throw std::overflow_error("Gemma 3 F16 KV storage produced nonfinite values");
        result.execution = decode_stats(stats);
        account_contiguous_attention(result.execution, target.config(), position, 2U);
        result.execution.selector_dispatch_count = selector_dispatches;
        if (requests) {
            result.candidates = command.read_token_candidates_width_two(target.selection_workspace());
            result.execution.compact_readback_bytes = result.candidates->at(0).readback_bytes
                + result.candidates->at(1).readback_bytes;
            result.execution.candidate_count = std::max(result.candidates->at(0).count,
                result.candidates->at(1).count);
        }
    } catch (...) {
        if (completed) {
            try { static_cast<void>(commit_gemma2_metal_width_two_verification(target, 0U)); }
            catch (...) { Gemma2MetalVerificationStateAccess::poison(target); }
        }
        throw;
    }
    return result;
}

}  // namespace local_gguf_engine::engine
