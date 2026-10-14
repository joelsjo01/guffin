#include "local_gguf_engine/frontends/cli/metrics.hpp"
#include "build_info.hpp"

#include "local_gguf_engine/engine/gemma_chat_session.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace local_gguf_engine::frontends::cli {
namespace {
using engine::model_architecture_name;

[[nodiscard]] const char* stop_reason_name(engine::Gemma2GenerationStopReason reason) {
    switch (reason) {
    case engine::Gemma2GenerationStopReason::token_limit: return "token_limit";
    case engine::Gemma2GenerationStopReason::stop_token: return "stop_token";
    case engine::Gemma2GenerationStopReason::callback_stop: return "callback_stop";
    case engine::Gemma2GenerationStopReason::reverse_stop: return "reverse_stop";
    case engine::Gemma2GenerationStopReason::interrupted: return "interrupted";
    }
    throw std::logic_error("invalid generation stop reason in diagnostics");
}

[[nodiscard]] std::string format_tokens_per_second(
    std::uint32_t token_count,
    std::uint64_t duration_ns
) {
    if (token_count == 0U || duration_ns == 0U) {
        return "n/a";
    }

    const auto rate = static_cast<double>(token_count) * 1.0e9
        / static_cast<double>(duration_ns);
    if (!std::isfinite(rate)) {
        return "n/a";
    }

    std::ostringstream formatted;
    formatted << std::fixed << std::setprecision(1) << rate << " t/s";
    return formatted.str();
}

[[nodiscard]] std::string format_seconds(std::uint64_t duration_ns) {
    std::ostringstream formatted;
    formatted << std::fixed << std::setprecision(6)
              << static_cast<double>(duration_ns) / 1.0e9 << " s";
    return formatted.str();
}

} // namespace

std::string format_chat_timing_line(const engine::GemmaChatTurnMetrics& metrics) {
    return "[ Prompt: "
        + format_tokens_per_second(
            metrics.generation.newly_evaluated_prompt_token_count,
            metrics.prompt_processing_duration_ns
        )
        + " | Generation: "
        + format_tokens_per_second(
            metrics.generation.selected_token_count,
            metrics.generation_duration_ns
        )
        + " | TTFT: "
        + (metrics.prompt_processing_duration_ns != 0U
            ? format_seconds(metrics.prompt_processing_duration_ns)
            : std::string("n/a"))
        + " ]";
}

std::string format_chat_timing_detail_line(const engine::GemmaChatTurnMetrics& metrics) {
    return "[ Detail: Evaluated: "
        + std::to_string(metrics.generation.newly_evaluated_prompt_token_count)
        + " | Cached: " + std::to_string(metrics.reused_cached_token_count)
        + " | Decoded: " + std::to_string(metrics.generation.selected_token_count)
        + " | Prompt: " + format_seconds(metrics.prompt_processing_duration_ns)
        + " | Generation: " + format_seconds(metrics.generation_duration_ns) + " ]";
}

std::string format_chat_decode_attribution_line(const engine::GemmaChatTurnMetrics& metrics) {
    constexpr std::uint32_t schema_version = 6U;
    if (metrics.standalone_draft != metrics.standalone_effective_sampler.has_value()) {
        throw std::logic_error("standalone diagnostics require the actual effective draft sampler");
    }
    if (metrics.standalone_effective_sampler) {
        const auto& sampler = *metrics.standalone_effective_sampler;
        if (!std::isfinite(sampler.temperature) || sampler.temperature < 0.0F
            || !std::isfinite(sampler.top_p) || sampler.top_p <= 0.0F || sampler.top_p > 1.0F
            || !std::isfinite(sampler.min_p) || sampler.min_p < 0.0F || sampler.min_p > 1.0F
            || !std::isfinite(sampler.repetition_penalty) || sampler.repetition_penalty <= 0.0F) {
            throw std::logic_error("standalone diagnostics contain invalid effective draft scalars");
        }
    }
    const auto decode_tokens = static_cast<std::uint64_t>(metrics.generation.decode_token_count);
    const auto checked_multiply = [](std::uint64_t left, std::uint64_t right, const char* name) {
        if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
            throw std::overflow_error(std::string("decode attribution overflow: ") + name);
        }
        return left * right;
    };
    const auto checked_add = [](std::uint64_t left, std::uint64_t right, const char* name) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            throw std::overflow_error(std::string("decode attribution overflow: ") + name);
        }
        return left + right;
    };
    const auto& bytes = metrics.decode_scheduled_weight_bytes;
    std::uint64_t byte_sum = 0U;
    for (const auto value : {
             bytes.embedding, bytes.attention_projections, bytes.feed_forward_gate_up,
             bytes.feed_forward_down, bytes.output_head, bytes.norms,
         }) {
        byte_sum = checked_add(byte_sum, value, "scheduled byte sum");
    }
    if (byte_sum != bytes.total) {
        throw std::logic_error("decode attribution scheduled byte categories do not sum to total");
    }
    const auto& execution = metrics.decode_execution;
    if (decode_tokens > 0U && (execution.command_buffer_count == 0U || execution.gpu_duration_ns == 0U)) {
        throw std::logic_error("decode attribution has decoded tokens without GPU execution");
    }
    if (execution.gpu_duration_ns > metrics.generation.decode_duration_ns) {
        throw std::logic_error("decode attribution GPU duration exceeds decode wall duration");
    }
    if (execution.output_head_elision_count > decode_tokens) {
        throw std::logic_error("decode attribution output-head elisions exceed decoded tokens");
    }
    const auto final_cache_position = checked_add(
        checked_add(
            metrics.initial_cache_position,
            metrics.generation.newly_evaluated_prompt_token_count,
            "prompt cache position"
        ),
        decode_tokens,
        "final cache position"
    );
    const auto full_scheduled_bytes = checked_multiply(bytes.total, decode_tokens, "scheduled bytes");
    const auto minimum_elided_bytes = checked_multiply(
        bytes.output_head, execution.output_head_elision_count, "elided output-head bytes"
    );
    if (execution.output_elided_weight_bytes < minimum_elided_bytes
        || execution.output_elided_weight_bytes > full_scheduled_bytes) {
        throw std::logic_error("decode attribution elided byte count is invalid");
    }
    const auto total_scheduled_bytes = full_scheduled_bytes - execution.output_elided_weight_bytes;
    const auto wall_residual_ns = metrics.generation.decode_duration_ns - execution.gpu_duration_ns;

    std::ostringstream token_hash;
    token_hash << std::hex << std::setw(16) << std::setfill('0')
               << metrics.generation.selected_token_hash;

    std::ostringstream json;
    json << "[ Decode: {\"schema_version\":" << schema_version
         << ",\"engine_build\":\"" << build_identity << "\""
         << ",\"model_architecture\":\"" << model_architecture_name(metrics.model_architecture) << "\""
         << ",\"retrieval_decode\":" << (metrics.retrieval_decode ? "true" : "false")
         << ",\"context_capacity\":" << metrics.context_capacity
         << ",\"kv_cache_dtype\":\""
         << engine::gemma_kv_precision_name(metrics.kv_precision) << "\""
         << ",\"selected_tokens\":" << metrics.generation.selected_token_count
         << ",\"stop_reason\":\"" << stop_reason_name(metrics.generation.stop_reason) << "\""
         << ",\"selected_token_hash\":\"" << token_hash.str() << "\""
         << ",\"decoded_tokens\":" << metrics.generation.decode_token_count
         << ",\"initial_cache_position\":" << metrics.initial_cache_position
         << ",\"final_cache_position\":" << final_cache_position
         << ",\"evaluated_prompt_tokens\":" << metrics.generation.newly_evaluated_prompt_token_count
         << ",\"reused_cache_tokens\":" << metrics.reused_cached_token_count
         << ",\"decode_wall_ns\":" << metrics.generation.decode_duration_ns
         << ",\"gpu_ns\":" << execution.gpu_duration_ns
         << ",\"selection_ns\":" << metrics.generation.selection_duration_ns
         << ",\"proposer_ns\":" << metrics.generation.proposer_duration_ns
         << ",\"generation_completion_ns\":" << metrics.generation_completion_duration_ns
         << ",\"full_turn_ns\":" << metrics.full_turn_duration_ns
         << ",\"first_text_ns\":" << metrics.first_text_duration_ns
         << ",\"learned_owned_bytes\":" << metrics.learned_owned_byte_count
         << ",\"learned_horizon\":" << metrics.learned_horizon
         << ",\"speculative_draft_model\":\""
         << (metrics.standalone_draft ? "standalone_gemma3_tli" : "none") << "\""
         << ",\"speculative_sampling\":\""
         << (metrics.speculative_sampling == engine::SpeculativeSamplingMode::lossless
             ? "lossless" : "exact-match") << "\""
         << ",\"learned_verification_schedule\":\""
         << (metrics.standalone_execution.conditional_verification ? "conditional" : "full") << "\""
         << ",\"learned_draft_head\":\""
         << (metrics.standalone_draft ? "q8_0" : "f16") << "\""
         << ",\"scheduled_bytes_basis\":\"one_row_equivalent_not_physical_bandwidth\""
         << ",\"learned_cycles\":" << metrics.generation.learned_cycle_count
         << ",\"learned_accepted_edges\":" << metrics.generation.learned_accepted_edge_count
         << ",\"learned_proposed_edges\":" << metrics.generation.learned_proposed_edge_count
         << ",\"learned_numeric_refusals\":" << metrics.generation.learned_numeric_refusal_count
         << ",\"learned_draft_ns\":" << metrics.generation.learned_draft_duration_ns
         << ",\"learned_draft_gpu_ns\":" << metrics.generation.learned_draft_gpu_ns
         << ",\"learned_target_gpu_ns\":" << metrics.generation.learned_target_gpu_ns
         << ",\"learned_cycle_ns\":" << metrics.generation.learned_cycle_duration_ns
         << ",\"standalone_prefill_ns\":" << metrics.generation.standalone_prefill_ns
         << ",\"standalone_rebuild_ns\":" << metrics.generation.standalone_rebuild_ns
         << ",\"standalone_draft_uploaded_bytes\":" << metrics.generation.standalone_draft_uploaded_bytes
         << ",\"standalone_draft_readback_bytes\":" << metrics.generation.standalone_draft_readback_bytes
         << ",\"standalone_draft_wall_ns\":" << metrics.generation.standalone_draft_wall_ns
         << ",\"standalone_draft_gpu_ns\":" << metrics.generation.standalone_draft_gpu_ns
         << ",\"standalone_draft_command_buffer_count\":" << metrics.generation.standalone_draft_command_buffer_count
         << ",\"standalone_draft_compute_dispatch_count\":" << metrics.generation.standalone_draft_compute_dispatch_count
         << ",\"standalone_draft_sampler\":\""
         << (metrics.standalone_draft
             ? (metrics.standalone_bounded_repetition_history
                 ? "explicit_or_target_scalars_target_window_v2"
                 : "explicit_or_target_scalars_native_history_v2")
             : "none") << "\""
         << ",\"standalone_repetition_window\":";
    if (metrics.standalone_bounded_repetition_history && metrics.standalone_effective_sampler)
        json << metrics.standalone_effective_sampler->repetition_window;
    else json << "null";
    json << ",\"standalone_effective_sampler\":";
    if (metrics.standalone_effective_sampler) {
        const auto& sampler = *metrics.standalone_effective_sampler;
        // Emit round-trippable F32 values actually consumed by the draft, not
        // the original CLI strings or unchanged target configuration.
        const auto precision = json.precision();
        json << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "{\"temperature\":" << sampler.temperature
             << ",\"top_k\":" << sampler.top_k
             << ",\"top_p\":" << sampler.top_p
             << ",\"min_p\":" << sampler.min_p
             << ",\"repetition_penalty\":" << sampler.repetition_penalty << '}';
        json.precision(precision);
    } else json << "null";
    json << ",\"learned_horizon_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_horizon_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_horizon_histogram[index];
    }
    json << "],\"learned_reached_depth_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_reached_depth_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_reached_depth_histogram[index];
    }
    json << "],\"learned_accepted_prefix_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_accepted_prefix_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_accepted_prefix_histogram[index];
    }
    json << "],\"learned_rejection_depth_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_rejection_depth_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_rejection_depth_histogram[index];
    }
    json << ']'
         << ",\"learned_censored_cycle_count\":" << metrics.generation.learned_censored_cycle_count
         << ",\"learned_full_accept_count\":" << metrics.generation.learned_full_accept_count
         << ",\"learned_horizon_limited_cycle_count\":" << metrics.generation.learned_horizon_limited_cycle_count
         << ",\"speculative_cycles\":" << metrics.generation.speculative_cycle_count
         << ",\"speculative_matches\":" << metrics.generation.speculative_second_token_match_count
         << ",\"speculative_misses\":" << metrics.generation.speculative_second_token_miss_count
         << ",\"speculative_fallbacks\":" << metrics.generation.speculative_fallback_count
         << ",\"speculative_rollbacks\":" << metrics.generation.speculative_rollback_count
         << ",\"proposal_buckets\":[";
    bool first_bucket = true;
    constexpr std::array<const char*, 4> length_labels {"2", "3-4", "5-8", "9-16"};
    constexpr std::array<const char*, 3> support_labels {"2", "3", "4+"};
    constexpr std::array<const char*, 3> confidence_labels {"0.5-2/3", "2/3-1", "1"};
    for (std::size_t length = 0; length < length_labels.size(); ++length) {
        for (std::size_t support = 0; support < support_labels.size(); ++support) {
            for (std::size_t confidence = 0; confidence < confidence_labels.size(); ++confidence) {
                const auto index = (length * support_labels.size() + support)
                    * confidence_labels.size() + confidence;
                const auto& bucket = metrics.generation.speculative_proposal_buckets[index];
                if (bucket.cycle_count == 0U) continue;
                if (!first_bucket) json << ',';
                first_bucket = false;
                json << "{\"match_length\":\"" << length_labels[length]
                     << "\",\"support\":\"" << support_labels[support]
                     << "\",\"confidence\":\"" << confidence_labels[confidence]
                     << "\",\"cycles\":" << bucket.cycle_count
                     << ",\"matches\":" << bucket.match_count << '}';
            }
        }
    }
    json << ']'
         << ",\"wall_minus_gpu_ns\":" << wall_residual_ns
         << ",\"resident_weights_bytes\":" << metrics.resident_weight_byte_count
         << ",\"output_head_policy\":\""
         << engine::gemma_output_head_policy_name(metrics.output_head_policy) << "\""
         << ",\"resident_kv_bytes\":" << metrics.resident_kv_cache_byte_count
         << ",\"resident_scratch_bytes\":" << metrics.resident_scratch_byte_count
         << ",\"warmup_kv_clear_passes\":" << metrics.warmup_kv_clear_pass_count
         << ",\"warmup_host_cleared_kv_bytes\":" << metrics.warmup_host_cleared_kv_bytes
         << ",\"command_buffers\":" << execution.command_buffer_count
         << ",\"commits\":" << execution.commit_count
         << ",\"waits\":" << execution.synchronous_wait_count
         << ",\"compute_encoders\":" << execution.compute_encoder_count
         << ",\"compute_dispatches\":" << execution.compute_dispatch_count
         << ",\"blit_encoders\":" << execution.blit_encoder_count
         << ",\"barriers\":" << execution.memory_barrier_count
         << ",\"transient_allocations\":" << execution.transient_buffer_allocation_count
         << ",\"selector_dispatches\":" << execution.selector_dispatch_count
         << ",\"max_candidates\":" << execution.candidate_count
         << ",\"compact_readback_bytes\":" << execution.compact_readback_bytes
         << ",\"q4_decode\":" << execution.q4_decode_projection_count
         << ",\"q8_decode\":" << execution.q8_decode_projection_count
         << ",\"q8_width_two\":" << execution.q8_width_two_projection_count
         << ",\"q4_small_batch\":" << execution.q4_small_batch_projection_count
         << ",\"q4_width_two\":" << execution.q4_width_two_projection_count
         << ",\"q4_exact_16\":" << execution.q4_exact_16_projection_count
         << ",\"q4_generic_tiled\":" << execution.q4_generic_tiled_projection_count
         << ",\"q6_decode\":" << execution.q6_decode_projection_count
         << ",\"q6_small_batch\":" << execution.q6_small_batch_projection_count
         << ",\"q6_width_two\":" << execution.q6_width_two_projection_count
         << ",\"q6_exact_16\":" << execution.q6_exact_16_projection_count
         << ",\"q6_generic_tiled\":" << execution.q6_generic_tiled_projection_count
         << ",\"f16\":" << execution.f16_projection_count
         << ",\"output_head_elisions\":" << execution.output_head_elision_count
         << ",\"output_elided_weight_bytes\":" << execution.output_elided_weight_bytes
         << ",\"attention_dispatches\":" << execution.decode_attention_dispatch_count
         << ",\"attention_partitions\":" << execution.decode_attention_partition_count
         << ",\"attention_tokens\":" << execution.decode_attention_token_count
         << ",\"fused_embedding_norm\":" << execution.fused_embedding_norm_count
         << ",\"direct_qkv_cache\":" << execution.direct_qkv_cache_count
         << ",\"dual_norm_bridge\":" << execution.dual_norm_bridge_count
         << ",\"fused_gate_up_gelu\":" << execution.fused_gate_up_gelu_count
         << ",\"scheduled_bytes_per_token\":{\"embedding\":" << bytes.embedding
         << ",\"attention_projections\":" << bytes.attention_projections
         << ",\"ffn_gate_up\":" << bytes.feed_forward_gate_up
         << ",\"ffn_down\":" << bytes.feed_forward_down
         << ",\"output_head\":" << bytes.output_head
         << ",\"norms\":" << bytes.norms
         << ",\"total\":" << bytes.total
         << "},\"scheduled_bytes_total\":" << total_scheduled_bytes << "} ]";
    return json.str();
}

} // namespace local_gguf_engine::frontends::cli
