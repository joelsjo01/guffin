#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"

#include "../tensor/backends/metal/metal_command_context.hpp"

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
    const Gemma2MetalTokenSelector& select_token,
    const Gemma2MetalGeneratedTokenCallback& on_token,
    const Gemma2MonotonicClock& clock,
    bool use_resident_top_logit
);

}  // namespace

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
    metal::MetalCommandContext command("Gemma2 resident decode", true);
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
    command.encode_memory_barrier();

    const auto head_dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    for (std::size_t offset = 0; offset < decode.layers().size(); ++offset) {
        const auto& scheduled_layer = decode.layers()[offset];
        const auto& block = plan.weights().blocks[scheduled_layer.layer_index];
        const auto& attention = block.attention;
        const auto& feed_forward = block.feed_forward;
        const auto& cache = plan.kv_cache()[scheduled_layer.layer_index];
        const auto key_row_bytes = static_cast<std::uint64_t>(config.attention_head_count_kv)
            * config.attention_key_length * sizeof(float);
        const auto value_row_bytes = static_cast<std::uint64_t>(config.attention_head_count_kv)
            * config.attention_value_length * sizeof(float);
        const auto key_cache_offset = static_cast<std::uint64_t>(decode.cache_position()) * key_row_bytes;
        const auto value_cache_offset = static_cast<std::uint64_t>(decode.cache_position()) * value_row_bytes;

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
            cache.value,
            0,
            value_cache_offset
        );
        command.encode_memory_barrier();
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
            cache.key,
            decode.cache_position(),
            config.attention_head_count_kv,
            head_dimension,
            head_dimension,
            gemma2_rope_frequency_base,
            0,
            key_cache_offset
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
            config.attention_logit_softcap
        );
        command.encode_memory_barrier();
        command.encode_matmul(
            scratch.attention_output,
            attention.output_weight,
            scratch.projected_attention_output
        );
        command.encode_memory_barrier();
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
    }

    command.encode_rms_norm(
        scratch.feed_forward_residual,
        plan.weights().output_norm_weight,
        scratch.normalized_hidden,
        config.rms_norm_epsilon
    );
    command.encode_memory_barrier();
    command.encode_matmul(scratch.normalized_hidden, plan.weights().output_weight, scratch.logits);
    command.encode_memory_barrier();
    command.encode_softcap(scratch.logits, scratch.logits, config.final_logit_softcap);
    if (select_top_logit) {
        command.encode_memory_barrier();
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
            .compute_dispatch_count = metal_stats.compute_dispatch_count,
            .blit_encoder_count = metal_stats.blit_encoder_count,
            .memory_barrier_count = metal_stats.memory_barrier_count,
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
        execution.compute_dispatch_count += stats.compute_dispatch_count;
        execution.blit_encoder_count += stats.blit_encoder_count;
        execution.memory_barrier_count += stats.memory_barrier_count;
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
