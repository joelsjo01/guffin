#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

using Gemma2MonotonicClock = std::function<std::uint64_t()>;

enum class Gemma2GenerationStopReason {
    token_limit,
    stop_token,
    callback_stop,
    reverse_stop,
    interrupted,
};

struct Gemma2GenerationMetrics {
    std::uint64_t prefill_started_ns = 0;
    std::uint64_t prefill_completed_ns = 0;
    std::uint64_t first_token_selected_ns = 0;
    std::uint64_t last_token_selected_ns = 0;
    std::uint64_t decode_started_ns = 0;
    std::uint64_t decode_completed_ns = 0;
    std::uint64_t generation_completed_ns = 0;
    std::uint64_t prompt_evaluation_duration_ns = 0;
    std::uint64_t decode_duration_ns = 0;
    std::uint64_t runtime_first_token_selected_ns = 0;
    std::uint64_t selection_duration_ns = 0;
    std::uint64_t compact_readback_bytes = 0;
    std::uint32_t selector_dispatch_count = 0;
    std::uint32_t max_candidate_count = 0;
    std::uint32_t newly_evaluated_prompt_token_count = 0;
    std::uint32_t generated_token_count = 0;
    std::uint32_t selected_token_count = 0;
    std::uint32_t decode_token_count = 0;
    Gemma2GenerationStopReason stop_reason = Gemma2GenerationStopReason::token_limit;
};

struct Gemma2MetalDecodeExecutionStats {
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint32_t selector_dispatch_count = 0;
    std::uint32_t compact_readback_bytes = 0;
    std::uint32_t candidate_count = 0;
    std::uint64_t gpu_duration_ns = 0;
    std::uint32_t q4_decode_projection_count = 0;
    std::uint32_t q4_small_batch_projection_count = 0;
    std::uint32_t q4_exact_16_projection_count = 0;
    std::uint32_t q4_generic_tiled_projection_count = 0;
    std::uint32_t q6_decode_projection_count = 0;
    std::uint32_t q6_small_batch_projection_count = 0;
    std::uint32_t q6_exact_16_projection_count = 0;
    std::uint32_t q6_generic_tiled_projection_count = 0;
    std::uint32_t f16_projection_count = 0;
    std::uint32_t decode_attention_dispatch_count = 0;
    std::uint32_t decode_attention_partition_count = 0;
    // Schema-4 reserves the fused-family counters so baseline and future
    // candidates remain directly comparable. The retained legacy schedule
    // reports zero for all four.
    std::uint32_t fused_embedding_norm_count = 0;
    std::uint32_t direct_qkv_cache_count = 0;
    std::uint32_t dual_norm_bridge_count = 0;
    std::uint32_t fused_gate_up_gelu_count = 0;
};

struct Gemma2MetalResidentDecodeResult {
    tensor::BackendBuffer logits;
    std::optional<CompactTokenCandidates> candidates;
    Gemma2MetalDecodeExecutionStats execution;
};

struct Gemma2MetalPrefillExecutionStats {
    std::uint32_t chunk_count = 0;
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint32_t scheduled_layer_execution_count = 0;
    std::uint32_t kv_write_count = 0;
    std::uint32_t selector_dispatch_count = 0;
    std::uint32_t compact_readback_bytes = 0;
    std::uint32_t candidate_count = 0;
    std::uint64_t gpu_duration_ns = 0;
    std::uint32_t q4_decode_projection_count = 0;
    std::uint32_t q4_small_batch_projection_count = 0;
    std::uint32_t q4_exact_16_projection_count = 0;
    std::uint32_t q4_generic_tiled_projection_count = 0;
    std::uint32_t q6_decode_projection_count = 0;
    std::uint32_t q6_small_batch_projection_count = 0;
    std::uint32_t q6_exact_16_projection_count = 0;
    std::uint32_t q6_generic_tiled_projection_count = 0;
    std::uint32_t f16_projection_count = 0;
    std::uint32_t decode_attention_dispatch_count = 0;
    std::uint32_t decode_attention_partition_count = 0;
    std::uint32_t fused_embedding_norm_count = 0;
    std::uint32_t direct_qkv_cache_count = 0;
    std::uint32_t dual_norm_bridge_count = 0;
    std::uint32_t fused_gate_up_gelu_count = 0;
};

struct Gemma2MetalResidentPrefillResult {
    tensor::BackendBuffer logits;
    std::optional<CompactTokenCandidates> candidates;
    Gemma2MetalPrefillExecutionStats execution;
};

struct Gemma2MetalPlanGeneratedTokens {
    std::vector<std::uint32_t> generated_token_ids;
    // Includes every selected token, including a stop/control token that is
    // intentionally absent from generated_token_ids and visible output.
    std::vector<std::uint32_t> selected_token_ids;
    std::optional<std::uint32_t> stop_token_id;
    std::uint32_t prefill_cache_position = 0;
    std::uint32_t prefill_token_count = 0;
    std::uint32_t cached_token_count = 0;
    Gemma2GenerationMetrics metrics;
    Gemma2MetalPrefillExecutionStats prefill_execution;
    Gemma2MetalDecodeExecutionStats decode_execution;
};

enum class Gemma2MetalTokenStreamDecision {
    continue_generation,
    stop_generation,
};

struct Gemma2MetalGeneratedTokenEvent {
    std::uint32_t token_id = 0;
    std::size_t generated_token_count = 0;
    std::uint32_t cached_token_count = 0;
};

using Gemma2MetalGeneratedTokenCallback =
    std::function<Gemma2MetalTokenStreamDecision(const Gemma2MetalGeneratedTokenEvent&)>;
[[nodiscard]] Gemma2MetalResidentDecodeResult run_gemma2_metal_resident_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request = std::nullopt
);

[[nodiscard]] Gemma2MetalResidentPrefillResult run_gemma2_metal_resident_prefill(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request = std::nullopt
);

[[nodiscard]] std::vector<std::uint32_t> generate_gemma2_metal_argmax_tokens(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    std::optional<std::uint32_t> stop_token_id = std::nullopt
);

[[nodiscard]] Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_argmax_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    const Gemma2MetalGeneratedTokenCallback& on_token = {},
    const Gemma2MonotonicClock& clock = {}
);

[[nodiscard]] Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_sampled_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token = {},
    const Gemma2MonotonicClock& clock = {}
);

}  // namespace local_gguf_engine::engine
