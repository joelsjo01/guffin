#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/retrieval_proposer.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <array>
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

inline constexpr std::size_t gemma2_proposal_match_length_bucket_count = 4U;
inline constexpr std::size_t gemma2_proposal_support_bucket_count = 3U;
inline constexpr std::size_t gemma2_proposal_confidence_bucket_count = 3U;
inline constexpr std::size_t gemma2_proposal_bucket_count =
    gemma2_proposal_match_length_bucket_count
    * gemma2_proposal_support_bucket_count
    * gemma2_proposal_confidence_bucket_count;

struct Gemma2ProposalBucketMetrics {
    std::uint32_t cycle_count = 0;
    std::uint32_t match_count = 0;
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
    std::uint32_t speculative_cycle_count = 0;
    std::uint32_t speculative_second_token_match_count = 0;
    std::uint32_t speculative_second_token_miss_count = 0;
    std::uint32_t speculative_fallback_count = 0;
    std::uint32_t speculative_rollback_count = 0;
    std::uint64_t proposer_duration_ns = 0;
    std::uint32_t learned_cycle_count = 0;
    std::uint32_t learned_accepted_edge_count = 0;
    // Durable accepted-prefix lengths, not eagerly computed previews. Depth
    // zero denotes rejection of the first proposal. Terminal truncations are
    // kept separate so they cannot be mistaken for failed draft predictions.
    std::array<std::uint32_t, external_speculation_max_horizon + 1> learned_accepted_prefix_histogram {};
    std::array<std::uint32_t, external_speculation_max_horizon> learned_rejection_depth_histogram {};
    std::uint32_t learned_censored_cycle_count = 0;
    std::uint32_t learned_full_accept_count = 0;
    std::uint32_t learned_horizon_limited_cycle_count = 0;
    std::uint32_t learned_numeric_refusal_count = 0;
    std::uint64_t learned_draft_duration_ns = 0;
    std::uint64_t learned_draft_gpu_ns = 0;
    std::uint64_t learned_target_gpu_ns = 0;
    std::uint64_t learned_cycle_duration_ns = 0;
    std::uint64_t standalone_prefill_ns = 0;
    std::uint64_t standalone_rebuild_ns = 0;
    std::uint64_t standalone_draft_uploaded_bytes = 0;
    std::uint64_t standalone_draft_readback_bytes = 0;
    // All standalone work, including prompt synchronization and ordinary or
    // retrieval catch-up outside speculative cycles. Cycle-only learned_*
    // fields above retain their existing meaning.
    std::uint64_t standalone_draft_wall_ns = 0;
    std::uint64_t standalone_draft_gpu_ns = 0;
    std::uint32_t standalone_draft_command_buffer_count = 0;
    std::uint32_t standalone_draft_compute_dispatch_count = 0;
    // FNV-1a over each selected uint32 token in little-endian byte order.
    // This is diagnostic evidence, not a cryptographic identity.
    std::uint64_t selected_token_hash = 0;
    std::array<Gemma2ProposalBucketMetrics, gemma2_proposal_bucket_count>
        speculative_proposal_buckets {};
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
    std::uint32_t q8_decode_projection_count = 0;
    std::uint32_t q8_width_two_projection_count = 0;
    std::uint32_t q4_small_batch_projection_count = 0;
    std::uint32_t q4_width_two_projection_count = 0;
    std::uint32_t q4_exact_16_projection_count = 0;
    std::uint32_t q4_generic_tiled_projection_count = 0;
    std::uint32_t q6_decode_projection_count = 0;
    std::uint32_t q6_small_batch_projection_count = 0;
    std::uint32_t q6_width_two_projection_count = 0;
    std::uint32_t q6_exact_16_projection_count = 0;
    std::uint32_t q6_generic_tiled_projection_count = 0;
    std::uint32_t f16_projection_count = 0;
    std::uint32_t output_head_elision_count = 0;
    std::uint64_t output_elided_weight_bytes = 0;
    std::uint32_t decode_attention_dispatch_count = 0;
    std::uint32_t decode_attention_partition_count = 0;
    std::uint64_t decode_attention_token_count = 0;
    // Schema-4 reserves the fused-family counters so baseline and future
    // candidates remain directly comparable. The retained legacy schedule
    // reports zero for all four.
    std::uint32_t fused_embedding_norm_count = 0;
    std::uint32_t direct_qkv_cache_count = 0;
    std::uint32_t dual_norm_bridge_count = 0;
    std::uint32_t fused_gate_up_gelu_count = 0;
};

enum class Gemma2MetalDecodeOutputMode {
    logits,
    kv_only,
};

struct Gemma2MetalResidentDecodeResult {
    tensor::BackendBuffer logits;
    bool produced_logits = true;
    std::optional<CompactTokenCandidates> candidates;
    Gemma2MetalDecodeExecutionStats execution;
};

struct Gemma2MetalVerifierResult {
    tensor::BackendBuffer logits;
    std::optional<std::array<CompactTokenCandidates, 2>> candidates;
    std::uint32_t cache_position = 0;
    Gemma2MetalDecodeExecutionStats execution;
};

struct Gemma2MetalVerifierCommitResult {
    std::uint32_t accepted_token_count = 0;
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
    // Wrapped sliding layers add one device copy after attention. Logical
    // bytes count both the F16 stage read and resident-ring write.
    std::uint32_t staged_kv_copy_dispatch_count = 0;
    std::uint64_t staged_kv_copy_logical_bytes = 0;
    std::uint32_t selector_dispatch_count = 0;
    std::uint32_t compact_readback_bytes = 0;
    std::uint32_t candidate_count = 0;
    std::uint64_t gpu_duration_ns = 0;
    std::uint32_t q4_decode_projection_count = 0;
    std::uint32_t q8_decode_projection_count = 0;
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
    std::uint64_t decode_attention_token_count = 0;
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

struct Gemma2MetalWarmupStats {
    std::uint64_t wall_duration_ns = 0;
    std::uint32_t prefill_token_count = 0;
    std::uint32_t kv_clear_pass_count = 0;
    std::size_t host_cleared_kv_bytes = 0;
    Gemma2MetalPrefillExecutionStats prefill;
    Gemma2MetalDecodeExecutionStats decode;
    Gemma2MetalDecodeExecutionStats verifier;
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

struct Gemma2TargetSelectionTraceEvent {
    std::uint32_t target_position = 0U;
    std::uint32_t input_token_id = 0U;
    std::uint32_t selected_token_id = 0U;
    std::uint32_t remaining_token_budget = 0U;
    bool terminal = false;
    TokenSamplerOptions sampler_options;
    TokenSelectionRequest request;
    SamplerCheckpoint sampler_pre;
    SamplerTicket draw_ticket;
    SamplerCheckpoint sampler_post;
    std::span<const TokenLogit> compact_candidates;
};

using Gemma2TargetSelectionTraceCallback =
    std::function<void(const Gemma2TargetSelectionTraceEvent&)>;

struct Gemma2GenerationTraceBinding {
    FeatureSinkBinding feature_sink;
    Gemma2TargetSelectionTraceCallback on_target_selection;
};
struct Gemma2ExecutionReceipt {
    bool completion_unknown = false;
};
[[nodiscard]] Gemma2MetalResidentDecodeResult run_gemma2_metal_resident_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request = std::nullopt,
    Gemma2MetalDecodeOutputMode output_mode = Gemma2MetalDecodeOutputMode::logits,
    FeatureSinkBinding feature_sink = {},
    Gemma2ExecutionReceipt* completion = nullptr
);

[[nodiscard]] Gemma2MetalVerifierResult run_gemma2_metal_width_two_verifier(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& verifier,
    std::span<const std::uint32_t, 2> token_ids,
    float embedding_scale,
    std::optional<std::array<TokenSelectionRequest, 2>> selection_requests = std::nullopt
);

[[nodiscard]] Gemma2MetalVerifierCommitResult commit_gemma2_metal_width_two_verification(
    Gemma2MetalModelPlan& plan,
    std::uint32_t accepted_token_count
);

[[nodiscard]] Gemma2MetalVerifierCommitResult
provisionally_commit_gemma2_metal_width_two_verification(
    Gemma2MetalModelPlan& plan,
    std::uint32_t accepted_token_count
);

[[nodiscard]] Gemma2MetalVerifierCommitResult finalize_gemma2_metal_width_two_verification(
    Gemma2MetalModelPlan& plan,
    std::uint32_t retained_token_count
);

void discard_gemma2_metal_width_two_verification(Gemma2MetalModelPlan& plan);

[[nodiscard]] Gemma2MetalResidentPrefillResult run_gemma2_metal_resident_prefill(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    float embedding_scale,
    std::optional<TokenSelectionRequest> selection_request = std::nullopt,
    FeatureSinkBinding feature_sink = {},
    Gemma2ExecutionReceipt* completion = nullptr
);

[[nodiscard]] Gemma2MetalWarmupStats warm_up_gemma2_metal_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t, 16> token_ids,
    TokenSampler sampler_copy
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

class StandaloneChatPlan;
// Borrowed, preallocated turn storage. Updated only once the selected token's
// K/V is recoverable, before any irreversible callback for that token.
struct Gemma2GenerationCommitJournal {
    std::span<std::uint32_t> token_storage;
    std::uint32_t count = 0;
    std::uint32_t cached_token_count = 0;
    SamplerCheckpoint sampler;
    Gemma2ExecutionReceipt completion;
    void record(std::uint32_t token, std::uint32_t cached, const TokenSampler& authoritative);
};
[[nodiscard]] Gemma2MetalPlanGeneratedTokens generate_gemma2_metal_sampled_tokens_with_plan(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    std::size_t max_new_tokens,
    std::span<const std::uint32_t> stop_token_ids,
    TokenSampler& sampler,
    const Gemma2MetalGeneratedTokenCallback& on_token = {},
    const Gemma2MonotonicClock& clock = {},
    RetrievalProposer* proposer = nullptr,
    const Gemma2GenerationTraceBinding* trace = nullptr,
    Gemma2GenerationCommitJournal* journal = nullptr,
    StandaloneChatPlan* standalone = nullptr,
    std::span<const std::uint32_t> standalone_context = {}
);

}  // namespace local_gguf_engine::engine
