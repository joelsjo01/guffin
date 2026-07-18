#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
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
    std::uint64_t decode_started_ns = 0;
    std::uint64_t decode_completed_ns = 0;
    std::uint64_t generation_completed_ns = 0;
    std::uint64_t prefill_duration_ns = 0;
    std::uint64_t decode_duration_ns = 0;
    std::uint64_t runtime_ttft_ns = 0;
    std::uint32_t newly_evaluated_prompt_token_count = 0;
    std::uint32_t generated_token_count = 0;
    std::uint32_t decode_token_count = 0;
    Gemma2GenerationStopReason stop_reason = Gemma2GenerationStopReason::token_limit;
};

struct Gemma2MetalInputActivations {
    std::vector<std::uint32_t> token_ids;
    tensor::BackendBuffer hidden_states;
    float embedding_scale = 0.0F;
};

struct Gemma2MetalLayerInputActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    tensor::BackendBuffer normalized_attention_input;
};

struct Gemma2MetalAttentionProjectionActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    tensor::BackendBuffer query;
    tensor::BackendBuffer key;
    tensor::BackendBuffer value;
};

struct Gemma2MetalAttentionPositionActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t position_start = 0;
    std::uint32_t query_head_count = 0;
    std::uint32_t key_head_count = 0;
    std::uint32_t head_dimension = 0;
    std::uint32_t rope_dimension_count = 0;
    float rope_frequency_base = 0.0F;
    tensor::BackendBuffer query;
    tensor::BackendBuffer key;
};

struct Gemma2MetalAttentionScoreActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t query_position_start = 0;
    std::uint32_t attention_token_count = 0;
    std::uint32_t query_head_count = 0;
    std::uint32_t key_head_count = 0;
    std::uint32_t key_length = 0;
    float scale = 0.0F;
    float logit_softcap = 0.0F;
    tensor::BackendBuffer scores;
};

struct Gemma2MetalAttentionProbabilityActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t attention_token_count = 0;
    std::uint32_t query_head_count = 0;
    tensor::BackendBuffer probabilities;
};

struct Gemma2MetalAttentionValueActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t query_position_start = 0;
    std::uint32_t attention_token_count = 0;
    std::uint32_t query_head_count = 0;
    std::uint32_t value_head_count = 0;
    std::uint32_t value_length = 0;
    tensor::BackendBuffer attention_output;
};

struct Gemma2MetalAttentionOutputProjectionActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t attention_output_width = 0;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer projected_attention_output;
};

struct Gemma2MetalPostAttentionResidualActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer post_attention_residual;
};

struct Gemma2MetalFeedForwardGateUpActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t hidden_width = 0;
    std::uint32_t feed_forward_width = 0;
    tensor::BackendBuffer normalized_feed_forward_input;
    tensor::BackendBuffer gate;
    tensor::BackendBuffer up;
};

struct Gemma2MetalFeedForwardOutputActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t hidden_width = 0;
    std::uint32_t feed_forward_width = 0;
    tensor::BackendBuffer feed_forward_gated;
    tensor::BackendBuffer projected_feed_forward_output;
    tensor::BackendBuffer feed_forward_residual;
};

struct Gemma2MetalLayerHandoffActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer normalized_attention_input;
};

struct Gemma2MetalDecodeKVCacheWriteActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t cache_position = 0;
    std::uint32_t token_count = 0;
    std::uint32_t total_kv_token_count = 0;
    std::uint32_t key_head_count = 0;
    std::uint32_t key_length = 0;
    std::uint32_t value_head_count = 0;
    std::uint32_t value_length = 0;
    tensor::BackendBuffer key_cache;
    tensor::BackendBuffer value_cache;
};

struct Gemma2MetalDecodeBlockActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    Gemma2MetalAttentionProjectionActivations attention_projection;
    Gemma2MetalAttentionPositionActivations attention_position;
    Gemma2MetalDecodeKVCacheWriteActivations kv_cache_write;
    Gemma2MetalAttentionScoreActivations attention_scores;
    Gemma2MetalAttentionProbabilityActivations attention_probabilities;
    Gemma2MetalAttentionValueActivations attention_values;
    Gemma2MetalAttentionOutputProjectionActivations attention_output_projection;
    Gemma2MetalPostAttentionResidualActivations post_attention_residual;
    Gemma2MetalFeedForwardGateUpActivations feed_forward_gate_up;
    Gemma2MetalFeedForwardOutputActivations feed_forward_output;
};

struct Gemma2MetalDecodeOutputNormActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer normalized_hidden_states;
};

struct Gemma2MetalDecodeLogitsActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t hidden_width = 0;
    std::uint32_t vocab_size = 0;
    float final_logit_softcap = 0.0F;
    tensor::BackendBuffer raw_logits;
    tensor::BackendBuffer logits;
};

struct Gemma2MetalPrefillNextTokenSelection {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_prefill;
    std::uint32_t logits_row_index = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t token_id = 0;
    float logit = 0.0F;
};

struct Gemma2MetalDecodeNextTokenSelection {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t logits_row_index = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t token_id = 0;
    float logit = 0.0F;
};

struct Gemma2MetalDecodeStepActivations {
    Gemma2MetalSchedule decode;
    Gemma2MetalInputActivations input;
    Gemma2MetalLayerInputActivations first_layer_input;
    Gemma2MetalDecodeBlockActivations first_block;
    std::vector<Gemma2MetalLayerHandoffActivations> reusable_layer_inputs;
    std::vector<Gemma2MetalDecodeBlockActivations> reusable_blocks;
    Gemma2MetalFeedForwardOutputActivations terminal_feed_forward_output;
    Gemma2MetalDecodeOutputNormActivations output_norm;
    Gemma2MetalDecodeLogitsActivations logits;
    Gemma2MetalDecodeNextTokenSelection next_token_selection;
};

struct Gemma2MetalDecodeExecutionStats {
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint64_t gpu_duration_ns = 0;
};

// Production decode exposes only values whose lifetime extends beyond scratch
// reuse. The all-intermediate DecodeStepActivations API remains the explicit
// diagnostic path for parity tests.
struct Gemma2MetalResidentDecodeResult {
    tensor::BackendBuffer logits;
    std::optional<Gemma2MetalDecodeNextTokenSelection> next_token_selection;
    Gemma2MetalDecodeExecutionStats execution;
};

struct Gemma2MetalPrefillExecutionStats {
    std::uint32_t chunk_count = 0;
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint32_t scheduled_layer_execution_count = 0;
    std::uint32_t kv_write_count = 0;
    std::uint64_t gpu_duration_ns = 0;
};

struct Gemma2MetalResidentPrefillResult {
    tensor::BackendBuffer logits;
    std::optional<Gemma2MetalPrefillNextTokenSelection> next_token_selection;
    Gemma2MetalPrefillExecutionStats execution;
};

struct Gemma2MetalPlanGeneratedTokens {
    std::vector<std::uint32_t> generated_token_ids;
    std::optional<std::uint32_t> stop_token_id;
    std::uint32_t prefill_cache_position = 0;
    std::uint32_t prefill_token_count = 0;
    std::uint32_t cached_token_count = 0;
    Gemma2GenerationMetrics metrics;
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
using Gemma2MetalTokenSelector =
    std::function<std::uint32_t(const tensor::BackendBuffer& logits)>;

[[nodiscard]] std::optional<Gemma2MetalLayerHandoffActivations>
create_gemma2_metal_scheduled_decode_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
);

[[nodiscard]] Gemma2MetalDecodeBlockActivations run_gemma2_metal_scheduled_decode_block(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const tensor::BackendBuffer& normalized_attention_input,
    const tensor::BackendBuffer& residual_input
);

[[nodiscard]] Gemma2MetalDecodeStepActivations run_gemma2_metal_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale
);

[[nodiscard]] Gemma2MetalResidentDecodeResult run_gemma2_metal_resident_decode_step(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t token_id,
    float embedding_scale,
    bool select_top_logit = true
);

[[nodiscard]] Gemma2MetalResidentPrefillResult run_gemma2_metal_resident_prefill(
    Gemma2MetalModelPlan& plan,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t cache_position,
    float embedding_scale,
    bool select_top_logit = false
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
    const Gemma2MetalTokenSelector& select_token,
    const Gemma2MetalGeneratedTokenCallback& on_token = {},
    const Gemma2MonotonicClock& clock = {}
);

}  // namespace local_gguf_engine::engine
