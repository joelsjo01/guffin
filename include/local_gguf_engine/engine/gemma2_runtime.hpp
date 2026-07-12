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
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    tensor::BackendBuffer normalized_attention_input;
};

struct Gemma2MetalAttentionProjectionActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    tensor::BackendBuffer query;
    tensor::BackendBuffer key;
    tensor::BackendBuffer value;
};

struct Gemma2MetalAttentionPositionActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
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
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
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
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t attention_token_count = 0;
    std::uint32_t query_head_count = 0;
    tensor::BackendBuffer probabilities;
};

struct Gemma2MetalAttentionValueActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
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
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t attention_output_width = 0;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer projected_attention_output;
};

struct Gemma2MetalPostAttentionResidualActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer post_attention_residual;
};

struct Gemma2MetalFeedForwardGateUpActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t hidden_width = 0;
    std::uint32_t feed_forward_width = 0;
    tensor::BackendBuffer normalized_feed_forward_input;
    tensor::BackendBuffer gate;
    tensor::BackendBuffer up;
};

struct Gemma2MetalFeedForwardOutputActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t hidden_width = 0;
    std::uint32_t feed_forward_width = 0;
    tensor::BackendBuffer feed_forward_gated;
    tensor::BackendBuffer projected_feed_forward_output;
    tensor::BackendBuffer feed_forward_residual;
};

struct Gemma2MetalLayerHandoffActivations {
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer normalized_attention_input;
};

struct Gemma2MetalPrefillKVCacheWriteActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t cache_position = 0;
    std::uint32_t token_count = 0;
    std::uint32_t key_head_count = 0;
    std::uint32_t key_length = 0;
    std::uint32_t value_head_count = 0;
    std::uint32_t value_length = 0;
    tensor::BackendBuffer key_cache;
    tensor::BackendBuffer value_cache;
};

struct Gemma2MetalPrefillBlockActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    Gemma2MetalAttentionProjectionActivations attention_projection;
    Gemma2MetalAttentionPositionActivations attention_position;
    Gemma2MetalPrefillKVCacheWriteActivations kv_cache_write;
    Gemma2MetalAttentionScoreActivations attention_scores;
    Gemma2MetalAttentionProbabilityActivations attention_probabilities;
    Gemma2MetalAttentionValueActivations attention_values;
    Gemma2MetalAttentionOutputProjectionActivations attention_output_projection;
    Gemma2MetalPostAttentionResidualActivations post_attention_residual;
    Gemma2MetalFeedForwardGateUpActivations feed_forward_gate_up;
    Gemma2MetalFeedForwardOutputActivations feed_forward_output;
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

struct Gemma2MetalPrefillOutputNormActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t hidden_width = 0;
    tensor::BackendBuffer normalized_hidden_states;
};

struct Gemma2MetalPrefillLogitsActivations {
    std::uint32_t schedule_layer_offset = 0;
    std::uint32_t layer_index = 0;
    Gemma2MetalSchedulePhase phase = Gemma2MetalSchedulePhase::prefill;
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
    std::uint32_t hidden_width = 0;
    std::uint32_t vocab_size = 0;
    float final_logit_softcap = 0.0F;
    tensor::BackendBuffer raw_logits;
    tensor::BackendBuffer logits;
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
    Gemma2MetalScratchOwnership scratch_ownership = Gemma2MetalScratchOwnership::phase_local;
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

struct Gemma2MetalGenerationRuntimeShell {
    Gemma2MetalModelPlan plan;
    Gemma2MetalSchedule prefill;
    Gemma2MetalSchedule first_decode;
    Gemma2MetalInputActivations prefill_input;
    Gemma2MetalLayerInputActivations prefill_first_layer_input;
    Gemma2MetalAttentionProjectionActivations prefill_first_layer_attention_projection;
    Gemma2MetalAttentionPositionActivations prefill_first_layer_attention_position;
    Gemma2MetalPrefillKVCacheWriteActivations prefill_first_layer_kv_cache_write;
    Gemma2MetalAttentionScoreActivations prefill_first_layer_attention_scores;
    Gemma2MetalAttentionProbabilityActivations prefill_first_layer_attention_probabilities;
    Gemma2MetalAttentionValueActivations prefill_first_layer_attention_values;
    Gemma2MetalAttentionOutputProjectionActivations prefill_first_layer_attention_output_projection;
    Gemma2MetalPostAttentionResidualActivations prefill_first_layer_post_attention_residual;
    Gemma2MetalFeedForwardGateUpActivations prefill_first_layer_feed_forward_gate_up;
    Gemma2MetalFeedForwardOutputActivations prefill_first_layer_feed_forward_output;
    std::vector<Gemma2MetalLayerHandoffActivations> prefill_reusable_layer_inputs;
    std::vector<Gemma2MetalPrefillBlockActivations> prefill_reusable_blocks;
    std::vector<Gemma2MetalPrefillKVCacheWriteActivations> prefill_kv_cache_writes;
    Gemma2MetalPrefillOutputNormActivations prefill_output_norm;
    Gemma2MetalPrefillLogitsActivations prefill_logits;
    Gemma2MetalPrefillNextTokenSelection prefill_next_token_selection;
    Gemma2MetalInputActivations first_decode_input;
    Gemma2MetalLayerInputActivations first_decode_first_layer_input;
    Gemma2MetalDecodeBlockActivations first_decode_block;
    std::vector<Gemma2MetalLayerHandoffActivations> first_decode_reusable_layer_inputs;
    std::vector<Gemma2MetalDecodeBlockActivations> first_decode_reusable_blocks;
    Gemma2MetalFeedForwardOutputActivations first_decode_terminal_feed_forward_output;
    Gemma2MetalDecodeOutputNormActivations first_decode_output_norm;
    Gemma2MetalDecodeLogitsActivations first_decode_logits;
    Gemma2MetalDecodeNextTokenSelection first_decode_next_token_selection;
};

struct Gemma2MetalGeneratedTokenLoopActivations {
    Gemma2MetalGenerationRuntimeShell runtime_shell;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<Gemma2MetalDecodeStepActivations> decode_steps;
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
create_gemma2_metal_scheduled_prefill_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
);

[[nodiscard]] std::optional<Gemma2MetalLayerHandoffActivations>
create_gemma2_metal_scheduled_decode_layer_input(
    const Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& decode,
    std::uint32_t schedule_layer_offset,
    const Gemma2MetalFeedForwardOutputActivations& previous_feed_forward_output
);

[[nodiscard]] Gemma2MetalPrefillBlockActivations run_gemma2_metal_scheduled_prefill_block(
    Gemma2MetalModelPlan& plan,
    const Gemma2MetalSchedule& prefill,
    std::uint32_t schedule_layer_offset,
    const tensor::BackendBuffer& normalized_attention_input,
    const tensor::BackendBuffer& residual_input
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

[[nodiscard]] Gemma2MetalGenerationRuntimeShell create_gemma2_metal_generation_runtime_shell(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids
);

[[nodiscard]] Gemma2MetalGeneratedTokenLoopActivations create_gemma2_metal_generated_token_loop(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    std::optional<std::uint32_t> stop_token_id = std::nullopt
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
