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

struct Gemma2MetalDecodeExecutionStats {
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint64_t gpu_duration_ns = 0;
};

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
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t memory_barrier_count = 0;
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
