#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

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

struct Gemma2MetalGenerationRuntimeShell {
    Gemma2MetalModelPlan plan;
    Gemma2MetalSchedule prefill;
    Gemma2MetalSchedule first_decode;
    Gemma2MetalInputActivations prefill_input;
    Gemma2MetalLayerInputActivations prefill_first_layer_input;
    Gemma2MetalAttentionProjectionActivations prefill_first_layer_attention_projection;
    Gemma2MetalAttentionPositionActivations prefill_first_layer_attention_position;
    Gemma2MetalAttentionScoreActivations prefill_first_layer_attention_scores;
};

[[nodiscard]] Gemma2MetalGenerationRuntimeShell create_gemma2_metal_generation_runtime_shell(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids
);

}  // namespace local_gguf_engine::engine
