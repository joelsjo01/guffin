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

struct Gemma2MetalGenerationRuntimeShell {
    Gemma2MetalModelPlan plan;
    Gemma2MetalSchedule prefill;
    Gemma2MetalSchedule first_decode;
    Gemma2MetalInputActivations prefill_input;
    Gemma2MetalLayerInputActivations prefill_first_layer_input;
};

[[nodiscard]] Gemma2MetalGenerationRuntimeShell create_gemma2_metal_generation_runtime_shell(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids
);

}  // namespace local_gguf_engine::engine
