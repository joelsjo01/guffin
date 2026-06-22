#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"

#include <cstdint>
#include <span>

namespace local_gguf_engine::engine {

struct Gemma2MetalGenerationRuntimeShell {
    Gemma2MetalModelPlan plan;
    Gemma2MetalSchedule prefill;
    Gemma2MetalSchedule first_decode;
};

[[nodiscard]] Gemma2MetalGenerationRuntimeShell create_gemma2_metal_generation_runtime_shell(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids
);

}  // namespace local_gguf_engine::engine
