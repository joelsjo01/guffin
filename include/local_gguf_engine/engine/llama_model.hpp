#pragma once

#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <vector>

namespace local_gguf_engine::engine {

struct LlamaModelWeights {
    // Borrowed tensor references. The EngineModel used for binding must outlive this view.
    std::vector<LlamaBlockWeights> blocks;
};

[[nodiscard]] LlamaModelWeights bind_llama_model_weights(const EngineModel& model);

[[nodiscard]] tensor::TensorStorage run_llama_model_blocks_f32(
    const tensor::TensorStorage& input,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

}  // namespace local_gguf_engine::engine
