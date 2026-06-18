#pragma once

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::tests {

local_gguf_engine::engine::LlamaModelConfig tiny_feed_forward_layer_config();
local_gguf_engine::engine::LlamaModelConfig tiny_logits_config();

local_gguf_engine::engine::EngineTensor engine_tensor(
    std::string name,
    local_gguf_engine::tensor::TensorStorage storage
);
[[nodiscard]] std::string llama_block_tensor_name(std::uint32_t block_index, std::string_view suffix);
void add_tiny_llama_block_tensors(
    std::vector<local_gguf_engine::engine::EngineTensor>& tensors,
    std::uint32_t block_index,
    float projection_scale,
    bool query_as_f32,
    bool include_down_weight
);
void add_tiny_llama_top_level_tensors(std::vector<local_gguf_engine::engine::EngineTensor>& tensors);
[[nodiscard]] local_gguf_engine::engine::EngineModel tiny_bound_llama_model(
    local_gguf_engine::engine::LlamaModelConfig config,
    std::vector<local_gguf_engine::engine::EngineTensor> tensors
);

}  // namespace local_gguf_engine::tests
