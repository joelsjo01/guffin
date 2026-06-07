#pragma once

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::engine {

struct LlamaAttentionLayerWeights {
    const tensor::TensorStorage& attention_norm_weight;
    const tensor::TensorStorage& query_weight;
    const tensor::TensorStorage& key_weight;
    const tensor::TensorStorage& value_weight;
    const tensor::TensorStorage& output_weight;
};

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

}  // namespace local_gguf_engine::engine
