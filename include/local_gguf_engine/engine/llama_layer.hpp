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

struct LlamaFeedForwardLayerWeights {
    const tensor::TensorStorage& feed_forward_norm_weight;
    const tensor::TensorStorage& gate_weight;
    const tensor::TensorStorage& up_weight;
    const tensor::TensorStorage& down_weight;
};

struct LlamaBlockWeights {
    LlamaAttentionLayerWeights attention;
    LlamaFeedForwardLayerWeights feed_forward;
};

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_feed_forward_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaFeedForwardLayerWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage run_llama_block_f32(
    const tensor::TensorStorage& input,
    const LlamaBlockWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

}  // namespace local_gguf_engine::engine
