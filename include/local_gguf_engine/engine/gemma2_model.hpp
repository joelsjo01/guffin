#pragma once

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <vector>

namespace local_gguf_engine::engine {

struct Gemma2AttentionWeights {
    const tensor::TensorStorage& attention_norm_weight;
    const tensor::TensorStorage& post_attention_norm_weight;
    const tensor::TensorStorage& query_weight;
    const tensor::TensorStorage& key_weight;
    const tensor::TensorStorage& value_weight;
    const tensor::TensorStorage& output_weight;
};

struct Gemma2FeedForwardWeights {
    const tensor::TensorStorage& feed_forward_norm_weight;
    const tensor::TensorStorage& post_feed_forward_norm_weight;
    const tensor::TensorStorage& gate_weight;
    const tensor::TensorStorage& up_weight;
    const tensor::TensorStorage& down_weight;
};

struct Gemma2BlockWeights {
    Gemma2AttentionWeights attention;
    Gemma2FeedForwardWeights feed_forward;
};

struct Gemma2ModelWeights {
    // Borrowed tensor references. The EngineModel used for binding must outlive this view.
    const tensor::TensorStorage& token_embedding_weight;
    const tensor::TensorStorage& output_norm_weight;
    const tensor::TensorStorage& output_weight;
    std::vector<Gemma2BlockWeights> blocks;
};

[[nodiscard]] Gemma2ModelWeights bind_gemma2_model_weights(const EngineModel& model);

}  // namespace local_gguf_engine::engine
