#pragma once

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <vector>

namespace local_gguf_engine::engine {

struct Gemma2AttentionWeights {
    tensor::TensorView attention_norm_weight;
    tensor::TensorView post_attention_norm_weight;
    tensor::TensorView query_weight;
    tensor::TensorView key_weight;
    tensor::TensorView value_weight;
    tensor::TensorView output_weight;
    std::optional<tensor::TensorView> query_norm_weight;
    std::optional<tensor::TensorView> key_norm_weight;
};

struct Gemma2FeedForwardWeights {
    tensor::TensorView feed_forward_norm_weight;
    tensor::TensorView post_feed_forward_norm_weight;
    tensor::TensorView gate_weight;
    tensor::TensorView up_weight;
    tensor::TensorView down_weight;
};

struct Gemma2BlockWeights {
    Gemma2AttentionWeights attention;
    Gemma2FeedForwardWeights feed_forward;
};

struct Gemma2ModelWeights {
    tensor::TensorView token_embedding_weight;
    tensor::TensorView output_norm_weight;
    tensor::TensorView output_weight;
    std::vector<Gemma2BlockWeights> blocks;
};

[[nodiscard]] Gemma2ModelWeights bind_gemma2_model_weights(const EngineModel& model);
// Used before residency allocation, including callers supplying borrowed
// weights directly rather than going through the GGUF binding entry point.
void validate_gemma3_model_weights(const Gemma2ModelConfig& config, const Gemma2ModelWeights& weights);

}  // namespace local_gguf_engine::engine
