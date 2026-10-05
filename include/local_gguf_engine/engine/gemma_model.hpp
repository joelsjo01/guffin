#pragma once

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <vector>

namespace local_gguf_engine::engine {

struct GemmaAttentionWeights {
    tensor::TensorView attention_norm_weight;
    tensor::TensorView post_attention_norm_weight;
    tensor::TensorView query_weight;
    tensor::TensorView key_weight;
    tensor::TensorView value_weight;
    tensor::TensorView output_weight;
    std::optional<tensor::TensorView> query_norm_weight;
    std::optional<tensor::TensorView> key_norm_weight;
};

struct GemmaFeedForwardWeights {
    tensor::TensorView feed_forward_norm_weight;
    tensor::TensorView post_feed_forward_norm_weight;
    tensor::TensorView gate_weight;
    tensor::TensorView up_weight;
    tensor::TensorView down_weight;
};

struct GemmaBlockWeights {
    GemmaAttentionWeights attention;
    GemmaFeedForwardWeights feed_forward;
};

struct GemmaModelWeights {
    tensor::TensorView token_embedding_weight;
    tensor::TensorView output_norm_weight;
    tensor::TensorView output_weight;
    std::vector<GemmaBlockWeights> blocks;
};

[[nodiscard]] GemmaModelWeights bind_gemma_model_weights(const EngineModel& model);
// Used before residency allocation, including callers supplying borrowed
// weights directly rather than going through the GGUF binding entry point.
void validate_gemma3_model_weights(const GemmaModelConfig& config, const GemmaModelWeights& weights);

}  // namespace local_gguf_engine::engine
