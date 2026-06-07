#pragma once

#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

struct LlamaModelWeights {
    // Borrowed tensor references. The EngineModel used for binding must outlive this view.
    const tensor::TensorStorage& token_embedding_weight;
    const tensor::TensorStorage& output_norm_weight;
    const tensor::TensorStorage& output_weight;
    std::vector<LlamaBlockWeights> blocks;
};

[[nodiscard]] LlamaModelWeights bind_llama_model_weights(const EngineModel& model);

[[nodiscard]] tensor::TensorStorage embed_llama_token_ids_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage run_llama_model_blocks_f32(
    const tensor::TensorStorage& input,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage project_llama_logits_f32(
    const tensor::TensorStorage& hidden_states,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage run_llama_model_logits_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_argmax_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] std::vector<std::uint32_t> generate_llama_argmax_tokens_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
);

}  // namespace local_gguf_engine::engine
