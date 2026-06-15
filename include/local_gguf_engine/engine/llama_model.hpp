#pragma once

#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
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

struct LlamaModelAttentionBackendWeights {
    std::vector<LlamaAttentionLayerBackendWeights> blocks;
};

struct LlamaModelFeedForwardBackendWeights {
    std::vector<LlamaFeedForwardLayerBackendWeights> blocks;
};

struct LlamaModelBlockBackendWeights {
    std::vector<LlamaBlockBackendWeights> blocks;
};

struct LlamaModelLogitsBackendWeights {
    const tensor::TensorStorage& token_embedding_weight;
    const tensor::TensorStorage& output_norm_weight;
    tensor::BackendBuffer output_weight;
    std::vector<LlamaBlockBackendWeights> blocks;
};

[[nodiscard]] LlamaModelWeights bind_llama_model_weights(const EngineModel& model);
[[nodiscard]] LlamaModelAttentionBackendWeights copy_llama_model_attention_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
);
[[nodiscard]] LlamaModelFeedForwardBackendWeights copy_llama_model_feed_forward_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
);
[[nodiscard]] LlamaModelBlockBackendWeights copy_llama_model_block_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
);
[[nodiscard]] LlamaModelLogitsBackendWeights copy_llama_model_logits_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
);

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

[[nodiscard]] tensor::TensorStorage run_llama_model_blocks_f32(
    const tensor::TensorStorage& input,
    const LlamaModelBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_model_blocks_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_model_blocks_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaModelBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_model_blocks_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaModelBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaBackendKVCache& cache
);

[[nodiscard]] tensor::TensorStorage project_llama_logits_f32(
    const tensor::TensorStorage& hidden_states,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage project_llama_logits_f32(
    const tensor::TensorStorage& hidden_states,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage run_llama_model_logits_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_model_logits_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_model_logits_with_kv_cache_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_argmax_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_argmax_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_argmax_with_kv_cache_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_sampled_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    const SamplingConfig& sampling,
    SamplingState& sampling_state,
    std::span<const std::uint32_t> repetition_token_ids = {}
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_sampled_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    const SamplingConfig& sampling,
    SamplingState& sampling_state,
    std::span<const std::uint32_t> repetition_token_ids = {}
);

[[nodiscard]] std::uint32_t run_llama_model_next_token_sampled_with_kv_cache_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache,
    const SamplingConfig& sampling,
    SamplingState& sampling_state,
    std::span<const std::uint32_t> repetition_token_ids = {}
);

[[nodiscard]] std::vector<std::uint32_t> generate_llama_argmax_tokens_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::optional<std::uint32_t> stop_token_id = std::nullopt
);

[[nodiscard]] std::vector<std::uint32_t> generate_llama_argmax_tokens_with_kv_cache_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::optional<std::uint32_t> stop_token_id = std::nullopt
);

[[nodiscard]] std::vector<std::uint32_t> generate_llama_sampled_tokens_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    const SamplingConfig& sampling,
    std::optional<std::uint32_t> stop_token_id = std::nullopt
);

[[nodiscard]] std::vector<std::uint32_t> generate_llama_sampled_tokens_with_kv_cache_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    const SamplingConfig& sampling,
    std::optional<std::uint32_t> stop_token_id = std::nullopt
);

}  // namespace local_gguf_engine::engine
