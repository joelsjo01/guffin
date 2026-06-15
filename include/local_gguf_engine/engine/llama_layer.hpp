#pragma once

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
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

struct LlamaAttentionLayerBackendWeights {
    const tensor::TensorStorage& attention_norm_weight;
    tensor::BackendBuffer query_weight;
    tensor::BackendBuffer key_weight;
    tensor::BackendBuffer value_weight;
    tensor::BackendBuffer output_weight;
};

struct LlamaFeedForwardLayerWeights {
    const tensor::TensorStorage& feed_forward_norm_weight;
    const tensor::TensorStorage& gate_weight;
    const tensor::TensorStorage& up_weight;
    const tensor::TensorStorage& down_weight;
};

struct LlamaFeedForwardLayerBackendWeights {
    const tensor::TensorStorage& feed_forward_norm_weight;
    tensor::BackendBuffer feed_forward_norm_weight_buffer;
    tensor::BackendBuffer gate_weight;
    tensor::BackendBuffer up_weight;
    tensor::BackendBuffer down_weight;
};

struct LlamaBlockWeights {
    LlamaAttentionLayerWeights attention;
    LlamaFeedForwardLayerWeights feed_forward;
};

struct LlamaBlockBackendWeights {
    LlamaAttentionLayerBackendWeights attention;
    LlamaFeedForwardLayerBackendWeights feed_forward;
};

[[nodiscard]] LlamaAttentionLayerBackendWeights copy_llama_attention_layer_weights_to_backend(
    const LlamaAttentionLayerWeights& weights,
    runtime::BackendKind backend
);

[[nodiscard]] LlamaFeedForwardLayerBackendWeights copy_llama_feed_forward_layer_weights_to_backend(
    const LlamaFeedForwardLayerWeights& weights,
    runtime::BackendKind backend
);

[[nodiscard]] LlamaBlockBackendWeights copy_llama_block_weights_to_backend(
    const LlamaBlockWeights& weights,
    runtime::BackendKind backend
);

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCacheLayer& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCacheLayer& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_attention_layer_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaAttentionLayerBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaBackendKVCacheLayer& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_feed_forward_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaFeedForwardLayerWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage run_llama_feed_forward_layer_f32(
    const tensor::TensorStorage& input,
    const LlamaFeedForwardLayerBackendWeights& weights,
    const LlamaModelConfig& config
);

[[nodiscard]] tensor::TensorStorage run_llama_block_f32(
    const tensor::TensorStorage& input,
    const LlamaBlockWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_block_f32(
    const tensor::TensorStorage& input,
    const LlamaBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage run_llama_block_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaBlockWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCacheLayer& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_block_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCacheLayer& cache
);

[[nodiscard]] tensor::TensorStorage run_llama_block_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaBackendKVCacheLayer& cache
);

}  // namespace local_gguf_engine::engine
