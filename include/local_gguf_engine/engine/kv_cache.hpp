#pragma once

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <vector>

namespace local_gguf_engine::engine {

struct LlamaKVCacheLayer {
    tensor::TensorStorage key;
    tensor::TensorStorage value;
    std::uint64_t filled_token_count = 0;
};

struct LlamaKVCache {
    std::vector<LlamaKVCacheLayer> layers;
    std::uint64_t context_length = 0;
    std::uint64_t key_width = 0;
    std::uint64_t value_width = 0;
};

[[nodiscard]] LlamaKVCache create_llama_kv_cache(const LlamaModelConfig& config);

void append_llama_kv_cache_layer_f32(
    LlamaKVCacheLayer& layer,
    const tensor::TensorStorage& key_rows,
    const tensor::TensorStorage& value_rows,
    std::uint64_t position
);

[[nodiscard]] tensor::TensorStorage llama_kv_cache_layer_key_prefix_f32(
    const LlamaKVCacheLayer& layer
);

[[nodiscard]] tensor::TensorStorage llama_kv_cache_layer_value_prefix_f32(
    const LlamaKVCacheLayer& layer
);

}  // namespace local_gguf_engine::engine
