#include "local_gguf_engine/engine/kv_cache.hpp"

#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorError;
using tensor::TensorErrorCode;
using tensor::TensorStorage;

constexpr std::size_t matrix_rank = 2;

[[nodiscard]] std::uint64_t dtype_value(DType dtype) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype));
}

void require_positive_config_dimension(std::uint32_t value) {
    if (value == 0) {
        throw std::invalid_argument("Llama KV cache config dimensions must be positive");
    }
}

void require_dtype(const TensorDescriptor& descriptor, DType expected) {
    if (descriptor.dtype != expected) {
        throw TensorError(TensorErrorCode::unsupported_operator_dtype, dtype_value(expected), dtype_value(descriptor.dtype));
    }
}

void require_rank(const TensorDescriptor& descriptor, std::size_t expected_rank) {
    if (descriptor.shape.rank() != expected_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(expected_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_dimension(
    const TensorDescriptor& descriptor,
    std::size_t dimension_index,
    std::uint64_t expected_dimension
) {
    const auto actual_dimension = descriptor.shape.dimension(dimension_index);
    if (actual_dimension != expected_dimension) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_dimension, actual_dimension);
    }
}

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
}

[[nodiscard]] TensorStorage allocate_cache_tensor(std::uint64_t context_length, std::uint64_t width) {
    return TensorStorage::allocate(TensorDescriptor::from(Shape::from({context_length, width}), DType::f32));
}

[[nodiscard]] TensorStorage copy_cache_prefix(const TensorStorage& storage, std::uint64_t row_count) {
    if (row_count == 0) {
        throw std::invalid_argument("Llama KV cache prefix must contain at least one token");
    }

    require_dtype(storage.descriptor(), DType::f32);
    require_rank(storage.descriptor(), matrix_rank);
    const auto capacity = storage.descriptor().shape.dimension(0);
    const auto width = storage.descriptor().shape.dimension(1);
    if (row_count > capacity) {
        throw TensorError(TensorErrorCode::tensor_index_out_of_range, capacity, row_count);
    }

    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({row_count, width}), DType::f32));
    for (std::uint64_t row = 0; row < row_count; ++row) {
        for (std::uint64_t column = 0; column < width; ++column) {
            write_f32_linear(output, row * width + column, read_f32_linear(storage, row * width + column));
        }
    }

    return output;
}

}  // namespace

LlamaKVCache create_llama_kv_cache(const LlamaModelConfig& config) {
    require_positive_config_dimension(config.context_length);
    require_positive_config_dimension(config.block_count);
    require_positive_config_dimension(config.attention_head_count_kv);
    require_positive_config_dimension(config.attention_key_length);
    require_positive_config_dimension(config.attention_value_length);

    const auto context_length = static_cast<std::uint64_t>(config.context_length);
    const auto kv_head_count = static_cast<std::uint64_t>(config.attention_head_count_kv);
    const auto key_width = kv_head_count * static_cast<std::uint64_t>(config.attention_key_length);
    const auto value_width = kv_head_count * static_cast<std::uint64_t>(config.attention_value_length);

    std::vector<LlamaKVCacheLayer> layers;
    layers.reserve(config.block_count);
    for (std::uint32_t block_index = 0; block_index < config.block_count; ++block_index) {
        layers.push_back(LlamaKVCacheLayer {
            .key = allocate_cache_tensor(context_length, key_width),
            .value = allocate_cache_tensor(context_length, value_width),
            .filled_token_count = 0,
        });
    }

    return LlamaKVCache {
        .layers = std::move(layers),
        .context_length = context_length,
        .key_width = key_width,
        .value_width = value_width,
    };
}

void append_llama_kv_cache_layer_f32(
    LlamaKVCacheLayer& layer,
    const TensorStorage& key_rows,
    const TensorStorage& value_rows,
    std::uint64_t position
) {
    require_dtype(layer.key.descriptor(), DType::f32);
    require_dtype(layer.value.descriptor(), DType::f32);
    require_rank(layer.key.descriptor(), matrix_rank);
    require_rank(layer.value.descriptor(), matrix_rank);

    const auto capacity = layer.key.descriptor().shape.dimension(0);
    const auto key_width = layer.key.descriptor().shape.dimension(1);
    const auto value_width = layer.value.descriptor().shape.dimension(1);
    require_dimension(layer.value.descriptor(), 0, capacity);

    if (position != layer.filled_token_count) {
        throw std::invalid_argument("Llama KV cache append position must match filled token count");
    }

    require_dtype(key_rows.descriptor(), DType::f32);
    require_dtype(value_rows.descriptor(), DType::f32);
    require_rank(key_rows.descriptor(), matrix_rank);
    require_rank(value_rows.descriptor(), matrix_rank);
    require_dimension(key_rows.descriptor(), 1, key_width);
    require_dimension(value_rows.descriptor(), 1, value_width);

    const auto row_count = key_rows.descriptor().shape.dimension(0);
    if (row_count == 0) {
        throw std::invalid_argument("Llama KV cache append rows must be positive");
    }
    require_dimension(value_rows.descriptor(), 0, row_count);

    if (layer.filled_token_count > capacity || row_count > capacity - layer.filled_token_count) {
        throw std::invalid_argument("Llama KV cache append would exceed context length");
    }

    for (std::uint64_t row = 0; row < row_count; ++row) {
        const auto cache_row = layer.filled_token_count + row;
        for (std::uint64_t column = 0; column < key_width; ++column) {
            write_f32_linear(
                layer.key,
                cache_row * key_width + column,
                read_f32_linear(key_rows, row * key_width + column)
            );
        }
        for (std::uint64_t column = 0; column < value_width; ++column) {
            write_f32_linear(
                layer.value,
                cache_row * value_width + column,
                read_f32_linear(value_rows, row * value_width + column)
            );
        }
    }

    layer.filled_token_count += row_count;
}

TensorStorage llama_kv_cache_layer_key_prefix_f32(const LlamaKVCacheLayer& layer) {
    return copy_cache_prefix(layer.key, layer.filled_token_count);
}

TensorStorage llama_kv_cache_layer_value_prefix_f32(const LlamaKVCacheLayer& layer) {
    return copy_cache_prefix(layer.value, layer.filled_token_count);
}

}  // namespace local_gguf_engine::engine
