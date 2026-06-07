#include "local_gguf_engine/engine/llama_layer.hpp"

#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace local_gguf_engine::engine {

namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorError;
using tensor::TensorErrorCode;
using tensor::TensorStorage;
using tensor::ops::add_f32;
using tensor::ops::causal_scaled_dot_product_attention_f32;
using tensor::ops::matmul_f32_f16;
using tensor::ops::multiply_f32;
using tensor::ops::rms_norm_f32;
using tensor::ops::rope_f32;
using tensor::ops::silu_f32;

constexpr std::size_t vector_rank = 1;
constexpr std::size_t matrix_rank = 2;

[[nodiscard]] std::uint64_t dtype_value(DType dtype) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype));
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

void require_f32_matrix_shape(const TensorStorage& storage, std::uint64_t rows, std::uint64_t columns) {
    require_dtype(storage.descriptor(), DType::f32);
    require_rank(storage.descriptor(), matrix_rank);
    require_dimension(storage.descriptor(), 0, rows);
    require_dimension(storage.descriptor(), 1, columns);
}

void require_f32_vector_shape(const TensorStorage& storage, std::uint64_t length) {
    require_dtype(storage.descriptor(), DType::f32);
    require_rank(storage.descriptor(), vector_rank);
    require_dimension(storage.descriptor(), 0, length);
}

void require_f16_matrix_shape(const TensorStorage& storage, std::uint64_t rows, std::uint64_t columns) {
    require_dtype(storage.descriptor(), DType::f16);
    require_rank(storage.descriptor(), matrix_rank);
    require_dimension(storage.descriptor(), 0, rows);
    require_dimension(storage.descriptor(), 1, columns);
}

void require_positive_config_dimension(std::uint32_t value) {
    if (value == 0) {
        throw std::invalid_argument("Llama layer config dimensions must be positive");
    }
}

void require_supported_config(const LlamaModelConfig& config) {
    require_positive_config_dimension(config.embedding_length);
    require_positive_config_dimension(config.attention_head_count);
    require_positive_config_dimension(config.attention_head_count_kv);
    require_positive_config_dimension(config.attention_key_length);
    require_positive_config_dimension(config.attention_value_length);
    require_positive_config_dimension(config.rope_dimension_count);

    if (config.attention_head_count % config.attention_head_count_kv != 0) {
        throw std::invalid_argument("Llama attention head count must be divisible by KV head count");
    }

    if (config.rope_dimension_count > config.attention_key_length) {
        throw std::invalid_argument("Llama RoPE dimension must not exceed attention key length");
    }
}

[[nodiscard]] std::uint64_t query_position_start(std::uint64_t final_position, std::uint64_t token_count) {
    if (final_position < token_count - 1) {
        throw std::invalid_argument("Llama attention position must cover input rows");
    }

    return final_position - (token_count - 1);
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

void require_column_range(
    const TensorDescriptor& descriptor,
    std::uint64_t first_column,
    std::uint64_t column_count
) {
    const auto columns = descriptor.shape.dimension(1);
    const auto end_column = first_column + column_count;
    if (end_column < first_column || end_column > columns) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, columns, end_column);
    }
}

[[nodiscard]] TensorStorage copy_f32_columns(
    const TensorStorage& input,
    std::uint64_t first_column,
    std::uint64_t column_count
) {
    require_dtype(input.descriptor(), DType::f32);
    require_rank(input.descriptor(), matrix_rank);
    require_column_range(input.descriptor(), first_column, column_count);

    const auto rows = input.descriptor().shape.dimension(0);
    const auto columns = input.descriptor().shape.dimension(1);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, column_count}), DType::f32));

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < column_count; ++column) {
            write_f32_linear(
                output,
                row * column_count + column,
                read_f32_linear(input, row * columns + first_column + column)
            );
        }
    }

    return output;
}

void write_f32_columns(
    TensorStorage& output,
    std::uint64_t first_column,
    const TensorStorage& input
) {
    require_dtype(output.descriptor(), DType::f32);
    require_dtype(input.descriptor(), DType::f32);
    require_rank(output.descriptor(), matrix_rank);
    require_rank(input.descriptor(), matrix_rank);
    require_dimension(input.descriptor(), 0, output.descriptor().shape.dimension(0));
    require_column_range(output.descriptor(), first_column, input.descriptor().shape.dimension(1));

    const auto rows = output.descriptor().shape.dimension(0);
    const auto output_columns = output.descriptor().shape.dimension(1);
    const auto input_columns = input.descriptor().shape.dimension(1);

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < input_columns; ++column) {
            write_f32_linear(
                output,
                row * output_columns + first_column + column,
                read_f32_linear(input, row * input_columns + column)
            );
        }
    }
}

}  // namespace

TensorStorage run_llama_attention_layer_f32(
    const TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    require_supported_config(config);

    const auto token_count = input.descriptor().shape.dimension(0);
    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto head_count = static_cast<std::uint64_t>(config.attention_head_count);
    const auto kv_head_count = static_cast<std::uint64_t>(config.attention_head_count_kv);
    const auto key_length = static_cast<std::uint64_t>(config.attention_key_length);
    const auto value_length = static_cast<std::uint64_t>(config.attention_value_length);
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto attention_output_width = head_count * value_length;

    require_f32_matrix_shape(input, token_count, hidden);
    require_f32_vector_shape(weights.attention_norm_weight, hidden);
    require_f16_matrix_shape(weights.query_weight, hidden, query_width);
    require_f16_matrix_shape(weights.key_weight, hidden, key_width);
    require_f16_matrix_shape(weights.value_weight, hidden, value_width);
    require_f16_matrix_shape(weights.output_weight, attention_output_width, hidden);

    const auto normalized = rms_norm_f32(input, weights.attention_norm_weight, config.rms_norm_epsilon);
    const auto query = matmul_f32_f16(normalized, weights.query_weight);
    const auto key = matmul_f32_f16(normalized, weights.key_weight);
    const auto value = matmul_f32_f16(normalized, weights.value_weight);

    auto attention_output = TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({token_count, attention_output_width}), DType::f32)
    );
    const auto kv_group_size = head_count / kv_head_count;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(key_length));
    const auto attention_query_position_start = query_position_start(position, token_count);

    for (std::uint64_t head = 0; head < head_count; ++head) {
        const auto kv_head = head / kv_group_size;
        const auto query_head = copy_f32_columns(query, head * key_length, key_length);
        const auto key_head = copy_f32_columns(key, kv_head * key_length, key_length);
        const auto value_head = copy_f32_columns(value, kv_head * value_length, value_length);
        const auto rotated_query = rope_f32(query_head, position, config.rope_dimension_count, config.rope_freq_base);
        const auto rotated_key = rope_f32(key_head, position, config.rope_dimension_count, config.rope_freq_base);
        const auto head_output = causal_scaled_dot_product_attention_f32(
            rotated_query,
            rotated_key,
            value_head,
            attention_scale,
            attention_query_position_start
        );

        write_f32_columns(attention_output, head * value_length, head_output);
    }

    const auto projected = matmul_f32_f16(attention_output, weights.output_weight);
    return add_f32(input, projected);
}

TensorStorage run_llama_feed_forward_layer_f32(
    const TensorStorage& input,
    const LlamaFeedForwardLayerWeights& weights,
    const LlamaModelConfig& config
) {
    require_positive_config_dimension(config.embedding_length);
    require_positive_config_dimension(config.feed_forward_length);

    const auto token_count = input.descriptor().shape.dimension(0);
    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto feed_forward = static_cast<std::uint64_t>(config.feed_forward_length);

    require_f32_matrix_shape(input, token_count, hidden);
    require_f32_vector_shape(weights.feed_forward_norm_weight, hidden);
    require_f16_matrix_shape(weights.gate_weight, hidden, feed_forward);
    require_f16_matrix_shape(weights.up_weight, hidden, feed_forward);
    require_f16_matrix_shape(weights.down_weight, feed_forward, hidden);

    const auto normalized = rms_norm_f32(input, weights.feed_forward_norm_weight, config.rms_norm_epsilon);
    const auto gate = matmul_f32_f16(normalized, weights.gate_weight);
    const auto up = matmul_f32_f16(normalized, weights.up_weight);
    const auto activated_gate = silu_f32(gate);
    const auto gated = multiply_f32(activated_gate, up);
    const auto projected = matmul_f32_f16(gated, weights.down_weight);
    return add_f32(input, projected);
}

TensorStorage run_llama_block_f32(
    const TensorStorage& input,
    const LlamaBlockWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    const auto attention_output = run_llama_attention_layer_f32(input, weights.attention, config, position);
    return run_llama_feed_forward_layer_f32(attention_output, weights.feed_forward, config);
}

}  // namespace local_gguf_engine::engine
