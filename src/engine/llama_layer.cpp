#include "local_gguf_engine/engine/llama_layer.hpp"

#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/f16.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::engine {

namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorError;
using tensor::TensorErrorCode;
using tensor::TensorStorage;
using tensor::f16_bits_to_f32;
using tensor::copy_tensor_to_backend;
using tensor::ops::add_f32;
using tensor::ops::causal_scaled_dot_product_attention_f32;
using tensor::ops::matmul_f32_f16_gguf_weight;
using tensor::ops::multiply_f32;
using tensor::ops::rms_norm_f32;
using tensor::ops::rope_f32_with_position_start;
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

void require_f16_matrix_shape(const TensorDescriptor& descriptor, std::uint64_t rows, std::uint64_t columns) {
    require_dtype(descriptor, DType::f16);
    require_rank(descriptor, matrix_rank);
    require_dimension(descriptor, 0, rows);
    require_dimension(descriptor, 1, columns);
}

void require_f16_matrix_shape(const TensorStorage& storage, std::uint64_t rows, std::uint64_t columns) {
    require_f16_matrix_shape(storage.descriptor(), rows, columns);
}

[[nodiscard]] const TensorDescriptor& projection_weight_descriptor(const TensorStorage& storage) noexcept {
    return storage.descriptor();
}

[[nodiscard]] const TensorDescriptor& projection_weight_descriptor(const tensor::BackendBuffer& buffer) noexcept {
    return buffer.descriptor();
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

void require_valid_rope_dimension_count(std::uint64_t rope_dimension_count) {
    if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 2, rope_dimension_count);
    }
}

void require_positive_frequency_base(float frequency_base) {
    if (!(frequency_base > 0.0F)) {
        throw std::invalid_argument("RoPE frequency base must be positive");
    }
}

void require_row_position_window(std::uint64_t row_count, std::uint64_t position_start) {
    if (row_count == 0) {
        return;
    }

    if (position_start > std::numeric_limits<std::uint64_t>::max() - (row_count - 1)) {
        throw std::invalid_argument("RoPE row positions overflow");
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

[[nodiscard]] float read_f16_linear_as_f32(const TensorStorage& storage, std::uint64_t linear_index) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(std::uint16_t));
    const auto bytes = storage.bytes();
    const auto bits = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[byte_offset])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[byte_offset + 1]) << 8U)
    );
    return f16_bits_to_f32(bits);
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

[[nodiscard]] float read_f32_column_window(
    const TensorStorage& storage,
    std::uint64_t row,
    std::uint64_t first_column,
    std::uint64_t column
) {
    const auto columns = storage.descriptor().shape.dimension(1);
    return read_f32_linear(storage, row * columns + first_column + column);
}

[[nodiscard]] double rope_angle(
    std::uint64_t position,
    std::uint64_t pair_offset,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

[[nodiscard]] float read_rope_f32_column_window(
    const TensorStorage& storage,
    std::uint64_t row,
    std::uint64_t first_column,
    std::uint64_t column,
    std::uint64_t position_start,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    if (column >= rope_dimension_count) {
        return read_f32_column_window(storage, row, first_column, column);
    }

    const auto pair_offset = column - (column % 2);
    const auto angle = rope_angle(position_start + row, pair_offset, rope_dimension_count, frequency_base);
    const auto angle_cos = static_cast<float>(std::cos(angle));
    const auto angle_sin = static_cast<float>(std::sin(angle));
    const auto first = read_f32_column_window(storage, row, first_column, pair_offset);
    const auto second = read_f32_column_window(storage, row, first_column, pair_offset + 1);

    if (column == pair_offset) {
        return first * angle_cos - second * angle_sin;
    }

    return first * angle_sin + second * angle_cos;
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

void write_f32_rows_columns(
    TensorStorage& output,
    std::uint64_t first_row,
    std::uint64_t first_column,
    const TensorStorage& input
) {
    require_dtype(output.descriptor(), DType::f32);
    require_dtype(input.descriptor(), DType::f32);
    require_rank(output.descriptor(), matrix_rank);
    require_rank(input.descriptor(), matrix_rank);
    require_column_range(output.descriptor(), first_column, input.descriptor().shape.dimension(1));

    const auto output_rows = output.descriptor().shape.dimension(0);
    const auto input_rows = input.descriptor().shape.dimension(0);
    const auto end_row = first_row + input_rows;
    if (end_row < first_row || end_row > output_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, output_rows, end_row);
    }

    const auto output_columns = output.descriptor().shape.dimension(1);
    const auto input_columns = input.descriptor().shape.dimension(1);
    for (std::uint64_t row = 0; row < input_rows; ++row) {
        for (std::uint64_t column = 0; column < input_columns; ++column) {
            write_f32_linear(
                output,
                (first_row + row) * output_columns + first_column + column,
                read_f32_linear(input, row * input_columns + column)
            );
        }
    }
}

[[nodiscard]] double rope_angle(
    std::uint64_t position,
    std::uint64_t pair_offset,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto exponent = static_cast<double>(pair_offset) / static_cast<double>(rope_dimension_count);
    return static_cast<double>(position) / std::pow(static_cast<double>(frequency_base), exponent);
}

void write_rope_f32_rows_columns(
    TensorStorage& output,
    std::uint64_t output_first_row,
    std::uint64_t output_first_column,
    const TensorStorage& input,
    std::uint64_t input_first_column,
    std::uint64_t column_count,
    std::uint64_t position_start,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    require_dtype(output.descriptor(), DType::f32);
    require_dtype(input.descriptor(), DType::f32);
    require_rank(output.descriptor(), matrix_rank);
    require_rank(input.descriptor(), matrix_rank);
    require_column_range(output.descriptor(), output_first_column, column_count);
    require_column_range(input.descriptor(), input_first_column, column_count);
    require_valid_rope_dimension_count(rope_dimension_count);
    if (rope_dimension_count > column_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, column_count, rope_dimension_count);
    }
    require_positive_frequency_base(frequency_base);

    const auto input_rows = input.descriptor().shape.dimension(0);
    require_row_position_window(input_rows, position_start);
    const auto output_rows = output.descriptor().shape.dimension(0);
    const auto output_end_row = output_first_row + input_rows;
    if (output_end_row < output_first_row || output_end_row > output_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, output_rows, output_end_row);
    }

    const auto input_columns = input.descriptor().shape.dimension(1);
    const auto output_columns = output.descriptor().shape.dimension(1);
    for (std::uint64_t row = 0; row < input_rows; ++row) {
        const auto position = position_start + row;
        for (std::uint64_t pair_offset = 0; pair_offset < rope_dimension_count; pair_offset += 2) {
            const auto angle = rope_angle(position, pair_offset, rope_dimension_count, frequency_base);
            const auto angle_cos = static_cast<float>(std::cos(angle));
            const auto angle_sin = static_cast<float>(std::sin(angle));
            const auto input_first_index = row * input_columns + input_first_column + pair_offset;
            const auto output_first_index = (output_first_row + row) * output_columns
                + output_first_column
                + pair_offset;
            const auto first = read_f32_linear(input, input_first_index);
            const auto second = read_f32_linear(input, input_first_index + 1);

            write_f32_linear(output, output_first_index, first * angle_cos - second * angle_sin);
            write_f32_linear(output, output_first_index + 1, first * angle_sin + second * angle_cos);
        }

        for (std::uint64_t column = rope_dimension_count; column < column_count; ++column) {
            write_f32_linear(
                output,
                (output_first_row + row) * output_columns + output_first_column + column,
                read_f32_linear(input, row * input_columns + input_first_column + column)
            );
        }
    }
}

void require_cache_append_window(
    const LlamaKVCacheLayer& cache,
    std::uint64_t position,
    std::uint64_t row_count,
    std::uint64_t key_width,
    std::uint64_t value_width
) {
    require_f32_matrix_shape(cache.key, cache.key.descriptor().shape.dimension(0), key_width);
    require_f32_matrix_shape(cache.value, cache.key.descriptor().shape.dimension(0), value_width);

    if (position != cache.filled_token_count) {
        throw std::invalid_argument("Llama KV cache append position must match filled token count");
    }

    if (row_count == 0) {
        throw std::invalid_argument("Llama KV cache append rows must be positive");
    }

    const auto capacity = cache.key.descriptor().shape.dimension(0);
    if (cache.filled_token_count > capacity || row_count > capacity - cache.filled_token_count) {
        throw std::invalid_argument("Llama KV cache append would exceed context length");
    }
}

void append_rotated_key_value_to_cache_f32(
    LlamaKVCacheLayer& cache,
    const TensorStorage& key,
    const TensorStorage& value,
    const LlamaModelConfig& config,
    std::uint64_t position_start
) {
    const auto token_count = key.descriptor().shape.dimension(0);
    const auto kv_head_count = static_cast<std::uint64_t>(config.attention_head_count_kv);
    const auto key_length = static_cast<std::uint64_t>(config.attention_key_length);
    const auto key_width = kv_head_count * key_length;

    require_f32_matrix_shape(key, token_count, key_width);
    require_f32_matrix_shape(value, token_count, kv_head_count * static_cast<std::uint64_t>(config.attention_value_length));
    require_cache_append_window(
        cache,
        position_start,
        token_count,
        key_width,
        kv_head_count * static_cast<std::uint64_t>(config.attention_value_length)
    );

    for (std::uint64_t kv_head = 0; kv_head < kv_head_count; ++kv_head) {
        write_rope_f32_rows_columns(
            cache.key,
            position_start,
            kv_head * key_length,
            key,
            kv_head * key_length,
            key_length,
            position_start,
            config.rope_dimension_count,
            config.rope_freq_base
        );
    }

    write_f32_rows_columns(cache.value, position_start, 0, value);
    cache.filled_token_count += token_count;
}

[[nodiscard]] float cached_attention_score_f32(
    const TensorStorage& query,
    const LlamaKVCacheLayer& cache,
    std::uint64_t query_row,
    std::uint64_t key_row,
    std::uint64_t query_first_column,
    std::uint64_t key_first_column,
    std::uint64_t key_length,
    std::uint64_t position_start,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    float dot = 0.0F;
    for (std::uint64_t column = 0; column < key_length; ++column) {
        dot += read_rope_f32_column_window(
            query,
            query_row,
            query_first_column,
            column,
            position_start,
            rope_dimension_count,
            frequency_base
        ) * read_f32_column_window(cache.key, key_row, key_first_column, column);
    }

    return dot * scale;
}

void accumulate_cached_attention_projection_f32(
    TensorStorage& projected,
    const TensorStorage& query,
    const TensorStorage& output_weight,
    const LlamaKVCacheLayer& cache,
    const LlamaModelConfig& config,
    std::uint64_t token_count,
    std::uint64_t cached_token_count,
    std::uint64_t position_start,
    float scale
) {
    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto head_count = static_cast<std::uint64_t>(config.attention_head_count);
    const auto kv_head_count = static_cast<std::uint64_t>(config.attention_head_count_kv);
    const auto key_length = static_cast<std::uint64_t>(config.attention_key_length);
    const auto value_length = static_cast<std::uint64_t>(config.attention_value_length);
    const auto kv_group_size = head_count / kv_head_count;
    const auto output_rows = output_weight.descriptor().shape.dimension(0);
    std::vector<float> value_accumulator(static_cast<std::size_t>(value_length));

    for (std::uint64_t head = 0; head < head_count; ++head) {
        const auto kv_head = head / kv_group_size;
        const auto query_first_column = head * key_length;
        const auto key_first_column = kv_head * key_length;
        const auto value_first_column = kv_head * value_length;
        const auto output_first_row = head * value_length;

        for (std::uint64_t query_row = 0; query_row < token_count; ++query_row) {
            for (auto& value : value_accumulator) {
                value = 0.0F;
            }

            float max_score = -std::numeric_limits<float>::infinity();
            float denominator = 0.0F;
            const auto last_visible_key_row = position_start + query_row;

            for (std::uint64_t key_row = 0; key_row < cached_token_count; ++key_row) {
                if (key_row > last_visible_key_row) {
                    continue;
                }

                const auto score = cached_attention_score_f32(
                    query,
                    cache,
                    query_row,
                    key_row,
                    query_first_column,
                    key_first_column,
                    key_length,
                    position_start,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                    scale
                );

                if (score > max_score) {
                    const auto rescale = std::isfinite(max_score) ? std::exp(max_score - score) : 0.0F;
                    denominator *= rescale;
                    for (auto& value : value_accumulator) {
                        value *= rescale;
                    }
                    max_score = score;
                }

                const auto weight = std::exp(score - max_score);
                denominator += weight;
                for (std::uint64_t value_column = 0; value_column < value_length; ++value_column) {
                    value_accumulator[static_cast<std::size_t>(value_column)] += weight * read_f32_column_window(
                        cache.value,
                        key_row,
                        value_first_column,
                        value_column
                    );
                }
            }

            if (!(denominator > 0.0F)) {
                throw std::invalid_argument("Llama cached attention row has no visible keys");
            }

            for (std::uint64_t value_column = 0; value_column < value_length; ++value_column) {
                const auto attention_value = value_accumulator[static_cast<std::size_t>(value_column)] / denominator;
                for (std::uint64_t hidden_column = 0; hidden_column < hidden; ++hidden_column) {
                    const auto projected_index = query_row * hidden + hidden_column;
                    const auto output_weight_index = hidden_column * output_rows + output_first_row + value_column;
                    const auto current = read_f32_linear(projected, projected_index);
                    write_f32_linear(
                        projected,
                        projected_index,
                        current + attention_value * read_f16_linear_as_f32(output_weight, output_weight_index)
                    );
                }
            }
        }
    }
}

template <typename Weights>
[[nodiscard]] TensorStorage run_llama_attention_layer_impl(
    const TensorStorage& input,
    const Weights& weights,
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
    require_f16_matrix_shape(projection_weight_descriptor(weights.query_weight), hidden, query_width);
    require_f16_matrix_shape(projection_weight_descriptor(weights.key_weight), hidden, key_width);
    require_f16_matrix_shape(projection_weight_descriptor(weights.value_weight), hidden, value_width);
    require_f16_matrix_shape(projection_weight_descriptor(weights.output_weight), attention_output_width, hidden);

    const auto normalized = rms_norm_f32(input, weights.attention_norm_weight, config.rms_norm_epsilon);
    const auto query = matmul_f32_f16_gguf_weight(normalized, weights.query_weight);
    const auto key = matmul_f32_f16_gguf_weight(normalized, weights.key_weight);
    const auto value = matmul_f32_f16_gguf_weight(normalized, weights.value_weight);

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
        const auto rotated_query = rope_f32_with_position_start(
            query_head,
            attention_query_position_start,
            config.rope_dimension_count,
            config.rope_freq_base
        );
        const auto rotated_key = rope_f32_with_position_start(
            key_head,
            attention_query_position_start,
            config.rope_dimension_count,
            config.rope_freq_base
        );
        const auto head_output = causal_scaled_dot_product_attention_f32(
            rotated_query,
            rotated_key,
            value_head,
            attention_scale,
            attention_query_position_start
        );

        write_f32_columns(attention_output, head * value_length, head_output);
    }

    const auto projected = matmul_f32_f16_gguf_weight(attention_output, weights.output_weight);
    return add_f32(input, projected);
}

template <typename Weights>
[[nodiscard]] TensorStorage run_llama_feed_forward_layer_impl(
    const TensorStorage& input,
    const Weights& weights,
    const LlamaModelConfig& config
) {
    require_positive_config_dimension(config.embedding_length);
    require_positive_config_dimension(config.feed_forward_length);

    const auto token_count = input.descriptor().shape.dimension(0);
    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto feed_forward = static_cast<std::uint64_t>(config.feed_forward_length);

    require_f32_matrix_shape(input, token_count, hidden);
    require_f32_vector_shape(weights.feed_forward_norm_weight, hidden);
    require_f16_matrix_shape(projection_weight_descriptor(weights.gate_weight), hidden, feed_forward);
    require_f16_matrix_shape(projection_weight_descriptor(weights.up_weight), hidden, feed_forward);
    require_f16_matrix_shape(projection_weight_descriptor(weights.down_weight), feed_forward, hidden);

    const auto normalized = rms_norm_f32(input, weights.feed_forward_norm_weight, config.rms_norm_epsilon);
    const auto gate = matmul_f32_f16_gguf_weight(normalized, weights.gate_weight);
    const auto up = matmul_f32_f16_gguf_weight(normalized, weights.up_weight);
    const auto activated_gate = silu_f32(gate);
    const auto gated = multiply_f32(activated_gate, up);
    const auto projected = matmul_f32_f16_gguf_weight(gated, weights.down_weight);
    return add_f32(input, projected);
}

}  // namespace

LlamaAttentionLayerBackendWeights copy_llama_attention_layer_weights_to_backend(
    const LlamaAttentionLayerWeights& weights,
    runtime::BackendKind backend
) {
    return LlamaAttentionLayerBackendWeights {
        .attention_norm_weight = weights.attention_norm_weight,
        .query_weight = copy_tensor_to_backend(weights.query_weight, backend),
        .key_weight = copy_tensor_to_backend(weights.key_weight, backend),
        .value_weight = copy_tensor_to_backend(weights.value_weight, backend),
        .output_weight = copy_tensor_to_backend(weights.output_weight, backend),
    };
}

LlamaFeedForwardLayerBackendWeights copy_llama_feed_forward_layer_weights_to_backend(
    const LlamaFeedForwardLayerWeights& weights,
    runtime::BackendKind backend
) {
    return LlamaFeedForwardLayerBackendWeights {
        .feed_forward_norm_weight = weights.feed_forward_norm_weight,
        .gate_weight = copy_tensor_to_backend(weights.gate_weight, backend),
        .up_weight = copy_tensor_to_backend(weights.up_weight, backend),
        .down_weight = copy_tensor_to_backend(weights.down_weight, backend),
    };
}

LlamaBlockBackendWeights copy_llama_block_weights_to_backend(
    const LlamaBlockWeights& weights,
    runtime::BackendKind backend
) {
    return LlamaBlockBackendWeights {
        .attention = copy_llama_attention_layer_weights_to_backend(weights.attention, backend),
        .feed_forward = copy_llama_feed_forward_layer_weights_to_backend(weights.feed_forward, backend),
    };
}

TensorStorage run_llama_attention_layer_f32(
    const TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    return run_llama_attention_layer_impl(input, weights, config, position);
}

TensorStorage run_llama_attention_layer_f32(
    const TensorStorage& input,
    const LlamaAttentionLayerBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    return run_llama_attention_layer_impl(input, weights, config, position);
}

TensorStorage run_llama_attention_layer_with_kv_cache_f32(
    const TensorStorage& input,
    const LlamaAttentionLayerWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCacheLayer& cache
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

    const auto attention_query_position_start = query_position_start(position, token_count);
    if (cache.filled_token_count != attention_query_position_start) {
        throw std::invalid_argument("Llama KV cache append position must match filled token count");
    }

    const auto normalized = rms_norm_f32(input, weights.attention_norm_weight, config.rms_norm_epsilon);
    const auto query = matmul_f32_f16_gguf_weight(normalized, weights.query_weight);
    const auto key = matmul_f32_f16_gguf_weight(normalized, weights.key_weight);
    const auto value = matmul_f32_f16_gguf_weight(normalized, weights.value_weight);

    append_rotated_key_value_to_cache_f32(cache, key, value, config, attention_query_position_start);
    const auto cached_token_count = cache.filled_token_count;

    auto projected = TensorStorage::allocate(TensorDescriptor::from(Shape::from({token_count, hidden}), DType::f32));
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(key_length));
    accumulate_cached_attention_projection_f32(
        projected,
        query,
        weights.output_weight,
        cache,
        config,
        token_count,
        cached_token_count,
        attention_query_position_start,
        attention_scale
    );
    return add_f32(input, projected);
}

TensorStorage run_llama_feed_forward_layer_f32(
    const TensorStorage& input,
    const LlamaFeedForwardLayerWeights& weights,
    const LlamaModelConfig& config
) {
    return run_llama_feed_forward_layer_impl(input, weights, config);
}

TensorStorage run_llama_feed_forward_layer_f32(
    const TensorStorage& input,
    const LlamaFeedForwardLayerBackendWeights& weights,
    const LlamaModelConfig& config
) {
    return run_llama_feed_forward_layer_impl(input, weights, config);
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

TensorStorage run_llama_block_f32(
    const TensorStorage& input,
    const LlamaBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    const auto attention_output = run_llama_attention_layer_f32(input, weights.attention, config, position);
    return run_llama_feed_forward_layer_f32(attention_output, weights.feed_forward, config);
}

TensorStorage run_llama_block_with_kv_cache_f32(
    const TensorStorage& input,
    const LlamaBlockWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCacheLayer& cache
) {
    const auto attention_output = run_llama_attention_layer_with_kv_cache_f32(
        input,
        weights.attention,
        config,
        position,
        cache
    );
    return run_llama_feed_forward_layer_f32(attention_output, weights.feed_forward, config);
}

}  // namespace local_gguf_engine::engine
