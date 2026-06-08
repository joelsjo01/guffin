#include "local_gguf_engine/tensor/ops/attention.hpp"

#include "validation.hpp"

#include "local_gguf_engine/tensor/ops/softmax.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::tensor::ops {

namespace {

constexpr std::size_t matrix_rank = 2;

void require_matrix_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.shape.rank() != matrix_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(matrix_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_query_key_hidden_match(const TensorDescriptor& query, const TensorDescriptor& key) {
    const auto query_hidden = query.shape.dimension(1);
    const auto key_hidden = key.shape.dimension(1);
    if (query_hidden != key_hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_hidden, key_hidden);
    }
}

void require_key_value_row_match(const TensorDescriptor& key, const TensorDescriptor& value) {
    const auto key_rows = key.shape.dimension(0);
    const auto value_rows = value.shape.dimension(0);
    if (key_rows != value_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_rows, value_rows);
    }
}

void require_column_view_range(
    const TensorDescriptor& descriptor,
    std::uint64_t row_count,
    std::uint64_t first_column,
    std::uint64_t column_count
) {
    const auto rows = descriptor.shape.dimension(0);
    if (row_count == 0 || row_count > rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, rows, row_count);
    }

    const auto columns = descriptor.shape.dimension(1);
    const auto end_column = first_column + column_count;
    if (column_count == 0 || end_column < first_column || end_column > columns) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, columns, end_column);
    }
}

void require_query_key_column_view_hidden_match(
    const TensorDescriptor& query,
    const F32MatrixColumnView& key
) {
    const auto query_hidden = query.shape.dimension(1);
    if (query_hidden != key.column_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_hidden, key.column_count);
    }
}

void require_key_value_column_view_row_match(
    const F32MatrixColumnView& key,
    const F32MatrixColumnView& value
) {
    if (key.row_count != value.row_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key.row_count, value.row_count);
    }
}

void require_positive_finite_scale(float scale) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("attention scale must be positive and finite");
    }
}

void require_causal_position_window(
    std::uint64_t query_rows,
    std::uint64_t key_rows,
    std::uint64_t query_position_start
) {
    if (query_position_start > std::numeric_limits<std::uint64_t>::max() - query_rows) {
        throw std::invalid_argument("attention causal query positions overflow");
    }

    const auto query_position_end = query_position_start + query_rows;
    if (query_position_end > key_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_rows, query_position_end);
    }
}

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

[[nodiscard]] float read_f32_column_view(
    const F32MatrixColumnView& view,
    std::uint64_t row,
    std::uint64_t column
) {
    const auto storage_columns = view.storage.descriptor().shape.dimension(1);
    return read_f32_linear(view.storage, row * storage_columns + view.first_column + column);
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
}

[[nodiscard]] TensorStorage scaled_dot_product_attention_impl_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    const TensorStorage& value,
    float scale,
    bool causal,
    std::uint64_t query_position_start
) {
    require_f32_operand(query);
    require_f32_operand(key);
    require_f32_operand(value);
    require_matrix_descriptor(query.descriptor());
    require_matrix_descriptor(key.descriptor());
    require_matrix_descriptor(value.descriptor());
    require_query_key_hidden_match(query.descriptor(), key.descriptor());
    require_key_value_row_match(key.descriptor(), value.descriptor());
    require_positive_finite_scale(scale);

    const auto query_rows = query.descriptor().shape.dimension(0);
    const auto key_rows = key.descriptor().shape.dimension(0);
    const auto hidden = query.descriptor().shape.dimension(1);
    const auto value_hidden = value.descriptor().shape.dimension(1);
    if (causal) {
        require_causal_position_window(query_rows, key_rows, query_position_start);
    }

    auto scores = TensorStorage::allocate(TensorDescriptor::from(Shape::from({query_rows, key_rows}), DType::f32));
    for (std::uint64_t query_row = 0; query_row < query_rows; ++query_row) {
        for (std::uint64_t key_row = 0; key_row < key_rows; ++key_row) {
            if (causal && key_row > query_position_start + query_row) {
                write_f32_linear(scores, query_row * key_rows + key_row, -std::numeric_limits<float>::infinity());
                continue;
            }

            float dot = 0.0F;
            for (std::uint64_t column = 0; column < hidden; ++column) {
                dot += read_f32_linear(query, query_row * hidden + column)
                    * read_f32_linear(key, key_row * hidden + column);
            }

            write_f32_linear(scores, query_row * key_rows + key_row, dot * scale);
        }
    }

    const auto probabilities = softmax_f32(scores);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({query_rows, value_hidden}), DType::f32));

    for (std::uint64_t query_row = 0; query_row < query_rows; ++query_row) {
        for (std::uint64_t value_column = 0; value_column < value_hidden; ++value_column) {
            float sum = 0.0F;
            for (std::uint64_t key_row = 0; key_row < key_rows; ++key_row) {
                sum += read_f32_linear(probabilities, query_row * key_rows + key_row)
                    * read_f32_linear(value, key_row * value_hidden + value_column);
            }

            write_f32_linear(output, query_row * value_hidden + value_column, sum);
        }
    }

    return output;
}

[[nodiscard]] TensorStorage causal_scaled_dot_product_attention_column_view_impl_f32(
    const TensorStorage& query,
    F32MatrixColumnView key,
    F32MatrixColumnView value,
    float scale,
    std::uint64_t query_position_start
) {
    require_f32_operand(query);
    require_f32_operand(key.storage);
    require_f32_operand(value.storage);
    require_matrix_descriptor(query.descriptor());
    require_matrix_descriptor(key.storage.descriptor());
    require_matrix_descriptor(value.storage.descriptor());
    require_column_view_range(key.storage.descriptor(), key.row_count, key.first_column, key.column_count);
    require_column_view_range(value.storage.descriptor(), value.row_count, value.first_column, value.column_count);
    require_query_key_column_view_hidden_match(query.descriptor(), key);
    require_key_value_column_view_row_match(key, value);
    require_positive_finite_scale(scale);

    const auto query_rows = query.descriptor().shape.dimension(0);
    const auto hidden = query.descriptor().shape.dimension(1);
    const auto key_rows = key.row_count;
    const auto value_hidden = value.column_count;
    require_causal_position_window(query_rows, key_rows, query_position_start);

    auto scores = TensorStorage::allocate(TensorDescriptor::from(Shape::from({query_rows, key_rows}), DType::f32));
    for (std::uint64_t query_row = 0; query_row < query_rows; ++query_row) {
        for (std::uint64_t key_row = 0; key_row < key_rows; ++key_row) {
            if (key_row > query_position_start + query_row) {
                write_f32_linear(scores, query_row * key_rows + key_row, -std::numeric_limits<float>::infinity());
                continue;
            }

            float dot = 0.0F;
            for (std::uint64_t column = 0; column < hidden; ++column) {
                dot += read_f32_linear(query, query_row * hidden + column)
                    * read_f32_column_view(key, key_row, column);
            }

            write_f32_linear(scores, query_row * key_rows + key_row, dot * scale);
        }
    }

    const auto probabilities = softmax_f32(scores);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({query_rows, value_hidden}), DType::f32));

    for (std::uint64_t query_row = 0; query_row < query_rows; ++query_row) {
        for (std::uint64_t value_column = 0; value_column < value_hidden; ++value_column) {
            float sum = 0.0F;
            for (std::uint64_t key_row = 0; key_row < key_rows; ++key_row) {
                sum += read_f32_linear(probabilities, query_row * key_rows + key_row)
                    * read_f32_column_view(value, key_row, value_column);
            }

            write_f32_linear(output, query_row * value_hidden + value_column, sum);
        }
    }

    return output;
}

}  // namespace

TensorStorage scaled_dot_product_attention_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    const TensorStorage& value,
    float scale
) {
    return scaled_dot_product_attention_impl_f32(query, key, value, scale, false, 0);
}

TensorStorage causal_scaled_dot_product_attention_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    const TensorStorage& value,
    float scale,
    std::uint64_t query_position_start
) {
    return scaled_dot_product_attention_impl_f32(query, key, value, scale, true, query_position_start);
}

TensorStorage causal_scaled_dot_product_attention_f32(
    const TensorStorage& query,
    F32MatrixColumnView key,
    F32MatrixColumnView value,
    float scale,
    std::uint64_t query_position_start
) {
    return causal_scaled_dot_product_attention_column_view_impl_f32(
        query,
        key,
        value,
        scale,
        query_position_start
    );
}

}  // namespace local_gguf_engine::tensor::ops
