#include "local_gguf_engine/tensor/ops/rope.hpp"

#include "validation.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::tensor::ops {

namespace {

void require_supported_input_rank(const TensorDescriptor& descriptor) {
    const auto rank = descriptor.shape.rank();
    if (rank != 1 && rank != 2) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            2,
            static_cast<std::uint64_t>(rank)
        );
    }
}

[[nodiscard]] std::uint64_t hidden_dimension(const TensorDescriptor& descriptor) {
    if (descriptor.shape.rank() == 1) {
        return descriptor.shape.dimension(0);
    }

    return descriptor.shape.dimension(1);
}

void require_matrix_descriptor(const TensorDescriptor& descriptor) {
    constexpr std::size_t matrix_rank = 2;
    if (descriptor.shape.rank() != matrix_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(matrix_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_valid_rope_dimension_count(std::uint64_t rope_dimension_count) {
    if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 2, rope_dimension_count);
    }
}

void require_rope_dimension_within_hidden(std::uint64_t hidden, std::uint64_t rope_dimension_count) {
    if (rope_dimension_count > hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, rope_dimension_count);
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

void require_positive_head_layout(std::uint64_t head_count, std::uint64_t head_dimension) {
    if (head_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, head_count);
    }
    if (head_dimension == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, head_dimension);
    }
}

void require_head_width(
    const TensorDescriptor& descriptor,
    std::uint64_t head_count,
    std::uint64_t head_dimension
) {
    if (head_count > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
        throw TensorError(TensorErrorCode::element_count_overflow, head_count, head_dimension);
    }

    const auto expected_width = head_count * head_dimension;
    const auto actual_width = descriptor.shape.dimension(1);
    if (actual_width != expected_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_width, actual_width);
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

[[nodiscard]] double rope_angle(
    std::uint64_t position,
    std::uint64_t pair_offset,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto exponent = static_cast<double>(pair_offset) / static_cast<double>(rope_dimension_count);
    return static_cast<double>(position) / std::pow(static_cast<double>(frequency_base), exponent);
}

[[nodiscard]] TensorStorage rope_impl_f32(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    bool rowwise_positions
) {
    require_f32_operand(input);
    require_supported_input_rank(input.descriptor());
    require_valid_rope_dimension_count(rope_dimension_count);
    require_rope_dimension_within_hidden(hidden_dimension(input.descriptor()), rope_dimension_count);
    require_positive_frequency_base(frequency_base);

    const auto hidden = hidden_dimension(input.descriptor());
    const auto row_count = input.descriptor().element_count / hidden;
    if (rowwise_positions) {
        require_row_position_window(row_count, position_start);
    }

    auto output = TensorStorage::from_bytes(input.descriptor(), input.bytes());

    for (std::uint64_t row = 0; row < row_count; ++row) {
        const auto row_offset = row * hidden;
        const auto position = rowwise_positions ? position_start + row : position_start;
        for (std::uint64_t pair_offset = 0; pair_offset < rope_dimension_count; pair_offset += 2) {
            const auto angle = rope_angle(position, pair_offset, rope_dimension_count, frequency_base);
            const auto angle_cos = static_cast<float>(std::cos(angle));
            const auto angle_sin = static_cast<float>(std::sin(angle));
            const auto first_index = row_offset + pair_offset;
            const auto second_index = first_index + 1;
            const auto first = read_f32_linear(input, first_index);
            const auto second = read_f32_linear(input, second_index);

            write_f32_linear(output, first_index, first * angle_cos - second * angle_sin);
            write_f32_linear(output, second_index, first * angle_sin + second * angle_cos);
        }
    }

    return output;
}

[[nodiscard]] TensorStorage rope_by_heads_impl_f32(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    require_f32_operand(input);
    require_matrix_descriptor(input.descriptor());
    require_positive_head_layout(head_count, head_dimension);
    require_head_width(input.descriptor(), head_count, head_dimension);
    require_valid_rope_dimension_count(rope_dimension_count);
    require_rope_dimension_within_hidden(head_dimension, rope_dimension_count);
    require_positive_frequency_base(frequency_base);

    const auto row_count = input.descriptor().shape.dimension(0);
    require_row_position_window(row_count, position_start);
    auto output = TensorStorage::from_bytes(input.descriptor(), input.bytes());
    const auto row_width = head_count * head_dimension;

    for (std::uint64_t row = 0; row < row_count; ++row) {
        const auto position = position_start + row;
        const auto row_offset = row * row_width;
        for (std::uint64_t head = 0; head < head_count; ++head) {
            const auto head_offset = row_offset + head * head_dimension;
            for (std::uint64_t pair_offset = 0; pair_offset < rope_dimension_count; pair_offset += 2) {
                const auto angle = rope_angle(position, pair_offset, rope_dimension_count, frequency_base);
                const auto angle_cos = static_cast<float>(std::cos(angle));
                const auto angle_sin = static_cast<float>(std::sin(angle));
                const auto first_index = head_offset + pair_offset;
                const auto second_index = first_index + 1;
                const auto first = read_f32_linear(input, first_index);
                const auto second = read_f32_linear(input, second_index);

                write_f32_linear(output, first_index, first * angle_cos - second * angle_sin);
                write_f32_linear(output, second_index, first * angle_sin + second * angle_cos);
            }
        }
    }

    return output;
}

}  // namespace

TensorStorage rope_f32(
    const TensorStorage& input,
    std::uint64_t position,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    return rope_impl_f32(input, position, rope_dimension_count, frequency_base, false);
}

TensorStorage rope_f32_with_position_start(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    return rope_impl_f32(input, position_start, rope_dimension_count, frequency_base, true);
}

TensorStorage rope_f32_by_heads_with_position_start(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    return rope_by_heads_impl_f32(
        input,
        position_start,
        head_count,
        head_dimension,
        rope_dimension_count,
        frequency_base
    );
}

TensorStorage rope_neox_f32_by_heads_with_position_start(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    require_f32_operand(input);
    require_matrix_descriptor(input.descriptor());
    require_positive_head_layout(head_count, head_dimension);
    require_head_width(input.descriptor(), head_count, head_dimension);
    require_valid_rope_dimension_count(rope_dimension_count);
    require_rope_dimension_within_hidden(head_dimension, rope_dimension_count);
    require_positive_frequency_base(frequency_base);
    require_row_position_window(input.descriptor().shape.dimension(0), position_start);

    auto output = TensorStorage::from_bytes(input.descriptor(), input.bytes());
    const auto row_count = input.descriptor().shape.dimension(0);
    const auto row_width = input.descriptor().shape.dimension(1);
    const auto half_dimension = rope_dimension_count / 2U;
    for (std::uint64_t row = 0; row < row_count; ++row) {
        const auto position = position_start + row;
        const auto row_offset = row * row_width;
        for (std::uint64_t head = 0; head < head_count; ++head) {
            const auto head_offset = row_offset + head * head_dimension;
            for (std::uint64_t pair = 0; pair < half_dimension; ++pair) {
                const auto frequency_offset = pair * 2U;
                const auto angle = rope_angle(position, frequency_offset, rope_dimension_count, frequency_base);
                const auto angle_cos = static_cast<float>(std::cos(angle));
                const auto angle_sin = static_cast<float>(std::sin(angle));
                const auto first_index = head_offset + pair;
                const auto second_index = first_index + half_dimension;
                const auto first = read_f32_linear(input, first_index);
                const auto second = read_f32_linear(input, second_index);
                write_f32_linear(output, first_index, first * angle_cos - second * angle_sin);
                write_f32_linear(output, second_index, first * angle_sin + second * angle_cos);
            }
        }
    }
    return output;
}

}  // namespace local_gguf_engine::tensor::ops
