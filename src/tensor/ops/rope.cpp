#include "local_gguf_engine/tensor/ops/rope.hpp"

#include "validation.hpp"

#include <cmath>
#include <cstring>
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

}  // namespace

TensorStorage rope_f32(
    const TensorStorage& input,
    std::uint64_t position,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    require_f32_operand(input);
    require_supported_input_rank(input.descriptor());
    require_valid_rope_dimension_count(rope_dimension_count);
    require_rope_dimension_within_hidden(hidden_dimension(input.descriptor()), rope_dimension_count);
    require_positive_frequency_base(frequency_base);

    const auto hidden = hidden_dimension(input.descriptor());
    const auto row_count = input.descriptor().element_count / hidden;
    auto output = TensorStorage::from_bytes(input.descriptor(), input.bytes());

    for (std::uint64_t row = 0; row < row_count; ++row) {
        const auto row_offset = row * hidden;
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

}  // namespace local_gguf_engine::tensor::ops
