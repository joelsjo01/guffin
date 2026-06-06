#include "local_gguf_engine/tensor/ops/matmul.hpp"

#include "validation.hpp"

#include "local_gguf_engine/tensor/f16.hpp"

#include <cstring>

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

void require_matmul_inner_dimensions(const TensorDescriptor& left, const TensorDescriptor& right) {
    const auto left_columns = left.shape.dimension(1);
    const auto right_rows = right.shape.dimension(0);
    if (left_columns != right_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_columns, right_rows);
    }
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

template <typename RightReader>
[[nodiscard]] TensorStorage matmul_f32_with_right_reader(
    const TensorStorage& left,
    const TensorStorage& right,
    RightReader read_right
) {
    require_matrix_descriptor(left.descriptor());
    require_matrix_descriptor(right.descriptor());
    require_matmul_inner_dimensions(left.descriptor(), right.descriptor());

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::uint64_t inner = 0; inner < shared; ++inner) {
                const auto left_index = row * shared + inner;
                const auto right_index = inner * columns + column;
                sum += read_f32_linear(left, left_index) * read_right(right, right_index);
            }

            write_f32_linear(output, row * columns + column, sum);
        }
    }

    return output;
}

}  // namespace

TensorStorage matmul_f32(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f32_operand(right);
    return matmul_f32_with_right_reader(left, right, read_f32_linear);
}

TensorStorage matmul_f32_f16(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f16_operand(right);
    return matmul_f32_with_right_reader(left, right, read_f16_linear_as_f32);
}

}  // namespace local_gguf_engine::tensor::ops
