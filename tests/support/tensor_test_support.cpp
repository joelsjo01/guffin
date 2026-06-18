#include "tensor_test_support.hpp"

#include "test_support.hpp"

#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>

namespace local_gguf_engine::tests {

using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::f32_to_f16_bits;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::write_f16_from_f32;
using local_gguf_engine::tensor::write_f32;

void expect_close(float actual, float expected, float tolerance, std::string_view message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream stream;
        stream << message << ": expected " << expected << ", got " << actual;
        throw TestFailure(stream.str());
    }
}

TensorStorage tensor_f32_vector(std::span<const float> values) {
    auto storage = TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({static_cast<std::uint64_t>(values.size())}), DType::f32)
    );
    for (std::uint64_t index = 0; index < values.size(); ++index) {
        const std::array<std::uint64_t, 1> tensor_index {index};
        write_f32(storage, std::span<const std::uint64_t>(tensor_index), values[static_cast<std::size_t>(index)]);
    }

    return storage;
}

TensorStorage tensor_f32_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F32 matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> tensor_index {row, column};
            write_f32(
                storage,
                std::span<const std::uint64_t>(tensor_index),
                values[static_cast<std::size_t>(row * columns + column)]
            );
        }
    }

    return storage;
}

TensorStorage tensor_f16_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F16 matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> tensor_index {row, column};
            write_f16_from_f32(
                storage,
                std::span<const std::uint64_t>(tensor_index),
                values[static_cast<std::size_t>(row * columns + column)]
            );
        }
    }

    return storage;
}

TensorStorage tensor_f16_gguf_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F16 GGUF matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    auto bytes = storage.mutable_bytes();
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto bits = f32_to_f16_bits(values[static_cast<std::size_t>(row * columns + column)]);
            const auto byte_offset = static_cast<std::size_t>((column * rows + row) * sizeof(std::uint16_t));
            bytes[byte_offset] = static_cast<std::uint8_t>(bits & 0xFFU);
            bytes[byte_offset + 1] = static_cast<std::uint8_t>(bits >> 8U);
        }
    }

    return storage;
}

}  // namespace local_gguf_engine::tests
