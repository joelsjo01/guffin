#include "local_gguf_engine/tensor/ops/softmax.hpp"

#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace local_gguf_engine::tensor::ops {

namespace {

void require_supported_input_rank(const TensorDescriptor& descriptor) {
    const auto rank = descriptor.shape.rank();
    if (rank < 1 || rank > 3) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            3,
            static_cast<std::uint64_t>(rank)
        );
    }
}

[[nodiscard]] std::uint64_t softmax_dimension(const TensorDescriptor& descriptor) {
    return descriptor.shape.dimension(descriptor.shape.rank() - 1U);
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

}  // namespace

TensorStorage softmax_f32(const TensorStorage& input) {
    require_f32_operand(input);
    require_supported_input_rank(input.descriptor());

    const auto columns = softmax_dimension(input.descriptor());
    const auto row_count = input.descriptor().element_count / columns;
    auto output = TensorStorage::allocate(input.descriptor());

    for (std::uint64_t row = 0; row < row_count; ++row) {
        const auto row_offset = row * columns;
        auto max_value = read_f32_linear(input, row_offset);
        for (std::uint64_t column = 1; column < columns; ++column) {
            max_value = std::max(max_value, read_f32_linear(input, row_offset + column));
        }

        double exp_sum = 0.0;
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto exp_value = std::exp(static_cast<double>(read_f32_linear(input, row_offset + column) - max_value));
            exp_sum += exp_value;
            write_f32_linear(output, row_offset + column, static_cast<float>(exp_value));
        }

        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto exp_value = static_cast<double>(read_f32_linear(output, row_offset + column));
            write_f32_linear(output, row_offset + column, static_cast<float>(exp_value / exp_sum));
        }
    }

    return output;
}

}  // namespace local_gguf_engine::tensor::ops
