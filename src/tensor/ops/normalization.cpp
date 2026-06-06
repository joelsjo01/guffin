#include "local_gguf_engine/tensor/ops/normalization.hpp"

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

void require_weight_rank(const TensorDescriptor& descriptor) {
    if (descriptor.shape.rank() != 1) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            1,
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

[[nodiscard]] std::uint64_t hidden_dimension(const TensorDescriptor& descriptor) {
    if (descriptor.shape.rank() == 1) {
        return descriptor.shape.dimension(0);
    }

    return descriptor.shape.dimension(1);
}

void require_matching_hidden_dimension(const TensorDescriptor& input, const TensorDescriptor& weight) {
    const auto input_hidden = hidden_dimension(input);
    const auto weight_hidden = weight.shape.dimension(0);
    if (input_hidden != weight_hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, input_hidden, weight_hidden);
    }
}

void require_positive_epsilon(float epsilon) {
    if (!(epsilon > 0.0F)) {
        throw std::invalid_argument("RMSNorm epsilon must be positive");
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

}  // namespace

TensorStorage rms_norm_f32(const TensorStorage& input, const TensorStorage& weight, float epsilon) {
    require_f32_operand(input);
    require_f32_operand(weight);
    require_supported_input_rank(input.descriptor());
    require_weight_rank(weight.descriptor());
    require_matching_hidden_dimension(input.descriptor(), weight.descriptor());
    require_positive_epsilon(epsilon);

    const auto hidden = hidden_dimension(input.descriptor());
    const auto row_count = input.descriptor().element_count / hidden;
    auto output = TensorStorage::allocate(input.descriptor());

    for (std::uint64_t row = 0; row < row_count; ++row) {
        float sum_squares = 0.0F;
        for (std::uint64_t column = 0; column < hidden; ++column) {
            const auto value = read_f32_linear(input, row * hidden + column);
            sum_squares += value * value;
        }

        const auto mean_square = sum_squares / static_cast<float>(hidden);
        const auto inverse_rms = 1.0F / std::sqrt(mean_square + epsilon);
        for (std::uint64_t column = 0; column < hidden; ++column) {
            const auto input_value = read_f32_linear(input, row * hidden + column);
            const auto weight_value = read_f32_linear(weight, column);
            write_f32_linear(output, row * hidden + column, input_value * inverse_rms * weight_value);
        }
    }

    return output;
}

}  // namespace local_gguf_engine::tensor::ops
