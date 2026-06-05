#include "local_gguf_engine/tensor/ops/reduction.hpp"

#include "validation.hpp"

#include <cmath>
#include <cstring>

namespace local_gguf_engine::tensor::ops {

namespace {

[[nodiscard]] float read_f32_byte_offset(const TensorStorage& storage, std::size_t byte_offset) {
    float value = 0.0F;
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

}  // namespace

float sum_f32(const TensorStorage& input) {
    require_f32_operand(input);

    float sum = 0.0F;
    for (std::uint64_t index = 0; index < input.descriptor().element_count; ++index) {
        sum += read_f32_byte_offset(input, static_cast<std::size_t>(index * sizeof(float)));
    }

    return sum;
}

float sum_squares_f32(const TensorStorage& input) {
    require_f32_operand(input);

    float sum = 0.0F;
    for (std::uint64_t index = 0; index < input.descriptor().element_count; ++index) {
        const auto value = read_f32_byte_offset(input, static_cast<std::size_t>(index * sizeof(float)));
        sum += value * value;
    }

    return sum;
}

float mean_square_f32(const TensorStorage& input) {
    require_f32_operand(input);
    return sum_squares_f32(input) / static_cast<float>(input.descriptor().element_count);
}

float rms_f32(const TensorStorage& input) {
    return std::sqrt(mean_square_f32(input));
}

}  // namespace local_gguf_engine::tensor::ops
