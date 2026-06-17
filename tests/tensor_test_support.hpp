#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace local_gguf_engine::tests {

void expect_close(float actual, float expected, float tolerance, std::string_view message);

local_gguf_engine::tensor::TensorStorage tensor_f32_vector(std::span<const float> values);
local_gguf_engine::tensor::TensorStorage tensor_f32_matrix(
    std::uint64_t rows,
    std::uint64_t columns,
    std::span<const float> values
);
local_gguf_engine::tensor::TensorStorage tensor_f16_matrix(
    std::uint64_t rows,
    std::uint64_t columns,
    std::span<const float> values
);
local_gguf_engine::tensor::TensorStorage tensor_f16_gguf_matrix(
    std::uint64_t rows,
    std::uint64_t columns,
    std::span<const float> values
);

}  // namespace local_gguf_engine::tests
