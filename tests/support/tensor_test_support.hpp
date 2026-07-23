#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

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
local_gguf_engine::tensor::TensorStorage q4_k_reference_tensor();
local_gguf_engine::tensor::TensorStorage q6_k_reference_tensor();
local_gguf_engine::tensor::TensorStorage tensor_q4_k_gguf_weight_matrix(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint64_t seed = 0
);
local_gguf_engine::tensor::TensorStorage tensor_q4_k_stress_gguf_weight_matrix(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint64_t seed = 0
);
local_gguf_engine::tensor::TensorStorage tensor_q6_k_gguf_weight_matrix(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint64_t seed = 0
);
local_gguf_engine::tensor::TensorStorage tensor_q6_k_stress_gguf_weight_matrix(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint64_t seed = 0
);
std::vector<float> tensor_f32_projection_values(std::uint64_t rows, std::uint64_t columns);
local_gguf_engine::tensor::TensorStorage reference_f32_quantized_gguf_weight_matmul(
    const local_gguf_engine::tensor::TensorStorage& left,
    const local_gguf_engine::tensor::TensorStorage& right
);
local_gguf_engine::tensor::TensorStorage reference_f16_staged_quantized_gguf_weight_matmul(
    const local_gguf_engine::tensor::TensorStorage& left,
    const local_gguf_engine::tensor::TensorStorage& right
);

}  // namespace local_gguf_engine::tests
