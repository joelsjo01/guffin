#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage matmul_f32(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage matmul_f32_f16(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const TensorStorage& right);

}  // namespace local_gguf_engine::tensor::ops
