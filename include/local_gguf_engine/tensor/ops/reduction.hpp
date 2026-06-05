#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] float sum_f32(const TensorStorage& input);
[[nodiscard]] float sum_squares_f32(const TensorStorage& input);
[[nodiscard]] float mean_square_f32(const TensorStorage& input);
[[nodiscard]] float rms_f32(const TensorStorage& input);

}  // namespace local_gguf_engine::tensor::ops
