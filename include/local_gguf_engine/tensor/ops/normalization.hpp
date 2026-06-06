#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage rms_norm_f32(const TensorStorage& input, const TensorStorage& weight, float epsilon);

}  // namespace local_gguf_engine::tensor::ops
