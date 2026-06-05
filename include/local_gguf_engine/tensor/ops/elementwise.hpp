#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage add_f32(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage copy_f32(const TensorStorage& input);
[[nodiscard]] TensorStorage fill_f32(const TensorDescriptor& descriptor, float value);

}  // namespace local_gguf_engine::tensor::ops
