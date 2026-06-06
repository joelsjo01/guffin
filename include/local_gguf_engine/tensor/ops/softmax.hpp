#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage softmax_f32(const TensorStorage& input);

}  // namespace local_gguf_engine::tensor::ops
