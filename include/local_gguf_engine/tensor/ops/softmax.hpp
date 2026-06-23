#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage softmax_f32(const TensorStorage& input);
[[nodiscard]] BackendBuffer softmax_f32_to_backend(const BackendBuffer& input);

}  // namespace local_gguf_engine::tensor::ops
