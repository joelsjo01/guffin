#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage rms_norm_f32(const TensorStorage& input, const TensorStorage& weight, float epsilon);
[[nodiscard]] BackendBuffer rms_norm_f32_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    float epsilon
);

}  // namespace local_gguf_engine::tensor::ops
