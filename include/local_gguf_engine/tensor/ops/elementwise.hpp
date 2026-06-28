#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage add_f32(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] BackendBuffer add_f32_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
);
[[nodiscard]] TensorStorage multiply_f32(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage silu_f32(const TensorStorage& input);
[[nodiscard]] BackendBuffer silu_multiply_f32_to_backend(
    const BackendBuffer& gate,
    const BackendBuffer& up
);
[[nodiscard]] TensorStorage softcap_f32(const TensorStorage& input, float softcap);
[[nodiscard]] BackendBuffer softcap_f32_to_backend(const BackendBuffer& input, float softcap);
[[nodiscard]] TensorStorage copy_f32(const TensorStorage& input);
[[nodiscard]] TensorStorage fill_f32(const TensorDescriptor& descriptor, float value);

}  // namespace local_gguf_engine::tensor::ops
