#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <string>

namespace local_gguf_engine::tensor::backends::cpu {

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string unavailable_reason();

class CpuBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    explicit CpuBackendBufferStorage(const TensorStorage& storage);

    [[nodiscard]] runtime::BackendKind backend() const noexcept override;
    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override;
    [[nodiscard]] const TensorStorage& storage() const noexcept;

private:
    TensorStorage storage_;
};

[[nodiscard]] const CpuBackendBufferStorage& require_cpu_backend_buffer(
    const detail::BackendBufferStorage& storage
);
[[nodiscard]] BackendBuffer copy_tensor_to_cpu_backend(const TensorStorage& storage);
[[nodiscard]] TensorStorage copy_tensor_from_cpu_backend(
    const detail::BackendBufferStorage& storage
);

[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
);
[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight_with_cpu_right_buffer(
    const TensorStorage& left,
    const CpuBackendBufferStorage& right
);
[[nodiscard]] BackendBuffer matmul_f32_f16_gguf_weight_with_cpu_buffers(
    const CpuBackendBufferStorage& left,
    const CpuBackendBufferStorage& right
);
[[nodiscard]] ops::F32F16GGUFWeightTripleMatmulBackendBuffers
matmul_f32_f16_gguf_weight_triple_with_cpu_buffers(
    const CpuBackendBufferStorage& left,
    const CpuBackendBufferStorage& first_right,
    const CpuBackendBufferStorage& second_right,
    const CpuBackendBufferStorage& third_right
);
[[nodiscard]] BackendBuffer silu_multiply_f32_with_cpu_buffers(
    const CpuBackendBufferStorage& gate,
    const CpuBackendBufferStorage& up
);
[[nodiscard]] BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
    const CpuBackendBufferStorage& left,
    const CpuBackendBufferStorage& gate_weight,
    const CpuBackendBufferStorage& up_weight,
    const CpuBackendBufferStorage& down_weight
);
[[nodiscard]] BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
    const CpuBackendBufferStorage& input,
    const CpuBackendBufferStorage& attention_output,
    const CpuBackendBufferStorage& attention_output_weight,
    const CpuBackendBufferStorage& feed_forward_norm_weight,
    const CpuBackendBufferStorage& gate_weight,
    const CpuBackendBufferStorage& up_weight,
    const CpuBackendBufferStorage& down_weight,
    float epsilon
);

}  // namespace local_gguf_engine::tensor::backends::cpu
