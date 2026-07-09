#pragma once

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor::ops {

struct F32F16GGUFWeightTripleMatmulBackendBuffers {
    BackendBuffer first;
    BackendBuffer second;
    BackendBuffer third;
};

[[nodiscard]] TensorStorage matmul_f32(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage matmul_f32_f16(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const TensorStorage& right);
[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right,
    runtime::BackendKind backend
);
[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const BackendBuffer& right
);
[[nodiscard]] TensorStorage matmul_f32_quantized_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
);
[[nodiscard]] TensorStorage matmul_f32_quantized_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right,
    runtime::BackendKind backend
);
[[nodiscard]] BackendBuffer matmul_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
);
[[nodiscard]] BackendBuffer matmul_f32_quantized_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
);
[[nodiscard]] F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& first_right,
    const BackendBuffer& second_right,
    const BackendBuffer& third_right
);
[[nodiscard]] BackendBuffer gated_silu_projection_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight
);
[[nodiscard]] BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    float epsilon
);
}  // namespace local_gguf_engine::tensor::ops
