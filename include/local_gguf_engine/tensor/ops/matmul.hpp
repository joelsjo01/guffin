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
[[nodiscard]] BackendBuffer llama_cached_block_f32_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_norm_weight,
    const BackendBuffer& query_weight,
    const BackendBuffer& key_weight,
    const BackendBuffer& value_weight,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float attention_scale,
    float rms_norm_epsilon
);

}  // namespace local_gguf_engine::tensor::ops
