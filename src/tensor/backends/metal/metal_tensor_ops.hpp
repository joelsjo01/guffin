#pragma once

#include "metal_buffer.hpp"

#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"

#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor::backends::metal {

[[nodiscard]] std::uint32_t require_u32_dimension(std::uint64_t value, const char* name);

void require_matrix_descriptor(const TensorDescriptor& descriptor);
void require_f32_f16_matmul_gguf_weight_operands(
    const TensorStorage& left,
    const TensorDescriptor& right
);
void require_f32_f16_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
);
void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorStorage& left,
    const TensorDescriptor& right
);
void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
);
void require_same_backend(
    runtime::BackendKind left,
    runtime::BackendKind right,
    const char* operation
);
void require_silu_multiply_f32_operands(
    const TensorDescriptor& gate,
    const TensorDescriptor& up
);
void require_gated_silu_projection_f32_f16_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight
);
void require_f32_vector_descriptor(const TensorDescriptor& descriptor);
void require_positive_epsilon(float epsilon);
void require_rms_norm_f32_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& weight,
    float epsilon
);
void require_rope_f32_by_heads_operands(
    const TensorDescriptor& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
);
void require_causal_grouped_attention_scores_f32_operands(
    const TensorDescriptor& query,
    const TensorDescriptor& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
);
void require_causal_grouped_attention_values_f32_operands(
    const TensorDescriptor& probabilities,
    const TensorDescriptor& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
);
void require_softmax_f32_operand(const TensorDescriptor& input);
void require_softcap_f32_operand(const TensorDescriptor& input, float softcap);
void require_top_logit_f32_operand(const TensorDescriptor& input);
void require_positive_finite_scale(float scale);
void require_positive_frequency_base(float frequency_base);
void require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& attention_output,
    const TensorDescriptor& attention_output_weight,
    const TensorDescriptor& feed_forward_norm_weight,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight,
    float epsilon
);
void require_f32_f16_matmul_gguf_weight_triple_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& first_right,
    const TensorDescriptor& second_right,
    const TensorDescriptor& third_right
);

[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight_with_metal_right_buffer(
    const TensorStorage& left,
    const MetalBackendBufferStorage& right
);
[[nodiscard]] BackendBuffer matmul_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
);
[[nodiscard]] BackendBuffer embed_f16_gguf_weight_token_ids_with_metal_buffer(
    std::span<const std::uint32_t> token_ids,
    const MetalBackendBufferStorage& embedding_weight,
    float scale
);
[[nodiscard]] BackendBuffer rms_norm_f32_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& weight,
    float epsilon
);
[[nodiscard]] BackendBuffer rope_f32_by_heads_with_metal_buffer(
    const MetalBackendBufferStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
);
[[nodiscard]] BackendBuffer causal_grouped_attention_scores_f32_with_metal_buffers(
    const MetalBackendBufferStorage& query,
    const MetalBackendBufferStorage& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
);
[[nodiscard]] BackendBuffer causal_grouped_attention_values_f32_with_metal_buffers(
    const MetalBackendBufferStorage& probabilities,
    const MetalBackendBufferStorage& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
);
[[nodiscard]] BackendBuffer softmax_f32_with_metal_buffer(
    const MetalBackendBufferStorage& input
);
[[nodiscard]] BackendBuffer softcap_f32_with_metal_buffer(
    const MetalBackendBufferStorage& input,
    float softcap
);
[[nodiscard]] ops::TopLogit top_logit_f32_with_metal_buffer(
    const MetalBackendBufferStorage& input
);
[[nodiscard]] BackendBuffer matmul_f32_quantized_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
);
[[nodiscard]] ops::F32F16GGUFWeightTripleMatmulBackendBuffers
matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& first_right,
    const MetalBackendBufferStorage& second_right,
    const MetalBackendBufferStorage& third_right
);
[[nodiscard]] BackendBuffer silu_multiply_f32_with_metal_buffers(
    const MetalBackendBufferStorage& gate,
    const MetalBackendBufferStorage& up
);
[[nodiscard]] BackendBuffer add_f32_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
);
[[nodiscard]] BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight
);
[[nodiscard]] BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& attention_output,
    const MetalBackendBufferStorage& attention_output_weight,
    const MetalBackendBufferStorage& feed_forward_norm_weight,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight,
    float epsilon
);

}  // namespace local_gguf_engine::tensor::backends::metal
