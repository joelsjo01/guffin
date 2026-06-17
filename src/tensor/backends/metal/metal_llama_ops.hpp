#pragma once

#include "metal_buffer.hpp"
#include "metal_context.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor::backends::metal {

struct MatmulDimensions {
    std::uint32_t rows = 0;
    std::uint32_t shared = 0;
    std::uint32_t columns = 0;
};

struct ElementwiseDimensions {
    std::uint32_t elements = 0;
};

struct RMSNormDimensions {
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
    float epsilon = 0.0F;
};

struct LlamaCachedAttentionDimensions {
    std::uint32_t token_count = 0;
    std::uint32_t cache_position = 0;
    std::uint32_t cached_token_count = 0;
    std::uint32_t head_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t key_length = 0;
    std::uint32_t value_length = 0;
    std::uint32_t rope_dimension_count = 0;
    float frequency_base = 0.0F;
    float scale = 0.0F;
};

[[nodiscard]] id<MTLBuffer> reusable_scratch_buffer(
    id<MTLDevice> device,
    __strong id<MTLBuffer>& buffer,
    std::uint64_t byte_count,
    const char* label
);

void require_llama_cached_attention_f32_operands(
    const TensorDescriptor& query,
    const TensorDescriptor& key,
    const TensorDescriptor& value,
    const TensorDescriptor& cache_key,
    const TensorDescriptor& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
);

void encode_llama_cached_qkv_projection(
    MatmulPipelineState& state,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input_buffer,
    id<MTLBuffer> query_weight_buffer,
    id<MTLBuffer> key_weight_buffer,
    id<MTLBuffer> value_weight_buffer,
    id<MTLBuffer> query_buffer,
    id<MTLBuffer> key_buffer,
    id<MTLBuffer> value_buffer,
    const MatmulDimensions& query_dims,
    const MatmulDimensions& key_dims,
    const MatmulDimensions& value_dims,
    std::uint64_t rows,
    std::uint64_t hidden,
    std::uint64_t query_width,
    std::uint64_t key_width,
    std::uint64_t value_width,
    const char* error_context
);

void encode_llama_cached_attention_output(
    MatmulPipelineState& state,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> query_buffer,
    id<MTLBuffer> cache_key_buffer,
    id<MTLBuffer> cache_value_buffer,
    id<MTLBuffer> output_buffer,
    const LlamaCachedAttentionDimensions& dims,
    std::uint64_t output_width
);

void encode_llama_cached_block_tail(
    MatmulPipelineState& state,
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> input_buffer,
    id<MTLBuffer> attention_output_buffer,
    id<MTLBuffer> attention_output_weight_buffer,
    id<MTLBuffer> feed_forward_norm_weight_buffer,
    id<MTLBuffer> gate_weight_buffer,
    id<MTLBuffer> up_weight_buffer,
    id<MTLBuffer> down_weight_buffer,
    id<MTLBuffer> attention_residual_buffer,
    id<MTLBuffer> feed_forward_normalized_buffer,
    id<MTLBuffer> gate_buffer,
    id<MTLBuffer> up_buffer,
    id<MTLBuffer> gated_buffer,
    id<MTLBuffer> output_buffer,
    const RMSNormDimensions& rms_norm_dims,
    const MatmulDimensions& attention_output_dims,
    const MatmulDimensions& gate_dims,
    const MatmulDimensions& down_dims,
    const ElementwiseDimensions& feed_forward_elementwise_dims,
    std::uint64_t rows,
    std::uint64_t hidden,
    std::uint64_t attention_output_width,
    std::uint64_t feed_forward,
    float rms_norm_epsilon,
    const char* error_context
);

}  // namespace local_gguf_engine::tensor::backends::metal
