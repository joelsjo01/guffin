#include "metal_llama_ops.hpp"

#include "metal_tensor_ops.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "../../ops/validation.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

constexpr std::uint64_t llama_cached_decode_attention_thread_count = 256;
constexpr std::uint64_t llama_cached_decode_block_tail_thread_count = 256;
constexpr std::uint64_t llama_cached_decode_qkv_projection_thread_count = 256;

struct LlamaCachedDecodeBlockTailDimensions {
    std::uint32_t hidden = 0;
    std::uint32_t attention_output_width = 0;
    std::uint32_t feed_forward = 0;
    float epsilon = 0.0F;
};

struct LlamaCachedDecodeQKVProjectionDimensions {
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
    std::uint32_t query_width = 0;
    std::uint32_t key_width = 0;
    std::uint32_t value_width = 0;
};

}  // namespace

[[nodiscard]] id<MTLBuffer> reusable_scratch_buffer(
    id<MTLDevice> device,
    __strong id<MTLBuffer>& buffer,
    std::uint64_t byte_count,
    const char* label
) {
    const auto length = static_cast<NSUInteger>(byte_count);
    if (buffer == nil || [buffer length] < length) {
        buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal Llama cached block scratch buffer allocation failed: ") + label
            );
        }
    }

    return buffer;
}

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
) {
    ops::require_f32_descriptor(query);
    ops::require_f32_descriptor(key);
    ops::require_f32_descriptor(value);
    ops::require_f32_descriptor(cache_key);
    ops::require_f32_descriptor(cache_value);
    require_matrix_descriptor(query);
    require_matrix_descriptor(key);
    require_matrix_descriptor(value);
    require_matrix_descriptor(cache_key);
    require_matrix_descriptor(cache_value);

    if (head_count == 0 || kv_head_count == 0 || key_length == 0 || value_length == 0) {
        throw std::invalid_argument("Llama cached attention dimensions must be positive");
    }
    if (head_count % kv_head_count != 0) {
        throw std::invalid_argument("Llama attention head count must be divisible by KV head count");
    }
    if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 2, rope_dimension_count);
    }
    if (rope_dimension_count > key_length) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_length, rope_dimension_count);
    }
    require_positive_frequency_base(frequency_base);
    require_positive_finite_scale(scale);

    const auto token_count = query.shape.dimension(0);
    if (token_count == 0) {
        throw std::invalid_argument("Llama cached attention input rows must be positive");
    }
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto key_rows = key.shape.dimension(0);
    const auto value_rows = value.shape.dimension(0);
    if (key_rows != token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, token_count, key_rows);
    }
    if (value_rows != token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, token_count, value_rows);
    }
    if (query.shape.dimension(1) != query_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_width, query.shape.dimension(1));
    }
    if (key.shape.dimension(1) != key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, key.shape.dimension(1));
    }
    if (value.shape.dimension(1) != value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, value.shape.dimension(1));
    }
    if (cache_key.shape.dimension(1) != key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, cache_key.shape.dimension(1));
    }
    if (cache_value.shape.dimension(1) != value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, cache_value.shape.dimension(1));
    }

    const auto cache_capacity = cache_key.shape.dimension(0);
    if (cache_value.shape.dimension(0) != cache_capacity) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, cache_capacity, cache_value.shape.dimension(0));
    }
    if (cache_position > cache_capacity || token_count > cache_capacity - cache_position) {
        throw std::invalid_argument("Llama backend KV cache append would exceed context length");
    }
}

void require_llama_cached_block_f32_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& attention_norm_weight,
    const TensorDescriptor& query_weight,
    const TensorDescriptor& key_weight,
    const TensorDescriptor& value_weight,
    const TensorDescriptor& cache_key,
    const TensorDescriptor& cache_value,
    const TensorDescriptor& attention_output_weight,
    const TensorDescriptor& feed_forward_norm_weight,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float attention_scale,
    float rms_norm_epsilon
) {
    ops::require_f32_descriptor(input);
    require_matrix_descriptor(input);
    const auto token_count = input.shape.dimension(0);
    const auto hidden = input.shape.dimension(1);
    if (token_count == 0) {
        throw std::invalid_argument("Llama cached block input rows must be positive");
    }

    require_f32_vector_descriptor(attention_norm_weight);
    if (attention_norm_weight.shape.dimension(0) != hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, attention_norm_weight.shape.dimension(0));
    }

    if (head_count == 0 || kv_head_count == 0 || key_length == 0 || value_length == 0) {
        throw std::invalid_argument("Llama cached block attention dimensions must be positive");
    }
    if (head_count % kv_head_count != 0) {
        throw std::invalid_argument("Llama attention head count must be divisible by KV head count");
    }

    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto attention_output_width = head_count * value_length;
    require_f32_f16_matmul_gguf_weight_operands(input, query_weight);
    require_f32_f16_matmul_gguf_weight_operands(input, key_weight);
    require_f32_f16_matmul_gguf_weight_operands(input, value_weight);
    if (query_weight.shape.dimension(1) != query_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_width, query_weight.shape.dimension(1));
    }
    if (key_weight.shape.dimension(1) != key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, key_weight.shape.dimension(1));
    }
    if (value_weight.shape.dimension(1) != value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, value_weight.shape.dimension(1));
    }

    const auto query_descriptor = TensorDescriptor::from(Shape::from({token_count, query_width}), DType::f32);
    const auto key_descriptor = TensorDescriptor::from(Shape::from({token_count, key_width}), DType::f32);
    const auto value_descriptor = TensorDescriptor::from(Shape::from({token_count, value_width}), DType::f32);
    require_llama_cached_attention_f32_operands(
        query_descriptor,
        key_descriptor,
        value_descriptor,
        cache_key,
        cache_value,
        cache_position,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );

    const auto attention_output_descriptor =
        TensorDescriptor::from(Shape::from({token_count, attention_output_width}), DType::f32);
    require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
        input,
        attention_output_descriptor,
        attention_output_weight,
        feed_forward_norm_weight,
        gate_weight,
        up_weight,
        down_weight,
        rms_norm_epsilon
    );
}

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
}

[[nodiscard]] double llama_rope_angle(
    std::uint64_t position,
    std::uint64_t pair_offset,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto exponent = static_cast<double>(pair_offset) / static_cast<double>(rope_dimension_count);
    return static_cast<double>(position) / std::pow(static_cast<double>(frequency_base), exponent);
}

[[nodiscard]] float read_rope_f32_columns(
    const TensorStorage& storage,
    std::uint64_t row,
    std::uint64_t first_column,
    std::uint64_t column,
    std::uint64_t position,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto columns = storage.descriptor().shape.dimension(1);
    if (column >= rope_dimension_count) {
        return read_f32_linear(storage, row * columns + first_column + column);
    }

    const auto pair_offset = column - (column % 2);
    const auto angle = llama_rope_angle(position, pair_offset, rope_dimension_count, frequency_base);
    const auto angle_cos = static_cast<float>(std::cos(angle));
    const auto angle_sin = static_cast<float>(std::sin(angle));
    const auto first = read_f32_linear(storage, row * columns + first_column + pair_offset);
    const auto second = read_f32_linear(storage, row * columns + first_column + pair_offset + 1);

    if (column == pair_offset) {
        return first * angle_cos - second * angle_sin;
    }

    return first * angle_sin + second * angle_cos;
}

[[nodiscard]] bool can_use_llama_cached_decode_qkv_projection_kernel(
    const MatmulPipelineState& state,
    std::uint64_t rows,
    const LlamaCachedDecodeQKVProjectionDimensions& dims
) {
    if (rows != 1 || dims.hidden == 0 || dims.query_width == 0 || dims.key_width == 0 || dims.value_width == 0) {
        return false;
    }

    const auto pipeline_thread_limit = static_cast<std::uint64_t>(
        [state.llama_cached_decode_qkv_projection_pipeline maxTotalThreadsPerThreadgroup]
    );
    return pipeline_thread_limit >= llama_cached_decode_qkv_projection_thread_count;
}

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
) {
    const LlamaCachedDecodeQKVProjectionDimensions decode_qkv_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .query_width = require_u32_dimension(query_width, "query_width"),
        .key_width = require_u32_dimension(key_width, "key_width"),
        .value_width = require_u32_dimension(value_width, "value_width"),
    };

    if (can_use_llama_cached_decode_qkv_projection_kernel(state, rows, decode_qkv_dims)) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " decode QKV projection command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.llama_cached_decode_qkv_projection_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:query_weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:key_weight_buffer offset:0 atIndex:2];
        [encoder setBuffer:value_weight_buffer offset:0 atIndex:3];
        [encoder setBuffer:query_buffer offset:0 atIndex:4];
        [encoder setBuffer:key_buffer offset:0 atIndex:5];
        [encoder setBuffer:value_buffer offset:0 atIndex:6];
        [encoder setBytes:&decode_qkv_dims
                   length:sizeof(LlamaCachedDecodeQKVProjectionDimensions)
                  atIndex:7];

        const auto qkv_width = query_width + key_width + value_width;
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((qkv_width + 255U) / 256U), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(
                     static_cast<NSUInteger>(llama_cached_decode_qkv_projection_thread_count),
                     1,
                     1
                 )];
        [encoder endEncoding];
        return;
    }

    auto encode_matmul = [&](
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> output_buffer,
        const MatmulDimensions& dims,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&dims length:sizeof(MatmulDimensions) atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(query_weight_buffer, query_buffer, query_dims, query_width);
    encode_matmul(key_weight_buffer, key_buffer, key_dims, key_width);
    encode_matmul(value_weight_buffer, value_buffer, value_dims, value_width);
}

[[nodiscard]] std::uint64_t llama_cached_decode_attention_scratch_byte_count(
    const LlamaCachedAttentionDimensions& dims
) noexcept {
    return (static_cast<std::uint64_t>(dims.cached_token_count) + llama_cached_decode_attention_thread_count)
        * sizeof(float);
}

[[nodiscard]] bool can_use_llama_cached_decode_attention_output_kernel(
    const MatmulPipelineState& state,
    const LlamaCachedAttentionDimensions& dims
) {
    if (dims.token_count != 1 || dims.cached_token_count == 0) {
        return false;
    }

    const auto pipeline_thread_limit = static_cast<std::uint64_t>(
        [state.llama_cached_decode_attention_output_pipeline maxTotalThreadsPerThreadgroup]
    );
    if (pipeline_thread_limit < llama_cached_decode_attention_thread_count) {
        return false;
    }

    const auto threadgroup_memory_limit = static_cast<std::uint64_t>([state.device maxThreadgroupMemoryLength]);
    return llama_cached_decode_attention_scratch_byte_count(dims) <= threadgroup_memory_limit;
}

void encode_llama_cached_attention_output(
    MatmulPipelineState& state,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> query_buffer,
    id<MTLBuffer> cache_key_buffer,
    id<MTLBuffer> cache_value_buffer,
    id<MTLBuffer> output_buffer,
    const LlamaCachedAttentionDimensions& dims,
    std::uint64_t output_width
) {
    if (can_use_llama_cached_decode_attention_output_kernel(state, dims)) {
        [encoder setComputePipelineState:state.llama_cached_decode_attention_output_pipeline];
        [encoder setBuffer:query_buffer offset:0 atIndex:0];
        [encoder setBuffer:cache_key_buffer offset:0 atIndex:1];
        [encoder setBuffer:cache_value_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBytes:&dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];
        [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(
            llama_cached_decode_attention_scratch_byte_count(dims)
        )
                                    atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(dims.head_count), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(
                     static_cast<NSUInteger>(llama_cached_decode_attention_thread_count),
                     1,
                     1
                 )];
        return;
    }

    [encoder setComputePipelineState:state.llama_cached_attention_output_pipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:cache_key_buffer offset:0 atIndex:1];
    [encoder setBuffer:cache_value_buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:0 atIndex:3];
    [encoder setBytes:&dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];

    const MTLSize attention_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize attention_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((output_width + 15U) / 16U),
        static_cast<NSUInteger>((static_cast<std::uint64_t>(dims.token_count) + 15U) / 16U),
        1
    );
    [encoder dispatchThreadgroups:attention_threadgroups
            threadsPerThreadgroup:attention_threads_per_threadgroup];
}

[[nodiscard]] std::uint64_t llama_cached_decode_block_tail_scratch_byte_count(
    const LlamaCachedDecodeBlockTailDimensions& dims
) noexcept {
    return (static_cast<std::uint64_t>(dims.hidden) + llama_cached_decode_block_tail_thread_count) * sizeof(float);
}

[[nodiscard]] bool can_use_llama_cached_decode_block_tail_kernels(
    const MatmulPipelineState& state,
    std::uint64_t rows,
    const LlamaCachedDecodeBlockTailDimensions& dims
) {
    if (rows != 1 || dims.hidden == 0 || dims.attention_output_width == 0 || dims.feed_forward == 0) {
        return false;
    }

    const auto residual_norm_thread_limit = static_cast<std::uint64_t>(
        [state.llama_cached_decode_attention_residual_norm_pipeline maxTotalThreadsPerThreadgroup]
    );
    const auto gate_up_thread_limit = static_cast<std::uint64_t>(
        [state.llama_cached_decode_ffn_gate_up_silu_pipeline maxTotalThreadsPerThreadgroup]
    );
    if (residual_norm_thread_limit < llama_cached_decode_block_tail_thread_count
        || gate_up_thread_limit < llama_cached_decode_block_tail_thread_count) {
        return false;
    }

    const auto threadgroup_memory_limit = static_cast<std::uint64_t>([state.device maxThreadgroupMemoryLength]);
    return llama_cached_decode_block_tail_scratch_byte_count(dims) <= threadgroup_memory_limit;
}

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
) {
    const LlamaCachedDecodeBlockTailDimensions decode_tail_dims {
        .hidden = require_u32_dimension(hidden, "hidden"),
        .attention_output_width = require_u32_dimension(attention_output_width, "attention_output_width"),
        .feed_forward = require_u32_dimension(feed_forward, "feed_forward"),
        .epsilon = rms_norm_epsilon,
    };

    auto encode_matmul_add = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> residual_buffer,
        id<MTLBuffer> matmul_output_buffer,
        const MatmulDimensions& dims,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " matmul-add command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_add_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:residual_buffer offset:0 atIndex:2];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:3];
        [encoder setBytes:&dims length:sizeof(MatmulDimensions) atIndex:4];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    if (can_use_llama_cached_decode_block_tail_kernels(state, rows, decode_tail_dims)) {
        id<MTLComputeCommandEncoder> residual_norm_encoder = [command_buffer computeCommandEncoder];
        if (residual_norm_encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " decode residual norm command encoding failed"
            );
        }

        [residual_norm_encoder setComputePipelineState:state.llama_cached_decode_attention_residual_norm_pipeline];
        [residual_norm_encoder setBuffer:input_buffer offset:0 atIndex:0];
        [residual_norm_encoder setBuffer:attention_output_buffer offset:0 atIndex:1];
        [residual_norm_encoder setBuffer:attention_output_weight_buffer offset:0 atIndex:2];
        [residual_norm_encoder setBuffer:feed_forward_norm_weight_buffer offset:0 atIndex:3];
        [residual_norm_encoder setBuffer:attention_residual_buffer offset:0 atIndex:4];
        [residual_norm_encoder setBuffer:feed_forward_normalized_buffer offset:0 atIndex:5];
        [residual_norm_encoder setBytes:&decode_tail_dims
                                 length:sizeof(LlamaCachedDecodeBlockTailDimensions)
                                atIndex:6];
        [residual_norm_encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(
            llama_cached_decode_block_tail_scratch_byte_count(decode_tail_dims)
        )
                                                   atIndex:0];
        [residual_norm_encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(
                                  static_cast<NSUInteger>(llama_cached_decode_block_tail_thread_count),
                                  1,
                                  1
                              )];
        [residual_norm_encoder endEncoding];

        id<MTLComputeCommandEncoder> gate_up_encoder = [command_buffer computeCommandEncoder];
        if (gate_up_encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " decode gate/up command encoding failed"
            );
        }

        [gate_up_encoder setComputePipelineState:state.llama_cached_decode_ffn_gate_up_silu_pipeline];
        [gate_up_encoder setBuffer:feed_forward_normalized_buffer offset:0 atIndex:0];
        [gate_up_encoder setBuffer:gate_weight_buffer offset:0 atIndex:1];
        [gate_up_encoder setBuffer:up_weight_buffer offset:0 atIndex:2];
        [gate_up_encoder setBuffer:gated_buffer offset:0 atIndex:3];
        [gate_up_encoder setBytes:&decode_tail_dims
                           length:sizeof(LlamaCachedDecodeBlockTailDimensions)
                          atIndex:4];
        [gate_up_encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((feed_forward + 255U) / 256U), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(
                            static_cast<NSUInteger>(llama_cached_decode_block_tail_thread_count),
                            1,
                            1
                        )];
        [gate_up_encoder endEncoding];

        encode_matmul_add(gated_buffer, down_weight_buffer, attention_residual_buffer, output_buffer, down_dims, hidden);
        return;
    }

    auto encode_rms_norm = [&](id<MTLBuffer> source, id<MTLBuffer> weight, id<MTLBuffer> output) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " RMSNorm command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.rms_norm_pipeline];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:weight offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&rms_norm_dims length:sizeof(RMSNormDimensions) atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(64, 1, 1);
        const MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    auto encode_matmul = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> matmul_output_buffer,
        const MatmulDimensions& dims,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + error_context + " matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:2];
        [encoder setBytes:&dims length:sizeof(MatmulDimensions) atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul_add(
        attention_output_buffer,
        attention_output_weight_buffer,
        input_buffer,
        attention_residual_buffer,
        attention_output_dims,
        hidden
    );
    encode_rms_norm(attention_residual_buffer, feed_forward_norm_weight_buffer, feed_forward_normalized_buffer);
    encode_matmul(feed_forward_normalized_buffer, gate_weight_buffer, gate_buffer, gate_dims, feed_forward);
    encode_matmul(feed_forward_normalized_buffer, up_weight_buffer, up_buffer, gate_dims, feed_forward);

    id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
    if (silu_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            std::string("Metal ") + error_context + " activation command encoding failed"
        );
    }
    [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
    [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
    [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
    [silu_encoder setBytes:&feed_forward_elementwise_dims length:sizeof(ElementwiseDimensions) atIndex:3];

    const MTLSize activation_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto activation_threadgroup_count = static_cast<NSUInteger>(
        (static_cast<std::uint64_t>(feed_forward_elementwise_dims.elements) + 255U) / 256U
    );
    const MTLSize activation_threadgroups = MTLSizeMake(activation_threadgroup_count, 1, 1);
    [silu_encoder dispatchThreadgroups:activation_threadgroups
                 threadsPerThreadgroup:activation_threads_per_threadgroup];
    [silu_encoder endEncoding];

    encode_matmul_add(gated_buffer, down_weight_buffer, attention_residual_buffer, output_buffer, down_dims, hidden);
}

void append_llama_cached_kv_to_tensors_f32(
    TensorStorage& cache_key,
    TensorStorage& cache_value,
    const TensorStorage& key,
    const TensorStorage& value,
    std::uint64_t cache_position,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto token_count = key.descriptor().shape.dimension(0);
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;

    for (std::uint64_t row = 0; row < token_count; ++row) {
        const auto cache_row = cache_position + row;
        for (std::uint64_t kv_head = 0; kv_head < kv_head_count; ++kv_head) {
            const auto first_column = kv_head * key_length;
            for (std::uint64_t column = 0; column < key_length; ++column) {
                write_f32_linear(
                    cache_key,
                    cache_row * key_width + first_column + column,
                    read_rope_f32_columns(
                        key,
                        row,
                        first_column,
                        column,
                        cache_row,
                        rope_dimension_count,
                        frequency_base
                    )
                );
            }
        }

        for (std::uint64_t column = 0; column < value_width; ++column) {
            write_f32_linear(
                cache_value,
                cache_row * value_width + column,
                read_f32_linear(value, row * value_width + column)
            );
        }
    }
}

[[nodiscard]] TensorStorage materialize_llama_cached_attention_output_f32(
    const TensorStorage& query,
    const TensorStorage& cache_key,
    const TensorStorage& cache_value,
    std::uint64_t cache_position,
    std::uint64_t cached_token_count,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    const auto token_count = query.descriptor().shape.dimension(0);
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto output_width = head_count * value_length;
    const auto kv_group_size = head_count / kv_head_count;
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({token_count, output_width}), DType::f32));

    for (std::uint64_t head = 0; head < head_count; ++head) {
        const auto kv_head = head / kv_group_size;
        const auto query_first_column = head * key_length;
        const auto key_first_column = kv_head * key_length;
        const auto value_first_column = kv_head * value_length;
        const auto output_first_column = head * value_length;

        for (std::uint64_t query_row = 0; query_row < token_count; ++query_row) {
            const auto query_position = cache_position + query_row;
            for (std::uint64_t value_column = 0; value_column < value_length; ++value_column) {
                float max_score = -std::numeric_limits<float>::infinity();
                float denominator = 0.0F;
                float accumulator = 0.0F;

                for (std::uint64_t key_row = 0; key_row < cached_token_count; ++key_row) {
                    if (key_row > query_position) {
                        continue;
                    }

                    float dot = 0.0F;
                    for (std::uint64_t column = 0; column < key_length; ++column) {
                        dot += read_rope_f32_columns(
                            query,
                            query_row,
                            query_first_column,
                            column,
                            query_position,
                            rope_dimension_count,
                            frequency_base
                        ) * read_f32_linear(cache_key, key_row * key_width + key_first_column + column);
                    }

                    const auto score = dot * scale;
                    if (score > max_score) {
                        const auto rescale = std::isfinite(max_score) ? std::exp(max_score - score) : 0.0F;
                        denominator *= rescale;
                        accumulator *= rescale;
                        max_score = score;
                    }

                    const auto weight = std::exp(score - max_score);
                    denominator += weight;
                    accumulator += weight * read_f32_linear(
                        cache_value,
                        key_row * value_width + value_first_column + value_column
                    );
                }

                if (!(denominator > 0.0F)) {
                    throw std::invalid_argument("Llama cached attention row has no visible keys");
                }
                write_f32_linear(
                    output,
                    query_row * output_width + output_first_column + value_column,
                    accumulator / denominator
                );
            }
        }
    }

    (void)query_width;
    return output;
}

[[nodiscard]] BackendBuffer llama_cached_attention_f32_with_metal_buffers(
    const MetalBackendBufferStorage& query,
    const MetalBackendBufferStorage& key,
    const MetalBackendBufferStorage& value,
    const MetalBackendBufferStorage& cache_key,
    const MetalBackendBufferStorage& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    auto& state = shared_matmul_pipeline_state();

    const auto token_count = query.descriptor().shape.dimension(0);
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto output_width = head_count * value_length;
    const auto cached_token_count = cache_position + token_count;
    const LlamaCachedAttentionDimensions dims {
        .token_count = require_u32_dimension(token_count, "token_count"),
        .cache_position = require_u32_dimension(cache_position, "cache_position"),
        .cached_token_count = require_u32_dimension(cached_token_count, "cached_token_count"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
        .key_length = require_u32_dimension(key_length, "key_length"),
        .value_length = require_u32_dimension(value_length, "value_length"),
        .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
        .frequency_base = frequency_base,
        .scale = scale,
    };
    const auto output_descriptor = TensorDescriptor::from(Shape::from({token_count, output_width}), DType::f32);

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    if (output_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention command buffer creation failed"
        );
    }

    id<MTLComputeCommandEncoder> append_encoder = [command_buffer computeCommandEncoder];
    if (append_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached KV append command encoding failed"
        );
    }
    [append_encoder setComputePipelineState:state.llama_cached_kv_append_pipeline];
    [append_encoder setBuffer:key.buffer() offset:0 atIndex:0];
    [append_encoder setBuffer:value.buffer() offset:0 atIndex:1];
    [append_encoder setBuffer:cache_key.buffer() offset:0 atIndex:2];
    [append_encoder setBuffer:cache_value.buffer() offset:0 atIndex:3];
    [append_encoder setBytes:&dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];

    const auto append_columns = key_width > value_width ? key_width : value_width;
    const MTLSize append_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize append_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((append_columns + 15U) / 16U),
        static_cast<NSUInteger>((token_count + 15U) / 16U),
        1
    );
    [append_encoder dispatchThreadgroups:append_threadgroups threadsPerThreadgroup:append_threads_per_threadgroup];
    [append_encoder endEncoding];

    id<MTLComputeCommandEncoder> attention_encoder = [command_buffer computeCommandEncoder];
    if (attention_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention output command encoding failed"
        );
    }
    encode_llama_cached_attention_output(
        state,
        attention_encoder,
        query.buffer(),
        cache_key.buffer(),
        cache_value.buffer(),
        output_buffer,
        dims,
        output_width
    );
    [attention_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

[[nodiscard]] BackendBuffer llama_cached_block_f32_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& attention_norm_weight,
    const MetalBackendBufferStorage& query_weight,
    const MetalBackendBufferStorage& key_weight,
    const MetalBackendBufferStorage& value_weight,
    const MetalBackendBufferStorage& cache_key,
    const MetalBackendBufferStorage& cache_value,
    const MetalBackendBufferStorage& attention_output_weight,
    const MetalBackendBufferStorage& feed_forward_norm_weight,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float attention_scale,
    float rms_norm_epsilon
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = input.descriptor().shape.dimension(0);
    const auto hidden = input.descriptor().shape.dimension(1);
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto attention_output_width = head_count * value_length;
    const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
    const auto cached_token_count = cache_position + rows;
    const auto hidden_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
    const auto query_descriptor = TensorDescriptor::from(Shape::from({rows, query_width}), DType::f32);
    const auto key_descriptor = TensorDescriptor::from(Shape::from({rows, key_width}), DType::f32);
    const auto value_descriptor = TensorDescriptor::from(Shape::from({rows, value_width}), DType::f32);
    const auto attention_output_descriptor =
        TensorDescriptor::from(Shape::from({rows, attention_output_width}), DType::f32);
    const auto feed_forward_descriptor = TensorDescriptor::from(Shape::from({rows, feed_forward}), DType::f32);
    const RMSNormDimensions rms_norm_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .epsilon = rms_norm_epsilon,
    };
    const MatmulDimensions query_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(query_width, "query_width"),
    };
    const MatmulDimensions key_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(key_width, "key_width"),
    };
    const MatmulDimensions value_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(value_width, "value_width"),
    };
    const LlamaCachedAttentionDimensions attention_dims {
        .token_count = require_u32_dimension(rows, "token_count"),
        .cache_position = require_u32_dimension(cache_position, "cache_position"),
        .cached_token_count = require_u32_dimension(cached_token_count, "cached_token_count"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
        .key_length = require_u32_dimension(key_length, "key_length"),
        .value_length = require_u32_dimension(value_length, "value_length"),
        .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
        .frequency_base = frequency_base,
        .scale = attention_scale,
    };
    const MatmulDimensions attention_output_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(attention_output_width, "attention_output_width"),
        .columns = require_u32_dimension(hidden, "hidden"),
    };
    const MatmulDimensions gate_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(feed_forward, "feed_forward"),
    };
    const MatmulDimensions down_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(feed_forward, "feed_forward"),
        .columns = require_u32_dimension(hidden, "hidden"),
    };
    const ElementwiseDimensions feed_forward_elementwise_dims {
        .elements = require_u32_dimension(feed_forward_descriptor.element_count, "feed_forward_elements"),
    };

    std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
    auto& scratch = state.llama_cached_block_scratch;
    // The fused block waits before returning, so intra-block work buffers can be
    // reused across calls. The returned output buffer stays per-call owned
    // because it may become the next layer's input.
    id<MTLBuffer> attention_normalized_buffer =
        reusable_scratch_buffer(state.device, scratch.attention_normalized, hidden_descriptor.byte_count, "attention_norm");
    id<MTLBuffer> query_buffer =
        reusable_scratch_buffer(state.device, scratch.query, query_descriptor.byte_count, "query");
    id<MTLBuffer> key_buffer =
        reusable_scratch_buffer(state.device, scratch.key, key_descriptor.byte_count, "key");
    id<MTLBuffer> value_buffer =
        reusable_scratch_buffer(state.device, scratch.value, value_descriptor.byte_count, "value");
    id<MTLBuffer> attention_output_buffer =
        reusable_scratch_buffer(state.device, scratch.attention_output, attention_output_descriptor.byte_count, "attention_output");
    id<MTLBuffer> attention_residual_buffer =
        reusable_scratch_buffer(state.device, scratch.attention_residual, hidden_descriptor.byte_count, "attention_residual");
    id<MTLBuffer> feed_forward_normalized_buffer =
        reusable_scratch_buffer(state.device, scratch.feed_forward_normalized, hidden_descriptor.byte_count, "ffn_norm");
    id<MTLBuffer> gate_buffer =
        reusable_scratch_buffer(state.device, scratch.gate, feed_forward_descriptor.byte_count, "gate");
    id<MTLBuffer> up_buffer =
        reusable_scratch_buffer(state.device, scratch.up, feed_forward_descriptor.byte_count, "up");
    id<MTLBuffer> gated_buffer =
        reusable_scratch_buffer(state.device, scratch.gated, feed_forward_descriptor.byte_count, "gated");
    id<MTLBuffer> output_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    if (output_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block command buffer creation failed"
        );
    }

    auto encode_rms_norm = [&](id<MTLBuffer> source, id<MTLBuffer> weight, id<MTLBuffer> output) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal Llama cached block RMSNorm command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.rms_norm_pipeline];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:weight offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&rms_norm_dims length:sizeof(RMSNormDimensions) atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(64, 1, 1);
        const MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_rms_norm(input.buffer(), attention_norm_weight.buffer(), attention_normalized_buffer);
    encode_llama_cached_qkv_projection(
        state,
        command_buffer,
        attention_normalized_buffer,
        query_weight.buffer(),
        key_weight.buffer(),
        value_weight.buffer(),
        query_buffer,
        key_buffer,
        value_buffer,
        query_dims,
        key_dims,
        value_dims,
        rows,
        hidden,
        query_width,
        key_width,
        value_width,
        "Llama cached block QKV projection"
    );

    id<MTLComputeCommandEncoder> append_encoder = [command_buffer computeCommandEncoder];
    if (append_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block KV append command encoding failed"
        );
    }
    [append_encoder setComputePipelineState:state.llama_cached_kv_append_pipeline];
    [append_encoder setBuffer:key_buffer offset:0 atIndex:0];
    [append_encoder setBuffer:value_buffer offset:0 atIndex:1];
    [append_encoder setBuffer:cache_key.buffer() offset:0 atIndex:2];
    [append_encoder setBuffer:cache_value.buffer() offset:0 atIndex:3];
    [append_encoder setBytes:&attention_dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];

    const auto append_columns = key_width > value_width ? key_width : value_width;
    const MTLSize append_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize append_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((append_columns + 15U) / 16U),
        static_cast<NSUInteger>((rows + 15U) / 16U),
        1
    );
    [append_encoder dispatchThreadgroups:append_threadgroups threadsPerThreadgroup:append_threads_per_threadgroup];
    [append_encoder endEncoding];

    id<MTLComputeCommandEncoder> attention_encoder = [command_buffer computeCommandEncoder];
    if (attention_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block attention command encoding failed"
        );
    }
    encode_llama_cached_attention_output(
        state,
        attention_encoder,
        query_buffer,
        cache_key.buffer(),
        cache_value.buffer(),
        attention_output_buffer,
        attention_dims,
        attention_output_width
    );
    [attention_encoder endEncoding];

    encode_llama_cached_block_tail(
        state,
        command_buffer,
        input.buffer(),
        attention_output_buffer,
        attention_output_weight.buffer(),
        feed_forward_norm_weight.buffer(),
        gate_weight.buffer(),
        up_weight.buffer(),
        down_weight.buffer(),
        attention_residual_buffer,
        feed_forward_normalized_buffer,
        gate_buffer,
        up_buffer,
        gated_buffer,
        output_buffer,
        rms_norm_dims,
        attention_output_dims,
        gate_dims,
        down_dims,
        feed_forward_elementwise_dims,
        rows,
        hidden,
        attention_output_width,
        feed_forward,
        rms_norm_epsilon,
        "Llama cached block tail"
    );

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(hidden_descriptor, output_buffer);
}


}  // namespace local_gguf_engine::tensor::backends::metal

namespace local_gguf_engine::tensor::ops {

BackendBuffer llama_cached_attention_f32_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& key,
    const BackendBuffer& value,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    backends::metal::require_llama_cached_attention_f32_operands(
        query.descriptor(),
        key.descriptor(),
        value.descriptor(),
        cache_key.descriptor(),
        cache_value.descriptor(),
        cache_position,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        scale
    );
    backends::metal::require_same_backend(query.backend(), key.backend(), "Llama cached attention");
    backends::metal::require_same_backend(query.backend(), value.backend(), "Llama cached attention");
    backends::metal::require_same_backend(query.backend(), cache_key.backend(), "Llama cached attention");
    backends::metal::require_same_backend(query.backend(), cache_value.backend(), "Llama cached attention");

    switch (query.backend()) {
        case runtime::BackendKind::cpu: {
            const auto query_storage = copy_tensor_from_backend(query);
            const auto key_storage = copy_tensor_from_backend(key);
            const auto value_storage = copy_tensor_from_backend(value);
            auto cache_key_storage = copy_tensor_from_backend(cache_key);
            auto cache_value_storage = copy_tensor_from_backend(cache_value);

            backends::metal::append_llama_cached_kv_to_tensors_f32(
                cache_key_storage,
                cache_value_storage,
                key_storage,
                value_storage,
                cache_position,
                kv_head_count,
                key_length,
                value_length,
                rope_dimension_count,
                frequency_base
            );
            const auto output = backends::metal::materialize_llama_cached_attention_output_f32(
                query_storage,
                cache_key_storage,
                cache_value_storage,
                cache_position,
                cache_position + query_storage.descriptor().shape.dimension(0),
                head_count,
                kv_head_count,
                key_length,
                value_length,
                rope_dimension_count,
                frequency_base,
                scale
            );

            cache_key = copy_tensor_to_backend(cache_key_storage, runtime::BackendKind::cpu);
            cache_value = copy_tensor_to_backend(cache_value_storage, runtime::BackendKind::cpu);
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::llama_cached_attention_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(query)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(key)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(value)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_key)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_value)),
                    cache_position,
                    head_count,
                    kv_head_count,
                    key_length,
                    value_length,
                    rope_dimension_count,
                    frequency_base,
                    scale
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        query.backend(),
        "unsupported backend buffers for Llama cached attention"
    );
}

BackendBuffer llama_cached_block_f32_to_backend(
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
) {
    backends::metal::require_llama_cached_block_f32_operands(
        input.descriptor(),
        attention_norm_weight.descriptor(),
        query_weight.descriptor(),
        key_weight.descriptor(),
        value_weight.descriptor(),
        cache_key.descriptor(),
        cache_value.descriptor(),
        attention_output_weight.descriptor(),
        feed_forward_norm_weight.descriptor(),
        gate_weight.descriptor(),
        up_weight.descriptor(),
        down_weight.descriptor(),
        cache_position,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale,
        rms_norm_epsilon
    );
    backends::metal::require_same_backend(input.backend(), attention_norm_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), query_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), key_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), value_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), cache_key.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), cache_value.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), attention_output_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), feed_forward_norm_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), gate_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), up_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), down_weight.backend(), "Llama cached block");

    switch (input.backend()) {
        case runtime::BackendKind::cpu: {
            const auto input_storage = copy_tensor_from_backend(input);
            const auto attention_norm_storage = copy_tensor_from_backend(attention_norm_weight);
            const auto normalized = rms_norm_f32(input_storage, attention_norm_storage, rms_norm_epsilon);
            const auto normalized_buffer = copy_tensor_to_backend(normalized, runtime::BackendKind::cpu);
            const auto projections = matmul_f32_f16_gguf_weight_triple_to_backend(
                normalized_buffer,
                query_weight,
                key_weight,
                value_weight
            );
            const auto attention_output = llama_cached_attention_f32_to_backend(
                projections.first,
                projections.second,
                projections.third,
                cache_key,
                cache_value,
                cache_position,
                head_count,
                kv_head_count,
                key_length,
                value_length,
                rope_dimension_count,
                frequency_base,
                attention_scale
            );
            return residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
                input,
                attention_output,
                attention_output_weight,
                feed_forward_norm_weight,
                gate_weight,
                up_weight,
                down_weight,
                rms_norm_epsilon
            );
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::llama_cached_block_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_norm_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(query_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(key_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(value_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_key)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_value)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_output_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(down_weight)),
                    cache_position,
                    head_count,
                    kv_head_count,
                    key_length,
                    value_length,
                    rope_dimension_count,
                    frequency_base,
                    attention_scale,
                    rms_norm_epsilon
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffers for Llama cached block"
    );
}

}  // namespace local_gguf_engine::tensor::ops
