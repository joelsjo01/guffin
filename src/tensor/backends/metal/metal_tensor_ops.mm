#include "metal_tensor_ops.hpp"

#include "../cpu_backend.hpp"
#include "../metal_backend.hpp"
#include "metal_context.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "../../ops/validation.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

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

void require_matmul_inner_dimensions(const TensorDescriptor& left, const TensorDescriptor& right) {
    const auto left_columns = left.shape.dimension(1);
    const auto right_rows = right.shape.dimension(0);
    if (left_columns != right_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_columns, right_rows);
    }
}

void require_quantized_gguf_weight_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.dtype == DType::q4_k || descriptor.dtype == DType::q6_k) {
        return;
    }

    throw TensorError(
        TensorErrorCode::unsupported_operator_dtype,
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::q4_k)),
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
    );
}

void require_quantized_gguf_weight_inner_block_alignment(const TensorDescriptor& descriptor) {
    constexpr std::uint64_t k_quant_block_elements = 256;
    const auto inner_width = descriptor.shape.dimension(0);
    if (inner_width % k_quant_block_elements != 0) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            k_quant_block_elements,
            inner_width
        );
    }
}

id<MTLComputePipelineState> quantized_matmul_pipeline(
    MatmulPipelineState& state,
    DType dtype
) {
    switch (dtype) {
        case DType::q4_k:
            return state.matmul_q4_k_pipeline;
        case DType::q6_k:
            return state.matmul_q6_k_pipeline;
        case DType::f32:
        case DType::f16:
        case DType::q8_0:
            break;
    }

    throw TensorError(
        TensorErrorCode::unsupported_operator_dtype,
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::q4_k)),
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype))
    );
}

}  // namespace

std::uint32_t require_u32_dimension(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            std::string("Metal F32xF16 GGUF weight matmul dimension is too large: ") + name
        );
    }

    return static_cast<std::uint32_t>(value);
}

void require_matrix_descriptor(const TensorDescriptor& descriptor) {
    constexpr std::size_t matrix_rank = 2;
    if (descriptor.shape.rank() != matrix_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(matrix_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_f32_f16_matmul_gguf_weight_operands(
    const TensorStorage& left,
    const TensorDescriptor& right
) {
    ops::require_f32_descriptor(left.descriptor());
    ops::require_f16_descriptor(right);
    require_matrix_descriptor(left.descriptor());
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left.descriptor(), right);
}

void require_f32_f16_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
) {
    ops::require_f32_descriptor(left);
    ops::require_f16_descriptor(right);
    require_matrix_descriptor(left);
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left, right);
}

void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorStorage& left,
    const TensorDescriptor& right
) {
    require_f32_quantized_matmul_gguf_weight_operands(left.descriptor(), right);
}

void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
) {
    ops::require_f32_descriptor(left);
    require_quantized_gguf_weight_descriptor(right);
    require_matrix_descriptor(left);
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left, right);
    require_quantized_gguf_weight_inner_block_alignment(right);
}

void require_same_backend(
    runtime::BackendKind left,
    runtime::BackendKind right,
    const char* operation
) {
    if (left != right) {
        throw std::invalid_argument(std::string(operation) + " backend buffers must use the same backend");
    }
}

void require_silu_multiply_f32_operands(
    const TensorDescriptor& gate,
    const TensorDescriptor& up
) {
    ops::require_f32_descriptor(gate);
    ops::require_f32_descriptor(up);
    ops::require_same_shape(gate, up);
}

void require_gated_silu_projection_f32_f16_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight
) {
    require_f32_f16_matmul_gguf_weight_operands(left, gate_weight);
    require_f32_f16_matmul_gguf_weight_operands(left, up_weight);
    ops::require_f16_descriptor(down_weight);
    require_matrix_descriptor(down_weight);

    const auto feed_forward = gate_weight.shape.dimension(1);
    const auto up_feed_forward = up_weight.shape.dimension(1);
    if (feed_forward != up_feed_forward) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, feed_forward, up_feed_forward);
    }

    const auto down_rows = down_weight.shape.dimension(0);
    if (down_rows != feed_forward) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, feed_forward, down_rows);
    }
}

void require_f32_vector_descriptor(const TensorDescriptor& descriptor) {
    constexpr std::size_t vector_rank = 1;
    ops::require_f32_descriptor(descriptor);
    if (descriptor.shape.rank() != vector_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(vector_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_positive_epsilon(float epsilon) {
    if (!(epsilon > 0.0F)) {
        throw std::invalid_argument("RMSNorm epsilon must be positive");
    }
}

void require_positive_finite_scale(float scale) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("attention scale must be positive and finite");
    }
}

void require_positive_frequency_base(float frequency_base) {
    if (!(frequency_base > 0.0F)) {
        throw std::invalid_argument("RoPE frequency base must be positive");
    }
}

void require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& attention_output,
    const TensorDescriptor& attention_output_weight,
    const TensorDescriptor& feed_forward_norm_weight,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight,
    float epsilon
) {
    ops::require_f32_descriptor(input);
    ops::require_f32_descriptor(attention_output);
    require_matrix_descriptor(input);
    require_matrix_descriptor(attention_output);

    const auto token_count = input.shape.dimension(0);
    const auto attention_token_count = attention_output.shape.dimension(0);
    if (token_count != attention_token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, token_count, attention_token_count);
    }

    require_f32_f16_matmul_gguf_weight_operands(attention_output, attention_output_weight);
    const auto hidden = input.shape.dimension(1);
    const auto attention_output_columns = attention_output_weight.shape.dimension(1);
    if (hidden != attention_output_columns) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, attention_output_columns);
    }

    require_f32_vector_descriptor(feed_forward_norm_weight);
    const auto norm_hidden = feed_forward_norm_weight.shape.dimension(0);
    if (hidden != norm_hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, norm_hidden);
    }

    require_gated_silu_projection_f32_f16_gguf_weight_operands(
        input,
        gate_weight,
        up_weight,
        down_weight
    );
    const auto down_columns = down_weight.shape.dimension(1);
    if (hidden != down_columns) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, down_columns);
    }

    require_positive_epsilon(epsilon);
}

void require_f32_f16_matmul_gguf_weight_triple_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& first_right,
    const TensorDescriptor& second_right,
    const TensorDescriptor& third_right
) {
    require_f32_f16_matmul_gguf_weight_operands(left, first_right);
    require_f32_f16_matmul_gguf_weight_operands(left, second_right);
    require_f32_f16_matmul_gguf_weight_operands(left, third_right);
}

TensorStorage matmul_f32_f16_gguf_weight_with_metal_right_buffer(
    const TensorStorage& left,
    const MetalBackendBufferStorage& right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    const MatmulDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(columns, "columns"),
    };
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

    id<MTLBuffer> left_buffer = [state.device newBufferWithBytes:left.bytes().data()
                                                           length:static_cast<NSUInteger>(left.bytes().size())
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output.bytes().size())
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(MatmulDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (left_buffer == nil || output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.matmul_pipeline];
    [encoder setBuffer:left_buffer offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
    const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    std::memcpy(
        output.mutable_bytes().data(),
        [output_buffer contents],
        output.bytes().size()
    );
    return output;
}

BackendBuffer matmul_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    const MatmulDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(columns, "columns"),
    };
    const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, columns}), DType::f32);

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(MatmulDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer matmul command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.matmul_pipeline];
    [encoder setBuffer:left.buffer() offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
    const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer matmul command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer matmul_f32_quantized_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    const MatmulDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(columns, "columns"),
    };
    const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, columns}), DType::f32);

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(MatmulDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer K-quant matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer K-quant matmul command encoding failed"
        );
    }

    [encoder setComputePipelineState:quantized_matmul_pipeline(state, right.descriptor().dtype)];
    [encoder setBuffer:left.buffer() offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
    const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer K-quant matmul command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

ops::F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& first_right,
    const MetalBackendBufferStorage& second_right,
    const MetalBackendBufferStorage& third_right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto first_columns = first_right.descriptor().shape.dimension(1);
    const auto second_columns = second_right.descriptor().shape.dimension(1);
    const auto third_columns = third_right.descriptor().shape.dimension(1);
    const MatmulDimensions first_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(first_columns, "first_columns"),
    };
    const MatmulDimensions second_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(second_columns, "second_columns"),
    };
    const MatmulDimensions third_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(third_columns, "third_columns"),
    };
    const auto first_descriptor = TensorDescriptor::from(Shape::from({rows, first_columns}), DType::f32);
    const auto second_descriptor = TensorDescriptor::from(Shape::from({rows, second_columns}), DType::f32);
    const auto third_descriptor = TensorDescriptor::from(Shape::from({rows, third_columns}), DType::f32);

    id<MTLBuffer> first_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                     first_descriptor.byte_count
                                  )
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> second_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      second_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> third_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                     third_descriptor.byte_count
                                  )
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> first_dims_buffer = [state.device newBufferWithBytes:&first_dims
                                                                length:sizeof(MatmulDimensions)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> second_dims_buffer = [state.device newBufferWithBytes:&second_dims
                                                                 length:sizeof(MatmulDimensions)
                                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> third_dims_buffer = [state.device newBufferWithBytes:&third_dims
                                                                length:sizeof(MatmulDimensions)
                                                               options:MTLResourceStorageModeShared];
    if (first_buffer == nil || second_buffer == nil || third_buffer == nil
        || first_dims_buffer == nil || second_dims_buffer == nil || third_dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal triple matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal triple matmul command buffer creation failed"
        );
    }

    auto encode_matmul = [&](
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> output_buffer,
        id<MTLBuffer> dims_buffer,
        std::uint64_t output_columns
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal triple matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left.buffer() offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(first_right.buffer(), first_buffer, first_dims_buffer, first_columns);
    encode_matmul(second_right.buffer(), second_buffer, second_dims_buffer, second_columns);
    encode_matmul(third_right.buffer(), third_buffer, third_dims_buffer, third_columns);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal triple matmul command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return ops::F32F16GGUFWeightTripleMatmulBackendBuffers {
        .first = make_metal_backend_buffer(first_descriptor, first_buffer),
        .second = make_metal_backend_buffer(second_descriptor, second_buffer),
        .third = make_metal_backend_buffer(third_descriptor, third_buffer),
    };
}

BackendBuffer silu_multiply_f32_with_metal_buffers(
    const MetalBackendBufferStorage& gate,
    const MetalBackendBufferStorage& up
) {
    auto& state = shared_matmul_pipeline_state();

    const auto element_count = gate.descriptor().element_count;
    const ElementwiseDimensions dims {
        .elements = require_u32_dimension(element_count, "elements"),
    };
    const auto output_descriptor = gate.descriptor();

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(ElementwiseDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.silu_multiply_pipeline];
    [encoder setBuffer:gate.buffer() offset:0 atIndex:0];
    [encoder setBuffer:up.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto threadgroup_count = static_cast<NSUInteger>((element_count + 255U) / 256U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_count, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto hidden = left.descriptor().shape.dimension(1);
    const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
    const auto output_columns = down_weight.descriptor().shape.dimension(1);
    const MatmulDimensions gate_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(feed_forward, "feed_forward"),
    };
    const MatmulDimensions down_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(feed_forward, "feed_forward"),
        .columns = require_u32_dimension(output_columns, "output_columns"),
    };
    const auto intermediate_descriptor = TensorDescriptor::from(
        Shape::from({rows, feed_forward}),
        DType::f32
    );
    const auto output_descriptor = TensorDescriptor::from(
        Shape::from({rows, output_columns}),
        DType::f32
    );
    const ElementwiseDimensions elementwise_dims {
        .elements = require_u32_dimension(intermediate_descriptor.element_count, "elements"),
    };

    id<MTLBuffer> gate_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                    intermediate_descriptor.byte_count
                                 )
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> up_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                  intermediate_descriptor.byte_count
                               )
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                     intermediate_descriptor.byte_count
                                  )
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_dims_buffer = [state.device newBufferWithBytes:&gate_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> down_dims_buffer = [state.device newBufferWithBytes:&down_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> elementwise_dims_buffer = [state.device newBufferWithBytes:&elementwise_dims
                                                                      length:sizeof(ElementwiseDimensions)
                                                                     options:MTLResourceStorageModeShared];
    if (gate_buffer == nil || up_buffer == nil || gated_buffer == nil || output_buffer == nil
        || gate_dims_buffer == nil || down_dims_buffer == nil || elementwise_dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection command buffer creation failed"
        );
    }

    auto encode_matmul = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> matmul_output_buffer,
        id<MTLBuffer> dims_buffer,
        std::uint64_t output_rows,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal gated SiLU projection matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((output_rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(left.buffer(), gate_weight.buffer(), gate_buffer, gate_dims_buffer, rows, feed_forward);
    encode_matmul(left.buffer(), up_weight.buffer(), up_buffer, gate_dims_buffer, rows, feed_forward);

    id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
    if (silu_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection activation command encoding failed"
        );
    }
    [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
    [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
    [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
    [silu_encoder setBuffer:elementwise_dims_buffer offset:0 atIndex:3];

    const MTLSize activation_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto activation_threadgroup_count = static_cast<NSUInteger>(
        (intermediate_descriptor.element_count + 255U) / 256U
    );
    const MTLSize activation_threadgroups = MTLSizeMake(activation_threadgroup_count, 1, 1);
    [silu_encoder dispatchThreadgroups:activation_threadgroups
                 threadsPerThreadgroup:activation_threads_per_threadgroup];
    [silu_encoder endEncoding];

    encode_matmul(gated_buffer, down_weight.buffer(), output_buffer, down_dims_buffer, rows, output_columns);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& attention_output,
    const MetalBackendBufferStorage& attention_output_weight,
    const MetalBackendBufferStorage& feed_forward_norm_weight,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight,
    float epsilon
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = input.descriptor().shape.dimension(0);
    const auto hidden = input.descriptor().shape.dimension(1);
    const auto attention_output_width = attention_output.descriptor().shape.dimension(1);
    const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
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
    const auto hidden_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
    const auto feed_forward_descriptor = TensorDescriptor::from(Shape::from({rows, feed_forward}), DType::f32);
    const ElementwiseDimensions hidden_elementwise_dims {
        .elements = require_u32_dimension(hidden_descriptor.element_count, "hidden_elements"),
    };
    const ElementwiseDimensions feed_forward_elementwise_dims {
        .elements = require_u32_dimension(feed_forward_descriptor.element_count, "feed_forward_elements"),
    };
    const RMSNormDimensions rms_norm_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .epsilon = epsilon,
    };

    id<MTLBuffer> attention_projected_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_residual_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> normalized_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(feed_forward_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> up_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(feed_forward_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(feed_forward_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> feed_forward_projected_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_output_dims_buffer = [state.device newBufferWithBytes:&attention_output_dims
                                                                           length:sizeof(MatmulDimensions)
                                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_dims_buffer = [state.device newBufferWithBytes:&gate_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> down_dims_buffer = [state.device newBufferWithBytes:&down_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden_elementwise_dims_buffer = [state.device newBufferWithBytes:&hidden_elementwise_dims
                                                                             length:sizeof(ElementwiseDimensions)
                                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> feed_forward_elementwise_dims_buffer =
        [state.device newBufferWithBytes:&feed_forward_elementwise_dims
                                  length:sizeof(ElementwiseDimensions)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> rms_norm_dims_buffer = [state.device newBufferWithBytes:&rms_norm_dims
                                                                   length:sizeof(RMSNormDimensions)
                                                                  options:MTLResourceStorageModeShared];
    if (attention_projected_buffer == nil || attention_residual_buffer == nil || normalized_buffer == nil
        || gate_buffer == nil || up_buffer == nil || gated_buffer == nil || feed_forward_projected_buffer == nil
        || output_buffer == nil || attention_output_dims_buffer == nil || gate_dims_buffer == nil
        || down_dims_buffer == nil || hidden_elementwise_dims_buffer == nil
        || feed_forward_elementwise_dims_buffer == nil || rms_norm_dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection command buffer creation failed"
        );
    }

    auto encode_matmul = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> matmul_output_buffer,
        id<MTLBuffer> dims_buffer,
        std::uint64_t output_rows,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal residual gated SiLU projection matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((output_rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    auto encode_add = [&](id<MTLBuffer> left_buffer, id<MTLBuffer> right_buffer, id<MTLBuffer> add_output_buffer) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal residual gated SiLU projection add command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.add_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:add_output_buffer offset:0 atIndex:2];
        [encoder setBuffer:hidden_elementwise_dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
        const auto threadgroup_count = static_cast<NSUInteger>((hidden_descriptor.element_count + 255U) / 256U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_count, 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(
        attention_output.buffer(),
        attention_output_weight.buffer(),
        attention_projected_buffer,
        attention_output_dims_buffer,
        rows,
        hidden
    );
    encode_add(input.buffer(), attention_projected_buffer, attention_residual_buffer);

    id<MTLComputeCommandEncoder> rms_norm_encoder = [command_buffer computeCommandEncoder];
    if (rms_norm_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection RMSNorm command encoding failed"
        );
    }
    [rms_norm_encoder setComputePipelineState:state.rms_norm_pipeline];
    [rms_norm_encoder setBuffer:attention_residual_buffer offset:0 atIndex:0];
    [rms_norm_encoder setBuffer:feed_forward_norm_weight.buffer() offset:0 atIndex:1];
    [rms_norm_encoder setBuffer:normalized_buffer offset:0 atIndex:2];
    [rms_norm_encoder setBuffer:rms_norm_dims_buffer offset:0 atIndex:3];

    const MTLSize rms_norm_threads_per_threadgroup = MTLSizeMake(64, 1, 1);
    const auto rms_norm_threadgroup_count = static_cast<NSUInteger>((rows + 63U) / 64U);
    const MTLSize rms_norm_threadgroups = MTLSizeMake(rms_norm_threadgroup_count, 1, 1);
    [rms_norm_encoder dispatchThreadgroups:rms_norm_threadgroups
                     threadsPerThreadgroup:rms_norm_threads_per_threadgroup];
    [rms_norm_encoder endEncoding];

    encode_matmul(normalized_buffer, gate_weight.buffer(), gate_buffer, gate_dims_buffer, rows, feed_forward);
    encode_matmul(normalized_buffer, up_weight.buffer(), up_buffer, gate_dims_buffer, rows, feed_forward);

    id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
    if (silu_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection activation command encoding failed"
        );
    }
    [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
    [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
    [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
    [silu_encoder setBuffer:feed_forward_elementwise_dims_buffer offset:0 atIndex:3];

    const MTLSize activation_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto activation_threadgroup_count = static_cast<NSUInteger>(
        (feed_forward_descriptor.element_count + 255U) / 256U
    );
    const MTLSize activation_threadgroups = MTLSizeMake(activation_threadgroup_count, 1, 1);
    [silu_encoder dispatchThreadgroups:activation_threadgroups
                 threadsPerThreadgroup:activation_threads_per_threadgroup];
    [silu_encoder endEncoding];

    encode_matmul(gated_buffer, down_weight.buffer(), feed_forward_projected_buffer, down_dims_buffer, rows, hidden);
    encode_add(attention_residual_buffer, feed_forward_projected_buffer, output_buffer);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(hidden_descriptor, output_buffer);
}

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const TensorStorage& right) {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();

        const auto rows = left.descriptor().shape.dimension(0);
        const auto shared = left.descriptor().shape.dimension(1);
        const auto columns = right.descriptor().shape.dimension(1);
        const MatmulDimensions dims {
            .rows = require_u32_dimension(rows, "rows"),
            .shared = require_u32_dimension(shared, "shared"),
            .columns = require_u32_dimension(columns, "columns"),
        };
        auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

        id<MTLBuffer> left_buffer = [state.device newBufferWithBytes:left.bytes().data()
                                                               length:static_cast<NSUInteger>(left.bytes().size())
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> right_buffer = [state.device newBufferWithBytes:right.bytes().data()
                                                                length:static_cast<NSUInteger>(right.bytes().size())
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output.bytes().size())
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
        if (left_buffer == nil || right_buffer == nil || output_buffer == nil || dims_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal matmul buffer allocation failed"
            );
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal matmul command failed: " + ns_error_message(execution_error, "unknown error")
            );
        }

        std::memcpy(
            output.mutable_bytes().data(),
            [output_buffer contents],
            output.bytes().size()
        );
        return output;
    }
}

TensorStorage matmul_f32_quantized_gguf_weight(const TensorStorage& left, const TensorStorage& right) {
    @autoreleasepool {
        const auto left_buffer = copy_tensor_to_metal_backend(left);
        const auto right_buffer = copy_tensor_to_metal_backend(right);
        const auto output_buffer = matmul_f32_quantized_gguf_weight_with_metal_buffers(
            require_metal_backend_buffer(backend_buffer_storage(left_buffer)),
            require_metal_backend_buffer(backend_buffer_storage(right_buffer))
        );
        return copy_tensor_from_metal_backend(backend_buffer_storage(output_buffer));
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal

namespace local_gguf_engine::tensor::ops {

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const BackendBuffer& right) {
    backends::metal::require_f32_f16_matmul_gguf_weight_operands(left, right.descriptor());

    switch (right.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight_with_cpu_right_buffer(
                left,
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_f16_gguf_weight_with_metal_right_buffer(
                    left,
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        right.backend(),
        "unsupported backend buffer for F32xF16 GGUF weight matmul"
    );
}

BackendBuffer matmul_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
) {
    backends::metal::require_f32_f16_matmul_gguf_weight_operands(left.descriptor(), right.descriptor());
    backends::metal::require_same_backend(
        left.backend(),
        right.backend(),
        "F32xF16 GGUF weight matmul"
    );

    switch (right.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_f16_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        right.backend(),
        "unsupported backend buffers for F32xF16 GGUF weight matmul"
    );
}

BackendBuffer matmul_f32_quantized_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
) {
    backends::metal::require_f32_quantized_matmul_gguf_weight_operands(left.descriptor(), right.descriptor());
    backends::metal::require_same_backend(
        left.backend(),
        right.backend(),
        "F32xK-quant GGUF weight matmul"
    );

    switch (left.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_quantized_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_quantized_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for F32xK-quant GGUF weight matmul"
    );
}

F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& first_right,
    const BackendBuffer& second_right,
    const BackendBuffer& third_right
) {
    backends::metal::require_f32_f16_matmul_gguf_weight_triple_operands(
        left.descriptor(),
        first_right.descriptor(),
        second_right.descriptor(),
        third_right.descriptor()
    );
    backends::metal::require_same_backend(left.backend(), first_right.backend(), "F32xF16 GGUF triple matmul");
    backends::metal::require_same_backend(left.backend(), second_right.backend(), "F32xF16 GGUF triple matmul");
    backends::metal::require_same_backend(left.backend(), third_right.backend(), "F32xF16 GGUF triple matmul");

    switch (left.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight_triple_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(first_right)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(second_right)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(third_right))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(first_right)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(second_right)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(third_right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for F32xF16 GGUF triple matmul"
    );
}

BackendBuffer silu_multiply_f32_to_backend(
    const BackendBuffer& gate,
    const BackendBuffer& up
) {
    backends::metal::require_silu_multiply_f32_operands(gate.descriptor(), up.descriptor());
    backends::metal::require_same_backend(gate.backend(), up.backend(), "SiLU multiply");

    switch (gate.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::silu_multiply_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::silu_multiply_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        gate.backend(),
        "unsupported backend buffers for SiLU multiply"
    );
}

BackendBuffer gated_silu_projection_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight
) {
    backends::metal::require_gated_silu_projection_f32_f16_gguf_weight_operands(
        left.descriptor(),
        gate_weight.descriptor(),
        up_weight.descriptor(),
        down_weight.descriptor()
    );
    backends::metal::require_same_backend(
        left.backend(),
        gate_weight.backend(),
        "gated SiLU projection"
    );
    backends::metal::require_same_backend(
        left.backend(),
        up_weight.backend(),
        "gated SiLU projection"
    );
    backends::metal::require_same_backend(
        left.backend(),
        down_weight.backend(),
        "gated SiLU projection"
    );

    switch (left.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(down_weight))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(down_weight))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for gated SiLU projection"
    );
}

BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    float epsilon
) {
    backends::metal::require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
        input.descriptor(),
        attention_output.descriptor(),
        attention_output_weight.descriptor(),
        feed_forward_norm_weight.descriptor(),
        gate_weight.descriptor(),
        up_weight.descriptor(),
        down_weight.descriptor(),
        epsilon
    );
    backends::metal::require_same_backend(input.backend(), attention_output.backend(), "residual gated SiLU projection");
    backends::metal::require_same_backend(
        input.backend(),
        attention_output_weight.backend(),
        "residual gated SiLU projection"
    );
    backends::metal::require_same_backend(
        input.backend(),
        feed_forward_norm_weight.backend(),
        "residual gated SiLU projection"
    );
    backends::metal::require_same_backend(input.backend(), gate_weight.backend(), "residual gated SiLU projection");
    backends::metal::require_same_backend(input.backend(), up_weight.backend(), "residual gated SiLU projection");
    backends::metal::require_same_backend(input.backend(), down_weight.backend(), "residual gated SiLU projection");

    switch (input.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::residual_gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(attention_output)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(attention_output_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(down_weight)),
                epsilon
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_output)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_output_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(down_weight)),
                    epsilon
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffers for residual gated SiLU projection"
    );
}

}  // namespace local_gguf_engine::tensor::ops
