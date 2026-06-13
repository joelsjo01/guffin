#include "metal_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "../ops/validation.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

constexpr const char* matmul_f32_f16_gguf_weight_source = R"metal(
#include <metal_stdlib>
using namespace metal;

struct MatmulDimensions {
    uint rows;
    uint shared;
    uint columns;
};

kernel void matmul_f32_f16_gguf_weight_kernel(
    const device float* left [[buffer(0)]],
    const device half* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    if (row >= dims.rows || column >= dims.columns) {
        return;
    }

    float sum = 0.0f;
    for (uint inner = 0; inner < dims.shared; ++inner) {
        sum += left[row * dims.shared + inner] * float(right[column * dims.shared + inner]);
    }

    output[row * dims.columns + column] = sum;
}

struct ElementwiseDimensions {
    uint elements;
};

kernel void silu_multiply_f32_kernel(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseDimensions& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.elements) {
        return;
    }

    const float value = gate[gid];
    output[gid] = (value / (1.0f + exp(-value))) * up[gid];
}

kernel void add_f32_kernel(
    const device float* left [[buffer(0)]],
    const device float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseDimensions& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.elements) {
        return;
    }

    output[gid] = left[gid] + right[gid];
}

struct RMSNormDimensions {
    uint rows;
    uint hidden;
    float epsilon;
};

kernel void rms_norm_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormDimensions& dims [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= dims.rows) {
        return;
    }

    float sum_squares = 0.0f;
    const uint row_offset = row * dims.hidden;
    for (uint column = 0; column < dims.hidden; ++column) {
        const float value = input[row_offset + column];
        sum_squares += value * value;
    }

    const float inverse_rms = rsqrt((sum_squares / float(dims.hidden)) + dims.epsilon);
    for (uint column = 0; column < dims.hidden; ++column) {
        output[row_offset + column] = input[row_offset + column] * inverse_rms * weight[column];
    }
}
)metal";

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

struct MatmulPipelineState {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> silu_multiply_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLCommandQueue> queue = nil;

    MatmulPipelineState();
};

class CpuBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    explicit CpuBackendBufferStorage(const TensorStorage& storage)
        : storage_(TensorStorage::from_bytes(storage.descriptor(), storage.bytes())) {}

    [[nodiscard]] runtime::BackendKind backend() const noexcept override {
        return runtime::BackendKind::cpu;
    }

    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override {
        return storage_.descriptor();
    }

    [[nodiscard]] const TensorStorage& storage() const noexcept {
        return storage_;
    }

private:
    TensorStorage storage_;
};

class MetalBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    MetalBackendBufferStorage(const TensorStorage& storage, id<MTLBuffer> buffer)
        : MetalBackendBufferStorage(storage.descriptor(), buffer) {}

    MetalBackendBufferStorage(TensorDescriptor descriptor, id<MTLBuffer> buffer)
        : descriptor_(std::move(descriptor)),
          buffer_(buffer) {}

    [[nodiscard]] runtime::BackendKind backend() const noexcept override {
        return runtime::BackendKind::metal;
    }

    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }

    [[nodiscard]] id<MTLBuffer> buffer() const noexcept {
        return buffer_;
    }

private:
    TensorDescriptor descriptor_;
    id<MTLBuffer> buffer_ = nil;
};

[[nodiscard]] std::string ns_error_message(NSError* error, const char* fallback) {
    if (error == nil) {
        return fallback;
    }

    NSString* description = [error localizedDescription];
    if (description == nil) {
        return fallback;
    }

    const char* utf8 = [description UTF8String];
    if (utf8 == nullptr) {
        return fallback;
    }

    return std::string(utf8);
}

[[nodiscard]] runtime::BackendError metal_error(
    runtime::BackendErrorCode code,
    const std::string& detail
) {
    return runtime::BackendError(code, runtime::BackendKind::metal, detail);
}

[[nodiscard]] id<MTLDevice> metal_device_or_throw() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw metal_error(
            runtime::BackendErrorCode::unavailable_metal_device,
            "Metal backend unavailable: no system default Metal device"
        );
    }

    return device;
}

[[nodiscard]] std::uint32_t require_u32_dimension(std::uint64_t value, const char* name) {
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

void require_matmul_inner_dimensions(const TensorDescriptor& left, const TensorDescriptor& right) {
    const auto left_columns = left.shape.dimension(1);
    const auto right_rows = right.shape.dimension(0);
    if (left_columns != right_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_columns, right_rows);
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

[[nodiscard]] const CpuBackendBufferStorage& require_cpu_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const CpuBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain CPU storage"
        );
    }

    return *typed_storage;
}

[[nodiscard]] const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const MetalBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain Metal storage"
        );
    }

    return *typed_storage;
}

MatmulPipelineState::MatmulPipelineState() {
    device = metal_device_or_throw();

    NSError* library_error = nil;
    NSString* source = [NSString stringWithUTF8String:matmul_f32_f16_gguf_weight_source];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&library_error];
    if (library == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul library compilation failed: " + ns_error_message(library_error, "unknown error")
        );
    }

    id<MTLFunction> matmul_function = [library newFunctionWithName:@"matmul_f32_f16_gguf_weight_kernel"];
    if (matmul_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul kernel function was not found"
        );
    }

    NSError* pipeline_error = nil;
    matmul_pipeline = [device newComputePipelineStateWithFunction:matmul_function error:&pipeline_error];
    if (matmul_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul pipeline creation failed: " + ns_error_message(pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> silu_multiply_function = [library newFunctionWithName:@"silu_multiply_f32_kernel"];
    if (silu_multiply_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply kernel function was not found"
        );
    }

    NSError* silu_multiply_pipeline_error = nil;
    silu_multiply_pipeline = [device newComputePipelineStateWithFunction:silu_multiply_function
                                                                   error:&silu_multiply_pipeline_error];
    if (silu_multiply_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply pipeline creation failed: "
                + ns_error_message(silu_multiply_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> add_function = [library newFunctionWithName:@"add_f32_kernel"];
    if (add_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add kernel function was not found"
        );
    }

    NSError* add_pipeline_error = nil;
    add_pipeline = [device newComputePipelineStateWithFunction:add_function error:&add_pipeline_error];
    if (add_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add pipeline creation failed: " + ns_error_message(add_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> rms_norm_function = [library newFunctionWithName:@"rms_norm_f32_kernel"];
    if (rms_norm_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm kernel function was not found"
        );
    }

    NSError* rms_norm_pipeline_error = nil;
    rms_norm_pipeline = [device newComputePipelineStateWithFunction:rms_norm_function
                                                             error:&rms_norm_pipeline_error];
    if (rms_norm_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm pipeline creation failed: " + ns_error_message(rms_norm_pipeline_error, "unknown error")
        );
    }

    queue = [device newCommandQueue];
    if (queue == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul command queue creation failed"
        );
    }
}

[[nodiscard]] MatmulPipelineState& shared_matmul_pipeline_state() {
    static MatmulPipelineState state;
    return state;
}

[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight_with_metal_right_buffer(
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

[[nodiscard]] BackendBuffer matmul_f32_f16_gguf_weight_with_metal_buffers(
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

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] ops::F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
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
        .first = detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(first_descriptor, first_buffer)
        ),
        .second = detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(second_descriptor, second_buffer)
        ),
        .third = detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(third_descriptor, third_buffer)
        ),
    };
}

[[nodiscard]] BackendBuffer silu_multiply_f32_with_metal_buffers(
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

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
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

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
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

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(hidden_descriptor, output_buffer)
    );
}

}  // namespace

bool is_available() noexcept {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

std::string unavailable_reason() {
    @autoreleasepool {
        if (MTLCreateSystemDefaultDevice() != nil) {
            return {};
        }

        return "Metal backend unavailable: no system default Metal device";
    }
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

}  // namespace local_gguf_engine::tensor::backends::metal

namespace local_gguf_engine::tensor::detail {

BackendBufferStorage::~BackendBufferStorage() = default;

BackendBuffer make_backend_buffer(std::shared_ptr<BackendBufferStorage> storage) {
    return BackendBuffer(std::move(storage));
}

}  // namespace local_gguf_engine::tensor::detail

namespace local_gguf_engine::tensor {

BackendBuffer::BackendBuffer(std::shared_ptr<detail::BackendBufferStorage> storage)
    : storage_(std::move(storage)) {}

runtime::BackendKind BackendBuffer::backend() const noexcept {
    return storage_->backend();
}

const TensorDescriptor& BackendBuffer::descriptor() const noexcept {
    return storage_->descriptor();
}

BackendBuffer copy_tensor_to_backend(const TensorStorage& storage, runtime::BackendKind backend) {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return BackendBuffer(std::make_shared<backends::metal::CpuBackendBufferStorage>(storage));
        case runtime::BackendKind::metal: {
            @autoreleasepool {
                auto& state = backends::metal::shared_matmul_pipeline_state();
                id<MTLBuffer> buffer = [state.device newBufferWithBytes:storage.bytes().data()
                                                                  length:static_cast<NSUInteger>(storage.bytes().size())
                                                                 options:MTLResourceStorageModeShared];
                if (buffer == nil) {
                    throw runtime::BackendError(
                        runtime::BackendErrorCode::metal_execution_failed,
                        runtime::BackendKind::metal,
                        "Metal backend buffer allocation failed"
                    );
                }

                return BackendBuffer(
                    std::make_shared<backends::metal::MetalBackendBufferStorage>(storage, buffer)
                );
            }
        }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for tensor buffer copy"
    );
}

TensorStorage copy_tensor_from_backend(const BackendBuffer& buffer) {
    const auto& storage = backend_buffer_storage(buffer);
    switch (storage.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& cpu_storage = backends::metal::require_cpu_backend_buffer(storage).storage();
            return TensorStorage::from_bytes(cpu_storage.descriptor(), cpu_storage.bytes());
        }
        case runtime::BackendKind::metal: {
            @autoreleasepool {
                const auto& metal_storage = backends::metal::require_metal_backend_buffer(storage);
                auto output = TensorStorage::allocate(metal_storage.descriptor());
                std::memcpy(
                    output.mutable_bytes().data(),
                    [metal_storage.buffer() contents],
                    output.bytes().size()
                );
                return output;
            }
        }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        storage.backend(),
        "unsupported backend for tensor buffer readback"
    );
}

const detail::BackendBufferStorage& backend_buffer_storage(const BackendBuffer& buffer) noexcept {
    return *buffer.storage_;
}

}  // namespace local_gguf_engine::tensor

namespace local_gguf_engine::tensor::ops {

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const BackendBuffer& right) {
    backends::metal::require_f32_f16_matmul_gguf_weight_operands(left, right.descriptor());

    switch (right.backend()) {
        case runtime::BackendKind::cpu:
            return matmul_f32_f16_gguf_weight(
                left,
                backends::metal::require_cpu_backend_buffer(backend_buffer_storage(right)).storage(),
                runtime::BackendKind::cpu
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
        case runtime::BackendKind::cpu: {
            const auto& left_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(left)
            ).storage();
            const auto& right_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(right)
            ).storage();
            const auto output = matmul_f32_f16_gguf_weight(
                left_storage,
                right_storage,
                runtime::BackendKind::cpu
            );
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
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
        case runtime::BackendKind::cpu: {
            const auto& left_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(left)
            ).storage();
            const auto& first_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(first_right)
            ).storage();
            const auto& second_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(second_right)
            ).storage();
            const auto& third_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(third_right)
            ).storage();
            const auto first_output = matmul_f32_f16_gguf_weight(
                left_storage,
                first_storage,
                runtime::BackendKind::cpu
            );
            const auto second_output = matmul_f32_f16_gguf_weight(
                left_storage,
                second_storage,
                runtime::BackendKind::cpu
            );
            const auto third_output = matmul_f32_f16_gguf_weight(
                left_storage,
                third_storage,
                runtime::BackendKind::cpu
            );
            return F32F16GGUFWeightTripleMatmulBackendBuffers {
                .first = copy_tensor_to_backend(first_output, runtime::BackendKind::cpu),
                .second = copy_tensor_to_backend(second_output, runtime::BackendKind::cpu),
                .third = copy_tensor_to_backend(third_output, runtime::BackendKind::cpu),
            };
        }
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
        case runtime::BackendKind::cpu: {
            const auto& gate_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(gate)
            ).storage();
            const auto& up_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(up)
            ).storage();
            const auto output = multiply_f32(silu_f32(gate_storage), up_storage);
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
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
        case runtime::BackendKind::cpu: {
            const auto& left_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(left)
            ).storage();
            const auto& gate_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(gate_weight)
            ).storage();
            const auto& up_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(up_weight)
            ).storage();
            const auto& down_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(down_weight)
            ).storage();
            const auto gate = matmul_f32_f16_gguf_weight(
                left_storage,
                gate_storage,
                runtime::BackendKind::cpu
            );
            const auto up = matmul_f32_f16_gguf_weight(
                left_storage,
                up_storage,
                runtime::BackendKind::cpu
            );
            const auto gated = multiply_f32(silu_f32(gate), up);
            const auto output = matmul_f32_f16_gguf_weight(
                gated,
                down_storage,
                runtime::BackendKind::cpu
            );
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
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
        case runtime::BackendKind::cpu: {
            const auto& input_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(input)
            ).storage();
            const auto& attention_output_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(attention_output)
            ).storage();
            const auto& attention_output_weight_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(attention_output_weight)
            ).storage();
            const auto& feed_forward_norm_weight_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(feed_forward_norm_weight)
            ).storage();
            const auto& gate_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(gate_weight)
            ).storage();
            const auto& up_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(up_weight)
            ).storage();
            const auto& down_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(down_weight)
            ).storage();

            const auto attention_projected = matmul_f32_f16_gguf_weight(
                attention_output_storage,
                attention_output_weight_storage,
                runtime::BackendKind::cpu
            );
            const auto attention_residual = add_f32(input_storage, attention_projected);
            const auto normalized = rms_norm_f32(
                attention_residual,
                feed_forward_norm_weight_storage,
                epsilon
            );
            const auto gate = matmul_f32_f16_gguf_weight(
                normalized,
                gate_storage,
                runtime::BackendKind::cpu
            );
            const auto up = matmul_f32_f16_gguf_weight(
                normalized,
                up_storage,
                runtime::BackendKind::cpu
            );
            const auto gated = multiply_f32(silu_f32(gate), up);
            const auto feed_forward_projected = matmul_f32_f16_gguf_weight(
                gated,
                down_storage,
                runtime::BackendKind::cpu
            );
            const auto output = add_f32(attention_residual, feed_forward_projected);
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
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
