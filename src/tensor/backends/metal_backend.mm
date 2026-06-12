#include "metal_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <limits>

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
)metal";

struct MatmulDimensions {
    std::uint32_t rows = 0;
    std::uint32_t shared = 0;
    std::uint32_t columns = 0;
};

struct MatmulPipelineState {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandQueue> queue = nil;

    MatmulPipelineState();
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

    id<MTLFunction> function = [library newFunctionWithName:@"matmul_f32_f16_gguf_weight_kernel"];
    if (function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul kernel function was not found"
        );
    }

    NSError* pipeline_error = nil;
    pipeline = [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul pipeline creation failed: " + ns_error_message(pipeline_error, "unknown error")
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

        [encoder setComputePipelineState:state.pipeline];
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
