#include "metal_benchmark.hpp"

#include "metal_buffer.hpp"
#include "metal_context.hpp"
#include "metal_llama_ops.hpp"
#include "metal_tensor_ops.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "../../ops/validation.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <mutex>
#include <stdexcept>

namespace local_gguf_engine::tensor::backends::metal {

void benchmark_noop_dispatch() {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark no-op command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.noop_pipeline];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark no-op command failed: " + ns_error_message(execution_error, "unknown error")
            );
        }
    }
}

BackendBuffer benchmark_llama_cached_decode_attention_norm_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_norm_weight,
    float rms_norm_epsilon
) {
    @autoreleasepool {
        require_same_backend(input.backend(), runtime::BackendKind::metal, "benchmark attention norm");
        require_same_backend(input.backend(), attention_norm_weight.backend(), "benchmark attention norm");
        ops::require_f32_descriptor(input.descriptor());
        require_matrix_descriptor(input.descriptor());
        require_f32_vector_descriptor(attention_norm_weight.descriptor());
        require_positive_epsilon(rms_norm_epsilon);

        const auto rows = input.descriptor().shape.dimension(0);
        const auto hidden = input.descriptor().shape.dimension(1);
        if (rows == 0) {
            throw std::invalid_argument("benchmark attention norm input rows must be positive");
        }
        if (attention_norm_weight.descriptor().shape.dimension(0) != hidden) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                hidden,
                attention_norm_weight.descriptor().shape.dimension(0)
            );
        }

        auto& state = shared_matmul_pipeline_state();
        const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
        const RMSNormDimensions rms_norm_dims {
            .rows = require_u32_dimension(rows, "rows"),
            .hidden = require_u32_dimension(hidden, "hidden"),
            .epsilon = rms_norm_epsilon,
        };

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> output_buffer = reusable_scratch_buffer(
            state.device,
            scratch.attention_normalized,
            output_descriptor.byte_count,
            "benchmark_attention_norm"
        );

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention norm command buffer creation failed"
            );
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention norm command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.rms_norm_pipeline];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(input)).buffer() offset:0 atIndex:0];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(attention_norm_weight)).buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&rms_norm_dims length:sizeof(RMSNormDimensions) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention norm command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return make_metal_backend_buffer(output_descriptor, output_buffer);
    }
}

std::array<BackendBuffer, 3> benchmark_llama_cached_decode_qkv_projection_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& query_weight,
    const BackendBuffer& key_weight,
    const BackendBuffer& value_weight
) {
    @autoreleasepool {
        require_same_backend(input.backend(), runtime::BackendKind::metal, "benchmark QKV projection");
        require_same_backend(input.backend(), query_weight.backend(), "benchmark QKV projection");
        require_same_backend(input.backend(), key_weight.backend(), "benchmark QKV projection");
        require_same_backend(input.backend(), value_weight.backend(), "benchmark QKV projection");
        require_f32_f16_matmul_gguf_weight_operands(input.descriptor(), query_weight.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(input.descriptor(), key_weight.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(input.descriptor(), value_weight.descriptor());

        const auto rows = input.descriptor().shape.dimension(0);
        const auto hidden = input.descriptor().shape.dimension(1);
        const auto query_width = query_weight.descriptor().shape.dimension(1);
        const auto key_width = key_weight.descriptor().shape.dimension(1);
        const auto value_width = value_weight.descriptor().shape.dimension(1);
        if (rows == 0) {
            throw std::invalid_argument("benchmark QKV projection input rows must be positive");
        }

        auto& state = shared_matmul_pipeline_state();
        const auto query_descriptor = TensorDescriptor::from(Shape::from({rows, query_width}), DType::f32);
        const auto key_descriptor = TensorDescriptor::from(Shape::from({rows, key_width}), DType::f32);
        const auto value_descriptor = TensorDescriptor::from(Shape::from({rows, value_width}), DType::f32);
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

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> query_buffer = reusable_scratch_buffer(
            state.device,
            scratch.query,
            query_descriptor.byte_count,
            "benchmark_query"
        );
        id<MTLBuffer> key_buffer = reusable_scratch_buffer(
            state.device,
            scratch.key,
            key_descriptor.byte_count,
            "benchmark_key"
        );
        id<MTLBuffer> value_buffer = reusable_scratch_buffer(
            state.device,
            scratch.value,
            value_descriptor.byte_count,
            "benchmark_value"
        );

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark QKV projection command buffer creation failed"
            );
        }

        encode_llama_cached_qkv_projection(
            state,
            command_buffer,
            require_metal_backend_buffer(backend_buffer_storage(input)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(query_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(key_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(value_weight)).buffer(),
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
            "benchmark QKV projection"
        );

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark QKV projection command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return std::array<BackendBuffer, 3> {
            make_metal_backend_buffer(query_descriptor, query_buffer),
            make_metal_backend_buffer(key_descriptor, key_buffer),
            make_metal_backend_buffer(value_descriptor, value_buffer),
        };
    }
}

void benchmark_llama_cached_decode_kv_append_to_backend(
    const BackendBuffer& key,
    const BackendBuffer& value,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    @autoreleasepool {
        require_same_backend(key.backend(), runtime::BackendKind::metal, "benchmark KV append");
        require_same_backend(key.backend(), value.backend(), "benchmark KV append");
        require_same_backend(key.backend(), cache_key.backend(), "benchmark KV append");
        require_same_backend(key.backend(), cache_value.backend(), "benchmark KV append");
        ops::require_f32_descriptor(key.descriptor());
        ops::require_f32_descriptor(value.descriptor());
        ops::require_f32_descriptor(cache_key.descriptor());
        ops::require_f32_descriptor(cache_value.descriptor());
        require_matrix_descriptor(key.descriptor());
        require_matrix_descriptor(value.descriptor());
        require_matrix_descriptor(cache_key.descriptor());
        require_matrix_descriptor(cache_value.descriptor());
        require_positive_frequency_base(frequency_base);

        const auto rows = key.descriptor().shape.dimension(0);
        const auto key_width = kv_head_count * key_length;
        const auto value_width = kv_head_count * value_length;
        if (rows == 0) {
            throw std::invalid_argument("benchmark KV append input rows must be positive");
        }
        if (kv_head_count == 0 || key_length == 0 || value_length == 0) {
            throw std::invalid_argument("benchmark KV append dimensions must be positive");
        }
        if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0 || rope_dimension_count > key_length) {
            throw TensorError(TensorErrorCode::invalid_shape_dimension, key_length, rope_dimension_count);
        }
        if (value.descriptor().shape.dimension(0) != rows) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, rows, value.descriptor().shape.dimension(0));
        }
        if (key.descriptor().shape.dimension(1) != key_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, key.descriptor().shape.dimension(1));
        }
        if (value.descriptor().shape.dimension(1) != value_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, value.descriptor().shape.dimension(1));
        }
        if (cache_key.descriptor().shape.dimension(1) != key_width) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                key_width,
                cache_key.descriptor().shape.dimension(1)
            );
        }
        if (cache_value.descriptor().shape.dimension(1) != value_width) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                value_width,
                cache_value.descriptor().shape.dimension(1)
            );
        }
        if (cache_key.descriptor().shape.dimension(0) != cache_value.descriptor().shape.dimension(0)) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                cache_key.descriptor().shape.dimension(0),
                cache_value.descriptor().shape.dimension(0)
            );
        }
        if (cache_position > cache_key.descriptor().shape.dimension(0)
            || rows > cache_key.descriptor().shape.dimension(0) - cache_position) {
            throw std::invalid_argument("benchmark KV append would exceed context length");
        }

        auto& state = shared_matmul_pipeline_state();
        const LlamaCachedAttentionDimensions attention_dims {
            .token_count = require_u32_dimension(rows, "token_count"),
            .cache_position = require_u32_dimension(cache_position, "cache_position"),
            .cached_token_count = require_u32_dimension(cache_position + rows, "cached_token_count"),
            .head_count = require_u32_dimension(kv_head_count, "head_count"),
            .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
            .key_length = require_u32_dimension(key_length, "key_length"),
            .value_length = require_u32_dimension(value_length, "value_length"),
            .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
            .frequency_base = frequency_base,
            .scale = 1.0F,
        };

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark KV append command buffer creation failed"
            );
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark KV append command encoding failed"
            );
        }
        [encoder setComputePipelineState:state.llama_cached_kv_append_pipeline];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(key)).buffer() offset:0 atIndex:0];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(value)).buffer() offset:0 atIndex:1];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(cache_key)).buffer() offset:0 atIndex:2];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(cache_value)).buffer() offset:0 atIndex:3];
        [encoder setBytes:&attention_dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];
        const auto append_columns = key_width > value_width ? key_width : value_width;
        [encoder dispatchThreadgroups:MTLSizeMake(
                                            static_cast<NSUInteger>((append_columns + 15U) / 16U),
                                            static_cast<NSUInteger>((rows + 15U) / 16U),
                                            1
                                        )
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark KV append command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }
    }
}

std::array<BackendBuffer, 3> benchmark_llama_cached_decode_attention_frontend_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_norm_weight,
    const BackendBuffer& query_weight,
    const BackendBuffer& key_weight,
    const BackendBuffer& value_weight,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float rms_norm_epsilon
) {
    @autoreleasepool {
        require_same_backend(input.backend(), runtime::BackendKind::metal, "benchmark attention front end");
        require_same_backend(input.backend(), attention_norm_weight.backend(), "benchmark attention front end");
        require_same_backend(input.backend(), query_weight.backend(), "benchmark attention front end");
        require_same_backend(input.backend(), key_weight.backend(), "benchmark attention front end");
        require_same_backend(input.backend(), value_weight.backend(), "benchmark attention front end");
        require_same_backend(input.backend(), cache_key.backend(), "benchmark attention front end");
        require_same_backend(input.backend(), cache_value.backend(), "benchmark attention front end");

        ops::require_f32_descriptor(input.descriptor());
        require_matrix_descriptor(input.descriptor());
        require_f32_vector_descriptor(attention_norm_weight.descriptor());
        require_positive_epsilon(rms_norm_epsilon);
        require_positive_frequency_base(frequency_base);

        const auto rows = input.descriptor().shape.dimension(0);
        const auto hidden = input.descriptor().shape.dimension(1);
        const auto query_width = head_count * key_length;
        const auto key_width = kv_head_count * key_length;
        const auto value_width = kv_head_count * value_length;
        if (rows == 0) {
            throw std::invalid_argument("benchmark attention front end input rows must be positive");
        }
        if (attention_norm_weight.descriptor().shape.dimension(0) != hidden) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                hidden,
                attention_norm_weight.descriptor().shape.dimension(0)
            );
        }
        require_f32_f16_matmul_gguf_weight_operands(input.descriptor(), query_weight.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(input.descriptor(), key_weight.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(input.descriptor(), value_weight.descriptor());
        if (query_weight.descriptor().shape.dimension(1) != query_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_width, query_weight.descriptor().shape.dimension(1));
        }
        if (key_weight.descriptor().shape.dimension(1) != key_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, key_weight.descriptor().shape.dimension(1));
        }
        if (value_weight.descriptor().shape.dimension(1) != value_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, value_weight.descriptor().shape.dimension(1));
        }

        const auto query_descriptor = TensorDescriptor::from(Shape::from({rows, query_width}), DType::f32);
        const auto key_descriptor = TensorDescriptor::from(Shape::from({rows, key_width}), DType::f32);
        const auto value_descriptor = TensorDescriptor::from(Shape::from({rows, value_width}), DType::f32);
        require_llama_cached_attention_f32_operands(
            query_descriptor,
            key_descriptor,
            value_descriptor,
            cache_key.descriptor(),
            cache_value.descriptor(),
            cache_position,
            head_count,
            kv_head_count,
            key_length,
            value_length,
            rope_dimension_count,
            frequency_base,
            1.0F
        );

        auto& state = shared_matmul_pipeline_state();
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
            .cached_token_count = require_u32_dimension(cache_position + rows, "cached_token_count"),
            .head_count = require_u32_dimension(head_count, "head_count"),
            .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
            .key_length = require_u32_dimension(key_length, "key_length"),
            .value_length = require_u32_dimension(value_length, "value_length"),
            .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
            .frequency_base = frequency_base,
            .scale = 1.0F,
        };

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> normalized_buffer = reusable_scratch_buffer(
            state.device,
            scratch.attention_normalized,
            TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32).byte_count,
            "benchmark_attention_norm"
        );
        id<MTLBuffer> query_buffer = reusable_scratch_buffer(
            state.device,
            scratch.query,
            query_descriptor.byte_count,
            "benchmark_query"
        );
        id<MTLBuffer> key_buffer = reusable_scratch_buffer(
            state.device,
            scratch.key,
            key_descriptor.byte_count,
            "benchmark_key"
        );
        id<MTLBuffer> value_buffer = reusable_scratch_buffer(
            state.device,
            scratch.value,
            value_descriptor.byte_count,
            "benchmark_value"
        );

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention front end command buffer creation failed"
            );
        }

        auto encode_rms_norm = [&]() {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw metal_error(
                    runtime::BackendErrorCode::metal_execution_failed,
                    "Metal benchmark attention front end RMSNorm encoding failed"
                );
            }

            [encoder setComputePipelineState:state.rms_norm_pipeline];
            [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(input)).buffer() offset:0 atIndex:0];
            [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(attention_norm_weight)).buffer()
                         offset:0
                        atIndex:1];
            [encoder setBuffer:normalized_buffer offset:0 atIndex:2];
            [encoder setBytes:&rms_norm_dims length:sizeof(RMSNormDimensions) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
        };

        encode_rms_norm();
        encode_llama_cached_qkv_projection(
            state,
            command_buffer,
            normalized_buffer,
            require_metal_backend_buffer(backend_buffer_storage(query_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(key_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(value_weight)).buffer(),
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
            "benchmark attention front end QKV projection"
        );

        id<MTLComputeCommandEncoder> append_encoder = [command_buffer computeCommandEncoder];
        if (append_encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention front end KV append encoding failed"
            );
        }
        [append_encoder setComputePipelineState:state.llama_cached_kv_append_pipeline];
        [append_encoder setBuffer:key_buffer offset:0 atIndex:0];
        [append_encoder setBuffer:value_buffer offset:0 atIndex:1];
        [append_encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(cache_key)).buffer()
                          offset:0
                         atIndex:2];
        [append_encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(cache_value)).buffer()
                          offset:0
                         atIndex:3];
        [append_encoder setBytes:&attention_dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];
        const auto append_columns = key_width > value_width ? key_width : value_width;
        [append_encoder dispatchThreadgroups:MTLSizeMake(
                                                   static_cast<NSUInteger>((append_columns + 15U) / 16U),
                                                   static_cast<NSUInteger>((rows + 15U) / 16U),
                                                   1
                                               )
                        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [append_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention front end command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return std::array<BackendBuffer, 3> {
            make_metal_backend_buffer(query_descriptor, query_buffer),
            make_metal_backend_buffer(key_descriptor, key_buffer),
            make_metal_backend_buffer(value_descriptor, value_buffer),
        };
    }
}

BackendBuffer benchmark_llama_cached_decode_attention_output_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& cache_key,
    const BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    @autoreleasepool {
        require_same_backend(query.backend(), runtime::BackendKind::metal, "benchmark cached attention output");
        require_same_backend(query.backend(), cache_key.backend(), "benchmark cached attention output");
        require_same_backend(query.backend(), cache_value.backend(), "benchmark cached attention output");
        ops::require_f32_descriptor(query.descriptor());
        ops::require_f32_descriptor(cache_key.descriptor());
        ops::require_f32_descriptor(cache_value.descriptor());
        require_matrix_descriptor(query.descriptor());
        require_matrix_descriptor(cache_key.descriptor());
        require_matrix_descriptor(cache_value.descriptor());
        require_positive_frequency_base(frequency_base);
        require_positive_finite_scale(scale);

        const auto rows = query.descriptor().shape.dimension(0);
        const auto query_width = head_count * key_length;
        const auto key_width = kv_head_count * key_length;
        const auto value_width = kv_head_count * value_length;
        const auto output_width = head_count * value_length;
        if (rows == 0) {
            throw std::invalid_argument("benchmark cached attention output query rows must be positive");
        }
        if (head_count == 0 || kv_head_count == 0 || key_length == 0 || value_length == 0) {
            throw std::invalid_argument("benchmark cached attention output dimensions must be positive");
        }
        if (head_count % kv_head_count != 0) {
            throw std::invalid_argument("Llama attention head count must be divisible by KV head count");
        }
        if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0 || rope_dimension_count > key_length) {
            throw TensorError(TensorErrorCode::invalid_shape_dimension, key_length, rope_dimension_count);
        }
        if (query.descriptor().shape.dimension(1) != query_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_width, query.descriptor().shape.dimension(1));
        }
        if (cache_key.descriptor().shape.dimension(1) != key_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, cache_key.descriptor().shape.dimension(1));
        }
        if (cache_value.descriptor().shape.dimension(1) != value_width) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, cache_value.descriptor().shape.dimension(1));
        }
        if (cache_value.descriptor().shape.dimension(0) != cache_key.descriptor().shape.dimension(0)) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                cache_key.descriptor().shape.dimension(0),
                cache_value.descriptor().shape.dimension(0)
            );
        }
        if (cache_position > cache_key.descriptor().shape.dimension(0)
            || rows > cache_key.descriptor().shape.dimension(0) - cache_position) {
            throw std::invalid_argument("benchmark cached attention output would exceed context length");
        }

        auto& state = shared_matmul_pipeline_state();
        const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, output_width}), DType::f32);
        const LlamaCachedAttentionDimensions attention_dims {
            .token_count = require_u32_dimension(rows, "token_count"),
            .cache_position = require_u32_dimension(cache_position, "cache_position"),
            .cached_token_count = require_u32_dimension(cache_position + rows, "cached_token_count"),
            .head_count = require_u32_dimension(head_count, "head_count"),
            .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
            .key_length = require_u32_dimension(key_length, "key_length"),
            .value_length = require_u32_dimension(value_length, "value_length"),
            .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
            .frequency_base = frequency_base,
            .scale = scale,
        };

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> output_buffer = reusable_scratch_buffer(
            state.device,
            scratch.attention_output,
            output_descriptor.byte_count,
            "benchmark_attention_output"
        );

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark cached attention output command encoding failed"
            );
        }
        encode_llama_cached_attention_output(
            state,
            encoder,
            require_metal_backend_buffer(backend_buffer_storage(query)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(cache_key)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(cache_value)).buffer(),
            output_buffer,
            attention_dims,
            output_width
        );
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark cached attention output command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return make_metal_backend_buffer(output_descriptor, output_buffer);
    }
}

BackendBuffer benchmark_llama_cached_decode_block_tail_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    float rms_norm_epsilon
) {
    @autoreleasepool {
        require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
            input.descriptor(),
            attention_output.descriptor(),
            attention_output_weight.descriptor(),
            feed_forward_norm_weight.descriptor(),
            gate_weight.descriptor(),
            up_weight.descriptor(),
            down_weight.descriptor(),
            rms_norm_epsilon
        );
        require_same_backend(input.backend(), runtime::BackendKind::metal, "benchmark cached block tail");
        require_same_backend(input.backend(), attention_output.backend(), "benchmark cached block tail");
        require_same_backend(input.backend(), attention_output_weight.backend(), "benchmark cached block tail");
        require_same_backend(input.backend(), feed_forward_norm_weight.backend(), "benchmark cached block tail");
        require_same_backend(input.backend(), gate_weight.backend(), "benchmark cached block tail");
        require_same_backend(input.backend(), up_weight.backend(), "benchmark cached block tail");
        require_same_backend(input.backend(), down_weight.backend(), "benchmark cached block tail");

        auto& state = shared_matmul_pipeline_state();
        const auto rows = input.descriptor().shape.dimension(0);
        const auto hidden = input.descriptor().shape.dimension(1);
        const auto attention_output_width = attention_output.descriptor().shape.dimension(1);
        const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
        const auto hidden_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
        const auto feed_forward_descriptor = TensorDescriptor::from(Shape::from({rows, feed_forward}), DType::f32);
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
        const RMSNormDimensions rms_norm_dims {
            .rows = require_u32_dimension(rows, "rows"),
            .hidden = require_u32_dimension(hidden, "hidden"),
            .epsilon = rms_norm_epsilon,
        };

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> attention_residual_buffer = reusable_scratch_buffer(
            state.device,
            scratch.attention_residual,
            hidden_descriptor.byte_count,
            "benchmark_attention_residual"
        );
        id<MTLBuffer> feed_forward_normalized_buffer = reusable_scratch_buffer(
            state.device,
            scratch.feed_forward_normalized,
            hidden_descriptor.byte_count,
            "benchmark_ffn_norm"
        );
        id<MTLBuffer> gate_buffer = reusable_scratch_buffer(
            state.device,
            scratch.gate,
            feed_forward_descriptor.byte_count,
            "benchmark_gate"
        );
        id<MTLBuffer> up_buffer = reusable_scratch_buffer(
            state.device,
            scratch.up,
            feed_forward_descriptor.byte_count,
            "benchmark_up"
        );
        id<MTLBuffer> gated_buffer = reusable_scratch_buffer(
            state.device,
            scratch.gated,
            feed_forward_descriptor.byte_count,
            "benchmark_gated"
        );
        id<MTLBuffer> output_buffer =
            [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                      options:MTLResourceStorageModeShared];
        if (output_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark cached block tail output buffer allocation failed"
            );
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark cached block tail command buffer creation failed"
            );
        }

        const auto& input_storage = require_metal_backend_buffer(backend_buffer_storage(input));
        const auto& attention_output_storage = require_metal_backend_buffer(backend_buffer_storage(attention_output));
        encode_llama_cached_block_tail(
            state,
            command_buffer,
            input_storage.buffer(),
            attention_output_storage.buffer(),
            require_metal_backend_buffer(backend_buffer_storage(attention_output_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(gate_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(up_weight)).buffer(),
            require_metal_backend_buffer(backend_buffer_storage(down_weight)).buffer(),
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
            "benchmark cached block tail"
        );

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark cached block tail command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return make_metal_backend_buffer(hidden_descriptor, output_buffer);
    }
}

BackendBuffer benchmark_llama_cached_decode_attention_residual_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight
) {
    @autoreleasepool {
        require_same_backend(input.backend(), runtime::BackendKind::metal, "benchmark attention residual");
        require_same_backend(input.backend(), attention_output.backend(), "benchmark attention residual");
        require_same_backend(input.backend(), attention_output_weight.backend(), "benchmark attention residual");
        ops::require_f32_descriptor(input.descriptor());
        ops::require_f32_descriptor(attention_output.descriptor());
        require_matrix_descriptor(input.descriptor());
        require_matrix_descriptor(attention_output.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(
            attention_output.descriptor(),
            attention_output_weight.descriptor()
        );

        const auto rows = input.descriptor().shape.dimension(0);
        const auto hidden = input.descriptor().shape.dimension(1);
        const auto attention_rows = attention_output.descriptor().shape.dimension(0);
        const auto attention_output_width = attention_output.descriptor().shape.dimension(1);
        if (rows == 0) {
            throw std::invalid_argument("benchmark attention residual input rows must be positive");
        }
        if (attention_rows != rows) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, rows, attention_rows);
        }
        if (attention_output_weight.descriptor().shape.dimension(1) != hidden) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                hidden,
                attention_output_weight.descriptor().shape.dimension(1)
            );
        }

        auto& state = shared_matmul_pipeline_state();
        const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
        const MatmulDimensions attention_output_dims {
            .rows = require_u32_dimension(rows, "rows"),
            .shared = require_u32_dimension(attention_output_width, "attention_output_width"),
            .columns = require_u32_dimension(hidden, "hidden"),
        };

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> output_buffer = reusable_scratch_buffer(
            state.device,
            scratch.attention_residual,
            output_descriptor.byte_count,
            "benchmark_attention_residual"
        );

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention residual command buffer creation failed"
            );
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention residual command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_add_pipeline];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(attention_output)).buffer()
                    offset:0
                   atIndex:0];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(attention_output_weight)).buffer()
                    offset:0
                   atIndex:1];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(input)).buffer() offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBytes:&attention_output_dims length:sizeof(MatmulDimensions) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                                            static_cast<NSUInteger>((hidden + 15U) / 16U),
                                            static_cast<NSUInteger>((rows + 15U) / 16U),
                                            1
                                        )
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark attention residual command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return make_metal_backend_buffer(output_descriptor, output_buffer);
    }
}

BackendBuffer benchmark_llama_cached_decode_ffn_gate_up_to_backend(
    const BackendBuffer& attention_residual,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    float rms_norm_epsilon
) {
    @autoreleasepool {
        require_same_backend(attention_residual.backend(), runtime::BackendKind::metal, "benchmark FFN gate/up");
        require_same_backend(attention_residual.backend(), feed_forward_norm_weight.backend(), "benchmark FFN gate/up");
        require_same_backend(attention_residual.backend(), gate_weight.backend(), "benchmark FFN gate/up");
        require_same_backend(attention_residual.backend(), up_weight.backend(), "benchmark FFN gate/up");
        ops::require_f32_descriptor(attention_residual.descriptor());
        require_matrix_descriptor(attention_residual.descriptor());
        require_f32_vector_descriptor(feed_forward_norm_weight.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(attention_residual.descriptor(), gate_weight.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(attention_residual.descriptor(), up_weight.descriptor());
        require_positive_epsilon(rms_norm_epsilon);

        const auto rows = attention_residual.descriptor().shape.dimension(0);
        const auto hidden = attention_residual.descriptor().shape.dimension(1);
        const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
        if (rows == 0) {
            throw std::invalid_argument("benchmark FFN gate/up input rows must be positive");
        }
        if (feed_forward_norm_weight.descriptor().shape.dimension(0) != hidden) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                hidden,
                feed_forward_norm_weight.descriptor().shape.dimension(0)
            );
        }
        if (up_weight.descriptor().shape.dimension(1) != feed_forward) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                feed_forward,
                up_weight.descriptor().shape.dimension(1)
            );
        }

        auto& state = shared_matmul_pipeline_state();
        const auto hidden_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
        const auto feed_forward_descriptor = TensorDescriptor::from(Shape::from({rows, feed_forward}), DType::f32);
        const RMSNormDimensions rms_norm_dims {
            .rows = require_u32_dimension(rows, "rows"),
            .hidden = require_u32_dimension(hidden, "hidden"),
            .epsilon = rms_norm_epsilon,
        };
        const MatmulDimensions gate_dims {
            .rows = require_u32_dimension(rows, "rows"),
            .shared = require_u32_dimension(hidden, "hidden"),
            .columns = require_u32_dimension(feed_forward, "feed_forward"),
        };
        const ElementwiseDimensions feed_forward_elementwise_dims {
            .elements = require_u32_dimension(feed_forward_descriptor.element_count, "feed_forward_elements"),
        };

        std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
        auto& scratch = state.llama_cached_block_scratch;
        id<MTLBuffer> normalized_buffer = reusable_scratch_buffer(
            state.device,
            scratch.feed_forward_normalized,
            hidden_descriptor.byte_count,
            "benchmark_ffn_norm"
        );
        id<MTLBuffer> gate_buffer = reusable_scratch_buffer(
            state.device,
            scratch.gate,
            feed_forward_descriptor.byte_count,
            "benchmark_gate"
        );
        id<MTLBuffer> up_buffer = reusable_scratch_buffer(
            state.device,
            scratch.up,
            feed_forward_descriptor.byte_count,
            "benchmark_up"
        );
        id<MTLBuffer> gated_buffer = reusable_scratch_buffer(
            state.device,
            scratch.gated,
            feed_forward_descriptor.byte_count,
            "benchmark_gated"
        );

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN gate/up command buffer creation failed"
            );
        }

        auto encode_rms_norm = [&]() {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw metal_error(
                    runtime::BackendErrorCode::metal_execution_failed,
                    "Metal benchmark FFN gate/up RMSNorm encoding failed"
                );
            }

            [encoder setComputePipelineState:state.rms_norm_pipeline];
            [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(attention_residual)).buffer()
                        offset:0
                       atIndex:0];
            [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)).buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:normalized_buffer offset:0 atIndex:2];
            [encoder setBytes:&rms_norm_dims length:sizeof(RMSNormDimensions) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
        };

        auto encode_matmul = [&](const BackendBuffer& right, id<MTLBuffer> output_buffer) {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw metal_error(
                    runtime::BackendErrorCode::metal_execution_failed,
                    "Metal benchmark FFN gate/up matmul encoding failed"
                );
            }

            [encoder setComputePipelineState:state.matmul_pipeline];
            [encoder setBuffer:normalized_buffer offset:0 atIndex:0];
            [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(right)).buffer() offset:0 atIndex:1];
            [encoder setBuffer:output_buffer offset:0 atIndex:2];
            [encoder setBytes:&gate_dims length:sizeof(MatmulDimensions) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(
                                                static_cast<NSUInteger>((feed_forward + 15U) / 16U),
                                                static_cast<NSUInteger>((rows + 15U) / 16U),
                                                1
                                            )
                     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [encoder endEncoding];
        };

        encode_rms_norm();
        encode_matmul(gate_weight, gate_buffer);
        encode_matmul(up_weight, up_buffer);

        id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
        if (silu_encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN gate/up activation encoding failed"
            );
        }
        [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
        [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
        [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
        [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
        [silu_encoder setBytes:&feed_forward_elementwise_dims length:sizeof(ElementwiseDimensions) atIndex:3];
        [silu_encoder dispatchThreadgroups:MTLSizeMake(
                                                static_cast<NSUInteger>(
                                                    (feed_forward_descriptor.element_count + 255U) / 256U
                                                ),
                                                1,
                                                1
                                            )
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [silu_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN gate/up command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return make_metal_backend_buffer(feed_forward_descriptor, gated_buffer);
    }
}

BackendBuffer benchmark_llama_cached_decode_ffn_down_to_backend(
    const BackendBuffer& gated,
    const BackendBuffer& attention_residual,
    const BackendBuffer& down_weight
) {
    @autoreleasepool {
        require_same_backend(gated.backend(), runtime::BackendKind::metal, "benchmark FFN down");
        require_same_backend(gated.backend(), attention_residual.backend(), "benchmark FFN down");
        require_same_backend(gated.backend(), down_weight.backend(), "benchmark FFN down");
        ops::require_f32_descriptor(gated.descriptor());
        ops::require_f32_descriptor(attention_residual.descriptor());
        require_matrix_descriptor(gated.descriptor());
        require_matrix_descriptor(attention_residual.descriptor());
        require_f32_f16_matmul_gguf_weight_operands(gated.descriptor(), down_weight.descriptor());

        const auto rows = gated.descriptor().shape.dimension(0);
        const auto feed_forward = gated.descriptor().shape.dimension(1);
        const auto residual_rows = attention_residual.descriptor().shape.dimension(0);
        const auto hidden = attention_residual.descriptor().shape.dimension(1);
        if (rows == 0) {
            throw std::invalid_argument("benchmark FFN down input rows must be positive");
        }
        if (residual_rows != rows) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, rows, residual_rows);
        }
        if (down_weight.descriptor().shape.dimension(1) != hidden) {
            throw TensorError(
                TensorErrorCode::tensor_shape_mismatch,
                hidden,
                down_weight.descriptor().shape.dimension(1)
            );
        }

        auto& state = shared_matmul_pipeline_state();
        const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
        const MatmulDimensions down_dims {
            .rows = require_u32_dimension(rows, "rows"),
            .shared = require_u32_dimension(feed_forward, "feed_forward"),
            .columns = require_u32_dimension(hidden, "hidden"),
        };

        id<MTLBuffer> output_buffer =
            [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                      options:MTLResourceStorageModeShared];
        if (output_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN down output buffer allocation failed"
            );
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        if (command_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN down command buffer creation failed"
            );
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN down command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_add_pipeline];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(gated)).buffer() offset:0 atIndex:0];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(down_weight)).buffer() offset:0 atIndex:1];
        [encoder setBuffer:require_metal_backend_buffer(backend_buffer_storage(attention_residual)).buffer()
                    offset:0
                   atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBytes:&down_dims length:sizeof(MatmulDimensions) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(
                                            static_cast<NSUInteger>((hidden + 15U) / 16U),
                                            static_cast<NSUInteger>((rows + 15U) / 16U),
                                            1
                                        )
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal benchmark FFN down command failed: "
                    + ns_error_message(execution_error, "unknown error")
            );
        }

        return make_metal_backend_buffer(output_descriptor, output_buffer);
    }
}


}  // namespace local_gguf_engine::tensor::backends::metal
