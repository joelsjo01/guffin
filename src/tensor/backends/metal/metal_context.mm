#include "metal_context.hpp"

#include <stdexcept>

namespace local_gguf_engine::tensor::backends::metal {

std::string ns_error_message(NSError* error, const char* fallback) {
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

runtime::BackendError metal_error(
    runtime::BackendErrorCode code,
    const std::string& detail
) {
    return runtime::BackendError(code, runtime::BackendKind::metal, detail);
}

id<MTLDevice> system_metal_device() noexcept {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLDevice> metal_device_or_throw() {
    id<MTLDevice> device = system_metal_device();
    if (device == nil) {
        throw metal_error(
            runtime::BackendErrorCode::unavailable_metal_device,
            "Metal backend unavailable: no system default Metal device"
        );
    }

    return device;
}

MatmulPipelineState::MatmulPipelineState() {
    device = metal_device_or_throw();

    NSError* library_error = nil;
    NSString* source = [NSString stringWithUTF8String:metal_kernel_source()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&library_error];
    if (library == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul library compilation failed: " + ns_error_message(library_error, "unknown error")
        );
    }

    id<MTLFunction> matmul_function =
        [library newFunctionWithName:@"matmul_f32_f16_gguf_weight_simd_kernel"];
    if (matmul_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul kernel function was not found"
        );
    }

    NSError* pipeline_error = nil;
    matmul_f16_simd_pipeline =
        [device newComputePipelineStateWithFunction:matmul_function error:&pipeline_error];
    if (matmul_f16_simd_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul pipeline creation failed: " + ns_error_message(pipeline_error, "unknown error")
        );
    }
    if (matmul_f16_simd_pipeline.threadExecutionWidth != 32
        || matmul_f16_simd_pipeline.maxTotalThreadsPerThreadgroup < 64U) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal F16 SIMD matmul requires two 32-lane SIMD groups"
        );
    }

    id<MTLFunction> embedding_function =
        [library newFunctionWithName:@"embed_f16_gguf_weight_token_ids_kernel"];
    if (embedding_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal F16 GGUF token embedding kernel function was not found"
        );
    }

    NSError* embedding_pipeline_error = nil;
    embedding_pipeline = [device newComputePipelineStateWithFunction:embedding_function
                                                               error:&embedding_pipeline_error];
    if (embedding_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal F16 GGUF token embedding pipeline creation failed: "
                + ns_error_message(embedding_pipeline_error, "unknown error")
        );
    }

    auto make_pipeline = [&](NSString* function_name, const char* description) {
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + description + " kernel function was not found"
            );
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + description + " pipeline creation failed: "
                    + ns_error_message(error, "unknown error")
            );
        }
        return pipeline;
    };

    auto make_specialized_pipeline = [library, this](
        NSString* function_name,
        NSUInteger constant_index,
        std::uint16_t constant_value,
        const char* description
    ) {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        [constants setConstantValue:&constant_value type:MTLDataTypeUShort atIndex:constant_index];
        NSError* function_error = nil;
        id<MTLFunction> function = [library newFunctionWithName:function_name
                                                 constantValues:constants
                                                          error:&function_error];
        if (function == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + description + " specialized function creation failed: "
                    + ns_error_message(function_error, "unknown error")
            );
        }
        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + description + " pipeline creation failed: "
                    + ns_error_message(pipeline_error, "unknown error")
            );
        }
        return pipeline;
    };

    auto make_tiled_pipeline = [library, this](
        std::uint16_t dtype,
        bool exact_16,
        const char* description
    ) {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        [constants setConstantValue:&dtype type:MTLDataTypeUShort atIndex:2];
        [constants setConstantValue:&exact_16 type:MTLDataTypeBool atIndex:3];
        NSError* function_error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"matmul_f32_kquant_tiled_kernel"
                                                 constantValues:constants
                                                          error:&function_error];
        if (function == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + description + " specialized function creation failed: "
                    + ns_error_message(function_error, "unknown error")
            );
        }
        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal ") + description + " pipeline creation failed: "
                    + ns_error_message(pipeline_error, "unknown error")
            );
        }
        return pipeline;
    };

    matmul_q4_k_tiled_pipeline = make_tiled_pipeline(0, false, "Q4_K tiled matmul");
    matmul_q4_k_tiled_exact_16_pipeline =
        make_tiled_pipeline(0, true, "Q4_K exact 16-row tiled matmul");
    matmul_q4_k_small_batch_pipeline = make_specialized_pipeline(
        @"matmul_f32_kquant_small_batch_kernel", 2, 0, "Q4_K packed small-batch matmul"
    );

    matvec_q4_k_packed_pipeline =
        make_specialized_pipeline(@"matvec_f32_q4_k_packed_kernel", 0, 0, "Q4_K packed decode matvec");
    matvec_q4_k_packed_14_pipeline = make_specialized_pipeline(
        @"matvec_f32_q4_k_packed_kernel", 0, 14, "Q4_K 14-block packed decode matvec"
    );
    matvec_q4_k_packed_16_pipeline = make_specialized_pipeline(
        @"matvec_f32_q4_k_packed_kernel", 0, 16, "Q4_K 16-block packed decode matvec"
    );

    matmul_q6_k_tiled_pipeline = make_tiled_pipeline(1, false, "Q6_K tiled matmul");
    matmul_q6_k_tiled_exact_16_pipeline =
        make_tiled_pipeline(1, true, "Q6_K exact 16-row tiled matmul");
    matmul_q6_k_small_batch_pipeline = make_specialized_pipeline(
        @"matmul_f32_kquant_small_batch_kernel", 2, 1, "Q6_K packed small-batch matmul"
    );

    matvec_q6_k_packed_pipeline =
        make_specialized_pipeline(@"matvec_f32_q6_k_packed_kernel", 1, 0, "Q6_K packed decode matvec");
    matvec_q6_k_packed_14_pipeline = make_specialized_pipeline(
        @"matvec_f32_q6_k_packed_kernel", 1, 14, "Q6_K 14-block packed decode matvec"
    );
    matvec_q6_k_packed_56_pipeline = make_specialized_pipeline(
        @"matvec_f32_q6_k_packed_kernel", 1, 56, "Q6_K 56-block packed decode matvec"
    );

    id<MTLFunction> matmul_add_function =
        [library newFunctionWithName:@"matmul_f32_f16_gguf_weight_add_kernel"];
    if (matmul_add_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul-add kernel function was not found"
        );
    }

    NSError* matmul_add_pipeline_error = nil;
    matmul_add_pipeline = [device newComputePipelineStateWithFunction:matmul_add_function
                                                                error:&matmul_add_pipeline_error];
    if (matmul_add_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul-add pipeline creation failed: "
                + ns_error_message(matmul_add_pipeline_error, "unknown error")
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

    id<MTLFunction> gelu_tanh_multiply_function =
        [library newFunctionWithName:@"gelu_tanh_multiply_f32_kernel"];
    if (gelu_tanh_multiply_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal GELU-tanh multiply kernel function was not found"
        );
    }

    NSError* gelu_tanh_multiply_pipeline_error = nil;
    gelu_tanh_multiply_pipeline =
        [device newComputePipelineStateWithFunction:gelu_tanh_multiply_function
                                              error:&gelu_tanh_multiply_pipeline_error];
    if (gelu_tanh_multiply_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal GELU-tanh multiply pipeline creation failed: "
                + ns_error_message(gelu_tanh_multiply_pipeline_error, "unknown error")
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

    id<MTLFunction> softcap_function = [library newFunctionWithName:@"softcap_f32_kernel"];
    if (softcap_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softcap kernel function was not found"
        );
    }

    NSError* softcap_pipeline_error = nil;
    softcap_pipeline = [device newComputePipelineStateWithFunction:softcap_function
                                                            error:&softcap_pipeline_error];
    if (softcap_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softcap pipeline creation failed: "
                + ns_error_message(softcap_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> noop_function = [library newFunctionWithName:@"noop_kernel"];
    if (noop_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal no-op kernel function was not found"
        );
    }

    NSError* noop_pipeline_error = nil;
    noop_pipeline = [device newComputePipelineStateWithFunction:noop_function error:&noop_pipeline_error];
    if (noop_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal no-op pipeline creation failed: " + ns_error_message(noop_pipeline_error, "unknown error")
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

    id<MTLFunction> rms_norm_add_function = [library newFunctionWithName:@"rms_norm_add_f32_kernel"];
    if (rms_norm_add_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal fused RMSNorm-add kernel function was not found"
        );
    }
    NSError* rms_norm_add_pipeline_error = nil;
    rms_norm_add_pipeline = [device newComputePipelineStateWithFunction:rms_norm_add_function
                                                                  error:&rms_norm_add_pipeline_error];
    if (rms_norm_add_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal fused RMSNorm-add pipeline creation failed: "
                + ns_error_message(rms_norm_add_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> rope_by_heads_function = [library newFunctionWithName:@"rope_f32_by_heads_kernel"];
    if (rope_by_heads_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RoPE-by-heads kernel function was not found"
        );
    }

    NSError* rope_by_heads_pipeline_error = nil;
    rope_by_heads_pipeline = [device newComputePipelineStateWithFunction:rope_by_heads_function
                                                                   error:&rope_by_heads_pipeline_error];
    if (rope_by_heads_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RoPE-by-heads pipeline creation failed: "
                + ns_error_message(rope_by_heads_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> rope_neox_by_heads_function =
        [library newFunctionWithName:@"rope_neox_f32_by_heads_kernel"];
    if (rope_neox_by_heads_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal NeoX RoPE-by-heads kernel function was not found"
        );
    }
    NSError* rope_neox_by_heads_pipeline_error = nil;
    rope_neox_by_heads_pipeline =
        [device newComputePipelineStateWithFunction:rope_neox_by_heads_function
                                              error:&rope_neox_by_heads_pipeline_error];
    if (rope_neox_by_heads_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal NeoX RoPE-by-heads pipeline creation failed: "
                + ns_error_message(rope_neox_by_heads_pipeline_error, "unknown error")
        );
    }

    gemma2_kv_cache_store_pipeline = make_pipeline(
        @"gemma2_kv_cache_store_f16_kernel",
        "Gemma 2 F16 KV-cache store"
    );

    id<MTLFunction> causal_grouped_attention_scores_function =
        [library newFunctionWithName:@"causal_grouped_attention_scores_f32_kernel"];
    if (causal_grouped_attention_scores_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention scores kernel function was not found"
        );
    }

    NSError* causal_grouped_attention_scores_pipeline_error = nil;
    causal_grouped_attention_scores_pipeline =
        [device newComputePipelineStateWithFunction:causal_grouped_attention_scores_function
                                              error:&causal_grouped_attention_scores_pipeline_error];
    if (causal_grouped_attention_scores_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention scores pipeline creation failed: "
                + ns_error_message(causal_grouped_attention_scores_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> causal_grouped_attention_values_function =
        [library newFunctionWithName:@"causal_grouped_attention_values_f32_kernel"];
    if (causal_grouped_attention_values_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention values kernel function was not found"
        );
    }

    NSError* causal_grouped_attention_values_pipeline_error = nil;
    causal_grouped_attention_values_pipeline =
        [device newComputePipelineStateWithFunction:causal_grouped_attention_values_function
                                              error:&causal_grouped_attention_values_pipeline_error];
    if (causal_grouped_attention_values_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention values pipeline creation failed: "
                + ns_error_message(causal_grouped_attention_values_pipeline_error, "unknown error")
        );
    }

    gemma2_prefill_attention_scores_pipeline = make_pipeline(
        @"causal_grouped_attention_scores_f16_cache_kernel",
        "Gemma 2 F16-cache prefill attention scores"
    );
    gemma2_prefill_attention_values_pipeline = make_pipeline(
        @"causal_grouped_attention_values_f16_cache_kernel",
        "Gemma 2 F16-cache prefill attention values"
    );

    id<MTLFunction> gemma2_decode_attention_function =
        [library newFunctionWithName:@"gemma2_decode_attention_f16_cache_kernel"];
    if (gemma2_decode_attention_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Gemma 2 decode attention kernel function was not found"
        );
    }
    NSError* gemma2_decode_attention_pipeline_error = nil;
    gemma2_decode_attention_pipeline =
        [device newComputePipelineStateWithFunction:gemma2_decode_attention_function
                                              error:&gemma2_decode_attention_pipeline_error];
    if (gemma2_decode_attention_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Gemma 2 decode attention pipeline creation failed: "
                + ns_error_message(gemma2_decode_attention_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> gemma2_decode_attention_reduce_function =
        [library newFunctionWithName:@"gemma2_decode_attention_reduce_f32_kernel"];
    if (gemma2_decode_attention_reduce_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Gemma 2 decode attention reduction kernel function was not found"
        );
    }
    NSError* gemma2_decode_attention_reduce_pipeline_error = nil;
    gemma2_decode_attention_reduce_pipeline =
        [device newComputePipelineStateWithFunction:gemma2_decode_attention_reduce_function
                                              error:&gemma2_decode_attention_reduce_pipeline_error];
    if (gemma2_decode_attention_reduce_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Gemma 2 decode attention reduction pipeline creation failed: "
                + ns_error_message(gemma2_decode_attention_reduce_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> softmax_function = [library newFunctionWithName:@"softmax_f32_kernel"];
    if (softmax_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softmax kernel function was not found"
        );
    }

    NSError* softmax_pipeline_error = nil;
    softmax_pipeline = [device newComputePipelineStateWithFunction:softmax_function
                                                             error:&softmax_pipeline_error];
    if (softmax_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softmax pipeline creation failed: "
                + ns_error_message(softmax_pipeline_error, "unknown error")
        );
    }

    const auto make_candidate_pipeline = [&](NSString* function_name, const char* detail) {
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            throw metal_error(runtime::BackendErrorCode::metal_execution_failed, detail);
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string(detail) + ": " + ns_error_message(error, "unknown error")
            );
        }
        return pipeline;
    };
    token_candidate_tile_pipeline = make_candidate_pipeline(
        @"token_candidate_tile_f32_kernel", "Metal token-candidate tile pipeline creation failed"
    );
    token_candidate_merge_pipeline = make_candidate_pipeline(
        @"token_candidate_merge_f32_kernel", "Metal token-candidate merge pipeline creation failed"
    );
    token_candidate_finalize_pipeline = make_candidate_pipeline(
        @"token_candidate_finalize_f32_kernel", "Metal token-candidate finalization pipeline creation failed"
    );
    token_candidate_top1_pipeline = make_candidate_pipeline(
        @"token_candidate_top1_f32_kernel", "Metal token-candidate top-1 pipeline creation failed"
    );
    if (token_candidate_tile_pipeline.maxTotalThreadsPerThreadgroup < 512U
        || token_candidate_top1_pipeline.maxTotalThreadsPerThreadgroup < 512U) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal token-candidate selection requires 512 threads per threadgroup"
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

MatmulPipelineState& shared_matmul_pipeline_state() {
    static MatmulPipelineState state;
    return state;
}

F16MatmulDispatch select_f16_matmul_dispatch(
    MatmulPipelineState& state,
    std::uint64_t rows,
    std::uint64_t columns
) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Metal F16 matmul dimensions are unsupported");
    }
    return F16MatmulDispatch {
        .pipeline = state.matmul_f16_simd_pipeline,
        .threadgroups = MTLSizeMake(
            static_cast<NSUInteger>((columns + 3U) / 4U),
            static_cast<NSUInteger>(rows),
            1
        ),
        .threads_per_threadgroup = MTLSizeMake(64, 1, 1),
    };
}

QuantizedMatmulDispatch select_quantized_matmul_dispatch(
    MatmulPipelineState& state,
    DType dtype,
    std::uint64_t rows,
    std::uint64_t shared,
    std::uint64_t columns
) {
    if (dtype != DType::q4_k && dtype != DType::q6_k) {
        throw std::invalid_argument("Metal K-quant matmul weight dtype is unsupported");
    }
    if (rows == 0 || shared == 0 || columns == 0 || shared % 256 != 0) {
        throw std::invalid_argument("Metal K-quant matmul dimensions are unsupported");
    }

    id<MTLComputePipelineState> pipeline = nil;
    std::uint64_t row_tile = 1;
    std::uint64_t threads = 64;
    if (rows == 1) {
        const auto blocks = shared / 256;
        if (dtype == DType::q4_k) {
            pipeline = blocks == 14
                ? state.matvec_q4_k_packed_14_pipeline
                : blocks == 16
                    ? state.matvec_q4_k_packed_16_pipeline
                    : state.matvec_q4_k_packed_pipeline;
        } else {
            pipeline = blocks == 14
                ? state.matvec_q6_k_packed_14_pipeline
                : blocks == 56
                    ? state.matvec_q6_k_packed_56_pipeline
                    : state.matvec_q6_k_packed_pipeline;
        }
    } else if (rows <= 8) {
        row_tile = 4;
        threads = 64;
        pipeline = dtype == DType::q4_k
            ? state.matmul_q4_k_small_batch_pipeline
            : state.matmul_q6_k_small_batch_pipeline;
    } else {
        const bool exact_16 = rows == 16 && columns % 64 == 0;
        row_tile = exact_16 ? 16 : 32;
        threads = exact_16 ? 64 : 128;
        pipeline = dtype == DType::q4_k
            ? (exact_16 ? state.matmul_q4_k_tiled_exact_16_pipeline : state.matmul_q4_k_tiled_pipeline)
            : (exact_16 ? state.matmul_q6_k_tiled_exact_16_pipeline : state.matmul_q6_k_tiled_pipeline);
    }

    if (pipeline == nil || pipeline.threadExecutionWidth != 32
        || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal K-quant matmul requires 32-lane SIMD groups and its selected threadgroup size"
        );
    }

    return QuantizedMatmulDispatch {
        .pipeline = pipeline,
        .threadgroups = MTLSizeMake(
            static_cast<NSUInteger>(
                rows == 1 ? (columns + 3) / 4
                : rows <= 8 ? (columns + 1) / 2
                : (columns + 63) / 64
            ),
            static_cast<NSUInteger>((rows + row_tile - 1) / row_tile),
            1
        ),
        .threads_per_threadgroup = MTLSizeMake(static_cast<NSUInteger>(threads), 1, 1),
    };
}

}  // namespace local_gguf_engine::tensor::backends::metal
