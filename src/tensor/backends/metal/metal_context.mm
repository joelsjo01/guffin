#include "metal_context.hpp"

#if defined(LOCAL_GGUF_ENGINE_TESTING)
#include "metal_test_hooks.hpp"

#include <atomic>
#endif
#include <stdexcept>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

#if defined(LOCAL_GGUF_ENGINE_TESTING)
std::atomic_bool metal_device_unavailable_for_testing {false};
#endif

}  // namespace

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
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    if (metal_device_unavailable_for_testing.load(std::memory_order_acquire)) {
        return nil;
    }
#endif
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

#if defined(LOCAL_GGUF_ENGINE_TESTING)
void set_metal_device_unavailable_for_testing(bool unavailable) noexcept {
    metal_device_unavailable_for_testing.store(unavailable, std::memory_order_release);
}
#endif

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

MatmulPipelineState::MatmulPipelineState(bool precise_math) {
    device = metal_device_or_throw();

    NSError* library_error = nil;
    NSString* source = [NSString stringWithUTF8String:metal_kernel_source()];
    // Gemma 3 has one explicit numerical policy for ordinary and batched
    // execution. Preserve the established Gemma 2 compiler defaults.
    MTLCompileOptions* options = nil;
    if (precise_math) {
        options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    }
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&library_error];
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
        const char* description,
        std::uint16_t parallel_rows = 1U,
        bool columns8 = false
    ) {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        [constants setConstantValue:&constant_value type:MTLDataTypeUShort atIndex:constant_index];
        if ([function_name isEqualToString:@"matmul_f32_q4_k_packed_row_batch_kernel"]
            || [function_name isEqualToString:@"matmul_f32_q6_k_packed_row_batch_kernel"]) {
            [constants setConstantValue:&parallel_rows type:MTLDataTypeUShort atIndex:5];
        }
        if ([function_name isEqualToString:@"gemma3_matmul_f32_kquant_shared_kernel"]
            || [function_name isEqualToString:@"gemma3_matmul_f32_columns8_kernel"]
            || [function_name isEqualToString:@"gemma3_matmul_f32_columns8_prepared_kernel"]) {
            [constants setConstantValue:&parallel_rows type:MTLDataTypeUShort atIndex:6];
        }
        if ([function_name hasPrefix:@"gemma3_matmul_f32_kquant_matrix"]) {
            [constants setConstantValue:&columns8 type:MTLDataTypeBool atIndex:7];
        }
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
        const char* description,
        bool columns8 = false
    ) {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        [constants setConstantValue:&dtype type:MTLDataTypeUShort atIndex:2];
        [constants setConstantValue:&exact_16 type:MTLDataTypeBool atIndex:3];
        [constants setConstantValue:&columns8 type:MTLDataTypeBool atIndex:7];
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

    {
        if (precise_math) {
            verifier_kv_store_f16_checked_pipeline = make_pipeline(
                @"verifier_kv_store_f16_checked_kernel", "F16 K/V store with numerical status");
        }
        // The packed Gemma 2 body keeps the established compiler policy. Its
        // R1/R2 consumers and fused producers come from that same library.
        if (!precise_math) {
            columns8_gemma2_prompt[0] = make_tiled_pipeline(0, false,
                "Gemma2 packed half-operand prompt", true);
            columns8_gemma2_prompt[1] = make_tiled_pipeline(0, true,
                "Gemma2 packed exact16 half-operand prompt", true);
        }
        const std::array<NSString*, 2> prompt_matrix_names = {
            @"gemma3_matmul_f32_kquant_matrix_kernel",
            @"gemma3_matmul_f32_kquant_matrix_m32_kernel"};
        for (std::uint16_t packing = 0; precise_math && packing < 2; ++packing) {
            for (std::uint16_t dtype = 0; dtype < 2; ++dtype) {
                for (std::uint16_t tile = 0; tile < prompt_matrix_names.size(); ++tile) {
                    auto& pipeline = gemma3_prompt_matrix[packing][dtype][tile];
                    pipeline = make_specialized_pipeline(prompt_matrix_names[tile], 2, dtype,
                        "Gemma3 F32 prompt matrix", 1, packing == 1);
                    if (pipeline.threadExecutionWidth != 32U
                        || pipeline.maxTotalThreadsPerThreadgroup < 128U)
                        throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                            "Metal prompt matrices require four 32-lane SIMD groups");
                }
            }
        }
        if (precise_math) {
            gemma3_q4_shared_two_pipeline = make_specialized_pipeline(
                @"gemma3_matmul_f32_kquant_shared_kernel", 2, 0, "Gemma3 Q4 two shared rows", 2
            );
            gemma3_q6_shared_one_pipeline = make_specialized_pipeline(
                @"gemma3_matmul_f32_kquant_shared_kernel", 2, 1, "Gemma3 Q6 one shared row", 1
            );
            gemma3_q6_shared_two_pipeline = make_specialized_pipeline(
                @"gemma3_matmul_f32_kquant_shared_kernel", 2, 1, "Gemma3 Q6 two shared rows", 2
            );
        }
        for (std::uint16_t rows = 1; rows <= 2; ++rows) {
            columns8_q4_rows[rows - 1] = make_specialized_pipeline(
                @"gemma3_matmul_f32_columns8_kernel", 2, 0, "Columns8 Q4 rows", rows);
            columns8_q4_prepared_rows[rows - 1] = make_specialized_pipeline(
                @"gemma3_matmul_f32_columns8_prepared_kernel", 2, 0,
                "Columns8 Q4 rows with producer sums", rows);
            if (precise_math) columns8_q6_rows[rows - 1] = make_specialized_pipeline(
                @"gemma3_matmul_f32_columns8_kernel", 2, 1, "Columns8 Q6 rows", rows);
        }
        rms_norm_q4_sums_pipeline = make_pipeline(
            @"rms_norm_q4_sums_f32_kernel", "RMSNorm with Q4 input sums");
        gelu_tanh_multiply_q4_sums_pipeline = make_pipeline(
            @"gelu_tanh_multiply_q4_sums_f32_kernel", "GELU with Q4 input sums");
        verifier_attention_q4_sums_pipeline = make_pipeline(
            @"verifier_attention_q4_sums_f32_cache_kernel", "Attention with Q4 input sums");
        verifier_attention_q4_sums_f16_pipeline = make_pipeline(
            @"verifier_attention_q4_sums_f16_cache_kernel", "F16 cache attention with Q4 input sums");
        verifier_attention_reduce_q4_sums_pipeline = make_pipeline(
            @"verifier_attention_reduce_q4_sums_f32_kernel", "Attention reduction with Q4 input sums");
        for (const auto& pipeline : {rms_norm_q4_sums_pipeline,
                 gelu_tanh_multiply_q4_sums_pipeline}) {
            if (pipeline.threadExecutionWidth != 32U || pipeline.maxTotalThreadsPerThreadgroup < 256U)
                throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                    "Metal Q4 sum producers require eight 32-lane SIMD groups");
        }
        for (const auto& pipeline : {verifier_attention_q4_sums_pipeline,
                 verifier_attention_q4_sums_f16_pipeline}) {
            if (pipeline.threadExecutionWidth != 32U || pipeline.maxTotalThreadsPerThreadgroup < 128U)
                throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                    "Metal Q4 attention sum producers require four 32-lane SIMD groups");
        }
        if (verifier_attention_reduce_q4_sums_pipeline.threadExecutionWidth != 32U
            || verifier_attention_reduce_q4_sums_pipeline.maxTotalThreadsPerThreadgroup < 32U)
            throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                "Metal Q4 attention sum producers require 32-lane SIMD groups");
        for (const auto& pipeline : {
                 columns8_q4_rows[0], columns8_q4_rows[1],
                 columns8_q4_prepared_rows[0], columns8_q4_prepared_rows[1],
                 columns8_q6_rows[0], columns8_q6_rows[1],
             }) {
            if (pipeline == nil && !precise_math) continue;
            if (pipeline == nil || pipeline.threadExecutionWidth != 32U
                || pipeline.maxTotalThreadsPerThreadgroup < 128U) {
                throw metal_error(
                    runtime::BackendErrorCode::metal_execution_failed,
                    "Metal columns8 projections require four 32-lane SIMD groups"
                );
            }
        }
    }

    matvec_q8_0_packed_pipeline = make_specialized_pipeline(
        @"matvec_f32_q8_0_packed_kernel", 4, 0, "Q8_0 packed decode matvec"
    );
    matvec_q8_0_packed_112_pipeline = make_specialized_pipeline(
        @"matvec_f32_q8_0_packed_kernel", 4, 112, "Q8_0 112-block packed decode matvec"
    );
    matmul_q8_0_width_two_pipeline = make_specialized_pipeline(
        @"matmul_f32_q8_0_width_two_kernel", 4, 0, "Q8_0 packed width-two matmul"
    );
    matmul_q8_0_width_two_112_pipeline = make_specialized_pipeline(
        @"matmul_f32_q8_0_width_two_kernel", 4, 112, "Q8_0 112-block width-two matmul"
    );
    target_head_q8_row_parallel_pipeline = make_specialized_pipeline(
        @"matmul_f32_q8_0_target_rows_kernel", 4, 0,
        "Q8_0 target-head row-parallel matmul"
    );
    target_head_q8_row_parallel_112_pipeline = make_specialized_pipeline(
        @"matmul_f32_q8_0_target_rows_kernel", 4, 112,
        "Q8_0 112-block target-head row-parallel matmul"
    );
    feature_tap_copy_pipeline = make_pipeline(
        @"feature_tap_copy_f32_kernel", "feature tap copy"
    );
    feature_tap_store_f16_pipeline = make_pipeline(
        @"feature_tap_store_f16_kernel", "F16 trace feature tap store"
    );
    feature_scatter_f32_pipeline = make_pipeline(
        @"feature_scatter_f32_kernel", "F32 feature scatter"
    );
    feature_scatter_f16_pipeline = make_pipeline(
        @"feature_scatter_f16_kernel", "F16 feature scatter"
    );
    f16_row_scatter_pipeline = make_pipeline(
        @"f16_row_scatter_kernel", "F16 row scatter"
    );

    matmul_q4_k_tiled_pipeline = make_tiled_pipeline(0, false, "Q4_K tiled matmul");
    matmul_q4_k_tiled_exact_16_pipeline =
        make_tiled_pipeline(0, true, "Q4_K exact 16-row tiled matmul");
    matmul_q4_k_width_two_pipeline = make_specialized_pipeline(
        @"matmul_f32_q4_k_packed_width_two_kernel", 0, 0, "Q4_K packed width-two matmul"
    );
    matmul_q4_k_width_two_14_pipeline = make_specialized_pipeline(
        @"matmul_f32_q4_k_packed_width_two_kernel", 0, 14, "Q4_K 14-block width-two matmul"
    );
    matmul_q4_k_width_two_16_pipeline = make_specialized_pipeline(
        @"matmul_f32_q4_k_packed_width_two_kernel", 0, 16, "Q4_K 16-block width-two matmul"
    );

    matmul_q4_k_packed_row_batch_pipeline = make_specialized_pipeline(
        @"matmul_f32_q4_k_packed_row_batch_kernel", 0, 0, "Q4_K packed row-batch matmul"
    );
    matmul_q4_k_packed_row_batch_14_pipeline = make_specialized_pipeline(
        @"matmul_f32_q4_k_packed_row_batch_kernel", 0, 14, "Q4_K 14-block packed row-batch matmul"
    );
    matmul_q4_k_packed_row_batch_16_pipeline = make_specialized_pipeline(
        @"matmul_f32_q4_k_packed_row_batch_kernel", 0, 16, "Q4_K 16-block packed row-batch matmul"
    );

    matmul_q6_k_tiled_pipeline = make_tiled_pipeline(1, false, "Q6_K tiled matmul");
    matmul_q6_k_tiled_exact_16_pipeline =
        make_tiled_pipeline(1, true, "Q6_K exact 16-row tiled matmul");
    matmul_q6_k_width_two_pipeline = make_specialized_pipeline(
        @"matmul_f32_q6_k_packed_width_two_kernel", 1, 0, "Q6_K packed width-two matmul"
    );
    matmul_q6_k_width_two_14_pipeline = make_specialized_pipeline(
        @"matmul_f32_q6_k_packed_width_two_kernel", 1, 14, "Q6_K 14-block width-two matmul"
    );
    matmul_q6_k_width_two_56_pipeline = make_specialized_pipeline(
        @"matmul_f32_q6_k_packed_width_two_kernel", 1, 56, "Q6_K 56-block width-two matmul"
    );

    matmul_q6_k_packed_row_batch_pipeline = make_specialized_pipeline(
        @"matmul_f32_q6_k_packed_row_batch_kernel", 1, 0, "Q6_K packed row-batch matmul"
    );
    matmul_q6_k_packed_row_batch_14_pipeline = make_specialized_pipeline(
        @"matmul_f32_q6_k_packed_row_batch_kernel", 1, 14, "Q6_K 14-block packed row-batch matmul"
    );
    matmul_q6_k_packed_row_batch_56_pipeline = make_specialized_pipeline(
        @"matmul_f32_q6_k_packed_row_batch_kernel", 1, 56, "Q6_K 56-block packed row-batch matmul"
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
    gemma2_kv_cache_copy_pipeline = make_pipeline(
        @"gemma2_kv_cache_copy_f16_kernel",
        "Gemma 2 F16 KV-cache copy"
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
    gemma2_wrapped_prefill_attention_scores_pipeline = make_pipeline(
        @"causal_grouped_attention_scores_f16_hybrid_kernel",
        "Gemma 2 F16-cache wrapped prefill attention scores"
    );
    gemma2_wrapped_prefill_attention_values_pipeline = make_pipeline(
        @"causal_grouped_attention_values_f16_hybrid_kernel",
        "Gemma 2 F16-cache wrapped prefill attention values"
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

    embedding_q6_pipeline = make_pipeline(
        @"embed_q6_k_gguf_weight_token_ids_kernel", "native Q6_K token embedding"
    );
    head_rms_norm_pipeline = make_pipeline(
        @"head_rms_norm_f32_kernel", "per-head RMS normalization"
    );
    kv_cache_copy_f32_pipeline = make_pipeline(
        @"kv_cache_copy_f32_kernel", "F32 K/V cache copy"
    );
    verifier_kv_store_f32_pipeline = make_pipeline(
        @"verifier_kv_store_f32_kernel", "F32 private K/V store"
    );
    verifier_kv_scatter_f32_pipeline = make_pipeline(
        @"verifier_kv_scatter_f32_kernel", "F32 accepted-path K/V scatter"
    );
    verifier_attention_f32_pipeline = make_pipeline(
        @"verifier_attention_f32_cache_kernel", "F32 prefix and private attention"
    );
    verifier_rope_pipeline = make_pipeline(
        @"verifier_rope_f32_by_positions_kernel", "target verifier query RoPE"
    );
    verifier_kv_store_pipeline = make_pipeline(
        @"verifier_kv_store_f16_kernel", "target verifier private K/V store"
    );
    verifier_attention_pipeline = make_pipeline(
        @"verifier_attention_f16_cache_kernel", "target verifier tree attention"
    );
    verifier_attention_reduce_pipeline = make_pipeline(
        @"verifier_attention_reduce_f32_kernel", "target verifier tree attention reduction"
    );
    verifier_kv_scatter_pipeline = make_pipeline(
        @"verifier_kv_scatter_f16_kernel", "target verifier accepted-path K/V scatter"
    );
    verifier_input_validate_pipeline = make_pipeline(
        @"verifier_input_validate_u32_kernel", "target verifier device input validation"
    );

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
    token_candidate_batch_tile_pipeline = make_candidate_pipeline(
        @"token_candidate_batch_tile_f32_kernel", "Metal batched token-candidate tile pipeline creation failed"
    );
    token_candidate_batch_merge_pipeline = make_candidate_pipeline(
        @"token_candidate_batch_merge_f32_kernel", "Metal batched token-candidate merge pipeline creation failed"
    );
    token_candidate_batch_finalize_pipeline = make_candidate_pipeline(
        @"token_candidate_batch_finalize_f32_kernel", "Metal batched token-candidate finalization pipeline creation failed"
    );
    token_candidate_batch_top1_pipeline = make_candidate_pipeline(
        @"token_candidate_batch_top1_f32_kernel", "Metal batched token-candidate top-1 pipeline creation failed"
    );
    verifier_request_resolve_pipeline = make_candidate_pipeline(
        @"verifier_selection_request_resolve_kernel",
        "Metal verifier selection-request resolver pipeline creation failed"
    );
    target_candidate_batch_tile_pipeline = make_candidate_pipeline(
        @"target_candidate_batch_tile_f32_kernel",
        "Metal target-candidate batch tile pipeline creation failed"
    );
    target_candidate_batch_merge_pipeline = make_candidate_pipeline(
        @"target_candidate_batch_merge_f32_kernel",
        "Metal target-candidate batch merge pipeline creation failed"
    );
    target_candidate_batch_finalize_pipeline = make_candidate_pipeline(
        @"target_candidate_batch_finalize_f32_kernel",
        "Metal target-candidate batch finalization pipeline creation failed"
    );
    target_candidate_batch_top1_pipeline = make_candidate_pipeline(
        @"target_candidate_batch_top1_f32_kernel",
        "Metal target-candidate batch top-1 pipeline creation failed"
    );
    draft_topk_merge_pipeline = make_candidate_pipeline(
        @"draft_topk_merge_f32_kernel", "Metal draft top-k merge pipeline creation failed"
    );
    if (token_candidate_tile_pipeline.maxTotalThreadsPerThreadgroup < 512U
        || token_candidate_top1_pipeline.maxTotalThreadsPerThreadgroup < 512U
        || token_candidate_batch_tile_pipeline.maxTotalThreadsPerThreadgroup < 512U
        || token_candidate_batch_top1_pipeline.maxTotalThreadsPerThreadgroup < 512U
        || target_candidate_batch_tile_pipeline.maxTotalThreadsPerThreadgroup < 512U
        || target_candidate_batch_top1_pipeline.maxTotalThreadsPerThreadgroup < 512U) {
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
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    // A previously initialized process-lifetime state must not bypass the
    // injected unavailable-device boundary.
    if (metal_device_unavailable_for_testing.load(std::memory_order_acquire)) {
        static_cast<void>(metal_device_or_throw());
    }
#endif
    static MatmulPipelineState state;
    return state;
}

MatmulPipelineState& shared_precise_matmul_pipeline_state() {
    // Check availability even after process-lifetime pipeline initialization.
    static_cast<void>(metal_device_or_throw());
    static MatmulPipelineState state(true);
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
    if (dtype != DType::q8_0 && dtype != DType::q4_k && dtype != DType::q6_k) {
        throw std::invalid_argument("Metal K-quant matmul weight dtype is unsupported");
    }
    if (rows == 0 || shared == 0 || columns == 0) {
        throw std::invalid_argument("Metal K-quant matmul dimensions are unsupported");
    }

    if (dtype == DType::q8_0) {
        if ((rows != 1U && rows != 2U) || shared % 32U != 0U) {
            throw std::invalid_argument("Metal Q8_0 output projection requires one or two rows and 32-aligned shared width");
        }
        auto pipeline = rows == 2U
            ? (shared == 3584U
                ? state.matmul_q8_0_width_two_112_pipeline
                : state.matmul_q8_0_width_two_pipeline)
            : (shared == 3584U
                ? state.matvec_q8_0_packed_112_pipeline
                : state.matvec_q8_0_packed_pipeline);
        if (pipeline == nil || pipeline.threadExecutionWidth != 32
            || pipeline.maxTotalThreadsPerThreadgroup < 64U) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal Q8_0 matvec requires two 32-lane SIMD groups"
            );
        }
        return QuantizedMatmulDispatch {
            .pipeline = pipeline,
            .threadgroups = MTLSizeMake(static_cast<NSUInteger>((columns + 3U) / 4U), 1U, 1U),
            .threads_per_threadgroup = MTLSizeMake(64U, 1U, 1U),
        };
    }
    if (shared % 256U != 0U) {
        throw std::invalid_argument("Metal K-quant matmul shared width must be 256-aligned");
    }

    id<MTLComputePipelineState> pipeline = nil;
    std::uint64_t row_tile = 1;
    std::uint64_t threads = 64;
    const bool packed_row_batch = rows == 1U || (rows >= 3U && rows <= 8U);
    if (packed_row_batch) {
        // One threadgroup-grid row owns one activation row. This is the same
        // packed kernel for scalar decode and 3-8 row tails, so worker mapping
        // and floating-point accumulation order cannot drift by batch width.
        row_tile = 1;
        const auto blocks = shared / 256;
        if (dtype == DType::q4_k) {
            pipeline = blocks == 14
                ? state.matmul_q4_k_packed_row_batch_14_pipeline
                : blocks == 16
                    ? state.matmul_q4_k_packed_row_batch_16_pipeline
                    : state.matmul_q4_k_packed_row_batch_pipeline;
        } else {
            pipeline = blocks == 14
                ? state.matmul_q6_k_packed_row_batch_14_pipeline
                : blocks == 56
                    ? state.matmul_q6_k_packed_row_batch_56_pipeline
                    : state.matmul_q6_k_packed_row_batch_pipeline;
        }
    } else if (rows == 2) {
        row_tile = 2;
        const auto blocks = shared / 256;
        if (dtype == DType::q4_k) {
            pipeline = blocks == 14 ? state.matmul_q4_k_width_two_14_pipeline
                : blocks == 16 ? state.matmul_q4_k_width_two_16_pipeline
                : state.matmul_q4_k_width_two_pipeline;
        } else {
            pipeline = blocks == 14 ? state.matmul_q6_k_width_two_14_pipeline
                : blocks == 56 ? state.matmul_q6_k_width_two_56_pipeline
                : state.matmul_q6_k_width_two_pipeline;
        }
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
                rows <= 8 ? (columns + 3) / 4
                : (columns + 63) / 64
            ),
            static_cast<NSUInteger>((rows + row_tile - 1) / row_tile),
            1
        ),
        .threads_per_threadgroup = MTLSizeMake(static_cast<NSUInteger>(threads), 1, 1),
    };
}

}  // namespace local_gguf_engine::tensor::backends::metal
