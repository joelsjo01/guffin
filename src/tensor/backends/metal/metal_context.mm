#include "metal_context.hpp"

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

id<MTLDevice> metal_device_or_throw() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
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

    id<MTLFunction> matmul_q4_k_function =
        [library newFunctionWithName:@"matmul_f32_q4_k_gguf_weight_kernel"];
    if (matmul_q4_k_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Q4_K matmul kernel function was not found"
        );
    }

    NSError* matmul_q4_k_pipeline_error = nil;
    matmul_q4_k_pipeline = [device newComputePipelineStateWithFunction:matmul_q4_k_function
                                                                  error:&matmul_q4_k_pipeline_error];
    if (matmul_q4_k_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Q4_K matmul pipeline creation failed: "
                + ns_error_message(matmul_q4_k_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> matmul_q6_k_function =
        [library newFunctionWithName:@"matmul_f32_q6_k_gguf_weight_kernel"];
    if (matmul_q6_k_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Q6_K matmul kernel function was not found"
        );
    }

    NSError* matmul_q6_k_pipeline_error = nil;
    matmul_q6_k_pipeline = [device newComputePipelineStateWithFunction:matmul_q6_k_function
                                                                  error:&matmul_q6_k_pipeline_error];
    if (matmul_q6_k_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Q6_K matmul pipeline creation failed: "
                + ns_error_message(matmul_q6_k_pipeline_error, "unknown error")
        );
    }

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

    id<MTLFunction> llama_cached_decode_qkv_projection_function =
        [library newFunctionWithName:@"llama_cached_decode_qkv_projection_f32_f16_kernel"];
    if (llama_cached_decode_qkv_projection_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode QKV projection kernel function was not found"
        );
    }

    NSError* llama_cached_decode_qkv_projection_pipeline_error = nil;
    llama_cached_decode_qkv_projection_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_decode_qkv_projection_function
                                              error:&llama_cached_decode_qkv_projection_pipeline_error];
    if (llama_cached_decode_qkv_projection_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode QKV projection pipeline creation failed: "
                + ns_error_message(llama_cached_decode_qkv_projection_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_kv_append_function =
        [library newFunctionWithName:@"llama_cached_kv_append_f32_kernel"];
    if (llama_cached_kv_append_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached KV append kernel function was not found"
        );
    }

    NSError* llama_cached_kv_append_pipeline_error = nil;
    llama_cached_kv_append_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_kv_append_function
                                              error:&llama_cached_kv_append_pipeline_error];
    if (llama_cached_kv_append_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached KV append pipeline creation failed: "
                + ns_error_message(llama_cached_kv_append_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_attention_output_function =
        [library newFunctionWithName:@"llama_cached_attention_output_f32_kernel"];
    if (llama_cached_attention_output_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention output kernel function was not found"
        );
    }

    NSError* llama_cached_attention_output_pipeline_error = nil;
    llama_cached_attention_output_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_attention_output_function
                                              error:&llama_cached_attention_output_pipeline_error];
    if (llama_cached_attention_output_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention output pipeline creation failed: "
                + ns_error_message(llama_cached_attention_output_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_decode_attention_output_function =
        [library newFunctionWithName:@"llama_cached_decode_attention_output_f32_kernel"];
    if (llama_cached_decode_attention_output_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode attention output kernel function was not found"
        );
    }

    NSError* llama_cached_decode_attention_output_pipeline_error = nil;
    llama_cached_decode_attention_output_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_decode_attention_output_function
                                              error:&llama_cached_decode_attention_output_pipeline_error];
    if (llama_cached_decode_attention_output_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode attention output pipeline creation failed: "
                + ns_error_message(llama_cached_decode_attention_output_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_decode_attention_residual_norm_function =
        [library newFunctionWithName:@"llama_cached_decode_attention_residual_norm_f32_kernel"];
    if (llama_cached_decode_attention_residual_norm_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode attention residual norm kernel function was not found"
        );
    }

    NSError* llama_cached_decode_attention_residual_norm_pipeline_error = nil;
    llama_cached_decode_attention_residual_norm_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_decode_attention_residual_norm_function
                                              error:&llama_cached_decode_attention_residual_norm_pipeline_error];
    if (llama_cached_decode_attention_residual_norm_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode attention residual norm pipeline creation failed: "
                + ns_error_message(llama_cached_decode_attention_residual_norm_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_decode_ffn_gate_up_silu_function =
        [library newFunctionWithName:@"llama_cached_decode_ffn_gate_up_silu_f32_kernel"];
    if (llama_cached_decode_ffn_gate_up_silu_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode FFN gate/up SiLU kernel function was not found"
        );
    }

    NSError* llama_cached_decode_ffn_gate_up_silu_pipeline_error = nil;
    llama_cached_decode_ffn_gate_up_silu_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_decode_ffn_gate_up_silu_function
                                              error:&llama_cached_decode_ffn_gate_up_silu_pipeline_error];
    if (llama_cached_decode_ffn_gate_up_silu_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached decode FFN gate/up SiLU pipeline creation failed: "
                + ns_error_message(llama_cached_decode_ffn_gate_up_silu_pipeline_error, "unknown error")
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

}  // namespace local_gguf_engine::tensor::backends::metal
