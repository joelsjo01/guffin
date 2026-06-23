#pragma once

#include "local_gguf_engine/runtime/backend_error.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <mutex>
#include <string>

namespace local_gguf_engine::tensor::backends::metal {

struct LlamaCachedBlockScratchBuffers {
    id<MTLBuffer> attention_normalized = nil;
    id<MTLBuffer> query = nil;
    id<MTLBuffer> key = nil;
    id<MTLBuffer> value = nil;
    id<MTLBuffer> attention_output = nil;
    id<MTLBuffer> attention_residual = nil;
    id<MTLBuffer> feed_forward_normalized = nil;
    id<MTLBuffer> gate = nil;
    id<MTLBuffer> up = nil;
    id<MTLBuffer> gated = nil;
};

struct MatmulPipelineState {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_pipeline = nil;
    id<MTLComputePipelineState> matmul_add_pipeline = nil;
    id<MTLComputePipelineState> silu_multiply_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> noop_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLComputePipelineState> rope_by_heads_pipeline = nil;
    id<MTLComputePipelineState> causal_grouped_attention_scores_pipeline = nil;
    id<MTLComputePipelineState> causal_grouped_attention_values_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_decode_qkv_projection_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_kv_append_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_attention_output_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_decode_attention_output_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_decode_attention_residual_norm_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_decode_ffn_gate_up_silu_pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    std::mutex llama_cached_block_scratch_mutex;
    LlamaCachedBlockScratchBuffers llama_cached_block_scratch;

    MatmulPipelineState();
};

[[nodiscard]] const char* metal_kernel_source() noexcept;
[[nodiscard]] std::string ns_error_message(NSError* error, const char* fallback);
[[nodiscard]] runtime::BackendError metal_error(
    runtime::BackendErrorCode code,
    const std::string& detail
);
[[nodiscard]] id<MTLDevice> metal_device_or_throw();
[[nodiscard]] MatmulPipelineState& shared_matmul_pipeline_state();

}  // namespace local_gguf_engine::tensor::backends::metal
