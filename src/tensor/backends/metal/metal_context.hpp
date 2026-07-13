#pragma once

#include "local_gguf_engine/runtime/backend_error.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

namespace local_gguf_engine::tensor::backends::metal {

struct MatmulPipelineState {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_pipeline = nil;
    id<MTLComputePipelineState> matvec_q4_k_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_pipeline = nil;
    id<MTLComputePipelineState> matvec_q6_k_pipeline = nil;
    id<MTLComputePipelineState> matmul_add_pipeline = nil;
    id<MTLComputePipelineState> silu_multiply_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> softcap_pipeline = nil;
    id<MTLComputePipelineState> noop_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLComputePipelineState> rope_by_heads_pipeline = nil;
    id<MTLComputePipelineState> causal_grouped_attention_scores_pipeline = nil;
    id<MTLComputePipelineState> causal_grouped_attention_values_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> top_logit_pipeline = nil;
    id<MTLCommandQueue> queue = nil;

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
