#pragma once

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/dtype.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <array>

namespace local_gguf_engine::tensor::backends::metal {

struct MatmulPipelineState {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> matmul_f16_simd_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> matvec_q8_0_packed_pipeline = nil;
    id<MTLComputePipelineState> matvec_q8_0_packed_112_pipeline = nil;
    id<MTLComputePipelineState> matmul_q8_0_width_two_pipeline = nil;
    id<MTLComputePipelineState> matmul_q8_0_width_two_112_pipeline = nil;
    id<MTLComputePipelineState> target_head_q8_row_parallel_pipeline = nil;
    id<MTLComputePipelineState> target_head_q8_row_parallel_112_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_packed_row_batch_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_packed_row_batch_14_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_packed_row_batch_16_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_tiled_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_tiled_exact_16_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_width_two_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_width_two_14_pipeline = nil;
    id<MTLComputePipelineState> matmul_q4_k_width_two_16_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_packed_row_batch_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_packed_row_batch_14_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_packed_row_batch_56_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_tiled_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_tiled_exact_16_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_width_two_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_width_two_14_pipeline = nil;
    id<MTLComputePipelineState> matmul_q6_k_width_two_56_pipeline = nil;
    id<MTLComputePipelineState> matmul_add_pipeline = nil;
    id<MTLComputePipelineState> silu_multiply_pipeline = nil;
    id<MTLComputePipelineState> gelu_tanh_multiply_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> softcap_pipeline = nil;
    id<MTLComputePipelineState> noop_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_add_pipeline = nil;
    id<MTLComputePipelineState> rope_by_heads_pipeline = nil;
    id<MTLComputePipelineState> rope_neox_by_heads_pipeline = nil;
    id<MTLComputePipelineState> gemma2_kv_cache_store_pipeline = nil;
    id<MTLComputePipelineState> gemma2_kv_cache_copy_pipeline = nil;
    id<MTLComputePipelineState> causal_grouped_attention_scores_pipeline = nil;
    id<MTLComputePipelineState> causal_grouped_attention_values_pipeline = nil;
    id<MTLComputePipelineState> gemma2_prefill_attention_scores_pipeline = nil;
    id<MTLComputePipelineState> gemma2_prefill_attention_values_pipeline = nil;
    id<MTLComputePipelineState> gemma2_wrapped_prefill_attention_scores_pipeline = nil;
    id<MTLComputePipelineState> gemma2_wrapped_prefill_attention_values_pipeline = nil;
    id<MTLComputePipelineState> gemma2_decode_attention_pipeline = nil;
    id<MTLComputePipelineState> gemma2_decode_attention_reduce_pipeline = nil;
    id<MTLComputePipelineState> verifier_rope_pipeline = nil;
    id<MTLComputePipelineState> verifier_kv_store_pipeline = nil;
    id<MTLComputePipelineState> verifier_attention_pipeline = nil;
    id<MTLComputePipelineState> verifier_attention_reduce_pipeline = nil;
    id<MTLComputePipelineState> verifier_kv_scatter_pipeline = nil;
    id<MTLComputePipelineState> verifier_input_validate_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_tile_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_merge_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_finalize_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_top1_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_batch_tile_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_batch_merge_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_batch_finalize_pipeline = nil;
    id<MTLComputePipelineState> token_candidate_batch_top1_pipeline = nil;
    id<MTLComputePipelineState> verifier_request_resolve_pipeline = nil;
    id<MTLComputePipelineState> target_candidate_batch_tile_pipeline = nil;
    id<MTLComputePipelineState> target_candidate_batch_merge_pipeline = nil;
    id<MTLComputePipelineState> target_candidate_batch_finalize_pipeline = nil;
    id<MTLComputePipelineState> target_candidate_batch_top1_pipeline = nil;
    id<MTLComputePipelineState> feature_tap_copy_pipeline = nil;
    id<MTLComputePipelineState> feature_tap_store_f16_pipeline = nil;
    id<MTLComputePipelineState> feature_scatter_f32_pipeline = nil;
    id<MTLComputePipelineState> feature_scatter_f16_pipeline = nil;
    id<MTLComputePipelineState> f16_row_scatter_pipeline = nil;
    id<MTLComputePipelineState> draft_topk_tile_pipeline = nil;
    id<MTLComputePipelineState> draft_topk_merge_pipeline = nil;
    id<MTLComputePipelineState> draft_clear_u32_pipeline = nil;
    id<MTLComputePipelineState> draft_union_mark_selector_pipeline = nil;
    id<MTLComputePipelineState> draft_union_mark_aux_pipeline = nil;
    id<MTLComputePipelineState> draft_union_count_tiles_pipeline = nil;
    id<MTLComputePipelineState> draft_union_scan_tiles_pipeline = nil;
    id<MTLComputePipelineState> draft_union_scatter_pipeline = nil;
    id<MTLComputePipelineState> draft_indexed_f16_pipeline = nil;
    id<MTLComputePipelineState> draft_indexed_q8_pipeline = nil;
    id<MTLComputePipelineState> draft_indexed_q6_pipeline = nil;
    id<MTLComputePipelineState> draft_indexed_q4_pipeline = nil;
    id<MTLComputePipelineState> draft_score_correct_pipeline = nil;
    id<MTLComputePipelineState> draft_proposal_pipeline = nil;
    id<MTLComputePipelineState> corrected_logsumexp_pipeline = nil;
    id<MTLComputePipelineState> corrected_rank_finalize_pipeline = nil;
    id<MTLComputePipelineState> arm_s_transition_score_pipeline = nil;
    std::array<id<MTLComputePipelineState>, 7> recurrent_pipelines {};
    id<MTLCommandQueue> queue = nil;

    MatmulPipelineState();
};

struct QuantizedMatmulDispatch {
    id<MTLComputePipelineState> pipeline = nil;
    MTLSize threadgroups = MTLSizeMake(0, 0, 0);
    MTLSize threads_per_threadgroup = MTLSizeMake(0, 0, 0);
};

struct F16MatmulDispatch {
    id<MTLComputePipelineState> pipeline = nil;
    MTLSize threadgroups = MTLSizeMake(0, 0, 0);
    MTLSize threads_per_threadgroup = MTLSizeMake(0, 0, 0);
};

[[nodiscard]] const char* metal_kernel_source() noexcept;
[[nodiscard]] std::string ns_error_message(NSError* error, const char* fallback);
[[nodiscard]] runtime::BackendError metal_error(
    runtime::BackendErrorCode code,
    const std::string& detail
);
// Retain one device for the process lifetime. Repeatedly creating temporary
// MTLDevice objects can flood the IOKit termination queue during large suites.
[[nodiscard]] id<MTLDevice> system_metal_device() noexcept;
[[nodiscard]] id<MTLDevice> metal_device_or_throw();
[[nodiscard]] MatmulPipelineState& shared_matmul_pipeline_state();
void prepare_recurrent_pipelines();
[[nodiscard]] F16MatmulDispatch select_f16_matmul_dispatch(
    MatmulPipelineState& state,
    std::uint64_t rows,
    std::uint64_t columns
);
[[nodiscard]] QuantizedMatmulDispatch select_quantized_matmul_dispatch(
    MatmulPipelineState& state,
    DType dtype,
    std::uint64_t rows,
    std::uint64_t shared,
    std::uint64_t columns
);

}  // namespace local_gguf_engine::tensor::backends::metal
