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
    // Packed Gemma 2 retains its half-operand prompt tiles and F32 accumulation.
    // [M32/M16] for Q4; separate from Gemma 3's F32 prompt contract.
    id<MTLComputePipelineState> columns8_gemma2_prompt[2] = {};
    // [canonical/columns8][Q4/Q6][M8/N32/K64, M32/N64/K32], F32 prompt arithmetic.
    std::array<std::array<std::array<id<MTLComputePipelineState>, 2>, 2>, 2>
        gemma3_prompt_matrix = {};
    id<MTLComputePipelineState> gemma3_q4_shared_two_pipeline = nil;
    id<MTLComputePipelineState> gemma3_q6_shared_one_pipeline = nil;
    id<MTLComputePipelineState> gemma3_q6_shared_two_pipeline = nil;
    std::array<id<MTLComputePipelineState>, 2> columns8_q4_rows = {nil, nil};
    std::array<id<MTLComputePipelineState>, 2> columns8_q4_prepared_rows = {nil, nil};
    id<MTLComputePipelineState> rms_norm_q4_sums_pipeline = nil;
    id<MTLComputePipelineState> gelu_tanh_multiply_q4_sums_pipeline = nil;
    id<MTLComputePipelineState> verifier_attention_q4_sums_pipeline = nil;
    id<MTLComputePipelineState> verifier_attention_q4_sums_f16_pipeline = nil;
    id<MTLComputePipelineState> verifier_attention_reduce_q4_sums_pipeline = nil;
    std::array<id<MTLComputePipelineState>, 2> columns8_q6_rows = {nil, nil};
    id<MTLComputePipelineState> matmul_f16_simd_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> embedding_q6_pipeline = nil;
    id<MTLComputePipelineState> head_rms_norm_pipeline = nil;
    id<MTLComputePipelineState> kv_cache_copy_f32_pipeline = nil;
    id<MTLComputePipelineState> verifier_kv_store_f32_pipeline = nil;
    id<MTLComputePipelineState> verifier_kv_store_f16_checked_pipeline = nil;
    id<MTLComputePipelineState> verifier_kv_scatter_f32_pipeline = nil;
    id<MTLComputePipelineState> verifier_attention_f32_pipeline = nil;
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
    id<MTLComputePipelineState> gemma2_prefill_attention_pipeline = nil;
    id<MTLComputePipelineState> gemma2_prefill_attention_generic_pipeline = nil;
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
    id<MTLCommandQueue> queue = nil;

    explicit MatmulPipelineState(bool precise_math = false);
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

[[nodiscard]] std::string ns_error_message(NSError* error, const char* fallback);
[[nodiscard]] runtime::BackendError metal_error(
    runtime::BackendErrorCode code,
    const std::string& detail
);
// Retain one device for the process lifetime. Repeatedly creating temporary
// MTLDevice objects can flood the IOKit termination queue during large suites.
[[nodiscard]] id<MTLDevice> system_metal_device() noexcept;
[[nodiscard]] id<MTLDevice> metal_device_or_throw();
// The default target and standalone drafter share a queue independently of
// target pipeline preparation. Precise targets retain their separate queue.
[[nodiscard]] id<MTLCommandQueue> shared_metal_command_queue();
[[nodiscard]] MatmulPipelineState& shared_matmul_pipeline_state();
[[nodiscard]] MatmulPipelineState& shared_precise_matmul_pipeline_state();
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
