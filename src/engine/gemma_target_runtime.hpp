#pragma once

#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"
#include "../tensor/backends/metal/metal_command_context.hpp"

namespace local_gguf_engine::engine {

tensor::backends::metal::MetalCommandContext& target_workspace_command(
    Gemma2MetalTargetVerifierPlan& workspace
);

CompactTokenCandidates target_workspace_completed_candidates(
    Gemma2MetalTargetVerifierPlan&, const Gemma2MetalSelectionWorkspace&
);

// Packed retrieval shares the graph block schedule. Gemma 2 ordinary
// decode/prefill retain their original operation graphs; their projection
// encoder reads the same sole packed resident representation.
inline bool gemma_target_uses_shared_retrieval(const Gemma2MetalModelPlan& target) {
    if (target.config().architecture == ModelArchitecture::gemma3) return true;
    for (const auto& block : target.weights().blocks) {
        for (const auto* weight : {&block.attention.query_weight, &block.attention.key_weight,
                 &block.attention.value_weight, &block.attention.output_weight,
                 &block.feed_forward.gate_weight, &block.feed_forward.up_weight,
                 &block.feed_forward.down_weight}) {
            if (weight->descriptor().packing == tensor::TensorPacking::kquant_columns8)
                return true;
        }
    }
    return false;
}

// The same block schedule consumes either a contiguous prompt or validated
// speculative ancestry. Only the caller owns publication and rollback.
void encode_gemma_target_blocks(
    tensor::backends::metal::MetalCommandContext& command,
    Gemma2MetalModelPlan& target,
    const Gemma2MetalTargetVerifierScratch& scratch,
    std::uint32_t row_capacity, std::uint32_t row_count,
    const tensor::BackendBuffer& token_source, std::uint32_t token_offset,
    float embedding_scale, FeatureSinkBinding feature_sink = {},
    bool produce_final_hidden = true, bool prefill_projection = false
);

Gemma2MetalResidentDecodeResult run_gemma3_resident_decode(
    Gemma2MetalModelPlan&, std::uint32_t position, std::uint32_t token,
    float embedding_scale, std::optional<TokenSelectionRequest>,
    FeatureSinkBinding, Gemma2MetalDecodeOutputMode, Gemma2ExecutionReceipt*
);
Gemma2MetalResidentPrefillResult run_gemma3_resident_prefill(
    Gemma2MetalModelPlan&, std::span<const std::uint32_t>, std::uint32_t position,
    float embedding_scale, std::optional<TokenSelectionRequest>,
    FeatureSinkBinding, Gemma2ExecutionReceipt*
);
Gemma2MetalVerifierResult run_gemma_target_width_two_verifier(
    Gemma2MetalModelPlan&, std::uint32_t position,
    std::span<const std::uint32_t, 2>, float embedding_scale,
    std::optional<std::array<TokenSelectionRequest, 2>>
);

}  // namespace local_gguf_engine::engine
