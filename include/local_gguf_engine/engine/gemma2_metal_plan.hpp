#pragma once

#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace local_gguf_engine::engine {

struct Gemma2MetalAttentionWeightBuffers {
    tensor::BackendBuffer attention_norm_weight;
    tensor::BackendBuffer post_attention_norm_weight;
    tensor::BackendBuffer query_weight;
    tensor::BackendBuffer key_weight;
    tensor::BackendBuffer value_weight;
    tensor::BackendBuffer output_weight;
};

struct Gemma2MetalFeedForwardWeightBuffers {
    tensor::BackendBuffer feed_forward_norm_weight;
    tensor::BackendBuffer post_feed_forward_norm_weight;
    tensor::BackendBuffer gate_weight;
    tensor::BackendBuffer up_weight;
    tensor::BackendBuffer down_weight;
};

struct Gemma2MetalBlockWeightBuffers {
    Gemma2MetalAttentionWeightBuffers attention;
    Gemma2MetalFeedForwardWeightBuffers feed_forward;
};

struct Gemma2MetalModelWeightBuffers {
    tensor::BackendBuffer token_embedding_weight;
    tensor::BackendBuffer output_norm_weight;
    tensor::BackendBuffer output_weight;
    std::vector<Gemma2MetalBlockWeightBuffers> blocks;
};

struct Gemma2MetalKVCacheLayer {
    tensor::BackendBuffer key;
    tensor::BackendBuffer value;
};

struct Gemma2MetalDecodeScratch {
    tensor::BackendBuffer normalized_hidden;
    tensor::BackendBuffer query;
    tensor::BackendBuffer key;
    tensor::BackendBuffer value;
    tensor::BackendBuffer attention_scores;
    tensor::BackendBuffer attention_output;
    tensor::BackendBuffer feed_forward_gate;
    tensor::BackendBuffer feed_forward_up;
    tensor::BackendBuffer feed_forward_gated;
    tensor::BackendBuffer logits;
};

struct Gemma2MetalLayerState {
    std::uint32_t layer_index = 0;
    bool uses_sliding_window = false;
};

class Gemma2MetalModelPlan final {
public:
    Gemma2MetalModelPlan(
        Gemma2ModelConfig config,
        Gemma2MetalModelWeightBuffers weights,
        std::vector<Gemma2MetalKVCacheLayer> kv_cache,
        Gemma2MetalDecodeScratch decode_scratch,
        std::vector<Gemma2MetalLayerState> layers,
        std::size_t uploaded_weight_byte_count,
        std::size_t kv_cache_byte_count,
        std::size_t scratch_byte_count
    );

    [[nodiscard]] runtime::BackendKind backend() const noexcept;
    [[nodiscard]] const Gemma2ModelConfig& config() const noexcept;
    [[nodiscard]] const Gemma2MetalModelWeightBuffers& weights() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalKVCacheLayer>& kv_cache() const noexcept;
    [[nodiscard]] const Gemma2MetalDecodeScratch& decode_scratch() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalLayerState>& layers() const noexcept;
    [[nodiscard]] std::size_t uploaded_weight_byte_count() const noexcept;
    [[nodiscard]] std::size_t kv_cache_byte_count() const noexcept;
    [[nodiscard]] std::size_t scratch_byte_count() const noexcept;

private:
    Gemma2ModelConfig config_;
    Gemma2MetalModelWeightBuffers weights_;
    std::vector<Gemma2MetalKVCacheLayer> kv_cache_;
    Gemma2MetalDecodeScratch decode_scratch_;
    std::vector<Gemma2MetalLayerState> layers_;
    std::size_t uploaded_weight_byte_count_ = 0;
    std::size_t kv_cache_byte_count_ = 0;
    std::size_t scratch_byte_count_ = 0;
};

[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(const EngineModel& model);
[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights
);

}  // namespace local_gguf_engine::engine
