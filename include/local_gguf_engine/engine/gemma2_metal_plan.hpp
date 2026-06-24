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
    tensor::BackendBuffer attention_probabilities;
    tensor::BackendBuffer attention_output;
    tensor::BackendBuffer projected_attention_output;
    tensor::BackendBuffer post_attention_residual;
    tensor::BackendBuffer normalized_feed_forward_input;
    tensor::BackendBuffer feed_forward_gate;
    tensor::BackendBuffer feed_forward_up;
    tensor::BackendBuffer feed_forward_gated;
    tensor::BackendBuffer projected_feed_forward_output;
    tensor::BackendBuffer feed_forward_residual;
    tensor::BackendBuffer logits;
};

struct Gemma2MetalLayerState {
    std::uint32_t layer_index = 0;
    bool uses_sliding_window = false;
};

enum class Gemma2MetalSchedulePhase {
    prefill,
    decode,
};

enum class Gemma2MetalScratchOwnership {
    phase_local,
    resident_decode,
};

struct Gemma2MetalScheduleScratch {
    tensor::TensorDescriptor normalized_hidden;
    tensor::TensorDescriptor query;
    tensor::TensorDescriptor key;
    tensor::TensorDescriptor value;
    tensor::TensorDescriptor attention_scores;
    tensor::TensorDescriptor attention_probabilities;
    tensor::TensorDescriptor attention_output;
    tensor::TensorDescriptor projected_attention_output;
    tensor::TensorDescriptor post_attention_residual;
    tensor::TensorDescriptor normalized_feed_forward_input;
    tensor::TensorDescriptor feed_forward_gate;
    tensor::TensorDescriptor feed_forward_up;
    tensor::TensorDescriptor feed_forward_gated;
    tensor::TensorDescriptor projected_feed_forward_output;
    tensor::TensorDescriptor feed_forward_residual;
    tensor::TensorDescriptor logits;
};

struct Gemma2MetalLayerSchedule {
    std::uint32_t layer_index = 0;
    bool uses_sliding_window = false;
    std::uint32_t attention_token_count = 0;
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

class Gemma2MetalSchedule final {
public:
    [[nodiscard]] runtime::BackendKind backend() const noexcept;
    [[nodiscard]] const Gemma2ModelConfig& config() const noexcept;
    [[nodiscard]] Gemma2MetalSchedulePhase phase() const noexcept;
    [[nodiscard]] std::uint32_t token_count() const noexcept;
    [[nodiscard]] std::uint32_t cache_position() const noexcept;
    [[nodiscard]] std::uint32_t total_kv_token_count() const noexcept;
    [[nodiscard]] Gemma2MetalScratchOwnership scratch_ownership() const noexcept;
    [[nodiscard]] const Gemma2MetalScheduleScratch& scratch() const noexcept;
    [[nodiscard]] const Gemma2MetalModelWeightBuffers& weights() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalKVCacheLayer>& kv_cache() const noexcept;
    [[nodiscard]] const Gemma2MetalDecodeScratch& resident_decode_scratch() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalLayerSchedule>& layers() const noexcept;
    [[nodiscard]] std::size_t resident_weight_byte_count() const noexcept;
    [[nodiscard]] std::size_t resident_kv_cache_byte_count() const noexcept;
    [[nodiscard]] std::size_t resident_decode_scratch_byte_count() const noexcept;
    [[nodiscard]] std::size_t phase_scratch_byte_count() const noexcept;

private:
    Gemma2MetalSchedule(
        Gemma2ModelConfig config,
        Gemma2MetalModelWeightBuffers weights,
        std::vector<Gemma2MetalKVCacheLayer> kv_cache,
        Gemma2MetalDecodeScratch resident_decode_scratch,
        std::vector<Gemma2MetalLayerSchedule> layers,
        Gemma2MetalSchedulePhase phase,
        Gemma2MetalScratchOwnership scratch_ownership,
        Gemma2MetalScheduleScratch scratch,
        std::uint32_t token_count,
        std::uint32_t cache_position,
        std::uint32_t total_kv_token_count,
        std::size_t resident_weight_byte_count,
        std::size_t resident_kv_cache_byte_count,
        std::size_t resident_decode_scratch_byte_count,
        std::size_t phase_scratch_byte_count
    );

    friend Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
        const Gemma2MetalModelPlan& plan,
        std::uint32_t prompt_token_count
    );
    friend Gemma2MetalSchedule create_gemma2_metal_decode_schedule(
        const Gemma2MetalModelPlan& plan,
        std::uint32_t cached_token_count
    );

    Gemma2ModelConfig config_;
    Gemma2MetalModelWeightBuffers weights_;
    std::vector<Gemma2MetalKVCacheLayer> kv_cache_;
    Gemma2MetalDecodeScratch resident_decode_scratch_;
    std::vector<Gemma2MetalLayerSchedule> layers_;
    Gemma2MetalSchedulePhase phase_ = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership_ = Gemma2MetalScratchOwnership::resident_decode;
    Gemma2MetalScheduleScratch scratch_;
    std::uint32_t token_count_ = 0;
    std::uint32_t cache_position_ = 0;
    std::uint32_t total_kv_token_count_ = 0;
    std::size_t resident_weight_byte_count_ = 0;
    std::size_t resident_kv_cache_byte_count_ = 0;
    std::size_t resident_decode_scratch_byte_count_ = 0;
    std::size_t phase_scratch_byte_count_ = 0;
};

[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(const EngineModel& model);
[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights
);
[[nodiscard]] Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t prompt_token_count
);
[[nodiscard]] Gemma2MetalSchedule create_gemma2_metal_decode_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t cached_token_count
);

}  // namespace local_gguf_engine::engine
