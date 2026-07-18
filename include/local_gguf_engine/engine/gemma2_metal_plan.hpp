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
    // The decode executor treats these buffers as a fixed liveness map. Raw
    // Q/K must not alias their RoPE destinations because paired lanes are read
    // concurrently by the current kernel.
    tensor::BackendBuffer token_id_staging;
    tensor::BackendBuffer normalized_hidden;
    tensor::BackendBuffer raw_query;
    tensor::BackendBuffer raw_key;
    tensor::BackendBuffer query;
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
    tensor::BackendBuffer top_logit_staging;
};

inline constexpr std::uint32_t gemma2_metal_prefill_chunk_size = 16U;

struct Gemma2MetalPrefillScratch {
    // Token IDs are staged by absolute cache position so consecutive command
    // buffers never race with CPU writes while sharing activation scratch.
    tensor::BackendBuffer token_id_staging;
    tensor::BackendBuffer normalized_hidden;
    tensor::BackendBuffer raw_query;
    tensor::BackendBuffer raw_key;
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
    tensor::BackendBuffer final_normalized_hidden;
    tensor::BackendBuffer logits;
    tensor::BackendBuffer top_logit_staging;
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
    resident_prefill,
    resident_decode,
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
        Gemma2MetalPrefillScratch prefill_scratch,
        Gemma2MetalDecodeScratch decode_scratch,
        std::vector<Gemma2MetalLayerState> layers,
        std::size_t uploaded_weight_byte_count,
        std::size_t kv_cache_byte_count,
        std::size_t scratch_byte_count
    );

    [[nodiscard]] runtime::BackendKind backend() const noexcept;
    [[nodiscard]] const Gemma2ModelConfig& config() const noexcept;
    [[nodiscard]] const Gemma2MetalModelWeightBuffers& weights() const noexcept;
    [[nodiscard]] std::vector<Gemma2MetalKVCacheLayer>& kv_cache() noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalKVCacheLayer>& kv_cache() const noexcept;
    [[nodiscard]] const Gemma2MetalPrefillScratch& prefill_scratch() const noexcept;
    [[nodiscard]] const Gemma2MetalDecodeScratch& decode_scratch() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalLayerState>& layers() const noexcept;
    [[nodiscard]] std::size_t uploaded_weight_byte_count() const noexcept;
    [[nodiscard]] std::size_t kv_cache_byte_count() const noexcept;
    [[nodiscard]] std::size_t scratch_byte_count() const noexcept;

private:
    Gemma2ModelConfig config_;
    Gemma2MetalModelWeightBuffers weights_;
    std::vector<Gemma2MetalKVCacheLayer> kv_cache_;
    Gemma2MetalPrefillScratch prefill_scratch_;
    Gemma2MetalDecodeScratch decode_scratch_;
    std::vector<Gemma2MetalLayerState> layers_;
    std::size_t uploaded_weight_byte_count_ = 0;
    std::size_t kv_cache_byte_count_ = 0;
    std::size_t scratch_byte_count_ = 0;
};

class Gemma2MetalSchedule final {
public:
    [[nodiscard]] runtime::BackendKind backend() const noexcept;
    [[nodiscard]] Gemma2MetalSchedulePhase phase() const noexcept;
    [[nodiscard]] std::uint32_t token_count() const noexcept;
    [[nodiscard]] std::uint32_t cache_position() const noexcept;
    [[nodiscard]] std::uint32_t total_kv_token_count() const noexcept;
    [[nodiscard]] Gemma2MetalScratchOwnership scratch_ownership() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalLayerSchedule>& layers() const noexcept;

private:
    Gemma2MetalSchedule(
        std::vector<Gemma2MetalLayerSchedule> layers,
        Gemma2MetalSchedulePhase phase,
        Gemma2MetalScratchOwnership scratch_ownership,
        std::uint32_t token_count,
        std::uint32_t cache_position,
        std::uint32_t total_kv_token_count
    );

    friend Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
        const Gemma2MetalModelPlan& plan,
        std::uint32_t prompt_token_count
    );
    friend Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
        const Gemma2MetalModelPlan& plan,
        std::uint32_t token_count,
        std::uint32_t cache_position
    );
    friend Gemma2MetalSchedule create_gemma2_metal_decode_schedule(
        const Gemma2MetalModelPlan& plan,
        std::uint32_t cached_token_count
    );

    std::vector<Gemma2MetalLayerSchedule> layers_;
    Gemma2MetalSchedulePhase phase_ = Gemma2MetalSchedulePhase::decode;
    Gemma2MetalScratchOwnership scratch_ownership_ = Gemma2MetalScratchOwnership::resident_decode;
    std::uint32_t token_count_ = 0;
    std::uint32_t cache_position_ = 0;
    std::uint32_t total_kv_token_count_ = 0;
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
[[nodiscard]] Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t token_count,
    std::uint32_t cache_position
);
[[nodiscard]] Gemma2MetalSchedule create_gemma2_metal_decode_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t cached_token_count
);

}  // namespace local_gguf_engine::engine
