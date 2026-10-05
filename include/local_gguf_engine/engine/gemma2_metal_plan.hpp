#pragma once

#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <array>
#include <vector>

namespace local_gguf_engine::engine {

enum class Gemma2OutputHeadPolicy : std::uint32_t {
    automatic = 0U,
    f16 = 1U,
    q8_0 = 2U,
    q6_k = 3U,
};

[[nodiscard]] const char* gemma2_output_head_policy_name(Gemma2OutputHeadPolicy policy) noexcept;
[[nodiscard]] Gemma2OutputHeadPolicy resolve_gemma2_output_head_policy(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights,
    Gemma2OutputHeadPolicy requested_policy
);

struct Gemma2MetalAttentionWeightBuffers {
    tensor::BackendBuffer attention_norm_weight;
    tensor::BackendBuffer post_attention_norm_weight;
    tensor::BackendBuffer query_weight;
    tensor::BackendBuffer key_weight;
    tensor::BackendBuffer value_weight;
    tensor::BackendBuffer output_weight;
    std::optional<tensor::BackendBuffer> query_norm_weight;
    std::optional<tensor::BackendBuffer> key_norm_weight;
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

// Logical bytes read by one retained one-token decode. These are derived from
// the plan's validated weight descriptors; they are not hardware traffic
// counters. The embedding entry is one selected F16 row rather than the whole
// tied embedding/output allocation.
struct Gemma2DecodeScheduledWeightBytes {
    std::uint64_t embedding = 0;
    std::uint64_t attention_projections = 0;
    std::uint64_t feed_forward_gate_up = 0;
    std::uint64_t feed_forward_down = 0;
    std::uint64_t output_head = 0;
    std::uint64_t norms = 0;
    std::uint64_t total = 0;
};

struct Gemma2MetalKVCacheLayer {
    tensor::BackendBuffer key;
    tensor::BackendBuffer value;
    std::uint32_t row_capacity = 0;
    std::uint32_t row_mask = 0;
};

inline constexpr std::uint32_t gemma2_decode_attention_max_partitions = 32U;
inline constexpr std::uint32_t gemma2_decode_attention_query_heads = 16U;
inline constexpr std::uint32_t gemma2_decode_attention_head_dimension = 256U;
inline constexpr std::uint32_t gemma2_metal_verifier_row_max = 1'025U;

struct Gemma2MetalDecodeScratch {
    // The decode executor treats these buffers as a fixed liveness map. Raw
    // Q/K must not alias their RoPE destinations because paired lanes are read
    // concurrently by the current kernel.
    tensor::BackendBuffer token_id_staging;
    tensor::BackendBuffer normalized_hidden;
    tensor::BackendBuffer raw_query;
    tensor::BackendBuffer raw_key;
    tensor::BackendBuffer raw_value;
    tensor::BackendBuffer query;
    tensor::BackendBuffer attention_partial_output;
    tensor::BackendBuffer attention_partial_maxima;
    tensor::BackendBuffer attention_partial_denominators;
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

struct Gemma2MetalVerifierKVBackupLayer {
    tensor::BackendBuffer backup_key;
    tensor::BackendBuffer backup_value;
};

struct Gemma2MetalVerifierScratch {
    tensor::BackendBuffer token_id_staging;
    tensor::BackendBuffer normalized_hidden;
    tensor::BackendBuffer raw_query;
    tensor::BackendBuffer raw_key;
    tensor::BackendBuffer raw_value;
    tensor::BackendBuffer query;
    tensor::BackendBuffer attention_partial_output;
    tensor::BackendBuffer attention_partial_maxima;
    tensor::BackendBuffer attention_partial_denominators;
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
    // The verifier leaves both computed rows in the resident cache. Original
    // rows remain here until the controller accepts both rows or restores a
    // rejected suffix. This makes full acceptance a zero-dispatch operation.
    std::vector<Gemma2MetalVerifierKVBackupLayer> kv_backups;
};

inline constexpr std::uint32_t gemma2_metal_prefill_chunk_size = 128U;
static_assert(
    (gemma2_metal_prefill_chunk_size & (gemma2_metal_prefill_chunk_size - 1U)) == 0U
);

struct Gemma2MetalPrefillScratch {
    // Token IDs are staged by absolute cache position so consecutive command
    // buffers never race with CPU writes while sharing activation scratch.
    // Equal-descriptor fields may share storage only according to the serial
    // liveness schedule established by allocate_prefill_scratch().
    tensor::BackendBuffer token_id_staging;
    tensor::BackendBuffer normalized_hidden;
    tensor::BackendBuffer raw_query;
    tensor::BackendBuffer raw_key;
    tensor::BackendBuffer query;
    tensor::BackendBuffer value;
    // Once a sliding ring wraps, publishing a multi-row chunk before its
    // early queries run would overwrite old K/V that those queries still
    // need. These F16 buffers retain the current chunk until attention has
    // consumed the old ring, then one device copy publishes the live rows.
    tensor::BackendBuffer current_chunk_key;
    tensor::BackendBuffer current_chunk_value;
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
};

inline constexpr std::uint32_t gemma2_metal_selection_tile_size = 512U;

struct Gemma2MetalSelectionWorkspace {
    // Candidate buffers store packed { uint32 token, float logit } pairs. They
    // are shared by prefill, decode, and the two-row verifier, which cannot
    // select concurrently. Storage has two fixed row lanes; one-row paths use
    // lane zero without a separate allocation.
    tensor::BackendBuffer candidate_runs_a;
    tensor::BackendBuffer candidate_runs_b;
    tensor::BackendBuffer compact_result;
    tensor::BackendBuffer history_token_staging;
};

struct Gemma2MetalLayerState {
    std::uint32_t layer_index = 0;
    bool uses_sliding_window = false;
};

enum class Gemma2MetalSchedulePhase {
    prefill,
    decode,
    verify,
};

enum class Gemma2MetalScratchOwnership {
    resident_prefill,
    resident_decode,
    resident_verifier,
};

enum class VerifierTransactionState : std::uint8_t {
    idle,
    encoded,
    submitted,
    completed,
    provisional,
    poisoned,
};

struct Gemma2MetalLayerSchedule {
    std::uint32_t layer_index = 0;
    bool uses_sliding_window = false;
    std::uint32_t attention_token_count = 0;
    std::uint32_t second_attention_token_count = 0;
};

struct Gemma2MetalVerificationStateAccess;
struct ResidentTargetPayloadReceipt;
class Gemma2MetalTargetVerifierPlan;

class Gemma2MetalModelPlan final {
public:
    Gemma2MetalModelPlan(
        Gemma2ModelConfig config,
        Gemma2MetalModelWeightBuffers weights,
        std::vector<Gemma2MetalKVCacheLayer> kv_cache,
        Gemma2MetalPrefillScratch prefill_scratch,
        Gemma2MetalDecodeScratch decode_scratch,
        Gemma2MetalVerifierScratch verifier_scratch,
        Gemma2MetalSelectionWorkspace selection_workspace,
        std::vector<Gemma2MetalLayerState> layers,
        Gemma2OutputHeadPolicy output_head_policy,
        Gemma2DecodeScheduledWeightBytes decode_scheduled_weight_bytes,
        std::size_t uploaded_weight_byte_count,
        std::size_t kv_cache_byte_count,
        std::size_t scratch_byte_count
    );

    [[nodiscard]] runtime::BackendKind backend() const noexcept;
    [[nodiscard]] const Gemma2ModelConfig& config() const noexcept;
    [[nodiscard]] Gemma2OutputHeadPolicy output_head_policy() const noexcept;
    [[nodiscard]] const Gemma2MetalModelWeightBuffers& weights() const noexcept;
    [[nodiscard]] std::vector<Gemma2MetalKVCacheLayer>& kv_cache() noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalKVCacheLayer>& kv_cache() const noexcept;
    [[nodiscard]] const Gemma2MetalPrefillScratch& prefill_scratch() const noexcept;
    [[nodiscard]] const Gemma2MetalDecodeScratch& decode_scratch() const noexcept;
    [[nodiscard]] const Gemma2MetalVerifierScratch& verifier_scratch() const noexcept;
    [[nodiscard]] const Gemma2MetalSelectionWorkspace& selection_workspace() const noexcept;
    [[nodiscard]] const std::vector<Gemma2MetalLayerState>& layers() const noexcept;
    [[nodiscard]] const Gemma2DecodeScheduledWeightBytes& decode_scheduled_weight_bytes() const noexcept;
    [[nodiscard]] std::size_t uploaded_weight_byte_count() const noexcept;
    [[nodiscard]] std::size_t kv_cache_byte_count() const noexcept;
    [[nodiscard]] std::size_t scratch_byte_count() const noexcept;
    [[nodiscard]] std::shared_ptr<Gemma2MetalTargetVerifierPlan>& target_execution_workspace() noexcept;
    [[nodiscard]] std::array<std::uintptr_t, 4> allocation_identities() const noexcept;
    [[nodiscard]] bool has_pending_verification() const noexcept;
    [[nodiscard]] bool pending_verification_completed() const noexcept;
    [[nodiscard]] VerifierTransactionState verification_state() const noexcept;
    [[nodiscard]] bool verification_poisoned() const noexcept;
    [[nodiscard]] std::uint32_t pending_verification_position() const;
    [[nodiscard]] std::uint32_t pending_verification_row_count() const;
    [[nodiscard]] std::uint32_t provisionally_committed_token_count() const noexcept;

private:
    friend struct Gemma2MetalVerificationStateAccess;

    void set_pending_verification(
        std::uint32_t cache_position,
        std::uint32_t row_count,
        VerifierTransactionState state = VerifierTransactionState::submitted
    );
    void mark_pending_verification_submitted();
    void mark_pending_verification_completed();
    void set_provisionally_committed_token_count(std::uint32_t token_count);
    void clear_pending_verification() noexcept;
    void poison_pending_verification() noexcept;

    Gemma2ModelConfig config_;
    Gemma2MetalModelWeightBuffers weights_;
    std::vector<Gemma2MetalKVCacheLayer> kv_cache_;
    Gemma2MetalPrefillScratch prefill_scratch_;
    Gemma2MetalDecodeScratch decode_scratch_;
    Gemma2MetalVerifierScratch verifier_scratch_;
    Gemma2MetalSelectionWorkspace selection_workspace_;
    std::vector<Gemma2MetalLayerState> layers_;
    Gemma2OutputHeadPolicy output_head_policy_ = Gemma2OutputHeadPolicy::f16;
    Gemma2DecodeScheduledWeightBytes decode_scheduled_weight_bytes_;
    std::size_t uploaded_weight_byte_count_ = 0;
    std::size_t kv_cache_byte_count_ = 0;
    std::size_t scratch_byte_count_ = 0;
    VerifierTransactionState verification_state_ = VerifierTransactionState::idle;
    std::uint32_t pending_verification_position_ = 0;
    std::uint32_t pending_verification_row_count_ = 0;
    std::uint32_t provisionally_committed_token_count_ = 0;
    std::shared_ptr<Gemma2MetalTargetVerifierPlan> target_execution_workspace_;
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
    friend Gemma2MetalSchedule create_gemma2_metal_verifier_schedule(
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

struct Gemma3MetalMemoryEstimate {
    std::size_t weights = 0;
    std::size_t kv_cache = 0;
    // Includes the lazily prepared ordinary execution arena, not a speculative arena.
    std::size_t scratch = 0;
};
// Descriptor accounting only: validates the native Gemma 3 layout without
// uploading weights or allocating device storage. Tested against owned ledgers.
[[nodiscard]] Gemma3MetalMemoryEstimate estimate_gemma3_metal_memory(
    const Gemma2ModelConfig& config, const Gemma2ModelWeights& weights);

[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const EngineModel& model,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic,
    std::uint32_t context_length = 0U,
    GemmaKVPrecision kv_precision = GemmaKVPrecision::automatic
);
[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const EngineModel& model,
    Gemma2OutputHeadPolicy policy,
    ResidentTargetPayloadReceipt& resident_receipt,
    std::uint32_t context_length = 0U,
    GemmaKVPrecision kv_precision = GemmaKVPrecision::automatic
);
[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic
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
[[nodiscard]] Gemma2MetalSchedule create_gemma2_metal_verifier_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t cached_token_count
);

}  // namespace local_gguf_engine::engine
