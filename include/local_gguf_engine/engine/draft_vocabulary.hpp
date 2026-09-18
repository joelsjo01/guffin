#pragma once

#include "local_gguf_engine/engine/draft_vocabulary_types.hpp"
#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/speculative_artifact.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <optional>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::uint32_t draft_vocabulary_selection_tile_size = 512U;
inline constexpr std::uint32_t draft_vocabulary_auxiliary_width = 64U;
inline constexpr std::uint32_t draft_vocabulary_proposal_lut_size = 4'097U;

struct DraftVocabularyCandidate {
    std::uint32_t token_id = 0U;
    float score = 0.0F;

    constexpr bool operator==(const DraftVocabularyCandidate&) const = default;
};
static_assert(sizeof(DraftVocabularyCandidate) == 8U);

enum class DraftProposalStatus : std::uint32_t {
    selected = 0U,
    conservative_numeric_fallback = 1U,
    malformed_device_input = 2U,
};

struct DraftProposalResult {
    DraftProposalStatus status = DraftProposalStatus::malformed_device_input;
    std::uint32_t token_id = 0U;
    std::uint64_t total_weight = 0U;
    std::uint32_t selected_rank = 0U;

    constexpr bool operator==(const DraftProposalResult&) const = default;
};

[[nodiscard]] bool is_supported_draft_vocabulary_state_batch(
    std::uint32_t row_count
) noexcept;
[[nodiscard]] bool is_supported_draft_vocabulary_candidate_k(
    std::uint32_t candidate_k,
    std::uint32_t vocabulary_size
) noexcept;

[[nodiscard]] std::vector<DraftVocabularyCandidate> stable_top_k_cpu(
    std::span<const float> scores,
    std::uint32_t row_count,
    std::uint32_t vocabulary_size,
    std::uint32_t candidate_k
);

[[nodiscard]] std::vector<std::uint32_t> candidate_union_cpu(
    std::span<const DraftVocabularyCandidate> selector_candidates,
    std::uint32_t row_count,
    std::uint32_t selector_stride,
    std::span<const std::uint32_t> selector_counts,
    std::span<const std::uint32_t> history_ids,
    std::span<const std::uint32_t> history_counts,
    std::span<const std::uint32_t> required_ids,
    std::span<const std::uint32_t> required_counts,
    std::span<const std::uint32_t> retrieval_ids,
    std::span<const std::uint32_t> retrieval_counts,
    std::span<const std::uint32_t> transition_counts,
    std::span<const std::uint32_t> transition_tokens,
    std::uint32_t transition_width,
    std::uint32_t vocabulary_size,
    std::span<std::uint32_t> output_counts
);

[[nodiscard]] std::vector<float> indexed_vocabulary_dot_cpu(
    std::span<const float> states,
    std::uint32_t row_count,
    std::uint32_t state_width,
    const tensor::TensorView& packed_columns,
    std::span<const std::uint32_t> candidate_ids,
    std::span<const std::uint32_t> candidate_counts,
    std::uint32_t candidate_stride
);

[[nodiscard]] DraftProposalResult select_draft_proposal_cpu(
    std::span<const std::uint32_t> candidate_ids,
    std::span<const float> corrected_scores,
    std::span<const std::uint32_t> exp_lut,
    float temperature,
    std::uint32_t common_draw
);

struct DraftVocabularyDeviceInputs {
    tensor::BackendBuffer semantic_states;
    tensor::BackendBuffer history_ids;
    tensor::BackendBuffer history_counts;
    tensor::BackendBuffer required_ids;
    tensor::BackendBuffer required_counts;
    tensor::BackendBuffer retrieval_ids;
    tensor::BackendBuffer retrieval_counts;
    tensor::BackendBuffer correction_scores;
    tensor::BackendBuffer common_draws;
};

struct DraftVocabularyExecution {
    std::uint32_t row_count = 0U;
    std::uint32_t candidate_k = 0U;
    float temperature = 1.0F;
};

struct DraftVocabularyMetalScratch {
    tensor::BackendBuffer selector_states;
    tensor::BackendBuffer selector_logits;
    tensor::BackendBuffer candidate_runs_a;
    tensor::BackendBuffer candidate_runs_b;
    tensor::BackendBuffer candidate_bits;
    tensor::BackendBuffer candidate_tile_counts;
    tensor::BackendBuffer candidate_tile_offsets;
    tensor::BackendBuffer candidate_ids;
    tensor::BackendBuffer candidate_counts;
    tensor::BackendBuffer target_states;
    tensor::BackendBuffer normalized_target_states;
    tensor::BackendBuffer indexed_scores;
    tensor::BackendBuffer proposal_weights;
    tensor::BackendBuffer proposal_results;
    tensor::BackendBuffer status;
};

struct DraftVocabularyMetalResult;

class DraftVocabularyMetalPlan final {
public:
    DraftVocabularyMetalPlan(const DraftVocabularyMetalPlan&) = delete;
    DraftVocabularyMetalPlan& operator=(const DraftVocabularyMetalPlan&) = delete;
    DraftVocabularyMetalPlan(DraftVocabularyMetalPlan&&) noexcept = default;
    DraftVocabularyMetalPlan& operator=(DraftVocabularyMetalPlan&&) noexcept = default;

    [[nodiscard]] const DraftVocabularyArtifactSpec& spec() const noexcept;
    [[nodiscard]] const DraftVocabularyMetalScratch& scratch() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer* selector_projection() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer& selector_head() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer& candidate_head() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer* target_up_projection() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer* transition_counts() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer* transition_tokens() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer& proposal_exp_lut() const noexcept;
    [[nodiscard]] bool candidate_head_aliases_target() const noexcept;
    [[nodiscard]] std::size_t owned_weight_bytes() const noexcept;
    [[nodiscard]] std::size_t scratch_bytes() const noexcept;

private:
    friend DraftVocabularyMetalPlan create_draft_vocabulary_metal_plan(
        const AuthenticatedSpeculativeArtifact&, const EngineModelPayloadReceipt&,
        const ResidentTargetPayloadReceipt&, const Gemma2MetalModelPlan&
    );
    friend DraftVocabularyMetalResult execute_draft_vocabulary_metal(
        DraftVocabularyMetalPlan&, const Gemma2MetalModelPlan&,
        const DraftVocabularyDeviceInputs&, const DraftVocabularyExecution&
    );
    DraftVocabularyMetalPlan(
        DraftVocabularyArtifactSpec,
        DraftVocabularyMetalScratch,
        std::optional<tensor::BackendBuffer>,
        tensor::BackendBuffer,
        tensor::BackendBuffer,
        std::optional<tensor::BackendBuffer>,
        std::optional<tensor::BackendBuffer>,
        std::optional<tensor::BackendBuffer>,
        tensor::BackendBuffer,
        bool,
        std::size_t,
        std::size_t
    );
    DraftVocabularyArtifactSpec spec_;
    DraftVocabularyMetalScratch scratch_;
    std::optional<tensor::BackendBuffer> selector_projection_;
    tensor::BackendBuffer selector_head_;
    tensor::BackendBuffer candidate_head_;
    std::optional<tensor::BackendBuffer> target_up_projection_;
    std::optional<tensor::BackendBuffer> transition_counts_;
    std::optional<tensor::BackendBuffer> transition_tokens_;
    tensor::BackendBuffer proposal_exp_lut_;
    bool candidate_head_aliases_target_ = false;
    std::size_t owned_weight_bytes_ = 0U;
    std::size_t scratch_bytes_ = 0U;
    std::array<DraftProposalResult, draft_vocabulary_external_state_max> host_proposals_ {};
    std::array<std::uint32_t, draft_vocabulary_external_state_max> host_candidate_counts_ {};
};

[[nodiscard]] DraftVocabularyMetalPlan create_draft_vocabulary_metal_plan(
    const AuthenticatedSpeculativeArtifact& artifact,
    const EngineModelPayloadReceipt& model_receipt,
    const ResidentTargetPayloadReceipt& resident_receipt,
    const Gemma2MetalModelPlan& target
);

struct DraftVocabularyMetalResult {
    std::span<const DraftProposalResult> proposals;
    std::span<const std::uint32_t> candidate_counts;
    std::uint64_t gpu_duration_ns = 0U;
    std::uint64_t wall_duration_ns = 0U;
    std::uint32_t dispatch_count = 0U;
};

[[nodiscard]] DraftVocabularyMetalResult execute_draft_vocabulary_metal(
    DraftVocabularyMetalPlan& plan,
    const Gemma2MetalModelPlan& target,
    const DraftVocabularyDeviceInputs& inputs,
    const DraftVocabularyExecution& execution
);

enum class CorrectedVocabularyWorkload : std::uint8_t {
    treeflash_released_h15_m16_top16,
    treeflash_h15_m64_top16,
    treeflash_h15_m64_top64,
    treeflash_h15_m64_full_v_lazy,
    treeflash_paper_h16_m64_full_v_lazy,
    treeflash_code_h16_m64_full_v_lazy,
    arm_s_cross_root_excluded,
    arm_s_cross_root_included,
    arm_s_deep_gamma23_k1,
    arm_s_threshold_gamma4_k3,
    arm_s_post_walk_selected_anchor,
};

enum class CorrectedVocabularySourceFidelity : std::uint8_t {
    released_source_exact,
    paper_semantic_transfer,
    local_complete_interpretation,
};

struct CorrectedVocabularyWorkloadSpec {
    CorrectedVocabularyWorkload workload;
    CorrectedVocabularySourceFidelity fidelity;
    std::uint32_t semantic_rows = 0U;
    std::uint32_t physical_head_rows = 0U;
    std::uint32_t conditioning_width = 0U;
    std::uint32_t rank_width = 0U;
    std::uint32_t rank_entry_capacity = 0U;
    std::uint32_t raw_lower_main_rows = 0U;
    std::uint32_t upper_main_rows = 0U;
    std::uint32_t upper_stream_rows = 0U;
    std::uint64_t head_macs = 0U;
    std::uint64_t forbidden_dense_f32_bytes = 0U;
};

[[nodiscard]] CorrectedVocabularyWorkloadSpec corrected_vocabulary_workload_spec(
    CorrectedVocabularyWorkload workload,
    std::uint32_t target_width = 3'584U,
    std::uint32_t vocabulary_size = 256'000U
);

[[nodiscard]] std::vector<DraftVocabularyCandidate>
corrected_full_vocabulary_cpu(
    std::span<const float> states,
    std::uint32_t row_count,
    std::span<const float> final_norm_weight,
    float epsilon,
    float softcap,
    float temperature,
    const tensor::TensorView& target_q8_head,
    std::uint32_t rank_width
);

inline constexpr std::uint32_t arm_s_gamma4_k3_raw_edge_count = 120U;
inline constexpr std::uint32_t arm_s_raw_prior_node_max = 256U;
inline constexpr std::uint32_t arm_s_gamma4_k3_compact_node_max = 14'641U;

struct ArmSSurvivingPriorTree {
    std::array<std::uint32_t, arm_s_raw_prior_node_max> raw_node_indices {};
    std::uint32_t survivor_count = 0U;
};

// Applies lossless prefix pruning in raw-tree order. The root is always kept;
// an edge at the frozen threshold is kept, and a rejected edge removes its
// complete descendant subtree through the parent-kept predicate.
[[nodiscard]] ArmSSurvivingPriorTree prune_arm_s_raw_tree_cpu(
    std::span<const std::uint32_t> parent_indices,
    std::span<const float> transition_probabilities,
    float threshold
);

struct ArmSCompactNextTree {
    std::array<std::uint32_t, arm_s_gamma4_k3_compact_node_max> token_ids {};
    std::array<std::uint32_t, arm_s_gamma4_k3_compact_node_max> parent_indices {};
    std::array<std::uint32_t, arm_s_gamma4_k3_compact_node_max> anchor_indices {};
    std::array<std::uint16_t, arm_s_gamma4_k3_compact_node_max> depths {};
    std::uint32_t node_count = 0U;
};

// Materializes each survivor's prefix-closed next raw tree from its gamma
// compact stream menus. The root token for every anchor is explicit, matching
// the 121-node gamma4/k3 accounting used by the Arm-S registry.
[[nodiscard]] ArmSCompactNextTree build_arm_s_compact_next_tree_cpu(
    std::span<const std::uint32_t> anchor_root_token_ids,
    std::span<const DraftVocabularyCandidate> stream_rank_entries,
    std::uint32_t gamma,
    std::uint32_t branch_width,
    std::uint32_t output_capacity = arm_s_gamma4_k3_compact_node_max
);

enum class ArmSEarlyExitInterpretation : std::uint8_t {
    // The paper/code leave the projection algebra unresolved. This complete,
    // bias-free rank-8 Gemma transfer is never labeled source-exact.
    local_bias_free_rank8_projection,
};

struct ArmSEarlyExitMetalScratch {
    tensor::BackendBuffer rank8_states;
    tensor::BackendBuffer projected_target_states;
    tensor::BackendBuffer normalized_rows;
    tensor::BackendBuffer logits_tile;
    tensor::BackendBuffer logsumexp;
    tensor::BackendBuffer transition_scores;
    tensor::BackendBuffer status;
};

struct ArmSEarlyExitMetalResult;

class ArmSEarlyExitMetalPlan final {
public:
    ArmSEarlyExitMetalPlan(const ArmSEarlyExitMetalPlan&) = delete;
    ArmSEarlyExitMetalPlan& operator=(const ArmSEarlyExitMetalPlan&) = delete;
    ArmSEarlyExitMetalPlan(ArmSEarlyExitMetalPlan&&) noexcept = default;
    ArmSEarlyExitMetalPlan& operator=(ArmSEarlyExitMetalPlan&&) noexcept = default;

    [[nodiscard]] ArmSEarlyExitInterpretation interpretation() const noexcept;
    [[nodiscard]] std::uint32_t row_capacity() const noexcept;
    [[nodiscard]] std::size_t owned_weight_bytes() const noexcept;
    [[nodiscard]] std::size_t scratch_bytes() const noexcept;

private:
    friend ArmSEarlyExitMetalPlan create_arm_s_early_exit_metal_plan(
        const AuthenticatedSpeculativeArtifact&, const EngineModelPayloadReceipt&,
        const ResidentTargetPayloadReceipt&, const Gemma2MetalModelPlan&
    );
    friend struct ArmSEarlyExitMetalResult;
    friend ArmSEarlyExitMetalResult execute_arm_s_early_exit_metal(
        ArmSEarlyExitMetalPlan&, const Gemma2MetalModelPlan&,
        const tensor::BackendBuffer&, const tensor::BackendBuffer&,
        std::uint32_t
    );
    ArmSEarlyExitMetalPlan(
        tensor::BackendBuffer,
        tensor::BackendBuffer,
        ArmSEarlyExitMetalScratch,
        std::size_t,
        std::size_t
    );
    tensor::BackendBuffer down_;
    tensor::BackendBuffer up_;
    ArmSEarlyExitMetalScratch scratch_;
    std::size_t owned_weight_bytes_ = 0U;
    std::size_t scratch_bytes_ = 0U;
    std::array<float, arm_s_gamma4_k3_raw_edge_count> host_scores_ {};
};

struct ArmSEarlyExitMetalResult {
    std::span<const float> transition_probabilities;
    std::uint64_t gpu_duration_ns = 0U;
    std::uint64_t wall_duration_ns = 0U;
    std::uint32_t dispatch_count = 0U;
    std::uint32_t full_vocabulary_rows = 0U;
    std::size_t scratch_bytes = 0U;
};

[[nodiscard]] ArmSEarlyExitMetalPlan create_arm_s_early_exit_metal_plan(
    const AuthenticatedSpeculativeArtifact& artifact,
    const EngineModelPayloadReceipt& model_receipt,
    const ResidentTargetPayloadReceipt& resident_receipt,
    const Gemma2MetalModelPlan& target
);

[[nodiscard]] ArmSEarlyExitMetalResult execute_arm_s_early_exit_metal(
    ArmSEarlyExitMetalPlan& plan,
    const Gemma2MetalModelPlan& target,
    const tensor::BackendBuffer& parent_states,
    const tensor::BackendBuffer& child_token_ids,
    std::uint32_t edge_count
);

struct CorrectedVocabularyMetalScratch {
    tensor::BackendBuffer normalized_rows;
    tensor::BackendBuffer logits_tile;
    tensor::BackendBuffer candidate_runs_a;
    tensor::BackendBuffer candidate_runs_b;
    tensor::BackendBuffer logsumexp;
    tensor::BackendBuffer rank_entries;
    tensor::BackendBuffer status;
};

struct CorrectedVocabularyMetalResult;

class CorrectedVocabularyMetalPlan final {
public:
    CorrectedVocabularyMetalPlan(const CorrectedVocabularyMetalPlan&) = delete;
    CorrectedVocabularyMetalPlan& operator=(const CorrectedVocabularyMetalPlan&) = delete;
    CorrectedVocabularyMetalPlan(CorrectedVocabularyMetalPlan&&) noexcept = default;
    CorrectedVocabularyMetalPlan& operator=(CorrectedVocabularyMetalPlan&&) noexcept = default;

    [[nodiscard]] const CorrectedVocabularyWorkloadSpec& spec() const noexcept;
    [[nodiscard]] std::size_t scratch_bytes() const noexcept;

private:
    friend CorrectedVocabularyMetalPlan create_corrected_vocabulary_metal_plan(
        const Gemma2MetalModelPlan&, CorrectedVocabularyWorkload
    );
    friend struct CorrectedVocabularyMetalResult;
    friend CorrectedVocabularyMetalResult execute_corrected_vocabulary_metal(
        CorrectedVocabularyMetalPlan&, const Gemma2MetalModelPlan&,
        const tensor::BackendBuffer&, float
    );
    CorrectedVocabularyMetalPlan(
        CorrectedVocabularyWorkloadSpec,
        CorrectedVocabularyMetalScratch,
        std::size_t
    );
    CorrectedVocabularyWorkloadSpec spec_;
    CorrectedVocabularyMetalScratch scratch_;
    std::size_t scratch_bytes_ = 0U;
    std::vector<DraftVocabularyCandidate> host_rank_entries_;
};

struct CorrectedVocabularyMetalResult {
    std::span<const DraftVocabularyCandidate> rank_entries;
    std::uint64_t gpu_duration_ns = 0U;
    std::uint64_t wall_duration_ns = 0U;
    std::uint32_t dispatch_count = 0U;
    std::uint32_t command_buffer_count = 0U;
    std::uint32_t physical_head_rows = 0U;
    std::uint32_t rescanned_head_rows = 0U;
    std::uint64_t head_macs = 0U;
    std::uint64_t head_flops = 0U;
    std::uint64_t logical_head_weight_bytes = 0U;
    std::size_t scratch_bytes = 0U;
};

[[nodiscard]] CorrectedVocabularyMetalPlan create_corrected_vocabulary_metal_plan(
    const Gemma2MetalModelPlan& target,
    CorrectedVocabularyWorkload workload
);

[[nodiscard]] CorrectedVocabularyMetalResult execute_corrected_vocabulary_metal(
    CorrectedVocabularyMetalPlan& plan,
    const Gemma2MetalModelPlan& target,
    const tensor::BackendBuffer& corrected_states,
    float temperature = 1.0F
);

struct TreeFlashRankTableView {
    std::span<const DraftVocabularyCandidate> ranks;
    std::span<const std::int32_t> transition_next_indices;
    std::uint32_t depth_count = 0U;
    std::uint32_t conditioning_width = 0U;
    std::uint32_t rank_width = 0U;
};

struct TreeFlashTree {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> parent_indices;
    std::vector<std::uint16_t> depths;
};

[[nodiscard]] TreeFlashTree build_treeflash_tree_cpu(
    const TreeFlashRankTableView& table,
    std::uint32_t tree_size
);

}  // namespace local_gguf_engine::engine
