#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/speculative_controller.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::size_t verifier_layer_max = 64U;
inline constexpr std::size_t verifier_ancestor_entry_max =
    2U * VerifierRowMax * (external_speculation_max_horizon + 1U);
inline constexpr std::size_t verifier_tile_max = external_speculation_max_horizon;
inline constexpr std::size_t target_selection_batch_rows = 4U;
static_assert(VerifierRowMax == gemma2_metal_verifier_row_max);

enum class VerifierTopologyKind : std::uint8_t {
    unique_root_tree,
    repeated_root_beam_batch,
};

enum class VerifierInputResidency : std::uint8_t {
    host_staged,
    device_resident,
};

struct VerifierTile {
    std::uint16_t first_row = 0;
    std::uint16_t row_count = 0;
    std::uint32_t proposal_edge_count = 0;
};

struct VerifierRowAttention {
    std::uint32_t base_first_position = 0;
    std::uint32_t base_token_count = 0;
    std::uint32_t ancestor_offset = 0;
    std::uint16_t ancestor_count = 0;

    friend bool operator==(const VerifierRowAttention&, const VerifierRowAttention&) = default;
};

struct VerifierAttentionSchedule {
    std::array<VerifierRowAttention, VerifierRowMax> sliding_rows {};
    std::array<VerifierRowAttention, VerifierRowMax> global_rows {};
    std::array<std::uint32_t, verifier_ancestor_entry_max> ancestor_node_indices {};
    std::array<bool, verifier_layer_max> layer_uses_sliding_window {};
    std::array<std::uint32_t, verifier_layer_max * VerifierRowMax>
        per_layer_attention_counts {};
    std::uint32_t row_count = 0;
    std::uint32_t ancestor_entry_count = 0;
    std::uint32_t layer_count = 0;

    friend bool operator==(
        const VerifierAttentionSchedule&,
        const VerifierAttentionSchedule&
    ) = default;
};

struct VerifierTopologyDescriptor {
    VerifierTopologyKind kind = VerifierTopologyKind::unique_root_tree;
    ProposalGraphKind proposal_kind = ProposalGraphKind::chain;
    std::uint64_t graph_identity = 0;
    std::uint64_t device_arena_epoch = 0;
    std::uint32_t horizon = 0;
    std::uint32_t input_row_count = 0;
    std::uint32_t proposal_edge_count = 0;
    std::uint32_t materialized_edge_count = 0;
    std::uint32_t output_frontier_edge_count = 0;
    std::uint32_t root_count = 0;
    std::array<std::uint32_t, ReDrafterMLXRepeatedVerifierRootMax> root_indices {};
    std::uint32_t repeated_beam_width = 0;
    std::uint32_t repeated_beam_length = 0;
    // A conditional chain tile can begin below the source graph's root after
    // earlier tiles have published their accepted prefix to resident K/V.
    std::uint16_t root_source_depth = 0;
    bool has_source_budget = false;
    SourceTreeBudget source_budget;
    ResolvedTreeBudget resolved_budget;
    std::array<std::uint32_t, VerifierRowMax> token_ids {};
    std::array<std::uint32_t, VerifierRowMax> source_node_indices {};
    std::array<std::uint32_t, VerifierRowMax> parent_indices {};
    std::array<std::uint16_t, VerifierRowMax> depths {};
    std::array<std::uint16_t, VerifierRowMax> causal_history_counts {};
    std::array<std::uint32_t, VerifierRowMax> absolute_positions {};
    std::array<std::uint32_t, VerifierRowMax> component_roots {};
    std::array<ProposalRowOutputRole, VerifierRowMax> output_roles {};
    std::array<std::uint16_t, VerifierRowMax> selection_lanes {};
    MaterializedProposalEdgeArrays materialized_edges;
    OutputFrontierProposalEdgeArrays output_frontier_edges;
    std::array<VerifierTile, verifier_tile_max> tiles {};
    std::uint16_t tile_count = 0;
    VerifierAttentionSchedule attention;
};

struct VerifierDeviceInputBinding {
    tensor::BackendBuffer token_ids;
    tensor::BackendBuffer raw_selection_requests;
    std::uint32_t first_token = 0;
    std::uint32_t first_request = 0;
    std::uint64_t arena_epoch = 0;
};

// This all-u32 ABI is consumed directly by Metal. It intentionally mirrors
// DeviceSelectionRequest without depending on C++ aggregate padding.
struct VerifierDeviceSelectionRow {
    std::array<std::uint32_t, token_sampler_max_repetition_window> history_token_ids {};
    std::array<std::uint32_t, external_speculation_max_horizon> suffix_token_ids {};
    std::uint32_t history_head = 0;
    std::uint32_t history_length = 0;
    std::uint32_t history_window_limit = 0;
    std::uint32_t suffix_count = 0;
};

struct VerifierEncodeRequest {
    VerifierTopologyDescriptor topology;
    std::array<DeviceSelectionRequest, VerifierRowMax> selection_requests {};
    std::uint32_t selection_request_count = 0;
    TokenSamplerOptions sampler_options;
    VerifierInputResidency input_residency = VerifierInputResidency::host_staged;
    std::optional<VerifierDeviceInputBinding> device_inputs;
};

class ValidatedVerifierEncodeRequest final {
public:
    ValidatedVerifierEncodeRequest(const ValidatedVerifierEncodeRequest&) = delete;
    ValidatedVerifierEncodeRequest& operator=(const ValidatedVerifierEncodeRequest&) = delete;
    ValidatedVerifierEncodeRequest(ValidatedVerifierEncodeRequest&&) noexcept = default;
    ValidatedVerifierEncodeRequest& operator=(ValidatedVerifierEncodeRequest&&) noexcept = default;

    [[nodiscard]] const VerifierEncodeRequest& request() const noexcept;
    [[nodiscard]] bool validated_for(const Gemma2ModelConfig& target) const noexcept;

private:
    friend ValidatedVerifierEncodeRequest validate_verifier_encode_request(
        const VerifierEncodeRequest&,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t
    );
    explicit ValidatedVerifierEncodeRequest(
        const VerifierEncodeRequest& request,
        std::array<std::uint32_t, 5> validation_shape
    );
    // A validated request is an immutable, non-owning capability. Its request
    // must outlive the capability and the synchronous encode that consumes it.
    // The large fixed-capacity request therefore remains in caller-owned,
    // preallocated storage rather than being copied through stack frames or
    // retained by an asynchronous target transaction.
    const VerifierEncodeRequest* request_ = nullptr;
    // Vocabulary, layers, context capacity, local window, global-layer period.
    // Encoding binds these proof inputs to the actual target in constant time;
    // a proof for a larger context or different mask cannot authorize this one.
    std::array<std::uint32_t, 5> validation_shape_ {};
};

static_assert(sizeof(ValidatedVerifierEncodeRequest) <= sizeof(const void*) + 6U * sizeof(std::uint32_t));

[[nodiscard]] VerifierTopologyDescriptor make_verifier_topology_descriptor(
    const ValidatedProposalGraph& graph,
    const ChainVerifierSchedule& chain_schedule,
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period = 2U
);

[[nodiscard]] ValidatedVerifierEncodeRequest validate_verifier_encode_request(
    const VerifierEncodeRequest& request,
    std::uint32_t vocabulary_size,
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period = 2U
);

ValidatedVerifierEncodeRequest validate_verifier_encode_request(
    VerifierEncodeRequest&&,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t = 2U
) = delete;

// Produces the one causally eligible tile view of a validated full chain.
// Source-node identities and complete sampler histories remain global while
// private attention ancestry is rebased to the tile root.
void make_chain_verifier_tile_request(
    VerifierEncodeRequest& output,
    const ValidatedVerifierEncodeRequest& full_request,
    std::uint32_t tile_index,
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period = 2U
);

struct TargetSelectionWorkspace {
    tensor::BackendBuffer logits_tile;
    tensor::BackendBuffer candidate_runs_a;
    tensor::BackendBuffer candidate_runs_b;
    tensor::BackendBuffer compact_results;
    tensor::BackendBuffer raw_request_staging;
    tensor::BackendBuffer resolved_request_staging;
    tensor::BackendBuffer history_token_staging;
    std::uint32_t row_capacity = 0;
    std::uint32_t batch_row_capacity = 0;
};

struct Gemma2MetalTargetVerifierKVLayer {
    tensor::BackendBuffer private_key;
    tensor::BackendBuffer private_value;
    tensor::BackendBuffer accepted_backup_key;
    tensor::BackendBuffer accepted_backup_value;
};

struct Gemma2MetalTargetVerifierScratch {
    tensor::BackendBuffer token_id_staging;
    tensor::BackendBuffer expected_token_id_staging;
    tensor::BackendBuffer absolute_position_staging;
    tensor::BackendBuffer attention_schedule_staging;
    tensor::BackendBuffer ancestor_index_staging;
    tensor::BackendBuffer selected_path_staging;
    tensor::BackendBuffer input_status;
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
    // Reused after each existing projection barrier, bounded to scalar/H1 rows.
    std::optional<tensor::BackendBuffer> q4_input_sums;
    std::vector<Gemma2MetalTargetVerifierKVLayer> kv_layers;
    TargetSelectionWorkspace selection;
};

struct TargetVerifierRuntimeState;
struct TargetVerifierRuntimeAccess;
class Gemma2MetalTargetVerifierPlan final {
public:
    Gemma2MetalTargetVerifierPlan(
        Gemma2MetalTargetVerifierScratch scratch,
        std::uint32_t row_capacity,
        std::uint32_t layer_count,
        std::uint32_t vocabulary_size,
        std::uintptr_t target_weight_identity,
        std::size_t allocated_byte_count,
        bool precise_math = false
    );

    [[nodiscard]] const Gemma2MetalTargetVerifierScratch& scratch() const noexcept;
    [[nodiscard]] std::uint32_t row_capacity() const noexcept;
    [[nodiscard]] std::uint32_t layer_count() const noexcept;
    [[nodiscard]] std::uint32_t vocabulary_size() const noexcept;
    [[nodiscard]] std::uintptr_t target_weight_identity() const noexcept;
    [[nodiscard]] std::size_t allocated_byte_count() const noexcept;
    // Preallocated host mirror for compact per-row results. A completion view
    // remains valid while its target-only ticket owns the completed state.
    [[nodiscard]] std::span<CompactTokenCandidates> completion_candidates() noexcept;

private:
    friend struct TargetVerifierRuntimeAccess;
    std::shared_ptr<TargetVerifierRuntimeState> runtime_;
    Gemma2MetalTargetVerifierScratch scratch_;
    std::uint32_t row_capacity_ = 0;
    std::uint32_t layer_count_ = 0;
    std::uint32_t vocabulary_size_ = 0;
    std::uintptr_t target_weight_identity_ = 0;
    std::size_t allocated_byte_count_ = 0;
    std::vector<CompactTokenCandidates> completion_candidates_;
};

// The same descriptor ledger as the prepared arena; no device work is performed.
[[nodiscard]] std::size_t estimate_target_verifier_bytes(
    const Gemma2ModelConfig& config, std::uint32_t row_capacity);

[[nodiscard]] Gemma2MetalTargetVerifierPlan create_gemma2_metal_target_verifier_plan(
    const Gemma2MetalModelPlan& target,
    std::uint32_t row_capacity
);

struct TargetVerifierExecutionStats {
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint32_t selector_dispatch_count = 0;
    std::uint32_t compact_readback_bytes = 0;
    std::uint32_t input_row_count = 0;
    std::uint32_t proposal_edge_count = 0;
    std::uint32_t attention_dispatch_count = 0;
    std::uint32_t attention_partition_count = 0;
    std::uint64_t attention_token_count = 0;
    std::uint64_t gpu_duration_ns = 0;
    std::uint32_t q4_decode_projection_count = 0;
    std::uint32_t q8_decode_projection_count = 0;
    std::uint32_t q8_width_two_projection_count = 0;
    std::uint32_t q4_small_batch_projection_count = 0;
    std::uint32_t q4_width_two_projection_count = 0;
    std::uint32_t q4_exact_16_projection_count = 0;
    std::uint32_t q4_generic_tiled_projection_count = 0;
    std::uint32_t q6_decode_projection_count = 0;
    std::uint32_t q6_small_batch_projection_count = 0;
    std::uint32_t q6_width_two_projection_count = 0;
    std::uint32_t q6_exact_16_projection_count = 0;
    std::uint32_t q6_generic_tiled_projection_count = 0;
    std::uint32_t f16_projection_count = 0;
};

struct TargetVerifierCompletionResult {
    std::span<const CompactTokenCandidates> candidates;
    std::uint32_t candidate_row_count = 0;
    std::uint32_t cache_position = 0;
    TargetVerifierExecutionStats execution;
};

struct TargetVerifierCommitReceipt {
    std::array<std::uint32_t, external_speculation_max_horizon + 1U> retained_node_indices {};
    std::uint32_t retained_row_count = 0;
    TargetVerifierExecutionStats execution;
};

struct TargetVerifierEpisodeWorkspace {
    // This request is intentionally caller-owned and reusable. It is roughly
    // VerifierRowMax-shaped, so placing it inside a per-call stack frame would
    // both risk stack exhaustion and violate the steady-state allocation
    // contract.
    VerifierEncodeRequest tile_request;
    SpeculativeControllerScenario controller_scenario;
};

struct TargetVerifierEpisodeResult {
    SpeculativeControllerOutcome controller;
    TargetVerifierExecutionStats target_execution;
    Gemma2MetalDecodeExecutionStats endpoint_execution;
    Gemma2MetalDecodeExecutionStats ordinary_fallback_execution;
    CompactTokenCandidates next_candidates;
    CompactTokenCandidates ordinary_fallback_candidates;
    std::uint64_t controller_wall_ns = 0;
    std::uint64_t target_wall_ns = 0;
    std::uint64_t target_gpu_ns = 0;
    std::uint64_t commit_wall_ns = 0;
    std::uint64_t commit_gpu_ns = 0;
    std::uint64_t endpoint_wall_ns = 0;
    std::uint64_t ordinary_fallback_wall_ns = 0;
    std::uint32_t executed_tile_count = 0;
    std::uint32_t endpoint_count = 0;
    bool has_next_candidates = false;
    bool executed_ordinary_fallback = false;
};

class TargetVerificationTicket final {
public:
    TargetVerificationTicket(TargetVerificationTicket&&) noexcept;
    TargetVerificationTicket& operator=(TargetVerificationTicket&&) noexcept = delete;
    ~TargetVerificationTicket();

    TargetVerificationTicket(const TargetVerificationTicket&) = delete;
    TargetVerificationTicket& operator=(const TargetVerificationTicket&) = delete;

    [[nodiscard]] VerifierTransactionState state() const noexcept;
    [[nodiscard]] std::uint32_t input_row_count() const noexcept;
    [[nodiscard]] std::uint64_t graph_identity() const noexcept;

private:
    explicit TargetVerificationTicket(std::shared_ptr<TargetVerifierRuntimeState> impl);
    friend TargetVerificationTicket encode_gemma2_metal_target_verifier(
        Gemma2MetalModelPlan&,
        Gemma2MetalTargetVerifierPlan&,
        const ValidatedVerifierEncodeRequest&,
        float,
        FeatureSinkBinding,
        DraftCacheBinding
    );
    friend void submit_gemma2_metal_target_verifier(TargetVerificationTicket&);
    friend TargetVerifierCompletionResult complete_gemma2_metal_target_verifier(
        TargetVerificationTicket&
    );
    friend TargetVerifierCommitReceipt provisionally_commit_gemma2_metal_target_verifier(
        TargetVerificationTicket&,
        std::span<const std::uint32_t>
    );
    friend TargetVerifierCommitReceipt finalize_gemma2_metal_target_verifier(
        TargetVerificationTicket&,
        std::uint32_t
    );
    friend void discard_gemma2_metal_target_verifier(TargetVerificationTicket&);
    std::shared_ptr<TargetVerifierRuntimeState> impl_;
    std::uint64_t lease_epoch_ = 0;
};

[[nodiscard]] TargetVerificationTicket encode_gemma2_metal_target_verifier(
    Gemma2MetalModelPlan& target,
    Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& request,
    float embedding_scale,
    FeatureSinkBinding feature_sink = {},
    DraftCacheBinding draft_cache = {}
);

void submit_gemma2_metal_target_verifier(TargetVerificationTicket& ticket);

[[nodiscard]] TargetVerifierCompletionResult complete_gemma2_metal_target_verifier(
    TargetVerificationTicket& ticket
);

[[nodiscard]] TargetVerifierCommitReceipt provisionally_commit_gemma2_metal_target_verifier(
    TargetVerificationTicket& ticket,
    std::span<const std::uint32_t> selected_node_path
);

[[nodiscard]] TargetVerifierCommitReceipt finalize_gemma2_metal_target_verifier(
    TargetVerificationTicket& ticket,
    std::uint32_t retained_row_count
);

void discard_gemma2_metal_target_verifier(TargetVerificationTicket& ticket);

// Drives a validated chain through the exact controller one available target
// sample at a time. A later tile is encoded only after the controller reaches
// it; all preceding gate, context, stop, budget, callback, and mismatch
// decisions therefore suppress that Metal submission rather than merely
// discounting it after execution. The authoritative sampler must already be
// at scenario.origin_checkpoint (the checkpoint after selecting x0).
[[nodiscard]] TargetVerifierEpisodeResult
run_gemma2_metal_target_verifier_chain_episode(
    Gemma2MetalModelPlan& target,
    Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& full_request,
    const SpeculativeControllerScenario& scenario,
    TokenSampler& authoritative_sampler,
    TargetVerifierEpisodeWorkspace& workspace,
    float embedding_scale
);

// Single-submission prefix-closed forest/repeated-root counterpart. Sampling
// walks only the target-selected causal path; unused leaf rows never consume a
// sampler ticket, and only that path can reach resident K/V.
[[nodiscard]] TargetVerifierEpisodeResult
run_gemma2_metal_target_verifier_forest_episode(
    Gemma2MetalModelPlan& target,
    Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& full_request,
    const SpeculativeControllerScenario& scenario,
    TokenSampler& authoritative_sampler,
    TargetVerifierEpisodeWorkspace& workspace,
    float embedding_scale
);

// Caches an authoritative terminal token that was selected from a verifier
// row but was not itself among the verifier inputs. This is the single target
// endpoint to which Phase 3 attaches feature and draft-cache maintenance.
[[nodiscard]] Gemma2MetalDecodeExecutionStats run_gemma2_metal_target_verifier_endpoint(
    Gemma2MetalModelPlan& target,
    std::uint32_t token_id,
    std::uint32_t cache_position,
    float embedding_scale
);

}  // namespace local_gguf_engine::engine
