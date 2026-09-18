#pragma once

#include "local_gguf_engine/engine/sampling.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace local_gguf_engine::engine {

inline constexpr std::size_t StreamRawPriorNodeMax = 256;
inline constexpr std::size_t StreamSurvivorMax = 256;
inline constexpr std::size_t StreamUpperRowMax = 6'144;
inline constexpr std::size_t StreamTransitionBranchMax = 8;
inline constexpr std::size_t SNextNodeMax = 65'536;
inline constexpr std::size_t SRawChildMax = 47'104;
inline constexpr std::size_t ExternalTreeDraftNodeMax = 1'024;
inline constexpr std::size_t VerifierRowMax = 1'025;
inline constexpr std::size_t MaterializedTreeEdgeMax = 1'024;
inline constexpr std::size_t MaterializedDegreeMax = 1'024;
inline constexpr std::size_t OutputFrontierBranchMax = 32;
inline constexpr std::size_t OutputFrontierEdgeMax = 32'800;
inline constexpr std::size_t BestFirstCandidateRankMax = 1'024;
inline constexpr std::size_t DDTreeHeapLiveMax = 1'025;
inline constexpr std::size_t CaDDTreeHeapLiveMax = 1'025;
inline constexpr std::size_t TreeFlashCursorHeapLiveMax = 513;
inline constexpr std::size_t JetSpecHeapLiveMax = 255;
inline constexpr std::size_t FullPathJetSpecHeapLiveMax = 1'024;
inline constexpr std::size_t FullPathDDTreeHeapLiveMax = 1'025;
inline constexpr std::size_t ReDrafterRawBeamEntryMax = 225;
inline constexpr std::size_t ReDrafterRectangularKVRowMax = 225;
inline constexpr std::size_t ReDrafterMLXPhysicalRowMax = 16;
inline constexpr std::size_t ReDrafterMLXRepeatedVerifierRootMax = 4;
inline constexpr std::size_t ReDrafterPyRawRootOccurrenceMax = 45;
inline constexpr std::size_t ReDrafterPyUnpackedCacheRowMax = 225;
inline constexpr std::size_t ReDrafterPyTreeRowMax = 181;
inline constexpr std::size_t EAGLE3RawCandidateNodeMax = 710;
inline constexpr std::size_t EAGLE3RawSupertreeRowMax = 711;
inline constexpr std::size_t EAGLE3ScoreParentRowMax = 71;
inline constexpr std::size_t EAGLE3ExpansionActiveMax = 10;
inline constexpr std::size_t EAGLE3ExpansionForwardRowMax = 70;
inline constexpr std::size_t EAGLE3ExpansionMaskKeyMax = 80;
inline constexpr std::size_t EAGLE3RetrievePathRowMax = 9;
inline constexpr std::size_t EAGLE3VerifierRowMax = 60;
inline constexpr std::size_t DBlastCategoryTrieRowMax = 121;
inline constexpr std::size_t PCTreePoolMax = 969;
inline constexpr std::size_t TreeFlashReleasedCorrectionStateMax = 225;
inline constexpr std::size_t TreeFlashCodeTransferCorrectionStateMax = 897;
inline constexpr std::size_t TreeFlashSemanticCorrectionStateMax = 961;
inline constexpr std::size_t TreeFlashReleasedPhysicalCorrectionRowMax = 240;
inline constexpr std::size_t TreeFlashCodePhysicalCorrectionRowMax = 960;
inline constexpr std::size_t TreeFlashH16PhysicalCorrectionRowMax = 1'024;
inline constexpr std::size_t DARTreeSupertreeMax = 512;
inline constexpr std::size_t DARTreeScoredPerDepthMax = 2'048;
inline constexpr std::size_t DARTreeScoreEventMax = 30'784;
inline constexpr std::size_t TreeFlashReleasedRankEntryMax = 3'840;
inline constexpr std::size_t TreeFlashCodeTop16RankEntryMax = 15'360;
inline constexpr std::size_t TreeFlashCodeTop64RankEntryMax = 61'440;
inline constexpr std::size_t TreeFlashH15FullVRankEntryMax = 491'520;
inline constexpr std::size_t TreeFlashH16FullVRankEntryMax = 492'032;
inline constexpr std::size_t TreeFlashH16CodeFullVRankEntryMax = 524'288;
inline constexpr std::size_t DominoGeneratedChildMax = 512;
inline constexpr std::size_t DominoHeapLiveMax = 449;
inline constexpr std::size_t DominoFrontierActiveMax = 64;
inline constexpr std::size_t DominoConditionalScoreTileMax = 16'384;
inline constexpr std::size_t DominoFrontierGeneratedPerDepthMax = 512;
inline constexpr std::size_t DominoFrontierGeneratedLedgerMax = 7'680;
inline constexpr std::size_t DominoCorrectionLaneStepMax = 960;
inline constexpr std::size_t FullPathSupertreeMax = 1'025;
inline constexpr std::size_t FullPathScaffoldEdgeMax = 1'024;
inline constexpr std::size_t ConditionalExplicitMenuMax = 256;
inline constexpr std::size_t DominoTreeBranchMax = 8;
inline constexpr std::size_t DARTreeCandidateWidth = 64;
inline constexpr std::size_t DARTreeLayerWidthMax = 32;
inline constexpr std::size_t JetSpecBranchMax = 7;
inline constexpr std::size_t ReportedSharedTable3PostPruneDraftNodeCount = 32;

inline constexpr std::uint32_t proposal_node_none = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint16_t proposal_ticket_none = std::numeric_limits<std::uint16_t>::max();
inline constexpr std::size_t controller_max_selected_tokens = external_speculation_max_horizon + 1U;
inline constexpr std::size_t controller_max_trace_events = 256;
inline constexpr std::size_t controller_max_chain_tiles = external_speculation_max_horizon;
inline constexpr std::size_t controller_max_conditional_costs = 256;

enum class ProposalGraphKind : std::uint8_t {
    chain,
    forest,
    repeated_root_beam_batch,
};

enum class ProposalRowOutputRole : std::uint8_t {
    verifies_proposals,
    correction_candidate,
    post_full_candidate,
};

enum class ProposalFeatureSource : std::uint8_t {
    target_hidden,
    draft_hidden,
    recurrent_state,
    latent_state,
    target_integrated_stream,
    retrieval,
};

enum class ProposalPrimaryPolicy : std::uint8_t {
    scalar_common_draw,
    deterministic,
    greedy,
};

enum class ProposalSiblingPolicy : std::uint8_t {
    deterministic_stable_top_k,
    private_domain_separated_rng,
};

enum class SourceRootConvention : std::uint8_t {
    excluded,
    included,
};

struct SourceTreeBudget {
    std::uint32_t value = 0;
    SourceRootConvention root_convention = SourceRootConvention::excluded;
};

struct ResolvedTreeBudget {
    std::uint32_t draft_node_cap = 0;
    std::uint32_t input_row_cap = 0;
};

[[nodiscard]] ResolvedTreeBudget resolve_source_tree_budget(SourceTreeBudget source);

struct ProposalPathHistoryDescriptor {
    std::uint64_t digest = 0;
    std::uint32_t length = 0;
};

struct ProposalFeatureProvenance {
    ProposalFeatureSource source = ProposalFeatureSource::target_hidden;
    std::uint16_t source_layer = 0;
    std::uint64_t arena_epoch = 0;
};

struct ProposalGraphHeader {
    ProposalGraphKind kind = ProposalGraphKind::chain;
    std::uint32_t horizon = 0;
    std::uint32_t materialized_node_count = 0;
    std::uint32_t materialized_edge_count = 0;
    std::uint32_t output_frontier_edge_count = 0;
    std::uint32_t root_count = 0;
    std::array<std::uint32_t, ReDrafterMLXRepeatedVerifierRootMax> root_indices {};
    std::uint32_t repeated_beam_width = 0;
    std::uint32_t repeated_beam_length = 0;
    std::uint64_t graph_identity = 0;
    std::uint64_t device_arena_epoch = 0;
    ProposalPrimaryPolicy primary_policy = ProposalPrimaryPolicy::scalar_common_draw;
    ProposalSiblingPolicy sibling_policy = ProposalSiblingPolicy::deterministic_stable_top_k;
    bool has_source_budget = false;
    SourceTreeBudget source_budget;
    ResolvedTreeBudget resolved_budget;
};

struct ProposalNodeArrays {
    std::array<std::uint32_t, VerifierRowMax> token_ids {};
    std::array<std::uint32_t, VerifierRowMax> parent_indices {};
    std::array<std::uint16_t, VerifierRowMax> depths {};
    std::array<std::uint32_t, VerifierRowMax> absolute_positions {};
    std::array<std::uint32_t, VerifierRowMax> mask_component_roots {};
    std::array<std::uint16_t, VerifierRowMax> mask_ancestor_counts {};
    std::array<float, VerifierRowMax> conditional_confidences {};
    std::array<ProposalPathHistoryDescriptor, VerifierRowMax> path_histories {};
    std::array<std::uint16_t, VerifierRowMax> proposal_ticket_offsets {};
    std::array<ProposalFeatureProvenance, VerifierRowMax> feature_provenance {};
    std::array<ProposalRowOutputRole, VerifierRowMax> output_roles {};
};

struct MaterializedProposalEdgeArrays {
    std::array<std::uint32_t, MaterializedTreeEdgeMax> parent_node_indices {};
    std::array<std::uint32_t, MaterializedTreeEdgeMax> child_node_indices {};
    std::array<std::uint32_t, MaterializedTreeEdgeMax> token_ids {};
    std::array<float, MaterializedTreeEdgeMax> conditional_confidences {};
    std::array<std::uint16_t, MaterializedTreeEdgeMax> sibling_ranks {};
    std::array<std::uint16_t, MaterializedTreeEdgeMax> proposal_ticket_offsets {};
};

struct OutputFrontierProposalEdgeArrays {
    std::array<std::uint32_t, OutputFrontierEdgeMax> parent_node_indices {};
    std::array<std::uint32_t, OutputFrontierEdgeMax> token_ids {};
    std::array<float, OutputFrontierEdgeMax> conditional_confidences {};
    std::array<std::uint16_t, OutputFrontierEdgeMax> sibling_ranks {};
    std::array<std::uint16_t, OutputFrontierEdgeMax> proposal_ticket_offsets {};
};

struct ProposalGraphStorage {
    ProposalGraphHeader header;
    ProposalNodeArrays nodes;
    MaterializedProposalEdgeArrays materialized_edges;
    OutputFrontierProposalEdgeArrays output_frontier_edges;
};

struct ProposalGraphValidationContext {
    std::uint32_t vocabulary_size = 0;
    std::uint32_t context_length = 0;
    std::uint32_t expected_root_token = 0;
    std::uint32_t expected_root_position = 0;
    ProposalPathHistoryDescriptor expected_root_history;
    std::uint64_t expected_device_arena_epoch = 0;
};

class ValidatedProposalGraph final {
public:
    ValidatedProposalGraph(const ValidatedProposalGraph&) = delete;
    ValidatedProposalGraph& operator=(const ValidatedProposalGraph&) = delete;
    ValidatedProposalGraph(ValidatedProposalGraph&&) noexcept = default;
    ValidatedProposalGraph& operator=(ValidatedProposalGraph&&) noexcept = default;

    [[nodiscard]] const ProposalGraphStorage& storage() const noexcept;

private:
    friend ValidatedProposalGraph validate_proposal_graph(
        ProposalGraphStorage,
        const ProposalGraphValidationContext&
    );
    explicit ValidatedProposalGraph(ProposalGraphStorage storage);

    ProposalGraphStorage storage_;
};

[[nodiscard]] ValidatedProposalGraph validate_proposal_graph(
    ProposalGraphStorage storage,
    const ProposalGraphValidationContext& context
);

enum class ChainTileEndpoint : std::uint8_t {
    terminal_carry,
    post_full_candidate,
};

struct ChainVerifierTile {
    std::uint16_t validated_edge_count = 0;
    std::uint16_t input_row_count = 0;
};

struct ChainVerifierSchedule {
    std::array<ChainVerifierTile, controller_max_chain_tiles> tiles {};
    std::uint16_t tile_count = 0;
    ChainTileEndpoint final_endpoint = ChainTileEndpoint::post_full_candidate;
};

void validate_chain_verifier_schedule(
    const ChainVerifierSchedule& schedule,
    std::uint32_t horizon
);

// Engine-registered causal partitions for the external H=3/4/7/8/15/16/23
// frontier. Parts are proposal-edge counts; only the final post-full tile adds
// the candidate row that caches dH.
[[nodiscard]] ChainVerifierSchedule make_registered_conditional_chain_schedule(
    std::uint32_t horizon,
    ChainTileEndpoint final_endpoint = ChainTileEndpoint::post_full_candidate
);

enum class SpeculativeProposalSource : std::uint8_t {
    none,
    retrieval,
    learned_external,
    target_integrated_stream,
};

enum class SpeculativeDraftSchedule : std::uint8_t {
    eager,
    conditional_tiles,
    dynamic_horizon,
    target_integrated,
};

enum class SpeculativeRouteMode : std::uint8_t {
    retrieval_first,
    retrieval_anchored_learned,
    learned_only,
    target_integrated,
};

enum class TargetVerifierTopology : std::uint8_t {
    conditional_tile_cascade,
    single_invocation_terminal_carry,
    single_invocation_post_full,
    prefix_closed_forest,
    repeated_root_beam_batch,
    target_integrated_forest,
};

enum class FirstCallCommandTopology : std::uint8_t {
    fused_eager,
    host_gated,
    gpu_gated,
    target_integrated,
};

enum class ControllerCallbackDecision : std::uint8_t {
    continue_generation,
    stop_generation,
    throw_exception,
};

enum class ControllerTileGateDecision : std::uint8_t {
    accept,
    decline,
};

enum class ControllerDisposition : std::uint8_t {
    completed,
    carry_queued,
    ordinary_fallback,
    stopped,
    callback_threw,
};

enum class ControllerStopReason : std::uint8_t {
    none,
    stop_token,
    selected_token_budget,
    callback_stop,
    callback_throw,
    ineligible,
    context_exhausted,
    tile_gate_decline,
    second_call_confidence_decline,
    target_mismatch,
    terminal_carry,
    full_match,
    materialized_leaf,
};

enum class TargetStateLeaseSource : std::uint8_t {
    none,
    ordinary_decode,
    selected_verifier_row,
    target_integrated_main_row,
};

struct TargetStateLeaseDescriptor {
    TargetStateLeaseSource source = TargetStateLeaseSource::none;
    std::uint64_t owner_epoch = 0;
    std::uint32_t absolute_position = 0;
    std::uint32_t node_index = proposal_node_none;
};

struct ControllerTargetSample {
    std::uint32_t token_id = 0;
    SamplerCheckpoint checkpoint;
};

struct ControllerCallbackRecord {
    std::uint32_t token_id = 0;
    std::uint32_t logical_selected_count = 0;
    std::uint32_t logical_cached_count = 0;
    ControllerCallbackDecision decision = ControllerCallbackDecision::continue_generation;
};

enum class ControllerCostKind : std::uint8_t {
    none,
    draft_execution,
    target_verifier,
    target_selection,
    synchronization,
    readback,
    tile_gate,
    callback,
    cache_finalize,
    cache_restore,
    ordinary_fallback,
    tree_publish,
};

struct ConditionalCostKey {
    ControllerCostKind kind = ControllerCostKind::none;
    ProposalGraphKind topology = ProposalGraphKind::chain;
    FirstCallCommandTopology command_topology = FirstCallCommandTopology::host_gated;
    std::uint16_t shape = 0;
    std::uint16_t outcome = 0;

    friend bool operator==(const ConditionalCostKey&, const ConditionalCostKey&) = default;
};

enum class ControllerTraceEventKind : std::uint8_t {
    draft_execution,
    verifier_tile,
    synchronization,
    candidate_readback,
    target_preview,
    callback,
    checkpoint_adopt,
    retain_row,
    discard_preview,
    endpoint_kv_only,
    tile_gate,
    carry,
    next_candidate,
    complete,
};

struct ControllerTraceEvent {
    ControllerTraceEventKind kind = ControllerTraceEventKind::complete;
    std::uint16_t depth = 0;
    std::uint16_t tile = 0;
    std::uint32_t token_id = 0;
    std::uint32_t node_index = proposal_node_none;
    std::uint32_t logical_selected_count = 0;
    std::uint32_t logical_cached_count = 0;
    ConditionalCostKey cost;
};

struct SpeculativeControllerScenario {
    const ValidatedProposalGraph* graph = nullptr;
    ChainVerifierSchedule chain_schedule;
    SpeculativeProposalSource proposal_source = SpeculativeProposalSource::none;
    SpeculativeDraftSchedule draft_schedule = SpeculativeDraftSchedule::eager;
    SpeculativeRouteMode route_mode = SpeculativeRouteMode::learned_only;
    TargetVerifierTopology verifier_topology = TargetVerifierTopology::single_invocation_post_full;
    FirstCallCommandTopology first_call_topology = FirstCallCommandTopology::host_gated;
    bool eligible = true;
    std::uint32_t origin_token = 0;
    SamplerCheckpoint origin_checkpoint;
    std::uint32_t remaining_selected_token_budget = 0;
    std::uint32_t initial_logical_cached_count = 0;
    std::uint32_t available_context_rows = 0;
    std::array<ControllerTargetSample, external_speculation_max_horizon> target_samples {};
    std::uint32_t target_sample_count = 0;
    std::array<bool, controller_max_selected_tokens> token_is_stop {};
    std::array<ControllerCallbackDecision, controller_max_selected_tokens> callbacks {};
    // Scripted goldens provide every decision. Live adapters start at zero
    // and resume only after the actual callback has returned.
    std::uint32_t callback_decision_count = controller_max_selected_tokens;
    std::array<ControllerTileGateDecision, controller_max_chain_tiles> tile_gates {};
};

struct SpeculativeControllerOutcome {
    ControllerDisposition disposition = ControllerDisposition::completed;
    ControllerStopReason stop_reason = ControllerStopReason::none;
    SpeculativeProposalSource proposal_source = SpeculativeProposalSource::none;
    SpeculativeRouteMode route_mode = SpeculativeRouteMode::learned_only;
    TargetVerifierTopology verifier_topology = TargetVerifierTopology::single_invocation_post_full;
    FirstCallCommandTopology first_call_topology = FirstCallCommandTopology::host_gated;
    ProposalPrimaryPolicy primary_policy = ProposalPrimaryPolicy::scalar_common_draw;
    ProposalSiblingPolicy sibling_policy = ProposalSiblingPolicy::deterministic_stable_top_k;
    std::array<std::uint32_t, controller_max_selected_tokens> selected_tokens {};
    std::array<std::uint32_t, controller_max_selected_tokens> selected_node_path {};
    std::uint32_t selected_count = 0;
    std::uint32_t accepted_edge_count = 0;
    std::uint32_t sampled_preview_count = 0;
    std::array<std::uint32_t, controller_max_selected_tokens> retained_row_nodes {};
    std::uint32_t retained_row_count = 0;
    std::array<std::uint32_t, controller_max_selected_tokens> endpoint_kv_only_tokens {};
    std::uint32_t endpoint_kv_only_count = 0;
    std::array<ControllerCallbackRecord, controller_max_selected_tokens> callback_records {};
    std::uint32_t callback_count = 0;
    std::uint32_t logical_cached_count = 0;
    SamplerCheckpoint adopted_checkpoint;
    bool has_carry = false;
    std::uint32_t carry_token = 0;
    SamplerCheckpoint carry_checkpoint;
    bool carry_is_cached = false;
    TargetStateLeaseDescriptor next_candidate_lease;
    // run_available() returns at the exact point where target execution must
    // supply the next authoritative sample. No later tile/controller action
    // has occurred when this is set.
    bool requires_target_sample = false;
    std::uint32_t required_target_sample_offset = 0;
    std::uint16_t required_target_sample_tile = 0;
    bool requires_callback = false;
    std::uint32_t required_callback_depth = 0;
    std::array<ControllerTraceEvent, controller_max_trace_events> trace {};
    std::uint32_t trace_count = 0;
};

class BoundedSpeculativeController final {
public:
    [[nodiscard]] SpeculativeControllerOutcome run(
        const SpeculativeControllerScenario& scenario
    ) const;

    // Executes every decision for which target samples are already present,
    // then pauses before consuming an unavailable sample. This is the same
    // state machine as run(); it exists so the engine can launch conditional
    // verifier tiles only when their first sample is causally reachable.
    [[nodiscard]] SpeculativeControllerOutcome run_available(
        const SpeculativeControllerScenario& scenario
    ) const;

private:
    [[nodiscard]] SpeculativeControllerOutcome run_impl(
        const SpeculativeControllerScenario& scenario,
        bool pause_before_missing_target_sample
    ) const;
};

struct MeasuredConditionalCost {
    ConditionalCostKey key;
    double duration_ns = 0.0;
};

struct MeasuredConditionalCostTable {
    std::array<MeasuredConditionalCost, controller_max_conditional_costs> values {};
    std::uint32_t count = 0;
};

struct CorrectedTrajectoryCost {
    double duration_ns = 0.0;
    std::uint32_t charged_event_count = 0;
};

[[nodiscard]] CorrectedTrajectoryCost replay_corrected_trajectory_cost(
    const SpeculativeControllerOutcome& trajectory,
    const MeasuredConditionalCostTable& costs
);

[[nodiscard]] double diagnostic_iid_expected_tokens(double acceptance, std::uint32_t horizon);

struct StreamTransitionShape {
    std::uint16_t gamma = 0;
    std::uint16_t branch_count = 0;
    std::uint32_t raw_prior_node_count = 0;
    std::uint32_t survivor_count = 0;
    std::uint32_t upper_row_count = 0;
    std::uint32_t next_node_count = 0;
    std::uint32_t raw_child_count = 0;
};

struct StreamTransitionShapeTable {
    std::array<StreamTransitionShape, StreamRawPriorNodeMax + 1U> values {};
    std::uint16_t count = 0;
};

[[nodiscard]] StreamTransitionShape make_stream_transition_shape(
    std::uint32_t gamma,
    std::uint32_t branch_count,
    std::uint32_t survivor_count
);

[[nodiscard]] StreamTransitionShapeTable generate_registered_stream_transition_table(
    std::uint32_t gamma,
    std::uint32_t branch_count
);

struct StreamAnchorTicketSlice {
    std::uint16_t accepted_depth = 0;
    std::uint16_t first_ticket_offset = 0;
    std::uint16_t ticket_count = 0;
};

struct StreamAnchorTicketMap {
    std::array<StreamAnchorTicketSlice, stream_speculation_max_horizon + 1U> values {};
    std::uint16_t count = 0;
    std::uint16_t pool_size = 0;
};

[[nodiscard]] StreamAnchorTicketMap generate_stream_anchor_ticket_map(std::uint32_t gamma);

enum class StreamLeasePhase : std::uint8_t {
    unavailable,
    pending_bootstrap,
    pending_root_bind,
    steady,
    pending_correction_bind,
    draining,
    poisoned,
};

enum class StreamLeaseEventKind : std::uint8_t {
    begin_bootstrap,
    bootstrap_completed,
    bind_selected_root,
    verification_mismatch,
    verification_full_match,
    bind_correction,
    publish_next_tree,
    discard_next_tree,
    intentional_switch,
    endpoint,
    begin_rebase,
    drain_completed,
    poison,
    authenticated_reset,
};

enum class NextTreeAction : std::uint8_t {
    none,
    publish,
    discard,
    rebase,
};

struct StreamTreeLeaseDescriptor {
    std::uint64_t lease_identity = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t arena_epoch = 0;
    std::uint64_t graph_identity = 0;
    std::uint64_t history_digest = 0;
    std::uint32_t root_token = 0;
    std::uint32_t root_position = 0;
};

struct StreamLeaseState {
    StreamLeasePhase phase = StreamLeasePhase::unavailable;
    std::uint64_t session_epoch = 0;
    bool has_prior_tree = false;
    StreamTreeLeaseDescriptor prior_tree;
    bool has_pending_tree = false;
    StreamTreeLeaseDescriptor pending_tree;
};

struct StreamLeaseEvent {
    StreamLeaseEventKind kind = StreamLeaseEventKind::begin_bootstrap;
    bool has_tree = false;
    StreamTreeLeaseDescriptor tree;
};

struct StreamLeaseTransition {
    StreamLeaseState next_state;
    NextTreeAction tree_action = NextTreeAction::none;
};

[[nodiscard]] StreamLeaseTransition transition_stream_lease(
    const StreamLeaseState& state,
    const StreamLeaseEvent& event
);

struct RegisteredHorizonTransition {
    std::uint16_t accepted_depth = 0;
    std::uint16_t target_ticket_offset = 0;
    std::uint16_t selected_token_count = 0;
};

struct RegisteredHorizonTransitionTable {
    std::array<RegisteredHorizonTransition, controller_max_selected_tokens> values {};
    std::uint16_t count = 0;
};

[[nodiscard]] RegisteredHorizonTransitionTable generate_registered_horizon_transition_table(
    std::uint32_t horizon
);

}  // namespace local_gguf_engine::engine
