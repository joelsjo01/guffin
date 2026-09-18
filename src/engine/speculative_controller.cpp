#include "local_gguf_engine/engine/speculative_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

static_assert(StreamUpperRowMax == StreamSurvivorMax * (stream_speculation_max_horizon + 1U));
static_assert(SNextNodeMax == StreamSurvivorMax * StreamRawPriorNodeMax);
static_assert(
    SRawChildMax
    == StreamSurvivorMax * stream_speculation_max_horizon * StreamTransitionBranchMax
);
static_assert(OutputFrontierEdgeMax == VerifierRowMax * OutputFrontierBranchMax);

namespace {

[[nodiscard]] bool valid_confidence(float value) noexcept {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
}

[[nodiscard]] bool same_history_descriptor(
    const ProposalPathHistoryDescriptor& left,
    const ProposalPathHistoryDescriptor& right
) noexcept {
    return left.digest == right.digest && left.length == right.length;
}

[[nodiscard]] bool is_registered_horizon(std::uint32_t horizon) noexcept {
    for (const auto registered : {3U, 4U, 7U, 8U, 15U, 16U, 23U}) {
        if (horizon == registered) return true;
    }
    return false;
}

[[nodiscard]] bool is_registered_stream_geometry(
    std::uint32_t gamma,
    std::uint32_t branch_count
) noexcept {
    return (gamma == 4U && (branch_count == 1U || branch_count == 2U || branch_count == 3U))
        || (gamma == 3U && branch_count == 3U)
        || ((gamma == 7U || gamma == 15U || gamma == 23U) && branch_count == 1U);
}

struct EdgeMatch {
    bool matched = false;
    bool materialized = false;
    std::uint32_t child_node = proposal_node_none;
};

[[nodiscard]] EdgeMatch find_edge_match(
    const ProposalGraphStorage& graph,
    std::uint32_t parent,
    std::uint32_t token
) noexcept {
    for (std::size_t index = 0; index < graph.header.materialized_edge_count; ++index) {
        if (graph.materialized_edges.parent_node_indices[index] == parent
            && graph.materialized_edges.token_ids[index] == token) {
            return EdgeMatch {
                .matched = true,
                .materialized = true,
                .child_node = graph.materialized_edges.child_node_indices[index],
            };
        }
    }
    for (std::size_t index = 0; index < graph.header.output_frontier_edge_count; ++index) {
        if (graph.output_frontier_edges.parent_node_indices[index] == parent
            && graph.output_frontier_edges.token_ids[index] == token) {
            return EdgeMatch {.matched = true};
        }
    }
    return {};
}

[[nodiscard]] bool node_has_outgoing_edge(
    const ProposalGraphStorage& graph,
    std::uint32_t parent
) noexcept {
    for (std::size_t index = 0; index < graph.header.materialized_edge_count; ++index) {
        if (graph.materialized_edges.parent_node_indices[index] == parent) return true;
    }
    for (std::size_t index = 0; index < graph.header.output_frontier_edge_count; ++index) {
        if (graph.output_frontier_edges.parent_node_indices[index] == parent) return true;
    }
    return false;
}

class OutcomeWriter final {
public:
    OutcomeWriter(
        SpeculativeControllerOutcome& outcome,
        const SpeculativeControllerScenario& scenario,
        ProposalGraphKind topology
    ) : outcome_(outcome), scenario_(scenario), topology_(topology) {}

    void trace(
        ControllerTraceEventKind kind,
        std::uint32_t depth,
        std::uint32_t tile,
        std::uint32_t token,
        std::uint32_t node,
        ConditionalCostKey cost = {}
    ) {
        if (outcome_.trace_count >= outcome_.trace.size()) {
            throw std::overflow_error("speculative controller trace capacity exceeded");
        }
        outcome_.trace[outcome_.trace_count++] = ControllerTraceEvent {
            .kind = kind,
            .depth = static_cast<std::uint16_t>(depth),
            .tile = static_cast<std::uint16_t>(tile),
            .token_id = token,
            .node_index = node,
            .logical_selected_count = outcome_.selected_count,
            .logical_cached_count = outcome_.logical_cached_count,
            .cost = cost,
        };
    }

    void select(
        std::uint32_t token,
        std::uint32_t node,
        const SamplerCheckpoint& checkpoint
    ) {
        if (outcome_.selected_count >= outcome_.selected_tokens.size()) {
            throw std::overflow_error("speculative selected-token capacity exceeded");
        }
        outcome_.selected_tokens[outcome_.selected_count] = token;
        outcome_.selected_node_path[outcome_.selected_count] = node;
        ++outcome_.selected_count;
        outcome_.adopted_checkpoint = checkpoint;
        trace(
            ControllerTraceEventKind::checkpoint_adopt,
            outcome_.selected_count - 1U,
            0U,
            token,
            node
        );
    }

    void retain(std::uint32_t token, std::uint32_t node, std::uint32_t depth, std::uint32_t tile) {
        for (std::size_t index = 0; index < outcome_.retained_row_count; ++index) {
            if (outcome_.retained_row_nodes[index] == node) return;
        }
        if (outcome_.retained_row_count >= outcome_.retained_row_nodes.size()) {
            throw std::overflow_error("speculative retained-row path capacity exceeded");
        }
        outcome_.retained_row_nodes[outcome_.retained_row_count++] = node;
        ++outcome_.logical_cached_count;
        trace(
            ControllerTraceEventKind::retain_row,
            depth,
            tile,
            token,
            node,
            ConditionalCostKey {
                .kind = ControllerCostKind::cache_finalize,
                .topology = topology_,
                .command_topology = scenario_.first_call_topology,
                .shape = 1U,
                .outcome = 1U,
            }
        );
    }

    void endpoint_kv(std::uint32_t token, std::uint32_t depth) {
        if (outcome_.endpoint_kv_only_count >= outcome_.endpoint_kv_only_tokens.size()) {
            throw std::overflow_error("speculative endpoint K/V capacity exceeded");
        }
        outcome_.endpoint_kv_only_tokens[outcome_.endpoint_kv_only_count++] = token;
        ++outcome_.logical_cached_count;
        trace(
            ControllerTraceEventKind::endpoint_kv_only,
            depth,
            0U,
            token,
            proposal_node_none,
            ConditionalCostKey {
                .kind = ControllerCostKind::cache_finalize,
                .topology = topology_,
                .command_topology = scenario_.first_call_topology,
                .shape = 1U,
                .outcome = 2U,
            }
        );
    }

    [[nodiscard]] bool callback(std::uint32_t depth, std::uint32_t token) {
        if (scenario_.token_is_stop[depth]) return true;
        if (depth >= scenario_.callback_decision_count) {
            outcome_.requires_callback = true;
            outcome_.required_callback_depth = depth;
            return false;
        }
        if (outcome_.callback_count >= outcome_.callback_records.size()) {
            throw std::overflow_error("speculative callback record capacity exceeded");
        }
        const auto decision = scenario_.callbacks[depth];
        outcome_.callback_records[outcome_.callback_count++] = ControllerCallbackRecord {
            .token_id = token,
            .logical_selected_count = outcome_.selected_count,
            .logical_cached_count = outcome_.logical_cached_count,
            .decision = decision,
        };
        trace(
            ControllerTraceEventKind::callback,
            depth,
            0U,
            token,
            outcome_.selected_node_path[depth],
            ConditionalCostKey {
                .kind = ControllerCostKind::callback,
                .topology = topology_,
                .command_topology = scenario_.first_call_topology,
                .shape = 1U,
                .outcome = static_cast<std::uint16_t>(decision),
            }
        );
        if (decision == ControllerCallbackDecision::continue_generation) return true;
        if (decision == ControllerCallbackDecision::stop_generation) {
            outcome_.disposition = ControllerDisposition::stopped;
            outcome_.stop_reason = ControllerStopReason::callback_stop;
        } else {
            outcome_.disposition = ControllerDisposition::callback_threw;
            outcome_.stop_reason = ControllerStopReason::callback_throw;
        }
        return false;
    }

    void carry(
        std::uint32_t token,
        const SamplerCheckpoint& checkpoint,
        bool cached,
        ControllerDisposition disposition,
        ControllerStopReason reason,
        std::uint32_t depth,
        std::uint32_t node
    ) {
        outcome_.has_carry = true;
        outcome_.carry_token = token;
        outcome_.carry_checkpoint = checkpoint;
        outcome_.carry_is_cached = cached;
        outcome_.disposition = disposition;
        outcome_.stop_reason = reason;
        trace(
            ControllerTraceEventKind::carry,
            depth,
            0U,
            token,
            node,
            ConditionalCostKey {
                .kind = disposition == ControllerDisposition::ordinary_fallback
                    ? ControllerCostKind::ordinary_fallback : ControllerCostKind::none,
                .topology = topology_,
                .command_topology = scenario_.first_call_topology,
                .shape = 1U,
                .outcome = static_cast<std::uint16_t>(reason),
            }
        );
    }

private:
    SpeculativeControllerOutcome& outcome_;
    const SpeculativeControllerScenario& scenario_;
    ProposalGraphKind topology_;
};

[[nodiscard]] bool tree_descriptor_valid(
    const StreamTreeLeaseDescriptor& tree,
    std::uint64_t session_epoch
) noexcept {
    return tree.lease_identity != 0U && tree.session_epoch == session_epoch
        && tree.arena_epoch != 0U && tree.graph_identity != 0U;
}

[[nodiscard]] bool same_tree(
    const StreamTreeLeaseDescriptor& left,
    const StreamTreeLeaseDescriptor& right
) noexcept {
    return left.lease_identity == right.lease_identity
        && left.session_epoch == right.session_epoch
        && left.arena_epoch == right.arena_epoch
        && left.graph_identity == right.graph_identity
        && left.history_digest == right.history_digest
        && left.root_token == right.root_token
        && left.root_position == right.root_position;
}

}  // namespace

ResolvedTreeBudget resolve_source_tree_budget(SourceTreeBudget source) {
    if (source.root_convention == SourceRootConvention::excluded) {
        if (source.value > ExternalTreeDraftNodeMax) {
            throw std::overflow_error("root-excluded source tree budget exceeds 1,024 draft nodes");
        }
        return ResolvedTreeBudget {
            .draft_node_cap = source.value,
            .input_row_cap = source.value + 1U,
        };
    }
    if (source.value == 0U || source.value > VerifierRowMax) {
        throw std::overflow_error("root-included source tree budget must be in [1, 1,025]");
    }
    return ResolvedTreeBudget {
        .draft_node_cap = source.value - 1U,
        .input_row_cap = source.value,
    };
}

ValidatedProposalGraph::ValidatedProposalGraph(ProposalGraphStorage storage)
    : storage_(std::move(storage)) {}

const ProposalGraphStorage& ValidatedProposalGraph::storage() const noexcept { return storage_; }

ValidatedProposalGraph validate_proposal_graph(
    ProposalGraphStorage storage,
    const ProposalGraphValidationContext& context
) {
    const auto& header = storage.header;
    if (context.vocabulary_size == 0U || context.context_length == 0U) {
        throw std::invalid_argument("proposal graph validation context is empty");
    }
    if (header.graph_identity == 0U || header.device_arena_epoch == 0U
        || header.device_arena_epoch != context.expected_device_arena_epoch) {
        throw std::invalid_argument("proposal graph identity or device-arena epoch is stale");
    }
    if (header.horizon > external_speculation_max_horizon
        || header.materialized_node_count == 0U
        || header.materialized_node_count > VerifierRowMax
        || header.materialized_edge_count > MaterializedTreeEdgeMax
        || header.output_frontier_edge_count > OutputFrontierEdgeMax) {
        throw std::invalid_argument("proposal graph exceeds a fixed safety capacity");
    }

    const bool repeated = header.kind == ProposalGraphKind::repeated_root_beam_batch;
    const auto required_roots = repeated ? header.repeated_beam_width : 1U;
    if (header.root_count != required_roots || header.root_count == 0U
        || header.root_count > ReDrafterMLXRepeatedVerifierRootMax) {
        throw std::invalid_argument("proposal graph root count does not match its topology tag");
    }
    if (header.materialized_edge_count
        != header.materialized_node_count - header.root_count) {
        throw std::invalid_argument("proposal graph materialized edges do not form a rooted forest");
    }
    if (header.has_source_budget) {
        const auto resolved = resolve_source_tree_budget(header.source_budget);
        if (resolved.draft_node_cap != header.resolved_budget.draft_node_cap
            || resolved.input_row_cap != header.resolved_budget.input_row_cap
            || header.materialized_node_count > resolved.input_row_cap
            || header.materialized_node_count - header.root_count > resolved.draft_node_cap) {
            throw std::invalid_argument("proposal graph source budget resolution is inconsistent");
        }
    }

    if (!repeated) {
        if (header.root_indices[0] != 0U || header.repeated_beam_width != 0U
            || header.repeated_beam_length != 0U) {
            throw std::invalid_argument("unique-root proposal graph has repeated-root metadata");
        }
    } else {
        if (header.output_frontier_edge_count != 0U
            || header.repeated_beam_length == 0U
            || header.repeated_beam_width * header.repeated_beam_length
                != header.materialized_node_count
            || header.materialized_node_count > ReDrafterMLXPhysicalRowMax
            || header.horizon + 1U != header.repeated_beam_length) {
            throw std::invalid_argument("repeated-root beam dimensions are invalid");
        }
        for (std::size_t root = 0; root < header.root_count; ++root) {
            if (header.root_indices[root] != root * header.repeated_beam_length) {
                throw std::invalid_argument("repeated-root beams must be stored as isolated blocks");
            }
        }
    }

    std::array<bool, VerifierRowMax> is_root {};
    for (std::size_t root = 0; root < header.root_count; ++root) {
        const auto node = header.root_indices[root];
        if (node >= header.materialized_node_count || is_root[node]) {
            throw std::invalid_argument("proposal graph root index is invalid or duplicated");
        }
        is_root[node] = true;
    }

    for (std::size_t node = 0; node < header.materialized_node_count; ++node) {
        if (storage.nodes.token_ids[node] >= context.vocabulary_size
            || storage.nodes.absolute_positions[node] >= context.context_length
            || !valid_confidence(storage.nodes.conditional_confidences[node])
            || storage.nodes.feature_provenance[node].arena_epoch != header.device_arena_epoch) {
            throw std::invalid_argument("proposal graph node value or provenance is invalid");
        }
        if (is_root[node]) {
            if (storage.nodes.parent_indices[node] != proposal_node_none
                || storage.nodes.depths[node] != 0U
                || storage.nodes.mask_component_roots[node] != node
                || storage.nodes.mask_ancestor_counts[node] != 1U
                || storage.nodes.token_ids[node] != context.expected_root_token
                || storage.nodes.absolute_positions[node] != context.expected_root_position
                || !same_history_descriptor(
                    storage.nodes.path_histories[node], context.expected_root_history
                )
                || storage.nodes.proposal_ticket_offsets[node] != proposal_ticket_none) {
                throw std::invalid_argument("proposal graph root identity is incompatible");
            }
            continue;
        }
        const auto parent = storage.nodes.parent_indices[node];
        if (parent >= node || parent >= header.materialized_node_count
            || storage.nodes.depths[node] != storage.nodes.depths[parent] + 1U
            || storage.nodes.depths[node] > header.horizon
            || storage.nodes.absolute_positions[node]
                != storage.nodes.absolute_positions[parent] + 1U
            || storage.nodes.mask_component_roots[node]
                != storage.nodes.mask_component_roots[parent]
            || storage.nodes.mask_ancestor_counts[node]
                != storage.nodes.mask_ancestor_counts[parent] + 1U
            || storage.nodes.path_histories[node].length
                != storage.nodes.path_histories[parent].length + 1U
            || (header.primary_policy == ProposalPrimaryPolicy::scalar_common_draw
                ? storage.nodes.proposal_ticket_offsets[node] >= sampler_max_preview_ticket_count
                : storage.nodes.proposal_ticket_offsets[node] != proposal_ticket_none)) {
            throw std::invalid_argument("proposal graph node is not a causal parent-before-child row");
        }
    }

    if (repeated) {
        const auto root = header.root_indices[0];
        for (std::size_t index = 1; index < header.root_count; ++index) {
            const auto duplicate = header.root_indices[index];
            if (storage.nodes.token_ids[duplicate] != storage.nodes.token_ids[root]
                || storage.nodes.absolute_positions[duplicate]
                    != storage.nodes.absolute_positions[root]
                || !same_history_descriptor(
                    storage.nodes.path_histories[duplicate], storage.nodes.path_histories[root]
                )) {
                throw std::invalid_argument("repeated verifier roots are not byte-identical identities");
            }
        }
        for (std::size_t left_beam = 0; left_beam < header.root_count; ++left_beam) {
            for (std::size_t right_beam = left_beam + 1U;
                 right_beam < header.root_count; ++right_beam) {
                bool prefix_equal = true;
                const auto left_root = header.root_indices[left_beam];
                const auto right_root = header.root_indices[right_beam];
                for (std::size_t depth = 0; depth < header.repeated_beam_length; ++depth) {
                    const auto left = left_root + depth;
                    const auto right = right_root + depth;
                    prefix_equal = prefix_equal
                        && storage.nodes.token_ids[left] == storage.nodes.token_ids[right];
                    if (!prefix_equal) continue;
                    const auto& left_feature = storage.nodes.feature_provenance[left];
                    const auto& right_feature = storage.nodes.feature_provenance[right];
                    if (storage.nodes.absolute_positions[left]
                            != storage.nodes.absolute_positions[right]
                        || storage.nodes.mask_ancestor_counts[left]
                            != storage.nodes.mask_ancestor_counts[right]
                        || !same_history_descriptor(
                            storage.nodes.path_histories[left], storage.nodes.path_histories[right]
                        )
                        || left_feature.source != right_feature.source
                        || left_feature.source_layer != right_feature.source_layer
                        || left_feature.arena_epoch != right_feature.arena_epoch) {
                        throw std::invalid_argument(
                            "byte-identical repeated-beam prefixes have incompatible row identity"
                        );
                    }
                }
            }
        }
    }

    std::array<std::uint16_t, VerifierRowMax> incoming {};
    std::array<std::uint16_t, VerifierRowMax> materialized_degree {};
    std::array<std::uint16_t, VerifierRowMax> frontier_degree {};
    std::size_t group_start = 0;
    std::uint32_t prior_parent = 0;
    for (std::size_t edge = 0; edge < header.materialized_edge_count; ++edge) {
        const auto parent = storage.materialized_edges.parent_node_indices[edge];
        const auto child = storage.materialized_edges.child_node_indices[edge];
        if (parent >= header.materialized_node_count || child >= header.materialized_node_count
            || parent >= child || storage.nodes.parent_indices[child] != parent
            || storage.materialized_edges.token_ids[edge] != storage.nodes.token_ids[child]
            || !valid_confidence(storage.materialized_edges.conditional_confidences[edge])
            || storage.materialized_edges.proposal_ticket_offsets[edge]
                != storage.nodes.proposal_ticket_offsets[child]) {
            throw std::invalid_argument("materialized proposal edge is inconsistent with its child");
        }
        if (edge == 0U || parent != prior_parent) {
            if (edge != 0U && parent < prior_parent) {
                throw std::invalid_argument("materialized proposal edges are not stably grouped");
            }
            group_start = edge;
            prior_parent = parent;
        }
        if (storage.materialized_edges.sibling_ranks[edge] != edge - group_start) {
            throw std::invalid_argument("materialized proposal sibling ranks are not contiguous");
        }
        for (std::size_t prior = group_start; prior < edge; ++prior) {
            if (storage.materialized_edges.token_ids[prior]
                == storage.materialized_edges.token_ids[edge]) {
                throw std::invalid_argument("materialized proposal siblings contain duplicate tokens");
            }
        }
        if (++incoming[child] != 1U || ++materialized_degree[parent] > MaterializedDegreeMax) {
            throw std::invalid_argument("materialized proposal incoming edge or degree is invalid");
        }
    }
    group_start = 0;
    prior_parent = 0;
    for (std::size_t edge = 0; edge < header.output_frontier_edge_count; ++edge) {
        const auto parent = storage.output_frontier_edges.parent_node_indices[edge];
        if (parent >= header.materialized_node_count
            || storage.output_frontier_edges.token_ids[edge] >= context.vocabulary_size
            || !valid_confidence(storage.output_frontier_edges.conditional_confidences[edge])
            || (header.primary_policy == ProposalPrimaryPolicy::scalar_common_draw
                ? storage.output_frontier_edges.proposal_ticket_offsets[edge]
                    >= sampler_max_preview_ticket_count
                : storage.output_frontier_edges.proposal_ticket_offsets[edge]
                    != proposal_ticket_none)
            || storage.nodes.depths[parent] + 1U > header.horizon) {
            throw std::invalid_argument("output-frontier proposal edge is invalid");
        }
        if (edge == 0U || parent != prior_parent) {
            if (edge != 0U && parent < prior_parent) {
                throw std::invalid_argument("output-frontier edges are not stably grouped");
            }
            group_start = edge;
            prior_parent = parent;
        }
        if (storage.output_frontier_edges.sibling_ranks[edge] != edge - group_start) {
            throw std::invalid_argument("output-frontier sibling ranks are not contiguous");
        }
        for (std::size_t prior = group_start; prior < edge; ++prior) {
            if (storage.output_frontier_edges.token_ids[prior]
                == storage.output_frontier_edges.token_ids[edge]) {
                throw std::invalid_argument("output-frontier siblings contain duplicate tokens");
            }
        }
        for (std::size_t materialized = 0; materialized < header.materialized_edge_count;
             ++materialized) {
            if (storage.materialized_edges.parent_node_indices[materialized] == parent
                && storage.materialized_edges.token_ids[materialized]
                    == storage.output_frontier_edges.token_ids[edge]) {
                throw std::invalid_argument("materialized and output-frontier siblings overlap");
            }
        }
        if (++frontier_degree[parent] > OutputFrontierBranchMax) {
            throw std::invalid_argument("output-frontier branch exceeds 32 proposals");
        }
    }

    for (std::size_t node = 0; node < header.materialized_node_count; ++node) {
        if ((is_root[node] && incoming[node] != 0U) || (!is_root[node] && incoming[node] != 1U)) {
            throw std::invalid_argument("proposal graph is not prefix-closed");
        }
        const auto degree = static_cast<std::uint32_t>(materialized_degree[node])
            + frontier_degree[node];
        if ((degree != 0U
                && storage.nodes.output_roles[node] != ProposalRowOutputRole::verifies_proposals)
            || (degree == 0U
                && storage.nodes.output_roles[node] == ProposalRowOutputRole::verifies_proposals)) {
            throw std::invalid_argument("proposal row output role does not match its outgoing edges");
        }
        if (header.kind == ProposalGraphKind::chain && degree > 1U) {
            throw std::invalid_argument("chain proposal graph branches");
        }
        if (repeated && degree > 1U) {
            throw std::invalid_argument("repeated-root beam component branches");
        }
    }

    if (header.kind == ProposalGraphKind::chain) {
        if (header.materialized_edge_count + header.output_frontier_edge_count != header.horizon
            || header.output_frontier_edge_count > 1U) {
            throw std::invalid_argument("chain proposal count does not equal its horizon");
        }
    }
    if (repeated) {
        for (std::size_t beam = 0; beam < header.root_count; ++beam) {
            const auto begin = beam * header.repeated_beam_length;
            const auto end = begin + header.repeated_beam_length;
            for (std::size_t node = begin; node < end; ++node) {
                if (storage.nodes.depths[node] != node - begin
                    || (node != begin && storage.nodes.parent_indices[node] != node - 1U)) {
                    throw std::invalid_argument("repeated-root beam block is not a causal chain");
                }
            }
        }
    }

    return ValidatedProposalGraph(std::move(storage));
}

void validate_chain_verifier_schedule(
    const ChainVerifierSchedule& schedule,
    std::uint32_t horizon
) {
    if (horizon == 0U || horizon > external_speculation_max_horizon
        || schedule.tile_count == 0U || schedule.tile_count > horizon) {
        throw std::invalid_argument("chain verifier schedule has invalid horizon or tile count");
    }
    std::uint32_t total_edges = 0;
    for (std::size_t tile = 0; tile < schedule.tile_count; ++tile) {
        const auto descriptor = schedule.tiles[tile];
        if (descriptor.validated_edge_count == 0U) {
            throw std::invalid_argument("chain verifier tile validates zero edges");
        }
        const bool final = tile + 1U == schedule.tile_count;
        const auto required_rows = descriptor.validated_edge_count
            + (final && schedule.final_endpoint == ChainTileEndpoint::post_full_candidate ? 1U : 0U);
        if (descriptor.input_row_count != required_rows) {
            throw std::invalid_argument("chain verifier tile row/edge mapping is invalid");
        }
        total_edges += descriptor.validated_edge_count;
    }
    if (total_edges != horizon) {
        throw std::invalid_argument("chain verifier tile edge partition does not equal horizon");
    }
}

ChainVerifierSchedule make_registered_conditional_chain_schedule(
    std::uint32_t horizon,
    ChainTileEndpoint final_endpoint
) {
    ChainVerifierSchedule schedule;
    schedule.final_endpoint = final_endpoint;
    std::array<std::uint16_t, 3> edge_parts {};
    switch (horizon) {
        case 3U: edge_parts = {2U, 1U, 0U}; schedule.tile_count = 2U; break;
        case 4U: edge_parts = {2U, 2U, 0U}; schedule.tile_count = 2U; break;
        case 7U: edge_parts = {4U, 3U, 0U}; schedule.tile_count = 2U; break;
        case 8U: edge_parts = {4U, 4U, 0U}; schedule.tile_count = 2U; break;
        case 15U: edge_parts = {8U, 7U, 0U}; schedule.tile_count = 2U; break;
        case 16U: edge_parts = {8U, 8U, 0U}; schedule.tile_count = 2U; break;
        case 23U: edge_parts = {8U, 8U, 7U}; schedule.tile_count = 3U; break;
        default:
            throw std::invalid_argument(
                "conditional chain schedule horizon is not in the registered frontier"
            );
    }
    for (std::uint32_t tile = 0U; tile < schedule.tile_count; ++tile) {
        const auto final = tile + 1U == schedule.tile_count;
        const auto rows = static_cast<std::uint16_t>(
            edge_parts[tile]
            + (final && final_endpoint == ChainTileEndpoint::post_full_candidate ? 1U : 0U)
        );
        schedule.tiles[tile] = ChainVerifierTile {
            .validated_edge_count = edge_parts[tile],
            .input_row_count = rows,
        };
    }
    validate_chain_verifier_schedule(schedule, horizon);
    return schedule;
}

SpeculativeControllerOutcome BoundedSpeculativeController::run(
    const SpeculativeControllerScenario& scenario
) const {
    return run_impl(scenario, false);
}

SpeculativeControllerOutcome BoundedSpeculativeController::run_available(
    const SpeculativeControllerScenario& scenario
) const {
    return run_impl(scenario, true);
}

SpeculativeControllerOutcome BoundedSpeculativeController::run_impl(
    const SpeculativeControllerScenario& scenario,
    bool pause_before_missing_target_sample
) const {
    if (scenario.graph == nullptr || scenario.remaining_selected_token_budget == 0U
        || scenario.target_sample_count > external_speculation_max_horizon
        || scenario.callback_decision_count > controller_max_selected_tokens) {
        throw std::invalid_argument("speculative controller scenario is incomplete");
    }
    const auto& graph = scenario.graph->storage();
    if (graph.header.kind == ProposalGraphKind::chain) {
        validate_chain_verifier_schedule(scenario.chain_schedule, graph.header.horizon);
    } else if (scenario.chain_schedule.tile_count != 0U) {
        throw std::invalid_argument("non-chain proposal graph carries a chain tile schedule");
    }
    if (graph.nodes.token_ids[graph.header.root_indices[0]] != scenario.origin_token) {
        throw std::invalid_argument("controller origin token does not match proposal root");
    }
    const bool compatible_route =
        (scenario.proposal_source == SpeculativeProposalSource::none)
        || (scenario.proposal_source == SpeculativeProposalSource::retrieval
            && scenario.route_mode == SpeculativeRouteMode::retrieval_first)
        || (scenario.proposal_source == SpeculativeProposalSource::learned_external
            && (scenario.route_mode == SpeculativeRouteMode::retrieval_anchored_learned
                || scenario.route_mode == SpeculativeRouteMode::learned_only))
        || (scenario.proposal_source == SpeculativeProposalSource::target_integrated_stream
            && scenario.route_mode == SpeculativeRouteMode::target_integrated);
    if (!compatible_route) {
        throw std::invalid_argument("proposal source is incompatible with controller route mode");
    }
    TargetVerifierTopology expected_verifier = TargetVerifierTopology::prefix_closed_forest;
    if (scenario.proposal_source == SpeculativeProposalSource::target_integrated_stream) {
        if (graph.header.kind != ProposalGraphKind::forest) {
            throw std::invalid_argument("target-integrated route requires a prefix-closed forest");
        }
        expected_verifier = TargetVerifierTopology::target_integrated_forest;
        if (scenario.first_call_topology != FirstCallCommandTopology::target_integrated) {
            throw std::invalid_argument("target-integrated route has an external command topology");
        }
    } else {
        if (scenario.first_call_topology == FirstCallCommandTopology::target_integrated) {
            throw std::invalid_argument("external route has a target-integrated command topology");
        }
        if (graph.header.kind == ProposalGraphKind::chain) {
            expected_verifier = scenario.chain_schedule.tile_count > 1U
                ? TargetVerifierTopology::conditional_tile_cascade
                : (scenario.chain_schedule.final_endpoint == ChainTileEndpoint::terminal_carry
                    ? TargetVerifierTopology::single_invocation_terminal_carry
                    : TargetVerifierTopology::single_invocation_post_full);
        } else if (graph.header.kind == ProposalGraphKind::repeated_root_beam_batch) {
            expected_verifier = TargetVerifierTopology::repeated_root_beam_batch;
        }
    }
    if (scenario.verifier_topology != expected_verifier) {
        throw std::invalid_argument("declared verifier topology does not match the proposal plan");
    }

    SpeculativeControllerOutcome outcome;
    outcome.proposal_source = scenario.proposal_source;
    outcome.route_mode = scenario.route_mode;
    outcome.verifier_topology = scenario.verifier_topology;
    outcome.first_call_topology = scenario.first_call_topology;
    outcome.primary_policy = graph.header.primary_policy;
    outcome.sibling_policy = graph.header.sibling_policy;
    outcome.logical_cached_count = scenario.initial_logical_cached_count;
    outcome.adopted_checkpoint = scenario.origin_checkpoint;
    std::fill(outcome.selected_node_path.begin(), outcome.selected_node_path.end(), proposal_node_none);
    std::fill(outcome.retained_row_nodes.begin(), outcome.retained_row_nodes.end(), proposal_node_none);
    const auto topology = graph.header.kind;
    const bool repeated = topology == ProposalGraphKind::repeated_root_beam_batch;
    OutcomeWriter writer(outcome, scenario, topology);
    const auto root = graph.header.root_indices[0];
    writer.select(scenario.origin_token, root, scenario.origin_checkpoint);

    auto endpoint = [&](std::uint32_t depth, std::uint32_t token, bool cached) {
        if (!cached) writer.endpoint_kv(token, depth);
        if (scenario.token_is_stop[depth]) {
            outcome.disposition = ControllerDisposition::stopped;
            outcome.stop_reason = ControllerStopReason::stop_token;
            return;
        }
        if (!writer.callback(depth, token)) return;
        outcome.disposition = ControllerDisposition::stopped;
        outcome.stop_reason = ControllerStopReason::selected_token_budget;
    };

    if (scenario.token_is_stop[0] || scenario.remaining_selected_token_budget == 1U) {
        endpoint(0U, scenario.origin_token, false);
        writer.trace(ControllerTraceEventKind::complete, 0U, 0U, scenario.origin_token, root);
        return outcome;
    }
    if (!scenario.eligible || scenario.proposal_source == SpeculativeProposalSource::none) {
        writer.carry(
            scenario.origin_token,
            scenario.origin_checkpoint,
            false,
            ControllerDisposition::ordinary_fallback,
            ControllerStopReason::ineligible,
            0U,
            root
        );
        return outcome;
    }
    const auto first_required_context = topology == ProposalGraphKind::chain
        ? static_cast<std::uint32_t>(scenario.chain_schedule.tiles[0].input_row_count)
        : graph.header.horizon + 1U;
    if (first_required_context > scenario.available_context_rows) {
        writer.carry(
            scenario.origin_token,
            scenario.origin_checkpoint,
            false,
            ControllerDisposition::ordinary_fallback,
            ControllerStopReason::context_exhausted,
            0U,
            root
        );
        return outcome;
    }

    if (scenario.proposal_source != SpeculativeProposalSource::target_integrated_stream) {
        writer.trace(
            ControllerTraceEventKind::draft_execution,
            0U,
            0U,
            scenario.origin_token,
            root,
            ConditionalCostKey {
                .kind = ControllerCostKind::draft_execution,
                .topology = topology,
                .command_topology = scenario.first_call_topology,
                .shape = static_cast<std::uint16_t>(graph.header.horizon),
                .outcome = static_cast<std::uint16_t>(scenario.draft_schedule),
            }
        );
    }

    std::uint32_t target_sample_cursor = 0;
    std::uint32_t current_depth = 0;
    std::uint32_t current_node = root;
    auto current_checkpoint = scenario.origin_checkpoint;
    std::uint32_t remaining_context = scenario.available_context_rows;
    std::uint32_t last_preview_trace_index = 0;

    auto take_sample = [&](std::uint32_t tile) -> std::optional<ControllerTargetSample> {
        if (target_sample_cursor >= scenario.target_sample_count) {
            if (pause_before_missing_target_sample) {
                outcome.requires_target_sample = true;
                outcome.required_target_sample_offset = target_sample_cursor;
                outcome.required_target_sample_tile = static_cast<std::uint16_t>(tile);
                return std::nullopt;
            }
            throw std::out_of_range("controller requested an unavailable scripted target sample");
        }
        const auto sample = scenario.target_samples[target_sample_cursor++];
        if (!sample.checkpoint.follows_sample(current_checkpoint, sample.token_id)) {
            throw std::invalid_argument("scripted target sample checkpoint does not follow its anchor");
        }
        ++outcome.sampled_preview_count;
        last_preview_trace_index = outcome.trace_count;
        writer.trace(
            ControllerTraceEventKind::target_preview,
            current_depth + 1U,
            tile,
            sample.token_id,
            proposal_node_none,
            ConditionalCostKey {
                .kind = ControllerCostKind::target_selection,
                .topology = topology,
                .command_topology = scenario.first_call_topology,
                .shape = static_cast<std::uint16_t>(current_depth + 1U),
                .outcome = 0U,
            }
        );
        return sample;
    };

    auto select_sample = [&](const ControllerTargetSample& sample, const EdgeMatch& match) {
        ++current_depth;
        current_checkpoint = sample.checkpoint;
        current_node = match.materialized ? match.child_node : proposal_node_none;
        writer.select(sample.token_id, current_node, current_checkpoint);
        if (match.matched) ++outcome.accepted_edge_count;
    };

    auto finish_carry = [&](
        std::uint32_t token,
        const SamplerCheckpoint& checkpoint,
        ControllerStopReason reason,
        std::uint32_t node
    ) {
        writer.carry(
            token,
            checkpoint,
            false,
            ControllerDisposition::carry_queued,
            reason,
            current_depth,
            node
        );
    };

    if (topology == ProposalGraphKind::chain) {
        for (std::size_t tile = 0; tile < scenario.chain_schedule.tile_count; ++tile) {
            const auto descriptor = scenario.chain_schedule.tiles[tile];
            if (tile != 0U) {
                writer.trace(
                    ControllerTraceEventKind::tile_gate,
                    current_depth,
                    static_cast<std::uint32_t>(tile),
                    outcome.selected_tokens[current_depth],
                    current_node,
                    ConditionalCostKey {
                        .kind = ControllerCostKind::tile_gate,
                        .topology = topology,
                        .command_topology = scenario.first_call_topology,
                        .shape = descriptor.input_row_count,
                        .outcome = static_cast<std::uint16_t>(scenario.tile_gates[tile - 1U]),
                    }
                );
                if (scenario.tile_gates[tile - 1U] == ControllerTileGateDecision::decline) {
                    const auto reason = graph.header.horizon == 3U && tile == 1U
                        ? ControllerStopReason::second_call_confidence_decline
                        : ControllerStopReason::tile_gate_decline;
                    writer.carry(
                        outcome.selected_tokens[current_depth], current_checkpoint, false,
                        ControllerDisposition::ordinary_fallback,
                        reason,
                        current_depth, current_node
                    );
                    return outcome;
                }
            }
            if (descriptor.input_row_count > remaining_context) {
                writer.carry(
                    outcome.selected_tokens[current_depth], current_checkpoint, false,
                    ControllerDisposition::ordinary_fallback,
                    ControllerStopReason::context_exhausted,
                    current_depth, current_node
                );
                return outcome;
            }
            writer.trace(
                ControllerTraceEventKind::verifier_tile,
                current_depth,
                static_cast<std::uint32_t>(tile),
                outcome.selected_tokens[current_depth],
                current_node,
                ConditionalCostKey {
                    .kind = ControllerCostKind::target_verifier,
                    .topology = topology,
                    .command_topology = scenario.first_call_topology,
                    .shape = descriptor.input_row_count,
                    .outcome = 0U,
                }
            );
            writer.trace(
                ControllerTraceEventKind::synchronization,
                current_depth,
                static_cast<std::uint32_t>(tile),
                outcome.selected_tokens[current_depth],
                current_node,
                ConditionalCostKey {
                    .kind = ControllerCostKind::synchronization,
                    .topology = topology,
                    .command_topology = scenario.first_call_topology,
                    .shape = descriptor.input_row_count,
                    .outcome = 0U,
                }
            );
            writer.trace(
                ControllerTraceEventKind::candidate_readback,
                current_depth,
                static_cast<std::uint32_t>(tile),
                outcome.selected_tokens[current_depth],
                current_node,
                ConditionalCostKey {
                    .kind = ControllerCostKind::readback,
                    .topology = topology,
                    .command_topology = scenario.first_call_topology,
                    .shape = descriptor.input_row_count,
                    .outcome = 0U,
                }
            );

            for (std::size_t relative_edge = 1; relative_edge <= descriptor.validated_edge_count;
                 ++relative_edge) {
                const auto parent_depth = current_depth;
                const auto parent_node = current_node;
                const auto parent_token = outcome.selected_tokens[parent_depth];
                const auto pending_sample = take_sample(static_cast<std::uint32_t>(tile));
                if (!pending_sample.has_value()) return outcome;
                const auto sample = *pending_sample;

                if (relative_edge == 1U) {
                    writer.retain(parent_token, parent_node, parent_depth, static_cast<std::uint32_t>(tile));
                    --remaining_context;
                    if (!writer.callback(parent_depth, parent_token)) {
                        writer.trace(
                            ControllerTraceEventKind::discard_preview,
                            parent_depth + 1U,
                            static_cast<std::uint32_t>(tile),
                            sample.token_id,
                            proposal_node_none,
                            ConditionalCostKey {
                                .kind = ControllerCostKind::cache_restore,
                                .topology = topology,
                                .command_topology = scenario.first_call_topology,
                                .shape = descriptor.input_row_count,
                                .outcome = 1U,
                            }
                        );
                        return outcome;
                    }
                }

                const auto match = find_edge_match(graph, parent_node, sample.token_id);
                outcome.trace[last_preview_trace_index].cost.outcome = match.matched ? 1U : 2U;
                select_sample(sample, match);
                const bool input_row = relative_edge < descriptor.input_row_count;
                if (match.matched && match.materialized && input_row) {
                    writer.retain(
                        sample.token_id, match.child_node, current_depth, static_cast<std::uint32_t>(tile)
                    );
                    --remaining_context;
                }

                const bool selected_endpoint = scenario.token_is_stop[current_depth]
                    || outcome.selected_count >= scenario.remaining_selected_token_budget;
                if (selected_endpoint) {
                    endpoint(current_depth, sample.token_id,
                        match.matched && match.materialized && input_row);
                    return outcome;
                }
                if (!match.matched) {
                    finish_carry(
                        sample.token_id, current_checkpoint,
                        ControllerStopReason::target_mismatch, proposal_node_none
                    );
                    return outcome;
                }
                if (relative_edge < descriptor.validated_edge_count) {
                    if (!writer.callback(current_depth, sample.token_id)) return outcome;
                    continue;
                }

                const bool final_tile = tile + 1U == scenario.chain_schedule.tile_count;
                if (!final_tile) break;
                if (scenario.chain_schedule.final_endpoint == ChainTileEndpoint::terminal_carry) {
                    finish_carry(
                        sample.token_id, current_checkpoint,
                        ControllerStopReason::terminal_carry, current_node
                    );
                    return outcome;
                }
                if (!match.materialized) {
                    throw std::logic_error("post-full chain endpoint is not a materialized row");
                }
                if (!writer.callback(current_depth, sample.token_id)) return outcome;
                outcome.disposition = ControllerDisposition::completed;
                outcome.stop_reason = ControllerStopReason::full_match;
                outcome.next_candidate_lease = TargetStateLeaseDescriptor {
                    .source = TargetStateLeaseSource::selected_verifier_row,
                    .owner_epoch = graph.header.device_arena_epoch,
                    .absolute_position = graph.nodes.absolute_positions[current_node],
                    .node_index = current_node,
                };
                writer.trace(
                    ControllerTraceEventKind::next_candidate,
                    current_depth,
                    static_cast<std::uint32_t>(tile),
                    sample.token_id,
                    current_node
                );
                return outcome;
            }
        }
        throw std::logic_error("chain controller exhausted its schedule without an outcome");
    }

    const auto required_context = graph.header.horizon + 1U;
    if (required_context > remaining_context) {
        writer.carry(
            scenario.origin_token, scenario.origin_checkpoint, false,
            ControllerDisposition::ordinary_fallback,
            ControllerStopReason::context_exhausted,
            0U, root
        );
        return outcome;
    }
    writer.trace(
        ControllerTraceEventKind::verifier_tile,
        0U,
        0U,
        scenario.origin_token,
        root,
        ConditionalCostKey {
            .kind = ControllerCostKind::target_verifier,
            .topology = topology,
            .command_topology = scenario.first_call_topology,
            .shape = static_cast<std::uint16_t>(graph.header.materialized_node_count),
            .outcome = 0U,
        }
    );
    writer.trace(
        ControllerTraceEventKind::synchronization,
        0U,
        0U,
        scenario.origin_token,
        root,
        ConditionalCostKey {
            .kind = ControllerCostKind::synchronization,
            .topology = topology,
            .command_topology = scenario.first_call_topology,
            .shape = static_cast<std::uint16_t>(graph.header.materialized_node_count),
            .outcome = 0U,
        }
    );
    writer.trace(
        ControllerTraceEventKind::candidate_readback,
        0U,
        0U,
        scenario.origin_token,
        root,
        ConditionalCostKey {
            .kind = ControllerCostKind::readback,
            .topology = topology,
            .command_topology = scenario.first_call_topology,
            .shape = static_cast<std::uint16_t>(graph.header.materialized_node_count),
            .outcome = 0U,
        }
    );

    std::array<std::uint32_t, ReDrafterMLXRepeatedVerifierRootMax> active_nodes {};
    std::uint32_t active_count = graph.header.root_count;
    for (std::size_t index = 0; index < active_count; ++index) {
        active_nodes[index] = graph.header.root_indices[index];
    }

    if (graph.header.horizon == 0U || !node_has_outgoing_edge(graph, current_node)) {
        writer.retain(scenario.origin_token, root, 0U, 0U);
        if (!writer.callback(0U, scenario.origin_token)) return outcome;
        outcome.disposition = ControllerDisposition::completed;
        outcome.stop_reason = ControllerStopReason::materialized_leaf;
        outcome.next_candidate_lease = TargetStateLeaseDescriptor {
            .source = scenario.proposal_source == SpeculativeProposalSource::target_integrated_stream
                ? TargetStateLeaseSource::target_integrated_main_row
                : TargetStateLeaseSource::selected_verifier_row,
            .owner_epoch = graph.header.device_arena_epoch,
            .absolute_position = graph.nodes.absolute_positions[root],
            .node_index = root,
        };
        return outcome;
    }

    while (current_depth < graph.header.horizon) {
        const auto parent_depth = current_depth;
        const auto parent_token = outcome.selected_tokens[parent_depth];
        const auto pending_sample = take_sample(0U);
        if (!pending_sample.has_value()) return outcome;
        const auto sample = *pending_sample;
        if (parent_depth == 0U) {
            writer.retain(parent_token, current_node, parent_depth, 0U);
            if (!writer.callback(parent_depth, parent_token)) {
                writer.trace(
                    ControllerTraceEventKind::discard_preview,
                    parent_depth + 1U, 0U, sample.token_id, proposal_node_none,
                    ConditionalCostKey {
                        .kind = ControllerCostKind::cache_restore,
                        .topology = topology,
                        .command_topology = scenario.first_call_topology,
                        .shape = static_cast<std::uint16_t>(graph.header.materialized_node_count),
                        .outcome = 1U,
                    }
                );
                return outcome;
            }
        }

        EdgeMatch match;
        if (!repeated) {
            match = find_edge_match(graph, current_node, sample.token_id);
        } else {
            std::array<std::uint32_t, ReDrafterMLXRepeatedVerifierRootMax> next_active {};
            std::uint32_t next_count = 0;
            for (std::size_t index = 0; index < active_count; ++index) {
                const auto candidate = find_edge_match(graph, active_nodes[index], sample.token_id);
                if (candidate.matched && candidate.materialized) {
                    next_active[next_count++] = candidate.child_node;
                }
            }
            if (next_count != 0U) {
                active_nodes = next_active;
                active_count = next_count;
                const auto selected_beam_root =
                    (active_nodes[0] / graph.header.repeated_beam_length)
                    * graph.header.repeated_beam_length;
                for (std::uint32_t depth = 0; depth <= parent_depth; ++depth) {
                    outcome.selected_node_path[depth] = selected_beam_root + depth;
                    if (depth < outcome.retained_row_count) {
                        outcome.retained_row_nodes[depth] = selected_beam_root + depth;
                    }
                }
                match = EdgeMatch {
                    .matched = true,
                    .materialized = true,
                    .child_node = active_nodes[0],
                };
            }
        }
        outcome.trace[last_preview_trace_index].cost.outcome = match.matched ? 1U : 2U;
        select_sample(sample, match);
        if (match.matched && match.materialized) {
            writer.retain(sample.token_id, match.child_node, current_depth, 0U);
        }

        if (scenario.token_is_stop[current_depth]
            || outcome.selected_count >= scenario.remaining_selected_token_budget) {
            endpoint(current_depth, sample.token_id, match.matched && match.materialized);
            return outcome;
        }
        if (!match.matched || !match.materialized) {
            finish_carry(
                sample.token_id, current_checkpoint,
                match.matched ? ControllerStopReason::terminal_carry
                              : ControllerStopReason::target_mismatch,
                proposal_node_none
            );
            return outcome;
        }
        if (!node_has_outgoing_edge(graph, current_node) || current_depth == graph.header.horizon) {
            if (!writer.callback(current_depth, sample.token_id)) return outcome;
            outcome.disposition = ControllerDisposition::completed;
            outcome.stop_reason = ControllerStopReason::materialized_leaf;
            outcome.next_candidate_lease = TargetStateLeaseDescriptor {
                .source = scenario.proposal_source == SpeculativeProposalSource::target_integrated_stream
                    ? TargetStateLeaseSource::target_integrated_main_row
                    : TargetStateLeaseSource::selected_verifier_row,
                .owner_epoch = graph.header.device_arena_epoch,
                .absolute_position = graph.nodes.absolute_positions[current_node],
                .node_index = current_node,
            };
            return outcome;
        }
        if (!writer.callback(current_depth, sample.token_id)) return outcome;
    }
    throw std::logic_error("forest controller reached an invalid horizon state");
}

CorrectedTrajectoryCost replay_corrected_trajectory_cost(
    const SpeculativeControllerOutcome& trajectory,
    const MeasuredConditionalCostTable& costs
) {
    if (costs.count > costs.values.size() || trajectory.trace_count > trajectory.trace.size()) {
        throw std::invalid_argument("trajectory replay table exceeds fixed capacity");
    }
    for (std::size_t index = 0; index < costs.count; ++index) {
        if (!std::isfinite(costs.values[index].duration_ns)
            || costs.values[index].duration_ns < 0.0) {
            throw std::invalid_argument("conditional trajectory cost is invalid");
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (costs.values[index].key == costs.values[prior].key) {
                throw std::invalid_argument("conditional trajectory cost key is duplicated");
            }
        }
    }

    CorrectedTrajectoryCost replay;
    for (std::size_t event_index = 0; event_index < trajectory.trace_count; ++event_index) {
        const auto& key = trajectory.trace[event_index].cost;
        if (key.kind == ControllerCostKind::none) continue;
        const MeasuredConditionalCost* matched = nullptr;
        for (std::size_t cost_index = 0; cost_index < costs.count; ++cost_index) {
            if (costs.values[cost_index].key == key) {
                matched = &costs.values[cost_index];
                break;
            }
        }
        if (matched == nullptr) {
            throw std::invalid_argument("trajectory replay lacks a measured conditional cost");
        }
        replay.duration_ns += matched->duration_ns;
        ++replay.charged_event_count;
    }
    if (!std::isfinite(replay.duration_ns)) {
        throw std::overflow_error("trajectory replay cost overflowed");
    }
    return replay;
}

double diagnostic_iid_expected_tokens(double acceptance, std::uint32_t horizon) {
    if (!std::isfinite(acceptance) || acceptance < 0.0 || acceptance > 1.0
        || horizon > external_speculation_max_horizon) {
        throw std::invalid_argument("iid diagnostic inputs are invalid");
    }
    double total = 1.0;
    double survival = 1.0;
    for (std::uint32_t depth = 1; depth <= horizon; ++depth) {
        survival *= acceptance;
        total += survival;
    }
    return total;
}

StreamTransitionShape make_stream_transition_shape(
    std::uint32_t gamma,
    std::uint32_t branch_count,
    std::uint32_t survivor_count
) {
    if (gamma == 0U || gamma > stream_speculation_max_horizon
        || branch_count == 0U || branch_count > StreamTransitionBranchMax) {
        throw std::invalid_argument("Arm-S gamma or transition branching is outside safety bounds");
    }
    std::uint64_t raw_nodes = 1U;
    std::uint64_t level_nodes = 1U;
    for (std::uint32_t depth = 1; depth <= gamma; ++depth) {
        level_nodes *= branch_count;
        raw_nodes += level_nodes;
        if (raw_nodes > StreamRawPriorNodeMax) {
            throw std::overflow_error("Arm-S raw prior tree exceeds its declared legal geometry");
        }
    }
    if (survivor_count > raw_nodes || survivor_count > StreamSurvivorMax) {
        throw std::invalid_argument("Arm-S survivor count exceeds raw prior nodes");
    }
    const auto upper_rows = static_cast<std::uint64_t>(survivor_count) * (gamma + 1U);
    const auto next_nodes = static_cast<std::uint64_t>(survivor_count) * raw_nodes;
    const auto raw_children = static_cast<std::uint64_t>(survivor_count)
        * gamma * branch_count;
    if (upper_rows > StreamUpperRowMax || next_nodes > SNextNodeMax
        || raw_children > SRawChildMax) {
        throw std::overflow_error("Arm-S derived transition storage exceeds a fixed safety bound");
    }
    return StreamTransitionShape {
        .gamma = static_cast<std::uint16_t>(gamma),
        .branch_count = static_cast<std::uint16_t>(branch_count),
        .raw_prior_node_count = static_cast<std::uint32_t>(raw_nodes),
        .survivor_count = survivor_count,
        .upper_row_count = static_cast<std::uint32_t>(upper_rows),
        .next_node_count = static_cast<std::uint32_t>(next_nodes),
        .raw_child_count = static_cast<std::uint32_t>(raw_children),
    };
}

StreamTransitionShapeTable generate_registered_stream_transition_table(
    std::uint32_t gamma,
    std::uint32_t branch_count
) {
    if (!is_registered_stream_geometry(gamma, branch_count)) {
        throw std::invalid_argument("Arm-S transition geometry is not in the finite registry");
    }
    const auto raw = make_stream_transition_shape(gamma, branch_count, 0U);
    StreamTransitionShapeTable table;
    table.count = static_cast<std::uint16_t>(raw.raw_prior_node_count + 1U);
    for (std::uint32_t survivors = 0; survivors <= raw.raw_prior_node_count; ++survivors) {
        table.values[survivors] = make_stream_transition_shape(
            gamma, branch_count, survivors
        );
    }
    return table;
}

StreamAnchorTicketMap generate_stream_anchor_ticket_map(std::uint32_t gamma) {
    if (gamma == 0U || gamma > stream_speculation_max_horizon) {
        throw std::invalid_argument("Arm-S ticket map gamma is outside [1, 23]");
    }
    StreamAnchorTicketMap map;
    map.count = static_cast<std::uint16_t>(gamma + 1U);
    map.pool_size = static_cast<std::uint16_t>(2U * gamma + 1U);
    for (std::uint32_t accepted_depth = 0; accepted_depth <= gamma; ++accepted_depth) {
        // The slice starts at the target draw that selects this anchor's
        // correction/root and includes gamma following primary-path draws.
        map.values[accepted_depth] = StreamAnchorTicketSlice {
            .accepted_depth = static_cast<std::uint16_t>(accepted_depth),
            .first_ticket_offset = static_cast<std::uint16_t>(accepted_depth),
            .ticket_count = static_cast<std::uint16_t>(gamma + 1U),
        };
    }
    return map;
}

StreamLeaseTransition transition_stream_lease(
    const StreamLeaseState& state,
    const StreamLeaseEvent& event
) {
    if (state.session_epoch == 0U) {
        throw std::invalid_argument("Arm-S lease state has no session epoch");
    }
    if ((state.has_prior_tree && !tree_descriptor_valid(state.prior_tree, state.session_epoch))
        || (state.has_pending_tree && !tree_descriptor_valid(state.pending_tree, state.session_epoch))) {
        throw std::invalid_argument("Arm-S lease state contains a stale tree");
    }
    if (event.has_tree && !tree_descriptor_valid(event.tree, state.session_epoch)) {
        throw std::invalid_argument("Arm-S lease event contains a stale tree");
    }

    if (event.kind == StreamLeaseEventKind::poison) {
        auto poisoned = state;
        poisoned.phase = StreamLeasePhase::poisoned;
        poisoned.has_prior_tree = false;
        poisoned.has_pending_tree = false;
        return StreamLeaseTransition {.next_state = poisoned, .tree_action = NextTreeAction::discard};
    }
    if (state.phase == StreamLeasePhase::poisoned) {
        if (event.kind != StreamLeaseEventKind::authenticated_reset || event.has_tree) {
            throw std::logic_error("poisoned Arm-S lease requires authenticated reset");
        }
        return StreamLeaseTransition {
            .next_state = StreamLeaseState {
                .phase = StreamLeasePhase::unavailable,
                .session_epoch = state.session_epoch,
            },
        };
    }

    auto next = state;
    switch (state.phase) {
    case StreamLeasePhase::unavailable:
        if (event.kind != StreamLeaseEventKind::begin_bootstrap || event.has_tree) break;
        next.phase = StreamLeasePhase::pending_bootstrap;
        return {.next_state = next};
    case StreamLeasePhase::pending_bootstrap:
        if (event.kind == StreamLeaseEventKind::bootstrap_completed && event.has_tree) {
            next.phase = StreamLeasePhase::pending_root_bind;
            next.has_pending_tree = true;
            next.pending_tree = event.tree;
            return {.next_state = next};
        }
        if ((event.kind == StreamLeaseEventKind::endpoint
                || event.kind == StreamLeaseEventKind::intentional_switch) && !event.has_tree) {
            next.phase = StreamLeasePhase::draining;
            return {.next_state = next, .tree_action = NextTreeAction::discard};
        }
        break;
    case StreamLeasePhase::pending_root_bind:
        if (event.kind == StreamLeaseEventKind::bind_selected_root && event.has_tree
            && state.has_pending_tree && same_tree(event.tree, state.pending_tree)) {
            next.phase = StreamLeasePhase::steady;
            next.has_prior_tree = true;
            next.prior_tree = event.tree;
            next.has_pending_tree = false;
            return {.next_state = next, .tree_action = NextTreeAction::publish};
        }
        if (event.kind == StreamLeaseEventKind::discard_next_tree && !event.has_tree) {
            next.phase = StreamLeasePhase::unavailable;
            next.has_pending_tree = false;
            return {.next_state = next, .tree_action = NextTreeAction::discard};
        }
        break;
    case StreamLeasePhase::steady:
        if ((event.kind == StreamLeaseEventKind::verification_mismatch
                || event.kind == StreamLeaseEventKind::verification_full_match)
            && event.has_tree) {
            next.phase = StreamLeasePhase::pending_correction_bind;
            next.has_pending_tree = true;
            next.pending_tree = event.tree;
            return {.next_state = next};
        }
        if (event.kind == StreamLeaseEventKind::publish_next_tree && event.has_tree) {
            next.has_prior_tree = true;
            next.prior_tree = event.tree;
            return {.next_state = next, .tree_action = NextTreeAction::publish};
        }
        if (event.kind == StreamLeaseEventKind::discard_next_tree && !event.has_tree) {
            return {.next_state = next, .tree_action = NextTreeAction::discard};
        }
        if (event.kind == StreamLeaseEventKind::begin_rebase && event.has_tree) {
            next.phase = StreamLeasePhase::pending_root_bind;
            next.has_pending_tree = true;
            next.pending_tree = event.tree;
            next.has_prior_tree = false;
            return {.next_state = next, .tree_action = NextTreeAction::rebase};
        }
        if ((event.kind == StreamLeaseEventKind::endpoint
                || event.kind == StreamLeaseEventKind::intentional_switch) && !event.has_tree) {
            next.phase = StreamLeasePhase::draining;
            next.has_prior_tree = false;
            return {.next_state = next, .tree_action = NextTreeAction::discard};
        }
        break;
    case StreamLeasePhase::pending_correction_bind:
        if (event.kind == StreamLeaseEventKind::bind_correction && event.has_tree
            && state.has_pending_tree && same_tree(event.tree, state.pending_tree)) {
            next.phase = StreamLeasePhase::steady;
            next.has_prior_tree = true;
            next.prior_tree = event.tree;
            next.has_pending_tree = false;
            return {.next_state = next, .tree_action = NextTreeAction::publish};
        }
        if (event.kind == StreamLeaseEventKind::begin_rebase && event.has_tree) {
            next.phase = StreamLeasePhase::pending_root_bind;
            next.has_prior_tree = false;
            next.has_pending_tree = true;
            next.pending_tree = event.tree;
            return {.next_state = next, .tree_action = NextTreeAction::rebase};
        }
        if ((event.kind == StreamLeaseEventKind::discard_next_tree
                || event.kind == StreamLeaseEventKind::endpoint
                || event.kind == StreamLeaseEventKind::intentional_switch) && !event.has_tree) {
            next.phase = StreamLeasePhase::draining;
            next.has_prior_tree = false;
            next.has_pending_tree = false;
            return {.next_state = next, .tree_action = NextTreeAction::discard};
        }
        break;
    case StreamLeasePhase::draining:
        if (event.kind == StreamLeaseEventKind::drain_completed && !event.has_tree) {
            next.phase = StreamLeasePhase::unavailable;
            next.has_prior_tree = false;
            next.has_pending_tree = false;
            return {.next_state = next};
        }
        if (event.kind == StreamLeaseEventKind::begin_rebase && event.has_tree) {
            next.phase = StreamLeasePhase::pending_root_bind;
            next.has_pending_tree = true;
            next.pending_tree = event.tree;
            return {.next_state = next, .tree_action = NextTreeAction::rebase};
        }
        break;
    case StreamLeasePhase::poisoned:
        break;
    }
    throw std::logic_error("illegal Arm-S lease transition");
}

RegisteredHorizonTransitionTable generate_registered_horizon_transition_table(
    std::uint32_t horizon
) {
    if (!is_registered_horizon(horizon)) {
        throw std::invalid_argument("transition table horizon is not registered");
    }
    RegisteredHorizonTransitionTable table;
    table.count = static_cast<std::uint16_t>(horizon + 1U);
    for (std::uint32_t depth = 0; depth <= horizon; ++depth) {
        table.values[depth] = RegisteredHorizonTransition {
            .accepted_depth = static_cast<std::uint16_t>(depth),
            .target_ticket_offset = depth < horizon
                ? static_cast<std::uint16_t>(depth) : proposal_ticket_none,
            .selected_token_count = static_cast<std::uint16_t>(depth + 1U),
        };
    }
    return table;
}

}  // namespace local_gguf_engine::engine
