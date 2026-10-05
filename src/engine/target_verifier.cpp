#include "local_gguf_engine/engine/target_verifier.hpp"

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/dtype.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#if defined(LOCAL_GGUF_ENGINE_TESTING)
#include "target_verifier_test_hooks.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;

[[nodiscard]] TensorDescriptor f32_descriptor(std::initializer_list<std::uint64_t> dimensions) {
    return TensorDescriptor::from(Shape::from(dimensions), DType::f32);
}

[[nodiscard]] std::optional<TensorDescriptor> q4_input_sums_descriptor(
    const GemmaModelConfig& config, std::uint32_t row_capacity
) {
    if (config.architecture != ModelArchitecture::gemma3
        && (config.embedding_length % 256U != 0U
            || config.feed_forward_length % 256U != 0U
            || config.attention_head_count * config.attention_value_length % 256U != 0U))
        return std::nullopt;
    const auto width = std::max({std::uint64_t(config.embedding_length),
        std::uint64_t(config.feed_forward_length),
        std::uint64_t(config.attention_head_count) * config.attention_value_length});
    // Ordinary and larger verifier arenas also execute scalar/H1 rows. Bound
    // the reusable sums independently of their maximum prefill row capacity.
    return f32_descriptor({std::min(row_capacity, 2U), width / 8U});
}

[[nodiscard]] tensor::BackendBuffer allocate_metal(
    const TensorDescriptor& descriptor,
    std::size_t& bytes
) {
    if (descriptor.byte_count > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error("target verifier allocation byte count overflow");
    }
    bytes += descriptor.byte_count;
    return tensor::allocate_backend_buffer(descriptor, runtime::BackendKind::metal);
}

[[nodiscard]] bool is_power_of_two(std::uint32_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

void require_model_shape(
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    bool require_window_within_context = true
) {
    if (layer_count == 0U || layer_count > verifier_layer_max) {
        throw std::invalid_argument("target verifier layer count exceeds its fixed capacity");
    }
    if (!is_power_of_two(context_length) || !is_power_of_two(sliding_window)
        || (require_window_within_context && sliding_window > context_length)) {
        throw std::invalid_argument("target verifier requires power-of-two context windows");
    }
}

[[nodiscard]] std::uint32_t append_ancestor_path(
    VerifierAttentionSchedule& schedule,
    const VerifierTopologyDescriptor& topology,
    std::uint32_t row,
    std::uint32_t keep_last
) {
    std::array<std::uint32_t, external_speculation_max_horizon + 1U> reverse {};
    auto count = std::uint32_t {0};
    auto node = row;
    while (node != proposal_node_none) {
        if (count >= reverse.size()) {
            throw std::invalid_argument("target verifier path exceeds ExternalHmax+1 rows");
        }
        reverse[count++] = node;
        node = topology.parent_indices[node];
    }
    if (count != static_cast<std::uint32_t>(topology.depths[row])
            - topology.root_source_depth + 1U) {
        throw std::invalid_argument("target verifier ancestor path does not match node depth");
    }
    const auto retained = std::min(count, keep_last);
    if (retained > verifier_ancestor_entry_max - schedule.ancestor_entry_count) {
        throw std::length_error("target verifier ancestor arena overflow");
    }
    const auto offset = schedule.ancestor_entry_count;
    for (std::uint32_t index = 0; index < retained; ++index) {
        schedule.ancestor_node_indices[offset + index] = reverse[retained - index - 1U];
    }
    schedule.ancestor_entry_count += retained;
    return offset;
}

[[nodiscard]] bool same_history_snapshot(
    const SamplerHistorySnapshot& left,
    const SamplerHistorySnapshot& right
) noexcept {
    if (left.head != right.head || left.length != right.length
        || left.window_limit != right.window_limit) return false;
    if (left.window_limit == 0U) return true;
    for (std::uint32_t index = 0; index < left.length; ++index) {
        const auto left_slot = (left.head + index) % left.window_limit;
        const auto right_slot = (right.head + index) % right.window_limit;
        if (left.token_ids[left_slot] != right.token_ids[right_slot]) return false;
    }
    return true;
}

void validate_device_binding(
    const VerifierDeviceInputBinding& binding,
    const VerifierTopologyDescriptor& topology
) {
    if (binding.token_ids.backend() != runtime::BackendKind::metal
        || binding.raw_selection_requests.backend() != runtime::BackendKind::metal) {
        throw std::invalid_argument("target verifier device inputs must be Metal resident");
    }
    if (binding.arena_epoch != topology.device_arena_epoch) {
        throw std::invalid_argument("target verifier device input epoch is stale");
    }
    const auto token_bytes = static_cast<std::uint64_t>(binding.first_token)
            * sizeof(std::uint32_t)
        + static_cast<std::uint64_t>(topology.input_row_count) * sizeof(std::uint32_t);
    const auto request_bytes = static_cast<std::uint64_t>(binding.first_request)
            * sizeof(VerifierDeviceSelectionRow)
        + static_cast<std::uint64_t>(topology.input_row_count)
            * sizeof(VerifierDeviceSelectionRow);
    if (token_bytes > binding.token_ids.descriptor().byte_count
        || request_bytes > binding.raw_selection_requests.descriptor().byte_count) {
        throw std::invalid_argument("target verifier device input binding exceeds its buffers");
    }
}

void populate_attention_schedule(
    VerifierTopologyDescriptor& topology,
    std::uint32_t layer_count,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period
) {
    if (global_layer_period == 0U) throw std::invalid_argument("target global layer period must be positive");
    topology.attention = VerifierAttentionSchedule {};
    auto& attention = topology.attention;
    attention.row_count = topology.input_row_count;
    attention.layer_count = layer_count;
    for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
        attention.layer_uses_sliding_window[layer] = ((layer + 1U) % global_layer_period) != 0U;
    }
    for (std::uint32_t row = 0; row < topology.input_row_count; ++row) {
        if (topology.depths[row] < topology.root_source_depth) {
            throw std::invalid_argument("target verifier row precedes its tile depth origin");
        }
        const auto local_depth = static_cast<std::uint32_t>(topology.depths[row])
            - topology.root_source_depth;
        const auto path_count = local_depth + 1U;
        if (topology.absolute_positions[row] < local_depth) {
            throw std::invalid_argument("target verifier row position precedes its private path");
        }
        const auto root_position = topology.absolute_positions[row] - local_depth;

        const auto global_offset = append_ancestor_path(
            attention, topology, row, external_speculation_max_horizon + 1U
        );
        attention.global_rows[row] = VerifierRowAttention {
            .base_first_position = 0U,
            .base_token_count = root_position,
            .ancestor_offset = global_offset,
            .ancestor_count = static_cast<std::uint16_t>(path_count),
        };

        const auto total_sliding = std::min(
            topology.absolute_positions[row] + 1U, sliding_window
        );
        const auto private_sliding = std::min(path_count, total_sliding);
        const auto base_sliding = total_sliding - private_sliding;
        if (base_sliding > root_position) {
            throw std::invalid_argument("target verifier sliding prefix precedes position zero");
        }
        const auto sliding_offset = append_ancestor_path(
            attention, topology, row, private_sliding
        );
        attention.sliding_rows[row] = VerifierRowAttention {
            .base_first_position = root_position - base_sliding,
            .base_token_count = base_sliding,
            .ancestor_offset = sliding_offset,
            .ancestor_count = static_cast<std::uint16_t>(private_sliding),
        };
        for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
            const auto& row_schedule = attention.layer_uses_sliding_window[layer]
                ? attention.sliding_rows[row] : attention.global_rows[row];
            attention.per_layer_attention_counts[layer * VerifierRowMax + row] =
                row_schedule.base_token_count + row_schedule.ancestor_count;
        }
    }
}

void validate_topology_again(
    const VerifierTopologyDescriptor& topology,
    std::uint32_t vocabulary_size,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period
) {
    if (topology.input_row_count == 0U || topology.input_row_count > VerifierRowMax) {
        throw std::invalid_argument("target verifier input row count is outside fixed capacity");
    }
    const auto root_only_forest = topology.proposal_kind == ProposalGraphKind::forest
        && topology.input_row_count == 1U
        && topology.materialized_edge_count == 0U
        && topology.output_frontier_edge_count == 0U;
    if (topology.horizon > external_speculation_max_horizon
        || (topology.horizon == 0U && !root_only_forest)) {
        throw std::invalid_argument("target verifier horizon is outside registered bounds");
    }
    if (topology.root_source_depth > topology.horizon) {
        throw std::invalid_argument("target verifier tile depth origin exceeds its horizon");
    }
    if (topology.materialized_edge_count > MaterializedTreeEdgeMax
        || topology.output_frontier_edge_count > OutputFrontierEdgeMax
        || topology.proposal_edge_count != topology.materialized_edge_count
            + topology.output_frontier_edge_count) {
        throw std::invalid_argument("target verifier proposal-edge counts are invalid");
    }
    const auto repeated = topology.kind == VerifierTopologyKind::repeated_root_beam_batch;
    if ((!repeated && topology.root_count != 1U)
        || (repeated && (topology.root_count == 0U
            || topology.root_count > ReDrafterMLXRepeatedVerifierRootMax))) {
        throw std::invalid_argument("target verifier root count does not match topology kind");
    }
    if ((repeated && topology.proposal_kind != ProposalGraphKind::repeated_root_beam_batch)
        || (!repeated && topology.proposal_kind == ProposalGraphKind::repeated_root_beam_batch)
        || (topology.proposal_kind != ProposalGraphKind::chain
            && topology.root_source_depth != 0U)) {
        throw std::invalid_argument("target verifier topology and source kind are incompatible");
    }
    if (topology.has_source_budget) {
        const auto resolved = resolve_source_tree_budget(topology.source_budget);
        if (resolved.draft_node_cap != topology.resolved_budget.draft_node_cap
            || resolved.input_row_cap != topology.resolved_budget.input_row_cap
            || topology.input_row_count > resolved.input_row_cap) {
            throw std::invalid_argument("target verifier source budget is inconsistent");
        }
    }
    std::array<bool, VerifierRowMax> roots {};
    for (std::uint32_t root = 0U; root < topology.root_count; ++root) {
        const auto row = topology.root_indices[root];
        if (row >= topology.input_row_count || roots[row]) {
            throw std::invalid_argument("target verifier root index is invalid or duplicated");
        }
        roots[row] = true;
    }
    std::array<bool, VerifierRowMax> source_nodes {};
    std::array<std::uint16_t, VerifierRowMax> incoming {};
    for (std::uint32_t row = 0; row < topology.input_row_count; ++row) {
        if (topology.token_ids[row] >= vocabulary_size
            || topology.absolute_positions[row] >= context_length
            || topology.selection_lanes[row] != row
            || topology.source_node_indices[row] >= VerifierRowMax
            || source_nodes[topology.source_node_indices[row]]
            || topology.causal_history_counts[row] != topology.depths[row]
            || topology.depths[row] > topology.horizon) {
            throw std::invalid_argument("target verifier row token, position, or selection lane is invalid");
        }
        source_nodes[topology.source_node_indices[row]] = true;
        const auto parent = topology.parent_indices[row];
        if (roots[row]) {
            if (topology.depths[row] != topology.root_source_depth
                || parent != proposal_node_none || topology.component_roots[row] != row) {
                throw std::invalid_argument("target verifier root metadata is invalid");
            }
        } else if (topology.depths[row] == topology.root_source_depth) {
            throw std::invalid_argument("target verifier non-root is at the root depth");
        } else if (parent >= row
            || topology.depths[row] != topology.depths[parent] + 1U
            || topology.absolute_positions[row] != topology.absolute_positions[parent] + 1U
            || topology.component_roots[row] != topology.component_roots[parent]) {
            throw std::invalid_argument("target verifier parent/depth/position metadata is invalid");
        }
    }
    std::uint32_t prior_parent = 0U;
    std::uint32_t group_start = 0U;
    for (std::uint32_t edge = 0U; edge < topology.materialized_edge_count; ++edge) {
        const auto parent = topology.materialized_edges.parent_node_indices[edge];
        const auto child = topology.materialized_edges.child_node_indices[edge];
        if (parent >= topology.input_row_count || child >= topology.input_row_count
            || topology.parent_indices[child] != parent
            || topology.materialized_edges.token_ids[edge] != topology.token_ids[child]) {
            throw std::invalid_argument("target verifier materialized edge is inconsistent");
        }
        if (edge == 0U || parent != prior_parent) {
            if (edge != 0U && parent < prior_parent) {
                throw std::invalid_argument("target verifier materialized edges are not grouped");
            }
            prior_parent = parent;
            group_start = edge;
        }
        if (topology.materialized_edges.sibling_ranks[edge] != edge - group_start
            || ++incoming[child] != 1U) {
            throw std::invalid_argument("target verifier sibling rank or incoming edge is invalid");
        }
    }
    if (topology.materialized_edge_count != topology.input_row_count - topology.root_count) {
        throw std::invalid_argument("target verifier materialized rows do not form a forest");
    }
    for (std::uint32_t row = 0U; row < topology.input_row_count; ++row) {
        if ((!roots[row] && incoming[row] != 1U) || (roots[row] && incoming[row] != 0U)) {
            throw std::invalid_argument("target verifier row has invalid materialized indegree");
        }
    }
    prior_parent = 0U;
    group_start = 0U;
    for (std::uint32_t edge = 0U; edge < topology.output_frontier_edge_count; ++edge) {
        const auto parent = topology.output_frontier_edges.parent_node_indices[edge];
        if (parent >= topology.input_row_count
            || topology.output_frontier_edges.token_ids[edge] >= vocabulary_size) {
            throw std::invalid_argument("target verifier output-frontier edge is invalid");
        }
        if (edge == 0U || parent != prior_parent) {
            if (edge != 0U && parent < prior_parent) {
                throw std::invalid_argument("target verifier output-frontier edges are not grouped");
            }
            prior_parent = parent;
            group_start = edge;
        }
        if (topology.output_frontier_edges.sibling_ranks[edge] != edge - group_start) {
            throw std::invalid_argument("target verifier output-frontier sibling rank is invalid");
        }
    }
    if (topology.tile_count == 0U || topology.tile_count > verifier_tile_max) {
        throw std::invalid_argument("target verifier tile count is outside fixed capacity");
    }
    auto expected_first = std::uint32_t {0U};
    for (std::uint32_t tile = 0U; tile < topology.tile_count; ++tile) {
        const auto& descriptor = topology.tiles[tile];
        if (descriptor.first_row != expected_first || descriptor.row_count == 0U
            || descriptor.first_row > topology.input_row_count
            || descriptor.row_count > topology.input_row_count - descriptor.first_row) {
            throw std::invalid_argument("target verifier tile schedule is inconsistent");
        }
        if (topology.proposal_kind == ProposalGraphKind::chain) {
            if (descriptor.proposal_edge_count < descriptor.row_count - 1U
                || descriptor.proposal_edge_count > descriptor.row_count) {
                throw std::invalid_argument("target verifier chain tile edge count is inconsistent");
            }
            expected_first += descriptor.proposal_edge_count;
        } else {
            if (tile != 0U || descriptor.proposal_edge_count != topology.proposal_edge_count) {
                throw std::invalid_argument("target verifier forest tile edge count is inconsistent");
            }
            expected_first = topology.input_row_count;
        }
    }
    if (topology.proposal_kind != ProposalGraphKind::chain
        && (topology.tile_count != 1U || topology.tiles[0].first_row != 0U
            || topology.tiles[0].row_count != topology.input_row_count)) {
        throw std::invalid_argument("target verifier forest carries a chain tile schedule");
    }
    if (topology.attention.row_count != topology.input_row_count
        || topology.attention.layer_count == 0U
        || topology.attention.layer_count > verifier_layer_max
        || topology.attention.ancestor_entry_count > verifier_ancestor_entry_max) {
        throw std::invalid_argument("target verifier attention schedule extent is invalid");
    }
    if (global_layer_period == 0U) throw std::invalid_argument("target global layer period must be positive");
    for (std::uint32_t layer = 0; layer < topology.attention.layer_count; ++layer) {
        if (topology.attention.layer_uses_sliding_window[layer]
            != (((layer + 1U) % global_layer_period) != 0U)) {
            throw std::invalid_argument("target verifier attention pattern does not match target");
        }
    }
    auto expected_ancestor_offset = std::uint32_t {0U};
    for (std::uint32_t row = 0U; row < topology.input_row_count; ++row) {
        const auto local_depth = static_cast<std::uint32_t>(topology.depths[row])
            - topology.root_source_depth;
        const auto path_count = local_depth + 1U;
        std::array<std::uint32_t, external_speculation_max_horizon + 1U> reverse {};
        auto node = row;
        for (std::uint32_t index = 0U; index < path_count; ++index) {
            if (node == proposal_node_none) {
                throw std::invalid_argument("target verifier attention path ended early");
            }
            reverse[index] = node;
            node = topology.parent_indices[node];
        }
        if (node != proposal_node_none) {
            throw std::invalid_argument("target verifier attention path did not reach its root");
        }
        const auto root_position = topology.absolute_positions[row] - local_depth;
        const auto& global = topology.attention.global_rows[row];
        if (global.base_first_position != 0U || global.base_token_count != root_position
            || global.ancestor_offset != expected_ancestor_offset
            || global.ancestor_count != path_count) {
            throw std::invalid_argument("target verifier global attention row is not canonical");
        }
        for (std::uint32_t index = 0U; index < path_count; ++index) {
            if (topology.attention.ancestor_node_indices[expected_ancestor_offset + index]
                != reverse[path_count - index - 1U]) {
                throw std::invalid_argument("target verifier global ancestor path is not canonical");
            }
        }
        expected_ancestor_offset += path_count;

        const auto total_sliding = std::min(
            topology.absolute_positions[row] + 1U, sliding_window
        );
        const auto private_sliding = std::min(path_count, total_sliding);
        const auto base_sliding = total_sliding - private_sliding;
        const auto& sliding = topology.attention.sliding_rows[row];
        if (sliding.base_first_position != root_position - base_sliding
            || sliding.base_token_count != base_sliding
            || sliding.ancestor_offset != expected_ancestor_offset
            || sliding.ancestor_count != private_sliding) {
            throw std::invalid_argument("target verifier sliding attention row is not canonical");
        }
        for (std::uint32_t index = 0U; index < private_sliding; ++index) {
            if (topology.attention.ancestor_node_indices[expected_ancestor_offset + index]
                != reverse[private_sliding - index - 1U]) {
                throw std::invalid_argument("target verifier sliding ancestor path is not canonical");
            }
        }
        expected_ancestor_offset += private_sliding;

        const auto validate_row = [&](const VerifierRowAttention& schedule) {
            if (schedule.ancestor_offset > topology.attention.ancestor_entry_count
                || schedule.ancestor_count > topology.attention.ancestor_entry_count
                    - schedule.ancestor_offset
                || schedule.base_first_position > topology.absolute_positions[row]
                || schedule.base_token_count > topology.absolute_positions[row] + 1U) {
                throw std::invalid_argument("target verifier attention row is out of bounds");
            }
        };
        validate_row(topology.attention.sliding_rows[row]);
        validate_row(topology.attention.global_rows[row]);
        for (std::uint32_t layer = 0U; layer < topology.attention.layer_count; ++layer) {
            const auto& schedule = topology.attention.layer_uses_sliding_window[layer]
                ? topology.attention.sliding_rows[row] : topology.attention.global_rows[row];
            if (topology.attention.per_layer_attention_counts[
                    layer * VerifierRowMax + row
                ] != schedule.base_token_count + schedule.ancestor_count) {
                throw std::invalid_argument("target verifier per-layer attention count is stale");
            }
        }
    }
    if (expected_ancestor_offset != topology.attention.ancestor_entry_count) {
        throw std::invalid_argument("target verifier ancestor arena has stale entries");
    }
}

}  // namespace

VerifierTopologyDescriptor make_verifier_topology_descriptor(
    const ValidatedProposalGraph& graph,
    const ChainVerifierSchedule& chain_schedule,
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period
) {
    require_model_shape(layer_count, context_length, sliding_window);
    const auto& source = graph.storage();
    const auto& header = source.header;
    if (header.materialized_node_count == 0U
        || header.materialized_node_count > VerifierRowMax) {
        throw std::invalid_argument("proposal graph has no verifier rows or exceeds VerifierRowMax");
    }

    VerifierTopologyDescriptor topology;
    topology.kind = header.kind == ProposalGraphKind::repeated_root_beam_batch
        ? VerifierTopologyKind::repeated_root_beam_batch
        : VerifierTopologyKind::unique_root_tree;
    topology.proposal_kind = header.kind;
    topology.graph_identity = header.graph_identity;
    topology.device_arena_epoch = header.device_arena_epoch;
    topology.horizon = header.horizon;
    topology.input_row_count = header.materialized_node_count;
    topology.materialized_edge_count = header.materialized_edge_count;
    topology.output_frontier_edge_count = header.output_frontier_edge_count;
    topology.proposal_edge_count = header.materialized_edge_count
        + header.output_frontier_edge_count;
    topology.root_count = header.root_count;
    topology.root_indices = header.root_indices;
    topology.repeated_beam_width = header.repeated_beam_width;
    topology.repeated_beam_length = header.repeated_beam_length;
    topology.has_source_budget = header.has_source_budget;
    topology.source_budget = header.source_budget;
    topology.resolved_budget = header.resolved_budget;
    topology.materialized_edges = source.materialized_edges;
    topology.output_frontier_edges = source.output_frontier_edges;
    for (std::uint32_t row = 0; row < topology.input_row_count; ++row) {
        topology.token_ids[row] = source.nodes.token_ids[row];
        topology.source_node_indices[row] = row;
        topology.parent_indices[row] = source.nodes.parent_indices[row];
        topology.depths[row] = source.nodes.depths[row];
        topology.causal_history_counts[row] = source.nodes.depths[row];
        topology.absolute_positions[row] = source.nodes.absolute_positions[row];
        topology.component_roots[row] = source.nodes.mask_component_roots[row];
        topology.output_roles[row] = source.nodes.output_roles[row];
        topology.selection_lanes[row] = static_cast<std::uint16_t>(row);
    }

    if (header.kind == ProposalGraphKind::chain) {
        validate_chain_verifier_schedule(chain_schedule, header.horizon);
        topology.tile_count = chain_schedule.tile_count;
        auto first_row = std::uint32_t {0};
        for (std::uint32_t tile = 0; tile < chain_schedule.tile_count; ++tile) {
            const auto& input = chain_schedule.tiles[tile];
            if (first_row + input.input_row_count > topology.input_row_count) {
                throw std::invalid_argument("chain verifier tile exceeds materialized rows");
            }
            topology.tiles[tile] = VerifierTile {
                .first_row = static_cast<std::uint16_t>(first_row),
                .row_count = input.input_row_count,
                .proposal_edge_count = input.validated_edge_count,
            };
            first_row += input.validated_edge_count;
        }
    } else {
        if (chain_schedule.tile_count != 0U) {
            throw std::invalid_argument("forest verifier topology cannot carry a chain tile schedule");
        }
        topology.tile_count = 1U;
        topology.tiles[0] = VerifierTile {
            .first_row = 0U,
            .row_count = static_cast<std::uint16_t>(topology.input_row_count),
            .proposal_edge_count = topology.proposal_edge_count,
        };
    }

    populate_attention_schedule(topology, layer_count, sliding_window, global_layer_period);
    validate_topology_again(
        topology, std::numeric_limits<std::uint32_t>::max(), context_length, sliding_window, global_layer_period
    );
    return topology;
}

ValidatedVerifierEncodeRequest::ValidatedVerifierEncodeRequest(
    const VerifierEncodeRequest& request,
    std::array<std::uint32_t, 5> validation_shape
) : request_(&request), validation_shape_(validation_shape) {}

const VerifierEncodeRequest& ValidatedVerifierEncodeRequest::request() const noexcept {
    return *request_;
}

bool ValidatedVerifierEncodeRequest::validated_for(const GemmaModelConfig& target) const noexcept {
    return validation_shape_ == std::array<std::uint32_t, 5> {
        target.vocab_size, target.block_count, target.context_length,
        target.sliding_window, target.attention_global_layer_period,
    };
}

ValidatedVerifierEncodeRequest validate_verifier_encode_request(
    const VerifierEncodeRequest& request,
    std::uint32_t vocabulary_size,
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period
) {
    require_model_shape(layer_count, context_length, sliding_window);
    validate_topology_again(request.topology, vocabulary_size, context_length, sliding_window, global_layer_period);
    if (request.topology.attention.layer_count != layer_count) {
        throw std::invalid_argument("target verifier topology layer count does not match target plan");
    }
    if (request.selection_request_count != request.topology.input_row_count) {
        throw std::invalid_argument("target verifier requires one selection request per row");
    }

    const auto empty_history = SamplerHistorySnapshot {
        .window_limit = static_cast<std::uint32_t>(request.sampler_options.repetition_window),
    };
    static_cast<void>(make_token_selection_request(request.sampler_options, empty_history));
    if (request.input_residency == VerifierInputResidency::device_resident) {
        if (!request.device_inputs.has_value()) {
            throw std::invalid_argument("target verifier device residency requires an input binding");
        }
        validate_device_binding(*request.device_inputs, request.topology);
    } else if (request.device_inputs.has_value()) {
        throw std::invalid_argument("host-staged target verifier cannot carry a device input binding");
    }

    const auto& root_history = request.selection_requests[0].history;
    for (std::uint32_t row = 0; row < request.selection_request_count; ++row) {
        const auto& selection = request.selection_requests[row];
        static_cast<void>(make_token_selection_request(request.sampler_options, selection));
        if (!same_history_snapshot(selection.history, root_history)) {
            throw std::invalid_argument("target verifier rows do not share the authoritative root history");
        }
        const auto history_count = static_cast<std::uint32_t>(
            request.topology.causal_history_counts[row]
        );
        if (selection.suffix_count != history_count) {
            throw std::invalid_argument("target verifier selection suffix does not match row depth");
        }
        const auto local_path_count = static_cast<std::uint32_t>(
            request.topology.depths[row] - request.topology.root_source_depth
        ) + (request.topology.root_source_depth == 0U ? 0U : 1U);
        auto node = row;
        for (std::uint32_t reverse = local_path_count; reverse > 0U; --reverse) {
            const auto suffix_index = history_count - (local_path_count - reverse) - 1U;
            if (selection.suffix_token_ids[suffix_index] != request.topology.token_ids[node]) {
                throw std::invalid_argument("target verifier selection suffix does not match its causal path");
            }
            node = request.topology.parent_indices[node];
        }
        if (request.topology.root_source_depth == 0U
            && request.topology.depths[node] != 0U) {
            throw std::invalid_argument("target verifier selection suffix did not reach a root");
        }
    }
    return ValidatedVerifierEncodeRequest(request, {
        vocabulary_size, layer_count, context_length, sliding_window, global_layer_period,
    });
}

void make_chain_verifier_tile_request(
    VerifierEncodeRequest& output,
    const ValidatedVerifierEncodeRequest& full_request,
    std::uint32_t tile_index,
    std::uint32_t layer_count,
    std::uint32_t context_length,
    std::uint32_t sliding_window,
    std::uint32_t global_layer_period
) {
    require_model_shape(layer_count, context_length, sliding_window);
    if (output.selection_request_count != 0U || output.topology.input_row_count != 0U
        || output.device_inputs.has_value()) {
        throw std::invalid_argument("target verifier tile destination is not empty");
    }
    const auto& full = full_request.request();
    const auto& source = full.topology;
    if (source.proposal_kind != ProposalGraphKind::chain
        || source.kind != VerifierTopologyKind::unique_root_tree
        || tile_index >= source.tile_count) {
        throw std::invalid_argument("target verifier tile view requires a valid chain tile");
    }
    const auto tile = source.tiles[tile_index];
    const auto first = static_cast<std::uint32_t>(tile.first_row);
    const auto rows = static_cast<std::uint32_t>(tile.row_count);
    const auto proposal_edges = tile.proposal_edge_count;
    if (rows == 0U || first > source.input_row_count
        || rows > source.input_row_count - first
        || proposal_edges < rows - 1U || proposal_edges > rows) {
        throw std::invalid_argument("target verifier chain tile extent is invalid");
    }

    auto& topology = output.topology;
    topology.kind = VerifierTopologyKind::unique_root_tree;
    topology.proposal_kind = ProposalGraphKind::chain;
    topology.graph_identity = source.graph_identity;
    topology.device_arena_epoch = source.device_arena_epoch;
    topology.horizon = source.horizon;
    topology.input_row_count = rows;
    topology.proposal_edge_count = proposal_edges;
    topology.materialized_edge_count = rows - 1U;
    topology.output_frontier_edge_count = proposal_edges - topology.materialized_edge_count;
    topology.root_count = 1U;
    topology.root_indices[0] = 0U;
    topology.root_source_depth = source.depths[first];
    topology.has_source_budget = source.has_source_budget;
    topology.source_budget = source.source_budget;
    topology.resolved_budget = source.resolved_budget;
    topology.tile_count = 1U;
    topology.tiles[0] = VerifierTile {
        .first_row = 0U,
        .row_count = static_cast<std::uint16_t>(rows),
        .proposal_edge_count = proposal_edges,
    };
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto source_row = first + row;
        topology.token_ids[row] = source.token_ids[source_row];
        topology.source_node_indices[row] = source.source_node_indices[source_row];
        topology.parent_indices[row] = row == 0U ? proposal_node_none : row - 1U;
        topology.depths[row] = source.depths[source_row];
        topology.causal_history_counts[row] = source.causal_history_counts[source_row];
        topology.absolute_positions[row] = source.absolute_positions[source_row];
        topology.component_roots[row] = 0U;
        topology.output_roles[row] = source.output_roles[source_row];
        topology.selection_lanes[row] = static_cast<std::uint16_t>(row);
        output.selection_requests[row] = full.selection_requests[source_row];
    }
    output.selection_request_count = rows;
    output.sampler_options = full.sampler_options;
    output.input_residency = full.input_residency;
    if (full.device_inputs.has_value()) {
        output.device_inputs = *full.device_inputs;
        if (first > std::numeric_limits<std::uint32_t>::max()
                - output.device_inputs->first_token
            || first > std::numeric_limits<std::uint32_t>::max()
                - output.device_inputs->first_request) {
            throw std::overflow_error("target verifier tile device offset overflows");
        }
        output.device_inputs->first_token += first;
        output.device_inputs->first_request += first;
    }

    const auto find_materialized = [&](std::uint32_t parent, std::uint32_t child) {
        for (std::uint32_t edge = 0U; edge < source.materialized_edge_count; ++edge) {
            if (source.materialized_edges.parent_node_indices[edge] == parent
                && source.materialized_edges.child_node_indices[edge] == child) {
                return edge;
            }
        }
        throw std::invalid_argument("target verifier chain tile lost a materialized edge");
    };
    for (std::uint32_t edge = 0U; edge < topology.materialized_edge_count; ++edge) {
        const auto source_edge = find_materialized(first + edge, first + edge + 1U);
        topology.materialized_edges.parent_node_indices[edge] = edge;
        topology.materialized_edges.child_node_indices[edge] = edge + 1U;
        topology.materialized_edges.token_ids[edge] =
            source.materialized_edges.token_ids[source_edge];
        topology.materialized_edges.conditional_confidences[edge] =
            source.materialized_edges.conditional_confidences[source_edge];
        topology.materialized_edges.sibling_ranks[edge] =
            source.materialized_edges.sibling_ranks[source_edge];
        topology.materialized_edges.proposal_ticket_offsets[edge] =
            source.materialized_edges.proposal_ticket_offsets[source_edge];
    }
    if (topology.output_frontier_edge_count != 0U) {
        const auto source_parent = first + rows - 1U;
        auto copied = false;
        if (source_parent + 1U < source.input_row_count) {
            const auto source_edge = find_materialized(source_parent, source_parent + 1U);
            topology.output_frontier_edges.parent_node_indices[0] = rows - 1U;
            topology.output_frontier_edges.token_ids[0] =
                source.materialized_edges.token_ids[source_edge];
            topology.output_frontier_edges.conditional_confidences[0] =
                source.materialized_edges.conditional_confidences[source_edge];
            topology.output_frontier_edges.sibling_ranks[0] =
                source.materialized_edges.sibling_ranks[source_edge];
            topology.output_frontier_edges.proposal_ticket_offsets[0] =
                source.materialized_edges.proposal_ticket_offsets[source_edge];
            copied = true;
        }
        for (std::uint32_t edge = 0U; !copied && edge < source.output_frontier_edge_count;
             ++edge) {
            if (source.output_frontier_edges.parent_node_indices[edge] != source_parent) continue;
            topology.output_frontier_edges.parent_node_indices[0] = rows - 1U;
            topology.output_frontier_edges.token_ids[0] =
                source.output_frontier_edges.token_ids[edge];
            topology.output_frontier_edges.conditional_confidences[0] =
                source.output_frontier_edges.conditional_confidences[edge];
            topology.output_frontier_edges.sibling_ranks[0] =
                source.output_frontier_edges.sibling_ranks[edge];
            topology.output_frontier_edges.proposal_ticket_offsets[0] =
                source.output_frontier_edges.proposal_ticket_offsets[edge];
            copied = true;
        }
        if (!copied) {
            throw std::invalid_argument("target verifier chain tile lost its output-frontier edge");
        }
    }
    populate_attention_schedule(topology, layer_count, sliding_window, global_layer_period);
    validate_topology_again(
        topology, std::numeric_limits<std::uint32_t>::max(), context_length, sliding_window, global_layer_period
    );
}

// Defined with the Metal command implementation, allocated only during plan setup.
std::shared_ptr<TargetVerifierRuntimeState> make_target_verifier_runtime_state(bool precise_math);

Gemma2MetalTargetVerifierPlan::Gemma2MetalTargetVerifierPlan(
    Gemma2MetalTargetVerifierScratch scratch,
    std::uint32_t row_capacity,
    std::uint32_t layer_count,
    std::uint32_t vocabulary_size,
    std::uintptr_t target_weight_identity,
    std::size_t allocated_byte_count,
    bool precise_math
) : storage_(std::make_shared<Storage>(Storage {
        .runtime_ = make_target_verifier_runtime_state(precise_math),
        .scratch_ = std::move(scratch),
        .row_capacity_ = row_capacity,
        .layer_count_ = layer_count,
        .vocabulary_size_ = vocabulary_size,
        .target_weight_identity_ = target_weight_identity,
        .allocated_byte_count_ = allocated_byte_count,
        .completion_candidates_ = std::vector<CompactTokenCandidates>(row_capacity),
    })) {}

std::size_t Gemma2MetalTargetVerifierPlan::host_storage_byte_count() noexcept {
    return sizeof(Storage);
}

const Gemma2MetalTargetVerifierScratch& Gemma2MetalTargetVerifierPlan::scratch() const noexcept {
    return storage_->scratch_;
}

std::uint32_t Gemma2MetalTargetVerifierPlan::row_capacity() const noexcept { return storage_->row_capacity_; }
std::uint32_t Gemma2MetalTargetVerifierPlan::layer_count() const noexcept { return storage_->layer_count_; }
std::uint32_t Gemma2MetalTargetVerifierPlan::vocabulary_size() const noexcept {
    return storage_->vocabulary_size_;
}
std::uintptr_t Gemma2MetalTargetVerifierPlan::target_weight_identity() const noexcept {
    return storage_->target_weight_identity_;
}
std::size_t Gemma2MetalTargetVerifierPlan::allocated_byte_count() const noexcept {
    return storage_->allocated_byte_count_;
}
std::span<CompactTokenCandidates>
Gemma2MetalTargetVerifierPlan::completion_candidates() noexcept {
    return storage_->completion_candidates_;
}

std::size_t estimate_target_verifier_bytes(const GemmaModelConfig& config, std::uint32_t row_capacity) {
    // An ordinary workspace may have a runtime context shorter than the
    // model's local window. It never creates ancestry; topology admission
    // below the public verifier boundary retains its stricter window check.
    require_model_shape(config.block_count, config.context_length, config.sliding_window, false);
    if (row_capacity == 0U || row_capacity > VerifierRowMax)
        throw std::invalid_argument("target verifier row capacity must be in [1, 1025]");
    std::size_t bytes = 0;
    const auto add = [&](std::initializer_list<std::uint64_t> shape, DType dtype = DType::f32) {
        const auto count = TensorDescriptor::from(Shape::from(shape), dtype).byte_count;
        if (count > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("target verifier allocation byte count overflow");
        bytes += count;
    };
    const auto rows = std::uint64_t(row_capacity);
    const auto accepted = std::uint64_t(std::bit_ceil(std::min(row_capacity,
        std::uint32_t(external_speculation_max_horizon + 1U))));
    const auto kv = gemma_kv_dtype(config);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        add({rows, config.attention_head_count_kv, config.attention_key_length}, kv);
        add({rows, config.attention_head_count_kv, config.attention_value_length}, kv);
        add({accepted, config.attention_head_count_kv, config.attention_key_length}, kv);
        add({accepted, config.attention_head_count_kv, config.attention_value_length}, kv);
    }
    constexpr auto batches = target_selection_batch_rows;
    const auto tiles = (std::uint64_t(config.vocab_size) + gemma2_metal_selection_tile_size - 1U)
        / gemma2_metal_selection_tile_size;
    add({batches, config.vocab_size});
    add({2U * batches, tiles, token_sampler_max_base_candidates, sizeof(TokenLogit) / sizeof(float)});
    add({rows, 2U + token_sampler_max_candidates * sizeof(TokenLogit) / sizeof(float)});
    add({rows, sizeof(VerifierDeviceSelectionRow) / sizeof(float)});
    add({rows, 4U + token_sampler_max_repetition_window});
    add({rows, token_sampler_max_repetition_window});
    add({3U, rows}); // Token, expected-token and position staging.
    add({config.block_count, rows, 4U});
    add({verifier_ancestor_entry_max});
    add({std::max(rows, accepted)});
    add({1U});
    add({6U, rows, config.embedding_length});
    add({2U, rows, config.attention_head_count, config.attention_key_length});
    add({rows, config.attention_head_count_kv, config.attention_key_length});
    add({rows, config.attention_head_count_kv, config.attention_value_length});
    add({rows, gemma2_decode_attention_max_partitions, config.attention_head_count,
        std::uint64_t(config.attention_value_length) + 2U});
    add({rows, config.attention_head_count, config.attention_value_length});
    add({3U, rows, config.feed_forward_length});
    if (const auto sums = q4_input_sums_descriptor(config, row_capacity))
        add({sums->shape.dimension(0), sums->shape.dimension(1)});
    return bytes;
}

Gemma2MetalTargetVerifierPlan create_gemma2_metal_target_verifier_plan(
    const Gemma2MetalModelPlan& target,
    std::uint32_t row_capacity
) {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::allocation);
#endif
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) {
        throw runtime::BackendError(
            runtime::BackendErrorCode::unavailable_metal_device,
            runtime::BackendKind::metal,
            "target verifier requires an available Metal device"
        );
    }
    if (target.output_head_policy() != GemmaOutputHeadPolicy::q8_0
        && target.output_head_policy() != GemmaOutputHeadPolicy::q6_k) {
        throw std::invalid_argument("target verifier requires a native Q8 or Q6 output head");
    }
    if (row_capacity == 0U || row_capacity > VerifierRowMax) {
        throw std::invalid_argument("target verifier row capacity must be in [1, 1025]");
    }
    const auto& config = target.config();
    // An ordinary workspace may have a runtime context shorter than the
    // model's local window. It never creates ancestry; topology admission
    // below the public verifier boundary retains its stricter window check.
    require_model_shape(config.block_count, config.context_length, config.sliding_window, false);
    if (config.attention_head_count != 2U * config.attention_head_count_kv
        || config.attention_key_length != 256U
        || config.attention_value_length != config.attention_key_length) {
        throw std::invalid_argument(
            "target verifier requires the retained Gemma2 grouped-attention geometry"
        );
    }
    const auto query_width = static_cast<std::uint64_t>(config.attention_head_count)
        * config.attention_key_length;
    const auto key_width = static_cast<std::uint64_t>(config.attention_head_count_kv)
        * config.attention_key_length;
    const auto value_width = static_cast<std::uint64_t>(config.attention_head_count_kv)
        * config.attention_value_length;
    const auto attention_width = static_cast<std::uint64_t>(config.attention_head_count)
        * config.attention_value_length;
    const auto rows = static_cast<std::uint64_t>(row_capacity);
    const auto partitions = static_cast<std::uint64_t>(gemma2_decode_attention_max_partitions);
    const auto query_heads = static_cast<std::uint64_t>(config.attention_head_count);
    const auto value_dimension = static_cast<std::uint64_t>(config.attention_value_length);
    const auto selection_batches = static_cast<std::uint64_t>(target_selection_batch_rows);
    const auto selection_tiles = (static_cast<std::uint64_t>(config.vocab_size)
        + gemma2_metal_selection_tile_size - 1U) / gemma2_metal_selection_tile_size;
    const auto candidate_f32 = sizeof(TokenLogit) / sizeof(float);
    const auto result_f32 = 2U + token_sampler_max_candidates * candidate_f32;
    const auto request_f32 = sizeof(VerifierDeviceSelectionRow) / sizeof(float);
    constexpr std::uint64_t resolved_request_f32 = 4U + token_sampler_max_repetition_window;

    const auto kv_descriptor = [&](std::initializer_list<std::uint64_t> shape) {
        return TensorDescriptor::from(Shape::from(shape), target.kv_cache().front().key.descriptor().dtype);
    };
    std::size_t allocated = 0;
    std::vector<Gemma2MetalTargetVerifierKVLayer> kv_layers;
    kv_layers.reserve(config.block_count);
    // A selected path cannot outgrow its arena or the controller's horizon.
    // The ordinary 128-row workspace retains its existing 32-slot allocation.
    const auto accepted_capacity = static_cast<std::uint64_t>(std::bit_ceil(std::min(
        row_capacity, static_cast<std::uint32_t>(external_speculation_max_horizon + 1U))));
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        kv_layers.push_back(Gemma2MetalTargetVerifierKVLayer {
            .private_key = allocate_metal(kv_descriptor({rows, key_width}), allocated),
            .private_value = allocate_metal(kv_descriptor({rows, value_width}), allocated),
            .accepted_backup_key = allocate_metal(
                kv_descriptor({
                    accepted_capacity, config.attention_head_count_kv,
                    config.attention_key_length,
                }),
                allocated
            ),
            .accepted_backup_value = allocate_metal(
                kv_descriptor({
                    accepted_capacity, config.attention_head_count_kv,
                    config.attention_value_length,
                }),
                allocated
            ),
        });
    }

    TargetSelectionWorkspace selection {
        .logits_tile = allocate_metal(
            f32_descriptor({selection_batches, config.vocab_size}), allocated
        ),
        .candidate_runs_a = allocate_metal(f32_descriptor({
            selection_batches * selection_tiles * token_sampler_max_base_candidates
                * candidate_f32,
        }), allocated),
        .candidate_runs_b = allocate_metal(f32_descriptor({
            selection_batches * selection_tiles * token_sampler_max_base_candidates
                * candidate_f32,
        }), allocated),
        .compact_results = allocate_metal(f32_descriptor({rows * result_f32}), allocated),
        .raw_request_staging = allocate_metal(f32_descriptor({rows * request_f32}), allocated),
        .resolved_request_staging = allocate_metal(
            f32_descriptor({rows * resolved_request_f32}), allocated
        ),
        .history_token_staging = allocate_metal(f32_descriptor({
            rows * token_sampler_max_repetition_window,
        }), allocated),
        .row_capacity = row_capacity,
        .batch_row_capacity = static_cast<std::uint32_t>(target_selection_batch_rows),
    };

    std::optional<tensor::BackendBuffer> q4_input_sums;
    if (const auto descriptor = q4_input_sums_descriptor(config, row_capacity))
        q4_input_sums = allocate_metal(*descriptor, allocated);
    Gemma2MetalTargetVerifierScratch scratch {
        .token_id_staging = allocate_metal(f32_descriptor({rows}), allocated),
        .expected_token_id_staging = allocate_metal(f32_descriptor({rows}), allocated),
        .absolute_position_staging = allocate_metal(f32_descriptor({rows}), allocated),
        .attention_schedule_staging = allocate_metal(f32_descriptor({
            static_cast<std::uint64_t>(config.block_count) * rows * 4U,
        }), allocated),
        .ancestor_index_staging = allocate_metal(f32_descriptor({
            verifier_ancestor_entry_max,
        }), allocated),
        .selected_path_staging = allocate_metal(f32_descriptor({std::max(rows, accepted_capacity)}), allocated),
        .input_status = allocate_metal(f32_descriptor({1U}), allocated),
        .normalized_hidden = allocate_metal(f32_descriptor({rows, config.embedding_length}), allocated),
        .raw_query = allocate_metal(f32_descriptor({rows, query_width}), allocated),
        .raw_key = allocate_metal(f32_descriptor({rows, key_width}), allocated),
        .raw_value = allocate_metal(f32_descriptor({rows, value_width}), allocated),
        .query = allocate_metal(f32_descriptor({rows, query_width}), allocated),
        .attention_partial_output = allocate_metal(f32_descriptor({
            rows, partitions, query_heads, value_dimension,
        }), allocated),
        .attention_partial_maxima = allocate_metal(f32_descriptor({
            rows, partitions, query_heads,
        }), allocated),
        .attention_partial_denominators = allocate_metal(f32_descriptor({
            rows, partitions, query_heads,
        }), allocated),
        .attention_output = allocate_metal(f32_descriptor({rows, attention_width}), allocated),
        .projected_attention_output = allocate_metal(
            f32_descriptor({rows, config.embedding_length}), allocated
        ),
        .post_attention_residual = allocate_metal(
            f32_descriptor({rows, config.embedding_length}), allocated
        ),
        .normalized_feed_forward_input = allocate_metal(
            f32_descriptor({rows, config.embedding_length}), allocated
        ),
        .feed_forward_gate = allocate_metal(
            f32_descriptor({rows, config.feed_forward_length}), allocated
        ),
        .feed_forward_up = allocate_metal(
            f32_descriptor({rows, config.feed_forward_length}), allocated
        ),
        .feed_forward_gated = allocate_metal(
            f32_descriptor({rows, config.feed_forward_length}), allocated
        ),
        .projected_feed_forward_output = allocate_metal(
            f32_descriptor({rows, config.embedding_length}), allocated
        ),
        .feed_forward_residual = allocate_metal(
            f32_descriptor({rows, config.embedding_length}), allocated
        ),
        .q4_input_sums = std::move(q4_input_sums),
        .kv_layers = std::move(kv_layers),
        .selection = std::move(selection),
    };
    return Gemma2MetalTargetVerifierPlan(
        std::move(scratch), row_capacity, config.block_count, config.vocab_size,
        target.allocation_identities()[0], allocated, config.architecture == ModelArchitecture::gemma3
    );
}

}  // namespace local_gguf_engine::engine
