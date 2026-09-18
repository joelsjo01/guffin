#include "local_gguf_engine/engine/target_verifier.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace local_gguf_engine::engine {
namespace {

using EpisodeClock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t elapsed_ns(
    EpisodeClock::time_point begin,
    EpisodeClock::time_point end
) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
    );
}

struct ActiveChainTile {
    std::array<std::uint32_t, external_speculation_max_horizon + 1U> token_ids {};
    std::array<std::uint32_t, external_speculation_max_horizon + 1U> source_node_indices {};
    std::uint32_t input_row_count = 0U;
    std::uint32_t root_source_depth = 0U;
};

template <class Value>
void checked_add(Value& destination, Value value, const char* label) {
    if (value > std::numeric_limits<Value>::max() - destination) {
        throw std::overflow_error(label);
    }
    destination += value;
}

void accumulate_target_execution(
    TargetVerifierExecutionStats& total,
    const TargetVerifierExecutionStats& value,
    bool include_encoded_shape = true
) {
#define ADD_TARGET_FIELD(field) \
    checked_add(total.field, value.field, "target verifier episode " #field " overflow")
    ADD_TARGET_FIELD(command_buffer_count);
    ADD_TARGET_FIELD(commit_count);
    ADD_TARGET_FIELD(synchronous_wait_count);
    ADD_TARGET_FIELD(compute_encoder_count);
    ADD_TARGET_FIELD(compute_dispatch_count);
    ADD_TARGET_FIELD(blit_encoder_count);
    ADD_TARGET_FIELD(memory_barrier_count);
    ADD_TARGET_FIELD(transient_buffer_allocation_count);
    ADD_TARGET_FIELD(selector_dispatch_count);
    ADD_TARGET_FIELD(compact_readback_bytes);
    if (include_encoded_shape) {
        ADD_TARGET_FIELD(input_row_count);
        ADD_TARGET_FIELD(proposal_edge_count);
        ADD_TARGET_FIELD(attention_dispatch_count);
        ADD_TARGET_FIELD(attention_partition_count);
        ADD_TARGET_FIELD(attention_token_count);
    }
    ADD_TARGET_FIELD(gpu_duration_ns);
    ADD_TARGET_FIELD(q4_decode_projection_count);
    ADD_TARGET_FIELD(q8_decode_projection_count);
    ADD_TARGET_FIELD(q8_width_two_projection_count);
    ADD_TARGET_FIELD(q4_small_batch_projection_count);
    ADD_TARGET_FIELD(q4_width_two_projection_count);
    ADD_TARGET_FIELD(q4_exact_16_projection_count);
    ADD_TARGET_FIELD(q4_generic_tiled_projection_count);
    ADD_TARGET_FIELD(q6_decode_projection_count);
    ADD_TARGET_FIELD(q6_small_batch_projection_count);
    ADD_TARGET_FIELD(q6_width_two_projection_count);
    ADD_TARGET_FIELD(q6_exact_16_projection_count);
    ADD_TARGET_FIELD(q6_generic_tiled_projection_count);
    ADD_TARGET_FIELD(f16_projection_count);
#undef ADD_TARGET_FIELD
}

void accumulate_endpoint_execution(
    Gemma2MetalDecodeExecutionStats& total,
    const Gemma2MetalDecodeExecutionStats& value
) {
#define ADD_ENDPOINT_FIELD(field) \
    checked_add(total.field, value.field, "target verifier endpoint " #field " overflow")
    ADD_ENDPOINT_FIELD(command_buffer_count);
    ADD_ENDPOINT_FIELD(commit_count);
    ADD_ENDPOINT_FIELD(synchronous_wait_count);
    ADD_ENDPOINT_FIELD(compute_encoder_count);
    ADD_ENDPOINT_FIELD(compute_dispatch_count);
    ADD_ENDPOINT_FIELD(blit_encoder_count);
    ADD_ENDPOINT_FIELD(memory_barrier_count);
    ADD_ENDPOINT_FIELD(transient_buffer_allocation_count);
    ADD_ENDPOINT_FIELD(selector_dispatch_count);
    ADD_ENDPOINT_FIELD(compact_readback_bytes);
    ADD_ENDPOINT_FIELD(candidate_count);
    ADD_ENDPOINT_FIELD(gpu_duration_ns);
    ADD_ENDPOINT_FIELD(q4_decode_projection_count);
    ADD_ENDPOINT_FIELD(q8_decode_projection_count);
    ADD_ENDPOINT_FIELD(q8_width_two_projection_count);
    ADD_ENDPOINT_FIELD(q4_small_batch_projection_count);
    ADD_ENDPOINT_FIELD(q4_width_two_projection_count);
    ADD_ENDPOINT_FIELD(q4_exact_16_projection_count);
    ADD_ENDPOINT_FIELD(q4_generic_tiled_projection_count);
    ADD_ENDPOINT_FIELD(q6_decode_projection_count);
    ADD_ENDPOINT_FIELD(q6_small_batch_projection_count);
    ADD_ENDPOINT_FIELD(q6_width_two_projection_count);
    ADD_ENDPOINT_FIELD(q6_exact_16_projection_count);
    ADD_ENDPOINT_FIELD(q6_generic_tiled_projection_count);
    ADD_ENDPOINT_FIELD(f16_projection_count);
    ADD_ENDPOINT_FIELD(output_head_elision_count);
    ADD_ENDPOINT_FIELD(output_elided_weight_bytes);
    ADD_ENDPOINT_FIELD(decode_attention_dispatch_count);
    ADD_ENDPOINT_FIELD(decode_attention_partition_count);
    ADD_ENDPOINT_FIELD(decode_attention_token_count);
    ADD_ENDPOINT_FIELD(fused_embedding_norm_count);
    ADD_ENDPOINT_FIELD(direct_qkv_cache_count);
    ADD_ENDPOINT_FIELD(dual_norm_bridge_count);
    ADD_ENDPOINT_FIELD(fused_gate_up_gelu_count);
#undef ADD_ENDPOINT_FIELD
}

void validate_chain_episode(
    const Gemma2MetalModelPlan& target,
    const Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& full_request,
    const SpeculativeControllerScenario& scenario,
    const TokenSampler& authoritative_sampler
) {
    if (scenario.graph == nullptr || scenario.target_sample_count != 0U
        || !authoritative_sampler.checkpoint().same_state(scenario.origin_checkpoint)) {
        throw std::invalid_argument(
            "target verifier chain episode requires an unsampled scenario at its authoritative checkpoint"
        );
    }
    const auto& graph = scenario.graph->storage();
    const auto& topology = full_request.request().topology;
    if (graph.header.kind != ProposalGraphKind::chain
        || topology.proposal_kind != ProposalGraphKind::chain
        || topology.kind != VerifierTopologyKind::unique_root_tree
        || graph.header.graph_identity != topology.graph_identity
        || graph.header.device_arena_epoch != topology.device_arena_epoch
        || graph.header.horizon != topology.horizon
        || graph.header.materialized_node_count != topology.input_row_count
        || scenario.chain_schedule.tile_count != topology.tile_count
        || topology.absolute_positions[0] != scenario.initial_logical_cached_count) {
        throw std::invalid_argument(
            "target verifier chain episode graph, request, schedule, or cache origin is incompatible"
        );
    }
    if (scenario.available_context_rows
        > target.config().context_length - topology.absolute_positions[0]) {
        throw std::invalid_argument("target verifier chain episode overstates available context");
    }
    for (std::uint32_t tile = 0U; tile < topology.tile_count; ++tile) {
        const auto& controller_tile = scenario.chain_schedule.tiles[tile];
        const auto& target_tile = topology.tiles[tile];
        if (controller_tile.validated_edge_count != target_tile.proposal_edge_count
            || controller_tile.input_row_count != target_tile.row_count
            || target_tile.row_count > verifier.row_capacity()) {
            throw std::invalid_argument(
                "target verifier chain episode tile does not fit its validated target arena"
            );
        }
    }
}

[[nodiscard]] std::uint32_t retained_local_prefix(
    const SpeculativeControllerOutcome& outcome,
    const ActiveChainTile& tile
) {
    auto count = std::uint32_t {0U};
    for (std::uint32_t local = 0U; local < tile.input_row_count; ++local) {
        const auto source = tile.source_node_indices[local];
        const auto present = std::find(
            outcome.retained_row_nodes.begin(),
            outcome.retained_row_nodes.begin() + outcome.retained_row_count,
            source
        ) != outcome.retained_row_nodes.begin() + outcome.retained_row_count;
        if (!present) break;
        ++count;
    }
    for (std::uint32_t local = count; local < tile.input_row_count; ++local) {
        const auto source = tile.source_node_indices[local];
        if (std::find(
                outcome.retained_row_nodes.begin(),
                outcome.retained_row_nodes.begin() + outcome.retained_row_count,
                source
            ) != outcome.retained_row_nodes.begin() + outcome.retained_row_count) {
            throw std::logic_error("target verifier controller retained a non-prefix chain row");
        }
    }
    return count;
}

[[nodiscard]] std::uint32_t provisional_local_prefix(
    const SpeculativeControllerScenario& scenario,
    const ActiveChainTile& tile
) {
    auto count = std::uint32_t {1U};
    const auto first_depth = tile.root_source_depth;
    for (std::uint32_t local_edge = 0U;
         local_edge + 1U < tile.input_row_count;
         ++local_edge) {
        const auto sample_offset = first_depth + local_edge;
        if (sample_offset >= scenario.target_sample_count
            || scenario.target_samples[sample_offset].token_id
                != tile.token_ids[local_edge + 1U]) {
            break;
        }
        ++count;
    }
    return count;
}

void execute_terminal_target_actions(
    Gemma2MetalModelPlan& target,
    const SpeculativeControllerScenario& scenario,
    const SpeculativeControllerOutcome& outcome,
    TokenSampler& authoritative_sampler,
    float embedding_scale,
    TargetVerifierEpisodeResult& result
) {
    authoritative_sampler.adopt_checkpoint(outcome.adopted_checkpoint);
    for (std::uint32_t endpoint = 0U;
         endpoint < outcome.endpoint_kv_only_count;
         ++endpoint) {
        const auto position = scenario.initial_logical_cached_count
            + outcome.retained_row_count + endpoint;
        const auto begin = EpisodeClock::now();
        const auto execution = run_gemma2_metal_target_verifier_endpoint(
            target, outcome.endpoint_kv_only_tokens[endpoint], position,
            embedding_scale
        );
        checked_add(
            result.endpoint_wall_ns, elapsed_ns(begin, EpisodeClock::now()),
            "target verifier endpoint wall time overflow"
        );
        accumulate_endpoint_execution(result.endpoint_execution, execution);
        ++result.endpoint_count;
    }
    if (outcome.has_carry && !outcome.carry_is_cached) {
        const auto position = scenario.initial_logical_cached_count
            + outcome.retained_row_count;
        const auto begin = EpisodeClock::now();
        const auto fallback = run_gemma2_metal_resident_decode_step(
            target, create_gemma2_metal_decode_schedule(target, position),
            outcome.carry_token, embedding_scale,
            authoritative_sampler.selection_request()
        );
        if (!fallback.produced_logits || !fallback.candidates.has_value()) {
            throw std::logic_error(
                "target verifier ordinary carry fallback produced no target candidates"
            );
        }
        result.ordinary_fallback_candidates = *fallback.candidates;
        result.ordinary_fallback_execution = fallback.execution;
        checked_add(
            result.ordinary_fallback_wall_ns,
            elapsed_ns(begin, EpisodeClock::now()),
            "target verifier ordinary fallback wall time overflow"
        );
        result.executed_ordinary_fallback = true;
    }
}

struct ForestPath {
    std::array<std::uint32_t, external_speculation_max_horizon + 1U> nodes {};
    std::uint32_t count = 0U;
};

[[nodiscard]] ForestPath provisional_forest_path(
    const ProposalGraphStorage& graph,
    const SpeculativeControllerScenario& scenario
) {
    ForestPath path;
    if (graph.header.kind != ProposalGraphKind::repeated_root_beam_batch) {
        auto node = graph.header.root_indices[0];
        path.nodes[path.count++] = node;
        for (std::uint32_t sample = 0U; sample < scenario.target_sample_count; ++sample) {
            auto child = proposal_node_none;
            for (std::uint32_t edge = 0U; edge < graph.header.materialized_edge_count; ++edge) {
                if (graph.materialized_edges.parent_node_indices[edge] == node
                    && graph.materialized_edges.token_ids[edge]
                        == scenario.target_samples[sample].token_id) {
                    child = graph.materialized_edges.child_node_indices[edge];
                    break;
                }
            }
            if (child == proposal_node_none) break;
            if (path.count >= path.nodes.size()) {
                throw std::overflow_error("target verifier provisional forest path overflow");
            }
            path.nodes[path.count++] = child;
            node = child;
        }
        return path;
    }

    std::array<std::uint32_t, ReDrafterMLXRepeatedVerifierRootMax> active {};
    auto active_count = graph.header.root_count;
    std::copy_n(graph.header.root_indices.begin(), active_count, active.begin());
    auto matched_depth = std::uint32_t {0U};
    for (std::uint32_t sample = 0U; sample < scenario.target_sample_count; ++sample) {
        std::array<std::uint32_t, ReDrafterMLXRepeatedVerifierRootMax> next {};
        auto next_count = std::uint32_t {0U};
        for (std::uint32_t candidate = 0U; candidate < active_count; ++candidate) {
            for (std::uint32_t edge = 0U; edge < graph.header.materialized_edge_count; ++edge) {
                if (graph.materialized_edges.parent_node_indices[edge] == active[candidate]
                    && graph.materialized_edges.token_ids[edge]
                        == scenario.target_samples[sample].token_id) {
                    next[next_count++] = graph.materialized_edges.child_node_indices[edge];
                    break;
                }
            }
        }
        if (next_count == 0U) break;
        active = next;
        active_count = next_count;
        ++matched_depth;
    }
    auto node = matched_depth == 0U ? graph.header.root_indices[0] : active[0];
    path.count = matched_depth + 1U;
    for (std::uint32_t reverse = path.count; reverse > 0U; --reverse) {
        path.nodes[reverse - 1U] = node;
        node = graph.nodes.parent_indices[node];
    }
    return path;
}

[[nodiscard]] std::uint32_t local_row_for_source(
    const VerifierTopologyDescriptor& topology,
    std::uint32_t source_node
) {
    for (std::uint32_t row = 0U; row < topology.input_row_count; ++row) {
        if (topology.source_node_indices[row] == source_node) return row;
    }
    throw std::logic_error("target verifier source node is absent from its encoded topology");
}

}  // namespace

TargetVerifierEpisodeResult run_gemma2_metal_target_verifier_chain_episode(
    Gemma2MetalModelPlan& target,
    Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& full_request,
    const SpeculativeControllerScenario& scenario,
    TokenSampler& authoritative_sampler,
    TargetVerifierEpisodeWorkspace& workspace,
    float embedding_scale
) {
    if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0F) {
        throw std::invalid_argument(
            "target verifier chain episode embedding scale must be finite and positive"
        );
    }
    validate_chain_episode(target, verifier, full_request, scenario, authoritative_sampler);
    workspace.controller_scenario = scenario;
    workspace.controller_scenario.target_sample_count = 0U;

    const auto tickets = authoritative_sampler.preview_external_tickets();
    auto sampler_fork = authoritative_sampler.fork(scenario.origin_checkpoint);
    const auto& full_topology = full_request.request().topology;
    BoundedSpeculativeController controller;
    TargetVerifierEpisodeResult result;
    std::optional<TargetVerificationTicket> active_ticket;
    TargetVerifierCompletionResult active_completion;
    ActiveChainTile active_tile_metadata;
    auto active_tile = std::numeric_limits<std::uint32_t>::max();

    const auto run_controller = [&]() {
        const auto begin = EpisodeClock::now();
        auto progress = controller.run_available(workspace.controller_scenario);
        checked_add(
            result.controller_wall_ns, elapsed_ns(begin, EpisodeClock::now()),
            "target verifier controller wall time overflow"
        );
        return progress;
    };

    const auto commit_active = [&](
        std::uint32_t provisional_count,
        std::uint32_t retained_count
    ) {
        const auto begin = EpisodeClock::now();
        if (!active_ticket.has_value() || provisional_count == 0U
            || retained_count > provisional_count) {
            throw std::logic_error("target verifier chain episode commit extent is invalid");
        }
        std::array<std::uint32_t, external_speculation_max_horizon + 1U> local_path {};
        for (std::uint32_t row = 0U; row < provisional_count; ++row) {
            local_path[row] = row;
        }
        const auto provisional = provisionally_commit_gemma2_metal_target_verifier(
            *active_ticket, std::span<const std::uint32_t>(local_path).first(provisional_count)
        );
        accumulate_target_execution(result.target_execution, provisional.execution, false);
        checked_add(
            result.commit_gpu_ns, provisional.execution.gpu_duration_ns,
            "target verifier commit GPU time overflow"
        );
        const auto finalized = finalize_gemma2_metal_target_verifier(
            *active_ticket, retained_count
        );
        accumulate_target_execution(result.target_execution, finalized.execution, false);
        checked_add(
            result.commit_gpu_ns, finalized.execution.gpu_duration_ns,
            "target verifier finalize GPU time overflow"
        );
        active_ticket.reset();
        active_tile = std::numeric_limits<std::uint32_t>::max();
        active_completion = {};
        active_tile_metadata = {};
        checked_add(
            result.commit_wall_ns, elapsed_ns(begin, EpisodeClock::now()),
            "target verifier commit wall time overflow"
        );
    };

    while (true) {
        auto progress = run_controller();
        if (!progress.requires_target_sample) {
            if (active_ticket.has_value()) {
                const auto provisional = provisional_local_prefix(
                    workspace.controller_scenario, active_tile_metadata
                );
                const auto retained = retained_local_prefix(progress, active_tile_metadata);
                if (progress.next_candidate_lease.source
                    == TargetStateLeaseSource::selected_verifier_row) {
                    for (std::uint32_t row = 0U;
                         row < active_tile_metadata.input_row_count;
                         ++row) {
                        if (active_tile_metadata.source_node_indices[row]
                            == progress.next_candidate_lease.node_index) {
                            result.next_candidates = active_completion.candidates[row];
                            result.has_next_candidates = true;
                            break;
                        }
                    }
                    if (!result.has_next_candidates) {
                        throw std::logic_error(
                            "target verifier next-candidate lease is outside the completing tile"
                        );
                    }
                }
                commit_active(provisional, retained);
            }

            execute_terminal_target_actions(
                target, scenario, progress, authoritative_sampler,
                embedding_scale, result
            );
            result.controller = std::move(progress);
            return result;
        }

        const auto tile_index = static_cast<std::uint32_t>(
            progress.required_target_sample_tile
        );
        const auto sample_offset = progress.required_target_sample_offset;
        if (sample_offset != workspace.controller_scenario.target_sample_count
            || tile_index >= full_topology.tile_count) {
            throw std::logic_error("target verifier controller requested a non-causal sample");
        }

        if (active_ticket.has_value() && tile_index != active_tile) {
            const auto retained = retained_local_prefix(progress, active_tile_metadata);
            if (retained != active_tile_metadata.input_row_count
                || provisional_local_prefix(
                    workspace.controller_scenario, active_tile_metadata
                ) != active_tile_metadata.input_row_count) {
                throw std::logic_error(
                    "target verifier controller advanced past an uncommitted tile prefix"
                );
            }
            commit_active(active_tile_metadata.input_row_count, retained);
            authoritative_sampler.adopt_checkpoint(sampler_fork.checkpoint());
        }

        if (!active_ticket.has_value()) {
            const auto target_begin = EpisodeClock::now();
            workspace.tile_request.selection_request_count = 0U;
            workspace.tile_request.topology.input_row_count = 0U;
            workspace.tile_request.device_inputs.reset();
            make_chain_verifier_tile_request(
                workspace.tile_request, full_request, tile_index,
                target.config().block_count, target.config().context_length,
                target.config().sliding_window
            );
            active_tile_metadata.input_row_count =
                workspace.tile_request.topology.input_row_count;
            active_tile_metadata.root_source_depth =
                workspace.tile_request.topology.root_source_depth;
            std::copy_n(
                workspace.tile_request.topology.token_ids.begin(),
                active_tile_metadata.input_row_count,
                active_tile_metadata.token_ids.begin()
            );
            std::copy_n(
                workspace.tile_request.topology.source_node_indices.begin(),
                active_tile_metadata.input_row_count,
                active_tile_metadata.source_node_indices.begin()
            );
            auto validated_tile = validate_verifier_encode_request(
                workspace.tile_request, target.config().vocab_size,
                target.config().block_count, target.config().context_length,
                target.config().sliding_window
            );
            active_ticket.emplace(encode_gemma2_metal_target_verifier(
                target, verifier, validated_tile, embedding_scale
            ));
            submit_gemma2_metal_target_verifier(*active_ticket);
            active_completion = complete_gemma2_metal_target_verifier(*active_ticket);
            checked_add(
                result.target_wall_ns,
                elapsed_ns(target_begin, EpisodeClock::now()),
                "target verifier target wall time overflow"
            );
            accumulate_target_execution(result.target_execution, active_completion.execution);
            checked_add(
                result.target_gpu_ns, active_completion.execution.gpu_duration_ns,
                "target verifier target GPU time overflow"
            );
            ++result.executed_tile_count;
            active_tile = tile_index;
        }
        if (active_tile != tile_index) {
            throw std::logic_error("target verifier episode active tile identity drifted");
        }

        const auto first_depth = active_tile_metadata.root_source_depth;
        if (sample_offset < first_depth
            || sample_offset - first_depth >= active_completion.candidate_row_count) {
            throw std::logic_error("target verifier sample does not map to a tile output row");
        }
        const auto candidate_row = sample_offset - first_depth;
        const auto& candidates = active_completion.candidates[candidate_row];
        const auto selected = sampler_fork.sample_with_ticket(
            tickets.at(sample_offset),
            std::span<const TokenLogit>(candidates.values).first(candidates.count)
        );
        sampler_fork.observe_tokens(std::span<const std::uint32_t>(&selected, 1U));
        workspace.controller_scenario.target_samples[sample_offset] = ControllerTargetSample {
            .token_id = selected,
            .checkpoint = sampler_fork.checkpoint(),
        };
        ++workspace.controller_scenario.target_sample_count;
    }
}

TargetVerifierEpisodeResult run_gemma2_metal_target_verifier_forest_episode(
    Gemma2MetalModelPlan& target,
    Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& full_request,
    const SpeculativeControllerScenario& scenario,
    TokenSampler& authoritative_sampler,
    TargetVerifierEpisodeWorkspace& workspace,
    float embedding_scale
) {
    if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0F
        || scenario.graph == nullptr || scenario.target_sample_count != 0U
        || !authoritative_sampler.checkpoint().same_state(scenario.origin_checkpoint)) {
        throw std::invalid_argument(
            "target verifier forest episode requires an unsampled scenario at its authoritative checkpoint"
        );
    }
    const auto& graph = scenario.graph->storage();
    const auto& topology = full_request.request().topology;
    if (graph.header.kind == ProposalGraphKind::chain
        || topology.proposal_kind != graph.header.kind
        || topology.graph_identity != graph.header.graph_identity
        || topology.device_arena_epoch != graph.header.device_arena_epoch
        || topology.input_row_count != graph.header.materialized_node_count
        || topology.input_row_count > verifier.row_capacity()
        || topology.tile_count != 1U || scenario.chain_schedule.tile_count != 0U
        || topology.absolute_positions[topology.root_indices[0]]
            != scenario.initial_logical_cached_count) {
        throw std::invalid_argument(
            "target verifier forest episode graph, request, arena, or cache origin is incompatible"
        );
    }
    if (scenario.available_context_rows
        > target.config().context_length
            - topology.absolute_positions[topology.root_indices[0]]) {
        throw std::invalid_argument("target verifier forest episode overstates available context");
    }

    workspace.controller_scenario = scenario;
    workspace.controller_scenario.target_sample_count = 0U;
    BoundedSpeculativeController controller;
    TargetVerifierEpisodeResult result;
    const auto run_controller = [&]() {
        const auto begin = EpisodeClock::now();
        auto output = controller.run_available(workspace.controller_scenario);
        checked_add(
            result.controller_wall_ns, elapsed_ns(begin, EpisodeClock::now()),
            "target verifier controller wall time overflow"
        );
        return output;
    };
    auto progress = run_controller();
    const auto needs_leaf_candidate = !progress.requires_target_sample
        && progress.next_candidate_lease.source
            == TargetStateLeaseSource::selected_verifier_row;
    if (!progress.requires_target_sample && !needs_leaf_candidate) {
        execute_terminal_target_actions(
            target, scenario, progress, authoritative_sampler, embedding_scale, result
        );
        result.controller = std::move(progress);
        return result;
    }

    const auto target_begin = EpisodeClock::now();
    workspace.tile_request = full_request.request();
    auto validated = validate_verifier_encode_request(
        workspace.tile_request, target.config().vocab_size,
        target.config().block_count, target.config().context_length,
        target.config().sliding_window
    );
    auto ticket = encode_gemma2_metal_target_verifier(
        target, verifier, validated, embedding_scale
    );
    submit_gemma2_metal_target_verifier(ticket);
    const auto completion = complete_gemma2_metal_target_verifier(ticket);
    result.target_wall_ns = elapsed_ns(target_begin, EpisodeClock::now());
    accumulate_target_execution(result.target_execution, completion.execution);
    result.target_gpu_ns = completion.execution.gpu_duration_ns;
    result.executed_tile_count = 1U;

    const auto tickets = authoritative_sampler.preview_external_tickets();
    auto sampler_fork = authoritative_sampler.fork(scenario.origin_checkpoint);
    while (progress.requires_target_sample) {
        const auto sample_offset = progress.required_target_sample_offset;
        if (sample_offset != workspace.controller_scenario.target_sample_count
            || progress.required_target_sample_tile != 0U
            || progress.selected_count == 0U) {
            throw std::logic_error("target verifier forest controller requested a non-causal sample");
        }
        const auto parent_source = progress.selected_node_path[progress.selected_count - 1U];
        if (parent_source == proposal_node_none) {
            throw std::logic_error("target verifier forest requested a sample after leaving its tree");
        }
        const auto row = local_row_for_source(topology, parent_source);
        const auto& candidates = completion.candidates[row];
        const auto selected = sampler_fork.sample_with_ticket(
            tickets.at(sample_offset),
            std::span<const TokenLogit>(candidates.values).first(candidates.count)
        );
        sampler_fork.observe_tokens(std::span<const std::uint32_t>(&selected, 1U));
        workspace.controller_scenario.target_samples[sample_offset] = ControllerTargetSample {
            .token_id = selected,
            .checkpoint = sampler_fork.checkpoint(),
        };
        ++workspace.controller_scenario.target_sample_count;
        progress = run_controller();
    }

    if (progress.next_candidate_lease.source
        == TargetStateLeaseSource::selected_verifier_row) {
        const auto row = local_row_for_source(
            topology, progress.next_candidate_lease.node_index
        );
        result.next_candidates = completion.candidates[row];
        result.has_next_candidates = true;
    }

    auto provisional = provisional_forest_path(graph, workspace.controller_scenario);
    const auto retained_count = progress.retained_row_count;
    if (retained_count == 0U || retained_count > provisional.count) {
        throw std::logic_error("target verifier forest controller returned an invalid retained path");
    }
    auto prefix_compatible = true;
    for (std::uint32_t index = 0U; index < retained_count; ++index) {
        prefix_compatible = prefix_compatible
            && provisional.nodes[index] == progress.retained_row_nodes[index];
    }
    if (!prefix_compatible) {
        provisional.count = retained_count;
        std::copy_n(
            progress.retained_row_nodes.begin(), retained_count, provisional.nodes.begin()
        );
    }

    std::array<std::uint32_t, external_speculation_max_horizon + 1U> local_path {};
    for (std::uint32_t index = 0U; index < provisional.count; ++index) {
        local_path[index] = local_row_for_source(topology, provisional.nodes[index]);
    }
    const auto commit_begin = EpisodeClock::now();
    const auto provisional_commit = provisionally_commit_gemma2_metal_target_verifier(
        ticket, std::span<const std::uint32_t>(local_path).first(provisional.count)
    );
    accumulate_target_execution(result.target_execution, provisional_commit.execution, false);
    result.commit_gpu_ns = provisional_commit.execution.gpu_duration_ns;
    const auto finalized = finalize_gemma2_metal_target_verifier(ticket, retained_count);
    accumulate_target_execution(result.target_execution, finalized.execution, false);
    checked_add(
        result.commit_gpu_ns, finalized.execution.gpu_duration_ns,
        "target verifier forest finalize GPU time overflow"
    );
    result.commit_wall_ns = elapsed_ns(commit_begin, EpisodeClock::now());
    execute_terminal_target_actions(
        target, scenario, progress, authoritative_sampler, embedding_scale, result
    );
    result.controller = std::move(progress);
    return result;
}

}  // namespace local_gguf_engine::engine
