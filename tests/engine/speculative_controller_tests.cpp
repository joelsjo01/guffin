#include "speculative_controller_tests.hpp"

#include "local_gguf_engine/engine/speculative_controller.hpp"
#include "local_gguf_engine/engine/target_verifier_registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace local_gguf_engine::engine;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

constexpr std::uint32_t kRootToken = 100U;
constexpr std::uint32_t kRootPosition = 10U;
constexpr std::uint64_t kHistoryDigest = 55U;
constexpr std::uint32_t kHistoryLength = 5U;
constexpr std::uint64_t kArenaEpoch = 7U;

[[nodiscard]] ProposalGraphValidationContext validation_context() {
    return ProposalGraphValidationContext {
        .vocabulary_size = 1'000U,
        .context_length = 4'096U,
        .expected_root_token = kRootToken,
        .expected_root_position = kRootPosition,
        .expected_root_history = {.digest = kHistoryDigest, .length = kHistoryLength},
        .expected_device_arena_epoch = kArenaEpoch,
    };
}

void initialize_node(
    ProposalGraphStorage& graph,
    std::uint32_t node,
    std::uint32_t token,
    std::uint32_t parent,
    std::uint16_t depth,
    ProposalRowOutputRole role
) {
    graph.nodes.token_ids[node] = token;
    graph.nodes.parent_indices[node] = parent;
    graph.nodes.depths[node] = depth;
    graph.nodes.absolute_positions[node] = kRootPosition + depth;
    graph.nodes.mask_component_roots[node] = depth == 0U
        ? node : graph.nodes.mask_component_roots[parent];
    graph.nodes.mask_ancestor_counts[node] = static_cast<std::uint16_t>(depth + 1U);
    graph.nodes.conditional_confidences[node] = 0.75F;
    graph.nodes.path_histories[node] = ProposalPathHistoryDescriptor {
        .digest = kHistoryDigest + depth,
        .length = kHistoryLength + depth,
    };
    graph.nodes.proposal_ticket_offsets[node] = depth == 0U
        ? proposal_ticket_none : static_cast<std::uint16_t>(depth - 1U);
    graph.nodes.feature_provenance[node] = ProposalFeatureProvenance {
        .source = ProposalFeatureSource::draft_hidden,
        .source_layer = 17U,
        .arena_epoch = kArenaEpoch,
    };
    graph.nodes.output_roles[node] = role;
}

[[nodiscard]] ProposalGraphStorage chain_storage(
    std::uint32_t horizon,
    ChainTileEndpoint endpoint
) {
    ProposalGraphStorage graph;
    const bool post_full = endpoint == ChainTileEndpoint::post_full_candidate;
    const auto materialized_edges = post_full ? horizon : horizon - 1U;
    graph.header = ProposalGraphHeader {
        .kind = ProposalGraphKind::chain,
        .horizon = horizon,
        .materialized_node_count = materialized_edges + 1U,
        .materialized_edge_count = materialized_edges,
        .output_frontier_edge_count = post_full ? 0U : 1U,
        .root_count = 1U,
        .root_indices = {0U},
        .graph_identity = 10'000U + horizon + static_cast<std::uint32_t>(post_full),
        .device_arena_epoch = kArenaEpoch,
        .has_source_budget = true,
        .source_budget = {.value = horizon, .root_convention = SourceRootConvention::excluded},
        .resolved_budget = {.draft_node_cap = horizon, .input_row_cap = horizon + 1U},
    };
    initialize_node(graph, 0U, kRootToken, proposal_node_none, 0U,
        horizon == 0U ? ProposalRowOutputRole::correction_candidate
                      : ProposalRowOutputRole::verifies_proposals);
    for (std::uint32_t depth = 1; depth <= materialized_edges; ++depth) {
        initialize_node(
            graph, depth, kRootToken + depth, depth - 1U,
            static_cast<std::uint16_t>(depth),
            depth == materialized_edges
                ? ProposalRowOutputRole::post_full_candidate
                : ProposalRowOutputRole::verifies_proposals
        );
        const auto edge = depth - 1U;
        graph.materialized_edges.parent_node_indices[edge] = depth - 1U;
        graph.materialized_edges.child_node_indices[edge] = depth;
        graph.materialized_edges.token_ids[edge] = kRootToken + depth;
        graph.materialized_edges.conditional_confidences[edge] = 0.7F;
        graph.materialized_edges.sibling_ranks[edge] = 0U;
        graph.materialized_edges.proposal_ticket_offsets[edge] =
            static_cast<std::uint16_t>(depth - 1U);
    }
    if (!post_full) {
        const auto parent = materialized_edges;
        graph.nodes.output_roles[parent] = ProposalRowOutputRole::verifies_proposals;
        graph.output_frontier_edges.parent_node_indices[0] = parent;
        graph.output_frontier_edges.token_ids[0] = kRootToken + horizon;
        graph.output_frontier_edges.conditional_confidences[0] = 0.6F;
        graph.output_frontier_edges.sibling_ranks[0] = 0U;
        graph.output_frontier_edges.proposal_ticket_offsets[0] =
            static_cast<std::uint16_t>(horizon - 1U);
    }
    return graph;
}

[[nodiscard]] ValidatedProposalGraph make_chain_graph(
    std::uint32_t horizon,
    ChainTileEndpoint endpoint = ChainTileEndpoint::post_full_candidate
) {
    return validate_proposal_graph(chain_storage(horizon, endpoint), validation_context());
}

[[nodiscard]] ChainVerifierSchedule single_tile_schedule(
    std::uint32_t horizon,
    ChainTileEndpoint endpoint = ChainTileEndpoint::post_full_candidate
) {
    ChainVerifierSchedule schedule;
    schedule.tile_count = 1U;
    schedule.final_endpoint = endpoint;
    schedule.tiles[0] = ChainVerifierTile {
        .validated_edge_count = static_cast<std::uint16_t>(horizon),
        .input_row_count = static_cast<std::uint16_t>(
            horizon + (endpoint == ChainTileEndpoint::post_full_candidate ? 1U : 0U)
        ),
    };
    return schedule;
}

void add_target_samples(
    SpeculativeControllerScenario& scenario,
    std::span<const std::uint32_t> tokens
) {
    TokenSampler sampler(TokenSamplerOptions {
        .temperature = 0.0F,
        .top_k = 2U,
        .top_p = 1.0F,
        .min_p = 0.0F,
        .repetition_penalty = 1.0F,
        .seed = 42U,
    });
    sampler.observe_tokens(std::span<const std::uint32_t>(&scenario.origin_token, 1U));
    scenario.origin_checkpoint = sampler.checkpoint();
    const auto tickets = sampler.preview_external_tickets();
    auto fork = sampler.fork();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::array<TokenLogit, 2> candidates {{
            {tokens[index], 1.0F},
            {static_cast<std::uint32_t>((tokens[index] + 1U) % 1'000U), 0.0F},
        }};
        const auto selected = fork.sample_with_ticket(tickets.at(index), candidates);
        expect(selected == tokens[index], "script fixture failed to select requested target token");
        fork.observe_tokens(std::span<const std::uint32_t>(&selected, 1U));
        scenario.target_samples[index] = ControllerTargetSample {
            .token_id = selected,
            .checkpoint = fork.checkpoint(),
        };
    }
    scenario.target_sample_count = static_cast<std::uint32_t>(tokens.size());
}

[[nodiscard]] SpeculativeControllerScenario chain_scenario(
    const ValidatedProposalGraph& graph,
    std::span<const std::uint32_t> target_tokens,
    ChainVerifierSchedule schedule
) {
    SpeculativeControllerScenario scenario;
    scenario.graph = &graph;
    scenario.chain_schedule = schedule;
    scenario.proposal_source = SpeculativeProposalSource::learned_external;
    scenario.draft_schedule = schedule.tile_count == 1U
        ? SpeculativeDraftSchedule::eager : SpeculativeDraftSchedule::conditional_tiles;
    if (graph.storage().header.kind == ProposalGraphKind::chain) {
        scenario.verifier_topology = schedule.tile_count > 1U
            ? TargetVerifierTopology::conditional_tile_cascade
            : (schedule.final_endpoint == ChainTileEndpoint::terminal_carry
                ? TargetVerifierTopology::single_invocation_terminal_carry
                : TargetVerifierTopology::single_invocation_post_full);
    } else {
        scenario.verifier_topology =
            graph.storage().header.kind == ProposalGraphKind::repeated_root_beam_batch
            ? TargetVerifierTopology::repeated_root_beam_batch
            : TargetVerifierTopology::prefix_closed_forest;
    }
    scenario.origin_token = kRootToken;
    scenario.remaining_selected_token_budget = 64U;
    scenario.initial_logical_cached_count = 20U;
    scenario.available_context_rows = 64U;
    scenario.callbacks.fill(ControllerCallbackDecision::continue_generation);
    scenario.tile_gates.fill(ControllerTileGateDecision::accept);
    add_target_samples(scenario, target_tokens);
    return scenario;
}

template <class Exception, class Function>
// Keep independent maximum-capacity graph cases in separate stack frames.
// Inlining all adversarial lambdas makes a >16 MiB frame on the normal 8 MiB
// macOS CLI stack; this preserves every tested capacity and assertion.
[[gnu::noinline]] void expect_rejected(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const Exception&) { rejected = true; }
    expect(rejected, message);
}

void test_registered_chain_transitions_are_exhaustive() {
    BoundedSpeculativeController controller;
    for (const auto horizon : {3U, 4U, 7U, 8U, 15U, 16U, 23U}) {
        const auto graph = make_chain_graph(horizon);
        const auto schedule = single_tile_schedule(horizon);
        const auto table = generate_registered_horizon_transition_table(horizon);
        expect(table.count == horizon + 1U, "registered transition table length mismatch");
        for (std::uint32_t depth = 0; depth <= horizon; ++depth) {
            expect(table.values[depth].accepted_depth == depth
                    && table.values[depth].selected_token_count == depth + 1U
                    && table.values[depth].target_ticket_offset
                        == (depth < horizon ? depth : proposal_ticket_none),
                "registered transition table entry mismatch");
        }

        for (std::uint32_t accepted = 0; accepted <= horizon; ++accepted) {
            std::array<std::uint32_t, external_speculation_max_horizon> tokens {};
            std::size_t token_count = 0;
            for (std::uint32_t depth = 1; depth <= accepted; ++depth) {
                tokens[token_count++] = kRootToken + depth;
            }
            if (accepted < horizon) tokens[token_count++] = 900U - accepted;
            auto scenario = chain_scenario(
                graph,
                std::span<const std::uint32_t>(tokens).first(token_count),
                schedule
            );
            const auto outcome = controller.run(scenario);
            expect(outcome.accepted_edge_count == accepted,
                "chain controller accepted-edge count mismatch");
            expect(outcome.sampled_preview_count == token_count,
                "chain controller target sample count mismatch");
            expect(outcome.selected_count == (accepted < horizon ? accepted + 2U : horizon + 1U),
                "chain controller selected-token count mismatch");
            expect(outcome.retained_row_count == accepted + 1U,
                "chain controller retained-row count mismatch");
            expect(outcome.callback_count == accepted + 1U,
                "chain controller callback count mismatch");
            expect(outcome.adopted_checkpoint.same_state(
                    scenario.target_samples[token_count - 1U].checkpoint),
                "chain controller adopted the wrong sampler checkpoint");
            if (accepted < horizon) {
                expect(outcome.disposition == ControllerDisposition::carry_queued
                        && outcome.stop_reason == ControllerStopReason::target_mismatch
                        && outcome.has_carry && !outcome.carry_is_cached,
                    "chain mismatch did not produce the uncached target carry");
            } else {
                expect(outcome.disposition == ControllerDisposition::completed
                        && outcome.stop_reason == ControllerStopReason::full_match
                        && !outcome.has_carry
                        && outcome.next_candidate_lease.source
                            == TargetStateLeaseSource::selected_verifier_row,
                    "chain full match did not retain the post-full candidate row");
            }
        }
    }
}

void test_chain_budgets_stops_and_callbacks_at_every_depth() {
    BoundedSpeculativeController controller;
    for (const auto horizon : {3U, 4U, 7U, 8U, 15U, 16U, 23U}) {
        const auto graph = make_chain_graph(horizon);
        const auto schedule = single_tile_schedule(horizon);
        std::array<std::uint32_t, external_speculation_max_horizon> full_tokens {};
        for (std::uint32_t depth = 1; depth <= horizon; ++depth) {
            full_tokens[depth - 1U] = kRootToken + depth;
        }

        for (std::uint32_t budget = 1; budget <= horizon + 1U; ++budget) {
            auto scenario = chain_scenario(
                graph, std::span<const std::uint32_t>(full_tokens).first(horizon), schedule
            );
            scenario.remaining_selected_token_budget = budget;
            const auto outcome = controller.run(scenario);
            expect(outcome.disposition == ControllerDisposition::stopped
                    && outcome.stop_reason == ControllerStopReason::selected_token_budget
                    && outcome.selected_count == budget
                    && outcome.callback_count == budget
                    && outcome.sampled_preview_count == budget - 1U,
                "chain budget boundary changed token/callback/sample counts");
            if (budget == 1U) {
                expect(outcome.endpoint_kv_only_count == 1U && outcome.retained_row_count == 0U,
                    "x0 budget endpoint did not use ordinary K/V-only caching");
            }
        }

        for (std::uint32_t stop_depth = 0; stop_depth <= horizon; ++stop_depth) {
            auto scenario = chain_scenario(
                graph, std::span<const std::uint32_t>(full_tokens).first(horizon), schedule
            );
            scenario.token_is_stop[stop_depth] = true;
            const auto outcome = controller.run(scenario);
            expect(outcome.disposition == ControllerDisposition::stopped
                    && outcome.stop_reason == ControllerStopReason::stop_token
                    && outcome.selected_count == stop_depth + 1U
                    && outcome.callback_count == stop_depth
                    && outcome.sampled_preview_count == stop_depth,
                "chain stop-token boundary changed target authority");
        }

        for (const auto decision : {
            ControllerCallbackDecision::stop_generation,
            ControllerCallbackDecision::throw_exception,
        }) {
            for (std::uint32_t callback_depth = 0; callback_depth <= horizon; ++callback_depth) {
                auto scenario = chain_scenario(
                    graph, std::span<const std::uint32_t>(full_tokens).first(horizon), schedule
                );
                scenario.callbacks[callback_depth] = decision;
                const auto outcome = controller.run(scenario);
                expect(outcome.selected_count == callback_depth + 1U
                        && outcome.callback_count == callback_depth + 1U
                        && outcome.sampled_preview_count
                            == (callback_depth == 0U ? 1U : callback_depth),
                    "callback boundary selected or sampled an extra authoritative token");
                expect(outcome.adopted_checkpoint.same_state(
                        callback_depth == 0U
                            ? scenario.origin_checkpoint
                            : scenario.target_samples[callback_depth - 1U].checkpoint),
                    "callback boundary adopted a preview checkpoint");
                expect(outcome.disposition
                        == (decision == ControllerCallbackDecision::stop_generation
                            ? ControllerDisposition::stopped
                            : ControllerDisposition::callback_threw),
                    "callback boundary disposition mismatch");
            }
        }
    }
}

void test_h3_conditional_tile_order_gates_context_and_terminal_carry() {
    BoundedSpeculativeController controller;
    const auto graph = make_chain_graph(3U);
    ChainVerifierSchedule cascade;
    cascade.tile_count = 2U;
    cascade.final_endpoint = ChainTileEndpoint::post_full_candidate;
    cascade.tiles[0] = {.validated_edge_count = 2U, .input_row_count = 2U};
    cascade.tiles[1] = {.validated_edge_count = 1U, .input_row_count = 2U};
    const std::array<std::uint32_t, 3> full {101U, 102U, 103U};

    auto scenario = chain_scenario(graph, full, cascade);
    scenario.available_context_rows = 4U;
    scenario.callbacks[2] = ControllerCallbackDecision::stop_generation;
    const auto stopped = controller.run(scenario);
    expect(stopped.selected_count == 3U && stopped.sampled_preview_count == 3U
            && stopped.accepted_edge_count == 2U
            && stopped.adopted_checkpoint.same_state(scenario.target_samples[1].checkpoint),
        "tile-boundary callback did not discard x3/S3 preview");
    std::size_t preview_x3 = stopped.trace_count;
    std::size_t callback_x2 = stopped.trace_count;
    for (std::size_t index = 0; index < stopped.trace_count; ++index) {
        if (stopped.trace[index].kind == ControllerTraceEventKind::target_preview
            && stopped.trace[index].token_id == 103U) preview_x3 = index;
        if (stopped.trace[index].kind == ControllerTraceEventKind::callback
            && stopped.trace[index].token_id == 102U) callback_x2 = index;
    }
    expect(preview_x3 < callback_x2,
        "next-tile first preview must precede the carried parent's callback");

    auto gate_decline = chain_scenario(graph, full, cascade);
    gate_decline.available_context_rows = 4U;
    gate_decline.tile_gates[0] = ControllerTileGateDecision::decline;
    const auto declined = controller.run(gate_decline);
    expect(declined.disposition == ControllerDisposition::ordinary_fallback
            && declined.stop_reason == ControllerStopReason::second_call_confidence_decline
            && declined.selected_count == 3U && declined.sampled_preview_count == 2U
            && declined.carry_token == 102U && !declined.carry_is_cached,
        "tile-gate decline did not return the already-authoritative carry");

    auto context_decline = chain_scenario(graph, full, cascade);
    context_decline.available_context_rows = 3U;
    const auto context = controller.run(context_decline);
    expect(context.disposition == ControllerDisposition::ordinary_fallback
            && context.stop_reason == ControllerStopReason::context_exhausted
            && context.carry_token == 102U && context.selected_count == 3U,
        "later-tile context exhaustion did not preserve its carry");

    auto first_context = chain_scenario(graph, full, cascade);
    first_context.available_context_rows = 1U;
    const auto first = controller.run(first_context);
    expect(first.disposition == ControllerDisposition::ordinary_fallback
            && first.selected_count == 1U && first.sampled_preview_count == 0U
            && first.carry_token == kRootToken,
        "first-tile context exhaustion performed speculative work");

    const auto terminal_graph = make_chain_graph(3U, ChainTileEndpoint::terminal_carry);
    auto terminal = chain_scenario(
        terminal_graph, full, single_tile_schedule(3U, ChainTileEndpoint::terminal_carry)
    );
    const auto terminal_outcome = controller.run(terminal);
    expect(terminal_outcome.accepted_edge_count == 3U
            && terminal_outcome.retained_row_count == 3U
            && terminal_outcome.has_carry && terminal_outcome.carry_token == 103U
            && !terminal_outcome.carry_is_cached
            && terminal_outcome.stop_reason == ControllerStopReason::terminal_carry,
        "terminal-carry schedule incorrectly treated dH as an input row");
}

void test_dynamic_horizons_and_every_conditional_tile_gate() {
    BoundedSpeculativeController controller;
    for (std::uint32_t horizon = 1U; horizon <= external_speculation_max_horizon; ++horizon) {
        const auto graph = make_chain_graph(horizon);
        ChainVerifierSchedule schedule;
        schedule.tile_count = static_cast<std::uint16_t>(horizon);
        schedule.final_endpoint = ChainTileEndpoint::post_full_candidate;
        for (std::uint32_t tile = 0; tile < horizon; ++tile) {
            schedule.tiles[tile] = ChainVerifierTile {
                .validated_edge_count = 1U,
                .input_row_count = static_cast<std::uint16_t>(tile + 1U == horizon ? 2U : 1U),
            };
        }
        std::array<std::uint32_t, external_speculation_max_horizon> full {};
        for (std::uint32_t depth = 1; depth <= horizon; ++depth) {
            full[depth - 1U] = kRootToken + depth;
        }
        auto scenario = chain_scenario(
            graph, std::span<const std::uint32_t>(full).first(horizon), schedule
        );
        scenario.draft_schedule = SpeculativeDraftSchedule::dynamic_horizon;
        scenario.available_context_rows = horizon + 1U;
        const auto completed = controller.run(scenario);
        expect(completed.selected_count == horizon + 1U
                && completed.accepted_edge_count == horizon
                && completed.stop_reason == ControllerStopReason::full_match,
            "dynamic horizon was routed through a neighboring fixed horizon");

        for (std::uint32_t gate = 0; gate + 1U < horizon; ++gate) {
            auto declined = scenario;
            declined.tile_gates[gate] = ControllerTileGateDecision::decline;
            const auto outcome = controller.run(declined);
            const auto expected_reason = horizon == 3U && gate == 0U
                ? ControllerStopReason::second_call_confidence_decline
                : ControllerStopReason::tile_gate_decline;
            expect(outcome.disposition == ControllerDisposition::ordinary_fallback
                    && outcome.stop_reason == expected_reason
                    && outcome.accepted_edge_count == gate + 1U
                    && outcome.sampled_preview_count == gate + 1U
                    && outcome.selected_count == gate + 2U,
                "conditional tile gate did not preserve its exact accepted anchor");
        }
    }

    constexpr std::uint32_t context_horizon = 8U;
    const auto context_graph = make_chain_graph(context_horizon);
    ChainVerifierSchedule context_schedule;
    context_schedule.tile_count = context_horizon;
    context_schedule.final_endpoint = ChainTileEndpoint::post_full_candidate;
    for (std::uint32_t tile = 0; tile < context_horizon; ++tile) {
        context_schedule.tiles[tile] = {
            .validated_edge_count = 1U,
            .input_row_count = static_cast<std::uint16_t>(
                tile + 1U == context_horizon ? 2U : 1U
            ),
        };
    }
    std::array<std::uint32_t, context_horizon> context_tokens {};
    for (std::uint32_t depth = 1; depth <= context_horizon; ++depth) {
        context_tokens[depth - 1U] = kRootToken + depth;
    }
    for (std::uint32_t available = 0; available <= context_horizon; ++available) {
        auto scenario = chain_scenario(context_graph, context_tokens, context_schedule);
        scenario.available_context_rows = available;
        const auto outcome = controller.run(scenario);
        const auto expected_accepted = available == 0U
            ? 0U : std::min(available, context_horizon - 1U);
        expect(outcome.disposition == ControllerDisposition::ordinary_fallback
                && outcome.stop_reason == ControllerStopReason::context_exhausted
                && outcome.accepted_edge_count == expected_accepted
                && outcome.sampled_preview_count == expected_accepted
                && outcome.selected_count == expected_accepted + 1U,
            "context exhaustion at a conditional call lost or invented an accepted token");
    }
}

void test_resumable_controller_pauses_before_each_causal_target_sample() {
    BoundedSpeculativeController controller;
    const auto graph = make_chain_graph(3U);
    ChainVerifierSchedule cascade;
    cascade.tile_count = 2U;
    cascade.final_endpoint = ChainTileEndpoint::post_full_candidate;
    cascade.tiles[0] = {.validated_edge_count = 2U, .input_row_count = 2U};
    cascade.tiles[1] = {.validated_edge_count = 1U, .input_row_count = 2U};
    constexpr std::array<std::uint32_t, 3> tokens {101U, 102U, 103U};
    const auto complete_scenario = chain_scenario(graph, tokens, cascade);
    const auto complete = controller.run(complete_scenario);

    auto incremental = complete_scenario;
    incremental.target_sample_count = 0U;
    constexpr std::array<std::uint16_t, 3> expected_tiles {0U, 0U, 1U};
    for (std::uint32_t offset = 0U; offset < tokens.size(); ++offset) {
        const auto paused = controller.run_available(incremental);
        expect(paused.requires_target_sample
                && paused.required_target_sample_offset == offset
                && paused.required_target_sample_tile == expected_tiles[offset]
                && paused.sampled_preview_count == offset,
            "resumable controller did not pause before the exact causal target sample");
        incremental.target_samples[offset] = complete_scenario.target_samples[offset];
        ++incremental.target_sample_count;
    }
    const auto resumed = controller.run_available(incremental);
    expect(!resumed.requires_target_sample
            && resumed.disposition == complete.disposition
            && resumed.stop_reason == complete.stop_reason
            && resumed.selected_count == complete.selected_count
            && resumed.accepted_edge_count == complete.accepted_edge_count
            && resumed.retained_row_count == complete.retained_row_count
            && resumed.adopted_checkpoint.same_state(complete.adopted_checkpoint),
        "resumable controller completion diverged from the exact one-shot controller");

    auto declined = complete_scenario;
    declined.target_sample_count = 2U;
    declined.tile_gates[0] = ControllerTileGateDecision::decline;
    const auto gated = controller.run_available(declined);
    expect(!gated.requires_target_sample
            && gated.stop_reason == ControllerStopReason::second_call_confidence_decline
            && gated.sampled_preview_count == 2U,
        "resumable controller requested a later tile after its causal gate declined");
}

void test_terminal_carry_stop_and_budget_endpoints_are_cached_once() {
    BoundedSpeculativeController controller;
    for (const auto horizon : {3U, 4U, 8U, 16U, 23U}) {
        const auto graph = make_chain_graph(horizon, ChainTileEndpoint::terminal_carry);
        const auto schedule = single_tile_schedule(horizon, ChainTileEndpoint::terminal_carry);
        std::array<std::uint32_t, external_speculation_max_horizon> full {};
        for (std::uint32_t depth = 1; depth <= horizon; ++depth) {
            full[depth - 1U] = kRootToken + depth;
        }

        auto stopped = chain_scenario(
            graph, std::span<const std::uint32_t>(full).first(horizon), schedule
        );
        stopped.token_is_stop[horizon] = true;
        const auto stop_outcome = controller.run(stopped);
        expect(stop_outcome.stop_reason == ControllerStopReason::stop_token
                && stop_outcome.endpoint_kv_only_count == 1U
                && stop_outcome.endpoint_kv_only_tokens[0] == kRootToken + horizon
                && !stop_outcome.has_carry
                && stop_outcome.callback_count == horizon,
            "uncached terminal stop token was not K/V-cached exactly once");

        auto budget = chain_scenario(
            graph, std::span<const std::uint32_t>(full).first(horizon), schedule
        );
        budget.remaining_selected_token_budget = horizon + 1U;
        const auto budget_outcome = controller.run(budget);
        expect(budget_outcome.stop_reason == ControllerStopReason::selected_token_budget
                && budget_outcome.endpoint_kv_only_count == 1U
                && budget_outcome.endpoint_kv_only_tokens[0] == kRootToken + horizon
                && !budget_outcome.has_carry
                && budget_outcome.callback_count == horizon + 1U,
            "uncached terminal budget token was not cached before callback");
    }
}

[[nodiscard]] ValidatedProposalGraph make_forest_graph() {
    ProposalGraphStorage graph;
    graph.header = ProposalGraphHeader {
        .kind = ProposalGraphKind::forest,
        .horizon = 3U,
        .materialized_node_count = 4U,
        .materialized_edge_count = 3U,
        .output_frontier_edge_count = 1U,
        .root_count = 1U,
        .root_indices = {0U},
        .graph_identity = 20'001U,
        .device_arena_epoch = kArenaEpoch,
    };
    initialize_node(graph, 0U, 100U, proposal_node_none, 0U, ProposalRowOutputRole::verifies_proposals);
    initialize_node(graph, 1U, 101U, 0U, 1U, ProposalRowOutputRole::verifies_proposals);
    initialize_node(graph, 2U, 201U, 0U, 1U, ProposalRowOutputRole::correction_candidate);
    initialize_node(graph, 3U, 102U, 1U, 2U, ProposalRowOutputRole::verifies_proposals);
    const std::array<std::uint32_t, 3> parents {0U, 0U, 1U};
    const std::array<std::uint32_t, 3> children {1U, 2U, 3U};
    for (std::size_t edge = 0; edge < 3U; ++edge) {
        graph.materialized_edges.parent_node_indices[edge] = parents[edge];
        graph.materialized_edges.child_node_indices[edge] = children[edge];
        graph.materialized_edges.token_ids[edge] = graph.nodes.token_ids[children[edge]];
        graph.materialized_edges.conditional_confidences[edge] = 0.7F;
        graph.materialized_edges.sibling_ranks[edge] = edge == 1U ? 1U : 0U;
        graph.materialized_edges.proposal_ticket_offsets[edge] =
            graph.nodes.proposal_ticket_offsets[children[edge]];
    }
    graph.output_frontier_edges.parent_node_indices[0] = 3U;
    graph.output_frontier_edges.token_ids[0] = 103U;
    graph.output_frontier_edges.conditional_confidences[0] = 0.5F;
    graph.output_frontier_edges.sibling_ranks[0] = 0U;
    graph.output_frontier_edges.proposal_ticket_offsets[0] = 2U;
    return validate_proposal_graph(std::move(graph), validation_context());
}

void test_forest_and_output_frontier_walk_one_causal_path() {
    BoundedSpeculativeController controller;
    const auto graph = make_forest_graph();
    const std::array<std::uint32_t, 3> frontier_path {101U, 102U, 103U};
    auto frontier = chain_scenario(graph, frontier_path, {});
    const auto frontier_outcome = controller.run(frontier);
    expect(frontier_outcome.accepted_edge_count == 3U
            && frontier_outcome.selected_count == 4U
            && frontier_outcome.retained_row_count == 3U
            && frontier_outcome.has_carry && frontier_outcome.carry_token == 103U
            && !frontier_outcome.carry_is_cached
            && frontier_outcome.callback_count == 3U,
        "forest output-only frontier was not returned as an uncached carry");

    const std::array<std::uint32_t, 1> leaf_path {201U};
    auto leaf = chain_scenario(graph, leaf_path, {});
    const auto leaf_outcome = controller.run(leaf);
    expect(leaf_outcome.accepted_edge_count == 1U
            && leaf_outcome.selected_count == 2U
            && leaf_outcome.retained_row_count == 2U
            && !leaf_outcome.has_carry
            && leaf_outcome.stop_reason == ControllerStopReason::materialized_leaf
            && leaf_outcome.next_candidate_lease.node_index == 2U,
        "forest materialized leaf did not supply the next candidate state");
}

[[nodiscard]] ValidatedProposalGraph make_repeated_root_graph() {
    ProposalGraphStorage graph;
    constexpr std::uint32_t width = 4U;
    constexpr std::uint32_t length = 4U;
    graph.header = ProposalGraphHeader {
        .kind = ProposalGraphKind::repeated_root_beam_batch,
        .horizon = 3U,
        .materialized_node_count = width * length,
        .materialized_edge_count = width * (length - 1U),
        .root_count = width,
        .root_indices = {0U, 4U, 8U, 12U},
        .repeated_beam_width = width,
        .repeated_beam_length = length,
        .graph_identity = 30'001U,
        .device_arena_epoch = kArenaEpoch,
    };
    const std::array<std::array<std::uint32_t, length>, width> beams {{
        {{100U, 101U, 102U, 103U}},
        {{100U, 101U, 202U, 203U}},
        {{100U, 301U, 302U, 303U}},
        {{100U, 401U, 402U, 403U}},
    }};
    std::size_t edge = 0;
    for (std::uint32_t beam = 0; beam < width; ++beam) {
        const auto root = beam * length;
        for (std::uint32_t depth = 0; depth < length; ++depth) {
            const auto node = root + depth;
            initialize_node(
                graph, node, beams[beam][depth],
                depth == 0U ? proposal_node_none : node - 1U,
                static_cast<std::uint16_t>(depth),
                depth + 1U == length
                    ? ProposalRowOutputRole::correction_candidate
                    : ProposalRowOutputRole::verifies_proposals
            );
            if (depth == 0U) continue;
            graph.materialized_edges.parent_node_indices[edge] = node - 1U;
            graph.materialized_edges.child_node_indices[edge] = node;
            graph.materialized_edges.token_ids[edge] = beams[beam][depth];
            graph.materialized_edges.conditional_confidences[edge] = 0.7F;
            graph.materialized_edges.sibling_ranks[edge] = 0U;
            graph.materialized_edges.proposal_ticket_offsets[edge] =
                static_cast<std::uint16_t>(depth - 1U);
            ++edge;
        }
    }
    return validate_proposal_graph(std::move(graph), validation_context());
}

void test_repeated_root_beams_preserve_prefix_candidates() {
    BoundedSpeculativeController controller;
    const auto graph = make_repeated_root_graph();
    const std::array<std::uint32_t, 3> target {101U, 202U, 203U};
    auto scenario = chain_scenario(graph, target, {});
    const auto outcome = controller.run(scenario);
    expect(outcome.disposition == ControllerDisposition::completed
            && outcome.accepted_edge_count == 3U
            && outcome.selected_count == 4U
            && outcome.selected_tokens[1] == 101U
            && outcome.selected_tokens[2] == 202U
            && outcome.selected_tokens[3] == 203U
            && outcome.selected_node_path[0] == 4U
            && outcome.selected_node_path[1] == 5U
            && outcome.selected_node_path[2] == 6U
            && outcome.next_candidate_lease.node_index == 7U,
        "repeated-root beam walker collapsed distinct candidate prefixes");
}

[[nodiscard]] ValidatedProposalGraph make_redrafter_l5_h4_graph() {
    ProposalGraphStorage graph;
    constexpr std::uint32_t width = 3U;
    constexpr std::uint32_t length = 5U;
    graph.header = ProposalGraphHeader {
        .kind = ProposalGraphKind::repeated_root_beam_batch,
        .horizon = 4U,
        .materialized_node_count = width * length,
        .materialized_edge_count = width * (length - 1U),
        .root_count = width,
        .root_indices = {0U, 5U, 10U},
        .repeated_beam_width = width,
        .repeated_beam_length = length,
        .graph_identity = 30'002U,
        .device_arena_epoch = kArenaEpoch,
    };
    const std::array<std::array<std::uint32_t, length>, width> beams {{
        {{100U, 101U, 102U, 103U, 104U}},
        {{100U, 101U, 202U, 203U, 204U}},
        {{100U, 301U, 302U, 303U, 304U}},
    }};
    std::size_t edge = 0;
    for (std::uint32_t beam = 0; beam < width; ++beam) {
        const auto root = beam * length;
        for (std::uint32_t depth = 0; depth < length; ++depth) {
            const auto node = root + depth;
            initialize_node(
                graph, node, beams[beam][depth],
                depth == 0U ? proposal_node_none : node - 1U,
                static_cast<std::uint16_t>(depth),
                depth + 1U == length
                    ? ProposalRowOutputRole::correction_candidate
                    : ProposalRowOutputRole::verifies_proposals
            );
            if (depth == 0U) continue;
            graph.materialized_edges.parent_node_indices[edge] = node - 1U;
            graph.materialized_edges.child_node_indices[edge] = node;
            graph.materialized_edges.token_ids[edge] = beams[beam][depth];
            graph.materialized_edges.conditional_confidences[edge] = 0.7F;
            graph.materialized_edges.sibling_ranks[edge] = 0U;
            graph.materialized_edges.proposal_ticket_offsets[edge] =
                static_cast<std::uint16_t>(depth - 1U);
            ++edge;
        }
    }
    return validate_proposal_graph(std::move(graph), validation_context());
}

[[nodiscard]] ValidatedProposalGraph make_eagle3_depth8_final_tree() {
    ProposalGraphStorage graph;
    graph.header = ProposalGraphHeader {
        .kind = ProposalGraphKind::forest,
        .horizon = 8U,
        .materialized_node_count = EAGLE3VerifierRowMax,
        .materialized_edge_count = EAGLE3VerifierRowMax - 1U,
        .root_count = 1U,
        .root_indices = {0U},
        .graph_identity = 40'001U,
        .device_arena_epoch = kArenaEpoch,
    };
    initialize_node(graph, 0U, 100U, proposal_node_none, 0U,
        ProposalRowOutputRole::verifies_proposals);
    for (std::uint32_t depth = 1; depth <= 8U; ++depth) {
        initialize_node(
            graph, depth, 100U + depth, depth - 1U,
            static_cast<std::uint16_t>(depth),
            depth == 8U ? ProposalRowOutputRole::correction_candidate
                        : ProposalRowOutputRole::verifies_proposals
        );
    }
    for (std::uint32_t node = 9U; node < EAGLE3VerifierRowMax; ++node) {
        initialize_node(
            graph, node, 300U + node, 0U, 1U,
            ProposalRowOutputRole::correction_candidate
        );
    }

    std::size_t edge = 0;
    auto add_edge = [&](std::uint32_t parent, std::uint32_t child, std::uint16_t rank) {
        graph.materialized_edges.parent_node_indices[edge] = parent;
        graph.materialized_edges.child_node_indices[edge] = child;
        graph.materialized_edges.token_ids[edge] = graph.nodes.token_ids[child];
        graph.materialized_edges.conditional_confidences[edge] = 0.7F;
        graph.materialized_edges.sibling_ranks[edge] = rank;
        graph.materialized_edges.proposal_ticket_offsets[edge] =
            graph.nodes.proposal_ticket_offsets[child];
        ++edge;
    };
    add_edge(0U, 1U, 0U);
    std::uint16_t root_rank = 1U;
    for (std::uint32_t node = 9U; node < EAGLE3VerifierRowMax; ++node) {
        add_edge(0U, node, root_rank++);
    }
    for (std::uint32_t parent = 1U; parent < 8U; ++parent) {
        add_edge(parent, parent + 1U, 0U);
    }
    expect(edge == EAGLE3VerifierRowMax - 1U,
        "EAGLE-3 fixture did not materialize exactly 59 edges");
    return validate_proposal_graph(std::move(graph), validation_context());
}

void test_source_shaped_redrafter_l5_and_eagle3_depth8() {
    BoundedSpeculativeController controller;
    const auto redrafter = make_redrafter_l5_h4_graph();
    const std::array<std::uint32_t, 4> redrafter_path {101U, 202U, 203U, 204U};
    const auto redrafter_outcome = controller.run(chain_scenario(redrafter, redrafter_path, {}));
    expect(redrafter_outcome.accepted_edge_count == 4U
            && redrafter_outcome.selected_count == 5U
            && redrafter_outcome.next_candidate_lease.node_index == 9U,
        "ReDrafter L5/H4 repeated-root control was truncated to a neighboring horizon");

    const auto eagle = make_eagle3_depth8_final_tree();
    const std::array<std::uint32_t, 8> eagle_path {
        101U, 102U, 103U, 104U, 105U, 106U, 107U, 108U,
    };
    const auto eagle_outcome = controller.run(chain_scenario(eagle, eagle_path, {}));
    expect(eagle.storage().header.materialized_node_count == 60U
            && eagle_outcome.accepted_edge_count == 8U
            && eagle_outcome.selected_count == 9U
            && eagle_outcome.next_candidate_lease.node_index == 8U,
        "EAGLE-3 depth-8/60-row final tree was routed through a smaller control");
}

void test_target_verifier_source_shape_registry_is_complete_and_typed() {
    const auto shapes = target_verifier_source_shapes();
    expect(shapes.size() == target_verifier_source_shape_count,
        "target verifier source-shape registry count drifted");
    std::array<std::uint32_t, 12> family_counts {};
    auto saw_cadd_root_only = false;
    auto saw_cadd_maximum = false;
    auto saw_treeflash_released = false;
    auto saw_treeflash_code_m64 = false;
    auto saw_treeflash_h16_m64 = false;
    auto saw_pctree_pool_maximum = false;
    auto saw_dartree_score_maximum = false;
    auto saw_domino_maximum = false;
    auto saw_full_path_scaffold = false;
    auto saw_dblast_maximum = false;
    auto saw_eagle3_maximum = false;
    auto saw_redrafter_maximum = false;

    for (std::size_t index = 0U; index < shapes.size(); ++index) {
        const auto& shape = shapes[index];
        expect(shape.registry_index == index
                && shape.physical_target_rows == shape.resolved_budget.input_row_cap
                && shape.physical_target_rows <= VerifierRowMax
                && shape.resolved_budget.draft_node_cap <= ExternalTreeDraftNodeMax
                && shape.maximum_depth <= external_speculation_max_horizon
                && shape.child_branch_width <= MaterializedDegreeMax
                && target_verifier_source_mechanism_name(shape.mechanism) != "unknown",
            "target verifier source shape has stale identity or capacity metadata");
        const auto family = static_cast<std::size_t>(shape.family);
        expect(family < family_counts.size(),
            "target verifier source shape has unknown family");
        ++family_counts[family];

        if (shape.topology == ProposalGraphKind::repeated_root_beam_batch) {
            expect(shape.mechanism
                        == TargetVerifierSourceMechanism::redrafter_mlx_repeated_root
                    && shape.physical_target_rows == 4U * shape.active_width
                    && shape.resolved_budget.draft_node_cap == 3U * shape.active_width,
                "repeated-root source shape was flattened into a unique-root budget");
        } else {
            expect(shape.topology == ProposalGraphKind::forest
                    && shape.resolved_budget.draft_node_cap + 1U
                        == shape.physical_target_rows,
                "unique-root source shape has inconsistent resolved rows");
        }
        expect((shape.candidate_domain == TargetVerifierCandidateDomain::full_vocabulary)
                    == (shape.candidate_width == 0U),
            "source shape conflated full-V and finite candidate domains");

        saw_cadd_root_only = saw_cadd_root_only
            || (shape.mechanism == TargetVerifierSourceMechanism::caddtree_adaptive
                && shape.source_budget.value == 0U && shape.physical_target_rows == 1U);
        saw_cadd_maximum = saw_cadd_maximum
            || (shape.mechanism == TargetVerifierSourceMechanism::caddtree_adaptive
                && shape.source_budget.value == 1'024U
                && shape.physical_target_rows == 1'025U);
        saw_treeflash_released = saw_treeflash_released
            || (shape.mechanism
                    == TargetVerifierSourceMechanism::treeflash_executable_h15_top_k
                && shape.source_budget.value == 64U
                && shape.conditioning_width == 16U
                && shape.candidate_width == 16U
                && shape.builder.raw_row_count == TreeFlashReleasedCorrectionStateMax
                && shape.builder.packed_row_count
                    == TreeFlashReleasedPhysicalCorrectionRowMax);
        saw_treeflash_code_m64 = saw_treeflash_code_m64
            || (shape.mechanism
                    == TargetVerifierSourceMechanism::treeflash_executable_h15_top_k
                && shape.conditioning_width == 64U
                && shape.builder.raw_row_count == TreeFlashCodeTransferCorrectionStateMax
                && shape.builder.packed_row_count == TreeFlashCodePhysicalCorrectionRowMax);
        saw_treeflash_h16_m64 = saw_treeflash_h16_m64
            || (shape.mechanism
                    == TargetVerifierSourceMechanism::treeflash_paper_h16_full_v_lazy
                && shape.conditioning_width == 64U
                && shape.builder.raw_row_count == TreeFlashSemanticCorrectionStateMax
                && shape.builder.packed_row_count == TreeFlashH16PhysicalCorrectionRowMax);
        saw_pctree_pool_maximum = saw_pctree_pool_maximum
            || shape.builder.raw_row_count == PCTreePoolMax;
        saw_dartree_score_maximum = saw_dartree_score_maximum
            || shape.builder.score_event_count == DARTreeScoreEventMax;
        saw_domino_maximum = saw_domino_maximum
            || (shape.family == TargetVerifierSourceFamily::dominotree
                && shape.source_budget.value == 64U
                && shape.conditioning_width == 256U
                && shape.builder.raw_row_count == DominoFrontierGeneratedLedgerMax
                && shape.builder.kv_row_count == DominoCorrectionLaneStepMax);
        saw_full_path_scaffold = saw_full_path_scaffold
            || (shape.family == TargetVerifierSourceFamily::full_path
                && shape.builder.raw_row_count == FullPathSupertreeMax);
        saw_dblast_maximum = saw_dblast_maximum
            || (shape.family == TargetVerifierSourceFamily::dblast
                && shape.physical_target_rows == DBlastCategoryTrieRowMax);
        saw_eagle3_maximum = saw_eagle3_maximum
            || (shape.family == TargetVerifierSourceFamily::eagle3
                && shape.builder.raw_row_count == EAGLE3RawSupertreeRowMax
                && shape.physical_target_rows == EAGLE3VerifierRowMax);
        saw_redrafter_maximum = saw_redrafter_maximum
            || (shape.mechanism
                    == TargetVerifierSourceMechanism::redrafter_pytorch_l5_w45_compressed
                && shape.builder.raw_row_count == ReDrafterRawBeamEntryMax
                && shape.physical_target_rows == ReDrafterPyTreeRowMax
                && shape.builder.kv_row_count == ReDrafterPyUnpackedCacheRowMax);

        for (std::size_t prior = 0U; prior < index; ++prior) {
            const auto& other = shapes[prior];
            const auto duplicate = shape.mechanism == other.mechanism
                && shape.source_budget.value == other.source_budget.value
                && shape.source_budget.root_convention
                    == other.source_budget.root_convention
                && shape.maximum_depth == other.maximum_depth
                && shape.child_branch_width == other.child_branch_width
                && shape.candidate_domain == other.candidate_domain
                && shape.candidate_width == other.candidate_width
                && shape.conditioning_width == other.conditioning_width
                && shape.active_width == other.active_width
                && shape.physical_target_rows == other.physical_target_rows;
            expect(!duplicate, "target verifier source registry contains a duplicate identity");
        }
    }

    constexpr std::array<std::uint32_t, 12> expected_family_counts {
        56U, 7U, 1'025U, 54U, 26U, 18U,
        12U, 18U, 12U, 4U, 1U, 13U,
    };
    expect(family_counts == expected_family_counts,
        "target verifier source family was narrowed or accidentally crossed");
    expect(saw_cadd_root_only && saw_cadd_maximum
            && saw_treeflash_released && saw_treeflash_code_m64
            && saw_treeflash_h16_m64 && saw_pctree_pool_maximum
            && saw_dartree_score_maximum && saw_domino_maximum
            && saw_full_path_scaffold && saw_dblast_maximum
            && saw_eagle3_maximum && saw_redrafter_maximum,
        "target verifier source registry lost a required boundary fixture");
    expect(TreeFlashReleasedRankEntryMax == 3'840U
            && TreeFlashCodeTop16RankEntryMax == 15'360U
            && TreeFlashCodeTop64RankEntryMax == 61'440U
            && TreeFlashH15FullVRankEntryMax == 491'520U
            && TreeFlashH16FullVRankEntryMax == 492'032U
            && TreeFlashH16CodeFullVRankEntryMax == 524'288U,
        "TreeFlash rank-storage capacity contract drifted");
}

void test_routes_and_proposal_randomness_remain_distinct() {
    BoundedSpeculativeController controller;
    const std::array<std::uint32_t, 3> full {101U, 102U, 103U};

    for (const auto policy : {
        ProposalPrimaryPolicy::scalar_common_draw,
        ProposalPrimaryPolicy::deterministic,
        ProposalPrimaryPolicy::greedy,
    }) {
        auto storage = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        storage.header.primary_policy = policy;
        if (policy != ProposalPrimaryPolicy::scalar_common_draw) {
            for (std::size_t node = 1; node < storage.header.materialized_node_count; ++node) {
                storage.nodes.proposal_ticket_offsets[node] = proposal_ticket_none;
            }
            for (std::size_t edge = 0; edge < storage.header.materialized_edge_count; ++edge) {
                storage.materialized_edges.proposal_ticket_offsets[edge] = proposal_ticket_none;
            }
        }
        const auto graph = validate_proposal_graph(std::move(storage), validation_context());
        const auto outcome = controller.run(
            chain_scenario(graph, full, single_tile_schedule(3U))
        );
        expect(outcome.primary_policy == policy && outcome.selected_count == 4U,
            "proposal randomness policy was collapsed or changed target output");
    }

    const auto graph = make_chain_graph(3U);
    auto retrieval = chain_scenario(graph, full, single_tile_schedule(3U));
    retrieval.proposal_source = SpeculativeProposalSource::retrieval;
    retrieval.route_mode = SpeculativeRouteMode::retrieval_first;
    expect(controller.run(retrieval).proposal_source == SpeculativeProposalSource::retrieval,
        "retrieval-first route identity was lost");

    auto anchored = chain_scenario(graph, full, single_tile_schedule(3U));
    anchored.route_mode = SpeculativeRouteMode::retrieval_anchored_learned;
    expect(controller.run(anchored).route_mode
            == SpeculativeRouteMode::retrieval_anchored_learned,
        "retrieval-anchored learned route identity was lost");

    for (const auto command_topology : {
        FirstCallCommandTopology::fused_eager,
        FirstCallCommandTopology::host_gated,
        FirstCallCommandTopology::gpu_gated,
    }) {
        auto topology_scenario = chain_scenario(graph, full, single_tile_schedule(3U));
        topology_scenario.first_call_topology = command_topology;
        expect(controller.run(topology_scenario).first_call_topology == command_topology,
            "first-call command topology identity was collapsed");
    }

    auto ineligible = chain_scenario(graph, full, single_tile_schedule(3U));
    ineligible.eligible = false;
    const auto declined = controller.run(ineligible);
    expect(declined.disposition == ControllerDisposition::ordinary_fallback
            && declined.stop_reason == ControllerStopReason::ineligible
            && declined.sampled_preview_count == 0U,
        "ineligible route performed speculative target sampling");

    auto incompatible = chain_scenario(graph, full, single_tile_schedule(3U));
    incompatible.route_mode = SpeculativeRouteMode::target_integrated;
    expect_rejected<std::invalid_argument>([&] {
        static_cast<void>(controller.run(incompatible));
    }, "incompatible target-integrated route accepted an external proposal graph");
}

void test_graph_schedule_and_capacity_validation_fail_closed() {
    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.header.device_arena_epoch = 9U;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "stale proposal arena was accepted");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.nodes.parent_indices[2] = 2U;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "cyclic/non-topological proposal node was accepted");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.nodes.mask_component_roots[2] = 2U;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "sibling/nonancestor attention-mask component was accepted");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.nodes.output_roles[0] = ProposalRowOutputRole::correction_candidate;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "proposal row with children was labeled as a correction row");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.nodes.proposal_ticket_offsets[1] =
            static_cast<std::uint16_t>(sampler_max_preview_ticket_count);
        graph.materialized_edges.proposal_ticket_offsets[0] =
            static_cast<std::uint16_t>(sampler_max_preview_ticket_count);
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "proposal ticket beyond the 47-offset pool was accepted");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(2U, ChainTileEndpoint::post_full_candidate);
        graph.nodes.parent_indices[2] = 0U;
        graph.nodes.depths[2] = 1U;
        graph.nodes.absolute_positions[2] = kRootPosition + 1U;
        graph.nodes.path_histories[2].length = kHistoryLength + 1U;
        graph.nodes.token_ids[2] = graph.nodes.token_ids[1];
        graph.materialized_edges.parent_node_indices[1] = 0U;
        graph.materialized_edges.token_ids[1] = graph.nodes.token_ids[1];
        graph.materialized_edges.sibling_ranks[1] = 1U;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "duplicate proposal siblings were accepted");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.header.resolved_budget.input_row_cap = 3U;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "inconsistent root convention resolution was accepted");

    expect_rejected<std::invalid_argument>([] {
        auto graph = chain_storage(3U, ChainTileEndpoint::post_full_candidate);
        graph.header.materialized_node_count = VerifierRowMax + 1U;
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "proposal node-capacity overflow was accepted");

    expect_rejected<std::invalid_argument>([] {
        ProposalGraphStorage graph;
        graph.header = ProposalGraphHeader {
            .kind = ProposalGraphKind::forest,
            .horizon = 1U,
            .materialized_node_count = 1U,
            .materialized_edge_count = 0U,
            .output_frontier_edge_count = OutputFrontierBranchMax + 1U,
            .root_count = 1U,
            .root_indices = {0U},
            .graph_identity = 88U,
            .device_arena_epoch = kArenaEpoch,
        };
        initialize_node(
            graph, 0U, kRootToken, proposal_node_none, 0U,
            ProposalRowOutputRole::verifies_proposals
        );
        for (std::size_t edge = 0; edge < graph.header.output_frontier_edge_count; ++edge) {
            graph.output_frontier_edges.parent_node_indices[edge] = 0U;
            graph.output_frontier_edges.token_ids[edge] = 500U + static_cast<std::uint32_t>(edge);
            graph.output_frontier_edges.conditional_confidences[edge] = 0.5F;
            graph.output_frontier_edges.sibling_ranks[edge] = static_cast<std::uint16_t>(edge);
            graph.output_frontier_edges.proposal_ticket_offsets[edge] = 0U;
        }
        static_cast<void>(validate_proposal_graph(std::move(graph), validation_context()));
    }, "output-frontier branch beyond 32 was accepted");

    expect_rejected<std::invalid_argument>([] {
        ChainVerifierSchedule schedule;
        schedule.tile_count = 2U;
        schedule.tiles[0] = {.validated_edge_count = 2U, .input_row_count = 2U};
        schedule.tiles[1] = {.validated_edge_count = 1U, .input_row_count = 1U};
        schedule.final_endpoint = ChainTileEndpoint::post_full_candidate;
        validate_chain_verifier_schedule(schedule, 3U);
    }, "post-full verifier tile omitted its candidate row");

    expect_rejected<std::invalid_argument>([] {
        static_cast<void>(generate_registered_horizon_transition_table(5U));
    }, "unregistered horizon transition table was generated");
}

[[nodiscard]] StreamTreeLeaseDescriptor tree(
    std::uint64_t identity,
    std::uint32_t root_token = 100U
) {
    return StreamTreeLeaseDescriptor {
        .lease_identity = identity,
        .session_epoch = 1U,
        .arena_epoch = 2U,
        .graph_identity = 100U + identity,
        .history_digest = 200U + identity,
        .root_token = root_token,
        .root_position = 10U,
    };
}

void test_arm_s_lease_state_machine_and_poisoning() {
    StreamLeaseState state {.phase = StreamLeasePhase::unavailable, .session_epoch = 1U};
    state = transition_stream_lease(state, {.kind = StreamLeaseEventKind::begin_bootstrap}).next_state;
    expect(state.phase == StreamLeasePhase::pending_bootstrap,
        "Arm-S bootstrap did not enter pending state");
    const auto first_tree = tree(1U);
    state = transition_stream_lease(state, {
        .kind = StreamLeaseEventKind::bootstrap_completed,
        .has_tree = true,
        .tree = first_tree,
    }).next_state;
    expect(state.phase == StreamLeasePhase::pending_root_bind && state.has_pending_tree,
        "Arm-S bootstrap completion did not retain its unpublished tree");
    const auto bound = transition_stream_lease(state, {
        .kind = StreamLeaseEventKind::bind_selected_root,
        .has_tree = true,
        .tree = first_tree,
    });
    state = bound.next_state;
    expect(state.phase == StreamLeasePhase::steady && state.has_prior_tree
            && bound.tree_action == NextTreeAction::publish,
        "Arm-S root bind did not publish exactly one prior-tree lease");

    const auto correction_tree = tree(2U, 333U);
    state = transition_stream_lease(state, {
        .kind = StreamLeaseEventKind::verification_mismatch,
        .has_tree = true,
        .tree = correction_tree,
    }).next_state;
    expect(state.phase == StreamLeasePhase::pending_correction_bind,
        "Arm-S mismatch did not retain the main-row correction tree");
    const auto corrected = transition_stream_lease(state, {
        .kind = StreamLeaseEventKind::bind_correction,
        .has_tree = true,
        .tree = correction_tree,
    });
    state = corrected.next_state;
    expect(state.phase == StreamLeasePhase::steady
            && state.prior_tree.lease_identity == correction_tree.lease_identity
            && state.prior_tree.graph_identity == correction_tree.graph_identity
            && corrected.tree_action == NextTreeAction::publish,
        "Arm-S correction bind published the wrong tree");

    const auto rebased_tree = tree(3U, 444U);
    const auto rebasing = transition_stream_lease(state, {
        .kind = StreamLeaseEventKind::begin_rebase,
        .has_tree = true,
        .tree = rebased_tree,
    });
    expect(rebasing.next_state.phase == StreamLeasePhase::pending_root_bind
            && rebasing.tree_action == NextTreeAction::rebase,
        "Arm-S rebase did not invalidate the old prior tree");

    const auto poisoned = transition_stream_lease(state, {.kind = StreamLeaseEventKind::poison});
    expect(poisoned.next_state.phase == StreamLeasePhase::poisoned
            && !poisoned.next_state.has_prior_tree,
        "Arm-S poison retained reusable state");
    expect_rejected<std::logic_error>([&] {
        static_cast<void>(transition_stream_lease(
            poisoned.next_state, {.kind = StreamLeaseEventKind::begin_bootstrap}
        ));
    }, "poisoned Arm-S state accepted ordinary reuse");
    const auto reset = transition_stream_lease(
        poisoned.next_state, {.kind = StreamLeaseEventKind::authenticated_reset}
    );
    expect(reset.next_state.phase == StreamLeasePhase::unavailable,
        "authenticated Arm-S reset did not clear poison");

    expect_rejected<std::invalid_argument>([&] {
        auto stale = tree(9U);
        stale.session_epoch = 2U;
        static_cast<void>(transition_stream_lease(state, {
            .kind = StreamLeaseEventKind::publish_next_tree,
            .has_tree = true,
            .tree = stale,
        }));
    }, "stale Arm-S session lease was accepted");
    expect_rejected<std::logic_error>([&] {
        static_cast<void>(transition_stream_lease(
            state, {.kind = StreamLeaseEventKind::drain_completed}
        ));
    }, "illegal Arm-S steady-to-unavailable transition was accepted");

    const auto full_tree = tree(4U, 555U);
    auto full_pending = transition_stream_lease(state, {
        .kind = StreamLeaseEventKind::verification_full_match,
        .has_tree = true,
        .tree = full_tree,
    }).next_state;
    expect(full_pending.phase == StreamLeasePhase::pending_correction_bind,
        "Arm-S full match did not retain PendingTargetCandidateWithTree state");
    const auto endpoint = transition_stream_lease(
        full_pending, {.kind = StreamLeaseEventKind::endpoint}
    );
    expect(endpoint.next_state.phase == StreamLeasePhase::draining
            && endpoint.tree_action == NextTreeAction::discard,
        "Arm-S endpoint published unused future-tree work");
    const auto drained = transition_stream_lease(
        endpoint.next_state, {.kind = StreamLeaseEventKind::drain_completed}
    );
    expect(drained.next_state.phase == StreamLeasePhase::unavailable,
        "Arm-S drain/shutdown did not reach unavailable state");

    expect_rejected<std::logic_error>([&] {
        static_cast<void>(transition_stream_lease(
            state, {.kind = StreamLeaseEventKind::verification_mismatch}
        ));
    }, "Arm-S mismatch without an attached main-row tree was accepted");
}

void test_target_integrated_callbacks_discard_unpublished_tree_path() {
    BoundedSpeculativeController controller;
    const auto graph = make_forest_graph();
    const std::array<std::uint32_t, 3> path {101U, 102U, 103U};
    for (const auto decision : {
        ControllerCallbackDecision::stop_generation,
        ControllerCallbackDecision::throw_exception,
    }) {
        auto scenario = chain_scenario(graph, path, {});
        scenario.proposal_source = SpeculativeProposalSource::target_integrated_stream;
        scenario.route_mode = SpeculativeRouteMode::target_integrated;
        scenario.draft_schedule = SpeculativeDraftSchedule::target_integrated;
        scenario.verifier_topology = TargetVerifierTopology::target_integrated_forest;
        scenario.first_call_topology = FirstCallCommandTopology::target_integrated;
        scenario.callbacks[1] = decision;
        const auto outcome = controller.run(scenario);
        expect(outcome.selected_count == 2U && outcome.sampled_preview_count == 1U
                && outcome.next_candidate_lease.source == TargetStateLeaseSource::none
                && !outcome.has_carry,
            "Arm-S callback boundary published or carried later speculative state");
        for (std::size_t event = 0; event < outcome.trace_count; ++event) {
            expect(outcome.trace[event].kind != ControllerTraceEventKind::draft_execution,
                "target-integrated route was charged an external drafter call");
        }
    }
}

void test_arm_s_raw_pruned_shapes_and_ticket_slices_are_exhaustive() {
    const std::array<std::array<std::uint32_t, 3>, 7> cells {{
        {{4U, 3U, 121U}},
        {{3U, 3U, 40U}},
        {{4U, 1U, 5U}},
        {{4U, 2U, 31U}},
        {{7U, 1U, 8U}},
        {{15U, 1U, 16U}},
        {{23U, 1U, 24U}},
    }};
    for (const auto& cell : cells) {
        const auto gamma = cell[0];
        const auto branching = cell[1];
        const auto raw_nodes = cell[2];
        const auto table = generate_registered_stream_transition_table(gamma, branching);
        expect(table.count == raw_nodes + 1U,
            "Arm-S survivor transition table omitted a legal pruned count");
        for (std::uint32_t survivors = 0; survivors <= raw_nodes; ++survivors) {
            const auto& shape = table.values[survivors];
            expect(shape.raw_prior_node_count == raw_nodes
                    && shape.survivor_count == survivors
                    && shape.upper_row_count == survivors * (gamma + 1U)
                    && shape.next_node_count == survivors * raw_nodes
                    && shape.raw_child_count == survivors * gamma * branching,
                "Arm-S raw/pruned/upper transition shape mismatch");
        }
    }
    const auto primary = make_stream_transition_shape(4U, 3U, 121U);
    expect(primary.raw_prior_node_count == 121U
            && primary.upper_row_count == 605U
            && primary.next_node_count == 14'641U
            && primary.raw_child_count == 1'452U,
        "paper-primary gamma4/k3 Arm-S geometry mismatch");
    expect_rejected<std::overflow_error>([] {
        static_cast<void>(make_stream_transition_shape(23U, 8U, 1U));
    }, "impossible exponential Arm-S cross was silently clipped");
    expect_rejected<std::invalid_argument>([] {
        static_cast<void>(generate_registered_stream_transition_table(8U, 2U));
    }, "unregistered Arm-S Cartesian geometry was generated");

    constexpr std::uint32_t gamma = 4U;
    const auto map = generate_stream_anchor_ticket_map(gamma);
    expect(map.count == gamma + 1U && map.pool_size == 2U * gamma + 1U,
        "Arm-S anchor ticket pool size mismatch");
    for (std::uint32_t anchor = 0; anchor <= gamma; ++anchor) {
        expect(map.values[anchor].accepted_depth == anchor
                && map.values[anchor].first_ticket_offset == anchor
                && map.values[anchor].ticket_count == gamma + 1U
                && map.values[anchor].first_ticket_offset
                    + map.values[anchor].ticket_count <= map.pool_size,
            "Arm-S anchor ticket slice overlaps the wrong target offsets");
    }

    const auto options = TokenSamplerOptions {
        .temperature = 0.8F, .top_k = 4U, .top_p = 1.0F, .min_p = 0.0F,
        .repetition_penalty = 1.0F, .seed = 91U,
    };
    TokenSampler sampler(options);
    const std::uint32_t origin = 77U;
    sampler.observe_tokens(std::span<const std::uint32_t>(&origin, 1U));
    const auto pool = sampler.preview_tickets(map.pool_size);
    std::array<SamplerCheckpoint, gamma + 1U> anchors {};
    anchors[0] = sampler.checkpoint();
    auto prefix = sampler.fork();
    const std::array<TokenLogit, 4> candidates {{
        {1U, 1.0F}, {2U, 0.4F}, {3U, -0.2F}, {4U, -1.0F},
    }};
    for (std::uint32_t depth = 0; depth < gamma; ++depth) {
        const auto token = prefix.sample_with_ticket(pool.at(depth), candidates);
        prefix.observe_tokens(std::span<const std::uint32_t>(&token, 1U));
        anchors[depth + 1U] = prefix.checkpoint();
    }
    for (std::uint32_t anchor = 0; anchor <= gamma; ++anchor) {
        auto ticketed = sampler.fork(anchors[anchor]);
        auto direct = sampler;
        direct.adopt_checkpoint(anchors[anchor]);
        const auto slice = map.values[anchor];
        for (std::uint32_t offset = 0; offset < slice.ticket_count; ++offset) {
            const auto absolute = slice.first_ticket_offset + offset;
            const auto ticketed_token = ticketed.sample_with_ticket(pool.at(absolute), candidates);
            const auto direct_token = direct.sample(candidates);
            expect(ticketed_token == direct_token,
                "Arm-S anchor slice changed a future target draw");
            ticketed.observe_tokens(std::span<const std::uint32_t>(&ticketed_token, 1U));
            direct.observe_tokens(std::span<const std::uint32_t>(&direct_token, 1U));
            expect(ticketed.checkpoint().same_state(direct.checkpoint()),
                "Arm-S anchor slice diverged from its eventual direct checkpoint");
        }
    }
}

void test_controller_matches_independent_ordinary_token_callback_and_cache_loop() {
    BoundedSpeculativeController controller;
    for (const auto horizon : {3U, 8U, 16U, 23U}) {
        const auto graph = make_chain_graph(horizon);
        const auto schedule = single_tile_schedule(horizon);
        for (std::uint32_t accepted = 0; accepted <= horizon; ++accepted) {
            std::array<std::uint32_t, external_speculation_max_horizon> tokens {};
            std::size_t token_count = 0;
            for (std::uint32_t depth = 1; depth <= accepted; ++depth) {
                tokens[token_count++] = kRootToken + depth;
            }
            if (accepted < horizon) tokens[token_count++] = 900U - accepted;
            auto scenario = chain_scenario(
                graph, std::span<const std::uint32_t>(tokens).first(token_count), schedule
            );
            const auto outcome = controller.run(scenario);

            // Independent ordinary decoding of the same authoritative target
            // samples caches and calls back each selected non-stop token in
            // sequence. Complete an uncached speculative carry through that
            // ordinary boundary before comparing states.
            std::array<std::uint32_t, controller_max_selected_tokens> ordinary_selected {};
            ordinary_selected[0] = kRootToken;
            for (std::size_t index = 0; index < token_count; ++index) {
                ordinary_selected[index + 1U] = tokens[index];
            }
            const auto ordinary_count = static_cast<std::uint32_t>(token_count + 1U);
            std::uint32_t completed_cached = outcome.logical_cached_count;
            std::uint32_t completed_callbacks = outcome.callback_count;
            if (outcome.has_carry && !outcome.carry_is_cached) {
                ++completed_cached;
                ++completed_callbacks;
            }
            expect(outcome.selected_count == ordinary_count,
                "controller selected sequence length differs from ordinary target decoding");
            for (std::size_t index = 0; index < ordinary_count; ++index) {
                expect(outcome.selected_tokens[index] == ordinary_selected[index],
                    "controller token sequence differs from ordinary target decoding");
            }
            expect(completed_cached
                    == scenario.initial_logical_cached_count + ordinary_count
                    && completed_callbacks == ordinary_count,
                "controller plus ordinary carry completion changed callback/cache semantics");
            expect(outcome.adopted_checkpoint.same_state(
                    scenario.target_samples[token_count - 1U].checkpoint),
                "controller final sampler state differs from ordinary target decoding");
        }
    }
}

void test_corrected_trajectory_replay_charges_only_incurred_events() {
    BoundedSpeculativeController controller;
    const auto graph = make_chain_graph(3U);
    const std::array<std::uint32_t, 2> target {101U, 900U};
    const auto outcome = controller.run(chain_scenario(graph, target, single_tile_schedule(3U)));

    MeasuredConditionalCostTable costs;
    for (std::size_t event = 0; event < outcome.trace_count; ++event) {
        const auto key = outcome.trace[event].cost;
        if (key.kind == ControllerCostKind::none) continue;
        bool exists = false;
        for (std::size_t index = 0; index < costs.count; ++index) {
            exists = exists || costs.values[index].key == key;
        }
        if (!exists) {
            costs.values[costs.count++] = MeasuredConditionalCost {
                .key = key,
                .duration_ns = 10.0 + static_cast<double>(static_cast<std::uint8_t>(key.kind)),
            };
        }
    }
    double expected = 0.0;
    std::uint32_t expected_events = 0;
    for (std::size_t event = 0; event < outcome.trace_count; ++event) {
        if (outcome.trace[event].cost.kind == ControllerCostKind::none) continue;
        for (std::size_t index = 0; index < costs.count; ++index) {
            if (costs.values[index].key == outcome.trace[event].cost) {
                expected += costs.values[index].duration_ns;
                ++expected_events;
                break;
            }
        }
    }
    const auto replay = replay_corrected_trajectory_cost(outcome, costs);
    expect(replay.duration_ns == expected && replay.charged_event_count == expected_events,
        "corrected replay did not charge the realized conditional trajectory");

    auto missing = costs;
    --missing.count;
    expect_rejected<std::invalid_argument>([&] {
        static_cast<void>(replay_corrected_trajectory_cost(outcome, missing));
    }, "trajectory replay accepted a missing measured cost");

    expect(std::abs(diagnostic_iid_expected_tokens(0.5, 3U) - 1.875) < 1.0e-12,
        "iid diagnostic equation mismatch");
    expect_rejected<std::invalid_argument>([] {
        static_cast<void>(diagnostic_iid_expected_tokens(1.1, 3U));
    }, "iid diagnostic accepted an invalid probability");
}

void test_live_callback_pause_matches_scripted_controller() {
    BoundedSpeculativeController controller;
    for (auto horizon : {3U, 23U}) {
        const auto graph = make_chain_graph(horizon);
        std::array<std::uint32_t,23> tokens {};
        for (unsigned i = 0; i < horizon; ++i) tokens[i] = kRootToken + i + 1;
        for (unsigned stopping_depth = 0; stopping_depth <= horizon; ++stopping_depth) {
            auto scripted = chain_scenario(graph, std::span(tokens).first(horizon), single_tile_schedule(horizon));
            scripted.callbacks[stopping_depth] = ControllerCallbackDecision::stop_generation;
            const auto expected = controller.run(scripted);
            auto live = scripted;
            live.callback_decision_count = 0;
            for (unsigned depth = 0; depth <= stopping_depth; ++depth) {
                const auto paused = controller.run_available(live);
                expect(paused.requires_callback && paused.required_callback_depth == depth
                    && paused.selected_count == depth + 1 && paused.retained_row_count == depth + 1,
                    "live controller advanced beyond an unresolved callback receipt");
                expect(paused.callback_count == depth,
                    "uninvoked callback was reported as completed");
                live.callback_decision_count = depth + 1;
            }
            const auto actual = controller.run_available(live);
            expect(!actual.requires_callback && actual.stop_reason == expected.stop_reason
                && actual.selected_tokens == expected.selected_tokens
                && actual.retained_row_count == expected.retained_row_count
                && actual.adopted_checkpoint.same_state(expected.adopted_checkpoint),
                "live callback decisions diverged from scripted state machine");
        }
    }
}

void test_lossless_controller_adopts_only_causal_randomness() {
    BoundedSpeculativeController controller;
    for (const auto horizon : {3U, 23U}) {
        auto storage = chain_storage(horizon, ChainTileEndpoint::post_full_candidate);
        storage.header.primary_policy = ProposalPrimaryPolicy::independent_distribution;
        storage.nodes.proposal_ticket_offsets.fill(proposal_ticket_none);
        storage.materialized_edges.proposal_ticket_offsets.fill(proposal_ticket_none);
        const auto graph = validate_proposal_graph(std::move(storage), validation_context());
        for (const auto rejection_depth : {0U, 1U, horizon - 1U, horizon}) {
            auto scenario = chain_scenario(graph, {}, single_tile_schedule(horizon));
            scenario.sampling_mode = SpeculativeSamplingMode::lossless;
            TokenSampler sampler({.temperature=0.8F, .top_k=1, .seed=42});
            sampler.observe_tokens(std::span(&kRootToken, 1));
            scenario.origin_checkpoint = sampler.checkpoint();
            auto fork = sampler.fork();
            for (unsigned i = 0; i < horizon; ++i) {
                const auto proposal = kRootToken + i + 1;
                DraftTokenDistribution q;
                q.count = 1;
                q.values[0] = {proposal, std::uint64_t{1} << 32};
                const std::array<TokenLogit, 1> p {{{i == rejection_depth ? 900U : proposal, 1.0F}}};
                const auto selected = fork.sample_speculative(proposal, q, p);
                fork.observe_tokens(std::span(&selected.token, 1));
                scenario.target_samples[i] = {selected.token, fork.checkpoint(), selected.accepted
                    ? ControllerSampleLaw::rejection_accept : ControllerSampleLaw::rejection_residual};
                ++scenario.target_sample_count;
                if (!selected.accepted) break;
            }
            const auto completed = controller.run(scenario);
            expect(completed.accepted_edge_count == rejection_depth,
                "lossless controller retained a rejected proposal or lost an acceptance");
            expect(completed.adopted_checkpoint.same_state(fork.checkpoint()),
                "lossless controller did not adopt the causal proposal/acceptance RNG state");
            for (unsigned depth = 0; depth < completed.selected_count; ++depth) {
                auto stopped = scenario;
                stopped.callbacks[depth] = ControllerCallbackDecision::stop_generation;
                const auto result = controller.run(stopped);
                const auto& expected = depth == 0 ? scenario.origin_checkpoint
                    : scenario.target_samples[depth - 1].checkpoint;
                expect(result.adopted_checkpoint.same_state(expected),
                    "callback stop adopted unused future lossless random draws");
            }
            auto malformed = scenario;
            malformed.target_samples[0].law = ControllerSampleLaw::target_draw;
            expect_rejected<std::invalid_argument>([&] { static_cast<void>(controller.run(malformed)); },
                "lossless controller accepted an unaudited target-draw transition");
            malformed = scenario;
            malformed.sampling_mode = SpeculativeSamplingMode::exact_match;
            expect_rejected<std::invalid_argument>([&] { static_cast<void>(controller.run(malformed)); },
                "legacy controller accepted a rejection-sampling checkpoint");
        }
    }
}

void test_lossless_conditional_tiles_preserve_decisions_and_randomness() {
    BoundedSpeculativeController controller;
    for(const auto horizon:{3U,4U,7U,8U,15U,16U,23U}) {
        auto storage=chain_storage(horizon,ChainTileEndpoint::post_full_candidate);
        storage.header.primary_policy=ProposalPrimaryPolicy::independent_distribution;
        storage.nodes.proposal_ticket_offsets.fill(proposal_ticket_none);
        storage.materialized_edges.proposal_ticket_offsets.fill(proposal_ticket_none);
        const auto graph=validate_proposal_graph(std::move(storage),validation_context());
        for(std::uint32_t rejected=0;rejected<=horizon;++rejected) {
            auto reference=chain_scenario(graph,{},single_tile_schedule(horizon));
            reference.sampling_mode=SpeculativeSamplingMode::lossless;
            TokenSampler sampler({.temperature=0.8F,.top_k=1,.seed=42});
            sampler.observe_tokens(std::span(&kRootToken,1));
            reference.origin_checkpoint=sampler.checkpoint();
            auto fork=sampler.fork();
            for(std::uint32_t edge=0;edge<horizon;++edge) {
                const auto proposal=kRootToken+edge+1;
                DraftTokenDistribution q;
                q.count=1;
                q.values[0]={proposal,draft_probability_mass_total};
                const std::array<TokenLogit,1> p {{{edge==rejected?900U:proposal,1.0F}}};
                const auto selected=fork.sample_speculative(proposal,q,p);
                fork.observe_tokens(std::span(&selected.token,1));
                reference.target_samples[edge]={selected.token,fork.checkpoint(),selected.accepted
                    ?ControllerSampleLaw::rejection_accept:ControllerSampleLaw::rejection_residual};
                ++reference.target_sample_count;
                if(!selected.accepted) break;
            }
            const auto compare=[&](const SpeculativeControllerScenario& single) {
                auto conditional=single;
                conditional.chain_schedule=make_registered_conditional_chain_schedule(
                    horizon,ChainTileEndpoint::post_full_candidate);
                conditional.verifier_topology=TargetVerifierTopology::conditional_tile_cascade;
                conditional.draft_schedule=SpeculativeDraftSchedule::conditional_tiles;
                const auto expected=controller.run(single);
                const auto actual=controller.run(conditional);
                expect(actual.selected_count==expected.selected_count
                    && actual.selected_tokens==expected.selected_tokens
                    && actual.accepted_edge_count==expected.accepted_edge_count
                    && actual.adopted_checkpoint.same_state(expected.adopted_checkpoint),
                    "conditional lossless scheduling changed selection or causal RNG ownership");
                expect(actual.logical_cached_count==expected.logical_cached_count
                    && actual.retained_row_count+actual.endpoint_kv_only_count
                        ==expected.retained_row_count+expected.endpoint_kv_only_count
                    && actual.stop_reason==expected.stop_reason
                    && actual.disposition==expected.disposition
                    && actual.has_carry==expected.has_carry
                    && (!actual.has_carry || actual.carry_token==expected.carry_token)
                    && actual.callback_count==expected.callback_count,
                    "conditional lossless scheduling changed terminal cache/carry/callback state");
                for(std::uint32_t callback=0;callback<actual.callback_count;++callback) {
                    const auto& a=actual.callback_records[callback];
                    const auto& b=expected.callback_records[callback];
                    expect(a.token_id==b.token_id && a.logical_selected_count==b.logical_selected_count
                        && a.logical_cached_count==b.logical_cached_count && a.decision==b.decision,
                        "conditional lossless callback observed a different cache receipt");
                }
                expect(actual.next_candidate_lease.source==expected.next_candidate_lease.source
                    && actual.next_candidate_lease.node_index==expected.next_candidate_lease.node_index,
                    "conditional lossless scheduling lost the full-prefix next candidate");
            };
            compare(reference);
            const auto selected_count=std::min(rejected+2,horizon+1);
            for(std::uint32_t depth=0;depth<selected_count;++depth) {
                for(const auto decision:{ControllerCallbackDecision::stop_generation,
                    ControllerCallbackDecision::throw_exception}) {
                    auto stopped=reference;
                    stopped.callbacks[depth]=decision;
                    compare(stopped);
                }
                auto stopped=reference;
                stopped.token_is_stop[depth]=true;
                compare(stopped);
                stopped=reference;
                stopped.remaining_selected_token_budget=depth+1;
                compare(stopped);
            }
        }
    }
}

const TestCase kTests[] = {
    {"lossless conditional tiles preserve all-horizon decisions randomness and terminal receipts",
        test_lossless_conditional_tiles_preserve_decisions_and_randomness},
    {"lossless controller adopts only causal proposal and acceptance randomness",
        test_lossless_controller_adopts_only_causal_randomness},
    {"live callback pauses preserve the pure controller's exact commit floor",
        test_live_callback_pause_matches_scripted_controller},
    {"registered speculative chain transitions are exhaustive",
        test_registered_chain_transitions_are_exhaustive},
    {"speculative chain budgets stops and callbacks at every depth",
        test_chain_budgets_stops_and_callbacks_at_every_depth},
    {"H3 conditional tile order gates context and terminal carry",
        test_h3_conditional_tile_order_gates_context_and_terminal_carry},
    {"dynamic horizons and every conditional tile gate",
        test_dynamic_horizons_and_every_conditional_tile_gate},
    {"resumable controller pauses before each causal target sample",
        test_resumable_controller_pauses_before_each_causal_target_sample},
    {"terminal carry stop and budget endpoints are cached once",
        test_terminal_carry_stop_and_budget_endpoints_are_cached_once},
    {"speculative forest and output frontier walk one causal path",
        test_forest_and_output_frontier_walk_one_causal_path},
    {"repeated-root beams preserve prefix candidates",
        test_repeated_root_beams_preserve_prefix_candidates},
    {"source-shaped ReDrafter L5 and EAGLE3 depth8 are exact",
        test_source_shaped_redrafter_l5_and_eagle3_depth8},
    {"target verifier source shape registry is complete and typed",
        test_target_verifier_source_shape_registry_is_complete_and_typed},
    {"routes and proposal randomness remain distinct",
        test_routes_and_proposal_randomness_remain_distinct},
    {"proposal graph schedule and capacity validation fail closed",
        test_graph_schedule_and_capacity_validation_fail_closed},
    {"Arm-S lease state machine and poisoning", test_arm_s_lease_state_machine_and_poisoning},
    {"target-integrated callbacks discard unpublished tree path",
        test_target_integrated_callbacks_discard_unpublished_tree_path},
    {"Arm-S raw pruned shapes and ticket slices are exhaustive",
        test_arm_s_raw_pruned_shapes_and_ticket_slices_are_exhaustive},
    {"controller matches independent ordinary token callback and cache loop",
        test_controller_matches_independent_ordinary_token_callback_and_cache_loop},
    {"corrected trajectory replay charges only incurred events",
        test_corrected_trajectory_replay_charges_only_incurred_events},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> speculative_controller_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
