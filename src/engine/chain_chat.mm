#include "local_gguf_engine/engine/chain_chat.hpp"
#include "gemma2_verification_state_access.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::engine {
namespace {
bool is_stop(std::uint32_t token, std::span<const std::uint32_t> stop_tokens) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
}

void add_target_stats(TargetVerifierExecutionStats& total,
    const TargetVerifierExecutionStats& value,
    bool encoded_shape = true) {
#define ADD_TARGET_FIELD(field)                                                                    \
    if (value.field > std::numeric_limits<decltype(total.field)>::max() - total.field)             \
        throw std::overflow_error("chain target " #field " overflow");                             \
    total.field += value.field
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
    if (encoded_shape) {
        ADD_TARGET_FIELD(input_row_count);
        ADD_TARGET_FIELD(proposal_edge_count);
    }
    ADD_TARGET_FIELD(attention_dispatch_count);
    ADD_TARGET_FIELD(attention_partition_count);
    ADD_TARGET_FIELD(attention_token_count);
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

std::uint64_t append_digest(std::uint64_t value, std::uint32_t token) {
    for (unsigned byte = 0; byte < 4; ++byte) {
        value ^= (token >> (8 * byte)) & 255U;
        value *= 1099511628211ULL;
    }
    return value;
}

ChainVerifierSchedule unbounded_verification_schedule(std::uint32_t horizon, bool conditional) {
    if (conditional) {
        switch (horizon) {
            case 3:
            case 4:
            case 7:
            case 8:
            case 15:
            case 16:
            case 23:
                return make_registered_conditional_chain_schedule(
                    horizon, ChainTileEndpoint::post_full_candidate);
            default: {
                // Standalone models also support unregistered chain lengths.
                // The established conditional partitions remain unchanged.
                ChainVerifierSchedule schedule;
                for (std::uint32_t remaining = horizon; remaining;) {
                    const auto count = std::min(remaining, 2U);
                    remaining -= count;
                    schedule.tiles[schedule.tile_count++] = {static_cast<std::uint16_t>(count),
                        static_cast<std::uint16_t>(count + (remaining == 0 ? 1U : 0U))};
                }
                validate_chain_verifier_schedule(schedule, horizon);
                return schedule;
            }
        }
    }
    ChainVerifierSchedule schedule;
    schedule.tiles[0] = {
        static_cast<std::uint16_t>(horizon), static_cast<std::uint16_t>(horizon + 1)};
    schedule.tile_count = 1;
    return schedule;
}

ChainVerifierSchedule verification_schedule(std::uint32_t horizon,
    bool conditional,
    std::uint32_t row_capacity = external_speculation_max_horizon + 1) {
    auto schedule = unbounded_verification_schedule(horizon, conditional);
    bool fits = true;
    for (std::uint32_t tile = 0; tile < schedule.tile_count; ++tile)
        fits = fits && schedule.tiles[tile].input_row_count <= row_capacity;
    if (fits)
        return schedule;
    if (!conditional || row_capacity < 2)
        throw std::invalid_argument("chain schedule exceeds its prepared row capacity");
    // Registered widths are not monotonic in H: H=3 owns two rows whereas
    // H=2 ordinarily asks for three; H=23 owns eight whereas H=16 asks for
    // nine. Repartition shortened cycles within their existing arena. The
    // last tile reserves one row for the post-full candidate.
    schedule = {};
    for (std::uint32_t remaining = horizon; remaining;) {
        const auto count =
            remaining < row_capacity ? remaining : std::min(remaining - 1, row_capacity);
        remaining -= count;
        schedule.tiles[schedule.tile_count++] = {static_cast<std::uint16_t>(count),
            static_cast<std::uint16_t>(count + (remaining == 0 ? 1U : 0U))};
    }
    validate_chain_verifier_schedule(schedule, horizon);
    return schedule;
}

std::uint32_t prepared_rows(std::uint32_t horizon, bool conditional) {
    if (horizon == 0 || horizon > external_speculation_max_horizon)
        throw std::invalid_argument("invalid chain verifier horizon");
    const auto schedule = verification_schedule(horizon, conditional);
    std::uint32_t rows = 0;
    for (std::uint32_t index = 0; index < schedule.tile_count; ++index)
        rows = std::max(rows, std::uint32_t(schedule.tiles[index].input_row_count));
    return rows;
}
} // namespace

struct ChainChatVerifier::Impl {
    Gemma2MetalTargetVerifierPlan verifier;
    ProposalGraphStorage graph;
    VerifierEncodeRequest request;
    VerifierEncodeRequest tile_request;
    std::optional<ValidatedProposalGraph> validated_graph;
    SpeculativeControllerScenario scenario;
    std::uint32_t horizon;
    SpeculativeSamplingMode sampling_mode;
    bool conditional;
    bool warmed_tiles = false;
    std::uint64_t epoch = 0;
    Impl(const Gemma2MetalModelPlan& target,
        std::uint32_t proposal_horizon,
        SpeculativeSamplingMode mode,
        bool conditional_value)
        : verifier(create_gemma2_metal_target_verifier_plan(
              target, prepared_rows(proposal_horizon, conditional_value))),
          horizon(proposal_horizon), sampling_mode(mode), conditional(conditional_value) {
        if (mode != SpeculativeSamplingMode::exact_match &&
            mode != SpeculativeSamplingMode::lossless)
            throw std::invalid_argument("invalid chain verifier sampling mode");
    }
};

ChainChatVerifier::ChainChatVerifier(const Gemma2MetalModelPlan& target,
    std::uint32_t horizon,
    SpeculativeSamplingMode mode,
    bool conditional)
    : impl_(std::make_unique<Impl>(target, horizon, mode, conditional)) {}

ChainChatVerifier::~ChainChatVerifier() = default;
ChainChatVerifier::ChainChatVerifier(ChainChatVerifier&&) noexcept = default;
ChainChatVerifier& ChainChatVerifier::operator=(ChainChatVerifier&&) noexcept = default;
std::size_t ChainChatVerifier::owned_bytes() const noexcept {
    return sizeof(Impl) + Gemma2MetalTargetVerifierPlan::host_storage_byte_count()
        + impl_->verifier.allocated_byte_count();
}

std::size_t ChainChatVerifier::estimate_owned_bytes(
    const GemmaModelConfig& config, std::uint32_t horizon, bool conditional) {
    const auto rows = prepared_rows(horizon, conditional);
    const auto arena = estimate_target_verifier_bytes(config, rows);
    const auto host = sizeof(Impl) + Gemma2MetalTargetVerifierPlan::host_storage_byte_count();
    if (arena > std::numeric_limits<std::size_t>::max() - host)
        throw std::length_error("chain verifier storage estimate overflow");
    return arena + host;
}

#if defined(LOCAL_GGUF_ENGINE_TESTING)
ChainVerifierSchedule ChainChatVerifier::schedule_for_testing(
    std::uint32_t prepared_horizon, std::uint32_t active_horizon, bool conditional) {
    if (active_horizon == 0 || active_horizon > prepared_horizon)
        throw std::invalid_argument("active chain horizon exceeds its preparation");
    return verification_schedule(
        active_horizon, conditional, prepared_rows(prepared_horizon, conditional));
}
#endif

void ChainChatVerifier::warm_up_tiles(Gemma2MetalModelPlan& target) {
    auto& state = *impl_;
    if (state.warmed_tiles || state.epoch == 0 || !state.validated_graph)
        throw std::logic_error("chain verifier tile warmup is not at its first setup boundary");
    state.warmed_tiles = true;
    if (!state.conditional)
        return;
    const auto& config = target.config();
    const auto request = validate_verifier_encode_request(state.request,
        config.vocab_size,
        config.block_count,
        config.context_length,
        config.sliding_window,
        config.attention_global_layer_period);
    std::array<std::uint32_t, external_speculation_max_horizon + 1> row_path{};
    for (std::uint32_t row = 0; row < row_path.size(); ++row)
        row_path[row] = row;
    // Setup rows are not a generated sequence. Publishing the complete setup
    // prefix initializes attention for each later shape; generation restarts
    // from logical position zero and can never observe these extra rows.
    for (std::uint32_t tile = 0; tile < state.request.topology.tile_count; ++tile) {
        state.tile_request.selection_request_count = 0;
        state.tile_request.topology.input_row_count = 0;
        state.tile_request.device_inputs.reset();
        make_chain_verifier_tile_request(state.tile_request,
            request,
            tile,
            config.block_count,
            config.context_length,
            config.sliding_window,
            config.attention_global_layer_period);
        auto validated = validate_verifier_encode_request(state.tile_request,
            config.vocab_size,
            config.block_count,
            config.context_length,
            config.sliding_window,
            config.attention_global_layer_period);
        auto ticket = encode_gemma2_metal_target_verifier(target,
            state.verifier,
            validated,
            gemma_input_embedding_scale(config.embedding_length));
        submit_gemma2_metal_target_verifier(ticket);
        static_cast<void>(complete_gemma2_metal_target_verifier(ticket));
        const auto rows = state.tile_request.topology.input_row_count;
        static_cast<void>(provisionally_commit_gemma2_metal_target_verifier(
            ticket, std::span<const std::uint32_t>(row_path).first(rows)));
        static_cast<void>(finalize_gemma2_metal_target_verifier(ticket, rows));
    }
}

ChainCycleResult ChainChatVerifier::run(Gemma2MetalModelPlan& target,
    TokenSampler& sampler,
    std::uint32_t root_token,
    std::uint32_t position,
    std::size_t remaining,
    std::span<const std::uint32_t> stop_tokens,
    const ChainProposalBatch& batch,
    const ChainSampleCallback& sample,
    const ChainCommitCallback& callback) {
    auto& state = *impl_;
    const auto& config = target.config();
    const auto proposal_horizon = static_cast<std::uint32_t>(batch.tokens.size());
    if (remaining < 2 || batch.tokens.empty() || batch.tokens.size() > state.horizon ||
        position > config.context_length ||
        proposal_horizon + 1 > config.context_length - position ||
        is_stop(root_token, stop_tokens) || batch.device.arena_epoch == 0 ||
        batch.device.arena_epoch <= state.epoch ||
        (!batch.confidence_logits.empty() && batch.confidence_logits.size() != proposal_horizon))
        throw std::invalid_argument("chain proposal has incompatible epoch, horizon or capacity");
    state.epoch = batch.device.arena_epoch;
    ChainCycleResult result;
    result.proposal_count = proposal_horizon;
    const bool lossless = state.sampling_mode == SpeculativeSamplingMode::lossless;
    auto& graph = state.graph;
    graph.header = {.kind = ProposalGraphKind::chain,
        .horizon = proposal_horizon,
        .materialized_node_count = proposal_horizon + 1,
        .materialized_edge_count = proposal_horizon,
        .root_count = 1,
        .root_indices = {0},
        .graph_identity = state.epoch,
        .device_arena_epoch = state.epoch};
    graph.header.primary_policy = lossless ? ProposalPrimaryPolicy::independent_distribution
                                           : ProposalPrimaryPolicy::scalar_common_draw;
    auto digest = std::uint64_t{14695981039346656037ULL};
    const auto& history = sampler.history_snapshot();
    for (std::uint32_t index = 0; index < history.length; ++index)
        digest =
            append_digest(digest, history.token_ids[(history.head + index) % history.window_limit]);
    const ProposalPathHistoryDescriptor root_history{digest, position + 1};
    for (std::uint32_t row = 0; row <= proposal_horizon; ++row) {
        const auto token = row == 0 ? root_token : batch.tokens[row - 1];
        if (row)
            digest = append_digest(digest, token);
        graph.nodes.token_ids[row] = token;
        graph.nodes.parent_indices[row] = row == 0 ? proposal_node_none : row - 1;
        graph.nodes.depths[row] = static_cast<std::uint16_t>(row);
        graph.nodes.absolute_positions[row] = position + row;
        graph.nodes.mask_component_roots[row] = 0;
        graph.nodes.mask_ancestor_counts[row] = static_cast<std::uint16_t>(row + 1);
        graph.nodes.conditional_confidences[row] = row == 0
            ? 1.0F
            : (batch.confidence_logits.empty()
                      ? 1.0F
                      : 1.0F / (1.0F + std::exp(-batch.confidence_logits[row - 1])));
        graph.nodes.path_histories[row] = {digest, position + row + 1};
        graph.nodes.proposal_ticket_offsets[row] =
            row == 0 || lossless ? proposal_ticket_none : static_cast<std::uint16_t>(row - 1);
        graph.nodes.feature_provenance[row] = {.source = batch.source, .arena_epoch = state.epoch};
        graph.nodes.output_roles[row] = row == proposal_horizon
            ? ProposalRowOutputRole::post_full_candidate
            : ProposalRowOutputRole::verifies_proposals;
        if (row) {
            const auto edge = row - 1;
            graph.materialized_edges.parent_node_indices[edge] = edge;
            graph.materialized_edges.child_node_indices[edge] = row;
            graph.materialized_edges.token_ids[edge] = token;
            graph.materialized_edges.conditional_confidences[edge] =
                graph.nodes.conditional_confidences[row];
            graph.materialized_edges.sibling_ranks[edge] = 0;
            graph.materialized_edges.proposal_ticket_offsets[edge] =
                lossless ? proposal_ticket_none : static_cast<std::uint16_t>(edge);
        }
        state.request.selection_requests[row] = sampler.device_selection_request(
            std::span<const std::uint32_t>(batch.tokens).first(row));
    }
    state.validated_graph.emplace(validate_proposal_graph(graph,
        {.vocabulary_size = config.vocab_size,
            .context_length = config.context_length,
            .expected_root_token = root_token,
            .expected_root_position = position,
            .expected_root_history = root_history,
            .expected_device_arena_epoch = state.epoch}));
    const auto schedule =
        verification_schedule(proposal_horizon, state.conditional, state.verifier.row_capacity());
    state.request.topology = make_verifier_topology_descriptor(*state.validated_graph,
        schedule,
        config.block_count,
        config.context_length,
        config.sliding_window,
        config.attention_global_layer_period);
    state.request.selection_request_count = proposal_horizon + 1;
    state.request.sampler_options = sampler.options();
    state.request.input_residency = VerifierInputResidency::device_resident;
    state.request.device_inputs = batch.device;
    auto request = validate_verifier_encode_request(state.request,
        config.vocab_size,
        config.block_count,
        config.context_length,
        config.sliding_window,
        config.attention_global_layer_period);
    // Previews stay on a fork. The pure controller, resumed after each live
    // callback, owns selection/adoption/stop/budget/carry decisions.
    auto& scenario = state.scenario;
    scenario = {};
    scenario.graph = &*state.validated_graph;
    scenario.chain_schedule = schedule;
    scenario.verifier_topology = schedule.tile_count > 1
        ? TargetVerifierTopology::conditional_tile_cascade
        : TargetVerifierTopology::single_invocation_post_full;
    scenario.draft_schedule = schedule.tile_count > 1 ? SpeculativeDraftSchedule::conditional_tiles
                                                      : SpeculativeDraftSchedule::eager;
    scenario.tile_gates.fill(ControllerTileGateDecision::accept);
    scenario.proposal_source = SpeculativeProposalSource::learned_external;
    scenario.sampling_mode = state.sampling_mode;
    scenario.origin_token = root_token;
    scenario.origin_checkpoint = sampler.checkpoint();
    scenario.remaining_selected_token_budget = static_cast<std::uint32_t>(remaining);
    scenario.initial_logical_cached_count = position;
    scenario.available_context_rows = config.context_length - position;
    scenario.callback_decision_count = 0;
    auto sampler_fork = sampler.fork();
    std::array<std::uint32_t, external_speculation_max_horizon + 1> row_path{};
    for (std::uint32_t index = 0; index < row_path.size(); ++index)
        row_path[index] = index;
    BoundedSpeculativeController controller;
    std::optional<TargetVerificationTicket> ticket;
    TargetVerifierCompletionResult completion;
    std::uint32_t active_tile_index = 0;
    std::uint32_t active_first_row = 0;
    std::uint32_t active_row_count = 0;
    std::uint32_t committed_rows = 0;
    bool provisionally_committed = false;
    bool endpoint_done = false;
    std::uint32_t delivered_rows = 0;
    std::uint32_t reported_tokens = 0;
    SpeculativeControllerOutcome progress;
    const auto finalize = [&](std::uint32_t retained) {
        if (!ticket)
            return;
        if (!provisionally_committed || retained < active_first_row ||
            retained - active_first_row > active_row_count)
            throw std::logic_error("chain tile retained prefix is outside its transaction");
        add_target_stats(result.target,
            finalize_gemma2_metal_target_verifier(*ticket, retained - active_first_row).execution,
            false);
        committed_rows = retained;
        ticket.reset();
        provisionally_committed = false;
    };
    try {
        while (true) {
            progress = controller.run_available(scenario);
            if (progress.requires_target_sample) {
                const auto tile = std::uint32_t(progress.required_target_sample_tile);
                const auto sample_offset = progress.required_target_sample_offset;
                if (tile >= schedule.tile_count || sample_offset != scenario.target_sample_count ||
                    (ticket && tile != active_tile_index + 1))
                    throw std::logic_error("live chain controller requested a non-causal tile");
                if (ticket) {
                    if (progress.retained_row_count != active_first_row + active_row_count)
                        throw std::logic_error("chain tile advanced beyond an uncommitted prefix");
                    finalize(progress.retained_row_count);
                }
                active_tile_index = tile;
                active_first_row = state.request.topology.tiles[tile].first_row;
                active_row_count = state.request.topology.tiles[tile].row_count;
                if (sample_offset != active_first_row || committed_rows != active_first_row)
                    throw std::logic_error("chain tile does not follow its committed prefix");
                // The tile rebases attention ancestry, not sampler history or
                // device tokens. Earlier accepted rows already live in K/V;
                // no proposal upload or accepted-prefix replay is involved.
                if (schedule.tile_count > 1) {
                    state.tile_request.selection_request_count = 0;
                    state.tile_request.topology.input_row_count = 0;
                    state.tile_request.device_inputs.reset();
                    make_chain_verifier_tile_request(state.tile_request,
                        request,
                        tile,
                        config.block_count,
                        config.context_length,
                        config.sliding_window,
                        config.attention_global_layer_period);
                    auto validated = validate_verifier_encode_request(state.tile_request,
                        config.vocab_size,
                        config.block_count,
                        config.context_length,
                        config.sliding_window,
                        config.attention_global_layer_period);
                    ticket.emplace(encode_gemma2_metal_target_verifier(target,
                        state.verifier,
                        validated,
                        gemma_input_embedding_scale(config.embedding_length)));
                } else {
                    ticket.emplace(encode_gemma2_metal_target_verifier(target,
                        state.verifier,
                        request,
                        gemma_input_embedding_scale(config.embedding_length)));
                }
                submit_gemma2_metal_target_verifier(*ticket);
                completion = complete_gemma2_metal_target_verifier(*ticket);
                add_target_stats(result.target, completion.execution);
                std::uint32_t matched_edges = 0;
                const auto verified_edge_end =
                    active_first_row + schedule.tiles[tile].validated_edge_count;
                for (std::uint32_t index = active_first_row;
                    index < verified_edge_end && index + 1 < remaining;
                    ++index) {
                    const auto& candidates = completion.candidates[index - active_first_row];
                    const auto values =
                        std::span<const TokenLogit>(candidates.values).first(candidates.count);
                    std::uint32_t token;
                    auto law = ControllerSampleLaw::target_draw;
                    const auto selected = sample(sampler_fork, index, values);
                    token = selected.token;
                    if (lossless)
                        law = selected.accepted ? ControllerSampleLaw::rejection_accept
                                                : ControllerSampleLaw::rejection_residual;
                    sampler_fork.observe_tokens(std::span<const std::uint32_t>(&token, 1));
                    scenario.target_samples[index] = {token, sampler_fork.checkpoint(), law};
                    scenario.token_is_stop[index + 1] = is_stop(token, stop_tokens);
                    ++scenario.target_sample_count;
                    if (token != batch.tokens[index])
                        break;
                    ++matched_edges;
                    if (is_stop(token, stop_tokens))
                        break;
                }
                add_target_stats(result.target,
                    provisionally_commit_gemma2_metal_target_verifier(*ticket,
                        std::span<const std::uint32_t>(row_path).first(
                            std::min(matched_edges + 1, active_row_count)))
                        .execution,
                    false);
                provisionally_committed = true;
                continue;
            }
            sampler.adopt_checkpoint(progress.adopted_checkpoint);
            if (progress.endpoint_kv_only_count && !endpoint_done) {
                if (progress.endpoint_kv_only_count != 1)
                    throw std::logic_error("chain endpoint count");
                finalize(progress.retained_row_count);
                try {
                    result.endpoint = run_gemma2_metal_target_verifier_endpoint(target,
                        progress.endpoint_kv_only_tokens[0],
                        position + progress.retained_row_count,
                        gemma_input_embedding_scale(config.embedding_length));
                } catch (...) {
                    Gemma2MetalVerificationStateAccess::poison(target);
                    throw;
                }
                endpoint_done = true;
            }
            if (!progress.requires_callback)
                break;
            const auto depth = progress.required_callback_depth;
            if (depth != reported_tokens)
                throw std::logic_error("non-causal live callback depth");
            delivered_rows = progress.retained_row_count;
            ++reported_tokens; // the throwing callback is an irreversible receipt
            const auto decision = callback(
                {progress.selected_tokens[depth], depth, progress.logical_cached_count, false});
            scenario.callbacks[depth] =
                decision == Gemma2MetalTokenStreamDecision::continue_generation
                ? ControllerCallbackDecision::continue_generation
                : ControllerCallbackDecision::stop_generation;
            scenario.callback_decision_count = depth + 1;
        }
        if (progress.next_candidate_lease.source == TargetStateLeaseSource::selected_verifier_row) {
            const auto row = progress.next_candidate_lease.node_index;
            if (row != proposal_horizon || progress.retained_row_count != proposal_horizon + 1 ||
                !ticket || row < active_first_row || row - active_first_row >= active_row_count)
                throw std::logic_error("invalid post-full lease");
            const auto local_row = row - active_first_row;
            result.next_candidates = completion.candidates[local_row];
            result.has_next_candidates = true;
        }
        finalize(progress.retained_row_count);
        result.retained_rows = progress.retained_row_count + progress.endpoint_kv_only_count;
        result.accepted_edges = progress.accepted_edge_count;
        result.rejected = progress.selected_count > progress.accepted_edge_count + 1U;
        result.callback_stopped = progress.stop_reason == ControllerStopReason::callback_stop;
        if (progress.stop_reason == ControllerStopReason::stop_token) {
            const auto depth = progress.selected_count - 1;
            result.stop_token = progress.selected_tokens[depth];
            // Internal commit notification only: stop tokens never invoke the
            // external text callback, but their sampler/K/V receipt is durable.
            static_cast<void>(
                callback({*result.stop_token, depth, progress.logical_cached_count, true}));
        }
        if (progress.has_carry && !progress.carry_is_cached)
            result.carry = progress.carry_token;
    } catch (...) {
        if (ticket && ticket->state() == VerifierTransactionState::provisional)
            finalize(delivered_rows);
        throw;
    }
    return result;
}
} // namespace local_gguf_engine::engine
