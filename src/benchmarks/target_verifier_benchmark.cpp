#include "local_gguf_engine/benchmarks/target_verifier_benchmark.hpp"

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::benchmarks {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
    );
}

[[nodiscard]] engine::ValidatedProposalGraph make_linear_graph(
    std::span<const std::uint32_t> tokens,
    std::uint32_t root_position,
    std::uint32_t vocabulary_size,
    std::uint64_t arena_epoch
) {
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    const auto horizon = rows - 1U;
    engine::ProposalGraphStorage graph;
    graph.header = engine::ProposalGraphHeader {
        .kind = engine::ProposalGraphKind::chain,
        .horizon = horizon,
        .materialized_node_count = rows,
        .materialized_edge_count = horizon,
        .root_count = 1U,
        .root_indices = {0U},
        .graph_identity = 0x5441524700000000ULL + rows,
        .device_arena_epoch = arena_epoch,
        .has_source_budget = true,
        .source_budget = {
            .value = horizon,
            .root_convention = engine::SourceRootConvention::excluded,
        },
        .resolved_budget = {.draft_node_cap = horizon, .input_row_cap = rows},
    };
    for (std::uint32_t row = 0U; row < rows; ++row) {
        graph.nodes.token_ids[row] = tokens[row];
        graph.nodes.parent_indices[row] = row == 0U ? engine::proposal_node_none : row - 1U;
        graph.nodes.depths[row] = static_cast<std::uint16_t>(row);
        graph.nodes.absolute_positions[row] = root_position + row;
        graph.nodes.mask_component_roots[row] = 0U;
        graph.nodes.mask_ancestor_counts[row] = static_cast<std::uint16_t>(row + 1U);
        graph.nodes.conditional_confidences[row] = 1.0F;
        graph.nodes.path_histories[row] = {.digest = 0x12345678U + row, .length = 17U + row};
        graph.nodes.proposal_ticket_offsets[row] = row == 0U
            ? engine::proposal_ticket_none : static_cast<std::uint16_t>(row - 1U);
        graph.nodes.feature_provenance[row] = {
            .source = engine::ProposalFeatureSource::target_hidden,
            .source_layer = 0U,
            .arena_epoch = arena_epoch,
        };
        graph.nodes.output_roles[row] = row + 1U == rows
            ? engine::ProposalRowOutputRole::post_full_candidate
            : engine::ProposalRowOutputRole::verifies_proposals;
        if (row == 0U) continue;
        const auto edge = row - 1U;
        graph.materialized_edges.parent_node_indices[edge] = row - 1U;
        graph.materialized_edges.child_node_indices[edge] = row;
        graph.materialized_edges.token_ids[edge] = tokens[row];
        graph.materialized_edges.conditional_confidences[edge] = 1.0F;
        graph.materialized_edges.sibling_ranks[edge] = 0U;
        graph.materialized_edges.proposal_ticket_offsets[edge] =
            static_cast<std::uint16_t>(row - 1U);
    }
    return engine::validate_proposal_graph(
        std::move(graph),
        engine::ProposalGraphValidationContext {
            .vocabulary_size = vocabulary_size,
            .context_length = 8'192U,
            .expected_root_token = tokens[0],
            .expected_root_position = root_position,
            .expected_root_history = {.digest = 0x12345678U, .length = 17U},
            .expected_device_arena_epoch = arena_epoch,
        }
    );
}

[[nodiscard]] engine::ValidatedProposalGraph make_prefix_closed_forest_graph(
    std::span<const std::uint32_t> selected_path_tokens,
    std::uint32_t physical_row_count,
    std::uint32_t root_position,
    std::uint32_t vocabulary_size,
    std::uint64_t arena_epoch
) {
    const auto path_rows = static_cast<std::uint32_t>(selected_path_tokens.size());
    if (path_rows == 0U || physical_row_count < path_rows
        || physical_row_count > engine::VerifierRowMax
        || vocabulary_size <= physical_row_count + path_rows) {
        throw std::invalid_argument("target verifier forest benchmark shape is invalid");
    }
    engine::ProposalGraphStorage graph;
    graph.header = engine::ProposalGraphHeader {
        .kind = engine::ProposalGraphKind::forest,
        .horizon = path_rows - 1U,
        .materialized_node_count = physical_row_count,
        .materialized_edge_count = physical_row_count - 1U,
        .root_count = 1U,
        .root_indices = {0U},
        .graph_identity = 0x464f524500000000ULL
            + (static_cast<std::uint64_t>(physical_row_count) << 16U) + path_rows,
        .device_arena_epoch = arena_epoch,
        .primary_policy = engine::ProposalPrimaryPolicy::deterministic,
        .has_source_budget = true,
        .source_budget = {
            .value = physical_row_count,
            .root_convention = engine::SourceRootConvention::included,
        },
        .resolved_budget = {
            .draft_node_cap = physical_row_count - 1U,
            .input_row_cap = physical_row_count,
        },
    };
    const auto initialize_node = [&](
        std::uint32_t node,
        std::uint32_t token,
        std::uint32_t parent,
        std::uint16_t depth
    ) {
        graph.nodes.token_ids[node] = token;
        graph.nodes.parent_indices[node] = parent;
        graph.nodes.depths[node] = depth;
        graph.nodes.absolute_positions[node] = root_position + depth;
        graph.nodes.mask_component_roots[node] = 0U;
        graph.nodes.mask_ancestor_counts[node] = static_cast<std::uint16_t>(depth + 1U);
        graph.nodes.conditional_confidences[node] = 1.0F;
        graph.nodes.path_histories[node] = {
            .digest = 0x12345678U + node,
            .length = 17U + depth,
        };
        graph.nodes.proposal_ticket_offsets[node] = engine::proposal_ticket_none;
        graph.nodes.feature_provenance[node] = {
            .source = engine::ProposalFeatureSource::target_hidden,
            .source_layer = 0U,
            .arena_epoch = arena_epoch,
        };
        graph.nodes.output_roles[node] = engine::ProposalRowOutputRole::correction_candidate;
    };
    initialize_node(
        0U, selected_path_tokens[0], engine::proposal_node_none, 0U
    );
    for (std::uint32_t node = 1U; node < path_rows; ++node) {
        initialize_node(
            node, selected_path_tokens[node], node - 1U,
            static_cast<std::uint16_t>(node)
        );
    }
    for (std::uint32_t node = path_rows; node < physical_row_count; ++node) {
        auto token = 2U + (vocabulary_size - 2U - physical_row_count + node);
        const auto conflicts_with_root_child = [&]() {
            if (path_rows > 1U && token == selected_path_tokens[1]) return true;
            for (std::uint32_t prior = path_rows; prior < node; ++prior) {
                if (graph.nodes.token_ids[prior] == token) return true;
            }
            return false;
        };
        while (conflicts_with_root_child()) {
            token = token + 1U == vocabulary_size ? 2U : token + 1U;
        }
        initialize_node(node, token, 0U, 1U);
    }

    auto edge = std::uint32_t {0U};
    auto root_rank = std::uint16_t {0U};
    const auto add_edge = [&](
        std::uint32_t parent,
        std::uint32_t child,
        std::uint16_t rank
    ) {
        graph.materialized_edges.parent_node_indices[edge] = parent;
        graph.materialized_edges.child_node_indices[edge] = child;
        graph.materialized_edges.token_ids[edge] = graph.nodes.token_ids[child];
        graph.materialized_edges.conditional_confidences[edge] = 1.0F;
        graph.materialized_edges.sibling_ranks[edge] = rank;
        graph.materialized_edges.proposal_ticket_offsets[edge] =
            engine::proposal_ticket_none;
        graph.nodes.output_roles[parent] = engine::ProposalRowOutputRole::verifies_proposals;
        ++edge;
    };
    if (path_rows > 1U) add_edge(0U, 1U, root_rank++);
    for (std::uint32_t node = path_rows; node < physical_row_count; ++node) {
        add_edge(0U, node, root_rank++);
    }
    for (std::uint32_t node = 2U; node < path_rows; ++node) {
        add_edge(node - 1U, node, 0U);
    }
    if (edge != physical_row_count - 1U) {
        throw std::logic_error("target verifier forest benchmark edge count drifted");
    }
    return engine::validate_proposal_graph(
        std::move(graph),
        engine::ProposalGraphValidationContext {
            .vocabulary_size = vocabulary_size,
            .context_length = 8'192U,
            .expected_root_token = selected_path_tokens[0],
            .expected_root_position = root_position,
            .expected_root_history = {.digest = 0x12345678U, .length = 17U},
            .expected_device_arena_epoch = arena_epoch,
        }
    );
}

[[nodiscard]] engine::ValidatedProposalGraph make_repeated_root_graph(
    std::span<const std::uint32_t> selected_path_tokens,
    std::uint32_t beam_width,
    std::uint32_t root_position,
    std::uint32_t vocabulary_size,
    std::uint64_t arena_epoch
) {
    constexpr auto beam_length = std::uint32_t {4U};
    if (selected_path_tokens.size() != beam_length || beam_width == 0U
        || beam_width > engine::ReDrafterMLXRepeatedVerifierRootMax
        || beam_width * beam_length > engine::ReDrafterMLXPhysicalRowMax
        || vocabulary_size <= beam_width * beam_length + beam_length) {
        throw std::invalid_argument("repeated-root target verifier benchmark shape is invalid");
    }
    engine::ProposalGraphStorage graph;
    graph.header = engine::ProposalGraphHeader {
        .kind = engine::ProposalGraphKind::repeated_root_beam_batch,
        .horizon = beam_length - 1U,
        .materialized_node_count = beam_width * beam_length,
        .materialized_edge_count = beam_width * (beam_length - 1U),
        .root_count = beam_width,
        .repeated_beam_width = beam_width,
        .repeated_beam_length = beam_length,
        .graph_identity = 0x5245445200000000ULL + beam_width,
        .device_arena_epoch = arena_epoch,
        .primary_policy = engine::ProposalPrimaryPolicy::deterministic,
    };
    for (std::uint32_t beam = 0U; beam < beam_width; ++beam) {
        const auto root = beam * beam_length;
        graph.header.root_indices[beam] = root;
        for (std::uint32_t depth = 0U; depth < beam_length; ++depth) {
            const auto node = root + depth;
            auto token = selected_path_tokens[depth];
            if (beam != 0U && depth != 0U) {
                token = vocabulary_size - 1U - beam * beam_length - depth;
            }
            graph.nodes.token_ids[node] = token;
            graph.nodes.parent_indices[node] = depth == 0U
                ? engine::proposal_node_none : node - 1U;
            graph.nodes.depths[node] = static_cast<std::uint16_t>(depth);
            graph.nodes.absolute_positions[node] = root_position + depth;
            graph.nodes.mask_component_roots[node] = root;
            graph.nodes.mask_ancestor_counts[node] = static_cast<std::uint16_t>(depth + 1U);
            graph.nodes.conditional_confidences[node] = 1.0F;
            graph.nodes.path_histories[node] = {
                .digest = 0x12345678U + depth,
                .length = 17U + depth,
            };
            graph.nodes.proposal_ticket_offsets[node] = engine::proposal_ticket_none;
            graph.nodes.feature_provenance[node] = {
                .source = engine::ProposalFeatureSource::recurrent_state,
                .source_layer = 0U,
                .arena_epoch = arena_epoch,
            };
            graph.nodes.output_roles[node] = depth + 1U == beam_length
                ? engine::ProposalRowOutputRole::correction_candidate
                : engine::ProposalRowOutputRole::verifies_proposals;
            if (depth == 0U) continue;
            const auto edge = beam * (beam_length - 1U) + depth - 1U;
            graph.materialized_edges.parent_node_indices[edge] = node - 1U;
            graph.materialized_edges.child_node_indices[edge] = node;
            graph.materialized_edges.token_ids[edge] = token;
            graph.materialized_edges.conditional_confidences[edge] = 1.0F;
            graph.materialized_edges.sibling_ranks[edge] = 0U;
            graph.materialized_edges.proposal_ticket_offsets[edge] =
                engine::proposal_ticket_none;
        }
    }
    return engine::validate_proposal_graph(
        std::move(graph),
        engine::ProposalGraphValidationContext {
            .vocabulary_size = vocabulary_size,
            .context_length = 8'192U,
            .expected_root_token = selected_path_tokens[0],
            .expected_root_position = root_position,
            .expected_root_history = {.digest = 0x12345678U, .length = 17U},
            .expected_device_arena_epoch = arena_epoch,
        }
    );
}

[[nodiscard]] engine::ChainVerifierSchedule linear_schedule(std::uint32_t rows) {
    engine::ChainVerifierSchedule schedule;
    schedule.tile_count = 1U;
    schedule.final_endpoint = engine::ChainTileEndpoint::post_full_candidate;
    schedule.tiles[0] = {
        .validated_edge_count = static_cast<std::uint16_t>(rows - 1U),
        .input_row_count = static_cast<std::uint16_t>(rows),
    };
    return schedule;
}

[[nodiscard]] bool same_candidates(
    const engine::CompactTokenCandidates& left,
    const engine::CompactTokenCandidates& right
) {
    if (left.count != right.count) return false;
    for (std::uint32_t index = 0U; index < left.count; ++index) {
        if (left.values[index].token != right.values[index].token
            || left.values[index].logit != right.values[index].logit) return false;
    }
    return true;
}

[[nodiscard]] std::vector<float> copy_f32_values(
    const tensor::BackendBuffer& buffer
) {
    const auto storage = tensor::copy_tensor_from_backend(buffer);
    if (storage.bytes().size() % sizeof(float) != 0U) {
        throw std::logic_error("target verifier diagnostic expected an F32 buffer");
    }
    std::vector<float> values(storage.bytes().size() / sizeof(float));
    std::memcpy(values.data(), storage.bytes().data(), storage.bytes().size());
    return values;
}

}  // namespace

TargetVerifierBenchmarkResult run_target_verifier_benchmark(
    const std::string& model_path,
    std::uint32_t shape,
    TargetVerifierBenchmarkTopology benchmark_topology
) {
    if (model_path.empty()) {
        throw std::invalid_argument("target verifier benchmark model path is empty");
    }
    const auto conditional = benchmark_topology
        == TargetVerifierBenchmarkTopology::registered_conditional;
    const auto forest = benchmark_topology
        == TargetVerifierBenchmarkTopology::prefix_closed_forest;
    const auto repeated = benchmark_topology
        == TargetVerifierBenchmarkTopology::repeated_root_beam_batch;
    constexpr std::array<std::uint32_t, 11> registered_rows {
        3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U, 17U, 23U, 24U,
    };
    constexpr std::array<std::uint32_t, 7> registered_horizons {
        3U, 4U, 7U, 8U, 15U, 16U, 23U,
    };
    if ((!conditional && !forest && !repeated
            && std::find(registered_rows.begin(), registered_rows.end(), shape)
                == registered_rows.end())
        || (conditional
            && std::find(registered_horizons.begin(), registered_horizons.end(), shape)
                == registered_horizons.end())
        || (forest && (shape == 0U || shape > engine::VerifierRowMax))
        || (repeated && (shape == 0U
            || shape > engine::ReDrafterMLXRepeatedVerifierRootMax))) {
        throw std::invalid_argument("target verifier benchmark row count is not registered");
    }
    constexpr auto repeated_beam_length = std::uint32_t {4U};
    const auto input_row_count = conditional ? shape + 1U
        : repeated ? shape * repeated_beam_length : shape;
    const auto path_row_count = repeated ? repeated_beam_length : forest
        ? std::min<std::uint32_t>(input_row_count, engine::external_speculation_max_horizon + 1U)
        : input_row_count;
    const auto chain_schedule = forest || repeated
        ? engine::ChainVerifierSchedule {}
        : conditional
            ? engine::make_registered_conditional_chain_schedule(shape)
            : linear_schedule(input_row_count);
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    const auto gemma3 = model.architecture() == engine::ModelArchitecture::gemma3;
    auto plan = engine::create_gemma2_metal_model_plan(
        model, gemma3 ? engine::Gemma2OutputHeadPolicy::automatic
                      : engine::Gemma2OutputHeadPolicy::q8_0
    );
    const auto& config = plan.config();
    constexpr auto root_position = std::uint32_t {16U};
    if (config.context_length < root_position + path_row_count
        || config.vocab_size < 40U
        || ((forest || repeated) && config.vocab_size <= input_row_count + path_row_count)) {
        throw std::invalid_argument("target verifier benchmark model is too small");
    }
    const auto scale = std::sqrt(static_cast<float>(config.embedding_length));
    std::array<std::uint32_t, root_position> prefix {};
    for (std::uint32_t index = 0U; index < prefix.size(); ++index) {
        prefix[index] = 2U + index % (config.vocab_size - 2U);
    }
    (void)engine::run_gemma2_metal_resident_prefill(plan, prefix, 0U, scale);

    const auto options = engine::TokenSamplerOptions {
        .temperature = 0.8F,
        .top_k = 40U,
        .top_p = 0.95F,
        .min_p = 0.05F,
        .repetition_penalty = 1.0F,
        .seed = 42U,
        .repetition_window = 64U,
    };
    engine::TokenSampler root_sampler(options);
    root_sampler.observe_tokens(prefix);
    auto current_token = std::uint32_t {5U};
    root_sampler.observe_tokens(std::span<const std::uint32_t>(&current_token, 1U));
    auto path_sampler = root_sampler;
    std::vector<std::uint32_t> tokens(input_row_count);
    std::vector<engine::CompactTokenCandidates> ordinary_candidates(path_row_count);
    std::vector<float> ordinary_final_hidden(
        static_cast<std::size_t>(path_row_count) * config.embedding_length
    );
    std::uint64_t ordinary_gpu_ns = 0U;
    const auto ordinary_begin = Clock::now();
    for (std::uint32_t row = 0U; row < path_row_count; ++row) {
        tokens[row] = current_token;
        const auto decoded = engine::run_gemma2_metal_resident_decode_step(
            plan, engine::create_gemma2_metal_decode_schedule(plan, root_position + row),
            current_token, scale, path_sampler.selection_request()
        );
        if (!decoded.candidates.has_value()) {
            throw std::logic_error("target verifier benchmark ordinary row has no candidates");
        }
        ordinary_candidates[row] = *decoded.candidates;
        auto* hidden_destination = ordinary_final_hidden.data()
            + static_cast<std::size_t>(row) * config.embedding_length;
        if (gemma3) {
            // Ordinary Gemma 3 uses row zero of the shared block workspace.
            // Read only that row; copying its full 128-row capacity would
            // inflate the ordinary timing with unrelated diagnostic traffic.
            const auto& workspace = plan.target_execution_workspace();
            if (!workspace) {
                throw std::logic_error("Gemma 3 ordinary benchmark has no execution workspace");
            }
            const auto hidden = tensor::copy_backend_buffer_bytes(
                workspace->scratch().normalized_hidden, 0U,
                static_cast<std::size_t>(config.embedding_length) * sizeof(float)
            );
            std::memcpy(hidden_destination, hidden.data(), hidden.size());
        } else {
            const auto hidden = copy_f32_values(plan.decode_scratch().normalized_hidden);
            std::copy_n(hidden.begin(), config.embedding_length, hidden_destination);
        }
        ordinary_gpu_ns += decoded.execution.gpu_duration_ns;
        if (row + 1U < path_row_count) {
            current_token = path_sampler.sample(
                std::span<const engine::TokenLogit>(decoded.candidates->values)
                    .first(decoded.candidates->count)
            );
            path_sampler.observe_tokens(
                std::span<const std::uint32_t>(&current_token, 1U)
            );
        }
    }
    const auto ordinary_end = Clock::now();
    // Re-establish the logical prefix. Stale later ring rows are intentionally
    // ignored by the verifier's absolute causal schedule.
    (void)engine::run_gemma2_metal_resident_prefill(plan, prefix, 0U, scale);

    constexpr auto arena_epoch = std::uint64_t {0x54415247U};
    auto graph = std::make_unique<engine::ValidatedProposalGraph>(
        repeated
            ? make_repeated_root_graph(
                std::span<const std::uint32_t>(tokens).first(path_row_count),
                shape, root_position, config.vocab_size, arena_epoch
            )
        : forest
            ? make_prefix_closed_forest_graph(
                std::span<const std::uint32_t>(tokens).first(path_row_count),
                input_row_count, root_position, config.vocab_size, arena_epoch
            )
            : make_linear_graph(tokens, root_position, config.vocab_size, arena_epoch)
    );
    auto request = std::make_unique<engine::VerifierEncodeRequest>();
    request->topology = engine::make_verifier_topology_descriptor(
        *graph, chain_schedule, config.block_count,
        config.context_length, config.sliding_window, config.attention_global_layer_period
    );
    request->selection_request_count = input_row_count;
    request->sampler_options = options;
    for (std::uint32_t row = 0U; row < input_row_count; ++row) {
        if (repeated) {
            const auto beam_root = (row / repeated_beam_length) * repeated_beam_length;
            const auto depth = row % repeated_beam_length;
            request->selection_requests[row] = root_sampler.device_selection_request(
                std::span<const std::uint32_t>(graph->storage().nodes.token_ids)
                    .subspan(beam_root + 1U, depth)
            );
        } else if (!forest || row < path_row_count) {
            request->selection_requests[row] = root_sampler.device_selection_request(
                std::span<const std::uint32_t>(tokens).subspan(1U, row)
            );
        } else {
            const auto token = graph->storage().nodes.token_ids[row];
            request->selection_requests[row] = root_sampler.device_selection_request(
                std::span<const std::uint32_t>(&token, 1U)
            );
        }
    }
    auto verifier_rows = input_row_count;
    if (conditional) {
        verifier_rows = 0U;
        for (std::uint32_t tile = 0U; tile < chain_schedule.tile_count; ++tile) {
            verifier_rows = std::max<std::uint32_t>(
                verifier_rows, chain_schedule.tiles[tile].input_row_count
            );
        }
    }
    const auto allocation_begin = Clock::now();
    auto verifier = engine::create_gemma2_metal_target_verifier_plan(plan, verifier_rows);
    const auto allocation_end = Clock::now();

    auto run_diagnostic_once = [&]() {
        if (conditional) {
            return std::unique_ptr<engine::TargetVerifierCompletionResult> {};
        }
        auto local_request = std::make_unique<engine::VerifierEncodeRequest>(*request);
        auto ticket = engine::encode_gemma2_metal_target_verifier(
            plan, verifier,
            engine::validate_verifier_encode_request(
                *local_request, config.vocab_size, config.block_count,
                config.context_length, config.sliding_window, config.attention_global_layer_period
            ),
            scale
        );
        engine::submit_gemma2_metal_target_verifier(ticket);
        auto completion = std::make_unique<engine::TargetVerifierCompletionResult>(
            engine::complete_gemma2_metal_target_verifier(ticket)
        );
        engine::discard_gemma2_metal_target_verifier(ticket);
        return completion;
    };
    std::unique_ptr<engine::TargetVerifierCompletionResult> completion;
    if (!conditional) {
        // Warm the complete width-specific route before recording its ceiling.
        static_cast<void>(run_diagnostic_once());
        completion = run_diagnostic_once();
    }

    bool candidates_match = !conditional;
    bool candidate_tokens_match = !conditional;
    auto maximum_candidate_absolute_error = 0.0F;
    auto maximum_candidate_relative_error = 0.0F;
    auto candidate_token_mismatch_count = std::uint32_t {0};
    auto first_candidate_token_mismatch_row = std::uint32_t {0};
    auto first_candidate_token_mismatch_rank = std::uint32_t {0};
    auto first_candidate_token_expected = std::uint32_t {0};
    auto first_candidate_token_actual = std::uint32_t {0};
    auto maximum_final_hidden_absolute_error = 0.0F;
    auto maximum_final_hidden_error_row = std::uint32_t {0};
    auto first_row_final_hidden_absolute_error = 0.0F;
    auto mismatch_row_final_hidden_absolute_error = 0.0F;
    for (std::uint32_t row = 0U; completion && row < path_row_count; ++row) {
        candidates_match = candidates_match
            && same_candidates(completion->candidates[row], ordinary_candidates[row]);
        candidate_tokens_match = candidate_tokens_match
            && completion->candidates[row].count == ordinary_candidates[row].count;
        const auto compared = std::min(
            completion->candidates[row].count, ordinary_candidates[row].count
        );
        for (std::uint32_t index = 0U; index < compared; ++index) {
            const auto& actual = completion->candidates[row].values[index];
            const auto& expected = ordinary_candidates[row].values[index];
            if (actual.token != expected.token) {
                if (candidate_token_mismatch_count == 0U) {
                    first_candidate_token_mismatch_row = row;
                    first_candidate_token_mismatch_rank = index;
                    first_candidate_token_expected = expected.token;
                    first_candidate_token_actual = actual.token;
                }
                ++candidate_token_mismatch_count;
            }
            candidate_tokens_match = candidate_tokens_match
                && actual.token == expected.token;
            const auto absolute = std::abs(actual.logit - expected.logit);
            const auto relative_scale = std::max(std::abs(expected.logit), 1.0e-12F);
            maximum_candidate_absolute_error = std::max(
                maximum_candidate_absolute_error, absolute
            );
            maximum_candidate_relative_error = std::max(
                maximum_candidate_relative_error, absolute / relative_scale
            );
        }
    }
    if (completion) {
        const auto verifier_hidden = copy_f32_values(verifier.scratch().normalized_hidden);
        for (std::uint32_t row = 0U; row < path_row_count; ++row) {
            for (std::uint32_t column = 0U; column < config.embedding_length; ++column) {
                const auto index = static_cast<std::size_t>(row) * config.embedding_length
                    + column;
                const auto error = std::abs(
                    verifier_hidden[index] - ordinary_final_hidden[index]
                );
                if (error > maximum_final_hidden_absolute_error) {
                    maximum_final_hidden_absolute_error = error;
                    maximum_final_hidden_error_row = row;
                }
                if (row == 0U) {
                    first_row_final_hidden_absolute_error = std::max(
                        first_row_final_hidden_absolute_error, error
                    );
                }
                if (row == first_candidate_token_mismatch_row) {
                    mismatch_row_final_hidden_absolute_error = std::max(
                        mismatch_row_final_hidden_absolute_error, error
                    );
                }
            }
        }
    }
    engine::SpeculativeControllerScenario scenario;
    scenario.graph = graph.get();
    scenario.chain_schedule = chain_schedule;
    scenario.proposal_source = engine::SpeculativeProposalSource::learned_external;
    scenario.draft_schedule = conditional
        ? engine::SpeculativeDraftSchedule::conditional_tiles
        : engine::SpeculativeDraftSchedule::eager;
    scenario.route_mode = engine::SpeculativeRouteMode::learned_only;
    scenario.verifier_topology = repeated
        ? engine::TargetVerifierTopology::repeated_root_beam_batch
        : forest
        ? engine::TargetVerifierTopology::prefix_closed_forest
        : conditional
        ? engine::TargetVerifierTopology::conditional_tile_cascade
        : engine::TargetVerifierTopology::single_invocation_post_full;
    scenario.first_call_topology = conditional
        ? engine::FirstCallCommandTopology::host_gated
        : engine::FirstCallCommandTopology::fused_eager;
    scenario.origin_token = tokens[0];
    scenario.origin_checkpoint = root_sampler.checkpoint();
    scenario.remaining_selected_token_budget = path_row_count + 1U;
    scenario.initial_logical_cached_count = root_position;
    scenario.available_context_rows = config.context_length - root_position;
    scenario.callbacks.fill(engine::ControllerCallbackDecision::continue_generation);
    scenario.tile_gates.fill(engine::ControllerTileGateDecision::accept);
    auto episode_request = std::make_unique<engine::VerifierEncodeRequest>(*request);
    auto validated_episode = std::make_unique<engine::ValidatedVerifierEncodeRequest>(
        engine::validate_verifier_encode_request(
            *episode_request, config.vocab_size, config.block_count,
            config.context_length, config.sliding_window, config.attention_global_layer_period
        )
    );
    auto episode_workspace = std::make_unique<engine::TargetVerifierEpisodeWorkspace>();
    if (conditional) {
        auto warm_sampler = root_sampler;
        static_cast<void>(engine::run_gemma2_metal_target_verifier_chain_episode(
            plan, verifier, *validated_episode, scenario, warm_sampler,
            *episode_workspace, scale
        ));
        (void)engine::run_gemma2_metal_resident_prefill(plan, prefix, 0U, scale);
    }
    auto episode_sampler = root_sampler;
    const auto episode_begin = Clock::now();
    const auto episode = forest || repeated
        ? engine::run_gemma2_metal_target_verifier_forest_episode(
            plan, verifier, *validated_episode, scenario, episode_sampler,
            *episode_workspace, scale
        )
        : engine::run_gemma2_metal_target_verifier_chain_episode(
            plan, verifier, *validated_episode, scenario, episode_sampler,
            *episode_workspace, scale
        );
    const auto episode_end = Clock::now();
    const auto commit_wall_ns = episode.commit_wall_ns;
    const auto commit_gpu_ns = episode.commit_gpu_ns;
    const auto episode_wall_ns = elapsed_ns(episode_begin, episode_end);
    const auto ordinary_wall_ns = elapsed_ns(ordinary_begin, ordinary_end);
    return TargetVerifierBenchmarkResult {
        .model_path = model_path,
        .input_row_count = input_row_count,
        .proposal_horizon = path_row_count - 1U,
        .allocation_wall_ns = elapsed_ns(allocation_begin, allocation_end),
        .ordinary_rows_wall_ns = ordinary_wall_ns,
        .ordinary_rows_gpu_ns = ordinary_gpu_ns,
        .verifier_wall_ns = episode.target_wall_ns,
        .verifier_gpu_ns = episode.target_gpu_ns,
        .controller_wall_ns = episode.controller_wall_ns,
        .commit_wall_ns = commit_wall_ns,
        .commit_gpu_ns = commit_gpu_ns,
        .complete_episode_wall_ns = episode_wall_ns,
        .verifier_allocated_bytes = verifier.allocated_byte_count(),
        .compact_readback_bytes = episode.target_execution.compact_readback_bytes,
        .command_buffer_count = episode.target_execution.command_buffer_count
            + episode.endpoint_execution.command_buffer_count
            + episode.ordinary_fallback_execution.command_buffer_count,
        .executed_tile_count = episode.executed_tile_count,
        .conditional_topology = conditional,
        .forest_topology = forest || repeated,
        .candidate_identity_measured = !conditional,
        .exact_candidate_match = candidates_match,
        .candidate_token_order_match = candidate_tokens_match,
        .maximum_candidate_absolute_error = maximum_candidate_absolute_error,
        .maximum_candidate_relative_error = maximum_candidate_relative_error,
        .candidate_token_mismatch_count = candidate_token_mismatch_count,
        .first_candidate_token_mismatch_row = first_candidate_token_mismatch_row,
        .first_candidate_token_mismatch_rank = first_candidate_token_mismatch_rank,
        .first_candidate_token_expected = first_candidate_token_expected,
        .first_candidate_token_actual = first_candidate_token_actual,
        .maximum_final_hidden_absolute_error = maximum_final_hidden_absolute_error,
        .maximum_final_hidden_error_row = maximum_final_hidden_error_row,
        .first_row_final_hidden_absolute_error = first_row_final_hidden_absolute_error,
        .mismatch_row_final_hidden_absolute_error = mismatch_row_final_hidden_absolute_error,
        .controller_full_match = episode.controller.stop_reason
                == (forest || repeated ? engine::ControllerStopReason::materialized_leaf
                           : engine::ControllerStopReason::full_match)
            && episode.controller.retained_row_count == path_row_count,
        .authoritative_token_match = episode.controller.stop_reason
                == (forest || repeated ? engine::ControllerStopReason::materialized_leaf
                           : engine::ControllerStopReason::full_match)
            && episode.controller.selected_count == path_row_count
            && std::equal(
                tokens.begin(), tokens.begin() + path_row_count,
                episode.controller.selected_tokens.begin()
            ),
        .perfect_progress_ceiling = static_cast<double>(ordinary_wall_ns)
            / static_cast<double>(episode_wall_ns),
    };
}

}  // namespace local_gguf_engine::benchmarks
