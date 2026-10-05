#include "local_gguf_engine/engine/target_verifier.hpp"

#include "feature_bank_runtime.hpp"
#include "draft_cache_runtime.hpp"
#include "gemma2_verification_state_access.hpp"
#include "target_verifier_test_hooks.hpp"
#include "gemma_target_runtime.hpp"
#include "../tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

namespace metal = tensor::backends::metal;

namespace {

class UnsubmittedFeatureCaptureGuard final {
public:
    explicit UnsubmittedFeatureCaptureGuard(FeatureSinkBinding binding) noexcept
        : binding_(binding), armed_(binding.sinks != nullptr) {}
    UnsubmittedFeatureCaptureGuard(const UnsubmittedFeatureCaptureGuard&) = delete;
    UnsubmittedFeatureCaptureGuard& operator=(const UnsubmittedFeatureCaptureGuard&) = delete;
    ~UnsubmittedFeatureCaptureGuard() {
        if (armed_) FeatureSinkRuntimeAccess::abort_unsubmitted(binding_);
    }
    void release() noexcept { armed_ = false; }

private:
    FeatureSinkBinding binding_;
    bool armed_ = false;
};

struct TargetVerifierTicketMetadata {
    std::array<std::uint32_t, VerifierRowMax> parent_indices {};
    std::array<std::uint32_t, VerifierRowMax> absolute_positions {};
    std::array<std::uint16_t, VerifierRowMax> depths {};
    std::uint64_t graph_identity = 0;
    std::uint64_t attention_token_count = 0;
    std::uint32_t cache_position = 0;
    std::uint32_t input_row_count = 0;
    std::uint32_t proposal_edge_count = 0;
    std::uint32_t attention_dispatch_count = 0;
    std::uint32_t attention_partition_count = 0;
    std::uint16_t root_source_depth = 0;
};

static_assert(sizeof(TargetVerifierTicketMetadata) <= 16U * 1024U);

[[nodiscard]] TargetVerifierTicketMetadata make_ticket_metadata(
    const VerifierTopologyDescriptor& topology
) {
    TargetVerifierTicketMetadata metadata {
        .graph_identity = topology.graph_identity,
        .cache_position = topology.absolute_positions[topology.root_indices[0]],
        .input_row_count = topology.input_row_count,
        .proposal_edge_count = topology.proposal_edge_count,
        .root_source_depth = topology.root_source_depth,
    };
    std::copy_n(
        topology.parent_indices.begin(), topology.input_row_count,
        metadata.parent_indices.begin()
    );
    std::copy_n(
        topology.absolute_positions.begin(), topology.input_row_count,
        metadata.absolute_positions.begin()
    );
    std::copy_n(
        topology.depths.begin(), topology.input_row_count,
        metadata.depths.begin()
    );
    for (std::uint32_t layer = 0; layer < topology.attention.layer_count; ++layer) {
        for (std::uint32_t row = 0; row < topology.input_row_count; ++row) {
            const auto tokens = topology.attention.per_layer_attention_counts[
                layer * VerifierRowMax + row
            ];
            ++metadata.attention_dispatch_count;
            metadata.attention_partition_count +=
                metal::gemma2_decode_attention_partition_count(tokens);
            if (tokens > std::numeric_limits<std::uint64_t>::max()
                    - metadata.attention_token_count) {
                throw std::overflow_error("target verifier attention accounting overflow");
            }
            metadata.attention_token_count += tokens;
        }
    }
    return metadata;
}

[[nodiscard]] TargetVerifierExecutionStats make_execution_stats(
    const metal::MetalCommandStats& stats,
    const TargetVerifierTicketMetadata& metadata,
    std::uint32_t selector_dispatch_count,
    std::uint32_t compact_readback_bytes
) {
    TargetVerifierExecutionStats output {
        .command_buffer_count = stats.command_buffer_count,
        .commit_count = stats.commit_count,
        .synchronous_wait_count = stats.synchronous_wait_count,
        .compute_encoder_count = stats.compute_encoder_count,
        .compute_dispatch_count = stats.compute_dispatch_count,
        .blit_encoder_count = stats.blit_encoder_count,
        .memory_barrier_count = stats.memory_barrier_count,
        .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
        .selector_dispatch_count = selector_dispatch_count,
        .compact_readback_bytes = compact_readback_bytes,
        .input_row_count = metadata.input_row_count,
        .proposal_edge_count = metadata.proposal_edge_count,
        .attention_dispatch_count = metadata.attention_dispatch_count,
        .attention_partition_count = metadata.attention_partition_count,
        .attention_token_count = metadata.attention_token_count,
        .gpu_duration_ns = stats.gpu_duration_ns,
        .q4_decode_projection_count = stats.q4_decode_projection_count,
        .q8_decode_projection_count = stats.q8_decode_projection_count,
        .q8_width_two_projection_count = stats.q8_width_two_projection_count,
        .q4_small_batch_projection_count = stats.q4_small_batch_projection_count,
        .q4_width_two_projection_count = stats.q4_width_two_projection_count,
        .q4_exact_16_projection_count = stats.q4_exact_16_projection_count,
        .q4_generic_tiled_projection_count = stats.q4_generic_tiled_projection_count,
        .q6_decode_projection_count = stats.q6_decode_projection_count,
        .q6_small_batch_projection_count = stats.q6_small_batch_projection_count,
        .q6_width_two_projection_count = stats.q6_width_two_projection_count,
        .q6_exact_16_projection_count = stats.q6_exact_16_projection_count,
        .q6_generic_tiled_projection_count = stats.q6_generic_tiled_projection_count,
        .f16_projection_count = stats.f16_projection_count,
    };
    return output;
}

[[nodiscard]] VerifierDeviceSelectionRow pack_selection_request(
    const DeviceSelectionRequest& request
) {
    return VerifierDeviceSelectionRow {
        .history_token_ids = request.history.token_ids,
        .suffix_token_ids = request.suffix_token_ids,
        .history_head = request.history.head,
        .history_length = request.history.length,
        .history_window_limit = request.history.window_limit,
        .suffix_count = request.suffix_count,
    };
}

void write_selection_request(
    metal::MetalCommandContext& command,
    const tensor::BackendBuffer& destination,
    std::uint32_t row,
    const DeviceSelectionRequest& request
) {
    const auto packed = pack_selection_request(request);
    static_assert(sizeof(packed) % sizeof(std::uint32_t) == 0U);
    const auto* words = reinterpret_cast<const std::uint32_t*>(&packed);
    command.write_u32s(
        destination,
        row * static_cast<std::uint32_t>(sizeof(packed) / sizeof(std::uint32_t)),
        std::span<const std::uint32_t>(words, sizeof(packed) / sizeof(std::uint32_t))
    );
}

void stage_topology(
    metal::MetalCommandContext& command,
    const Gemma2MetalTargetVerifierPlan& verifier,
    const VerifierEncodeRequest& request
) {
    const auto& topology = request.topology;
    const auto& scratch = verifier.scratch();
    command.write_u32s(scratch.expected_token_id_staging, 0U,
        std::span<const std::uint32_t>(topology.token_ids).first(topology.input_row_count));
    command.write_u32s(scratch.absolute_position_staging, 0U,
        std::span<const std::uint32_t>(topology.absolute_positions).first(topology.input_row_count));
    command.write_u32s(scratch.ancestor_index_staging, 0U,
        std::span<const std::uint32_t>(topology.attention.ancestor_node_indices)
            .first(topology.attention.ancestor_entry_count));

    for (std::uint32_t layer = 0; layer < topology.attention.layer_count; ++layer) {
        for (std::uint32_t row = 0; row < topology.input_row_count; ++row) {
            const auto& source = topology.attention.layer_uses_sliding_window[layer]
                ? topology.attention.sliding_rows[row]
                : topology.attention.global_rows[row];
            const std::array<std::uint32_t, 4> packed {
                source.base_first_position,
                source.base_token_count,
                source.ancestor_offset,
                source.ancestor_count,
            };
            const auto output_row = layer * verifier.row_capacity() + row;
            command.write_u32s(scratch.attention_schedule_staging, output_row * 4U, packed);
        }
    }
    if (request.input_residency == VerifierInputResidency::host_staged) {
        command.write_u32s(scratch.token_id_staging, 0U,
            std::span<const std::uint32_t>(topology.token_ids).first(topology.input_row_count));
        for (std::uint32_t row = 0; row < topology.input_row_count; ++row) {
            write_selection_request(command, scratch.selection.raw_request_staging, row,
                request.selection_requests[row]);
        }
    }
    command.write_u32(scratch.input_status, 0U);
}

[[nodiscard]] std::uint32_t first_root_position(const VerifierTopologyDescriptor& topology) {
    return topology.absolute_positions[topology.root_indices[0]];
}

void validate_plan_binding(
    const Gemma2MetalModelPlan& target,
    const Gemma2MetalTargetVerifierPlan& verifier,
    const VerifierEncodeRequest& request
) {
    if (target.has_pending_verification()) {
        throw std::logic_error("target verifier cannot encode while another transaction is pending");
    }
    if (verifier.target_weight_identity() != target.allocation_identities()[0]
        || verifier.layer_count() != target.config().block_count
        || verifier.vocabulary_size() != target.config().vocab_size) {
        throw std::invalid_argument("target verifier arena belongs to a different target plan");
    }
    for (std::uint32_t layer = 0; layer < target.config().block_count; ++layer) {
        if (request.topology.attention.layer_uses_sliding_window[layer]
            != gemma_layer_uses_sliding_window(target.config(), layer)) {
            throw std::invalid_argument("target verifier attention pattern does not match bound model");
        }
    }
    if (target.config().architecture == ModelArchitecture::gemma3) {
        for (std::uint32_t row = 0; row < request.topology.input_row_count; ++row) {
            if (static_cast<std::uint32_t>(request.topology.depths[row])
                    - request.topology.root_source_depth + 1U > target.config().sliding_window) {
                throw std::invalid_argument("target verifier causal path exceeds the local cache window");
            }
        }
    }
    if (request.topology.input_row_count > verifier.row_capacity()) {
        throw std::length_error("target verifier request exceeds its shape-owned arena");
    }
    if (target.config().attention_head_count != 2U * target.config().attention_head_count_kv
        || target.config().attention_key_length != 256U
        || target.config().attention_value_length != target.config().attention_key_length) {
        throw std::invalid_argument("target verifier plan has incompatible grouped-attention geometry");
    }
}

[[nodiscard]] TargetVerifierExecutionStats execution_from_commit(
    const metal::MetalCommandStats& stats,
    const TargetVerifierTicketMetadata& metadata
) {
    return TargetVerifierExecutionStats {
        .command_buffer_count = stats.command_buffer_count,
        .commit_count = stats.commit_count,
        .synchronous_wait_count = stats.synchronous_wait_count,
        .compute_encoder_count = stats.compute_encoder_count,
        .compute_dispatch_count = stats.compute_dispatch_count,
        .blit_encoder_count = stats.blit_encoder_count,
        .memory_barrier_count = stats.memory_barrier_count,
        .transient_buffer_allocation_count = stats.transient_buffer_allocation_count,
        .input_row_count = metadata.input_row_count,
        .proposal_edge_count = metadata.proposal_edge_count,
        .gpu_duration_ns = stats.gpu_duration_ns,
    };
}

}  // namespace

struct TargetVerifierRuntimeState final {
    Gemma2MetalModelPlan* target = nullptr;
    Gemma2MetalTargetVerifierPlan* verifier = nullptr;
    TargetVerifierTicketMetadata metadata;
    explicit TargetVerifierRuntimeState(bool precise)
        : command("Gemma target verifier", true, precise),
          commit_command("Gemma verifier K/V commit or restore", true, precise) {}
    metal::MetalCommandContext command;
    metal::MetalCommandContext commit_command;
    std::uint64_t epoch = 0;
    VerifierTransactionState state = VerifierTransactionState::idle;
    std::uint32_t selector_dispatch_count = 0;
    std::array<std::uint32_t, external_speculation_max_horizon + 1U> provisional_path {};
    std::uint32_t provisional_count = 0;
    FeatureSinkBinding feature_sink;
    DraftCacheBinding draft_cache;

};
std::shared_ptr<TargetVerifierRuntimeState> make_target_verifier_runtime_state(bool precise_math) {
    return std::make_shared<TargetVerifierRuntimeState>(precise_math);
}
struct TargetVerifierRuntimeAccess {
    static std::shared_ptr<TargetVerifierRuntimeState> get(Gemma2MetalTargetVerifierPlan& plan) {
        return plan.runtime_;
    }
};

metal::MetalCommandContext& target_workspace_command(Gemma2MetalTargetVerifierPlan& workspace) {
    auto runtime = TargetVerifierRuntimeAccess::get(workspace);
    if (runtime->state != VerifierTransactionState::idle)
        throw std::logic_error("ordinary execution workspace has an outstanding ticket");
    runtime->command.restart();
    return runtime->command;
}

CompactTokenCandidates target_workspace_completed_candidates(
    Gemma2MetalTargetVerifierPlan& workspace, const Gemma2MetalSelectionWorkspace& selection) {
    return TargetVerifierRuntimeAccess::get(workspace)->command.read_token_candidates(selection);
}

TargetVerificationTicket::TargetVerificationTicket(std::shared_ptr<TargetVerifierRuntimeState> impl)
    : impl_(std::move(impl)), lease_epoch_(impl_->epoch) {}

TargetVerificationTicket::TargetVerificationTicket(TargetVerificationTicket&&) noexcept = default;

TargetVerificationTicket::~TargetVerificationTicket() {
    if (!impl_ || impl_->epoch != lease_epoch_ || impl_->state == VerifierTransactionState::idle) return;
    if (impl_->state == VerifierTransactionState::encoded
        || impl_->state == VerifierTransactionState::completed) {
        if (impl_->state == VerifierTransactionState::encoded) {
            FeatureSinkRuntimeAccess::abort_unsubmitted(impl_->feature_sink);
        } else {
            FeatureSinkRuntimeAccess::discard_speculative(impl_->feature_sink);
        }
        Gemma2MetalVerificationStateAccess::clear(*impl_->target);
        impl_->state = VerifierTransactionState::idle;
        return;
    }
    if (impl_->state == VerifierTransactionState::provisional) {
        try {
            static_cast<void>(finalize_gemma2_metal_target_verifier(*this, 0U));
        } catch (...) {
            FeatureSinkRuntimeAccess::poison(impl_->feature_sink);
            DraftCacheRuntimeAccess::poison(impl_->draft_cache);
            Gemma2MetalVerificationStateAccess::poison(*impl_->target);
            impl_->state = VerifierTransactionState::poisoned;
        }
        return;
    }
    FeatureSinkRuntimeAccess::poison(impl_->feature_sink);
    DraftCacheRuntimeAccess::poison(impl_->draft_cache);
    Gemma2MetalVerificationStateAccess::poison(*impl_->target);
    impl_->state = VerifierTransactionState::poisoned;
}

VerifierTransactionState TargetVerificationTicket::state() const noexcept {
    return impl_ && impl_->epoch == lease_epoch_ ? impl_->state : VerifierTransactionState::idle;
}

std::uint32_t TargetVerificationTicket::input_row_count() const noexcept {
    return impl_ && impl_->epoch == lease_epoch_ ? impl_->metadata.input_row_count : 0U;
}

std::uint64_t TargetVerificationTicket::graph_identity() const noexcept {
    return impl_ && impl_->epoch == lease_epoch_ ? impl_->metadata.graph_identity : 0U;
}

TargetVerificationTicket encode_gemma2_metal_target_verifier(
    Gemma2MetalModelPlan& target,
    Gemma2MetalTargetVerifierPlan& verifier,
    const ValidatedVerifierEncodeRequest& request,
    float embedding_scale,
    FeatureSinkBinding feature_sink,
    DraftCacheBinding draft_cache
) {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::encode);
#endif
    if (!std::isfinite(embedding_scale) || embedding_scale <= 0.0F) {
        throw std::invalid_argument("target verifier embedding scale must be finite and positive");
    }
    if (!request.validated_for(target.config())) {
        throw std::invalid_argument("target verifier validation shape does not match bound model");
    }
    const auto& encoded = request.request();
    validate_plan_binding(target, verifier, encoded);
    auto runtime = TargetVerifierRuntimeAccess::get(verifier);
    if (runtime->state != VerifierTransactionState::idle || runtime->epoch == UINT64_MAX) {
        throw std::logic_error("target verifier arena has an active, poisoned, or exhausted lease");
    }
    auto* command = &runtime->command;
    command->restart();
    UnsubmittedFeatureCaptureGuard feature_guard(feature_sink);
    stage_topology(*command, verifier, encoded);
    const auto& topology = encoded.topology;
    const auto& scratch = verifier.scratch();
    DraftCacheRuntimeAccess::validate_binding(
        draft_cache, topology.device_arena_epoch, topology.input_row_count
    );
    if (feature_sink_bound(feature_sink)) {
        if (feature_sink.destination != FeatureCaptureDestination::speculative) {
            throw std::invalid_argument("target verifier feature sink must be speculative");
        }
        FeatureSinkRuntimeAccess::begin_capture(
            feature_sink, *command,
            std::span<const std::uint32_t>(topology.absolute_positions)
                .first(topology.input_row_count),
            std::span<const std::uint32_t>(topology.token_ids)
                .first(topology.input_row_count),
            topology.input_row_count
        );
    }
    const tensor::BackendBuffer* token_source = &scratch.token_id_staging;
    const tensor::BackendBuffer* request_source = &scratch.selection.raw_request_staging;
    auto token_offset = std::uint32_t {0};
    auto request_byte_offset = std::uint64_t {0};
    if (encoded.input_residency == VerifierInputResidency::device_resident) {
        const auto& binding = *encoded.device_inputs;
        token_source = &binding.token_ids;
        request_source = &binding.raw_selection_requests;
        token_offset = binding.first_token;
        request_byte_offset = static_cast<std::uint64_t>(binding.first_request)
            * sizeof(VerifierDeviceSelectionRow);
    }
    command->encode_verifier_input_validation(
        *token_source,
        static_cast<std::uint64_t>(token_offset) * sizeof(std::uint32_t),
        scratch.expected_token_id_staging,
        scratch.input_status,
        topology.input_row_count,
        target.config().vocab_size
    );
    command->encode_verifier_selection_requests(
        *request_source, request_byte_offset,
        scratch.selection.resolved_request_staging,
        scratch.selection.history_token_staging,
        scratch.input_status,
        topology.input_row_count,
        encoded.sampler_options,
        target.config().vocab_size
    );
    command->encode_memory_barrier();
    encode_gemma_target_blocks(*command, target, scratch, verifier.row_capacity(),
        topology.input_row_count, *token_source, token_offset, embedding_scale, feature_sink);
    const auto& config = target.config();
    auto selector_dispatches = std::uint32_t {0};
    for (std::uint32_t first = 0; first < topology.input_row_count;
         first += static_cast<std::uint32_t>(target_selection_batch_rows)) {
        const auto rows = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(target_selection_batch_rows),
            topology.input_row_count - first
        );
        command->encode_target_head_rows(
            scratch.normalized_hidden, first, target.weights().output_weight,
            scratch.selection.logits_tile, rows
        );
        command->encode_memory_barrier();
        selector_dispatches += command->encode_target_token_candidates(
            scratch.selection.logits_tile, scratch.selection, first, rows,
            config.final_logit_softcap,
            encoded.sampler_options.temperature == 0.0F
                && encoded.sampler_options.repetition_penalty == 1.0F
        );
        command->encode_memory_barrier();
    }

    auto metadata = make_ticket_metadata(topology);
    runtime->target = &target;
    runtime->verifier = &verifier;
    runtime->metadata = metadata;
    runtime->selector_dispatch_count = selector_dispatches;
    runtime->feature_sink = feature_sink;
    runtime->draft_cache = draft_cache;
    runtime->provisional_count = 0;
    runtime->state = VerifierTransactionState::encoded;
    ++runtime->epoch;
    Gemma2MetalVerificationStateAccess::set_pending(
        target, first_root_position(topology), topology.input_row_count,
        VerifierTransactionState::encoded
    );
    feature_guard.release();
    return TargetVerificationTicket(std::move(runtime));
}

void submit_gemma2_metal_target_verifier(TargetVerificationTicket& ticket) {
    if (!ticket.impl_ || ticket.state() != VerifierTransactionState::encoded) {
        throw std::logic_error("target verifier submit requires an encoded ticket");
    }
    try {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
        inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::commit);
#endif
        static_cast<void>(ticket.impl_->command.commit());
        Gemma2MetalVerificationStateAccess::mark_submitted(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::submitted;
    } catch (...) {
        FeatureSinkRuntimeAccess::abort_unsubmitted(ticket.impl_->feature_sink);
        Gemma2MetalVerificationStateAccess::clear(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::idle;
        throw;
    }
}

TargetVerifierCompletionResult complete_gemma2_metal_target_verifier(
    TargetVerificationTicket& ticket
) {
    if (!ticket.impl_ || ticket.state() != VerifierTransactionState::submitted) {
        throw std::logic_error("target verifier completion requires a submitted ticket");
    }
    metal::MetalCommandStats stats;
    try {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
        inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::wait);
#endif
        stats = ticket.impl_->command.wait_until_completed();
        FeatureSinkRuntimeAccess::complete_capture(
            ticket.impl_->feature_sink, ticket.impl_->command,
            std::span<const std::uint32_t>(ticket.impl_->metadata.absolute_positions)
                .first(ticket.impl_->metadata.input_row_count),
            ticket.impl_->metadata.input_row_count
        );
        Gemma2MetalVerificationStateAccess::mark_completed(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::completed;
    } catch (...) {
        FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
        DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
        Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::poisoned;
        throw;
    }

    const auto& metadata = ticket.impl_->metadata;
    TargetVerifierCompletionResult result;
    try {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
        inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::readback);
#endif
        const auto status = ticket.impl_->command.read_verifier_status(
            ticket.impl_->verifier->scratch().input_status
        );
        if ((status & metal::verifier_kv_nonfinite_status) != 0U) {
            throw std::overflow_error("Gemma 3 F16 KV storage produced nonfinite values");
        }
        if (status != 0U) {
            throw std::invalid_argument("target verifier rejected malformed device input");
        }
        result.candidate_row_count = metadata.input_row_count;
        result.cache_position = metadata.cache_position;
        const auto candidates = ticket.impl_->verifier->completion_candidates()
            .first(result.candidate_row_count);
        ticket.impl_->command.read_target_token_candidates(
            ticket.impl_->verifier->scratch().selection,
            0U,
            candidates
        );
        result.candidates = candidates;
    } catch (...) {
        FeatureSinkRuntimeAccess::discard_speculative(ticket.impl_->feature_sink);
        Gemma2MetalVerificationStateAccess::clear(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::idle;
        throw;
    }
    auto bytes = std::uint32_t {0};
    try {
        for (std::uint32_t row = 0; row < result.candidate_row_count; ++row) {
            if (result.candidates[row].readback_bytes
                > std::numeric_limits<std::uint32_t>::max() - bytes) {
                throw std::overflow_error("target verifier compact readback byte count overflow");
            }
            bytes += result.candidates[row].readback_bytes;
        }
    } catch (...) {
        FeatureSinkRuntimeAccess::discard_speculative(ticket.impl_->feature_sink);
        Gemma2MetalVerificationStateAccess::clear(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::idle;
        throw;
    }
    result.execution = make_execution_stats(
        stats, metadata, ticket.impl_->selector_dispatch_count, bytes
    );
    return result;
}

TargetVerifierCommitReceipt provisionally_commit_gemma2_metal_target_verifier(
    TargetVerificationTicket& ticket,
    std::span<const std::uint32_t> selected_node_path
) {
    if (!ticket.impl_ || ticket.state() != VerifierTransactionState::completed) {
        throw std::logic_error("target verifier provisional commit requires completion");
    }
    const auto& metadata = ticket.impl_->metadata;
    if (selected_node_path.empty() || selected_node_path.size() > metadata.input_row_count
        || selected_node_path.size() > external_speculation_max_horizon + 1U) {
        throw std::invalid_argument("target verifier selected path extent is invalid");
    }
    for (std::size_t index = 0; index < selected_node_path.size(); ++index) {
        const auto node = selected_node_path[index];
        if (node >= metadata.input_row_count
            || (index == 0U && metadata.depths[node] != metadata.root_source_depth)
            || (index > 0U
                && metadata.parent_indices[node] != selected_node_path[index - 1U])) {
            throw std::invalid_argument("target verifier selected nodes are not one causal path");
        }
    }

    const auto count = static_cast<std::uint32_t>(selected_node_path.size());
    const auto position = metadata.cache_position;
    auto& command = ticket.impl_->commit_command;
    command.restart();
    command.write_u32s(ticket.impl_->verifier->scratch().selected_path_staging, 0U,
        selected_node_path);
    std::array<std::uint32_t, external_speculation_max_horizon + 1U>
        selected_absolute_positions {};
    for (std::uint32_t row = 0U; row < count; ++row) {
        selected_absolute_positions[row] =
            metadata.absolute_positions[selected_node_path[row]];
    }
    try {
        FeatureSinkRuntimeAccess::encode_speculative_promotion(
            ticket.impl_->feature_sink, command, selected_node_path,
            std::span<const std::uint32_t>(selected_absolute_positions).first(count)
        );
        DraftCacheRuntimeAccess::encode_promotion(
            ticket.impl_->draft_cache, command, selected_node_path,
            std::span<const std::uint32_t>(selected_absolute_positions).first(count)
        );
        command.encode_memory_barrier();
        const auto& config = ticket.impl_->target->config();
        const auto row_width = config.attention_head_count_kv * config.attention_key_length;
        // Backups read independent durable layers; join all old-row reads once
        // before publishing any selected private rows into those ring slots.
        for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
            const auto& cache = ticket.impl_->target->kv_cache()[layer];
            const auto& private_cache = ticket.impl_->verifier->scratch().kv_layers[layer];
            const auto backup_mask = static_cast<std::uint32_t>(
                private_cache.accepted_backup_key.descriptor().shape.dimension(0)) - 1U;
            command.encode_gemma2_kv_cache_copy(cache.key, cache.value,
                private_cache.accepted_backup_key, private_cache.accepted_backup_value,
                position, count, cache.row_mask, backup_mask);
        }
        command.encode_memory_barrier();
        for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
            const auto& cache = ticket.impl_->target->kv_cache()[layer];
            const auto& private_cache = ticket.impl_->verifier->scratch().kv_layers[layer];
            command.encode_verifier_kv_scatter(
                private_cache.private_key, private_cache.private_value,
                ticket.impl_->verifier->scratch().selected_path_staging,
                cache.key, cache.value, position, count, row_width, cache.row_mask
            );
        }
    } catch (...) {
        FeatureSinkRuntimeAccess::abort_promotion(ticket.impl_->feature_sink);
        DraftCacheRuntimeAccess::abort_promotion(ticket.impl_->draft_cache);
        throw;
    }
    metal::MetalCommandStats stats;
    try {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
        inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::commit);
#endif
        stats = command.commit_and_wait();
        FeatureSinkRuntimeAccess::complete_speculative_promotion(
            ticket.impl_->feature_sink, command
        );
        DraftCacheRuntimeAccess::complete_promotion(
            ticket.impl_->draft_cache, command
        );
    } catch (...) {
        FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
        DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
        Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::poisoned;
        throw;
    }
    Gemma2MetalVerificationStateAccess::set_provisional(*ticket.impl_->target, count);
    ticket.impl_->state = VerifierTransactionState::provisional;
    ticket.impl_->provisional_count = count;
    std::copy(selected_node_path.begin(), selected_node_path.end(),
        ticket.impl_->provisional_path.begin());
    TargetVerifierCommitReceipt receipt;
    receipt.retained_row_count = count;
    std::copy(selected_node_path.begin(), selected_node_path.end(),
        receipt.retained_node_indices.begin());
    receipt.execution = execution_from_commit(stats, metadata);
    return receipt;
}

TargetVerifierCommitReceipt finalize_gemma2_metal_target_verifier(
    TargetVerificationTicket& ticket,
    std::uint32_t retained_row_count
) {
    if (!ticket.impl_ || ticket.state() != VerifierTransactionState::provisional) {
        throw std::logic_error("target verifier finalize requires a provisional commit");
    }
    if (retained_row_count > ticket.impl_->provisional_count) {
        throw std::invalid_argument("target verifier cannot retain an uncommitted row");
    }
    metal::MetalCommandStats stats;
    if (retained_row_count < ticket.impl_->provisional_count) {
        const auto position = ticket.impl_->metadata.cache_position + retained_row_count;
        const auto count = ticket.impl_->provisional_count - retained_row_count;
        auto& command = ticket.impl_->commit_command;
        command.restart();
#if defined(LOCAL_GGUF_ENGINE_TESTING)
        try {
            inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::restore);
        } catch (...) {
            FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
            DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
            Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
            ticket.impl_->state = VerifierTransactionState::poisoned;
            throw;
        }
#endif
        for (std::uint32_t layer = 0; layer < ticket.impl_->target->config().block_count; ++layer) {
            const auto& cache = ticket.impl_->target->kv_cache()[layer];
            const auto& private_cache = ticket.impl_->verifier->scratch().kv_layers[layer];
            const auto backup_mask = static_cast<std::uint32_t>(
                private_cache.accepted_backup_key.descriptor().shape.dimension(0)) - 1U;
            command.encode_gemma2_kv_cache_copy(
                private_cache.accepted_backup_key, private_cache.accepted_backup_value,
                cache.key, cache.value, position, count,
                backup_mask, cache.row_mask
            );
        }
        try {
            stats = command.commit_and_wait();
        } catch (...) {
            FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
            DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
            Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
            ticket.impl_->state = VerifierTransactionState::poisoned;
            throw;
        }
    }
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    try {
        inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint::finalize);
    } catch (...) {
        FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
        DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
        Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::poisoned;
        throw;
    }
#endif
    if (ticket.impl_->feature_sink.sinks != nullptr) {
        try {
            FeatureSinkRuntimeAccess::truncate(
                *ticket.impl_->feature_sink.sinks,
                ticket.impl_->metadata.cache_position + retained_row_count
            );
        } catch (...) {
            FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
            DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
            Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
            ticket.impl_->state = VerifierTransactionState::poisoned;
            throw;
        }
    }
    if (ticket.impl_->draft_cache.arena != nullptr) {
        try {
            DraftCacheRuntimeAccess::truncate(
                *ticket.impl_->draft_cache.arena,
                ticket.impl_->metadata.cache_position + retained_row_count
            );
        } catch (...) {
            FeatureSinkRuntimeAccess::poison(ticket.impl_->feature_sink);
            DraftCacheRuntimeAccess::poison(ticket.impl_->draft_cache);
            Gemma2MetalVerificationStateAccess::poison(*ticket.impl_->target);
            ticket.impl_->state = VerifierTransactionState::poisoned;
            throw;
        }
    }
    const auto execution = execution_from_commit(stats, ticket.impl_->metadata);
    Gemma2MetalVerificationStateAccess::clear(*ticket.impl_->target);
    ticket.impl_->state = VerifierTransactionState::idle;
    TargetVerifierCommitReceipt receipt;
    receipt.retained_row_count = retained_row_count;
    std::copy_n(ticket.impl_->provisional_path.begin(), retained_row_count,
        receipt.retained_node_indices.begin());
    receipt.execution = execution;
    return receipt;
}

void discard_gemma2_metal_target_verifier(TargetVerificationTicket& ticket) {
    if (!ticket.impl_ || ticket.state() == VerifierTransactionState::idle) return;
    if (ticket.impl_->state == VerifierTransactionState::provisional) {
        static_cast<void>(finalize_gemma2_metal_target_verifier(ticket, 0U));
        return;
    }
    if (ticket.impl_->state == VerifierTransactionState::encoded
        || ticket.impl_->state == VerifierTransactionState::completed) {
        if (ticket.impl_->state == VerifierTransactionState::encoded) {
            FeatureSinkRuntimeAccess::abort_unsubmitted(ticket.impl_->feature_sink);
        } else {
            FeatureSinkRuntimeAccess::discard_speculative(ticket.impl_->feature_sink);
        }
        Gemma2MetalVerificationStateAccess::clear(*ticket.impl_->target);
        ticket.impl_->state = VerifierTransactionState::idle;
        return;
    }
    throw std::logic_error("submitted or poisoned target verifier cannot be discarded safely");
}

Gemma2MetalDecodeExecutionStats run_gemma2_metal_target_verifier_endpoint(
    Gemma2MetalModelPlan& target,
    std::uint32_t token_id,
    std::uint32_t cache_position,
    float embedding_scale
) {
    const auto result = run_gemma2_metal_resident_decode_step(
        target,
        create_gemma2_metal_decode_schedule(target, cache_position),
        token_id,
        embedding_scale,
        std::nullopt,
        Gemma2MetalDecodeOutputMode::kv_only
    );
    if (result.produced_logits || result.candidates.has_value()) {
        throw std::logic_error("target verifier endpoint unexpectedly produced selection output");
    }
    return result.execution;
}

}  // namespace local_gguf_engine::engine
