#include "local_gguf_engine/benchmarks/corrected_vocabulary_benchmark.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace local_gguf_engine::benchmarks {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
    );
}

void write_u16(std::span<std::uint8_t> bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[offset + 1U] = static_cast<std::uint8_t>(value >> 8U);
}

void append_u32(runtime::Sha256& digest, std::uint32_t value) {
    std::array<std::uint8_t, 4U> bytes {};
    for (std::uint32_t index = 0U; index < 4U; ++index) {
        bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
    digest.update(bytes);
}

void append_u64(runtime::Sha256& digest, std::uint64_t value) {
    std::array<std::uint8_t, 8U> bytes {};
    for (std::uint32_t index = 0U; index < 8U; ++index) {
        bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
    digest.update(bytes);
}

void append_proposal_result(
    runtime::Sha256& digest,
    const engine::DraftProposalResult& proposal,
    std::uint32_t candidate_count
) {
    append_u32(digest, static_cast<std::uint32_t>(proposal.status));
    append_u32(digest, proposal.token_id);
    append_u64(digest, proposal.total_weight);
    append_u32(digest, proposal.selected_rank);
    append_u32(digest, candidate_count);
}

struct FixtureArtifactBytes {
    std::vector<std::uint8_t> encoded;
    std::uint64_t projection_bytes = 0U;
    std::uint64_t selector_head_bytes = 0U;
};

[[nodiscard]] FixtureArtifactBytes make_draft_shape_fixture(
    const engine::EngineModelPayloadReceipt& model_receipt,
    const engine::ResidentTargetPayloadReceipt& resident_receipt,
    const engine::Gemma2MetalModelPlan& target
) {
    constexpr auto selector_width = std::uint32_t {224U};
    constexpr auto candidate_k = std::uint32_t {1'024U};
    const auto hidden = target.config().embedding_length;
    const auto vocabulary = target.config().vocab_size;
    if (hidden == 0U || vocabulary < candidate_k || hidden % 32U != 0U
        || target.config().block_count > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument("draft fixture target dimensions are unsupported");
    }
    engine::SpeculativeArtifactDescriptor descriptor;
    descriptor.kind = engine::SpeculativeArtifactKind::fixture_feature_cache;
    descriptor.artifact_id = 0x5048'3456U;
    descriptor.target.model = model_receipt;
    descriptor.target.resident = resident_receipt;
    descriptor.target.config = target.config();
    descriptor.target.output_head_policy = static_cast<std::uint32_t>(
        target.output_head_policy()
    );
    descriptor.feature_bank_count = 1U;
    descriptor.feature_banks[0].bank_id = 1U;
    descriptor.feature_banks[0].tap_count = 1U;
    descriptor.feature_banks[0].taps[0] = {
        .stage = engine::FeatureTapStage::final_normalized,
        .layer_index = 0U,
    };
    descriptor.feature_banks[0].child_count = 1U;
    descriptor.feature_banks[0].children[0] = {
        .child_id = 1U,
        .output_width = selector_width,
        .weight_dtype = tensor::DType::f16,
        .schedule = engine::FeatureProjectionSchedule::lazy,
    };
    descriptor.draft_cache = {
        .layer_count = static_cast<std::uint16_t>(target.config().block_count),
        .stream_count = 1U,
        .key_width = 1U,
        .value_width = 1U,
        .maximum_depth = engine::draft_vocabulary_external_state_max,
        .maximum_active_width = 64U,
        .linear_row_capacity = engine::draft_vocabulary_external_state_max,
        .tree_row_capacity = 1'025U,
    };
    auto& draft = descriptor.draft_vocabulary;
    draft.version = engine::draft_vocabulary_artifact_version;
    draft.semantic_width = hidden;
    draft.selector_width = selector_width;
    draft.selector_storage_width = selector_width;
    draft.target_width = hidden;
    draft.vocabulary_size = vocabulary;
    draft.state_capacity = engine::draft_vocabulary_external_state_max;
    draft.candidate_capacity = candidate_k;

    std::vector<std::uint8_t> payload;
    const auto append_payload_tensor = [&](engine::DraftArtifactTensorRole role,
                                           engine::DraftArtifactElementType type,
                                           std::uint32_t d0,
                                           std::uint32_t d1,
                                           std::uint64_t byte_count) {
        if (byte_count > std::numeric_limits<std::size_t>::max() - payload.size()) {
            throw std::overflow_error("draft fixture payload exceeds host size");
        }
        auto& spec = draft.tensors[engine::draft_artifact_tensor_index(role)];
        spec.storage = engine::DraftArtifactTensorStorage::payload;
        spec.element_type = type;
        spec.dimension_0 = d0;
        spec.dimension_1 = d1;
        spec.payload_offset = payload.size();
        spec.payload_byte_count = byte_count;
        payload.resize(payload.size() + static_cast<std::size_t>(byte_count), 0U);
        return spec.payload_offset;
    };
    const auto projection_bytes = static_cast<std::uint64_t>(hidden)
        * selector_width * sizeof(std::uint16_t);
    const auto projection_offset = append_payload_tensor(
        engine::DraftArtifactTensorRole::selector_projection,
        engine::DraftArtifactElementType::f16, hidden, selector_width,
        projection_bytes
    );
    // A sparse deterministic fixture projection preserves coordinates from a
    // live target residual without pretending to be a learned selector.
    auto payload_span = std::span<std::uint8_t>(payload);
    for (std::uint32_t column = 0U; column < selector_width; ++column) {
        const auto first = (column * 17U) % hidden;
        const auto second = (first + 131U) % hidden;
        write_u16(payload_span,
            static_cast<std::size_t>(projection_offset
                + (static_cast<std::uint64_t>(column) * hidden + first) * 2U),
            tensor::f32_to_f16_bits(1.0F));
        write_u16(payload_span,
            static_cast<std::size_t>(projection_offset
                + (static_cast<std::uint64_t>(column) * hidden + second) * 2U),
            tensor::f32_to_f16_bits(0.5F));
    }

    const auto selector_descriptor = tensor::TensorDescriptor::from_quantized_gguf_weight_matrix(
        tensor::Shape::from({selector_width, vocabulary}), tensor::DType::q8_0
    );
    const auto selector_offset = append_payload_tensor(
        engine::DraftArtifactTensorRole::selector_head,
        engine::DraftArtifactElementType::q8_0, selector_width, vocabulary,
        selector_descriptor.byte_count
    );
    payload_span = std::span<std::uint8_t>(payload);
    const auto scale = tensor::f32_to_f16_bits(1.0F / 128.0F);
    const auto blocks_per_column = selector_width / 32U;
    for (std::uint32_t token = 0U; token < vocabulary; ++token) {
        for (std::uint32_t block = 0U; block < blocks_per_column; ++block) {
            const auto block_index = static_cast<std::uint64_t>(token) * blocks_per_column
                + block;
            const auto offset = selector_offset + block_index
                * tensor::Q8_0BlockView::byte_count;
            write_u16(payload_span, static_cast<std::size_t>(offset), scale);
            for (std::uint32_t element = 0U; element < 32U; ++element) {
                const auto coordinate = block * 32U + element;
                const auto raw = static_cast<std::int32_t>(
                    (static_cast<std::uint64_t>(token) * 131U
                        + static_cast<std::uint64_t>(coordinate) * 17U) % 255U
                ) - 127;
                payload_span[static_cast<std::size_t>(offset + 2U + element)] =
                    std::bit_cast<std::uint8_t>(static_cast<std::int8_t>(raw));
            }
        }
    }
    auto& candidate = draft.tensors[engine::draft_artifact_tensor_index(
        engine::DraftArtifactTensorRole::candidate_head
    )];
    candidate.storage = engine::DraftArtifactTensorStorage::resident_target_output_q8_0;
    candidate.element_type = engine::DraftArtifactElementType::q8_0;
    candidate.dimension_0 = hidden;
    candidate.dimension_1 = vocabulary;

    const auto arm_down_bytes = static_cast<std::uint64_t>(hidden) * 8U * 2U;
    const auto arm_down_offset = append_payload_tensor(
        engine::DraftArtifactTensorRole::arm_s_rank8_down,
        engine::DraftArtifactElementType::f16, hidden, 8U, arm_down_bytes
    );
    payload_span = std::span<std::uint8_t>(payload);
    for (std::uint32_t rank = 0U; rank < 8U; ++rank) {
        const auto coordinate = (rank * 421U) % hidden;
        write_u16(payload_span,
            static_cast<std::size_t>(arm_down_offset
                + (static_cast<std::uint64_t>(rank) * hidden + coordinate) * 2U),
            tensor::f32_to_f16_bits(1.0F));
    }
    const auto arm_up_bytes = static_cast<std::uint64_t>(8U) * hidden * 2U;
    const auto arm_up_offset = append_payload_tensor(
        engine::DraftArtifactTensorRole::arm_s_rank8_up,
        engine::DraftArtifactElementType::f16, 8U, hidden, arm_up_bytes
    );
    payload_span = std::span<std::uint8_t>(payload);
    for (std::uint32_t coordinate = 0U; coordinate < hidden; ++coordinate) {
        write_u16(payload_span,
            static_cast<std::size_t>(arm_up_offset
                + (static_cast<std::uint64_t>(coordinate) * 8U + coordinate % 8U) * 2U),
            tensor::f32_to_f16_bits(0.125F));
    }

    const auto lut_offset = append_payload_tensor(
        engine::DraftArtifactTensorRole::proposal_exp_lut,
        engine::DraftArtifactElementType::u32,
        engine::draft_vocabulary_proposal_lut_size, 1U,
        static_cast<std::uint64_t>(engine::draft_vocabulary_proposal_lut_size) * 4U
    );
    payload_span = std::span<std::uint8_t>(payload);
    for (std::uint32_t index = 0U;
         index < engine::draft_vocabulary_proposal_lut_size; ++index) {
        const auto value = static_cast<std::uint32_t>(std::max(
            1.0, std::round(std::exp(-static_cast<double>(index) / 256.0)
                * 16'777'216.0)
        ));
        for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
            payload_span[static_cast<std::size_t>(lut_offset + index * 4U + byte)] =
                static_cast<std::uint8_t>(value >> (byte * 8U));
        }
    }
    auto& feature_child = descriptor.feature_banks[0].children[0];
    feature_child.payload_offset = projection_offset;
    feature_child.payload_byte_count = projection_bytes;
    return {
        .encoded = engine::encode_fixture_speculative_artifact(descriptor, payload),
        .projection_bytes = projection_bytes,
        .selector_head_bytes = selector_descriptor.byte_count,
    };
}

[[nodiscard]] engine::DraftVocabularyDeviceInputs make_shape_inputs(
    std::span<const float> features,
    std::uint32_t row_count,
    std::uint32_t hidden,
    std::uint32_t vocabulary,
    std::uint32_t first_feature_row
) {
    if (features.size() < static_cast<std::size_t>(first_feature_row + row_count) * hidden) {
        throw std::invalid_argument("draft fixture feature span is too small");
    }
    const auto make_zero = [](std::uint32_t rows, std::uint32_t columns) {
        return tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
            tensor::Shape::from({rows, columns}), tensor::DType::f32
        ));
    };
    auto semantic = make_zero(row_count, hidden);
    std::memcpy(semantic.mutable_bytes().data(),
        features.data() + static_cast<std::size_t>(first_feature_row) * hidden,
        static_cast<std::size_t>(row_count) * hidden * sizeof(float));
    auto history_ids = make_zero(row_count, engine::draft_vocabulary_auxiliary_width);
    auto history_counts = make_zero(row_count, 1U);
    auto required_ids = make_zero(row_count, engine::draft_vocabulary_auxiliary_width);
    auto required_counts = make_zero(row_count, 1U);
    auto retrieval_ids = make_zero(row_count, engine::draft_vocabulary_auxiliary_width);
    auto retrieval_counts = make_zero(row_count, 1U);
    auto corrections = make_zero(row_count, vocabulary);
    auto draws = make_zero(row_count, 1U);
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        const auto draw = 0x9e37'79b9U ^ ((first_feature_row + row) * 0x85eb'ca6bU);
        std::memcpy(draws.mutable_bytes().data() + static_cast<std::size_t>(row) * 4U,
            &draw, sizeof(draw));
    }
    constexpr auto metal = runtime::BackendKind::metal;
    return {
        .semantic_states = tensor::copy_tensor_to_backend(semantic, metal),
        .history_ids = tensor::copy_tensor_to_backend(history_ids, metal),
        .history_counts = tensor::copy_tensor_to_backend(history_counts, metal),
        .required_ids = tensor::copy_tensor_to_backend(required_ids, metal),
        .required_counts = tensor::copy_tensor_to_backend(required_counts, metal),
        .retrieval_ids = tensor::copy_tensor_to_backend(retrieval_ids, metal),
        .retrieval_counts = tensor::copy_tensor_to_backend(retrieval_counts, metal),
        .correction_scores = tensor::copy_tensor_to_backend(corrections, metal),
        .common_draws = tensor::copy_tensor_to_backend(draws, metal),
    };
}

[[nodiscard]] DraftVocabularyShapeMeasurement measure_parallel_shape(
    engine::DraftVocabularyMetalPlan& plan,
    const engine::Gemma2MetalModelPlan& target,
    const engine::DraftVocabularyDeviceInputs& inputs,
    std::uint32_t candidate_k,
    std::uint64_t scheduled_weight_bytes
) {
    const auto result = engine::execute_draft_vocabulary_metal(
        plan, target, inputs, {engine::draft_vocabulary_external_state_max, candidate_k, 1.0F}
    );
    runtime::Sha256 digest;
    auto fallback = std::uint32_t {0U};
    for (std::uint32_t row = 0U; row < result.proposals.size(); ++row) {
        append_proposal_result(digest, result.proposals[row], result.candidate_counts[row]);
        fallback += result.proposals[row].status
            == engine::DraftProposalStatus::conservative_numeric_fallback ? 1U : 0U;
    }
    return {
        .row_count = engine::draft_vocabulary_external_state_max,
        .invocation_count = 1U,
        .wall_duration_ns = result.wall_duration_ns,
        .gpu_duration_ns = result.gpu_duration_ns,
        .dispatch_count = result.dispatch_count,
        .optimistic_scheduled_weight_bytes = scheduled_weight_bytes,
        .optimistic_effective_bandwidth_gbps = result.gpu_duration_ns == 0U ? 0.0
            : static_cast<double>(scheduled_weight_bytes)
                / static_cast<double>(result.gpu_duration_ns),
        .conservative_fallback_count = fallback,
        .proposal_digest = digest.finalize(),
    };
}

[[nodiscard]] DraftVocabularyShapeMeasurement measure_recurrent_shape(
    engine::DraftVocabularyMetalPlan& plan,
    const engine::Gemma2MetalModelPlan& target,
    std::span<const engine::DraftVocabularyDeviceInputs> inputs,
    std::uint32_t candidate_k,
    std::uint64_t scheduled_weight_bytes
) {
    runtime::Sha256 digest;
    auto wall = std::uint64_t {0U};
    auto gpu = std::uint64_t {0U};
    auto dispatches = std::uint64_t {0U};
    auto fallback = std::uint32_t {0U};
    for (const auto& input : inputs) {
        const auto result = engine::execute_draft_vocabulary_metal(
            plan, target, input, {1U, candidate_k, 1.0F}
        );
        wall += result.wall_duration_ns;
        gpu += result.gpu_duration_ns;
        dispatches += result.dispatch_count;
        append_proposal_result(digest, result.proposals[0], result.candidate_counts[0]);
        fallback += result.proposals[0].status
            == engine::DraftProposalStatus::conservative_numeric_fallback ? 1U : 0U;
    }
    return {
        .row_count = 1U,
        .invocation_count = static_cast<std::uint32_t>(inputs.size()),
        .wall_duration_ns = wall,
        .gpu_duration_ns = gpu,
        .dispatch_count = dispatches,
        .optimistic_scheduled_weight_bytes = scheduled_weight_bytes,
        .optimistic_effective_bandwidth_gbps = gpu == 0U ? 0.0
            : static_cast<double>(scheduled_weight_bytes) / static_cast<double>(gpu),
        .conservative_fallback_count = fallback,
        .proposal_digest = digest.finalize(),
    };
}

[[nodiscard]] ArmSPrunerShapeMeasurement measure_arm_s_pruner_shape(
    engine::ArmSEarlyExitMetalPlan& plan,
    const engine::Gemma2MetalModelPlan& target,
    const tensor::BackendBuffer& parent_states,
    const tensor::BackendBuffer& child_token_ids
) {
    constexpr auto edges = engine::arm_s_gamma4_k3_raw_edge_count;
    const auto result = engine::execute_arm_s_early_exit_metal(
        plan, target, parent_states, child_token_ids, edges
    );
    runtime::Sha256 digest;
    for (const auto probability : result.transition_probabilities) {
        append_u32(digest, std::bit_cast<std::uint32_t>(probability));
    }
    std::array<std::uint32_t, edges + 1U> parents {};
    parents[0] = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t node = 1U; node < parents.size(); ++node) {
        parents[node] = (node - 1U) / 3U;
    }
    const auto threshold = 1.0F / static_cast<float>(target.config().vocab_size);
    const auto surviving = engine::prune_arm_s_raw_tree_cpu(
        parents, result.transition_probabilities, threshold
    );
    const auto head_macs = static_cast<std::uint64_t>(edges)
        * target.config().embedding_length * target.config().vocab_size;
    return {
        .interpretation_label =
            "local bias-free rank-8 fixture; source-exact algebra/threshold unresolved",
        .raw_edge_count = edges,
        .survivor_count = surviving.survivor_count,
        .frozen_fixture_threshold = threshold,
        .wall_duration_ns = result.wall_duration_ns,
        .gpu_duration_ns = result.gpu_duration_ns,
        .dispatch_count = result.dispatch_count,
        .head_macs = head_macs,
        .head_flops = head_macs * 2U,
        .owned_weight_bytes = plan.owned_weight_bytes(),
        .scratch_bytes = plan.scratch_bytes(),
        .probability_digest = digest.finalize(),
    };
}

}  // namespace

engine::CorrectedVocabularyWorkload parse_corrected_vocabulary_workload(
    std::string_view name
) {
    using W = engine::CorrectedVocabularyWorkload;
    if (name == "treeflash-h15-m16-top16") return W::treeflash_released_h15_m16_top16;
    if (name == "treeflash-h15-m64-top16") return W::treeflash_h15_m64_top16;
    if (name == "treeflash-h15-m64-top64") return W::treeflash_h15_m64_top64;
    if (name == "treeflash-h15-m64-full") return W::treeflash_h15_m64_full_v_lazy;
    if (name == "treeflash-h16-paper-full") return W::treeflash_paper_h16_m64_full_v_lazy;
    if (name == "treeflash-h16-code-full") return W::treeflash_code_h16_m64_full_v_lazy;
    if (name == "arm-s-cross-160") return W::arm_s_cross_root_excluded;
    if (name == "arm-s-cross-165") return W::arm_s_cross_root_included;
    if (name == "arm-s-gamma23-k1") return W::arm_s_deep_gamma23_k1;
    if (name == "arm-s-threshold-605") return W::arm_s_threshold_gamma4_k3;
    if (name == "arm-s-post-walk") return W::arm_s_post_walk_selected_anchor;
    throw std::invalid_argument("unknown corrected-vocabulary workload");
}

const char* corrected_vocabulary_workload_name(
    engine::CorrectedVocabularyWorkload workload
) noexcept {
    using W = engine::CorrectedVocabularyWorkload;
    switch (workload) {
        case W::treeflash_released_h15_m16_top16: return "treeflash-h15-m16-top16";
        case W::treeflash_h15_m64_top16: return "treeflash-h15-m64-top16";
        case W::treeflash_h15_m64_top64: return "treeflash-h15-m64-top64";
        case W::treeflash_h15_m64_full_v_lazy: return "treeflash-h15-m64-full";
        case W::treeflash_paper_h16_m64_full_v_lazy: return "treeflash-h16-paper-full";
        case W::treeflash_code_h16_m64_full_v_lazy: return "treeflash-h16-code-full";
        case W::arm_s_cross_root_excluded: return "arm-s-cross-160";
        case W::arm_s_cross_root_included: return "arm-s-cross-165";
        case W::arm_s_deep_gamma23_k1: return "arm-s-gamma23-k1";
        case W::arm_s_threshold_gamma4_k3: return "arm-s-threshold-605";
        case W::arm_s_post_walk_selected_anchor: return "arm-s-post-walk";
    }
    return "unknown";
}

CorrectedVocabularyBenchmarkResult run_corrected_vocabulary_benchmark(
    const std::string& model_path,
    engine::CorrectedVocabularyWorkload workload
) {
    if (model_path.empty()) {
        throw std::invalid_argument("corrected-vocabulary benchmark model path is empty");
    }
    const auto load_begin = Clock::now();
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_verified_engine_model_from_gguf(
        file_model.view(), file_model.bytes()
    );
    const auto load_end = Clock::now();
    engine::ResidentTargetPayloadReceipt resident;
    const auto upload_begin = Clock::now();
    auto target = engine::create_gemma2_metal_model_plan(
        model, engine::Gemma2OutputHeadPolicy::q8_0, resident
    );
    const auto upload_end = Clock::now();
    if (target.config().context_length < 16U || target.config().vocab_size < 18U) {
        throw std::invalid_argument("corrected-vocabulary benchmark target is too small");
    }
    std::vector<std::uint32_t> tokens(16U);
    for (std::uint32_t index = 0U; index < tokens.size(); ++index) {
        tokens[index] = 2U + index;
    }
    const auto prefill_begin = Clock::now();
    static_cast<void>(engine::run_gemma2_metal_resident_prefill(
        target, tokens, 0U,
        engine::gemma2_input_embedding_scale(target.config().embedding_length)
    ));
    const auto prefill_end = Clock::now();
    // The benchmark setup duplicates a real final-layer residual from the
    // production target prefill. Only setup is host-staged; every measured
    // normalization/head/top-rank operation uses the production Metal path.
    const auto live_rows = tensor::copy_tensor_from_backend(
        target.prefill_scratch().feed_forward_residual
    );
    const auto hidden = target.config().embedding_length;
    const auto input_row_capacity = static_cast<std::uint32_t>(
        live_rows.descriptor().shape.dimension(0)
    );
    const auto source_row = input_row_capacity - 1U;
    const auto allocation_begin = Clock::now();
    auto corrected_plan = engine::create_corrected_vocabulary_metal_plan(target, workload);
    const auto rows = corrected_plan.spec().physical_head_rows;
    auto repeated = tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
        tensor::Shape::from({rows, hidden}), tensor::DType::f32
    ));
    const auto row_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
    const auto* source = live_rows.bytes().data()
        + static_cast<std::size_t>(source_row) * row_bytes;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        std::memcpy(repeated.mutable_bytes().data() + static_cast<std::size_t>(row) * row_bytes,
            source, row_bytes);
        // Keep rows numerically distinct without replacing the live feature.
        auto* values = reinterpret_cast<float*>(
            repeated.mutable_bytes().data() + static_cast<std::size_t>(row) * row_bytes
        );
        values[row % hidden] += static_cast<float>(row % 17U) * 0.0001F;
    }
    const auto corrected_states = tensor::copy_tensor_to_backend(
        repeated, runtime::BackendKind::metal
    );
    const auto allocation_end = Clock::now();
    auto execution = engine::execute_corrected_vocabulary_metal(
        corrected_plan, target, corrected_states, 1.0F
    );
    const auto rank_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(execution.rank_entries.data()),
        execution.rank_entries.size_bytes()
    );
    const auto rank_digest = runtime::sha256(rank_bytes);
    auto compact_nodes = std::uint32_t {0U};
    runtime::Sha256Digest compact_digest;
    if (corrected_plan.spec().upper_stream_rows != 0U) {
        const auto gamma = corrected_plan.spec().conditioning_width;
        if (gamma == 0U || corrected_plan.spec().upper_stream_rows % gamma != 0U) {
            throw std::logic_error("Arm-S corrected workload has an invalid stream ledger");
        }
        const auto anchors = corrected_plan.spec().upper_stream_rows / gamma;
        std::vector<std::uint32_t> root_tokens(anchors);
        for (std::uint32_t anchor = 0U; anchor < anchors; ++anchor) {
            root_tokens[anchor] = 2U + anchor;
        }
        const auto compact = engine::build_arm_s_compact_next_tree_cpu(
            root_tokens, execution.rank_entries, gamma, corrected_plan.spec().rank_width
        );
        compact_nodes = compact.node_count;
        runtime::Sha256 digest;
        for (std::uint32_t node = 0U; node < compact.node_count; ++node) {
            append_u32(digest, compact.token_ids[node]);
            append_u32(digest, compact.parent_indices[node]);
            append_u32(digest, compact.anchor_indices[node]);
            append_u32(digest, compact.depths[node]);
        }
        compact_digest = digest.finalize();
    }
    execution.rank_entries = {};
    return {
        .model_path = model_path,
        .workload = corrected_plan.spec(),
        .execution = execution,
        .verified_load_wall_ns = elapsed_ns(load_begin, load_end),
        .target_upload_wall_ns = elapsed_ns(upload_begin, upload_end),
        .scratch_allocation_wall_ns = elapsed_ns(allocation_begin, allocation_end),
        .live_feature_prefill_wall_ns = elapsed_ns(prefill_begin, prefill_end),
        .rank_digest = rank_digest,
        .compact_next_tree_nodes = compact_nodes,
        .compact_next_tree_digest = compact_digest,
    };
}

DraftVocabularyShapeBenchmarkResult run_draft_vocabulary_shape_benchmark(
    const std::string& model_path
) {
    constexpr auto rows = engine::draft_vocabulary_external_state_max;
    constexpr auto candidate_k = std::uint32_t {1'024U};
    if (model_path.empty()) {
        throw std::invalid_argument("draft-vocabulary shape benchmark model path is empty");
    }
    const auto load_begin = Clock::now();
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_verified_engine_model_from_gguf(
        file_model.view(), file_model.bytes()
    );
    const auto load_end = Clock::now();
    if (!model.payload_receipt()) {
        throw std::runtime_error("verified model omitted its payload receipt");
    }
    engine::ResidentTargetPayloadReceipt resident;
    const auto upload_begin = Clock::now();
    auto target = engine::create_gemma2_metal_model_plan(
        model, engine::Gemma2OutputHeadPolicy::q8_0, resident
    );
    const auto upload_end = Clock::now();
    if (target.config().context_length < rows || target.config().vocab_size < candidate_k) {
        throw std::invalid_argument("draft-vocabulary shape benchmark target is too small");
    }
    std::vector<std::uint32_t> tokens(rows);
    for (std::uint32_t index = 0U; index < rows; ++index) tokens[index] = 2U + index;
    const auto prefill_begin = Clock::now();
    static_cast<void>(engine::run_gemma2_metal_resident_prefill(
        target, tokens, 0U,
        engine::gemma2_input_embedding_scale(target.config().embedding_length)
    ));
    const auto prefill_end = Clock::now();
    const auto live = tensor::copy_tensor_from_backend(
        target.prefill_scratch().feed_forward_residual
    );
    const auto hidden = target.config().embedding_length;
    const auto live_capacity = static_cast<std::uint32_t>(
        live.descriptor().shape.dimension(0)
    );
    const auto source_row = live_capacity - 1U;
    std::vector<float> live_feature(hidden);
    std::memcpy(live_feature.data(),
        live.bytes().data() + static_cast<std::size_t>(source_row) * hidden * sizeof(float),
        static_cast<std::size_t>(hidden) * sizeof(float));
    std::vector<float> features(static_cast<std::size_t>(rows) * hidden);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        std::copy(live_feature.begin(), live_feature.end(),
            features.begin() + static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(row) * hidden));
        features[static_cast<std::size_t>(row) * hidden + row % hidden]
            += static_cast<float>(row) * 0.0001F;
    }

    const auto setup_begin = Clock::now();
    auto fixture = make_draft_shape_fixture(*model.payload_receipt(), resident, target);
    const auto artifact = engine::load_speculative_artifact(fixture.encoded);
    auto plan = engine::create_draft_vocabulary_metal_plan(
        artifact, *model.payload_receipt(), resident, target
    );
    auto arm_plan = engine::create_arm_s_early_exit_metal_plan(
        artifact, *model.payload_receipt(), resident, target
    );
    auto parallel_inputs = make_shape_inputs(
        features, rows, hidden, target.config().vocab_size, 0U
    );
    std::vector<engine::DraftVocabularyDeviceInputs> recurrent_inputs;
    recurrent_inputs.reserve(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        recurrent_inputs.push_back(make_shape_inputs(
            features, 1U, hidden, target.config().vocab_size, row
        ));
    }
    constexpr auto arm_edges = engine::arm_s_gamma4_k3_raw_edge_count;
    auto arm_parent_storage = tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
        tensor::Shape::from({arm_edges, hidden}), tensor::DType::f32
    ));
    for (std::uint32_t edge = 0U; edge < arm_edges; ++edge) {
        std::memcpy(
            arm_parent_storage.mutable_bytes().data()
                + static_cast<std::size_t>(edge) * hidden * sizeof(float),
            features.data() + static_cast<std::size_t>(edge % rows) * hidden,
            static_cast<std::size_t>(hidden) * sizeof(float)
        );
    }
    auto arm_child_storage = tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
        tensor::Shape::from({arm_edges, 1U}), tensor::DType::f32
    ));
    for (std::uint32_t edge = 0U; edge < arm_edges; ++edge) {
        const auto token = 2U + (edge * 17U) % (target.config().vocab_size - 2U);
        std::memcpy(arm_child_storage.mutable_bytes().data()
                + static_cast<std::size_t>(edge) * sizeof(std::uint32_t),
            &token, sizeof(token));
    }
    const auto arm_parent_states = tensor::copy_tensor_to_backend(
        arm_parent_storage, runtime::BackendKind::metal
    );
    const auto arm_child_tokens = tensor::copy_tensor_to_backend(
        arm_child_storage, runtime::BackendKind::metal
    );
    const auto setup_end = Clock::now();

    // Warm both command shapes once. These invocations are not reported.
    static_cast<void>(engine::execute_draft_vocabulary_metal(
        plan, target, parallel_inputs, {rows, candidate_k, 1.0F}
    ));
    for (const auto& inputs : recurrent_inputs) {
        static_cast<void>(engine::execute_draft_vocabulary_metal(
            plan, target, inputs, {1U, candidate_k, 1.0F}
        ));
    }
    static_cast<void>(engine::execute_arm_s_early_exit_metal(
        arm_plan, target, arm_parent_states, arm_child_tokens, arm_edges
    ));

    const auto candidate_column_bytes = static_cast<std::uint64_t>(hidden / 32U)
        * tensor::Q8_0BlockView::byte_count;
    const auto per_row_candidate_bytes = candidate_column_bytes * candidate_k;
    const auto parallel_scheduled = fixture.projection_bytes + fixture.selector_head_bytes
        + per_row_candidate_bytes * rows;
    const auto recurrent_scheduled = rows * (fixture.projection_bytes
        + fixture.selector_head_bytes + per_row_candidate_bytes);
    auto parallel = measure_parallel_shape(
        plan, target, parallel_inputs, candidate_k, parallel_scheduled
    );
    auto recurrent = measure_recurrent_shape(
        plan, target, recurrent_inputs, candidate_k, recurrent_scheduled
    );
    auto arm_s_pruner = measure_arm_s_pruner_shape(
        arm_plan, target, arm_parent_states, arm_child_tokens
    );
    if (parallel.proposal_digest != recurrent.proposal_digest
        || parallel.conservative_fallback_count != recurrent.conservative_fallback_count) {
        throw std::runtime_error(
            "parallel and recurrent draft fixture rollouts disagree"
        );
    }
    return {
        .model_path = model_path,
        .interpretation_label = "fixture-weight component lower bound; not trained end-to-end",
        .state_count = rows,
        .candidate_k = candidate_k,
        .verified_load_wall_ns = elapsed_ns(load_begin, load_end),
        .target_upload_wall_ns = elapsed_ns(upload_begin, upload_end),
        .live_feature_prefill_wall_ns = elapsed_ns(prefill_begin, prefill_end),
        .fixture_pack_plan_input_wall_ns = elapsed_ns(setup_begin, setup_end),
        .authenticated_fixture_bytes = fixture.encoded.size(),
        .owned_weight_bytes = plan.owned_weight_bytes(),
        .scratch_bytes = plan.scratch_bytes(),
        .measurements = {recurrent, parallel},
        .arm_s_pruner = std::move(arm_s_pruner),
    };
}

}  // namespace local_gguf_engine::benchmarks
