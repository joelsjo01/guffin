#include "draft_vocabulary_tests.hpp"

#include "local_gguf_engine/engine/draft_vocabulary.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "tensor_test_support.hpp"
#include "../../src/tensor/backends/metal/metal_command_context.hpp"
#include "../../src/tensor/backends/metal/metal_test_hooks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace local_gguf_engine;
using namespace local_gguf_engine::engine;
using runtime::BackendError;
using runtime::BackendErrorCode;
using runtime::BackendKind;
using tensor::allocate_backend_buffer;
using tensor::copy_tensor_from_backend;
using tensor::copy_tensor_to_backend;
using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorStorage;
using tensor::backends::metal::MetalCommandContext;
using tensor::backends::metal::set_metal_device_unavailable_for_testing;
using tests::expect;
using tests::expect_close;
using tests::reference_f16_staged_quantized_gguf_weight_matmul;
using tests::reference_f32_quantized_gguf_weight_matmul;
using tests::tensor_f16_gguf_matrix;
using tests::tensor_f32_matrix;
using tests::tensor_f32_projection_values;
using tests::tensor_q4_k_stress_gguf_weight_matrix;
using tests::tensor_q6_k_stress_gguf_weight_matrix;
using tests::TestCase;

[[nodiscard]] TensorStorage raw_u32_matrix(
    std::uint32_t rows,
    std::uint32_t columns,
    std::span<const std::uint32_t> values
) {
    expect(values.size() == static_cast<std::size_t>(rows) * columns,
        "raw u32 fixture extent mismatch");
    auto storage = TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({rows, columns}), DType::f32)
    );
    std::memcpy(storage.mutable_bytes().data(), values.data(), values.size_bytes());
    return storage;
}

[[nodiscard]] std::span<const DraftVocabularyCandidate> candidate_bytes(
    const TensorStorage& storage
) {
    return {
        reinterpret_cast<const DraftVocabularyCandidate*>(storage.bytes().data()),
        storage.bytes().size() / sizeof(DraftVocabularyCandidate),
    };
}

void test_phase4_registry_shapes_are_exact() {
    for (const auto rows : draft_vocabulary_state_batches) {
        expect(is_supported_draft_vocabulary_state_batch(rows),
            "registered draft state batch was rejected");
    }
    expect(!is_supported_draft_vocabulary_state_batch(2U)
            && !is_supported_draft_vocabulary_state_batch(22U),
        "unregistered draft state batch was accepted");
    for (std::uint32_t k = 1'024U; k <= 131'072U; k *= 2U) {
        expect(is_supported_draft_vocabulary_candidate_k(k, 256'000U),
            "registered draft candidate k was rejected");
    }
    expect(is_supported_draft_vocabulary_candidate_k(256'000U, 256'000U),
        "full-vocabulary control was rejected");
    expect(!is_supported_draft_vocabulary_candidate_k(3'000U, 256'000U),
        "unregistered draft candidate k was accepted");

    const auto released = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::treeflash_released_h15_m16_top16
    );
    const auto paper = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::treeflash_paper_h16_m64_full_v_lazy
    );
    const auto code = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::treeflash_code_h16_m64_full_v_lazy
    );
    const auto h15_m64_top16 = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::treeflash_h15_m64_top16
    );
    expect(released.physical_head_rows == 240U
            && released.rank_entry_capacity == 3'840U,
        "released TreeFlash physical rank ledger changed");
    expect(h15_m64_top16.physical_head_rows == 960U
            && h15_m64_top16.rank_entry_capacity == 15'360U,
        "TreeFlash H15/M64/topK16 physical rank ledger changed");
    expect(paper.physical_head_rows == 961U
            && paper.rank_entry_capacity == 492'032U
            && paper.head_macs == 881'721'344'000ULL,
        "paper TreeFlash H16 semantic ledger changed");
    expect(code.physical_head_rows == 1'024U
            && code.rank_entry_capacity == 524'288U
            && code.head_macs == 939'524'096'000ULL,
        "code-shaped TreeFlash H16 physical ledger changed");

    const auto cross = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::arm_s_cross_root_excluded
    );
    const auto carried = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::arm_s_cross_root_included
    );
    const auto deep = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::arm_s_deep_gamma23_k1
    );
    const auto threshold = corrected_vocabulary_workload_spec(
        CorrectedVocabularyWorkload::arm_s_threshold_gamma4_k3
    );
    expect(cross.physical_head_rows == 160U && cross.raw_lower_main_rows == 121U,
        "Arm-S root-excluded ledger changed");
    expect(carried.physical_head_rows == 165U && carried.raw_lower_main_rows == 121U,
        "Arm-S carried-root ledger changed");
    expect(deep.upper_stream_rows == 552U && deep.physical_head_rows == 552U,
        "Arm-S gamma23/k1 ledger changed");
    expect(threshold.upper_main_rows == 121U
            && threshold.upper_stream_rows == 484U
            && threshold.physical_head_rows == 605U,
        "Arm-S threshold-only ledger changed");
}

void test_cpu_stable_topk_union_and_proposal_contract() {
    constexpr std::uint32_t rows = 2U;
    constexpr std::uint32_t vocabulary = 8U;
    constexpr std::uint32_t k = 4U;
    const std::array<float, rows * vocabulary> scores {
        1.0F, 4.0F, 4.0F, -1.0F, 3.0F, 2.0F, 4.0F, 0.0F,
        0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
    };
    const auto selected = stable_top_k_cpu(scores, rows, vocabulary, k);
    expect(selected[0].token_id == 1U && selected[1].token_id == 2U
            && selected[2].token_id == 6U,
        "stable top-k did not break equal scores by token ID");

    std::array<std::uint32_t, rows> selector_counts {3U, 2U};
    std::array<std::uint32_t, rows * draft_vocabulary_auxiliary_width> history {};
    std::array<std::uint32_t, rows * draft_vocabulary_auxiliary_width> required {};
    std::array<std::uint32_t, rows * draft_vocabulary_auxiliary_width> retrieval {};
    history[0] = 7U;
    required[0] = 2U;
    retrieval[0] = 5U;
    history[draft_vocabulary_auxiliary_width] = 0U;
    std::array<std::uint32_t, rows> history_counts {1U, 1U};
    std::array<std::uint32_t, rows> required_counts {1U, 0U};
    std::array<std::uint32_t, rows> retrieval_counts {1U, 0U};
    std::array<std::uint32_t, vocabulary> transition_counts {};
    transition_counts[1] = 2U;
    std::array<std::uint32_t, vocabulary * 2U> transitions {};
    transitions[2U] = 3U;
    transitions[3U] = 5U;
    std::array<std::uint32_t, rows> output_counts {};
    const auto united = candidate_union_cpu(
        selected, rows, k, selector_counts,
        history, history_counts, required, required_counts, retrieval, retrieval_counts,
        transition_counts, transitions, 2U, vocabulary, output_counts
    );
    const std::array<std::uint32_t, 6U> expected_first {1U, 2U, 3U, 5U, 6U, 7U};
    expect(output_counts[0] == expected_first.size()
            && std::equal(expected_first.begin(), expected_first.end(), united.begin()),
        "candidate union did not stably deduplicate selector/transition/auxiliary IDs");

    std::array<std::uint32_t, draft_vocabulary_proposal_lut_size> lut {};
    for (std::size_t index = 0U; index < lut.size(); ++index) {
        lut[index] = static_cast<std::uint32_t>(std::max(
            1.0, std::round(std::exp(-static_cast<double>(index) / 256.0)
                * 16'777'216.0)
        ));
    }
    const std::array<std::uint32_t, 3U> ids {2U, 5U, 7U};
    const std::array<float, 3U> proposal_scores {1.0F, 0.75F, -0.125F};
    const auto proposal = select_draft_proposal_cpu(
        ids, proposal_scores, lut, 0.8F, 0x8123'4567U
    );
    expect(proposal.status == DraftProposalStatus::selected
            && proposal.selected_rank < ids.size()
            && proposal.token_id == ids[proposal.selected_rank],
        "CPU proposal transform did not select from its stable ticket order");
    const std::array<float, 2U> boundary_scores {1.0F, 1.0F - (0.5F / 256.0F)};
    const std::array<std::uint32_t, 2U> boundary_ids {1U, 2U};
    expect(select_draft_proposal_cpu(
        boundary_ids, boundary_scores, lut, 1.0F, 0U
    ).status == DraftProposalStatus::conservative_numeric_fallback,
        "proposal transform did not fail conservatively at a quantization boundary");
}

void test_indexed_cpu_supports_all_artifact_table_formats() {
    constexpr std::uint32_t rows = 3U;
    constexpr std::uint32_t shared = 256U;
    constexpr std::uint32_t vocabulary = 7U;
    const auto values = tensor_f32_projection_values(rows, shared);
    const auto f16_values = tensor_f32_projection_values(shared, vocabulary);
    const auto f16 = tensor_f16_gguf_matrix(shared, vocabulary, f16_values);
    const auto q8 = tensor::quantize_f16_gguf_weight_q8_0(f16.view());
    const auto q4 = tensor_q4_k_stress_gguf_weight_matrix(shared, vocabulary, 7U);
    const auto q6 = tensor_q6_k_stress_gguf_weight_matrix(shared, vocabulary, 11U);
    const std::array<std::uint32_t, rows * vocabulary> ids {
        0U, 2U, 6U, 0U, 0U, 0U, 0U,
        1U, 3U, 4U, 5U, 0U, 0U, 0U,
        6U, 0U, 0U, 0U, 0U, 0U, 0U,
    };
    const std::array<std::uint32_t, rows> counts {3U, 4U, 1U};
    for (const auto* weight : {&f16, &q8, &q6, &q4}) {
        const auto dots = indexed_vocabulary_dot_cpu(
            values, rows, shared, weight->view(), ids, counts, vocabulary
        );
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t candidate = 0U; candidate < counts[row]; ++candidate) {
                expect(std::isfinite(dots[static_cast<std::size_t>(row) * vocabulary + candidate]),
                    "indexed CPU packed-column dot produced a non-finite score");
            }
        }
    }
}

void test_metal_full_v_scans_every_registered_state_batch_and_format() {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) return;
    constexpr std::uint32_t shared = 256U;
    constexpr std::uint32_t columns = 7U;
    const auto weight_values = tensor_f32_projection_values(shared, columns);
    const auto f16 = tensor_f16_gguf_matrix(shared, columns, weight_values);
    const auto q8 = tensor::quantize_f16_gguf_weight_q8_0(f16.view());
    const auto q4 = tensor_q4_k_stress_gguf_weight_matrix(shared, columns, 19U);
    const auto q6 = tensor_q6_k_stress_gguf_weight_matrix(shared, columns, 23U);
    for (const auto rows : draft_vocabulary_state_batches) {
        auto input_values = tensor_f32_projection_values(rows, shared);
        // H16 must contain a real sixteenth row; poisoning it catches an H15
        // implementation hidden behind padding.
        if (rows == 16U) {
            for (std::uint32_t column = 0U; column < shared; ++column) {
                input_values[15U * shared + column] += 0.375F;
            }
        }
        const auto input = tensor_f32_matrix(rows, shared, input_values);
        for (const auto* weight : {&f16, &q8, &q6, &q4}) {
            TensorStorage expected = weight->descriptor().dtype == DType::f16
                ? tensor::ops::matmul_f32_f16_gguf_weight(input, *weight)
                : (weight->descriptor().dtype == DType::q8_0 || rows <= 8U
                    ? reference_f32_quantized_gguf_weight_matmul(input, *weight)
                    : reference_f16_staged_quantized_gguf_weight_matmul(input, *weight));
            const auto input_buffer = copy_tensor_to_backend(input, BackendKind::metal);
            const auto weight_buffer = copy_tensor_to_backend(*weight, BackendKind::metal);
            const auto output = allocate_backend_buffer(
                TensorDescriptor::from(Shape::from({rows, columns}), DType::f32),
                BackendKind::metal
            );
            MetalCommandContext command("draft full-V state-batch parity");
            if (weight->descriptor().dtype == DType::q8_0) {
                command.encode_target_head_q8_rows(
                    input_buffer, 0U, weight_buffer, output, rows
                );
            } else {
                command.encode_matmul(input_buffer, weight_buffer, output, rows);
            }
            static_cast<void>(command.commit_and_wait());
            const auto actual = copy_tensor_from_backend(output);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                for (std::uint32_t column = 0U; column < columns; ++column) {
                    const std::array<std::uint64_t, 2U> at {row, column};
                    const auto wanted = tensor::read_f32(expected, at);
                    const auto got = tensor::read_f32(actual, at);
                    const auto dtype = weight->descriptor().dtype;
                    const auto absolute = (dtype == DType::q4_k && rows > 8U) ? 2.0F
                        : (dtype == DType::q6_k && rows > 8U) ? 0.02F : 0.002F;
                    const auto relative = (dtype == DType::q4_k && rows > 8U)
                        ? 0.02F : 1.0e-5F;
                    expect(std::abs(got - wanted)
                            <= absolute + relative * std::abs(wanted),
                        "Metal draft full-V scan differs from independent packed golden");
                }
            }
        }
    }
}

void test_metal_scalable_topk_union_indexed_and_proposal_match_cpu() {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) return;
    constexpr std::uint32_t rows = 3U;
    constexpr std::uint32_t vocabulary = 1'025U;
    constexpr std::uint32_t k = 1'024U;
    std::vector<float> scores(static_cast<std::size_t>(rows) * vocabulary);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t token = 0U; token < vocabulary; ++token) {
            scores[static_cast<std::size_t>(row) * vocabulary + token]
                = static_cast<float>((token * 37U + row * 11U) % 97U)
                - static_cast<float>(token) * 0.00001F;
        }
    }
    scores[4U] = scores[5U] = 100.0F;
    const auto golden = stable_top_k_cpu(scores, rows, vocabulary, k);
    const auto score_storage = tensor_f32_matrix(rows, vocabulary, scores);
    const auto score_buffer = copy_tensor_to_backend(score_storage, BackendKind::metal);
    const auto runs_a = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary * 2U}), DType::f32),
        BackendKind::metal
    );
    const auto runs_b = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary * 2U}), DType::f32),
        BackendKind::metal
    );
    const auto status = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext topk_command("draft scalable top-k parity");
    topk_command.write_u32(status, 0U);
    const auto topk = topk_command.encode_draft_stable_top_k(
        score_buffer, runs_a, runs_b, status, rows, vocabulary, k
    );
    static_cast<void>(topk_command.commit_and_wait());
    expect(topk_command.read_verifier_status(status) == 0U,
        "Metal scalable top-k reported an input error");
    const auto selected_buffer = topk.final_buffer_is_a ? runs_a : runs_b;
    const auto selected_storage = copy_tensor_from_backend(selected_buffer);
    const auto selected = candidate_bytes(selected_storage);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t rank = 0U; rank < k; ++rank) {
            expect(selected[static_cast<std::size_t>(row) * vocabulary + rank]
                    == golden[static_cast<std::size_t>(row) * k + rank],
                "Metal scalable top-k changed a candidate ID, tie, or score");
        }
    }

    std::vector<std::uint32_t> aux_ids(rows * draft_vocabulary_auxiliary_width);
    std::vector<std::uint32_t> aux_counts(rows, 1U);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        aux_ids[row * draft_vocabulary_auxiliary_width] = vocabulary - 1U - row;
    }
    std::vector<std::uint32_t> transition_counts(vocabulary);
    std::vector<std::uint32_t> transition_tokens(vocabulary * 2U);
    transition_counts[selected[0].token_id] = 2U;
    transition_tokens[selected[0].token_id * 2U] = vocabulary - 1U;
    transition_tokens[selected[0].token_id * 2U + 1U] = 3U;
    const auto transition_count_buffer = copy_tensor_to_backend(
        raw_u32_matrix(vocabulary, 1U, transition_counts), BackendKind::metal
    );
    const auto transition_token_buffer = copy_tensor_to_backend(
        raw_u32_matrix(vocabulary, 2U, transition_tokens), BackendKind::metal
    );
    const auto aux_id_buffer = copy_tensor_to_backend(
        raw_u32_matrix(rows, draft_vocabulary_auxiliary_width, aux_ids), BackendKind::metal
    );
    const auto aux_count_buffer = copy_tensor_to_backend(
        raw_u32_matrix(rows, 1U, aux_counts), BackendKind::metal
    );
    const auto bit_words = (vocabulary + 31U) / 32U;
    const auto tile_count = (vocabulary + 511U) / 512U;
    const auto bits = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, bit_words}), DType::f32), BackendKind::metal
    );
    const auto tile_counts = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, tile_count}), DType::f32), BackendKind::metal
    );
    const auto tile_offsets = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, tile_count}), DType::f32), BackendKind::metal
    );
    const auto union_ids = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary}), DType::f32), BackendKind::metal
    );
    const auto union_counts = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, 1U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext union_command("draft candidate union parity");
    union_command.write_u32(status, 0U);
    union_command.encode_draft_candidate_union(
        selected_buffer, transition_count_buffer, transition_token_buffer,
        aux_id_buffer, aux_count_buffer, aux_id_buffer, aux_count_buffer,
        aux_id_buffer, aux_count_buffer, bits, tile_counts, tile_offsets,
        union_ids, union_counts, status, rows, vocabulary, k, 2U, false
    );
    static_cast<void>(union_command.commit_and_wait());
    expect(union_command.read_verifier_status(status) == 0U,
        "Metal candidate union reported an input error");
    const auto count_storage = copy_tensor_from_backend(union_counts);
    const auto id_storage = copy_tensor_from_backend(union_ids);
    const auto* actual_counts = reinterpret_cast<const std::uint32_t*>(
        count_storage.bytes().data()
    );
    const auto* actual_ids = reinterpret_cast<const std::uint32_t*>(id_storage.bytes().data());
    std::array<std::uint32_t, rows> selector_counts {k, k, k};
    std::array<std::uint32_t, rows> golden_counts {};
    const auto union_golden = candidate_union_cpu(
        selected, rows, vocabulary, selector_counts,
        aux_ids, aux_counts, aux_ids, aux_counts, aux_ids, aux_counts,
        transition_counts, transition_tokens, 2U, vocabulary, golden_counts
    );
    for (std::uint32_t row = 0U; row < rows; ++row) {
        expect(actual_counts[row] == golden_counts[row],
            "Metal candidate union count differs from CPU");
        expect(std::equal(
            union_golden.begin() + static_cast<std::ptrdiff_t>(row * vocabulary),
            union_golden.begin() + static_cast<std::ptrdiff_t>(row * vocabulary + golden_counts[row]),
            actual_ids + row * vocabulary
        ), "Metal candidate union order differs from stable CPU deduplication");
    }
}

void test_metal_topk_128k_and_full_vocabulary_union_boundaries() {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) return;
    constexpr auto vocabulary = std::uint32_t {131'073U};
    constexpr auto candidate_k = std::uint32_t {131'072U};
    std::vector<float> scores(vocabulary);
    for (std::uint32_t token = 0U; token < vocabulary; ++token) {
        scores[token] = static_cast<float>((token * 29U) % 4'099U);
    }
    scores[17U] = scores[18U] = 10'000.0F;
    const auto golden = stable_top_k_cpu(scores, 1U, vocabulary, candidate_k);
    const auto score_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(1U, vocabulary, scores), BackendKind::metal
    );
    const auto runs_a = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, vocabulary * 2U}), DType::f32),
        BackendKind::metal
    );
    const auto runs_b = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, vocabulary * 2U}), DType::f32),
        BackendKind::metal
    );
    const auto status = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext topk_command("draft 128K top-k boundary");
    topk_command.write_u32(status, 0U);
    const auto topk = topk_command.encode_draft_stable_top_k(
        score_buffer, runs_a, runs_b, status, 1U, vocabulary, candidate_k
    );
    static_cast<void>(topk_command.commit_and_wait());
    expect(topk_command.read_verifier_status(status) == 0U,
        "Metal 128K top-k boundary reported an error");
    const auto selected_storage = copy_tensor_from_backend(
        topk.final_buffer_is_a ? runs_a : runs_b
    );
    const auto selected = candidate_bytes(selected_storage);
    expect(std::equal(golden.begin(), golden.end(), selected.begin()),
        "Metal 128K top-k boundary changed stable ranks");

    constexpr auto narrow_k = std::uint32_t {3U};
    const auto narrow_golden = stable_top_k_cpu(scores, 1U, vocabulary, narrow_k);
    MetalCommandContext narrow_command("draft full-V narrow top-k boundary");
    narrow_command.write_u32(status, 0U);
    const auto narrow = narrow_command.encode_draft_stable_top_k(
        score_buffer, runs_a, runs_b, status, 1U, vocabulary, narrow_k
    );
    static_cast<void>(narrow_command.commit_and_wait());
    expect(narrow_command.read_verifier_status(status) == 0U,
        "Metal full-V narrow top-k boundary reported an error");
    const auto narrow_storage = copy_tensor_from_backend(
        narrow.final_buffer_is_a ? runs_a : runs_b
    );
    const auto narrow_selected = candidate_bytes(narrow_storage);
    expect(std::equal(narrow_golden.begin(), narrow_golden.end(), narrow_selected.begin()),
        "Metal full-V narrow top-k boundary changed stable ranks");

    std::vector<std::uint32_t> empty_aux(draft_vocabulary_auxiliary_width);
    std::array<std::uint32_t, 1U> zero_count {};
    const auto aux_ids = copy_tensor_to_backend(
        raw_u32_matrix(1U, draft_vocabulary_auxiliary_width, empty_aux), BackendKind::metal
    );
    const auto counts = copy_tensor_to_backend(
        raw_u32_matrix(1U, 1U, zero_count), BackendKind::metal
    );
    const auto bit_words = (vocabulary + 31U) / 32U;
    const auto tile_count = (vocabulary + 511U) / 512U;
    const auto bits = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, bit_words}), DType::f32), BackendKind::metal
    );
    const auto tile_counts = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, tile_count}), DType::f32), BackendKind::metal
    );
    const auto tile_offsets = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, tile_count}), DType::f32), BackendKind::metal
    );
    const auto union_ids = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, vocabulary}), DType::f32), BackendKind::metal
    );
    const auto union_counts = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext union_command("draft full-vocabulary union boundary");
    union_command.write_u32(status, 0U);
    union_command.encode_draft_candidate_union(
        runs_a, status, status,
        aux_ids, counts, aux_ids, counts, aux_ids, counts,
        bits, tile_counts, tile_offsets, union_ids, union_counts, status,
        1U, vocabulary, vocabulary, 0U, true
    );
    static_cast<void>(union_command.commit_and_wait());
    expect(union_command.read_verifier_status(status) == 0U,
        "Metal full-vocabulary union boundary reported an error");
    const auto full_counts = copy_tensor_from_backend(union_counts);
    const auto full_ids = copy_tensor_from_backend(union_ids);
    std::uint32_t count = 0U;
    std::memcpy(&count, full_counts.bytes().data(), sizeof(count));
    expect(count == vocabulary, "full-vocabulary union lost candidate IDs");
    const auto* ids = reinterpret_cast<const std::uint32_t*>(full_ids.bytes().data());
    for (std::uint32_t token = 0U; token < vocabulary; ++token) {
        expect(ids[token] == token, "full-vocabulary union order is not stable by token ID");
    }
}

void test_metal_indexed_all_formats_and_proposal_transform_match_cpu() {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) return;
    constexpr std::uint32_t rows = 3U;
    constexpr std::uint32_t shared = 256U;
    constexpr std::uint32_t vocabulary = 7U;
    const auto states_values = tensor_f32_projection_values(rows, shared);
    const auto states = tensor_f32_matrix(rows, shared, states_values);
    const auto weight_values = tensor_f32_projection_values(shared, vocabulary);
    const auto f16 = tensor_f16_gguf_matrix(shared, vocabulary, weight_values);
    const auto q8 = tensor::quantize_f16_gguf_weight_q8_0(f16.view());
    const auto q4 = tensor_q4_k_stress_gguf_weight_matrix(shared, vocabulary, 29U);
    const auto q6 = tensor_q6_k_stress_gguf_weight_matrix(shared, vocabulary, 31U);
    const std::array<std::uint32_t, rows * vocabulary> ids {
        0U, 2U, 6U, 0U, 0U, 0U, 0U,
        1U, 3U, 4U, 5U, 0U, 0U, 0U,
        6U, 0U, 0U, 0U, 0U, 0U, 0U,
    };
    const std::array<std::uint32_t, rows> counts {3U, 4U, 1U};
    const auto state_buffer = copy_tensor_to_backend(states, BackendKind::metal);
    const auto id_buffer = copy_tensor_to_backend(
        raw_u32_matrix(rows, vocabulary, ids), BackendKind::metal
    );
    const auto count_buffer = copy_tensor_to_backend(
        raw_u32_matrix(rows, 1U, counts), BackendKind::metal
    );
    const auto status = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32), BackendKind::metal
    );
    for (const auto* weight : {&f16, &q8, &q6, &q4}) {
        const auto golden = indexed_vocabulary_dot_cpu(
            states_values, rows, shared, weight->view(), ids, counts, vocabulary
        );
        const auto output = allocate_backend_buffer(
            TensorDescriptor::from(Shape::from({rows, vocabulary}), DType::f32),
            BackendKind::metal
        );
        MetalCommandContext command("draft indexed packed parity");
        command.write_u32(status, 0U);
        command.encode_draft_indexed_dot(
            state_buffer, copy_tensor_to_backend(*weight, BackendKind::metal),
            id_buffer, count_buffer, output, status, rows, vocabulary, 4U
        );
        static_cast<void>(command.commit_and_wait());
        expect(command.read_verifier_status(status) == 0U,
            "Metal indexed packed dot reported an input error");
        const auto actual = copy_tensor_from_backend(output);
        const auto* actual_scores = reinterpret_cast<const float*>(actual.bytes().data());
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t rank = 0U; rank < counts[row]; ++rank) {
                const auto index = static_cast<std::size_t>(row) * vocabulary + rank;
                const auto tolerance = weight->descriptor().dtype == DType::q4_k
                    ? 0.02F : 0.005F;
                expect(std::abs(actual_scores[index] - golden[index])
                        <= tolerance + 1.0e-5F * std::abs(golden[index]),
                    "Metal indexed packed-column dot differs from CPU golden");
            }
        }
    }

    std::array<float, rows * vocabulary> raw_correction_scores {};
    std::array<float, rows * vocabulary> corrections {};
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t rank = 0U; rank < counts[row]; ++rank) {
            const auto index = static_cast<std::size_t>(row) * vocabulary + rank;
            raw_correction_scores[index] = static_cast<float>(index) * 7.0F - 20.0F;
            corrections[index] = static_cast<float>(rank + 1U) * 0.125F;
        }
    }
    const auto corrected_score_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(rows, vocabulary, raw_correction_scores), BackendKind::metal
    );
    const auto correction_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(rows, vocabulary, corrections), BackendKind::metal
    );
    MetalCommandContext correction_command("draft softcap and correction parity");
    correction_command.write_u32(status, 0U);
    correction_command.encode_draft_score_correction(
        corrected_score_buffer, correction_buffer, count_buffer, status,
        rows, vocabulary, vocabulary, 30.0F
    );
    static_cast<void>(correction_command.commit_and_wait());
    expect(correction_command.read_verifier_status(status) == 0U,
        "Metal draft score correction rejected finite values");
    const auto corrected_storage = copy_tensor_from_backend(corrected_score_buffer);
    const auto* corrected = reinterpret_cast<const float*>(corrected_storage.bytes().data());
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t rank = 0U; rank < counts[row]; ++rank) {
            const auto index = static_cast<std::size_t>(row) * vocabulary + rank;
            const auto expected = 30.0F * std::tanh(raw_correction_scores[index] / 30.0F)
                + corrections[index];
            expect_close(corrected[index], expected, 3.0e-6F,
                "Metal draft softcap/correction differs from CPU equation");
        }
    }
    corrections[0] = std::numeric_limits<float>::quiet_NaN();
    const auto bad_correction_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(rows, vocabulary, corrections), BackendKind::metal
    );
    const auto bad_score_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(rows, vocabulary, raw_correction_scores), BackendKind::metal
    );
    MetalCommandContext bad_correction_command("draft invalid score correction");
    bad_correction_command.write_u32(status, 0U);
    bad_correction_command.encode_draft_score_correction(
        bad_score_buffer, bad_correction_buffer, count_buffer, status,
        rows, vocabulary, vocabulary, 30.0F
    );
    static_cast<void>(bad_correction_command.commit_and_wait());
    expect(bad_correction_command.read_verifier_status(status) != 0U,
        "Metal draft score correction accepted a non-finite correction");

    std::array<std::uint32_t, draft_vocabulary_proposal_lut_size> lut {};
    for (std::size_t index = 0U; index < lut.size(); ++index) {
        lut[index] = static_cast<std::uint32_t>(std::max(
            1.0, std::round(std::exp(-static_cast<double>(index) / 256.0)
                * 16'777'216.0)
        ));
    }
    const std::array<std::uint32_t, rows> proposal_counts {3U, 0U, 0U};
    const std::array<std::uint32_t, rows> draws {0x8123'4567U, 0U, 0U};
    std::array<float, rows * vocabulary> proposal_scores {};
    proposal_scores[0] = 1.0F;
    proposal_scores[1] = 0.75F;
    proposal_scores[2] = -0.125F;
    const auto score_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(rows, vocabulary, proposal_scores), BackendKind::metal
    );
    const auto proposal_count_buffer = copy_tensor_to_backend(
        raw_u32_matrix(rows, 1U, proposal_counts), BackendKind::metal
    );
    const auto draw_buffer = copy_tensor_to_backend(
        raw_u32_matrix(rows, 1U, draws), BackendKind::metal
    );
    const auto lut_buffer = copy_tensor_to_backend(
        raw_u32_matrix(draft_vocabulary_proposal_lut_size, 1U, lut), BackendKind::metal
    );
    const auto weights = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary}), DType::f32), BackendKind::metal
    );
    const auto results = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, 6U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext command("draft proposal transform parity");
    command.encode_draft_proposals(
        id_buffer, proposal_count_buffer, score_buffer, lut_buffer, draw_buffer,
        weights, results, 1U, vocabulary, 0.8F
    );
    static_cast<void>(command.commit_and_wait());
    const auto result_storage = copy_tensor_from_backend(results);
    const auto* result = reinterpret_cast<const DraftProposalResult*>(
        result_storage.bytes().data()
    );
    const auto golden = select_draft_proposal_cpu(
        std::span<const std::uint32_t>(ids.data(), 3U),
        std::span<const float>(proposal_scores.data(), 3U), lut, 0.8F, draws[0]
    );
    expect(result[0] == golden,
        "Metal artifact-bound proposal transform differs from CPU arithmetic contract");
}

void test_corrected_full_vocabulary_cpu_metal_parity() {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) return;
    constexpr std::uint32_t rows = 4U;
    constexpr std::uint32_t hidden = 256U;
    constexpr std::uint32_t vocabulary = 7U;
    constexpr std::uint32_t rank_width = 3U;
    auto state_values = tensor_f32_projection_values(rows, hidden);
    std::vector<float> norm_values(hidden, 1.0F);
    const auto weight_values = tensor_f32_projection_values(hidden, vocabulary);
    const auto f16 = tensor_f16_gguf_matrix(hidden, vocabulary, weight_values);
    const auto q8 = tensor::quantize_f16_gguf_weight_q8_0(f16.view());
    const auto golden = corrected_full_vocabulary_cpu(
        state_values, rows, norm_values, 1.0e-6F, 30.0F, 0.8F,
        q8.view(), rank_width
    );
    const auto states = copy_tensor_to_backend(
        tensor_f32_matrix(rows, hidden, state_values), BackendKind::metal
    );
    const auto norm = copy_tensor_to_backend(
        tensor_f32_matrix(1U, hidden, norm_values), BackendKind::metal
    );
    const auto head = copy_tensor_to_backend(q8, BackendKind::metal);
    const auto normalized = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32), BackendKind::metal
    );
    const auto logits = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary}), DType::f32), BackendKind::metal
    );
    const auto runs_a = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary * 2U}), DType::f32), BackendKind::metal
    );
    const auto runs_b = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, vocabulary * 2U}), DType::f32), BackendKind::metal
    );
    const auto logsumexp = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, 1U}), DType::f32), BackendKind::metal
    );
    const auto ranks = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows * rank_width, 2U}), DType::f32), BackendKind::metal
    );
    const auto status = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext command("corrected full-vocabulary parity");
    command.write_u32(status, 0U);
    command.encode_rms_norm(states, norm, normalized, 1.0e-6F, rows);
    command.encode_memory_barrier();
    command.encode_target_head_q8_rows(normalized, 0U, head, logits, rows);
    command.encode_memory_barrier();
    command.encode_softcap(logits, logits, 30.0F,
        static_cast<std::uint64_t>(rows) * vocabulary);
    command.encode_memory_barrier();
    command.encode_corrected_logsumexp(
        logits, logsumexp, status, rows, vocabulary, 0.8F
    );
    command.encode_memory_barrier();
    const auto topk = command.encode_draft_stable_top_k(
        logits, runs_a, runs_b, status, rows, vocabulary, rank_width
    );
    command.encode_memory_barrier();
    command.encode_corrected_rank_finalize(
        topk.final_buffer_is_a ? runs_a : runs_b, logsumexp, ranks, status,
        rows, vocabulary, rank_width, 0U, 0.8F
    );
    static_cast<void>(command.commit_and_wait());
    expect(command.read_verifier_status(status) == 0U,
        "corrected Metal scorer reported a numerical error");
    const auto actual_storage = copy_tensor_from_backend(ranks);
    const auto actual = candidate_bytes(actual_storage);
    expect(actual.size() == golden.size(), "corrected rank extent changed");
    for (std::size_t index = 0U; index < golden.size(); ++index) {
        expect(actual[index].token_id == golden[index].token_id,
            "corrected Metal scorer changed a stable top-rank token");
        expect_close(actual[index].score, golden[index].score, 3.0e-5F,
            "corrected Metal log probability differs from CPU golden");
    }
}

void test_arm_s_transition_probabilities_match_cpu_and_reject_bad_tokens() {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) return;
    constexpr auto rows = std::uint32_t {3U};
    constexpr auto vocabulary = std::uint32_t {5U};
    const std::array<float, rows * vocabulary> logits {
        -2.0F, -1.0F, 0.0F, 1.0F, 2.0F,
        0.5F, -0.5F, 1.5F, -1.5F, 0.0F,
        4.0F, 1.0F, -2.0F, 3.0F, 0.5F,
    };
    const std::array<std::uint32_t, rows> child_ids {4U, 2U, 0U};
    const auto logits_buffer = copy_tensor_to_backend(
        tensor_f32_matrix(rows, vocabulary, logits), BackendKind::metal
    );
    auto child_storage = TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({rows, 1U}), DType::f32)
    );
    std::memcpy(child_storage.mutable_bytes().data(), child_ids.data(), sizeof(child_ids));
    const auto child_buffer = copy_tensor_to_backend(child_storage, BackendKind::metal);
    const auto logsum = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, 1U}), DType::f32), BackendKind::metal
    );
    const auto probabilities = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, 1U}), DType::f32), BackendKind::metal
    );
    const auto status = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32), BackendKind::metal
    );
    MetalCommandContext command("Arm-S transition probability parity");
    command.write_u32(status, 0U);
    command.encode_corrected_logsumexp(
        logits_buffer, logsum, status, rows, vocabulary, 1.0F
    );
    command.encode_memory_barrier();
    command.encode_arm_s_transition_scores(
        logits_buffer, logsum, child_buffer, 0U, probabilities, status,
        rows, vocabulary, 0U
    );
    static_cast<void>(command.commit_and_wait());
    expect(command.read_verifier_status(status) == 0U,
        "Arm-S transition scoring rejected finite valid inputs");
    const auto actual_storage = copy_tensor_from_backend(probabilities);
    std::array<float, rows> actual {};
    std::memcpy(actual.data(), actual_storage.bytes().data(), sizeof(actual));
    for (std::uint32_t row = 0U; row < rows; ++row) {
        auto denominator = 0.0;
        for (std::uint32_t token = 0U; token < vocabulary; ++token) {
            denominator += std::exp(static_cast<double>(logits[row * vocabulary + token]));
        }
        const auto expected = static_cast<float>(
            std::exp(static_cast<double>(logits[row * vocabulary + child_ids[row]]))
                / denominator
        );
        expect_close(actual[row], expected, 2.0e-6F,
            "Arm-S Metal transition probability differs from CPU softmax");
    }

    auto bad_child_storage = child_storage;
    const auto bad_token = vocabulary;
    std::memcpy(bad_child_storage.mutable_bytes().data(), &bad_token, sizeof(bad_token));
    const auto bad_children = copy_tensor_to_backend(bad_child_storage, BackendKind::metal);
    MetalCommandContext bad_command("Arm-S invalid transition token");
    bad_command.write_u32(status, 0U);
    bad_command.encode_arm_s_transition_scores(
        logits_buffer, logsum, bad_children, 0U, probabilities, status,
        rows, vocabulary, 0U
    );
    static_cast<void>(bad_command.commit_and_wait());
    expect(bad_command.read_verifier_status(status) != 0U,
        "Arm-S transition scoring accepted an out-of-vocabulary child");
}

void test_treeflash_constructor_pop_and_tie_order() {
    constexpr std::uint32_t depths = 3U;
    constexpr std::uint32_t conditions = 2U;
    constexpr std::uint32_t ranks_per_row = 3U;
    std::array<DraftVocabularyCandidate, depths * conditions * ranks_per_row> ranks {};
    std::array<std::int32_t, depths * conditions * ranks_per_row> next {};
    for (std::uint32_t depth = 0U; depth < depths; ++depth) {
        for (std::uint32_t condition = 0U; condition < conditions; ++condition) {
            for (std::uint32_t rank = 0U; rank < ranks_per_row; ++rank) {
                const auto index = (depth * conditions + condition) * ranks_per_row + rank;
                ranks[index] = {
                    .token_id = 10U * depth + 3U * condition + rank + 1U,
                    .score = -static_cast<float>(rank + 1U),
                };
                next[index] = depth + 1U < depths
                    ? static_cast<std::int32_t>((condition + rank) % conditions) : -1;
            }
        }
    }
    // Equal sibling/child cumulative scores force the released tuple tie
    // rule: the lexicographically smaller rank path is popped first.
    ranks[0].score = -1.0F;
    ranks[1].score = -2.0F;
    ranks[conditions * ranks_per_row].score = -1.0F;
    const auto tree = build_treeflash_tree_cpu(
        {.ranks = ranks, .transition_next_indices = next,
         .depth_count = depths, .conditioning_width = conditions,
         .rank_width = ranks_per_row},
        8U
    );
    expect(tree.token_ids.size() == 8U && tree.parent_indices[0] == 0U,
        "TreeFlash constructor did not produce its requested compact tree");
    expect(tree.depths[1] == 1U && tree.token_ids[1] == ranks[0].token_id,
        "TreeFlash constructor changed its first released heap pop");
    for (std::size_t node = 1U; node < tree.parent_indices.size(); ++node) {
        expect(tree.parent_indices[node] < node,
            "TreeFlash compact tree is not prefix closed");
    }
}

void test_arm_s_pruning_is_prefix_closed_and_threshold_exact() {
    constexpr auto raw_nodes = std::uint32_t {121U};
    std::array<std::uint32_t, raw_nodes> parents {};
    parents[0] = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t node = 1U; node < raw_nodes; ++node) {
        parents[node] = (node - 1U) / 3U;
    }
    std::array<float, raw_nodes - 1U> probabilities {};
    probabilities.fill(0.9F);
    probabilities[0] = 0.1F;
    const auto pruned = prune_arm_s_raw_tree_cpu(parents, probabilities, 0.5F);
    expect(pruned.survivor_count == 81U,
        "Arm-S pruning did not remove the failed root edge's full subtree");
    for (std::uint32_t index = 0U; index < pruned.survivor_count; ++index) {
        expect(pruned.raw_node_indices[index] != 1U,
            "Arm-S pruning retained the explicitly failed child");
    }
    probabilities[0] = 0.5F;
    const auto threshold_tie = prune_arm_s_raw_tree_cpu(parents, probabilities, 0.5F);
    expect(threshold_tie.survivor_count == raw_nodes,
        "Arm-S pruning rejected an edge exactly at the frozen threshold");
    probabilities[7] = std::numeric_limits<float>::quiet_NaN();
    auto rejected = false;
    try {
        static_cast<void>(prune_arm_s_raw_tree_cpu(parents, probabilities, 0.5F));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "Arm-S pruning accepted a non-finite transition probability");
}

void test_arm_s_compact_builder_preserves_registered_capacities() {
    constexpr auto gamma = std::uint32_t {4U};
    constexpr auto branch = std::uint32_t {3U};
    constexpr auto anchors = std::uint32_t {121U};
    std::array<std::uint32_t, anchors> roots {};
    std::array<DraftVocabularyCandidate, anchors * gamma * branch> ranks {};
    for (std::uint32_t anchor = 0U; anchor < anchors; ++anchor) {
        roots[anchor] = 2U + anchor;
        for (std::uint32_t depth = 0U; depth < gamma; ++depth) {
            for (std::uint32_t rank = 0U; rank < branch; ++rank) {
                const auto index = (anchor * gamma + depth) * branch + rank;
                ranks[index] = {
                    .token_id = 1'000U + index,
                    .score = -static_cast<float>(rank),
                };
            }
        }
    }
    const auto maximum = build_arm_s_compact_next_tree_cpu(roots, ranks, gamma, branch);
    expect(maximum.node_count == arm_s_gamma4_k3_compact_node_max,
        "Arm-S threshold-only compact capacity changed");
    auto root_count = std::uint32_t {0U};
    for (std::uint32_t node = 0U; node < maximum.node_count; ++node) {
        if (maximum.depths[node] == 0U) {
            ++root_count;
            expect(maximum.parent_indices[node] == std::numeric_limits<std::uint32_t>::max(),
                "Arm-S compact root has a parent");
        } else {
            expect(maximum.parent_indices[node] < node,
                "Arm-S compact next tree is not prefix closed");
        }
    }
    expect(root_count == anchors, "Arm-S compact next tree lost an anchor root");
    const auto cross32 = build_arm_s_compact_next_tree_cpu(
        std::span<const std::uint32_t>(roots.data(), 32U),
        std::span<const DraftVocabularyCandidate>(ranks.data(), 32U * gamma * branch),
        gamma, branch
    );
    const auto cross33 = build_arm_s_compact_next_tree_cpu(
        std::span<const std::uint32_t>(roots.data(), 33U),
        std::span<const DraftVocabularyCandidate>(ranks.data(), 33U * gamma * branch),
        gamma, branch
    );
    expect(cross32.node_count == 3'872U && cross33.node_count == 3'993U,
        "Arm-S cross-mode root-convention compact ledgers changed");
    auto overflow_rejected = false;
    try {
        static_cast<void>(build_arm_s_compact_next_tree_cpu(
            roots, ranks, gamma, branch, arm_s_gamma4_k3_compact_node_max - 1U
        ));
    } catch (const std::length_error&) {
        overflow_rejected = true;
    }
    expect(overflow_rejected, "Arm-S compact builder exceeded its fixed capacity");
}

void test_metal_unavailable_fails_closed_for_phase4_storage() {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    struct Restore final {
        ~Restore() { set_metal_device_unavailable_for_testing(false); }
    } restore;
    set_metal_device_unavailable_for_testing(true);
    auto rejected = false;
    try {
        static_cast<void>(allocate_backend_buffer(
            TensorDescriptor::from(Shape::from({1U, 1U}), DType::f32),
            BackendKind::metal
        ));
    } catch (const BackendError& error) {
        rejected = error.code() == BackendErrorCode::unavailable_metal_device;
    }
    expect(rejected, "Phase-4 plan storage did not fail closed when Metal was unavailable");
#endif
}

const TestCase kTests[] {
    {"Phase 4 registry preserves exact TreeFlash and Arm-S ledgers",
        test_phase4_registry_shapes_are_exact},
    {"draft CPU stable top-k union and proposal contract is deterministic",
        test_cpu_stable_topk_union_and_proposal_contract},
    {"indexed CPU vocabulary dots support every artifact table format",
        test_indexed_cpu_supports_all_artifact_table_formats},
    {"Metal draft full-V scan covers every state batch and packed format",
        test_metal_full_v_scans_every_registered_state_batch_and_format},
    {"Metal scalable top-k and device union match CPU",
        test_metal_scalable_topk_union_indexed_and_proposal_match_cpu},
    {"Metal top-k 128K and full-vocabulary union boundaries are exact",
        test_metal_topk_128k_and_full_vocabulary_union_boundaries},
    {"Metal indexed tables and proposal arithmetic match CPU",
        test_metal_indexed_all_formats_and_proposal_transform_match_cpu},
    {"corrected full-vocabulary CPU and Metal scoring agree",
        test_corrected_full_vocabulary_cpu_metal_parity},
    {"Arm-S transition probabilities match CPU and reject malformed children",
        test_arm_s_transition_probabilities_match_cpu_and_reject_bad_tokens},
    {"TreeFlash compact constructor preserves released heap tie order",
        test_treeflash_constructor_pop_and_tie_order},
    {"Arm-S pruning is prefix closed and threshold exact",
        test_arm_s_pruning_is_prefix_closed_and_threshold_exact},
    {"Arm-S compact builder preserves every registered capacity",
        test_arm_s_compact_builder_preserves_registered_capacities},
    {"Phase 4 Metal storage fails closed when the device is unavailable",
        test_metal_unavailable_fails_closed_for_phase4_storage},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> draft_vocabulary_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
