#include "feature_bank_tests.hpp"

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "../../src/engine/feature_bank_runtime.hpp"
#include "../../src/engine/draft_cache_runtime.hpp"
#include "../../src/tensor/backends/metal/metal_command_context.hpp"
#include "../../src/tensor/backends/metal/metal_test_hooks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace local_gguf_engine::engine;
using local_gguf_engine::runtime::BackendKind;
using namespace local_gguf_engine::tensor;
using local_gguf_engine::tensor::backends::metal::MetalCommandContext;
using local_gguf_engine::tensor::backends::metal::set_metal_device_unavailable_for_testing;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

[[nodiscard]] std::vector<std::uint8_t> projection_payload() {
    // Two column-major-by-row [shared, output] F16 matrices: identity, then
    // diag(2,3). The independent expected equations below use the same
    // mathematical orientation, not the Metal kernel implementation.
    const std::array<float, 8U> values {1.0F, 0.0F, 0.0F, 1.0F,
                                        2.0F, 0.0F, 0.0F, 3.0F};
    std::vector<std::uint8_t> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const auto bits = f32_to_f16_bits(values[index]);
        std::memcpy(bytes.data() + index * sizeof(bits), &bits, sizeof(bits));
    }
    return bytes;
}

[[nodiscard]] FeatureBankSpec bank_spec(std::span<const std::uint8_t> payload) {
    FeatureBankSpec spec;
    spec.bank_id = 1U;
    spec.tap_count = 1U;
    spec.taps[0] = {
        .stage = FeatureTapStage::layer_residual,
        .layer_index = 0U,
    };
    spec.child_count = 2U;
    spec.children[0] = {
        .child_id = 1U,
        .output_width = 2U,
        .weight_dtype = DType::f16,
        .schedule = FeatureProjectionSchedule::eager,
        .payload_offset = 0U,
        .payload_byte_count = 8U,
        .payload_digest = local_gguf_engine::runtime::sha256(payload.first(8U)),
    };
    spec.children[1] = {
        .child_id = 2U,
        .output_width = 2U,
        .weight_dtype = DType::f16,
        .schedule = FeatureProjectionSchedule::lazy,
        .payload_offset = 8U,
        .payload_byte_count = 8U,
        .payload_digest = local_gguf_engine::runtime::sha256(payload.subspan(8U, 8U)),
    };
    return spec;
}

[[nodiscard]] TensorStorage input_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({2U, 2U}), DType::f32
    ));
    write_f32(input, std::array<std::uint64_t, 2U> {0U, 0U}, 1.25F);
    write_f32(input, std::array<std::uint64_t, 2U> {0U, 1U}, -2.0F);
    write_f32(input, std::array<std::uint64_t, 2U> {1U, 0U}, 3.0F);
    write_f32(input, std::array<std::uint64_t, 2U> {1U, 1U}, 4.0F);
    return input;
}

[[nodiscard]] std::uint32_t packed_f16(float first, float second) {
    return static_cast<std::uint32_t>(f32_to_f16_bits(first))
        | (static_cast<std::uint32_t>(f32_to_f16_bits(second)) << 16U);
}

[[nodiscard]] std::vector<std::uint8_t> five_tap_projection_payload() {
    std::vector<std::uint8_t> payload(40U);
    for (std::size_t child = 0U; child < 2U; ++child) {
        for (std::size_t column = 0U; column < 10U; ++column) {
            const auto weight = static_cast<float>(column / 2U + 1U)
                * (column % 2U == 0U ? 1.0F : -1.0F);
            const auto bits = f32_to_f16_bits(weight);
            std::memcpy(payload.data() + (child * 10U + column) * sizeof(bits),
                &bits, sizeof(bits));
        }
    }
    return payload;
}

[[nodiscard]] FeatureBankSpec five_tap_bank(std::span<const std::uint8_t> payload) {
    auto spec = bank_spec(payload);
    spec.tap_count = 5U;
    for (std::uint16_t tap = 0U; tap < spec.tap_count; ++tap) {
        spec.taps[tap] = {.stage = FeatureTapStage::layer_residual, .layer_index = tap};
    }
    for (std::uint16_t child = 0U; child < spec.child_count; ++child) {
        spec.children[child].output_width = 1U;
        spec.children[child].payload_offset = child * 20U;
        spec.children[child].payload_byte_count = 20U;
        spec.children[child].payload_digest = local_gguf_engine::runtime::sha256(
            payload.subspan(child * 20U, 20U));
    }
    return spec;
}

void test_five_tap_feature_bank_parity_promotion_and_rollback() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto payload = five_tap_projection_payload();
    const std::array specs {five_tap_bank(payload)};
    auto sinks = create_feature_sink_set(specs, payload, 2U, 5U, 4U, 2U);
    std::array<FeatureTapSpec, 6U> trace_taps {};
    std::copy_n(specs[0].taps.begin(), specs[0].tap_count, trace_taps.begin());
    trace_taps[5] = {.stage = FeatureTapStage::final_normalized};
    std::optional<TargetFeatureTraceBatch> traced;
    auto trace = create_target_feature_trace_sink(trace_taps, 2U, 5U, 2U,
        [&](TargetFeatureTraceBatch&& batch) { traced = std::move(batch); });
    const auto capture = [&](FeatureSinkBinding binding,
                             std::array<std::uint32_t, 2U> positions, float scale) {
        binding.trace_sink = &trace;
        traced.reset();
        MetalCommandContext command("five-tap feature parity", true);
        FeatureSinkRuntimeAccess::begin_capture(binding, command, positions, positions, 2U);
        for (std::uint16_t tap = 0U; tap < 5U; ++tap) {
            auto input = input_rows();
            for (std::uint32_t row = 0U; row < 2U; ++row) {
                write_f32(input, std::array<std::uint64_t, 2U> {row, 0U},
                    scale * static_cast<float>((tap + 1U) * (row + 1U)));
                write_f32(input, std::array<std::uint64_t, 2U> {row, 1U},
                    -scale * static_cast<float>(tap + row + 1U));
            }
            const auto device = copy_tensor_to_backend(input, BackendKind::metal);
            FeatureSinkRuntimeAccess::encode_tap(binding, command,
                FeatureTapStage::layer_residual, tap, device, 0U, 2U);
        }
        const auto final = copy_tensor_to_backend(input_rows(), BackendKind::metal);
        FeatureSinkRuntimeAccess::encode_tap(binding, command,
            FeatureTapStage::final_normalized, 0U, final, 0U, 2U);
        static_cast<void>(command.commit_and_wait());
        FeatureSinkRuntimeAccess::complete_capture(binding, command, positions, 2U);
        expect(traced.has_value() && traced->taps.size() == 6U
            && traced->taps.back().stage == FeatureTapStage::final_normalized
            && traced->bytes.size() == 2U * 6U * 2U * sizeof(std::uint16_t),
            "five runtime taps prevented separate final-normalized teacher capture");
        std::uint16_t final_first = 0U;
        std::memcpy(&final_first, traced->bytes.data() + 5U * 2U * sizeof(final_first),
            sizeof(final_first));
        expect(final_first == f32_to_f16_bits(1.25F),
            "teacher trace final tap was replaced by a runtime conditioning tap");
    };
    const auto check = [&](std::uint32_t child, std::uint32_t start,
                           std::uint32_t count, float scale) {
        const auto actual = sinks.copy_canonical_child_feature_rows(child, start, count);
        for (std::uint32_t row = 0U; row < count; ++row) {
            float expected = 0.0F;
            for (std::uint32_t tap = 0U; tap < 5U; ++tap) {
                expected += scale * static_cast<float>(tap + 1U)
                    * static_cast<float>((tap + 1U) * (row + 1U) + tap + row + 1U);
            }
            expect(read_f16_as_f32(actual, std::array<std::uint64_t, 2U> {row, 0U})
                    == f16_bits_to_f32(f32_to_f16_bits(expected)),
                "five-tap Metal projection differs from independent scalar equation");
        }
    };
    capture({.sinks = &sinks, .execution_epoch = 1U}, {0U, 1U}, 1.0F);
    check(1U, 0U, 2U, 1.0F);
    sinks.rebuild_lazy_child(2U, 0U, 2U);
    check(2U, 0U, 2U, 1.0F);

    const FeatureSinkBinding speculative {
        .sinks = &sinks, .destination = FeatureCaptureDestination::speculative,
        .execution_epoch = 2U, .arena_epoch = 7U,
    };
    capture(speculative, {2U, 3U}, 2.0F);
    expect(!sinks.target_state_view(1U, 2U).has_value(),
        "five-tap speculative rows became canonical before promotion");
    constexpr std::array<std::uint32_t, 1U> selected {0U}, positions {2U};
    auto stale = speculative;
    ++stale.arena_epoch;
    auto rejected = false;
    try {
        MetalCommandContext command("five-tap stale promotion", true);
        FeatureSinkRuntimeAccess::encode_speculative_promotion(
            stale, command, selected, positions);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    expect(rejected, "five-tap promotion accepted stale speculative provenance");
    MetalCommandContext promotion("five-tap selected promotion", true);
    FeatureSinkRuntimeAccess::encode_speculative_promotion(
        speculative, promotion, selected, positions);
    static_cast<void>(promotion.commit_and_wait());
    FeatureSinkRuntimeAccess::complete_speculative_promotion(speculative, promotion);
    check(1U, 2U, 1U, 2.0F);
    expect(!sinks.target_state_view(2U, 2U).has_value(),
        "five-tap lazy projection skipped required rebuild after promotion");
    sinks.rebuild_lazy_child(2U, 2U, 1U);
    check(2U, 2U, 1U, 2.0F);
    expect(sinks.target_state_view(1U, 2U)->source_layer == 4U
        && sinks.target_state_view(1U, 2U)->coordinate.draft_position == 3U
        && !sinks.target_state_view(1U, 3U).has_value(),
        "five-tap promotion lost final-tap provenance or retained a rejected row");
    FeatureSinkRuntimeAccess::truncate(sinks, 2U);
    expect(sinks.target_state_view(1U, 1U).has_value()
        && !sinks.target_state_view(1U, 2U).has_value()
        && !sinks.target_state_view(2U, 2U).has_value(),
        "five-tap rollback retained an eager or lazy rejected feature");
}

void test_five_tap_feature_bank_rejects_malformed_and_incomplete_capture() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto payload = five_tap_projection_payload();
    const auto valid = five_tap_bank(payload);
    for (std::uint32_t failure = 0U; failure < 5U; ++failure) {
        auto invalid = valid;
        auto hidden_width = 2U;
        if (failure == 0U) invalid.tap_count = 6U;
        if (failure == 1U) invalid.taps[4].layer_index = 5U;
        if (failure == 2U) invalid.taps[4] = invalid.taps[3];
        if (failure == 3U) --invalid.children[0].payload_byte_count;
        if (failure == 4U) hidden_width = std::numeric_limits<std::uint32_t>::max();
        auto rejected = false;
        try {
            static_cast<void>(create_feature_sink_set(
                std::array {invalid}, payload, hidden_width, 5U, 4U, 2U));
        } catch (const std::invalid_argument&) {
            rejected = failure != 4U;
        } catch (const std::overflow_error&) {
            rejected = failure == 4U;
        }
        expect(rejected, "five-tap feature bank accepted malformed bounds or dimensions");
    }
    const auto input = copy_tensor_to_backend(input_rows(), BackendKind::metal);
    constexpr std::array<std::uint32_t, 2U> positions {0U, 1U};
    for (std::uint32_t failure = 0U; failure < 3U; ++failure) {
        auto sinks = create_feature_sink_set(std::array {valid}, payload, 2U, 5U, 4U, 2U);
        const FeatureSinkBinding binding {.sinks = &sinks, .execution_epoch = 1U};
        MetalCommandContext command("five-tap malformed capture", true);
        FeatureSinkRuntimeAccess::begin_capture(binding, command, positions, positions, 2U);
        auto rejected = false;
        try {
            if (failure == 0U) {
                FeatureSinkRuntimeAccess::encode_tap(binding, command,
                    FeatureTapStage::layer_residual, 4U, input, 0U, 2U);
            } else {
                for (std::uint16_t tap = 0U; tap < 4U; ++tap) {
                    FeatureSinkRuntimeAccess::encode_tap(binding, command,
                        FeatureTapStage::layer_residual, tap, input, 0U, 2U);
                }
                if (failure == 1U) {
                    FeatureSinkRuntimeAccess::encode_tap(binding, command,
                        FeatureTapStage::layer_residual, 3U, input, 0U, 2U);
                } else {
                    static_cast<void>(command.commit_and_wait());
                    FeatureSinkRuntimeAccess::complete_capture(binding, command, positions, 2U);
                }
            }
        } catch (const std::logic_error&) {
            rejected = failure != 2U;
        } catch (const std::runtime_error&) {
            rejected = failure == 2U && sinks.poisoned();
        }
        expect(rejected && !sinks.target_state_view(1U, 0U).has_value(),
            "five-tap capture accepted out-of-order, duplicate, or missing target taps");
        FeatureSinkRuntimeAccess::abort_unsubmitted(binding);
    }
}

void test_feature_bank_eager_and_lazy_metal_match_scalar_equations() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto payload = projection_payload();
    const std::array specs {bank_spec(payload)};
    auto sinks = create_feature_sink_set(specs, payload, 2U, 1U, 4U, 2U);
    const auto input = copy_tensor_to_backend(input_rows(), BackendKind::metal);
    const std::array<std::uint32_t, 2U> positions {0U, 1U};
    const FeatureSinkBinding binding {
        .sinks = &sinks,
        .destination = FeatureCaptureDestination::canonical,
        .execution_epoch = 1U,
        .arena_epoch = 0U,
    };
    MetalCommandContext command("feature-bank parity", true);
    FeatureSinkRuntimeAccess::begin_capture(binding, command, positions, positions, 2U);
    FeatureSinkRuntimeAccess::encode_tap(
        binding, command, FeatureTapStage::layer_residual, 0U, input, 0U, 2U
    );
    static_cast<void>(command.commit_and_wait());
    FeatureSinkRuntimeAccess::complete_capture(binding, command, positions, 2U);

    const auto eager = sinks.copy_canonical_child_feature_rows(1U, 0U, 2U);
    const std::array<std::array<float, 2U>, 2U> source {{
        {1.25F, -2.0F}, {3.0F, 4.0F},
    }};
    for (std::uint32_t row = 0U; row < 2U; ++row) {
        for (std::uint32_t column = 0U; column < 2U; ++column) {
            const auto actual = read_f16_as_f32(
                eager, std::array<std::uint64_t, 2U> {row, column}
            );
            expect(actual == f16_bits_to_f32(f32_to_f16_bits(source[row][column])),
                "eager feature projection differs from scalar identity equation");
        }
    }
    const auto eager_view = sinks.target_state_view(1U, 1U);
    expect(eager_view.has_value()
        && eager_view->coordinate.target_position == 1U
        && eager_view->coordinate.draft_position == 2U,
        "eager feature p/p+1 provenance changed");
    expect(!sinks.target_state_view(2U, 0U).has_value(),
        "lazy feature became visible before its charged rebuild");

    sinks.rebuild_lazy_child(2U, 0U, 2U);
    const auto lazy = sinks.copy_canonical_child_feature_rows(2U, 0U, 2U);
    const std::array<float, 2U> scales {2.0F, 3.0F};
    for (std::uint32_t row = 0U; row < 2U; ++row) {
        for (std::uint32_t column = 0U; column < 2U; ++column) {
            const auto expected = f16_bits_to_f32(f32_to_f16_bits(
                source[row][column] * scales[column]
            ));
            const auto actual = read_f16_as_f32(
                lazy, std::array<std::uint64_t, 2U> {row, column}
            );
            expect(std::abs(actual - expected) == 0.0F,
                "lazy feature rebuild differs from scalar projection equation");
        }
    }
    expect(sinks.target_state_view(2U, 0U).has_value()
        && sinks.metrics().lazy_rebuild_row_count == 2U
        && sinks.metrics().trace_readback_row_count == 4U,
        "lazy feature rebuild did not publish provenance and charge rows");

    auto rejected = false;
    try {
        static_cast<void>(sinks.copy_canonical_child_feature_rows(1U, 1U, 2U));
    } catch (const std::logic_error&) {
        rejected = true;
    }
    expect(rejected, "bounded feature readback crossed the canonical frontier");
}

void test_feature_bank_fails_closed_without_metal() {
    class ScopedUnavailable final {
    public:
        ScopedUnavailable() { set_metal_device_unavailable_for_testing(true); }
        ~ScopedUnavailable() { set_metal_device_unavailable_for_testing(false); }
    } unavailable;
    const auto payload = five_tap_projection_payload();
    const std::array specs {five_tap_bank(payload)};
    auto rejected = false;
    try {
        static_cast<void>(create_feature_sink_set(specs, payload, 2U, 5U, 4U, 2U));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    expect(rejected, "feature-bank creation did not fail closed without Metal");

    const std::array trace_taps {FeatureTapSpec {
        .stage = FeatureTapStage::final_normalized,
    }};
    rejected = false;
    try {
        static_cast<void>(create_target_feature_trace_sink(
            trace_taps, 2U, 1U, 2U, [](TargetFeatureTraceBatch&&) {}
        ));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    expect(rejected, "raw target-feature trace sink did not fail closed without Metal");
}

void test_target_feature_trace_sink_streams_exact_f16_taps() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const std::array taps {
        FeatureTapSpec {.stage = FeatureTapStage::embedding_residual},
        FeatureTapSpec {.stage = FeatureTapStage::layer_residual, .layer_index = 0U},
    };
    std::optional<TargetFeatureTraceBatch> captured;
    auto sink = create_target_feature_trace_sink(
        taps, 2U, 1U, 2U,
        [&](TargetFeatureTraceBatch&& batch) { captured = std::move(batch); }
    );
    auto first = input_rows();
    auto second = input_rows();
    write_f32(second, std::array<std::uint64_t, 2U> {0U, 0U}, -7.0F);
    write_f32(second, std::array<std::uint64_t, 2U> {0U, 1U}, 8.0F);
    write_f32(second, std::array<std::uint64_t, 2U> {1U, 0U}, 9.0F);
    write_f32(second, std::array<std::uint64_t, 2U> {1U, 1U}, -10.0F);
    const auto first_buffer = copy_tensor_to_backend(first, BackendKind::metal);
    const auto second_buffer = copy_tensor_to_backend(second, BackendKind::metal);
    const std::array<std::uint32_t, 2U> positions {11U, 12U};
    const FeatureSinkBinding binding {
        .trace_sink = &sink,
        .execution_epoch = 5U,
    };
    MetalCommandContext command("raw feature trace parity", true);
    FeatureSinkRuntimeAccess::begin_capture(binding, command, positions, positions, 2U);
    FeatureSinkRuntimeAccess::encode_tap(
        binding, command, FeatureTapStage::embedding_residual, 0U,
        first_buffer, 0U, 2U
    );
    FeatureSinkRuntimeAccess::encode_tap(
        binding, command, FeatureTapStage::layer_residual, 0U,
        second_buffer, 0U, 2U
    );
    static_cast<void>(command.commit_and_wait());
    FeatureSinkRuntimeAccess::complete_capture(binding, command, positions, 2U);
    expect(captured.has_value() && captured->execution_epoch == 5U
            && captured->absolute_positions == std::vector<std::uint32_t>({11U, 12U})
            && captured->input_token_ids == std::vector<std::uint32_t>({11U, 12U})
            && captured->taps == std::vector<FeatureTapSpec>(taps.begin(), taps.end())
            && captured->bytes.size() == 2U * 2U * 2U * sizeof(std::uint16_t),
        "raw feature trace batch metadata or bounds changed");
    const std::array<float, 8U> expected {
        1.25F, -2.0F, -7.0F, 8.0F,
        3.0F, 4.0F, 9.0F, -10.0F,
    };
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        std::uint16_t actual = 0U;
        std::memcpy(&actual, captured->bytes.data() + index * sizeof(actual), sizeof(actual));
        expect(actual == f32_to_f16_bits(expected[index]),
            "raw feature trace F16 row/tap/hidden layout changed");
    }

    captured.reset();
    MetalCommandContext missing("raw feature trace missing tap", true);
    FeatureSinkRuntimeAccess::begin_capture(binding, missing, positions, positions, 2U);
    FeatureSinkRuntimeAccess::encode_tap(
        binding, missing, FeatureTapStage::embedding_residual, 0U,
        first_buffer, 0U, 2U
    );
    static_cast<void>(missing.commit_and_wait());
    auto rejected = false;
    try {
        FeatureSinkRuntimeAccess::complete_capture(binding, missing, positions, 2U);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    expect(rejected && sink.poisoned() && !captured.has_value(),
        "raw feature trace accepted an incomplete target tap set");
}

void test_hybrid_feature_bank_concatenates_taps_in_causal_order() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const std::array<float, 4U> weights {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<std::uint8_t, 8U> payload {};
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        const auto bits = f32_to_f16_bits(weights[index]);
        std::memcpy(payload.data() + index * sizeof(bits), &bits, sizeof(bits));
    }
    FeatureBankSpec spec;
    spec.bank_id = 3U;
    spec.tap_count = 2U;
    spec.taps[0] = {.stage = FeatureTapStage::embedding_residual};
    spec.taps[1] = {.stage = FeatureTapStage::layer_residual, .layer_index = 0U};
    spec.child_count = 1U;
    spec.children[0] = {
        .child_id = 5U,
        .output_width = 1U,
        .weight_dtype = DType::f16,
        .schedule = FeatureProjectionSchedule::eager,
        .payload_offset = 0U,
        .payload_byte_count = payload.size(),
        .payload_digest = local_gguf_engine::runtime::sha256(payload),
    };
    const std::array specs {spec};
    auto sinks = create_feature_sink_set(specs, payload, 2U, 1U, 4U, 2U);

    auto embedding = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({1U, 2U}), DType::f32
    ));
    auto layer = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({1U, 2U}), DType::f32
    ));
    write_f32(embedding, std::array<std::uint64_t, 2U> {0U, 0U}, 1.0F);
    write_f32(embedding, std::array<std::uint64_t, 2U> {0U, 1U}, 2.0F);
    write_f32(layer, std::array<std::uint64_t, 2U> {0U, 0U}, 3.0F);
    write_f32(layer, std::array<std::uint64_t, 2U> {0U, 1U}, 4.0F);
    const auto embedding_buffer = copy_tensor_to_backend(embedding, BackendKind::metal);
    const auto layer_buffer = copy_tensor_to_backend(layer, BackendKind::metal);
    constexpr std::array<std::uint32_t, 1U> positions {0U};
    const FeatureSinkBinding binding {
        .sinks = &sinks,
        .destination = FeatureCaptureDestination::canonical,
        .execution_epoch = 9U,
    };
    MetalCommandContext command("hybrid feature bank parity", true);
    FeatureSinkRuntimeAccess::begin_capture(binding, command, positions, positions, 1U);
    FeatureSinkRuntimeAccess::encode_tap(
        binding, command, FeatureTapStage::embedding_residual, 0U,
        embedding_buffer, 0U, 1U
    );
    FeatureSinkRuntimeAccess::encode_tap(
        binding, command, FeatureTapStage::layer_residual, 0U,
        layer_buffer, 0U, 1U
    );
    static_cast<void>(command.commit_and_wait());
    FeatureSinkRuntimeAccess::complete_capture(binding, command, positions, 1U);
    const auto output = sinks.copy_canonical_child_feature_rows(5U, 0U, 1U);
    expect(read_f16_as_f32(output, std::array<std::uint64_t, 2U> {0U, 0U}) == 30.0F,
        "hybrid feature bank changed concatenation order or projection equation");

    auto invalid = spec;
    std::swap(invalid.taps[0], invalid.taps[1]);
    const std::array invalid_specs {invalid};
    auto rejected = false;
    try {
        static_cast<void>(create_feature_sink_set(
            invalid_specs, payload, 2U, 1U, 4U, 2U
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "feature bank accepted taps outside causal execution order");
}

void test_draft_cache_tree_rows_promote_and_rebuild_identically_on_metal() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const DraftCacheSpec spec {
        .layer_count = 1U,
        .stream_count = 1U,
        .key_width = 4U,
        .value_width = 2U,
        .maximum_depth = 23U,
        .maximum_active_width = 3U,
        .linear_row_capacity = 23U,
        .tree_row_capacity = 3U,
    };
    const auto build = [&]() {
        auto arena = create_draft_cache_arena(spec, 8U);
        expect(arena.layers()[0].linear_key.descriptor().shape.dimension(0) == 24U,
            "draft cache narrowed ExternalHmax+root linear capacity");
        const std::array<std::uint32_t, 6U> keys {
            packed_f16(1.0F, 2.0F), packed_f16(3.0F, 4.0F),
            packed_f16(5.0F, 6.0F), packed_f16(7.0F, 8.0F),
            packed_f16(9.0F, 10.0F), packed_f16(11.0F, 12.0F),
        };
        const std::array<std::uint32_t, 3U> values {
            packed_f16(13.0F, 14.0F), packed_f16(15.0F, 16.0F),
            packed_f16(17.0F, 18.0F),
        };
        MetalCommandContext initialize("draft-cache fixture initialization", true);
        initialize.write_u32s(arena.layers()[0].tree_key, 0U, keys);
        initialize.write_u32s(arena.layers()[0].tree_value, 0U, values);
        static_cast<void>(initialize.commit_and_wait());

        const DraftCacheBinding binding {
            .arena = &arena,
            .scratch = DraftCacheScratchKind::tree,
            .arena_epoch = 7U,
        };
        const std::array<std::uint32_t, 2U> source_rows {2U, 0U};
        const std::array<std::uint32_t, 2U> positions {0U, 1U};
        MetalCommandContext promote("draft-cache selected path", true);
        DraftCacheRuntimeAccess::encode_promotion(
            binding, promote, source_rows, positions
        );
        static_cast<void>(promote.commit_and_wait());
        DraftCacheRuntimeAccess::complete_promotion(binding, promote);
        return arena;
    };

    auto first = build();
    auto rebuilt = build();
    const auto first_key = first.copy_canonical_key(0U, 0U);
    const auto rebuilt_key = rebuilt.copy_canonical_key(0U, 0U);
    const auto first_value = first.copy_canonical_value(0U, 0U);
    const auto rebuilt_value = rebuilt.copy_canonical_value(0U, 0U);
    expect(std::equal(first_key.bytes().begin(), first_key.bytes().end(),
               rebuilt_key.bytes().begin())
        && std::equal(first_value.bytes().begin(), first_value.bytes().end(),
               rebuilt_value.bytes().begin()),
        "independent draft cache rebuild changed canonical K/V bytes");
    expect(read_f16_as_f32(first_key, std::array<std::uint64_t, 2U> {0U, 0U}) == 9.0F
        && read_f16_as_f32(first_key, std::array<std::uint64_t, 2U> {1U, 0U}) == 1.0F
        && first.state_view(0U).has_value()
        && first.state_view(0U)->draft_position == 1U,
        "draft cache promotion mapping or p/p+1 provenance changed");
    DraftCacheRuntimeAccess::truncate(first, 1U);
    expect(first.state_view(0U).has_value() && !first.state_view(1U).has_value(),
        "draft cache truncate retained a rejected suffix");
    DraftCacheRuntimeAccess::poison(DraftCacheBinding {
        .arena = &first,
        .scratch = DraftCacheScratchKind::tree,
        .arena_epoch = 7U,
    });
    first.reset();
    expect(first.poisoned(), "draft cache reset reused unknown-completion state");
}

const TestCase kTests[] = {
    {"five-tap feature banks preserve Metal parity, promotion, and rollback",
        test_five_tap_feature_bank_parity_promotion_and_rollback},
    {"five-tap feature banks reject malformed bounds and incomplete capture",
        test_five_tap_feature_bank_rejects_malformed_and_incomplete_capture},
    {"feature bank eager and lazy Metal projections match scalar equations",
        test_feature_bank_eager_and_lazy_metal_match_scalar_equations},
    {"feature bank fails closed without Metal",
        test_feature_bank_fails_closed_without_metal},
    {"hybrid feature bank preserves causal tap concatenation",
        test_hybrid_feature_bank_concatenates_taps_in_causal_order},
    {"raw target feature trace streams exact F16 taps",
        test_target_feature_trace_sink_streams_exact_f16_taps},
    {"draft cache tree promotion and independent rebuild preserve exact K/V",
        test_draft_cache_tree_rows_promote_and_rebuild_identically_on_metal},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> feature_bank_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
