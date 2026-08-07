#include "quick_benchmark_tests.hpp"

#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::tests {
namespace {

using benchmarks::QuickKernelMeasurementMode;
using benchmarks::quick_kernel_isolated_dispatch_count;
using benchmarks::quick_kernel_json_schema_version;
using benchmarks::quick_kernel_repetition_count;
using benchmarks::quick_kernel_sustained_dispatch_count;
using benchmarks::quick_kernel_warmup_count;
using benchmarks::quick_kernel_measurement_mode_name;
using benchmarks::run_quick_kernel_benchmark;
using benchmarks::summarize_kernel_measurement;
using benchmarks::summarize_values;

void test_quick_kernel_benchmark_contract_is_explicit() {
    expect(quick_kernel_repetition_count == 3U, "quick kernel repetition count mismatch");
    expect(quick_kernel_warmup_count == 1U, "quick kernel warm-up count mismatch");
    expect(quick_kernel_isolated_dispatch_count == 1U, "isolated dispatch count mismatch");
    expect(quick_kernel_sustained_dispatch_count == 42U, "sustained dispatch count mismatch");
    expect(quick_kernel_json_schema_version == 2U, "quick kernel JSON schema mismatch");
    expect(
        quick_kernel_measurement_mode_name(QuickKernelMeasurementMode::isolated) == "isolated",
        "isolated measurement mode name mismatch"
    );
    expect(
        quick_kernel_measurement_mode_name(QuickKernelMeasurementMode::sustained) == "sustained",
        "sustained measurement mode name mismatch"
    );
}

void test_quick_kernel_statistics_report_median_and_range() {
    const std::array values {4.0, 2.0, 3.0};
    const auto result = summarize_values(values);
    expect(result.median == 3.0, "quick kernel median mismatch");
    expect(result.minimum == 2.0, "quick kernel minimum mismatch");
    expect(result.maximum == 4.0, "quick kernel maximum mismatch");

    const std::array even_values {4.0, 2.0, 8.0, 6.0};
    const auto even = summarize_values(even_values);
    expect(even.median == 5.0, "quick kernel even median mismatch");
    expect(even.minimum == 2.0, "quick kernel even minimum mismatch");
    expect(even.maximum == 8.0, "quick kernel even maximum mismatch");
}

void test_quick_kernel_measurement_normalizes_sustained_commands() {
    const std::array<std::uint64_t, 3> raw {1260U, 420U, 840U};
    const auto result = summarize_kernel_measurement(
        QuickKernelMeasurementMode::sustained, 42U, 100U, raw
    );
    expect(result.mode == QuickKernelMeasurementMode::sustained, "measurement mode mismatch");
    expect(result.dispatches_per_command == 42U, "measurement dispatch count mismatch");
    expect(result.gpu_duration_per_dispatch_ns.median == 20.0, "normalized median mismatch");
    expect(result.gpu_duration_per_dispatch_ns.minimum == 10.0, "normalized minimum mismatch");
    expect(result.gpu_duration_per_dispatch_ns.maximum == 30.0, "normalized maximum mismatch");
    expect(result.effective_weight_bandwidth_gb_per_second == 5.0, "normalized bandwidth mismatch");
    expect(result.raw_command_gpu_duration_ns.size() == raw.size(), "raw duration count mismatch");
    expect(result.raw_command_gpu_duration_ns[0] == raw[0], "raw duration order mismatch");
}

void test_quick_kernel_measurement_rejects_invalid_inputs() {
    const std::array<std::uint64_t, 1> valid {100U};
    const std::array<std::uint64_t, 1> zero {0U};
    const std::span<const std::uint64_t> empty;

    auto rejected = false;
    try {
        static_cast<void>(summarize_kernel_measurement(
            QuickKernelMeasurementMode::isolated, 0U, 100U, valid
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "zero dispatch count should be rejected");

    rejected = false;
    try {
        static_cast<void>(summarize_kernel_measurement(
            QuickKernelMeasurementMode::isolated, 1U, 0U, valid
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "zero weight bytes should be rejected");

    rejected = false;
    try {
        static_cast<void>(summarize_kernel_measurement(
            QuickKernelMeasurementMode::isolated, 1U, 100U, empty
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "empty measurement samples should be rejected");

    rejected = false;
    try {
        static_cast<void>(summarize_kernel_measurement(
            QuickKernelMeasurementMode::isolated, 1U, 100U, zero
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "zero measurement duration should be rejected");

    rejected = false;
    try {
        static_cast<void>(summarize_kernel_measurement(
            QuickKernelMeasurementMode::sustained,
            2U,
            std::numeric_limits<std::uint64_t>::max(),
            valid
        ));
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    expect(rejected, "streamed byte overflow should be rejected");
}

void test_quick_kernel_statistics_reject_non_finite_values() {
    const std::array values {1.0, std::numeric_limits<double>::infinity()};
    bool rejected = false;
    try {
        static_cast<void>(summarize_values(values));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "non-finite benchmark value should be rejected");
}

void test_quick_kernel_benchmark_rejects_empty_model_path() {
    bool rejected = false;
    try {
        static_cast<void>(run_quick_kernel_benchmark(""));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "quick kernel benchmark should reject empty model path");
}

const std::array<TestCase, 6> kTests {{
    {"quick kernel benchmark contract is explicit", test_quick_kernel_benchmark_contract_is_explicit},
    {"quick kernel statistics report median and range", test_quick_kernel_statistics_report_median_and_range},
    {"quick kernel measurement normalizes sustained commands", test_quick_kernel_measurement_normalizes_sustained_commands},
    {"quick kernel measurement rejects invalid inputs", test_quick_kernel_measurement_rejects_invalid_inputs},
    {"quick kernel statistics reject non-finite values", test_quick_kernel_statistics_reject_non_finite_values},
    {"quick kernel benchmark rejects empty model path", test_quick_kernel_benchmark_rejects_empty_model_path},
}};

}  // namespace

std::span<const TestCase> quick_benchmark_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
