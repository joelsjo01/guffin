#include "quick_benchmark_tests.hpp"

#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include <array>
#include <stdexcept>

namespace local_gguf_engine::tests {
namespace {

using benchmarks::quick_kernel_repetition_count;
using benchmarks::run_quick_kernel_benchmark;
using benchmarks::summarize_values;

void test_quick_kernel_benchmark_contract_is_small() {
    expect(quick_kernel_repetition_count == 3U, "quick kernel repetition count mismatch");
}

void test_quick_kernel_statistics_report_median_and_range() {
    const std::array values {4.0, 2.0, 3.0};
    const auto result = summarize_values(values);
    expect(result.median == 3.0, "quick kernel median mismatch");
    expect(result.minimum == 2.0, "quick kernel minimum mismatch");
    expect(result.maximum == 4.0, "quick kernel maximum mismatch");
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

const std::array<TestCase, 3> kTests {{
    {"quick kernel benchmark contract is small", test_quick_kernel_benchmark_contract_is_small},
    {"quick kernel statistics report median and range", test_quick_kernel_statistics_report_median_and_range},
    {"quick kernel benchmark rejects empty model path", test_quick_kernel_benchmark_rejects_empty_model_path},
}};

}  // namespace

std::span<const TestCase> quick_benchmark_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
