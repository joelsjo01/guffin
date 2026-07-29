#include "conversation_benchmark_tests.hpp"

#include "local_gguf_engine/benchmarks/conversation_benchmark.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::tests {
namespace {

using namespace local_gguf_engine::benchmarks;

void test_conversation_scenario_is_fixed_and_versioned() {
    const auto& scenario = primary_conversation_scenario();
    expect(conversation_benchmark_schema_version == 4U, "conversation schema version mismatch");
    expect(scenario.version == 1U, "conversation scenario version mismatch");
    expect(scenario.turns.size() == 4U, "conversation turn count mismatch");
    expect(scenario.turns[0].user_text == "Hello, how are you?", "conversation first turn mismatch");
    expect(scenario.turns[2].generated_token_limit > scenario.turns[0].generated_token_limit,
        "long conversation turn should have a larger output budget");
}

void test_conversation_statistics_include_mad_and_missing_values() {
    const std::array<std::optional<double>, 5> values {1.0, 2.0, 3.0, 4.0, 100.0};
    const auto stats = summarize_benchmark_values(values);
    expect(stats.sample_count == 5U, "conversation statistics count mismatch");
    expect(std::abs(*stats.median - 3.0) < 1.0e-12, "conversation median mismatch");
    expect(std::abs(*stats.median_absolute_deviation - 1.0) < 1.0e-12, "conversation MAD mismatch");
    expect(*stats.minimum == 1.0 && *stats.maximum == 100.0, "conversation range mismatch");
    const std::array<std::optional<double>, 2> missing {std::nullopt, std::nullopt};
    expect(summarize_benchmark_values(missing).sample_count == 0U,
        "missing conversation values should remain unavailable");
}

void test_conversation_rates_use_explicit_zero_unavailable_contract() {
    expect(!benchmark_rate(0U, 10U).has_value(), "zero items should have no rate");
    expect(!benchmark_rate(10U, 0U).has_value(), "zero duration should have no rate");
    expect(std::abs(*benchmark_rate(5U, 1'000'000'000U) - 5.0) < 1.0e-12,
        "conversation rate mismatch");
}

void test_conversation_runner_rejects_invalid_options_before_loading() {
    const auto rejects = [](ConversationBenchmarkRunnerOptions options) {
        try {
            static_cast<void>(run_conversation_benchmark(options));
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(rejects({}), "empty conversation model path should be rejected");
    expect(rejects({.model_path = "missing.gguf", .repetition_count = 0U}),
        "zero conversation repetitions should be rejected");
    expect(rejects({.model_path = "missing.gguf", .backend = runtime::BackendKind::cpu}),
        "CPU conversation benchmark should be rejected");
    expect(rejects({
        .model_path = "missing.gguf",
        .sampler = engine::TokenSamplerOptions {.temperature = 0.8F},
    }), "sampled primary conversation should be rejected");
}

const TestCase kTests[] = {
    {"conversation benchmark scenario is fixed and versioned", test_conversation_scenario_is_fixed_and_versioned},
    {"conversation benchmark statistics include MAD and missing values", test_conversation_statistics_include_mad_and_missing_values},
    {"conversation benchmark rates use explicit zero unavailable contract", test_conversation_rates_use_explicit_zero_unavailable_contract},
    {"conversation benchmark runner rejects invalid options before loading", test_conversation_runner_rejects_invalid_options_before_loading},
};

}  // namespace

std::span<const TestCase> conversation_benchmark_test_cases() { return kTests; }
}  // namespace local_gguf_engine::tests
