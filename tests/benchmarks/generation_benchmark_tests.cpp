#include "generation_benchmark_tests.hpp"

#include "local_gguf_engine/benchmarks/generation_benchmark.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using local_gguf_engine::benchmarks::generation_benchmark_case_name;
using local_gguf_engine::benchmarks::generation_benchmark_prompt_tokens_per_second;
using local_gguf_engine::benchmarks::generation_benchmark_schema_version;
using local_gguf_engine::benchmarks::GenerationBenchmarkCase;
using local_gguf_engine::benchmarks::GenerationBenchmarkRepetitionRecord;
using local_gguf_engine::benchmarks::GenerationBenchmarkRunnerOptions;
using local_gguf_engine::benchmarks::run_generation_benchmark;
using local_gguf_engine::benchmarks::summarize_generation_benchmark_case;
using local_gguf_engine::benchmarks::summarize_generation_benchmark_values;
using local_gguf_engine::engine::Gemma2TokenSelectionMode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

void expect_close(double actual, double expected, double tolerance, std::string_view message) {
    expect(std::abs(actual - expected) <= tolerance, message);
}

void test_generation_benchmark_schema_and_case_names_are_stable() {
    expect(generation_benchmark_schema_version == 1U, "generation benchmark schema version mismatch");
    const std::array<std::pair<GenerationBenchmarkCase, std::string_view>, 4> cases {{
        {GenerationBenchmarkCase::cold_first_turn, "cold_first_turn"},
        {GenerationBenchmarkCase::warm_first_turn, "warm_first_turn"},
        {GenerationBenchmarkCase::cached_second_turn, "cached_second_turn"},
        {GenerationBenchmarkCase::seeded_sampled_turn, "seeded_sampled_turn"},
    }};
    for (const auto& [benchmark_case, expected_name] : cases) {
        expect(generation_benchmark_case_name(benchmark_case) == expected_name,
            "generation benchmark case name mismatch");
    }
}

void test_generation_benchmark_statistics_cover_odd_even_and_sample_deviation() {
    const std::array<std::optional<double>, 3> odd {1.0, 3.0, 5.0};
    const auto odd_summary = summarize_generation_benchmark_values(odd);
    expect(odd_summary.sample_count == 3U, "odd benchmark sample count mismatch");
    expect_close(*odd_summary.mean, 3.0, 1.0e-12, "odd benchmark mean mismatch");
    expect_close(*odd_summary.median, 3.0, 1.0e-12, "odd benchmark median mismatch");
    expect_close(*odd_summary.minimum, 1.0, 1.0e-12, "odd benchmark minimum mismatch");
    expect_close(*odd_summary.maximum, 5.0, 1.0e-12, "odd benchmark maximum mismatch");
    expect_close(*odd_summary.sample_standard_deviation, 2.0, 1.0e-12,
        "odd benchmark sample deviation mismatch");

    const std::array<std::optional<double>, 4> even {1.0, 2.0, 3.0, 4.0};
    const auto even_summary = summarize_generation_benchmark_values(even);
    expect_close(*even_summary.mean, 2.5, 1.0e-12, "even benchmark mean mismatch");
    expect_close(*even_summary.median, 2.5, 1.0e-12, "even benchmark median mismatch");
    expect_close(*even_summary.sample_standard_deviation, std::sqrt(5.0 / 3.0), 1.0e-12,
        "even benchmark sample deviation mismatch");
}

void test_generation_benchmark_statistics_handle_single_repeated_and_unavailable_values() {
    const std::array<std::optional<double>, 1> single {7.0};
    const auto single_summary = summarize_generation_benchmark_values(single);
    expect(single_summary.sample_count == 1U, "single benchmark sample count mismatch");
    expect(*single_summary.sample_standard_deviation == 0.0, "single benchmark deviation should be zero");

    const std::array<std::optional<double>, 3> repeated {4.0, 4.0, 4.0};
    const auto repeated_summary = summarize_generation_benchmark_values(repeated);
    expect(*repeated_summary.mean == 4.0 && *repeated_summary.median == 4.0,
        "repeated benchmark center mismatch");
    expect(*repeated_summary.sample_standard_deviation == 0.0,
        "repeated benchmark deviation should be zero");

    const std::array<std::optional<double>, 3> unavailable {
        std::nullopt,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
    };
    const auto unavailable_summary = summarize_generation_benchmark_values(unavailable);
    expect(unavailable_summary.sample_count == 0U, "unavailable benchmark sample count mismatch");
    expect(!unavailable_summary.mean.has_value() && !unavailable_summary.median.has_value(),
        "unavailable benchmark center should be absent");
    expect(!unavailable_summary.sample_standard_deviation.has_value(),
        "unavailable benchmark deviation should be absent");
}

GenerationBenchmarkRepetitionRecord benchmark_record(
    std::size_t repetition_index,
    bool warmup,
    GenerationBenchmarkCase benchmark_case,
    std::uint64_t prefill_duration_ns,
    std::optional<std::uint64_t> end_to_end_ttft_ns
) {
    GenerationBenchmarkRepetitionRecord record {
        .benchmark_case = benchmark_case,
        .warmup = warmup,
        .repetition_index = repetition_index,
    };
    record.setup.model_label = "tiny-gemma2";
    record.setup.backend = BackendKind::metal;
    record.setup.model_load_duration_ns = 50U;
    record.setup.plan_creation_duration_ns = 70U;
    record.turn.generation.prefill_duration_ns = prefill_duration_ns;
    record.turn.generation.newly_evaluated_prompt_token_count = 10U;
    record.turn.generation.decode_duration_ns = 200U;
    record.turn.generation.decode_token_count = 20U;
    record.turn.generation.runtime_ttft_ns = prefill_duration_ns + 10U;
    record.turn.end_to_end_ttft_ns = end_to_end_ttft_ns;
    record.turn.total_turn_duration_ns = prefill_duration_ns + 500U;
    return record;
}

void test_generation_benchmark_summary_excludes_warmups_and_preserves_measured_order() {
    std::vector<GenerationBenchmarkRepetitionRecord> records;
    records.push_back(benchmark_record(0U, true, GenerationBenchmarkCase::warm_first_turn, 900U, 950U));
    records.push_back(benchmark_record(2U, false, GenerationBenchmarkCase::warm_first_turn, 100U, 150U));
    records.push_back(benchmark_record(3U, false, GenerationBenchmarkCase::cached_second_turn, 800U, 850U));
    records.push_back(benchmark_record(4U, false, GenerationBenchmarkCase::warm_first_turn, 300U, std::nullopt));

    const auto summary = summarize_generation_benchmark_case(records, GenerationBenchmarkCase::warm_first_turn);
    expect(summary.measured_repetitions.size() == 2U, "benchmark measured repetition count mismatch");
    expect(summary.measured_repetitions[0].repetition_index == 2U,
        "benchmark first measured repetition order mismatch");
    expect(summary.measured_repetitions[1].repetition_index == 4U,
        "benchmark second measured repetition order mismatch");
    expect(summary.prefill_duration_ns.sample_count == 2U, "benchmark prefill sample count mismatch");
    expect_close(*summary.prefill_duration_ns.mean, 200.0, 1.0e-12, "benchmark prefill mean mismatch");
    expect(summary.end_to_end_ttft_ns.sample_count == 1U,
        "benchmark absent TTFT should not become zero");
    expect(summary.prompt_tokens_per_second.sample_count == 2U,
        "benchmark prompt rate sample count mismatch");
    expect(summary.generation_tokens_per_second.sample_count == 2U,
        "benchmark generation rate sample count mismatch");
}

void test_generation_benchmark_record_carries_complete_identity_and_derived_rates() {
    auto record = benchmark_record(5U, false, GenerationBenchmarkCase::seeded_sampled_turn, 2'000'000'000U, 3U);
    record.turn.selection_mode = Gemma2TokenSelectionMode::sampled;
    record.sampler.temperature = 0.8F;
    record.sampler.seed = 42U;
    record.setup.resident_weight_byte_count = 1000U;
    record.setup.resident_kv_cache_byte_count = 2000U;
    record.setup.resident_scratch_byte_count = 3000U;

    expect(record.schema_version == generation_benchmark_schema_version, "benchmark record schema mismatch");
    expect(record.setup.schema_version == generation_benchmark_schema_version, "benchmark setup schema mismatch");
    expect(record.setup.model_label == "tiny-gemma2" && record.setup.backend == BackendKind::metal,
        "benchmark model/backend identity mismatch");
    expect(record.turn.selection_mode == Gemma2TokenSelectionMode::sampled && record.sampler.seed == 42U,
        "benchmark sampler identity mismatch");
    expect(record.setup.resident_weight_byte_count == 1000U
        && record.setup.resident_kv_cache_byte_count == 2000U
        && record.setup.resident_scratch_byte_count == 3000U,
        "benchmark resident memory facts mismatch");
    const auto prompt_rate = generation_benchmark_prompt_tokens_per_second(record);
    expect(prompt_rate.has_value(), "benchmark prompt rate should be available");
    expect_close(*prompt_rate, 5.0, 1.0e-12, "benchmark prompt rate mismatch");
}

void test_generation_benchmark_runner_rejects_invalid_options_before_loading() {
    const auto rejects = [](const GenerationBenchmarkRunnerOptions& options) {
        bool rejected = false;
        try {
            static_cast<void>(run_generation_benchmark(options));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        return rejected;
    };

    expect(rejects(GenerationBenchmarkRunnerOptions {.prompt = "Hello"}),
        "benchmark runner should reject empty model path");
    expect(rejects(GenerationBenchmarkRunnerOptions {
        .model_path = "/not/read",
        .prompt = "Hello",
        .repetition_count = 0U,
    }), "benchmark runner should reject zero repetitions before loading");
    expect(rejects(GenerationBenchmarkRunnerOptions {
        .model_path = "/not/read",
        .prompt = "Hello",
        .backend = BackendKind::cpu,
    }), "benchmark runner should reject CPU backend before loading");
    expect(rejects(GenerationBenchmarkRunnerOptions {
        .model_path = "/not/read",
        .prompt = "Hello",
        .sampled_sampler = local_gguf_engine::engine::TokenSamplerOptions {.temperature = 0.0F},
    }), "benchmark runner should reject greedy sampled-case settings before loading");
}

const TestCase kGenerationBenchmarkTests[] = {
    {"generation benchmark schema and case names are stable", test_generation_benchmark_schema_and_case_names_are_stable},
    {"generation benchmark statistics cover odd even and sample deviation", test_generation_benchmark_statistics_cover_odd_even_and_sample_deviation},
    {"generation benchmark statistics handle single repeated and unavailable values", test_generation_benchmark_statistics_handle_single_repeated_and_unavailable_values},
    {"generation benchmark summary excludes warmups and preserves measured order", test_generation_benchmark_summary_excludes_warmups_and_preserves_measured_order},
    {"generation benchmark record carries complete identity and derived rates", test_generation_benchmark_record_carries_complete_identity_and_derived_rates},
    {"generation benchmark runner rejects invalid options before loading", test_generation_benchmark_runner_rejects_invalid_options_before_loading},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> generation_benchmark_test_cases() {
    return kGenerationBenchmarkTests;
}

}  // namespace local_gguf_engine::tests
