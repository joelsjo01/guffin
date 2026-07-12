#pragma once

#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::benchmarks {

inline constexpr std::uint32_t generation_benchmark_schema_version = 1U;

enum class GenerationBenchmarkCase {
    cold_first_turn,
    warm_first_turn,
    cached_second_turn,
    seeded_sampled_turn,
};

[[nodiscard]] std::string_view generation_benchmark_case_name(
    GenerationBenchmarkCase benchmark_case
) noexcept;

struct GenerationBenchmarkSetupRecord {
    std::uint32_t schema_version = generation_benchmark_schema_version;
    std::string model_label;
    runtime::BackendKind backend = runtime::BackendKind::metal;
    std::optional<std::uint64_t> model_load_duration_ns;
    std::optional<std::uint64_t> plan_creation_duration_ns;
    std::size_t resident_weight_byte_count = 0;
    std::size_t resident_kv_cache_byte_count = 0;
    std::size_t resident_scratch_byte_count = 0;
};

struct GenerationBenchmarkRepetitionRecord {
    std::uint32_t schema_version = generation_benchmark_schema_version;
    GenerationBenchmarkCase benchmark_case = GenerationBenchmarkCase::cold_first_turn;
    bool warmup = false;
    std::size_t repetition_index = 0;
    GenerationBenchmarkSetupRecord setup;
    engine::Gemma2ChatTurnMetrics turn;
    std::vector<std::uint32_t> generated_token_ids;
    std::optional<std::uint32_t> stop_token_id;
    engine::TokenSamplerOptions sampler;
};

struct GenerationBenchmarkStatistics {
    std::size_t sample_count = 0;
    std::optional<double> mean;
    std::optional<double> median;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> sample_standard_deviation;
};

struct GenerationBenchmarkSummaryRecord {
    std::uint32_t schema_version = generation_benchmark_schema_version;
    GenerationBenchmarkCase benchmark_case = GenerationBenchmarkCase::cold_first_turn;
    std::vector<GenerationBenchmarkRepetitionRecord> measured_repetitions;
    GenerationBenchmarkStatistics model_load_duration_ns;
    GenerationBenchmarkStatistics plan_creation_duration_ns;
    GenerationBenchmarkStatistics prefill_duration_ns;
    GenerationBenchmarkStatistics prompt_tokens_per_second;
    GenerationBenchmarkStatistics decode_duration_ns;
    GenerationBenchmarkStatistics generation_tokens_per_second;
    GenerationBenchmarkStatistics runtime_ttft_ns;
    GenerationBenchmarkStatistics end_to_end_ttft_ns;
    GenerationBenchmarkStatistics total_turn_duration_ns;
};

struct GenerationBenchmarkRunnerOptions {
    std::string model_path;
    std::string prompt;
    std::string cached_second_prompt;
    std::size_t max_new_tokens = 128U;
    std::size_t warmup_count = 1U;
    std::size_t repetition_count = 5U;
    runtime::BackendKind backend = runtime::BackendKind::metal;
    engine::TokenSamplerOptions sampled_sampler {
        .temperature = 0.8F,
        .top_k = 40U,
        .top_p = 0.95F,
        .repetition_penalty = 1.0F,
        .seed = 42U,
    };
    engine::Gemma2MonotonicClock clock;
};

struct GenerationBenchmarkRunResult {
    GenerationBenchmarkSetupRecord setup;
    std::vector<GenerationBenchmarkRepetitionRecord> repetitions;
    std::vector<GenerationBenchmarkSummaryRecord> summaries;
};

[[nodiscard]] std::optional<double> generation_benchmark_tokens_per_second(
    std::uint32_t token_count,
    std::uint64_t duration_ns
) noexcept;

[[nodiscard]] std::optional<double> generation_benchmark_prompt_tokens_per_second(
    const GenerationBenchmarkRepetitionRecord& record
) noexcept;

[[nodiscard]] std::optional<double> generation_benchmark_generation_tokens_per_second(
    const GenerationBenchmarkRepetitionRecord& record
) noexcept;

[[nodiscard]] GenerationBenchmarkStatistics summarize_generation_benchmark_values(
    std::span<const std::optional<double>> values
);

[[nodiscard]] GenerationBenchmarkSummaryRecord summarize_generation_benchmark_case(
    std::span<const GenerationBenchmarkRepetitionRecord> records,
    GenerationBenchmarkCase benchmark_case
);

[[nodiscard]] GenerationBenchmarkRunResult run_generation_benchmark(
    const GenerationBenchmarkRunnerOptions& options
);

}  // namespace local_gguf_engine::benchmarks
