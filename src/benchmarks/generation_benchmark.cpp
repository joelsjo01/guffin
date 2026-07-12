#include "local_gguf_engine/benchmarks/generation_benchmark.hpp"

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace local_gguf_engine::benchmarks {

namespace {

[[nodiscard]] std::optional<double> finite_double(long double value) noexcept {
    const auto converted = static_cast<double>(value);
    return std::isfinite(converted) ? std::optional<double>(converted) : std::nullopt;
}

[[nodiscard]] std::uint64_t monotonic_now_ns(const engine::Gemma2MonotonicClock& clock) {
    if (clock) {
        return clock();
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count());
}

[[nodiscard]] std::uint64_t elapsed_ns(std::uint64_t start, std::uint64_t end) {
    if (end < start) {
        throw std::runtime_error("generation benchmark clock moved backwards");
    }
    return end - start;
}

template <typename ValueFunction>
[[nodiscard]] GenerationBenchmarkStatistics summarize_records(
    std::span<const GenerationBenchmarkRepetitionRecord> records,
    ValueFunction value
) {
    std::vector<std::optional<double>> values;
    values.reserve(records.size());
    for (const auto& record : records) {
        values.push_back(value(record));
    }
    return summarize_generation_benchmark_values(values);
}

[[nodiscard]] std::optional<double> optional_u64_as_double(
    std::optional<std::uint64_t> value
) noexcept {
    if (!value.has_value()) {
        return std::nullopt;
    }
    return finite_double(static_cast<long double>(*value));
}

[[nodiscard]] std::optional<double> u64_as_double(std::uint64_t value) noexcept {
    return finite_double(static_cast<long double>(value));
}

}  // namespace

std::string_view generation_benchmark_case_name(
    GenerationBenchmarkCase benchmark_case
) noexcept {
    switch (benchmark_case) {
        case GenerationBenchmarkCase::cold_first_turn:
            return "cold_first_turn";
        case GenerationBenchmarkCase::warm_first_turn:
            return "warm_first_turn";
        case GenerationBenchmarkCase::cached_second_turn:
            return "cached_second_turn";
        case GenerationBenchmarkCase::seeded_sampled_turn:
            return "seeded_sampled_turn";
    }
    return "unknown";
}

std::optional<double> generation_benchmark_tokens_per_second(
    std::uint32_t token_count,
    std::uint64_t duration_ns
) noexcept {
    if (token_count == 0U || duration_ns == 0U) {
        return std::nullopt;
    }
    return finite_double(
        static_cast<long double>(token_count) * 1.0e9L
            / static_cast<long double>(duration_ns)
    );
}

std::optional<double> generation_benchmark_prompt_tokens_per_second(
    const GenerationBenchmarkRepetitionRecord& record
) noexcept {
    return generation_benchmark_tokens_per_second(
        record.turn.generation.newly_evaluated_prompt_token_count,
        record.turn.generation.prefill_duration_ns
    );
}

std::optional<double> generation_benchmark_generation_tokens_per_second(
    const GenerationBenchmarkRepetitionRecord& record
) noexcept {
    return generation_benchmark_tokens_per_second(
        record.turn.generation.decode_token_count,
        record.turn.generation.decode_duration_ns
    );
}

GenerationBenchmarkStatistics summarize_generation_benchmark_values(
    std::span<const std::optional<double>> values
) {
    std::vector<double> finite_values;
    finite_values.reserve(values.size());
    for (const auto value : values) {
        if (value.has_value() && std::isfinite(*value)) {
            finite_values.push_back(*value);
        }
    }
    if (finite_values.empty()) {
        return {};
    }

    std::sort(finite_values.begin(), finite_values.end());
    long double sum = 0.0L;
    for (const auto value : finite_values) {
        sum += static_cast<long double>(value);
    }
    const auto mean = sum / static_cast<long double>(finite_values.size());
    const auto middle = finite_values.size() / 2U;
    const auto median = finite_values.size() % 2U == 0U
        ? (static_cast<long double>(finite_values[middle - 1U])
            + static_cast<long double>(finite_values[middle])) / 2.0L
        : static_cast<long double>(finite_values[middle]);

    long double sample_variance = 0.0L;
    if (finite_values.size() > 1U) {
        for (const auto value : finite_values) {
            const auto difference = static_cast<long double>(value) - mean;
            sample_variance += difference * difference;
        }
        sample_variance /= static_cast<long double>(finite_values.size() - 1U);
    }

    return GenerationBenchmarkStatistics {
        .sample_count = finite_values.size(),
        .mean = finite_double(mean),
        .median = finite_double(median),
        .minimum = finite_values.front(),
        .maximum = finite_values.back(),
        // A single observation has no spread in this benchmark contract.
        .sample_standard_deviation = finite_double(std::sqrt(sample_variance)),
    };
}

GenerationBenchmarkSummaryRecord summarize_generation_benchmark_case(
    std::span<const GenerationBenchmarkRepetitionRecord> records,
    GenerationBenchmarkCase benchmark_case
) {
    GenerationBenchmarkSummaryRecord summary {
        .benchmark_case = benchmark_case,
    };
    for (const auto& record : records) {
        if (record.benchmark_case == benchmark_case && !record.warmup) {
            summary.measured_repetitions.push_back(record);
        }
    }
    const auto measured = std::span<const GenerationBenchmarkRepetitionRecord>(
        summary.measured_repetitions
    );
    summary.model_load_duration_ns = summarize_records(measured, [](const auto& record) {
        return optional_u64_as_double(record.setup.model_load_duration_ns);
    });
    summary.plan_creation_duration_ns = summarize_records(measured, [](const auto& record) {
        return optional_u64_as_double(record.setup.plan_creation_duration_ns);
    });
    summary.prefill_duration_ns = summarize_records(measured, [](const auto& record) {
        return u64_as_double(record.turn.generation.prefill_duration_ns);
    });
    summary.prompt_tokens_per_second = summarize_records(measured, [](const auto& record) {
        return generation_benchmark_prompt_tokens_per_second(record);
    });
    summary.decode_duration_ns = summarize_records(measured, [](const auto& record) {
        return u64_as_double(record.turn.generation.decode_duration_ns);
    });
    summary.generation_tokens_per_second = summarize_records(measured, [](const auto& record) {
        return generation_benchmark_generation_tokens_per_second(record);
    });
    summary.runtime_ttft_ns = summarize_records(measured, [](const auto& record) {
        return u64_as_double(record.turn.generation.runtime_ttft_ns);
    });
    summary.end_to_end_ttft_ns = summarize_records(measured, [](const auto& record) {
        return optional_u64_as_double(record.turn.end_to_end_ttft_ns);
    });
    summary.total_turn_duration_ns = summarize_records(measured, [](const auto& record) {
        return u64_as_double(record.turn.total_turn_duration_ns);
    });
    return summary;
}

GenerationBenchmarkRunResult run_generation_benchmark(
    const GenerationBenchmarkRunnerOptions& options
) {
    if (options.model_path.empty()) {
        throw std::invalid_argument("generation benchmark model path must not be empty");
    }
    if (options.prompt.empty()) {
        throw std::invalid_argument("generation benchmark prompt must not be empty");
    }
    if (options.max_new_tokens == 0U) {
        throw std::invalid_argument("generation benchmark max_new_tokens must be positive");
    }
    if (options.repetition_count == 0U) {
        throw std::invalid_argument("generation benchmark repetition_count must be positive");
    }
    if (options.backend != runtime::BackendKind::metal) {
        throw std::invalid_argument("generation benchmark Gemma 2 runner requires Metal backend");
    }
    if (options.sampled_sampler.temperature <= 0.0F) {
        throw std::invalid_argument("generation benchmark sampled case requires positive temperature");
    }
    static_cast<void>(engine::TokenSampler(options.sampled_sampler));

    const auto load_started_ns = monotonic_now_ns(options.clock);
    auto file_model = model_format::read_file_backed_gguf_model(options.model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    auto tokenizer = tokenizer::load_tokenizer_from_gguf_metadata(file_model.view().directory.metadata);
    const auto load_completed_ns = monotonic_now_ns(options.clock);

    const auto plan_started_ns = monotonic_now_ns(options.clock);
    auto plan = engine::create_gemma2_metal_model_plan(model);
    const auto plan_completed_ns = monotonic_now_ns(options.clock);

    GenerationBenchmarkRunResult result;
    result.setup = GenerationBenchmarkSetupRecord {
        .model_label = options.model_path,
        .backend = options.backend,
        .model_load_duration_ns = elapsed_ns(load_started_ns, load_completed_ns),
        .plan_creation_duration_ns = elapsed_ns(plan_started_ns, plan_completed_ns),
        .resident_weight_byte_count = plan.uploaded_weight_byte_count(),
        .resident_kv_cache_byte_count = plan.kv_cache_byte_count(),
        .resident_scratch_byte_count = plan.scratch_byte_count(),
    };

    auto session = engine::Gemma2ChatSession(
        engine::Gemma2ChatSessionState {
            .file_model = std::move(file_model),
            .model = std::move(model),
            .tokenizer = std::move(tokenizer),
            .plan = std::move(plan),
        },
        engine::Gemma2ChatSessionOptions {
            .max_new_tokens = options.max_new_tokens,
            .backend = options.backend,
            .sampler = engine::TokenSamplerOptions {},
            .clock = options.clock,
        }
    );

    const auto record_turn = [&](
        GenerationBenchmarkCase benchmark_case,
        bool warmup,
        std::size_t repetition_index,
        const engine::TokenSamplerOptions& sampler,
        engine::Gemma2ChatTurnResult turn
    ) {
        result.repetitions.push_back(GenerationBenchmarkRepetitionRecord {
            .benchmark_case = benchmark_case,
            .warmup = warmup,
            .repetition_index = repetition_index,
            .setup = result.setup,
            .turn = turn.metrics,
            .generated_token_ids = std::move(turn.generated_token_ids),
            .stop_token_id = turn.stop_token_id,
            .sampler = sampler,
        });
    };
    const auto generate_visible_turn = [&](std::string_view prompt) {
        return session.generate_user_turn(prompt, [](std::string_view) {});
    };

    const engine::TokenSamplerOptions greedy_sampler;
    record_turn(
        GenerationBenchmarkCase::cold_first_turn,
        false,
        0U,
        greedy_sampler,
        generate_visible_turn(options.prompt)
    );

    const auto run_repeated_case = [&](
        GenerationBenchmarkCase benchmark_case,
        const engine::TokenSamplerOptions& sampler,
        const auto& run_one
    ) {
        session.set_sampler_options(sampler);
        for (std::size_t warmup_index = 0; warmup_index < options.warmup_count; ++warmup_index) {
            session.clear();
            record_turn(benchmark_case, true, warmup_index, sampler, run_one());
        }
        for (std::size_t repetition_index = 0; repetition_index < options.repetition_count; ++repetition_index) {
            session.clear();
            record_turn(benchmark_case, false, repetition_index, sampler, run_one());
        }
    };

    run_repeated_case(
        GenerationBenchmarkCase::warm_first_turn,
        greedy_sampler,
        [&]() { return generate_visible_turn(options.prompt); }
    );
    const auto cached_second_prompt = options.cached_second_prompt.empty()
        ? std::string_view(options.prompt)
        : std::string_view(options.cached_second_prompt);
    run_repeated_case(
        GenerationBenchmarkCase::cached_second_turn,
        greedy_sampler,
        [&]() {
            static_cast<void>(generate_visible_turn(options.prompt));
            return generate_visible_turn(cached_second_prompt);
        }
    );
    run_repeated_case(
        GenerationBenchmarkCase::seeded_sampled_turn,
        options.sampled_sampler,
        [&]() { return generate_visible_turn(options.prompt); }
    );

    if (session.plan_revision() != 0U) {
        throw std::runtime_error("generation benchmark rebuilt the resident Metal plan");
    }
    const std::array<GenerationBenchmarkCase, 4> cases {
        GenerationBenchmarkCase::cold_first_turn,
        GenerationBenchmarkCase::warm_first_turn,
        GenerationBenchmarkCase::cached_second_turn,
        GenerationBenchmarkCase::seeded_sampled_turn,
    };
    result.summaries.reserve(cases.size());
    for (const auto benchmark_case : cases) {
        result.summaries.push_back(summarize_generation_benchmark_case(
            result.repetitions,
            benchmark_case
        ));
    }
    return result;
}

}  // namespace local_gguf_engine::benchmarks
