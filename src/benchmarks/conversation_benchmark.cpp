#include "local_gguf_engine/benchmarks/conversation_benchmark.hpp"

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include "../tensor/backends/metal_backend.hpp"

#include <mach/mach.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace local_gguf_engine::benchmarks {
namespace {

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

[[nodiscard]] std::uint64_t now_ns(const engine::Gemma2MonotonicClock& clock) {
    if (clock) return clock();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count());
}

[[nodiscard]] std::uint64_t elapsed(std::uint64_t start, std::uint64_t finish) {
    if (finish < start) throw std::runtime_error("conversation benchmark clock moved backwards");
    return finish - start;
}

[[nodiscard]] std::string uname_field(bool machine) {
    utsname value {};
    if (uname(&value) != 0) return {};
    return machine ? value.machine : std::string(value.sysname) + " " + value.release;
}

[[nodiscard]] std::string hardware_name() {
    std::size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) != 0 || size == 0U) {
        return uname_field(true);
    }
    std::string value(size, '\0');
    if (sysctlbyname("machdep.cpu.brand_string", value.data(), &size, nullptr, 0) != 0) {
        return uname_field(true);
    }
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

[[nodiscard]] std::optional<std::uint64_t> physical_footprint_bytes() noexcept {
    task_vm_info_data_t info {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count)
        != KERN_SUCCESS) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(info.phys_footprint);
}

[[nodiscard]] std::optional<std::uint64_t> physical_footprint_peak_bytes() noexcept {
    task_vm_info_data_t info {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count)
        != KERN_SUCCESS || info.ledger_phys_footprint_peak < 0) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(info.ledger_phys_footprint_peak);
}

[[nodiscard]] std::optional<double> as_double(std::optional<std::uint64_t> value) {
    return value.has_value() ? std::optional<double>(static_cast<double>(*value)) : std::nullopt;
}

[[nodiscard]] std::string model_identity(const std::string& path) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    const auto resolved = error ? std::filesystem::path(path) : canonical;
    error.clear();
    const auto size = std::filesystem::file_size(resolved, error);
    if (error) throw std::runtime_error("conversation benchmark could not identify model file");
    struct stat file_info {};
    if (::stat(resolved.c_str(), &file_info) != 0) {
        throw std::runtime_error("conversation benchmark could not identify model timestamp");
    }
    std::ostringstream identity;
    const auto modified_ns = static_cast<std::int64_t>(file_info.st_mtimespec.tv_sec) * 1'000'000'000LL
        + static_cast<std::int64_t>(file_info.st_mtimespec.tv_nsec);
    identity << resolved.string() << ":size=" << size << ":mtime=" << modified_ns;
    return identity.str();
}

template <typename Projection>
ConversationBenchmarkStatistics summarize_turns(
    std::span<const ConversationBenchmarkTurnRecord> records,
    std::size_t turn_index,
    Projection projection
) {
    std::vector<std::optional<double>> values;
    for (const auto& record : records) {
        if (!record.warmup && record.turn_index == turn_index) values.push_back(projection(record));
    }
    return summarize_benchmark_values(values);
}

}  // namespace

const ConversationBenchmarkScenario& primary_conversation_scenario() {
    static const ConversationBenchmarkScenario scenario {
        .name = "product_conversation_v1",
        .turns = {
            {"Hello, how are you?", 32U},
            {"Okay, that is good.", 32U},
            {"Tell me about Gothenburg in about 120 words.", 160U},
            {"What was the second place you mentioned, and why might a visitor choose it?", 128U},
        },
    };
    return scenario;
}

std::optional<double> benchmark_rate(std::uint64_t item_count, std::uint64_t duration_ns) noexcept {
    if (item_count == 0U || duration_ns == 0U) return std::nullopt;
    const auto rate = static_cast<double>(item_count) * 1.0e9 / static_cast<double>(duration_ns);
    return std::isfinite(rate) ? std::optional<double>(rate) : std::nullopt;
}

ConversationBenchmarkStatistics summarize_benchmark_values(
    std::span<const std::optional<double>> values
) {
    std::vector<double> finite;
    for (const auto value : values) if (value.has_value() && std::isfinite(*value)) finite.push_back(*value);
    if (finite.empty()) return {};
    std::sort(finite.begin(), finite.end());
    const auto median_of_sorted = [](std::span<const double> sorted) {
        const auto middle = sorted.size() / 2U;
        return sorted.size() % 2U == 0U
            ? (sorted[middle - 1U] + sorted[middle]) / 2.0
            : sorted[middle];
    };
    const auto median = median_of_sorted(finite);
    const auto mean = std::accumulate(finite.begin(), finite.end(), 0.0) / static_cast<double>(finite.size());
    double variance = 0.0;
    if (finite.size() > 1U) {
        for (const auto value : finite) variance += (value - mean) * (value - mean);
        variance /= static_cast<double>(finite.size() - 1U);
    }
    std::vector<double> deviations;
    deviations.reserve(finite.size());
    for (const auto value : finite) deviations.push_back(std::abs(value - median));
    std::sort(deviations.begin(), deviations.end());
    return {
        .sample_count = finite.size(),
        .median = median,
        .minimum = finite.front(),
        .maximum = finite.back(),
        .sample_standard_deviation = std::sqrt(variance),
        .median_absolute_deviation = median_of_sorted(deviations),
    };
}

ConversationBenchmarkRunResult run_conversation_benchmark(
    const ConversationBenchmarkRunnerOptions& options
) {
    if (options.model_path.empty()) throw std::invalid_argument("conversation benchmark model path must not be empty");
    if (options.repetition_count == 0U) throw std::invalid_argument("conversation benchmark repetition count must be positive");
    if (options.backend != runtime::BackendKind::metal) {
        throw std::invalid_argument("conversation benchmark Gemma 2 runner requires Metal backend");
    }
    if (!engine::TokenSampler(options.sampler).is_greedy()) {
        throw std::invalid_argument("primary conversation benchmark requires greedy sampling");
    }
    if (!options.frontend_stream_callback || options.frontend_name.empty()) {
        throw std::invalid_argument(
            "conversation benchmark requires an identified synchronous frontend stream callback"
        );
    }

    ConversationBenchmarkIdentity identity {
        .engine_build = options.engine_build.empty() ? LOCAL_GGUF_ENGINE_BUILD : options.engine_build,
        .model_label = options.model_path,
        .model_identity = model_identity(options.model_path),
        .backend = options.backend,
        .operating_system = uname_field(false),
        .hardware = hardware_name(),
        .frontend = options.frontend_name,
    };
    const auto load_start = now_ns(options.clock);
    auto file_model = model_format::read_file_backed_gguf_model(options.model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    auto tokenizer = tokenizer::load_tokenizer_from_gguf_metadata(file_model.view().directory.metadata);
    const auto load_finish = now_ns(options.clock);
    const auto plan_start = now_ns(options.clock);
    auto plan = engine::create_gemma2_metal_model_plan(model);
    const auto plan_finish = now_ns(options.clock);
    const auto sample_memory = [&]() {
        if (options.memory_sampler) return options.memory_sampler();
        return ConversationBenchmarkMemorySnapshot {
            .metal_current_allocated_bytes = tensor::backends::metal::current_allocated_size_bytes(),
            .process_physical_footprint_bytes = physical_footprint_bytes(),
            .process_physical_footprint_peak_bytes = physical_footprint_peak_bytes(),
        };
    };
    const auto setup_memory = sample_memory();

    ConversationBenchmarkRunResult run;
    run.setup = {
        .identity = identity,
        .model_load_duration_ns = elapsed(load_start, load_finish),
        .plan_creation_duration_ns = elapsed(plan_start, plan_finish),
        .process_start_to_ready_duration_ns = options.process_started_ns.has_value()
            ? std::optional<std::uint64_t>(elapsed(*options.process_started_ns, plan_finish))
            : std::nullopt,
        .resident_weight_byte_count = plan.uploaded_weight_byte_count(),
        .resident_kv_cache_byte_count = plan.kv_cache_byte_count(),
        .resident_scratch_byte_count = plan.scratch_byte_count(),
        .metal_current_allocated_bytes = setup_memory.metal_current_allocated_bytes,
        .process_physical_footprint_bytes = setup_memory.process_physical_footprint_bytes,
        .process_physical_footprint_peak_bytes = setup_memory.process_physical_footprint_peak_bytes,
    };
    auto session = engine::Gemma2ChatSession(
        engine::Gemma2ChatSessionState {
            .file_model = std::move(file_model), .model = std::move(model),
            .tokenizer = std::move(tokenizer), .plan = std::move(plan),
        },
        engine::Gemma2ChatSessionOptions {.backend = options.backend, .sampler = options.sampler, .clock = options.clock}
    );
    const auto& scenario = primary_conversation_scenario();
    const auto run_repetition = [&](
        bool warmup,
        std::size_t repetition_index,
        bool timing_eligible,
        bool memory_audit
    ) {
        session.clear();
        const auto revision = session.plan_revision();
        const auto allocation_identities = session.plan_allocation_identities();
        ConversationBenchmarkConversationRecord conversation {
            .identity = identity, .sampler = options.sampler,
            .warmup = warmup, .repetition_index = repetition_index,
            .timing_eligible = timing_eligible, .memory_audit = memory_audit,
        };
        for (std::size_t turn_index = 0; turn_index < scenario.turns.size(); ++turn_index) {
            const auto& scenario_turn = scenario.turns[turn_index];
            session.set_max_new_tokens(scenario_turn.generated_token_limit);
            auto turn = session.generate_user_turn(
                scenario_turn.user_text,
                options.frontend_stream_callback
            );
            conversation.complete_conversation_duration_ns += turn.metrics.total_turn_duration_ns;
            conversation.per_turn_frontend_complete_ttft_ns.push_back(turn.metrics.frontend_complete_ttft_ns);
            const auto memory = memory_audit ? sample_memory() : ConversationBenchmarkMemorySnapshot {};
            run.turns.push_back({
                .identity = identity, .warmup = warmup, .repetition_index = repetition_index,
                .turn_index = turn_index, .timing_eligible = timing_eligible,
                .memory_audit = memory_audit, .user_text = scenario_turn.user_text,
                .generated_token_limit = scenario_turn.generated_token_limit,
                .sampler = options.sampler,
                .configured_stop_token_ids = std::vector<std::uint32_t>(
                    session.stop_token_ids().begin(), session.stop_token_ids().end()),
                .configured_reverse_stop_strings = std::vector<std::string>(
                    session.reverse_stop_strings().begin(), session.reverse_stop_strings().end()),
                .result = std::move(turn),
                .plan_revision = session.plan_revision(),
                .plan_allocation_identities = session.plan_allocation_identities(),
                .metal_current_allocated_bytes = memory.metal_current_allocated_bytes,
                .process_physical_footprint_bytes = memory.process_physical_footprint_bytes,
                .process_physical_footprint_peak_bytes = memory.process_physical_footprint_peak_bytes,
            });
            if (session.plan_revision() != revision) throw std::runtime_error("conversation benchmark rebuilt resident plan");
            if (session.plan_allocation_identities() != allocation_identities) {
                throw std::runtime_error("conversation benchmark replaced resident plan allocations");
            }
        }
        run.conversations.push_back(std::move(conversation));
    };
    for (std::size_t i = 0; i < options.warmup_count; ++i) run_repetition(true, i, false, false);
    for (std::size_t i = 0; i < options.repetition_count; ++i) run_repetition(false, i, true, false);
    if (options.run_memory_audit) run_repetition(false, 0U, false, true);

    bool workload_stable_through_turn = true;
    for (std::size_t turn_index = 0; turn_index < scenario.turns.size(); ++turn_index) {
        ConversationBenchmarkSummaryRecord summary {
            .identity = identity, .sampler = options.sampler, .turn_index = turn_index,
        };
        const std::vector<std::uint32_t>* expected_selected = nullptr;
        const std::vector<std::uint32_t>* expected_prompt = nullptr;
        for (const auto& record : run.turns) {
            if (!record.timing_eligible || record.turn_index != turn_index) continue;
            if (expected_selected == nullptr) expected_selected = &record.result.selected_token_ids;
            else if (record.result.selected_token_ids != *expected_selected) {
                summary.exact_selected_tokens_stable = false;
            }
            if (expected_prompt == nullptr) expected_prompt = &record.result.rendered_prompt_token_ids;
            else if (record.result.rendered_prompt_token_ids != *expected_prompt) {
                summary.exact_rendered_prompts_stable = false;
            }
        }
        workload_stable_through_turn = workload_stable_through_turn
            && summary.exact_selected_tokens_stable && summary.exact_rendered_prompts_stable;
        summary.workload_stable = workload_stable_through_turn;
        if (summary.workload_stable) {
            summary.frontend_complete_ttft_ns = summarize_turns(run.turns, turn_index, [](const auto& r) {
                return r.timing_eligible ? as_double(r.result.metrics.frontend_complete_ttft_ns) : std::nullopt;
            });
            summary.complete_duration_ns = summarize_turns(run.turns, turn_index, [](const auto& r) {
                return r.timing_eligible
                    ? std::optional<double>(static_cast<double>(r.result.metrics.total_turn_duration_ns))
                    : std::nullopt;
            });
            summary.prompt_tokens_per_second = summarize_turns(run.turns, turn_index, [](const auto& r) {
                return r.timing_eligible
                    ? benchmark_rate(r.result.metrics.generation.newly_evaluated_prompt_token_count,
                        r.result.metrics.generation.prompt_evaluation_duration_ns)
                    : std::nullopt;
            });
            summary.decode_tokens_per_second = summarize_turns(run.turns, turn_index, [](const auto& r) {
                return r.timing_eligible
                    ? benchmark_rate(r.result.metrics.generation.decode_token_count,
                        r.result.metrics.generation.decode_duration_ns)
                    : std::nullopt;
            });
            summary.visible_tokens_per_second = summarize_turns(run.turns, turn_index, [](const auto& r) {
                if (!r.timing_eligible || r.result.metrics.visible_emission_count < 2U
                    || !r.result.metrics.visible_stream_interval_ns) {
                    return std::optional<double> {};
                }
                return benchmark_rate(r.result.metrics.visible_emission_count - 1U,
                    *r.result.metrics.visible_stream_interval_ns);
            });
        } else {
            run.valid = false;
            run.validation_errors.push_back(
                "turn " + std::to_string(turn_index)
                    + " measured repetitions do not have identical rendered prompts and selected tokens"
            );
        }
        run.summaries.push_back(std::move(summary));
    }
    std::vector<std::optional<double>> complete_values;
    for (const auto& conversation : run.conversations) if (conversation.timing_eligible) {
        complete_values.push_back(static_cast<double>(conversation.complete_conversation_duration_ns));
    }
    ConversationBenchmarkSummaryRecord complete {.identity = identity, .sampler = options.sampler};
    complete.exact_selected_tokens_stable = std::all_of(
        run.summaries.begin(), run.summaries.end(), [](const auto& summary) {
            return summary.exact_selected_tokens_stable;
        }
    );
    complete.exact_rendered_prompts_stable = std::all_of(
        run.summaries.begin(), run.summaries.end(), [](const auto& summary) {
            return summary.exact_rendered_prompts_stable;
        }
    );
    complete.workload_stable = run.valid;
    if (run.valid) complete.complete_duration_ns = summarize_benchmark_values(complete_values);
    run.summaries.push_back(std::move(complete));
    return run;
}

}  // namespace local_gguf_engine::benchmarks
