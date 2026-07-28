#pragma once

#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace local_gguf_engine::benchmarks {

inline constexpr std::uint32_t conversation_benchmark_schema_version = 3U;
inline constexpr std::uint32_t conversation_benchmark_scenario_version = 1U;

struct ConversationBenchmarkScenarioTurn {
    std::string user_text;
    std::size_t generated_token_limit = 0;
};

struct ConversationBenchmarkScenario {
    std::uint32_t version = conversation_benchmark_scenario_version;
    std::string name;
    std::vector<ConversationBenchmarkScenarioTurn> turns;
};

[[nodiscard]] const ConversationBenchmarkScenario& primary_conversation_scenario();

struct ConversationBenchmarkIdentity {
    std::uint32_t schema_version = conversation_benchmark_schema_version;
    std::uint32_t scenario_version = conversation_benchmark_scenario_version;
    std::string engine = "local-gguf-engine";
    std::string engine_build;
    std::string model_label;
    std::string model_identity;
    runtime::BackendKind backend = runtime::BackendKind::metal;
    std::string operating_system;
    std::string hardware;
    std::string frontend;
};

struct ConversationBenchmarkSetupRecord {
    ConversationBenchmarkIdentity identity;
    std::optional<std::uint64_t> model_load_duration_ns;
    std::optional<std::uint64_t> plan_creation_duration_ns;
    std::optional<std::uint64_t> process_start_to_ready_duration_ns;
    std::size_t resident_weight_byte_count = 0;
    std::size_t resident_kv_cache_byte_count = 0;
    std::size_t resident_scratch_byte_count = 0;
    std::optional<std::uint64_t> metal_current_allocated_bytes;
    std::optional<std::uint64_t> process_physical_footprint_bytes;
    std::optional<std::uint64_t> process_physical_footprint_peak_bytes;
};

struct ConversationBenchmarkTurnRecord {
    ConversationBenchmarkIdentity identity;
    bool warmup = false;
    std::size_t repetition_index = 0;
    std::size_t turn_index = 0;
    bool timing_eligible = true;
    bool memory_audit = false;
    std::string user_text;
    std::size_t generated_token_limit = 0;
    engine::TokenSamplerOptions sampler;
    std::vector<std::uint32_t> configured_stop_token_ids;
    std::vector<std::string> configured_reverse_stop_strings;
    engine::Gemma2ChatTurnResult result;
    std::size_t plan_revision = 0;
    std::array<std::uintptr_t, 4> plan_allocation_identities {};
    std::optional<std::uint64_t> metal_current_allocated_bytes;
    std::optional<std::uint64_t> process_physical_footprint_bytes;
    std::optional<std::uint64_t> process_physical_footprint_peak_bytes;
};

struct ConversationBenchmarkConversationRecord {
    ConversationBenchmarkIdentity identity;
    engine::TokenSamplerOptions sampler;
    bool warmup = false;
    std::size_t repetition_index = 0;
    bool timing_eligible = true;
    bool memory_audit = false;
    std::uint64_t complete_conversation_duration_ns = 0;
    std::vector<std::optional<std::uint64_t>> per_turn_frontend_complete_ttft_ns;
};

struct ConversationBenchmarkStatistics {
    std::size_t sample_count = 0;
    std::optional<double> median;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> sample_standard_deviation;
    std::optional<double> median_absolute_deviation;
};

struct ConversationBenchmarkMemorySnapshot {
    std::optional<std::uint64_t> metal_current_allocated_bytes;
    std::optional<std::uint64_t> process_physical_footprint_bytes;
    std::optional<std::uint64_t> process_physical_footprint_peak_bytes;
};

using ConversationBenchmarkMemorySampler =
    std::function<ConversationBenchmarkMemorySnapshot()>;

struct ConversationBenchmarkSummaryRecord {
    ConversationBenchmarkIdentity identity;
    engine::TokenSamplerOptions sampler;
    std::optional<std::size_t> turn_index;
    bool exact_selected_tokens_stable = true;
    bool exact_rendered_prompts_stable = true;
    bool workload_stable = true;
    ConversationBenchmarkStatistics frontend_complete_ttft_ns;
    ConversationBenchmarkStatistics complete_duration_ns;
    ConversationBenchmarkStatistics prompt_tokens_per_second;
    ConversationBenchmarkStatistics decode_tokens_per_second;
    ConversationBenchmarkStatistics visible_tokens_per_second;
};

struct ConversationBenchmarkRunnerOptions {
    std::string model_path;
    std::size_t warmup_count = 1U;
    std::size_t repetition_count = 5U;
    runtime::BackendKind backend = runtime::BackendKind::metal;
    engine::TokenSamplerOptions sampler;
    engine::Gemma2MonotonicClock clock;
    engine::Gemma2ChatStreamCallback frontend_stream_callback;
    std::string frontend_name;
    std::optional<std::uint64_t> process_started_ns;
    std::string engine_build;
    bool run_memory_audit = true;
    ConversationBenchmarkMemorySampler memory_sampler;
};

struct ConversationBenchmarkRunResult {
    ConversationBenchmarkSetupRecord setup;
    std::vector<ConversationBenchmarkTurnRecord> turns;
    std::vector<ConversationBenchmarkConversationRecord> conversations;
    std::vector<ConversationBenchmarkSummaryRecord> summaries;
    bool valid = true;
    std::vector<std::string> validation_errors;
};

[[nodiscard]] std::optional<double> benchmark_rate(
    std::uint64_t item_count,
    std::uint64_t duration_ns
) noexcept;
[[nodiscard]] ConversationBenchmarkStatistics summarize_benchmark_values(
    std::span<const std::optional<double>> values
);
[[nodiscard]] ConversationBenchmarkRunResult run_conversation_benchmark(
    const ConversationBenchmarkRunnerOptions& options
);

}  // namespace local_gguf_engine::benchmarks
