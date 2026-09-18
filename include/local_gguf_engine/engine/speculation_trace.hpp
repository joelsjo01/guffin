#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/speculative_artifact.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"
#include "local_gguf_engine/tensor/dtype.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::uint32_t speculation_trace_schema_version = 2U;
inline constexpr std::uint64_t speculation_trace_record_max = 100'000'000ULL;
inline constexpr std::uint32_t speculation_trace_feature_max_per_record = 64U;
inline constexpr std::uint32_t speculation_trace_feature_byte_max = 64U * 1024U * 1024U;
inline constexpr std::uint64_t speculation_trace_file_byte_max = 1ULL << 36U;

enum class SpeculationTraceErrorCode : std::uint8_t {
    truncated,
    invalid_magic,
    unsupported_version,
    invalid_identity,
    invalid_record,
    invalid_causality,
    capacity_overflow,
    digest_mismatch,
    io_failure,
};

class SpeculationTraceError final : public std::runtime_error {
public:
    SpeculationTraceError(SpeculationTraceErrorCode code, std::string detail);
    [[nodiscard]] SpeculationTraceErrorCode code() const noexcept;

private:
    SpeculationTraceErrorCode code_;
};

enum class SpeculationTraceRowKind : std::uint8_t {
    prompt,
    generated,
    speculative,
};

enum class SpeculationTraceSplit : std::uint8_t {
    training_development = 1U,
    architecture_selection = 2U,
    calibration = 3U,
    sealed_admission = 4U,
};

enum class SpeculationTraceWorkload : std::uint8_t {
    instruction_chat = 1U,
    factual_grounded_qa = 2U,
    code = 3U,
    mathematics_reasoning = 4U,
    summarization_translation = 5U,
    structured_tool = 6U,
    rare_non_english = 7U,
    repeated_retrieval = 8U,
};

enum SpeculationTraceContextFlag : std::uint32_t {
    speculation_trace_context_early = 1U << 0U,
    speculation_trace_context_middle = 1U << 1U,
    speculation_trace_context_sliding_wrap = 1U << 2U,
    speculation_trace_context_global_tail = 1U << 3U,
    speculation_trace_context_global_boundary = 1U << 4U,
    speculation_trace_context_near_limit = 1U << 5U,
    speculation_trace_context_rebuilt = 1U << 6U,
};

inline constexpr std::uint32_t speculation_trace_known_context_flags =
    speculation_trace_context_early
    | speculation_trace_context_middle
    | speculation_trace_context_sliding_wrap
    | speculation_trace_context_global_tail
    | speculation_trace_context_global_boundary
    | speculation_trace_context_near_limit
    | speculation_trace_context_rebuilt;

struct SpeculationTraceSource {
    runtime::Sha256Digest source_group;
    runtime::Sha256Digest source_cluster;
    runtime::Sha256Digest conversation;
    runtime::Sha256Digest prompt_derivation;
    SpeculationTraceSplit split = SpeculationTraceSplit::training_development;
    SpeculationTraceWorkload workload = SpeculationTraceWorkload::instruction_chat;
    std::uint32_t context_generation = 0U;
    std::uint32_t context_flags = 0U;
    std::uint64_t generation_seed = 0U;

    constexpr bool operator==(const SpeculationTraceSource&) const = default;
};

struct SpeculationTraceSamplerState {
    SamplerHistorySnapshot history;
    std::uint64_t draw_count = 0U;

    constexpr bool operator==(const SpeculationTraceSamplerState&) const = default;
};

struct SpeculationTraceDrawEvidence {
    std::array<std::uint64_t, sampler_ticket_max_raw_engine_outputs> raw_engine_outputs {};
    std::uint32_t raw_engine_output_count = 0U;
    std::uint64_t draw_index = 0U;
    double unit_interval_draw = 0.0;
    double mapped_draw = 0.0;
    bool consumes_random_draw = false;

    constexpr bool operator==(const SpeculationTraceDrawEvidence&) const = default;
};

struct SpeculationTracePreparedCandidate {
    std::uint32_t token = 0U;
    float compact_logit = 0.0F;
    double probability = 0.0;
    double sampling_weight = 0.0;

    constexpr bool operator==(const SpeculationTracePreparedCandidate&) const = default;
};

struct SpeculationTraceSelection {
    TokenSelectionRequest request;
    SpeculationTraceSamplerState sampler_pre;
    SpeculationTraceSamplerState sampler_post;
    SpeculationTraceDrawEvidence draw;
    std::vector<TokenLogit> compact_candidates;
    std::vector<SpeculationTracePreparedCandidate> prepared_candidates;
    double total_sampling_weight = 0.0;

    bool operator==(const SpeculationTraceSelection&) const = default;
};

struct SpeculationTraceIdentity {
    EngineModelPayloadReceipt model;
    ResidentTargetPayloadReceipt resident;
    runtime::Sha256Digest artifact;
    runtime::Sha256Digest tokenizer;
    runtime::Sha256Digest chat_template;
    runtime::Sha256Digest source_manifest;
    runtime::Sha256Digest registry;
    TokenSamplerOptions sampler;

    constexpr bool operator==(const SpeculationTraceIdentity&) const = default;
};

struct SpeculationTraceFeature {
    TargetStateView state;
    tensor::DType dtype = tensor::DType::f16;
    std::vector<std::uint8_t> bytes;
    runtime::Sha256Digest digest;

    bool operator==(const SpeculationTraceFeature&) const = default;
};

struct SpeculationTraceRecord {
    std::uint64_t sequence_id = 0U;
    std::uint32_t turn_index = 0U;
    SpeculationTraceRowKind kind = SpeculationTraceRowKind::prompt;
    std::optional<std::uint64_t> parent_record_index;
    std::uint32_t target_position = 0U;
    std::uint32_t draft_position = 1U;
    std::uint32_t token_id = 0U;
    std::optional<std::uint32_t> target_label_token_id;
    std::uint64_t sampler_draw_count = 0U;
    bool terminal = false;
    SpeculationTraceSource source;
    std::uint32_t remaining_token_budget = 0U;
    bool retrieval_eligible = false;
    std::optional<std::uint32_t> retrieval_proposal_token_id;
    std::optional<SpeculationTraceSelection> selection;
    std::vector<SpeculationTraceFeature> features;

    bool operator==(const SpeculationTraceRecord&) const = default;
};

struct SpeculationTraceAudit {
    std::uint64_t record_count = 0U;
    std::uint64_t sequence_count = 0U;
    std::uint64_t feature_count = 0U;
    std::uint64_t feature_byte_count = 0U;
    std::uint32_t maximum_target_position = 0U;
    std::uint64_t selection_count = 0U;
    std::uint64_t prepared_candidate_count = 0U;
    std::array<std::uint64_t, 4U> split_record_counts {};
    std::array<std::uint64_t, 8U> workload_record_counts {};
    runtime::Sha256Digest envelope_digest;
};

[[nodiscard]] SpeculationTraceSelection make_speculation_trace_selection(
    std::span<const TokenLogit> compact_candidates,
    const TokenSamplerOptions& sampler_options,
    const TokenSelectionRequest& request,
    const SamplerCheckpoint& sampler_pre,
    const SamplerTicket& ticket,
    const SamplerCheckpoint& sampler_post,
    std::uint32_t selected_token
);
[[nodiscard]] std::vector<SpeculationTraceFeature> make_speculation_trace_features(
    const TargetFeatureTraceBatch& batch,
    std::size_t row_index
);

class AuthenticatedSpeculationTrace final {
public:
    AuthenticatedSpeculationTrace(const AuthenticatedSpeculationTrace&) = delete;
    AuthenticatedSpeculationTrace& operator=(const AuthenticatedSpeculationTrace&) = delete;
    AuthenticatedSpeculationTrace(AuthenticatedSpeculationTrace&&) noexcept = default;
    AuthenticatedSpeculationTrace& operator=(AuthenticatedSpeculationTrace&&) noexcept = default;

    [[nodiscard]] const SpeculationTraceIdentity& identity() const noexcept;
    [[nodiscard]] std::span<const SpeculationTraceRecord> records() const noexcept;
    [[nodiscard]] const runtime::Sha256Digest& envelope_digest() const noexcept;

private:
    friend AuthenticatedSpeculationTrace load_speculation_trace(std::span<const std::uint8_t>);
    AuthenticatedSpeculationTrace(
        SpeculationTraceIdentity,
        std::vector<SpeculationTraceRecord>,
        runtime::Sha256Digest
    );

    SpeculationTraceIdentity identity_;
    std::vector<SpeculationTraceRecord> records_;
    runtime::Sha256Digest envelope_digest_;
};

// Writes one causally ordered sequence without retaining its feature payloads
// in RAM. Publication is atomic; abandoning or failing the writer removes both
// private files and never exposes a partial trace.
class SpeculationTraceSequenceWriter final {
public:
    SpeculationTraceSequenceWriter(
        std::filesystem::path path,
        SpeculationTraceIdentity identity
    );
    SpeculationTraceSequenceWriter(const SpeculationTraceSequenceWriter&) = delete;
    SpeculationTraceSequenceWriter& operator=(const SpeculationTraceSequenceWriter&) = delete;
    SpeculationTraceSequenceWriter(SpeculationTraceSequenceWriter&&) noexcept;
    SpeculationTraceSequenceWriter& operator=(SpeculationTraceSequenceWriter&&) noexcept;
    ~SpeculationTraceSequenceWriter();

    void append(const SpeculationTraceRecord& record);
    [[nodiscard]] runtime::Sha256Digest finish();
    [[nodiscard]] std::uint64_t record_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::vector<std::uint8_t> encode_speculation_trace(
    const SpeculationTraceIdentity& identity,
    std::span<const SpeculationTraceRecord> records
);
void write_speculation_trace(
    const std::filesystem::path& path,
    const SpeculationTraceIdentity& identity,
    std::span<const SpeculationTraceRecord> records
);
[[nodiscard]] AuthenticatedSpeculationTrace load_speculation_trace(
    std::span<const std::uint8_t> bytes
);
[[nodiscard]] AuthenticatedSpeculationTrace load_speculation_trace_file(
    const std::filesystem::path& path
);
[[nodiscard]] SpeculationTraceAudit audit_speculation_trace(
    const AuthenticatedSpeculationTrace& trace
);

}  // namespace local_gguf_engine::engine
