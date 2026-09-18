#include "local_gguf_engine/engine/speculation_trace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace local_gguf_engine::engine {
namespace {

constexpr std::array<std::uint8_t, 8U> trace_magic {
    'L', 'G', 'T', 'R', 'C', '5', 0U, 0U,
};
constexpr std::size_t digest_offset = 12U;
constexpr std::size_t digest_size = 32U;
constexpr std::uint64_t no_parent = std::numeric_limits<std::uint64_t>::max();

[[noreturn]] void fail(SpeculationTraceErrorCode code, const std::string& detail) {
    throw SpeculationTraceError(code, detail);
}

class Writer final {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }
    void u16(std::uint16_t value) {
        for (std::uint32_t shift = 0U; shift < 16U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u32(std::uint32_t value) {
        for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u64(std::uint64_t value) {
        for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void f32(float value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }
    void digest(const runtime::Sha256Digest& value) {
        raw(value.bytes);
    }
    void raw(std::span<const std::uint8_t> value) {
        if (bytes_.size() > speculation_trace_file_byte_max
            || value.size() > speculation_trace_file_byte_max - bytes_.size()) {
            fail(SpeculationTraceErrorCode::capacity_overflow, "encoded file bytes");
        }
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }
    [[nodiscard]] std::vector<std::uint8_t> finish() && { return std::move(bytes_); }

private:
    std::vector<std::uint8_t> bytes_;
};

class FileWriter final {
public:
    explicit FileWriter(const std::filesystem::path& path)
        : path_(path), output_(path, std::ios::binary | std::ios::out | std::ios::trunc) {
        if (!output_) fail(SpeculationTraceErrorCode::io_failure, path.string());
    }

    void u8(std::uint8_t value) { raw(std::span<const std::uint8_t>(&value, 1U)); }
    void u16(std::uint16_t value) {
        for (std::uint32_t shift = 0U; shift < 16U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u32(std::uint32_t value) {
        for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u64(std::uint64_t value) {
        for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void f32(float value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }
    void digest(const runtime::Sha256Digest& value) { raw(value.bytes); }
    void raw(std::span<const std::uint8_t> value) {
        if (byte_count_ > speculation_trace_file_byte_max
            || value.size() > speculation_trace_file_byte_max - byte_count_) {
            fail(SpeculationTraceErrorCode::capacity_overflow, "encoded file bytes");
        }
        if (!value.empty() && !output_.write(
            reinterpret_cast<const char*>(value.data()),
            static_cast<std::streamsize>(value.size())
        )) {
            fail(SpeculationTraceErrorCode::io_failure, path_.string());
        }
        hash_.update(value);
        byte_count_ += value.size();
    }
    [[nodiscard]] runtime::Sha256Digest finish() {
        const auto digest_value = hash_.finalize();
        output_.seekp(static_cast<std::streamoff>(digest_offset));
        if (!output_ || !output_.write(
            reinterpret_cast<const char*>(digest_value.bytes.data()),
            static_cast<std::streamsize>(digest_value.bytes.size())
        )) {
            fail(SpeculationTraceErrorCode::io_failure, path_.string());
        }
        output_.flush();
        if (!output_) fail(SpeculationTraceErrorCode::io_failure, path_.string());
        output_.close();
        return digest_value;
    }

private:
    std::filesystem::path path_;
    std::ofstream output_;
    runtime::Sha256 hash_;
    std::uint64_t byte_count_ = 0U;
};

class RecordFileWriter final {
public:
    explicit RecordFileWriter(const std::filesystem::path& path)
        : path_(path), output_(path, std::ios::binary | std::ios::out | std::ios::trunc) {
        if (!output_) fail(SpeculationTraceErrorCode::io_failure, path.string());
    }
    ~RecordFileWriter() { if (output_.is_open()) output_.close(); }

    void u8(std::uint8_t value) { raw(std::span<const std::uint8_t>(&value, 1U)); }
    void u16(std::uint16_t value) {
        for (std::uint32_t shift = 0U; shift < 16U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u32(std::uint32_t value) {
        for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u64(std::uint64_t value) {
        for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void f32(float value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }
    void digest(const runtime::Sha256Digest& value) { raw(value.bytes); }
    void raw(std::span<const std::uint8_t> value) {
        if (byte_count_ > speculation_trace_file_byte_max
            || value.size() > speculation_trace_file_byte_max - byte_count_) {
            fail(SpeculationTraceErrorCode::capacity_overflow, "streamed record bytes");
        }
        if (!value.empty() && !output_.write(
            reinterpret_cast<const char*>(value.data()),
            static_cast<std::streamsize>(value.size())
        )) fail(SpeculationTraceErrorCode::io_failure, path_.string());
        byte_count_ += value.size();
    }
    void close() {
        output_.flush();
        if (!output_) fail(SpeculationTraceErrorCode::io_failure, path_.string());
        output_.close();
        if (!output_) fail(SpeculationTraceErrorCode::io_failure, path_.string());
    }

private:
    std::filesystem::path path_;
    std::ofstream output_;
    std::uint64_t byte_count_ = 0U;
};

class Reader final {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}
    [[nodiscard]] std::uint8_t u8() {
        require(1U);
        return bytes_[offset_++];
    }
    [[nodiscard]] std::uint16_t u16() {
        auto value = std::uint16_t {0U};
        for (std::uint32_t shift = 0U; shift < 16U; shift += 8U) {
            value |= static_cast<std::uint16_t>(u8()) << shift;
        }
        return value;
    }
    [[nodiscard]] std::uint32_t u32() {
        auto value = std::uint32_t {0U};
        for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
            value |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return value;
    }
    [[nodiscard]] std::uint64_t u64() {
        auto value = std::uint64_t {0U};
        for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
            value |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return value;
    }
    [[nodiscard]] float f32() { return std::bit_cast<float>(u32()); }
    [[nodiscard]] double f64() { return std::bit_cast<double>(u64()); }
    [[nodiscard]] runtime::Sha256Digest digest() {
        runtime::Sha256Digest value;
        const auto data = raw(value.bytes.size());
        std::copy(data.begin(), data.end(), value.bytes.begin());
        return value;
    }
    [[nodiscard]] std::span<const std::uint8_t> raw(std::size_t count) {
        require(count);
        const auto result = bytes_.subspan(offset_, count);
        offset_ += count;
        return result;
    }
    [[nodiscard]] bool at_end() const noexcept { return offset_ == bytes_.size(); }

private:
    void require(std::size_t count) const {
        if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
            fail(SpeculationTraceErrorCode::truncated, "trace field");
        }
    }
    std::span<const std::uint8_t> bytes_;
    std::size_t offset_ = 0U;
};

template <typename Sink>
void write_model_receipt(Sink& writer, const EngineModelPayloadReceipt& value) {
    writer.u32(value.version);
    writer.u64(value.whole_gguf_byte_count);
    writer.u32(value.source_tensor_count);
    writer.digest(value.whole_gguf);
    writer.digest(value.canonical_consumed_metadata);
    writer.digest(value.source_tensors);
}

[[nodiscard]] EngineModelPayloadReceipt read_model_receipt(Reader& reader) {
    EngineModelPayloadReceipt value;
    value.version = reader.u32();
    value.whole_gguf_byte_count = reader.u64();
    value.source_tensor_count = reader.u32();
    value.whole_gguf = reader.digest();
    value.canonical_consumed_metadata = reader.digest();
    value.source_tensors = reader.digest();
    return value;
}

template <typename Sink>
void write_resident_receipt(Sink& writer, const ResidentTargetPayloadReceipt& value) {
    writer.u32(value.version);
    writer.u32(value.allocation_count);
    writer.u64(value.byte_count);
    writer.digest(value.bytes);
}

[[nodiscard]] ResidentTargetPayloadReceipt read_resident_receipt(Reader& reader) {
    ResidentTargetPayloadReceipt value;
    value.version = reader.u32();
    value.allocation_count = reader.u32();
    value.byte_count = reader.u64();
    value.bytes = reader.digest();
    return value;
}

template <typename Sink>
void write_sampler(Sink& writer, const TokenSamplerOptions& sampler) {
    writer.u32(std::bit_cast<std::uint32_t>(sampler.temperature));
    writer.u64(sampler.top_k);
    writer.u32(std::bit_cast<std::uint32_t>(sampler.top_p));
    writer.u32(std::bit_cast<std::uint32_t>(sampler.min_p));
    writer.u32(std::bit_cast<std::uint32_t>(sampler.repetition_penalty));
    writer.u64(sampler.seed);
    writer.u64(sampler.repetition_window);
}

[[nodiscard]] TokenSamplerOptions read_sampler(Reader& reader) {
    return TokenSamplerOptions {
        .temperature = std::bit_cast<float>(reader.u32()),
        .top_k = static_cast<std::size_t>(reader.u64()),
        .top_p = std::bit_cast<float>(reader.u32()),
        .min_p = std::bit_cast<float>(reader.u32()),
        .repetition_penalty = std::bit_cast<float>(reader.u32()),
        .seed = reader.u64(),
        .repetition_window = static_cast<std::size_t>(reader.u64()),
    };
}

template <typename Sink>
void write_history(Sink& writer, const SamplerHistorySnapshot& history) {
    writer.u32(history.head);
    writer.u32(history.length);
    writer.u32(history.window_limit);
    for (const auto token : history.token_ids) writer.u32(token);
}

[[nodiscard]] SamplerHistorySnapshot read_history(Reader& reader) {
    SamplerHistorySnapshot history;
    history.head = reader.u32();
    history.length = reader.u32();
    history.window_limit = reader.u32();
    for (auto& token : history.token_ids) token = reader.u32();
    return history;
}

template <typename Sink>
void write_selection_request(Sink& writer, const TokenSelectionRequest& request) {
    writer.u32(request.top_k);
    writer.u32(request.base_candidate_count);
    writer.u32(request.history_token_count);
    for (const auto token : request.history_token_ids) writer.u32(token);
}

[[nodiscard]] TokenSelectionRequest read_selection_request(Reader& reader) {
    TokenSelectionRequest request;
    request.top_k = reader.u32();
    request.base_candidate_count = reader.u32();
    request.history_token_count = reader.u32();
    for (auto& token : request.history_token_ids) token = reader.u32();
    return request;
}

[[nodiscard]] bool same_history(
    const SamplerHistorySnapshot& left,
    const SamplerHistorySnapshot& right
) noexcept {
    return left.head == right.head && left.length == right.length
        && left.window_limit == right.window_limit && left.token_ids == right.token_ids;
}

void validate_history(const SamplerHistorySnapshot& history) {
    if (history.window_limit > token_sampler_max_repetition_window
        || history.length > history.window_limit
        || (history.window_limit == 0U ? history.head != 0U
                                      : history.head >= history.window_limit)) {
        fail(SpeculationTraceErrorCode::invalid_record, "sampler history bounds");
    }
}

[[nodiscard]] bool valid_split(SpeculationTraceSplit split) noexcept {
    return split >= SpeculationTraceSplit::training_development
        && split <= SpeculationTraceSplit::sealed_admission;
}

[[nodiscard]] bool valid_workload(SpeculationTraceWorkload workload) noexcept {
    return workload >= SpeculationTraceWorkload::instruction_chat
        && workload <= SpeculationTraceWorkload::repeated_retrieval;
}

void validate_source(const SpeculationTraceSource& source) {
    if (source.source_group.is_zero() || source.source_cluster.is_zero()
        || source.conversation.is_zero() || source.prompt_derivation.is_zero()
        || !valid_split(source.split) || !valid_workload(source.workload)
        || (source.context_flags & ~speculation_trace_known_context_flags) != 0U) {
        fail(SpeculationTraceErrorCode::invalid_record, "source/split provenance");
    }
}

void validate_selection(
    const SpeculationTraceSelection& selection,
    const TokenSamplerOptions& options,
    std::uint32_t selected_token
) {
    validate_history(selection.sampler_pre.history);
    validate_history(selection.sampler_post.history);
    if (selection.request.history_token_count > token_sampler_max_repetition_window
        || selection.request.top_k == 0U
        || selection.request.top_k > token_sampler_max_top_k
        || selection.request.base_candidate_count
            != selection.request.top_k + selection.request.history_token_count
        || selection.compact_candidates.empty()
        || selection.compact_candidates.size() > token_sampler_max_candidates
        || selection.prepared_candidates.empty()
        || selection.prepared_candidates.size() > token_sampler_max_top_k
        || selection.draw.raw_engine_output_count > sampler_ticket_max_raw_engine_outputs
        || !std::isfinite(selection.draw.unit_interval_draw)
        || selection.draw.unit_interval_draw < 0.0
        || selection.draw.unit_interval_draw >= 1.0
        || !std::isfinite(selection.draw.mapped_draw)
        || !std::isfinite(selection.total_sampling_weight)
        || selection.total_sampling_weight <= 0.0) {
        fail(SpeculationTraceErrorCode::invalid_record, "prepared selection bounds");
    }
    const auto expected_request = make_token_selection_request(
        options, selection.sampler_pre.history
    );
    if (selection.request.top_k != expected_request.top_k
        || selection.request.base_candidate_count != expected_request.base_candidate_count
        || selection.request.history_token_count != expected_request.history_token_count
        || selection.request.history_token_ids != expected_request.history_token_ids) {
        fail(SpeculationTraceErrorCode::invalid_causality, "selection request/pre-state");
    }
    const auto prepared = prepare_token_distribution(
        selection.compact_candidates, options, selection.request
    );
    if (prepared.count != selection.prepared_candidates.size()
        || std::bit_cast<std::uint64_t>(prepared.total_sampling_weight)
            != std::bit_cast<std::uint64_t>(selection.total_sampling_weight)) {
        fail(SpeculationTraceErrorCode::invalid_record, "prepared distribution total");
    }
    auto probability_sum = 0.0;
    for (std::size_t index = 0U; index < prepared.count; ++index) {
        const auto& stored = selection.prepared_candidates[index];
        const auto& recomputed = prepared.values[index];
        const auto compact = std::find_if(
            selection.compact_candidates.begin(), selection.compact_candidates.end(),
            [&](const TokenLogit& candidate) { return candidate.token == stored.token; }
        );
        if (compact == selection.compact_candidates.end()
            || !std::isfinite(stored.compact_logit)
            || stored.token != recomputed.token
            || std::bit_cast<std::uint64_t>(stored.probability)
                != std::bit_cast<std::uint64_t>(recomputed.probability)
            || std::bit_cast<std::uint64_t>(stored.sampling_weight)
                != std::bit_cast<std::uint64_t>(recomputed.sampling_weight)) {
            fail(SpeculationTraceErrorCode::invalid_record, "prepared distribution entry");
        }
        probability_sum += stored.probability;
    }
    if (!std::isfinite(probability_sum)
        || std::abs(probability_sum - 1.0) > 8.0 * std::numeric_limits<double>::epsilon()) {
        fail(SpeculationTraceErrorCode::invalid_record, "prepared probability normalization");
    }
    const auto expected_post_history = append_sampler_history(
        selection.sampler_pre.history,
        std::span<const std::uint32_t>(&selected_token, 1U)
    );
    const auto expected_draws = options.temperature == 0.0F ? 0U : 1U;
    if (!same_history(selection.sampler_post.history, expected_post_history)
        || selection.sampler_post.draw_count
            != selection.sampler_pre.draw_count + expected_draws
        || selection.draw.draw_index != selection.sampler_pre.draw_count
        || selection.draw.consumes_random_draw != (expected_draws != 0U)
        || selection.draw.raw_engine_output_count
            != (selection.draw.consumes_random_draw
                ? selection.draw.raw_engine_output_count : 0U)
        || (selection.draw.consumes_random_draw
            && (selection.draw.raw_engine_output_count == 0U
                || selection.draw.mapped_draw < 0.0
                || selection.draw.mapped_draw >= selection.total_sampling_weight))) {
        fail(SpeculationTraceErrorCode::invalid_causality, "sampler pre/post transition");
    }
    auto cumulative = 0.0;
    auto reproduced = selection.prepared_candidates.back().token;
    if (!selection.draw.consumes_random_draw) {
        reproduced = selection.prepared_candidates.front().token;
    } else {
        for (const auto& candidate : selection.prepared_candidates) {
            cumulative += candidate.sampling_weight;
            if (selection.draw.mapped_draw < cumulative) {
                reproduced = candidate.token;
                break;
            }
        }
    }
    if (reproduced != selected_token) {
        fail(SpeculationTraceErrorCode::invalid_causality, "raw draw/token selection");
    }
}

void validate_identity(const SpeculationTraceIdentity& identity) {
    if (identity.model.version != engine_model_payload_receipt_version
        || identity.model.whole_gguf_byte_count == 0U
        || identity.model.source_tensor_count == 0U
        || identity.model.whole_gguf.is_zero()
        || identity.model.canonical_consumed_metadata.is_zero()
        || identity.model.source_tensors.is_zero()
        || identity.resident.version != 1U
        || identity.resident.allocation_count == 0U
        || identity.resident.byte_count == 0U
        || identity.resident.bytes.is_zero()
        || identity.artifact.is_zero() || identity.tokenizer.is_zero()
        || identity.chat_template.is_zero() || identity.source_manifest.is_zero()
        || identity.registry.is_zero()
        || !std::isfinite(identity.sampler.temperature)
        || identity.sampler.temperature < 0.0F
        || identity.sampler.top_k == 0U
        || identity.sampler.top_k > token_sampler_max_top_k
        || !std::isfinite(identity.sampler.top_p)
        || identity.sampler.top_p <= 0.0F || identity.sampler.top_p > 1.0F
        || !std::isfinite(identity.sampler.min_p)
        || identity.sampler.min_p < 0.0F || identity.sampler.min_p > 1.0F
        || !std::isfinite(identity.sampler.repetition_penalty)
        || identity.sampler.repetition_penalty <= 0.0F
        || identity.sampler.repetition_window > token_sampler_max_repetition_window) {
        fail(SpeculationTraceErrorCode::invalid_identity, "identity or sampler bounds");
    }
}

[[nodiscard]] std::uint32_t element_byte_count(tensor::DType dtype) {
    switch (dtype) {
        case tensor::DType::f16: return 2U;
        case tensor::DType::f32: return 4U;
        default:
            fail(SpeculationTraceErrorCode::invalid_record, "trace feature dtype");
    }
}

void validate_feature(const SpeculationTraceFeature& feature, bool verify_digest) {
    if (feature.state.execution_epoch == 0U || feature.state.bank_id == 0U
        || feature.state.child_id == 0U || feature.state.row_count != 1U
        || feature.state.feature_width == 0U
        || feature.state.coordinate.target_position
            == std::numeric_limits<std::uint32_t>::max()
        || feature.state.coordinate.draft_position
            != feature.state.coordinate.target_position + 1U
        || (feature.state.source_stage != FeatureTapStage::embedding_residual
            && feature.state.source_stage != FeatureTapStage::layer_residual
            && feature.state.source_stage != FeatureTapStage::final_normalized)
        || (feature.state.source_stage != FeatureTapStage::layer_residual
            && feature.state.source_layer != 0U)) {
        fail(SpeculationTraceErrorCode::invalid_record, "feature state provenance");
    }
    const auto required = static_cast<std::uint64_t>(feature.state.feature_width)
        * element_byte_count(feature.dtype);
    if (required != feature.bytes.size()
        || required > speculation_trace_feature_byte_max) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "feature bytes");
    }
    const auto actual = runtime::sha256(feature.bytes);
    if (verify_digest && feature.digest != actual) {
        fail(SpeculationTraceErrorCode::digest_mismatch, "feature payload");
    }
}

void validate_records(
    std::span<const SpeculationTraceRecord> records,
    bool verify_digests,
    const TokenSamplerOptions& sampler_options
) {
    if (records.empty() || records.size() > speculation_trace_record_max) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "record count");
    }
    std::unordered_set<std::uint64_t> sequences;
    for (std::size_t index = 0U; index < records.size(); ++index) {
        auto& record = records[index];
        if (record.sequence_id == 0U
            || record.target_position == std::numeric_limits<std::uint32_t>::max()
            || record.draft_position != record.target_position + 1U
            || record.features.empty()
            || record.features.size() > speculation_trace_feature_max_per_record
            || (record.kind != SpeculationTraceRowKind::prompt
                && record.kind != SpeculationTraceRowKind::generated
                && record.kind != SpeculationTraceRowKind::speculative)) {
            fail(SpeculationTraceErrorCode::invalid_record, std::to_string(index));
        }
        validate_source(record.source);
        if (record.retrieval_proposal_token_id.has_value()
            && !record.retrieval_eligible) {
            fail(SpeculationTraceErrorCode::invalid_record, "ineligible retrieval proposal");
        }
        if (record.selection.has_value()) {
            if (!record.target_label_token_id.has_value()) {
                fail(SpeculationTraceErrorCode::invalid_record, "selection without target label");
            }
            validate_selection(*record.selection, sampler_options, *record.target_label_token_id);
            if (record.sampler_draw_count != record.selection->sampler_pre.draw_count) {
                fail(SpeculationTraceErrorCode::invalid_causality, "legacy draw count/pre-state");
            }
        } else if (record.sampler_draw_count != 0U
                   || (record.kind != SpeculationTraceRowKind::prompt
                       && !(record.kind == SpeculationTraceRowKind::generated
                            && record.terminal
                            && !record.target_label_token_id.has_value()))) {
            fail(SpeculationTraceErrorCode::invalid_record, "unlabeled row is not a terminal context leaf");
        }
        if (record.parent_record_index.has_value()) {
            const auto parent_index = *record.parent_record_index;
            if (parent_index >= index
                || records[parent_index].sequence_id != record.sequence_id
                || records[parent_index].draft_position != record.target_position
                || records[parent_index].terminal
                || !records[parent_index].target_label_token_id.has_value()
                || *records[parent_index].target_label_token_id != record.token_id
                || records[parent_index].source.source_group != record.source.source_group
                || records[parent_index].source.source_cluster != record.source.source_cluster
                || records[parent_index].source.conversation != record.source.conversation
                || records[parent_index].source.split != record.source.split
                || records[parent_index].source.workload != record.source.workload
                || records[parent_index].source.context_generation
                    != record.source.context_generation) {
                fail(SpeculationTraceErrorCode::invalid_causality, std::to_string(index));
            }
        } else if (!sequences.insert(record.sequence_id).second) {
            fail(SpeculationTraceErrorCode::invalid_causality, "second root in sequence");
        }
        for (const auto& feature : record.features) {
            validate_feature(feature, verify_digests);
            if (feature.state.coordinate.target_position != record.target_position
                || feature.state.coordinate.draft_position != record.draft_position
                || (record.kind == SpeculationTraceRowKind::speculative
                    ? feature.state.arena_epoch == 0U
                    : feature.state.arena_epoch != 0U)) {
                fail(SpeculationTraceErrorCode::invalid_causality, "future/stale feature row");
            }
        }
    }
}

template <typename Sink>
void write_trace_header(
    Sink& writer, const SpeculationTraceIdentity& identity, std::uint64_t record_count
) {
    writer.raw(trace_magic);
    writer.u32(speculation_trace_schema_version);
    writer.raw(std::array<std::uint8_t, digest_size> {});
    write_model_receipt(writer, identity.model);
    write_resident_receipt(writer, identity.resident);
    writer.digest(identity.artifact);
    writer.digest(identity.tokenizer);
    writer.digest(identity.chat_template);
    writer.digest(identity.source_manifest);
    writer.digest(identity.registry);
    write_sampler(writer, identity.sampler);
    writer.u64(record_count);
}

template <typename Sink>
void write_trace_record(Sink& writer, const SpeculationTraceRecord& record) {
        writer.u64(record.sequence_id);
        writer.u32(record.turn_index);
        writer.u8(static_cast<std::uint8_t>(record.kind));
        writer.u8(record.terminal ? 1U : 0U);
        writer.u64(record.parent_record_index.value_or(no_parent));
        writer.u32(record.target_position);
        writer.u32(record.draft_position);
        writer.u32(record.token_id);
        writer.u8(record.target_label_token_id.has_value() ? 1U : 0U);
        writer.u32(record.target_label_token_id.value_or(0U));
        writer.u64(record.sampler_draw_count);
        writer.digest(record.source.source_group);
        writer.digest(record.source.source_cluster);
        writer.digest(record.source.conversation);
        writer.digest(record.source.prompt_derivation);
        writer.u8(static_cast<std::uint8_t>(record.source.split));
        writer.u8(static_cast<std::uint8_t>(record.source.workload));
        writer.u32(record.source.context_generation);
        writer.u32(record.source.context_flags);
        writer.u64(record.source.generation_seed);
        writer.u32(record.remaining_token_budget);
        writer.u8(record.retrieval_eligible ? 1U : 0U);
        writer.u8(record.retrieval_proposal_token_id.has_value() ? 1U : 0U);
        writer.u32(record.retrieval_proposal_token_id.value_or(0U));
        writer.u8(record.selection.has_value() ? 1U : 0U);
        if (record.selection.has_value()) {
            const auto& selection = *record.selection;
            write_selection_request(writer, selection.request);
            write_history(writer, selection.sampler_pre.history);
            writer.u64(selection.sampler_pre.draw_count);
            write_history(writer, selection.sampler_post.history);
            writer.u64(selection.sampler_post.draw_count);
            writer.u32(selection.draw.raw_engine_output_count);
            for (const auto raw : selection.draw.raw_engine_outputs) writer.u64(raw);
            writer.u64(selection.draw.draw_index);
            writer.f64(selection.draw.unit_interval_draw);
            writer.f64(selection.draw.mapped_draw);
            writer.u8(selection.draw.consumes_random_draw ? 1U : 0U);
            writer.u32(static_cast<std::uint32_t>(selection.compact_candidates.size()));
            for (const auto& candidate : selection.compact_candidates) {
                writer.u32(candidate.token);
                writer.f32(candidate.logit);
            }
            writer.u32(static_cast<std::uint32_t>(selection.prepared_candidates.size()));
            for (const auto& candidate : selection.prepared_candidates) {
                writer.u32(candidate.token);
                writer.f32(candidate.compact_logit);
                writer.f64(candidate.probability);
                writer.f64(candidate.sampling_weight);
            }
            writer.f64(selection.total_sampling_weight);
        }
        writer.u16(static_cast<std::uint16_t>(record.features.size()));
        for (const auto& feature : record.features) {
            const auto& state = feature.state;
            writer.u64(state.execution_epoch);
            writer.u64(state.arena_epoch);
            writer.u32(state.bank_id);
            writer.u32(state.child_id);
            writer.u8(static_cast<std::uint8_t>(state.source_stage));
            writer.u16(state.source_layer);
            writer.u32(state.coordinate.target_position);
            writer.u32(state.coordinate.draft_position);
            writer.u32(state.row_count);
            writer.u32(state.feature_width);
            writer.u8(static_cast<std::uint8_t>(feature.dtype));
            writer.u32(static_cast<std::uint32_t>(feature.bytes.size()));
            writer.digest(runtime::sha256(feature.bytes));
            writer.raw(feature.bytes);
        }
}

template <typename Sink>
void write_trace_payload(
    Sink& writer,
    const SpeculationTraceIdentity& identity,
    std::span<const SpeculationTraceRecord> records
) {
    write_trace_header(writer, identity, records.size());
    for (const auto& record : records) write_trace_record(writer, record);
}

[[nodiscard]] runtime::Sha256Digest compute_envelope_digest(
    std::span<const std::uint8_t> bytes
) {
    if (bytes.size() < digest_offset + digest_size) {
        fail(SpeculationTraceErrorCode::truncated, "envelope digest");
    }
    runtime::Sha256 hash;
    hash.update(bytes.first(digest_offset));
    constexpr std::array<std::uint8_t, digest_size> zeros {};
    hash.update(zeros);
    hash.update(bytes.subspan(digest_offset + digest_size));
    return hash.finalize();
}

}  // namespace

SpeculationTraceError::SpeculationTraceError(
    SpeculationTraceErrorCode code, std::string detail
) : std::runtime_error("speculation trace: " + std::move(detail)), code_(code) {}

SpeculationTraceErrorCode SpeculationTraceError::code() const noexcept { return code_; }

AuthenticatedSpeculationTrace::AuthenticatedSpeculationTrace(
    SpeculationTraceIdentity identity,
    std::vector<SpeculationTraceRecord> records,
    runtime::Sha256Digest digest
) : identity_(std::move(identity)), records_(std::move(records)), envelope_digest_(digest) {}

const SpeculationTraceIdentity& AuthenticatedSpeculationTrace::identity() const noexcept {
    return identity_;
}
std::span<const SpeculationTraceRecord> AuthenticatedSpeculationTrace::records() const noexcept {
    return records_;
}
const runtime::Sha256Digest& AuthenticatedSpeculationTrace::envelope_digest() const noexcept {
    return envelope_digest_;
}

struct SpeculationTraceSequenceWriter::Impl {
    struct LastRecord {
        std::uint64_t sequence_id = 0U;
        std::uint32_t draft_position = 0U;
        std::optional<std::uint32_t> target_label_token_id;
        SpeculationTraceSource source;
        bool terminal = false;
    };

    std::filesystem::path path;
    std::filesystem::path partial;
    std::filesystem::path body;
    SpeculationTraceIdentity identity;
    std::unique_ptr<RecordFileWriter> output;
    std::optional<LastRecord> last;
    std::uint64_t count = 0U;
    bool finished = false;

    Impl(std::filesystem::path value_path, SpeculationTraceIdentity value_identity)
        : path(std::move(value_path)), identity(std::move(value_identity)) {
        validate_identity(identity);
        partial = path;
        partial += ".partial";
        body = path;
        body += ".body.partial";
        if (std::filesystem::exists(path) || std::filesystem::exists(partial)
            || std::filesystem::exists(body)) {
            fail(SpeculationTraceErrorCode::io_failure,
                "refusing to overwrite streamed trace output");
        }
        output = std::make_unique<RecordFileWriter>(body);
    }

    ~Impl() {
        output.reset();
        if (!finished) {
            std::error_code ignored;
            std::filesystem::remove(partial, ignored);
            std::filesystem::remove(body, ignored);
        }
    }
};

SpeculationTraceSequenceWriter::SpeculationTraceSequenceWriter(
    std::filesystem::path path,
    SpeculationTraceIdentity identity
) : impl_(std::make_unique<Impl>(std::move(path), std::move(identity))) {}
SpeculationTraceSequenceWriter::SpeculationTraceSequenceWriter(
    SpeculationTraceSequenceWriter&&
) noexcept = default;
SpeculationTraceSequenceWriter& SpeculationTraceSequenceWriter::operator=(
    SpeculationTraceSequenceWriter&&
) noexcept = default;
SpeculationTraceSequenceWriter::~SpeculationTraceSequenceWriter() = default;

void SpeculationTraceSequenceWriter::append(const SpeculationTraceRecord& record) {
    if (!impl_ || impl_->finished || !impl_->output) {
        fail(SpeculationTraceErrorCode::io_failure, "streamed trace writer is closed");
    }
    if (impl_->count >= speculation_trace_record_max) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "streamed record count");
    }
    if (record.sequence_id == 0U
        || record.target_position == std::numeric_limits<std::uint32_t>::max()
        || record.draft_position != record.target_position + 1U
        || record.features.empty()
        || record.features.size() > speculation_trace_feature_max_per_record
        || (record.kind != SpeculationTraceRowKind::prompt
            && record.kind != SpeculationTraceRowKind::generated
            && record.kind != SpeculationTraceRowKind::speculative)) {
        fail(SpeculationTraceErrorCode::invalid_record, "streamed record fields");
    }
    validate_source(record.source);
    if (record.retrieval_proposal_token_id.has_value() && !record.retrieval_eligible) {
        fail(SpeculationTraceErrorCode::invalid_record, "ineligible retrieval proposal");
    }
    if (record.selection.has_value()) {
        if (!record.target_label_token_id.has_value()) {
            fail(SpeculationTraceErrorCode::invalid_record, "selection without target label");
        }
        validate_selection(
            *record.selection, impl_->identity.sampler, *record.target_label_token_id
        );
        if (record.sampler_draw_count != record.selection->sampler_pre.draw_count) {
            fail(SpeculationTraceErrorCode::invalid_causality, "legacy draw count/pre-state");
        }
    } else if (record.sampler_draw_count != 0U
               || (record.kind != SpeculationTraceRowKind::prompt
                   && !(record.kind == SpeculationTraceRowKind::generated
                        && record.terminal
                        && !record.target_label_token_id.has_value()))) {
        fail(SpeculationTraceErrorCode::invalid_record, "unlabeled row is not a terminal context leaf");
    }
    for (const auto& feature : record.features) {
        validate_feature(feature, false);
        if (feature.state.coordinate.target_position != record.target_position
            || feature.state.coordinate.draft_position != record.draft_position
            || (record.kind == SpeculationTraceRowKind::speculative
                ? feature.state.arena_epoch == 0U : feature.state.arena_epoch != 0U)) {
            fail(SpeculationTraceErrorCode::invalid_causality, "streamed feature provenance");
        }
    }
    if (!impl_->last.has_value()) {
        if (record.parent_record_index.has_value()) {
            fail(SpeculationTraceErrorCode::invalid_causality, "streamed sequence root parent");
        }
    } else {
        const auto& previous = *impl_->last;
        if (!record.parent_record_index.has_value()
            || *record.parent_record_index != impl_->count - 1U
            || previous.terminal || previous.sequence_id != record.sequence_id
            || previous.draft_position != record.target_position
            || !previous.target_label_token_id.has_value()
            || *previous.target_label_token_id != record.token_id
            || previous.source.source_group != record.source.source_group
            || previous.source.source_cluster != record.source.source_cluster
            || previous.source.conversation != record.source.conversation
            || previous.source.split != record.source.split
            || previous.source.workload != record.source.workload
            || previous.source.context_generation != record.source.context_generation) {
            fail(SpeculationTraceErrorCode::invalid_causality, "streamed sequence edge");
        }
    }
    write_trace_record(*impl_->output, record);
    impl_->last = Impl::LastRecord {
        .sequence_id = record.sequence_id,
        .draft_position = record.draft_position,
        .target_label_token_id = record.target_label_token_id,
        .source = record.source,
        .terminal = record.terminal,
    };
    ++impl_->count;
}

runtime::Sha256Digest SpeculationTraceSequenceWriter::finish() {
    if (!impl_ || impl_->finished || !impl_->output || impl_->count == 0U
        || !impl_->last.has_value() || !impl_->last->terminal) {
        fail(SpeculationTraceErrorCode::invalid_causality,
            "streamed trace must finish at one terminal record");
    }
    try {
        impl_->output->close();
        impl_->output.reset();
        FileWriter writer(impl_->partial);
        write_trace_header(writer, impl_->identity, impl_->count);
        std::ifstream input(impl_->body, std::ios::binary);
        if (!input) fail(SpeculationTraceErrorCode::io_failure, impl_->body.string());
        std::array<std::uint8_t, 1U << 20U> chunk {};
        while (input) {
            input.read(reinterpret_cast<char*>(chunk.data()),
                static_cast<std::streamsize>(chunk.size()));
            const auto read = input.gcount();
            if (read < 0) fail(SpeculationTraceErrorCode::io_failure, impl_->body.string());
            writer.raw(std::span<const std::uint8_t>(chunk).first(
                static_cast<std::size_t>(read)
            ));
        }
        if (!input.eof()) fail(SpeculationTraceErrorCode::io_failure, impl_->body.string());
        const auto digest = writer.finish();
        std::error_code error;
        std::filesystem::rename(impl_->partial, impl_->path, error);
        if (error) fail(SpeculationTraceErrorCode::io_failure, error.message());
        std::filesystem::remove(impl_->body, error);
        if (error) fail(SpeculationTraceErrorCode::io_failure, error.message());
        impl_->finished = true;
        return digest;
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(impl_->partial, ignored);
        std::filesystem::remove(impl_->body, ignored);
        throw;
    }
}

std::uint64_t SpeculationTraceSequenceWriter::record_count() const noexcept {
    return impl_ ? impl_->count : 0U;
}

std::vector<std::uint8_t> encode_speculation_trace(
    const SpeculationTraceIdentity& identity,
    std::span<const SpeculationTraceRecord> input_records
) {
    validate_identity(identity);
    validate_records(input_records, false, identity.sampler);
    Writer writer;
    write_trace_payload(writer, identity, input_records);
    auto bytes = std::move(writer).finish();
    const auto digest = compute_envelope_digest(bytes);
    std::copy(digest.bytes.begin(), digest.bytes.end(), bytes.begin() + digest_offset);
    return bytes;
}

void write_speculation_trace(
    const std::filesystem::path& path,
    const SpeculationTraceIdentity& identity,
    std::span<const SpeculationTraceRecord> records
) {
    if (std::filesystem::exists(path)) {
        fail(SpeculationTraceErrorCode::io_failure, "refusing to overwrite " + path.string());
    }
    validate_identity(identity);
    validate_records(records, false, identity.sampler);
    auto partial = path;
    partial += ".partial";
    if (std::filesystem::exists(partial)) {
        fail(SpeculationTraceErrorCode::io_failure,
            "refusing to overwrite " + partial.string());
    }
    try {
        FileWriter writer(partial);
        write_trace_payload(writer, identity, records);
        static_cast<void>(writer.finish());
        std::error_code error;
        std::filesystem::rename(partial, path, error);
        if (error) fail(SpeculationTraceErrorCode::io_failure, error.message());
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(partial, ignored);
        throw;
    }
}

AuthenticatedSpeculationTrace load_speculation_trace(
    std::span<const std::uint8_t> bytes
) {
    if (bytes.size() > speculation_trace_file_byte_max) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "file bytes");
    }
    Reader reader(bytes);
    if (!std::equal(trace_magic.begin(), trace_magic.end(), reader.raw(trace_magic.size()).begin())) {
        fail(SpeculationTraceErrorCode::invalid_magic, "magic");
    }
    const auto version = reader.u32();
    if (version != speculation_trace_schema_version) {
        fail(SpeculationTraceErrorCode::unsupported_version, std::to_string(version));
    }
    const auto stored_digest = reader.digest();
    const auto actual_digest = compute_envelope_digest(bytes);
    if (stored_digest != actual_digest) {
        fail(SpeculationTraceErrorCode::digest_mismatch, "envelope");
    }
    SpeculationTraceIdentity identity;
    identity.model = read_model_receipt(reader);
    identity.resident = read_resident_receipt(reader);
    identity.artifact = reader.digest();
    identity.tokenizer = reader.digest();
    identity.chat_template = reader.digest();
    identity.source_manifest = reader.digest();
    identity.registry = reader.digest();
    identity.sampler = read_sampler(reader);
    validate_identity(identity);
    const auto record_count = reader.u64();
    if (record_count > speculation_trace_record_max
        || record_count > bytes.size()) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "record count");
    }
    std::vector<SpeculationTraceRecord> records;
    records.reserve(static_cast<std::size_t>(record_count));
    for (std::uint64_t index = 0U; index < record_count; ++index) {
        SpeculationTraceRecord record;
        record.sequence_id = reader.u64();
        record.turn_index = reader.u32();
        record.kind = static_cast<SpeculationTraceRowKind>(reader.u8());
        const auto terminal = reader.u8();
        if (terminal > 1U) fail(SpeculationTraceErrorCode::invalid_record, "terminal flag");
        record.terminal = terminal != 0U;
        const auto parent = reader.u64();
        if (parent != no_parent) record.parent_record_index = parent;
        record.target_position = reader.u32();
        record.draft_position = reader.u32();
        record.token_id = reader.u32();
        const auto has_label = reader.u8();
        if (has_label > 1U) fail(SpeculationTraceErrorCode::invalid_record, "label flag");
        const auto label = reader.u32();
        if (has_label != 0U) record.target_label_token_id = label;
        record.sampler_draw_count = reader.u64();
        record.source.source_group = reader.digest();
        record.source.source_cluster = reader.digest();
        record.source.conversation = reader.digest();
        record.source.prompt_derivation = reader.digest();
        record.source.split = static_cast<SpeculationTraceSplit>(reader.u8());
        record.source.workload = static_cast<SpeculationTraceWorkload>(reader.u8());
        record.source.context_generation = reader.u32();
        record.source.context_flags = reader.u32();
        record.source.generation_seed = reader.u64();
        record.remaining_token_budget = reader.u32();
        const auto retrieval_eligible = reader.u8();
        const auto has_retrieval_proposal = reader.u8();
        if (retrieval_eligible > 1U || has_retrieval_proposal > 1U) {
            fail(SpeculationTraceErrorCode::invalid_record, "retrieval flags");
        }
        record.retrieval_eligible = retrieval_eligible != 0U;
        const auto retrieval_proposal = reader.u32();
        if (has_retrieval_proposal != 0U) {
            record.retrieval_proposal_token_id = retrieval_proposal;
        }
        const auto has_selection = reader.u8();
        if (has_selection > 1U) {
            fail(SpeculationTraceErrorCode::invalid_record, "selection flag");
        }
        if (has_selection != 0U) {
            SpeculationTraceSelection selection;
            selection.request = read_selection_request(reader);
            selection.sampler_pre.history = read_history(reader);
            selection.sampler_pre.draw_count = reader.u64();
            selection.sampler_post.history = read_history(reader);
            selection.sampler_post.draw_count = reader.u64();
            selection.draw.raw_engine_output_count = reader.u32();
            for (auto& raw : selection.draw.raw_engine_outputs) raw = reader.u64();
            selection.draw.draw_index = reader.u64();
            selection.draw.unit_interval_draw = reader.f64();
            selection.draw.mapped_draw = reader.f64();
            const auto consumes_draw = reader.u8();
            if (consumes_draw > 1U) {
                fail(SpeculationTraceErrorCode::invalid_record, "draw flag");
            }
            selection.draw.consumes_random_draw = consumes_draw != 0U;
            const auto compact_count = reader.u32();
            if (compact_count == 0U || compact_count > token_sampler_max_candidates) {
                fail(SpeculationTraceErrorCode::capacity_overflow, "compact candidates");
            }
            selection.compact_candidates.resize(compact_count);
            for (auto& candidate : selection.compact_candidates) {
                candidate.token = reader.u32();
                candidate.logit = reader.f32();
            }
            const auto prepared_count = reader.u32();
            if (prepared_count == 0U || prepared_count > token_sampler_max_top_k) {
                fail(SpeculationTraceErrorCode::capacity_overflow, "prepared candidates");
            }
            selection.prepared_candidates.resize(prepared_count);
            for (auto& candidate : selection.prepared_candidates) {
                candidate.token = reader.u32();
                candidate.compact_logit = reader.f32();
                candidate.probability = reader.f64();
                candidate.sampling_weight = reader.f64();
            }
            selection.total_sampling_weight = reader.f64();
            record.selection = std::move(selection);
        }
        const auto feature_count = reader.u16();
        if (feature_count == 0U || feature_count > speculation_trace_feature_max_per_record) {
            fail(SpeculationTraceErrorCode::capacity_overflow, "feature count");
        }
        record.features.reserve(feature_count);
        for (std::uint16_t feature_index = 0U;
             feature_index < feature_count; ++feature_index) {
            SpeculationTraceFeature feature;
            feature.state.execution_epoch = reader.u64();
            feature.state.arena_epoch = reader.u64();
            feature.state.bank_id = reader.u32();
            feature.state.child_id = reader.u32();
            feature.state.source_stage = static_cast<FeatureTapStage>(reader.u8());
            feature.state.source_layer = reader.u16();
            feature.state.coordinate.target_position = reader.u32();
            feature.state.coordinate.draft_position = reader.u32();
            feature.state.row_count = reader.u32();
            feature.state.feature_width = reader.u32();
            feature.dtype = static_cast<tensor::DType>(reader.u8());
            const auto byte_count = reader.u32();
            if (byte_count > speculation_trace_feature_byte_max) {
                fail(SpeculationTraceErrorCode::capacity_overflow, "feature bytes");
            }
            feature.digest = reader.digest();
            const auto payload = reader.raw(byte_count);
            feature.bytes.assign(payload.begin(), payload.end());
            record.features.push_back(std::move(feature));
        }
        records.push_back(std::move(record));
    }
    if (!reader.at_end()) fail(SpeculationTraceErrorCode::invalid_record, "trailing bytes");
    validate_records(records, true, identity.sampler);
    return AuthenticatedSpeculationTrace(
        std::move(identity), std::move(records), actual_digest
    );
}

AuthenticatedSpeculationTrace load_speculation_trace_file(
    const std::filesystem::path& path
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail(SpeculationTraceErrorCode::io_failure, path.string());
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > speculation_trace_file_byte_max) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "file bytes");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!bytes.empty() && !input.read(
        reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())
    )) {
        fail(SpeculationTraceErrorCode::io_failure, path.string());
    }
    return load_speculation_trace(bytes);
}

SpeculationTraceAudit audit_speculation_trace(
    const AuthenticatedSpeculationTrace& trace
) {
    validate_identity(trace.identity());
    validate_records(trace.records(), true, trace.identity().sampler);
    SpeculationTraceAudit audit;
    audit.record_count = trace.records().size();
    audit.envelope_digest = trace.envelope_digest();
    std::unordered_set<std::uint64_t> sequences;
    for (const auto& record : trace.records()) {
        sequences.insert(record.sequence_id);
        const auto split_index = static_cast<std::size_t>(record.source.split) - 1U;
        const auto workload_index = static_cast<std::size_t>(record.source.workload) - 1U;
        ++audit.split_record_counts[split_index];
        ++audit.workload_record_counts[workload_index];
        if (record.selection.has_value()) {
            ++audit.selection_count;
            audit.prepared_candidate_count += record.selection->prepared_candidates.size();
        }
        audit.maximum_target_position = std::max(
            audit.maximum_target_position, record.target_position
        );
        audit.feature_count += record.features.size();
        for (const auto& feature : record.features) {
            audit.feature_byte_count += feature.bytes.size();
        }
    }
    audit.sequence_count = sequences.size();
    return audit;
}

SpeculationTraceSelection make_speculation_trace_selection(
    std::span<const TokenLogit> compact_candidates,
    const TokenSamplerOptions& sampler_options,
    const TokenSelectionRequest& request,
    const SamplerCheckpoint& sampler_pre,
    const SamplerTicket& ticket,
    const SamplerCheckpoint& sampler_post,
    std::uint32_t selected_token
) {
    const auto prepared = prepare_token_distribution(
        compact_candidates, sampler_options, request
    );
    SpeculationTraceSelection selection;
    selection.request = request;
    selection.sampler_pre = {
        .history = sampler_pre.history(),
        .draw_count = sampler_pre.draw_count(),
    };
    selection.sampler_post = {
        .history = sampler_post.history(),
        .draw_count = sampler_post.draw_count(),
    };
    selection.draw.raw_engine_output_count = ticket.raw_engine_output_count();
    for (std::size_t index = 0U; index < ticket.raw_engine_output_count(); ++index) {
        selection.draw.raw_engine_outputs[index] = ticket.raw_engine_output(index);
    }
    selection.draw.draw_index = ticket.draw_index();
    selection.draw.unit_interval_draw = ticket.unit_interval_draw();
    selection.draw.mapped_draw = ticket.mapped_draw(prepared.total_sampling_weight);
    selection.draw.consumes_random_draw = ticket.consumes_random_draw();
    selection.compact_candidates.assign(compact_candidates.begin(), compact_candidates.end());
    selection.prepared_candidates.reserve(prepared.count);
    for (std::size_t index = 0U; index < prepared.count; ++index) {
        const auto& value = prepared.values[index];
        const auto compact = std::find_if(
            compact_candidates.begin(), compact_candidates.end(),
            [&](const TokenLogit& candidate) { return candidate.token == value.token; }
        );
        if (compact == compact_candidates.end()) {
            fail(SpeculationTraceErrorCode::invalid_record, "prepared token lacks compact logit");
        }
        selection.prepared_candidates.push_back({
            .token = value.token,
            .compact_logit = compact->logit,
            .probability = value.probability,
            .sampling_weight = value.sampling_weight,
        });
    }
    selection.total_sampling_weight = prepared.total_sampling_weight;
    validate_selection(selection, sampler_options, selected_token);
    return selection;
}

std::vector<SpeculationTraceFeature> make_speculation_trace_features(
    const TargetFeatureTraceBatch& batch,
    std::size_t row_index
) {
    if (batch.execution_epoch == 0U || batch.hidden_width == 0U
        || batch.dtype != tensor::DType::f16 || batch.taps.empty()
        || batch.taps.size() > speculation_trace_feature_max_per_record
        || batch.input_token_ids.size() != batch.absolute_positions.size()
        || row_index >= batch.absolute_positions.size()) {
        fail(SpeculationTraceErrorCode::invalid_record, "trace feature batch metadata");
    }
    const auto feature_bytes = static_cast<std::uint64_t>(batch.hidden_width)
        * sizeof(std::uint16_t);
    const auto row_bytes = feature_bytes * batch.taps.size();
    const auto required = row_bytes * batch.absolute_positions.size();
    if (feature_bytes > speculation_trace_feature_byte_max
        || required != batch.bytes.size()) {
        fail(SpeculationTraceErrorCode::capacity_overflow, "trace feature batch bytes");
    }
    std::vector<SpeculationTraceFeature> result;
    result.reserve(batch.taps.size());
    for (std::size_t tap = 0U; tap < batch.taps.size(); ++tap) {
        const auto offset = row_index * static_cast<std::size_t>(row_bytes)
            + tap * static_cast<std::size_t>(feature_bytes);
        SpeculationTraceFeature feature;
        feature.state = {
            .execution_epoch = batch.execution_epoch,
            .arena_epoch = 0U,
            .bank_id = static_cast<std::uint32_t>(tap + 1U),
            .child_id = 1U,
            .source_stage = batch.taps[tap].stage,
            .source_layer = batch.taps[tap].layer_index,
            .coordinate = {
                .target_position = batch.absolute_positions[row_index],
                .draft_position = batch.absolute_positions[row_index] + 1U,
            },
            .row_count = 1U,
            .feature_width = batch.hidden_width,
        };
        feature.dtype = batch.dtype;
        feature.bytes.assign(
            batch.bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            batch.bytes.begin() + static_cast<std::ptrdiff_t>(offset + feature_bytes)
        );
        feature.digest = runtime::sha256(feature.bytes);
        result.push_back(std::move(feature));
    }
    return result;
}

}  // namespace local_gguf_engine::engine
