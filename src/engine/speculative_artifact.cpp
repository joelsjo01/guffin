#include "local_gguf_engine/engine/speculative_artifact.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <system_error>

namespace local_gguf_engine::engine {
namespace {

constexpr std::array<std::uint8_t, 8> artifact_magic {
    'L', 'G', 'S', 'P', 'E', 'C', '3', 0,
};
constexpr std::size_t header_size_offset = 16U;
constexpr std::size_t envelope_digest_offset = 32U;
constexpr std::size_t envelope_digest_size = 32U;

[[nodiscard]] std::string error_message(
    SpeculativeArtifactErrorCode code,
    const std::string& detail
) {
    std::string message;
    switch (code) {
        case SpeculativeArtifactErrorCode::truncated: message = "truncated speculative artifact"; break;
        case SpeculativeArtifactErrorCode::invalid_magic: message = "invalid speculative artifact magic"; break;
        case SpeculativeArtifactErrorCode::unsupported_version: message = "unsupported speculative artifact version"; break;
        case SpeculativeArtifactErrorCode::invalid_descriptor: message = "invalid speculative artifact descriptor"; break;
        case SpeculativeArtifactErrorCode::capacity_overflow: message = "speculative artifact capacity overflow"; break;
        case SpeculativeArtifactErrorCode::digest_mismatch: message = "speculative artifact digest mismatch"; break;
        case SpeculativeArtifactErrorCode::io_failure: message = "speculative artifact I/O failure"; break;
        case SpeculativeArtifactErrorCode::incompatible_target: message = "speculative artifact target mismatch"; break;
    }
    if (!detail.empty()) message += ": " + detail;
    return message;
}

class ByteWriter final {
public:
    void u8(std::uint8_t value) { bytes.push_back(value); }
    void u16(std::uint16_t value) {
        for (std::uint32_t shift = 0U; shift < 16U; shift += 8U) {
            bytes.push_back(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u32(std::uint32_t value) {
        for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
            bytes.push_back(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void u64(std::uint64_t value) {
        for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
            bytes.push_back(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void digest(const runtime::Sha256Digest& value) {
        bytes.insert(bytes.end(), value.bytes.begin(), value.bytes.end());
    }
    void raw(std::span<const std::uint8_t> value) {
        bytes.insert(bytes.end(), value.begin(), value.end());
    }

    std::vector<std::uint8_t> bytes;
};

class ByteReader final {
public:
    explicit ByteReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1U);
        return bytes_[offset_++];
    }
    [[nodiscard]] std::uint16_t u16() {
        require(2U);
        auto value = std::uint16_t {0U};
        for (std::uint32_t index = 0U; index < 2U; ++index) {
            value |= static_cast<std::uint16_t>(bytes_[offset_++]) << (index * 8U);
        }
        return value;
    }
    [[nodiscard]] std::uint32_t u32() {
        require(4U);
        auto value = std::uint32_t {0U};
        for (std::uint32_t index = 0U; index < 4U; ++index) {
            value |= static_cast<std::uint32_t>(bytes_[offset_++]) << (index * 8U);
        }
        return value;
    }
    [[nodiscard]] std::uint64_t u64() {
        require(8U);
        auto value = std::uint64_t {0U};
        for (std::uint32_t index = 0U; index < 8U; ++index) {
            value |= static_cast<std::uint64_t>(bytes_[offset_++]) << (index * 8U);
        }
        return value;
    }
    [[nodiscard]] runtime::Sha256Digest digest() {
        require(32U);
        runtime::Sha256Digest result;
        std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_), 32U,
            result.bytes.begin());
        offset_ += 32U;
        return result;
    }
    [[nodiscard]] std::span<const std::uint8_t> raw(std::size_t count) {
        require(count);
        const auto result = bytes_.subspan(offset_, count);
        offset_ += count;
        return result;
    }
    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - std::min(offset_, bytes_.size())) {
            throw SpeculativeArtifactError(
                SpeculativeArtifactErrorCode::truncated, "encoded field"
            );
        }
    }

    std::span<const std::uint8_t> bytes_;
    std::size_t offset_ = 0U;
};

void patch_u64(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value) {
    if (offset > bytes.size() || bytes.size() - offset < 8U) {
        throw std::logic_error("speculative artifact internal patch range is invalid");
    }
    for (std::uint32_t index = 0U; index < 8U; ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
}

void write_config(ByteWriter& writer, const Gemma2ModelConfig& config) {
    for (const auto value : {
             config.context_length, config.embedding_length, config.block_count,
             config.feed_forward_length, config.attention_head_count,
             config.attention_head_count_kv, config.attention_key_length,
             config.attention_value_length, config.sliding_window, config.vocab_size,
         }) writer.u32(value);
    writer.u32(std::bit_cast<std::uint32_t>(config.rms_norm_epsilon));
    writer.u32(std::bit_cast<std::uint32_t>(config.attention_logit_softcap));
    writer.u32(std::bit_cast<std::uint32_t>(config.final_logit_softcap));
    writer.u32(config.uses_tied_output_weight ? 1U : 0U);
}

[[nodiscard]] Gemma2ModelConfig read_config(ByteReader& reader) {
    Gemma2ModelConfig config;
    config.context_length = reader.u32();
    config.embedding_length = reader.u32();
    config.block_count = reader.u32();
    config.feed_forward_length = reader.u32();
    config.attention_head_count = reader.u32();
    config.attention_head_count_kv = reader.u32();
    config.attention_key_length = reader.u32();
    config.attention_value_length = reader.u32();
    config.sliding_window = reader.u32();
    config.vocab_size = reader.u32();
    config.rms_norm_epsilon = std::bit_cast<float>(reader.u32());
    config.attention_logit_softcap = std::bit_cast<float>(reader.u32());
    config.final_logit_softcap = std::bit_cast<float>(reader.u32());
    const auto tied = reader.u32();
    if (tied > 1U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor, "target tied-weight flag"
        );
    }
    config.uses_tied_output_weight = tied != 0U;
    return config;
}

[[nodiscard]] bool valid_dtype(tensor::DType dtype) noexcept {
    switch (dtype) {
        case tensor::DType::f32:
        case tensor::DType::f16:
        case tensor::DType::q8_0:
        case tensor::DType::q4_k:
        case tensor::DType::q6_k:
            return true;
    }
    return false;
}

[[nodiscard]] bool valid_draft_element_type(DraftArtifactElementType type) noexcept {
    switch (type) {
        case DraftArtifactElementType::f16:
        case DraftArtifactElementType::q8_0:
        case DraftArtifactElementType::q6_k:
        case DraftArtifactElementType::q4_k:
        case DraftArtifactElementType::u32:
            return true;
    }
    return false;
}

[[nodiscard]] std::uint64_t draft_tensor_required_bytes(
    const DraftArtifactTensorSpec& tensor
) {
    if (tensor.dimension_0 == 0U || tensor.dimension_1 == 0U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "draft tensor has an empty dimension"
        );
    }
    if (tensor.dimension_0 > std::numeric_limits<std::uint64_t>::max()
            / tensor.dimension_1) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::capacity_overflow, "draft tensor elements"
        );
    }
    const auto elements = static_cast<std::uint64_t>(tensor.dimension_0)
        * tensor.dimension_1;
    auto block_elements = std::uint64_t {1U};
    auto block_bytes = std::uint64_t {0U};
    switch (tensor.element_type) {
        case DraftArtifactElementType::f16: block_bytes = 2U; break;
        case DraftArtifactElementType::u32: block_bytes = 4U; break;
        case DraftArtifactElementType::q8_0:
            block_elements = 32U;
            block_bytes = 34U;
            break;
        case DraftArtifactElementType::q4_k:
            block_elements = 256U;
            block_bytes = 144U;
            break;
        case DraftArtifactElementType::q6_k:
            block_elements = 256U;
            block_bytes = 210U;
            break;
    }
    if (tensor.dimension_0 % block_elements != 0U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "draft packed tensor dimension-0 alignment"
        );
    }
    const auto blocks = elements / block_elements;
    if (blocks > std::numeric_limits<std::uint64_t>::max() / block_bytes) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::capacity_overflow, "draft tensor bytes"
        );
    }
    return blocks * block_bytes;
}

void validate_draft_tensor(
    const DraftArtifactTensorSpec& tensor,
    std::span<const std::uint8_t> payload,
    bool require_payload_digests,
    const char* name
) {
    if (tensor.storage == DraftArtifactTensorStorage::absent) {
        if (tensor.dimension_0 != 0U || tensor.dimension_1 != 0U
            || tensor.payload_offset != 0U || tensor.payload_byte_count != 0U
            || !tensor.payload_digest.is_zero()) {
            throw SpeculativeArtifactError(
                SpeculativeArtifactErrorCode::invalid_descriptor,
                std::string(name) + " absent tensor carries storage"
            );
        }
        return;
    }
    if (!valid_draft_element_type(tensor.element_type)) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            std::string(name) + " element type"
        );
    }
    const auto required = draft_tensor_required_bytes(tensor);
    if (tensor.storage == DraftArtifactTensorStorage::resident_target_output_q8_0) {
        if (tensor.element_type != DraftArtifactElementType::q8_0
            || tensor.payload_offset != 0U || tensor.payload_byte_count != 0U
            || !tensor.payload_digest.is_zero()) {
            throw SpeculativeArtifactError(
                SpeculativeArtifactErrorCode::invalid_descriptor,
                std::string(name) + " resident alias"
            );
        }
        return;
    }
    if (tensor.storage != DraftArtifactTensorStorage::payload
        || tensor.payload_byte_count != required
        || tensor.payload_offset > payload.size()
        || tensor.payload_byte_count > payload.size() - tensor.payload_offset) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            std::string(name) + " payload extent"
        );
    }
    if (require_payload_digests) {
        const auto slice = payload.subspan(
            static_cast<std::size_t>(tensor.payload_offset),
            static_cast<std::size_t>(tensor.payload_byte_count)
        );
        if (runtime::sha256(slice) != tensor.payload_digest) {
            throw SpeculativeArtifactError(
                SpeculativeArtifactErrorCode::digest_mismatch,
                std::string("draft tensor ") + name
            );
        }
    }
}

void validate_draft_vocabulary(
    const SpeculativeArtifactDescriptor& descriptor,
    std::span<const std::uint8_t> payload,
    bool require_payload_digests
) {
    const auto& draft = descriptor.draft_vocabulary;
    if (!draft.present()) return;
    if (draft.version != draft_vocabulary_artifact_version
        || (draft.selector_width != 112U && draft.selector_width != 224U
            && draft.selector_width != 448U)
        || draft.selector_storage_width < draft.selector_width
        || draft.selector_storage_width > 512U
        || draft.semantic_width == 0U || draft.semantic_width > 16'384U
        || draft.target_width != descriptor.target.config.embedding_length
        || draft.vocabulary_size != descriptor.target.config.vocab_size
        || draft.vocabulary_size > draft_vocabulary_maximum_vocabulary
        || draft.state_capacity != draft_vocabulary_external_state_max
        || draft.candidate_capacity == 0U
        || draft.candidate_capacity > draft.vocabulary_size
        || (draft.candidate_capacity != draft.vocabulary_size
            && (draft.candidate_capacity < 1'024U
                || draft.candidate_capacity > 131'072U
                || !std::has_single_bit(draft.candidate_capacity)))
        || draft.transition_width > 64U
        || draft.rank_entry_capacity > 524'288U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "draft vocabulary dimensions or capacities"
        );
    }
    constexpr std::array<const char*, draft_vocabulary_tensor_count> names {
        "selector projection", "selector head", "candidate head",
        "target up projection", "transition counts", "transition tokens",
        "proposal exp LUT", "Arm-S rank-8 down", "Arm-S rank-8 up",
    };
    for (std::size_t index = 0U; index < draft.tensors.size(); ++index) {
        validate_draft_tensor(
            draft.tensors[index], payload, require_payload_digests, names[index]
        );
    }
    const auto& selector_projection = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::selector_projection
    )];
    const auto& selector_head = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::selector_head
    )];
    const auto& candidate_head = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::candidate_head
    )];
    const auto& target_up = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::target_up_projection
    )];
    const auto& transition_counts = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::transition_counts
    )];
    const auto& transition_tokens = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::transition_tokens
    )];
    const auto& lut = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::proposal_exp_lut
    )];
    const auto payload_only_f16_matrix = [](const DraftArtifactTensorSpec& tensor,
                                            std::uint32_t d0,
                                            std::uint32_t d1) {
        return tensor.storage == DraftArtifactTensorStorage::payload
            && tensor.element_type == DraftArtifactElementType::f16
            && tensor.dimension_0 == d0 && tensor.dimension_1 == d1;
    };
    const auto scoring_type = [](DraftArtifactElementType type) {
        return type == DraftArtifactElementType::f16
            || type == DraftArtifactElementType::q8_0
            || type == DraftArtifactElementType::q6_k
            || type == DraftArtifactElementType::q4_k;
    };
    if ((draft.semantic_width == draft.selector_storage_width
            ? selector_projection.storage != DraftArtifactTensorStorage::absent
            : !payload_only_f16_matrix(
                selector_projection, draft.semantic_width,
                draft.selector_storage_width))
        || selector_head.storage != DraftArtifactTensorStorage::payload
        || !scoring_type(selector_head.element_type)
        || selector_head.dimension_0 != draft.selector_storage_width
        || selector_head.dimension_1 != draft.vocabulary_size
        || (draft.semantic_width == draft.target_width
            ? target_up.storage != DraftArtifactTensorStorage::absent
            : !payload_only_f16_matrix(target_up, draft.semantic_width, draft.target_width))
        || candidate_head.dimension_0 != draft.target_width
        || candidate_head.dimension_1 != draft.vocabulary_size
        || !scoring_type(candidate_head.element_type)
        || (candidate_head.storage != DraftArtifactTensorStorage::payload
            && candidate_head.storage
                != DraftArtifactTensorStorage::resident_target_output_q8_0)
        || (candidate_head.storage == DraftArtifactTensorStorage::resident_target_output_q8_0
            && descriptor.target.output_head_policy != 2U)
        || lut.storage != DraftArtifactTensorStorage::payload
        || lut.element_type != DraftArtifactElementType::u32
        || lut.dimension_0 != 4'097U || lut.dimension_1 != 1U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "draft vocabulary tensor roles"
        );
    }
    const auto transitions_absent = draft.transition_width == 0U;
    if (transitions_absent
        ? (transition_counts.storage != DraftArtifactTensorStorage::absent
            || transition_tokens.storage != DraftArtifactTensorStorage::absent)
        : (transition_counts.storage != DraftArtifactTensorStorage::payload
            || transition_counts.element_type != DraftArtifactElementType::u32
            || transition_counts.dimension_0 != draft.vocabulary_size
            || transition_counts.dimension_1 != 1U
            || transition_tokens.storage != DraftArtifactTensorStorage::payload
            || transition_tokens.element_type != DraftArtifactElementType::u32
            || transition_tokens.dimension_0 != draft.vocabulary_size
            || transition_tokens.dimension_1 != draft.transition_width)) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "draft transition tensors"
        );
    }
    const auto& arm_down = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::arm_s_rank8_down
    )];
    const auto& arm_up = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::arm_s_rank8_up
    )];
    const auto arm_absent = arm_down.storage == DraftArtifactTensorStorage::absent
        && arm_up.storage == DraftArtifactTensorStorage::absent;
    if (!arm_absent
        && (!payload_only_f16_matrix(arm_down, draft.semantic_width, 8U)
            || !payload_only_f16_matrix(arm_up, 8U, draft.target_width))) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "local Arm-S rank-8 interpretation tensors"
        );
    }
}

[[nodiscard]] std::uint32_t tap_order(const FeatureTapSpec& tap) {
    switch (tap.stage) {
        case FeatureTapStage::embedding_residual: return 0U;
        case FeatureTapStage::layer_residual:
            return 1U + tap.layer_index;
        case FeatureTapStage::final_normalized:
            return std::numeric_limits<std::uint32_t>::max();
    }
    throw SpeculativeArtifactError(
        SpeculativeArtifactErrorCode::invalid_descriptor, "feature tap stage"
    );
}

void validate_descriptor(
    const SpeculativeArtifactDescriptor& descriptor,
    std::span<const std::uint8_t> payload,
    bool require_payload_digests
) {
    const bool learned = descriptor.schema_version == 3U
        && descriptor.kind == SpeculativeArtifactKind::trained_candidate;
    if (!learned && descriptor.schema_version != speculative_artifact_schema_version) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::unsupported_version,
            std::to_string(descriptor.schema_version)
        );
    }
    if (descriptor.artifact_id == 0U || (!learned && descriptor.feature_bank_count == 0U)
        || descriptor.feature_bank_count > descriptor.feature_banks.size()) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor, "artifact id or feature-bank count"
        );
    }
    if (payload.size() > speculative_artifact_payload_byte_max) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::capacity_overflow, "artifact payload"
        );
    }
    if (descriptor.target.model.version != engine_model_payload_receipt_version
        || descriptor.target.model.whole_gguf_byte_count == 0U
        || descriptor.target.model.source_tensor_count == 0U
        || descriptor.target.model.whole_gguf.is_zero()
        || descriptor.target.model.canonical_consumed_metadata.is_zero()
        || descriptor.target.model.source_tensors.is_zero()
        || descriptor.target.resident.version != 1U
        || descriptor.target.resident.allocation_count == 0U
        || descriptor.target.resident.byte_count == 0U
        || descriptor.target.resident.bytes.is_zero()) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor, "target identity"
        );
    }
    if (learned) {
        const auto fail = [](const char* detail) {
            throw SpeculativeArtifactError(SpeculativeArtifactErrorCode::invalid_descriptor, detail);
        };
        const auto& r = descriptor.recurrent;
        const auto& c = descriptor.target.config;
        constexpr std::array<std::uint32_t, 7> horizons {3, 4, 7, 8, 15, 16, 23};
        if (r.version != 1U || r.role != 1U || r.numeric_contract != 1U
            || r.width == 0U || r.width > 3'584U
            || std::find(horizons.begin(), horizons.end(), r.horizon) == horizons.end()
            || c.embedding_length == 0U || c.embedding_length > 4'096U
            || c.vocab_size == 0U || c.vocab_size > 262'144U
            || !c.uses_tied_output_weight || descriptor.target.output_head_policy != 2U
            || !std::isfinite(c.rms_norm_epsilon) || c.rms_norm_epsilon <= 0.0F
            || !std::isfinite(c.final_logit_softcap) || c.final_logit_softcap <= 0.0F
            || r.top_k == 0U || r.top_k > std::min(c.vocab_size, 256U)
            || !std::isfinite(r.temperature) || r.temperature <= 0.0F
            || !std::isfinite(r.top_p) || r.top_p <= 0.0F || r.top_p > 1.0F
            || !std::isfinite(r.min_p) || r.min_p < 0.0F || r.min_p > 1.0F
            || descriptor.feature_bank_count != 0U
            || descriptor.draft_cache != DraftCacheSpec {}
            || descriptor.draft_vocabulary != DraftVocabularyArtifactSpec {}) {
            fail("trained recurrent mechanism, numeric, sampler, or ownership contract");
        }
        for (const auto& digest : r.provenance) {
            if (digest.is_zero()) fail("trained recurrent provenance is incomplete");
        }
        const auto w = r.width, d = c.embedding_length;
        const std::array<std::array<std::uint32_t, 2>, recurrent_tensor_count> shapes {{
            {w,d}, {1,w}, {w,d}, {w,2*w}, {1,w},
            {w,2*w}, {1,w}, {d,w}, {1,w}, {1,1},
        }};
        std::uint64_t end = 0;
        for (std::size_t i = 0; i < r.tensors.size(); ++i) {
            const auto& t = r.tensors[i];
            if (t.rows != shapes[i][0] || t.columns != shapes[i][1]
                || t.offset != end || t.bytes != std::uint64_t(t.rows)*t.columns*4U
                || t.offset > payload.size() || t.bytes > payload.size()-t.offset) {
                fail("trained recurrent tensor shape, contiguous layout, or extent");
            }
            const auto bytes = payload.subspan(static_cast<std::size_t>(t.offset),
                                              static_cast<std::size_t>(t.bytes));
            if (runtime::sha256(bytes) != t.digest) {
                throw SpeculativeArtifactError(SpeculativeArtifactErrorCode::digest_mismatch,
                                               "trained recurrent tensor " + std::to_string(i));
            }
            for (std::size_t at = 0; at < bytes.size(); at += 4) {
                std::uint32_t bits = 0;
                for (unsigned b = 0; b < 4; ++b) bits |= std::uint32_t(bytes[at+b]) << (8*b);
                if (!std::isfinite(std::bit_cast<float>(bits))) fail("nonfinite recurrent weight");
            }
            end += t.bytes;
        }
        if (end != payload.size()) fail("unclaimed recurrent payload bytes");
        return;
    }
    if (descriptor.recurrent.version != 0U) {
        throw SpeculativeArtifactError(SpeculativeArtifactErrorCode::invalid_descriptor,
                                       "recurrent payload requires trained schema 3");
    }
    const auto& cache = descriptor.draft_cache;
    if (cache.layer_count == 0U || cache.stream_count == 0U
        || cache.stream_count > 256U
        || cache.key_width == 0U || cache.value_width == 0U
        || cache.key_width > 4'096U || cache.value_width > 4'096U
        || cache.maximum_depth == 0U || cache.maximum_depth > 23U
        || cache.maximum_active_width == 0U
        || cache.maximum_active_width > 256U
        || cache.linear_row_capacity == 0U || cache.linear_row_capacity > 23U
        || cache.tree_row_capacity == 0U || cache.tree_row_capacity > 1'025U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor, "draft-cache bounds"
        );
    }

    auto previous_bank = std::uint32_t {0U};
    std::unordered_set<std::uint32_t> child_ids;
    for (std::uint16_t bank_index = 0U;
         bank_index < descriptor.feature_bank_count; ++bank_index) {
        const auto& bank = descriptor.feature_banks[bank_index];
        if (bank.bank_id == 0U || (bank_index != 0U && bank.bank_id <= previous_bank)
            || bank.fusion != FeatureFusionKind::concatenate
            || bank.tap_count == 0U || bank.tap_count > bank.taps.size()
            || bank.child_count == 0U || bank.child_count > bank.children.size()) {
            throw SpeculativeArtifactError(
                SpeculativeArtifactErrorCode::invalid_descriptor, "feature bank"
            );
        }
        previous_bank = bank.bank_id;
        auto previous_tap = std::uint32_t {0U};
        for (std::uint16_t tap_index = 0U; tap_index < bank.tap_count; ++tap_index) {
            const auto order = tap_order(bank.taps[tap_index]);
            if (bank.taps[tap_index].stage == FeatureTapStage::layer_residual
                && bank.taps[tap_index].layer_index
                    >= descriptor.target.config.block_count) {
                throw SpeculativeArtifactError(
                    SpeculativeArtifactErrorCode::invalid_descriptor,
                    "feature tap layer exceeds target"
                );
            }
            if (tap_index != 0U && order <= previous_tap) {
                throw SpeculativeArtifactError(
                    SpeculativeArtifactErrorCode::invalid_descriptor,
                    "feature taps are not in causal execution order"
                );
            }
            previous_tap = order;
        }
        auto previous_child = std::uint32_t {0U};
        for (std::uint16_t child_index = 0U; child_index < bank.child_count; ++child_index) {
            const auto& child = bank.children[child_index];
            if (child.child_id == 0U
                || !child_ids.insert(child.child_id).second
                || (child_index != 0U && child.child_id <= previous_child)
                || child.output_width == 0U
                || child.output_width > feature_projection_width_max
                || !valid_dtype(child.weight_dtype)
                || (child.schedule != FeatureProjectionSchedule::eager
                    && child.schedule != FeatureProjectionSchedule::lazy)
                || child.payload_byte_count == 0U
                || child.payload_offset > payload.size()
                || child.payload_byte_count > payload.size() - child.payload_offset) {
                throw SpeculativeArtifactError(
                    SpeculativeArtifactErrorCode::invalid_descriptor, "feature child"
                );
            }
            if (descriptor.kind == SpeculativeArtifactKind::fixture_feature_cache) {
                const auto fused_width = static_cast<std::uint64_t>(
                    descriptor.target.config.embedding_length
                ) * bank.tap_count;
                const auto required_bytes = fused_width
                    * child.output_width * sizeof(std::uint16_t);
                if (child.weight_dtype != tensor::DType::f16
                    || fused_width == 0U
                    || required_bytes != child.payload_byte_count) {
                    throw SpeculativeArtifactError(
                        SpeculativeArtifactErrorCode::invalid_descriptor,
                        "fixture projection shape"
                    );
                }
            }
            previous_child = child.child_id;
            if (require_payload_digests) {
                const auto slice = payload.subspan(
                    static_cast<std::size_t>(child.payload_offset),
                    static_cast<std::size_t>(child.payload_byte_count)
                );
                if (runtime::sha256(slice) != child.payload_digest) {
                    throw SpeculativeArtifactError(
                        SpeculativeArtifactErrorCode::digest_mismatch,
                        "feature child " + std::to_string(child.child_id)
                    );
                }
            }
        }
    }
    validate_draft_vocabulary(descriptor, payload, require_payload_digests);
}

void write_target(ByteWriter& writer, const SpeculativeArtifactTargetIdentity& target) {
    writer.u32(target.model.version);
    writer.u64(target.model.whole_gguf_byte_count);
    writer.u32(target.model.source_tensor_count);
    writer.digest(target.model.whole_gguf);
    writer.digest(target.model.canonical_consumed_metadata);
    writer.digest(target.model.source_tensors);
    writer.u32(target.resident.version);
    writer.u32(target.resident.allocation_count);
    writer.u64(target.resident.byte_count);
    writer.digest(target.resident.bytes);
    write_config(writer, target.config);
    writer.u32(target.output_head_policy);
}

[[nodiscard]] SpeculativeArtifactTargetIdentity read_target(ByteReader& reader) {
    SpeculativeArtifactTargetIdentity target;
    target.model.version = reader.u32();
    target.model.whole_gguf_byte_count = reader.u64();
    target.model.source_tensor_count = reader.u32();
    target.model.whole_gguf = reader.digest();
    target.model.canonical_consumed_metadata = reader.digest();
    target.model.source_tensors = reader.digest();
    target.resident.version = reader.u32();
    target.resident.allocation_count = reader.u32();
    target.resident.byte_count = reader.u64();
    target.resident.bytes = reader.digest();
    target.config = read_config(reader);
    target.output_head_policy = reader.u32();
    return target;
}

void write_bank(ByteWriter& writer, const FeatureBankSpec& bank) {
    writer.u32(bank.bank_id);
    writer.u8(static_cast<std::uint8_t>(bank.fusion));
    writer.u16(bank.tap_count);
    writer.u16(bank.child_count);
    for (std::uint16_t index = 0U; index < bank.tap_count; ++index) {
        writer.u8(static_cast<std::uint8_t>(bank.taps[index].stage));
        writer.u16(bank.taps[index].layer_index);
    }
    for (std::uint16_t index = 0U; index < bank.child_count; ++index) {
        const auto& child = bank.children[index];
        writer.u32(child.child_id);
        writer.u32(child.output_width);
        writer.u8(static_cast<std::uint8_t>(child.weight_dtype));
        writer.u8(static_cast<std::uint8_t>(child.schedule));
        writer.u64(child.payload_offset);
        writer.u64(child.payload_byte_count);
        writer.digest(child.payload_digest);
    }
}

[[nodiscard]] FeatureBankSpec read_bank(ByteReader& reader) {
    FeatureBankSpec bank;
    bank.bank_id = reader.u32();
    bank.fusion = static_cast<FeatureFusionKind>(reader.u8());
    bank.tap_count = reader.u16();
    bank.child_count = reader.u16();
    if (bank.tap_count > bank.taps.size() || bank.child_count > bank.children.size()) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::capacity_overflow, "feature bank entries"
        );
    }
    for (std::uint16_t index = 0U; index < bank.tap_count; ++index) {
        bank.taps[index].stage = static_cast<FeatureTapStage>(reader.u8());
        bank.taps[index].layer_index = reader.u16();
    }
    for (std::uint16_t index = 0U; index < bank.child_count; ++index) {
        auto& child = bank.children[index];
        child.child_id = reader.u32();
        child.output_width = reader.u32();
        child.weight_dtype = static_cast<tensor::DType>(reader.u8());
        child.schedule = static_cast<FeatureProjectionSchedule>(reader.u8());
        child.payload_offset = reader.u64();
        child.payload_byte_count = reader.u64();
        child.payload_digest = reader.digest();
    }
    return bank;
}

void write_cache(ByteWriter& writer, const DraftCacheSpec& cache) {
    writer.u16(cache.layer_count);
    writer.u16(cache.stream_count);
    writer.u16(cache.key_width);
    writer.u16(cache.value_width);
    writer.u16(cache.maximum_depth);
    writer.u16(cache.maximum_active_width);
    writer.u32(cache.linear_row_capacity);
    writer.u32(cache.tree_row_capacity);
}

[[nodiscard]] DraftCacheSpec read_cache(ByteReader& reader) {
    return DraftCacheSpec {
        .layer_count = reader.u16(),
        .stream_count = reader.u16(),
        .key_width = reader.u16(),
        .value_width = reader.u16(),
        .maximum_depth = reader.u16(),
        .maximum_active_width = reader.u16(),
        .linear_row_capacity = reader.u32(),
        .tree_row_capacity = reader.u32(),
    };
}

void write_draft_tensor(ByteWriter& writer, const DraftArtifactTensorSpec& tensor) {
    writer.u8(static_cast<std::uint8_t>(tensor.storage));
    writer.u8(static_cast<std::uint8_t>(tensor.element_type));
    writer.u32(tensor.dimension_0);
    writer.u32(tensor.dimension_1);
    writer.u64(tensor.payload_offset);
    writer.u64(tensor.payload_byte_count);
    writer.digest(tensor.payload_digest);
}

[[nodiscard]] DraftArtifactTensorSpec read_draft_tensor(ByteReader& reader) {
    DraftArtifactTensorSpec tensor;
    tensor.storage = static_cast<DraftArtifactTensorStorage>(reader.u8());
    tensor.element_type = static_cast<DraftArtifactElementType>(reader.u8());
    tensor.dimension_0 = reader.u32();
    tensor.dimension_1 = reader.u32();
    tensor.payload_offset = reader.u64();
    tensor.payload_byte_count = reader.u64();
    tensor.payload_digest = reader.digest();
    return tensor;
}

void write_draft_vocabulary(ByteWriter& writer, const DraftVocabularyArtifactSpec& draft) {
    writer.u32(draft.version);
    writer.u32(draft.semantic_width);
    writer.u32(draft.selector_width);
    writer.u32(draft.selector_storage_width);
    writer.u32(draft.target_width);
    writer.u32(draft.vocabulary_size);
    writer.u32(draft.state_capacity);
    writer.u32(draft.candidate_capacity);
    writer.u32(draft.transition_width);
    writer.u32(draft.rank_entry_capacity);
    for (const auto& tensor : draft.tensors) write_draft_tensor(writer, tensor);
}

[[nodiscard]] DraftVocabularyArtifactSpec read_draft_vocabulary(ByteReader& reader) {
    DraftVocabularyArtifactSpec draft;
    draft.version = reader.u32();
    draft.semantic_width = reader.u32();
    draft.selector_width = reader.u32();
    draft.selector_storage_width = reader.u32();
    draft.target_width = reader.u32();
    draft.vocabulary_size = reader.u32();
    draft.state_capacity = reader.u32();
    draft.candidate_capacity = reader.u32();
    draft.transition_width = reader.u32();
    draft.rank_entry_capacity = reader.u32();
    for (auto& tensor : draft.tensors) tensor = read_draft_tensor(reader);
    return draft;
}

[[nodiscard]] runtime::Sha256Digest envelope_digest(
    std::span<const std::uint8_t> bytes
) {
    if (bytes.size() < envelope_digest_offset + envelope_digest_size) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::truncated, "envelope digest"
        );
    }
    runtime::Sha256 hash;
    hash.update(bytes.first(envelope_digest_offset));
    constexpr std::array<std::uint8_t, envelope_digest_size> zero {};
    hash.update(zero);
    hash.update(bytes.subspan(envelope_digest_offset + envelope_digest_size));
    return hash.finalize();
}

void write_recurrent(ByteWriter& out, const LearnedRecurrentSpec& r) {
    for (auto v : {r.version, r.role, r.numeric_contract, r.width, r.horizon, r.top_k}) out.u32(v);
    for (auto v : {r.temperature, r.top_p, r.min_p}) out.u32(std::bit_cast<std::uint32_t>(v));
    for (const auto& v : r.provenance) out.digest(v);
    for (const auto& t : r.tensors) {
        out.u32(t.rows); out.u32(t.columns); out.u64(t.offset); out.u64(t.bytes); out.digest(t.digest);
    }
}

LearnedRecurrentSpec read_recurrent(ByteReader& in) {
    LearnedRecurrentSpec r;
    r.version=in.u32(); r.role=in.u32(); r.numeric_contract=in.u32();
    r.width=in.u32(); r.horizon=in.u32(); r.top_k=in.u32();
    r.temperature=std::bit_cast<float>(in.u32()); r.top_p=std::bit_cast<float>(in.u32());
    r.min_p=std::bit_cast<float>(in.u32());
    for (auto& v : r.provenance) v=in.digest();
    for (auto& t : r.tensors) {
        t.rows=in.u32(); t.columns=in.u32(); t.offset=in.u64(); t.bytes=in.u64(); t.digest=in.digest();
    }
    return r;
}

}  // namespace

SpeculativeArtifactError::SpeculativeArtifactError(
    SpeculativeArtifactErrorCode code,
    std::string detail
)
    : std::runtime_error(error_message(code, detail)), code_(code) {}

SpeculativeArtifactErrorCode SpeculativeArtifactError::code() const noexcept {
    return code_;
}

AuthenticatedSpeculativeArtifact::AuthenticatedSpeculativeArtifact(
    SpeculativeArtifactDescriptor descriptor,
    std::vector<std::uint8_t> payload,
    runtime::Sha256Digest envelope_digest_value
)
    : descriptor_(std::move(descriptor)),
      payload_(std::move(payload)),
      envelope_digest_(envelope_digest_value) {}

const SpeculativeArtifactDescriptor& AuthenticatedSpeculativeArtifact::descriptor() const noexcept {
    return descriptor_;
}

std::span<const std::uint8_t> AuthenticatedSpeculativeArtifact::payload() const noexcept {
    return payload_;
}

const runtime::Sha256Digest& AuthenticatedSpeculativeArtifact::envelope_digest() const noexcept {
    return envelope_digest_;
}

std::vector<std::uint8_t> encode_fixture_speculative_artifact(
    SpeculativeArtifactDescriptor descriptor,
    std::span<const std::uint8_t> payload
) {
    if (descriptor.kind != SpeculativeArtifactKind::fixture_feature_cache) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "fixture writer accepts fixture_feature_cache only"
        );
    }
    for (std::uint16_t bank = 0U; bank < descriptor.feature_bank_count; ++bank) {
        for (std::uint16_t child = 0U;
             child < descriptor.feature_banks[bank].child_count; ++child) {
            auto& projection = descriptor.feature_banks[bank].children[child];
            if (projection.payload_offset > payload.size()
                || projection.payload_byte_count > payload.size() - projection.payload_offset) {
                throw SpeculativeArtifactError(
                    SpeculativeArtifactErrorCode::invalid_descriptor,
                    "fixture projection payload range"
                );
            }
            projection.payload_digest = runtime::sha256(payload.subspan(
                static_cast<std::size_t>(projection.payload_offset),
                static_cast<std::size_t>(projection.payload_byte_count)
            ));
        }
    }
    if (descriptor.draft_vocabulary.present()) {
        for (auto& tensor : descriptor.draft_vocabulary.tensors) {
            if (tensor.storage != DraftArtifactTensorStorage::payload) continue;
            if (tensor.payload_offset > payload.size()
                || tensor.payload_byte_count > payload.size() - tensor.payload_offset) {
                throw SpeculativeArtifactError(
                    SpeculativeArtifactErrorCode::invalid_descriptor,
                    "draft tensor payload range"
                );
            }
            tensor.payload_digest = runtime::sha256(payload.subspan(
                static_cast<std::size_t>(tensor.payload_offset),
                static_cast<std::size_t>(tensor.payload_byte_count)
            ));
        }
    }
    validate_descriptor(descriptor, payload, true);

    ByteWriter writer;
    writer.raw(artifact_magic);
    writer.u32(descriptor.schema_version);
    writer.u32(static_cast<std::uint32_t>(descriptor.kind));
    writer.u64(0U);
    writer.u64(payload.size());
    writer.digest({});
    writer.u64(descriptor.artifact_id);
    write_target(writer, descriptor.target);
    writer.u16(descriptor.feature_bank_count);
    for (std::uint16_t index = 0U; index < descriptor.feature_bank_count; ++index) {
        write_bank(writer, descriptor.feature_banks[index]);
    }
    write_cache(writer, descriptor.draft_cache);
    write_draft_vocabulary(writer, descriptor.draft_vocabulary);
    patch_u64(writer.bytes, header_size_offset, writer.bytes.size());
    writer.raw(payload);
    const auto digest = envelope_digest(writer.bytes);
    std::copy(digest.bytes.begin(), digest.bytes.end(),
        writer.bytes.begin() + static_cast<std::ptrdiff_t>(envelope_digest_offset));
    return writer.bytes;
}

std::vector<std::uint8_t> encode_recurrent_weights_artifact(
    SpeculativeArtifactDescriptor descriptor, std::span<const std::uint8_t> payload
) {
    if (descriptor.schema_version != 3U || descriptor.kind != SpeculativeArtifactKind::trained_candidate) {
        throw SpeculativeArtifactError(SpeculativeArtifactErrorCode::invalid_descriptor,
                                       "recurrent writer requires trained schema 3");
    }
    validate_descriptor(descriptor, payload, true);
    ByteWriter writer;
    writer.raw(artifact_magic); writer.u32(3U); writer.u32(2U);
    writer.u64(0U); writer.u64(payload.size()); writer.digest({});
    writer.u64(descriptor.artifact_id); write_target(writer, descriptor.target);
    write_recurrent(writer, descriptor.recurrent);
    patch_u64(writer.bytes, header_size_offset, writer.bytes.size());
    writer.raw(payload);
    const auto digest = envelope_digest(writer.bytes);
    std::copy(digest.bytes.begin(), digest.bytes.end(), writer.bytes.begin()+32);
    return writer.bytes;
}

void write_fixture_speculative_artifact(
    const std::filesystem::path& path,
    SpeculativeArtifactDescriptor descriptor,
    std::span<const std::uint8_t> payload
) {
    std::error_code exists_error;
    const auto exists = std::filesystem::exists(path, exists_error);
    if (exists_error || exists) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::io_failure,
            exists ? "refusing to overwrite " + path.string()
                   : "cannot inspect " + path.string()
        );
    }
    const auto bytes = encode_fixture_speculative_artifact(std::move(descriptor), payload);
    std::ofstream stream(path, std::ios::binary | std::ios::out);
    if (!stream) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::io_failure, "cannot create " + path.string()
        );
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::io_failure, "cannot write " + path.string()
        );
    }
}

AuthenticatedSpeculativeArtifact load_speculative_artifact(
    std::span<const std::uint8_t> bytes
) {
    ByteReader reader(bytes);
    if (!std::equal(artifact_magic.begin(), artifact_magic.end(), reader.raw(artifact_magic.size()).begin())) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_magic, "expected LGSPEC3"
        );
    }
    SpeculativeArtifactDescriptor descriptor;
    descriptor.schema_version = reader.u32();
    if (descriptor.schema_version != speculative_artifact_schema_version && descriptor.schema_version != 3U) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::unsupported_version,
            std::to_string(descriptor.schema_version)
        );
    }
    const auto kind = reader.u32();
    if (kind < static_cast<std::uint32_t>(SpeculativeArtifactKind::fixture_feature_cache)
        || kind > static_cast<std::uint32_t>(SpeculativeArtifactKind::trained_bundle)) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor, "artifact kind"
        );
    }
    descriptor.kind = static_cast<SpeculativeArtifactKind>(kind);
    if (descriptor.schema_version == 3U && descriptor.kind != SpeculativeArtifactKind::trained_candidate) {
        throw SpeculativeArtifactError(SpeculativeArtifactErrorCode::invalid_descriptor, "schema-3 kind");
    }
    const auto encoded_header_size = reader.u64();
    const auto encoded_payload_size = reader.u64();
    const auto wanted_envelope_digest = reader.digest();
    if (envelope_digest(bytes) != wanted_envelope_digest) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::digest_mismatch, "envelope"
        );
    }
    descriptor.artifact_id = reader.u64();
    descriptor.target = read_target(reader);
    if (descriptor.schema_version == 3U) {
        descriptor.recurrent = read_recurrent(reader);
    } else {
    descriptor.feature_bank_count = reader.u16();
    if (descriptor.feature_bank_count > descriptor.feature_banks.size()) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::capacity_overflow, "feature-bank count"
        );
    }
    for (std::uint16_t index = 0U; index < descriptor.feature_bank_count; ++index) {
        descriptor.feature_banks[index] = read_bank(reader);
    }
    descriptor.draft_cache = read_cache(reader);
    descriptor.draft_vocabulary = read_draft_vocabulary(reader);
    }
    if (encoded_header_size != reader.offset()
        || encoded_header_size > bytes.size()
        || encoded_payload_size > bytes.size() - encoded_header_size
        || encoded_header_size + encoded_payload_size != bytes.size()
        || encoded_payload_size > speculative_artifact_payload_byte_max) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor, "header or payload extent"
        );
    }
    const auto payload = bytes.subspan(
        static_cast<std::size_t>(encoded_header_size),
        static_cast<std::size_t>(encoded_payload_size)
    );
    validate_descriptor(descriptor, payload, true);
    return AuthenticatedSpeculativeArtifact(
        std::move(descriptor), std::vector<std::uint8_t>(payload.begin(), payload.end()),
        wanted_envelope_digest
    );
}

AuthenticatedSpeculativeArtifact load_speculative_artifact_file(
    const std::filesystem::path& path
) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::io_failure, "cannot open " + path.string()
        );
    }
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > speculative_artifact_payload_byte_max) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::capacity_overflow, "artifact file"
        );
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
    }
    if (!stream) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::io_failure, "cannot read " + path.string()
        );
    }
    return load_speculative_artifact(bytes);
}

void validate_speculative_artifact_target(
    const SpeculativeArtifactDescriptor& artifact,
    const EngineModelPayloadReceipt& model,
    const ResidentTargetPayloadReceipt& resident,
    const Gemma2ModelConfig& config,
    std::uint32_t output_head_policy
) {
    if (artifact.target.model != model
        || artifact.target.resident != resident
        || artifact.target.config != config
        || artifact.target.output_head_policy != output_head_policy) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::incompatible_target,
            "GGUF, metadata, source tensor, resident bytes, config, or output policy"
        );
    }
}

}  // namespace local_gguf_engine::engine
