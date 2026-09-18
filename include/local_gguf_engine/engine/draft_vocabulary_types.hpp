#pragma once

#include "local_gguf_engine/runtime/sha256.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace local_gguf_engine::engine {

inline constexpr std::uint32_t draft_vocabulary_artifact_version = 1U;
inline constexpr std::uint32_t draft_vocabulary_external_state_max = 23U;
inline constexpr std::uint32_t draft_vocabulary_maximum_vocabulary = 262'144U;
inline constexpr std::uint32_t draft_vocabulary_tensor_count = 9U;
inline constexpr std::array<std::uint32_t, 8U> draft_vocabulary_state_batches {
    1U, 3U, 4U, 7U, 8U, 15U, 16U, 23U,
};

enum class DraftArtifactElementType : std::uint8_t {
    f16 = 1U,
    q8_0 = 2U,
    q6_k = 3U,
    q4_k = 4U,
    u32 = 5U,
};

enum class DraftArtifactTensorStorage : std::uint8_t {
    absent = 0U,
    payload = 1U,
    resident_target_output_q8_0 = 2U,
};

enum class DraftArtifactTensorRole : std::uint8_t {
    selector_projection = 0U,
    selector_head = 1U,
    candidate_head = 2U,
    target_up_projection = 3U,
    transition_counts = 4U,
    transition_tokens = 5U,
    proposal_exp_lut = 6U,
    arm_s_rank8_down = 7U,
    arm_s_rank8_up = 8U,
};

struct DraftArtifactTensorSpec {
    DraftArtifactTensorStorage storage = DraftArtifactTensorStorage::absent;
    DraftArtifactElementType element_type = DraftArtifactElementType::f16;
    std::uint32_t dimension_0 = 0U;
    std::uint32_t dimension_1 = 0U;
    std::uint64_t payload_offset = 0U;
    std::uint64_t payload_byte_count = 0U;
    runtime::Sha256Digest payload_digest;

    constexpr bool operator==(const DraftArtifactTensorSpec&) const = default;
};

struct DraftVocabularyArtifactSpec {
    std::uint32_t version = 0U;
    std::uint32_t semantic_width = 0U;
    std::uint32_t selector_width = 0U;
    // K-quant tables retain their native 256-element block width. Values in
    // [selector_width, selector_storage_width) must be authenticated zeros.
    std::uint32_t selector_storage_width = 0U;
    std::uint32_t target_width = 0U;
    std::uint32_t vocabulary_size = 0U;
    std::uint32_t state_capacity = 0U;
    std::uint32_t candidate_capacity = 0U;
    std::uint32_t transition_width = 0U;
    std::uint32_t rank_entry_capacity = 0U;
    std::array<DraftArtifactTensorSpec, draft_vocabulary_tensor_count> tensors {};

    [[nodiscard]] constexpr bool present() const noexcept { return version != 0U; }
    constexpr bool operator==(const DraftVocabularyArtifactSpec&) const = default;
};

[[nodiscard]] constexpr std::size_t draft_artifact_tensor_index(
    DraftArtifactTensorRole role
) noexcept {
    return static_cast<std::size_t>(role);
}

}  // namespace local_gguf_engine::engine
