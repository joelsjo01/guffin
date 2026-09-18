#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/engine/draft_vocabulary_types.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/learned_recurrent_types.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::uint32_t speculative_artifact_schema_version = 2U;
inline constexpr std::uint64_t speculative_artifact_payload_byte_max = 1ULL << 33U;

enum class SpeculativeArtifactKind : std::uint32_t {
    fixture_feature_cache = 1U,
    trained_candidate = 2U,
    trained_bundle = 3U,
};

enum class SpeculativeArtifactErrorCode : std::uint8_t {
    truncated,
    invalid_magic,
    unsupported_version,
    invalid_descriptor,
    capacity_overflow,
    digest_mismatch,
    io_failure,
    incompatible_target,
};

class SpeculativeArtifactError final : public std::runtime_error {
public:
    SpeculativeArtifactError(SpeculativeArtifactErrorCode code, std::string detail);
    [[nodiscard]] SpeculativeArtifactErrorCode code() const noexcept;

private:
    SpeculativeArtifactErrorCode code_;
};

struct ResidentTargetPayloadReceipt {
    std::uint32_t version = 1U;
    std::uint32_t allocation_count = 0U;
    std::uint64_t byte_count = 0U;
    runtime::Sha256Digest bytes;

    constexpr bool operator==(const ResidentTargetPayloadReceipt&) const = default;
};

struct SpeculativeArtifactTargetIdentity {
    EngineModelPayloadReceipt model;
    ResidentTargetPayloadReceipt resident;
    Gemma2ModelConfig config;
    std::uint32_t output_head_policy = 0U;

    constexpr bool operator==(const SpeculativeArtifactTargetIdentity&) const = default;
};

struct SpeculativeArtifactDescriptor {
    std::uint32_t schema_version = speculative_artifact_schema_version;
    SpeculativeArtifactKind kind = SpeculativeArtifactKind::fixture_feature_cache;
    std::uint64_t artifact_id = 0U;
    SpeculativeArtifactTargetIdentity target;
    std::array<FeatureBankSpec, feature_bank_max_count> feature_banks {};
    std::uint16_t feature_bank_count = 0U;
    DraftCacheSpec draft_cache;
    DraftVocabularyArtifactSpec draft_vocabulary;
    LearnedRecurrentSpec recurrent;

    constexpr bool operator==(const SpeculativeArtifactDescriptor&) const = default;
};

class AuthenticatedSpeculativeArtifact final {
public:
    AuthenticatedSpeculativeArtifact(const AuthenticatedSpeculativeArtifact&) = delete;
    AuthenticatedSpeculativeArtifact& operator=(const AuthenticatedSpeculativeArtifact&) = delete;
    AuthenticatedSpeculativeArtifact(AuthenticatedSpeculativeArtifact&&) noexcept = default;
    AuthenticatedSpeculativeArtifact& operator=(AuthenticatedSpeculativeArtifact&&) noexcept = default;

    [[nodiscard]] const SpeculativeArtifactDescriptor& descriptor() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> payload() const noexcept;
    [[nodiscard]] const runtime::Sha256Digest& envelope_digest() const noexcept;

private:
    friend AuthenticatedSpeculativeArtifact load_speculative_artifact(
        std::span<const std::uint8_t>
    );
    AuthenticatedSpeculativeArtifact(
        SpeculativeArtifactDescriptor descriptor,
        std::vector<std::uint8_t> payload,
        runtime::Sha256Digest envelope_digest
    );

    SpeculativeArtifactDescriptor descriptor_;
    std::vector<std::uint8_t> payload_;
    runtime::Sha256Digest envelope_digest_;
};

[[nodiscard]] std::vector<std::uint8_t> encode_fixture_speculative_artifact(
    SpeculativeArtifactDescriptor descriptor,
    std::span<const std::uint8_t> payload
);

// Schema 3 is a nonselectable trained weights payload. The schema-2 fixture
// format remains byte-for-byte independent of this trained mechanism.
[[nodiscard]] std::vector<std::uint8_t> encode_recurrent_weights_artifact(
    SpeculativeArtifactDescriptor descriptor, std::span<const std::uint8_t> payload
);

void write_fixture_speculative_artifact(
    const std::filesystem::path& path,
    SpeculativeArtifactDescriptor descriptor,
    std::span<const std::uint8_t> payload
);

[[nodiscard]] AuthenticatedSpeculativeArtifact load_speculative_artifact(
    std::span<const std::uint8_t> bytes
);

[[nodiscard]] AuthenticatedSpeculativeArtifact load_speculative_artifact_file(
    const std::filesystem::path& path
);

void validate_speculative_artifact_target(
    const SpeculativeArtifactDescriptor& artifact,
    const EngineModelPayloadReceipt& model,
    const ResidentTargetPayloadReceipt& resident,
    const Gemma2ModelConfig& config,
    std::uint32_t output_head_policy
);

}  // namespace local_gguf_engine::engine
