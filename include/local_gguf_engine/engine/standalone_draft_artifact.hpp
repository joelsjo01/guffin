#pragma once

#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <filesystem>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace local_gguf_engine::engine {

enum class StandaloneDraftKind { vicuna_68m_f16, gemma3_270m_q8 };

// Each kind binds one complete authenticated artifact, not configurable reduced
// geometry. Tensor views borrow the mmap and must not outlive this owner.
class StandaloneDraftArtifact final {
public:
    static constexpr std::uint32_t layer_count = 2;
    static constexpr std::uint32_t hidden_size = 768;
    static constexpr std::uint32_t intermediate_size = 3072;
    static constexpr std::uint32_t head_count = 12;
    static constexpr std::uint32_t head_dimension = 64;
    static constexpr std::uint32_t vocabulary_size = 32000;
    static constexpr std::uint32_t context_capacity = 2048;
    static constexpr float rms_epsilon = 1e-6F;
    static constexpr float rope_theta = 10000.0F;

    explicit StandaloneDraftArtifact(const std::filesystem::path& path);
    StandaloneDraftArtifact(const StandaloneDraftArtifact&) = delete;
    StandaloneDraftArtifact& operator=(const StandaloneDraftArtifact&) = delete;
    StandaloneDraftArtifact(StandaloneDraftArtifact&&) noexcept = default;
    StandaloneDraftArtifact& operator=(StandaloneDraftArtifact&&) noexcept = default;

    // GGUF shape order: contiguous input dimension first. Vicuna names are HF
    // names; Gemma3 names are native GGUF names and norm weights already include
    // the source Gemma +1 offset. Neither accessor changes payload values.
    [[nodiscard]] tensor::TensorView tensor(std::string_view hf_name) const;
    [[nodiscard]] const tokenizer::Tokenizer& tokenizer() const noexcept;
    [[nodiscard]] const runtime::Sha256Digest& digest() const noexcept;
    [[nodiscard]] StandaloneDraftKind kind() const noexcept;
    [[nodiscard]] std::uint32_t context_limit() const noexcept;

private:
    model_format::GGUFFileBackedModel mapped_;
    tokenizer::Tokenizer tokenizer_;
    runtime::Sha256Digest digest_;
    StandaloneDraftKind kind_;
};

class StandaloneTliMapping final {
public:
    static constexpr std::uint32_t unmapped = std::numeric_limits<std::uint32_t>::max();

    StandaloneTliMapping(const tokenizer::Tokenizer& target, const tokenizer::Tokenizer& draft);
    [[nodiscard]] std::span<const std::uint32_t> draft_to_target() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> target_to_draft() const noexcept;
    [[nodiscard]] std::size_t intersection_size() const noexcept;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    [[nodiscard]] std::size_t max_append_tokens() const noexcept;

    // Mapped corrections retain their exact draft ID. Vicuna uses released TLI
    // cleanup/dummy-prefix encoding and strips target controls. Gemma3 preserves
    // shared Gemma control roles, uses identity/no-dummy-prefix encoding without
    // punctuation cleanup, and never interprets literal correction text as a
    // control. Only the target owns stops. Borrowed tokenizers must outlive this
    // immutable mapping; callers retain actual accepted draft emissions.
    [[nodiscard]] std::size_t append_target_token(
        std::uint32_t target_token, std::span<std::uint32_t> output,
        std::span<char> text_scratch, tokenizer::TokenizerEncodeWorkspace& workspace
    ) const;
    [[nodiscard]] std::size_t encode_target_context(
        std::span<const std::uint32_t> target_tokens, std::span<std::uint32_t> output,
        std::span<char> text_scratch, tokenizer::TokenizerEncodeWorkspace& workspace
    ) const;

private:
    const tokenizer::Tokenizer* target_;
    const tokenizer::Tokenizer* draft_;
    std::vector<std::uint32_t> draft_to_target_;
    std::vector<std::uint32_t> target_to_draft_;
    // Opaque controls may condition the draft without becoming proposals.
    // This is deliberately not an alias of the probability-law intersection.
    std::vector<std::uint32_t> target_control_context_;
    std::size_t intersection_size_ = 0;
    std::size_t max_append_tokens_ = 0;
    bool preserve_gemma_controls_ = false;
};

}  // namespace local_gguf_engine::engine
