#pragma once

#include "local_gguf_engine/model_format/gguf_metadata.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace local_gguf_engine::tokenizer {

enum class TokenizerErrorCode {
    missing_metadata,
    invalid_metadata_type,
    unsupported_architecture,
    unsupported_tokenizer_model,
    invalid_vocabulary,
    missing_special_token,
    token_id_out_of_range,
    no_matching_token,
};

class TokenizerError final : public std::runtime_error {
public:
    TokenizerError(TokenizerErrorCode code, std::string detail);

    [[nodiscard]] TokenizerErrorCode code() const noexcept;
    [[nodiscard]] const std::string& detail() const noexcept;

private:
    TokenizerErrorCode code_;
    std::string detail_;
};

enum class TokenizerModel {
    llama,
    gpt2,
};

struct TokenizerToken {
    std::string text;
    std::optional<float> score;
    std::optional<std::int32_t> type;
};

struct TokenizerSpecialTokens {
    std::optional<std::uint32_t> bos_token_id;
    std::optional<std::uint32_t> eos_token_id;
    std::optional<std::uint32_t> unknown_token_id;
    bool add_bos_token = false;
    bool add_eos_token = false;
};

struct TokenizerEncodeOptions {
    bool add_bos = false;
    bool add_eos = false;
};

struct TokenizerDecodeOptions {
    bool skip_special_tokens = false;
};

class Tokenizer final {
public:
    [[nodiscard]] TokenizerModel model() const noexcept;
    [[nodiscard]] std::string_view model_name() const noexcept;
    [[nodiscard]] const std::vector<TokenizerToken>& vocabulary() const noexcept;
    [[nodiscard]] const TokenizerSpecialTokens& special_tokens() const noexcept;
    [[nodiscard]] TokenizerEncodeOptions default_encode_options() const noexcept;

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const;
    [[nodiscard]] std::vector<std::uint32_t> encode(
        std::string_view text,
        TokenizerEncodeOptions options
    ) const;

    [[nodiscard]] std::string decode(std::span<const std::uint32_t> token_ids) const;
    [[nodiscard]] std::string decode(
        std::span<const std::uint32_t> token_ids,
        TokenizerDecodeOptions options
    ) const;

private:
    friend Tokenizer load_tokenizer_from_gguf_metadata(
        const model_format::GGUFMetadataSection& metadata
    );

    Tokenizer(
        TokenizerModel model,
        std::vector<TokenizerToken> vocabulary,
        TokenizerSpecialTokens special_tokens,
        std::vector<std::string> bpe_merges,
        bool add_space_prefix
    );

    void append_exact_token_ids(std::vector<std::uint32_t>& token_ids, std::string_view text) const;
    void append_gpt2_bpe_token_ids(std::vector<std::uint32_t>& token_ids, std::string_view text) const;
    void append_token_id_for_text(std::vector<std::uint32_t>& token_ids, std::string_view token_text) const;

    [[nodiscard]] bool token_id_is_special(std::uint32_t token_id) const noexcept;
    [[nodiscard]] bool token_id_is_control(std::uint32_t token_id) const noexcept;
    [[nodiscard]] bool token_id_can_encode_text(std::uint32_t token_id) const noexcept;

    TokenizerModel model_;
    std::vector<TokenizerToken> vocabulary_;
    TokenizerSpecialTokens special_tokens_;
    std::vector<std::uint32_t> encode_candidate_token_ids_;
    std::unordered_map<std::string, std::uint32_t> token_text_to_id_;
    std::unordered_map<std::string, std::uint32_t> bpe_merge_ranks_;
    bool add_space_prefix_;
};

[[nodiscard]] std::string_view tokenizer_model_name(TokenizerModel model) noexcept;
[[nodiscard]] Tokenizer load_tokenizer_from_gguf_metadata(
    const model_format::GGUFMetadataSection& metadata
);

}  // namespace local_gguf_engine::tokenizer
