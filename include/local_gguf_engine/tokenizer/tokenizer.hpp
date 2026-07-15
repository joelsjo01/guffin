#pragma once

#include "local_gguf_engine/model_format/gguf_metadata.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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
    unsupported_chat_template,
    invalid_chat_message,
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
    bool encode_special_tokens = false;
};

struct TokenizerDecodeOptions {
    bool skip_special_tokens = false;
};

struct PromptFormatResult {
    std::string text;
    bool encode_special_tokens = false;
    std::optional<bool> add_bos;
    std::optional<bool> add_eos;
};

struct ChatMessage {
    std::string_view role;
    std::string_view content;
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
    [[nodiscard]] bool token_id_is_special(std::uint32_t token_id) const noexcept;

private:
    friend Tokenizer load_tokenizer_from_gguf_metadata(
        const model_format::GGUFMetadataSection& metadata
    );

    Tokenizer(
        TokenizerModel model,
        std::vector<TokenizerToken> vocabulary,
        TokenizerSpecialTokens special_tokens
    );

    void append_exact_token_ids(std::vector<std::uint32_t>& token_ids, std::string_view text) const;
    void append_piece_token_ids(
        std::vector<std::uint32_t>& token_ids,
        std::string_view text,
        bool encode_special_tokens
    ) const;
    void append_token_id_for_text(std::vector<std::uint32_t>& token_ids, std::string_view token_text) const;

    [[nodiscard]] bool token_id_is_control(std::uint32_t token_id) const noexcept;
    [[nodiscard]] bool token_id_can_encode_text(std::uint32_t token_id) const noexcept;
    [[nodiscard]] bool token_id_can_encode_special(std::uint32_t token_id) const noexcept;
    [[nodiscard]] std::optional<std::pair<std::uint32_t, std::size_t>> special_token_match_at(
        std::string_view text,
        std::size_t offset
    ) const noexcept;

    TokenizerModel model_;
    std::vector<TokenizerToken> vocabulary_;
    TokenizerSpecialTokens special_tokens_;
    std::vector<std::uint32_t> encode_candidate_token_ids_;
    std::vector<std::uint32_t> special_encode_candidate_token_ids_;
    std::unordered_map<std::string, std::uint32_t> token_text_to_id_;
    std::array<std::optional<std::uint32_t>, 256> byte_token_ids_;
};

[[nodiscard]] std::string_view tokenizer_model_name(TokenizerModel model) noexcept;
[[nodiscard]] PromptFormatResult format_chat_prompt_from_gguf_metadata(
    const model_format::GGUFMetadataSection& metadata,
    std::span<const ChatMessage> messages,
    bool add_generation_prompt
);
[[nodiscard]] PromptFormatResult format_generate_prompt_from_gguf_metadata(
    const model_format::GGUFMetadataSection& metadata,
    std::string_view prompt
);
[[nodiscard]] Tokenizer load_tokenizer_from_gguf_metadata(
    const model_format::GGUFMetadataSection& metadata
);

}  // namespace local_gguf_engine::tokenizer
