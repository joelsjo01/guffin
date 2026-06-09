#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>

namespace local_gguf_engine::tokenizer {

namespace {

using model_format::GGUFMetadataArray;
using model_format::GGUFMetadataEntry;
using model_format::GGUFMetadataSection;
using model_format::GGUFMetadataValueType;

constexpr std::int32_t gguf_token_type_unknown = 2;
constexpr std::int32_t gguf_token_type_control = 3;
constexpr std::int32_t gguf_token_type_unused = 5;

[[nodiscard]] std::string error_message(TokenizerErrorCode code, std::string_view detail) {
    std::string message;
    switch (code) {
        case TokenizerErrorCode::missing_metadata:
            message = "missing tokenizer metadata";
            break;
        case TokenizerErrorCode::invalid_metadata_type:
            message = "invalid tokenizer metadata type";
            break;
        case TokenizerErrorCode::unsupported_architecture:
            message = "unsupported tokenizer architecture";
            break;
        case TokenizerErrorCode::unsupported_tokenizer_model:
            message = "unsupported tokenizer model";
            break;
        case TokenizerErrorCode::invalid_vocabulary:
            message = "invalid tokenizer vocabulary";
            break;
        case TokenizerErrorCode::missing_special_token:
            message = "missing tokenizer special token";
            break;
        case TokenizerErrorCode::token_id_out_of_range:
            message = "token id out of range";
            break;
        case TokenizerErrorCode::no_matching_token:
            message = "no tokenizer vocabulary match";
            break;
    }

    if (!detail.empty()) {
        message += ": ";
        message += detail;
    }

    return message;
}

[[nodiscard]] const GGUFMetadataEntry* find_metadata_entry(
    const GGUFMetadataSection& metadata,
    std::string_view key
) noexcept {
    const auto found = std::find_if(
        metadata.entries.begin(),
        metadata.entries.end(),
        [key](const GGUFMetadataEntry& entry) {
            return entry.key == key;
        }
    );

    if (found == metadata.entries.end()) {
        return nullptr;
    }

    return &*found;
}

[[nodiscard]] const GGUFMetadataEntry& require_metadata_entry(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        throw TokenizerError(TokenizerErrorCode::missing_metadata, std::string(key));
    }

    return *entry;
}

[[nodiscard]] std::string require_metadata_string(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto& entry = require_metadata_entry(metadata, key);
    if (entry.value.type != GGUFMetadataValueType::string
        || !std::holds_alternative<std::string>(entry.value.data)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<std::string>(entry.value.data);
}

[[nodiscard]] std::optional<std::uint32_t> optional_metadata_u32(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        return std::nullopt;
    }

    if (entry->value.type != GGUFMetadataValueType::uint32
        || !std::holds_alternative<std::uint32_t>(entry->value.data)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<std::uint32_t>(entry->value.data);
}

[[nodiscard]] std::optional<bool> optional_metadata_bool(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        return std::nullopt;
    }

    if (entry->value.type != GGUFMetadataValueType::boolean
        || !std::holds_alternative<bool>(entry->value.data)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<bool>(entry->value.data);
}

[[nodiscard]] const GGUFMetadataArray& require_metadata_array(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto& entry = require_metadata_entry(metadata, key);
    if (entry.value.type != GGUFMetadataValueType::array
        || !std::holds_alternative<GGUFMetadataArray>(entry.value.data)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<GGUFMetadataArray>(entry.value.data);
}

template <typename Values>
[[nodiscard]] const Values& require_metadata_array_values(
    const GGUFMetadataSection& metadata,
    std::string_view key,
    GGUFMetadataValueType element_type
) {
    const auto& array = require_metadata_array(metadata, key);
    if (array.element_type != element_type || !std::holds_alternative<Values>(array.values)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<Values>(array.values);
}

template <typename Values>
[[nodiscard]] std::optional<std::reference_wrapper<const Values>> optional_metadata_array_values(
    const GGUFMetadataSection& metadata,
    std::string_view key,
    GGUFMetadataValueType element_type
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        return std::nullopt;
    }

    if (entry->value.type != GGUFMetadataValueType::array
        || !std::holds_alternative<GGUFMetadataArray>(entry->value.data)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    const auto& array = std::get<GGUFMetadataArray>(entry->value.data);
    if (array.element_type != element_type || !std::holds_alternative<Values>(array.values)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::cref(std::get<Values>(array.values));
}

[[nodiscard]] TokenizerModel parse_tokenizer_model(std::string_view model) {
    if (model == "llama") {
        return TokenizerModel::llama;
    }

    if (model == "gpt2") {
        return TokenizerModel::gpt2;
    }

    throw TokenizerError(TokenizerErrorCode::unsupported_tokenizer_model, std::string(model));
}

void validate_architecture(const GGUFMetadataSection& metadata) {
    const auto architecture = require_metadata_string(metadata, "general.architecture");
    if (architecture != "llama") {
        throw TokenizerError(TokenizerErrorCode::unsupported_architecture, architecture);
    }
}

void validate_optional_vocab_size(const GGUFMetadataSection& metadata, std::size_t token_count) {
    const auto vocab_size = optional_metadata_u32(metadata, "llama.vocab_size");
    if (!vocab_size.has_value()) {
        return;
    }

    if (static_cast<std::uint64_t>(*vocab_size) != static_cast<std::uint64_t>(token_count)) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "llama.vocab_size");
    }
}

void validate_optional_array_length(
    std::string_view key,
    std::size_t actual_size,
    std::size_t expected_size
) {
    if (actual_size != expected_size) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, std::string(key));
    }
}

void validate_token_id(
    std::string_view key,
    std::optional<std::uint32_t> token_id,
    std::size_t token_count
) {
    if (!token_id.has_value()) {
        return;
    }

    if (static_cast<std::uint64_t>(*token_id) >= static_cast<std::uint64_t>(token_count)) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, std::string(key));
    }
}

[[nodiscard]] std::vector<TokenizerToken> build_vocabulary(const GGUFMetadataSection& metadata) {
    const auto& token_texts = require_metadata_array_values<std::vector<std::string>>(
        metadata,
        "tokenizer.ggml.tokens",
        GGUFMetadataValueType::string
    );
    const auto scores = optional_metadata_array_values<std::vector<float>>(
        metadata,
        "tokenizer.ggml.scores",
        GGUFMetadataValueType::float32
    );
    const auto token_types = optional_metadata_array_values<std::vector<std::int32_t>>(
        metadata,
        "tokenizer.ggml.token_type",
        GGUFMetadataValueType::int32
    );

    if (token_texts.empty()) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "tokenizer.ggml.tokens");
    }

    if (scores.has_value()) {
        validate_optional_array_length("tokenizer.ggml.scores", scores->get().size(), token_texts.size());
    }

    if (token_types.has_value()) {
        validate_optional_array_length("tokenizer.ggml.token_type", token_types->get().size(), token_texts.size());
    }

    std::unordered_set<std::string> seen_tokens;
    seen_tokens.reserve(token_texts.size());

    std::vector<TokenizerToken> vocabulary;
    vocabulary.reserve(token_texts.size());
    for (std::size_t index = 0; index < token_texts.size(); ++index) {
        const auto& text = token_texts[index];
        if (text.empty()) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "empty token text");
        }

        if (!seen_tokens.insert(text).second) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, text);
        }

        vocabulary.push_back(TokenizerToken {
            .text = text,
            .score = scores.has_value()
                ? std::optional<float>(scores->get()[index])
                : std::nullopt,
            .type = token_types.has_value()
                ? std::optional<std::int32_t>(token_types->get()[index])
                : std::nullopt,
        });
    }

    validate_optional_vocab_size(metadata, vocabulary.size());

    return vocabulary;
}

[[nodiscard]] TokenizerSpecialTokens load_special_tokens(
    const GGUFMetadataSection& metadata,
    std::size_t token_count
) {
    TokenizerSpecialTokens special_tokens {
        .bos_token_id = optional_metadata_u32(metadata, "tokenizer.ggml.bos_token_id"),
        .eos_token_id = optional_metadata_u32(metadata, "tokenizer.ggml.eos_token_id"),
        .unknown_token_id = optional_metadata_u32(metadata, "tokenizer.ggml.unknown_token_id"),
        .add_bos_token = optional_metadata_bool(metadata, "tokenizer.ggml.add_bos_token").value_or(false),
        .add_eos_token = optional_metadata_bool(metadata, "tokenizer.ggml.add_eos_token").value_or(false),
    };

    validate_token_id("tokenizer.ggml.bos_token_id", special_tokens.bos_token_id, token_count);
    validate_token_id("tokenizer.ggml.eos_token_id", special_tokens.eos_token_id, token_count);
    validate_token_id("tokenizer.ggml.unknown_token_id", special_tokens.unknown_token_id, token_count);

    if (special_tokens.add_bos_token && !special_tokens.bos_token_id.has_value()) {
        throw TokenizerError(TokenizerErrorCode::missing_special_token, "tokenizer.ggml.bos_token_id");
    }

    if (special_tokens.add_eos_token && !special_tokens.eos_token_id.has_value()) {
        throw TokenizerError(TokenizerErrorCode::missing_special_token, "tokenizer.ggml.eos_token_id");
    }

    return special_tokens;
}

[[nodiscard]] bool starts_with_at(
    std::string_view text,
    std::size_t offset,
    std::string_view token_text
) noexcept {
    return token_text.size() <= text.size() - offset
        && text.compare(offset, token_text.size(), token_text) == 0;
}

[[nodiscard]] std::string byte_offset_detail(std::size_t offset) {
    return "byte offset " + std::to_string(offset);
}

[[nodiscard]] std::uint32_t require_special_token_id(
    std::optional<std::uint32_t> token_id,
    std::string_view key
) {
    if (!token_id.has_value()) {
        throw TokenizerError(TokenizerErrorCode::missing_special_token, std::string(key));
    }

    return *token_id;
}

[[nodiscard]] bool is_utf8_continuation(std::uint8_t byte) noexcept {
    return (byte & 0xC0U) == 0x80U;
}

struct Utf8CodePoint {
    char32_t value;
    std::size_t byte_count;
    bool valid;
};

[[nodiscard]] Utf8CodePoint read_utf8_code_point(std::string_view text, std::size_t offset) noexcept {
    const auto first = static_cast<std::uint8_t>(text[offset]);
    if (first < 0x80U) {
        return Utf8CodePoint {.value = static_cast<char32_t>(first), .byte_count = 1, .valid = true};
    }

    const auto remaining = text.size() - offset;
    if ((first & 0xE0U) == 0xC0U && remaining >= 2) {
        const auto second = static_cast<std::uint8_t>(text[offset + 1]);
        if (is_utf8_continuation(second)) {
            return Utf8CodePoint {
                .value = static_cast<char32_t>(((first & 0x1FU) << 6U) | (second & 0x3FU)),
                .byte_count = 2,
                .valid = true,
            };
        }
    }

    if ((first & 0xF0U) == 0xE0U && remaining >= 3) {
        const auto second = static_cast<std::uint8_t>(text[offset + 1]);
        const auto third = static_cast<std::uint8_t>(text[offset + 2]);
        if (is_utf8_continuation(second) && is_utf8_continuation(third)) {
            return Utf8CodePoint {
                .value = static_cast<char32_t>(
                    ((first & 0x0FU) << 12U)
                    | ((second & 0x3FU) << 6U)
                    | (third & 0x3FU)
                ),
                .byte_count = 3,
                .valid = true,
            };
        }
    }

    if ((first & 0xF8U) == 0xF0U && remaining >= 4) {
        const auto second = static_cast<std::uint8_t>(text[offset + 1]);
        const auto third = static_cast<std::uint8_t>(text[offset + 2]);
        const auto fourth = static_cast<std::uint8_t>(text[offset + 3]);
        if (is_utf8_continuation(second) && is_utf8_continuation(third) && is_utf8_continuation(fourth)) {
            return Utf8CodePoint {
                .value = static_cast<char32_t>(
                    ((first & 0x07U) << 18U)
                    | ((second & 0x3FU) << 12U)
                    | ((third & 0x3FU) << 6U)
                    | (fourth & 0x3FU)
                ),
                .byte_count = 4,
                .valid = true,
            };
        }
    }

    return Utf8CodePoint {.value = static_cast<char32_t>(first), .byte_count = 1, .valid = false};
}

[[nodiscard]] std::optional<std::uint8_t> gpt2_byte_from_code_point(char32_t code_point) noexcept {
    const auto code = static_cast<std::uint32_t>(code_point);
    if ((code >= 33U && code <= 126U) || (code >= 161U && code <= 172U) || (code >= 174U && code <= 255U)) {
        return static_cast<std::uint8_t>(code);
    }

    if (code < 0x0100U || code > 0x0143U) {
        return std::nullopt;
    }

    const auto compact = code - 0x0100U;
    if (compact <= 32U) {
        return static_cast<std::uint8_t>(compact);
    }

    if (compact <= 66U) {
        return static_cast<std::uint8_t>(127U + (compact - 33U));
    }

    return static_cast<std::uint8_t>(173U);
}

void append_gpt2_decoded_text(std::string& output, std::string_view token_text) {
    // GPT-2 vocab stores bytes in a reversible Unicode alphabet before BPE merges.
    std::size_t offset = 0;
    while (offset < token_text.size()) {
        const auto code_point = read_utf8_code_point(token_text, offset);
        if (code_point.valid) {
            const auto byte = gpt2_byte_from_code_point(code_point.value);
            if (byte.has_value()) {
                output.push_back(static_cast<char>(*byte));
                offset += code_point.byte_count;
                continue;
            }
        }

        output.append(token_text.substr(offset, code_point.byte_count));
        offset += code_point.byte_count;
    }
}

void append_decoded_token_text(
    std::string& output,
    TokenizerModel model,
    std::string_view token_text
) {
    switch (model) {
        case TokenizerModel::gpt2:
            append_gpt2_decoded_text(output, token_text);
            return;
        case TokenizerModel::llama:
            output += token_text;
            return;
    }
}

}  // namespace

TokenizerError::TokenizerError(TokenizerErrorCode code, std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

TokenizerErrorCode TokenizerError::code() const noexcept {
    return code_;
}

const std::string& TokenizerError::detail() const noexcept {
    return detail_;
}

std::string_view tokenizer_model_name(TokenizerModel model) noexcept {
    switch (model) {
        case TokenizerModel::llama:
            return "llama";
        case TokenizerModel::gpt2:
            return "gpt2";
    }

    return "unknown";
}

Tokenizer::Tokenizer(
    TokenizerModel model,
    std::vector<TokenizerToken> vocabulary,
    TokenizerSpecialTokens special_tokens
)
    : model_(model),
      vocabulary_(std::move(vocabulary)),
      special_tokens_(special_tokens),
      encode_candidate_token_ids_() {
    encode_candidate_token_ids_.reserve(vocabulary_.size());
    for (std::size_t index = 0; index < vocabulary_.size(); ++index) {
        if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "token id overflow");
        }

        const auto token_id = static_cast<std::uint32_t>(index);
        if (token_id_can_encode_text(token_id)) {
            encode_candidate_token_ids_.push_back(token_id);
        }
    }

    std::sort(
        encode_candidate_token_ids_.begin(),
        encode_candidate_token_ids_.end(),
        [this](std::uint32_t left, std::uint32_t right) {
            const auto left_size = vocabulary_[left].text.size();
            const auto right_size = vocabulary_[right].text.size();
            if (left_size != right_size) {
                return left_size > right_size;
            }

            return left < right;
        }
    );
}

TokenizerModel Tokenizer::model() const noexcept {
    return model_;
}

std::string_view Tokenizer::model_name() const noexcept {
    return tokenizer_model_name(model_);
}

const std::vector<TokenizerToken>& Tokenizer::vocabulary() const noexcept {
    return vocabulary_;
}

const TokenizerSpecialTokens& Tokenizer::special_tokens() const noexcept {
    return special_tokens_;
}

TokenizerEncodeOptions Tokenizer::default_encode_options() const noexcept {
    return TokenizerEncodeOptions {
        .add_bos = special_tokens_.add_bos_token,
        .add_eos = special_tokens_.add_eos_token,
    };
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
    return encode(text, default_encode_options());
}

std::vector<std::uint32_t> Tokenizer::encode(
    std::string_view text,
    TokenizerEncodeOptions options
) const {
    std::vector<std::uint32_t> token_ids;
    token_ids.reserve(text.size() + (options.add_bos ? 1U : 0U) + (options.add_eos ? 1U : 0U));

    if (options.add_bos) {
        token_ids.push_back(require_special_token_id(special_tokens_.bos_token_id, "tokenizer.ggml.bos_token_id"));
    }

    std::size_t offset = 0;
    while (offset < text.size()) {
        bool matched = false;
        for (const auto token_id : encode_candidate_token_ids_) {
            const auto& token_text = vocabulary_[token_id].text;
            if (!starts_with_at(text, offset, token_text)) {
                continue;
            }

            token_ids.push_back(token_id);
            offset += token_text.size();
            matched = true;
            break;
        }

        if (matched) {
            continue;
        }

        if (special_tokens_.unknown_token_id.has_value()) {
            token_ids.push_back(*special_tokens_.unknown_token_id);
            ++offset;
            continue;
        }

        throw TokenizerError(TokenizerErrorCode::no_matching_token, byte_offset_detail(offset));
    }

    if (options.add_eos) {
        token_ids.push_back(require_special_token_id(special_tokens_.eos_token_id, "tokenizer.ggml.eos_token_id"));
    }

    return token_ids;
}

std::string Tokenizer::decode(std::span<const std::uint32_t> token_ids) const {
    return decode(token_ids, TokenizerDecodeOptions {});
}

std::string Tokenizer::decode(
    std::span<const std::uint32_t> token_ids,
    TokenizerDecodeOptions options
) const {
    std::string output;
    for (const auto token_id : token_ids) {
        if (static_cast<std::uint64_t>(token_id) >= static_cast<std::uint64_t>(vocabulary_.size())) {
            throw TokenizerError(TokenizerErrorCode::token_id_out_of_range, std::to_string(token_id));
        }

        if (options.skip_special_tokens && token_id_is_special(token_id)) {
            continue;
        }

        append_decoded_token_text(output, model_, vocabulary_[token_id].text);
    }

    return output;
}

bool Tokenizer::token_id_is_special(std::uint32_t token_id) const noexcept {
    return special_tokens_.bos_token_id == token_id
        || special_tokens_.eos_token_id == token_id
        || token_id_is_control(token_id);
}

bool Tokenizer::token_id_is_control(std::uint32_t token_id) const noexcept {
    if (static_cast<std::uint64_t>(token_id) >= static_cast<std::uint64_t>(vocabulary_.size())) {
        return false;
    }

    const auto type = vocabulary_[token_id].type;
    return type == gguf_token_type_control || type == gguf_token_type_unused;
}

bool Tokenizer::token_id_can_encode_text(std::uint32_t token_id) const noexcept {
    if (static_cast<std::uint64_t>(token_id) >= static_cast<std::uint64_t>(vocabulary_.size())) {
        return false;
    }

    if (token_id_is_special(token_id)) {
        return false;
    }

    if (special_tokens_.unknown_token_id == token_id) {
        return false;
    }

    const auto type = vocabulary_[token_id].type;
    return type != gguf_token_type_unknown;
}

Tokenizer load_tokenizer_from_gguf_metadata(const GGUFMetadataSection& metadata) {
    validate_architecture(metadata);
    const auto model = parse_tokenizer_model(require_metadata_string(metadata, "tokenizer.ggml.model"));
    auto vocabulary = build_vocabulary(metadata);
    const auto special_tokens = load_special_tokens(metadata, vocabulary.size());

    return Tokenizer(model, std::move(vocabulary), special_tokens);
}

}  // namespace local_gguf_engine::tokenizer
