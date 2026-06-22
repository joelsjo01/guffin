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
        case TokenizerErrorCode::unsupported_chat_template:
            message = "unsupported chat template";
            break;
        case TokenizerErrorCode::invalid_chat_message:
            message = "invalid chat message";
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

[[nodiscard]] std::optional<std::string> optional_metadata_string(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        return std::nullopt;
    }

    if (entry->value.type != GGUFMetadataValueType::string
        || !std::holds_alternative<std::string>(entry->value.data)) {
        throw TokenizerError(TokenizerErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<std::string>(entry->value.data);
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
    if (architecture != "llama" && architecture != "gemma2") {
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

[[nodiscard]] std::vector<std::string> load_bpe_merges(const GGUFMetadataSection& metadata) {
    const auto merges = optional_metadata_array_values<std::vector<std::string>>(
        metadata,
        "tokenizer.ggml.merges",
        GGUFMetadataValueType::string
    );
    if (!merges.has_value()) {
        return {};
    }

    for (const auto& merge : merges->get()) {
        const auto separator = merge.find(' ');
        if (separator == std::string::npos || separator == 0 || separator + 1 == merge.size()) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "tokenizer.ggml.merges");
        }
    }

    return merges->get();
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

[[nodiscard]] bool load_add_space_prefix(const GGUFMetadataSection& metadata) {
    return optional_metadata_bool(metadata, "tokenizer.ggml.add_space_prefix").value_or(false);
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

[[nodiscard]] bool is_ascii_alpha(char character) noexcept {
    return (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z');
}

[[nodiscard]] bool is_ascii_digit(char character) noexcept {
    return character >= '0' && character <= '9';
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

[[nodiscard]] bool code_point_is_space(char32_t code_point) noexcept {
    return code_point == U' '
        || code_point == U'\t'
        || code_point == U'\n'
        || code_point == U'\r'
        || code_point == U'\v'
        || code_point == U'\f'
        || code_point == 0x0085U
        || code_point == 0x00A0U;
}

enum class GPT2PieceClass {
    letter,
    number,
    punctuation,
    space,
};

[[nodiscard]] GPT2PieceClass gpt2_piece_class(std::string_view text, std::size_t offset) noexcept {
    const auto code_point = read_utf8_code_point(text, offset);
    if (code_point_is_space(code_point.value)) {
        return GPT2PieceClass::space;
    }

    const auto first = text[offset];
    if (static_cast<unsigned char>(first) < 0x80U) {
        if (is_ascii_alpha(first)) {
            return GPT2PieceClass::letter;
        }

        if (is_ascii_digit(first)) {
            return GPT2PieceClass::number;
        }

        return GPT2PieceClass::punctuation;
    }

    return GPT2PieceClass::letter;
}

[[nodiscard]] bool starts_with_ascii(std::string_view text, std::size_t offset, std::string_view prefix) noexcept {
    return prefix.size() <= text.size() - offset && text.compare(offset, prefix.size(), prefix) == 0;
}

[[nodiscard]] std::size_t contraction_byte_count(std::string_view text, std::size_t offset) noexcept {
    constexpr std::string_view contractions[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
    for (const auto contraction : contractions) {
        if (starts_with_ascii(text, offset, contraction)) {
            return contraction.size();
        }
    }

    return 0;
}

[[nodiscard]] std::size_t code_point_byte_count(std::string_view text, std::size_t offset) noexcept {
    return read_utf8_code_point(text, offset).byte_count;
}

[[nodiscard]] std::vector<std::string_view> gpt2_pretokenize(std::string_view text) {
    std::vector<std::string_view> pieces;
    pieces.reserve(text.size());

    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto contraction_size = contraction_byte_count(text, offset);
        if (contraction_size != 0) {
            pieces.push_back(text.substr(offset, contraction_size));
            offset += contraction_size;
            continue;
        }

        const auto piece_start = offset;
        auto piece_class = gpt2_piece_class(text, offset);
        if (text[offset] == ' ' && offset + 1 < text.size()) {
            const auto next_offset = offset + 1;
            const auto next_class = gpt2_piece_class(text, next_offset);
            if (next_class != GPT2PieceClass::space) {
                offset = next_offset;
                piece_class = next_class;
            }
        }

        offset += code_point_byte_count(text, offset);
        while (offset < text.size() && contraction_byte_count(text, offset) == 0) {
            if (gpt2_piece_class(text, offset) != piece_class) {
                break;
            }
            offset += code_point_byte_count(text, offset);
        }

        pieces.push_back(text.substr(piece_start, offset - piece_start));
    }

    return pieces;
}

void append_utf8_code_point(std::string& output, char32_t code_point) {
    if (code_point <= 0x7FU) {
        output.push_back(static_cast<char>(code_point));
        return;
    }

    if (code_point <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | ((static_cast<std::uint32_t>(code_point) >> 6U) & 0x1FU)));
        output.push_back(static_cast<char>(0x80U | (static_cast<std::uint32_t>(code_point) & 0x3FU)));
        return;
    }

    if (code_point <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | ((static_cast<std::uint32_t>(code_point) >> 12U) & 0x0FU)));
        output.push_back(static_cast<char>(0x80U | ((static_cast<std::uint32_t>(code_point) >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (static_cast<std::uint32_t>(code_point) & 0x3FU)));
        return;
    }

    output.push_back(static_cast<char>(0xF0U | ((static_cast<std::uint32_t>(code_point) >> 18U) & 0x07U)));
    output.push_back(static_cast<char>(0x80U | ((static_cast<std::uint32_t>(code_point) >> 12U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | ((static_cast<std::uint32_t>(code_point) >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (static_cast<std::uint32_t>(code_point) & 0x3FU)));
}

[[nodiscard]] char32_t gpt2_code_point_from_byte(std::uint8_t byte) noexcept {
    if ((byte >= 33U && byte <= 126U) || (byte >= 161U && byte <= 172U) || (byte >= 174U && byte <= 255U)) {
        return static_cast<char32_t>(byte);
    }

    if (byte <= 32U) {
        return static_cast<char32_t>(0x0100U + byte);
    }

    if (byte >= 127U && byte <= 160U) {
        return static_cast<char32_t>(0x0100U + 33U + (byte - 127U));
    }

    return static_cast<char32_t>(0x0143U);
}

[[nodiscard]] std::string gpt2_byte_encode(std::string_view text) {
    std::string output;
    output.reserve(text.size());
    for (const auto character : text) {
        append_utf8_code_point(output, gpt2_code_point_from_byte(static_cast<std::uint8_t>(character)));
    }

    return output;
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

[[nodiscard]] std::vector<std::string> split_utf8_symbols(std::string_view text) {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());

    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto code_point = read_utf8_code_point(text, offset);
        symbols.emplace_back(text.substr(offset, code_point.byte_count));
        offset += code_point.byte_count;
    }

    return symbols;
}

[[nodiscard]] std::string bpe_merge_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + 1U + right.size());
    key += left;
    key.push_back('\0');
    key += right;
    return key;
}

[[nodiscard]] std::pair<std::string_view, std::string_view> parse_bpe_merge(std::string_view merge) {
    const auto separator = merge.find(' ');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 == merge.size()) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "tokenizer.ggml.merges");
    }

    return {merge.substr(0, separator), merge.substr(separator + 1)};
}

constexpr std::string_view smollm2_default_system_prompt =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

[[nodiscard]] bool supports_smollm2_single_user_template(std::string_view chat_template) noexcept {
    return chat_template.find(smollm2_default_system_prompt) != std::string_view::npos
        && chat_template.find("messages[0]['role'] != 'system'") != std::string_view::npos
        && chat_template.find("message['role']") != std::string_view::npos
        && chat_template.find("message['content']") != std::string_view::npos
        && chat_template.find("add_generation_prompt") != std::string_view::npos
        && chat_template.find("<|im_start|>assistant\n") != std::string_view::npos;
}

[[nodiscard]] std::string format_smollm2_single_user_prompt(std::string_view prompt) {
    // Keep chat-template support engine-owned without adding a production Jinja dependency.
    std::string formatted;
    formatted.reserve(prompt.size() + smollm2_default_system_prompt.size() + 80U);
    formatted += "<|im_start|>system\n";
    formatted += smollm2_default_system_prompt;
    formatted += "<|im_end|>\n<|im_start|>user\n";
    formatted += prompt;
    formatted += "<|im_end|>\n<|im_start|>assistant\n";
    return formatted;
}

[[nodiscard]] bool supports_gemma2_it_template(std::string_view chat_template) noexcept {
    return chat_template.find("{{ bos_token }}") != std::string_view::npos
        && chat_template.find("System role not supported") != std::string_view::npos
        && chat_template.find("Conversation roles must alternate") != std::string_view::npos
        && chat_template.find("message['role'] == 'assistant'") != std::string_view::npos
        && chat_template.find("{% set role = 'model' %}") != std::string_view::npos
        && chat_template.find("<start_of_turn>") != std::string_view::npos
        && chat_template.find("<end_of_turn>") != std::string_view::npos
        && chat_template.find("message['content'] | trim") != std::string_view::npos
        && chat_template.find("add_generation_prompt") != std::string_view::npos
        && chat_template.find("<start_of_turn>model") != std::string_view::npos;
}

[[nodiscard]] bool is_ascii_whitespace(char character) noexcept {
    return character == ' '
        || character == '\t'
        || character == '\n'
        || character == '\r'
        || character == '\v'
        || character == '\f';
}

[[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view text) noexcept {
    std::size_t begin = 0;
    while (begin < text.size() && is_ascii_whitespace(text[begin])) {
        ++begin;
    }

    auto end = text.size();
    while (end > begin && is_ascii_whitespace(text[end - 1U])) {
        --end;
    }

    return text.substr(begin, end - begin);
}

[[nodiscard]] std::string token_text_for_id(
    const GGUFMetadataSection& metadata,
    std::uint32_t token_id,
    std::string_view detail
) {
    const auto& token_texts = require_metadata_array_values<std::vector<std::string>>(
        metadata,
        "tokenizer.ggml.tokens",
        GGUFMetadataValueType::string
    );

    if (static_cast<std::uint64_t>(token_id) >= static_cast<std::uint64_t>(token_texts.size())) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, std::string(detail));
    }

    return token_texts[token_id];
}

[[nodiscard]] std::string bos_token_text(const GGUFMetadataSection& metadata) {
    const auto token_id = require_special_token_id(
        optional_metadata_u32(metadata, "tokenizer.ggml.bos_token_id"),
        "tokenizer.ggml.bos_token_id"
    );
    return token_text_for_id(metadata, token_id, "tokenizer.ggml.bos_token_id");
}

[[nodiscard]] std::string_view gemma2_output_role_for_message(const ChatMessage& message, std::size_t index) {
    if (message.role == "system") {
        throw TokenizerError(TokenizerErrorCode::invalid_chat_message, "system");
    }

    const auto expects_user = index % 2U == 0U;
    const auto is_user = message.role == "user";
    const auto is_model = message.role == "assistant" || message.role == "model";
    if (!is_user && !is_model) {
        throw TokenizerError(TokenizerErrorCode::invalid_chat_message, std::string(message.role));
    }

    if (expects_user != is_user) {
        throw TokenizerError(TokenizerErrorCode::invalid_chat_message, "role alternation");
    }

    return is_user ? std::string_view("user") : std::string_view("model");
}

[[nodiscard]] PromptFormatResult format_gemma2_it_chat_prompt(
    const GGUFMetadataSection& metadata,
    std::span<const ChatMessage> messages,
    bool add_generation_prompt
) {
    if (messages.empty()) {
        throw TokenizerError(TokenizerErrorCode::invalid_chat_message, "messages");
    }

    std::size_t content_size = 0;
    for (const auto& message : messages) {
        content_size += trim_ascii_whitespace(message.content).size();
    }

    std::string formatted = bos_token_text(metadata);
    formatted.reserve(formatted.size() + content_size + messages.size() * 36U + (add_generation_prompt ? 22U : 0U));
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto output_role = gemma2_output_role_for_message(messages[index], index);
        formatted += "<start_of_turn>";
        formatted += output_role;
        formatted.push_back('\n');
        formatted += trim_ascii_whitespace(messages[index].content);
        formatted += "<end_of_turn>\n";
    }

    if (add_generation_prompt) {
        formatted += "<start_of_turn>model\n";
    }

    return PromptFormatResult {
        .text = std::move(formatted),
        .encode_special_tokens = true,
        .add_bos = false,
        .add_eos = false,
    };
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

PromptFormatResult format_chat_prompt_from_gguf_metadata(
    const GGUFMetadataSection& metadata,
    std::span<const ChatMessage> messages,
    bool add_generation_prompt
) {
    const auto chat_template = optional_metadata_string(metadata, "tokenizer.chat_template");
    if (!chat_template.has_value()) {
        throw TokenizerError(TokenizerErrorCode::unsupported_chat_template, "tokenizer.chat_template");
    }

    const auto architecture = require_metadata_string(metadata, "general.architecture");
    if (architecture == "gemma2" && supports_gemma2_it_template(*chat_template)) {
        return format_gemma2_it_chat_prompt(metadata, messages, add_generation_prompt);
    }

    if (supports_smollm2_single_user_template(*chat_template)) {
        if (messages.size() != 1 || messages.front().role != "user" || !add_generation_prompt) {
            throw TokenizerError(TokenizerErrorCode::invalid_chat_message, "messages");
        }

        return PromptFormatResult {
            .text = format_smollm2_single_user_prompt(messages.front().content),
            .encode_special_tokens = true,
        };
    }

    throw TokenizerError(TokenizerErrorCode::unsupported_chat_template, "tokenizer.chat_template");
}

PromptFormatResult format_generate_prompt_from_gguf_metadata(
    const GGUFMetadataSection& metadata,
    std::string_view prompt
) {
    const auto chat_template = optional_metadata_string(metadata, "tokenizer.chat_template");
    if (!chat_template.has_value()) {
        return PromptFormatResult {
            .text = std::string(prompt),
            .encode_special_tokens = false,
        };
    }

    const auto architecture = require_metadata_string(metadata, "general.architecture");
    if (architecture == "gemma2" && supports_gemma2_it_template(*chat_template)) {
        const ChatMessage message {
            .role = "user",
            .content = prompt,
        };
        return format_chat_prompt_from_gguf_metadata(
            metadata,
            std::span<const ChatMessage>(&message, 1),
            true
        );
    }

    if (supports_smollm2_single_user_template(*chat_template)) {
        return PromptFormatResult {
            .text = format_smollm2_single_user_prompt(prompt),
            .encode_special_tokens = true,
        };
    }

    throw TokenizerError(TokenizerErrorCode::unsupported_chat_template, "tokenizer.chat_template");
}

Tokenizer::Tokenizer(
    TokenizerModel model,
    std::vector<TokenizerToken> vocabulary,
    TokenizerSpecialTokens special_tokens,
    std::vector<std::string> bpe_merges,
    bool add_space_prefix
)
    : model_(model),
      vocabulary_(std::move(vocabulary)),
      special_tokens_(special_tokens),
      encode_candidate_token_ids_(),
      special_encode_candidate_token_ids_(),
      token_text_to_id_(),
      bpe_merge_ranks_(),
      add_space_prefix_(add_space_prefix) {
    encode_candidate_token_ids_.reserve(vocabulary_.size());
    special_encode_candidate_token_ids_.reserve(vocabulary_.size());
    token_text_to_id_.reserve(vocabulary_.size());
    for (std::size_t index = 0; index < vocabulary_.size(); ++index) {
        if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "token id overflow");
        }

        const auto token_id = static_cast<std::uint32_t>(index);
        if (token_id_can_encode_text(token_id)) {
            encode_candidate_token_ids_.push_back(token_id);
            token_text_to_id_.emplace(vocabulary_[index].text, token_id);
        } else if (token_id_can_encode_special(token_id)) {
            special_encode_candidate_token_ids_.push_back(token_id);
        }
    }

    bpe_merge_ranks_.reserve(bpe_merges.size());
    for (std::size_t index = 0; index < bpe_merges.size(); ++index) {
        if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "tokenizer.ggml.merges");
        }

        const auto [left, right] = parse_bpe_merge(bpe_merges[index]);
        if (!bpe_merge_ranks_.emplace(bpe_merge_key(left, right), static_cast<std::uint32_t>(index)).second) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "tokenizer.ggml.merges");
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

    std::sort(
        special_encode_candidate_token_ids_.begin(),
        special_encode_candidate_token_ids_.end(),
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

    append_piece_token_ids(token_ids, text, options.encode_special_tokens);

    if (options.add_eos) {
        token_ids.push_back(require_special_token_id(special_tokens_.eos_token_id, "tokenizer.ggml.eos_token_id"));
    }

    return token_ids;
}

void Tokenizer::append_piece_token_ids(
    std::vector<std::uint32_t>& token_ids,
    std::string_view text,
    bool encode_special_tokens
) const {
    if (!encode_special_tokens) {
        if (model_ == TokenizerModel::gpt2 && !bpe_merge_ranks_.empty()) {
            append_gpt2_bpe_token_ids(token_ids, text);
        } else {
            append_exact_token_ids(token_ids, text);
        }
        return;
    }

    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto special_match = special_token_match_at(text, offset);
        if (special_match.has_value()) {
            token_ids.push_back(special_match->first);
            offset += special_match->second;
            continue;
        }

        const auto piece_start = offset;
        ++offset;
        while (offset < text.size() && !special_token_match_at(text, offset).has_value()) {
            ++offset;
        }

        append_piece_token_ids(token_ids, text.substr(piece_start, offset - piece_start), false);
    }
}

void Tokenizer::append_exact_token_ids(std::vector<std::uint32_t>& token_ids, std::string_view text) const {
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
}

void Tokenizer::append_gpt2_bpe_token_ids(std::vector<std::uint32_t>& token_ids, std::string_view text) const {
    std::string prefixed_text;
    std::string_view text_to_encode = text;
    if (add_space_prefix_ && !text.empty() && !code_point_is_space(read_utf8_code_point(text, 0).value)) {
        prefixed_text.reserve(text.size() + 1U);
        prefixed_text.push_back(' ');
        prefixed_text += text;
        text_to_encode = prefixed_text;
    }

    const auto pieces = gpt2_pretokenize(text_to_encode);
    for (const auto piece : pieces) {
        const auto byte_encoded_piece = gpt2_byte_encode(piece);
        auto symbols = split_utf8_symbols(byte_encoded_piece);

        while (symbols.size() > 1) {
            bool has_best = false;
            std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
            std::string best_left;
            std::string best_right;

            for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
                const auto found = bpe_merge_ranks_.find(bpe_merge_key(symbols[index], symbols[index + 1]));
                if (found == bpe_merge_ranks_.end() || (has_best && found->second >= best_rank)) {
                    continue;
                }

                has_best = true;
                best_rank = found->second;
                best_left = symbols[index];
                best_right = symbols[index + 1];
            }

            if (!has_best) {
                break;
            }

            std::vector<std::string> merged_symbols;
            merged_symbols.reserve(symbols.size());
            std::size_t index = 0;
            while (index < symbols.size()) {
                if (index + 1 < symbols.size() && symbols[index] == best_left && symbols[index + 1] == best_right) {
                    merged_symbols.push_back(symbols[index] + symbols[index + 1]);
                    index += 2;
                    continue;
                }

                merged_symbols.push_back(std::move(symbols[index]));
                ++index;
            }

            symbols = std::move(merged_symbols);
        }

        for (const auto& symbol : symbols) {
            append_token_id_for_text(token_ids, symbol);
        }
    }
}

void Tokenizer::append_token_id_for_text(std::vector<std::uint32_t>& token_ids, std::string_view token_text) const {
    const auto found = token_text_to_id_.find(std::string(token_text));
    if (found != token_text_to_id_.end()) {
        token_ids.push_back(found->second);
        return;
    }

    if (special_tokens_.unknown_token_id.has_value()) {
        token_ids.push_back(*special_tokens_.unknown_token_id);
        return;
    }

    throw TokenizerError(TokenizerErrorCode::no_matching_token, std::string(token_text));
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

bool Tokenizer::token_id_can_encode_special(std::uint32_t token_id) const noexcept {
    if (static_cast<std::uint64_t>(token_id) >= static_cast<std::uint64_t>(vocabulary_.size())) {
        return false;
    }

    if (special_tokens_.unknown_token_id == token_id) {
        return false;
    }

    return token_id_is_special(token_id);
}

std::optional<std::pair<std::uint32_t, std::size_t>> Tokenizer::special_token_match_at(
    std::string_view text,
    std::size_t offset
) const noexcept {
    for (const auto token_id : special_encode_candidate_token_ids_) {
        const auto& token_text = vocabulary_[token_id].text;
        if (starts_with_at(text, offset, token_text)) {
            return std::pair<std::uint32_t, std::size_t>(token_id, token_text.size());
        }
    }

    return std::nullopt;
}

Tokenizer load_tokenizer_from_gguf_metadata(const GGUFMetadataSection& metadata) {
    validate_architecture(metadata);
    const auto model = parse_tokenizer_model(require_metadata_string(metadata, "tokenizer.ggml.model"));
    auto vocabulary = build_vocabulary(metadata);
    auto bpe_merges = load_bpe_merges(metadata);
    const auto special_tokens = load_special_tokens(metadata, vocabulary.size());
    const auto add_space_prefix = load_add_space_prefix(metadata);

    return Tokenizer(model, std::move(vocabulary), special_tokens, std::move(bpe_merges), add_space_prefix);
}

}  // namespace local_gguf_engine::tokenizer
