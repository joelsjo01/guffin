#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
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
constexpr std::int32_t gguf_token_type_byte = 6;

[[nodiscard]] std::optional<std::uint8_t> parse_byte_token(std::string_view text) noexcept {
    if (text.size() != 6U || !text.starts_with("<0x") || text.back() != '>') {
        return std::nullopt;
    }

    const auto hex_value = [](char digit) -> std::optional<std::uint8_t> {
        if (digit >= '0' && digit <= '9') return static_cast<std::uint8_t>(digit - '0');
        if (digit >= 'A' && digit <= 'F') return static_cast<std::uint8_t>(digit - 'A' + 10);
        if (digit >= 'a' && digit <= 'f') return static_cast<std::uint8_t>(digit - 'a' + 10);
        return std::nullopt;
    };

    const auto high = hex_value(text[3]);
    const auto low = hex_value(text[4]);
    if (!high.has_value() || !low.has_value()) return std::nullopt;
    return static_cast<std::uint8_t>((*high << 4U) | *low);
}

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

    throw TokenizerError(TokenizerErrorCode::unsupported_tokenizer_model, std::string(model));
}

void validate_architecture(const GGUFMetadataSection& metadata) {
    const auto architecture = require_metadata_string(metadata, "general.architecture");
    if (architecture != "gemma2" && architecture != "llama" && architecture != "gemma3") {
        throw TokenizerError(TokenizerErrorCode::unsupported_architecture, architecture);
    }
    if (architecture == "llama"
        && (require_metadata_string(metadata, "tokenizer.ggml.normalizer")
                != "identity-no-collapse-byte-fallback"
            || optional_metadata_bool(metadata, "tokenizer.ggml.add_space_prefix") != true)) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "unsupported Llama normalization");
    }
    if (architecture == "gemma3"
        && (optional_metadata_bool(metadata, "tokenizer.ggml.add_space_prefix") != false
            || optional_metadata_string(metadata, "tokenizer.ggml.pre").value_or("default") != "default"
            || optional_metadata_string(metadata, "tokenizer.ggml.normalizer").value_or("identity-no-collapse-byte-fallback")
                != "identity-no-collapse-byte-fallback"
            || optional_metadata_bool(metadata, "tokenizer.ggml.remove_extra_whitespaces").value_or(false)
            || !optional_metadata_bool(metadata, "tokenizer.ggml.escape_whitespaces").value_or(true))) {
        throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "unsupported Gemma3 normalization");
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

[[nodiscard]] bool supports_gemma3_it_template(std::string_view chat_template) noexcept {
    return chat_template.find("{{ bos_token }}") != std::string_view::npos
        && chat_template.find("messages[0]['role'] == 'system'") != std::string_view::npos
        && chat_template.find("first_user_prefix = messages[0]['content'] + '\n\n'") != std::string_view::npos
        && chat_template.find("loop_messages = messages[1:]") != std::string_view::npos
        && chat_template.find("(message['role'] == 'user') != (loop.index0 % 2 == 0)") != std::string_view::npos
        && chat_template.find("Conversation roles must alternate") != std::string_view::npos
        && chat_template.find("set role = \"model\"") != std::string_view::npos
        && chat_template.find("(first_user_prefix if loop.first else \"\")") != std::string_view::npos
        && chat_template.find("message['content'] | trim") != std::string_view::npos
        && chat_template.find("<end_of_turn>\n") != std::string_view::npos
        && chat_template.find("add_generation_prompt") != std::string_view::npos
        && chat_template.find("<start_of_turn>model\n") != std::string_view::npos;
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

[[nodiscard]] std::size_t python_whitespace_prefix_bytes(std::string_view text) noexcept {
    if (text.empty()) return 0;
    const auto first = static_cast<unsigned char>(text[0]);
    // Jinja's trim filter uses Python str.strip(), including U+001C..U+001F.
    // Keep invalid UTF-8 and non-whitespace bytes intact for tokenizer fallback.
    if ((first >= 0x09 && first <= 0x0d) || (first >= 0x1c && first <= 0x20)) return 1;
    if (text.size() >= 2 && first == 0xc2
        && (static_cast<unsigned char>(text[1]) == 0x85 || static_cast<unsigned char>(text[1]) == 0xa0)) return 2;
    if (text.size() < 3) return 0;
    const auto second = static_cast<unsigned char>(text[1]);
    const auto third = static_cast<unsigned char>(text[2]);
    if ((first == 0xe1 && second == 0x9a && third == 0x80)
        || (first == 0xe2 && second == 0x80 && ((third >= 0x80 && third <= 0x8a) || third == 0xa8 || third == 0xa9 || third == 0xaf))
        || (first == 0xe2 && second == 0x81 && third == 0x9f)
        || (first == 0xe3 && second == 0x80 && third == 0x80)) return 3;
    return 0;
}

[[nodiscard]] std::string_view trim_python_whitespace(std::string_view text) noexcept {
    while (const auto count = python_whitespace_prefix_bytes(text)) text.remove_prefix(count);
    while (!text.empty()) {
        auto last = text.size() - 1;
        while (last > 0 && (static_cast<unsigned char>(text[last]) & 0xc0U) == 0x80U) --last;
        const auto count = python_whitespace_prefix_bytes(text.substr(last));
        if (count == 0 || count != text.size() - last) break;
        text.remove_suffix(count);
    }
    return text;
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

[[nodiscard]] PromptFormatResult format_gemma3_it_chat_prompt(
    const GGUFMetadataSection& metadata,
    std::span<const ChatMessage> messages,
    bool add_generation_prompt
) {
    if (messages.empty()) throw TokenizerError(TokenizerErrorCode::invalid_chat_message, "messages");
    const bool has_system = messages.front().role == "system";
    const auto conversation = messages.subspan(has_system ? 1 : 0);
    if (conversation.empty()) throw TokenizerError(TokenizerErrorCode::invalid_chat_message, "missing user message");
    std::string formatted = bos_token_text(metadata);
    for (std::size_t index = 0; index < conversation.size(); ++index) {
        // Public text-chat roles are user/assistant/model. System is only legal
        // first, where the released template folds its *untrimmed* content into
        // the first user turn. Later user/assistant contents use Unicode trim.
        const auto role = gemma2_output_role_for_message(conversation[index], index);
        formatted += "<start_of_turn>";
        formatted += role;
        formatted.push_back('\n');
        if (index == 0 && has_system) {
            formatted += messages.front().content;
            formatted += "\n\n";
        }
        formatted += trim_python_whitespace(conversation[index].content);
        formatted += "<end_of_turn>\n";
    }
    if (add_generation_prompt) formatted += "<start_of_turn>model\n";
    return {.text = std::move(formatted), .encode_special_tokens = true, .add_bos = false, .add_eos = false};
}

void append_sentencepiece_decoded_text(std::string& output, std::string_view token_text) {
    constexpr std::string_view space_marker = "\xE2\x96\x81";
    std::size_t offset = 0;
    while (offset < token_text.size()) {
        const auto marker_offset = token_text.find(space_marker, offset);
        if (marker_offset == std::string_view::npos) {
            output += token_text.substr(offset);
            return;
        }

        output += token_text.substr(offset, marker_offset - offset);
        output.push_back(' ');
        offset = marker_offset + space_marker.size();
    }
}

void append_decoded_token_text(
    std::string& output,
    TokenizerModel model,
    std::string_view token_text
) {
    switch (model) {
        case TokenizerModel::llama:
            append_sentencepiece_decoded_text(output, token_text);
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
    if (architecture == "gemma3" && supports_gemma3_it_template(*chat_template)) {
        return format_gemma3_it_chat_prompt(metadata, messages, add_generation_prompt);
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
    if ((architecture == "gemma2" && supports_gemma2_it_template(*chat_template))
        || (architecture == "gemma3" && supports_gemma3_it_template(*chat_template))) {
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

    throw TokenizerError(TokenizerErrorCode::unsupported_chat_template, "tokenizer.chat_template");
}

Tokenizer::Tokenizer(
    TokenizerModel model,
    std::vector<TokenizerToken> vocabulary,
    TokenizerSpecialTokens special_tokens,
    bool add_space_prefix,
    TokenizerProfile profile
)
    : model_(model),
      vocabulary_(std::move(vocabulary)),
      special_tokens_(special_tokens),
      add_space_prefix_(add_space_prefix),
      profile_(profile),
      encode_candidate_token_ids_(),
      special_encode_candidate_token_ids_(),
      token_text_to_id_(),
      byte_token_ids_() {
    encode_candidate_token_ids_.reserve(vocabulary_.size());
    special_encode_candidate_token_ids_.reserve(vocabulary_.size());
    token_text_to_id_.reserve(vocabulary_.size());
    for (std::size_t index = 0; index < vocabulary_.size(); ++index) {
        if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "token id overflow");
        }

        const auto token_id = static_cast<std::uint32_t>(index);
        if (vocabulary_[index].type == gguf_token_type_byte) {
            const auto byte = parse_byte_token(vocabulary_[index].text);
            if (!byte.has_value() || byte_token_ids_[*byte].has_value()) {
                throw TokenizerError(
                    TokenizerErrorCode::invalid_vocabulary,
                    "invalid byte token at id " + std::to_string(token_id)
                );
            }
            byte_token_ids_[*byte] = token_id;
            continue;
        }

        if (token_id_can_encode_text(token_id)) {
            encode_candidate_token_ids_.push_back(token_id);
            token_text_to_id_.emplace(vocabulary_[index].text, token_id);
        } else if (token_id_can_encode_special(token_id)) {
            special_encode_candidate_token_ids_.push_back(token_id);
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
    if (profile_ == TokenizerProfile::gemma3) {
        // SentencePiece user-defined symbols are longest-prefix, indivisible
        // pieces. A prepared trie avoids a vocabulary scan per input byte. The
        // authenticated GGUF converts source ▁-runs to literal spaces and marks
        // <unused*>/mask as controls; their source behavior is still atomic.
        added_token_trie_.emplace_back();
        const auto insert = [&](std::string_view piece, std::uint32_t token) {
            std::uint32_t node = 0;
            for (const char character : piece) {
                const auto byte = static_cast<unsigned char>(character);
                auto child = added_token_trie_[node].child;
                while (child && added_token_trie_[child].byte != byte) child = added_token_trie_[child].sibling;
                if (!child) {
                    if (added_token_trie_.size() == UINT32_MAX) throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "added-token trie overflow");
                    child = static_cast<std::uint32_t>(added_token_trie_.size());
                    added_token_trie_.push_back({0, added_token_trie_[node].child, UINT32_MAX, byte});
                    added_token_trie_[node].child = child;
                }
                node = child;
            }
            auto& prior = added_token_trie_[node].token;
            if (prior != UINT32_MAX && prior != token) throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "ambiguous Gemma3 added-token alias");
            prior = token;
        };
        for (std::uint32_t id = 0; id < vocabulary_.size(); ++id) {
            const auto& token = vocabulary_[id];
            if (token.type != 3 && token.type != 4 && token.type != 2) continue;
            insert(token.text, id);
            if (token.type == 4 && token.text.find_first_not_of(' ') == std::string::npos) {
                std::string markers;
                for (std::size_t i = 0; i < token.text.size(); ++i) markers += "▁";
                insert(markers, id);
            }
        }
    }
}

TokenizerModel Tokenizer::model() const noexcept {
    return model_;
}

TokenizerProfile Tokenizer::profile() const noexcept { return profile_; }

std::string_view Tokenizer::model_name() const noexcept {
    return tokenizer_model_name(model_);
}

const std::vector<TokenizerToken>& Tokenizer::vocabulary() const noexcept {
    return vocabulary_;
}

const TokenizerSpecialTokens& Tokenizer::special_tokens() const noexcept {
    return special_tokens_;
}

const runtime::Sha256Digest& Tokenizer::identity() const noexcept { return identity_; }

TokenizerEncodeOptions Tokenizer::default_encode_options() const noexcept {
    return TokenizerEncodeOptions {
        .add_bos = special_tokens_.add_bos_token,
        .add_eos = special_tokens_.add_eos_token,
    };
}

std::size_t Tokenizer::owned_bytes() const noexcept {
    std::size_t bytes = vocabulary_.capacity() * sizeof(TokenizerToken)
        + encode_candidate_token_ids_.capacity() * sizeof(std::uint32_t)
        + special_encode_candidate_token_ids_.capacity() * sizeof(std::uint32_t)
        + added_token_trie_.capacity() * sizeof(AddedTokenNode)
        + token_text_to_id_.bucket_count() * sizeof(void*)
        + token_text_to_id_.size() * sizeof(decltype(token_text_to_id_)::value_type);
    for (const auto& token : vocabulary_) bytes += token.text.capacity() + 1U;
    for (const auto& [text, id] : token_text_to_id_) { (void)id; bytes += text.capacity() + 1U; }
    return bytes;
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
    return encode(text, default_encode_options());
}

std::vector<std::uint32_t> Tokenizer::encode(
    std::string_view text,
    TokenizerEncodeOptions options
) const {
    if (add_space_prefix_ || profile_ == TokenizerProfile::gemma3) {
        TokenizerEncodeWorkspace workspace(text.size());
        std::vector<std::uint32_t> output(text.size() * 3U + 5U);
        output.resize(encode_into(text, output, workspace, options));
        return output;
    }
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
        append_exact_token_ids(token_ids, text);
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
    constexpr std::string_view sentencepiece_space = "\xE2\x96\x81";
    std::string escaped;
    escaped.reserve(text.size());
    for (const auto character : text) {
        if (character == ' ') {
            escaped += sentencepiece_space;
        } else {
            escaped.push_back(character);
        }
    }

    struct Symbol {
        std::size_t offset = 0;
        std::size_t size = 0;
        std::int64_t previous = -1;
        std::int64_t next = -1;
    };
    struct Bigram {
        std::size_t left = 0;
        std::size_t right = 0;
        float score = 0.0F;
        std::size_t size = 0;
    };
    struct HigherPriorityBigram {
        bool operator()(const Bigram& left, const Bigram& right) const noexcept {
            return left.score < right.score
                || (left.score == right.score && left.left > right.left);
        }
    };

    std::vector<Symbol> symbols;
    for (std::size_t offset = 0; offset < escaped.size();) {
        const auto lead = static_cast<unsigned char>(escaped[offset]);
        std::size_t size = 1;
        if ((lead & 0xE0U) == 0xC0U) size = 2;
        else if ((lead & 0xF0U) == 0xE0U) size = 3;
        else if ((lead & 0xF8U) == 0xF0U) size = 4;
        size = std::min(size, escaped.size() - offset);
        symbols.push_back(Symbol {
            .offset = offset,
            .size = size,
            .previous = static_cast<std::int64_t>(symbols.size()) - 1,
            .next = -1,
        });
        if (symbols.size() > 1U) {
            symbols[symbols.size() - 2U].next = static_cast<std::int64_t>(symbols.size() - 1U);
        }
        offset += size;
    }

    std::priority_queue<Bigram, std::vector<Bigram>, HigherPriorityBigram> candidates;
    const auto add_candidate = [&](std::int64_t left_index, std::int64_t right_index) {
        if (left_index < 0 || right_index < 0) return;
        const auto left = static_cast<std::size_t>(left_index);
        const auto right = static_cast<std::size_t>(right_index);
        const auto combined = escaped.substr(symbols[left].offset, symbols[left].size + symbols[right].size);
        const auto found = token_text_to_id_.find(combined);
        if (found == token_text_to_id_.end()) return;
        candidates.push(Bigram {
            .left = left,
            .right = right,
            .score = vocabulary_[found->second].score.value_or(0.0F),
            .size = combined.size(),
        });
    };

    for (std::size_t index = 1; index < symbols.size(); ++index) {
        add_candidate(static_cast<std::int64_t>(index - 1U), static_cast<std::int64_t>(index));
    }
    while (!candidates.empty()) {
        const auto candidate = candidates.top();
        candidates.pop();
        auto& left = symbols[candidate.left];
        auto& right = symbols[candidate.right];
        if (left.size == 0U || right.size == 0U || left.size + right.size != candidate.size
            || left.next != static_cast<std::int64_t>(candidate.right)) {
            continue;
        }
        left.size += right.size;
        right.size = 0;
        left.next = right.next;
        if (right.next >= 0) symbols[static_cast<std::size_t>(right.next)].previous = static_cast<std::int64_t>(candidate.left);
        add_candidate(left.previous, static_cast<std::int64_t>(candidate.left));
        add_candidate(static_cast<std::int64_t>(candidate.left), left.next);
    }

    for (std::int64_t index = symbols.empty() ? -1 : 0; index >= 0;) {
        const auto& symbol = symbols[static_cast<std::size_t>(index)];
        const auto piece = escaped.substr(symbol.offset, symbol.size);
        const auto found = token_text_to_id_.find(piece);
        if (found != token_text_to_id_.end()) {
            token_ids.push_back(found->second);
        } else {
            bool has_all_byte_tokens = true;
            for (const auto character : piece) {
                const auto byte = static_cast<unsigned char>(character);
                if (!byte_token_ids_[byte].has_value()) {
                    has_all_byte_tokens = false;
                    break;
                }
            }
            if (has_all_byte_tokens) {
                for (const auto character : piece) {
                    const auto byte = static_cast<unsigned char>(character);
                    token_ids.push_back(*byte_token_ids_[byte]);
                }
            } else if (special_tokens_.unknown_token_id.has_value()) {
                token_ids.push_back(*special_tokens_.unknown_token_id);
            } else {
                throw TokenizerError(TokenizerErrorCode::no_matching_token, byte_offset_detail(symbol.offset));
            }
        }
        index = symbol.next;
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
    if (add_space_prefix_) {
        std::size_t capacity = 0;
        for (const auto id : token_ids) {
            if (id >= vocabulary_.size()) throw TokenizerError(TokenizerErrorCode::token_id_out_of_range, std::to_string(id));
            capacity += vocabulary_[id].text.size();
        }
        std::string output(capacity, '\0');
        output.resize(decode_into(token_ids, output, options));
        return output;
    }
    std::string output;
    for (const auto token_id : token_ids) {
        append_decoded_token(output, token_id, options);
    }

    return output;
}

void Tokenizer::append_decoded_token(
    std::string& output,
    std::uint32_t token_id,
    TokenizerDecodeOptions options
) const {
    if (static_cast<std::uint64_t>(token_id) >= static_cast<std::uint64_t>(vocabulary_.size())) {
        throw TokenizerError(TokenizerErrorCode::token_id_out_of_range, std::to_string(token_id));
    }
    if (options.skip_special_tokens && token_id_is_special(token_id)) {
        return;
    }

    const auto& token = vocabulary_[token_id];
    if (token.type == gguf_token_type_byte) {
        const auto byte = parse_byte_token(token.text);
        if (!byte.has_value()) {
            throw TokenizerError(
                TokenizerErrorCode::invalid_vocabulary,
                "invalid byte token at id " + std::to_string(token_id)
            );
        }
        output.push_back(static_cast<char>(*byte));
        return;
    }
    append_decoded_token_text(output, model_, token.text);
}

bool Tokenizer::token_id_is_special(std::uint32_t token_id) const noexcept {
    if (profile_ == TokenizerProfile::gemma3 && token_id < vocabulary_.size()) {
        const auto& text = vocabulary_[token_id].text;
        return text == "<pad>" || text == "<eos>" || text == "<bos>" || text == "<unk>"
            || text == "<start_of_turn>" || text == "<end_of_turn>"
            || text == "<start_of_image>" || text == "<end_of_image>";
    }
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
    return type != gguf_token_type_unknown && type != gguf_token_type_byte;
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

std::optional<std::pair<std::uint32_t, std::size_t>> Tokenizer::added_token_match_at(
    std::string_view text, std::size_t offset, bool include_special, bool specials_only
) const noexcept {
    std::optional<std::pair<std::uint32_t, std::size_t>> result;
    std::uint32_t node = 0;
    for (auto cursor = offset; cursor < text.size();) {
        auto child = added_token_trie_[node].child;
        const auto byte = static_cast<unsigned char>(text[cursor]);
        while (child && added_token_trie_[child].byte != byte) child = added_token_trie_[child].sibling;
        if (!child) break;
        node = child;
        ++cursor;
        const auto id = added_token_trie_[node].token;
        if (id != UINT32_MAX && (include_special || !token_id_is_special(id))
            && (!specials_only || token_id_is_special(id))) result = {{id, cursor - offset}};
    }
    return result;
}

TokenizerEncodeWorkspace::TokenizerEncodeWorkspace(std::size_t max_input_bytes)
    : max_input_bytes_(max_input_bytes) {
    static_cast<void>(estimate_owned_bytes(max_input_bytes));
    const auto bound = max_input_bytes * 3U + 3U;
    escaped_.reserve(bound);
    // Normalization expands bytes, not symbol count: every input byte yields
    // at most one codepoint, plus the dummy prefix. A merge adds at most two
    // heap entries, and there are at most n-1 merges.
    symbols_.reserve(max_input_bytes + 1U);
    candidates_.reserve((max_input_bytes + 1U) * 3U);
}

std::size_t TokenizerEncodeWorkspace::estimate_owned_bytes(std::size_t max_input_bytes) {
    // libc++ reserves vector storage at the requested count and rounds string
    // storage to an allocation boundary. Thirty-two extra bytes cover that
    // rounding, the terminator and the empty string's inline capacity on the
    // supported Mac runtime. This excludes allocator bookkeeping like the
    // actual owned_bytes() ledger; it is not a process-RSS estimate.
    constexpr auto per_input = 3U + sizeof(Symbol) + 3U * sizeof(Bigram);
    constexpr auto fixed = 3U + 32U + sizeof(Symbol) + 3U * sizeof(Bigram);
    if (max_input_bytes > (std::numeric_limits<std::size_t>::max() - fixed) / per_input)
        throw std::length_error("tokenizer workspace capacity overflow");
    return max_input_bytes * per_input + fixed;
}

std::size_t TokenizerEncodeWorkspace::owned_bytes() const noexcept {
    return escaped_.capacity() + 1U + symbols_.capacity() * sizeof(Symbol)
        + candidates_.capacity() * sizeof(Bigram);
}

std::size_t Tokenizer::encode_into(
    std::string_view text, std::span<std::uint32_t> output,
    TokenizerEncodeWorkspace& workspace, TokenizerEncodeOptions options
) const {
    if (text.size() > workspace.max_input_bytes_) throw std::length_error("tokenizer input capacity exceeded");
    std::size_t written = 0;
    const auto emit = [&](std::uint32_t id) {
        if (written == output.size()) throw std::length_error("tokenizer output capacity exceeded");
        output[written++] = id;
    };
    if (options.add_bos) emit(require_special_token_id(special_tokens_.bos_token_id, "tokenizer.ggml.bos_token_id"));
    const auto special_at = [&](std::size_t offset) -> std::optional<std::pair<std::uint32_t, std::size_t>> {
        if (profile_ == TokenizerProfile::gemma3) {
            if (options.encode_special_tokens && text.substr(offset).starts_with("<image_soft_token>")) {
                throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "Gemma3 multimodal token is outside the text embedding vocabulary");
            }
            // Non-control user-defined pieces are frozen after normalization:
            // mixed ASCII spaces and literal ▁ must form the same longest run.
            return options.encode_special_tokens ? added_token_match_at(text, offset, true, true) : std::nullopt;
        }
        if (!options.encode_special_tokens) return std::nullopt;
        auto match = special_token_match_at(text, offset);
        if (!match && add_space_prefix_ && special_tokens_.unknown_token_id) {
            const auto id = *special_tokens_.unknown_token_id;
            if (text.substr(offset).starts_with(vocabulary_[id].text)) match = {{id, vocabulary_[id].text.size()}};
        }
        return match;
    };
    const auto lower_priority = [](const auto& left, const auto& right) {
        return left.score < right.score || (left.score == right.score && left.left > right.left);
    };
    auto& escaped = workspace.escaped_;
    auto& symbols = workspace.symbols_;
    auto& candidates = workspace.candidates_;
    for (std::size_t segment_start = 0; segment_start < text.size();) {
        if (const auto match = special_at(segment_start)) {
            emit(match->first);
            segment_start += match->second;
            continue;
        }
        auto segment_end = segment_start + 1;
        while (segment_end < text.size() && !special_at(segment_end)) ++segment_end;
        escaped.clear();
        symbols.clear();
        candidates.clear();
        constexpr std::string_view marker = "\xE2\x96\x81";
        if (add_space_prefix_) escaped += marker;
        // The pinned SP normalizer is identity with no whitespace collapsing.
        // This byte API first applies Unicode maximal-subpart replacement, like
        // decoding the target's byte stream to a Python Unicode string before
        // handing it to the reference Llama tokenizer.
        for (auto offset = segment_start; offset < segment_end;) {
            const auto lead = static_cast<unsigned char>(text[offset]);
            if (lead == ' ') { escaped += marker; ++offset; continue; }
            std::size_t size = lead < 0x80U ? 1U
                : (lead >= 0xC2U && lead <= 0xDFU ? 2U
                : (lead >= 0xE0U && lead <= 0xEFU ? 3U
                : (lead >= 0xF0U && lead <= 0xF4U ? 4U : 0U)));
            bool valid = size != 0 && size <= segment_end - offset;
            for (std::size_t byte = 1; valid && byte < size; ++byte) {
                const auto next = static_cast<unsigned char>(text[offset + byte]);
                valid = next >= 0x80U && next <= 0xBFU;
            }
            if (valid && size >= 3) {
                const auto second = static_cast<unsigned char>(text[offset + 1]);
                valid = !((lead == 0xE0U && second < 0xA0U) || (lead == 0xEDU && second >= 0xA0U)
                    || (lead == 0xF0U && second < 0x90U) || (lead == 0xF4U && second >= 0x90U));
            }
            if (!valid) {
                std::size_t consumed = 1;
                while (consumed < size && offset + consumed < segment_end) {
                    const auto next = static_cast<unsigned char>(text[offset + consumed]);
                    if (next < 0x80U || next > 0xBFU) break;
                    if (consumed == 1 && ((lead == 0xE0U && next < 0xA0U) || (lead == 0xEDU && next >= 0xA0U)
                        || (lead == 0xF0U && next < 0x90U) || (lead == 0xF4U && next >= 0x90U))) break;
                    ++consumed;
                }
                escaped += "\xEF\xBF\xBD";
                offset += consumed;
            }
            else { escaped.append(text.data() + offset, size); offset += size; }
        }
        for (std::size_t offset = 0; offset < escaped.size();) {
            const auto lead = static_cast<unsigned char>(escaped[offset]);
            const auto frozen = profile_ == TokenizerProfile::gemma3
                ? added_token_match_at(escaped, offset, options.encode_special_tokens) : std::nullopt;
            const std::size_t size = frozen ? frozen->second
                : (lead < 0x80U ? 1U : (lead < 0xE0U ? 2U : (lead < 0xF0U ? 3U : 4U)));
            symbols.push_back({offset, size, static_cast<std::int64_t>(symbols.size()) - 1, -1,
                frozen ? frozen->first : UINT32_MAX});
            if (symbols.size() > 1) symbols[symbols.size() - 2].next = static_cast<std::int64_t>(symbols.size() - 1);
            offset += size;
        }
        const auto add_candidate = [&](std::int64_t left_index, std::int64_t right_index) {
            if (left_index < 0 || right_index < 0) return;
            const auto left = static_cast<std::size_t>(left_index);
            const auto right = static_cast<std::size_t>(right_index);
            if (symbols[left].frozen_token != UINT32_MAX || symbols[right].frozen_token != UINT32_MAX) return;
            const std::string_view combined(escaped.data() + symbols[left].offset, symbols[left].size + symbols[right].size);
            const auto found = token_text_to_id_.find(combined);
            if (found == token_text_to_id_.end()) return;
            if (candidates.size() == candidates.capacity()) throw std::length_error("tokenizer merge capacity exceeded");
            candidates.push_back({left, right, vocabulary_[found->second].score.value_or(0.0F), combined.size()});
            std::push_heap(candidates.begin(), candidates.end(), lower_priority);
        };
        for (std::size_t index = 1; index < symbols.size(); ++index) add_candidate(static_cast<std::int64_t>(index - 1), static_cast<std::int64_t>(index));
        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end(), lower_priority);
            const auto candidate = candidates.back();
            candidates.pop_back();
            auto& left = symbols[candidate.left];
            auto& right = symbols[candidate.right];
            if (!left.size || !right.size || left.size + right.size != candidate.size
                || left.next != static_cast<std::int64_t>(candidate.right)) continue;
            left.size += right.size;
            right.size = 0;
            left.next = right.next;
            if (right.next >= 0) symbols[static_cast<std::size_t>(right.next)].previous = static_cast<std::int64_t>(candidate.left);
            add_candidate(left.previous, static_cast<std::int64_t>(candidate.left));
            add_candidate(static_cast<std::int64_t>(candidate.left), left.next);
        }
        for (std::int64_t index = symbols.empty() ? -1 : 0; index >= 0;) {
            const auto& symbol = symbols[static_cast<std::size_t>(index)];
            const std::string_view piece(escaped.data() + symbol.offset, symbol.size);
            const auto found = token_text_to_id_.find(piece);
            if (symbol.frozen_token != UINT32_MAX) emit(symbol.frozen_token);
            else if (found != token_text_to_id_.end()) emit(found->second);
            else {
                bool complete = true;
                for (const auto byte : piece) complete &= byte_token_ids_[static_cast<unsigned char>(byte)].has_value();
                if (complete) for (const auto byte : piece) emit(*byte_token_ids_[static_cast<unsigned char>(byte)]);
                else emit(require_special_token_id(special_tokens_.unknown_token_id, "tokenizer.ggml.unknown_token_id"));
            }
            index = symbol.next;
        }
        segment_start = segment_end;
    }
    if (options.add_eos) emit(require_special_token_id(special_tokens_.eos_token_id, "tokenizer.ggml.eos_token_id"));
    return written;
}

std::size_t Tokenizer::decode_into(
    std::span<const std::uint32_t> ids, std::span<char> output, TokenizerDecodeOptions options
) const {
    std::size_t written = 0;
    bool first_piece = true;
    const auto emit = [&](char byte) {
        if (written == output.size()) throw std::length_error("tokenizer decode capacity exceeded");
        output[written++] = byte;
    };
    constexpr std::string_view marker = "\xE2\x96\x81";
    for (const auto id : ids) {
        if (id >= vocabulary_.size()) throw TokenizerError(TokenizerErrorCode::token_id_out_of_range, std::to_string(id));
        if (options.skip_special_tokens && (token_id_is_special(id) || special_tokens_.unknown_token_id == id)) continue;
        const auto& token = vocabulary_[id];
        if (token.type == gguf_token_type_byte) emit(static_cast<char>(*parse_byte_token(token.text)));
        else {
            std::size_t offset = add_space_prefix_ && first_piece && token.text.starts_with(marker) ? marker.size() : 0;
            while (offset < token.text.size()) {
                if (std::string_view(token.text).substr(offset).starts_with(marker)) { emit(' '); offset += marker.size(); }
                else emit(token.text[offset++]);
            }
        }
        first_piece = false;
    }
    return written;
}

Tokenizer load_tokenizer_from_gguf_metadata(const GGUFMetadataSection& metadata) {
    validate_architecture(metadata);
    const auto model = parse_tokenizer_model(require_metadata_string(metadata, "tokenizer.ggml.model"));
    auto vocabulary = build_vocabulary(metadata);
    const auto special_tokens = load_special_tokens(metadata, vocabulary.size());

    const auto architecture = require_metadata_string(metadata, "general.architecture");
    const bool add_space_prefix = architecture == "llama";
    const bool gemma3 = architecture == "gemma3";
    if (add_space_prefix || gemma3) {
        std::array<bool, 256> bytes {};
        for (const auto& token : vocabulary) {
            if (!token.score.has_value() || !std::isfinite(*token.score) || !token.type.has_value()
                || (*token.type != 1 && *token.type != 2 && *token.type != 3 && *token.type != 6 && !(gemma3 && *token.type == 4))) {
                throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "unsupported standalone token profile");
            }
            if (*token.type == gguf_token_type_byte) {
                const auto byte = parse_byte_token(token.text);
                if (!byte || bytes[*byte]) throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "Llama byte fallback");
                bytes[*byte] = true;
            }
        }
        if (!std::all_of(bytes.begin(), bytes.end(), [](bool value) { return value; })) {
            throw TokenizerError(TokenizerErrorCode::invalid_vocabulary, "incomplete Llama byte fallback");
        }
    }
    auto tokenizer = Tokenizer(model, std::move(vocabulary), special_tokens, add_space_prefix,
        gemma3 ? TokenizerProfile::gemma3 : (add_space_prefix ? TokenizerProfile::llama_legacy : TokenizerProfile::gemma2));
    tokenizer.identity_ = tokenizer_identity_from_gguf_metadata(metadata);
    return tokenizer;
}

runtime::Sha256Digest tokenizer_identity_from_gguf_metadata(const GGUFMetadataSection& metadata) {
    validate_architecture(metadata);
    const auto model = parse_tokenizer_model(require_metadata_string(metadata, "tokenizer.ggml.model"));
    const auto& texts = require_metadata_array_values<std::vector<std::string>>(
        metadata, "tokenizer.ggml.tokens", GGUFMetadataValueType::string);
    const auto scores = optional_metadata_array_values<std::vector<float>>(
        metadata, "tokenizer.ggml.scores", GGUFMetadataValueType::float32);
    const auto types = optional_metadata_array_values<std::vector<std::int32_t>>(
        metadata, "tokenizer.ggml.token_type", GGUFMetadataValueType::int32);
    if (scores) validate_optional_array_length("tokenizer.ggml.scores", scores->get().size(), texts.size());
    if (types) validate_optional_array_length("tokenizer.ggml.token_type", types->get().size(), texts.size());
    const auto specials = load_special_tokens(metadata, texts.size());
    runtime::Sha256 hash;
    const auto integer = [&](std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes {};
        for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
        hash.update(bytes);
    };
    const auto text = [&](std::string_view value) {
        integer(value.size());
        hash.update({reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
    };
    text("local-gguf-engine/tokenizer-semantics/v1");
    integer(static_cast<std::uint64_t>(model));
    const auto architecture = require_metadata_string(metadata, "general.architecture");
    integer(architecture == "llama" ? 1U : (architecture == "gemma3" ? 2U : 0U));
    for (const auto id : {specials.bos_token_id, specials.eos_token_id, specials.unknown_token_id}) {
        integer(id.has_value() ? 1U : 0U);
        integer(id.value_or(0));
    }
    integer(specials.add_bos_token ? 1U : 0U);
    integer(specials.add_eos_token ? 1U : 0U);
    integer(scores.has_value() ? 1U : 0U);
    integer(types.has_value() ? 1U : 0U);
    integer(texts.size());
    for (std::size_t i = 0; i < texts.size(); ++i) {
        text(texts[i]);
        if (scores) integer(std::bit_cast<std::uint32_t>(scores->get()[i]));
        if (types) integer(static_cast<std::uint32_t>(types->get()[i]));
    }
    return hash.finalize();
}

}  // namespace local_gguf_engine::tokenizer
