#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
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
    if (architecture != "gemma2") {
        throw TokenizerError(TokenizerErrorCode::unsupported_architecture, architecture);
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

    throw TokenizerError(TokenizerErrorCode::unsupported_chat_template, "tokenizer.chat_template");
}

Tokenizer::Tokenizer(
    TokenizerModel model,
    std::vector<TokenizerToken> vocabulary,
    TokenizerSpecialTokens special_tokens
)
    : model_(model),
      vocabulary_(std::move(vocabulary)),
      special_tokens_(special_tokens),
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

Tokenizer load_tokenizer_from_gguf_metadata(const GGUFMetadataSection& metadata) {
    validate_architecture(metadata);
    const auto model = parse_tokenizer_model(require_metadata_string(metadata, "tokenizer.ggml.model"));
    auto vocabulary = build_vocabulary(metadata);
    const auto special_tokens = load_special_tokens(metadata, vocabulary.size());

    return Tokenizer(model, std::move(vocabulary), special_tokens);
}

}  // namespace local_gguf_engine::tokenizer
