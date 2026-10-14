#include "tokenizer_tests.hpp"
#include "gemma3_chat_template.hpp"

#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::metadata_bool;
using local_gguf_engine::tests::metadata_f32_array;
using local_gguf_engine::tests::metadata_i32_array;
using local_gguf_engine::tests::metadata_section;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_string_array;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::gemma2_chat_template_fixture;
using local_gguf_engine::tokenizer::ChatMessage;
using local_gguf_engine::tokenizer::format_chat_prompt_from_gguf_metadata;
using local_gguf_engine::tokenizer::format_generate_prompt_from_gguf_metadata;
using local_gguf_engine::tokenizer::load_tokenizer_from_gguf_metadata;
using local_gguf_engine::tokenizer::TokenizerDecodeOptions;
using local_gguf_engine::tokenizer::TokenizerEncodeOptions;
using local_gguf_engine::tokenizer::TokenizerEncodeWorkspace;
using local_gguf_engine::tokenizer::TokenizerError;
using local_gguf_engine::tokenizer::TokenizerErrorCode;
using local_gguf_engine::tokenizer::TokenizerModel;
using local_gguf_engine::tokenizer::tokenizer_model_name;

std::vector<std::string> tiny_gemma2_token_texts() {
    std::vector<std::string> tokens;
    tokens.reserve(110);
    for (std::uint32_t token_id = 0; token_id < 110; ++token_id) {
        tokens.push_back("<unused" + std::to_string(token_id) + ">");
    }

    tokens[0] = "<pad>";
    tokens[1] = "<eos>";
    tokens[2] = "<bos>";
    tokens[3] = "<unk>";
    tokens[4] = "<mask>";
    tokens[5] = "Hello";
    tokens[6] = "First";
    tokens[7] = "Second";
    tokens[8] = "Answer";
    tokens[9] = "user\n";
    tokens[10] = "model\n";
    tokens[11] = "\n";
    tokens[12] = "\xE2\x96\x81seems";
    tokens[13] = "\xE2\x96\x81question";
    tokens[14] = "\xE2\x96\x81world";
    tokens[15] = "!";
    tokens[16] = "Hello\xE2\x96\x81world";
    tokens[17] = "H";
    tokens[18] = "e";
    tokens[19] = "l";
    tokens[20] = "o";
    tokens[21] = "He";
    tokens[22] = "Hel";
    tokens[23] = "Hell";
    tokens[24] = "\xE2\x96\x81";
    tokens[25] = "w";
    tokens[26] = "r";
    tokens[27] = "d";
    tokens[28] = "\xE2\x96\x81w";
    tokens[29] = "\xE2\x96\x81wo";
    tokens[30] = "\xE2\x96\x81wor";
    tokens[31] = "\xE2\x96\x81worl";
    tokens[32] = "u";
    tokens[33] = "s";
    tokens[34] = "us";
    tokens[35] = "use";
    tokens[36] = "user";
    tokens[37] = "m";
    tokens[38] = "mo";
    tokens[39] = "mod";
    tokens[40] = "mode";
    tokens[41] = "model";
    tokens[42] = "<0xC3>";
    tokens[43] = "<0xA9>";
    tokens[44] = "<0xF0>";
    tokens[45] = "<0x9F>";
    tokens[46] = "<0x98>";
    tokens[47] = "<0x80>";
    tokens[106] = "<start_of_turn>";
    tokens[107] = "<end_of_turn>";
    return tokens;
}

std::vector<std::int32_t> tiny_gemma2_token_types() {
    std::vector<std::int32_t> token_types(110, 1);
    token_types[0] = 3;
    token_types[1] = 3;
    token_types[2] = 3;
    token_types[3] = 2;
    token_types[4] = 3;
    for (std::size_t token_id = 42; token_id <= 47; ++token_id) {
        token_types[token_id] = 6;
    }
    token_types[106] = 3;
    token_types[107] = 3;
    return token_types;
}

std::vector<GGUFMetadataEntry> tiny_gemma2_tokenizer_metadata() {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string_array("tokenizer.ggml.tokens", tiny_gemma2_token_texts()),
        metadata_f32_array("tokenizer.ggml.scores", std::vector<float>(110, 0.0F)),
        metadata_i32_array("tokenizer.ggml.token_type", tiny_gemma2_token_types()),
        metadata_u32("tokenizer.ggml.bos_token_id", 2),
        metadata_u32("tokenizer.ggml.eos_token_id", 1),
        metadata_u32("tokenizer.ggml.unknown_token_id", 3),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
        metadata_bool("tokenizer.ggml.add_space_prefix", false),
        metadata_string("tokenizer.chat_template", gemma2_chat_template_fixture()),
    };
}

constexpr std::uint32_t gemma2_added_byte_base = 27;

std::vector<GGUFMetadataEntry> gemma2_added_piece_metadata() {
    // Raw GGUF user-defined pieces are indivisible before space escaping. The
    // high-score ordinary pieces make crossing one of these boundaries tempting.
    std::vector<std::string> tokens {
        "<pad>", "<eos>", "<bos>", "<unk>", "A", "B", "▁", "▁B",
        "  ", "   ", "    ", "\n", "\n\n", "<code>", "</code>",
        "<start_of_turn>", "<end_of_turn>",
        "A▁", "A▁▁", "A▁▁▁", "A▁▁▁▁", "A▁▁▁▁B",
        "A\n", "A\n\n", "\n\nB", "A<code>", "</s>",
    };
    std::vector<std::int32_t> types(tokens.size(), 1);
    types[0] = types[1] = types[2] = types[15] = types[16] = 3;
    types[3] = 2;
    for (std::size_t id = 8; id <= 14; ++id) types[id] = 4;
    types[26] = 4;
    std::vector<float> scores(tokens.size(), 100.0F);
    for (std::size_t byte = 0; byte < 256; ++byte) {
        constexpr char hex[] = "0123456789ABCDEF";
        std::string piece = "<0x00>";
        piece[3] = hex[byte >> 4];
        piece[4] = hex[byte & 15];
        tokens.push_back(std::move(piece));
        types.push_back(6);
        scores.push_back(0.0F);
    }
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string_array("tokenizer.ggml.tokens", std::move(tokens)),
        metadata_f32_array("tokenizer.ggml.scores", std::move(scores)),
        metadata_i32_array("tokenizer.ggml.token_type", std::move(types)),
        metadata_u32("tokenizer.ggml.bos_token_id", 2),
        metadata_u32("tokenizer.ggml.eos_token_id", 1),
        metadata_u32("tokenizer.ggml.unknown_token_id", 3),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
        metadata_bool("tokenizer.ggml.add_space_prefix", false),
    };
}

std::vector<GGUFMetadataEntry> tiny_gemma3_tokenizer_metadata() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    auto tokens = tiny_gemma2_token_texts();
    auto types = tiny_gemma2_token_types();
    for (std::size_t id = 42; id <= 47; ++id) {
        tokens[id] = "<unused" + std::to_string(id) + ">";
        types[id] = 1;
    }
    constexpr std::string_view hex = "0123456789ABCDEF";
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        std::string spelling = "<0x00>";
        spelling[3] = hex[byte >> 4U];
        spelling[4] = hex[byte & 15U];
        tokens.push_back(std::move(spelling));
        types.push_back(6);
    }
    for (auto& entry : metadata) {
        if (entry.key == "general.architecture") entry = metadata_string(entry.key, "gemma3");
        if (entry.key == "tokenizer.chat_template") entry = metadata_string(entry.key, std::string(gemma3_chat_template_fixture));
        if (entry.key == "tokenizer.ggml.tokens") entry = metadata_string_array(entry.key, tokens);
        if (entry.key == "tokenizer.ggml.token_type") entry = metadata_i32_array(entry.key, types);
        if (entry.key == "tokenizer.ggml.scores") entry = metadata_f32_array(entry.key, std::vector<float>(tokens.size(), 0.0F));
    }
    return metadata;
}

void test_formats_gemma3_system_and_multiturn_template() {
    const auto metadata = metadata_section(tiny_gemma3_tokenizer_metadata());
    const std::array<ChatMessage, 4> messages {{
        {.role = "system", .content = "  Be helpful.\n"},
        {.role = "user", .content = "\xC2\xA0 First \xE3\x80\x80"},
        {.role = "assistant", .content = "\nAnswer\n"},
        {.role = "user", .content = "\tSecond\t"},
    }};
    const auto formatted = format_chat_prompt_from_gguf_metadata(metadata, messages, true);
    const std::string expected = "<bos><start_of_turn>user\n  Be helpful.\n\n\nFirst<end_of_turn>\n"
        "<start_of_turn>model\nAnswer<end_of_turn>\n"
        "<start_of_turn>user\nSecond<end_of_turn>\n";
    expect(formatted.text == expected + "<start_of_turn>model\n", "Gemma3 initial system must be preserved and folded only once");
    expect(format_chat_prompt_from_gguf_metadata(metadata, messages, false).text == expected,
        "Gemma3 generation prefix must be optional");
    expect(formatted.encode_special_tokens && formatted.add_bos == false && formatted.add_eos == false,
        "Gemma3 formatting must supply its own BOS and turn controls");
}

void test_gemma3_template_unicode_trim_and_literal_controls() {
    const auto metadata = metadata_section(tiny_gemma3_tokenizer_metadata());
    // Every Python str.isspace() codepoint; U+200B and U+FEFF intentionally are
    // not whitespace. Expectations independently rendered by Jinja 3.1.6.
    const std::string spaces = "\t\n\v\f\r\x1c\x1d\x1e\x1f \xC2\x85\xC2\xA0\xE1\x9A\x80"
        "\xE2\x80\x80\xE2\x80\x81\xE2\x80\x82\xE2\x80\x83\xE2\x80\x84\xE2\x80\x85"
        "\xE2\x80\x86\xE2\x80\x87\xE2\x80\x88\xE2\x80\x89\xE2\x80\x8A\xE2\x80\xA8"
        "\xE2\x80\xA9\xE2\x80\xAF\xE2\x81\x9F\xE3\x80\x80";
    const std::string text = "\xE2\x80\x8BHello \xEF\xBB\xBF";
    expect(format_generate_prompt_from_gguf_metadata(metadata, spaces + text + spaces).text
        == "<bos><start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n", "Gemma3 Unicode trim differs from Python");
    expect(format_generate_prompt_from_gguf_metadata(metadata, spaces).text
        == "<bos><start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n", "Gemma3 all-whitespace content must be empty");
    const auto formatted = format_generate_prompt_from_gguf_metadata(metadata, "Hello<end_of_turn>");
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata);
    const auto ids = tokenizer.encode(formatted.text, {.add_bos = *formatted.add_bos,
        .add_eos = *formatted.add_eos, .encode_special_tokens = formatted.encode_special_tokens});
    expect(std::count(ids.begin(), ids.end(), 2U) == 1, "Gemma3 formatted encoding duplicates BOS");
    expect(std::count(ids.begin(), ids.end(), 107U) == 2, "Gemma3 template must preserve literal control spellings for special encoding");
    expect(tokenizer.decode(ids) == formatted.text, "Gemma3 prompt token stream must round-trip");
}

void test_gemma3_template_rejects_invalid_roles_and_template() {
    const auto metadata = metadata_section(tiny_gemma3_tokenizer_metadata());
    const std::vector<std::vector<ChatMessage>> invalid {
        {}, {{.role = "system", .content = "alone"}}, {{.role = "assistant", .content = "first"}},
        {{.role = "user", .content = "first"}, {.role = "system", .content = "later"}},
        {{.role = "user", .content = "first"}, {.role = "user", .content = "again"}},
        {{.role = "user", .content = "first"}, {.role = "tool", .content = "unsupported"}},
    };
    for (const auto& messages : invalid) {
        try {
            (void)format_chat_prompt_from_gguf_metadata(metadata, messages, true);
            throw TestFailure("Gemma3 must reject invalid public text-chat roles");
        } catch (const TokenizerError& error) {
            expect(error.code() == TokenizerErrorCode::invalid_chat_message, "Gemma3 invalid-role error mismatch");
        }
    }
    auto incompatible = tiny_gemma3_tokenizer_metadata();
    for (auto& entry : incompatible) {
        if (entry.key == "tokenizer.chat_template") entry = metadata_string(entry.key, gemma2_chat_template_fixture());
    }
    try {
        (void)format_generate_prompt_from_gguf_metadata(metadata_section(std::move(incompatible)), "Hello");
        throw TestFailure("Gemma3 must reject unsupported template semantics");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::unsupported_chat_template, "Gemma3 template error mismatch");
    }
}

void remove_metadata_key(std::vector<GGUFMetadataEntry>& metadata, std::string_view key) {
    for (auto iterator = metadata.begin(); iterator != metadata.end(); ++iterator) {
        if (iterator->key == key) {
            metadata.erase(iterator);
            return;
        }
    }
}

void test_loads_gemma2_llama_tokenizer_from_metadata() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));

    expect(tokenizer.model() == TokenizerModel::llama, "Gemma2 tokenizer model enum mismatch");
    expect(tokenizer.model_name() == "llama", "Gemma2 tokenizer model name mismatch");
    expect(tokenizer_model_name(tokenizer.model()) == "llama", "Gemma2 tokenizer helper mismatch");
    expect(tokenizer.vocabulary().size() == 110, "Gemma2 tokenizer vocabulary size mismatch");
    expect(tokenizer.vocabulary()[0].text == "<pad>", "Gemma2 PAD token text mismatch");
    expect(tokenizer.vocabulary()[1].text == "<eos>", "Gemma2 EOS token text mismatch");
    expect(tokenizer.vocabulary()[2].text == "<bos>", "Gemma2 BOS token text mismatch");
    expect(tokenizer.vocabulary()[3].text == "<unk>", "Gemma2 unknown token text mismatch");
    expect(tokenizer.vocabulary()[5].score.has_value(), "Gemma2 tokenizer score should load");
    expect(*tokenizer.vocabulary()[5].score == 0.0F, "Gemma2 tokenizer score mismatch");
    expect(tokenizer.vocabulary()[106].text == "<start_of_turn>", "Gemma2 start-of-turn token text mismatch");
    expect(tokenizer.vocabulary()[107].text == "<end_of_turn>", "Gemma2 end-of-turn token text mismatch");
    expect(tokenizer.special_tokens().bos_token_id == 2, "Gemma2 BOS token id mismatch");
    expect(tokenizer.special_tokens().eos_token_id == 1, "Gemma2 EOS token id mismatch");
    expect(tokenizer.special_tokens().unknown_token_id == 3, "Gemma2 unknown token id mismatch");

    const auto options = tokenizer.default_encode_options();
    expect(options.add_bos, "Gemma2 tokenizer default should add BOS");
    expect(!options.add_eos, "Gemma2 tokenizer default should not add EOS");
}

void test_encodes_gemma2_tokenizer_known_ids_with_special_tokens() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));

    const auto default_ids = tokenizer.encode("Hello world!");

    expect(default_ids.size() == 3, "Gemma2 tokenizer default encoded count mismatch");
    expect(default_ids[0] == 2, "Gemma2 tokenizer default BOS id mismatch");
    expect(default_ids[1] == 16, "Gemma2 tokenizer should prefer longest matching token");
    expect(default_ids[2] == 15, "Gemma2 tokenizer punctuation id mismatch");

    const auto eos_ids = tokenizer.encode("Hello world!", TokenizerEncodeOptions {.add_bos = false, .add_eos = true});
    expect(eos_ids.size() == 3, "Gemma2 tokenizer explicit EOS count mismatch");
    expect(eos_ids[0] == 16, "Gemma2 tokenizer explicit first token mismatch");
    expect(eos_ids[1] == 15, "Gemma2 tokenizer explicit punctuation mismatch");
    expect(eos_ids[2] == 1, "Gemma2 tokenizer explicit EOS id mismatch");
}

void test_gemma2_freezes_raw_user_defined_pieces_with_either_special_option() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(gemma2_added_piece_metadata()));
    for (const bool parse_special : {false, true}) {
        const TokenizerEncodeOptions options {.encode_special_tokens = parse_special};
        expect(tokenizer.encode("A    B", options) == std::vector<std::uint32_t> {4, 10, 5},
            "Gemma2 must freeze the longest raw space piece before normalization");
        expect(tokenizer.encode("A         B", options) == std::vector<std::uint32_t> {4, 10, 10, 7},
            "Gemma2 must partition long runs greedily and normalize their ordinary remainder");
        expect(tokenizer.encode("A\n\n\n\n\nB", options) == std::vector<std::uint32_t> {4, 12, 12, 11, 5},
            "Gemma2 ordinary merges must not cross a frozen newline piece");
        expect(tokenizer.encode("A<code>B</code>", options) == std::vector<std::uint32_t> {4, 13, 5, 14},
            "Gemma2 HTML user-defined pieces must remain atomic next to ordinary text");
        expect(tokenizer.encode("A</s>B", options) == std::vector<std::uint32_t> {4, 26, 5},
            "Gemma2 must obey user-defined metadata without inferring EOS from token text");
    }
    const std::array<std::uint32_t, 6> ids {2, 4, 10, 13, 5, 1};
    expect(tokenizer.decode(ids, {.skip_special_tokens = true}) == "A    <code>B",
        "Gemma2 skip-special decoding must preserve user-defined spaces and HTML");
}

void test_gemma2_distinguishes_raw_spaces_from_literal_sentencepiece_markers() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(gemma2_added_piece_metadata()));
    expect(tokenizer.encode("A▁▁▁▁B", {}) == std::vector<std::uint32_t> {21},
        "literal SentencePiece markers must remain ordinary merge input, not raw-space aliases");
    expect(tokenizer.encode("A    B", {}) == std::vector<std::uint32_t> {4, 10, 5},
        "raw spaces must use the declared user-defined piece instead of the marker merge");
    expect(tokenizer.encode("    A    ", {}) == std::vector<std::uint32_t> {10, 4, 10},
        "Gemma2 must preserve leading and trailing raw-space pieces without a dummy prefix");
}

void test_gemma2_control_and_unknown_literals_follow_explicit_special_option() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(gemma2_added_piece_metadata()));
    constexpr TokenizerEncodeOptions parse {.encode_special_tokens = true};
    expect(tokenizer.encode("A<unk>B", parse) == std::vector<std::uint32_t> {4, 3, 5},
        "unknown token text must be recognized when special parsing is requested");
    expect(tokenizer.encode("A<start_of_turn>B", parse) == std::vector<std::uint32_t> {4, 15, 5},
        "control token text must be recognized when special parsing is requested");
    const auto ordinary = tokenizer.encode("A<unk>B", {});
    const std::vector<std::uint32_t> expected {
        4, gemma2_added_byte_base + '<', gemma2_added_byte_base + 'u',
        gemma2_added_byte_base + 'n', gemma2_added_byte_base + 'k', gemma2_added_byte_base + '>', 5,
    };
    expect(ordinary == expected, "disabled special parsing must encode unknown-token text as literal bytes");
    expect(tokenizer.encode("", {.add_bos = true, .add_eos = true}) == std::vector<std::uint32_t> {2, 1},
        "Gemma2 explicit BOS and EOS must remain defined for empty input");
    expect(tokenizer.encode("A    B") == std::vector<std::uint32_t> {2, 4, 10, 5},
        "Gemma2 default BOS must coexist with raw user-defined pieces");
}

void test_gemma2_bounded_encoder_reuses_workspace_after_capacity_errors() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(gemma2_added_piece_metadata()));
    TokenizerEncodeWorkspace workspace(16);
    const auto owned_bytes = workspace.owned_bytes();
    std::array<std::uint32_t, 32> output {};
    const auto verify = [&](std::string_view text, const std::vector<std::uint32_t>& expected) {
        const auto written = tokenizer.encode_into(text, output, workspace);
        expect(std::vector<std::uint32_t>(output.begin(), output.begin() + written) == expected,
            "bounded Gemma2 encoding mismatch after workspace reuse");
        expect(workspace.owned_bytes() == owned_bytes, "bounded encoding must not grow prepared storage");
    };
    verify("A    B", {4, 10, 5});
    verify("A▁▁▁▁B", {21});
    try {
        (void)tokenizer.encode_into("A    B", std::span(output).first(2), workspace);
        throw TestFailure("bounded Gemma2 encoder should reject insufficient output capacity");
    } catch (const std::length_error&) {}
    verify("A\n\nB", {4, 12, 5});
    try {
        (void)tokenizer.encode_into(std::string(17, 'A'), output, workspace);
        throw TestFailure("bounded Gemma2 encoder should reject its input byte bound");
    } catch (const std::length_error&) {}
    verify("A<code>B", {4, 13, 5});
    expect(tokenizer.encode_into("", output, workspace) == 0, "empty bounded input must clear prior workspace state");
}

void test_gemma2_preserves_original_bytes_for_invalid_utf8_fallback() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(gemma2_added_piece_metadata()));
    for (const std::string& text : {std::string("A\xFF" "B"), std::string("A\xC0\xAF" "B"), std::string("A\xE2\x82")}) {
        std::vector<std::uint32_t> expected {4};
        for (std::size_t index = 1; index < text.size(); ++index) {
            const auto byte = static_cast<unsigned char>(text[index]);
            expected.push_back(byte == 'B' ? 5 : gemma2_added_byte_base + byte);
        }
        const auto ids = tokenizer.encode(text, {});
        expect(ids == expected, "Gemma2 fallback must preserve malformed input bytes without inventing replacement characters");
        expect(tokenizer.decode(ids) == text, "Gemma2 raw byte fallback must round-trip malformed input");
    }
}

void test_gemma2_template_trims_python_unicode_whitespace() {
    const auto metadata = metadata_section(tiny_gemma2_tokenizer_metadata());
    const auto formatted = format_generate_prompt_from_gguf_metadata(metadata, "\u00A0\u3000Hello\u0085\u2003");
    expect(formatted.text == "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n",
        "Gemma2 template trim must match Python/Jinja for non-ASCII whitespace");
    const auto empty = format_generate_prompt_from_gguf_metadata(metadata, "\u00A0\u3000\n\u0085\u2003");
    expect(empty.text == "<bos><start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n",
        "Gemma2 template must trim a Unicode-whitespace-only message to empty");
}

void test_gemma2_rejects_unsupported_normalization_metadata() {
    const std::vector<GGUFMetadataEntry> incompatible {
        metadata_bool("tokenizer.ggml.add_space_prefix", true),
        metadata_string("tokenizer.ggml.pre", "llama3"),
        metadata_string("tokenizer.ggml.normalizer", "nmt_nfkc"),
        metadata_bool("tokenizer.ggml.remove_extra_whitespaces", true),
        metadata_bool("tokenizer.ggml.escape_whitespaces", false),
    };
    for (const auto& replacement : incompatible) {
        auto metadata = tiny_gemma2_tokenizer_metadata();
        const auto prior = std::find_if(metadata.begin(), metadata.end(), [&](const auto& entry) {
            return entry.key == replacement.key;
        });
        if (prior == metadata.end()) metadata.push_back(replacement);
        else *prior = replacement;
        try {
            (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
            throw TestFailure("Gemma2 must reject normalization metadata the engine does not implement");
        } catch (const TokenizerError& error) {
            expect(error.code() == TokenizerErrorCode::invalid_vocabulary,
                "Gemma2 unsupported normalization error code mismatch");
            expect(error.detail() == "unsupported Gemma2 normalization",
                "Gemma2 unsupported normalization error detail mismatch");
        }
    }
}

void test_rejects_nonfinite_tokenizer_scores() {
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        auto metadata = tiny_gemma2_tokenizer_metadata();
        for (auto& entry : metadata) {
            if (entry.key == "tokenizer.ggml.scores") {
                auto scores = std::vector<float>(110, 0.0F);
                scores[5] = invalid;
                entry = metadata_f32_array(entry.key, std::move(scores));
                break;
            }
        }
        try {
            (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
            throw TestFailure("tokenizer must reject scores without a well-defined merge ordering");
        } catch (const TokenizerError& error) {
            expect(error.code() == TokenizerErrorCode::invalid_vocabulary,
                "nonfinite tokenizer score error code mismatch");
            expect(error.detail() == "nonfinite tokenizer.ggml.scores",
                "nonfinite tokenizer score error detail mismatch");
        }
    }
}

void test_encodes_tokenizer_special_tokens_when_enabled() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));

    const auto ids = tokenizer.encode(
        "<start_of_turn>Hello<end_of_turn>",
        TokenizerEncodeOptions {.add_bos = false, .add_eos = false, .encode_special_tokens = true}
    );
    const auto ordinary_ids = tokenizer.encode(
        "<start_of_turn>",
        TokenizerEncodeOptions {.add_bos = false, .add_eos = false, .encode_special_tokens = false}
    );

    expect(ids.size() == 3, "special-token encode count mismatch");
    expect(ids[0] == 106, "special-token encode start-of-turn mismatch");
    expect(ids[1] == 5, "special-token encode text mismatch");
    expect(ids[2] == 107, "special-token encode end-of-turn mismatch");
    expect(ordinary_ids.size() > 1, "ordinary encode should not collapse control tokens");
    expect(ordinary_ids[0] == 3, "ordinary encode should route literal control text through unknown token");
}

void test_formats_gemma2_generate_prompt_from_chat_template() {
    const auto formatted = format_generate_prompt_from_gguf_metadata(
        metadata_section(tiny_gemma2_tokenizer_metadata()),
        "  Hello\n"
    );

    expect(formatted.encode_special_tokens, "Gemma2 prompt should encode special tokens");
    expect(formatted.add_bos.has_value() && !*formatted.add_bos, "Gemma2 prompt should suppress automatic BOS");
    expect(formatted.add_eos.has_value() && !*formatted.add_eos, "Gemma2 prompt should suppress automatic EOS");
    expect(
        formatted.text == "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n",
        "Gemma2 generate prompt mismatch"
    );
}

void test_formats_gemma2_multi_turn_chat_prompt() {
    const std::array<ChatMessage, 3> messages {{
        ChatMessage {.role = "user", .content = " First "},
        ChatMessage {.role = "assistant", .content = "\nAnswer\n"},
        ChatMessage {.role = "user", .content = "\tSecond\t"},
    }};

    const auto formatted = format_chat_prompt_from_gguf_metadata(
        metadata_section(tiny_gemma2_tokenizer_metadata()),
        std::span<const ChatMessage>(messages),
        true
    );

    expect(
        formatted.text == "<bos><start_of_turn>user\nFirst<end_of_turn>\n"
            "<start_of_turn>model\nAnswer<end_of_turn>\n"
            "<start_of_turn>user\nSecond<end_of_turn>\n"
            "<start_of_turn>model\n",
        "Gemma2 multi-turn chat prompt mismatch"
    );
}

void test_encodes_gemma2_prompt_special_tokens_without_double_bos() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));
    const auto formatted = format_generate_prompt_from_gguf_metadata(
        metadata_section(tiny_gemma2_tokenizer_metadata()),
        "Hello"
    );
    const auto defaults = tokenizer.default_encode_options();
    const auto ids = tokenizer.encode(
        formatted.text,
        TokenizerEncodeOptions {
            .add_bos = formatted.add_bos.value_or(defaults.add_bos),
            .add_eos = formatted.add_eos.value_or(defaults.add_eos),
            .encode_special_tokens = formatted.encode_special_tokens,
        }
    );

    const std::array<std::uint32_t, 8> expected {2, 106, 9, 5, 107, 11, 106, 10};
    expect(ids.size() == expected.size(), "Gemma2 encoded prompt count mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect(ids[index] == expected[index], "Gemma2 encoded prompt token mismatch");
    }
}

void test_rejects_gemma2_system_chat_message() {
    const std::array<ChatMessage, 1> messages {{
        ChatMessage {.role = "system", .content = "Nope"},
    }};

    try {
        (void)format_chat_prompt_from_gguf_metadata(
            metadata_section(tiny_gemma2_tokenizer_metadata()),
            std::span<const ChatMessage>(messages),
            true
        );
        throw TestFailure("Gemma2 formatter should reject system role");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_chat_message, "Gemma2 system role error mismatch");
        expect(error.detail() == "system", "Gemma2 system role detail mismatch");
    }
}

void test_rejects_gemma2_chat_role_alternation() {
    const std::array<ChatMessage, 2> messages {{
        ChatMessage {.role = "user", .content = "First"},
        ChatMessage {.role = "user", .content = "Second"},
    }};

    try {
        (void)format_chat_prompt_from_gguf_metadata(
            metadata_section(tiny_gemma2_tokenizer_metadata()),
            std::span<const ChatMessage>(messages),
            true
        );
        throw TestFailure("Gemma2 formatter should reject invalid role alternation");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_chat_message, "Gemma2 alternation error mismatch");
        expect(error.detail() == "role alternation", "Gemma2 alternation detail mismatch");
    }
}

void test_keeps_raw_generate_prompt_without_chat_template() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    remove_metadata_key(metadata, "tokenizer.chat_template");

    const auto formatted = format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");

    expect(!formatted.encode_special_tokens, "raw prompt should not enable special-token encoding");
    expect(formatted.text == "Hello", "raw prompt formatting mismatch");
}

void test_rejects_unsupported_chat_template() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.chat_template") {
            entry = metadata_string("tokenizer.chat_template", "{{ messages }}");
            break;
        }
    }

    try {
        (void)format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");
        throw TestFailure("prompt formatter should reject unsupported chat template");
    } catch (const TokenizerError& error) {
        expect(
            error.code() == TokenizerErrorCode::unsupported_chat_template,
            "unsupported chat-template error mismatch"
        );
        expect(error.detail() == "tokenizer.chat_template", "unsupported chat-template detail mismatch");
    }
}

void test_rejects_chat_template_invalid_type() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.chat_template") {
            entry = metadata_bool("tokenizer.chat_template", true);
            break;
        }
    }

    try {
        (void)format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");
        throw TestFailure("prompt formatter should reject invalid chat-template metadata type");
    } catch (const TokenizerError& error) {
        expect(
            error.code() == TokenizerErrorCode::invalid_metadata_type,
            "invalid chat-template metadata type error mismatch"
        );
        expect(error.detail() == "tokenizer.chat_template", "invalid chat-template metadata detail mismatch");
    }
}

void test_decodes_gemma2_tokenizer_known_ids_and_skips_special_tokens() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));
    const std::array<std::uint32_t, 5> ids {2, 5, 14, 15, 1};

    const auto decoded = tokenizer.decode(std::span<const std::uint32_t>(ids));
    const auto skipped = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );

    expect(decoded == "<bos>Hello world!<eos>", "Gemma2 tokenizer decoded text mismatch");
    expect(skipped == "Hello world!", "Gemma2 tokenizer skip-special decoded text mismatch");
}

void test_decodes_gemma2_sentencepiece_space_markers() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));
    const std::array<std::uint32_t, 5> ids {2, 5, 12, 13, 1};

    const auto decoded = tokenizer.decode(std::span<const std::uint32_t>(ids));
    const auto skipped = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );

    expect(decoded == "<bos>Hello seems question<eos>", "Gemma2 decoded space-marker text mismatch");
    expect(skipped == "Hello seems question", "Gemma2 skip-special space-marker text mismatch");
}

void test_sentencepiece_byte_fallback_round_trips_unknown_text() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));
    const auto ids = tokenizer.encode(
        "\xC3\xA9\xF0\x9F\x98\x80",
        TokenizerEncodeOptions {.add_bos = false}
    );
    const std::array<std::uint32_t, 6> expected {42, 43, 44, 45, 46, 47};

    expect(
        std::equal(ids.begin(), ids.end(), expected.begin(), expected.end()),
        "SentencePiece byte fallback IDs mismatch"
    );
    expect(
        tokenizer.decode(std::span<const std::uint32_t>(ids)) == "\xC3\xA9\xF0\x9F\x98\x80",
        "SentencePiece byte fallback should round-trip UTF-8"
    );
}

void test_incremental_decode_matches_batch_across_token_boundaries() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));
    const std::array<std::uint32_t, 11> ids {2, 5, 3, 12, 42, 43, 44, 45, 46, 47, 1};
    constexpr TokenizerDecodeOptions options {.skip_special_tokens = true};

    std::string incremental;
    for (const auto token_id : ids) {
        tokenizer.append_decoded_token(incremental, token_id, options);
    }
    constexpr std::string_view expected = "Hello seems\xC3\xA9\xF0\x9F\x98\x80";
    expect(incremental == expected
        && tokenizer.decode(std::span<const std::uint32_t>(ids), options) == expected,
        "incremental and batch decoding must skip controls and unknown tokens");
    std::array<char, 64> output {};
    const auto written = tokenizer.decode_into(ids, output, options);
    expect(std::string_view(output.data(), written) == expected,
        "bounded decoding must match incremental and batch skip-special semantics");

    const std::array<std::uint32_t, 1> unknown {3};
    std::string literal_unknown;
    tokenizer.append_decoded_token(literal_unknown, unknown.front());
    const auto unknown_written = tokenizer.decode_into(unknown, output);
    expect(literal_unknown == "<unk>" && tokenizer.decode(unknown) == "<unk>"
        && std::string_view(output.data(), unknown_written) == "<unk>",
        "all decode APIs must preserve unknown text when special skipping is disabled");

    try {
        tokenizer.append_decoded_token(incremental, 110U, options);
        throw TestFailure("incremental decode should reject an invalid token ID");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::token_id_out_of_range,
            "incremental invalid-ID error mismatch");
    }
}

void test_rejects_malformed_sentencepiece_byte_token() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.ggml.tokens") {
            auto& tokens = std::get<local_gguf_engine::model_format::GGUFMetadataArray>(entry.value.data);
            auto& values = std::get<std::vector<std::string>>(tokens.values);
            values[42] = "<0xZZ>";
            break;
        }
    }

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject malformed byte token text");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_vocabulary, "malformed byte token error mismatch");
        expect(error.detail() == "invalid byte token at id 42", "malformed byte token detail mismatch");
    }
}

void test_rejects_tokenizer_decode_id_out_of_range() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_gemma2_tokenizer_metadata()));
    const std::array<std::uint32_t, 1> ids {110};

    try {
        (void)tokenizer.decode(std::span<const std::uint32_t>(ids));
        throw TestFailure("tokenizer decode should reject invalid token id");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::token_id_out_of_range, "tokenizer invalid id error mismatch");
        expect(error.detail() == "110", "tokenizer invalid id detail mismatch");
    }
}

void test_rejects_tokenizer_malformed_token_metadata() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.ggml.token_type") {
            entry = metadata_i32_array("tokenizer.ggml.token_type", {3, 3});
            break;
        }
    }

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject token type length mismatch");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_vocabulary, "tokenizer malformed metadata error mismatch");
        expect(error.detail() == "tokenizer.ggml.token_type", "tokenizer malformed metadata detail mismatch");
    }
}

void test_rejects_tokenizer_unsupported_model() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.ggml.model") {
            entry = metadata_string("tokenizer.ggml.model", "unsupported-tokenizer");
            break;
        }
    }

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject unsupported model");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::unsupported_tokenizer_model, "tokenizer model error mismatch");
        expect(error.detail() == "unsupported-tokenizer", "tokenizer model detail mismatch");
    }
}

void test_rejects_tokenizer_without_matching_token() {
    auto metadata = tiny_gemma2_tokenizer_metadata();
    remove_metadata_key(metadata, "tokenizer.ggml.unknown_token_id");
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));

    try {
        (void)tokenizer.encode("Hello?");
        throw TestFailure("tokenizer should reject unmatched text without unknown token");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::no_matching_token, "tokenizer no-match error mismatch");
        expect(error.detail() == "byte offset 5", "tokenizer no-match detail mismatch");
    }
}

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> tokenizer_test_cases() {
    static constexpr TestCase tests[] = {
        {"formats Gemma3 system and multi-turn template", test_formats_gemma3_system_and_multiturn_template},
        {"Gemma3 template Unicode trim and literal controls", test_gemma3_template_unicode_trim_and_literal_controls},
        {"Gemma3 template rejects invalid roles and template", test_gemma3_template_rejects_invalid_roles_and_template},
        {"loads Gemma2 Llama tokenizer from metadata", test_loads_gemma2_llama_tokenizer_from_metadata},
        {"Gemma2 freezes raw user-defined pieces with either special option", test_gemma2_freezes_raw_user_defined_pieces_with_either_special_option},
        {"Gemma2 distinguishes raw spaces from literal SentencePiece markers", test_gemma2_distinguishes_raw_spaces_from_literal_sentencepiece_markers},
        {"Gemma2 control and unknown literals follow explicit special option", test_gemma2_control_and_unknown_literals_follow_explicit_special_option},
        {"Gemma2 bounded encoder reuses workspace after capacity errors", test_gemma2_bounded_encoder_reuses_workspace_after_capacity_errors},
        {"Gemma2 preserves original bytes for invalid UTF-8 fallback", test_gemma2_preserves_original_bytes_for_invalid_utf8_fallback},
        {"Gemma2 template trims Python Unicode whitespace", test_gemma2_template_trims_python_unicode_whitespace},
        {"Gemma2 rejects unsupported normalization metadata", test_gemma2_rejects_unsupported_normalization_metadata},
        {"rejects nonfinite tokenizer scores", test_rejects_nonfinite_tokenizer_scores},

        {
            "encodes Gemma2 tokenizer known IDs with special tokens",
            test_encodes_gemma2_tokenizer_known_ids_with_special_tokens,
        },
        {"encodes tokenizer special tokens when enabled", test_encodes_tokenizer_special_tokens_when_enabled},
        {
            "formats Gemma2 generate prompt from chat template",
            test_formats_gemma2_generate_prompt_from_chat_template,
        },
        {"formats Gemma2 multi-turn chat prompt", test_formats_gemma2_multi_turn_chat_prompt},
        {
            "encodes Gemma2 prompt special tokens without double BOS",
            test_encodes_gemma2_prompt_special_tokens_without_double_bos,
        },
        {"rejects Gemma2 system chat message", test_rejects_gemma2_system_chat_message},
        {"rejects Gemma2 chat role alternation", test_rejects_gemma2_chat_role_alternation},
        {"keeps raw generate prompt without chat template", test_keeps_raw_generate_prompt_without_chat_template},
        {"rejects unsupported chat template", test_rejects_unsupported_chat_template},
        {"rejects chat template invalid type", test_rejects_chat_template_invalid_type},
        {
            "decodes Gemma2 tokenizer known IDs and skips special tokens",
            test_decodes_gemma2_tokenizer_known_ids_and_skips_special_tokens,
        },
        {
            "decodes Gemma2 SentencePiece space markers",
            test_decodes_gemma2_sentencepiece_space_markers,
        },
        {
            "SentencePiece byte fallback round-trips unknown text",
            test_sentencepiece_byte_fallback_round_trips_unknown_text,
        },
        {
            "incremental decode matches batch across token boundaries",
            test_incremental_decode_matches_batch_across_token_boundaries,
        },
        {
            "rejects malformed SentencePiece byte token",
            test_rejects_malformed_sentencepiece_byte_token,
        },
        {"rejects tokenizer decode ID out of range", test_rejects_tokenizer_decode_id_out_of_range},
        {
            "rejects tokenizer malformed token metadata",
            test_rejects_tokenizer_malformed_token_metadata,
        },
        {"rejects tokenizer unsupported model", test_rejects_tokenizer_unsupported_model},
        {"rejects tokenizer without matching token", test_rejects_tokenizer_without_matching_token},
    };

    return tests;
}

}  // namespace local_gguf_engine::tests
