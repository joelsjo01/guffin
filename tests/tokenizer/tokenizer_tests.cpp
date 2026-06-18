#include "tokenizer_tests.hpp"

#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <array>
#include <cstdint>
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
using local_gguf_engine::tests::smollm2_chat_template_fixture;
using local_gguf_engine::tests::tiny_llama_metadata;
using local_gguf_engine::tokenizer::format_generate_prompt_from_gguf_metadata;
using local_gguf_engine::tokenizer::load_tokenizer_from_gguf_metadata;
using local_gguf_engine::tokenizer::TokenizerDecodeOptions;
using local_gguf_engine::tokenizer::TokenizerEncodeOptions;
using local_gguf_engine::tokenizer::TokenizerError;
using local_gguf_engine::tokenizer::TokenizerErrorCode;
using local_gguf_engine::tokenizer::TokenizerModel;
using local_gguf_engine::tokenizer::tokenizer_model_name;

std::vector<GGUFMetadataEntry> tiny_llama_tokenizer_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 7);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {"<s>", "</s>", "Hello", " world", "!", "Hello world", "<unk>"}
    ));
    metadata.push_back(metadata_f32_array(
        "tokenizer.ggml.scores",
        {0.0F, 0.0F, -1.0F, -1.1F, -1.2F, -0.2F, -10.0F}
    ));
    metadata.push_back(metadata_i32_array(
        "tokenizer.ggml.token_type",
        {3, 3, 1, 1, 1, 1, 2}
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 0));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 6));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", true));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    return metadata;
}

std::vector<GGUFMetadataEntry> tiny_gpt2_byte_decoder_tokenizer_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 8);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {"<s>", "</s>", "Hello", "\xC4\xA0world", "!", "\xC4\x8A", "\xC3\x83\xC2\xA5", "<unk>"}
    ));
    metadata.push_back(metadata_f32_array(
        "tokenizer.ggml.scores",
        {0.0F, 0.0F, -1.0F, -1.1F, -1.2F, -1.3F, -1.4F, -10.0F}
    ));
    metadata.push_back(metadata_i32_array(
        "tokenizer.ggml.token_type",
        {3, 3, 1, 1, 1, 1, 1, 2}
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 0));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 7));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", true));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    return metadata;
}

std::vector<GGUFMetadataEntry> tiny_gpt2_bpe_tokenizer_metadata(bool add_space_prefix) {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 25);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {
            "<s>", "</s>", "H", "e", "l", "o", "\xC4\xA0", "w", "r", "d", "!",
            "\xC3\x83", "\xC2\xA5", "He", "Hel", "Hell", "Hello",
            "\xC4\xA0w", "\xC4\xA0wo", "\xC4\xA0wor", "\xC4\xA0worl", "\xC4\xA0world",
            "\xC3\x83\xC2\xA5", "\xC4\xA0Hello", "<unk>",
        }
    ));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.merges",
        {
            "H e", "He l", "Hel l", "Hell o",
            "\xC4\xA0 w", "\xC4\xA0w o", "\xC4\xA0wo r", "\xC4\xA0wor l", "\xC4\xA0worl d",
            "\xC3\x83 \xC2\xA5", "\xC4\xA0 Hello",
        }
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 0));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 24));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", true));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_space_prefix", add_space_prefix));
    return metadata;
}

std::string smollm2_formatted_prompt_fixture(std::string_view prompt) {
    std::string formatted;
    formatted += "<|im_start|>system\n";
    formatted += "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
    formatted += "<|im_end|>\n<|im_start|>user\n";
    formatted += prompt;
    formatted += "<|im_end|>\n<|im_start|>assistant\n";
    return formatted;
}

std::vector<GGUFMetadataEntry> tiny_gpt2_special_tokenizer_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 4);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {"<unk>", "<|im_start|>", "<|im_end|>", "Hello"}
    ));
    metadata.push_back(metadata_i32_array(
        "tokenizer.ggml.token_type",
        {2, 3, 3, 1}
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 2));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 0));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", false));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    return metadata;
}

void test_loads_llama_tokenizer_from_metadata() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));

    expect(tokenizer.model() == TokenizerModel::gpt2, "tokenizer model enum mismatch");
    expect(tokenizer.model_name() == "gpt2", "tokenizer model name mismatch");
    expect(tokenizer_model_name(tokenizer.model()) == "gpt2", "tokenizer model helper mismatch");
    expect(tokenizer.vocabulary().size() == 7, "tokenizer vocabulary size mismatch");
    expect(tokenizer.vocabulary()[0].text == "<s>", "tokenizer BOS text mismatch");
    expect(tokenizer.vocabulary()[2].text == "Hello", "tokenizer normal token text mismatch");
    expect(tokenizer.vocabulary()[5].score.has_value(), "tokenizer score should load");
    expect(*tokenizer.vocabulary()[5].score == -0.2F, "tokenizer score mismatch");
    expect(tokenizer.vocabulary()[0].type.has_value(), "tokenizer token type should load");
    expect(*tokenizer.vocabulary()[0].type == 3, "tokenizer token type mismatch");
    expect(tokenizer.special_tokens().bos_token_id == 0, "tokenizer BOS id mismatch");
    expect(tokenizer.special_tokens().eos_token_id == 1, "tokenizer EOS id mismatch");
    expect(tokenizer.special_tokens().unknown_token_id == 6, "tokenizer unknown id mismatch");

    const auto options = tokenizer.default_encode_options();
    expect(options.add_bos, "tokenizer default should add BOS from metadata");
    expect(!options.add_eos, "tokenizer default should not add EOS from metadata");
}

void test_encodes_llama_tokenizer_known_ids_with_special_tokens() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));

    const auto default_ids = tokenizer.encode("Hello world!");

    expect(default_ids.size() == 3, "tokenizer default encoded count mismatch");
    expect(default_ids[0] == 0, "tokenizer default BOS id mismatch");
    expect(default_ids[1] == 5, "tokenizer should prefer longest matching token");
    expect(default_ids[2] == 4, "tokenizer punctuation id mismatch");

    const auto eos_ids = tokenizer.encode("Hello world!", TokenizerEncodeOptions {.add_bos = false, .add_eos = true});
    expect(eos_ids.size() == 3, "tokenizer explicit EOS count mismatch");
    expect(eos_ids[0] == 5, "tokenizer explicit first token mismatch");
    expect(eos_ids[1] == 4, "tokenizer explicit punctuation mismatch");
    expect(eos_ids[2] == 1, "tokenizer explicit EOS id mismatch");
}

void test_encodes_gpt2_bpe_prompt_pieces() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_bpe_tokenizer_metadata(false))
    );

    const auto prompt = std::string("Hello world!") + "\xC3\xA5";
    const auto ids = tokenizer.encode(prompt, TokenizerEncodeOptions {.add_bos = false, .add_eos = true});

    expect(ids.size() == 5, "GPT-2 BPE encoded count mismatch");
    expect(ids[0] == 16, "GPT-2 BPE should merge Hello");
    expect(ids[1] == 21, "GPT-2 BPE should merge space-prefixed world");
    expect(ids[2] == 10, "GPT-2 BPE punctuation token mismatch");
    expect(ids[3] == 22, "GPT-2 BPE UTF-8 byte merge mismatch");
    expect(ids[4] == 1, "GPT-2 BPE EOS id mismatch");

    const auto decoded = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );
    expect(decoded == prompt, "GPT-2 BPE encode/decode round-trip mismatch");
}

void test_encodes_gpt2_bpe_with_space_prefix() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_bpe_tokenizer_metadata(true))
    );

    const auto ids = tokenizer.encode("Hello", TokenizerEncodeOptions {.add_bos = false, .add_eos = false});

    expect(ids.size() == 1, "GPT-2 BPE space-prefix encoded count mismatch");
    expect(ids[0] == 23, "GPT-2 BPE should use space-prefixed token");
    expect(
        tokenizer.decode(std::span<const std::uint32_t>(ids)) == " Hello",
        "GPT-2 BPE space-prefix decode mismatch"
    );
}

void test_encodes_tokenizer_special_tokens_when_enabled() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_special_tokenizer_metadata())
    );

    const auto ids = tokenizer.encode(
        "<|im_start|>Hello<|im_end|>",
        TokenizerEncodeOptions {.add_bos = false, .add_eos = false, .encode_special_tokens = true}
    );
    const auto ordinary_ids = tokenizer.encode(
        "<|im_start|>",
        TokenizerEncodeOptions {.add_bos = false, .add_eos = false, .encode_special_tokens = false}
    );

    expect(ids.size() == 3, "special-token encode count mismatch");
    expect(ids[0] == 1, "special-token encode im_start mismatch");
    expect(ids[1] == 3, "special-token encode text mismatch");
    expect(ids[2] == 2, "special-token encode im_end mismatch");
    expect(ordinary_ids.size() > 1, "ordinary encode should not collapse control tokens");
    expect(ordinary_ids[0] == 0, "ordinary encode should route literal control text through unknown token");
}

void test_formats_smollm2_generate_prompt_from_chat_template() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.push_back(metadata_string("tokenizer.chat_template", smollm2_chat_template_fixture()));

    const auto formatted = format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");

    expect(formatted.encode_special_tokens, "chat-formatted prompt should enable special-token encoding");
    expect(
        formatted.text == smollm2_formatted_prompt_fixture("Hello"),
        "SmolLM2 chat-formatted prompt mismatch"
    );
}

void test_keeps_raw_generate_prompt_without_chat_template() {
    const auto formatted = format_generate_prompt_from_gguf_metadata(
        metadata_section(tiny_llama_tokenizer_metadata()),
        "Hello"
    );

    expect(!formatted.encode_special_tokens, "raw prompt should not enable special-token encoding");
    expect(formatted.text == "Hello", "raw prompt formatting mismatch");
}

void test_rejects_unsupported_chat_template() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.push_back(metadata_string("tokenizer.chat_template", "{{ messages }}"));

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
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.push_back(metadata_bool("tokenizer.chat_template", true));

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

void test_decodes_llama_tokenizer_known_ids_and_skips_special_tokens() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));
    const std::array<std::uint32_t, 5> ids {0, 2, 3, 4, 1};

    const auto decoded = tokenizer.decode(std::span<const std::uint32_t>(ids));
    const auto skipped = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );

    expect(decoded == "<s>Hello world!</s>", "tokenizer decoded text mismatch");
    expect(skipped == "Hello world!", "tokenizer skip-special decoded text mismatch");
}

void test_decodes_gpt2_byte_level_markers() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_byte_decoder_tokenizer_metadata())
    );
    const std::array<std::uint32_t, 7> ids {0, 2, 3, 4, 5, 6, 1};

    const auto decoded = tokenizer.decode(std::span<const std::uint32_t>(ids));
    const auto skipped = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );

    const auto expected = std::string("<s>Hello world!\n") + "\xC3\xA5" + "</s>";
    const auto expected_skipped = std::string("Hello world!\n") + "\xC3\xA5";

    expect(decoded == expected, "GPT-2 byte-level decoded text mismatch");
    expect(skipped == expected_skipped, "GPT-2 byte-level skip-special decoded text mismatch");
}

void test_rejects_tokenizer_decode_id_out_of_range() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));
    const std::array<std::uint32_t, 1> ids {7};

    try {
        (void)tokenizer.decode(std::span<const std::uint32_t>(ids));
        throw TestFailure("tokenizer decode should reject invalid token id");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::token_id_out_of_range, "tokenizer invalid id error mismatch");
        expect(error.detail() == "7", "tokenizer invalid id detail mismatch");
    }
}

void test_rejects_tokenizer_malformed_token_metadata() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata[17] = metadata_i32_array("tokenizer.ggml.token_type", {3, 3});

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject token type length mismatch");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_vocabulary, "tokenizer malformed metadata error mismatch");
        expect(error.detail() == "tokenizer.ggml.token_type", "tokenizer malformed metadata detail mismatch");
    }
}

void test_rejects_tokenizer_malformed_bpe_merges() {
    auto metadata = tiny_gpt2_bpe_tokenizer_metadata(false);
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.ggml.merges") {
            entry = metadata_string_array("tokenizer.ggml.merges", {"bad-merge"});
            break;
        }
    }

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject malformed BPE merge metadata");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_vocabulary, "tokenizer malformed BPE error mismatch");
        expect(error.detail() == "tokenizer.ggml.merges", "tokenizer malformed BPE detail mismatch");
    }
}

void test_rejects_tokenizer_without_matching_token() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.erase(metadata.begin() + 20);
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
        {"loads Llama tokenizer from metadata", test_loads_llama_tokenizer_from_metadata},
        {
            "encodes Llama tokenizer known IDs with special tokens",
            test_encodes_llama_tokenizer_known_ids_with_special_tokens,
        },
        {"encodes GPT-2 BPE prompt pieces", test_encodes_gpt2_bpe_prompt_pieces},
        {"encodes GPT-2 BPE with space prefix", test_encodes_gpt2_bpe_with_space_prefix},
        {"encodes tokenizer special tokens when enabled", test_encodes_tokenizer_special_tokens_when_enabled},
        {
            "formats SmolLM2 generate prompt from chat template",
            test_formats_smollm2_generate_prompt_from_chat_template,
        },
        {"keeps raw generate prompt without chat template", test_keeps_raw_generate_prompt_without_chat_template},
        {"rejects unsupported chat template", test_rejects_unsupported_chat_template},
        {"rejects chat template invalid type", test_rejects_chat_template_invalid_type},
        {
            "decodes Llama tokenizer known IDs and skips special tokens",
            test_decodes_llama_tokenizer_known_ids_and_skips_special_tokens,
        },
        {"decodes GPT-2 byte-level markers", test_decodes_gpt2_byte_level_markers},
        {"rejects tokenizer decode ID out of range", test_rejects_tokenizer_decode_id_out_of_range},
        {
            "rejects tokenizer malformed token metadata",
            test_rejects_tokenizer_malformed_token_metadata,
        },
        {"rejects tokenizer malformed BPE merges", test_rejects_tokenizer_malformed_bpe_merges},
        {"rejects tokenizer without matching token", test_rejects_tokenizer_without_matching_token},
    };

    return tests;
}

}  // namespace local_gguf_engine::tests
