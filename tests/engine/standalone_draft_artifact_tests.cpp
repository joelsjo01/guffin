#include "test_support.hpp"

#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"
#include "local_gguf_engine/engine/prepared_gemma_plan.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <unistd.h>

namespace local_gguf_engine::tests {
std::vector<std::uint8_t> standalone_chat_target_fixture_bytes(std::uint32_t context);
namespace {
using tokenizer::load_tokenizer_from_gguf_metadata;

model_format::GGUFMetadataSection fixture_tokenizer(bool llama) {
    std::vector<std::string> pieces {"<unk>", "<s>", "</s>"};
    std::vector<std::int32_t> types {2, 3, 3};
    std::vector<float> scores {0, 0, 0};
    for (unsigned byte = 0; byte < 256; ++byte) {
        std::array<char, 7> name {};
        std::snprintf(name.data(), name.size(), "<0x%02X>", byte);
        pieces.emplace_back(name.data()); types.push_back(6); scores.push_back(0);
    }
    for (const auto* piece : {"▁", "a", "b", "ab", "▁a", "▁ab", "x"}) {
        pieces.emplace_back(piece); types.push_back(1); scores.push_back(-10);
    }
    scores[262] = -2; scores[263] = -1; scores[264] = 0;
    return metadata_section({
        metadata_string("general.architecture", llama ? "llama" : "gemma2"),
        metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string("tokenizer.ggml.normalizer", "identity-no-collapse-byte-fallback"),
        metadata_bool("tokenizer.ggml.add_space_prefix", true),
        metadata_string_array("tokenizer.ggml.tokens", std::move(pieces)),
        metadata_f32_array("tokenizer.ggml.scores", std::move(scores)),
        metadata_i32_array("tokenizer.ggml.token_type", std::move(types)),
        metadata_u32("tokenizer.ggml.unknown_token_id", 0),
        metadata_u32("tokenizer.ggml.bos_token_id", 1),
        metadata_u32("tokenizer.ggml.eos_token_id", 2),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
    });
}

model_format::GGUFMetadataSection fixture_gemma3_tokenizer() {
    auto metadata = fixture_tokenizer(false);
    for (auto& entry : metadata.entries) {
        if (entry.key == "general.architecture") entry = metadata_string(entry.key, "gemma3");
        if (entry.key == "tokenizer.ggml.add_space_prefix") entry = metadata_bool(entry.key, false);
    }
    for (auto& entry : metadata.entries) {
        auto* array = std::get_if<model_format::GGUFMetadataArray>(&entry.value.data);
        if (!array) continue;
        if (entry.key == "tokenizer.ggml.tokens") {
            auto& values = std::get<std::vector<std::string>>(array->values);
            values[1] = "<bos>"; values[2] = "<eos>";
            values.insert(values.end(), {"  ", "\n", "\n\n", "<table>", "<start_of_turn>", "<end_of_turn>", "<unused0>"});
        } else if (entry.key == "tokenizer.ggml.scores") {
            auto& values = std::get<std::vector<float>>(array->values);
            values.insert(values.end(), 7, -1000);
        } else if (entry.key == "tokenizer.ggml.token_type") {
            auto& values = std::get<std::vector<std::int32_t>>(array->values);
            values.insert(values.end(), {4, 4, 4, 4, 3, 3, 3});
        }
    }
    return metadata;
}

void test_gemma3_bounded_user_defined_and_literal_controls() {
    const auto model = load_tokenizer_from_gguf_metadata(fixture_gemma3_tokenizer());
    expect(model.profile() == tokenizer::TokenizerProfile::gemma3, "Gemma3 tokenizer profile lost");
    tokenizer::TokenizerEncodeWorkspace workspace(128);
    std::array<std::uint32_t, 384> output {};
    const std::pair<std::string, std::vector<std::uint32_t>> cases[] {
        {"ab", {262}}, {" ab", {264}}, {"  ab", {266, 262}},
        {"\n\nab", {268, 262}}, {"<table>ab", {269, 262}},
        {"ab<unused0>a", {262, 272, 260}}, {" ▁", {266}}, {"  ▁", {266, 259}},
        {"<bos>ab<end_of_turn>", {1, 262, 271}},
    };
    for (const auto& [text, expected] : cases) {
        const auto count = model.encode_into(text, output, workspace, {.encode_special_tokens = true});
        expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == expected,
            "Gemma3 frozen-user-defined/no-dummy-prefix tokenization mismatch: " + text);
        expect(model.encode(text, {.encode_special_tokens = true}) == expected, "bounded and allocating Gemma3 encoders disagree");
    }
    for (const auto* text : {"<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "<unk>"}) {
        const auto count = model.encode_into(text, output, workspace);
        expect(count > 1, "literal control-looking text incorrectly became a control");
        for (std::size_t i = 0; i < count; ++i) expect(!model.token_id_is_special(output[i]), "ordinary text emitted special ID");
        expect(model.decode(std::span(output).first(count)) == text, "literal control text did not roundtrip");
    }
    const std::array<std::uint32_t, 5> decoded {1, 266, 269, 272, 271};
    expect(model.decode(decoded, {.skip_special_tokens = true}) == "  <table><unused0>", "Gemma3 non-special added pieces were discarded");
    bool failed = false;
    try { (void)model.encode_into("<image_soft_token>", output, workspace, {.encode_special_tokens = true}); }
    catch (const tokenizer::TokenizerError&) { failed = true; }
    expect(failed, "tokenizer-only multimodal ID outside text embedding must fail closed");
    for (const bool input : {false, true}) {
        failed = false;
        try {
            tokenizer::TokenizerEncodeWorkspace small(input ? 1 : 128);
            (void)model.encode_into("<table>ab", input ? std::span(output) : std::span<std::uint32_t>{}, small);
        } catch (const std::length_error&) { failed = true; }
        expect(failed, "Gemma3 bounded input/output overflow must fail");
    }
    expect(model.encode_into("ab", output, workspace) == 1, "Gemma3 workspace not reusable after failure");
    auto malformed = fixture_gemma3_tokenizer();
    for (auto& entry : malformed.entries) if (entry.key == "tokenizer.ggml.add_space_prefix") entry = metadata_bool(entry.key, true);
    failed = false;
    try { (void)load_tokenizer_from_gguf_metadata(malformed); }
    catch (const tokenizer::TokenizerError&) { failed = true; }
    expect(failed, "Gemma3 must not silently accept a dummy-prefix normalizer");
    for (const auto* malformed_key : {"tokenizer.ggml.scores", "tokenizer.ggml.token_type"}) {
        malformed = fixture_gemma3_tokenizer();
        for (auto& entry : malformed.entries) if (entry.key == malformed_key) {
            auto& values = std::get<model_format::GGUFMetadataArray>(entry.value.data).values;
            if (entry.key == "tokenizer.ggml.scores") std::get<std::vector<float>>(values)[260] = std::numeric_limits<float>::quiet_NaN();
            else std::get<std::vector<std::int32_t>>(values)[3] = 1; // remove byte 0
        }
        failed = false;
        try { (void)load_tokenizer_from_gguf_metadata(malformed); }
        catch (const tokenizer::TokenizerError&) { failed = true; }
        expect(failed, "Gemma3 malformed scores/incomplete byte fallback accepted");
    }
    malformed = fixture_gemma3_tokenizer();
    for (auto& entry : malformed.entries) if (entry.key == "tokenizer.ggml.tokens") {
        auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[267] = "▁▁";
    }
    failed = false;
    try { (void)load_tokenizer_from_gguf_metadata(malformed); }
    catch (const tokenizer::TokenizerError&) { failed = true; }
    expect(failed, "Gemma3 ambiguous whitespace aliases accepted");
}

void test_gemma3_control_preserving_injective_bridge() {
    auto target_metadata = fixture_gemma3_tokenizer();
    for (auto& entry : target_metadata.entries) if (entry.key == "general.architecture") entry = metadata_string(entry.key, "gemma2");
    // Explicitly different IDs for SOT/EOT, literal-vs-SP-marker whitespace,
    // and a correction whose ordinary text would merge to a control if enabled.
    for (auto& entry : target_metadata.entries) if (entry.key == "tokenizer.ggml.tokens") {
        auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[266] = "▁▁";
        std::swap(values[270], values[271]);
        values[265] = "<bos>ab";
    }
    const auto target = load_tokenizer_from_gguf_metadata(target_metadata);
    const auto draft = load_tokenizer_from_gguf_metadata(fixture_gemma3_tokenizer());
    const auto legacy = load_tokenizer_from_gguf_metadata(fixture_tokenizer(true));
    for (const bool unsupported_target : {false, true}) {
        bool rejected = false;
        try {
            const engine::StandaloneTliMapping invalid(unsupported_target ? legacy : target,
                unsupported_target ? draft : target);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "unsupported target/draft tokenizer pair must fail closed");
    }
    const engine::StandaloneTliMapping mapping(target, draft);
    expect(!mapping.direct_ids() && mapping.max_append_tokens() > 1, "Gemma2 TLI incorrectly became an identity bridge");
    expect(mapping.target_to_draft()[270] == 271 && mapping.target_to_draft()[271] == 270, "turn controls were mapped by numeric ID");
    expect(mapping.target_to_draft()[2] == 2 && mapping.target_to_draft()[270] != 2, "EOS and EOT conflated");
    expect(mapping.target_to_draft()[266] == 266, "typed whitespace alias missing");
    for (std::uint32_t id = 0; id < mapping.target_to_draft().size(); ++id) {
        const auto mapped = mapping.target_to_draft()[id];
        if (mapped != engine::StandaloneTliMapping::unmapped) expect(mapping.draft_to_target()[mapped] == id, "Gemma mapping is not injective");
    }
    tokenizer::TokenizerEncodeWorkspace workspace(128);
    std::array<std::uint32_t, 384> output {};
    std::array<char, 128> scratch {};
    const std::array<std::uint32_t, 7> prompt {1, 271, 262, 266, 262, 270, 271};
    const std::vector<std::uint32_t> expected {1, 270, 262, 266, 262, 271, 270};
    const auto count = mapping.encode_target_context(prompt, output, scratch, workspace);
    expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == expected,
        "Gemma context lost controls, inserted BOS/dummy prefix, or collapsed whitespace");
    expect(mapping.append_target_token(270, output, scratch, workspace) == 1 && output[0] == 271, "EOT correction was dropped");
    const auto correction_count = mapping.append_target_token(265, output, scratch, workspace);
    expect(correction_count > 1 && correction_count <= mapping.max_append_tokens(), "unmapped correction bound invalid");
    for (std::size_t i = 0; i < correction_count; ++i) expect(!draft.token_id_is_special(output[i]), "literal correction injected a control");
    expect(draft.decode(std::span(output).first(correction_count)) == "<bos>ab", "unmapped correction was altered");
    expect(mapping.append_target_token(272, output, scratch, workspace) == 1 && output[0] == 272,
        "opaque target control with an identical draft symbol was discarded");
    expect(mapping.append_target_token(0, output, scratch, workspace) == 1 && output[0] == 0,
        "unknown target context symbol was discarded");
    expect(mapping.target_to_draft()[272] == engine::StandaloneTliMapping::unmapped
        && mapping.target_to_draft()[0] == engine::StandaloneTliMapping::unmapped,
        "context-only opaque symbols leaked into proposal support");
    for (const bool invalid_id : {false, true}) {
        bool rejected = false;
        try {
            (void)mapping.append_target_token(invalid_id ? 999U : 270U, {}, scratch, workspace);
        } catch (const std::exception&) { rejected = true; }
        expect(rejected, "Gemma bridge accepted invalid token or exhausted correction output");
    }
    bool failed = false;
    try { (void)mapping.encode_target_context(prompt, {}, scratch, workspace); }
    catch (const std::length_error&) { failed = true; }
    expect(failed, "control-preserving context capacity overflow must fail");
    failed = false;
    try { (void)mapping.encode_target_context(prompt, output); }
    catch (const std::logic_error&) { failed = true; }
    expect(failed, "cross-generation context conversion cannot omit its workspace");

    auto ambiguous = target_metadata;
    for (auto& entry : ambiguous.entries) if (entry.key == "tokenizer.ggml.tokens") {
        auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[267] = "  ";
    }
    failed = false;
    try {
        const auto collision = load_tokenizer_from_gguf_metadata(ambiguous);
        const engine::StandaloneTliMapping invalid(collision, draft);
    } catch (const std::invalid_argument&) { failed = true; }
    expect(failed, "two target IDs with one whitespace alias must fail closed");
    auto opaque = target_metadata;
    for (auto& entry : opaque.entries) if (entry.key == "tokenizer.ggml.tokens") {
        auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[272] = "<opaque_control>";
    }
    const auto opaque_target = load_tokenizer_from_gguf_metadata(opaque);
    const engine::StandaloneTliMapping opaque_mapping(opaque_target, draft);
    const auto opaque_count = opaque_mapping.append_target_token(272, output, scratch, workspace);
    expect(opaque_count > 1 && draft.decode(std::span(output).first(opaque_count)) == "<opaque_control>",
        "opaque control without identical draft symbol did not retain its literal context");
}

model_format::GGUFMetadataSection fixture_direct_gemma3_tokenizer(bool draft) {
    auto metadata = fixture_gemma3_tokenizer();
    for (auto& entry : metadata.entries) {
        auto* array = std::get_if<model_format::GGUFMetadataArray>(&entry.value.data);
        if (!array) continue;
        if (entry.key == "tokenizer.ggml.tokens") {
            auto& values = std::get<std::vector<std::string>>(array->values);
            values.insert(values.end(), {"<start_of_image>", "<end_of_image>", "<pad>"});
        } else if (entry.key == "tokenizer.ggml.scores") {
            auto& values = std::get<std::vector<float>>(array->values);
            values.insert(values.end(), 3, -1000);
        } else if (entry.key == "tokenizer.ggml.token_type") {
            auto& values = std::get<std::vector<std::int32_t>>(array->values);
            values[0] = 3; // The released Gemma3 GGUF marks <unk> as CONTROL.
            values.insert(values.end(), {draft ? 4 : 3, draft ? 4 : 3, 3});
        }
    }
    if (draft) metadata.entries.push_back(metadata_bool("tokenizer.ggml.add_sep_token", false));
    return metadata;
}

void test_gemma3_direct_ids_preserve_total_conditioning() {
    const auto target = load_tokenizer_from_gguf_metadata(fixture_direct_gemma3_tokenizer(false));
    const auto draft = load_tokenizer_from_gguf_metadata(fixture_direct_gemma3_tokenizer(true));
    const auto target_identity = target.identity(), draft_identity = draft.identity();
    expect(target_identity != draft_identity, "image-type fixture must retain different authenticated identities");
    const engine::StandaloneTliMapping mapping(target, draft);
    expect(mapping.direct_ids() && mapping.max_append_tokens() == 1, "validated Gemma3 bridge must advertise one-ID conditioning");
    expect(target.identity() == target_identity && draft.identity() == draft_identity,
        "bridge compatibility must not rewrite authenticated tokenizer identities");
    std::array<std::uint32_t, 384> output {};
    for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
        expect(mapping.append_target_token(id, output) == 1 && output[0] == id,
            "one-token bound must cover every ID, including excluded controls");
        const auto mapped = mapping.draft_to_target()[id];
        expect(mapped == engine::StandaloneTliMapping::unmapped
            || (mapped == id && mapping.target_to_draft()[id] == id), "direct proposal map is not injective");
    }
    for (const auto id : {0U, 272U, 273U, 274U, 275U}) {
        expect(mapping.draft_to_target()[id] == engine::StandaloneTliMapping::unmapped
            && mapping.target_to_draft()[id] == engine::StandaloneTliMapping::unmapped,
            "context-only unknown/reserved/image/pad ID leaked into proposal support");
    }
    for (const auto id : {1U, 2U, 3U, 262U, 266U, 270U, 271U})
        expect(mapping.target_to_draft()[id] == id, "ordinary/byte/shared-control proposal support lost");
    // Deliberately noncanonical BPE segmentation, incomplete UTF-8 bytes and
    // every class of excluded control must survive without decode/re-encode.
    const std::array<std::uint32_t, 12> prompt {1, 270, 260, 261, 229, 133, 0, 272, 273, 274, 275, 271};
    expect(mapping.encode_target_context(prompt, output) == prompt.size()
        && std::equal(prompt.begin(), prompt.end(), output.begin()), "direct history was retokenized or lost controls");
    tokenizer::TokenizerEncodeWorkspace no_text_capacity(0);
    expect(mapping.encode_target_context(prompt, output, {}, no_text_capacity) == prompt.size(),
        "legacy overload should bypass text workspace in direct mode");
    expect(mapping.append_target_token(273, output, {}, no_text_capacity) == 1 && output[0] == 273,
        "legacy append overload should preserve context-only IDs without workspace");
    auto overlapping = prompt;
    expect(mapping.encode_target_context(std::span(overlapping).subspan(1), overlapping) == prompt.size() - 1
        && std::equal(prompt.begin() + 1, prompt.end(), overlapping.begin()), "direct in-place history copy corrupted IDs");
    expect(mapping.encode_target_context({}, {}) == 0, "empty direct history needs no storage");
    output.fill(9876);
    const std::array<std::uint32_t, 2> invalid {1, static_cast<std::uint32_t>(target.vocabulary().size())};
    bool failed = false;
    try { (void)mapping.encode_target_context(invalid, output); }
    catch (const std::invalid_argument&) { failed = true; }
    expect(failed && output[0] == 9876, "invalid direct history must fail before changing its destination");
    for (const bool append : {false, true}) {
        failed = false;
        try {
            if (append) (void)mapping.append_target_token(1, {});
            else (void)mapping.encode_target_context(prompt, std::span(output).first(prompt.size() - 1));
        } catch (const std::length_error&) { failed = true; }
        expect(failed, "direct history/correction must check output capacity");
    }
    failed = false;
    try { (void)mapping.append_target_token(invalid.back(), output); }
    catch (const std::invalid_argument&) { failed = true; }
    expect(failed, "direct correction must check vocabulary bounds");
    auto undeclared_metadata = fixture_direct_gemma3_tokenizer(false);
    std::erase_if(undeclared_metadata.entries, [](const auto& entry) {
        return entry.key == "tokenizer.ggml.unknown_token_id";
    });
    const auto undeclared = load_tokenizer_from_gguf_metadata(undeclared_metadata);
    const engine::StandaloneTliMapping undeclared_mapping(undeclared, undeclared);
    expect(undeclared_mapping.direct_ids() && undeclared_mapping.append_target_token(0, output) == 1
        && output[0] == 0 && undeclared_mapping.draft_to_target()[0] == engine::StandaloneTliMapping::unmapped,
        "matching optional unknown-ID absence must preserve special conditioning without a proposal");
}

void test_gemma3_direct_ids_reject_semantic_mismatches() {
    const auto target_metadata = fixture_direct_gemma3_tokenizer(false);
    const auto draft = load_tokenizer_from_gguf_metadata(fixture_direct_gemma3_tokenizer(true));
    for (const std::string_view mutation : {"permuted IDs", "piece", "score", "score sign", "type", "opaque control type",
            "image type", "size", "BOS ID", "EOS ID", "unknown ID", "add BOS", "add EOS", "normalization",
            "pretokenizer", "normalizer", "collapse", "escaping"}) {
        auto metadata = target_metadata;
        for (auto& entry : metadata.entries) {
            if (entry.key == "tokenizer.ggml.tokens") {
                auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
                if (mutation == "permuted IDs") std::swap(values[260], values[261]);
                if (mutation == "piece") values[265] = "z";
                if (mutation == "size") values.pop_back();
            } else if (entry.key == "tokenizer.ggml.scores") {
                auto& values = std::get<std::vector<float>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
                if (mutation == "score") values[260] += 1;
                if (mutation == "score sign") values[1] = -0.0F;
                if (mutation == "size") values.pop_back();
            } else if (entry.key == "tokenizer.ggml.token_type") {
                auto& values = std::get<std::vector<std::int32_t>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
                if (mutation == "type") values[260] = 4;
                if (mutation == "opaque control type") values[272] = 4;
                if (mutation == "image type") values[273] = 1;
                if (mutation == "size") values.pop_back();
            }
            if ((mutation == "BOS ID" && entry.key == "tokenizer.ggml.bos_token_id")
                || (mutation == "EOS ID" && entry.key == "tokenizer.ggml.eos_token_id")
                || (mutation == "unknown ID" && entry.key == "tokenizer.ggml.unknown_token_id"))
                entry = metadata_u32(entry.key, 260);
            if (mutation == "add BOS" && entry.key == "tokenizer.ggml.add_bos_token") entry = metadata_bool(entry.key, false);
            if (mutation == "add EOS" && entry.key == "tokenizer.ggml.add_eos_token") entry = metadata_bool(entry.key, true);
            if (mutation == "normalization" && entry.key == "tokenizer.ggml.add_space_prefix") entry = metadata_bool(entry.key, true);
            if (mutation == "normalizer" && entry.key == "tokenizer.ggml.normalizer") entry = metadata_string(entry.key, "nfkc");
        }
        if (mutation == "pretokenizer") metadata.entries.push_back(metadata_string("tokenizer.ggml.pre", "changed"));
        if (mutation == "collapse") metadata.entries.push_back(metadata_bool("tokenizer.ggml.remove_extra_whitespaces", true));
        if (mutation == "escaping") metadata.entries.push_back(metadata_bool("tokenizer.ggml.escape_whitespaces", false));
        bool failed = false;
        try {
            const auto target = load_tokenizer_from_gguf_metadata(metadata);
            const engine::StandaloneTliMapping mapping(target, draft);
        } catch (const std::exception&) { failed = true; }
        expect(failed, "direct-ID compatibility accepted changed " + std::string(mutation));
    }
    // Matching malformed role metadata is not enough: both sides must still
    // have the Gemma control contract, rather than merely matching each other.
    for (const bool missing_turn : {false, true}) {
        auto metadata = target_metadata;
        for (auto& entry : metadata.entries) {
            if (missing_turn && entry.key == "tokenizer.ggml.tokens") {
                auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
                values[271] = "<unused_end>";
            }
            if (!missing_turn && entry.key == "tokenizer.ggml.bos_token_id") entry = metadata_u32(entry.key, 260);
        }
        bool failed = false;
        try {
            const auto malformed = load_tokenizer_from_gguf_metadata(metadata);
            const engine::StandaloneTliMapping mapping(malformed, malformed);
        } catch (const std::invalid_argument&) { failed = true; }
        expect(failed, "matching malformed Gemma control semantics accepted");
    }
}

void test_bounded_legacy_tokenizer_reference() {
    const auto model = load_tokenizer_from_gguf_metadata(fixture_tokenizer(true));
    tokenizer::TokenizerEncodeWorkspace workspace(128);
    std::array<std::uint32_t, 128> output {};
    // Independently produced with SentencePiece 0.2.1 BPE ModelProto using this
    // exact vocabulary, identity normalization, byte fallback and dummy prefix.
    const std::pair<std::string, std::vector<std::uint32_t>> goldens[] {
        {"", {}}, {"ab", {264}}, {" ab", {259, 264}}, {"a b", {263, 259, 261}},
        {"  ", {259, 259, 259}}, {"\t\n", {259, 12, 13}},
        {"😀", {259, 243, 162, 155, 131}}, {std::string("a\0b", 3), {263, 3, 261}},
        {"\u00a0a", {259, 197, 163, 260}},
        {std::string("\xFF") + "a", {259, 242, 194, 192, 260}},
        {std::string("\xE2\x82") + "a", {259, 242, 194, 192, 260}},
        {std::string("\xED\xA0\x80") + "a", {259, 242, 194, 192, 242, 194, 192, 242, 194, 192, 260}},
        {"<s>ab</s>", {1, 264, 2}}, {"a<unk> b", {263, 0, 259, 259, 261}},
    };
    for (const auto& [text, expected] : goldens) {
        const auto count = model.encode_into(text, output, workspace, {.encode_special_tokens = true});
        expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == expected,
            "bounded legacy SP golden mismatch: " + text);
    }
    const auto count = model.encode_into("ab", output, workspace, {.add_bos = true, .add_eos = true});
    expect(count == 3 && output[0] == 1 && output[1] == 264 && output[2] == 2, "explicit special token insertion mismatch");
}

void test_bounded_tokenizer_capacity_and_metadata() {
    for (const std::size_t bound : {0U, 1U, 7U, 31U, 128U, 4096U}) {
        const tokenizer::TokenizerEncodeWorkspace prepared(bound);
        const auto estimate = tokenizer::TokenizerEncodeWorkspace::estimate_owned_bytes(bound);
        expect(estimate >= prepared.owned_bytes() && estimate - prepared.owned_bytes() <= 32,
            "tokenizer preallocation estimate must cover real reserved capacities with bounded slack");
    }
    bool estimate_failed = false;
    try { (void)tokenizer::TokenizerEncodeWorkspace::estimate_owned_bytes(std::numeric_limits<std::size_t>::max()); }
    catch (const std::length_error&) { estimate_failed = true; }
    expect(estimate_failed, "tokenizer preallocation estimate must reject arithmetic overflow without allocation");
    const auto model = load_tokenizer_from_gguf_metadata(fixture_tokenizer(true));
    tokenizer::TokenizerEncodeWorkspace workspace(2);
    std::array<std::uint32_t, 10> output {};
    for (const bool input_overflow : {false, true}) {
        bool failed = false;
        try {
            (void)model.encode_into(input_overflow ? "abc" : "ab",
                input_overflow ? std::span<std::uint32_t>(output) : std::span<std::uint32_t>{}, workspace);
        } catch (const std::length_error&) { failed = true; }
        expect(failed, "bounded tokenizer must reject capacity overflow");
    }
    expect(model.encode_into("ab", output, workspace) == 1, "workspace must remain reusable after overflow");
    bool failed = false;
    try { tokenizer::TokenizerEncodeWorkspace invalid(std::numeric_limits<std::size_t>::max()); }
    catch (const std::length_error&) { failed = true; }
    expect(failed, "workspace arithmetic overflow must fail");
    auto metadata = fixture_tokenizer(true);
    for (auto& entry : metadata.entries) if (entry.key == "tokenizer.ggml.normalizer") entry = metadata_string(entry.key, "nfkc");
    failed = false;
    try { (void)load_tokenizer_from_gguf_metadata(metadata); }
    catch (const tokenizer::TokenizerError&) { failed = true; }
    expect(failed, "unsupported normalization must not silently become identity");
}

void test_artifact_sha256_published_vectors_and_streaming() {
    // Standard SHA-256 known answers, including the 56-byte padding boundary
    // and the million-'a' multi-block vector. Chunking must not alter receipts.
    const std::pair<std::string, std::string_view> vectors[] {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const auto& [input, expected] : vectors) {
        const auto bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(input.data()), input.size());
        expect(runtime::sha256(bytes).hex() == expected, "SHA-256 published known answer differs");
        for (const std::size_t chunk : {1U, 55U, 56U, 63U, 64U, 65U, 4096U}) {
            runtime::Sha256 streamed;
            streamed.update({});
            for (std::size_t offset = 0; offset < bytes.size(); offset += chunk)
                streamed.update(bytes.subspan(offset, std::min(chunk, bytes.size() - offset)));
            streamed.update({});
            expect(streamed.finalize().hex() == expected, "SHA-256 streaming changed a published digest");
        }
    }
}

void test_artifact_sha256_state_lifecycle() {
    constexpr std::array<std::uint8_t, 3> abc {'a', 'b', 'c'};
    runtime::Sha256 prefix;
    prefix.update(std::span(abc).first(1));
    auto copied = prefix;
    copied.update(std::span(abc).subspan(1));
    expect(copied.finalize() == runtime::sha256(abc), "copied SHA-256 state lost its pending prefix");
    expect(prefix.finalize() == runtime::sha256(std::span(abc).first(1)),
        "updating a copied SHA-256 state changed the original");
    for (const auto action : {0U, 1U, 2U}) {
        bool failed = false;
        try {
            if (action == 0) prefix.update({});
            else if (action == 1) prefix.update(abc);
            else (void)prefix.finalize();
        } catch (const std::logic_error&) { failed = true; }
        expect(failed, "finalized SHA-256 state accepted another operation");
    }
}

void test_gemma3_artifact_payload_authentication() {
    const std::filesystem::path path("models/gemma-3-270m-it-Q8_0.gguf");
    if (!std::filesystem::exists(path)) return;
    // Corrupt a structurally valid tensor byte: directory/shape validation alone
    // must not authenticate a different model. This copy is test-owned only.
    std::string temporary_pattern = (std::filesystem::temp_directory_path() / "gemma3-auth-XXXXXX").string();
    const int descriptor = mkstemp(temporary_pattern.data());
    expect(descriptor >= 0, "could not create artifact authentication fixture");
    close(descriptor);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup {temporary_pattern};
    std::filesystem::copy_file(path, cleanup.path, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream stream(cleanup.path, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekg(-1, std::ios::end);
        char byte = 0;
        stream.read(&byte, 1);
        byte ^= 1;
        stream.seekp(-1, std::ios::end);
        stream.write(&byte, 1);
        expect(stream.good(), "could not mutate artifact authentication fixture");
    }
    bool failed = false;
    try { engine::StandaloneDraftArtifact corrupt(cleanup.path); }
    catch (const std::invalid_argument&) { failed = true; }
    expect(failed, "valid-layout changed tensor bytes must fail authentication");
}

void test_gemma3_artifact_and_tokenizer_goldens_when_available() {
    const std::filesystem::path path("models/gemma-3-270m-it-Q8_0.gguf");
    if (!std::filesystem::exists(path)) return;
    const engine::StandaloneDraftArtifact artifact(path);
    expect(artifact.kind() == engine::StandaloneDraftKind::gemma3_270m_q8 && artifact.context_limit() == 32768,
        "Gemma3 artifact kind/context geometry mismatch");
    expect(artifact.tensor("token_embd.weight").descriptor().shape == tensor::Shape::from({640, 262144}),
        "Gemma3 complete tied head/embedding shape mismatch");
    expect(artifact.tensor("blk.17.attn_q_norm.weight").descriptor().shape == tensor::Shape::from({256}),
        "Gemma3 final layer Q norm missing");
    tokenizer::TokenizerEncodeWorkspace workspace(512);
    std::array<std::uint32_t, 1536> output {};
    // Independent SentencePiece 0.2.1, official tokenizer.model SHA256
    // 1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c.
    // Retrieved from unsloth/gemma-3-270m-it; its Git LFS pointer blob equals
    // Google's 14f810a829755bae3fafd6f97096dbd2eac556bd. Identity normalizer,
    // no dummy prefix/collapse, escaped spaces, complete byte fallback, and
    // longest-prefix frozen USER_DEFINED pieces; GGUF's space aliases differ
    // in representation only. Explicit control insertion is a native API.
    const std::pair<std::string, std::vector<std::uint32_t>> goldens[] {
        {"", {}}, {"Hello, is AI getting smarter?", {9259, 236764, 563, 12498, 3978, 66843, 236881}},
        {" Hello  world\n", {26352, 138, 12392, 107}},
        {"\t\n café 😀", {255968, 107, 33443, 163543}}, {"你好世界", {144626, 12811}},
        {std::string("a\0b", 3), {236746, 238, 236763}},
        {"foo_bar = 12.5", {26537, 236779, 2230, 578, 236743, 236770, 236778, 236761, 236810}},
        {"\u3000x\u00a0y", {465, 366, 366, 236781, 432, 398, 236762}},
        {"▁a", {496}}, {"  ", {138}}, {std::string(32, ' '), {167, 236743}},
        {std::string(33, '\n'), {137, 108}}, {std::string(32, '\t'), {255998, 255968}},
        {" ▁ ", {139}}, {"  ▁", {139}}, {"▁  ", {139}}, {"▁   \t", {140, 255968}},
        {"<table>x</table>", {168, 236781, 176}},
        {"abc<unused0>def", {28180, 6, 2063}}, {"[multimodal]hello", {5, 23391}},
        {"<mask>abc", {4, 28180}}, {"a<start_of_turn>b", {236746, 105, 236763}},
        {"a<end_of_turn>b", {236746, 106, 236763}}, {"e\u0301é", {236744, 238288, 236859}},
        {" \t  \n", {236743, 255968, 138, 107}},
        {"👨‍👩‍👧‍👦", {243874, 237243, 243767, 237243, 244814, 237243, 244990}},
    };
    const auto owned = workspace.owned_bytes();
    for (const auto& [text, expected] : goldens) {
        const auto count = artifact.tokenizer().encode_into(text, output, workspace, {.encode_special_tokens = true});
        expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == expected,
            "real Gemma3 independent tokenizer golden mismatch: " + text);
        expect(workspace.owned_bytes() == owned, "Gemma3 bounded tokenizer grew steady-loop storage");
    }
    if (std::filesystem::exists("models/gemma-3-12b-it-Q4_K_M.gguf")) {
        const auto target_file = model_format::read_file_backed_gguf_model("models/gemma-3-12b-it-Q4_K_M.gguf");
        const auto target = load_tokenizer_from_gguf_metadata(target_file.view().directory.metadata);
        const engine::StandaloneTliMapping mapping(target, artifact.tokenizer());
        expect(mapping.direct_ids() && mapping.max_append_tokens() == 1 && target.vocabulary().size() == 262144,
            "downloaded Gemma3 pair did not establish total one-ID conditioning");
        expect(target.identity() != artifact.tokenizer().identity(),
            "downloaded image-token metadata differences must retain distinct authenticated identities");
        std::size_t differing_types = 0;
        for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
            expect(mapping.append_target_token(id, output) == 1 && output[0] == id,
                "downloaded Gemma3 correction map is not total");
            if (target.vocabulary()[id].type == artifact.tokenizer().vocabulary()[id].type) continue;
            ++differing_types;
            expect((id == 255999 || id == 256000) && target.vocabulary()[id].type == 3
                && artifact.tokenizer().vocabulary()[id].type == 4,
                "downloaded pair has an unaudited token-type difference");
        }
        expect(differing_types == 2, "downloaded pair no longer matches the audited image-marker conversion");
        for (const auto id : {0U, 3U, 4U, 6U, 255999U, 256000U}) {
            expect(mapping.target_to_draft()[id] == engine::StandaloneTliMapping::unmapped,
                "downloaded context-only controls leaked into proposal support");
        }
        expect(mapping.target_to_draft()[5] == 5, "downloaded ordinary [multimodal] added symbol lost its existing proposal semantics");
        for (const auto& [text, expected] : goldens) {
            const auto target_ids = target.encode(text, {.encode_special_tokens = true});
            expect(target_ids == expected, "Gemma3 target differs from independent shared tokenizer golden");
            const auto count = mapping.encode_target_context(target_ids, output);
            expect(count == expected.size() && std::equal(expected.begin(), expected.end(), output.begin()),
                "downloaded Gemma3 direct prompt changed independently verified IDs");
        }
    }
    if (!std::filesystem::exists("models/gemma-2-9b-Q4.gguf")) return;
    const auto target_file = model_format::read_file_backed_gguf_model("models/gemma-2-9b-Q4.gguf");
    const auto target = load_tokenizer_from_gguf_metadata(target_file.view().directory.metadata);
    const engine::StandaloneTliMapping mapping(target, artifact.tokenizer());
    expect(mapping.target_to_draft()[2] == 2 && mapping.target_to_draft()[106] == 105
        && mapping.target_to_draft()[107] == 106 && mapping.target_to_draft()[1] == 1,
        "real Gemma control role mapping mismatch");
    expect(mapping.target_to_draft()[108] == 107 && mapping.target_to_draft()[139] == 138,
        "real Gemma newline/space user-defined mapping mismatch");
    std::array<char, 512> scratch {};
    const tokenizer::ChatMessage user {"user", "Hello, is AI getting smarter?"};
    const auto prompt = tokenizer::format_chat_prompt_from_gguf_metadata(target_file.view().directory.metadata, std::span(&user, 1), true);
    const auto target_ids = target.encode(prompt.text, {.encode_special_tokens = true});
    const auto count = mapping.encode_target_context(target_ids, output, scratch, workspace);
    const auto draft_prompt = artifact.tokenizer().encode(prompt.text, {.encode_special_tokens = true});
    expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == draft_prompt,
        "real Gemma initial prompt/control bridge does not match independent draft tokenization");
    expect(std::count(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count), 2U) == 1,
        "initial Gemma draft prompt must contain one BOS");
    // Mapped IDs retain actual emissions, even where a fresh BPE pass over a
    // concatenation would choose different merges.
    for (const auto id : target_ids) {
        const auto converted = mapping.append_target_token(id, output, scratch, workspace);
        expect(converted <= mapping.max_append_tokens(), "real correction exceeds declared preallocation bound");
        if (mapping.target_to_draft()[id] != engine::StandaloneTliMapping::unmapped) {
            expect(converted == 1 && output[0] == mapping.target_to_draft()[id], "mapped actual emission was retokenized");
        }
    }
}

void test_standalone_rejects_missing_and_unauthenticated_artifacts() {
    bool failed = false;
    try { engine::StandaloneDraftArtifact artifact("models/nonexistent-gemma3-draft.gguf"); }
    catch (const std::exception&) { failed = true; }
    expect(failed, "missing standalone artifact must fail closed");
    if (std::filesystem::exists("models/gemma-2-9b-Q4.gguf")) {
        failed = false;
        try { engine::StandaloneDraftArtifact artifact("models/gemma-2-9b-Q4.gguf"); }
        catch (const std::exception&) { failed = true; }
        expect(failed, "unrelated model must not load as a pinned standalone drafter");
    }
}

void test_prepared_standalone_authenticates_target_tokenizer() {
    const auto bytes = standalone_chat_target_fixture_bytes(128);
    const auto view = model_format::read_gguf_model_view(bytes);
    const auto model = engine::load_verified_engine_model_from_gguf(view, bytes);
    const auto correct = load_tokenizer_from_gguf_metadata(view.directory.metadata);
    expect(model.tokenizer_identity() == correct.identity(), "verified model must authenticate its tokenizer semantics");
    auto changed = view.directory.metadata;
    for (auto& entry : changed.entries) if (entry.key == "tokenizer.ggml.tokens") {
        auto& values = std::get<std::vector<std::string>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[5] = "DifferentSameSizeVocabulary";
    }
    const auto wrong = load_tokenizer_from_gguf_metadata(changed);
    expect(correct.vocabulary().size() == wrong.vocabulary().size() && correct.identity() != wrong.identity(),
        "same vocabulary size must not authenticate different token semantics");
    auto altered_view = view;
    altered_view.directory.metadata = changed;
    const auto altered_model = engine::load_verified_engine_model_from_gguf(altered_view, bytes);
    expect(altered_model.tokenizer_identity() == correct.identity(),
        "caller-mutated parsed view must not change authenticated tokenizer identity");
    bool altered_failed = false;
    try { (void)engine::prepare_gemma_standalone_plan(altered_model, wrong, nullptr, {}); }
    catch (const std::invalid_argument& error) {
        altered_failed = std::string_view(error.what()).find("target tokenizer does not match") != std::string_view::npos;
    }
    expect(altered_failed, "mutated view and matching counterfeit tokenizer must fail before device preparation");
    for (const bool mismatch : {false, true}) {
        bool failed = false;
        try { (void)engine::prepare_gemma_standalone_plan(model, mismatch ? wrong : correct, nullptr, {}); }
        catch (const std::invalid_argument& error) {
            failed = std::string_view(error.what()).find(mismatch ? "target tokenizer does not match" : "authenticated model sources")
                != std::string_view::npos;
        }
        expect(failed, "prepared factory must authenticate tokenizer before device/source preparation");
    }
    auto reordered = view.directory.metadata;
    std::reverse(reordered.entries.begin(), reordered.entries.end());
    expect(load_tokenizer_from_gguf_metadata(reordered).identity() == correct.identity(),
        "semantic tokenizer identity must not depend on metadata ordering");
    auto rescored = fixture_tokenizer(true);
    const auto previous = load_tokenizer_from_gguf_metadata(rescored).identity();
    for (auto& entry : rescored.entries) if (entry.key == "tokenizer.ggml.scores") {
        auto& values = std::get<std::vector<float>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[262] += 1.0F;
    }
    expect(load_tokenizer_from_gguf_metadata(rescored).identity() != previous,
        "tokenizer authentication must bind merge scores as well as token strings");
}

}  // namespace

std::span<const TestCase> standalone_draft_artifact_test_cases() {
    static constexpr TestCase cases[] {
        {"bounded legacy Llama tokenizer matches independent SP equations", test_bounded_legacy_tokenizer_reference},
        {"bounded tokenizer rejects capacity and unsupported normalization", test_bounded_tokenizer_capacity_and_metadata},
        {"Gemma3 bounded tokenizer preserves atomic user-defined pieces and literal controls", test_gemma3_bounded_user_defined_and_literal_controls},
        {"Gemma3 TLI bridge preserves typed controls and rejects ambiguous aliases", test_gemma3_control_preserving_injective_bridge},
        {"Gemma3 direct-ID bridge preserves total conditioning independently of proposals", test_gemma3_direct_ids_preserve_total_conditioning},
        {"Gemma3 direct-ID bridge rejects semantic and control mismatches", test_gemma3_direct_ids_reject_semantic_mismatches},
        {"pinned Gemma3 artifact and independent real tokenizer goldens when available", test_gemma3_artifact_and_tokenizer_goldens_when_available},
        {"artifact SHA-256 matches published vectors across streaming boundaries", test_artifact_sha256_published_vectors_and_streaming},
        {"artifact SHA-256 copied state and finalization lifecycle remain isolated", test_artifact_sha256_state_lifecycle},
        {"pinned Gemma3 artifact rejects altered tensor payload bytes", test_gemma3_artifact_payload_authentication},
        {"standalone artifacts reject missing or unauthenticated sources", test_standalone_rejects_missing_and_unauthenticated_artifacts},
        {"prepared standalone plan authenticates target tokenizer semantics", test_prepared_standalone_authenticates_target_tokenizer},
    };
    return cases;
}
}  // namespace local_gguf_engine::tests
