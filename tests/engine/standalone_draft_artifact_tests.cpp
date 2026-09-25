#include "test_support.hpp"

#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"
#include "local_gguf_engine/engine/prepared_gemma2_plan.hpp"
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
    const engine::StandaloneTliMapping mapping(target, draft);
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
    bool failed = false;
    try { (void)mapping.encode_target_context(prompt, {}, scratch, workspace); }
    catch (const std::length_error&) { failed = true; }
    expect(failed, "control-preserving context capacity overflow must fail");

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

void test_tli_mapping_and_bounded_context() {
    const auto target = load_tokenizer_from_gguf_metadata(fixture_tokenizer(false));
    const auto draft = load_tokenizer_from_gguf_metadata(fixture_tokenizer(true));
    const engine::StandaloneTliMapping mapping(target, draft);
    expect(mapping.intersection_size() == 263, "normal/byte TLI intersection mismatch");
    expect(mapping.target_to_draft()[0] == engine::StandaloneTliMapping::unmapped, "unknown must not be proposed");
    expect(mapping.draft_to_target()[1] == engine::StandaloneTliMapping::unmapped, "control must not be proposed");
    tokenizer::TokenizerEncodeWorkspace workspace(64);
    std::array<std::uint32_t, 64> output {};
    std::array<char, 64> scratch {};
    expect(mapping.append_target_token(3, output, scratch, workspace) == 1 && output[0] == 3, "NUL byte mapping mismatch");
    expect(mapping.append_target_token(1, output, scratch, workspace) == 0, "target special must not add draft BOS");
    const std::array<std::uint32_t, 3> prompt {1, 262, 2};
    expect(mapping.encode_target_context(prompt, output, scratch, workspace) == 1 && output[0] == 264,
        "initial context must skip target specials and apply SP dummy prefix without BOS");
    const std::array<std::uint32_t, 3> punctuation {260, 35, 36}; // a + byte space + byte '!'
    expect(mapping.encode_target_context(punctuation, output, scratch, workspace) == 2 && output[0] == 263 && output[1] == 36,
        "reference target context punctuation cleanup mismatch");
    const std::array<std::uint32_t, 2> invalid_utf8 {229, 133}; // incomplete E2 82
    expect(mapping.encode_target_context(invalid_utf8, output, scratch, workspace) == 4
        && output[0] == 259 && output[1] == 242 && output[2] == 194 && output[3] == 192,
        "context byte stream must replace one incomplete UTF8 prefix, not each continuation byte");
    bool failed = false;
    try { (void)mapping.append_target_token(999, output, scratch, workspace); }
    catch (const std::invalid_argument&) { failed = true; }
    expect(failed, "out-of-range TLI token must fail");
    failed = false;
    try { (void)mapping.append_target_token(260, {}, scratch, workspace); }
    catch (const std::length_error&) { failed = true; }
    expect(failed, "mapped TLI output capacity must fail closed");
    // Type mismatches cannot be conflated merely because piece text is equal.
    auto changed = fixture_tokenizer(false);
    for (auto& entry : changed.entries) if (entry.key == "tokenizer.ggml.token_type") {
        auto& values = std::get<std::vector<std::int32_t>>(std::get<model_format::GGUFMetadataArray>(entry.value.data).values);
        values[263] = 4;
    }
    const auto target_variant = load_tokenizer_from_gguf_metadata(changed);
    const engine::StandaloneTliMapping variant(target_variant, draft);
    expect(variant.target_to_draft()[263] == engine::StandaloneTliMapping::unmapped, "type-mismatched TLI mapping admitted");
    expect(variant.append_target_token(263, output, scratch, workspace) == 2 && output[0] == 259 && output[1] == 263,
        "unmapped correction must use legacy dummy prefix, not continuation normalization");
}

void test_vicuna_artifact_goldens_when_available() {
    const std::filesystem::path path("models/vicuna_68m/vicuna-68m-f16.gguf");
    if (!std::filesystem::exists(path)) return;
    engine::StandaloneDraftArtifact artifact(path);
    expect(artifact.kind() == engine::StandaloneDraftKind::vicuna_68m_f16 && artifact.context_limit() == 2048,
        "existing Vicuna artifact identity/context changed");
    expect(artifact.tensor("lm_head.weight").descriptor().shape == tensor::Shape::from({768, 32000}), "Vicuna full head shape mismatch");
    expect(artifact.tensor("model.layers.1.mlp.down_proj.weight").descriptor().shape == tensor::Shape::from({3072, 768}), "Vicuna second layer missing/transposed");
    tokenizer::TokenizerEncodeWorkspace workspace(128);
    std::array<std::uint32_t, 128> output {};
    // Transformers 4.50.0 LlamaTokenizer(legacy=True), SentencePiece 0.2.1;
    // pinned source tokenizer SHA 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347.
    const std::pair<std::string, std::vector<std::uint32_t>> goldens[] {
        {"", {}}, {"Hello world!", {15043, 3186, 29991}},
        {" Hello  world\n", {29871, 15043, 29871, 3186, 13}},
        {"\t\n café 😀", {29871, 12, 13, 274, 28059, 29871, 243, 162, 155, 131}},
        {"你好世界", {29871, 30919, 31076, 30793, 30967}},
        {std::string("a\0b", 3), {263, 3, 29890}},
        {"<s>Hello</s>", {1, 15043, 2}}, {"x<unk> y", {921, 0, 29871, 343}},
        {"foo_bar = 12.5", {7953, 29918, 1646, 353, 29871, 29896, 29906, 29889, 29945}},
        {"\u3000x\u00a0y", {29871, 30358, 29916, 30081, 29891}},
        {"▁a", {29871, 263}},
    };
    for (const auto& [text, expected] : goldens) {
        const auto count = artifact.tokenizer().encode_into(text, output, workspace, {.encode_special_tokens = true});
        expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == expected,
            "real Vicuna tokenizer reference mismatch: " + text);
    }
    if (std::filesystem::exists("models/gemma-2-9b-Q4.gguf")) {
        const auto target_file = model_format::read_file_backed_gguf_model("models/gemma-2-9b-Q4.gguf");
        const auto target = load_tokenizer_from_gguf_metadata(target_file.view().directory.metadata);
        const engine::StandaloneTliMapping mapping(target, artifact.tokenizer());
        expect(mapping.intersection_size() > 30000, "real Gemma/Vicuna TLI vocabulary intersection unexpectedly missing");
        std::array<char, 128> scratch {};
        // Independently decoded target pieces and encoded them with pinned HF
        // LlamaTokenizer(add_special_tokens=False), including leading spaces.
        const std::pair<std::uint32_t, std::vector<std::uint32_t>> corrections[] {
            {613, {29871, 281, 29875}}, {636, {274, 2034}}, {662, {5539}},
            {802, {260, 293}}, {815, {330, 400}}, {817, {269, 291}},
            {906, {29871, 318, 333}}, {922, {259, 30112}},
        };
        for (const auto& [id, expected] : corrections) {
            const auto count = mapping.append_target_token(id, output, scratch, workspace);
            expect(std::vector<std::uint32_t>(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(count)) == expected,
                "real TLI unmapped correction tokenizer golden mismatch");
        }
    }
    bool failed = false;
    try { (void)artifact.tensor("missing.weight"); }
    catch (const std::invalid_argument&) { failed = true; }
    expect(failed, "missing Vicuna tensor must fail closed");

    // Corrupt a structurally valid tensor byte: directory/shape validation alone
    // must not authenticate a different model. This copy is test-owned only.
    std::string temporary_pattern = (std::filesystem::temp_directory_path() / "vicuna-auth-XXXXXX").string();
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
    failed = false;
    try { engine::StandaloneDraftArtifact corrupt(cleanup.path); }
    catch (const std::invalid_argument&) { failed = true; }
    expect(failed, "valid-layout changed tensor bytes must fail authentication");
}

void test_gemma3_artifact_and_tokenizer_goldens_when_available() {
    const std::filesystem::path path("models/gemma-3-270m-it-Q8_0.gguf");
    if (!std::filesystem::exists(path)) return;
    const engine::StandaloneDraftArtifact artifact(path);
    expect(artifact.kind() == engine::StandaloneDraftKind::gemma3_270m_q8 && artifact.context_limit() == 32768,
        "Gemma3 artifact inherited Vicuna kind/context geometry");
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

void test_vicuna_rejects_missing_and_unauthenticated_artifacts() {
    bool failed = false;
    try { engine::StandaloneDraftArtifact artifact("models/nonexistent-vicuna-draft.gguf"); }
    catch (const std::exception&) { failed = true; }
    expect(failed, "missing standalone artifact must fail closed");
    if (std::filesystem::exists("models/gemma-2-9b-Q4.gguf")) {
        failed = false;
        try { engine::StandaloneDraftArtifact artifact("models/gemma-2-9b-Q4.gguf"); }
        catch (const std::exception&) { failed = true; }
        expect(failed, "unrelated model must not load as Vicuna");
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
    try { (void)engine::prepare_gemma2_standalone_plan(altered_model, wrong, nullptr, {}); }
    catch (const std::invalid_argument& error) {
        altered_failed = std::string_view(error.what()).find("target tokenizer does not match") != std::string_view::npos;
    }
    expect(altered_failed, "mutated view and matching counterfeit tokenizer must fail before device preparation");
    for (const bool mismatch : {false, true}) {
        bool failed = false;
        try { (void)engine::prepare_gemma2_standalone_plan(model, mismatch ? wrong : correct, nullptr, {}); }
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
        {"TLI mapping owns exact token types and bounded context conversion", test_tli_mapping_and_bounded_context},
        {"Gemma3 bounded tokenizer preserves atomic user-defined pieces and literal controls", test_gemma3_bounded_user_defined_and_literal_controls},
        {"Gemma3 TLI bridge preserves typed controls and rejects ambiguous aliases", test_gemma3_control_preserving_injective_bridge},
        {"pinned Gemma3 artifact and independent real tokenizer goldens when available", test_gemma3_artifact_and_tokenizer_goldens_when_available},
        {"pinned Vicuna artifact and real tokenizer goldens when available", test_vicuna_artifact_goldens_when_available},
        {"Vicuna artifacts reject missing or unauthenticated sources", test_vicuna_rejects_missing_and_unauthenticated_artifacts},
        {"prepared standalone plan authenticates target tokenizer semantics", test_prepared_standalone_authenticates_target_tokenizer},
    };
    return cases;
}
}  // namespace local_gguf_engine::tests
