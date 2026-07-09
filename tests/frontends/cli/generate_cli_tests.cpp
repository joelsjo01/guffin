#include "generate_cli_tests.hpp"

#include "local_gguf_engine/frontends/cli/generate.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using local_gguf_engine::frontends::cli::run_generate;
using local_gguf_engine::model_format::GGUFMetadataArray;
using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::model_format::GGUFMetadataValue;
using local_gguf_engine::model_format::GGUFMetadataValueType;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::gemma2_chat_template_fixture;
using local_gguf_engine::tests::metadata_bool;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_i32_array;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_string_array;
using local_gguf_engine::tests::metadata_u32;

class TempBinaryFile final {
public:
    TempBinaryFile(std::string_view name, std::span<const std::uint8_t> bytes)
        : path_(std::filesystem::temp_directory_path() / ("local-gguf-engine-" + std::string(name) + ".gguf")) {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw TestFailure("could not create temp binary file: " + path_.string());
        }

        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw TestFailure("could not write temp binary file: " + path_.string());
        }
    }

    ~TempBinaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

int run_generate_for_path(
    const std::filesystem::path& path,
    std::string_view prompt,
    std::string_view max_new_tokens,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string prompt_string(prompt);
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        prompt_string.c_str(),
        max_new_tokens_string.c_str(),
    };
    return run_generate(4, argv, output, error);
}

int run_generate_for_path_with_greedy(
    const std::filesystem::path& path,
    std::string_view prompt,
    std::string_view max_new_tokens,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string prompt_string(prompt);
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        prompt_string.c_str(),
        max_new_tokens_string.c_str(),
        "--greedy",
    };
    return run_generate(5, argv, output, error);
}

void append_u8(std::vector<std::uint8_t>& bytes, std::uint8_t value) {
    bytes.push_back(value);
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (std::uint32_t shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_i32(std::vector<std::uint8_t>& bytes, std::int32_t value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void append_f32(std::vector<std::uint8_t>& bytes, float value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void append_gguf_string(std::vector<std::uint8_t>& bytes, std::string_view value) {
    append_u64(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t gguf_metadata_type_id(GGUFMetadataValueType type) {
    return static_cast<std::uint32_t>(type);
}

void append_metadata_array(std::vector<std::uint8_t>& bytes, const GGUFMetadataArray& array) {
    append_u32(bytes, gguf_metadata_type_id(array.element_type));

    switch (array.element_type) {
        case GGUFMetadataValueType::string: {
            const auto& values = std::get<std::vector<std::string>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto& value : values) {
                append_gguf_string(bytes, value);
            }
            return;
        }
        case GGUFMetadataValueType::int32: {
            const auto& values = std::get<std::vector<std::int32_t>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_i32(bytes, value);
            }
            return;
        }
        case GGUFMetadataValueType::float32: {
            const auto& values = std::get<std::vector<float>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_f32(bytes, value);
            }
            return;
        }
        default:
            throw TestFailure("test GGUF writer unsupported metadata array type");
    }
}

void append_metadata_value(std::vector<std::uint8_t>& bytes, const GGUFMetadataValue& value) {
    append_u32(bytes, gguf_metadata_type_id(value.type));

    switch (value.type) {
        case GGUFMetadataValueType::uint32:
            append_u32(bytes, std::get<std::uint32_t>(value.data));
            return;
        case GGUFMetadataValueType::float32:
            append_f32(bytes, std::get<float>(value.data));
            return;
        case GGUFMetadataValueType::boolean:
            append_u8(bytes, static_cast<std::uint8_t>(std::get<bool>(value.data) ? 1 : 0));
            return;
        case GGUFMetadataValueType::string:
            append_gguf_string(bytes, std::get<std::string>(value.data));
            return;
        case GGUFMetadataValueType::array:
            append_metadata_array(bytes, std::get<GGUFMetadataArray>(value.data));
            return;
        default:
            throw TestFailure("test GGUF writer unsupported metadata value type");
    }
}

void append_metadata_entry(std::vector<std::uint8_t>& bytes, const GGUFMetadataEntry& entry) {
    append_gguf_string(bytes, entry.key);
    append_metadata_value(bytes, entry.value);
}

std::uint32_t gguf_type_id_for_dtype(DType dtype) {
    switch (dtype) {
        case DType::f32:
            return 0;
        case DType::f16:
            return 1;
        case DType::q8_0:
            return 8;
        case DType::q4_k:
            return 12;
        case DType::q6_k:
            return 14;
    }

    throw TestFailure("test GGUF writer unsupported tensor dtype");
}

struct TinyGGUFTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type_id = 0;
    std::vector<std::uint8_t> bytes;
};

std::size_t align_size(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }

    return value + alignment - remainder;
}

std::vector<std::uint8_t> write_tiny_gguf_bytes(
    const std::vector<GGUFMetadataEntry>& metadata,
    const std::vector<TinyGGUFTensor>& tensors
) {
    constexpr std::size_t alignment = 32;
    std::vector<std::uint64_t> tensor_offsets;
    tensor_offsets.reserve(tensors.size());

    std::size_t payload_offset = 0;
    for (const auto& tensor : tensors) {
        tensor_offsets.push_back(static_cast<std::uint64_t>(payload_offset));
        payload_offset = align_size(payload_offset + tensor.bytes.size(), alignment);
    }

    std::vector<std::uint8_t> bytes;
    bytes.push_back(0x47);
    bytes.push_back(0x47);
    bytes.push_back(0x55);
    bytes.push_back(0x46);
    append_u32(bytes, 3);
    append_u64(bytes, static_cast<std::uint64_t>(tensors.size()));
    append_u64(bytes, static_cast<std::uint64_t>(metadata.size()));

    for (const auto& entry : metadata) {
        append_metadata_entry(bytes, entry);
    }

    for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto& tensor = tensors[tensor_index];
        append_gguf_string(bytes, tensor.name);
        append_u32(bytes, static_cast<std::uint32_t>(tensor.dimensions.size()));
        for (const auto dimension : tensor.dimensions) {
            append_u64(bytes, dimension);
        }
        append_u32(bytes, tensor.type_id);
        append_u64(bytes, tensor_offsets[tensor_index]);
    }

    const auto tensor_data_start = align_size(bytes.size(), alignment);
    bytes.resize(tensor_data_start, 0);

    for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto& tensor = tensors[tensor_index];
        const auto write_offset = tensor_data_start + static_cast<std::size_t>(tensor_offsets[tensor_index]);
        if (bytes.size() < write_offset) {
            bytes.resize(write_offset, 0);
        }
        bytes.insert(bytes.end(), tensor.bytes.begin(), tensor.bytes.end());
    }

    return bytes;
}

std::vector<std::uint8_t> zero_tensor_bytes(std::span<const std::uint64_t> dimensions, DType dtype) {
    return std::vector<std::uint8_t>(
        TensorDescriptor::from(Shape::from(dimensions), dtype).byte_count
    );
}

TinyGGUFTensor tiny_zero_tensor(std::string name, std::vector<std::uint64_t> dimensions, DType dtype) {
    return TinyGGUFTensor {
        .name = std::move(name),
        .dimensions = dimensions,
        .type_id = gguf_type_id_for_dtype(dtype),
        .bytes = zero_tensor_bytes(std::span<const std::uint64_t>(dimensions), dtype),
    };
}

[[nodiscard]] std::string gemma2_block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

std::vector<std::string> tiny_gemma2_generate_cli_token_texts() {
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
    tokens[9] = "user\n";
    tokens[10] = "model\n";
    tokens[11] = "\n";
    tokens[106] = "<start_of_turn>";
    tokens[107] = "<end_of_turn>";
    return tokens;
}

std::vector<std::int32_t> tiny_gemma2_generate_cli_token_types() {
    std::vector<std::int32_t> token_types(110, 1);
    token_types[0] = 3;
    token_types[1] = 3;
    token_types[2] = 3;
    token_types[3] = 2;
    token_types[4] = 3;
    token_types[106] = 3;
    token_types[107] = 3;
    return token_types;
}

std::vector<GGUFMetadataEntry> tiny_gemma2_generate_cli_metadata(std::uint32_t context_length = 16) {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Generate Fixture"),
        metadata_u32("gemma2.context_length", context_length),
        metadata_u32("gemma2.embedding_length", 256),
        metadata_u32("gemma2.block_count", 1),
        metadata_u32("gemma2.feed_forward_length", 256),
        metadata_u32("gemma2.attention.head_count", 1),
        metadata_u32("gemma2.attention.head_count_kv", 1),
        metadata_u32("gemma2.attention.key_length", 256),
        metadata_u32("gemma2.attention.value_length", 256),
        metadata_u32("gemma2.attention.sliding_window", 8),
        metadata_f32("gemma2.attention.layer_norm_rms_epsilon", 0.000001F),
        metadata_f32("gemma2.attn_logit_softcapping", 50.0F),
        metadata_f32("gemma2.final_logit_softcapping", 30.0F),
        metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string_array("tokenizer.ggml.tokens", tiny_gemma2_generate_cli_token_texts()),
        metadata_i32_array("tokenizer.ggml.token_type", tiny_gemma2_generate_cli_token_types()),
        metadata_u32("tokenizer.ggml.bos_token_id", 2),
        metadata_u32("tokenizer.ggml.eos_token_id", 1),
        metadata_u32("tokenizer.ggml.unknown_token_id", 3),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
        metadata_bool("tokenizer.ggml.add_space_prefix", false),
        metadata_string("tokenizer.chat_template", gemma2_chat_template_fixture()),
    };
}

std::vector<TinyGGUFTensor> tiny_gemma2_generate_cli_tensors() {
    std::vector<TinyGGUFTensor> tensors;
    tensors.reserve(13);
    tensors.push_back(tiny_zero_tensor("token_embd.weight", {256, 110}, DType::f16));
    tensors.push_back(tiny_zero_tensor("output_norm.weight", {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "post_attention_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_q.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_k.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_v.weight"), {256, 256}, DType::q6_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_output.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "post_ffw_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_gate.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_up.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_down.weight"), {256, 256}, DType::q6_k));
    return tensors;
}

std::vector<std::uint8_t> tiny_gemma2_generate_cli_gguf_bytes() {
    return write_tiny_gguf_bytes(tiny_gemma2_generate_cli_metadata(), tiny_gemma2_generate_cli_tensors());
}

void test_runs_generate_gemma2_default_greedy_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-default-greedy", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 default greedy exit code mismatch");
    expect(output.str() == "\n", "generate Gemma2 default greedy output mismatch");
    expect(error.str().empty(), "generate Gemma2 default greedy should not write stderr");
}

void test_runs_generate_gemma2_greedy_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-greedy", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path_with_greedy(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 greedy exit code mismatch");
    expect(output.str() == "\n", "generate Gemma2 greedy output mismatch");
    expect(error.str().empty(), "generate Gemma2 greedy should not write stderr");
}

void test_reports_generate_gemma2_metal_runtime_shell_unavailable_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-metal-missing", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path_with_greedy(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 unavailable runtime shell exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 unavailable runtime shell should not write stdout");
    expect(
        error.str().find("error: backend:") != std::string::npos,
        "generate Gemma2 unavailable runtime shell error prefix missing"
    );
    expect(
        error.str().find("Metal backend unavailable") != std::string::npos,
        "generate Gemma2 unavailable runtime shell message missing"
    );
}

void test_rejects_generate_gemma2_cpu_backend() {
    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-cpu", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "Hello",
        "1",
        "--greedy",
        "--backend",
        "cpu",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 CPU backend exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 CPU backend should not write stdout");
    expect(
        error.str().find("Gemma 2 generation requires Metal backend") != std::string::npos,
        "generate Gemma2 CPU backend error missing"
    );
}

void test_rejects_generate_gemma2_after_decode_schedule_context_full_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(8),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-decode-context-full", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path_with_greedy(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 decode schedule context-full exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 decode schedule context-full should not write stdout");
    expect(
        error.str().find("Gemma2 Metal generation token count exceeds context length") != std::string::npos,
        "generate Gemma2 decode schedule context-full error missing"
    );
}

void test_rejects_generate_gemma2_unsupported_chat_template_before_runtime() {
    auto metadata = tiny_gemma2_generate_cli_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.chat_template") {
            entry = metadata_string("tokenizer.chat_template", "{{ messages }}");
            break;
        }
    }

    const auto bytes = write_tiny_gguf_bytes(metadata, tiny_gemma2_generate_cli_tensors());
    const TempBinaryFile model("tiny-gemma2-bad-template", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 unsupported template exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 unsupported template should not write stdout");
    expect(
        error.str().find("unsupported chat template: tokenizer.chat_template") != std::string::npos,
        "generate Gemma2 unsupported template error missing"
    );
}

void test_rejects_generate_missing_arguments() {
    const char* argv[] = {"local-gguf-generate"};
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(1, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate missing-args exit code mismatch");
    expect(output.str().empty(), "generate missing-args should not write stdout");
    expect(
        error.str().find("usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>") != std::string::npos,
        "generate missing-args usage missing"
    );
    expect(error.str().find("missing required argument") != std::string::npos, "generate missing-args error missing");
}

void test_rejects_generate_invalid_max_new_tokens() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(path, "", "0", output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-max exit code mismatch");
    expect(output.str().empty(), "generate invalid-max should not write stdout");
    expect(
        error.str().find("max-new-tokens must be a positive integer") != std::string::npos,
        "generate invalid-max error missing"
    );
}

void test_rejects_generate_invalid_backend_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--backend",
        "gpu",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-backend exit code mismatch");
    expect(output.str().empty(), "generate invalid-backend should not write stdout");
    expect(
        error.str().find("--backend must be cpu or metal") != std::string::npos,
        "generate invalid-backend error missing"
    );
}

const local_gguf_engine::tests::TestCase kGenerateCliTests[] = {
    {
        "runs generate Gemma2 default greedy when available",
        test_runs_generate_gemma2_default_greedy_when_available,
    },
    {
        "runs generate Gemma2 greedy when available",
        test_runs_generate_gemma2_greedy_when_available,
    },
    {
        "reports generate Gemma2 Metal runtime shell unavailable when missing",
        test_reports_generate_gemma2_metal_runtime_shell_unavailable_when_missing,
    },
    {
        "rejects generate Gemma2 CPU backend",
        test_rejects_generate_gemma2_cpu_backend,
    },
    {
        "rejects generate Gemma2 after decode schedule context full when available",
        test_rejects_generate_gemma2_after_decode_schedule_context_full_when_available,
    },
    {
        "rejects generate Gemma2 unsupported chat template before runtime",
        test_rejects_generate_gemma2_unsupported_chat_template_before_runtime,
    },
    {"rejects generate missing arguments", test_rejects_generate_missing_arguments},
    {"rejects generate invalid max new tokens", test_rejects_generate_invalid_max_new_tokens},
    {"rejects generate invalid backend option", test_rejects_generate_invalid_backend_option},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> generate_cli_test_cases() {
    return kGenerateCliTests;
}

}  // namespace local_gguf_engine::tests
