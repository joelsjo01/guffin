#include "generate_cli_tests.hpp"

#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/benchmarks/generation_benchmark.hpp"
#include "local_gguf_engine/frontends/cli/generate.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include "tensor_test_support.hpp"

#include <algorithm>
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
using local_gguf_engine::frontends::cli::format_chat_timing_line;
using local_gguf_engine::benchmarks::GenerationBenchmarkCase;
using local_gguf_engine::benchmarks::GenerationBenchmarkRunnerOptions;
using local_gguf_engine::benchmarks::run_generation_benchmark;
using local_gguf_engine::engine::Gemma2ChatSession;
using local_gguf_engine::engine::Gemma2ChatContextError;
using local_gguf_engine::engine::Gemma2ChatSessionOptions;
using local_gguf_engine::engine::Gemma2ChatTurnOptions;
using local_gguf_engine::engine::Gemma2GenerationStopReason;
using local_gguf_engine::engine::Gemma2TokenSelectionMode;
using local_gguf_engine::engine::TokenSamplerOptions;
using local_gguf_engine::model_format::read_file_backed_gguf_model;
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
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_vector;

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

[[nodiscard]] std::size_t chat_timing_line_count(std::string_view text) {
    std::size_t count = 0;
    for (std::size_t position = 0; (position = text.find("[ Prompt: ", position)) != std::string_view::npos;) {
        ++count;
        position += 10U;
    }
    return count;
}

[[nodiscard]] std::string without_chat_timing_lines(std::string_view text) {
    std::string filtered;
    std::size_t line_start = 0;
    while (line_start < text.size()) {
        const auto newline = text.find('\n', line_start);
        const auto line_end = newline == std::string_view::npos ? text.size() : newline + 1U;
        const auto line = text.substr(line_start, line_end - line_start);
        if (!line.starts_with("[ Prompt: ")) {
            filtered.append(line);
        }
        line_start = line_end;
    }
    return filtered;
}

void test_formats_chat_timing_line_from_runtime_metrics() {
    local_gguf_engine::engine::Gemma2ChatTurnMetrics metrics;
    metrics.generation.newly_evaluated_prompt_token_count = 19U;
    metrics.generation.prefill_duration_ns = 1'000'000'000U;
    metrics.generation.decode_token_count = 78U;
    metrics.generation.decode_duration_ns = 5'000'000'000U;
    expect(
        format_chat_timing_line(metrics) == "[ Prompt: 19.0 t/s | Generation: 15.6 t/s ]",
        "chat timing normal formatting mismatch"
    );

    metrics.generation.newly_evaluated_prompt_token_count = 2U;
    metrics.generation.prefill_duration_ns = 3'000'000'000U;
    metrics.generation.decode_token_count = 0U;
    metrics.generation.decode_duration_ns = 0U;
    expect(
        format_chat_timing_line(metrics) == "[ Prompt: 0.7 t/s | Generation: n/a ]",
        "chat timing rounding/unavailable formatting mismatch"
    );

    metrics.generation.newly_evaluated_prompt_token_count = 0U;
    metrics.generation.prefill_duration_ns = 1U;
    metrics.generation.decode_token_count = 1U;
    metrics.generation.decode_duration_ns = 0U;
    expect(
        format_chat_timing_line(metrics) == "[ Prompt: n/a | Generation: n/a ]",
        "chat timing zero input formatting mismatch"
    );
}

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

int run_generate_chat_for_path(
    const std::filesystem::path& path,
    std::string_view max_new_tokens,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "--chat",
        "--max-new-tokens",
        max_new_tokens_string.c_str(),
    };
    return run_generate(5, argv, input, output, error);
}

int run_generate_chat_for_path_with_reverse_stop(
    const std::filesystem::path& path,
    std::string_view max_new_tokens,
    std::string_view reverse_stop,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string max_new_tokens_string(max_new_tokens);
    const std::string reverse_stop_string(reverse_stop);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "--chat",
        "--max-new-tokens",
        max_new_tokens_string.c_str(),
        "--reverse-prompt",
        reverse_stop_string.c_str(),
    };
    return run_generate(7, argv, input, output, error);
}

int run_generate_chat_for_path_with_sampling(
    const std::filesystem::path& path,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(), path_string.c_str(), "--chat", "--max-new-tokens", "3",
        "--temperature", "1", "--top-k", "2", "--top-p", "1",
        "--repeat-penalty", "1.1", "--seed", "42",
    };
    return run_generate(15, argv, input, output, error);
}

int run_generate_chat_with_extra_arguments(
    const std::filesystem::path& path,
    const std::vector<std::string>& extra_arguments,
    std::ostringstream& output,
    std::ostringstream& error
) {
    std::vector<std::string> arguments {
        "local-gguf-generate", path.string(), "--chat", "--max-new-tokens", "1",
    };
    arguments.insert(arguments.end(), extra_arguments.begin(), extra_arguments.end());
    std::vector<const char*> argv;
    argv.reserve(arguments.size());
    for (const auto& argument : arguments) {
        argv.push_back(argument.c_str());
    }
    std::istringstream input("/exit\n");
    return run_generate(static_cast<int>(argv.size()), argv.data(), input, output, error);
}

int run_generate_with_extra_arguments(
    const std::filesystem::path& path,
    const std::vector<std::string>& extra_arguments,
    std::ostringstream& output,
    std::ostringstream& error
) {
    std::vector<std::string> arguments {"local-gguf-generate", path.string(), "Hello", "1"};
    arguments.insert(arguments.end(), extra_arguments.begin(), extra_arguments.end());
    std::vector<const char*> argv;
    for (const auto& argument : arguments) argv.push_back(argument.c_str());
    return run_generate(static_cast<int>(argv.size()), argv.data(), output, error);
}

[[nodiscard]] bool looks_like_strict_json_object(std::string_view line) {
    if (line.size() < 2U || line.front() != '{' || line.back() != '}') return false;
    bool in_string = false;
    bool escaped = false;
    int depth = 0;
    for (const auto character : line) {
        if (escaped) { escaped = false; continue; }
        if (in_string && character == '\\') { escaped = true; continue; }
        if (character == '"') { in_string = !in_string; continue; }
        if (in_string) {
            if (static_cast<unsigned char>(character) < 0x20U) return false;
            continue;
        }
        if (character == '{' || character == '[') ++depth;
        if (character == '}' || character == ']') --depth;
        if (depth < 0) return false;
    }
    return !in_string && !escaped && depth == 0;
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
    tokens[6] = "▁Hello";
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

void replace_tiny_tensor(
    std::vector<TinyGGUFTensor>& tensors,
    std::string_view name,
    const local_gguf_engine::tensor::TensorStorage& storage
) {
    for (auto& tensor : tensors) {
        if (tensor.name != name) {
            continue;
        }

        tensor.bytes.assign(storage.bytes().begin(), storage.bytes().end());
        return;
    }

    throw TestFailure(std::string("missing tiny tensor fixture ") + std::string(name));
}

void insert_output_weight_tensor(
    std::vector<TinyGGUFTensor>& tensors,
    const local_gguf_engine::tensor::TensorStorage& storage
) {
    tensors.insert(
        tensors.begin() + 2,
        TinyGGUFTensor {
            .name = "output.weight",
            .dimensions = {256, 110},
            .type_id = gguf_type_id_for_dtype(DType::f16),
            .bytes = std::vector<std::uint8_t>(storage.bytes().begin(), storage.bytes().end()),
        }
    );
}

std::vector<TinyGGUFTensor> tiny_gemma2_generate_cli_tensors_for_forced_token(
    std::uint32_t token_id
) {
    auto tensors = tiny_gemma2_generate_cli_tensors();

    std::vector<float> embedding_values(256U * 110U, 0.0F);
    for (std::uint32_t token = 0; token < 110U; ++token) {
        embedding_values[token] = 1.0F;
    }
    const auto embedding = tensor_f16_gguf_matrix(
        256,
        110,
        std::span<const float>(embedding_values)
    );
    replace_tiny_tensor(tensors, "token_embd.weight", embedding);

    std::vector<float> output_norm_values(256U, 0.0F);
    output_norm_values[0] = 1.0F;
    const auto output_norm = tensor_f32_vector(std::span<const float>(output_norm_values));
    replace_tiny_tensor(tensors, "output_norm.weight", output_norm);

    std::vector<float> output_weight_values(256U * 110U, 0.0F);
    output_weight_values[token_id] = 1.0F;
    const auto output_weight = tensor_f16_gguf_matrix(
        256,
        110,
        std::span<const float>(output_weight_values)
    );
    insert_output_weight_tensor(tensors, output_weight);

    return tensors;
}

std::vector<TinyGGUFTensor> tiny_gemma2_generate_cli_tensors_for_sampled_tokens() {
    auto tensors = tiny_gemma2_generate_cli_tensors_for_forced_token(5);
    std::vector<float> output_weight_values(256U * 110U, 0.0F);
    output_weight_values[5] = 0.25F;
    output_weight_values[11] = 0.20F;
    const auto output_weight = tensor_f16_gguf_matrix(
        256,
        110,
        std::span<const float>(output_weight_values)
    );
    replace_tiny_tensor(tensors, "output.weight", output_weight);
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

void test_gemma2_chat_session_appends_second_turn_at_nonzero_cache_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-session", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    const auto first_turn = session.generate_user_turn("Hello");
    const auto second_turn = session.generate_user_turn("Hello");

    expect(first_turn.cache_position == 0, "Gemma2 chat first turn cache position mismatch");
    expect(second_turn.cache_position > 0, "Gemma2 chat second turn should append at nonzero cache position");
    expect(second_turn.appended_token_count > 0, "Gemma2 chat second turn should append prompt tokens");
    expect(session.message_count() == 4, "Gemma2 chat session message count mismatch after two turns");
}

void test_gemma2_chat_session_appends_after_truncated_output_retokenizes_differently_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(6)
    );
    const TempBinaryFile model("tiny-gemma2-truncated-retokenize", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1U}
    );
    const auto first = session.generate_user_turn("Hello");
    const auto second = session.generate_user_turn("Hello");
    expect(first.text == "Hello", "truncated retokenization fixture text mismatch");
    expect(first.generated_token_ids == std::vector<std::uint32_t> {6U},
        "truncated retokenization should retain exact generated ID");
    expect(second.cache_position > first.cache_position,
        "truncated retokenization second turn should reuse resident KV");
    expect(second.metrics.reused_cached_token_count == second.cache_position,
        "truncated retokenization cache metric mismatch");
}

void test_gemma2_chat_session_records_deterministic_turn_metrics_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-metrics", std::span<const std::uint8_t>(bytes));
    std::uint64_t now_ns = 100U;
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {
            .max_new_tokens = 1,
            .clock = [&now_ns]() {
                const auto current = now_ns;
                now_ns += 10U;
                return current;
            },
        }
    );

    std::string streamed;
    const auto first = session.generate_user_turn("Hello", [&](std::string_view delta) {
        streamed += delta;
    });
    const auto second = session.generate_user_turn("Hello");

    expect(streamed == "Hello", "Gemma2 measured turn stream mismatch");
    expect(first.metrics.turn_accepted_ns == 100U, "Gemma2 measured turn accepted boundary mismatch");
    expect(first.metrics.generation.prefill_started_ns == 110U, "Gemma2 measured prefill start mismatch");
    expect(first.metrics.generation.prefill_completed_ns == 120U, "Gemma2 measured prefill completion mismatch");
    expect(first.metrics.generation.first_token_selected_ns == 140U, "Gemma2 measured selection boundary mismatch");
    expect(first.metrics.generation.decode_started_ns == 150U, "Gemma2 measured decode start mismatch");
    expect(first.metrics.generation.decode_completed_ns == 160U, "Gemma2 measured decode completion mismatch");
    expect(first.metrics.generation.generation_completed_ns == 170U, "Gemma2 measured runtime completion mismatch");
    expect(first.metrics.first_visible_output_ns == 180U, "Gemma2 measured visible-output boundary mismatch");
    expect(first.metrics.turn_completed_ns == 190U, "Gemma2 measured turn completion mismatch");
    expect(first.metrics.generation.prefill_duration_ns == 30U, "Gemma2 measured prefill duration mismatch");
    expect(first.metrics.generation.runtime_ttft_ns == 30U, "Gemma2 measured runtime TTFT mismatch");
    expect(first.metrics.generation.decode_duration_ns == 10U, "Gemma2 measured decode duration mismatch");
    expect(first.metrics.end_to_end_ttft_ns == 80U, "Gemma2 measured end-to-end TTFT mismatch");
    expect(first.metrics.total_turn_duration_ns == 90U, "Gemma2 measured total turn duration mismatch");
    expect(first.metrics.generation.newly_evaluated_prompt_token_count == first.appended_token_count,
        "Gemma2 measured prompt token count mismatch");
    expect(first.metrics.generation.generated_token_count == 1U, "Gemma2 measured generated count mismatch");
    expect(first.metrics.generation.decode_token_count == 1U, "Gemma2 measured decode count mismatch");
    expect(first.metrics.generation.stop_reason == Gemma2GenerationStopReason::token_limit,
        "Gemma2 measured token-limit reason mismatch");
    expect(first.metrics.selection_mode == Gemma2TokenSelectionMode::greedy,
        "Gemma2 measured greedy mode mismatch");
    expect(first.metrics.initial_cache_position == 0U && first.metrics.reused_cached_token_count == 0U,
        "Gemma2 measured first-turn cache facts mismatch");
    expect(first.metrics.resident_weight_byte_count > 0U, "Gemma2 measured resident weight bytes missing");
    expect(first.metrics.resident_kv_cache_byte_count > 0U, "Gemma2 measured resident KV bytes missing");
    expect(first.metrics.resident_scratch_byte_count > 0U, "Gemma2 measured resident scratch bytes missing");
    expect(second.metrics.initial_cache_position == first.cached_token_count,
        "Gemma2 measured cached-turn initial position mismatch");
    expect(second.metrics.reused_cached_token_count == second.cache_position,
        "Gemma2 measured reused cache count mismatch");
    expect(second.metrics.generation.newly_evaluated_prompt_token_count == second.appended_token_count,
        "Gemma2 measured cached turn should count only appended prompt tokens");
}

void test_gemma2_chat_session_streams_visible_text_before_return_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-stream-session", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 2}
    );

    bool returned = false;
    std::vector<std::string> chunks;
    const auto turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        expect(!returned, "Gemma2 chat stream callback should run before generate_user_turn returns");
        chunks.push_back(std::string(text_delta));
    });
    returned = true;

    expect(chunks.size() == 2, "Gemma2 chat stream chunk count mismatch");
    expect(chunks[0] == "Hello", "Gemma2 chat first stream chunk mismatch");
    expect(chunks[1] == "Hello", "Gemma2 chat second stream chunk mismatch");
    expect(turn.text == "HelloHello", "Gemma2 chat streamed final text mismatch");
    expect(turn.generated_token_ids.size() == 2, "Gemma2 chat streamed token count mismatch");
}

void test_gemma2_chat_session_can_cancel_generation_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-cancel-session", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    bool should_stop = false;
    std::string streamed;
    const auto turn = session.generate_user_turn(
        "Hello",
        Gemma2ChatTurnOptions {
            .on_text_delta = [&](std::string_view text_delta) {
                streamed += text_delta;
                should_stop = true;
            },
            .should_stop = [&]() {
                return should_stop;
            },
        }
    );

    expect(turn.interrupted, "Gemma2 chat cancelled turn should report interrupted");
    expect(streamed == "Hello", "Gemma2 chat cancelled turn stream mismatch");
    expect(turn.text == streamed, "Gemma2 chat cancelled turn final text mismatch");
    expect(turn.generated_token_ids.size() == 1, "Gemma2 chat cancelled turn token count mismatch");
    expect(turn.metrics.generation.stop_reason == Gemma2GenerationStopReason::interrupted,
        "Gemma2 chat cancelled turn measurement reason mismatch");
}

void test_gemma2_chat_session_stops_on_eos_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(1)
    );
    const TempBinaryFile model("tiny-gemma2-chat-eos-stop", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    std::string streamed;
    const auto turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        streamed += text_delta;
    });

    expect(streamed.empty(), "Gemma2 chat EOS stop should not stream visible text");
    expect(turn.text.empty(), "Gemma2 chat EOS stop text should be empty");
    expect(turn.stop_token_id.has_value() && *turn.stop_token_id == 1U, "Gemma2 chat EOS stop token mismatch");
    expect(turn.generated_token_ids.empty(), "Gemma2 chat EOS stop should not store visible tokens");
    expect(turn.metrics.generation.stop_reason == Gemma2GenerationStopReason::stop_token,
        "Gemma2 chat EOS measurement reason mismatch");
    expect(!turn.metrics.end_to_end_ttft_ns.has_value(),
        "Gemma2 chat invisible EOS should not invent end-to-end TTFT");
}

void test_gemma2_chat_session_stops_on_end_of_turn_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(107)
    );
    const TempBinaryFile model("tiny-gemma2-chat-eot-stop", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    std::string streamed;
    const auto turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        streamed += text_delta;
    });

    expect(streamed.empty(), "Gemma2 chat EOT stop should not stream visible text");
    expect(turn.text.empty(), "Gemma2 chat EOT stop text should be empty");
    expect(turn.stop_token_id.has_value() && *turn.stop_token_id == 107U, "Gemma2 chat EOT stop token mismatch");
    expect(turn.generated_token_ids.empty(), "Gemma2 chat EOT stop should not store visible tokens");
}

void test_gemma2_chat_session_applies_reverse_stop_and_cleans_history_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-reverse-stop", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {
            .max_new_tokens = 3,
            .reverse_stop_strings = {"Hello"},
        }
    );

    std::string streamed;
    const auto first_turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        streamed += text_delta;
    });
    const auto second_turn = session.generate_user_turn("Hello");

    expect(streamed.empty(), "Gemma2 chat reverse stop should not stream the stop string");
    expect(first_turn.text.empty(), "Gemma2 chat reverse stop should trim final text");
    expect(
        first_turn.reverse_stop_string.has_value() && *first_turn.reverse_stop_string == "Hello",
        "Gemma2 chat reverse stop marker mismatch"
    );
    expect(first_turn.generated_token_ids.empty(), "Gemma2 chat reverse stop should not store stop token ids");
    expect(first_turn.metrics.generation.stop_reason == Gemma2GenerationStopReason::reverse_stop,
        "Gemma2 chat reverse-stop measurement reason mismatch");
    expect(!first_turn.metrics.end_to_end_ttft_ns.has_value(),
        "Gemma2 chat fully trimmed reverse stop should not invent end-to-end TTFT");
    expect(second_turn.cache_position > first_turn.cache_position, "Gemma2 chat reverse stop history should continue");
    expect(session.message_count() == 4, "Gemma2 chat reverse stop message count mismatch");
}

void test_gemma2_chat_session_clear_and_reset_restore_empty_context_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-clear-reset", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    const auto first_turn = session.generate_user_turn("Hello");
    expect(session.cached_token_count() > 0U, "Gemma2 chat clear setup should cache tokens");
    session.clear();
    expect(session.cached_token_count() == 0U, "Gemma2 chat clear should reset cached tokens");
    expect(session.message_count() == 0U, "Gemma2 chat clear should reset messages");
    const auto after_clear = session.generate_user_turn("Hello");
    expect(after_clear.cache_position == 0U, "Gemma2 chat after clear should restart at cache zero");
    expect(after_clear.generated_token_ids == first_turn.generated_token_ids, "Gemma2 chat clear generation mismatch");

    session.reset();
    expect(session.cached_token_count() == 0U, "Gemma2 chat reset should reset cached tokens");
    expect(session.message_count() == 0U, "Gemma2 chat reset should reset messages");
    const auto after_reset = session.generate_user_turn("Hello");
    expect(after_reset.cache_position == 0U, "Gemma2 chat after reset should restart at cache zero");
    expect(after_reset.generated_token_ids == first_turn.generated_token_ids, "Gemma2 chat reset generation mismatch");
}

void test_gemma2_chat_session_regenerates_last_turn_without_full_replay_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-regen", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    const auto first_turn = session.generate_user_turn("Hello");
    const auto second_turn = session.generate_user_turn("Hello");
    const auto regenerated = session.regenerate_last_turn();

    expect(regenerated.cache_position == second_turn.cache_position, "Gemma2 chat regen cache rollback mismatch");
    expect(regenerated.cache_position > first_turn.cache_position, "Gemma2 chat regen should retain earlier KV prefix");
    expect(regenerated.generated_token_ids == second_turn.generated_token_ids, "Gemma2 chat greedy regen mismatch");
    expect(session.message_count() == 4U, "Gemma2 chat regen should replace, not append, the last turn");
}

void test_gemma2_chat_session_reports_real_token_budget_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-token-budget", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    expect(session.context_length() == 64U, "Gemma2 chat context length mismatch");
    expect(session.remaining_context_tokens() == 64U, "Gemma2 chat initial remaining tokens mismatch");
    expect(session.max_new_tokens() == 3U, "Gemma2 chat configured max-new-tokens mismatch");
    const auto turn = session.generate_user_turn("Hello");
    expect(
        session.remaining_context_tokens() == session.context_length() - turn.cached_token_count,
        "Gemma2 chat remaining tokens should reflect cached context"
    );
}

void test_gemma2_chat_session_trims_oldest_turns_and_rebuilds_retained_context_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-context-trim", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    bool rebuilt = false;
    for (std::size_t turn_index = 0; turn_index < 8U; ++turn_index) {
        const auto turn = session.generate_user_turn("Hello");
        if (!turn.rebuilt_context) {
            continue;
        }

        rebuilt = true;
        expect(turn.trimmed_turn_count > 0U, "Gemma2 chat overflow should trim complete turns");
        expect(turn.cache_position == 0U, "Gemma2 chat rebuilt context should prefill from cache zero");
        expect(turn.generated_token_ids == std::vector<std::uint32_t> {5U}, "Gemma2 chat rebuilt output mismatch");
        expect(turn.cached_token_count <= session.context_length(), "Gemma2 chat rebuilt cache exceeds context");
        expect(turn.metrics.rebuilt_context, "Gemma2 chat measured rebuild flag mismatch");
        expect(turn.metrics.trimmed_turn_count == turn.trimmed_turn_count,
            "Gemma2 chat measured trimmed-turn count mismatch");
        expect(turn.metrics.initial_cache_position == 0U && turn.metrics.reused_cached_token_count == 0U,
            "Gemma2 rebuilt measurement should not report reused stale KV");
        break;
    }

    expect(rebuilt, "Gemma2 chat long conversation should trigger context rebuild");
}

void test_gemma2_chat_session_rejects_impossible_turn_without_mutating_context_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-context-reject", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );
    const auto first_turn = session.generate_user_turn("Hello");

    bool rejected = false;
    try {
        static_cast<void>(session.generate_user_turn(std::string(128U, 'x')));
    } catch (const Gemma2ChatContextError& error) {
        rejected = std::string_view(error.what()).find("use a shorter prompt or lower --max-new-tokens")
            != std::string_view::npos;
    }

    expect(rejected, "Gemma2 chat impossible turn should provide recovery guidance");
    expect(session.cached_token_count() == first_turn.cached_token_count, "rejected turn should preserve cached context");
    expect(session.message_count() == 2U, "rejected turn should preserve message history");
    const auto next_turn = session.generate_user_turn("Hello");
    expect(next_turn.cache_position == first_turn.cached_token_count, "turn after rejection should keep append position");
}

void test_gemma2_chat_session_seeded_sampling_and_regen_are_reproducible_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_sampled_tokens()
    );
    const TempBinaryFile first_model("tiny-gemma2-chat-sampled-a", std::span<const std::uint8_t>(bytes));
    const TempBinaryFile second_model("tiny-gemma2-chat-sampled-b", std::span<const std::uint8_t>(bytes));
    std::uint64_t now_ns = 1000U;
    const auto options = Gemma2ChatSessionOptions {
        .max_new_tokens = 3,
        .sampler = TokenSamplerOptions {
            .temperature = 1.0F,
            .top_k = 2,
            .top_p = 1.0F,
            .repetition_penalty = 1.1F,
            .seed = 42,
        },
        .clock = [&now_ns]() {
            const auto current = now_ns;
            now_ns += 10U;
            return current;
        },
    };
    auto first_session = Gemma2ChatSession(read_file_backed_gguf_model(first_model.path()), options);
    auto second_session = Gemma2ChatSession(read_file_backed_gguf_model(second_model.path()), options);

    const auto first = first_session.generate_user_turn("Hello");
    const auto same_seed = second_session.generate_user_turn("Hello");
    const auto regenerated = first_session.regenerate_last_turn();

    expect(first.generated_token_ids == same_seed.generated_token_ids, "seeded sampled sessions mismatch");
    expect(first.generated_token_ids == regenerated.generated_token_ids, "sampled /regen should replay sampler state");
    expect(first.metrics.selection_mode == Gemma2TokenSelectionMode::sampled,
        "sampled turn measurement mode mismatch");
    expect(first.metrics.generation.prefill_duration_ns == 30U,
        "sampled turn deterministic prefill timing mismatch");
    expect(first.metrics.generation.runtime_ttft_ns == 30U,
        "sampled turn deterministic runtime TTFT mismatch");
    expect(first.metrics.generation.decode_duration_ns == 50U,
        "sampled turn deterministic selection/decode timing mismatch");
    expect(first.metrics.generation.decode_token_count == 3U,
        "sampled turn decode count mismatch");
    expect(first_session.message_count() == 2U, "sampled /regen should replace the previous response");
}

void test_generation_benchmark_runner_reuses_one_plan_and_records_all_cases_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(128),
        tiny_gemma2_generate_cli_tensors_for_sampled_tokens()
    );
    const TempBinaryFile model("tiny-gemma2-benchmark-runner", std::span<const std::uint8_t>(bytes));
    std::uint64_t now_ns = 100U;
    const auto result = run_generation_benchmark(GenerationBenchmarkRunnerOptions {
        .model_path = model.path().string(),
        .prompt = "Hello",
        .cached_second_prompt = "Hello",
        .max_new_tokens = 1U,
        .warmup_count = 1U,
        .repetition_count = 2U,
        .sampled_sampler = TokenSamplerOptions {
            .temperature = 1.0F,
            .top_k = 2U,
            .top_p = 1.0F,
            .repetition_penalty = 1.0F,
            .seed = 42U,
        },
        .clock = [&now_ns]() {
            const auto current = now_ns;
            now_ns += 10U;
            return current;
        },
    });

    expect(result.setup.model_load_duration_ns == 10U, "benchmark runner load timing mismatch");
    expect(result.setup.plan_creation_duration_ns == 10U, "benchmark runner plan timing mismatch");
    expect(result.setup.resident_weight_byte_count > 0U, "benchmark runner resident weights missing");
    expect(result.setup.resident_kv_cache_byte_count > 0U, "benchmark runner resident KV missing");
    expect(result.setup.resident_scratch_byte_count > 0U, "benchmark runner resident scratch missing");
    expect(result.repetitions.size() == 10U, "benchmark runner repetition record count mismatch");
    expect(result.summaries.size() == 4U, "benchmark runner summary count mismatch");

    std::size_t cold_count = 0;
    std::size_t warmup_count = 0;
    std::vector<const local_gguf_engine::benchmarks::GenerationBenchmarkRepetitionRecord*> sampled_measured;
    for (const auto& record : result.repetitions) {
        expect(record.setup.model_label == model.path().string(), "benchmark runner model label mismatch");
        expect(record.setup.resident_weight_byte_count == result.setup.resident_weight_byte_count,
            "benchmark runner should share one resident setup record");
        if (record.benchmark_case == GenerationBenchmarkCase::cold_first_turn) {
            ++cold_count;
        }
        if (record.warmup) {
            ++warmup_count;
        }
        if (record.benchmark_case == GenerationBenchmarkCase::warm_first_turn) {
            expect(record.turn.initial_cache_position == 0U,
                "benchmark warm-first record should start at cache zero");
        }
        if (record.benchmark_case == GenerationBenchmarkCase::cached_second_turn) {
            expect(record.turn.initial_cache_position > 0U && record.turn.reused_cached_token_count > 0U,
                "benchmark cached-second record should reuse resident KV");
            expect(record.turn.generation.newly_evaluated_prompt_token_count > 0U,
                "benchmark cached-second record should evaluate an appended suffix");
        }
        if (record.benchmark_case == GenerationBenchmarkCase::seeded_sampled_turn) {
            expect(record.turn.selection_mode == Gemma2TokenSelectionMode::sampled,
                "benchmark sampled case mode mismatch");
            expect(record.sampler.seed == 42U, "benchmark sampled case seed mismatch");
            if (!record.warmup) {
                sampled_measured.push_back(&record);
            }
        }
    }
    expect(cold_count == 1U, "benchmark runner should record one true cold first turn");
    expect(warmup_count == 3U, "benchmark runner warmup count mismatch");
    expect(sampled_measured.size() == 2U, "benchmark sampled measured count mismatch");
    expect(sampled_measured[0]->generated_token_ids == sampled_measured[1]->generated_token_ids,
        "benchmark seeded sampled repetitions should reproduce token ids");
    expect(result.summaries[0].measured_repetitions.size() == 1U,
        "benchmark cold summary count mismatch");
    for (std::size_t index = 1U; index < result.summaries.size(); ++index) {
        expect(result.summaries[index].measured_repetitions.size() == 2U,
            "benchmark repeatable case summary should exclude warmup");
    }
}

void test_runs_generation_benchmark_human_and_jsonl_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(128),
        tiny_gemma2_generate_cli_tensors_for_sampled_tokens()
    );
    const TempBinaryFile model("tiny-gemma2-benchmark-cli", std::span<const std::uint8_t>(bytes));

    std::ostringstream human;
    std::ostringstream human_error;
    const auto human_exit = run_generate_with_extra_arguments(
        model.path(), {"--bench", "--warmup", "0", "--repetitions", "1", "--output", "human"},
        human, human_error
    );
    expect(human_exit == EXIT_SUCCESS && human_error.str().empty(), "human benchmark CLI failed");
    const auto human_text = human.str();
    expect(human_text.starts_with("benchmark schema=1"), "human benchmark setup missing");
    for (const auto case_name : {"cold_first_turn", "warm_first_turn", "cached_second_turn", "seeded_sampled_turn"}) {
        expect(human_text.find(case_name) != std::string::npos, "human benchmark case missing");
    }
    expect(human_text.find("mean=") != std::string::npos
        && human_text.find("median=") != std::string::npos
        && human_text.find("stddev=") != std::string::npos,
        "human benchmark statistics missing");

    std::ostringstream jsonl;
    std::ostringstream jsonl_error;
    const auto jsonl_exit = run_generate_with_extra_arguments(
        model.path(), {"--bench", "--warmup", "0", "--repetitions", "1", "--output", "jsonl"},
        jsonl, jsonl_error
    );
    expect(jsonl_exit == EXIT_SUCCESS && jsonl_error.str().empty(), "JSONL benchmark CLI failed");
    std::istringstream lines(jsonl.str());
    std::string line;
    std::size_t line_count = 0;
    while (std::getline(lines, line)) {
        ++line_count;
        expect(looks_like_strict_json_object(line), "benchmark JSONL line is not a strict object");
        expect(line.find("\"schema_version\":1") != std::string::npos, "benchmark JSONL schema missing");
        expect(line.find("\"units\":") != std::string::npos, "benchmark JSONL units missing");
    }
    expect(line_count == 9U, "benchmark JSONL setup/repetition/summary line count mismatch");
    expect(jsonl.str().find("benchmark schema=") == std::string::npos, "JSONL stdout contains human banner");
    expect(jsonl.str().find("\"record_kind\":\"setup\"") != std::string::npos, "JSONL setup missing");
    expect(jsonl.str().find("\"record_kind\":\"repetition\"") != std::string::npos, "JSONL repetition missing");
    expect(jsonl.str().find("\"record_kind\":\"summary\"") != std::string::npos, "JSONL summary missing");
}

void test_rejects_invalid_generation_benchmark_cli_options_before_loading() {
    const auto missing = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-benchmark.gguf";
    const std::vector<std::pair<std::vector<std::string>, std::string_view>> cases {
        {{"--warmup", "1"}, "benchmark options require --bench"},
        {{"--bench", "--warmup", "-1"}, "--warmup requires a nonnegative integer"},
        {{"--bench", "--repetitions", "0"}, "--repetitions requires a positive integer"},
        {{"--bench", "--output", "csv"}, "--output must be human or jsonl"},
        {{"--bench", "--backend", "cpu"}, "requires Metal backend"},
        {{"--bench", "--temperature", "0"}, "requires positive temperature"},
    };
    for (const auto& [arguments, expected] : cases) {
        std::ostringstream output;
        std::ostringstream error;
        const auto exit_code = run_generate_with_extra_arguments(missing, arguments, output, error);
        expect(exit_code == EXIT_FAILURE, "invalid benchmark CLI options should fail");
        expect(output.str().empty(), "invalid benchmark CLI options should not write stdout");
        expect(error.str().find(expected) != std::string::npos, "invalid benchmark CLI error mismatch");
    }

    std::ostringstream output;
    std::ostringstream error;
    const auto mixed_exit = run_generate_chat_with_extra_arguments(missing, {"--bench"}, output, error);
    expect(mixed_exit == EXIT_FAILURE && output.str().empty(), "benchmark/chat mixing should fail");
    expect(error.str().find("--bench cannot be combined with --chat") != std::string::npos,
        "benchmark/chat mixing error mismatch");
}

void test_runs_generate_gemma2_chat_loop_until_exit_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-cli", std::span<const std::uint8_t>(bytes));
    std::istringstream input("Hello\nHello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "1", input, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 chat CLI exit code mismatch");
    expect(
        without_chat_timing_lines(output.str()) == "local-gguf-engine chat\ncommands: /exit /quit /clear /reset /regen /tokens /read <path>\nmultiline: end a line with \\\n> \n> \n> ",
        "generate Gemma2 chat CLI output mismatch"
    );
    expect(chat_timing_line_count(output.str()) == 2U, "generate Gemma2 chat timing line count mismatch");
    expect(error.str().empty(), "generate Gemma2 chat CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_streams_visible_tokens_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-cli-stream", std::span<const std::uint8_t>(bytes));
    std::istringstream input("Hello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "2", input, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 chat stream CLI exit code mismatch");
    expect(
        without_chat_timing_lines(output.str()) == "local-gguf-engine chat\ncommands: /exit /quit /clear /reset /regen /tokens /read <path>\nmultiline: end a line with \\\n> HelloHello\n> ",
        "generate Gemma2 chat stream CLI output mismatch"
    );
    expect(chat_timing_line_count(output.str()) == 1U, "streamed chat timing line count mismatch");
    expect(error.str().empty(), "generate Gemma2 chat stream CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_reverse_stop_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-cli-reverse-stop", std::span<const std::uint8_t>(bytes));
    std::istringstream input("Hello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path_with_reverse_stop(
        model.path(),
        "3",
        "Hello",
        input,
        output,
        error
    );

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 chat reverse-stop CLI exit code mismatch");
    expect(
        without_chat_timing_lines(output.str()) == "local-gguf-engine chat\ncommands: /exit /quit /clear /reset /regen /tokens /read <path>\nmultiline: end a line with \\\n> \n> ",
        "generate Gemma2 chat reverse-stop CLI output mismatch"
    );
    expect(chat_timing_line_count(output.str()) == 1U, "reverse-stop chat timing line count mismatch");
    expect(error.str().empty(), "generate Gemma2 chat reverse-stop CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_gate3_commands_and_input_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(128),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-gate3-cli", std::span<const std::uint8_t>(bytes));
    const std::vector<std::uint8_t> prompt_bytes {'H', 'e', 'l', 'l', 'o'};
    const TempBinaryFile prompt_file("tiny-gemma2-chat-read-prompt", std::span<const std::uint8_t>(prompt_bytes));
    std::istringstream input(
        "/tokens\n/regen\n/bogus\n/read /definitely/missing/local-gguf-prompt\n"
        "Hello\\\nHello\n/clear\nHello\n/reset\n/read " + prompt_file.path().string()
        + "\n/regen\n/quit\n"
    );
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "1", input, output, error);
    const auto text = output.str();

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 Gate 3 CLI exit code mismatch");
    expect(text.find("tokens: cached=0 remaining=128 max-new-tokens=1") != std::string::npos, "/tokens output mismatch");
    expect(text.find("error: nothing to regenerate") != std::string::npos, "empty /regen output mismatch");
    expect(text.find("error: unknown command '/bogus'") != std::string::npos, "invalid command output mismatch");
    expect(text.find("error: could not read '/definitely/missing/local-gguf-prompt'") != std::string::npos, "missing /read output mismatch");
    expect(text.find("... Hello") != std::string::npos, "multiline continuation/output mismatch");
    expect(text.find("chat cleared") != std::string::npos, "/clear output mismatch");
    expect(text.find("session reset") != std::string::npos, "/reset output mismatch");
    std::size_t generated_hello_count = 0;
    for (std::size_t position = 0; (position = text.find("Hello", position)) != std::string::npos; position += 5U) {
        ++generated_hello_count;
    }
    expect(generated_hello_count == 4U, "Gate 3 multiline, /read, and /regen generation count mismatch");
    expect(chat_timing_line_count(text) == 4U, "Gate 3 successful turns should each print one timing line");
    expect(error.str().empty(), "generate Gemma2 Gate 3 CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_context_trim_and_recovers_from_impossible_turn_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-context-cli", std::span<const std::uint8_t>(bytes));
    std::istringstream input(
        "Hello\nHello\nHello\nHello\n"
        + std::string(128U, 'x') + "\n/tokens\n/exit\n"
    );
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "1", input, output, error);
    const auto text = output.str();

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 context CLI should survive turn rejection");
    expect(text.find("context: dropped ") != std::string::npos, "context trim notice missing");
    expect(text.find("oldest turn(s) and rebuilt retained KV") != std::string::npos, "context rebuild notice missing");
    expect(text.find("error: context: turn requires ") != std::string::npos, "context rejection prefix missing");
    expect(text.find("use a shorter prompt or lower --max-new-tokens") != std::string::npos, "context recovery guidance missing");
    expect(text.find("tokens: cached=") != std::string::npos, "REPL should continue after context rejection");
    expect(chat_timing_line_count(text) == 4U, "context rejection should not print an extra timing line");
    const auto timing_before_context = text.rfind("[ Prompt: ", text.find("context: dropped "));
    expect(timing_before_context != std::string::npos, "context rebuild timing should precede rebuild notice");
    expect(error.str().empty(), "generate Gemma2 context CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_seeded_sampling_flags_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_sampled_tokens()
    );
    const TempBinaryFile model("tiny-gemma2-chat-sampled-cli", std::span<const std::uint8_t>(bytes));
    std::istringstream first_input("Hello\n/regen\n/exit\n");
    std::istringstream second_input("Hello\n/regen\n/exit\n");
    std::ostringstream first_output;
    std::ostringstream second_output;
    std::ostringstream first_error;
    std::ostringstream second_error;

    const auto first_exit = run_generate_chat_for_path_with_sampling(
        model.path(), first_input, first_output, first_error
    );
    const auto second_exit = run_generate_chat_for_path_with_sampling(
        model.path(), second_input, second_output, second_error
    );

    expect(first_exit == EXIT_SUCCESS && second_exit == EXIT_SUCCESS, "sampled chat CLI exit mismatch");
    expect(
        without_chat_timing_lines(first_output.str()) == without_chat_timing_lines(second_output.str()),
        "seeded sampled CLI transcript mismatch"
    );
    expect(chat_timing_line_count(first_output.str()) == 2U, "sampled turn and /regen timing line count mismatch");
    expect(first_error.str().empty() && second_error.str().empty(), "sampled chat CLI should not write stderr");
}

void test_rejects_generate_gemma2_chat_invalid_sampling_flags() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-sampling.gguf";
    struct InvalidCase {
        std::vector<std::string> arguments;
        std::string_view expected_error;
    };
    const std::vector<InvalidCase> cases {
        {{"--temperature", "bad"}, "--temperature requires a finite number"},
        {{"--temperature", "1", "--top-k", "0"}, "--top-k must be a positive integer"},
        {{"--temperature", "1", "--top-p", "2"}, "--top-p must be in (0, 1]"},
        {{"--temperature", "1", "--repeat-penalty", "0"}, "--repeat-penalty must be positive"},
        {{"--temperature", "1", "--seed", "-1"}, "--seed must be an unsigned integer"},
        {{"--greedy", "--temperature", "1"}, "--greedy cannot be combined"},
        {{"--seed", "42"}, "sampling modifiers require --temperature greater than zero"},
    };

    for (const auto& test_case : cases) {
        std::ostringstream output;
        std::ostringstream error;
        const auto exit_code = run_generate_chat_with_extra_arguments(
            path, test_case.arguments, output, error
        );
        expect(exit_code == EXIT_FAILURE, "invalid sampling flags should fail");
        expect(output.str().empty(), "invalid sampling flags should not write stdout");
        expect(error.str().find(test_case.expected_error) != std::string::npos, "invalid sampling error mismatch");
    }
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
        "formats chat timing line from runtime metrics",
        test_formats_chat_timing_line_from_runtime_metrics,
    },
    {
        "runs generate Gemma2 default greedy when available",
        test_runs_generate_gemma2_default_greedy_when_available,
    },
    {
        "runs generate Gemma2 greedy when available",
        test_runs_generate_gemma2_greedy_when_available,
    },
    {
        "Gemma2 chat session appends second turn at nonzero cache when available",
        test_gemma2_chat_session_appends_second_turn_at_nonzero_cache_when_available,
    },
    {
        "Gemma2 chat session appends after truncated output retokenizes differently when available",
        test_gemma2_chat_session_appends_after_truncated_output_retokenizes_differently_when_available,
    },
    {
        "Gemma2 chat session records deterministic turn metrics when available",
        test_gemma2_chat_session_records_deterministic_turn_metrics_when_available,
    },
    {
        "Gemma2 chat session streams visible text before return when available",
        test_gemma2_chat_session_streams_visible_text_before_return_when_available,
    },
    {
        "Gemma2 chat session can cancel generation when available",
        test_gemma2_chat_session_can_cancel_generation_when_available,
    },
    {
        "Gemma2 chat session stops on EOS when available",
        test_gemma2_chat_session_stops_on_eos_when_available,
    },
    {
        "Gemma2 chat session stops on end-of-turn when available",
        test_gemma2_chat_session_stops_on_end_of_turn_when_available,
    },
    {
        "Gemma2 chat session applies reverse stop and cleans history when available",
        test_gemma2_chat_session_applies_reverse_stop_and_cleans_history_when_available,
    },
    {
        "Gemma2 chat session clear and reset restore empty context when available",
        test_gemma2_chat_session_clear_and_reset_restore_empty_context_when_available,
    },
    {
        "Gemma2 chat session regenerates last turn without full replay when available",
        test_gemma2_chat_session_regenerates_last_turn_without_full_replay_when_available,
    },
    {
        "Gemma2 chat session reports real token budget when available",
        test_gemma2_chat_session_reports_real_token_budget_when_available,
    },
    {
        "Gemma2 chat session trims oldest turns and rebuilds retained context when available",
        test_gemma2_chat_session_trims_oldest_turns_and_rebuilds_retained_context_when_available,
    },
    {
        "Gemma2 chat session rejects impossible turn without mutating context when available",
        test_gemma2_chat_session_rejects_impossible_turn_without_mutating_context_when_available,
    },
    {
        "Gemma2 chat session seeded sampling and regen are reproducible when available",
        test_gemma2_chat_session_seeded_sampling_and_regen_are_reproducible_when_available,
    },
    {
        "generation benchmark runner reuses one plan and records all cases when available",
        test_generation_benchmark_runner_reuses_one_plan_and_records_all_cases_when_available,
    },
    {
        "runs generation benchmark human and JSONL when available",
        test_runs_generation_benchmark_human_and_jsonl_when_available,
    },
    {
        "rejects invalid generation benchmark CLI options before loading",
        test_rejects_invalid_generation_benchmark_cli_options_before_loading,
    },
    {
        "runs generate Gemma2 chat loop until exit when available",
        test_runs_generate_gemma2_chat_loop_until_exit_when_available,
    },
    {
        "runs generate Gemma2 chat streams visible tokens when available",
        test_runs_generate_gemma2_chat_streams_visible_tokens_when_available,
    },
    {
        "runs generate Gemma2 chat reverse stop when available",
        test_runs_generate_gemma2_chat_reverse_stop_when_available,
    },
    {
        "runs generate Gemma2 chat Gate 3 commands and input when available",
        test_runs_generate_gemma2_chat_gate3_commands_and_input_when_available,
    },
    {
        "runs generate Gemma2 chat context trim and recovers from impossible turn when available",
        test_runs_generate_gemma2_chat_context_trim_and_recovers_from_impossible_turn_when_available,
    },
    {
        "runs generate Gemma2 chat seeded sampling flags when available",
        test_runs_generate_gemma2_chat_seeded_sampling_flags_when_available,
    },
    {
        "rejects generate Gemma2 chat invalid sampling flags",
        test_rejects_generate_gemma2_chat_invalid_sampling_flags,
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
