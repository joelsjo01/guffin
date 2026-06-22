#include "generate_cli_tests.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/frontends/cli/generate.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::ModelArchitecture;
using local_gguf_engine::frontends::cli::GenerateOptions;
using local_gguf_engine::frontends::cli::generate_text_from_gguf_bytes;
using local_gguf_engine::frontends::cli::run_generate;
using local_gguf_engine::model_format::GGUFMetadataArray;
using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::model_format::GGUFMetadataValue;
using local_gguf_engine::model_format::GGUFMetadataValueType;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::metadata_bool;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_f32_array;
using local_gguf_engine::tests::metadata_i32_array;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_string_array;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::gemma2_chat_template_fixture;
using local_gguf_engine::tests::smollm2_chat_template_fixture;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
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

        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
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
        case GGUFMetadataValueType::float32: {
            const auto& values = std::get<std::vector<float>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_f32(bytes, value);
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

std::vector<TinyGGUFTensor> tiny_gguf_tensors_from_engine_model(const EngineModel& model) {
    std::vector<TinyGGUFTensor> tensors;
    tensors.reserve(model.tensors().size());

    for (const auto& tensor : model.tensors()) {
        const auto dimensions = tensor.storage.descriptor().shape.dimensions();
        tensors.push_back(TinyGGUFTensor {
            .name = tensor.name,
            .dimensions = std::vector<std::uint64_t>(dimensions.begin(), dimensions.end()),
            .type_id = gguf_type_id_for_dtype(tensor.storage.descriptor().dtype),
            .bytes = std::vector<std::uint8_t>(tensor.storage.bytes().begin(), tensor.storage.bytes().end()),
        });
    }

    return tensors;
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

EngineTensor engine_tensor(std::string name, TensorStorage storage) {
    return EngineTensor {
        .name = std::move(name),
        .storage = std::move(storage),
    };
}

[[nodiscard]] std::string llama_block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

LlamaModelConfig tiny_generate_config() {
    return LlamaModelConfig {
        .context_length = 8,
        .embedding_length = 2,
        .block_count = 1,
        .feed_forward_length = 2,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .attention_key_length = 2,
        .attention_value_length = 2,
        .rope_dimension_count = 2,
        .vocab_size = 3,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 3.5F,
    };
}

void add_zero_llama_block_tensors(std::vector<EngineTensor>& tensors, std::uint32_t block_index) {
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> projection_values {
        0.0F, 0.0F,
        0.0F, 0.0F,
    };

    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_q.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_k.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_v.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_output.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_norm.weight"),
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_gate.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_up.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_down.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
}

[[nodiscard]] EngineModel tiny_bound_llama_model(LlamaModelConfig config, std::vector<EngineTensor> tensors) {
    return EngineModel(ModelArchitecture::llama, {}, config, std::move(tensors));
}

EngineModel tiny_argmax_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_zero_llama_block_tensors(tensors, 0);
    return tiny_bound_llama_model(config, std::move(tensors));
}

EngineModel tiny_tied_output_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    add_zero_llama_block_tensors(tensors, 0);
    return tiny_bound_llama_model(config, std::move(tensors));
}

EngineModel tiny_eos_then_visible_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        0.0F, 2.0F, 0.0F,
        0.0F, 0.0F, 2.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_zero_llama_block_tensors(tensors, 0);
    return tiny_bound_llama_model(config, std::move(tensors));
}

EngineModel tiny_chat_template_generation_model(LlamaModelConfig config) {
    std::vector<float> embedding_values(static_cast<std::size_t>(config.embedding_length) * config.vocab_size, 0.0F);
    std::vector<float> output_values(static_cast<std::size_t>(config.embedding_length) * config.vocab_size, 0.0F);
    for (std::uint32_t token = 0; token < config.vocab_size; ++token) {
        embedding_values[token] = 1.0F;
    }
    output_values[7] = 2.0F;

    const std::array<float, 2> norm_values {1.0F, 1.0F};
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(config.embedding_length, config.vocab_size, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(config.embedding_length, config.vocab_size, std::span<const float>(output_values))
    ));
    add_zero_llama_block_tensors(tensors, 0);
    return tiny_bound_llama_model(config, std::move(tensors));
}

std::vector<GGUFMetadataEntry> tiny_generate_cli_metadata(const LlamaModelConfig& config) {
    return {
        metadata_string("general.architecture", "llama"),
        metadata_string("general.name", "Tiny Generate Fixture"),
        metadata_u32("llama.context_length", config.context_length),
        metadata_u32("llama.embedding_length", config.embedding_length),
        metadata_u32("llama.block_count", config.block_count),
        metadata_u32("llama.feed_forward_length", config.feed_forward_length),
        metadata_u32("llama.attention.head_count", config.attention_head_count),
        metadata_u32("llama.attention.head_count_kv", config.attention_head_count_kv),
        metadata_u32("llama.attention.key_length", config.attention_key_length),
        metadata_u32("llama.attention.value_length", config.attention_value_length),
        metadata_u32("llama.rope.dimension_count", config.rope_dimension_count),
        metadata_u32("llama.vocab_size", config.vocab_size),
        metadata_f32("llama.rope.freq_base", config.rope_freq_base),
        metadata_f32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_epsilon),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array("tokenizer.ggml.tokens", {"<s>", "Hello", "!"}),
        metadata_f32_array("tokenizer.ggml.scores", {0.0F, -1.0F, -0.5F}),
        metadata_i32_array("tokenizer.ggml.token_type", {3, 1, 1}),
        metadata_u32("tokenizer.ggml.bos_token_id", 0),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
    };
}

std::vector<GGUFMetadataEntry> tiny_generate_cli_chat_metadata(const LlamaModelConfig& config) {
    return {
        metadata_string("general.architecture", "llama"),
        metadata_string("general.name", "Tiny Chat Generate Fixture"),
        metadata_u32("llama.context_length", config.context_length),
        metadata_u32("llama.embedding_length", config.embedding_length),
        metadata_u32("llama.block_count", config.block_count),
        metadata_u32("llama.feed_forward_length", config.feed_forward_length),
        metadata_u32("llama.attention.head_count", config.attention_head_count),
        metadata_u32("llama.attention.head_count_kv", config.attention_head_count_kv),
        metadata_u32("llama.attention.key_length", config.attention_key_length),
        metadata_u32("llama.attention.value_length", config.attention_value_length),
        metadata_u32("llama.rope.dimension_count", config.rope_dimension_count),
        metadata_u32("llama.vocab_size", config.vocab_size),
        metadata_f32("llama.rope.freq_base", config.rope_freq_base),
        metadata_f32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_epsilon),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array(
            "tokenizer.ggml.tokens",
            {
                "<pad>",
                "<|im_start|>",
                "<|im_end|>",
                "system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face",
                "\n",
                "user\nHello",
                "assistant\n",
                "!",
            }
        ),
        metadata_i32_array(
            "tokenizer.ggml.token_type",
            {1, 3, 3, 1, 1, 1, 1, 1}
        ),
        metadata_u32("tokenizer.ggml.bos_token_id", 1),
        metadata_u32("tokenizer.ggml.eos_token_id", 2),
        metadata_bool("tokenizer.ggml.add_bos_token", false),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
        metadata_string("tokenizer.chat_template", smollm2_chat_template_fixture()),
    };
}

std::vector<GGUFMetadataEntry> tiny_generate_cli_eos_metadata(const LlamaModelConfig& config) {
    auto metadata = tiny_generate_cli_metadata(config);
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    return metadata;
}

std::vector<std::uint8_t> tiny_generate_cli_gguf_bytes() {
    auto config = tiny_generate_config();
    const auto model = tiny_argmax_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

std::vector<std::uint8_t> tiny_tied_output_generate_cli_gguf_bytes() {
    auto config = tiny_generate_config();
    const auto model = tiny_tied_output_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

std::vector<std::uint8_t> tiny_eos_generate_cli_gguf_bytes() {
    auto config = tiny_generate_config();
    const auto model = tiny_eos_then_visible_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_eos_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

std::vector<std::uint8_t> tiny_chat_template_generate_cli_gguf_bytes() {
    auto config = tiny_generate_config();
    config.context_length = 16;
    config.vocab_size = 8;
    const auto model = tiny_chat_template_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_chat_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
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

std::vector<GGUFMetadataEntry> tiny_gemma2_generate_cli_metadata() {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Generate Fixture"),
        metadata_u32("gemma2.context_length", 16),
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
    tensors.push_back(tiny_zero_tensor(
        gemma2_block_tensor_name(0, "post_attention_norm.weight"),
        {256},
        DType::f32
    ));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_q.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_k.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_v.weight"), {256, 256}, DType::q6_k));
    tensors.push_back(tiny_zero_tensor(
        gemma2_block_tensor_name(0, "attn_output.weight"),
        {256, 256},
        DType::q4_k
    ));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "post_ffw_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_gate.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_up.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_down.weight"), {256, 256}, DType::q6_k));
    return tensors;
}

std::vector<std::uint8_t> tiny_gemma2_generate_cli_gguf_bytes() {
    return write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(),
        tiny_gemma2_generate_cli_tensors()
    );
}

void test_generates_text_from_gguf_bytes_with_cached_argmax() {
    const auto bytes = tiny_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2);

    expect(text == "!", "generate text should decode newly generated non-special tokens");
}

void test_generates_text_from_gguf_bytes_with_tied_output_weight() {
    const auto bytes = tiny_tied_output_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2);

    expect(text == "!!", "generate text should support tied output weights");
}

void test_generates_text_from_gguf_bytes_stops_at_tokenizer_eos() {
    const auto bytes = tiny_eos_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2);

    expect(text.empty(), "generate text should stop before visible token after EOS");
}

void test_generates_text_from_gguf_bytes_with_chat_template_prompt() {
    const auto bytes = tiny_chat_template_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "Hello", 1);

    expect(text == "!", "generate text should apply tokenizer chat template before encoding");
}

void test_generates_text_from_gguf_bytes_with_sampling_options() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const GenerateOptions options {
        .sampling = local_gguf_engine::engine::SamplingConfig {
            .temperature = 10.0F,
            .top_k = 2,
            .seed = 0,
        },
    };

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 1, options);

    expect(text == "Hello", "generate text should use seeded temperature/top-k sampling options");
}

void test_generates_text_from_gguf_bytes_with_cpu_backend_option() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const GenerateOptions options {
        .sampling = local_gguf_engine::engine::SamplingConfig {},
        .backend = BackendKind::cpu,
    };

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2, options);

    expect(text == "!", "generate text CPU backend option should preserve cached argmax output");
}

void test_rejects_generate_gemma2_runtime_not_implemented() {
    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 unsupported runtime exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 unsupported runtime should not write stdout");
    expect(
        error.str().find("Gemma 2 generation runtime is not implemented yet") != std::string::npos,
        "generate Gemma2 unsupported runtime error missing"
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
    expect(
        error.str().find("Gemma 2 generation runtime is not implemented yet") == std::string::npos,
        "generate Gemma2 unsupported template should fail before runtime"
    );
}

void test_runs_generate_success_path() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "", "2", output, error);

    expect(exit_code == EXIT_SUCCESS, "generate success exit code mismatch");
    expect(output.str() == "!!\n", "generate success output mismatch");
    expect(error.str().empty(), "generate success should not write stderr");
}

void test_runs_generate_with_greedy_option() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-greedy", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "2",
        "--greedy",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(5, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate greedy success exit code mismatch");
    expect(output.str() == "!\n", "generate greedy output mismatch");
    expect(error.str().empty(), "generate greedy should not write stderr");
}

void test_runs_generate_with_cpu_backend_option() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-backend-cpu", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "2",
        "--greedy",
        "--backend",
        "cpu",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate CPU backend success exit code mismatch");
    expect(output.str() == "!\n", "generate CPU backend output mismatch");
    expect(error.str().empty(), "generate CPU backend should not write stderr");
}

void test_runs_generate_with_metal_backend_option_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-backend-metal", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "2",
        "--greedy",
        "--backend",
        "metal",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Metal backend success exit code mismatch");
    expect(output.str() == "!\n", "generate Metal backend output mismatch");
    expect(error.str().empty(), "generate Metal backend should not write stderr");
}

void test_runs_generate_with_sampling_options() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-sampling", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--temperature",
        "10",
        "--top-k",
        "2",
        "--top-p",
        "1",
        "--repeat-penalty",
        "1",
        "--seed",
        "0",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(14, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate sampling success exit code mismatch");
    expect(output.str() == "Hello\n", "generate sampling success output mismatch");
    expect(error.str().empty(), "generate sampling success should not write stderr");
}

void test_runs_generate_with_top_p_option() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-top-p", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--temperature",
        "10",
        "--top-p",
        "0.01",
        "--seed",
        "0",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(10, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate top-p success exit code mismatch");
    expect(output.str() == "!\n", "generate top-p output mismatch");
    expect(error.str().empty(), "generate top-p should not write stderr");
}

void test_runs_generate_with_first_token_debug() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-debug", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--debug-first-token",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(5, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate debug success exit code mismatch");
    expect(output.str() == "!\n", "generate debug output mismatch");
    expect(error.str().find("debug: formatted_prompt=\"\"") != std::string::npos, "generate debug prompt missing");
    expect(
        error.str().find("debug: encode_special_tokens=false") != std::string::npos,
        "generate debug special-token flag missing"
    );
    expect(error.str().find("debug: prompt_token_ids=[0]") != std::string::npos, "generate debug token IDs missing");
    expect(
        error.str().find("debug:   token=2") != std::string::npos,
        "generate debug top token missing"
    );
    expect(
        error.str().find("text=\"!\"") != std::string::npos,
        "generate debug decoded top token missing"
    );
}

void test_reports_generate_metal_backend_unavailable_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-backend-metal-missing", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "2",
        "--greedy",
        "--backend",
        "metal",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate unavailable Metal backend exit code mismatch");
    expect(output.str().empty(), "generate unavailable Metal backend should not write stdout");
    expect(
        error.str().find("error: backend:") != std::string::npos,
        "generate unavailable Metal backend error prefix missing"
    );
    expect(
        error.str().find("Metal backend unavailable") != std::string::npos,
        "generate unavailable Metal backend message missing"
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

void test_rejects_generate_invalid_temperature_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--temperature",
        "-1",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-temperature exit code mismatch");
    expect(output.str().empty(), "generate invalid-temperature should not write stdout");
    expect(
        error.str().find("--temperature must be a nonnegative finite number") != std::string::npos,
        "generate invalid-temperature error missing"
    );
}

void test_rejects_generate_invalid_top_p_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--top-p",
        "0",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-top-p exit code mismatch");
    expect(output.str().empty(), "generate invalid-top-p should not write stdout");
    expect(
        error.str().find("--top-p must be in (0, 1]") != std::string::npos,
        "generate invalid-top-p error missing"
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

void test_rejects_generate_invalid_repeat_penalty_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--repeat-penalty",
        "0.5",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-repeat-penalty exit code mismatch");
    expect(output.str().empty(), "generate invalid-repeat-penalty should not write stdout");
    expect(
        error.str().find("--repeat-penalty must be at least 1") != std::string::npos,
        "generate invalid-repeat-penalty error missing"
    );
}

void test_rejects_generate_greedy_sampling_conflict() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--greedy",
        "--seed",
        "1",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate greedy-conflict exit code mismatch");
    expect(output.str().empty(), "generate greedy-conflict should not write stdout");
    expect(
        error.str().find("--greedy cannot be combined with sampling options") != std::string::npos,
        "generate greedy-conflict error missing"
    );
}

const local_gguf_engine::tests::TestCase kGenerateCliTests[] = {
    {
        "generates text from GGUF bytes with cached argmax",
        test_generates_text_from_gguf_bytes_with_cached_argmax,
    },
    {
        "generates text from GGUF bytes with tied output weight",
        test_generates_text_from_gguf_bytes_with_tied_output_weight,
    },
    {
        "generates text from GGUF bytes stops at tokenizer EOS",
        test_generates_text_from_gguf_bytes_stops_at_tokenizer_eos,
    },
    {
        "generates text from GGUF bytes with chat-template prompt",
        test_generates_text_from_gguf_bytes_with_chat_template_prompt,
    },
    {
        "generates text from GGUF bytes with sampling options",
        test_generates_text_from_gguf_bytes_with_sampling_options,
    },
    {
        "generates text from GGUF bytes with CPU backend option",
        test_generates_text_from_gguf_bytes_with_cpu_backend_option,
    },
    {"runs generate success path", test_runs_generate_success_path},
    {
        "rejects generate Gemma2 runtime not implemented",
        test_rejects_generate_gemma2_runtime_not_implemented,
    },
    {
        "rejects generate Gemma2 unsupported chat template before runtime",
        test_rejects_generate_gemma2_unsupported_chat_template_before_runtime,
    },
    {"runs generate with greedy option", test_runs_generate_with_greedy_option},
    {"runs generate with CPU backend option", test_runs_generate_with_cpu_backend_option},
    {
        "runs generate with Metal backend option when available",
        test_runs_generate_with_metal_backend_option_when_available,
    },
    {"runs generate with sampling options", test_runs_generate_with_sampling_options},
    {"runs generate with top-p option", test_runs_generate_with_top_p_option},
    {"runs generate with first-token debug", test_runs_generate_with_first_token_debug},
    {
        "reports generate Metal backend unavailable when missing",
        test_reports_generate_metal_backend_unavailable_when_missing,
    },
    {"rejects generate missing arguments", test_rejects_generate_missing_arguments},
    {"rejects generate invalid max new tokens", test_rejects_generate_invalid_max_new_tokens},
    {"rejects generate invalid temperature option", test_rejects_generate_invalid_temperature_option},
    {"rejects generate invalid top-p option", test_rejects_generate_invalid_top_p_option},
    {"rejects generate invalid backend option", test_rejects_generate_invalid_backend_option},
    {
        "rejects generate invalid repeat-penalty option",
        test_rejects_generate_invalid_repeat_penalty_option,
    },
    {"rejects generate greedy sampling conflict", test_rejects_generate_greedy_sampling_conflict},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> generate_cli_test_cases() {
    return kGenerateCliTests;
}

}  // namespace local_gguf_engine::tests
