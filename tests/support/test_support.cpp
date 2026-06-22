#include "test_support.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace local_gguf_engine::tests {

using model_format::GGUFHeader;
using model_format::GGUFMetadataArray;
using model_format::GGUFMetadataEntry;
using model_format::GGUFMetadataSection;
using model_format::GGUFMetadataValue;
using model_format::GGUFMetadataValueType;

TestFailure::TestFailure(std::string message) : message_(std::move(message)) {}

const char* TestFailure::what() const noexcept {
    return message_.c_str();
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

GGUFMetadataEntry metadata_string(std::string key, std::string value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::string,
            .data = std::move(value),
        },
    };
}

GGUFMetadataEntry metadata_u32(std::string key, std::uint32_t value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::uint32,
            .data = value,
        },
    };
}

GGUFMetadataEntry metadata_f32(std::string key, float value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::float32,
            .data = value,
        },
    };
}

GGUFMetadataEntry metadata_bool(std::string key, bool value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::boolean,
            .data = value,
        },
    };
}

GGUFMetadataEntry metadata_string_array(std::string key, std::vector<std::string> values) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::array,
            .data = GGUFMetadataArray {
                .element_type = GGUFMetadataValueType::string,
                .values = std::move(values),
            },
        },
    };
}

GGUFMetadataEntry metadata_f32_array(std::string key, std::vector<float> values) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::array,
            .data = GGUFMetadataArray {
                .element_type = GGUFMetadataValueType::float32,
                .values = std::move(values),
            },
        },
    };
}

GGUFMetadataEntry metadata_i32_array(std::string key, std::vector<std::int32_t> values) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::array,
            .data = GGUFMetadataArray {
                .element_type = GGUFMetadataValueType::int32,
                .values = std::move(values),
            },
        },
    };
}

std::vector<GGUFMetadataEntry> tiny_llama_metadata() {
    return {
        metadata_string("general.architecture", "llama"),
        metadata_string("general.name", "Tiny Llama Fixture"),
        metadata_u32("llama.context_length", 16),
        metadata_u32("llama.embedding_length", 8),
        metadata_u32("llama.block_count", 2),
        metadata_u32("llama.feed_forward_length", 32),
        metadata_u32("llama.attention.head_count", 4),
        metadata_u32("llama.attention.head_count_kv", 2),
        metadata_u32("llama.attention.key_length", 2),
        metadata_u32("llama.attention.value_length", 2),
        metadata_u32("llama.rope.dimension_count", 2),
        metadata_u32("llama.vocab_size", 64),
        metadata_f32("llama.rope.freq_base", 10000.0F),
        metadata_f32("llama.attention.layer_norm_rms_epsilon", 0.00001F),
    };
}

GGUFMetadataSection metadata_section(std::vector<GGUFMetadataEntry> entries) {
    const auto metadata_count = static_cast<std::uint64_t>(entries.size());
    return GGUFMetadataSection {
        .header = GGUFHeader {
            .version = 3,
            .tensor_count = 0,
            .metadata_kv_count = metadata_count,
        },
        .entries = std::move(entries),
        .encoded_byte_count = 0,
    };
}

std::string smollm2_chat_template_fixture() {
    return "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n"
        "' + message['content'] + '<|im_end|>' + '\n"
        "'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n"
        "' }}{% endif %}";
}

std::string gemma2_chat_template_fixture() {
    return "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
        "{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
        "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}"
        "{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n"
        "' + message['content'] | trim + '<end_of_turn>\n"
        "' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n"
        "'}}{% endif %}";
}

std::vector<std::uint8_t> load_hex_fixture(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw TestFailure("fixture missing: " + path.string());
    }

    std::vector<std::uint8_t> bytes;
    std::string token;
    while (input >> token) {
        const auto value = std::stoul(token, nullptr, 16);
        if (value > 0xFFU) {
            throw TestFailure("fixture byte is out of range: " + token);
        }
        bytes.push_back(static_cast<std::uint8_t>(value));
    }

    return bytes;
}

std::vector<std::uint8_t> fixture(std::string_view name) {
    return load_hex_fixture(std::filesystem::path("tests") / "fixtures" / (std::string(name) + ".hex"));
}

}  // namespace local_gguf_engine::tests
