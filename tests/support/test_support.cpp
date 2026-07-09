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
