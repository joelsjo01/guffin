#pragma once

#include "local_gguf_engine/model_format/gguf_metadata.hpp"

#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::tests {

class TestFailure final : public std::exception {
public:
    explicit TestFailure(std::string message);

    [[nodiscard]] const char* what() const noexcept override;

private:
    std::string message_;
};

struct TestCase {
    std::string_view name;
    void (*run)();
};

void expect(bool condition, std::string_view message);

local_gguf_engine::model_format::GGUFMetadataEntry metadata_string(std::string key, std::string value);
local_gguf_engine::model_format::GGUFMetadataEntry metadata_u32(std::string key, std::uint32_t value);
local_gguf_engine::model_format::GGUFMetadataEntry metadata_f32(std::string key, float value);
local_gguf_engine::model_format::GGUFMetadataEntry metadata_bool(std::string key, bool value);
local_gguf_engine::model_format::GGUFMetadataEntry metadata_string_array(
    std::string key,
    std::vector<std::string> values
);
local_gguf_engine::model_format::GGUFMetadataEntry metadata_f32_array(std::string key, std::vector<float> values);
local_gguf_engine::model_format::GGUFMetadataEntry metadata_i32_array(
    std::string key,
    std::vector<std::int32_t> values
);

local_gguf_engine::model_format::GGUFMetadataSection metadata_section(
    std::vector<local_gguf_engine::model_format::GGUFMetadataEntry> entries
);
std::string gemma2_chat_template_fixture();
std::vector<std::uint8_t> fixture(std::string_view name);

}  // namespace local_gguf_engine::tests
