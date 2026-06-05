#pragma once

#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace local_gguf_engine::model_format {

enum class GGUFReadErrorCode {
    truncated_header,
    invalid_magic,
    truncated_metadata,
    invalid_metadata_value_type,
    invalid_metadata_value,
    unsupported_metadata_value_type,
    truncated_tensor_directory,
    invalid_tensor_dimension_count,
    invalid_tensor_name,
    invalid_alignment,
    invalid_tensor_offset,
    invalid_tensor_padding,
    truncated_tensor_payload,
    unsupported_tensor_type,
    invalid_tensor_size,
};

class GGUFReadError final : public std::runtime_error {
public:
    GGUFReadError(
        GGUFReadErrorCode code,
        std::size_t expected_byte_count,
        std::size_t actual_byte_count,
        std::array<std::uint8_t, 4> actual_magic = {}
    );

    [[nodiscard]] GGUFReadErrorCode code() const noexcept;
    [[nodiscard]] std::size_t expected_byte_count() const noexcept;
    [[nodiscard]] std::size_t actual_byte_count() const noexcept;
    [[nodiscard]] std::array<std::uint8_t, 4> actual_magic() const noexcept;

private:
    GGUFReadErrorCode code_;
    std::size_t expected_byte_count_;
    std::size_t actual_byte_count_;
    std::array<std::uint8_t, 4> actual_magic_;
};

[[nodiscard]] constexpr std::array<std::uint8_t, 4> gguf_magic() noexcept {
    return {0x47, 0x47, 0x55, 0x46};
}

[[nodiscard]] GGUFHeader read_gguf_header(std::span<const std::uint8_t> bytes);
[[nodiscard]] GGUFMetadataSection read_gguf_metadata_section(std::span<const std::uint8_t> bytes);
[[nodiscard]] GGUFTensorDirectory read_gguf_tensor_directory(std::span<const std::uint8_t> bytes);
[[nodiscard]] GGUFModelView read_gguf_model_view(std::span<const std::uint8_t> bytes);

}  // namespace local_gguf_engine::model_format
