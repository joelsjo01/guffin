#include "local_gguf_engine/model_format/gguf_reader.hpp"

#include <algorithm>
#include <string>

namespace local_gguf_engine::model_format {

namespace {

[[nodiscard]] constexpr std::uint32_t read_little_endian_u32(
    std::span<const std::uint8_t> bytes,
    std::size_t offset
) noexcept {
    return static_cast<std::uint32_t>(bytes[offset])
        | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
        | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
        | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

[[nodiscard]] constexpr std::uint64_t read_little_endian_u64(
    std::span<const std::uint8_t> bytes,
    std::size_t offset
) noexcept {
    return static_cast<std::uint64_t>(bytes[offset])
        | (static_cast<std::uint64_t>(bytes[offset + 1]) << 8)
        | (static_cast<std::uint64_t>(bytes[offset + 2]) << 16)
        | (static_cast<std::uint64_t>(bytes[offset + 3]) << 24)
        | (static_cast<std::uint64_t>(bytes[offset + 4]) << 32)
        | (static_cast<std::uint64_t>(bytes[offset + 5]) << 40)
        | (static_cast<std::uint64_t>(bytes[offset + 6]) << 48)
        | (static_cast<std::uint64_t>(bytes[offset + 7]) << 56);
}

[[nodiscard]] std::string error_message(GGUFReadErrorCode code) {
    switch (code) {
        case GGUFReadErrorCode::truncated_header:
            return "truncated GGUF header";
        case GGUFReadErrorCode::invalid_magic:
            return "invalid GGUF magic";
    }
}

}  // namespace

GGUFReadError::GGUFReadError(
    GGUFReadErrorCode code,
    std::size_t expected_byte_count,
    std::size_t actual_byte_count,
    std::array<std::uint8_t, 4> actual_magic
)
    : std::runtime_error(error_message(code)),
      code_(code),
      expected_byte_count_(expected_byte_count),
      actual_byte_count_(actual_byte_count),
      actual_magic_(actual_magic) {}

GGUFReadErrorCode GGUFReadError::code() const noexcept {
    return code_;
}

std::size_t GGUFReadError::expected_byte_count() const noexcept {
    return expected_byte_count_;
}

std::size_t GGUFReadError::actual_byte_count() const noexcept {
    return actual_byte_count_;
}

std::array<std::uint8_t, 4> GGUFReadError::actual_magic() const noexcept {
    return actual_magic_;
}

GGUFHeader read_gguf_header(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < GGUFHeader::encoded_byte_count) {
        throw GGUFReadError(
            GGUFReadErrorCode::truncated_header,
            GGUFHeader::encoded_byte_count,
            bytes.size()
        );
    }

    std::array<std::uint8_t, 4> actual_magic {};
    std::copy_n(bytes.begin(), actual_magic.size(), actual_magic.begin());

    if (actual_magic != gguf_magic()) {
        throw GGUFReadError(
            GGUFReadErrorCode::invalid_magic,
            gguf_magic().size(),
            actual_magic.size(),
            actual_magic
        );
    }

    return GGUFHeader {
        .version = read_little_endian_u32(bytes, 4),
        .tensor_count = read_little_endian_u64(bytes, 8),
        .metadata_kv_count = read_little_endian_u64(bytes, 16),
    };
}

}  // namespace local_gguf_engine::model_format
