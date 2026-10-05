#include "local_gguf_engine/model_format/gguf_reader.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace local_gguf_engine::model_format {

namespace {

class UniqueFileDescriptor final {
public:
    explicit UniqueFileDescriptor(int descriptor) noexcept : descriptor_(descriptor) {}

    UniqueFileDescriptor(const UniqueFileDescriptor&) = delete;
    UniqueFileDescriptor& operator=(const UniqueFileDescriptor&) = delete;

    ~UniqueFileDescriptor() {
        if (descriptor_ >= 0) {
            (void)::close(descriptor_);
        }
    }

    [[nodiscard]] int get() const noexcept {
        return descriptor_;
    }

private:
    int descriptor_ = -1;
};

[[nodiscard]] std::system_error file_error(std::string_view operation, const std::filesystem::path& path) {
    const auto error_number = errno;
    return std::system_error(
        error_number,
        std::generic_category(),
        std::string(operation) + " '" + path.string() + "'"
    );
}

[[nodiscard]] constexpr std::uint32_t read_little_endian_u32(
    std::span<const std::uint8_t> bytes,
    std::size_t offset
) noexcept {
    return static_cast<std::uint32_t>(bytes[offset])
        | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
        | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
        | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

[[nodiscard]] constexpr std::uint16_t read_little_endian_u16(
    std::span<const std::uint8_t> bytes,
    std::size_t offset
) noexcept {
    return static_cast<std::uint16_t>(bytes[offset])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
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

void require_available(
    std::span<const std::uint8_t> bytes,
    std::size_t offset,
    std::size_t byte_count,
    GGUFReadErrorCode error_code
) {
    if (offset <= bytes.size() && byte_count <= bytes.size() - offset) {
        return;
    }

    const auto max_size = std::numeric_limits<std::size_t>::max();
    const auto expected_byte_count = byte_count > max_size - offset ? max_size : offset + byte_count;
    throw GGUFReadError(error_code, expected_byte_count, bytes.size());
}

[[nodiscard]] std::size_t checked_size(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw GGUFReadError(
            GGUFReadErrorCode::truncated_metadata,
            std::numeric_limits<std::size_t>::max(),
            0
        );
    }

    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::size_t remaining_byte_count(std::span<const std::uint8_t> bytes, std::size_t offset) noexcept {
    if (offset > bytes.size()) {
        return 0;
    }

    return bytes.size() - offset;
}

[[nodiscard]] std::size_t bounded_reserve_count(
    std::uint64_t requested_count,
    std::span<const std::uint8_t> bytes,
    std::size_t offset
) {
    return checked_size(std::min(requested_count, static_cast<std::uint64_t>(remaining_byte_count(bytes, offset))));
}

[[nodiscard]] std::uint64_t checked_u64_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_offset, 0, 0);
    }

    return left + right;
}

[[nodiscard]] std::uint64_t checked_u64_multiply(
    std::uint64_t left,
    std::uint64_t right,
    GGUFReadErrorCode error_code
) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw GGUFReadError(error_code, 0, 0);
    }

    return left * right;
}

[[nodiscard]] std::string read_gguf_string(
    std::span<const std::uint8_t> bytes,
    std::size_t& offset,
    GGUFReadErrorCode error_code
) {
    require_available(bytes, offset, sizeof(std::uint64_t), error_code);
    const auto length = checked_size(read_little_endian_u64(bytes, offset));
    offset += sizeof(std::uint64_t);

    require_available(bytes, offset, length, error_code);
    std::string value;
    value.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        value.push_back(static_cast<char>(bytes[offset + index]));
    }
    offset += length;

    return value;
}

[[nodiscard]] GGUFMetadataValueType metadata_value_type_from_raw(std::uint32_t raw_type) {
    switch (raw_type) {
        case 0:
            return GGUFMetadataValueType::uint8;
        case 1:
            return GGUFMetadataValueType::int8;
        case 2:
            return GGUFMetadataValueType::uint16;
        case 3:
            return GGUFMetadataValueType::int16;
        case 4:
            return GGUFMetadataValueType::uint32;
        case 5:
            return GGUFMetadataValueType::int32;
        case 6:
            return GGUFMetadataValueType::float32;
        case 7:
            return GGUFMetadataValueType::boolean;
        case 8:
            return GGUFMetadataValueType::string;
        case 9:
            return GGUFMetadataValueType::array;
        case 10:
            return GGUFMetadataValueType::uint64;
        case 11:
            return GGUFMetadataValueType::int64;
        case 12:
            return GGUFMetadataValueType::float64;
        default:
            throw GGUFReadError(GGUFReadErrorCode::invalid_metadata_value_type, 0, raw_type);
    }
}

[[nodiscard]] GGUFMetadataValueType read_metadata_value_type(
    std::span<const std::uint8_t> bytes,
    std::size_t& offset
) {
    require_available(bytes, offset, sizeof(std::uint32_t), GGUFReadErrorCode::truncated_metadata);
    const auto raw_type = read_little_endian_u32(bytes, offset);
    offset += sizeof(std::uint32_t);
    return metadata_value_type_from_raw(raw_type);
}

template <typename Element, typename Reader>
[[nodiscard]] std::vector<Element> read_metadata_array_elements(
    std::span<const std::uint8_t> bytes,
    std::size_t& offset,
    std::uint64_t element_count,
    Reader read_element
) {
    std::vector<Element> values;
    values.reserve(bounded_reserve_count(element_count, bytes, offset));

    for (std::uint64_t index = 0; index < element_count; ++index) {
        values.push_back(read_element(bytes, offset));
    }

    return values;
}

[[nodiscard]] std::uint8_t read_metadata_u8(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    require_available(bytes, offset, sizeof(std::uint8_t), GGUFReadErrorCode::truncated_metadata);
    const auto value = bytes[offset];
    offset += sizeof(std::uint8_t);
    return value;
}

[[nodiscard]] std::int8_t read_metadata_i8(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    return std::bit_cast<std::int8_t>(read_metadata_u8(bytes, offset));
}

[[nodiscard]] std::uint16_t read_metadata_u16(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    require_available(bytes, offset, sizeof(std::uint16_t), GGUFReadErrorCode::truncated_metadata);
    const auto value = read_little_endian_u16(bytes, offset);
    offset += sizeof(std::uint16_t);
    return value;
}

[[nodiscard]] std::int16_t read_metadata_i16(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    return std::bit_cast<std::int16_t>(read_metadata_u16(bytes, offset));
}

[[nodiscard]] std::uint32_t read_metadata_u32(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    require_available(bytes, offset, sizeof(std::uint32_t), GGUFReadErrorCode::truncated_metadata);
    const auto value = read_little_endian_u32(bytes, offset);
    offset += sizeof(std::uint32_t);
    return value;
}

[[nodiscard]] std::int32_t read_metadata_i32(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    return std::bit_cast<std::int32_t>(read_metadata_u32(bytes, offset));
}

[[nodiscard]] float read_metadata_f32(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    return std::bit_cast<float>(read_metadata_u32(bytes, offset));
}

[[nodiscard]] bool read_metadata_bool(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    const auto value = read_metadata_u8(bytes, offset);
    if (value > 1) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_metadata_value, 1, value);
    }
    return value == 1;
}

[[nodiscard]] std::uint64_t read_metadata_u64(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    require_available(bytes, offset, sizeof(std::uint64_t), GGUFReadErrorCode::truncated_metadata);
    const auto value = read_little_endian_u64(bytes, offset);
    offset += sizeof(std::uint64_t);
    return value;
}

[[nodiscard]] std::int64_t read_metadata_i64(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    return std::bit_cast<std::int64_t>(read_metadata_u64(bytes, offset));
}

[[nodiscard]] double read_metadata_f64(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    return std::bit_cast<double>(read_metadata_u64(bytes, offset));
}

[[nodiscard]] std::uint32_t read_tensor_u32(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    require_available(bytes, offset, sizeof(std::uint32_t), GGUFReadErrorCode::truncated_tensor_directory);
    const auto value = read_little_endian_u32(bytes, offset);
    offset += sizeof(std::uint32_t);
    return value;
}

[[nodiscard]] std::uint64_t read_tensor_u64(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    require_available(bytes, offset, sizeof(std::uint64_t), GGUFReadErrorCode::truncated_tensor_directory);
    const auto value = read_little_endian_u64(bytes, offset);
    offset += sizeof(std::uint64_t);
    return value;
}

[[nodiscard]] std::string read_tensor_name(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    auto name = read_gguf_string(bytes, offset, GGUFReadErrorCode::truncated_tensor_directory);
    if (name.size() > GGUFTensorInfo::max_name_byte_count) {
        throw GGUFReadError(
            GGUFReadErrorCode::invalid_tensor_name,
            GGUFTensorInfo::max_name_byte_count,
            name.size()
        );
    }

    return name;
}

[[nodiscard]] GGUFTensorInfo read_tensor_info(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    auto name = read_tensor_name(bytes, offset);
    const auto dimension_count = read_tensor_u32(bytes, offset);
    if (dimension_count > GGUFTensorInfo::max_dimension_count) {
        throw GGUFReadError(
            GGUFReadErrorCode::invalid_tensor_dimension_count,
            GGUFTensorInfo::max_dimension_count,
            dimension_count
        );
    }

    std::vector<std::uint64_t> dimensions;
    dimensions.reserve(dimension_count);
    for (std::uint32_t index = 0; index < dimension_count; ++index) {
        dimensions.push_back(read_tensor_u64(bytes, offset));
    }

    const auto type_id = read_tensor_u32(bytes, offset);
    const auto tensor_data_relative_offset = read_tensor_u64(bytes, offset);

    return GGUFTensorInfo {
        .name = std::move(name),
        .dimensions = std::move(dimensions),
        .type_id = type_id,
        .offset = tensor_data_relative_offset,
    };
}

[[nodiscard]] GGUFMetadataArray read_metadata_array(std::span<const std::uint8_t> bytes, std::size_t& offset) {
    const auto element_type = read_metadata_value_type(bytes, offset);
    const auto element_count = read_metadata_u64(bytes, offset);

    switch (element_type) {
        case GGUFMetadataValueType::uint8:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::uint8_t>(bytes, offset, element_count, read_metadata_u8),
            };
        case GGUFMetadataValueType::int8:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::int8_t>(bytes, offset, element_count, read_metadata_i8),
            };
        case GGUFMetadataValueType::uint16:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::uint16_t>(bytes, offset, element_count, read_metadata_u16),
            };
        case GGUFMetadataValueType::int16:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::int16_t>(bytes, offset, element_count, read_metadata_i16),
            };
        case GGUFMetadataValueType::uint32:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::uint32_t>(bytes, offset, element_count, read_metadata_u32),
            };
        case GGUFMetadataValueType::int32:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::int32_t>(bytes, offset, element_count, read_metadata_i32),
            };
        case GGUFMetadataValueType::float32:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<float>(bytes, offset, element_count, read_metadata_f32),
            };
        case GGUFMetadataValueType::boolean:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<bool>(bytes, offset, element_count, read_metadata_bool),
            };
        case GGUFMetadataValueType::string:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::string>(
                    bytes,
                    offset,
                    element_count,
                    [](std::span<const std::uint8_t> source, std::size_t& cursor) {
                        return read_gguf_string(source, cursor, GGUFReadErrorCode::truncated_metadata);
                    }
                ),
            };
        case GGUFMetadataValueType::uint64:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::uint64_t>(bytes, offset, element_count, read_metadata_u64),
            };
        case GGUFMetadataValueType::int64:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<std::int64_t>(bytes, offset, element_count, read_metadata_i64),
            };
        case GGUFMetadataValueType::float64:
            return GGUFMetadataArray {
                .element_type = element_type,
                .values = read_metadata_array_elements<double>(bytes, offset, element_count, read_metadata_f64),
            };
        case GGUFMetadataValueType::array:
            throw GGUFReadError(GGUFReadErrorCode::unsupported_metadata_value_type, 0, 9);
    }
}

[[nodiscard]] GGUFMetadataValue read_metadata_value(
    std::span<const std::uint8_t> bytes,
    std::size_t& offset,
    GGUFMetadataValueType type
) {
    switch (type) {
        case GGUFMetadataValueType::uint8:
            return GGUFMetadataValue {.type = type, .data = read_metadata_u8(bytes, offset)};
        case GGUFMetadataValueType::int8:
            return GGUFMetadataValue {.type = type, .data = read_metadata_i8(bytes, offset)};
        case GGUFMetadataValueType::uint16:
            return GGUFMetadataValue {.type = type, .data = read_metadata_u16(bytes, offset)};
        case GGUFMetadataValueType::int16:
            return GGUFMetadataValue {.type = type, .data = read_metadata_i16(bytes, offset)};
        case GGUFMetadataValueType::uint32:
            return GGUFMetadataValue {.type = type, .data = read_metadata_u32(bytes, offset)};
        case GGUFMetadataValueType::int32:
            return GGUFMetadataValue {.type = type, .data = read_metadata_i32(bytes, offset)};
        case GGUFMetadataValueType::float32:
            return GGUFMetadataValue {.type = type, .data = read_metadata_f32(bytes, offset)};
        case GGUFMetadataValueType::boolean:
            return GGUFMetadataValue {.type = type, .data = read_metadata_bool(bytes, offset)};
        case GGUFMetadataValueType::string:
            return GGUFMetadataValue {
                .type = type,
                .data = read_gguf_string(bytes, offset, GGUFReadErrorCode::truncated_metadata),
            };
        case GGUFMetadataValueType::array:
            return GGUFMetadataValue {.type = type, .data = read_metadata_array(bytes, offset)};
        case GGUFMetadataValueType::uint64:
            return GGUFMetadataValue {.type = type, .data = read_metadata_u64(bytes, offset)};
        case GGUFMetadataValueType::int64:
            return GGUFMetadataValue {.type = type, .data = read_metadata_i64(bytes, offset)};
        case GGUFMetadataValueType::float64:
            return GGUFMetadataValue {.type = type, .data = read_metadata_f64(bytes, offset)};
    }
}

[[nodiscard]] bool metadata_entry_is_uint32(
    const GGUFMetadataEntry& entry,
    std::string_view key
) {
    return entry.key == key && entry.value.type == GGUFMetadataValueType::uint32;
}

[[nodiscard]] std::uint32_t read_alignment(const GGUFMetadataSection& metadata) {
    constexpr std::uint32_t default_alignment = 32;

    for (const auto& entry : metadata.entries) {
        if (entry.key != "general.alignment") {
            continue;
        }

        if (!metadata_entry_is_uint32(entry, "general.alignment")) {
            throw GGUFReadError(GGUFReadErrorCode::invalid_alignment, 4, 0);
        }

        const auto alignment = std::get<std::uint32_t>(entry.value.data);
        if (alignment == 0 || alignment % 8 != 0) {
            throw GGUFReadError(GGUFReadErrorCode::invalid_alignment, 8, alignment);
        }

        return alignment;
    }

    return default_alignment;
}

[[nodiscard]] std::size_t align_offset(std::size_t offset, std::uint32_t alignment) {
    const auto remainder = offset % alignment;
    if (remainder == 0) {
        return offset;
    }

    const auto padding_byte_count = static_cast<std::size_t>(alignment - remainder);
    if (padding_byte_count > std::numeric_limits<std::size_t>::max() - offset) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_alignment, alignment, 0);
    }

    return offset + padding_byte_count;
}

void validate_tensor_padding(
    std::span<const std::uint8_t> bytes,
    std::size_t tensor_infos_end_offset,
    std::size_t padding_byte_count
) {
    require_available(
        bytes,
        tensor_infos_end_offset,
        padding_byte_count,
        GGUFReadErrorCode::truncated_tensor_directory
    );

    for (std::size_t index = 0; index < padding_byte_count; ++index) {
        if (bytes[tensor_infos_end_offset + index] != 0) {
            throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_padding, 0, bytes[tensor_infos_end_offset + index]);
        }
    }
}

void resolve_tensor_file_offsets(
    std::vector<GGUFTensorInfo>& tensors,
    std::uint32_t alignment,
    std::size_t tensor_data_start_offset
) {
    const auto tensor_data_start = static_cast<std::uint64_t>(tensor_data_start_offset);

    for (auto& tensor : tensors) {
        if (tensor.offset % alignment != 0) {
            throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_offset, alignment, tensor.offset);
        }

        tensor.file_offset = checked_u64_add(tensor_data_start, tensor.offset);
    }
}

struct TensorTypeLayout {
    std::uint64_t block_element_count;
    std::uint64_t block_byte_count;
};

[[nodiscard]] TensorTypeLayout tensor_type_layout(std::uint32_t type_id) {
    switch (type_id) {
        case 0:
            return TensorTypeLayout {.block_element_count = 1, .block_byte_count = 4};
        case 1:
            return TensorTypeLayout {.block_element_count = 1, .block_byte_count = 2};
        case 8:
            return TensorTypeLayout {.block_element_count = 32, .block_byte_count = 34};
        case 12:
            return TensorTypeLayout {.block_element_count = 256, .block_byte_count = 144};
        case 14:
            return TensorTypeLayout {.block_element_count = 256, .block_byte_count = 210};
        default:
            throw GGUFReadError(GGUFReadErrorCode::unsupported_tensor_type, 0, type_id);
    }
}

[[nodiscard]] std::uint64_t tensor_element_count(const GGUFTensorInfo& tensor) {
    std::uint64_t element_count = 1;
    for (const auto dimension : tensor.dimensions) {
        element_count = checked_u64_multiply(element_count, dimension, GGUFReadErrorCode::invalid_tensor_size);
    }

    return element_count;
}

[[nodiscard]] std::uint64_t tensor_row_element_count(const GGUFTensorInfo& tensor) {
    if (tensor.dimensions.empty()) {
        return tensor_element_count(tensor);
    }

    // GGUF stores the contiguous dimension first. The number of output rows
    // (including vocabulary size) does not need quantization-block alignment.
    return tensor.dimensions.front();
}

[[nodiscard]] std::size_t tensor_payload_byte_count(const GGUFTensorInfo& tensor) {
    const auto layout = tensor_type_layout(tensor.type_id);
    const auto row_element_count = tensor_row_element_count(tensor);
    if (row_element_count % layout.block_element_count != 0) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_size, layout.block_element_count, row_element_count);
    }

    const auto element_count = tensor_element_count(tensor);
    if (element_count % layout.block_element_count != 0) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_size, layout.block_element_count, element_count);
    }

    const auto block_count = element_count / layout.block_element_count;
    const auto byte_count = checked_u64_multiply(
        block_count,
        layout.block_byte_count,
        GGUFReadErrorCode::invalid_tensor_size
    );
    if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_size, 0, 0);
    }

    return static_cast<std::size_t>(byte_count);
}

[[nodiscard]] std::size_t tensor_file_offset(const GGUFTensorInfo& tensor) {
    if (tensor.file_offset > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw GGUFReadError(GGUFReadErrorCode::truncated_tensor_payload, 0, 0);
    }

    return static_cast<std::size_t>(tensor.file_offset);
}

[[nodiscard]] GGUFTensorPayloadView read_tensor_payload_view(
    std::span<const std::uint8_t> bytes,
    const GGUFTensorInfo& tensor
) {
    const auto file_offset = tensor_file_offset(tensor);
    const auto byte_count = tensor_payload_byte_count(tensor);
    require_available(bytes, file_offset, byte_count, GGUFReadErrorCode::truncated_tensor_payload);

    return GGUFTensorPayloadView {
        .tensor = tensor,
        .bytes = bytes.subspan(file_offset, byte_count),
    };
}

[[nodiscard]] std::string error_message(GGUFReadErrorCode code) {
    switch (code) {
        case GGUFReadErrorCode::truncated_header:
            return "truncated GGUF header";
        case GGUFReadErrorCode::invalid_magic:
            return "invalid GGUF magic";
        case GGUFReadErrorCode::truncated_metadata:
            return "truncated GGUF metadata";
        case GGUFReadErrorCode::invalid_metadata_value_type:
            return "invalid GGUF metadata value type";
        case GGUFReadErrorCode::invalid_metadata_value:
            return "invalid GGUF metadata value";
        case GGUFReadErrorCode::unsupported_metadata_value_type:
            return "unsupported GGUF metadata value type";
        case GGUFReadErrorCode::truncated_tensor_directory:
            return "truncated GGUF tensor directory";
        case GGUFReadErrorCode::invalid_tensor_dimension_count:
            return "invalid GGUF tensor dimension count";
        case GGUFReadErrorCode::invalid_tensor_name:
            return "invalid GGUF tensor name";
        case GGUFReadErrorCode::invalid_alignment:
            return "invalid GGUF alignment";
        case GGUFReadErrorCode::invalid_tensor_offset:
            return "invalid GGUF tensor offset";
        case GGUFReadErrorCode::invalid_tensor_padding:
            return "invalid GGUF tensor padding";
        case GGUFReadErrorCode::truncated_tensor_payload:
            return "truncated GGUF tensor payload";
        case GGUFReadErrorCode::unsupported_tensor_type:
            return "unsupported GGUF tensor type";
        case GGUFReadErrorCode::invalid_tensor_size:
            return "invalid GGUF tensor size";
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

GGUFFileBackedModel::GGUFFileBackedModel(GGUFFileBackedModel&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      byte_count_(std::exchange(other.byte_count_, 0)),
      model_(std::move(other.model_)) {}

GGUFFileBackedModel& GGUFFileBackedModel::operator=(GGUFFileBackedModel&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        byte_count_ = std::exchange(other.byte_count_, 0);
        model_ = std::move(other.model_);
    }

    return *this;
}

GGUFFileBackedModel::~GGUFFileBackedModel() {
    reset();
}

std::span<const std::uint8_t> GGUFFileBackedModel::bytes() const noexcept {
    return std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(data_), byte_count_);
}

const GGUFModelView& GGUFFileBackedModel::view() const noexcept {
    return model_;
}

GGUFFileBackedModel::GGUFFileBackedModel(void* data, std::size_t byte_count, GGUFModelView model)
    : data_(data),
      byte_count_(byte_count),
      model_(std::move(model)) {}

void GGUFFileBackedModel::reset() noexcept {
    if (data_ != nullptr) {
        (void)::munmap(data_, byte_count_);
        data_ = nullptr;
        byte_count_ = 0;
    }
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

GGUFMetadataSection read_gguf_metadata_section(std::span<const std::uint8_t> bytes) {
    const auto header = read_gguf_header(bytes);
    std::size_t offset = GGUFHeader::encoded_byte_count;

    std::vector<GGUFMetadataEntry> entries;
    entries.reserve(bounded_reserve_count(header.metadata_kv_count, bytes, offset));

    for (std::uint64_t index = 0; index < header.metadata_kv_count; ++index) {
        auto key = read_gguf_string(bytes, offset, GGUFReadErrorCode::truncated_metadata);
        const auto value_type = read_metadata_value_type(bytes, offset);
        auto value = read_metadata_value(bytes, offset, value_type);
        entries.push_back(GGUFMetadataEntry {.key = std::move(key), .value = std::move(value)});
    }

    return GGUFMetadataSection {
        .header = header,
        .entries = std::move(entries),
        .encoded_byte_count = offset,
    };
}

GGUFTensorDirectory read_gguf_tensor_directory(std::span<const std::uint8_t> bytes) {
    auto metadata = read_gguf_metadata_section(bytes);
    std::size_t offset = metadata.encoded_byte_count;

    std::vector<GGUFTensorInfo> tensors;
    tensors.reserve(bounded_reserve_count(metadata.header.tensor_count, bytes, offset));

    for (std::uint64_t index = 0; index < metadata.header.tensor_count; ++index) {
        tensors.push_back(read_tensor_info(bytes, offset));
    }

    const auto alignment = read_alignment(metadata);
    const auto tensor_infos_end_offset = offset;
    const auto tensor_data_start_offset = align_offset(tensor_infos_end_offset, alignment);
    const auto padding_byte_count = tensor_data_start_offset - tensor_infos_end_offset;

    validate_tensor_padding(bytes, tensor_infos_end_offset, padding_byte_count);
    resolve_tensor_file_offsets(tensors, alignment, tensor_data_start_offset);

    return GGUFTensorDirectory {
        .metadata = std::move(metadata),
        .tensors = std::move(tensors),
        .alignment = alignment,
        .tensor_infos_end_offset = tensor_infos_end_offset,
        .padding_byte_count = padding_byte_count,
        .tensor_data_start_offset = tensor_data_start_offset,
        .encoded_byte_count = tensor_data_start_offset,
    };
}

GGUFModelView read_gguf_model_view(std::span<const std::uint8_t> bytes) {
    auto directory = read_gguf_tensor_directory(bytes);

    std::vector<GGUFTensorPayloadView> tensor_payloads;
    tensor_payloads.reserve(directory.tensors.size());
    for (const auto& tensor : directory.tensors) {
        tensor_payloads.push_back(read_tensor_payload_view(bytes, tensor));
    }

    return GGUFModelView {
        .directory = std::move(directory),
        .tensor_payloads = std::move(tensor_payloads),
    };
}

GGUFFileBackedModel read_file_backed_gguf_model(const std::filesystem::path& path) {
    const auto descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw file_error("could not read", path);
    }

    UniqueFileDescriptor file(descriptor);

    struct stat status {};
    if (::fstat(file.get(), &status) != 0) {
        throw file_error("could not read", path);
    }

    if (status.st_size < 0) {
        throw std::runtime_error("could not read '" + path.string() + "': invalid file size");
    }

    const auto file_size = static_cast<std::uintmax_t>(status.st_size);
    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    const auto byte_count = static_cast<std::size_t>(file_size);
    if (byte_count == 0) {
        throw GGUFReadError(GGUFReadErrorCode::truncated_header, GGUFHeader::encoded_byte_count, 0);
    }

    void* data = ::mmap(nullptr, byte_count, PROT_READ, MAP_PRIVATE, file.get(), 0);
    if (data == MAP_FAILED) {
        throw file_error("could not map", path);
    }

    try {
        auto model = read_gguf_model_view(
            std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(data), byte_count)
        );
        return GGUFFileBackedModel(data, byte_count, std::move(model));
    } catch (...) {
        (void)::munmap(data, byte_count);
        throw;
    }
}

}  // namespace local_gguf_engine::model_format
