#pragma once

#include "local_gguf_engine/model_format/gguf_header.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace local_gguf_engine::model_format {

enum class GGUFMetadataValueType : std::uint32_t {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

struct GGUFMetadataArray {
    using Values = std::variant<
        std::vector<std::uint8_t>,
        std::vector<std::int8_t>,
        std::vector<std::uint16_t>,
        std::vector<std::int16_t>,
        std::vector<std::uint32_t>,
        std::vector<std::int32_t>,
        std::vector<float>,
        std::vector<bool>,
        std::vector<std::string>,
        std::vector<std::uint64_t>,
        std::vector<std::int64_t>,
        std::vector<double>>;

    GGUFMetadataValueType element_type = GGUFMetadataValueType::uint8;
    Values values {};

    constexpr bool operator==(const GGUFMetadataArray&) const = default;
};

struct GGUFMetadataValue {
    using Data = std::variant<
        std::uint8_t,
        std::int8_t,
        std::uint16_t,
        std::int16_t,
        std::uint32_t,
        std::int32_t,
        float,
        bool,
        std::string,
        GGUFMetadataArray,
        std::uint64_t,
        std::int64_t,
        double>;

    GGUFMetadataValueType type = GGUFMetadataValueType::uint8;
    Data data {};

    constexpr bool operator==(const GGUFMetadataValue&) const = default;
};

struct GGUFMetadataEntry {
    std::string key;
    GGUFMetadataValue value;

    constexpr bool operator==(const GGUFMetadataEntry&) const = default;
};

struct GGUFMetadataSection {
    GGUFHeader header;
    std::vector<GGUFMetadataEntry> entries;
    std::size_t encoded_byte_count = 0;

    constexpr bool operator==(const GGUFMetadataSection&) const = default;
};

}  // namespace local_gguf_engine::model_format
