#pragma once

#include "local_gguf_engine/model_format/gguf_metadata.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace local_gguf_engine::model_format {

struct GGUFTensorInfo {
    static constexpr std::uint32_t max_dimension_count = 4;
    static constexpr std::size_t max_name_byte_count = 64;

    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type_id = 0;
    std::uint64_t offset = 0;

    constexpr bool operator==(const GGUFTensorInfo&) const = default;
};

struct GGUFTensorDirectory {
    GGUFMetadataSection metadata;
    std::vector<GGUFTensorInfo> tensors;
    std::size_t encoded_byte_count = 0;

    constexpr bool operator==(const GGUFTensorDirectory&) const = default;
};

}  // namespace local_gguf_engine::model_format
