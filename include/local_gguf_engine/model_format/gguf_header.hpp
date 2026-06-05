#pragma once

#include <cstddef>
#include <cstdint>

namespace local_gguf_engine::model_format {

struct GGUFHeader {
    static constexpr std::size_t encoded_byte_count = 24;

    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_kv_count = 0;

    constexpr bool operator==(const GGUFHeader&) const = default;
};

}  // namespace local_gguf_engine::model_format
