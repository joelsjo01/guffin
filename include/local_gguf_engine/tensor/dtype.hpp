#pragma once

#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor {

enum class DType : std::uint8_t {
    f32 = 0,
    f16 = 1,
    q8_0 = 8,
    q4_k = 12,
    q6_k = 14,
};

struct DTypeLayout {
    std::uint64_t block_element_count;
    std::uint64_t block_byte_count;

    constexpr bool operator==(const DTypeLayout&) const = default;
};

[[nodiscard]] DTypeLayout dtype_layout(DType dtype);

}  // namespace local_gguf_engine::tensor
