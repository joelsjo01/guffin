#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor {

[[nodiscard]] std::size_t unquantized_element_byte_count(DType dtype);
[[nodiscard]] std::uint64_t linear_index(const Shape& shape, std::span<const std::uint64_t> indices);
[[nodiscard]] std::size_t element_byte_offset(
    const TensorDescriptor& descriptor,
    std::span<const std::uint64_t> indices
);
[[nodiscard]] std::span<const std::uint8_t> element_bytes(
    const TensorStorage& storage,
    std::span<const std::uint64_t> indices
);
[[nodiscard]] std::span<std::uint8_t> mutable_element_bytes(
    TensorStorage& storage,
    std::span<const std::uint64_t> indices
);

}  // namespace local_gguf_engine::tensor
