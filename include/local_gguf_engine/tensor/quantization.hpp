#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor {

class Q8_0BlockView final {
public:
    static constexpr std::size_t element_count = 32;
    static constexpr std::size_t byte_count = 34;

    Q8_0BlockView(std::uint16_t scale_bits, std::span<const std::uint8_t> quantized_bytes);

    [[nodiscard]] std::uint16_t scale_bits() const noexcept;
    [[nodiscard]] float scale() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> quantized_bytes() const noexcept;
    [[nodiscard]] std::int8_t quantized_value(std::size_t index) const;
    [[nodiscard]] float dequantized_value(std::size_t index) const;

private:
    std::uint16_t scale_bits_;
    std::span<const std::uint8_t> quantized_bytes_;
};

[[nodiscard]] std::uint64_t q8_0_block_count(const TensorDescriptor& descriptor);
[[nodiscard]] Q8_0BlockView q8_0_block_view(const TensorStorage& storage, std::uint64_t block_index);
[[nodiscard]] std::array<float, Q8_0BlockView::element_count> dequantize_q8_0_block(
    const TensorStorage& storage,
    std::uint64_t block_index
);

}  // namespace local_gguf_engine::tensor
