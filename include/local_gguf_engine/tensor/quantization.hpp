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

class Q4_KBlockView final {
public:
    static constexpr std::size_t element_count = 256;
    static constexpr std::size_t byte_count = 144;
    static constexpr std::size_t scale_min_byte_count = 12;
    static constexpr std::size_t quantized_byte_count = 128;

    Q4_KBlockView(
        std::uint16_t scale_bits,
        std::uint16_t min_scale_bits,
        std::span<const std::uint8_t> scale_min_bytes,
        std::span<const std::uint8_t> quantized_bytes
    );

    [[nodiscard]] std::uint16_t scale_bits() const noexcept;
    [[nodiscard]] std::uint16_t min_scale_bits() const noexcept;
    [[nodiscard]] float scale() const noexcept;
    [[nodiscard]] float min_scale() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> scale_min_bytes() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> quantized_bytes() const noexcept;
    [[nodiscard]] std::uint8_t quantized_nibble(std::size_t element_index) const;

private:
    std::uint16_t scale_bits_;
    std::uint16_t min_scale_bits_;
    std::span<const std::uint8_t> scale_min_bytes_;
    std::span<const std::uint8_t> quantized_bytes_;
};

class Q6_KBlockView final {
public:
    static constexpr std::size_t element_count = 256;
    static constexpr std::size_t byte_count = 210;
    static constexpr std::size_t low_bit_byte_count = 128;
    static constexpr std::size_t high_bit_byte_count = 64;
    static constexpr std::size_t scale_byte_count = 16;

    Q6_KBlockView(
        std::span<const std::uint8_t> low_bit_bytes,
        std::span<const std::uint8_t> high_bit_bytes,
        std::span<const std::uint8_t> scale_bytes,
        std::uint16_t scale_bits
    );

    [[nodiscard]] std::span<const std::uint8_t> low_bit_bytes() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> high_bit_bytes() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> scale_bytes() const noexcept;
    [[nodiscard]] std::uint16_t scale_bits() const noexcept;
    [[nodiscard]] float scale() const noexcept;
    [[nodiscard]] std::uint8_t low_nibble(std::size_t element_index) const;
    [[nodiscard]] std::uint8_t high_bits(std::size_t element_index) const;
    [[nodiscard]] std::int8_t scale_value(std::size_t group_index) const;

private:
    std::span<const std::uint8_t> low_bit_bytes_;
    std::span<const std::uint8_t> high_bit_bytes_;
    std::span<const std::uint8_t> scale_bytes_;
    std::uint16_t scale_bits_;
};

[[nodiscard]] std::uint64_t q8_0_block_count(const TensorDescriptor& descriptor);
[[nodiscard]] Q8_0BlockView q8_0_block_view(const TensorStorage& storage, std::uint64_t block_index);
[[nodiscard]] std::array<float, Q8_0BlockView::element_count> dequantize_q8_0_block(
    const TensorStorage& storage,
    std::uint64_t block_index
);
[[nodiscard]] std::uint64_t q4_k_block_count(const TensorDescriptor& descriptor);
[[nodiscard]] Q4_KBlockView q4_k_block_view(const TensorStorage& storage, std::uint64_t block_index);
[[nodiscard]] std::uint64_t q6_k_block_count(const TensorDescriptor& descriptor);
[[nodiscard]] Q6_KBlockView q6_k_block_view(const TensorStorage& storage, std::uint64_t block_index);

}  // namespace local_gguf_engine::tensor
