#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

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
    [[nodiscard]] std::uint8_t scale_value(std::size_t group_index) const;
    [[nodiscard]] std::uint8_t min_value(std::size_t group_index) const;
    [[nodiscard]] std::uint8_t quantized_nibble(std::size_t element_index) const;
    [[nodiscard]] float dequantized_value(std::size_t element_index) const;

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
    [[nodiscard]] std::int8_t quantized_value(std::size_t element_index) const;
    [[nodiscard]] std::int8_t scale_value(std::size_t group_index) const;
    [[nodiscard]] float dequantized_value(std::size_t element_index) const;

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
// Converts a GGUF/GGML F16 weight matrix shaped [shared, columns] into Q8_0.
// Blocks remain contiguous along the shared (dimension-0) width.
[[nodiscard]] TensorStorage quantize_f16_gguf_weight_q8_0(const TensorView& source);
// Writes directly into caller-owned storage. Destination metadata and its exact
// extent are validated before writing; source and destination must not overlap.
void quantize_f16_gguf_weight_q8_0_into(
    TensorView source, const TensorDescriptor& destination, std::span<std::uint8_t> bytes
);
[[nodiscard]] std::uint64_t q4_k_block_count(const TensorDescriptor& descriptor);
[[nodiscard]] Q4_KBlockView q4_k_block_view(const TensorStorage& storage, std::uint64_t block_index);
[[nodiscard]] std::array<float, Q4_KBlockView::element_count> dequantize_q4_k_block(
    const TensorStorage& storage,
    std::uint64_t block_index
);
[[nodiscard]] std::vector<float> dequantize_q4_k_row(const TensorStorage& storage, std::uint64_t row_index);
[[nodiscard]] float dot_q4_k_row_f32(
    const TensorStorage& storage,
    std::uint64_t row_index,
    std::span<const float> input
);
[[nodiscard]] std::uint64_t q6_k_block_count(const TensorDescriptor& descriptor);
[[nodiscard]] Q6_KBlockView q6_k_block_view(const TensorStorage& storage, std::uint64_t block_index);
[[nodiscard]] std::array<float, Q6_KBlockView::element_count> dequantize_q6_k_block(
    const TensorStorage& storage,
    std::uint64_t block_index
);
// Requantizes a GGUF Q6_K matrix shaped [shared, columns] into Q4_K while
// preserving its dimension-0 block order. Conversion is block-local and does
// not materialize an F32 matrix.
[[nodiscard]] TensorStorage requantize_q6_k_gguf_weight_q4_k(const TensorView& source);
// Canonical or columns8 Q4_K output uses the same block-local quantizer.
// Packed output is formed directly, without a full canonical staging tensor.
void requantize_q6_k_gguf_weight_q4_k_into(
    TensorView source, const TensorDescriptor& destination, std::span<std::uint8_t> bytes
);
[[nodiscard]] std::vector<float> dequantize_q6_k_row(const TensorStorage& storage, std::uint64_t row_index);
[[nodiscard]] float dot_q6_k_row_f32(
    const TensorStorage& storage,
    std::uint64_t row_index,
    std::span<const float> input
);

}  // namespace local_gguf_engine::tensor
