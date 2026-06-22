#include "local_gguf_engine/tensor/quantization.hpp"

#include "local_gguf_engine/tensor/f16.hpp"

#include <bit>
#include <limits>

namespace local_gguf_engine::tensor {

namespace {

void require_quantized_descriptor(const TensorDescriptor& descriptor, DType expected_dtype) {
    if (descriptor.dtype != expected_dtype) {
        throw TensorError(
            TensorErrorCode::unsupported_quantized_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(expected_dtype)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

[[nodiscard]] std::uint16_t read_little_endian_u16(std::span<const std::uint8_t> bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8);
}

[[nodiscard]] std::size_t checked_quantized_block_byte_offset(
    std::uint64_t block_index,
    std::uint64_t block_byte_count
) {
    if (block_index > std::numeric_limits<std::uint64_t>::max() / block_byte_count) {
        throw TensorError(TensorErrorCode::byte_size_overflow, 0, block_index);
    }

    const auto offset = block_index * block_byte_count;
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw TensorError(TensorErrorCode::byte_size_overflow, 0, offset);
    }

    return static_cast<std::size_t>(offset);
}

void require_span_size(
    std::span<const std::uint8_t> bytes,
    std::size_t expected_byte_count
) {
    if (bytes.size() != expected_byte_count) {
        throw TensorError(
            TensorErrorCode::invalid_storage_byte_count,
            static_cast<std::uint64_t>(expected_byte_count),
            static_cast<std::uint64_t>(bytes.size())
        );
    }
}

void require_block_element_index(std::size_t index, std::size_t element_count) {
    if (index >= element_count) {
        throw TensorError(
            TensorErrorCode::quantized_block_element_index_out_of_range,
            static_cast<std::uint64_t>(element_count),
            static_cast<std::uint64_t>(index)
        );
    }
}

}  // namespace

Q8_0BlockView::Q8_0BlockView(
    std::uint16_t scale_bits,
    std::span<const std::uint8_t> quantized_bytes
)
    : scale_bits_(scale_bits),
      quantized_bytes_(quantized_bytes) {
    require_span_size(quantized_bytes_, element_count);
}

std::uint16_t Q8_0BlockView::scale_bits() const noexcept {
    return scale_bits_;
}

float Q8_0BlockView::scale() const noexcept {
    return f16_bits_to_f32(scale_bits_);
}

std::span<const std::uint8_t> Q8_0BlockView::quantized_bytes() const noexcept {
    return quantized_bytes_;
}

std::int8_t Q8_0BlockView::quantized_value(std::size_t index) const {
    require_block_element_index(index, element_count);

    return std::bit_cast<std::int8_t>(quantized_bytes_[index]);
}

float Q8_0BlockView::dequantized_value(std::size_t index) const {
    return scale() * static_cast<float>(quantized_value(index));
}

Q4_KBlockView::Q4_KBlockView(
    std::uint16_t scale_bits,
    std::uint16_t min_scale_bits,
    std::span<const std::uint8_t> scale_min_bytes,
    std::span<const std::uint8_t> quantized_bytes
)
    : scale_bits_(scale_bits),
      min_scale_bits_(min_scale_bits),
      scale_min_bytes_(scale_min_bytes),
      quantized_bytes_(quantized_bytes) {
    require_span_size(scale_min_bytes_, scale_min_byte_count);
    require_span_size(quantized_bytes_, quantized_byte_count);
}

std::uint16_t Q4_KBlockView::scale_bits() const noexcept {
    return scale_bits_;
}

std::uint16_t Q4_KBlockView::min_scale_bits() const noexcept {
    return min_scale_bits_;
}

float Q4_KBlockView::scale() const noexcept {
    return f16_bits_to_f32(scale_bits_);
}

float Q4_KBlockView::min_scale() const noexcept {
    return f16_bits_to_f32(min_scale_bits_);
}

std::span<const std::uint8_t> Q4_KBlockView::scale_min_bytes() const noexcept {
    return scale_min_bytes_;
}

std::span<const std::uint8_t> Q4_KBlockView::quantized_bytes() const noexcept {
    return quantized_bytes_;
}

std::uint8_t Q4_KBlockView::quantized_nibble(std::size_t element_index) const {
    require_block_element_index(element_index, element_count);

    const auto byte = quantized_bytes_[element_index / 2];
    const auto shift = (element_index % 2) * 4U;
    return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

Q6_KBlockView::Q6_KBlockView(
    std::span<const std::uint8_t> low_bit_bytes,
    std::span<const std::uint8_t> high_bit_bytes,
    std::span<const std::uint8_t> scale_bytes,
    std::uint16_t scale_bits
)
    : low_bit_bytes_(low_bit_bytes),
      high_bit_bytes_(high_bit_bytes),
      scale_bytes_(scale_bytes),
      scale_bits_(scale_bits) {
    require_span_size(low_bit_bytes_, low_bit_byte_count);
    require_span_size(high_bit_bytes_, high_bit_byte_count);
    require_span_size(scale_bytes_, scale_byte_count);
}

std::span<const std::uint8_t> Q6_KBlockView::low_bit_bytes() const noexcept {
    return low_bit_bytes_;
}

std::span<const std::uint8_t> Q6_KBlockView::high_bit_bytes() const noexcept {
    return high_bit_bytes_;
}

std::span<const std::uint8_t> Q6_KBlockView::scale_bytes() const noexcept {
    return scale_bytes_;
}

std::uint16_t Q6_KBlockView::scale_bits() const noexcept {
    return scale_bits_;
}

float Q6_KBlockView::scale() const noexcept {
    return f16_bits_to_f32(scale_bits_);
}

std::uint8_t Q6_KBlockView::low_nibble(std::size_t element_index) const {
    require_block_element_index(element_index, element_count);

    const auto byte = low_bit_bytes_[element_index / 2];
    const auto shift = (element_index % 2) * 4U;
    return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

std::uint8_t Q6_KBlockView::high_bits(std::size_t element_index) const {
    require_block_element_index(element_index, element_count);

    const auto byte = high_bit_bytes_[element_index / 4];
    const auto shift = (element_index % 4) * 2U;
    return static_cast<std::uint8_t>((byte >> shift) & 0x03U);
}

std::int8_t Q6_KBlockView::scale_value(std::size_t group_index) const {
    require_block_element_index(group_index, scale_byte_count);
    return std::bit_cast<std::int8_t>(scale_bytes_[group_index]);
}

std::uint64_t q8_0_block_count(const TensorDescriptor& descriptor) {
    require_quantized_descriptor(descriptor, DType::q8_0);
    return descriptor.element_count / Q8_0BlockView::element_count;
}

Q8_0BlockView q8_0_block_view(const TensorStorage& storage, std::uint64_t block_index) {
    const auto block_count = q8_0_block_count(storage.descriptor());
    if (block_index >= block_count) {
        throw TensorError(TensorErrorCode::quantized_block_index_out_of_range, block_count, block_index);
    }

    const auto byte_offset = checked_quantized_block_byte_offset(block_index, Q8_0BlockView::byte_count);
    const auto block_bytes = storage.bytes().subspan(byte_offset, Q8_0BlockView::byte_count);
    return Q8_0BlockView(
        read_little_endian_u16(block_bytes.first<2>()),
        block_bytes.subspan(2, Q8_0BlockView::element_count)
    );
}

std::array<float, Q8_0BlockView::element_count> dequantize_q8_0_block(
    const TensorStorage& storage,
    std::uint64_t block_index
) {
    const auto block = q8_0_block_view(storage, block_index);

    std::array<float, Q8_0BlockView::element_count> values {};
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = block.dequantized_value(index);
    }

    return values;
}

std::uint64_t q4_k_block_count(const TensorDescriptor& descriptor) {
    require_quantized_descriptor(descriptor, DType::q4_k);
    return descriptor.element_count / Q4_KBlockView::element_count;
}

Q4_KBlockView q4_k_block_view(const TensorStorage& storage, std::uint64_t block_index) {
    const auto block_count = q4_k_block_count(storage.descriptor());
    if (block_index >= block_count) {
        throw TensorError(TensorErrorCode::quantized_block_index_out_of_range, block_count, block_index);
    }

    const auto byte_offset = checked_quantized_block_byte_offset(block_index, Q4_KBlockView::byte_count);
    const auto block_bytes = storage.bytes().subspan(byte_offset, Q4_KBlockView::byte_count);
    return Q4_KBlockView(
        read_little_endian_u16(block_bytes.first<2>()),
        read_little_endian_u16(block_bytes.subspan<2, 2>()),
        block_bytes.subspan(4, Q4_KBlockView::scale_min_byte_count),
        block_bytes.subspan(16, Q4_KBlockView::quantized_byte_count)
    );
}

std::uint64_t q6_k_block_count(const TensorDescriptor& descriptor) {
    require_quantized_descriptor(descriptor, DType::q6_k);
    return descriptor.element_count / Q6_KBlockView::element_count;
}

Q6_KBlockView q6_k_block_view(const TensorStorage& storage, std::uint64_t block_index) {
    const auto block_count = q6_k_block_count(storage.descriptor());
    if (block_index >= block_count) {
        throw TensorError(TensorErrorCode::quantized_block_index_out_of_range, block_count, block_index);
    }

    const auto byte_offset = checked_quantized_block_byte_offset(block_index, Q6_KBlockView::byte_count);
    const auto block_bytes = storage.bytes().subspan(byte_offset, Q6_KBlockView::byte_count);
    return Q6_KBlockView(
        block_bytes.subspan(0, Q6_KBlockView::low_bit_byte_count),
        block_bytes.subspan(Q6_KBlockView::low_bit_byte_count, Q6_KBlockView::high_bit_byte_count),
        block_bytes.subspan(
            Q6_KBlockView::low_bit_byte_count + Q6_KBlockView::high_bit_byte_count,
            Q6_KBlockView::scale_byte_count
        ),
        read_little_endian_u16(block_bytes.last<2>())
    );
}

}  // namespace local_gguf_engine::tensor
