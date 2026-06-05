#include "local_gguf_engine/tensor/quantization.hpp"

#include "local_gguf_engine/tensor/f16.hpp"

#include <bit>
#include <limits>

namespace local_gguf_engine::tensor {

namespace {

void require_q8_0_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.dtype != DType::q8_0) {
        throw TensorError(
            TensorErrorCode::unsupported_quantized_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::q8_0)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

[[nodiscard]] std::uint16_t read_little_endian_u16(std::span<const std::uint8_t> bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8);
}

[[nodiscard]] std::size_t checked_q8_0_block_byte_offset(std::uint64_t block_index) {
    constexpr auto block_byte_count = static_cast<std::uint64_t>(Q8_0BlockView::byte_count);
    if (block_index > std::numeric_limits<std::uint64_t>::max() / block_byte_count) {
        throw TensorError(TensorErrorCode::byte_size_overflow, 0, block_index);
    }

    const auto offset = block_index * block_byte_count;
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw TensorError(TensorErrorCode::byte_size_overflow, 0, offset);
    }

    return static_cast<std::size_t>(offset);
}

}  // namespace

Q8_0BlockView::Q8_0BlockView(
    std::uint16_t scale_bits,
    std::span<const std::uint8_t> quantized_bytes
)
    : scale_bits_(scale_bits),
      quantized_bytes_(quantized_bytes) {
    if (quantized_bytes_.size() != element_count) {
        throw TensorError(
            TensorErrorCode::invalid_storage_byte_count,
            element_count,
            static_cast<std::uint64_t>(quantized_bytes_.size())
        );
    }
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
    if (index >= quantized_bytes_.size()) {
        throw TensorError(
            TensorErrorCode::quantized_block_element_index_out_of_range,
            element_count,
            static_cast<std::uint64_t>(index)
        );
    }

    return std::bit_cast<std::int8_t>(quantized_bytes_[index]);
}

float Q8_0BlockView::dequantized_value(std::size_t index) const {
    return scale() * static_cast<float>(quantized_value(index));
}

std::uint64_t q8_0_block_count(const TensorDescriptor& descriptor) {
    require_q8_0_descriptor(descriptor);
    return descriptor.element_count / Q8_0BlockView::element_count;
}

Q8_0BlockView q8_0_block_view(const TensorStorage& storage, std::uint64_t block_index) {
    const auto block_count = q8_0_block_count(storage.descriptor());
    if (block_index >= block_count) {
        throw TensorError(TensorErrorCode::quantized_block_index_out_of_range, block_count, block_index);
    }

    const auto byte_offset = checked_q8_0_block_byte_offset(block_index);
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

}  // namespace local_gguf_engine::tensor
