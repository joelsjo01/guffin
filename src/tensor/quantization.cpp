#include "local_gguf_engine/tensor/quantization.hpp"

#include "local_gguf_engine/tensor/f16.hpp"

#include <bit>
#include <limits>
#include <numeric>
#include <vector>

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

void require_block_group_index(std::size_t index, std::size_t group_count) {
    if (index >= group_count) {
        throw TensorError(
            TensorErrorCode::quantized_block_element_index_out_of_range,
            static_cast<std::uint64_t>(group_count),
            static_cast<std::uint64_t>(index)
        );
    }
}

void require_quantized_matrix_shape(const TensorDescriptor& descriptor) {
    if (descriptor.shape.rank() != 2) {
        throw TensorError(
            TensorErrorCode::invalid_tensor_index_rank,
            2,
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_row_index(const TensorDescriptor& descriptor, std::uint64_t row_index) {
    const auto row_count = descriptor.shape.dimension(0);
    if (row_index >= row_count) {
        throw TensorError(TensorErrorCode::tensor_index_out_of_range, row_count, row_index);
    }
}

void require_input_width(std::span<const float> input, std::uint64_t width) {
    if (static_cast<std::uint64_t>(input.size()) != width) {
        throw TensorError(
            TensorErrorCode::tensor_shape_mismatch,
            width,
            static_cast<std::uint64_t>(input.size())
        );
    }
}

[[nodiscard]] std::uint8_t q4_k_scale_min_value(
    std::span<const std::uint8_t> scale_min_bytes,
    std::size_t group_index,
    bool min_value
) {
    require_block_group_index(group_index, Q4_KBlockView::element_count / 32U);

    if (group_index < 4U) {
        return static_cast<std::uint8_t>(scale_min_bytes[group_index + (min_value ? 4U : 0U)] & 0x3FU);
    }

    const auto low_source = scale_min_bytes[group_index + 4U];
    if (min_value) {
        const auto high_bits = static_cast<std::uint8_t>((scale_min_bytes[group_index] & 0xC0U) >> 2U);
        return static_cast<std::uint8_t>((low_source >> 4U) | high_bits);
    }

    const auto high_bits = static_cast<std::uint8_t>((scale_min_bytes[group_index - 4U] & 0xC0U) >> 2U);
    return static_cast<std::uint8_t>((low_source & 0x0FU) | high_bits);
}

[[nodiscard]] std::uint64_t quantized_matrix_block_index(
    const TensorDescriptor& descriptor,
    std::uint64_t row_index,
    std::uint64_t column_block_index,
    std::uint64_t block_element_count
) {
    const auto column_count = descriptor.shape.dimension(1);
    const auto row_block_count = column_count / block_element_count;
    if (row_index > std::numeric_limits<std::uint64_t>::max() / row_block_count) {
        throw TensorError(TensorErrorCode::byte_size_overflow, 0, row_index);
    }

    return row_index * row_block_count + column_block_index;
}

template <typename BlockView>
[[nodiscard]] std::array<float, BlockView::element_count> dequantize_block_values(const BlockView& block) {
    std::array<float, BlockView::element_count> values {};
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = block.dequantized_value(index);
    }

    return values;
}

template <typename BlockView, typename BlockLoader>
[[nodiscard]] std::vector<float> dequantize_quantized_row(
    const TensorStorage& storage,
    std::uint64_t row_index,
    DType dtype,
    BlockLoader block_loader
) {
    require_quantized_descriptor(storage.descriptor(), dtype);
    require_quantized_matrix_shape(storage.descriptor());
    require_row_index(storage.descriptor(), row_index);

    const auto column_count = storage.descriptor().shape.dimension(1);
    std::vector<float> row_values;
    row_values.reserve(static_cast<std::size_t>(column_count));

    const auto row_block_count = column_count / BlockView::element_count;
    for (std::uint64_t block_index = 0; block_index < row_block_count; ++block_index) {
        const auto block = block_loader(
            storage,
            quantized_matrix_block_index(storage.descriptor(), row_index, block_index, BlockView::element_count)
        );
        for (std::size_t element_index = 0; element_index < BlockView::element_count; ++element_index) {
            row_values.push_back(block.dequantized_value(element_index));
        }
    }

    return row_values;
}

template <typename RowDequantizer>
[[nodiscard]] float dot_quantized_row_f32(
    const TensorStorage& storage,
    std::uint64_t row_index,
    std::span<const float> input,
    RowDequantizer row_dequantizer
) {
    require_quantized_matrix_shape(storage.descriptor());
    require_row_index(storage.descriptor(), row_index);
    require_input_width(input, storage.descriptor().shape.dimension(1));

    const auto row_values = row_dequantizer(storage, row_index);
    return std::inner_product(row_values.begin(), row_values.end(), input.begin(), 0.0F);
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

std::uint8_t Q4_KBlockView::scale_value(std::size_t group_index) const {
    return q4_k_scale_min_value(scale_min_bytes_, group_index, false);
}

std::uint8_t Q4_KBlockView::min_value(std::size_t group_index) const {
    return q4_k_scale_min_value(scale_min_bytes_, group_index, true);
}

std::uint8_t Q4_KBlockView::quantized_nibble(std::size_t element_index) const {
    require_block_element_index(element_index, element_count);

    const auto super_group_base = (element_index / 64U) * 32U;
    const auto index_in_super_group = element_index % 64U;
    const auto byte = quantized_bytes_[super_group_base + (index_in_super_group % 32U)];
    const auto shift = index_in_super_group < 32U ? 0U : 4U;
    return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

float Q4_KBlockView::dequantized_value(std::size_t element_index) const {
    const auto group_index = element_index / 32U;
    return scale() * static_cast<float>(scale_value(group_index)) * static_cast<float>(quantized_nibble(element_index))
        - min_scale() * static_cast<float>(min_value(group_index));
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

    const auto half_block_base = (element_index / 128U) * 64U;
    const auto index_in_half_block = element_index % 128U;
    const auto lane_index = index_in_half_block % 32U;
    const auto low_plane_offset = ((index_in_half_block / 32U) % 2U) * 32U;
    const auto byte = low_bit_bytes_[half_block_base + low_plane_offset + lane_index];
    const auto shift = index_in_half_block < 64U ? 0U : 4U;
    return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

std::uint8_t Q6_KBlockView::high_bits(std::size_t element_index) const {
    require_block_element_index(element_index, element_count);

    const auto half_block_base = (element_index / 128U) * 32U;
    const auto index_in_half_block = element_index % 128U;
    const auto byte = high_bit_bytes_[half_block_base + (index_in_half_block % 32U)];
    const auto shift = (index_in_half_block / 32U) * 2U;
    return static_cast<std::uint8_t>((byte >> shift) & 0x03U);
}

std::int8_t Q6_KBlockView::quantized_value(std::size_t element_index) const {
    const auto value = static_cast<std::uint8_t>(low_nibble(element_index) | (high_bits(element_index) << 4U));
    return static_cast<std::int8_t>(static_cast<std::int16_t>(value) - 32);
}

std::int8_t Q6_KBlockView::scale_value(std::size_t group_index) const {
    require_block_group_index(group_index, scale_byte_count);
    return std::bit_cast<std::int8_t>(scale_bytes_[group_index]);
}

float Q6_KBlockView::dequantized_value(std::size_t element_index) const {
    const auto group_index = element_index / 16U;
    return scale() * static_cast<float>(scale_value(group_index)) * static_cast<float>(quantized_value(element_index));
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

std::array<float, Q4_KBlockView::element_count> dequantize_q4_k_block(
    const TensorStorage& storage,
    std::uint64_t block_index
) {
    return dequantize_block_values(q4_k_block_view(storage, block_index));
}

std::vector<float> dequantize_q4_k_row(const TensorStorage& storage, std::uint64_t row_index) {
    return dequantize_quantized_row<Q4_KBlockView>(storage, row_index, DType::q4_k, q4_k_block_view);
}

float dot_q4_k_row_f32(
    const TensorStorage& storage,
    std::uint64_t row_index,
    std::span<const float> input
) {
    require_quantized_descriptor(storage.descriptor(), DType::q4_k);
    return dot_quantized_row_f32(storage, row_index, input, dequantize_q4_k_row);
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

std::array<float, Q6_KBlockView::element_count> dequantize_q6_k_block(
    const TensorStorage& storage,
    std::uint64_t block_index
) {
    return dequantize_block_values(q6_k_block_view(storage, block_index));
}

std::vector<float> dequantize_q6_k_row(const TensorStorage& storage, std::uint64_t row_index) {
    return dequantize_quantized_row<Q6_KBlockView>(storage, row_index, DType::q6_k, q6_k_block_view);
}

float dot_q6_k_row_f32(
    const TensorStorage& storage,
    std::uint64_t row_index,
    std::span<const float> input
) {
    require_quantized_descriptor(storage.descriptor(), DType::q6_k);
    return dot_quantized_row_f32(storage, row_index, input, dequantize_q6_k_row);
}

}  // namespace local_gguf_engine::tensor
