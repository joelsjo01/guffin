#include "local_gguf_engine/tensor/quantization.hpp"

#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/kquant_packing.hpp"

#include "kquant_packing_internal.hpp"
#include "weight_preparation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>


namespace local_gguf_engine::tensor {

namespace {

void require_quantized_descriptor(const TensorDescriptor& descriptor, DType expected_dtype) {
    if (descriptor.packing != TensorPacking::canonical) {
        throw std::invalid_argument("native quantized block access requires canonical tensor packing");
    }
    if (descriptor.dtype != expected_dtype) {
        throw TensorError(
            TensorErrorCode::unsupported_quantized_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(expected_dtype)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

TensorDescriptor q8_0_conversion_descriptor(TensorView source) {
    const auto& source_descriptor = source.descriptor();
    if (source_descriptor.dtype != DType::f16) {
        throw TensorError(
            TensorErrorCode::unsupported_operator_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f16)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(source_descriptor.dtype))
        );
    }
    if (source_descriptor.shape.rank() != 2U) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            2U,
            static_cast<std::uint64_t>(source_descriptor.shape.rank())
        );
    }
    const auto shared = source_descriptor.shape.dimension(0);
    if (shared % Q8_0BlockView::element_count != 0U) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            Q8_0BlockView::element_count,
            shared
        );
    }

    return TensorDescriptor::from_quantized_gguf_weight_matrix(source_descriptor.shape, DType::q8_0);
}

TensorDescriptor q6_k_conversion_descriptor(TensorView source) {
    const auto& descriptor = source.descriptor();
    require_quantized_descriptor(descriptor, DType::q6_k);
    if (descriptor.shape.rank() != 2U) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            2U,
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
    const auto shared = descriptor.shape.dimension(0);
    if (shared % Q6_KBlockView::element_count != 0U) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            Q6_KBlockView::element_count,
            shared
        );
    }

    return TensorDescriptor::from_quantized_gguf_weight_matrix(descriptor.shape, DType::q4_k);
}

[[nodiscard]] std::uint16_t read_little_endian_u16(std::span<const std::uint8_t> bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8);
}

void write_little_endian_u16(std::span<std::uint8_t> bytes, std::uint16_t value) noexcept {
    bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

// This is the rounding primitive used by the pinned GGML Q4_K encoder. The
// input bound is far above any value produced by a finite F16 Q6_K block.
[[nodiscard]] int nearest_int(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value + 12582912.0F);
    return static_cast<int>(bits & 0x007FFFFFU) - 0x00400000;
}

[[nodiscard]] std::uint8_t q4_k_scale_min_value(
    std::span<const std::uint8_t> scale_min_bytes,
    std::size_t group_index,
    bool min_value
);

[[nodiscard]] float make_q4_k_group(
    std::span<const float, 32> values,
    std::span<const float, 32> weights,
    std::span<std::uint8_t, 32> levels,
    float& minimum_magnitude
) {
    float minimum = values[0];
    float maximum = values[0];
    float weight_sum = weights[0];
    float weighted_value_sum = weights[0] * values[0];
    for (std::size_t index = 1; index < values.size(); ++index) {
        minimum = std::min(minimum, values[index]);
        maximum = std::max(maximum, values[index]);
        weight_sum += weights[index];
        weighted_value_sum += weights[index] * values[index];
    }
    minimum = std::min(minimum, 0.0F);
    if (maximum == minimum) {
        std::fill(levels.begin(), levels.end(), 0U);
        minimum_magnitude = -minimum;
        return 0.0F;
    }

    constexpr int maximum_level = 15;
    float inverse_scale = maximum_level / (maximum - minimum);
    float scale = 1.0F / inverse_scale;
    float best_error = 0.0F;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto level = std::clamp(nearest_int(inverse_scale * (values[index] - minimum)), 0, maximum_level);
        levels[index] = static_cast<std::uint8_t>(level);
        const auto difference = scale * static_cast<float>(level) + minimum - values[index];
        best_error += weights[index] * difference * difference;
    }

    std::array<std::uint8_t, 32> candidate_levels {};
    for (int step = 0; step <= 20; ++step) {
        inverse_scale = (-1.0F + 0.1F * static_cast<float>(step) + maximum_level)
            / (maximum - minimum);
        float weighted_level_sum = 0.0F;
        float weighted_level_squared_sum = 0.0F;
        float weighted_level_value_sum = 0.0F;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto level = std::clamp(
                nearest_int(inverse_scale * (values[index] - minimum)), 0, maximum_level
            );
            candidate_levels[index] = static_cast<std::uint8_t>(level);
            const auto weighted_level = weights[index] * static_cast<float>(level);
            weighted_level_sum += weighted_level;
            weighted_level_squared_sum += weighted_level * static_cast<float>(level);
            weighted_level_value_sum += weighted_level * values[index];
        }
        const auto determinant = weight_sum * weighted_level_squared_sum
            - weighted_level_sum * weighted_level_sum;
        if (determinant <= 0.0F) continue;

        auto candidate_scale = (weight_sum * weighted_level_value_sum
            - weighted_value_sum * weighted_level_sum) / determinant;
        auto candidate_minimum = (weighted_level_squared_sum * weighted_value_sum
            - weighted_level_sum * weighted_level_value_sum) / determinant;
        if (candidate_minimum > 0.0F) {
            candidate_minimum = 0.0F;
            candidate_scale = weighted_level_value_sum / weighted_level_squared_sum;
        }
        float candidate_error = 0.0F;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto difference = candidate_scale * static_cast<float>(candidate_levels[index])
                + candidate_minimum - values[index];
            candidate_error += weights[index] * difference * difference;
        }
        if (candidate_error < best_error) {
            levels = candidate_levels;
            best_error = candidate_error;
            scale = candidate_scale;
            minimum = candidate_minimum;
        }
    }
    minimum_magnitude = -minimum;
    return scale;
}

void encode_q4_k_block(
    std::span<const float, Q4_KBlockView::element_count> values,
    std::span<std::uint8_t, Q4_KBlockView::byte_count> destination
) {
    std::array<std::uint8_t, Q4_KBlockView::element_count> levels {};
    std::array<float, 8> scales {};
    std::array<float, 8> minima {};
    float maximum_scale = 0.0F;
    float maximum_minimum = 0.0F;
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const std::span<const float, 32> group_values(values.data() + group * 32U, 32U);
        float sum_squares = 0.0F;
        for (const auto value : group_values) sum_squares += value * value;
        const auto average_magnitude = std::sqrt(sum_squares / 32.0F);
        std::array<float, 32> weights {};
        for (std::size_t index = 0; index < weights.size(); ++index) {
            weights[index] = average_magnitude + std::abs(group_values[index]);
        }
        scales[group] = make_q4_k_group(
            group_values,
            weights,
            std::span<std::uint8_t, 32>(levels.data() + group * 32U, 32U),
            minima[group]
        );
        maximum_scale = std::max(maximum_scale, scales[group]);
        maximum_minimum = std::max(maximum_minimum, minima[group]);
    }

    std::fill(destination.begin(), destination.end(), 0U);
    const auto inverse_scale = maximum_scale > 0.0F ? 63.0F / maximum_scale : 0.0F;
    const auto inverse_minimum = maximum_minimum > 0.0F ? 63.0F / maximum_minimum : 0.0F;
    auto packed = destination.subspan<4, Q4_KBlockView::scale_min_byte_count>();
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto scale = static_cast<std::uint8_t>(std::min(63, nearest_int(inverse_scale * scales[group])));
        const auto minimum = static_cast<std::uint8_t>(std::min(63, nearest_int(inverse_minimum * minima[group])));
        if (group < 4U) {
            packed[group] = scale;
            packed[group + 4U] = minimum;
        } else {
            packed[group + 4U] = static_cast<std::uint8_t>((scale & 0x0FU) | ((minimum & 0x0FU) << 4U));
            packed[group - 4U] |= static_cast<std::uint8_t>((scale >> 4U) << 6U);
            packed[group] |= static_cast<std::uint8_t>((minimum >> 4U) << 6U);
        }
    }
    const auto scale_bits = f32_to_f16_bits(maximum_scale / 63.0F);
    const auto minimum_bits = f32_to_f16_bits(maximum_minimum / 63.0F);
    write_little_endian_u16(destination.first<2>(), scale_bits);
    write_little_endian_u16(destination.subspan<2, 2>(), minimum_bits);

    const auto stored_scale = f16_bits_to_f32(scale_bits);
    const auto stored_minimum = f16_bits_to_f32(minimum_bits);
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto group_scale = stored_scale * q4_k_scale_min_value(packed, group, false);
        if (group_scale == 0.0F) continue;
        const auto group_minimum = stored_minimum * q4_k_scale_min_value(packed, group, true);
        for (std::size_t index = 0; index < 32U; ++index) {
            levels[group * 32U + index] = static_cast<std::uint8_t>(std::clamp(
                nearest_int((values[group * 32U + index] + group_minimum) / group_scale), 0, 15
            ));
        }
    }
    auto quantized = destination.subspan<16, Q4_KBlockView::quantized_byte_count>();
    for (std::size_t group = 0; group < 4U; ++group) {
        const auto base = group * 64U;
        for (std::size_t index = 0; index < 32U; ++index) {
            quantized[group * 32U + index] = static_cast<std::uint8_t>(
                levels[base + index] | (levels[base + index + 32U] << 4U)
            );
        }
    }
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

TensorStorage quantize_f16_gguf_weight_q8_0(const TensorView& source) {
    auto destination = TensorStorage::allocate(q8_0_conversion_descriptor(source));
    quantize_f16_gguf_weight_q8_0_into(source, destination.descriptor(), destination.mutable_bytes());
    return destination;
}

void quantize_f16_gguf_weight_q8_0_into(TensorView source,
    const TensorDescriptor& destination_descriptor, std::span<std::uint8_t> destination_bytes) {
    const auto expected = q8_0_conversion_descriptor(source);
    detail::validate_preparation_destination(source, destination_descriptor,
        expected, destination_bytes);
    const auto source_bytes = source.bytes();
    const auto block_count = destination_bytes.size() / Q8_0BlockView::byte_count;
    detail::prepare_weight_ranges(block_count, 4096U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t block_index = begin; block_index < end; ++block_index) {
            std::array<float, Q8_0BlockView::element_count> values {};
            float absolute_maximum = 0.0F;
            for (std::size_t element = 0; element < values.size(); ++element) {
                const auto linear_index = block_index * values.size() + element;
                const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(std::uint16_t));
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::uint16_t>(source_bytes[byte_offset])
                    | static_cast<std::uint16_t>(static_cast<std::uint16_t>(source_bytes[byte_offset + 1U]) << 8U)
                );
                values[element] = f16_bits_to_f32(bits);
                if (!std::isfinite(values[element])) {
                    throw std::invalid_argument("Q8_0 F16 weight quantization requires finite values");
                }
                absolute_maximum = std::max(absolute_maximum, std::abs(values[element]));
            }

            const float scale = absolute_maximum / 127.0F;
            const float inverse_scale = scale == 0.0F ? 0.0F : 1.0F / scale;
            const auto scale_bits = f32_to_f16_bits(scale);
            const auto destination_offset = static_cast<std::size_t>(block_index * Q8_0BlockView::byte_count);
            destination_bytes[destination_offset] = static_cast<std::uint8_t>(scale_bits & 0xFFU);
            destination_bytes[destination_offset + 1U] = static_cast<std::uint8_t>(scale_bits >> 8U);
            for (std::size_t element = 0; element < values.size(); ++element) {
                const auto rounded = std::round(values[element] * inverse_scale);
                const auto quantized = static_cast<std::int8_t>(rounded);
                destination_bytes[destination_offset + 2U + element] = std::bit_cast<std::uint8_t>(quantized);
            }
        }
    });
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

TensorStorage requantize_q6_k_gguf_weight_q4_k(const TensorView& source) {
    auto destination = TensorStorage::allocate(q6_k_conversion_descriptor(source));
    requantize_q6_k_gguf_weight_q4_k_into(source, destination.descriptor(), destination.mutable_bytes());
    return destination;
}

void requantize_q6_k_gguf_weight_q4_k_into(TensorView source,
    const TensorDescriptor& destination_descriptor, std::span<std::uint8_t> destination_bytes) {
    const auto native_descriptor = q6_k_conversion_descriptor(source);
    const auto& descriptor = source.descriptor();
    const auto shared = descriptor.shape.dimension(0);
    const bool packed = destination_descriptor.packing == TensorPacking::kquant_columns8;
    detail::validate_preparation_destination(source, destination_descriptor,
        packed ? TensorDescriptor::from_interleaved_kquant_weight_matrix(descriptor.shape, DType::q4_k)
               : native_descriptor,
        destination_bytes);
    const auto source_bytes = source.bytes();
    const auto convert_block = [&](std::size_t block_index,
        std::span<std::uint8_t, Q4_KBlockView::byte_count> output) {
        const auto source_offset = static_cast<std::size_t>(block_index * Q6_KBlockView::byte_count);
        const auto source_block = source_bytes.subspan(source_offset, Q6_KBlockView::byte_count);
        const Q6_KBlockView block(
            source_block.first<Q6_KBlockView::low_bit_byte_count>(),
            source_block.subspan<Q6_KBlockView::low_bit_byte_count, Q6_KBlockView::high_bit_byte_count>(),
            source_block.subspan<
                Q6_KBlockView::low_bit_byte_count + Q6_KBlockView::high_bit_byte_count,
                Q6_KBlockView::scale_byte_count
            >(),
            read_little_endian_u16(source_block.last<2>())
        );
        std::array<float, Q6_KBlockView::element_count> values {};
        const auto scale = block.scale();
        for (std::size_t group = 0U; group < Q6_KBlockView::scale_byte_count; ++group) {
            const auto group_scale = scale * static_cast<float>(block.scale_value(group));
            for (std::size_t lane = 0U; lane < 16U; ++lane) {
                const auto element = group * 16U + lane;
                values[element] = group_scale * static_cast<float>(block.quantized_value(element));
                if (!std::isfinite(values[element])) {
                    throw std::invalid_argument("Q6_K to Q4_K requantization requires finite decoded values");
                }
            }
        }
        encode_q4_k_block(values, output);
    };
    if (!packed) {
        const auto block_count = destination_bytes.size() / Q4_KBlockView::byte_count;
        detail::prepare_weight_ranges(block_count, 4096U, [&](std::size_t begin, std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                convert_block(index, std::span<std::uint8_t, Q4_KBlockView::byte_count>(
                    destination_bytes.data() + index * Q4_KBlockView::byte_count, Q4_KBlockView::byte_count));
            }
        });
        return;
    }

    const auto blocks = static_cast<std::size_t>(shared / Q6_KBlockView::element_count);
    const auto columns = static_cast<std::size_t>(descriptor.shape.dimension(1));
    const auto tile_count = destination_bytes.size() / kquant_q4_columns8_tile_bytes;
    // One worker owns each complete packed tile, including partial-column padding.
    detail::prepare_weight_ranges(tile_count, 512U, [&](std::size_t begin, std::size_t end) {
        std::array<std::uint8_t, Q4_KBlockView::byte_count> native {};
        for (std::size_t index = begin; index < end; ++index) {
            const auto first_column = index / blocks * kquant_packing_columns;
            const auto block = index % blocks;
            const auto lanes = std::min(kquant_packing_columns, columns - first_column);
            auto tile = destination_bytes.subspan(index * kquant_q4_columns8_tile_bytes,
                kquant_q4_columns8_tile_bytes);
            if (lanes < kquant_packing_columns) std::fill(tile.begin(), tile.end(), 0U);
            for (std::size_t lane = 0U; lane < lanes; ++lane) {
                convert_block((first_column + lane) * blocks + block, native);
                detail::pack_q4_k_columns8_lane(native, tile, lane);
            }
        }
    });
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
