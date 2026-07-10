#include "tensor_test_support.hpp"

#include "test_support.hpp"

#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

namespace local_gguf_engine::tests {

using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::f32_to_f16_bits;
using local_gguf_engine::tensor::q4_k_block_view;
using local_gguf_engine::tensor::q6_k_block_view;
using local_gguf_engine::tensor::Q4_KBlockView;
using local_gguf_engine::tensor::Q6_KBlockView;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::write_f16_from_f32;
using local_gguf_engine::tensor::write_f32;

void expect_close(float actual, float expected, float tolerance, std::string_view message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream stream;
        stream << message << ": expected " << expected << ", got " << actual;
        throw TestFailure(stream.str());
    }
}

TensorStorage tensor_f32_vector(std::span<const float> values) {
    auto storage = TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({static_cast<std::uint64_t>(values.size())}), DType::f32)
    );
    for (std::uint64_t index = 0; index < values.size(); ++index) {
        const std::array<std::uint64_t, 1> tensor_index {index};
        write_f32(storage, std::span<const std::uint64_t>(tensor_index), values[static_cast<std::size_t>(index)]);
    }

    return storage;
}

TensorStorage tensor_f32_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F32 matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> tensor_index {row, column};
            write_f32(
                storage,
                std::span<const std::uint64_t>(tensor_index),
                values[static_cast<std::size_t>(row * columns + column)]
            );
        }
    }

    return storage;
}

TensorStorage tensor_f16_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F16 matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> tensor_index {row, column};
            write_f16_from_f32(
                storage,
                std::span<const std::uint64_t>(tensor_index),
                values[static_cast<std::size_t>(row * columns + column)]
            );
        }
    }

    return storage;
}

TensorStorage tensor_f16_gguf_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F16 GGUF matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    auto bytes = storage.mutable_bytes();
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto bits = f32_to_f16_bits(values[static_cast<std::size_t>(row * columns + column)]);
            const auto byte_offset = static_cast<std::size_t>((column * rows + row) * sizeof(std::uint16_t));
            bytes[byte_offset] = static_cast<std::uint8_t>(bits & 0xFFU);
            bytes[byte_offset + 1] = static_cast<std::uint8_t>(bits >> 8U);
        }
    }

    return storage;
}

namespace {

void write_little_endian_u16(std::span<std::uint8_t> bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[offset + 1U] = static_cast<std::uint8_t>(value >> 8U);
}

void set_q4_k_scale_min(
    std::span<std::uint8_t> block_bytes,
    std::size_t group_index,
    std::uint8_t scale,
    std::uint8_t min
) {
    auto scale_min = block_bytes.subspan(4, Q4_KBlockView::scale_min_byte_count);
    scale = static_cast<std::uint8_t>(scale & 0x3FU);
    min = static_cast<std::uint8_t>(min & 0x3FU);

    if (group_index < 4U) {
        scale_min[group_index] = static_cast<std::uint8_t>((scale_min[group_index] & 0xC0U) | scale);
        scale_min[group_index + 4U] = static_cast<std::uint8_t>((scale_min[group_index + 4U] & 0xC0U) | min);
        return;
    }

    const auto scale_high_bits = static_cast<std::uint8_t>((scale & 0x30U) << 2U);
    const auto min_high_bits = static_cast<std::uint8_t>((min & 0x30U) << 2U);
    scale_min[group_index - 4U] =
        static_cast<std::uint8_t>((scale_min[group_index - 4U] & 0x3FU) | scale_high_bits);
    scale_min[group_index] = static_cast<std::uint8_t>((scale_min[group_index] & 0x3FU) | min_high_bits);
    scale_min[group_index + 4U] = static_cast<std::uint8_t>((scale & 0x0FU) | ((min & 0x0FU) << 4U));
}

void set_q4_k_quantized_nibble(
    std::span<std::uint8_t> block_bytes,
    std::size_t element_index,
    std::uint8_t value
) {
    auto quantized = block_bytes.subspan(16, Q4_KBlockView::quantized_byte_count);
    const auto super_group_base = (element_index / 64U) * 32U;
    const auto index_in_super_group = element_index % 64U;
    const auto byte_index = super_group_base + (index_in_super_group % 32U);
    const auto shift = index_in_super_group < 32U ? 0U : 4U;
    const auto mask = static_cast<std::uint8_t>(0x0FU << shift);
    quantized[byte_index] = static_cast<std::uint8_t>(
        (quantized[byte_index] & ~mask) | ((value & 0x0FU) << shift)
    );
}

void set_q6_k_scale(std::span<std::uint8_t> block_bytes, std::size_t group_index, std::int8_t value) {
    block_bytes[Q6_KBlockView::low_bit_byte_count + Q6_KBlockView::high_bit_byte_count + group_index] =
        std::bit_cast<std::uint8_t>(value);
}

void set_q6_k_quantized_value(
    std::span<std::uint8_t> block_bytes,
    std::size_t element_index,
    std::int8_t value
) {
    const auto encoded = static_cast<std::uint8_t>(static_cast<std::int16_t>(value) + 32);
    const auto low = static_cast<std::uint8_t>(encoded & 0x0FU);
    const auto high = static_cast<std::uint8_t>((encoded >> 4U) & 0x03U);
    const auto half_block_index = element_index / 128U;
    const auto index_in_half_block = element_index % 128U;
    const auto lane_index = index_in_half_block % 32U;

    const auto low_base = half_block_index * 64U;
    const auto low_plane_offset = ((index_in_half_block / 32U) % 2U) * 32U;
    const auto low_shift = index_in_half_block < 64U ? 0U : 4U;
    const auto low_mask = static_cast<std::uint8_t>(0x0FU << low_shift);
    auto& low_byte = block_bytes[low_base + low_plane_offset + lane_index];
    low_byte = static_cast<std::uint8_t>((low_byte & ~low_mask) | (low << low_shift));

    const auto high_base = Q6_KBlockView::low_bit_byte_count + half_block_index * 32U;
    const auto high_shift = (index_in_half_block / 32U) * 2U;
    const auto high_mask = static_cast<std::uint8_t>(0x03U << high_shift);
    auto& high_byte = block_bytes[high_base + lane_index];
    high_byte = static_cast<std::uint8_t>((high_byte & ~high_mask) | (high << high_shift));
}

float read_quantized_gguf_weight_value(
    const TensorStorage& storage,
    std::uint64_t column,
    std::uint64_t inner
) {
    const auto blocks_per_column = storage.descriptor().shape.dimension(0) / Q4_KBlockView::element_count;
    const auto block_index = column * blocks_per_column + (inner / Q4_KBlockView::element_count);
    const auto element_index = static_cast<std::size_t>(inner % Q4_KBlockView::element_count);

    switch (storage.descriptor().dtype) {
        case DType::q4_k:
            return q4_k_block_view(storage, block_index).dequantized_value(element_index);
        case DType::q6_k:
            return q6_k_block_view(storage, block_index).dequantized_value(element_index);
        case DType::f32:
        case DType::f16:
        case DType::q8_0:
            break;
    }

    throw TestFailure("unsupported quantized GGUF weight fixture dtype");
}

}  // namespace

TensorStorage q4_k_reference_tensor() {
    std::array<std::uint8_t, Q4_KBlockView::byte_count> bytes {};
    write_little_endian_u16(std::span<std::uint8_t>(bytes), 0, 0x3C00U);
    write_little_endian_u16(std::span<std::uint8_t>(bytes), 2, 0x3800U);

    set_q4_k_scale_min(std::span<std::uint8_t>(bytes), 0, 2, 1);
    set_q4_k_scale_min(std::span<std::uint8_t>(bytes), 1, 3, 2);
    set_q4_k_scale_min(std::span<std::uint8_t>(bytes), 4, 6, 5);
    set_q4_k_quantized_nibble(std::span<std::uint8_t>(bytes), 0, 4);
    set_q4_k_quantized_nibble(std::span<std::uint8_t>(bytes), 31, 15);
    set_q4_k_quantized_nibble(std::span<std::uint8_t>(bytes), 32, 5);
    set_q4_k_quantized_nibble(std::span<std::uint8_t>(bytes), 63, 6);
    set_q4_k_quantized_nibble(std::span<std::uint8_t>(bytes), 128, 7);

    return TensorStorage::from_bytes(
        TensorDescriptor::from(Shape::from({1, 256}), DType::q4_k),
        std::span<const std::uint8_t>(bytes)
    );
}

TensorStorage q6_k_reference_tensor() {
    std::array<std::uint8_t, Q6_KBlockView::byte_count> bytes {};
    write_little_endian_u16(std::span<std::uint8_t>(bytes), Q6_KBlockView::byte_count - 2U, 0x3800U);

    set_q6_k_scale(std::span<std::uint8_t>(bytes), 0, 2);
    set_q6_k_scale(std::span<std::uint8_t>(bytes), 1, 3);
    set_q6_k_scale(std::span<std::uint8_t>(bytes), 2, -3);
    set_q6_k_scale(std::span<std::uint8_t>(bytes), 4, 4);
    set_q6_k_scale(std::span<std::uint8_t>(bytes), 8, -2);
    set_q6_k_quantized_value(std::span<std::uint8_t>(bytes), 0, -32);
    set_q6_k_quantized_value(std::span<std::uint8_t>(bytes), 31, 31);
    set_q6_k_quantized_value(std::span<std::uint8_t>(bytes), 32, -1);
    set_q6_k_quantized_value(std::span<std::uint8_t>(bytes), 64, 5);
    set_q6_k_quantized_value(std::span<std::uint8_t>(bytes), 128, 7);

    return TensorStorage::from_bytes(
        TensorDescriptor::from(Shape::from({1, 256}), DType::q6_k),
        std::span<const std::uint8_t>(bytes)
    );
}

TensorStorage tensor_q4_k_gguf_weight_matrix(std::uint64_t shared, std::uint64_t columns, std::uint64_t seed) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({shared, columns}), DType::q4_k));
    auto bytes = storage.mutable_bytes();
    const auto blocks_per_column = shared / Q4_KBlockView::element_count;

    for (std::uint64_t column = 0; column < columns; ++column) {
        for (std::uint64_t column_block = 0; column_block < blocks_per_column; ++column_block) {
            const auto block_index = column * blocks_per_column + column_block;
            auto block = bytes.subspan(
                static_cast<std::size_t>(block_index * Q4_KBlockView::byte_count),
                Q4_KBlockView::byte_count
            );
            write_little_endian_u16(block, 0, 0x3C00U);
            write_little_endian_u16(block, 2, 0x0000U);

            for (std::size_t group = 0; group < 8U; ++group) {
                const auto scale = static_cast<std::uint8_t>(1U + ((seed + column + column_block + group) % 5U));
                set_q4_k_scale_min(block, group, scale, 0);
            }
            for (std::size_t element = 0; element < Q4_KBlockView::element_count; ++element) {
                const auto nibble = static_cast<std::uint8_t>(
                    (seed + column + (3U * column_block) + (5U * element)) % 16U
                );
                set_q4_k_quantized_nibble(block, element, nibble);
            }
        }
    }

    return storage;
}

TensorStorage tensor_q6_k_gguf_weight_matrix(std::uint64_t shared, std::uint64_t columns, std::uint64_t seed) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({shared, columns}), DType::q6_k));
    auto bytes = storage.mutable_bytes();
    const auto blocks_per_column = shared / Q6_KBlockView::element_count;

    for (std::uint64_t column = 0; column < columns; ++column) {
        for (std::uint64_t column_block = 0; column_block < blocks_per_column; ++column_block) {
            const auto block_index = column * blocks_per_column + column_block;
            auto block = bytes.subspan(
                static_cast<std::size_t>(block_index * Q6_KBlockView::byte_count),
                Q6_KBlockView::byte_count
            );
            write_little_endian_u16(block, Q6_KBlockView::byte_count - 2U, 0x3800U);

            for (std::size_t group = 0; group < Q6_KBlockView::scale_byte_count; ++group) {
                const auto raw = static_cast<int>((seed + column + column_block + group) % 7U) - 3;
                const auto scale = static_cast<std::int8_t>(raw == 0 ? 2 : raw);
                set_q6_k_scale(block, group, scale);
            }
            for (std::size_t element = 0; element < Q6_KBlockView::element_count; ++element) {
                const auto quantized = static_cast<std::int8_t>(
                    static_cast<int>((seed + column + (5U * column_block) + (7U * element)) % 31U) - 15
                );
                set_q6_k_quantized_value(block, element, quantized);
            }
        }
    }

    return storage;
}

std::vector<float> tensor_f32_projection_values(std::uint64_t rows, std::uint64_t columns) {
    std::vector<float> values(static_cast<std::size_t>(rows * columns));
    for (std::uint64_t index = 0; index < rows * columns; ++index) {
        const auto raw = static_cast<int>((index * 7U + 3U) % 11U) - 5;
        values[static_cast<std::size_t>(index)] = static_cast<float>(raw) * 0.125F;
    }

    return values;
}

TensorStorage reference_f32_quantized_gguf_weight_matmul(
    const TensorStorage& left,
    const TensorStorage& right
) {
    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::uint64_t inner = 0; inner < shared; ++inner) {
                const std::array<std::uint64_t, 2> left_index {row, inner};
                sum += read_f32(left, std::span<const std::uint64_t>(left_index))
                    * read_quantized_gguf_weight_value(right, column, inner);
            }

            const std::array<std::uint64_t, 2> output_index {row, column};
            write_f32(output, std::span<const std::uint64_t>(output_index), sum);
        }
    }

    return output;
}

}  // namespace local_gguf_engine::tests
