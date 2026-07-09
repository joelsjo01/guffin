#include "tensor_backend_tests.hpp"

#include "tensor_test_support.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::allocate_backend_buffer;
using local_gguf_engine::tensor::copy_tensor_from_backend;
using local_gguf_engine::tensor::copy_tensor_to_backend;
using local_gguf_engine::tensor::copy_tensor_view_to_backend;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::q4_k_block_view;
using local_gguf_engine::tensor::q6_k_block_view;
using local_gguf_engine::tensor::Q4_KBlockView;
using local_gguf_engine::tensor::Q6_KBlockView;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::tensor_backend_unavailable_reason;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::add_f32_to_backend;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32_to_backend;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32_with_kv_cache_to_backend;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32_to_backend;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32_with_kv_cache_to_backend;
using local_gguf_engine::tensor::ops::embed_f16_gguf_weight_token_ids_to_backend;
using local_gguf_engine::tensor::ops::gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_triple_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_quantized_gguf_weight;
using local_gguf_engine::tensor::ops::matmul_f32_quantized_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::multiply_f32;
using local_gguf_engine::tensor::ops::residual_gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rms_norm_f32_to_backend;
using local_gguf_engine::tensor::ops::rope_f32_by_heads_with_position_start;
using local_gguf_engine::tensor::ops::rope_f32_by_heads_with_position_start_to_backend;
using local_gguf_engine::tensor::ops::silu_f32;
using local_gguf_engine::tensor::ops::silu_multiply_f32_to_backend;
using local_gguf_engine::tensor::ops::softcap_f32;
using local_gguf_engine::tensor::ops::softcap_f32_to_backend;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::softmax_f32_to_backend;
using local_gguf_engine::tensor::ops::top_logit_f32;
using local_gguf_engine::tensor::ops::top_logit_f32_from_backend;
using local_gguf_engine::tensor::ops::write_prefill_kv_cache_f32_to_backend;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f16_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;

void expect_f32_matrix_close(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    expect(actual.descriptor().dtype == DType::f32, std::string(message) + " dtype mismatch");
    expect(actual.descriptor().shape.rank() == 2, std::string(message) + " rank mismatch");

    const auto rows = actual.descriptor().shape.dimension(0);
    const auto columns = actual.descriptor().shape.dimension(1);
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(actual, std::span<const std::uint64_t>(index)),
                read_f32(expected, std::span<const std::uint64_t>(index)),
                tolerance,
                message
            );
        }
    }
}

void expect_f32_rank3_close(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    expect(actual.descriptor().dtype == DType::f32, std::string(message) + " dtype mismatch");
    expect(actual.descriptor().shape.rank() == 3, std::string(message) + " rank mismatch");

    const auto dim0 = actual.descriptor().shape.dimension(0);
    const auto dim1 = actual.descriptor().shape.dimension(1);
    const auto dim2 = actual.descriptor().shape.dimension(2);
    for (std::uint64_t index0 = 0; index0 < dim0; ++index0) {
        for (std::uint64_t index1 = 0; index1 < dim1; ++index1) {
            for (std::uint64_t index2 = 0; index2 < dim2; ++index2) {
                const std::array<std::uint64_t, 3> index {index0, index1, index2};
                const auto actual_value = read_f32(actual, std::span<const std::uint64_t>(index));
                const auto expected_value = read_f32(expected, std::span<const std::uint64_t>(index));
                if (std::isinf(actual_value) || std::isinf(expected_value)) {
                    expect(
                        std::isinf(actual_value)
                            && std::isinf(expected_value)
                            && std::signbit(actual_value) == std::signbit(expected_value),
                        std::string(message) + " infinity mismatch"
                    );
                    continue;
                }

                expect_close(actual_value, expected_value, tolerance, message);
            }
        }
    }
}

TensorStorage attention_score_softmax_fixture() {
    auto scores = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2, 3}), DType::f32));
    const std::array<float, 12> values {
        -std::numeric_limits<float>::infinity(), 0.0F, 2.0F,
        1.0F, 1.0F, 1.0F,
        -1.0F, 0.0F, 1.0F,
        3.0F, 2.0F, 1.0F,
    };

    std::uint64_t linear_index = 0;
    for (std::uint64_t token = 0; token < 2; ++token) {
        for (std::uint64_t head = 0; head < 2; ++head) {
            for (std::uint64_t column = 0; column < 3; ++column) {
                const std::array<std::uint64_t, 3> index {token, head, column};
                write_f32(scores, std::span<const std::uint64_t>(index), values[linear_index]);
                ++linear_index;
            }
        }
    }

    return scores;
}

TensorStorage grouped_attention_value_probability_fixture() {
    auto probabilities = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2, 2}), DType::f32));
    const std::array<float, 12> values {
        1.0F, 0.0F,
        1.0F, 0.0F,
        0.25F, 0.75F,
        0.5F, 0.5F,
        0.2F, 0.8F,
        1.0F, 0.0F,
    };

    std::uint64_t linear_index = 0;
    for (std::uint64_t token = 0; token < 3; ++token) {
        for (std::uint64_t head = 0; head < 2; ++head) {
            for (std::uint64_t column = 0; column < 2; ++column) {
                const std::array<std::uint64_t, 3> index {token, head, column};
                write_f32(probabilities, std::span<const std::uint64_t>(index), values[linear_index]);
                ++linear_index;
            }
        }
    }

    return probabilities;
}

TensorStorage grouped_attention_value_fixture() {
    const std::array<float, 6> values {
        10.0F, 100.0F,
        20.0F, 200.0F,
        30.0F, 300.0F,
    };
    return tensor_f32_matrix(3, 2, std::span<const float>(values));
}

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

TensorStorage tensor_q4_k_gguf_weight_matrix(std::uint64_t shared, std::uint64_t columns) {
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
                const auto scale = static_cast<std::uint8_t>(1U + ((column + column_block + group) % 5U));
                set_q4_k_scale_min(block, group, scale, 0);
            }
            for (std::size_t element = 0; element < Q4_KBlockView::element_count; ++element) {
                const auto nibble = static_cast<std::uint8_t>((column + (3U * column_block) + (5U * element)) % 16U);
                set_q4_k_quantized_nibble(block, element, nibble);
            }
        }
    }

    return storage;
}

TensorStorage tensor_q6_k_gguf_weight_matrix(std::uint64_t shared, std::uint64_t columns) {
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
                const auto raw = static_cast<int>((column + column_block + group) % 7U) - 3;
                const auto scale = static_cast<std::int8_t>(raw == 0 ? 2 : raw);
                set_q6_k_scale(block, group, scale);
            }
            for (std::size_t element = 0; element < Q6_KBlockView::element_count; ++element) {
                const auto quantized = static_cast<std::int8_t>(
                    static_cast<int>((column + (5U * column_block) + (7U * element)) % 31U) - 15
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

void test_multiplies_f32_by_f16_gguf_weight_matrix_on_cpu_backend() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_matrix(3, 2, std::span<const float>(right_values));

    const auto output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 6.0F, "CPU backend GGUF matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 60.0F, "CPU backend GGUF matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 5.0F, "CPU backend GGUF matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 50.0F, "CPU backend GGUF matmul value 1,1 mismatch");
}

void test_multiplies_f32_by_q4_k_gguf_weight_matrix_on_cpu_backend() {
    constexpr std::uint64_t rows = 2;
    constexpr std::uint64_t shared = 512;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_q4_k_gguf_weight_matrix(shared, columns);

    const auto output = matmul_f32_quantized_gguf_weight(left, right, BackendKind::cpu);
    const auto expected = reference_f32_quantized_gguf_weight_matmul(left, right);

    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend Q4_K GGUF matmul output");
}

void test_multiplies_f32_by_q6_k_gguf_weight_matrix_on_cpu_backend() {
    constexpr std::uint64_t rows = 2;
    constexpr std::uint64_t shared = 512;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_q6_k_gguf_weight_matrix(shared, columns);

    const auto output = matmul_f32_quantized_gguf_weight(left, right, BackendKind::cpu);
    const auto expected = reference_f32_quantized_gguf_weight_matmul(left, right);

    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend Q6_K GGUF matmul output");
}

void test_copies_tensor_to_and_from_cpu_backend_buffer() {
    const std::array<float, 4> values {
        1.25F, -2.5F,
        3.75F, -4.5F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    const auto buffer = copy_tensor_to_backend(storage, BackendKind::cpu);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::cpu, "CPU backend buffer kind mismatch");
    expect(buffer.descriptor() == storage.descriptor(), "CPU backend buffer descriptor mismatch");
    expect(roundtrip.descriptor() == storage.descriptor(), "CPU backend buffer roundtrip descriptor mismatch");
    expect(roundtrip.bytes().data() != storage.bytes().data(), "CPU backend buffer should not alias original storage");
    expect(
        std::equal(storage.bytes().begin(), storage.bytes().end(), roundtrip.bytes().begin()),
        "CPU backend buffer roundtrip bytes mismatch"
    );
}

void test_copies_tensor_view_to_and_from_cpu_backend_buffer() {
    const std::array<float, 4> values {
        1.25F, -2.5F,
        3.75F, -4.5F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    const auto buffer = copy_tensor_view_to_backend(storage.view(), BackendKind::cpu);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::cpu, "CPU backend view buffer kind mismatch");
    expect(buffer.descriptor() == storage.descriptor(), "CPU backend view buffer descriptor mismatch");
    expect(roundtrip.descriptor() == storage.descriptor(), "CPU backend view buffer roundtrip descriptor mismatch");
    expect(
        std::equal(storage.bytes().begin(), storage.bytes().end(), roundtrip.bytes().begin()),
        "CPU backend view buffer roundtrip bytes mismatch"
    );
}

void test_allocates_cpu_backend_buffer() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);

    const auto buffer = allocate_backend_buffer(descriptor, BackendKind::cpu);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::cpu, "CPU allocated backend buffer kind mismatch");
    expect(buffer.descriptor() == descriptor, "CPU allocated backend buffer descriptor mismatch");
    expect(roundtrip.descriptor() == descriptor, "CPU allocated backend buffer roundtrip descriptor mismatch");
    expect(
        std::all_of(roundtrip.bytes().begin(), roundtrip.bytes().end(), [](std::uint8_t byte) {
            return byte == 0;
        }),
        "CPU allocated backend buffer should be zero initialized"
    );
}

void test_multiplies_f32_by_cpu_backend_f16_gguf_weight_buffer() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_matrix(3, 2, std::span<const float>(right_values));
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);

    const auto output = matmul_f32_f16_gguf_weight(left, right_buffer);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 6.0F, "CPU buffered GGUF matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 60.0F, "CPU buffered GGUF matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 5.0F, "CPU buffered GGUF matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 50.0F, "CPU buffered GGUF matmul value 1,1 mismatch");
}

void test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_cpu_backend_buffer() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(3, 2, std::span<const float>(right_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);

    const auto output_buffer = matmul_f32_f16_gguf_weight_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU backend-buffer GGUF matmul output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer GGUF matmul output");
}

void test_embeds_f16_gguf_token_ids_to_cpu_backend_buffer() {
    const std::array<float, 12> embedding_values {
        1.0F, 2.0F, 3.0F, 4.0F,
        10.0F, 20.0F, 30.0F, 40.0F,
        100.0F, 200.0F, 300.0F, 400.0F,
    };
    const std::array<std::uint32_t, 2> token_ids {3, 1};
    const auto embedding = tensor_f16_gguf_matrix(3, 4, std::span<const float>(embedding_values));
    const auto embedding_buffer = copy_tensor_to_backend(embedding, BackendKind::cpu);

    const auto output_buffer = embed_f16_gguf_weight_token_ids_to_backend(
        std::span<const std::uint32_t>(token_ids),
        embedding_buffer,
        0.5F
    );
    const auto output = copy_tensor_from_backend(output_buffer);

    const std::array<float, 6> expected_values {
        2.0F, 20.0F, 200.0F,
        1.0F, 10.0F, 100.0F,
    };
    const auto expected = tensor_f32_matrix(2, 3, std::span<const float>(expected_values));

    expect(output_buffer.backend() == BackendKind::cpu, "CPU token embedding output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU token embedding output");
}

void test_rejects_invalid_f16_gguf_token_embedding_inputs() {
    const std::array<float, 4> embedding_values {
        1.0F, 2.0F,
        10.0F, 20.0F,
    };
    const auto embedding = tensor_f16_gguf_matrix(2, 2, std::span<const float>(embedding_values));
    const auto embedding_buffer = copy_tensor_to_backend(embedding, BackendKind::cpu);

    try {
        (void)embed_f16_gguf_weight_token_ids_to_backend(
            std::span<const std::uint32_t>(),
            embedding_buffer,
            1.0F
        );
        throw TestFailure("token embedding should reject empty token ids");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "token embedding token ids must not be empty",
            "empty token embedding error mismatch"
        );
    }

    const std::array<std::uint32_t, 1> out_of_range_ids {2};
    try {
        (void)embed_f16_gguf_weight_token_ids_to_backend(
            std::span<const std::uint32_t>(out_of_range_ids),
            embedding_buffer,
            1.0F
        );
        throw TestFailure("token embedding should reject out-of-range token id");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_index_out_of_range, "token id range error code mismatch");
        expect(error.expected_value() == 2, "token id range expected value mismatch");
        expect(error.actual_value() == 2, "token id range actual value mismatch");
    }

    const std::array<std::uint32_t, 1> token_ids {1};
    try {
        (void)embed_f16_gguf_weight_token_ids_to_backend(
            std::span<const std::uint32_t>(token_ids),
            embedding_buffer,
            0.0F
        );
        throw TestFailure("token embedding should reject non-positive scale");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "token embedding scale must be positive and finite",
            "token embedding scale error mismatch"
        );
    }

    const auto wrong_dtype = tensor_f32_matrix(2, 2, std::span<const float>(embedding_values));
    const auto wrong_dtype_buffer = copy_tensor_to_backend(wrong_dtype, BackendKind::cpu);
    try {
        (void)embed_f16_gguf_weight_token_ids_to_backend(
            std::span<const std::uint32_t>(token_ids),
            wrong_dtype_buffer,
            1.0F
        );
        throw TestFailure("token embedding should reject non-F16 weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "token embedding dtype error code mismatch"
        );
        expect(
            error.expected_value() == static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f16)),
            "token embedding dtype expected value mismatch"
        );
        expect(
            error.actual_value() == static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f32)),
            "token embedding dtype actual value mismatch"
        );
    }
}

void test_applies_rms_norm_to_cpu_backend_buffers() {
    const std::array<float, 6> input_values {
        2.0F, -4.0F, 8.0F,
        0.5F, -1.5F, 3.0F,
    };
    const std::array<float, 3> weight_values {1.0F, 0.5F, 1.5F};
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto weight = tensor_f32_vector(std::span<const float>(weight_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);
    const auto weight_buffer = copy_tensor_to_backend(weight, BackendKind::cpu);

    const auto output_buffer = rms_norm_f32_to_backend(input_buffer, weight_buffer, 0.25F);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = rms_norm_f32(input, weight, 0.25F);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU RMSNorm output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer RMSNorm output");
}

void test_adds_f32_cpu_backend_buffers() {
    const std::array<float, 6> left_values {
        2.0F, -4.0F, 8.0F,
        0.5F, -1.5F, 3.0F,
    };
    const std::array<float, 6> right_values {
        -1.0F, 0.25F, 1.5F,
        4.0F, -2.0F, 0.75F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f32_matrix(2, 3, std::span<const float>(right_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);

    const auto output_buffer = add_f32_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = add_f32(left, right);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU F32 add output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer F32 add output");
}

void test_applies_headwise_rope_to_cpu_backend_buffer() {
    const std::array<float, 12> input_values {
        1.0F, 0.0F, 9.0F,
        0.0F, 1.0F, 8.0F,
        1.0F, 1.0F, 7.0F,
        2.0F, 0.0F, 6.0F,
    };
    const auto input = tensor_f32_matrix(2, 6, std::span<const float>(input_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);

    const auto output_buffer = rope_f32_by_heads_with_position_start_to_backend(
        input_buffer,
        1,
        2,
        3,
        2,
        1.0F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = rope_f32_by_heads_with_position_start(input, 1, 2, 3, 2, 1.0F);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU headwise RoPE output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer headwise RoPE output");
}

void test_applies_causal_grouped_attention_scores_to_cpu_backend_buffers() {
    const std::array<float, 12> query_values {
        1.0F, 0.0F, 0.0F, 1.0F,
        2.0F, 0.0F, 0.0F, 2.0F,
        1.0F, 1.0F, 2.0F, 1.0F,
    };
    const std::array<float, 6> key_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
        1.0F, 1.0F,
    };
    const auto query = tensor_f32_matrix(3, 4, std::span<const float>(query_values));
    const auto key = tensor_f32_matrix(3, 2, std::span<const float>(key_values));
    const auto query_buffer = copy_tensor_to_backend(query, BackendKind::cpu);
    const auto key_buffer = copy_tensor_to_backend(key, BackendKind::cpu);

    const auto output_buffer = causal_grouped_attention_scores_f32_to_backend(
        query_buffer,
        key_buffer,
        0,
        2,
        1,
        2,
        2,
        0.5F,
        10.0F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = causal_grouped_attention_scores_f32(query, key, 0, 2, 1, 2, 2, 0.5F, 10.0F);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU causal grouped attention score backend mismatch");
    expect_f32_rank3_close(output, expected, 0.0F, "CPU backend-buffer causal grouped attention scores");
}

void test_applies_causal_grouped_attention_values_to_cpu_backend_buffers() {
    const auto probabilities = grouped_attention_value_probability_fixture();
    const auto value = grouped_attention_value_fixture();
    const auto probability_buffer = copy_tensor_to_backend(probabilities, BackendKind::cpu);
    const auto value_buffer = copy_tensor_to_backend(value, BackendKind::cpu);

    const auto output_buffer = causal_grouped_attention_values_f32_to_backend(
        probability_buffer,
        value_buffer,
        0,
        2,
        1,
        2,
        2
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = causal_grouped_attention_values_f32(probabilities, value, 0, 2, 1, 2, 2);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU causal grouped attention value backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer causal grouped attention values");
}

void test_applies_softmax_to_cpu_backend_buffer() {
    const auto input = attention_score_softmax_fixture();
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);

    const auto output_buffer = softmax_f32_to_backend(input_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = softmax_f32(input);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU softmax backend mismatch");
    expect_f32_rank3_close(output, expected, 0.0F, "CPU backend-buffer softmax");
}

void test_applies_softcap_to_cpu_backend_buffer() {
    const std::array<float, 6> input_values {
        -6.0F, -1.0F, 0.0F,
        0.5F, 2.0F, 6.0F,
    };
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);

    const auto output_buffer = softcap_f32_to_backend(input_buffer, 2.0F);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = softcap_f32(input, 2.0F);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU softcap backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer softcap");
}

void test_selects_top_logit_from_last_row() {
    const std::array<float, 12> input_values {
        0.5F, 99.0F, 1.0F, 2.0F,
        -1.0F, 0.25F, 3.0F, 2.5F,
        -2.0F, 7.0F, 7.0F, 1.0F,
    };
    const auto input = tensor_f32_matrix(3, 4, std::span<const float>(input_values));

    const auto selected = top_logit_f32(input);

    expect(selected.token == 1U, "F32 top-logit should select first max token from final row");
    expect_close(selected.logit, 7.0F, 0.0F, "F32 top-logit selected logit mismatch");
}

void test_selects_top_logit_from_cpu_backend_buffer() {
    const std::array<float, 12> input_values {
        0.5F, 99.0F, 1.0F, 2.0F,
        -1.0F, 0.25F, 3.0F, 2.5F,
        -2.0F, 7.0F, 7.0F, 1.0F,
    };
    const auto input = tensor_f32_matrix(3, 4, std::span<const float>(input_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);

    const auto selected = top_logit_f32_from_backend(input_buffer);

    expect(selected.token == 1U, "CPU backend top-logit token mismatch");
    expect_close(selected.logit, 7.0F, 0.0F, "CPU backend top-logit logit mismatch");
}

void test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_cpu_backend_buffers() {
    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 6> first_values {
        0.25F, -0.5F,
        1.0F, 0.75F,
        -1.25F, 0.5F,
    };
    const std::array<float, 12> second_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 3> third_values {
        1.5F,
        -0.25F,
        0.375F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto first_right = tensor_f16_gguf_matrix(3, 2, std::span<const float>(first_values));
    const auto second_right = tensor_f16_gguf_matrix(3, 4, std::span<const float>(second_values));
    const auto third_right = tensor_f16_gguf_matrix(3, 1, std::span<const float>(third_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto first_buffer = copy_tensor_to_backend(first_right, BackendKind::cpu);
    const auto second_buffer = copy_tensor_to_backend(second_right, BackendKind::cpu);
    const auto third_buffer = copy_tensor_to_backend(third_right, BackendKind::cpu);

    const auto outputs = matmul_f32_f16_gguf_weight_triple_to_backend(
        left_buffer,
        first_buffer,
        second_buffer,
        third_buffer
    );
    const auto first_output = copy_tensor_from_backend(outputs.first);
    const auto second_output = copy_tensor_from_backend(outputs.second);
    const auto third_output = copy_tensor_from_backend(outputs.third);

    expect(outputs.first.backend() == BackendKind::cpu, "CPU triple matmul first output backend mismatch");
    expect(outputs.second.backend() == BackendKind::cpu, "CPU triple matmul second output backend mismatch");
    expect(outputs.third.backend() == BackendKind::cpu, "CPU triple matmul third output backend mismatch");
    expect_f32_matrix_close(
        first_output,
        matmul_f32_f16_gguf_weight(left, first_right, BackendKind::cpu),
        0.0F,
        "CPU triple matmul first output"
    );
    expect_f32_matrix_close(
        second_output,
        matmul_f32_f16_gguf_weight(left, second_right, BackendKind::cpu),
        0.0F,
        "CPU triple matmul second output"
    );
    expect_f32_matrix_close(
        third_output,
        matmul_f32_f16_gguf_weight(left, third_right, BackendKind::cpu),
        0.0F,
        "CPU triple matmul third output"
    );
}

void test_applies_silu_multiply_to_cpu_backend_buffers() {
    const std::array<float, 6> gate_values {
        -2.0F, -1.0F, 0.0F,
        0.5F, 1.0F, 2.0F,
    };
    const std::array<float, 6> up_values {
        0.25F, -0.5F, 1.0F,
        1.5F, -2.0F, 0.75F,
    };
    const auto gate = tensor_f32_matrix(2, 3, std::span<const float>(gate_values));
    const auto up = tensor_f32_matrix(2, 3, std::span<const float>(up_values));
    const auto gate_buffer = copy_tensor_to_backend(gate, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up, BackendKind::cpu);

    const auto output_buffer = silu_multiply_f32_to_backend(gate_buffer, up_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = multiply_f32(silu_f32(gate), up);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU SiLU multiply output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer SiLU multiply output");
}

void test_runs_gated_silu_projection_to_cpu_backend_buffer() {
    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 8> down_values {
        0.5F, -0.25F,
        -0.5F, 1.0F,
        0.25F, 0.375F,
        1.5F, -1.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 2, std::span<const float>(down_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::cpu);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::cpu);

    const auto output_buffer = gated_silu_projection_f32_f16_gguf_weight_to_backend(
        left_buffer,
        gate_buffer,
        up_buffer,
        down_buffer
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto gate = matmul_f32_f16_gguf_weight(left, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(left, up_weight, BackendKind::cpu);
    const auto expected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );

    expect(output_buffer.backend() == BackendKind::cpu, "CPU gated SiLU projection output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer gated SiLU projection output");
}

void test_runs_residual_gated_silu_projection_to_cpu_backend_buffer() {
    const std::array<float, 6> input_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 4> attention_values {
        1.25F, -0.5F,
        -0.75F, 0.875F,
    };
    const std::array<float, 6> attention_output_weight_values {
        0.25F, -0.5F, 0.75F,
        1.0F, -0.25F, 0.5F,
    };
    const std::array<float, 3> norm_values {1.0F, 0.75F, 1.25F};
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 12> down_values {
        0.5F, -0.25F, 0.125F,
        -0.5F, 1.0F, 0.625F,
        0.25F, 0.375F, -0.75F,
        1.5F, -1.0F, 0.5F,
    };
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto attention_output = tensor_f32_matrix(2, 2, std::span<const float>(attention_values));
    const auto attention_output_weight =
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(attention_output_weight_values));
    const auto norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 3, std::span<const float>(down_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);
    const auto attention_output_buffer = copy_tensor_to_backend(attention_output, BackendKind::cpu);
    const auto attention_output_weight_buffer = copy_tensor_to_backend(attention_output_weight, BackendKind::cpu);
    const auto norm_buffer = copy_tensor_to_backend(norm_weight, BackendKind::cpu);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::cpu);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::cpu);

    const auto output_buffer = residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
        input_buffer,
        attention_output_buffer,
        attention_output_weight_buffer,
        norm_buffer,
        gate_buffer,
        up_buffer,
        down_buffer,
        0.25F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto attention_projected = matmul_f32_f16_gguf_weight(
        attention_output,
        attention_output_weight,
        BackendKind::cpu
    );
    const auto attention_residual = add_f32(input, attention_projected);
    const auto normalized = rms_norm_f32(attention_residual, norm_weight, 0.25F);
    const auto gate = matmul_f32_f16_gguf_weight(normalized, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(normalized, up_weight, BackendKind::cpu);
    const auto projected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );
    const auto expected = add_f32(attention_residual, projected);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU residual gated SiLU output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer residual gated SiLU output");
}

void test_matches_cpu_f32_by_f16_gguf_weight_matmul_on_metal_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> left_values {
        0.5F, -1.0F, 2.0F, 3.5F,
        -2.0F, 0.25F, 1.5F, -0.75F,
        4.0F, -3.0F, 0.125F, 2.25F,
    };
    const std::array<float, 20> right_values {
        0.25F, -0.5F, 1.0F, 2.0F, -1.5F,
        1.25F, 0.75F, -2.0F, 0.5F, 3.0F,
        -0.25F, 1.5F, 0.125F, -1.0F, 2.25F,
        2.5F, -1.25F, 0.625F, 1.75F, -0.875F,
    };
    const auto left = tensor_f32_matrix(3, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 5, std::span<const float>(right_values));

    const auto cpu_output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
    const auto metal_output = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);

    expect(metal_output.descriptor() == cpu_output.descriptor(), "Metal GGUF matmul descriptor mismatch");
    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 5; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(metal_output, std::span<const std::uint64_t>(index)),
                read_f32(cpu_output, std::span<const std::uint64_t>(index)),
                1.0e-3F,
                "Metal GGUF matmul parity mismatch"
            );
        }
    }
}

void test_reuses_metal_f32_by_f16_gguf_weight_matmul_state() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 8> left_values {
        1.0F, -2.0F, 0.5F, 3.0F,
        -1.5F, 0.25F, 2.0F, -0.75F,
    };
    const std::array<float, 12> right_values {
        0.5F, -1.0F, 2.0F,
        1.5F, 0.25F, -0.5F,
        -2.5F, 0.75F, 1.25F,
        0.125F, -1.75F, 2.25F,
    };
    const auto left = tensor_f32_matrix(2, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 3, std::span<const float>(right_values));

    const auto first = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
    const auto second = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);

    expect(second.descriptor() == first.descriptor(), "reused Metal GGUF matmul descriptor mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(second, std::span<const std::uint64_t>(index)),
                read_f32(first, std::span<const std::uint64_t>(index)),
                0.0F,
                "reused Metal GGUF matmul value mismatch"
            );
        }
    }
}

void test_matches_cpu_f32_by_q4_k_gguf_weight_matmul_on_metal_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t rows = 2;
    constexpr std::uint64_t shared = 512;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_q4_k_gguf_weight_matrix(shared, columns);

    const auto cpu_output = matmul_f32_quantized_gguf_weight(left, right, BackendKind::cpu);
    const auto metal_output = matmul_f32_quantized_gguf_weight(left, right, BackendKind::metal);

    expect_f32_matrix_close(metal_output, cpu_output, 1.0e-3F, "Metal Q4_K GGUF matmul output");
}

void test_multiplies_backend_f32_by_backend_q4_k_gguf_weight_to_metal_backend_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t rows = 2;
    constexpr std::uint64_t shared = 512;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_q4_k_gguf_weight_matrix(shared, columns);
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);

    const auto output_buffer = matmul_f32_quantized_gguf_weight_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = matmul_f32_quantized_gguf_weight(left, right, BackendKind::cpu);

    expect(output_buffer.backend() == BackendKind::metal, "Metal Q4_K backend-buffer matmul output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer Q4_K GGUF matmul output");
}

void test_multiplies_backend_f32_by_backend_q6_k_gguf_weight_to_metal_backend_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t rows = 2;
    constexpr std::uint64_t shared = 512;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_q6_k_gguf_weight_matrix(shared, columns);
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);

    const auto output_buffer = matmul_f32_quantized_gguf_weight_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = matmul_f32_quantized_gguf_weight(left, right, BackendKind::cpu);

    expect(output_buffer.backend() == BackendKind::metal, "Metal Q6_K backend-buffer matmul output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer Q6_K GGUF matmul output");
}

void test_copies_tensor_to_and_from_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> values {
        1.25F, -2.5F,
        3.75F, -4.5F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    const auto buffer = copy_tensor_to_backend(storage, BackendKind::metal);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::metal, "Metal backend buffer kind mismatch");
    expect(buffer.descriptor() == storage.descriptor(), "Metal backend buffer descriptor mismatch");
    expect(roundtrip.descriptor() == storage.descriptor(), "Metal backend buffer roundtrip descriptor mismatch");
    expect(
        std::equal(storage.bytes().begin(), storage.bytes().end(), roundtrip.bytes().begin()),
        "Metal backend buffer roundtrip bytes mismatch"
    );
}

void test_copies_tensor_view_to_and_from_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> values {
        1.25F, -2.5F,
        3.75F, -4.5F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    const auto buffer = copy_tensor_view_to_backend(storage.view(), BackendKind::metal);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::metal, "Metal backend view buffer kind mismatch");
    expect(buffer.descriptor() == storage.descriptor(), "Metal backend view buffer descriptor mismatch");
    expect(roundtrip.descriptor() == storage.descriptor(), "Metal backend view buffer roundtrip descriptor mismatch");
    expect(
        std::equal(storage.bytes().begin(), storage.bytes().end(), roundtrip.bytes().begin()),
        "Metal backend view buffer roundtrip bytes mismatch"
    );
}

void test_allocates_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);

    const auto buffer = allocate_backend_buffer(descriptor, BackendKind::metal);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::metal, "Metal allocated backend buffer kind mismatch");
    expect(buffer.descriptor() == descriptor, "Metal allocated backend buffer descriptor mismatch");
    expect(roundtrip.descriptor() == descriptor, "Metal allocated backend buffer roundtrip descriptor mismatch");
}

void test_multiplies_f32_by_metal_backend_f16_gguf_weight_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> left_values {
        0.5F, -1.0F, 2.0F, 3.5F,
        -2.0F, 0.25F, 1.5F, -0.75F,
        4.0F, -3.0F, 0.125F, 2.25F,
    };
    const std::array<float, 20> right_values {
        0.25F, -0.5F, 1.0F, 2.0F, -1.5F,
        1.25F, 0.75F, -2.0F, 0.5F, 3.0F,
        -0.25F, 1.5F, 0.125F, -1.0F, 2.25F,
        2.5F, -1.25F, 0.625F, 1.75F, -0.875F,
    };
    const auto left = tensor_f32_matrix(3, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 5, std::span<const float>(right_values));
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);

    const auto cpu_output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
    const auto metal_output = matmul_f32_f16_gguf_weight(left, right_buffer);

    expect(metal_output.descriptor() == cpu_output.descriptor(), "Metal buffered GGUF matmul descriptor mismatch");
    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 5; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(metal_output, std::span<const std::uint64_t>(index)),
                read_f32(cpu_output, std::span<const std::uint64_t>(index)),
                1.0e-3F,
                "Metal buffered GGUF matmul parity mismatch"
            );
        }
    }
}

void test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> left_values {
        0.5F, -1.0F, 2.0F, 3.5F,
        -2.0F, 0.25F, 1.5F, -0.75F,
        4.0F, -3.0F, 0.125F, 2.25F,
    };
    const std::array<float, 20> right_values {
        0.25F, -0.5F, 1.0F, 2.0F, -1.5F,
        1.25F, 0.75F, -2.0F, 0.5F, 3.0F,
        -0.25F, 1.5F, 0.125F, -1.0F, 2.25F,
        2.5F, -1.25F, 0.625F, 1.75F, -0.875F,
    };
    const auto left = tensor_f32_matrix(3, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 5, std::span<const float>(right_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);

    const auto output_buffer = matmul_f32_f16_gguf_weight_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);

    expect(output_buffer.backend() == BackendKind::metal, "Metal backend-buffer GGUF matmul output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer GGUF matmul output");
}

void test_matches_f16_gguf_token_embedding_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 16> embedding_values {
        0.25F, -0.5F, 1.0F, 2.0F,
        1.25F, 0.75F, -2.0F, 0.5F,
        -0.25F, 1.5F, 0.125F, -1.0F,
        2.5F, -1.25F, 0.625F, 1.75F,
    };
    const std::array<std::uint32_t, 3> token_ids {2, 0, 3};
    const auto embedding = tensor_f16_gguf_matrix(4, 4, std::span<const float>(embedding_values));
    const auto cpu_embedding_buffer = copy_tensor_to_backend(embedding, BackendKind::cpu);
    const auto metal_embedding_buffer = copy_tensor_to_backend(embedding, BackendKind::metal);

    const auto cpu_output_buffer = embed_f16_gguf_weight_token_ids_to_backend(
        std::span<const std::uint32_t>(token_ids),
        cpu_embedding_buffer,
        1.25F
    );
    const auto metal_output_buffer = embed_f16_gguf_weight_token_ids_to_backend(
        std::span<const std::uint32_t>(token_ids),
        metal_embedding_buffer,
        1.25F
    );

    expect(
        metal_output_buffer.backend() == BackendKind::metal,
        "Metal token embedding output backend mismatch"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        0.0F,
        "Metal token embedding output"
    );
}

void test_matches_rms_norm_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 8> input_values {
        2.0F, -4.0F, 8.0F, -1.0F,
        0.5F, -1.5F, 3.0F, 6.0F,
    };
    const std::array<float, 4> weight_values {1.0F, 0.5F, 1.5F, 0.25F};
    const auto input = tensor_f32_matrix(2, 4, std::span<const float>(input_values));
    const auto weight = tensor_f32_vector(std::span<const float>(weight_values));
    const auto cpu_output_buffer = rms_norm_f32_to_backend(
        copy_tensor_to_backend(input, BackendKind::cpu),
        copy_tensor_to_backend(weight, BackendKind::cpu),
        0.25F
    );
    const auto metal_output_buffer = rms_norm_f32_to_backend(
        copy_tensor_to_backend(input, BackendKind::metal),
        copy_tensor_to_backend(weight, BackendKind::metal),
        0.25F
    );

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal RMSNorm output backend mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-5F,
        "Metal backend-buffer RMSNorm output"
    );
}

void test_matches_f32_add_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 8> left_values {
        2.0F, -4.0F, 8.0F, -1.0F,
        0.5F, -1.5F, 3.0F, 6.0F,
    };
    const std::array<float, 8> right_values {
        -0.5F, 2.0F, -3.0F, 0.25F,
        1.5F, 0.75F, -2.0F, 4.0F,
    };
    const auto left = tensor_f32_matrix(2, 4, std::span<const float>(left_values));
    const auto right = tensor_f32_matrix(2, 4, std::span<const float>(right_values));
    const auto cpu_output_buffer = add_f32_to_backend(
        copy_tensor_to_backend(left, BackendKind::cpu),
        copy_tensor_to_backend(right, BackendKind::cpu)
    );
    const auto metal_output_buffer = add_f32_to_backend(
        copy_tensor_to_backend(left, BackendKind::metal),
        copy_tensor_to_backend(right, BackendKind::metal)
    );

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal F32 add output backend mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        0.0F,
        "Metal backend-buffer F32 add output"
    );
}

void test_matches_headwise_rope_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> input_values {
        1.0F, 0.0F, 9.0F,
        0.0F, 1.0F, 8.0F,
        1.0F, 1.0F, 7.0F,
        2.0F, 0.0F, 6.0F,
    };
    const auto input = tensor_f32_matrix(2, 6, std::span<const float>(input_values));
    const auto cpu_output_buffer = rope_f32_by_heads_with_position_start_to_backend(
        copy_tensor_to_backend(input, BackendKind::cpu),
        1,
        2,
        3,
        2,
        1.0F
    );
    const auto metal_output_buffer = rope_f32_by_heads_with_position_start_to_backend(
        copy_tensor_to_backend(input, BackendKind::metal),
        1,
        2,
        3,
        2,
        1.0F
    );

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal headwise RoPE output backend mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-5F,
        "Metal backend-buffer headwise RoPE output"
    );
}

void test_matches_causal_grouped_attention_scores_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> query_values {
        1.0F, 0.0F, 0.0F, 1.0F,
        2.0F, 0.0F, 0.0F, 2.0F,
        1.0F, 1.0F, 2.0F, 1.0F,
    };
    const std::array<float, 6> key_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
        1.0F, 1.0F,
    };
    const auto query = tensor_f32_matrix(3, 4, std::span<const float>(query_values));
    const auto key = tensor_f32_matrix(3, 2, std::span<const float>(key_values));
    const auto cpu_output_buffer = causal_grouped_attention_scores_f32_to_backend(
        copy_tensor_to_backend(query, BackendKind::cpu),
        copy_tensor_to_backend(key, BackendKind::cpu),
        0,
        2,
        1,
        2,
        2,
        0.5F,
        10.0F
    );
    const auto metal_output_buffer = causal_grouped_attention_scores_f32_to_backend(
        copy_tensor_to_backend(query, BackendKind::metal),
        copy_tensor_to_backend(key, BackendKind::metal),
        0,
        2,
        1,
        2,
        2,
        0.5F,
        10.0F
    );

    expect(
        metal_output_buffer.backend() == BackendKind::metal,
        "Metal causal grouped attention score backend mismatch"
    );
    expect_f32_rank3_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-5F,
        "Metal backend-buffer causal grouped attention scores"
    );
}

void test_matches_causal_grouped_attention_values_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto probabilities = grouped_attention_value_probability_fixture();
    const auto value = grouped_attention_value_fixture();
    const auto cpu_output_buffer = causal_grouped_attention_values_f32_to_backend(
        copy_tensor_to_backend(probabilities, BackendKind::cpu),
        copy_tensor_to_backend(value, BackendKind::cpu),
        0,
        2,
        1,
        2,
        2
    );
    const auto metal_output_buffer = causal_grouped_attention_values_f32_to_backend(
        copy_tensor_to_backend(probabilities, BackendKind::metal),
        copy_tensor_to_backend(value, BackendKind::metal),
        0,
        2,
        1,
        2,
        2
    );

    expect(
        metal_output_buffer.backend() == BackendKind::metal,
        "Metal causal grouped attention value backend mismatch"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-5F,
        "Metal backend-buffer causal grouped attention values"
    );
}

void test_matches_softmax_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto input = attention_score_softmax_fixture();
    const auto cpu_output_buffer = softmax_f32_to_backend(copy_tensor_to_backend(input, BackendKind::cpu));
    const auto metal_output_buffer = softmax_f32_to_backend(copy_tensor_to_backend(input, BackendKind::metal));

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal softmax backend mismatch");
    expect_f32_rank3_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-6F,
        "Metal backend-buffer softmax"
    );
}

void test_matches_softcap_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> input_values {
        -6.0F, -1.0F, 0.0F,
        0.5F, 2.0F, 6.0F,
    };
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto cpu_output_buffer = softcap_f32_to_backend(copy_tensor_to_backend(input, BackendKind::cpu), 2.0F);
    const auto metal_output_buffer = softcap_f32_to_backend(copy_tensor_to_backend(input, BackendKind::metal), 2.0F);

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal softcap backend mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-6F,
        "Metal backend-buffer softcap"
    );
}

void test_matches_top_logit_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t rows = 2;
    constexpr std::uint64_t columns = 300;
    std::vector<float> values(static_cast<std::size_t>(rows * columns), -10.0F);
    values[17] = 500.0F;
    for (std::uint64_t token = 0; token < columns; ++token) {
        values[static_cast<std::size_t>(columns + token)] =
            -2.0F + 0.001F * static_cast<float>(token % 29U);
    }
    values[static_cast<std::size_t>(columns + 257U)] = 9.0F;
    values[static_cast<std::size_t>(columns + 299U)] = 9.0F;
    const auto input = tensor_f32_matrix(rows, columns, std::span<const float>(values));
    const auto expected = top_logit_f32(input);

    const auto selected = top_logit_f32_from_backend(copy_tensor_to_backend(input, BackendKind::metal));

    expect(expected.token == 257U, "Metal top-logit fixture expected token mismatch");
    expect(selected.token == expected.token, "Metal top-logit token mismatch");
    expect_close(selected.logit, expected.logit, 0.0F, "Metal top-logit logit mismatch");
}

void test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_metal_backend_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 6> first_values {
        0.25F, -0.5F,
        1.0F, 0.75F,
        -1.25F, 0.5F,
    };
    const std::array<float, 12> second_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 3> third_values {
        1.5F,
        -0.25F,
        0.375F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto first_right = tensor_f16_gguf_matrix(3, 2, std::span<const float>(first_values));
    const auto second_right = tensor_f16_gguf_matrix(3, 4, std::span<const float>(second_values));
    const auto third_right = tensor_f16_gguf_matrix(3, 1, std::span<const float>(third_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto first_buffer = copy_tensor_to_backend(first_right, BackendKind::metal);
    const auto second_buffer = copy_tensor_to_backend(second_right, BackendKind::metal);
    const auto third_buffer = copy_tensor_to_backend(third_right, BackendKind::metal);

    const auto outputs = matmul_f32_f16_gguf_weight_triple_to_backend(
        left_buffer,
        first_buffer,
        second_buffer,
        third_buffer
    );
    const auto first_output = copy_tensor_from_backend(outputs.first);
    const auto second_output = copy_tensor_from_backend(outputs.second);
    const auto third_output = copy_tensor_from_backend(outputs.third);

    expect(outputs.first.backend() == BackendKind::metal, "Metal triple matmul first output backend mismatch");
    expect(outputs.second.backend() == BackendKind::metal, "Metal triple matmul second output backend mismatch");
    expect(outputs.third.backend() == BackendKind::metal, "Metal triple matmul third output backend mismatch");
    expect_f32_matrix_close(
        first_output,
        matmul_f32_f16_gguf_weight(left, first_right, BackendKind::cpu),
        1.0e-3F,
        "Metal triple matmul first output"
    );
    expect_f32_matrix_close(
        second_output,
        matmul_f32_f16_gguf_weight(left, second_right, BackendKind::cpu),
        1.0e-3F,
        "Metal triple matmul second output"
    );
    expect_f32_matrix_close(
        third_output,
        matmul_f32_f16_gguf_weight(left, third_right, BackendKind::cpu),
        1.0e-3F,
        "Metal triple matmul third output"
    );
}

void test_applies_silu_multiply_to_metal_backend_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> gate_values {
        -2.0F, -1.0F, 0.0F,
        0.5F, 1.0F, 2.0F,
    };
    const std::array<float, 6> up_values {
        0.25F, -0.5F, 1.0F,
        1.5F, -2.0F, 0.75F,
    };
    const auto gate = tensor_f32_matrix(2, 3, std::span<const float>(gate_values));
    const auto up = tensor_f32_matrix(2, 3, std::span<const float>(up_values));
    const auto gate_buffer = copy_tensor_to_backend(gate, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up, BackendKind::metal);

    const auto output_buffer = silu_multiply_f32_to_backend(gate_buffer, up_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = multiply_f32(silu_f32(gate), up);

    expect(output_buffer.backend() == BackendKind::metal, "Metal SiLU multiply output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-5F, "Metal backend-buffer SiLU multiply output");
}

void test_runs_gated_silu_projection_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 8> down_values {
        0.5F, -0.25F,
        -0.5F, 1.0F,
        0.25F, 0.375F,
        1.5F, -1.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 2, std::span<const float>(down_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::metal);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::metal);

    const auto output_buffer = gated_silu_projection_f32_f16_gguf_weight_to_backend(
        left_buffer,
        gate_buffer,
        up_buffer,
        down_buffer
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto gate = matmul_f32_f16_gguf_weight(left, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(left, up_weight, BackendKind::cpu);
    const auto expected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );

    expect(output_buffer.backend() == BackendKind::metal, "Metal gated SiLU projection output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer gated SiLU projection output");
}

void test_runs_residual_gated_silu_projection_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> input_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 4> attention_values {
        1.25F, -0.5F,
        -0.75F, 0.875F,
    };
    const std::array<float, 6> attention_output_weight_values {
        0.25F, -0.5F, 0.75F,
        1.0F, -0.25F, 0.5F,
    };
    const std::array<float, 3> norm_values {1.0F, 0.75F, 1.25F};
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 12> down_values {
        0.5F, -0.25F, 0.125F,
        -0.5F, 1.0F, 0.625F,
        0.25F, 0.375F, -0.75F,
        1.5F, -1.0F, 0.5F,
    };
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto attention_output = tensor_f32_matrix(2, 2, std::span<const float>(attention_values));
    const auto attention_output_weight =
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(attention_output_weight_values));
    const auto norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 3, std::span<const float>(down_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::metal);
    const auto attention_output_buffer = copy_tensor_to_backend(attention_output, BackendKind::metal);
    const auto attention_output_weight_buffer = copy_tensor_to_backend(attention_output_weight, BackendKind::metal);
    const auto norm_buffer = copy_tensor_to_backend(norm_weight, BackendKind::metal);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::metal);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::metal);

    const auto output_buffer = residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
        input_buffer,
        attention_output_buffer,
        attention_output_weight_buffer,
        norm_buffer,
        gate_buffer,
        up_buffer,
        down_buffer,
        0.25F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto attention_projected = matmul_f32_f16_gguf_weight(
        attention_output,
        attention_output_weight,
        BackendKind::cpu
    );
    const auto attention_residual = add_f32(input, attention_projected);
    const auto normalized = rms_norm_f32(attention_residual, norm_weight, 0.25F);
    const auto gate = matmul_f32_f16_gguf_weight(normalized, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(normalized, up_weight, BackendKind::cpu);
    const auto projected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );
    const auto expected = add_f32(attention_residual, projected);

    expect(output_buffer.backend() == BackendKind::metal, "Metal residual gated SiLU output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer residual gated SiLU output");
}

void test_writes_prefill_kv_cache_to_cpu_backend_buffer() {
    const std::array<float, 8> key_values {
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
    };
    const std::array<float, 12> value_values {
        -1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F,
        -7.0F, -8.0F, -9.0F, -10.0F, -11.0F, -12.0F,
    };
    const auto key = tensor_f32_matrix(2, 4, std::span<const float>(key_values));
    const auto value = tensor_f32_matrix(2, 6, std::span<const float>(value_values));
    const auto key_cache_descriptor = TensorDescriptor::from(Shape::from({5, 2, 2}), DType::f32);
    const auto value_cache_descriptor = TensorDescriptor::from(Shape::from({5, 2, 3}), DType::f32);
    auto key_cache_buffer = copy_tensor_to_backend(TensorStorage::allocate(key_cache_descriptor), BackendKind::cpu);
    auto value_cache_buffer = copy_tensor_to_backend(
        TensorStorage::allocate(value_cache_descriptor),
        BackendKind::cpu
    );

    write_prefill_kv_cache_f32_to_backend(
        copy_tensor_to_backend(key, BackendKind::cpu),
        copy_tensor_to_backend(value, BackendKind::cpu),
        key_cache_buffer,
        value_cache_buffer,
        1,
        2,
        2,
        3
    );

    auto expected_key_cache = TensorStorage::allocate(key_cache_descriptor);
    auto expected_value_cache = TensorStorage::allocate(value_cache_descriptor);
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t head = 0; head < 2; ++head) {
            for (std::uint64_t column = 0; column < 2; ++column) {
                const std::array<std::uint64_t, 2> source_index {row, head * 2 + column};
                const std::array<std::uint64_t, 3> cache_index {row + 1, head, column};
                write_f32(
                    expected_key_cache,
                    std::span<const std::uint64_t>(cache_index),
                    read_f32(key, std::span<const std::uint64_t>(source_index))
                );
            }
            for (std::uint64_t column = 0; column < 3; ++column) {
                const std::array<std::uint64_t, 2> source_index {row, head * 3 + column};
                const std::array<std::uint64_t, 3> cache_index {row + 1, head, column};
                write_f32(
                    expected_value_cache,
                    std::span<const std::uint64_t>(cache_index),
                    read_f32(value, std::span<const std::uint64_t>(source_index))
                );
            }
        }
    }

    expect_f32_rank3_close(
        copy_tensor_from_backend(key_cache_buffer),
        expected_key_cache,
        0.0F,
        "CPU prefill KV key cache write"
    );
    expect_f32_rank3_close(
        copy_tensor_from_backend(value_cache_buffer),
        expected_value_cache,
        0.0F,
        "CPU prefill KV value cache write"
    );
}

void test_matches_prefill_kv_cache_write_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 8> key_values {
        0.25F, -0.5F, 1.25F, -1.5F,
        2.25F, -2.5F, 3.25F, -3.5F,
    };
    const std::array<float, 12> value_values {
        10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F,
        20.0F, 21.0F, 22.0F, 23.0F, 24.0F, 25.0F,
    };
    const auto key = tensor_f32_matrix(2, 4, std::span<const float>(key_values));
    const auto value = tensor_f32_matrix(2, 6, std::span<const float>(value_values));
    const auto key_cache_descriptor = TensorDescriptor::from(Shape::from({5, 2, 2}), DType::f32);
    const auto value_cache_descriptor = TensorDescriptor::from(Shape::from({5, 2, 3}), DType::f32);
    auto cpu_key_cache = copy_tensor_to_backend(TensorStorage::allocate(key_cache_descriptor), BackendKind::cpu);
    auto cpu_value_cache = copy_tensor_to_backend(TensorStorage::allocate(value_cache_descriptor), BackendKind::cpu);
    auto metal_key_cache = copy_tensor_to_backend(TensorStorage::allocate(key_cache_descriptor), BackendKind::metal);
    auto metal_value_cache = copy_tensor_to_backend(
        TensorStorage::allocate(value_cache_descriptor),
        BackendKind::metal
    );

    write_prefill_kv_cache_f32_to_backend(
        copy_tensor_to_backend(key, BackendKind::cpu),
        copy_tensor_to_backend(value, BackendKind::cpu),
        cpu_key_cache,
        cpu_value_cache,
        1,
        2,
        2,
        3
    );
    write_prefill_kv_cache_f32_to_backend(
        copy_tensor_to_backend(key, BackendKind::metal),
        copy_tensor_to_backend(value, BackendKind::metal),
        metal_key_cache,
        metal_value_cache,
        1,
        2,
        2,
        3
    );

    expect_f32_rank3_close(
        copy_tensor_from_backend(metal_key_cache),
        copy_tensor_from_backend(cpu_key_cache),
        0.0F,
        "Metal prefill KV key cache write"
    );
    expect_f32_rank3_close(
        copy_tensor_from_backend(metal_value_cache),
        copy_tensor_from_backend(cpu_value_cache),
        0.0F,
        "Metal prefill KV value cache write"
    );
}

void test_applies_causal_grouped_attention_to_cpu_backend_kv_cache_buffers() {
    constexpr std::uint64_t cache_capacity = 5;
    constexpr std::uint64_t head_count = 4;
    constexpr std::uint64_t kv_head_count = 2;
    constexpr std::uint64_t key_length = 2;
    constexpr std::uint64_t value_length = 3;
    constexpr std::uint64_t query_position_start = 3;
    constexpr std::uint64_t attention_token_count = 3;
    const std::array<float, head_count * key_length> query_values {
        0.25F, -0.5F,
        0.75F, 0.125F,
        -0.25F, 0.5F,
        0.375F, -0.125F,
    };
    const auto query = tensor_f32_matrix(1, head_count * key_length, std::span<const float>(query_values));
    auto key_cache = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({cache_capacity, kv_head_count, key_length}),
        DType::f32
    ));
    auto value_cache = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({cache_capacity, kv_head_count, value_length}),
        DType::f32
    ));
    for (std::uint64_t row = 0; row < cache_capacity; ++row) {
        for (std::uint64_t head = 0; head < kv_head_count; ++head) {
            for (std::uint64_t column = 0; column < key_length; ++column) {
                const std::array<std::uint64_t, 3> index {row, head, column};
                write_f32(
                    key_cache,
                    std::span<const std::uint64_t>(index),
                    0.1F * static_cast<float>(row + 1)
                        + 0.05F * static_cast<float>(head)
                        - 0.025F * static_cast<float>(column)
                );
            }
            for (std::uint64_t column = 0; column < value_length; ++column) {
                const std::array<std::uint64_t, 3> index {row, head, column};
                write_f32(
                    value_cache,
                    std::span<const std::uint64_t>(index),
                    -0.2F * static_cast<float>(row)
                        + 0.125F * static_cast<float>(head + 1)
                        + 0.0625F * static_cast<float>(column)
                );
            }
        }
    }

    const auto flat_key = TensorStorage::from_bytes(
        TensorDescriptor::from(Shape::from({cache_capacity, kv_head_count * key_length}), DType::f32),
        key_cache.bytes()
    );
    const auto flat_value = TensorStorage::from_bytes(
        TensorDescriptor::from(Shape::from({cache_capacity, kv_head_count * value_length}), DType::f32),
        value_cache.bytes()
    );
    const auto scores_buffer = causal_grouped_attention_scores_f32_with_kv_cache_to_backend(
        copy_tensor_to_backend(query, BackendKind::cpu),
        copy_tensor_to_backend(key_cache, BackendKind::cpu),
        query_position_start,
        head_count,
        kv_head_count,
        key_length,
        attention_token_count,
        0.5F,
        10.0F
    );
    const auto expected_scores = causal_grouped_attention_scores_f32(
        query,
        flat_key,
        query_position_start,
        head_count,
        kv_head_count,
        key_length,
        attention_token_count,
        0.5F,
        10.0F
    );
    const auto probabilities_buffer = softmax_f32_to_backend(scores_buffer);
    const auto values_buffer = causal_grouped_attention_values_f32_with_kv_cache_to_backend(
        probabilities_buffer,
        copy_tensor_to_backend(value_cache, BackendKind::cpu),
        query_position_start,
        head_count,
        kv_head_count,
        value_length,
        attention_token_count
    );
    const auto expected_values = causal_grouped_attention_values_f32(
        softmax_f32(expected_scores),
        flat_value,
        query_position_start,
        head_count,
        kv_head_count,
        value_length,
        attention_token_count
    );

    expect_f32_rank3_close(
        copy_tensor_from_backend(scores_buffer),
        expected_scores,
        1.0e-6F,
        "CPU grouped cache attention scores"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(values_buffer),
        expected_values,
        1.0e-6F,
        "CPU grouped cache attention values"
    );
}

void test_matches_causal_grouped_attention_kv_cache_to_metal_backend_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t cache_capacity = 6;
    constexpr std::uint64_t head_count = 4;
    constexpr std::uint64_t kv_head_count = 2;
    constexpr std::uint64_t key_length = 2;
    constexpr std::uint64_t value_length = 3;
    constexpr std::uint64_t query_position_start = 4;
    constexpr std::uint64_t attention_token_count = 4;
    const std::array<float, head_count * key_length> query_values {
        0.5F, -0.25F,
        0.125F, 0.75F,
        -0.375F, 0.625F,
        0.25F, -0.5F,
    };
    const auto query = tensor_f32_matrix(1, head_count * key_length, std::span<const float>(query_values));
    auto key_cache = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({cache_capacity, kv_head_count, key_length}),
        DType::f32
    ));
    auto value_cache = TensorStorage::allocate(TensorDescriptor::from(
        Shape::from({cache_capacity, kv_head_count, value_length}),
        DType::f32
    ));
    for (std::uint64_t row = 0; row < cache_capacity; ++row) {
        for (std::uint64_t head = 0; head < kv_head_count; ++head) {
            for (std::uint64_t column = 0; column < key_length; ++column) {
                const std::array<std::uint64_t, 3> index {row, head, column};
                write_f32(
                    key_cache,
                    std::span<const std::uint64_t>(index),
                    -0.075F * static_cast<float>(row)
                        + 0.2F * static_cast<float>(head + 1)
                        + 0.03125F * static_cast<float>(column)
                );
            }
            for (std::uint64_t column = 0; column < value_length; ++column) {
                const std::array<std::uint64_t, 3> index {row, head, column};
                write_f32(
                    value_cache,
                    std::span<const std::uint64_t>(index),
                    0.15F * static_cast<float>(row + 1)
                        - 0.05F * static_cast<float>(head)
                        + 0.025F * static_cast<float>(column)
                );
            }
        }
    }

    const auto cpu_scores_buffer = causal_grouped_attention_scores_f32_with_kv_cache_to_backend(
        copy_tensor_to_backend(query, BackendKind::cpu),
        copy_tensor_to_backend(key_cache, BackendKind::cpu),
        query_position_start,
        head_count,
        kv_head_count,
        key_length,
        attention_token_count,
        0.5F,
        10.0F
    );
    const auto metal_scores_buffer = causal_grouped_attention_scores_f32_with_kv_cache_to_backend(
        copy_tensor_to_backend(query, BackendKind::metal),
        copy_tensor_to_backend(key_cache, BackendKind::metal),
        query_position_start,
        head_count,
        kv_head_count,
        key_length,
        attention_token_count,
        0.5F,
        10.0F
    );
    const auto cpu_probabilities_buffer = softmax_f32_to_backend(cpu_scores_buffer);
    const auto metal_probabilities_buffer = softmax_f32_to_backend(metal_scores_buffer);
    const auto cpu_values_buffer = causal_grouped_attention_values_f32_with_kv_cache_to_backend(
        cpu_probabilities_buffer,
        copy_tensor_to_backend(value_cache, BackendKind::cpu),
        query_position_start,
        head_count,
        kv_head_count,
        value_length,
        attention_token_count
    );
    const auto metal_values_buffer = causal_grouped_attention_values_f32_with_kv_cache_to_backend(
        metal_probabilities_buffer,
        copy_tensor_to_backend(value_cache, BackendKind::metal),
        query_position_start,
        head_count,
        kv_head_count,
        value_length,
        attention_token_count
    );

    expect_f32_rank3_close(
        copy_tensor_from_backend(metal_scores_buffer),
        copy_tensor_from_backend(cpu_scores_buffer),
        1.0e-4F,
        "Metal grouped cache attention scores"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_values_buffer),
        copy_tensor_from_backend(cpu_values_buffer),
        1.0e-4F,
        "Metal grouped cache attention values"
    );
}

void test_reports_unavailable_metal_backend_for_gguf_matmul_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> left_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> right_values {
        5.0F, 6.0F,
        7.0F, 8.0F,
    };
    const auto left = tensor_f32_matrix(2, 2, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(2, 2, std::span<const float>(right_values));

    try {
        (void)matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
        throw TestFailure("Metal GGUF matmul should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal GGUF matmul unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal GGUF matmul unavailable backend mismatch");
        expect(
            !tensor_backend_unavailable_reason(BackendKind::metal).empty(),
            "Metal backend unavailable reason should be non-empty"
        );
    }
}

void test_reports_unavailable_metal_backend_for_quantized_gguf_matmul_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t rows = 1;
    constexpr std::uint64_t shared = 256;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_q4_k_gguf_weight_matrix(shared, columns);

    try {
        (void)matmul_f32_quantized_gguf_weight(left, right, BackendKind::metal);
        throw TestFailure("Metal K-quant GGUF matmul should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal K-quant GGUF matmul unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal K-quant GGUF matmul unavailable backend mismatch");
    }
}

void test_reports_unavailable_metal_backend_buffer_copy_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    try {
        (void)copy_tensor_to_backend(storage, BackendKind::metal);
        throw TestFailure("Metal backend buffer copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal backend buffer unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal backend buffer unavailable backend mismatch");
    }
}

void test_reports_unavailable_metal_backend_view_copy_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    try {
        (void)copy_tensor_view_to_backend(storage.view(), BackendKind::metal);
        throw TestFailure("Metal backend view copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal backend view-copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal backend view-copy unavailable backend mismatch");
    }
}

void test_reports_unavailable_metal_backend_allocation_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);

    try {
        (void)allocate_backend_buffer(descriptor, BackendKind::metal);
        throw TestFailure("Metal backend allocation should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal backend allocation unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal backend allocation unavailable backend mismatch");
    }
}

const TestCase tensor_backend_tests[] {
    {
        "multiplies F32 by F16 GGUF weight matrix on CPU backend",
        test_multiplies_f32_by_f16_gguf_weight_matrix_on_cpu_backend,
    },
    {
        "multiplies F32 by Q4_K GGUF weight matrix on CPU backend",
        test_multiplies_f32_by_q4_k_gguf_weight_matrix_on_cpu_backend,
    },
    {
        "multiplies F32 by Q6_K GGUF weight matrix on CPU backend",
        test_multiplies_f32_by_q6_k_gguf_weight_matrix_on_cpu_backend,
    },
    {"copies tensor to and from CPU backend buffer", test_copies_tensor_to_and_from_cpu_backend_buffer},
    {"copies tensor view to and from CPU backend buffer", test_copies_tensor_view_to_and_from_cpu_backend_buffer},
    {"allocates CPU backend buffer", test_allocates_cpu_backend_buffer},
    {
        "multiplies F32 by CPU backend F16 GGUF weight buffer",
        test_multiplies_f32_by_cpu_backend_f16_gguf_weight_buffer,
    },
    {
        "multiplies backend F32 by backend F16 GGUF weight to CPU backend buffer",
        test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_cpu_backend_buffer,
    },
    {
        "embeds F16 GGUF token ids to CPU backend buffer",
        test_embeds_f16_gguf_token_ids_to_cpu_backend_buffer,
    },
    {
        "rejects invalid F16 GGUF token embedding inputs",
        test_rejects_invalid_f16_gguf_token_embedding_inputs,
    },
    {"applies RMSNorm to CPU backend buffers", test_applies_rms_norm_to_cpu_backend_buffers},
    {"adds F32 CPU backend buffers", test_adds_f32_cpu_backend_buffers},
    {"applies headwise RoPE to CPU backend buffer", test_applies_headwise_rope_to_cpu_backend_buffer},
    {
        "applies causal grouped attention scores to CPU backend buffers",
        test_applies_causal_grouped_attention_scores_to_cpu_backend_buffers,
    },
    {
        "applies causal grouped attention values to CPU backend buffers",
        test_applies_causal_grouped_attention_values_to_cpu_backend_buffers,
    },
    {"applies softmax to CPU backend buffer", test_applies_softmax_to_cpu_backend_buffer},
    {"applies softcap to CPU backend buffer", test_applies_softcap_to_cpu_backend_buffer},
    {"selects top logit from last row", test_selects_top_logit_from_last_row},
    {"selects top logit from CPU backend buffer", test_selects_top_logit_from_cpu_backend_buffer},
    {
        "multiplies backend F32 by three backend F16 GGUF weights to CPU backend buffers",
        test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_cpu_backend_buffers,
    },
    {"applies SiLU multiply to CPU backend buffers", test_applies_silu_multiply_to_cpu_backend_buffers},
    {"runs gated SiLU projection to CPU backend buffer", test_runs_gated_silu_projection_to_cpu_backend_buffer},
    {
        "runs residual gated SiLU projection to CPU backend buffer",
        test_runs_residual_gated_silu_projection_to_cpu_backend_buffer,
    },
    {
        "matches CPU F32 by F16 GGUF weight matmul on Metal backend",
        test_matches_cpu_f32_by_f16_gguf_weight_matmul_on_metal_backend,
    },
    {"reuses Metal F32 by F16 GGUF weight matmul state", test_reuses_metal_f32_by_f16_gguf_weight_matmul_state},
    {
        "matches CPU F32 by Q4_K GGUF weight matmul on Metal backend",
        test_matches_cpu_f32_by_q4_k_gguf_weight_matmul_on_metal_backend,
    },
    {
        "multiplies backend F32 by backend Q4_K GGUF weight to Metal backend buffer when available",
        test_multiplies_backend_f32_by_backend_q4_k_gguf_weight_to_metal_backend_when_available,
    },
    {
        "multiplies backend F32 by backend Q6_K GGUF weight to Metal backend buffer when available",
        test_multiplies_backend_f32_by_backend_q6_k_gguf_weight_to_metal_backend_when_available,
    },
    {
        "copies tensor to and from Metal backend buffer when available",
        test_copies_tensor_to_and_from_metal_backend_buffer_when_available,
    },
    {
        "copies tensor view to and from Metal backend buffer when available",
        test_copies_tensor_view_to_and_from_metal_backend_buffer_when_available,
    },
    {
        "allocates Metal backend buffer when available",
        test_allocates_metal_backend_buffer_when_available,
    },
    {
        "multiplies F32 by Metal backend F16 GGUF weight buffer when available",
        test_multiplies_f32_by_metal_backend_f16_gguf_weight_buffer_when_available,
    },
    {
        "multiplies backend F32 by backend F16 GGUF weight to Metal backend buffer when available",
        test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_metal_backend_buffer_when_available,
    },
    {
        "matches F16 GGUF token embedding to Metal backend buffer when available",
        test_matches_f16_gguf_token_embedding_to_metal_backend_buffer_when_available,
    },
    {
        "matches RMSNorm to Metal backend buffer when available",
        test_matches_rms_norm_to_metal_backend_buffer_when_available,
    },
    {
        "matches F32 add to Metal backend buffer when available",
        test_matches_f32_add_to_metal_backend_buffer_when_available,
    },
    {
        "matches headwise RoPE to Metal backend buffer when available",
        test_matches_headwise_rope_to_metal_backend_buffer_when_available,
    },
    {
        "matches causal grouped attention scores to Metal backend buffer when available",
        test_matches_causal_grouped_attention_scores_to_metal_backend_buffer_when_available,
    },
    {
        "matches causal grouped attention values to Metal backend buffer when available",
        test_matches_causal_grouped_attention_values_to_metal_backend_buffer_when_available,
    },
    {
        "matches softmax to Metal backend buffer when available",
        test_matches_softmax_to_metal_backend_buffer_when_available,
    },
    {
        "matches softcap to Metal backend buffer when available",
        test_matches_softcap_to_metal_backend_buffer_when_available,
    },
    {
        "matches top logit to Metal backend buffer when available",
        test_matches_top_logit_to_metal_backend_buffer_when_available,
    },
    {
        "multiplies backend F32 by three backend F16 GGUF weights to Metal backend buffers when available",
        test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_metal_backend_buffers_when_available,
    },
    {
        "applies SiLU multiply to Metal backend buffers when available",
        test_applies_silu_multiply_to_metal_backend_buffers_when_available,
    },
    {
        "runs gated SiLU projection to Metal backend buffer when available",
        test_runs_gated_silu_projection_to_metal_backend_buffer_when_available,
    },
    {
        "runs residual gated SiLU projection to Metal backend buffer when available",
        test_runs_residual_gated_silu_projection_to_metal_backend_buffer_when_available,
    },
    {"writes prefill KV cache to CPU backend buffer", test_writes_prefill_kv_cache_to_cpu_backend_buffer},
    {
        "matches prefill KV cache write to Metal backend buffer when available",
        test_matches_prefill_kv_cache_write_to_metal_backend_buffer_when_available,
    },
    {
        "applies causal grouped attention to CPU backend KV cache buffers",
        test_applies_causal_grouped_attention_to_cpu_backend_kv_cache_buffers,
    },
    {
        "matches causal grouped attention KV cache to Metal backend buffers when available",
        test_matches_causal_grouped_attention_kv_cache_to_metal_backend_buffers_when_available,
    },
    {
        "reports unavailable Metal backend for GGUF matmul when missing",
        test_reports_unavailable_metal_backend_for_gguf_matmul_when_missing,
    },
    {
        "reports unavailable Metal backend for K-quant GGUF matmul when missing",
        test_reports_unavailable_metal_backend_for_quantized_gguf_matmul_when_missing,
    },
    {
        "reports unavailable Metal backend buffer copy when missing",
        test_reports_unavailable_metal_backend_buffer_copy_when_missing,
    },
    {
        "reports unavailable Metal backend view copy when missing",
        test_reports_unavailable_metal_backend_view_copy_when_missing,
    },
    {
        "reports unavailable Metal backend allocation when missing",
        test_reports_unavailable_metal_backend_allocation_when_missing,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> tensor_backend_test_cases() {
    return std::span<const TestCase>(tensor_backend_tests);
}

}  // namespace local_gguf_engine::tests
