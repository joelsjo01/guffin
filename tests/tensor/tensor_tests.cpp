#include "tensor_tests.hpp"

#include "tensor_test_support.hpp"
#include "test_support.hpp"

#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
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
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using local_gguf_engine::model_format::GGUFReadError;
using local_gguf_engine::model_format::GGUFReadErrorCode;
using local_gguf_engine::model_format::load_gguf_tensor_payload;
using local_gguf_engine::model_format::read_gguf_model_view;
using local_gguf_engine::tensor::byte_size;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::element_byte_offset;
using local_gguf_engine::tensor::element_bytes;
using local_gguf_engine::tensor::element_count;
using local_gguf_engine::tensor::linear_index;
using local_gguf_engine::tensor::mutable_element_bytes;
using local_gguf_engine::tensor::dequantize_q4_k_block;
using local_gguf_engine::tensor::dequantize_q4_k_row;
using local_gguf_engine::tensor::dequantize_q6_k_block;
using local_gguf_engine::tensor::dequantize_q6_k_row;
using local_gguf_engine::tensor::dequantize_q8_0_block;
using local_gguf_engine::tensor::dot_q4_k_row_f32;
using local_gguf_engine::tensor::dot_q6_k_row_f32;
using local_gguf_engine::tensor::q4_k_block_count;
using local_gguf_engine::tensor::q4_k_block_view;
using local_gguf_engine::tensor::q6_k_block_count;
using local_gguf_engine::tensor::q6_k_block_view;
using local_gguf_engine::tensor::q8_0_block_count;
using local_gguf_engine::tensor::q8_0_block_view;
using local_gguf_engine::tensor::Q4_KBlockView;
using local_gguf_engine::tensor::Q6_KBlockView;
using local_gguf_engine::tensor::Q8_0BlockView;
using local_gguf_engine::tensor::read_f16_as_f32;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::write_f16_from_f32;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32;
using local_gguf_engine::tensor::ops::causal_scaled_dot_product_attention_f32;
using local_gguf_engine::tensor::ops::copy_f32;
using local_gguf_engine::tensor::ops::F32MatrixColumnView;
using local_gguf_engine::tensor::ops::F32RoPEMatrixColumnView;
using local_gguf_engine::tensor::ops::fill_f32;
using local_gguf_engine::tensor::ops::gelu_tanh_f32;
using local_gguf_engine::tensor::ops::matmul_f32;
using local_gguf_engine::tensor::ops::matmul_f32_f16;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight;
using local_gguf_engine::tensor::ops::mean_square_f32;
using local_gguf_engine::tensor::ops::multiply_f32;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rms_f32;
using local_gguf_engine::tensor::ops::rope_f32;
using local_gguf_engine::tensor::ops::rope_f32_by_heads_with_position_start;
using local_gguf_engine::tensor::ops::rope_neox_f32_by_heads_with_position_start;
using local_gguf_engine::tensor::ops::rope_f32_with_position_start;
using local_gguf_engine::tensor::ops::scaled_dot_product_attention_f32;
using local_gguf_engine::tensor::ops::silu_f32;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::sum_f32;
using local_gguf_engine::tensor::ops::sum_squares_f32;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::fixture;
using local_gguf_engine::tests::q4_k_reference_tensor;
using local_gguf_engine::tests::q6_k_reference_tensor;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f16_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;

void test_default_backend_starts_on_cpu() {
    expect(
        local_gguf_engine::runtime::default_backend()
            == local_gguf_engine::runtime::BackendKind::cpu,
        "default backend should start on CPU"
    );
}

void test_tensor_shape_accepts_valid_shapes() {
    const auto vector = Shape::from({128});
    expect(vector.rank() == 1, "1D tensor shape rank mismatch");
    expect(vector.dimension(0) == 128, "1D tensor dimension mismatch");
    expect(element_count(vector) == 128, "1D tensor element count mismatch");

    const auto matrix = Shape::from({2, 3});
    expect(matrix.rank() == 2, "2D tensor shape rank mismatch");
    expect(matrix.dimensions().size() == 2, "2D tensor dimensions span size mismatch");
    expect(matrix.dimensions()[0] == 2, "2D tensor dimension 0 mismatch");
    expect(matrix.dimensions()[1] == 3, "2D tensor dimension 1 mismatch");
    expect(element_count(matrix) == 6, "2D tensor element count mismatch");

    const auto rank_four = Shape::from({2, 3, 4, 5});
    expect(rank_four.rank() == Shape::max_rank, "4D tensor shape rank mismatch");
    expect(rank_four.dimension(3) == 5, "4D tensor final dimension mismatch");
    expect(element_count(rank_four) == 120, "4D tensor element count mismatch");
    expect(sizeof(Shape) <= 40, "tensor shape should stay compact");
}

void test_rejects_tensor_shape_rank_over_four() {
    const std::array<std::uint64_t, 5> dimensions {1, 2, 3, 4, 5};

    try {
        (void)Shape::from(std::span<const std::uint64_t>(dimensions));
        throw TestFailure("rank above 4 should throw");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "error code should be invalid_shape_rank");
        expect(error.expected_value() == Shape::max_rank, "expected shape rank mismatch");
        expect(error.actual_value() == 5, "actual shape rank mismatch");
    }
}

void test_rejects_tensor_shape_element_count_overflow() {
    const auto shape = Shape::from({std::numeric_limits<std::uint64_t>::max(), 2});

    try {
        (void)element_count(shape);
        throw TestFailure("element count overflow should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::element_count_overflow,
            "error code should be element_count_overflow"
        );
    }
}

void test_calculates_tensor_byte_sizes() {
    expect(byte_size(Shape::from({2, 3}), DType::f32) == 24, "F32 tensor byte size mismatch");
    expect(byte_size(Shape::from({2, 3}), DType::f16) == 12, "F16 tensor byte size mismatch");
    expect(byte_size(Shape::from({32}), DType::q8_0) == 34, "Q8_0 single-block byte size mismatch");
    expect(byte_size(Shape::from({2, 32}), DType::q8_0) == 68, "Q8_0 multi-block byte size mismatch");
    expect(byte_size(Shape::from({256}), DType::q4_k) == 144, "Q4_K single-block byte size mismatch");
    expect(byte_size(Shape::from({2, 256}), DType::q4_k) == 288, "Q4_K multi-block byte size mismatch");
    expect(byte_size(Shape::from({256}), DType::q6_k) == 210, "Q6_K single-block byte size mismatch");
    expect(byte_size(Shape::from({2, 256}), DType::q6_k) == 420, "Q6_K multi-block byte size mismatch");
}

void test_rejects_unsupported_tensor_dtype() {
    try {
        (void)byte_size(Shape::from({1}), static_cast<DType>(255));
        throw TestFailure("unsupported dtype should throw");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_dtype, "error code should be unsupported_dtype");
        expect(error.actual_value() == 255, "unsupported dtype value mismatch");
    }
}

void test_rejects_invalid_quantized_tensor_element_count() {
    try {
        (void)byte_size(Shape::from({31}), DType::q8_0);
        throw TestFailure("invalid Q8_0 element count should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "error code should be invalid_quantized_element_count"
        );
        expect(error.expected_value() == 32, "expected Q8_0 block element count mismatch");
        expect(error.actual_value() == 31, "actual Q8_0 element count mismatch");
    }
}

void test_rejects_invalid_quantized_tensor_row_element_count() {
    try {
        (void)byte_size(Shape::from({2, 128}), DType::q4_k);
        throw TestFailure("invalid Q4_K row element count should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "error code should be invalid_quantized_element_count"
        );
        expect(error.expected_value() == 256, "expected Q4_K row block element count mismatch");
        expect(error.actual_value() == 128, "actual Q4_K row element count mismatch");
    }
}

void test_creates_quantized_gguf_weight_descriptor_with_output_tail() {
    const auto descriptor = TensorDescriptor::from_quantized_gguf_weight_matrix(
        Shape::from({512, 5}),
        DType::q4_k
    );
    expect(descriptor.shape == Shape::from({512, 5}), "Q4_K GGUF weight descriptor shape mismatch");
    expect(descriptor.element_count == 2560, "Q4_K GGUF weight descriptor element count mismatch");
    expect(descriptor.byte_count == 1440, "Q4_K GGUF weight descriptor byte count mismatch");

    try {
        (void)TensorDescriptor::from_quantized_gguf_weight_matrix(Shape::from({384, 5}), DType::q4_k);
        throw TestFailure("Q4_K GGUF weight descriptor should reject unaligned inner width");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "Q4_K GGUF weight descriptor alignment error code mismatch"
        );
        expect(error.expected_value() == 256, "Q4_K GGUF weight descriptor expected alignment mismatch");
        expect(error.actual_value() == 384, "Q4_K GGUF weight descriptor actual alignment mismatch");
    }
}

void test_rejects_tensor_byte_size_overflow() {
    const auto shape = Shape::from({std::numeric_limits<std::uint64_t>::max() / 4 + 1});

    try {
        (void)byte_size(shape, DType::f32);
        throw TestFailure("byte size overflow should throw");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::byte_size_overflow, "error code should be byte_size_overflow");
    }
}

void test_creates_tensor_descriptor_from_shape_and_dtype() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f16);

    expect(descriptor.shape == Shape::from({2, 3}), "tensor descriptor shape mismatch");
    expect(descriptor.dtype == DType::f16, "tensor descriptor dtype mismatch");
    expect(descriptor.element_count == 6, "tensor descriptor element count mismatch");
    expect(descriptor.byte_count == 12, "tensor descriptor byte count mismatch");
}

void test_allocates_zero_initialized_tensor_storage() {
    const auto descriptor = TensorDescriptor::from(Shape::from({4}), DType::f32);
    auto storage = TensorStorage::allocate(descriptor);

    expect(storage.descriptor() == descriptor, "allocated tensor descriptor mismatch");
    expect(storage.bytes().size() == 16, "allocated tensor byte count mismatch");

    for (const auto byte : storage.bytes()) {
        expect(byte == 0, "allocated tensor storage should be zero initialized");
    }

    storage.mutable_bytes()[0] = 0x7F;
    expect(storage.bytes()[0] == 0x7F, "mutable tensor storage byte should be visible through const view");
}

void test_copies_tensor_storage_from_bytes() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, 34> bytes {
        0x00, 0x3C, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
        0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
        0x1F, 0x20,
    };

    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    expect(storage.descriptor() == descriptor, "copied tensor descriptor mismatch");
    expect(storage.bytes().size() == bytes.size(), "copied tensor byte count mismatch");
    expect(storage.bytes().data() != bytes.data(), "tensor storage should own a byte copy");
    expect(storage.bytes()[0] == 0x00, "copied tensor byte 0 mismatch");
    expect(storage.bytes()[1] == 0x3C, "copied tensor byte 1 mismatch");
    expect(storage.bytes()[33] == 0x20, "copied tensor final byte mismatch");
}

void test_rejects_tensor_storage_byte_count_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const std::array<std::uint8_t, 23> bytes {};

    try {
        (void)TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));
        throw TestFailure("storage byte count mismatch should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_storage_byte_count,
            "error code should be invalid_storage_byte_count"
        );
        expect(error.expected_value() == 24, "expected storage byte count mismatch");
        expect(error.actual_value() == 23, "actual storage byte count mismatch");
    }
}

void test_rejects_tensor_descriptor_with_invalid_shape_dtype_combination() {
    try {
        (void)TensorDescriptor::from(Shape::from({2, 15}), DType::q8_0);
        throw TestFailure("invalid Q8_0 descriptor shape should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "error code should be invalid_quantized_element_count"
        );
        expect(error.expected_value() == 32, "expected Q8_0 descriptor block size mismatch");
        expect(error.actual_value() == 15, "actual Q8_0 descriptor row element count mismatch");
    }
}

void test_decodes_q8_0_block_values() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, Q8_0BlockView::byte_count> bytes {
        0x00, 0x38,
        0x00, 0x02, 0xFE, 0x7F, 0x80, 0x04, 0x06, 0x08,
        0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18,
        0x1A, 0x1C, 0x1E, 0x20, 0x22, 0x24, 0x26, 0x28,
        0x2A, 0x2C, 0x2E, 0x30, 0x32, 0x34, 0x36, 0x38,
    };
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    expect(q8_0_block_count(descriptor) == 1, "Q8_0 block count mismatch");

    const auto block = q8_0_block_view(storage, 0);
    expect(block.scale_bits() == 0x3800U, "Q8_0 scale bits mismatch");
    expect(block.scale() == 0.5F, "Q8_0 scale decode mismatch");
    expect(block.quantized_bytes().size() == Q8_0BlockView::element_count, "Q8_0 quantized byte count mismatch");
    expect(block.quantized_value(0) == 0, "Q8_0 quantized value 0 mismatch");
    expect(block.quantized_value(1) == 2, "Q8_0 quantized value 1 mismatch");
    expect(block.quantized_value(2) == static_cast<std::int8_t>(-2), "Q8_0 quantized value 2 mismatch");
    expect(block.quantized_value(3) == 127, "Q8_0 quantized value 3 mismatch");
    expect(block.quantized_value(4) == static_cast<std::int8_t>(-128), "Q8_0 quantized value 4 mismatch");

    const auto values = dequantize_q8_0_block(storage, 0);
    expect(values[0] == 0.0F, "Q8_0 dequantized value 0 mismatch");
    expect(values[1] == 1.0F, "Q8_0 dequantized value 1 mismatch");
    expect(values[2] == -1.0F, "Q8_0 dequantized value 2 mismatch");
    expect(values[3] == 63.5F, "Q8_0 dequantized value 3 mismatch");
    expect(values[4] == -64.0F, "Q8_0 dequantized value 4 mismatch");
}

void test_decodes_q4_k_reference_block_values() {
    const auto storage = q4_k_reference_tensor();
    const auto block = q4_k_block_view(storage, 0);

    expect(block.scale_value(0) == 2, "Q4_K scale group 0 mismatch");
    expect(block.min_value(0) == 1, "Q4_K min group 0 mismatch");
    expect(block.scale_value(4) == 6, "Q4_K scale group 4 mismatch");
    expect(block.min_value(4) == 5, "Q4_K min group 4 mismatch");
    expect(block.quantized_nibble(0) == 4, "Q4_K quant group 0 start mismatch");
    expect(block.quantized_nibble(31) == 15, "Q4_K quant group 0 final mismatch");
    expect(block.quantized_nibble(32) == 5, "Q4_K quant group 1 start mismatch");
    expect(block.quantized_nibble(63) == 6, "Q4_K quant group 1 final mismatch");
    expect(block.quantized_nibble(128) == 7, "Q4_K quant group 4 start mismatch");
    expect_close(block.dequantized_value(0), 7.5F, 0.0F, "Q4_K dequantized value 0 mismatch");
    expect_close(block.dequantized_value(32), 14.0F, 0.0F, "Q4_K dequantized value 32 mismatch");
    expect_close(block.dequantized_value(128), 39.5F, 0.0F, "Q4_K dequantized value 128 mismatch");

    const auto values = dequantize_q4_k_block(storage, 0);
    expect_close(values[31], 29.5F, 0.0F, "Q4_K dequantized value 31 mismatch");
    expect_close(values[63], 17.0F, 0.0F, "Q4_K dequantized value 63 mismatch");
}

void test_dequantizes_q4_k_reference_row_and_dot() {
    const auto storage = q4_k_reference_tensor();
    const auto row = dequantize_q4_k_row(storage, 0);
    std::array<float, 256> input {};
    input[0] = 1.0F;
    input[32] = 2.0F;
    input[128] = -1.0F;

    expect(row.size() == input.size(), "Q4_K row dequantized width mismatch");
    expect_close(row[0], 7.5F, 0.0F, "Q4_K row value 0 mismatch");
    expect_close(row[32], 14.0F, 0.0F, "Q4_K row value 32 mismatch");
    expect_close(row[128], 39.5F, 0.0F, "Q4_K row value 128 mismatch");
    expect_close(
        dot_q4_k_row_f32(storage, 0, std::span<const float>(input)),
        -4.0F,
        0.0F,
        "Q4_K row dot mismatch"
    );
}

void test_decodes_q6_k_reference_block_values() {
    const auto storage = q6_k_reference_tensor();
    const auto block = q6_k_block_view(storage, 0);

    expect(block.scale_value(0) == 2, "Q6_K scale group 0 mismatch");
    expect(block.scale_value(2) == -3, "Q6_K scale group 2 mismatch");
    expect(block.low_nibble(0) == 0, "Q6_K low nibble value 0 mismatch");
    expect(block.high_bits(0) == 0, "Q6_K high bits value 0 mismatch");
    expect(block.quantized_value(0) == -32, "Q6_K quantized value 0 mismatch");
    expect(block.low_nibble(32) == 15, "Q6_K low nibble value 32 mismatch");
    expect(block.high_bits(32) == 1, "Q6_K high bits value 32 mismatch");
    expect(block.quantized_value(32) == -1, "Q6_K quantized value 32 mismatch");
    expect(block.low_nibble(64) == 5, "Q6_K low nibble value 64 mismatch");
    expect(block.high_bits(64) == 2, "Q6_K high bits value 64 mismatch");
    expect(block.quantized_value(64) == 5, "Q6_K quantized value 64 mismatch");
    expect_close(block.dequantized_value(0), -32.0F, 0.0F, "Q6_K dequantized value 0 mismatch");
    expect_close(block.dequantized_value(32), 1.5F, 0.0F, "Q6_K dequantized value 32 mismatch");
    expect_close(block.dequantized_value(64), 10.0F, 0.0F, "Q6_K dequantized value 64 mismatch");

    const auto values = dequantize_q6_k_block(storage, 0);
    expect_close(values[31], 46.5F, 0.0F, "Q6_K dequantized value 31 mismatch");
    expect_close(values[128], -7.0F, 0.0F, "Q6_K dequantized value 128 mismatch");
}

void test_dequantizes_q6_k_reference_row_and_dot() {
    const auto storage = q6_k_reference_tensor();
    const auto row = dequantize_q6_k_row(storage, 0);
    std::array<float, 256> input {};
    input[0] = 1.0F;
    input[32] = 2.0F;
    input[64] = -1.0F;
    input[128] = 3.0F;

    expect(row.size() == input.size(), "Q6_K row dequantized width mismatch");
    expect_close(row[0], -32.0F, 0.0F, "Q6_K row value 0 mismatch");
    expect_close(row[32], 1.5F, 0.0F, "Q6_K row value 32 mismatch");
    expect_close(row[64], 10.0F, 0.0F, "Q6_K row value 64 mismatch");
    expect_close(
        dot_q6_k_row_f32(storage, 0, std::span<const float>(input)),
        -60.0F,
        0.0F,
        "Q6_K row dot mismatch"
    );
}

void test_rejects_q4_k_reference_row_width_mismatch() {
    const auto storage = q4_k_reference_tensor();
    std::array<float, 255> input {};

    try {
        (void)dot_q4_k_row_f32(storage, 0, std::span<const float>(input));
        throw TestFailure("Q4_K row dot should reject input width mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Q4_K width mismatch error code mismatch");
        expect(error.expected_value() == 256, "Q4_K width mismatch expected value mismatch");
        expect(error.actual_value() == 255, "Q4_K width mismatch actual value mismatch");
    }
}

void test_rejects_q6_k_reference_row_out_of_range() {
    const auto storage = q6_k_reference_tensor();

    try {
        (void)dequantize_q6_k_row(storage, 1);
        throw TestFailure("Q6_K row dequantize should reject row out of range");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_index_out_of_range, "Q6_K row range error code mismatch");
        expect(error.expected_value() == 1, "Q6_K row range expected value mismatch");
        expect(error.actual_value() == 1, "Q6_K row range actual value mismatch");
    }
}

void test_views_loaded_gguf_q8_0_block() {
    const auto bytes = fixture("valid-q8-0-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(q8_0_block_count(loaded.storage.descriptor()) == 1, "loaded Q8_0 block count mismatch");

    const auto block = q8_0_block_view(loaded.storage, 0);
    expect(block.scale_bits() == 0x0100U, "loaded Q8_0 scale bits mismatch");
    expect(block.quantized_value(0) == 2, "loaded Q8_0 quantized value 0 mismatch");
    expect(block.quantized_value(31) == 33, "loaded Q8_0 quantized final value mismatch");
}

void test_views_loaded_gguf_q4_k_block() {
    const auto bytes = fixture("valid-q4-k-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(loaded.storage.descriptor().dtype == DType::q4_k, "loaded Q4_K dtype mismatch");
    expect(q4_k_block_count(loaded.storage.descriptor()) == 1, "loaded Q4_K block count mismatch");

    const auto block = q4_k_block_view(loaded.storage, 0);
    expect(block.scale_bits() == 0x3800U, "Q4_K scale bits mismatch");
    expect(block.min_scale_bits() == 0x3C00U, "Q4_K min scale bits mismatch");
    expect(block.scale() == 0.5F, "Q4_K scale decode mismatch");
    expect(block.min_scale() == 1.0F, "Q4_K min scale decode mismatch");
    expect(block.scale_min_bytes().size() == Q4_KBlockView::scale_min_byte_count, "Q4_K scale byte count mismatch");
    expect(block.scale_min_bytes()[0] == 0x10, "Q4_K scale byte 0 mismatch");
    expect(block.scale_min_bytes()[11] == 0x1B, "Q4_K scale final byte mismatch");
    expect(block.quantized_bytes().size() == Q4_KBlockView::quantized_byte_count, "Q4_K quant byte count mismatch");
    expect(block.quantized_bytes()[127] == 0x7F, "Q4_K quant final byte mismatch");
    expect(block.quantized_nibble(2) == 2, "Q4_K quantized nibble 2 mismatch");
    expect(block.quantized_nibble(255) == 7, "Q4_K quantized final nibble mismatch");
}

void test_views_loaded_gguf_q6_k_block() {
    const auto bytes = fixture("valid-q6-k-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(loaded.storage.descriptor().dtype == DType::q6_k, "loaded Q6_K dtype mismatch");
    expect(q6_k_block_count(loaded.storage.descriptor()) == 1, "loaded Q6_K block count mismatch");

    const auto block = q6_k_block_view(loaded.storage, 0);
    expect(block.low_bit_bytes().size() == Q6_KBlockView::low_bit_byte_count, "Q6_K low byte count mismatch");
    expect(block.high_bit_bytes().size() == Q6_KBlockView::high_bit_byte_count, "Q6_K high byte count mismatch");
    expect(block.scale_bytes().size() == Q6_KBlockView::scale_byte_count, "Q6_K scale byte count mismatch");
    expect(block.low_bit_bytes()[127] == 0x7F, "Q6_K low final byte mismatch");
    expect(block.high_bit_bytes()[0] == 0x80, "Q6_K high byte 0 mismatch");
    expect(block.scale_bits() == 0x4000U, "Q6_K scale bits mismatch");
    expect(block.scale() == 2.0F, "Q6_K scale decode mismatch");
    expect(block.low_nibble(2) == 2, "Q6_K low nibble 2 mismatch");
    expect(block.high_bits(3) == 3, "Q6_K high bits 3 mismatch");
    expect(block.scale_value(0) == static_cast<std::int8_t>(-16), "Q6_K scale value 0 mismatch");
    expect(block.scale_value(15) == static_cast<std::int8_t>(-1), "Q6_K scale value 15 mismatch");
}

void test_rejects_q4_k_block_element_index_out_of_range() {
    const auto descriptor = TensorDescriptor::from(Shape::from({256}), DType::q4_k);
    const std::array<std::uint8_t, Q4_KBlockView::byte_count> bytes {};
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));
    const auto block = q4_k_block_view(storage, 0);

    try {
        (void)block.quantized_nibble(Q4_KBlockView::element_count);
        throw TestFailure("Q4_K block element index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::quantized_block_element_index_out_of_range,
            "error code should be quantized_block_element_index_out_of_range"
        );
        expect(error.expected_value() == Q4_KBlockView::element_count, "expected Q4_K element count mismatch");
        expect(error.actual_value() == Q4_KBlockView::element_count, "actual Q4_K element index mismatch");
    }
}

void test_rejects_q6_k_scale_index_out_of_range() {
    const auto descriptor = TensorDescriptor::from(Shape::from({256}), DType::q6_k);
    const std::array<std::uint8_t, Q6_KBlockView::byte_count> bytes {};
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));
    const auto block = q6_k_block_view(storage, 0);

    try {
        (void)block.scale_value(Q6_KBlockView::scale_byte_count);
        throw TestFailure("Q6_K scale index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::quantized_block_element_index_out_of_range,
            "error code should be quantized_block_element_index_out_of_range"
        );
        expect(error.expected_value() == Q6_KBlockView::scale_byte_count, "expected Q6_K scale count mismatch");
        expect(error.actual_value() == Q6_KBlockView::scale_byte_count, "actual Q6_K scale index mismatch");
    }
}

void test_rejects_q8_0_block_access_for_unquantized_dtype() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);

    try {
        (void)q8_0_block_count(descriptor);
        throw TestFailure("Q8_0 block count should reject F32 descriptor");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_quantized_dtype,
            "error code should be unsupported_quantized_dtype"
        );
        expect(error.expected_value() == 8, "expected Q8_0 dtype id mismatch");
        expect(error.actual_value() == 0, "actual F32 dtype id mismatch");
    }
}

void test_rejects_q8_0_block_index_out_of_range() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, Q8_0BlockView::byte_count> bytes {};
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    try {
        (void)q8_0_block_view(storage, 1);
        throw TestFailure("Q8_0 block index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::quantized_block_index_out_of_range,
            "error code should be quantized_block_index_out_of_range"
        );
        expect(error.expected_value() == 1, "expected Q8_0 block count mismatch");
        expect(error.actual_value() == 1, "actual Q8_0 block index mismatch");
    }
}

void test_rejects_q8_0_block_element_index_out_of_range() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, Q8_0BlockView::byte_count> bytes {};
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));
    const auto block = q8_0_block_view(storage, 0);

    try {
        (void)block.quantized_value(Q8_0BlockView::element_count);
        throw TestFailure("Q8_0 block element index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::quantized_block_element_index_out_of_range,
            "error code should be quantized_block_element_index_out_of_range"
        );
        expect(error.expected_value() == Q8_0BlockView::element_count, "expected Q8_0 element count mismatch");
        expect(error.actual_value() == Q8_0BlockView::element_count, "actual Q8_0 element index mismatch");
    }
}

void test_calculates_row_major_linear_indices() {
    const auto matrix = Shape::from({2, 3});
    const std::array<std::uint64_t, 2> matrix_index {1, 2};
    expect(
        linear_index(matrix, std::span<const std::uint64_t>(matrix_index)) == 5,
        "2D row-major linear index mismatch"
    );

    const auto rank_three = Shape::from({2, 3, 4});
    const std::array<std::uint64_t, 3> rank_three_index {1, 2, 3};
    expect(
        linear_index(rank_three, std::span<const std::uint64_t>(rank_three_index)) == 23,
        "3D row-major linear index mismatch"
    );
}

void test_calculates_unquantized_element_byte_offsets() {
    const auto f32_descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const std::array<std::uint64_t, 2> f32_index {1, 2};
    expect(
        element_byte_offset(f32_descriptor, std::span<const std::uint64_t>(f32_index)) == 20,
        "F32 element byte offset mismatch"
    );

    const auto f16_descriptor = TensorDescriptor::from(Shape::from({2, 3, 4}), DType::f16);
    const std::array<std::uint64_t, 3> f16_index {1, 0, 2};
    expect(
        element_byte_offset(f16_descriptor, std::span<const std::uint64_t>(f16_index)) == 28,
        "F16 element byte offset mismatch"
    );
}

void test_reads_and_mutates_unquantized_element_bytes() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    const std::array<std::uint8_t, 16> bytes {
        0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B,
        0x0C, 0x0D, 0x0E, 0x0F,
    };
    auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    const std::array<std::uint64_t, 2> read_index {1, 0};
    const auto read_bytes = element_bytes(storage, std::span<const std::uint64_t>(read_index));
    expect(read_bytes.size() == 4, "F32 element byte view size mismatch");
    expect(read_bytes[0] == 0x08, "F32 element byte view byte 0 mismatch");
    expect(read_bytes[3] == 0x0B, "F32 element byte view byte 3 mismatch");

    const std::array<std::uint64_t, 2> write_index {0, 1};
    auto write_bytes = mutable_element_bytes(storage, std::span<const std::uint64_t>(write_index));
    write_bytes[0] = 0xAA;
    write_bytes[3] = 0xDD;
    expect(storage.bytes()[4] == 0xAA, "mutable F32 element byte 0 mismatch");
    expect(storage.bytes()[7] == 0xDD, "mutable F32 element byte 3 mismatch");
}

void test_rejects_tensor_index_rank_mismatch() {
    const auto shape = Shape::from({2, 3});
    const std::array<std::uint64_t, 1> index {1};

    try {
        (void)linear_index(shape, std::span<const std::uint64_t>(index));
        throw TestFailure("tensor index rank mismatch should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_tensor_index_rank,
            "error code should be invalid_tensor_index_rank"
        );
        expect(error.expected_value() == 2, "expected tensor index rank mismatch");
        expect(error.actual_value() == 1, "actual tensor index rank mismatch");
    }
}

void test_rejects_tensor_index_out_of_range() {
    const auto shape = Shape::from({2, 3});
    const std::array<std::uint64_t, 2> index {2, 0};

    try {
        (void)linear_index(shape, std::span<const std::uint64_t>(index));
        throw TestFailure("tensor index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_index_out_of_range,
            "error code should be tensor_index_out_of_range"
        );
        expect(error.expected_value() == 2, "expected tensor index dimension size mismatch");
        expect(error.actual_value() == 2, "actual tensor index mismatch");
    }
}

void test_rejects_quantized_tensor_element_indexing() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint64_t, 1> index {0};

    try {
        (void)element_byte_offset(descriptor, std::span<const std::uint64_t>(index));
        throw TestFailure("quantized tensor element indexing should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_indexed_dtype,
            "error code should be unsupported_indexed_dtype"
        );
        expect(error.actual_value() == 8, "unsupported indexed dtype value mismatch");
    }
}

void test_reads_and_writes_f32_scalars_by_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto storage = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 2> first_index {0, 1};
    const std::array<std::uint64_t, 2> second_index {1, 0};
    write_f32(storage, std::span<const std::uint64_t>(first_index), 1.5F);
    write_f32(storage, std::span<const std::uint64_t>(second_index), -2.25F);

    expect(read_f32(storage, std::span<const std::uint64_t>(first_index)) == 1.5F, "F32 read/write value mismatch");
    expect(read_f32(storage, std::span<const std::uint64_t>(second_index)) == -2.25F, "F32 read/write negative value mismatch");

    const auto first_bytes = element_bytes(storage, std::span<const std::uint64_t>(first_index));
    expect(first_bytes.size() == sizeof(float), "F32 typed accessor byte size mismatch");
}

void test_reads_f32_scalars_from_loaded_gguf_payload() {
    const auto bytes = fixture("valid-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    const std::array<std::uint64_t, 1> first_index {0};
    const std::array<std::uint64_t, 1> second_index {1};
    expect(read_f32(loaded.storage, std::span<const std::uint64_t>(first_index)) == 1.0F, "loaded GGUF F32 scalar 0 mismatch");
    expect(read_f32(loaded.storage, std::span<const std::uint64_t>(second_index)) == 2.0F, "loaded GGUF F32 scalar 1 mismatch");
}

void test_rejects_f32_accessor_dtype_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f16);
    auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {0};

    try {
        (void)read_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F32 read accessor should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "error code should be unsupported_typed_accessor_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 dtype id mismatch");
    }

    try {
        write_f32(storage, std::span<const std::uint64_t>(index), 1.0F);
        throw TestFailure("F32 write accessor should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "write error code should be unsupported_typed_accessor_dtype"
        );
    }
}

void test_rejects_f32_accessor_invalid_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);
    const auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {2};

    try {
        (void)read_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F32 accessor invalid index should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_index_out_of_range,
            "error code should be tensor_index_out_of_range"
        );
        expect(error.expected_value() == 2, "expected F32 accessor dimension size mismatch");
        expect(error.actual_value() == 2, "actual F32 accessor index mismatch");
    }
}

void test_reads_and_writes_f16_scalars_by_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({3}), DType::f16);
    auto storage = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 1> first_index {0};
    const std::array<std::uint64_t, 1> second_index {1};
    const std::array<std::uint64_t, 1> third_index {2};
    write_f16_from_f32(storage, std::span<const std::uint64_t>(first_index), 1.0F);
    write_f16_from_f32(storage, std::span<const std::uint64_t>(second_index), -2.0F);
    write_f16_from_f32(storage, std::span<const std::uint64_t>(third_index), 0.5F);

    expect(read_f16_as_f32(storage, std::span<const std::uint64_t>(first_index)) == 1.0F, "F16 read/write value 0 mismatch");
    expect(read_f16_as_f32(storage, std::span<const std::uint64_t>(second_index)) == -2.0F, "F16 read/write value 1 mismatch");
    expect(read_f16_as_f32(storage, std::span<const std::uint64_t>(third_index)) == 0.5F, "F16 read/write value 2 mismatch");

    expect(storage.bytes()[0] == 0x00, "F16 value 0 low byte mismatch");
    expect(storage.bytes()[1] == 0x3C, "F16 value 0 high byte mismatch");
    expect(storage.bytes()[2] == 0x00, "F16 value 1 low byte mismatch");
    expect(storage.bytes()[3] == 0xC0, "F16 value 1 high byte mismatch");
    expect(storage.bytes()[4] == 0x00, "F16 value 2 low byte mismatch");
    expect(storage.bytes()[5] == 0x38, "F16 value 2 high byte mismatch");
}

void test_reads_f16_scalars_from_loaded_gguf_payload() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    const std::array<std::uint64_t, 1> first_index {0};
    const std::array<std::uint64_t, 1> second_index {1};
    expect(read_f16_as_f32(loaded.storage, std::span<const std::uint64_t>(first_index)) == 1.0F, "loaded GGUF F16 scalar 0 mismatch");
    expect(read_f16_as_f32(loaded.storage, std::span<const std::uint64_t>(second_index)) == 2.0F, "loaded GGUF F16 scalar 1 mismatch");
}

void test_rejects_f16_accessor_dtype_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);
    auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {0};

    try {
        (void)read_f16_as_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F16 read accessor should reject F32 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "error code should be unsupported_typed_accessor_dtype"
        );
        expect(error.expected_value() == 1, "expected F16 dtype id mismatch");
        expect(error.actual_value() == 0, "actual F32 dtype id mismatch");
    }

    try {
        write_f16_from_f32(storage, std::span<const std::uint64_t>(index), 1.0F);
        throw TestFailure("F16 write accessor should reject F32 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "write error code should be unsupported_typed_accessor_dtype"
        );
    }
}

void test_rejects_f16_accessor_invalid_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f16);
    const auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {2};

    try {
        (void)read_f16_as_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F16 accessor invalid index should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_index_out_of_range,
            "error code should be tensor_index_out_of_range"
        );
        expect(error.expected_value() == 2, "expected F16 accessor dimension size mismatch");
        expect(error.actual_value() == 2, "actual F16 accessor index mismatch");
    }
}

void test_adds_f32_tensors_elementwise() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto left = TensorStorage::allocate(descriptor);
    auto right = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};

    write_f32(left, std::span<const std::uint64_t>(index_00), 1.0F);
    write_f32(left, std::span<const std::uint64_t>(index_01), -2.0F);
    write_f32(left, std::span<const std::uint64_t>(index_10), 3.5F);
    write_f32(left, std::span<const std::uint64_t>(index_11), 4.0F);

    write_f32(right, std::span<const std::uint64_t>(index_00), 0.5F);
    write_f32(right, std::span<const std::uint64_t>(index_01), 2.0F);
    write_f32(right, std::span<const std::uint64_t>(index_10), -1.5F);
    write_f32(right, std::span<const std::uint64_t>(index_11), 8.0F);

    const auto output = add_f32(left, right);

    expect(output.descriptor() == descriptor, "F32 add output descriptor mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 1.5F, "F32 add value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 0.0F, "F32 add value 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 2.0F, "F32 add value 2 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 12.0F, "F32 add value 3 mismatch");
    expect(read_f32(left, std::span<const std::uint64_t>(index_00)) == 1.0F, "F32 add should not mutate left input");
    expect(read_f32(right, std::span<const std::uint64_t>(index_11)) == 8.0F, "F32 add should not mutate right input");
}

void test_multiplies_f32_tensors_elementwise() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto left = TensorStorage::allocate(descriptor);
    auto right = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};

    write_f32(left, std::span<const std::uint64_t>(index_00), 1.0F);
    write_f32(left, std::span<const std::uint64_t>(index_01), -2.0F);
    write_f32(left, std::span<const std::uint64_t>(index_10), 3.5F);
    write_f32(left, std::span<const std::uint64_t>(index_11), 4.0F);

    write_f32(right, std::span<const std::uint64_t>(index_00), 0.5F);
    write_f32(right, std::span<const std::uint64_t>(index_01), 2.0F);
    write_f32(right, std::span<const std::uint64_t>(index_10), -1.5F);
    write_f32(right, std::span<const std::uint64_t>(index_11), 8.0F);

    const auto output = multiply_f32(left, right);

    expect(output.descriptor() == descriptor, "F32 multiply output descriptor mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 0.5F, "F32 multiply value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == -4.0F, "F32 multiply value 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == -5.25F, "F32 multiply value 2 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 32.0F, "F32 multiply value 3 mismatch");
}

void test_rejects_invalid_f32_elementwise_inputs() {
    const auto f32_vector = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto f16_vector = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));
    const auto f32_matrix_2x3 = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto f32_matrix_2x2 = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)add_f32(f32_vector, f16_vector);
        throw TestFailure("F32 add should reject F16 operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 add dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 add expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 add actual dtype mismatch");
    }

    try {
        (void)add_f32(f32_matrix_2x3, f32_matrix_2x2);
        throw TestFailure("F32 add should reject mismatched shapes");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 add shape error code mismatch");
        expect(error.expected_value() == 3, "F32 add expected shape dimension mismatch");
        expect(error.actual_value() == 2, "F32 add actual shape dimension mismatch");
    }

    try {
        (void)multiply_f32(f32_vector, f16_vector);
        throw TestFailure("F32 multiply should reject F16 operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 multiply dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 multiply expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 multiply actual dtype mismatch");
    }

    try {
        (void)multiply_f32(f32_matrix_2x3, f32_matrix_2x2);
        throw TestFailure("F32 multiply should reject mismatched shapes");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 multiply shape error code mismatch");
        expect(error.expected_value() == 3, "F32 multiply expected shape dimension mismatch");
        expect(error.actual_value() == 2, "F32 multiply actual shape dimension mismatch");
    }
}

void test_applies_f32_silu() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    write_f32(input, std::span<const std::uint64_t>(index_0), -1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 2.0F);

    const auto output = silu_f32(input);

    expect(output.descriptor() == input.descriptor(), "F32 SiLU output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 SiLU output should not alias input");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(-1.0 / (1.0 + std::exp(1.0))),
        0.000001F,
        "F32 SiLU value 0 mismatch"
    );
    expect(read_f32(output, std::span<const std::uint64_t>(index_1)) == 0.0F, "F32 SiLU value 1 mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_2)),
        static_cast<float>(2.0 / (1.0 + std::exp(-2.0))),
        0.000001F,
        "F32 SiLU value 2 mismatch"
    );
}

void test_rejects_f32_silu_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)silu_f32(input);
        throw TestFailure("F32 SiLU should reject F16 input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 SiLU dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 SiLU expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 SiLU actual dtype mismatch");
    }
}

void test_applies_f32_gelu_tanh_approximation() {
    const std::array<float, 5> values {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F};
    const auto input = tensor_f32_vector(std::span<const float>(values));
    const auto output = gelu_tanh_f32(input);
    const std::array<float, 5> expected {
        -0.0454023F,
        -0.158808F,
        0.0F,
        0.841192F,
        1.95460F,
    };

    expect(output.descriptor() == input.descriptor(), "F32 GELU-tanh output descriptor mismatch");
    for (std::uint64_t index = 0; index < expected.size(); ++index) {
        const std::array<std::uint64_t, 1> tensor_index {index};
        expect_close(
            read_f32(output, std::span<const std::uint64_t>(tensor_index)),
            expected[static_cast<std::size_t>(index)],
            0.00001F,
            "F32 GELU-tanh value mismatch"
        );
    }
}

void test_fills_f32_tensor_with_scalar() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const auto output = fill_f32(descriptor, -3.25F);

    expect(output.descriptor() == descriptor, "F32 fill output descriptor mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect(read_f32(output, std::span<const std::uint64_t>(index)) == -3.25F, "F32 fill value mismatch");
        }
    }
}

void test_rejects_f32_fill_dtype_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f16);

    try {
        (void)fill_f32(descriptor, 1.0F);
        throw TestFailure("F32 fill should reject F16 descriptor");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 fill dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 fill dtype id mismatch");
    }
}

void test_copies_f32_tensor_storage() {
    const auto descriptor = TensorDescriptor::from(Shape::from({3}), DType::f32);
    auto input = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};

    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 2.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 3.0F);

    auto output = copy_f32(input);

    expect(output.descriptor() == descriptor, "F32 copy output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 copy should own distinct storage");
    expect(read_f32(output, std::span<const std::uint64_t>(index_0)) == 1.0F, "F32 copy value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_1)) == 2.0F, "F32 copy value 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_2)) == 3.0F, "F32 copy value 2 mismatch");

    write_f32(output, std::span<const std::uint64_t>(index_0), 9.0F);
    expect(read_f32(input, std::span<const std::uint64_t>(index_0)) == 1.0F, "F32 copy should not alias input");
    expect(read_f32(output, std::span<const std::uint64_t>(index_0)) == 9.0F, "F32 copy mutation mismatch");
}

void test_rejects_f32_copy_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({32}), DType::q8_0));

    try {
        (void)copy_f32(input);
        throw TestFailure("F32 copy should reject Q8_0 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 copy dtype id mismatch");
        expect(error.actual_value() == 8, "actual Q8_0 copy dtype id mismatch");
    }
}

void test_reduces_f32_tensor_sum_and_sum_squares() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto input = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};

    write_f32(input, std::span<const std::uint64_t>(index_00), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_01), -2.0F);
    write_f32(input, std::span<const std::uint64_t>(index_10), 3.5F);
    write_f32(input, std::span<const std::uint64_t>(index_11), 0.5F);

    expect(sum_f32(input) == 3.0F, "F32 sum mismatch");
    expect(sum_squares_f32(input) == 17.5F, "F32 sum squares mismatch");
}

void test_reduces_loaded_gguf_f32_payload() {
    const auto bytes = fixture("valid-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(sum_f32(loaded.storage) == 3.0F, "loaded GGUF F32 sum mismatch");
    expect(sum_squares_f32(loaded.storage) == 5.0F, "loaded GGUF F32 sum squares mismatch");
}

void test_rejects_f32_reduction_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)sum_f32(input);
        throw TestFailure("F32 sum should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "sum error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 sum dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 sum dtype id mismatch");
    }

    try {
        (void)sum_squares_f32(input);
        throw TestFailure("F32 sum squares should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "sum squares error code should be unsupported_operator_dtype"
        );
    }
}

void test_calculates_f32_mean_square_and_rms() {
    const auto descriptor = TensorDescriptor::from(Shape::from({4}), DType::f32);
    auto input = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    const std::array<std::uint64_t, 1> index_3 {3};

    write_f32(input, std::span<const std::uint64_t>(index_0), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 4.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_3), 0.0F);

    expect(mean_square_f32(input) == 6.25F, "F32 mean square mismatch");
    expect(rms_f32(input) == 2.5F, "F32 RMS mismatch");
}

void test_rejects_f32_rms_helpers_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)mean_square_f32(input);
        throw TestFailure("F32 mean square should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "mean square error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 mean square dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 mean square dtype id mismatch");
    }

    try {
        (void)rms_f32(input);
        throw TestFailure("F32 RMS should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "RMS error code should be unsupported_operator_dtype"
        );
    }
}

void test_applies_f32_rms_norm_to_vector() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    write_f32(input, std::span<const std::uint64_t>(index_0), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 4.0F);
    write_f32(weight, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(weight, std::span<const std::uint64_t>(index_1), 2.0F);

    const auto output = rms_norm_f32(input, weight, 3.5F);

    expect(output.descriptor() == input.descriptor(), "F32 RMSNorm vector output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 RMSNorm vector output should not alias input");
    expect(read_f32(output, std::span<const std::uint64_t>(index_0)) == 0.75F, "F32 RMSNorm vector value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_1)) == 2.0F, "F32 RMSNorm vector value 1 mismatch");
}

void test_applies_f32_rms_norm_to_batched_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 4.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), -3.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 4.0F);

    const std::array<std::uint64_t, 1> weight0 {0};
    const std::array<std::uint64_t, 1> weight1 {1};
    write_f32(weight, std::span<const std::uint64_t>(weight0), 1.0F);
    write_f32(weight, std::span<const std::uint64_t>(weight1), 2.0F);

    const auto output = rms_norm_f32(input, weight, 3.5F);

    expect(output.descriptor() == input.descriptor(), "F32 RMSNorm batched output descriptor mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row0_col0)) == 0.75F, "F32 RMSNorm batch value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row0_col1)) == 2.0F, "F32 RMSNorm batch value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row1_col0)) == -0.75F, "F32 RMSNorm batch value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row1_col1)) == 2.0F, "F32 RMSNorm batch value 1,1 mismatch");
}

void test_rejects_f32_rms_norm_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject F16 weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 RMSNorm dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 RMSNorm expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 RMSNorm actual dtype mismatch");
    }
}

void test_rejects_f32_rms_norm_input_rank_above_two() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2, 2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject rank-three input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 RMSNorm input rank error mismatch");
        expect(error.expected_value() == 2, "F32 RMSNorm expected input rank mismatch");
        expect(error.actual_value() == 3, "F32 RMSNorm actual input rank mismatch");
    }
}

void test_rejects_f32_rms_norm_weight_rank_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject rank-two weight");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 RMSNorm weight rank error mismatch");
        expect(error.expected_value() == 1, "F32 RMSNorm expected weight rank mismatch");
        expect(error.actual_value() == 2, "F32 RMSNorm actual weight rank mismatch");
    }
}

void test_rejects_f32_rms_norm_hidden_dimension_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject hidden dimension mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 RMSNorm hidden shape error mismatch");
        expect(error.expected_value() == 3, "F32 RMSNorm expected hidden dimension mismatch");
        expect(error.actual_value() == 2, "F32 RMSNorm actual hidden dimension mismatch");
    }
}

void test_rejects_f32_rms_norm_nonpositive_epsilon() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.0F);
        throw TestFailure("F32 RMSNorm should reject nonpositive epsilon");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RMSNorm epsilon must be positive",
            "F32 RMSNorm epsilon error mismatch"
        );
    }

    try {
        (void)rms_norm_f32(input, weight, std::numeric_limits<float>::quiet_NaN());
        throw TestFailure("F32 RMSNorm should reject NaN epsilon");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RMSNorm epsilon must be positive",
            "F32 RMSNorm NaN epsilon error mismatch"
        );
    }
}

void test_applies_f32_rope_to_vector() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({5}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    const std::array<std::uint64_t, 1> index_3 {3};
    const std::array<std::uint64_t, 1> index_4 {4};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_3), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_4), 9.0F);

    const auto output = rope_f32(input, 1, 4, 1.0F);
    const auto cosine = static_cast<float>(std::cos(1.0));
    const auto sine = static_cast<float>(std::sin(1.0));

    expect(output.descriptor() == input.descriptor(), "F32 RoPE vector output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 RoPE vector output should not alias input");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_0)), cosine, 0.000001F, "F32 RoPE vector value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_1)), sine, 0.000001F, "F32 RoPE vector value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_2)), -sine, 0.000001F, "F32 RoPE vector value 2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_3)), cosine, 0.000001F, "F32 RoPE vector value 3 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_4)) == 9.0F, "F32 RoPE trailing value mismatch");
}

void test_applies_f32_rope_to_batched_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 4}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_col2 {0, 2};
    const std::array<std::uint64_t, 2> row0_col3 {0, 3};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_col2 {1, 2};
    const std::array<std::uint64_t, 2> row1_col3 {1, 3};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col3), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col2), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col3), 0.0F);

    const auto output = rope_f32(input, 1, 4, 1.0F);
    const auto cosine = static_cast<float>(std::cos(1.0));
    const auto sine = static_cast<float>(std::sin(1.0));

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col0)), cosine, 0.000001F, "F32 RoPE batch value 0,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col1)), sine, 0.000001F, "F32 RoPE batch value 0,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col2)), -sine, 0.000001F, "F32 RoPE batch value 0,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col3)), cosine, 0.000001F, "F32 RoPE batch value 0,3 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col0)), -sine, 0.000001F, "F32 RoPE batch value 1,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col1)), cosine, 0.000001F, "F32 RoPE batch value 1,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col2)), cosine, 0.000001F, "F32 RoPE batch value 1,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col3)), sine, 0.000001F, "F32 RoPE batch value 1,3 mismatch");
}

void test_applies_f32_rope_to_batched_rows_with_position_start() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 4}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_col2 {0, 2};
    const std::array<std::uint64_t, 2> row0_col3 {0, 3};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_col2 {1, 2};
    const std::array<std::uint64_t, 2> row1_col3 {1, 3};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col3), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col2), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col3), 0.0F);

    const auto output = rope_f32_with_position_start(input, 1, 4, 1.0F);
    const auto row0_cosine = static_cast<float>(std::cos(1.0));
    const auto row0_sine = static_cast<float>(std::sin(1.0));
    const auto row1_cosine = static_cast<float>(std::cos(2.0));
    const auto row1_sine = static_cast<float>(std::sin(2.0));

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col0)), row0_cosine, 0.000001F, "F32 rowwise RoPE batch value 0,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col1)), row0_sine, 0.000001F, "F32 rowwise RoPE batch value 0,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col2)), -row0_sine, 0.000001F, "F32 rowwise RoPE batch value 0,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col3)), row0_cosine, 0.000001F, "F32 rowwise RoPE batch value 0,3 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col0)), -row1_sine, 0.000001F, "F32 rowwise RoPE batch value 1,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col1)), row1_cosine, 0.000001F, "F32 rowwise RoPE batch value 1,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col2)), row1_cosine, 0.000001F, "F32 rowwise RoPE batch value 1,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col3)), row1_sine, 0.000001F, "F32 rowwise RoPE batch value 1,3 mismatch");
}

void test_applies_f32_rope_to_batched_heads_with_position_start() {
    const std::array<float, 12> values {
        1.0F, 0.0F, 9.0F,
        0.0F, 1.0F, 8.0F,
        1.0F, 1.0F, 7.0F,
        2.0F, 0.0F, 6.0F,
    };
    const auto input = tensor_f32_matrix(2, 6, std::span<const float>(values));

    const auto output = rope_f32_by_heads_with_position_start(input, 1, 2, 3, 2, 1.0F);
    const auto row0_cosine = static_cast<float>(std::cos(1.0));
    const auto row0_sine = static_cast<float>(std::sin(1.0));
    const auto row1_cosine = static_cast<float>(std::cos(2.0));
    const auto row1_sine = static_cast<float>(std::sin(2.0));

    expect(output.descriptor() == input.descriptor(), "F32 headwise RoPE descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 headwise RoPE output should not alias input");

    const std::array<std::uint64_t, 2> row0_head0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_head0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_head0_trailing {0, 2};
    const std::array<std::uint64_t, 2> row0_head1_col0 {0, 3};
    const std::array<std::uint64_t, 2> row0_head1_col1 {0, 4};
    const std::array<std::uint64_t, 2> row0_head1_trailing {0, 5};
    const std::array<std::uint64_t, 2> row1_head0_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_head0_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_head0_trailing {1, 2};
    const std::array<std::uint64_t, 2> row1_head1_col0 {1, 3};
    const std::array<std::uint64_t, 2> row1_head1_col1 {1, 4};
    const std::array<std::uint64_t, 2> row1_head1_trailing {1, 5};
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head0_col0)), row0_cosine, 0.000001F, "F32 headwise RoPE row 0 head 0 col 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head0_col1)), row0_sine, 0.000001F, "F32 headwise RoPE row 0 head 0 col 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row0_head0_trailing)) == 9.0F, "F32 headwise RoPE row 0 head 0 trailing mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head1_col0)), -row0_sine, 0.000001F, "F32 headwise RoPE row 0 head 1 col 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head1_col1)), row0_cosine, 0.000001F, "F32 headwise RoPE row 0 head 1 col 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row0_head1_trailing)) == 8.0F, "F32 headwise RoPE row 0 head 1 trailing mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_head0_col0)),
        row1_cosine - row1_sine,
        0.000001F,
        "F32 headwise RoPE row 1 head 0 col 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_head0_col1)),
        row1_sine + row1_cosine,
        0.000001F,
        "F32 headwise RoPE row 1 head 0 col 1 mismatch"
    );
    expect(read_f32(output, std::span<const std::uint64_t>(row1_head0_trailing)) == 7.0F, "F32 headwise RoPE row 1 head 0 trailing mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_col0)), 2.0F * row1_cosine, 0.000001F, "F32 headwise RoPE row 1 head 1 col 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_col1)), 2.0F * row1_sine, 0.000001F, "F32 headwise RoPE row 1 head 1 col 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row1_head1_trailing)) == 6.0F, "F32 headwise RoPE row 1 head 1 trailing mismatch");
}

void test_applies_f32_neox_rope_by_split_halves() {
    const std::array<float, 4> values {1.0F, 2.0F, 3.0F, 4.0F};
    const auto input = tensor_f32_matrix(1, 4, std::span<const float>(values));

    const auto output = rope_neox_f32_by_heads_with_position_start(input, 1, 1, 4, 4, 1.0F);
    const auto cosine = static_cast<float>(std::cos(1.0));
    const auto sine = static_cast<float>(std::sin(1.0));

    expect_close(read_f32(output, std::array<std::uint64_t, 2> {0, 0}), 1.0F * cosine - 3.0F * sine, 0.000001F, "NeoX RoPE first split-half value mismatch");
    expect_close(read_f32(output, std::array<std::uint64_t, 2> {0, 1}), 2.0F * cosine - 4.0F * sine, 0.000001F, "NeoX RoPE second split-half value mismatch");
    expect_close(read_f32(output, std::array<std::uint64_t, 2> {0, 2}), 1.0F * sine + 3.0F * cosine, 0.000001F, "NeoX RoPE third split-half value mismatch");
    expect_close(read_f32(output, std::array<std::uint64_t, 2> {0, 3}), 2.0F * sine + 4.0F * cosine, 0.000001F, "NeoX RoPE fourth split-half value mismatch");
}

void test_applies_f32_rope_frequency_base_by_pair() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    const std::array<std::uint64_t, 1> index_3 {3};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_3), 0.0F);

    const auto output = rope_f32(input, 1, 4, 10000.0F);
    const auto first_angle = 1.0;
    const auto second_angle = 0.01;

    expect_close(read_f32(output, std::span<const std::uint64_t>(index_0)), static_cast<float>(std::cos(first_angle)), 0.000001F, "F32 RoPE first pair cosine mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_1)), static_cast<float>(std::sin(first_angle)), 0.000001F, "F32 RoPE first pair sine mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_2)), static_cast<float>(std::cos(second_angle)), 0.000001F, "F32 RoPE second pair cosine mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_3)), static_cast<float>(std::sin(second_angle)), 0.000001F, "F32 RoPE second pair sine mismatch");
}

void test_rejects_f32_rope_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f16));

    try {
        (void)rope_f32(input, 1, 4, 10000.0F);
        throw TestFailure("F32 RoPE should reject F16 input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 RoPE dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 RoPE expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 RoPE actual dtype mismatch");
    }
}

void test_rejects_f32_rope_input_rank_above_two() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2, 2}), DType::f32));

    try {
        (void)rope_f32(input, 1, 2, 10000.0F);
        throw TestFailure("F32 RoPE should reject rank-three input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 RoPE rank error code mismatch");
        expect(error.expected_value() == 2, "F32 RoPE expected rank mismatch");
        expect(error.actual_value() == 3, "F32 RoPE actual rank mismatch");
    }
}

void test_rejects_f32_rope_invalid_dimension_count() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));

    try {
        (void)rope_f32(input, 1, 3, 10000.0F);
        throw TestFailure("F32 RoPE should reject odd dimension count");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_dimension, "F32 RoPE odd dimension error mismatch");
        expect(error.expected_value() == 2, "F32 RoPE expected dimension multiple mismatch");
        expect(error.actual_value() == 3, "F32 RoPE actual odd dimension mismatch");
    }
}

void test_rejects_f32_rope_dimension_count_above_hidden() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));

    try {
        (void)rope_f32(input, 1, 6, 10000.0F);
        throw TestFailure("F32 RoPE should reject dimension count above hidden size");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 RoPE dimension shape error mismatch");
        expect(error.expected_value() == 4, "F32 RoPE expected hidden dimension mismatch");
        expect(error.actual_value() == 6, "F32 RoPE actual rope dimension mismatch");
    }
}

void test_rejects_f32_rope_nonpositive_frequency_base() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));

    try {
        (void)rope_f32(input, 1, 4, 0.0F);
        throw TestFailure("F32 RoPE should reject nonpositive frequency base");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RoPE frequency base must be positive",
            "F32 RoPE frequency base error mismatch"
        );
    }

    try {
        (void)rope_f32(input, 1, 4, std::numeric_limits<float>::quiet_NaN());
        throw TestFailure("F32 RoPE should reject NaN frequency base");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RoPE frequency base must be positive",
            "F32 RoPE NaN frequency base error mismatch"
        );
    }
}

void test_rejects_f32_rope_row_position_overflow() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)rope_f32_with_position_start(input, std::numeric_limits<std::uint64_t>::max(), 2, 1.0F);
        throw TestFailure("F32 rowwise RoPE should reject overflowing row positions");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RoPE row positions overflow",
            "F32 rowwise RoPE overflow error mismatch"
        );
    }
}

void test_applies_f32_softmax_to_vector() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 2.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 3.0F);

    const auto output = softmax_f32(input);
    const auto denominator = std::exp(-2.0) + std::exp(-1.0) + 1.0;

    expect(output.descriptor() == input.descriptor(), "F32 softmax vector output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 softmax vector output should not alias input");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(std::exp(-2.0) / denominator),
        0.000001F,
        "F32 softmax vector value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(std::exp(-1.0) / denominator),
        0.000001F,
        "F32 softmax vector value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_2)),
        static_cast<float>(1.0 / denominator),
        0.000001F,
        "F32 softmax vector value 2 mismatch"
    );
}

void test_applies_f32_softmax_to_batched_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_col2 {0, 2};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_col2 {1, 2};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col2), 3.0F);

    const auto output = softmax_f32(input);
    const auto row1_denominator = std::exp(-2.0) + 2.0;

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col0)), 1.0F / 3.0F, 0.000001F, "F32 softmax row 0,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col1)), 1.0F / 3.0F, 0.000001F, "F32 softmax row 0,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col2)), 1.0F / 3.0F, 0.000001F, "F32 softmax row 0,2 mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        static_cast<float>(std::exp(-2.0) / row1_denominator),
        0.000001F,
        "F32 softmax row 1,0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        static_cast<float>(1.0 / row1_denominator),
        0.000001F,
        "F32 softmax row 1,1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col2)),
        static_cast<float>(1.0 / row1_denominator),
        0.000001F,
        "F32 softmax row 1,2 mismatch"
    );
}

void test_applies_f32_softmax_to_rank3_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2, 3}), DType::f32));
    const std::array<std::uint64_t, 3> row0_head0_col0 {0, 0, 0};
    const std::array<std::uint64_t, 3> row0_head0_col1 {0, 0, 1};
    const std::array<std::uint64_t, 3> row0_head0_col2 {0, 0, 2};
    const std::array<std::uint64_t, 3> row1_head1_col0 {1, 1, 0};
    const std::array<std::uint64_t, 3> row1_head1_col1 {1, 1, 1};
    const std::array<std::uint64_t, 3> row1_head1_col2 {1, 1, 2};
    write_f32(
        input,
        std::span<const std::uint64_t>(row0_head0_col0),
        -std::numeric_limits<float>::infinity()
    );
    write_f32(input, std::span<const std::uint64_t>(row0_head0_col1), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_head0_col2), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_head1_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_head1_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_head1_col2), 0.0F);

    const auto output = softmax_f32(input);
    const auto denominator = std::exp(-2.0) + 1.0;

    expect(output.descriptor() == input.descriptor(), "F32 softmax rank-3 output descriptor mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head0_col0)), 0.0F, 0.0F, "F32 softmax masked rank-3 value mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_head0_col1)),
        static_cast<float>(std::exp(-2.0) / denominator),
        0.000001F,
        "F32 softmax rank-3 value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_head0_col2)),
        static_cast<float>(1.0 / denominator),
        0.000001F,
        "F32 softmax rank-3 value 2 mismatch"
    );
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_col0)), 1.0F / 3.0F, 0.000001F, "F32 softmax rank-3 row 1 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_col1)), 1.0F / 3.0F, 0.000001F, "F32 softmax rank-3 row 1 value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_col2)), 1.0F / 3.0F, 0.000001F, "F32 softmax rank-3 row 1 value 2 mismatch");
}

void test_applies_f32_softmax_to_large_values() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1000.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 1001.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 1002.0F);

    const auto output = softmax_f32(input);
    const auto denominator = std::exp(-2.0) + std::exp(-1.0) + 1.0;

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(std::exp(-2.0) / denominator),
        0.000001F,
        "F32 softmax stable value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(std::exp(-1.0) / denominator),
        0.000001F,
        "F32 softmax stable value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_2)),
        static_cast<float>(1.0 / denominator),
        0.000001F,
        "F32 softmax stable value 2 mismatch"
    );
}

void test_rejects_f32_softmax_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f16));

    try {
        (void)softmax_f32(input);
        throw TestFailure("F32 softmax should reject F16 input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 softmax dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 softmax expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 softmax actual dtype mismatch");
    }
}

void test_rejects_f32_softmax_input_rank_above_three() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 1, 2, 2}), DType::f32));

    try {
        (void)softmax_f32(input);
        throw TestFailure("F32 softmax should reject rank-four input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 softmax rank error code mismatch");
        expect(error.expected_value() == 3, "F32 softmax expected rank mismatch");
        expect(error.actual_value() == 4, "F32 softmax actual rank mismatch");
    }
}

void test_applies_f32_scaled_dot_product_attention() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2}), DType::f32));

    const float query_values[2][2] {
        {1.0F, 0.0F},
        {0.0F, 1.0F},
    };
    const float key_values[3][2] {
        {1.0F, 0.0F},
        {0.0F, 1.0F},
        {1.0F, 1.0F},
    };
    const float value_values[3][2] {
        {10.0F, 0.0F},
        {0.0F, 20.0F},
        {10.0F, 20.0F},
    };

    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(query, std::span<const std::uint64_t>(index), query_values[row][column]);
        }
    }

    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(key, std::span<const std::uint64_t>(index), key_values[row][column]);
            write_f32(value, std::span<const std::uint64_t>(index), value_values[row][column]);
        }
    }

    const auto output = scaled_dot_product_attention_f32(query, key, value, 0.5F);
    const auto denominator = 2.0 + std::exp(-0.5);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 attention output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "F32 attention output dtype mismatch");
    expect(output.bytes().data() != query.bytes().data(), "F32 attention output should not alias query");
    expect(output.bytes().data() != key.bytes().data(), "F32 attention output should not alias key");
    expect(output.bytes().data() != value.bytes().data(), "F32 attention output should not alias value");

    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        static_cast<float>(20.0 / denominator),
        0.000001F,
        "F32 attention value 0,0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col1)),
        static_cast<float>(20.0 * (1.0 + std::exp(-0.5)) / denominator),
        0.000001F,
        "F32 attention value 0,1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        static_cast<float>(10.0 * (1.0 + std::exp(-0.5)) / denominator),
        0.000001F,
        "F32 attention value 1,0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        static_cast<float>(40.0 / denominator),
        0.000001F,
        "F32 attention value 1,1 mismatch"
    );
}

void test_applies_f32_scaled_dot_product_attention_with_distinct_value_width() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const std::array<std::uint64_t, 2> query_0_0 {0, 0};
    const std::array<std::uint64_t, 2> query_0_1 {0, 1};
    const std::array<std::uint64_t, 2> key_0_0 {0, 0};
    const std::array<std::uint64_t, 2> key_0_1 {0, 1};
    const std::array<std::uint64_t, 2> key_1_0 {1, 0};
    const std::array<std::uint64_t, 2> key_1_1 {1, 1};
    const std::array<std::uint64_t, 2> value_0_0 {0, 0};
    const std::array<std::uint64_t, 2> value_1_0 {1, 0};
    write_f32(query, std::span<const std::uint64_t>(query_0_0), 1.0F);
    write_f32(query, std::span<const std::uint64_t>(query_0_1), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(key_0_0), 1.0F);
    write_f32(key, std::span<const std::uint64_t>(key_0_1), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(key_1_0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(key_1_1), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(value_0_0), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(value_1_0), 10.0F);

    const auto output = scaled_dot_product_attention_f32(query, key, value, 2.0F);
    const auto denominator = 1.0 + std::exp(-2.0);
    const std::array<std::uint64_t, 2> output_0_0 {0, 0};

    expect(output.descriptor().shape == Shape::from({1, 1}), "F32 attention distinct value width shape mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(output_0_0)),
        static_cast<float>(10.0 * std::exp(-2.0) / denominator),
        0.000001F,
        "F32 attention explicit scale value mismatch"
    );
}

void test_applies_f32_causal_scaled_dot_product_attention() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    write_f32(query, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(query, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(row0_col0), 10.0F);
    write_f32(value, std::span<const std::uint64_t>(row1_col0), 100.0F);

    const auto output = causal_scaled_dot_product_attention_f32(query, key, value, 1.0F, 0);

    expect(output.descriptor().shape == Shape::from({2, 1}), "F32 causal attention output shape mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        10.0F,
        0.000001F,
        "F32 causal attention row 0 should not see future value"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        55.0F,
        0.000001F,
        "F32 causal attention row 1 value mismatch"
    );
}

void test_applies_f32_causal_attention_with_query_position_offset() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 1}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 1}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 1}), DType::f32));
    const std::array<std::uint64_t, 2> query_index {0, 0};
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row2_col0 {2, 0};
    write_f32(query, std::span<const std::uint64_t>(query_index), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row2_col0), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(row0_col0), 10.0F);
    write_f32(value, std::span<const std::uint64_t>(row1_col0), 20.0F);
    write_f32(value, std::span<const std::uint64_t>(row2_col0), 300.0F);

    const auto output = causal_scaled_dot_product_attention_f32(query, key, value, 1.0F, 1);

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(query_index)),
        15.0F,
        0.000001F,
        "F32 causal attention offset should mask keys after query position"
    );
}

float softcap_value(float value, float softcap) {
    return std::tanh(value / softcap) * softcap;
}

void test_applies_f32_causal_grouped_attention_scores() {
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

    const auto scores = causal_grouped_attention_scores_f32(
        query,
        key,
        0,
        2,
        1,
        2,
        2,
        0.5F,
        10.0F
    );

    expect(
        scores.descriptor() == TensorDescriptor::from(Shape::from({3, 2, 2}), DType::f32),
        "F32 causal grouped attention scores descriptor mismatch"
    );
    expect(scores.bytes().data() != query.bytes().data(), "F32 causal grouped scores should not alias query");
    expect(scores.bytes().data() != key.bytes().data(), "F32 causal grouped scores should not alias key");

    const std::array<std::uint64_t, 3> row0_head0_col0 {0, 0, 0};
    const std::array<std::uint64_t, 3> row0_head0_col1 {0, 0, 1};
    const std::array<std::uint64_t, 3> row0_head1_col0 {0, 1, 0};
    const std::array<std::uint64_t, 3> row1_head0_col0 {1, 0, 0};
    const std::array<std::uint64_t, 3> row1_head0_col1 {1, 0, 1};
    const std::array<std::uint64_t, 3> row1_head1_col0 {1, 1, 0};
    const std::array<std::uint64_t, 3> row1_head1_col1 {1, 1, 1};
    const std::array<std::uint64_t, 3> row2_head0_col0 {2, 0, 0};
    const std::array<std::uint64_t, 3> row2_head0_col1 {2, 0, 1};
    const std::array<std::uint64_t, 3> row2_head1_col0 {2, 1, 0};
    const std::array<std::uint64_t, 3> row2_head1_col1 {2, 1, 1};

    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row0_head0_col0)),
        softcap_value(0.5F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 0 head 0 visible mismatch"
    );
    expect(
        std::isinf(read_f32(scores, std::span<const std::uint64_t>(row0_head0_col1))),
        "F32 causal grouped score row 0 future mask mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row0_head1_col0)),
        0.0F,
        0.0F,
        "F32 causal grouped score row 0 head 1 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row1_head0_col0)),
        softcap_value(1.0F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 1 head 0 key 0 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row1_head0_col1)),
        0.0F,
        0.0F,
        "F32 causal grouped score row 1 head 0 key 1 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row1_head1_col0)),
        0.0F,
        0.0F,
        "F32 causal grouped score row 1 head 1 key 0 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row1_head1_col1)),
        softcap_value(1.0F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 1 head 1 key 1 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row2_head0_col0)),
        softcap_value(0.5F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 2 head 0 key 1 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row2_head0_col1)),
        softcap_value(1.0F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 2 head 0 key 2 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row2_head1_col0)),
        softcap_value(0.5F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 2 head 1 key 1 mismatch"
    );
    expect_close(
        read_f32(scores, std::span<const std::uint64_t>(row2_head1_col1)),
        softcap_value(1.5F, 10.0F),
        0.000001F,
        "F32 causal grouped score row 2 head 1 key 2 mismatch"
    );
}

void test_applies_f32_causal_grouped_attention_values() {
    auto probabilities = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2, 2}), DType::f32));
    const std::array<float, 12> probability_values {
        1.0F, 0.0F,
        1.0F, 0.0F,
        0.25F, 0.75F,
        0.5F, 0.5F,
        0.2F, 0.8F,
        1.0F, 0.0F,
    };
    std::uint64_t linear_probability = 0;
    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t head = 0; head < 2; ++head) {
            for (std::uint64_t column = 0; column < 2; ++column) {
                const std::array<std::uint64_t, 3> index {row, head, column};
                write_f32(probabilities, std::span<const std::uint64_t>(index), probability_values[linear_probability]);
                ++linear_probability;
            }
        }
    }

    const std::array<float, 6> value_values {
        10.0F, 100.0F,
        20.0F, 200.0F,
        30.0F, 300.0F,
    };
    const auto value = tensor_f32_matrix(3, 2, std::span<const float>(value_values));

    const auto output = causal_grouped_attention_values_f32(probabilities, value, 0, 2, 1, 2, 2);

    expect(
        output.descriptor() == TensorDescriptor::from(Shape::from({3, 4}), DType::f32),
        "F32 causal grouped attention values descriptor mismatch"
    );
    expect(output.bytes().data() != probabilities.bytes().data(), "F32 causal grouped values should not alias probabilities");
    expect(output.bytes().data() != value.bytes().data(), "F32 causal grouped values should not alias value");

    const std::array<std::uint64_t, 2> row0_head0_value0 {0, 0};
    const std::array<std::uint64_t, 2> row0_head0_value1 {0, 1};
    const std::array<std::uint64_t, 2> row0_head1_value0 {0, 2};
    const std::array<std::uint64_t, 2> row0_head1_value1 {0, 3};
    const std::array<std::uint64_t, 2> row1_head0_value0 {1, 0};
    const std::array<std::uint64_t, 2> row1_head0_value1 {1, 1};
    const std::array<std::uint64_t, 2> row1_head1_value0 {1, 2};
    const std::array<std::uint64_t, 2> row1_head1_value1 {1, 3};
    const std::array<std::uint64_t, 2> row2_head0_value0 {2, 0};
    const std::array<std::uint64_t, 2> row2_head0_value1 {2, 1};
    const std::array<std::uint64_t, 2> row2_head1_value0 {2, 2};
    const std::array<std::uint64_t, 2> row2_head1_value1 {2, 3};

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head0_value0)), 10.0F, 0.000001F, "F32 grouped value row 0 head 0 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head0_value1)), 100.0F, 0.000001F, "F32 grouped value row 0 head 0 value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head1_value0)), 10.0F, 0.000001F, "F32 grouped value row 0 head 1 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_head1_value1)), 100.0F, 0.000001F, "F32 grouped value row 0 head 1 value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head0_value0)), 17.5F, 0.000001F, "F32 grouped value row 1 head 0 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head0_value1)), 175.0F, 0.000001F, "F32 grouped value row 1 head 0 value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_value0)), 15.0F, 0.000001F, "F32 grouped value row 1 head 1 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_head1_value1)), 150.0F, 0.000001F, "F32 grouped value row 1 head 1 value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row2_head0_value0)), 28.0F, 0.000001F, "F32 grouped value row 2 head 0 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row2_head0_value1)), 280.0F, 0.000001F, "F32 grouped value row 2 head 0 value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row2_head1_value0)), 20.0F, 0.000001F, "F32 grouped value row 2 head 1 value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row2_head1_value1)), 200.0F, 0.000001F, "F32 grouped value row 2 head 1 value 1 mismatch");
}

void test_applies_f32_causal_attention_with_column_views() {
    const std::array<float, 4> query_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 4> key_reference_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 4> value_reference_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
    };
    const std::array<float, 12> key_cache_values {
        99.0F, 1.0F, 0.0F, -99.0F,
        88.0F, 0.0F, 1.0F, -88.0F,
        77.0F, 5.0F, 5.0F, -77.0F,
    };
    const std::array<float, 12> value_cache_values {
        -1.0F, -2.0F, 10.0F, 20.0F,
        -3.0F, -4.0F, 30.0F, 40.0F,
        -5.0F, -6.0F, 50.0F, 60.0F,
    };
    const auto query = tensor_f32_matrix(2, 2, std::span<const float>(query_values));
    const auto key_reference = tensor_f32_matrix(2, 2, std::span<const float>(key_reference_values));
    const auto value_reference = tensor_f32_matrix(2, 2, std::span<const float>(value_reference_values));
    const auto key_cache = tensor_f32_matrix(3, 4, std::span<const float>(key_cache_values));
    const auto value_cache = tensor_f32_matrix(3, 4, std::span<const float>(value_cache_values));

    const auto reference = causal_scaled_dot_product_attention_f32(
        query,
        key_reference,
        value_reference,
        1.0F,
        0
    );
    const auto output = causal_scaled_dot_product_attention_f32(
        query,
        F32MatrixColumnView {
            .storage = key_cache,
            .row_count = 2,
            .first_column = 1,
            .column_count = 2,
        },
        F32MatrixColumnView {
            .storage = value_cache,
            .row_count = 2,
            .first_column = 2,
            .column_count = 2,
        },
        1.0F,
        0
    );

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 causal attention view output shape mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(output, std::span<const std::uint64_t>(index)),
                read_f32(reference, std::span<const std::uint64_t>(index)),
                0.000001F,
                "F32 causal attention view parity mismatch"
            );
        }
    }
}

void test_applies_f32_causal_attention_with_rope_query_column_view() {
    const auto row0_sine = static_cast<float>(std::sin(1.0));
    const auto row0_cosine = static_cast<float>(std::cos(1.0));
    const auto row1_sine = static_cast<float>(std::sin(2.0));
    const auto row1_cosine = static_cast<float>(std::cos(2.0));
    const std::array<float, 8> query_source_values {
        99.0F, 1.0F, 0.0F, -99.0F,
        88.0F, 0.0F, 1.0F, -88.0F,
    };
    const std::array<float, 4> query_reference_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 6> key_reference_values {
        1.0F, 0.0F,
        row0_cosine, row0_sine,
        -row1_sine, row1_cosine,
    };
    const std::array<float, 12> key_cache_values {
        99.0F, 1.0F, 0.0F, -99.0F,
        88.0F, row0_cosine, row0_sine, -88.0F,
        77.0F, -row1_sine, row1_cosine, -77.0F,
    };
    const std::array<float, 6> value_reference_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
        50.0F, 60.0F,
    };
    const std::array<float, 12> value_cache_values {
        -1.0F, 10.0F, 20.0F, -2.0F,
        -3.0F, 30.0F, 40.0F, -4.0F,
        -5.0F, 50.0F, 60.0F, -6.0F,
    };
    const auto query_source = tensor_f32_matrix(2, 4, std::span<const float>(query_source_values));
    const auto query_reference = tensor_f32_matrix(2, 2, std::span<const float>(query_reference_values));
    const auto rotated_query = rope_f32_with_position_start(query_reference, 1, 2, 1.0F);
    const auto key_reference = tensor_f32_matrix(3, 2, std::span<const float>(key_reference_values));
    const auto key_cache = tensor_f32_matrix(3, 4, std::span<const float>(key_cache_values));
    const auto value_reference = tensor_f32_matrix(3, 2, std::span<const float>(value_reference_values));
    const auto value_cache = tensor_f32_matrix(3, 4, std::span<const float>(value_cache_values));

    const auto reference = causal_scaled_dot_product_attention_f32(
        rotated_query,
        key_reference,
        value_reference,
        1.0F,
        1
    );
    const auto output = causal_scaled_dot_product_attention_f32(
        F32RoPEMatrixColumnView {
            .columns = F32MatrixColumnView {
                .storage = query_source,
                .row_count = 2,
                .first_column = 1,
                .column_count = 2,
            },
            .rope_dimension_count = 2,
            .frequency_base = 1.0F,
        },
        F32MatrixColumnView {
            .storage = key_cache,
            .row_count = 3,
            .first_column = 1,
            .column_count = 2,
        },
        F32MatrixColumnView {
            .storage = value_cache,
            .row_count = 3,
            .first_column = 1,
            .column_count = 2,
        },
        1.0F,
        1
    );

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 causal attention RoPE query view shape mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(output, std::span<const std::uint64_t>(index)),
                read_f32(reference, std::span<const std::uint64_t>(index)),
                0.000001F,
                "F32 causal attention RoPE query view parity mismatch"
            );
        }
    }
}

void test_rejects_f32_causal_attention_column_view_row_mismatch() {
    const std::array<float, 2> query_values {1.0F, 0.0F};
    const std::array<float, 6> key_cache_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
        1.0F, 1.0F,
    };
    const std::array<float, 6> value_cache_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
        50.0F, 60.0F,
    };
    const auto query = tensor_f32_matrix(1, 2, std::span<const float>(query_values));
    const auto key_cache = tensor_f32_matrix(3, 2, std::span<const float>(key_cache_values));
    const auto value_cache = tensor_f32_matrix(3, 2, std::span<const float>(value_cache_values));

    try {
        (void)causal_scaled_dot_product_attention_f32(
            query,
            F32MatrixColumnView {
                .storage = key_cache,
                .row_count = 2,
                .first_column = 0,
                .column_count = 2,
            },
            F32MatrixColumnView {
                .storage = value_cache,
                .row_count = 1,
                .first_column = 0,
                .column_count = 2,
            },
            1.0F,
            0
        );
        throw TestFailure("F32 causal attention should reject mismatched column view rows");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 causal attention view row error mismatch");
        expect(error.expected_value() == 2, "F32 causal attention view expected row mismatch");
        expect(error.actual_value() == 1, "F32 causal attention view actual row mismatch");
    }
}

void test_rejects_f32_causal_attention_rope_query_view_invalid_dimension_count() {
    const std::array<float, 2> query_values {1.0F, 0.0F};
    const std::array<float, 2> key_values {1.0F, 0.0F};
    const std::array<float, 1> value_values {10.0F};
    const auto query = tensor_f32_matrix(1, 2, std::span<const float>(query_values));
    const auto key = tensor_f32_matrix(1, 2, std::span<const float>(key_values));
    const auto value = tensor_f32_matrix(1, 1, std::span<const float>(value_values));

    try {
        (void)causal_scaled_dot_product_attention_f32(
            F32RoPEMatrixColumnView {
                .columns = F32MatrixColumnView {
                    .storage = query,
                    .row_count = 1,
                    .first_column = 0,
                    .column_count = 2,
                },
                .rope_dimension_count = 1,
                .frequency_base = 1.0F,
            },
            F32MatrixColumnView {
                .storage = key,
                .row_count = 1,
                .first_column = 0,
                .column_count = 2,
            },
            F32MatrixColumnView {
                .storage = value,
                .row_count = 1,
                .first_column = 0,
                .column_count = 1,
            },
            1.0F,
            0
        );
        throw TestFailure("F32 causal attention should reject invalid RoPE query dimension count");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_shape_dimension,
            "F32 causal attention RoPE query dimension error mismatch"
        );
        expect(error.expected_value() == 2, "F32 causal attention expected RoPE dimension mismatch");
        expect(error.actual_value() == 1, "F32 causal attention actual RoPE dimension mismatch");
    }
}

void test_rejects_f32_causal_attention_position_window_overflow() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));

    try {
        (void)causal_scaled_dot_product_attention_f32(query, key, value, 1.0F, 1);
        throw TestFailure("F32 causal attention should reject query positions beyond key rows");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 causal attention window error mismatch");
        expect(error.expected_value() == 2, "F32 causal attention expected key rows mismatch");
        expect(error.actual_value() == 3, "F32 causal attention actual position end mismatch");
    }
}

void test_rejects_f32_attention_dtype_mismatch() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f16));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject F16 key");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 attention dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 attention expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 attention actual dtype mismatch");
    }
}

void test_rejects_f32_attention_non_matrix_rank() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject rank-one query");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 attention rank error code mismatch");
        expect(error.expected_value() == 2, "F32 attention expected rank mismatch");
        expect(error.actual_value() == 1, "F32 attention actual rank mismatch");
    }
}

void test_rejects_f32_attention_query_key_hidden_mismatch() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 3}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject query/key hidden mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 attention hidden error code mismatch");
        expect(error.expected_value() == 2, "F32 attention expected hidden mismatch");
        expect(error.actual_value() == 3, "F32 attention actual hidden mismatch");
    }
}

void test_rejects_f32_attention_key_value_row_mismatch() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject key/value row mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 attention row error code mismatch");
        expect(error.expected_value() == 2, "F32 attention expected key row mismatch");
        expect(error.actual_value() == 3, "F32 attention actual value row mismatch");
    }
}

void test_rejects_f32_attention_nonpositive_scale() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 0.0F);
        throw TestFailure("F32 attention should reject nonpositive scale");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "attention scale must be positive and finite",
            "F32 attention scale error mismatch"
        );
    }

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, std::numeric_limits<float>::quiet_NaN());
        throw TestFailure("F32 attention should reject NaN scale");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "attention scale must be positive and finite",
            "F32 attention NaN scale error mismatch"
        );
    }
}

void test_multiplies_f32_matrices() {
    const auto left_descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const auto right_descriptor = TensorDescriptor::from(Shape::from({3, 2}), DType::f32);
    auto left = TensorStorage::allocate(left_descriptor);
    auto right = TensorStorage::allocate(right_descriptor);

    const float left_values[2][3] {
        {1.0F, 2.0F, 3.0F},
        {4.0F, 5.0F, 6.0F},
    };
    const float right_values[3][2] {
        {7.0F, 8.0F},
        {9.0F, 10.0F},
        {11.0F, 12.0F},
    };

    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(left, std::span<const std::uint64_t>(index), left_values[row][column]);
        }
    }

    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(right, std::span<const std::uint64_t>(index), right_values[row][column]);
        }
    }

    const auto output = matmul_f32(left, right);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 matmul output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "F32 matmul output dtype mismatch");
    expect(output.bytes().data() != left.bytes().data(), "F32 matmul output should not alias left");
    expect(output.bytes().data() != right.bytes().data(), "F32 matmul output should not alias right");

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 58.0F, "F32 matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 64.0F, "F32 matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 139.0F, "F32 matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 154.0F, "F32 matmul value 1,1 mismatch");
}

void test_rejects_invalid_f32_matmul_inputs() {
    const auto f32_matrix_2x2 = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto f16_matrix_2x2 = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f16));
    const auto f32_vector = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto f32_matrix_2x3 = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));

    try {
        (void)matmul_f32(f32_matrix_2x2, f16_matrix_2x2);
        throw TestFailure("F32 matmul should reject F16 right operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 matmul dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 matmul expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 matmul actual dtype mismatch");
    }

    try {
        (void)matmul_f32(f32_vector, f32_matrix_2x2);
        throw TestFailure("F32 matmul should reject rank-one operands");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 matmul rank error code mismatch");
        expect(error.expected_value() == 2, "F32 matmul expected rank mismatch");
        expect(error.actual_value() == 1, "F32 matmul actual rank mismatch");
    }

    try {
        (void)matmul_f32(f32_matrix_2x3, f32_matrix_2x2);
        throw TestFailure("F32 matmul should reject mismatched inner dimensions");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 matmul shape error code mismatch");
        expect(error.expected_value() == 3, "F32 matmul expected inner dimension mismatch");
        expect(error.actual_value() == 2, "F32 matmul actual inner dimension mismatch");
    }
}

void test_multiplies_f32_by_f16_matrices_into_f32() {
    const auto left_descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const auto right_descriptor = TensorDescriptor::from(Shape::from({3, 2}), DType::f16);
    auto left = TensorStorage::allocate(left_descriptor);
    auto right = TensorStorage::allocate(right_descriptor);

    const float left_values[2][3] {
        {1.0F, 2.0F, 3.0F},
        {4.0F, 5.0F, 6.0F},
    };
    const float right_values[3][2] {
        {7.0F, 8.0F},
        {9.0F, 10.0F},
        {11.0F, 12.0F},
    };

    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(left, std::span<const std::uint64_t>(index), left_values[row][column]);
        }
    }

    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f16_from_f32(right, std::span<const std::uint64_t>(index), right_values[row][column]);
        }
    }

    const auto output = matmul_f32_f16(left, right);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32xF16 matmul output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "F32xF16 matmul output dtype mismatch");

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 58.0F, "F32xF16 matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 64.0F, "F32xF16 matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 139.0F, "F32xF16 matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 154.0F, "F32xF16 matmul value 1,1 mismatch");
}

void test_multiplies_f32_by_f16_gguf_weight_matrix() {
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

    const auto output = matmul_f32_f16_gguf_weight(left, right);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32xF16 GGUF weight matmul output shape mismatch");
    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 6.0F, "F32xF16 GGUF matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 60.0F, "F32xF16 GGUF matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 5.0F, "F32xF16 GGUF matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 50.0F, "F32xF16 GGUF matmul value 1,1 mismatch");
}

void test_rejects_f32_f16_matmul_right_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)matmul_f32_f16(left, right);
        throw TestFailure("F32xF16 matmul should reject F32 right operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32xF16 matmul dtype error code mismatch"
        );
        expect(error.expected_value() == 1, "F32xF16 matmul expected dtype mismatch");
        expect(error.actual_value() == 0, "F32xF16 matmul actual dtype mismatch");
    }
}


const TestCase tensor_core_tests[] = {
    {"default backend starts on CPU", test_default_backend_starts_on_cpu},
    {"tensor shape accepts valid shapes", test_tensor_shape_accepts_valid_shapes},
    {"rejects tensor shape rank over four", test_rejects_tensor_shape_rank_over_four},
    {"rejects tensor shape element count overflow", test_rejects_tensor_shape_element_count_overflow},
    {"calculates tensor byte sizes", test_calculates_tensor_byte_sizes},
    {"rejects unsupported tensor dtype", test_rejects_unsupported_tensor_dtype},
    {"rejects invalid quantized tensor element count", test_rejects_invalid_quantized_tensor_element_count},
    {
        "rejects invalid quantized tensor row element count",
        test_rejects_invalid_quantized_tensor_row_element_count,
    },
    {
        "creates quantized GGUF weight descriptor with output tail",
        test_creates_quantized_gguf_weight_descriptor_with_output_tail,
    },
    {"rejects tensor byte size overflow", test_rejects_tensor_byte_size_overflow},
    {"creates tensor descriptor from shape and dtype", test_creates_tensor_descriptor_from_shape_and_dtype},
    {"allocates zero initialized tensor storage", test_allocates_zero_initialized_tensor_storage},
    {"copies tensor storage from bytes", test_copies_tensor_storage_from_bytes},
    {"rejects tensor storage byte count mismatch", test_rejects_tensor_storage_byte_count_mismatch},
    {
        "rejects tensor descriptor with invalid shape dtype combination",
        test_rejects_tensor_descriptor_with_invalid_shape_dtype_combination,
    },
    {"decodes Q8_0 block values", test_decodes_q8_0_block_values},
    {"decodes Q4_K reference block values", test_decodes_q4_k_reference_block_values},
    {"dequantizes Q4_K reference row and dot", test_dequantizes_q4_k_reference_row_and_dot},
    {"decodes Q6_K reference block values", test_decodes_q6_k_reference_block_values},
    {"dequantizes Q6_K reference row and dot", test_dequantizes_q6_k_reference_row_and_dot},
    {"rejects Q4_K reference row width mismatch", test_rejects_q4_k_reference_row_width_mismatch},
    {"rejects Q6_K reference row out of range", test_rejects_q6_k_reference_row_out_of_range},
    {"views loaded GGUF Q8_0 block", test_views_loaded_gguf_q8_0_block},
    {"views loaded GGUF Q4_K block", test_views_loaded_gguf_q4_k_block},
    {"views loaded GGUF Q6_K block", test_views_loaded_gguf_q6_k_block},
    {"rejects Q4_K block element index out of range", test_rejects_q4_k_block_element_index_out_of_range},
    {"rejects Q6_K scale index out of range", test_rejects_q6_k_scale_index_out_of_range},
    {
        "rejects Q8_0 block access for unquantized dtype",
        test_rejects_q8_0_block_access_for_unquantized_dtype,
    },
    {"rejects Q8_0 block index out of range", test_rejects_q8_0_block_index_out_of_range},
    {
        "rejects Q8_0 block element index out of range",
        test_rejects_q8_0_block_element_index_out_of_range,
    },
    {"calculates row-major linear indices", test_calculates_row_major_linear_indices},
    {"calculates unquantized element byte offsets", test_calculates_unquantized_element_byte_offsets},
    {"reads and mutates unquantized element bytes", test_reads_and_mutates_unquantized_element_bytes},
    {"rejects tensor index rank mismatch", test_rejects_tensor_index_rank_mismatch},
    {"rejects tensor index out of range", test_rejects_tensor_index_out_of_range},
    {"rejects quantized tensor element indexing", test_rejects_quantized_tensor_element_indexing},
    {"reads and writes F32 scalars by index", test_reads_and_writes_f32_scalars_by_index},
    {"reads F32 scalars from loaded GGUF payload", test_reads_f32_scalars_from_loaded_gguf_payload},
    {"rejects F32 accessor dtype mismatch", test_rejects_f32_accessor_dtype_mismatch},
    {"rejects F32 accessor invalid index", test_rejects_f32_accessor_invalid_index},
    {"reads and writes F16 scalars by index", test_reads_and_writes_f16_scalars_by_index},
    {"reads F16 scalars from loaded GGUF payload", test_reads_f16_scalars_from_loaded_gguf_payload},
    {"rejects F16 accessor dtype mismatch", test_rejects_f16_accessor_dtype_mismatch},
    {"rejects F16 accessor invalid index", test_rejects_f16_accessor_invalid_index},
    {"adds F32 tensors elementwise", test_adds_f32_tensors_elementwise},
    {"multiplies F32 tensors elementwise", test_multiplies_f32_tensors_elementwise},
    {"rejects invalid F32 elementwise inputs", test_rejects_invalid_f32_elementwise_inputs},
    {"applies F32 SiLU", test_applies_f32_silu},
    {"rejects F32 SiLU dtype mismatch", test_rejects_f32_silu_dtype_mismatch},
    {"applies F32 GELU-tanh approximation", test_applies_f32_gelu_tanh_approximation},
    {"fills F32 tensor with scalar", test_fills_f32_tensor_with_scalar},
    {"rejects F32 fill dtype mismatch", test_rejects_f32_fill_dtype_mismatch},
    {"copies F32 tensor storage", test_copies_f32_tensor_storage},
    {"rejects F32 copy dtype mismatch", test_rejects_f32_copy_dtype_mismatch},
    {"reduces F32 tensor sum and sum squares", test_reduces_f32_tensor_sum_and_sum_squares},
    {"reduces loaded GGUF F32 payload", test_reduces_loaded_gguf_f32_payload},
    {"rejects F32 reduction dtype mismatch", test_rejects_f32_reduction_dtype_mismatch},
    {"calculates F32 mean square and RMS", test_calculates_f32_mean_square_and_rms},
    {"rejects F32 RMS helpers dtype mismatch", test_rejects_f32_rms_helpers_dtype_mismatch},
    {"applies F32 RMSNorm to vector", test_applies_f32_rms_norm_to_vector},
    {"applies F32 RMSNorm to batched rows", test_applies_f32_rms_norm_to_batched_rows},
    {"rejects F32 RMSNorm dtype mismatch", test_rejects_f32_rms_norm_dtype_mismatch},
    {"rejects F32 RMSNorm input rank above two", test_rejects_f32_rms_norm_input_rank_above_two},
    {"rejects F32 RMSNorm weight rank mismatch", test_rejects_f32_rms_norm_weight_rank_mismatch},
    {
        "rejects F32 RMSNorm hidden dimension mismatch",
        test_rejects_f32_rms_norm_hidden_dimension_mismatch,
    },
    {"rejects F32 RMSNorm nonpositive epsilon", test_rejects_f32_rms_norm_nonpositive_epsilon},
    {"applies F32 RoPE to vector", test_applies_f32_rope_to_vector},
    {"applies F32 RoPE to batched rows", test_applies_f32_rope_to_batched_rows},
    {
        "applies F32 RoPE to batched rows with position start",
        test_applies_f32_rope_to_batched_rows_with_position_start,
    },
    {
        "applies F32 RoPE to batched heads with position start",
        test_applies_f32_rope_to_batched_heads_with_position_start,
    },
    {"applies F32 NeoX RoPE by split halves", test_applies_f32_neox_rope_by_split_halves},
    {"applies F32 RoPE frequency base by pair", test_applies_f32_rope_frequency_base_by_pair},
    {"rejects F32 RoPE dtype mismatch", test_rejects_f32_rope_dtype_mismatch},
    {"rejects F32 RoPE input rank above two", test_rejects_f32_rope_input_rank_above_two},
    {"rejects F32 RoPE invalid dimension count", test_rejects_f32_rope_invalid_dimension_count},
    {"rejects F32 RoPE dimension count above hidden", test_rejects_f32_rope_dimension_count_above_hidden},
    {"rejects F32 RoPE nonpositive frequency base", test_rejects_f32_rope_nonpositive_frequency_base},
    {"rejects F32 RoPE row position overflow", test_rejects_f32_rope_row_position_overflow},
    {"applies F32 softmax to vector", test_applies_f32_softmax_to_vector},
    {"applies F32 softmax to batched rows", test_applies_f32_softmax_to_batched_rows},
    {"applies F32 softmax to rank-3 rows", test_applies_f32_softmax_to_rank3_rows},
    {"applies F32 softmax to large values", test_applies_f32_softmax_to_large_values},
    {"rejects F32 softmax dtype mismatch", test_rejects_f32_softmax_dtype_mismatch},
    {"rejects F32 softmax input rank above three", test_rejects_f32_softmax_input_rank_above_three},
    {"applies F32 scaled dot product attention", test_applies_f32_scaled_dot_product_attention},
    {
        "applies F32 scaled dot product attention with distinct value width",
        test_applies_f32_scaled_dot_product_attention_with_distinct_value_width,
    },
    {
        "applies F32 causal scaled dot product attention",
        test_applies_f32_causal_scaled_dot_product_attention,
    },
    {
        "applies F32 causal attention with query position offset",
        test_applies_f32_causal_attention_with_query_position_offset,
    },
    {
        "applies F32 causal grouped attention scores",
        test_applies_f32_causal_grouped_attention_scores,
    },
    {
        "applies F32 causal grouped attention values",
        test_applies_f32_causal_grouped_attention_values,
    },
    {
        "applies F32 causal attention with column views",
        test_applies_f32_causal_attention_with_column_views,
    },
    {
        "applies F32 causal attention with RoPE query column view",
        test_applies_f32_causal_attention_with_rope_query_column_view,
    },
    {
        "rejects F32 causal attention column view row mismatch",
        test_rejects_f32_causal_attention_column_view_row_mismatch,
    },
    {
        "rejects F32 causal attention RoPE query view invalid dimension count",
        test_rejects_f32_causal_attention_rope_query_view_invalid_dimension_count,
    },
    {
        "rejects F32 causal attention position window overflow",
        test_rejects_f32_causal_attention_position_window_overflow,
    },
    {"rejects F32 attention dtype mismatch", test_rejects_f32_attention_dtype_mismatch},
    {"rejects F32 attention non-matrix rank", test_rejects_f32_attention_non_matrix_rank},
    {
        "rejects F32 attention query/key hidden mismatch",
        test_rejects_f32_attention_query_key_hidden_mismatch,
    },
    {
        "rejects F32 attention key/value row mismatch",
        test_rejects_f32_attention_key_value_row_mismatch,
    },
    {"rejects F32 attention nonpositive scale", test_rejects_f32_attention_nonpositive_scale},
    {"multiplies F32 matrices", test_multiplies_f32_matrices},
    {"rejects invalid F32 matmul inputs", test_rejects_invalid_f32_matmul_inputs},
    {"multiplies F32 by F16 matrices into F32", test_multiplies_f32_by_f16_matrices_into_f32},
    {"multiplies F32 by F16 GGUF weight matrix", test_multiplies_f32_by_f16_gguf_weight_matrix},
};

const TestCase tensor_post_backend_tests[] = {
    {"rejects F32xF16 matmul right dtype mismatch", test_rejects_f32_f16_matmul_right_dtype_mismatch},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> tensor_core_test_cases() {
    return std::span<const TestCase>(tensor_core_tests);
}

std::span<const TestCase> tensor_post_backend_test_cases() {
    return std::span<const TestCase>(tensor_post_backend_tests);
}

}  // namespace local_gguf_engine::tests
