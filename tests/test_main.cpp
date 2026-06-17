#include "benchmark_tests.hpp"
#include "engine_model_tests.hpp"
#include "generate_cli_tests.hpp"
#include "gguf_inspect_tests.hpp"
#include "gguf_reader_tests.hpp"
#include "gguf_tensor_bridge_tests.hpp"
#include "llama_attention_layer_tests.hpp"
#include "llama_block_tests.hpp"
#include "llama_feed_forward_layer_tests.hpp"
#include "llama_kv_cache_tests.hpp"
#include "llama_model_tests.hpp"
#include "llama_test_support.hpp"
#include "sampling_tests.hpp"
#include "tensor_test_support.hpp"
#include "test_support.hpp"
#include "tokenizer_tests.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::bind_llama_model_weights;
using local_gguf_engine::engine::copy_llama_backend_kv_cache_to_cpu;
using local_gguf_engine::engine::copy_llama_attention_layer_weights_to_backend;
using local_gguf_engine::engine::copy_llama_block_weights_to_backend;
using local_gguf_engine::engine::copy_llama_feed_forward_layer_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_attention_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_block_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_feed_forward_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_logits_weights_to_backend;
using local_gguf_engine::engine::create_llama_kv_cache;
using local_gguf_engine::engine::create_llama_backend_kv_cache;
using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::generate_llama_argmax_tokens_f32;
using local_gguf_engine::engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::generate_llama_sampled_tokens_f32;
using local_gguf_engine::engine::generate_llama_sampled_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::LlamaAttentionLayerBackendWeights;
using local_gguf_engine::engine::LlamaAttentionLayerWeights;
using local_gguf_engine::engine::LlamaBlockBackendWeights;
using local_gguf_engine::engine::LlamaBlockWeights;
using local_gguf_engine::engine::LlamaFeedForwardLayerWeights;
using local_gguf_engine::engine::LlamaKVCache;
using local_gguf_engine::engine::LlamaModelAttentionBackendWeights;
using local_gguf_engine::engine::LlamaModelBlockBackendWeights;
using local_gguf_engine::engine::LlamaModelFeedForwardBackendWeights;
using local_gguf_engine::engine::LlamaModelLogitsBackendWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::LlamaModelWeights;
using local_gguf_engine::engine::project_llama_logits_f32;
using local_gguf_engine::engine::run_llama_attention_layer_f32;
using local_gguf_engine::engine::run_llama_attention_layer_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_block_f32;
using local_gguf_engine::engine::run_llama_block_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_feed_forward_layer_f32;
using local_gguf_engine::engine::run_llama_model_blocks_f32;
using local_gguf_engine::engine::run_llama_model_blocks_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_model_logits_f32;
using local_gguf_engine::engine::run_llama_model_next_token_argmax_f32;
using local_gguf_engine::engine::run_llama_model_next_token_argmax_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_model_next_token_sampled_f32;
using local_gguf_engine::engine::SamplingConfig;
using local_gguf_engine::engine::SamplingState;
using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::model_format::GGUFReadError;
using local_gguf_engine::model_format::GGUFReadErrorCode;
using local_gguf_engine::model_format::load_gguf_tensor_payload;
using local_gguf_engine::model_format::read_gguf_model_view;
using local_gguf_engine::tensor::byte_size;
using local_gguf_engine::tensor::copy_tensor_from_backend;
using local_gguf_engine::tensor::copy_tensor_to_backend;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::element_byte_offset;
using local_gguf_engine::tensor::element_bytes;
using local_gguf_engine::tensor::element_count;
using local_gguf_engine::tensor::linear_index;
using local_gguf_engine::tensor::mutable_element_bytes;
using local_gguf_engine::tensor::dequantize_q8_0_block;
using local_gguf_engine::tensor::q8_0_block_count;
using local_gguf_engine::tensor::q8_0_block_view;
using local_gguf_engine::tensor::Q8_0BlockView;
using local_gguf_engine::tensor::read_f16_as_f32;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::tensor_backend_unavailable_reason;
using local_gguf_engine::tensor::write_f16_from_f32;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::causal_scaled_dot_product_attention_f32;
using local_gguf_engine::tensor::ops::copy_f32;
using local_gguf_engine::tensor::ops::F32MatrixColumnView;
using local_gguf_engine::tensor::ops::F32RoPEMatrixColumnView;
using local_gguf_engine::tensor::ops::fill_f32;
using local_gguf_engine::tensor::ops::gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::llama_cached_attention_f32_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32;
using local_gguf_engine::tensor::ops::matmul_f32_f16;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_triple_to_backend;
using local_gguf_engine::tensor::ops::mean_square_f32;
using local_gguf_engine::tensor::ops::multiply_f32;
using local_gguf_engine::tensor::ops::residual_gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rms_f32;
using local_gguf_engine::tensor::ops::rope_f32;
using local_gguf_engine::tensor::ops::rope_f32_with_position_start;
using local_gguf_engine::tensor::ops::scaled_dot_product_attention_f32;
using local_gguf_engine::tensor::ops::silu_f32;
using local_gguf_engine::tensor::ops::silu_multiply_f32_to_backend;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::sum_f32;
using local_gguf_engine::tensor::ops::sum_squares_f32;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::add_tiny_llama_block_tensors;
using local_gguf_engine::tests::add_tiny_llama_top_level_tensors;
using local_gguf_engine::tests::engine_tensor;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::fixture;
using local_gguf_engine::tests::llama_block_tensor_name;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f16_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::tiny_bound_llama_model;
using local_gguf_engine::tests::tiny_feed_forward_layer_config;
using local_gguf_engine::tests::tiny_logits_config;

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

void expect_llama_kv_cache_close(
    const LlamaKVCache& actual,
    const LlamaKVCache& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.layers.size() == expected.layers.size(), std::string(message) + " layer count mismatch");
    expect(actual.context_length == expected.context_length, std::string(message) + " context length mismatch");
    expect(actual.key_width == expected.key_width, std::string(message) + " key width mismatch");
    expect(actual.value_width == expected.value_width, std::string(message) + " value width mismatch");

    for (std::size_t layer_index = 0; layer_index < actual.layers.size(); ++layer_index) {
        const auto prefix = std::string(message) + " layer " + std::to_string(layer_index);
        expect(
            actual.layers[layer_index].filled_token_count == expected.layers[layer_index].filled_token_count,
            prefix + " filled token count mismatch"
        );
        expect_f32_matrix_close(
            actual.layers[layer_index].key,
            expected.layers[layer_index].key,
            tolerance,
            prefix + " key cache"
        );
        expect_f32_matrix_close(
            actual.layers[layer_index].value,
            expected.layers[layer_index].value,
            tolerance,
            prefix + " value cache"
        );
    }
}

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
        expect(error.actual_value() == 30, "actual Q8_0 descriptor element count mismatch");
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

void test_rejects_f32_add_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)add_f32(left, right);
        throw TestFailure("F32 add should reject F16 operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 operator dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 operator dtype id mismatch");
    }
}

void test_rejects_f32_add_shape_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)add_f32(left, right);
        throw TestFailure("F32 add should reject mismatched shapes");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "error code should be tensor_shape_mismatch");
        expect(error.expected_value() == 3, "expected shape dimension mismatch");
        expect(error.actual_value() == 2, "actual shape dimension mismatch");
    }
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

void test_rejects_f32_multiply_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)multiply_f32(left, right);
        throw TestFailure("F32 multiply should reject F16 operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 multiply dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 multiply expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 multiply actual dtype mismatch");
    }
}

void test_rejects_f32_multiply_shape_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)multiply_f32(left, right);
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

void test_rejects_f32_softmax_input_rank_above_two() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2, 2}), DType::f32));

    try {
        (void)softmax_f32(input);
        throw TestFailure("F32 softmax should reject rank-three input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 softmax rank error code mismatch");
        expect(error.expected_value() == 2, "F32 softmax expected rank mismatch");
        expect(error.actual_value() == 3, "F32 softmax actual rank mismatch");
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

void test_rejects_f32_matmul_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f16));

    try {
        (void)matmul_f32(left, right);
        throw TestFailure("F32 matmul should reject F16 right operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 matmul dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 matmul expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 matmul actual dtype mismatch");
    }
}

void test_rejects_f32_matmul_non_matrix_rank() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)matmul_f32(left, right);
        throw TestFailure("F32 matmul should reject rank-one operands");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 matmul rank error code mismatch");
        expect(error.expected_value() == 2, "F32 matmul expected rank mismatch");
        expect(error.actual_value() == 1, "F32 matmul actual rank mismatch");
    }
}

void test_rejects_f32_matmul_inner_dimension_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)matmul_f32(left, right);
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

void test_runs_llama_cached_attention_to_cpu_backend_buffer() {
    const std::array<float, 2> query_values {1.0F, 0.0F};
    const std::array<float, 2> key_values {0.5F, -1.0F};
    const std::array<float, 2> value_values {2.0F, -3.0F};
    const auto query = tensor_f32_matrix(1, 2, std::span<const float>(query_values));
    const auto key = tensor_f32_matrix(1, 2, std::span<const float>(key_values));
    const auto value = tensor_f32_matrix(1, 2, std::span<const float>(value_values));
    const auto cache_descriptor = TensorDescriptor::from(Shape::from({4, 2}), DType::f32);
    auto cache_key_buffer = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);
    auto cache_value_buffer = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);

    const auto output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query, BackendKind::cpu),
        copy_tensor_to_backend(key, BackendKind::cpu),
        copy_tensor_to_backend(value, BackendKind::cpu),
        cache_key_buffer,
        cache_value_buffer,
        0,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto cache_key = copy_tensor_from_backend(cache_key_buffer);
    const auto cache_value = copy_tensor_from_backend(cache_value_buffer);
    auto expected_cache_key = TensorStorage::allocate(cache_descriptor);
    auto expected_cache_value = TensorStorage::allocate(cache_descriptor);
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    write_f32(expected_cache_key, std::span<const std::uint64_t>(row0_col0), 0.5F);
    write_f32(expected_cache_key, std::span<const std::uint64_t>(row0_col1), -1.0F);
    write_f32(expected_cache_value, std::span<const std::uint64_t>(row0_col0), 2.0F);
    write_f32(expected_cache_value, std::span<const std::uint64_t>(row0_col1), -3.0F);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU Llama cached attention output backend mismatch");
    expect_f32_matrix_close(output, value, 1.0e-6F, "CPU backend Llama cached attention output");
    expect_f32_matrix_close(cache_key, expected_cache_key, 1.0e-6F, "CPU backend Llama cached attention key cache");
    expect_f32_matrix_close(cache_value, expected_cache_value, 1.0e-6F, "CPU backend Llama cached attention value cache");
}

void test_matches_llama_cached_attention_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 2> query0_values {0.5F, -0.25F};
    const std::array<float, 2> key0_values {1.0F, 0.5F};
    const std::array<float, 2> value0_values {2.0F, -1.0F};
    const std::array<float, 2> query1_values {0.25F, 0.75F};
    const std::array<float, 2> key1_values {-0.5F, 1.25F};
    const std::array<float, 2> value1_values {0.5F, 3.0F};
    const auto query0 = tensor_f32_matrix(1, 2, std::span<const float>(query0_values));
    const auto key0 = tensor_f32_matrix(1, 2, std::span<const float>(key0_values));
    const auto value0 = tensor_f32_matrix(1, 2, std::span<const float>(value0_values));
    const auto query1 = tensor_f32_matrix(1, 2, std::span<const float>(query1_values));
    const auto key1 = tensor_f32_matrix(1, 2, std::span<const float>(key1_values));
    const auto value1 = tensor_f32_matrix(1, 2, std::span<const float>(value1_values));
    const auto cache_descriptor = TensorDescriptor::from(Shape::from({4, 2}), DType::f32);
    auto cpu_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);
    auto cpu_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);
    auto metal_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::metal);
    auto metal_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::metal);

    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query0, BackendKind::cpu),
        copy_tensor_to_backend(key0, BackendKind::cpu),
        copy_tensor_to_backend(value0, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        0,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    const auto cpu_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query1, BackendKind::cpu),
        copy_tensor_to_backend(key1, BackendKind::cpu),
        copy_tensor_to_backend(value1, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        1,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query0, BackendKind::metal),
        copy_tensor_to_backend(key0, BackendKind::metal),
        copy_tensor_to_backend(value0, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        0,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    const auto metal_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query1, BackendKind::metal),
        copy_tensor_to_backend(key1, BackendKind::metal),
        copy_tensor_to_backend(value1, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        1,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal Llama cached attention output backend mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-4F,
        "Metal backend Llama cached attention output"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_key),
        copy_tensor_from_backend(cpu_cache_key),
        1.0e-4F,
        "Metal backend Llama cached attention key cache"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_value),
        copy_tensor_from_backend(cpu_cache_value),
        1.0e-4F,
        "Metal backend Llama cached attention value cache"
    );
}

void test_matches_llama_one_token_grouped_cached_attention_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t prefill_rows = 3;
    constexpr std::uint64_t decode_rows = 1;
    constexpr std::uint64_t head_count = 4;
    constexpr std::uint64_t kv_head_count = 2;
    constexpr std::uint64_t key_length = 4;
    constexpr std::uint64_t value_length = 3;
    constexpr std::uint64_t cache_capacity = 8;
    constexpr std::uint64_t query_width = head_count * key_length;
    constexpr std::uint64_t key_width = kv_head_count * key_length;
    constexpr std::uint64_t value_width = kv_head_count * value_length;
    constexpr std::uint64_t rope_dimension_count = key_length;
    constexpr float frequency_base = 10000.0F;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(key_length));
    auto make_values = [](std::size_t count, float scale, float bias) {
        std::vector<float> values;
        values.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto centered = static_cast<float>(static_cast<int>((index * 7U) % 17U) - 8);
            values.push_back(bias + (centered * scale) + (0.0003F * static_cast<float>(index)));
        }
        return values;
    };

    const auto prefill_query_values = make_values(prefill_rows * query_width, 0.013F, 0.002F);
    const auto prefill_key_values = make_values(prefill_rows * key_width, 0.011F, -0.003F);
    const auto prefill_value_values = make_values(prefill_rows * value_width, 0.017F, 0.005F);
    const auto decode_query_values = make_values(decode_rows * query_width, 0.019F, -0.004F);
    const auto decode_key_values = make_values(decode_rows * key_width, 0.007F, 0.006F);
    const auto decode_value_values = make_values(decode_rows * value_width, 0.023F, -0.001F);
    const auto prefill_query = tensor_f32_matrix(
        prefill_rows,
        query_width,
        std::span<const float>(prefill_query_values.data(), prefill_query_values.size())
    );
    const auto prefill_key = tensor_f32_matrix(
        prefill_rows,
        key_width,
        std::span<const float>(prefill_key_values.data(), prefill_key_values.size())
    );
    const auto prefill_value = tensor_f32_matrix(
        prefill_rows,
        value_width,
        std::span<const float>(prefill_value_values.data(), prefill_value_values.size())
    );
    const auto decode_query = tensor_f32_matrix(
        decode_rows,
        query_width,
        std::span<const float>(decode_query_values.data(), decode_query_values.size())
    );
    const auto decode_key = tensor_f32_matrix(
        decode_rows,
        key_width,
        std::span<const float>(decode_key_values.data(), decode_key_values.size())
    );
    const auto decode_value = tensor_f32_matrix(
        decode_rows,
        value_width,
        std::span<const float>(decode_value_values.data(), decode_value_values.size())
    );
    const auto cache_key_descriptor = TensorDescriptor::from(Shape::from({cache_capacity, key_width}), DType::f32);
    const auto cache_value_descriptor = TensorDescriptor::from(Shape::from({cache_capacity, value_width}), DType::f32);
    auto cpu_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_key_descriptor), BackendKind::cpu);
    auto cpu_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_value_descriptor), BackendKind::cpu);
    auto metal_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_key_descriptor), BackendKind::metal);
    auto metal_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_value_descriptor), BackendKind::metal);

    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(prefill_query, BackendKind::cpu),
        copy_tensor_to_backend(prefill_key, BackendKind::cpu),
        copy_tensor_to_backend(prefill_value, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        0,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );
    const auto cpu_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(decode_query, BackendKind::cpu),
        copy_tensor_to_backend(decode_key, BackendKind::cpu),
        copy_tensor_to_backend(decode_value, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        prefill_rows,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );

    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(prefill_query, BackendKind::metal),
        copy_tensor_to_backend(prefill_key, BackendKind::metal),
        copy_tensor_to_backend(prefill_value, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        0,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );
    const auto metal_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(decode_query, BackendKind::metal),
        copy_tensor_to_backend(decode_key, BackendKind::metal),
        copy_tensor_to_backend(decode_value, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        prefill_rows,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );

    expect(
        metal_output_buffer.backend() == BackendKind::metal,
        "Metal grouped one-token Llama cached attention output backend mismatch"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-4F,
        "Metal grouped one-token Llama cached attention output"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_key),
        copy_tensor_from_backend(cpu_cache_key),
        1.0e-4F,
        "Metal grouped one-token Llama cached attention key cache"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_value),
        copy_tensor_from_backend(cpu_cache_value),
        1.0e-4F,
        "Metal grouped one-token Llama cached attention value cache"
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

LlamaModelConfig backend_feed_forward_fixture_config() {
    return LlamaModelConfig {
        .context_length = 8,
        .embedding_length = 4,
        .block_count = 1,
        .feed_forward_length = 6,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .attention_key_length = 4,
        .attention_value_length = 4,
        .rope_dimension_count = 4,
        .vocab_size = 5,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 0.25F,
    };
}

void add_backend_feed_forward_fixture_tensors(std::vector<EngineTensor>& tensors) {
    const std::array<float, 20> token_embedding_values {
        0.5F, -1.0F, 1.5F, 0.25F, -0.75F,
        1.0F, 0.5F, -0.25F, 1.25F, -1.5F,
        -0.5F, 1.75F, 0.75F, -1.25F, 0.0F,
        1.5F, -0.75F, 0.25F, 0.5F, 1.0F,
    };
    const std::array<float, 20> output_values {
        0.25F, -0.5F, 1.0F, 0.75F, -1.25F,
        1.5F, 0.375F, -0.875F, 0.125F, 0.5F,
        -0.75F, 1.25F, 0.625F, -0.25F, 1.0F,
        0.875F, -1.5F, 0.5F, 1.125F, -0.375F,
    };
    const std::array<float, 4> attention_norm_values {1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 4> feed_forward_norm_values {1.0F, 0.75F, 1.25F, 0.5F};
    const std::array<float, 16> query_values {
        0.5F, -0.25F, 1.0F, 0.75F,
        -0.5F, 0.875F, 0.25F, -1.25F,
        1.5F, 0.0F, -0.75F, 0.5F,
        0.125F, 1.25F, -1.0F, 0.375F,
    };
    const std::array<float, 16> key_values {
        -0.75F, 0.5F, 0.25F, 1.25F,
        1.0F, -1.5F, 0.625F, -0.125F,
        0.375F, 0.875F, -0.5F, 0.75F,
        -1.25F, 0.25F, 1.5F, -0.375F,
    };
    const std::array<float, 16> value_values {
        1.25F, -0.5F, 0.875F, 0.0F,
        -0.25F, 1.5F, -1.0F, 0.625F,
        0.75F, 0.125F, 0.5F, -1.25F,
        -1.5F, 0.375F, 1.0F, 0.25F,
    };
    const std::array<float, 16> attention_output_values {
        0.625F, -0.875F, 0.25F, 1.0F,
        1.25F, 0.5F, -1.5F, 0.125F,
        -0.375F, 1.125F, 0.75F, -0.5F,
        0.0F, -0.25F, 1.5F, 0.875F,
    };
    const std::array<float, 24> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.5F,
        -1.0F, 0.125F, 0.375F, -0.625F, 0.875F, 0.25F,
        0.5F, 1.25F, -0.75F, 0.0F, 0.625F, -0.375F,
        1.5F, -1.25F, 0.25F, 0.5F, -0.875F, 0.125F,
    };
    const std::array<float, 24> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F, 0.375F, 1.25F,
        0.75F, -0.25F, 1.5F, 0.125F, -0.5F, 0.875F,
        1.0F, 0.25F, -1.25F, 0.5F, 0.75F, -0.625F,
        -0.75F, 1.125F, 0.0F, -0.375F, 1.375F, 0.25F,
    };
    const std::array<float, 24> down_values {
        0.5F, -0.25F, 0.125F, 0.75F,
        -0.5F, 1.0F, 0.625F, -0.125F,
        0.25F, 0.375F, -0.75F, 1.25F,
        1.5F, -1.0F, 0.5F, 0.0F,
        -0.875F, 0.25F, 1.125F, -0.5F,
        0.625F, 0.75F, -1.25F, 0.375F,
    };

    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(4, 5, std::span<const float>(token_embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(attention_norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(4, 5, std::span<const float>(output_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(attention_norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_q.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(query_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_k.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(key_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_v.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(value_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_output.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(attention_output_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_norm.weight"),
        tensor_f32_vector(std::span<const float>(feed_forward_norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_gate.weight"),
        tensor_f16_gguf_matrix(4, 6, std::span<const float>(gate_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_up.weight"),
        tensor_f16_gguf_matrix(4, 6, std::span<const float>(up_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_down.weight"),
        tensor_f16_gguf_matrix(6, 4, std::span<const float>(down_values))
    ));
}

EngineModel backend_feed_forward_fixture_model() {
    std::vector<EngineTensor> tensors;
    add_backend_feed_forward_fixture_tensors(tensors);
    return tiny_bound_llama_model(backend_feed_forward_fixture_config(), std::move(tensors));
}

TensorStorage backend_feed_forward_fixture_input() {
    const std::array<float, 8> input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    return tensor_f32_matrix(2, 4, std::span<const float>(input_values));
}

EngineModel backend_cached_model_blocks_fixture_model() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 2;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    add_tiny_llama_block_tensors(tensors, 1, 0.5F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

TensorStorage backend_cached_model_blocks_first_input() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    return tensor_f32_matrix(1, 2, std::span<const float>(input_values));
}

TensorStorage backend_cached_model_blocks_second_input() {
    const std::array<float, 2> input_values {30.0F, 40.0F};
    return tensor_f32_matrix(1, 2, std::span<const float>(input_values));
}

void expect_model_attention_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelAttentionBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model attention backend weight block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        const auto& expected = weights.blocks[block_index].attention;
        const auto& actual = backend_weights.blocks[block_index];
        expect(actual.attention_norm_weight_buffer.backend() == backend, "Llama model attention norm backend mismatch");
        expect(actual.query_weight.backend() == backend, "Llama model attention query backend mismatch");
        expect(actual.key_weight.backend() == backend, "Llama model attention key backend mismatch");
        expect(actual.value_weight.backend() == backend, "Llama model attention value backend mismatch");
        expect(actual.output_weight.backend() == backend, "Llama model attention output backend mismatch");
        expect(
            actual.attention_norm_weight_buffer.descriptor() == expected.attention_norm_weight.descriptor(),
            "Llama model attention norm backend descriptor mismatch"
        );
        expect(
            actual.query_weight.descriptor() == expected.query_weight.descriptor(),
            "Llama model attention query backend descriptor mismatch"
        );
        expect(
            actual.key_weight.descriptor() == expected.key_weight.descriptor(),
            "Llama model attention key backend descriptor mismatch"
        );
        expect(
            actual.value_weight.descriptor() == expected.value_weight.descriptor(),
            "Llama model attention value backend descriptor mismatch"
        );
        expect(
            actual.output_weight.descriptor() == expected.output_weight.descriptor(),
            "Llama model attention output backend descriptor mismatch"
        );
    }
}

void expect_model_feed_forward_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelFeedForwardBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model FFN backend weight block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        const auto& expected = weights.blocks[block_index].feed_forward;
        const auto& actual = backend_weights.blocks[block_index];
        expect(actual.feed_forward_norm_weight_buffer.backend() == backend, "Llama model FFN norm backend mismatch");
        expect(actual.gate_weight.backend() == backend, "Llama model FFN gate backend mismatch");
        expect(actual.up_weight.backend() == backend, "Llama model FFN up backend mismatch");
        expect(actual.down_weight.backend() == backend, "Llama model FFN down backend mismatch");
        expect(
            actual.feed_forward_norm_weight_buffer.descriptor() == expected.feed_forward_norm_weight.descriptor(),
            "Llama model FFN norm backend descriptor mismatch"
        );
        expect(
            actual.gate_weight.descriptor() == expected.gate_weight.descriptor(),
            "Llama model FFN gate backend descriptor mismatch"
        );
        expect(
            actual.up_weight.descriptor() == expected.up_weight.descriptor(),
            "Llama model FFN up backend descriptor mismatch"
        );
        expect(
            actual.down_weight.descriptor() == expected.down_weight.descriptor(),
            "Llama model FFN down backend descriptor mismatch"
        );
    }
}

void expect_block_backend_weights(
    const LlamaBlockWeights& expected,
    const LlamaBlockBackendWeights& actual,
    BackendKind backend
) {
    expect(actual.attention.attention_norm_weight_buffer.backend() == backend, "Llama block attention norm backend mismatch");
    expect(actual.attention.query_weight.backend() == backend, "Llama block attention query backend mismatch");
    expect(actual.attention.key_weight.backend() == backend, "Llama block attention key backend mismatch");
    expect(actual.attention.value_weight.backend() == backend, "Llama block attention value backend mismatch");
    expect(actual.attention.output_weight.backend() == backend, "Llama block attention output backend mismatch");
    expect(actual.feed_forward.feed_forward_norm_weight_buffer.backend() == backend, "Llama block FFN norm backend mismatch");
    expect(actual.feed_forward.gate_weight.backend() == backend, "Llama block FFN gate backend mismatch");
    expect(actual.feed_forward.up_weight.backend() == backend, "Llama block FFN up backend mismatch");
    expect(actual.feed_forward.down_weight.backend() == backend, "Llama block FFN down backend mismatch");
    expect(
        actual.attention.attention_norm_weight_buffer.descriptor()
            == expected.attention.attention_norm_weight.descriptor(),
        "Llama block attention norm backend descriptor mismatch"
    );
    expect(
        actual.attention.query_weight.descriptor() == expected.attention.query_weight.descriptor(),
        "Llama block attention query backend descriptor mismatch"
    );
    expect(
        actual.attention.key_weight.descriptor() == expected.attention.key_weight.descriptor(),
        "Llama block attention key backend descriptor mismatch"
    );
    expect(
        actual.attention.value_weight.descriptor() == expected.attention.value_weight.descriptor(),
        "Llama block attention value backend descriptor mismatch"
    );
    expect(
        actual.attention.output_weight.descriptor() == expected.attention.output_weight.descriptor(),
        "Llama block attention output backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.feed_forward_norm_weight_buffer.descriptor()
            == expected.feed_forward.feed_forward_norm_weight.descriptor(),
        "Llama block FFN norm backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.gate_weight.descriptor() == expected.feed_forward.gate_weight.descriptor(),
        "Llama block FFN gate backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.up_weight.descriptor() == expected.feed_forward.up_weight.descriptor(),
        "Llama block FFN up backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.down_weight.descriptor() == expected.feed_forward.down_weight.descriptor(),
        "Llama block FFN down backend descriptor mismatch"
    );
}

void expect_model_block_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelBlockBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model block backend weight block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        expect_block_backend_weights(weights.blocks[block_index], backend_weights.blocks[block_index], backend);
    }
}

void expect_model_logits_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelLogitsBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        &backend_weights.token_embedding_weight == &weights.token_embedding_weight,
        "Llama model logits backend should borrow token embedding weight"
    );
    expect(
        &backend_weights.output_norm_weight == &weights.output_norm_weight,
        "Llama model logits backend should borrow output norm weight"
    );
    expect(backend_weights.output_weight.backend() == backend, "Llama model logits output backend mismatch");
    expect(
        backend_weights.output_weight.descriptor() == weights.output_weight.descriptor(),
        "Llama model logits output backend descriptor mismatch"
    );
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model logits backend block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        expect_block_backend_weights(weights.blocks[block_index], backend_weights.blocks[block_index], backend);
    }
}

void test_runs_llama_attention_layer_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_attention_layer_weights_to_backend(
        weights.blocks[0].attention,
        BackendKind::cpu
    );
    const auto input = backend_feed_forward_fixture_input();

    const auto expected = run_llama_attention_layer_f32(input, weights.blocks[0].attention, model.llama_config(), 1);
    const auto output = run_llama_attention_layer_f32(input, backend_weights, model.llama_config(), 1);

    expect(backend_weights.query_weight.backend() == BackendKind::cpu, "Llama attention CPU query buffer backend mismatch");
    expect(backend_weights.key_weight.backend() == BackendKind::cpu, "Llama attention CPU key buffer backend mismatch");
    expect(backend_weights.value_weight.backend() == BackendKind::cpu, "Llama attention CPU value buffer backend mismatch");
    expect(backend_weights.output_weight.backend() == BackendKind::cpu, "Llama attention CPU output buffer backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "Llama attention CPU backend weight buffer output");
}

void test_runs_llama_attention_layer_with_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_attention_layer_weights_to_backend(
        weights.blocks[0].attention,
        BackendKind::cpu
    );
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        weights.blocks[0].attention,
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        weights.blocks[0].attention,
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect(backend_cache.layers[0].filled_token_count == 2, "Llama CPU backend cached attention fill mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-5F, "Llama CPU backend cached attention output");
    expect_f32_matrix_close(
        backend_cache.layers[0].key,
        expected_cache.layers[0].key,
        1.0e-5F,
        "Llama CPU backend cached attention key cache"
    );
    expect_f32_matrix_close(
        backend_cache.layers[0].value,
        expected_cache.layers[0].value,
        1.0e-5F,
        "Llama CPU backend cached attention value cache"
    );
}

void test_runs_llama_feed_forward_layer_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_feed_forward_layer_weights_to_backend(
        weights.blocks[0].feed_forward,
        BackendKind::cpu
    );
    const auto input = backend_feed_forward_fixture_input();

    const auto expected = run_llama_feed_forward_layer_f32(
        input,
        weights.blocks[0].feed_forward,
        model.llama_config()
    );
    const auto output = run_llama_feed_forward_layer_f32(input, backend_weights, model.llama_config());

    expect(backend_weights.gate_weight.backend() == BackendKind::cpu, "Llama FFN CPU gate buffer backend mismatch");
    expect(backend_weights.up_weight.backend() == BackendKind::cpu, "Llama FFN CPU up buffer backend mismatch");
    expect(backend_weights.down_weight.backend() == BackendKind::cpu, "Llama FFN CPU down buffer backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "Llama FFN CPU backend weight buffer output");
}

void test_runs_llama_block_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();

    const auto expected = run_llama_block_f32(input, weights.blocks[0], model.llama_config(), 1);
    const auto output = run_llama_block_f32(input, backend_weights, model.llama_config(), 1);

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama block CPU backend weight buffer output");
}

void test_runs_llama_block_with_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::cpu);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama CPU backend cached block output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 1.0e-5F, "Llama CPU backend cached block cache");
}

void test_matches_llama_block_with_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::metal);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::metal);
    expect_f32_matrix_close(output, expected, 2.0e-2F, "Llama Metal backend cached block output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 1.0e-2F, "Llama Metal backend cached block cache");
}

void test_runs_llama_block_with_backend_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::cpu);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::cpu);

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::cpu);
    expect(backend_cache.backend == BackendKind::cpu, "Llama block backend KV cache CPU backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama CPU backend resident-KV cached block output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        1.0e-5F,
        "Llama CPU backend resident-KV cached block cache"
    );
}

void test_matches_llama_block_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::metal);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::metal);

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::metal);
    expect(backend_cache.backend == BackendKind::metal, "Llama block backend KV cache Metal backend mismatch");
    expect_f32_matrix_close(output, expected, 5.0e-2F, "Llama Metal backend resident-KV cached block output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        2.0e-2F,
        "Llama Metal backend resident-KV cached block cache"
    );
}

void test_copies_llama_model_attention_weights_to_cpu_backend_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_attention_weights_to_backend(weights, BackendKind::cpu);

    expect_model_attention_backend_weights(weights, backend_weights, BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_attention_layer_f32(
        input,
        weights.blocks[0].attention,
        model.llama_config(),
        1
    );
    const auto output = run_llama_attention_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config(),
        1
    );
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU attention backend weight output");
}

void test_matches_llama_attention_layer_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_attention_weights_to_backend(weights, BackendKind::metal);

    expect_model_attention_backend_weights(weights, backend_weights, BackendKind::metal);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_attention_layer_f32(
        input,
        weights.blocks[0].attention,
        model.llama_config(),
        1
    );
    const auto output = run_llama_attention_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config(),
        1
    );
    expect_f32_matrix_close(output, expected, 2.0e-3F, "Llama model Metal attention backend weight output");
}

void test_matches_llama_attention_layer_with_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_attention_layer_weights_to_backend(
        weights.blocks[0].attention,
        BackendKind::metal
    );
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        weights.blocks[0].attention,
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        weights.blocks[0].attention,
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect(backend_cache.layers[0].filled_token_count == 2, "Llama Metal backend cached attention fill mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-2F, "Llama Metal backend cached attention output");
    expect_f32_matrix_close(
        backend_cache.layers[0].key,
        expected_cache.layers[0].key,
        1.0e-2F,
        "Llama Metal backend cached attention key cache"
    );
    expect_f32_matrix_close(
        backend_cache.layers[0].value,
        expected_cache.layers[0].value,
        1.0e-2F,
        "Llama Metal backend cached attention value cache"
    );
}

void test_reports_unavailable_llama_model_attention_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_attention_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal attention backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal attention backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal attention backend weight copy backend mismatch");
    }
}

void test_copies_llama_model_feed_forward_weights_to_cpu_backend_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_feed_forward_weights_to_backend(weights, BackendKind::cpu);

    expect_model_feed_forward_backend_weights(weights, backend_weights, BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_feed_forward_layer_f32(
        input,
        weights.blocks[0].feed_forward,
        model.llama_config()
    );
    const auto output = run_llama_feed_forward_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config()
    );
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU FFN backend weight output");
}

void test_matches_llama_feed_forward_layer_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_feed_forward_weights_to_backend(weights, BackendKind::metal);

    expect_model_feed_forward_backend_weights(weights, backend_weights, BackendKind::metal);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_feed_forward_layer_f32(
        input,
        weights.blocks[0].feed_forward,
        model.llama_config()
    );
    const auto output = run_llama_feed_forward_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config()
    );
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Llama model Metal FFN backend weight output");
}

void test_copies_llama_model_block_weights_to_cpu_backend_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::cpu);

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_model_blocks_f32(input, weights, model.llama_config(), 1);
    const auto output = run_llama_model_blocks_f32(input, backend_weights, model.llama_config(), 1);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU block backend weight output");
}

void test_matches_llama_model_blocks_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::metal);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_model_blocks_f32(input, weights, model.llama_config(), 1);
    const auto output = run_llama_model_blocks_f32(input, backend_weights, model.llama_config(), 1);
    expect_f32_matrix_close(output, expected, 4.0e-3F, "Llama model Metal block backend weight output");
}

void test_runs_llama_model_blocks_with_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::cpu);
    const auto first_input = backend_cached_model_blocks_first_input();
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        0,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama model CPU backend cached blocks output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 1.0e-5F, "Llama model CPU backend cached blocks cache");
}

void test_matches_llama_model_blocks_with_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
    const auto first_input = backend_cached_model_blocks_first_input();
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        0,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::metal);
    expect_f32_matrix_close(output, expected, 5.0e-2F, "Llama model Metal backend cached blocks output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 2.0e-2F, "Llama model Metal backend cached blocks cache");
}

void test_runs_llama_model_blocks_with_backend_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::cpu);
    const auto first_input = backend_cached_model_blocks_first_input();
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::cpu);

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        0,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect(backend_cache.backend == BackendKind::cpu, "Llama backend KV cache CPU backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama model CPU backend resident-KV cached blocks output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        1.0e-5F,
        "Llama model CPU backend resident-KV cached blocks cache"
    );
}

void test_matches_llama_model_blocks_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
    const std::array<float, 4> prefill_input_values {
        3.0F, 4.0F,
        5.0F, 6.0F,
    };
    const auto first_input = tensor_f32_matrix(2, 2, std::span<const float>(prefill_input_values));
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::metal);

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        2,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        2,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::metal);
    expect(backend_cache.backend == BackendKind::metal, "Llama backend KV cache Metal backend mismatch");
    expect_f32_matrix_close(output, expected, 5.0e-2F, "Llama model Metal backend resident-KV cached blocks output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        2.0e-2F,
        "Llama model Metal backend resident-KV cached blocks cache"
    );
}

void test_reports_unavailable_llama_backend_kv_cache_creation_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    try {
        (void)create_llama_backend_kv_cache(backend_cached_model_blocks_fixture_model().llama_config(), BackendKind::metal);
        throw TestFailure("Llama backend KV cache creation should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama backend KV cache creation unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama backend KV cache creation backend mismatch");
    }
}

void test_reports_unavailable_llama_model_blocks_with_kv_cache_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model cached-block Metal backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model cached-block Metal backend weight copy unavailable error code mismatch"
        );
        expect(
            error.backend() == BackendKind::metal,
            "Llama model cached-block Metal backend weight copy backend mismatch"
        );
    }
}

void test_reports_unavailable_llama_model_block_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal block backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal block backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal block backend weight copy backend mismatch");
    }
}

void test_projects_llama_logits_with_cpu_backend_output_weight_buffer() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const auto hidden_states = backend_feed_forward_fixture_input();

    const auto expected = project_llama_logits_f32(hidden_states, weights, model.llama_config());
    const auto output = project_llama_logits_f32(hidden_states, backend_weights, model.llama_config());

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama CPU logits backend output projection");
}

void test_runs_llama_model_logits_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const std::array<std::uint32_t, 2> token_ids {0, 3};

    const auto expected = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        1
    );
    const auto output = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        1
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU logits backend output");
}

void test_matches_llama_model_logits_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
    const std::array<std::uint32_t, 2> token_ids {0, 3};

    const auto expected = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        1
    );
    const auto output = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        1
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::metal);
    expect_f32_matrix_close(output, expected, 1.0e-2F, "Llama model Metal logits backend output");
}

void test_reports_unavailable_llama_model_logits_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal logits backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal logits backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal logits backend weight copy backend mismatch");
    }
}

void test_reports_unavailable_llama_model_feed_forward_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_feed_forward_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal FFN backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal FFN backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal FFN backend weight copy backend mismatch");
    }
}
EngineModel tiny_argmax_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}
void test_runs_llama_next_token_argmax_from_token_ids() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto token = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );

    expect(token == 2, "Llama next-token argmax mismatch");
}

void test_rejects_llama_next_token_empty_token_ids() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 0> token_ids {};

    try {
        (void)run_llama_model_next_token_argmax_f32(
            std::span<const std::uint32_t>(token_ids),
            weights,
            model.llama_config(),
            0
        );
        throw TestFailure("Llama next-token helper should reject empty token IDs");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama token id span must not be empty",
            "Llama next-token empty input error mismatch"
        );
    }
}

void test_runs_llama_next_token_argmax_with_cpu_backend_weight_buffers() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto expected = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );
    const auto token = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect(token == expected, "Llama CPU backend next-token argmax should match CPU reference");
    expect(token == 2, "Llama CPU backend next-token argmax token mismatch");
}

void test_runs_llama_next_token_sampled_with_cpu_backend_weight_buffers() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const std::array<std::uint32_t, 1> token_ids {0};
    const SamplingConfig sampling {
        .temperature = 10.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState expected_state(sampling.seed);
    SamplingState backend_state(sampling.seed);

    const auto expected = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0,
        sampling,
        expected_state,
        std::span<const std::uint32_t>(token_ids)
    );
    const auto token = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0,
        sampling,
        backend_state,
        std::span<const std::uint32_t>(token_ids)
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect(token == expected, "Llama CPU backend next-token sampled output should match CPU reference");
    expect(token == 1, "Llama CPU backend next-token sampled token mismatch");
}

void test_matches_llama_next_token_argmax_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto expected = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );
    const auto token = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::metal);
    expect(token == expected, "Llama Metal backend next-token argmax should match CPU reference");
}

void test_matches_llama_next_token_sampled_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
    const std::array<std::uint32_t, 1> token_ids {0};
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.01F,
        .seed = 17,
    };
    SamplingState expected_state(sampling.seed);
    SamplingState backend_state(sampling.seed);

    const auto expected = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0,
        sampling,
        expected_state,
        std::span<const std::uint32_t>(token_ids)
    );
    const auto token = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0,
        sampling,
        backend_state,
        std::span<const std::uint32_t>(token_ids)
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::metal);
    expect(token == expected, "Llama Metal backend next-token sampled output should match CPU reference");
}

void test_reports_unavailable_llama_next_token_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama next-token Metal backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama next-token Metal backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama next-token Metal backend weight copy backend mismatch");
    }
}

void test_generates_llama_argmax_tokens() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto tokens = generate_llama_argmax_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );

    expect(tokens.size() == 3, "Llama argmax generation token count mismatch");
    expect(tokens[0] == 0, "Llama argmax generation prompt token mismatch");
    expect(tokens[1] == 2, "Llama argmax generation first token mismatch");
    expect(tokens[2] == 0, "Llama argmax generation second token mismatch");
}

void test_generates_llama_argmax_tokens_stops_at_token() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto tokens = generate_llama_argmax_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        4,
        weights,
        model.llama_config(),
        std::optional<std::uint32_t>(2)
    );

    expect(tokens.size() == 2, "Llama argmax stop-token generation count mismatch");
    expect(tokens[0] == 0, "Llama argmax stop-token prompt mismatch");
    expect(tokens[1] == 2, "Llama argmax stop-token generated token mismatch");
}

void test_generates_llama_argmax_tokens_with_kv_cache_matches_no_cache() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto cached_tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );
    const auto no_cache_tokens = generate_llama_argmax_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );

    expect(cached_tokens == no_cache_tokens, "cached Llama argmax generation should match no-cache generation");
    expect(cached_tokens.size() == 3, "cached Llama argmax generation token count mismatch");
    expect(cached_tokens[0] == 0, "cached Llama argmax generation prompt token mismatch");
    expect(cached_tokens[1] == 2, "cached Llama argmax generation first token mismatch");
    expect(cached_tokens[2] == 0, "cached Llama argmax generation second token mismatch");
}

void test_generates_llama_argmax_tokens_with_kv_cache_stops_at_token() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        4,
        weights,
        model.llama_config(),
        std::optional<std::uint32_t>(2)
    );

    expect(tokens.size() == 2, "cached Llama argmax stop-token generation count mismatch");
    expect(tokens[0] == 0, "cached Llama argmax stop-token prompt mismatch");
    expect(tokens[1] == 2, "cached Llama argmax stop-token generated token mismatch");
}

void test_generates_llama_sampled_tokens_with_kv_cache_matches_no_cache() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};
    const SamplingConfig sampling {
        .temperature = 10.0F,
        .top_k = 2,
        .seed = 0,
    };

    const auto cached_tokens = generate_llama_sampled_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config(),
        sampling
    );
    const auto no_cache_tokens = generate_llama_sampled_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config(),
        sampling
    );

    expect(cached_tokens == no_cache_tokens, "cached Llama sampled generation should match no-cache generation");
    expect(cached_tokens.size() == 3, "cached Llama sampled generation token count mismatch");
    expect(cached_tokens[0] == 0, "cached Llama sampled generation prompt token mismatch");
    expect(cached_tokens[1] == 1, "cached Llama sampled generation first token mismatch");
    expect(cached_tokens[2] == 0, "cached Llama sampled generation second token mismatch");
}

void test_rejects_llama_cached_generation_zero_new_tokens() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_with_kv_cache_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            0,
            weights,
            model.llama_config()
        );
        throw TestFailure("cached Llama generation should reject zero new tokens");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation max_new_tokens must be positive",
            "cached Llama generation zero-token error mismatch"
        );
    }
}

void test_rejects_llama_cached_generation_empty_prompt() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 0> prompt_tokens {};

    try {
        (void)generate_llama_argmax_tokens_with_kv_cache_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            1,
            weights,
            model.llama_config()
        );
        throw TestFailure("cached Llama generation should reject empty prompt");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama token id span must not be empty",
            "cached Llama generation empty-prompt error mismatch"
        );
    }
}

void test_rejects_llama_cached_generation_context_overflow() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 2;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_with_kv_cache_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            2,
            weights,
            model.llama_config()
        );
        throw TestFailure("cached Llama generation should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation would exceed context length",
            "cached Llama generation context overflow error mismatch"
        );
    }
}

void test_rejects_llama_generation_zero_new_tokens() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            0,
            weights,
            model.llama_config()
        );
        throw TestFailure("Llama generation should reject zero new tokens");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation max_new_tokens must be positive",
            "Llama generation zero-token error mismatch"
        );
    }
}

void test_rejects_llama_generation_context_overflow() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 2;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            2,
            weights,
            model.llama_config()
        );
        throw TestFailure("Llama generation should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation would exceed context length",
            "Llama generation context overflow error mismatch"
        );
    }
}

}  // namespace

int main() {
    const TestCase tests_before_llama_kv_cache[] = {
        {"default backend starts on CPU", test_default_backend_starts_on_cpu},
        {"tensor shape accepts valid shapes", test_tensor_shape_accepts_valid_shapes},
        {"rejects tensor shape rank over four", test_rejects_tensor_shape_rank_over_four},
        {"rejects tensor shape element count overflow", test_rejects_tensor_shape_element_count_overflow},
        {"calculates tensor byte sizes", test_calculates_tensor_byte_sizes},
        {"rejects unsupported tensor dtype", test_rejects_unsupported_tensor_dtype},
        {"rejects invalid quantized tensor element count", test_rejects_invalid_quantized_tensor_element_count},
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
        {"views loaded GGUF Q8_0 block", test_views_loaded_gguf_q8_0_block},
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
        {"rejects F32 add dtype mismatch", test_rejects_f32_add_dtype_mismatch},
        {"rejects F32 add shape mismatch", test_rejects_f32_add_shape_mismatch},
        {"multiplies F32 tensors elementwise", test_multiplies_f32_tensors_elementwise},
        {"rejects F32 multiply dtype mismatch", test_rejects_f32_multiply_dtype_mismatch},
        {"rejects F32 multiply shape mismatch", test_rejects_f32_multiply_shape_mismatch},
        {"applies F32 SiLU", test_applies_f32_silu},
        {"rejects F32 SiLU dtype mismatch", test_rejects_f32_silu_dtype_mismatch},
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
        {"applies F32 RoPE frequency base by pair", test_applies_f32_rope_frequency_base_by_pair},
        {"rejects F32 RoPE dtype mismatch", test_rejects_f32_rope_dtype_mismatch},
        {"rejects F32 RoPE input rank above two", test_rejects_f32_rope_input_rank_above_two},
        {"rejects F32 RoPE invalid dimension count", test_rejects_f32_rope_invalid_dimension_count},
        {"rejects F32 RoPE dimension count above hidden", test_rejects_f32_rope_dimension_count_above_hidden},
        {"rejects F32 RoPE nonpositive frequency base", test_rejects_f32_rope_nonpositive_frequency_base},
        {"rejects F32 RoPE row position overflow", test_rejects_f32_rope_row_position_overflow},
        {"applies F32 softmax to vector", test_applies_f32_softmax_to_vector},
        {"applies F32 softmax to batched rows", test_applies_f32_softmax_to_batched_rows},
        {"applies F32 softmax to large values", test_applies_f32_softmax_to_large_values},
        {"rejects F32 softmax dtype mismatch", test_rejects_f32_softmax_dtype_mismatch},
        {"rejects F32 softmax input rank above two", test_rejects_f32_softmax_input_rank_above_two},
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
        {"rejects F32 matmul dtype mismatch", test_rejects_f32_matmul_dtype_mismatch},
        {"rejects F32 matmul non-matrix rank", test_rejects_f32_matmul_non_matrix_rank},
        {"rejects F32 matmul inner dimension mismatch", test_rejects_f32_matmul_inner_dimension_mismatch},
        {"multiplies F32 by F16 matrices into F32", test_multiplies_f32_by_f16_matrices_into_f32},
        {"multiplies F32 by F16 GGUF weight matrix", test_multiplies_f32_by_f16_gguf_weight_matrix},
        {
            "multiplies F32 by F16 GGUF weight matrix on CPU backend",
            test_multiplies_f32_by_f16_gguf_weight_matrix_on_cpu_backend,
        },
        {
            "copies tensor to and from CPU backend buffer",
            test_copies_tensor_to_and_from_cpu_backend_buffer,
        },
        {
            "multiplies F32 by CPU backend F16 GGUF weight buffer",
            test_multiplies_f32_by_cpu_backend_f16_gguf_weight_buffer,
        },
        {
            "multiplies backend F32 by backend F16 GGUF weight to CPU backend buffer",
            test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_cpu_backend_buffer,
        },
        {
            "multiplies backend F32 by three backend F16 GGUF weights to CPU backend buffers",
            test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_cpu_backend_buffers,
        },
        {
            "applies SiLU multiply to CPU backend buffers",
            test_applies_silu_multiply_to_cpu_backend_buffers,
        },
        {
            "runs gated SiLU projection to CPU backend buffer",
            test_runs_gated_silu_projection_to_cpu_backend_buffer,
        },
        {
            "runs residual gated SiLU projection to CPU backend buffer",
            test_runs_residual_gated_silu_projection_to_cpu_backend_buffer,
        },
        {
            "matches CPU F32 by F16 GGUF weight matmul on Metal backend",
            test_matches_cpu_f32_by_f16_gguf_weight_matmul_on_metal_backend,
        },
        {
            "copies tensor to and from Metal backend buffer when available",
            test_copies_tensor_to_and_from_metal_backend_buffer_when_available,
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
        {
            "runs Llama cached attention to CPU backend buffer",
            test_runs_llama_cached_attention_to_cpu_backend_buffer,
        },
        {
            "matches Llama cached attention to Metal backend buffer when available",
            test_matches_llama_cached_attention_to_metal_backend_buffer_when_available,
        },
        {
            "matches one-token grouped Llama cached attention to Metal backend buffer when available",
            test_matches_llama_one_token_grouped_cached_attention_to_metal_backend_buffer_when_available,
        },
        {
            "reuses Metal F32 by F16 GGUF weight matmul state",
            test_reuses_metal_f32_by_f16_gguf_weight_matmul_state,
        },
        {
            "reports unavailable Metal backend for GGUF matmul when missing",
            test_reports_unavailable_metal_backend_for_gguf_matmul_when_missing,
        },
        {
            "reports unavailable Metal backend buffer copy when missing",
            test_reports_unavailable_metal_backend_buffer_copy_when_missing,
        },
        {"rejects F32xF16 matmul right dtype mismatch", test_rejects_f32_f16_matmul_right_dtype_mismatch},
    };

    const TestCase tests_after_attention_layer_primary_before_attention_validation[] = {
        {
            "runs Llama attention layer with CPU backend weight buffers",
            test_runs_llama_attention_layer_with_cpu_backend_weight_buffers,
        },
        {
            "runs Llama attention layer with KV cache and CPU backend weight buffers",
            test_runs_llama_attention_layer_with_kv_cache_and_cpu_backend_weight_buffers,
        },
    };

    const TestCase tests_after_ffn_layer_primary_before_ffn_validation[] = {
        {
            "runs Llama FFN layer with CPU backend weight buffers",
            test_runs_llama_feed_forward_layer_with_cpu_backend_weight_buffers,
        },
    };

    const TestCase tests_after_block_primary_before_block_validation[] = {
        {
            "runs Llama block with CPU backend weight buffers",
            test_runs_llama_block_with_cpu_backend_weight_buffers,
        },
        {
            "runs Llama block with KV cache and CPU backend weight buffers",
            test_runs_llama_block_with_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama block with KV cache and Metal backend weight buffers when available",
            test_matches_llama_block_with_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "runs Llama block with backend KV cache and CPU backend weight buffers",
            test_runs_llama_block_with_backend_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama block with backend KV cache and Metal backend weight buffers when available",
            test_matches_llama_block_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
    };

    const TestCase tests_after_model_block_before_model_logits[] = {
        {
            "copies Llama model attention weights to CPU backend buffers",
            test_copies_llama_model_attention_weights_to_cpu_backend_buffers,
        },
        {
            "matches Llama attention layer with Metal backend weight buffers when available",
            test_matches_llama_attention_layer_with_metal_backend_weight_buffers_when_available,
        },
        {
            "matches Llama attention layer with KV cache and Metal backend weight buffers when available",
            test_matches_llama_attention_layer_with_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama model attention backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_attention_backend_weight_copy_when_metal_missing,
        },
        {
            "copies Llama model FFN weights to CPU backend buffers",
            test_copies_llama_model_feed_forward_weights_to_cpu_backend_buffers,
        },
        {
            "matches Llama FFN layer with Metal backend weight buffers when available",
            test_matches_llama_feed_forward_layer_with_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama model FFN backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_feed_forward_backend_weight_copy_when_metal_missing,
        },
        {
            "copies Llama model block weights to CPU backend buffers",
            test_copies_llama_model_block_weights_to_cpu_backend_buffers,
        },
        {
            "matches Llama model blocks with Metal backend weight buffers when available",
            test_matches_llama_model_blocks_with_metal_backend_weight_buffers_when_available,
        },
        {
            "runs Llama model blocks with KV cache and CPU backend weight buffers",
            test_runs_llama_model_blocks_with_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama model blocks with KV cache and Metal backend weight buffers when available",
            test_matches_llama_model_blocks_with_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "runs Llama model blocks with backend KV cache and CPU backend weight buffers",
            test_runs_llama_model_blocks_with_backend_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama model blocks with backend KV cache and Metal backend weight buffers when available",
            test_matches_llama_model_blocks_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama backend KV cache creation when Metal is missing",
            test_reports_unavailable_llama_backend_kv_cache_creation_when_metal_missing,
        },
        {
            "reports unavailable Llama model blocks with KV cache backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_blocks_with_kv_cache_backend_weight_copy_when_metal_missing,
        },
        {
            "reports unavailable Llama model block backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_block_backend_weight_copy_when_metal_missing,
        },
        {
            "projects Llama logits with CPU backend output weight buffer",
            test_projects_llama_logits_with_cpu_backend_output_weight_buffer,
        },
        {
            "runs Llama model logits with CPU backend weight buffers",
            test_runs_llama_model_logits_with_cpu_backend_weight_buffers,
        },
        {
            "matches Llama model logits with Metal backend weight buffers when available",
            test_matches_llama_model_logits_with_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama model logits backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_logits_backend_weight_copy_when_metal_missing,
        },
    };

    const TestCase tests_after_sampling[] = {
        {"runs Llama next-token argmax from token IDs", test_runs_llama_next_token_argmax_from_token_ids},
        {"rejects Llama next-token empty token IDs", test_rejects_llama_next_token_empty_token_ids},
        {
            "runs Llama next-token argmax with CPU backend weight buffers",
            test_runs_llama_next_token_argmax_with_cpu_backend_weight_buffers,
        },
        {
            "runs Llama next-token sampled with CPU backend weight buffers",
            test_runs_llama_next_token_sampled_with_cpu_backend_weight_buffers,
        },
        {
            "matches Llama next-token argmax with Metal backend weight buffers when available",
            test_matches_llama_next_token_argmax_with_metal_backend_weight_buffers_when_available,
        },
        {
            "matches Llama next-token sampled with Metal backend weight buffers when available",
            test_matches_llama_next_token_sampled_with_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama next-token backend weight copy when Metal is missing",
            test_reports_unavailable_llama_next_token_backend_weight_copy_when_metal_missing,
        },
        {"generates Llama argmax tokens", test_generates_llama_argmax_tokens},
        {"generates Llama argmax tokens stops at token", test_generates_llama_argmax_tokens_stops_at_token},
        {
            "generates Llama argmax tokens with KV cache matches no-cache",
            test_generates_llama_argmax_tokens_with_kv_cache_matches_no_cache,
        },
        {
            "generates Llama argmax tokens with KV cache stops at token",
            test_generates_llama_argmax_tokens_with_kv_cache_stops_at_token,
        },
        {
            "generates Llama sampled tokens with KV cache matches no-cache",
            test_generates_llama_sampled_tokens_with_kv_cache_matches_no_cache,
        },
        {
            "rejects Llama cached generation zero new tokens",
            test_rejects_llama_cached_generation_zero_new_tokens,
        },
        {
            "rejects Llama cached generation empty prompt",
            test_rejects_llama_cached_generation_empty_prompt,
        },
        {
            "rejects Llama cached generation context overflow",
            test_rejects_llama_cached_generation_context_overflow,
        },
        {"rejects Llama generation zero new tokens", test_rejects_llama_generation_zero_new_tokens},
        {"rejects Llama generation context overflow", test_rejects_llama_generation_context_overflow},
    };

    int failures = 0;
    std::size_t test_count = 0;
    const auto run_test = [&](const TestCase& test) {
        ++test_count;
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    };

    for (const auto& test : tests_before_llama_kv_cache) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_kv_cache_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_attention_layer_primary_test_cases()) {
        run_test(test);
    }
    for (const auto& test : tests_after_attention_layer_primary_before_attention_validation) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_attention_layer_validation_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_feed_forward_layer_primary_test_cases()) {
        run_test(test);
    }
    for (const auto& test : tests_after_ffn_layer_primary_before_ffn_validation) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_feed_forward_layer_validation_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_block_primary_test_cases()) {
        run_test(test);
    }
    for (const auto& test : tests_after_block_primary_before_block_validation) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_block_validation_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_model_block_test_cases()) {
        run_test(test);
    }
    for (const auto& test : tests_after_model_block_before_model_logits) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::llama_model_logits_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::sampling_test_cases()) {
        run_test(test);
    }
    for (const auto& test : tests_after_sampling) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::generate_cli_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gguf_tensor_bridge_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gguf_reader_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gguf_inspect_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::engine_model_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::tokenizer_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::benchmark_test_cases()) {
        run_test(test);
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << test_count << " test(s) passed\n";
    return EXIT_SUCCESS;
}
