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

#include "../../src/tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <array>
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
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::tensor_backend_unavailable_reason;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::quantize_f16_gguf_weight_q8_0;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::add_f32_to_backend;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32_to_backend;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32_to_backend;
using local_gguf_engine::tensor::ops::embed_f16_gguf_weight_token_ids_to_backend;
using local_gguf_engine::tensor::ops::gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::gelu_tanh_f32;
using local_gguf_engine::tensor::ops::gelu_tanh_multiply_f32_to_backend;
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
using local_gguf_engine::tensor::ops::rope_neox_f32_by_heads_with_position_start_to_backend;
using local_gguf_engine::tensor::ops::silu_f32;
using local_gguf_engine::tensor::ops::silu_multiply_f32_to_backend;
using local_gguf_engine::tensor::ops::softcap_f32;
using local_gguf_engine::tensor::ops::softcap_f32_to_backend;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::softmax_f32_to_backend;
using local_gguf_engine::tensor::ops::top_logit_f32;
using local_gguf_engine::tensor::requantize_q6_k_gguf_weight_q4_k;
using local_gguf_engine::tensor::backends::metal::fill_shared_metal_buffer_u32;
using local_gguf_engine::tensor::backends::metal::MetalCommandContext;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::reference_f32_quantized_gguf_weight_matmul;
using local_gguf_engine::tests::reference_f16_staged_quantized_gguf_weight_matmul;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f16_matrix;
using local_gguf_engine::tests::tensor_f32_projection_values;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::tensor_q4_k_gguf_weight_matrix;
using local_gguf_engine::tests::tensor_q4_k_stress_gguf_weight_matrix;
using local_gguf_engine::tests::tensor_q6_k_gguf_weight_matrix;
using local_gguf_engine::tests::tensor_q6_k_stress_gguf_weight_matrix;
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

void test_applies_gelu_tanh_multiply_to_cpu_backend_buffers() {
    const std::array<float, 8> gate_values {-14.181760F, -2.0F, -1.0F, 0.0F, 0.5F, 1.0F, 2.0F, 14.181760F};
    const std::array<float, 8> up_values {0.877355F, 0.25F, -0.5F, 1.0F, 1.5F, -2.0F, 0.75F, 0.877355F};
    const auto gate = tensor_f32_matrix(2, 4, std::span<const float>(gate_values));
    const auto up = tensor_f32_matrix(2, 4, std::span<const float>(up_values));
    const auto gate_buffer = copy_tensor_to_backend(gate, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up, BackendKind::cpu);

    const auto output_buffer = gelu_tanh_multiply_f32_to_backend(gate_buffer, up_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = multiply_f32(gelu_tanh_f32(gate), up);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU GELU-tanh multiply output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer GELU-tanh multiply output");
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

void expect_f32_matrix_close_absolute_relative(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float absolute_tolerance,
    float relative_tolerance,
    std::string_view message
);

void test_matches_cpu_f16_simd_matmul_boundaries_on_metal_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::array<std::uint64_t, 8> shared_widths {4, 31, 32, 33, 255, 256, 257, 3584};
    constexpr std::array<std::uint64_t, 6> output_counts {1, 2, 7, 8, 9, 257};
    for (const auto shared : shared_widths) {
        const auto left_values = tensor_f32_projection_values(1, shared);
        const auto left = tensor_f32_matrix(1, shared, std::span<const float>(left_values));
        for (const auto columns : output_counts) {
            std::vector<float> weight_values(static_cast<std::size_t>(shared * columns));
            for (std::size_t index = 0; index < weight_values.size(); ++index) {
                const auto signed_pattern = static_cast<int>(index % 29U) - 14;
                weight_values[index] = static_cast<float>(signed_pattern) * 0.03125F
                    + static_cast<float>((index / 29U) % 7U) * 0.00390625F;
            }
            const auto right = tensor_f16_gguf_matrix(
                shared, columns, std::span<const float>(weight_values)
            );
            const auto expected = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
            const auto actual = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
            expect_f32_matrix_close_absolute_relative(
                actual,
                expected,
                0.01F,
                1.0e-5F,
                "Metal F16 SIMD one-row boundary parity"
            );
        }
    }
}

void test_matches_cpu_f16_simd_multi_row_matmul_on_metal_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t rows = 3;
    constexpr std::uint64_t shared = 33;
    constexpr std::uint64_t columns = 9;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    std::vector<float> weight_values(static_cast<std::size_t>(shared * columns));
    for (std::size_t index = 0; index < weight_values.size(); ++index) {
        weight_values[index] = static_cast<float>(static_cast<int>(index % 19U) - 9) * 0.0625F;
    }
    const auto left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(shared, columns, std::span<const float>(weight_values));
    expect_f32_matrix_close_absolute_relative(
        matmul_f32_f16_gguf_weight(left, right, BackendKind::metal),
        matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu),
        0.01F,
        1.0e-5F,
        "Metal F16 SIMD multi-row parity"
    );
}

void expect_f32_matrix_close_absolute_relative(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float absolute_tolerance,
    float relative_tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    const auto rows = actual.descriptor().shape.dimension(0);
    const auto columns = actual.descriptor().shape.dimension(1);
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            const auto actual_value = read_f32(actual, std::span<const std::uint64_t>(index));
            const auto expected_value = read_f32(expected, std::span<const std::uint64_t>(index));
            const auto tolerance = absolute_tolerance + relative_tolerance * std::fabs(expected_value);
            expect_close(actual_value, expected_value, tolerance, message);
        }
    }
}

void test_matches_independent_q4_k_and_q6_k_small_batch_and_tiled_projection_on_metal() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    struct ProjectionCase {
        std::uint64_t rows;
        std::uint64_t packed_blocks;
        std::uint64_t columns;
    };
    // Distribute the small-batch/tile boundaries, production K-block counts,
    // and the 64-column output boundary without a combinatorial suite.
    constexpr std::array<ProjectionCase, 17> cases {{
        {2, 1, 1}, {2, 14, 3}, {2, 16, 4}, {2, 56, 5},
        {3, 2, 2}, {4, 14, 3}, {5, 16, 4},
        {6, 56, 5}, {7, 1, 63}, {8, 2, 64},
        {9, 14, 65}, {15, 16, 5}, {16, 56, 4},
        {17, 1, 3}, {31, 2, 2}, {32, 14, 1}, {33, 16, 5},
    }};

    for (const auto dtype : {DType::q4_k, DType::q6_k}) {
        for (const auto& test_case : cases) {
            const auto shared = test_case.packed_blocks * 256U;
            auto left_values = tensor_f32_projection_values(test_case.rows, shared);
            for (std::size_t index = 0; index < left_values.size(); ++index) {
                if (index % 17U == 0U) {
                    left_values[index] *= 16.0F;
                } else if (index % 13U == 0U) {
                    left_values[index] *= 0.03125F;
                }
            }
            const auto left = tensor_f32_matrix(
                test_case.rows, shared, std::span<const float>(left_values)
            );
            const auto right = dtype == DType::q4_k
                ? tensor_q4_k_stress_gguf_weight_matrix(shared, test_case.columns, test_case.rows)
                : tensor_q6_k_stress_gguf_weight_matrix(shared, test_case.columns, test_case.rows);
            const auto expected = reference_f32_quantized_gguf_weight_matmul(left, right);
            const auto staged_expected = reference_f16_staged_quantized_gguf_weight_matmul(
                left, right
            );

            const auto tensor_output = matmul_f32_quantized_gguf_weight(
                left, right, BackendKind::metal
            );
            const auto backend_output = copy_tensor_from_backend(
                matmul_f32_quantized_gguf_weight_to_backend(
                    copy_tensor_to_backend(left, BackendKind::metal),
                    copy_tensor_to_backend(right, BackendKind::metal)
                )
            );
            const bool small_batch = test_case.rows <= 8;
            const auto absolute_tolerance = small_batch
                ? (dtype == DType::q4_k ? 0.05F : 0.1F)
                : (dtype == DType::q4_k ? 2.0F : 0.01F);
            const auto relative_tolerance = small_batch || dtype == DType::q6_k
                ? 1.0e-5F
                : 2.0e-2F;
            expect_f32_matrix_close_absolute_relative(
                tensor_output,
                expected,
                absolute_tolerance,
                relative_tolerance,
                "Metal TensorStorage K-quant multi-row projection parity"
            );
            expect_f32_matrix_close_absolute_relative(
                backend_output,
                small_batch ? expected : staged_expected,
                absolute_tolerance,
                relative_tolerance,
                "Metal backend-buffer K-quant selected numerical-contract parity"
            );
            if (test_case.rows == 2U) {
                for (std::uint64_t row = 0; row < 2U; ++row) {
                    const auto row_begin = static_cast<std::size_t>(row * shared);
                    const auto row_input = tensor_f32_matrix(
                        1U,
                        shared,
                        std::span<const float>(left_values).subspan(
                            row_begin, static_cast<std::size_t>(shared)
                        )
                    );
                    const auto row_output = matmul_f32_quantized_gguf_weight(
                        row_input, right, BackendKind::metal
                    );
                    for (std::uint64_t column = 0; column < test_case.columns; ++column) {
                        const std::array<std::uint64_t, 2> batched_index {row, column};
                        const std::array<std::uint64_t, 2> row_index {0U, column};
                        expect(
                            read_f32(tensor_output, batched_index)
                                == read_f32(row_output, row_index),
                            "Metal K-quant width-two packed decode-order parity"
                        );
                    }
                }
            }
        }
    }
}

void test_exact_16_k_quant_projection_overwrites_poison_at_real_tiny_widths_on_metal() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    constexpr auto rows = std::uint64_t {16U};
    constexpr auto shared = std::uint64_t {256U};
    constexpr auto poison = std::uint32_t {0x7fc00000U};
    for (const auto dtype : {DType::q4_k, DType::q6_k}) {
        for (const auto columns : {std::uint64_t {256U}, std::uint64_t {512U}}) {
            auto left_values = tensor_f32_projection_values(rows, shared);
            for (std::size_t index = 0U; index < left_values.size(); ++index) {
                left_values[index] += static_cast<float>(index % 7U) * 0.00390625F;
            }
            const auto left = tensor_f32_matrix(rows, shared, left_values);
            const auto right = dtype == DType::q4_k
                ? tensor_q4_k_stress_gguf_weight_matrix(shared, columns, columns)
                : tensor_q6_k_stress_gguf_weight_matrix(shared, columns, columns);
            const auto expected = reference_f16_staged_quantized_gguf_weight_matmul(
                left, right
            );
            const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
            const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);
            const auto output = allocate_backend_buffer(
                TensorDescriptor::from(Shape::from({rows, columns}), DType::f32),
                BackendKind::metal
            );
            fill_shared_metal_buffer_u32(output, poison);

            MetalCommandContext command("exact-16 K-quant poison overwrite parity");
            command.encode_matmul(left_buffer, right_buffer, output, 16U);
            const auto stats = command.commit_and_wait();
            expect(
                (dtype == DType::q4_k ? stats.q4_exact_16_projection_count
                                      : stats.q6_exact_16_projection_count) == 1U,
                "Metal exact-16 K-quant parity did not select the specialization"
            );
            const auto actual = copy_tensor_from_backend(output);
            const auto absolute_tolerance = dtype == DType::q4_k ? 2.0F : 0.01F;
            const auto relative_tolerance = dtype == DType::q4_k ? 2.0e-2F : 1.0e-5F;
            expect_f32_matrix_close_absolute_relative(
                actual,
                expected,
                absolute_tolerance,
                relative_tolerance,
                "Metal exact-16 K-quant poison-overwrite parity"
            );
        }
    }
}

void test_matches_cpu_k_quant_decode_simd_boundaries_on_metal() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::array<std::uint64_t, 5> packed_block_counts {1, 2, 14, 16, 56};
    constexpr std::array<std::uint64_t, 6> output_counts {1, 2, 3, 4, 5, 7};
    for (const auto dtype : {DType::q4_k, DType::q6_k}) {
        for (const auto packed_block_count : packed_block_counts) {
            const auto shared = packed_block_count * 256U;
            auto left_values = tensor_f32_projection_values(1, shared);
            for (std::size_t index = 0; index < left_values.size(); ++index) {
                if (index % 17U == 0U) {
                    left_values[index] *= -8.0F;
                } else if (index % 11U == 0U) {
                    left_values[index] *= 0.125F;
                }
            }
            const auto left = tensor_f32_matrix(1, shared, std::span<const float>(left_values));
            for (const auto columns : output_counts) {
                const auto right = dtype == DType::q4_k
                    ? tensor_q4_k_stress_gguf_weight_matrix(shared, columns, packed_block_count)
                    : tensor_q6_k_stress_gguf_weight_matrix(shared, columns, packed_block_count);
                const auto expected = reference_f32_quantized_gguf_weight_matmul(left, right);
                const auto actual = copy_tensor_from_backend(
                    matmul_f32_quantized_gguf_weight_to_backend(
                        copy_tensor_to_backend(left, BackendKind::metal),
                        copy_tensor_to_backend(right, BackendKind::metal)
                    )
                );
                expect_f32_matrix_close_absolute_relative(
                    actual,
                    expected,
                    dtype == DType::q4_k ? 0.1F : 0.25F,
                    1.0e-5F,
                    "Metal K-quant decode SIMD boundary parity"
                );
            }
        }
    }
}

void test_matches_converted_q6_k_weight_on_q4_k_metal_paths() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    constexpr std::uint64_t shared = 56U * 256U;
    constexpr std::uint64_t columns = 5U;
    const auto source = tensor_q6_k_stress_gguf_weight_matrix(shared, columns, 23U);
    const auto converted = requantize_q6_k_gguf_weight_q4_k(source.view());
    for (const std::uint64_t rows : {1U, 15U}) {
        const auto values = tensor_f32_projection_values(rows, shared);
        const auto left = tensor_f32_matrix(rows, shared, values);
        const auto expected = rows == 1U
            ? reference_f32_quantized_gguf_weight_matmul(left, converted)
            : reference_f16_staged_quantized_gguf_weight_matmul(left, converted);
        const auto actual = copy_tensor_from_backend(
            matmul_f32_quantized_gguf_weight_to_backend(
                copy_tensor_to_backend(left, BackendKind::metal),
                copy_tensor_to_backend(converted, BackendKind::metal)
            )
        );
        expect_f32_matrix_close_absolute_relative(
            actual,
            expected,
            rows == 1U ? 0.1F : 2.0F,
            rows == 1U ? 1.0e-5F : 2.0e-2F,
            "converted Q6-to-Q4 Metal projection parity"
        );
    }
}

void test_matches_cpu_q8_0_output_matvec_on_metal() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    for (const std::uint64_t shared : {64U, 3584U}) {
        constexpr std::uint64_t columns = 5U;
        std::vector<float> activations(static_cast<std::size_t>(2U * shared));
        std::vector<float> weights(static_cast<std::size_t>(shared * columns));
        for (std::size_t index = 0; index < activations.size(); ++index) {
            activations[index] = static_cast<float>(static_cast<int>(index % 17U) - 8) / 9.0F;
        }
        for (std::size_t index = 0; index < weights.size(); ++index) {
            weights[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) / 37.0F;
        }
        const auto f16_weight = tensor_f16_gguf_matrix(shared, columns, std::span<const float>(weights));
        const auto q8_weight = quantize_f16_gguf_weight_q8_0(f16_weight.view());
        for (const std::uint64_t rows : {1U, 2U}) {
            const auto input = tensor_f32_matrix(
                rows,
                shared,
                std::span<const float>(activations).first(static_cast<std::size_t>(rows * shared))
            );
            const auto cpu = matmul_f32_quantized_gguf_weight(input, q8_weight, BackendKind::cpu);
            const auto metal = matmul_f32_quantized_gguf_weight(input, q8_weight, BackendKind::metal);
            expect_f32_matrix_close(metal, cpu, 2.0e-4F, "Metal Q8_0 output projection");
            if (rows == 2U) {
                for (std::uint64_t row = 0; row < 2U; ++row) {
                    const auto row_input = tensor_f32_matrix(
                        1U,
                        shared,
                        std::span<const float>(activations).subspan(
                            static_cast<std::size_t>(row * shared),
                            static_cast<std::size_t>(shared)
                        )
                    );
                    const auto row_output = matmul_f32_quantized_gguf_weight(
                        row_input, q8_weight, BackendKind::metal
                    );
                    for (std::uint64_t column = 0; column < columns; ++column) {
                        const std::array<std::uint64_t, 2> batched_index {row, column};
                        const std::array<std::uint64_t, 2> row_index {0U, column};
                        expect_close(
                            read_f32(metal, batched_index), read_f32(row_output, row_index),
                            2.0e-4F, "Metal Q8_0 width-two decode-order parity"
                        );
                    }
                }
            }
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
    const std::array<std::uint32_t, 17> token_ids {
        2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2,
    };
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

void test_matches_neox_rope_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 8> input_values {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    const auto input = tensor_f32_matrix(2, 4, std::span<const float>(input_values));
    const auto cpu_output = rope_neox_f32_by_heads_with_position_start_to_backend(
        copy_tensor_to_backend(input, BackendKind::cpu), 3, 1, 4, 4, 10000.0F
    );
    const auto metal_output = rope_neox_f32_by_heads_with_position_start_to_backend(
        copy_tensor_to_backend(input, BackendKind::metal), 3, 1, 4, 4, 10000.0F
    );

    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output),
        copy_tensor_from_backend(cpu_output),
        1.0e-5F,
        "Metal backend-buffer NeoX RoPE output"
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

void test_applies_gelu_tanh_multiply_to_metal_backend_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> gate_values {-2.0F, -1.0F, 0.0F, 0.5F, 1.0F, 2.0F};
    const std::array<float, 6> up_values {0.25F, -0.5F, 1.0F, 1.5F, -2.0F, 0.75F};
    const auto gate = tensor_f32_matrix(2, 3, std::span<const float>(gate_values));
    const auto up = tensor_f32_matrix(2, 3, std::span<const float>(up_values));
    const auto gate_buffer = copy_tensor_to_backend(gate, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up, BackendKind::metal);

    const auto output_buffer = gelu_tanh_multiply_f32_to_backend(gate_buffer, up_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = multiply_f32(gelu_tanh_f32(gate), up);

    expect(output_buffer.backend() == BackendKind::metal, "Metal GELU-tanh multiply output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-5F, "Metal backend-buffer GELU-tanh multiply output");
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

void test_reports_unavailable_metal_backend_operations_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> matrix_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> weight_values {
        5.0F, 6.0F,
        7.0F, 8.0F,
    };
    const auto left = tensor_f32_matrix(2, 2, std::span<const float>(matrix_values));
    const auto right = tensor_f16_gguf_matrix(2, 2, std::span<const float>(weight_values));

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

    constexpr std::uint64_t rows = 1;
    constexpr std::uint64_t shared = 256;
    constexpr std::uint64_t columns = 256;
    const auto left_values = tensor_f32_projection_values(rows, shared);
    const auto quantized_left = tensor_f32_matrix(rows, shared, std::span<const float>(left_values));
    const auto quantized_right = tensor_q4_k_gguf_weight_matrix(shared, columns);

    try {
        (void)matmul_f32_quantized_gguf_weight(quantized_left, quantized_right, BackendKind::metal);
        throw TestFailure("Metal K-quant GGUF matmul should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal K-quant GGUF matmul unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal K-quant GGUF matmul unavailable backend mismatch");
    }

    try {
        (void)copy_tensor_to_backend(left, BackendKind::metal);
        throw TestFailure("Metal backend buffer copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal backend buffer unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal backend buffer unavailable backend mismatch");
    }

    try {
        (void)copy_tensor_view_to_backend(left.view(), BackendKind::metal);
        throw TestFailure("Metal backend view copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal backend view-copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal backend view-copy unavailable backend mismatch");
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
    {
        "multiplies backend F32 by three backend F16 GGUF weights to CPU backend buffers",
        test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_cpu_backend_buffers,
    },
    {"applies SiLU multiply to CPU backend buffers", test_applies_silu_multiply_to_cpu_backend_buffers},
    {"applies GELU-tanh multiply to CPU backend buffers", test_applies_gelu_tanh_multiply_to_cpu_backend_buffers},
    {"runs gated SiLU projection to CPU backend buffer", test_runs_gated_silu_projection_to_cpu_backend_buffer},
    {
        "runs residual gated SiLU projection to CPU backend buffer",
        test_runs_residual_gated_silu_projection_to_cpu_backend_buffer,
    },
    {
        "matches CPU F16 SIMD matmul boundaries on Metal backend",
        test_matches_cpu_f16_simd_matmul_boundaries_on_metal_backend,
    },
    {
        "matches CPU F16 SIMD multi-row matmul on Metal backend",
        test_matches_cpu_f16_simd_multi_row_matmul_on_metal_backend,
    },
    {
        "matches independent Q4_K and Q6_K small-batch and tiled projection on Metal",
        test_matches_independent_q4_k_and_q6_k_small_batch_and_tiled_projection_on_metal,
    },
    {
        "exact-16 K-quant projection overwrites poison at real tiny widths on Metal",
        test_exact_16_k_quant_projection_overwrites_poison_at_real_tiny_widths_on_metal,
    },
    {
        "matches converted Q6_K weight on Q4_K Metal paths",
        test_matches_converted_q6_k_weight_on_q4_k_metal_paths,
    },
    {
        "matches CPU K-quant decode SIMD boundaries on Metal",
        test_matches_cpu_k_quant_decode_simd_boundaries_on_metal,
    },
    {"matches CPU Q8_0 output matvec on Metal", test_matches_cpu_q8_0_output_matvec_on_metal},
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
        "matches NeoX RoPE to Metal backend buffer when available",
        test_matches_neox_rope_to_metal_backend_buffer_when_available,
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
        "multiplies backend F32 by three backend F16 GGUF weights to Metal backend buffers when available",
        test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_metal_backend_buffers_when_available,
    },
    {
        "applies SiLU multiply to Metal backend buffers when available",
        test_applies_silu_multiply_to_metal_backend_buffers_when_available,
    },
    {
        "applies GELU-tanh multiply to Metal backend buffers when available",
        test_applies_gelu_tanh_multiply_to_metal_backend_buffers_when_available,
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
        "reports unavailable Metal backend operations when missing",
        test_reports_unavailable_metal_backend_operations_when_missing,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> tensor_backend_test_cases() {
    return std::span<const TestCase>(tensor_backend_tests);
}

}  // namespace local_gguf_engine::tests
