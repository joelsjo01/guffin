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
#include "../../src/tensor/backends/metal_backend.hpp"
#include "metal_test_hooks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>
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
#if defined(LOCAL_GGUF_ENGINE_TESTING)
using local_gguf_engine::tensor::backends::metal::set_metal_device_unavailable_for_testing;
#endif
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
            if (small_batch) {
                for (std::uint64_t row = 0; row < test_case.rows; ++row) {
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
                            "Metal K-quant small-batch packed decode-order parity"
                        );
                        expect(
                            read_f32(backend_output, batched_index)
                                == read_f32(row_output, row_index),
                            "Metal resident K-quant small-batch packed decode-order parity"
                        );
                    }
                }
            }
        }
    }
}

void test_gemma3_parallel_projection_preserves_scalar_bits_for_native_shapes() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    struct Case { std::uint32_t rows, blocks, columns, head_offset; };
    // Actual 12B K dimensions are 15 (Q/K/V, gate/up, head), 16 (O), and
    // 60 (down). Every width covers one-row execution and two-row tile tails;
    // odd columns and head offsets catch masked lanes and later selection tiles.
    constexpr std::array<Case, 17> cases {{
        {2, 1, 1, 1},
        {1, 15, 3, 0}, {2, 15, 5, 1}, {3, 15, 7, 3},
        {4, 15, 33, 4}, {5, 15, 65, 5},
        {1, 16, 5, 1}, {2, 16, 67, 3}, {3, 16, 3, 4},
        {4, 16, 4, 5}, {5, 16, 7, 1},
        {1, 60, 3, 3}, {2, 60, 4, 4}, {3, 60, 5, 5},
        {4, 60, 5, 1}, {5, 60, 7, 3}, {8, 60, 3, 1},
    }};
    for (const auto dtype : {DType::q4_k, DType::q6_k}) for (const auto shape : cases) {
        const auto shared = std::uint64_t(shape.blocks) * 256;
        const auto input_rows = shape.rows + shape.head_offset;
        auto values = tensor_f32_projection_values(input_rows, shared);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = values[i] * (i % 17 == 0 ? 16.0F : i % 13 == 0 ? 0.03125F : 1.0F)
                + 0.00037F * static_cast<float>(i % 11);
        const auto input = tensor_f32_matrix(input_rows, shared, values);
        const auto weight = dtype == DType::q4_k
            ? tensor_q4_k_stress_gguf_weight_matrix(shared, shape.columns, shape.rows)
            : tensor_q6_k_stress_gguf_weight_matrix(shared, shape.columns, shape.rows);
        const auto expected = reference_f32_quantized_gguf_weight_matmul(input, weight);
        const auto device_input = copy_tensor_to_backend(input, BackendKind::metal);
        const auto device_weight = copy_tensor_to_backend(weight, BackendKind::metal);
        const auto descriptor = TensorDescriptor::from(Shape::from({shape.rows + 1, shape.columns}), DType::f32);
        const auto batch = allocate_backend_buffer(descriptor, BackendKind::metal);
        const auto head = allocate_backend_buffer(descriptor, BackendKind::metal);
        fill_shared_metal_buffer_u32(batch, 0x7fc00000U);
        fill_shared_metal_buffer_u32(head, 0x7fc00000U);
        const auto ordinary_first = allocate_backend_buffer(
            TensorDescriptor::from(Shape::from({1, shape.columns}), DType::f32), BackendKind::metal);
        std::vector<local_gguf_engine::tensor::BackendBuffer> scalar;
        scalar.reserve(input_rows);
        for (std::uint32_t row = 0; row < input_rows; ++row)
            scalar.push_back(allocate_backend_buffer(
                TensorDescriptor::from(Shape::from({1, shape.columns}), DType::f32), BackendKind::metal));
        MetalCommandContext command("Gemma3 parallel rows retain canonical scalar arithmetic", false, true);
        command.encode_verifier_matmul(device_input, device_weight, batch, shape.rows);
        command.encode_verifier_matmul(device_input, device_weight, ordinary_first, 1U);
        command.encode_target_head_rows(device_input, shape.head_offset, device_weight, head, shape.rows);
        for (std::uint32_t row = 0; row < input_rows; ++row)
            command.encode_target_head_rows(device_input, row, device_weight, scalar[row], 1);
        const auto stats = command.commit_and_wait();
        expect(stats.transient_buffer_allocation_count == 0,
            "parallel row projection allocated transient device storage");
        const auto actual = copy_tensor_from_backend(batch);
        const auto actual_head = copy_tensor_from_backend(head);
        const auto actual_ordinary_first = copy_tensor_from_backend(ordinary_first);
        std::vector<TensorStorage> ordinary;
        ordinary.reserve(scalar.size());
        for (const auto& row : scalar) ordinary.push_back(copy_tensor_from_backend(row));
        for (std::uint32_t row = 0; row < shape.rows; ++row) {
            for (std::uint32_t column = 0; column < shape.columns; ++column) {
                const std::array<std::uint64_t, 2> index {row, column}, scalar_index {0, column};
                const auto value = read_f32(actual, index);
                const auto head_value = read_f32(actual_head, index);
                expect(std::isfinite(value) && std::isfinite(head_value),
                    "Gemma3 native projection must produce finite values for finite inputs");
                expect(std::bit_cast<std::uint32_t>(value)
                        == std::bit_cast<std::uint32_t>(read_f32(ordinary[row], scalar_index)),
                    "Gemma3 parallel projection changed canonical target arithmetic");
                expect(std::bit_cast<std::uint32_t>(head_value)
                        == std::bit_cast<std::uint32_t>(read_f32(ordinary[row + shape.head_offset], scalar_index)),
                    "Gemma3 parallel head changed canonical arithmetic or input-row offset");
                if (row == 0U) {
                    expect(std::bit_cast<std::uint32_t>(value)
                            == std::bit_cast<std::uint32_t>(read_f32(actual_ordinary_first, scalar_index)),
                        "Gemma3 ordinary transformer and batched/head projection arithmetic disagree");
                }
                const auto oracle = read_f32(expected, index);
                expect_close(value, oracle, (dtype == DType::q4_k ? 0.05F : 0.1F) + 1.e-5F * std::fabs(oracle),
                    "Gemma3 parallel projection differs from independent dequantized reference");
                const std::array<std::uint64_t, 2> head_index {row + shape.head_offset, column};
                const auto head_oracle = read_f32(expected, head_index);
                expect_close(head_value, head_oracle,
                    (dtype == DType::q4_k ? 0.05F : 0.1F) + 1.e-5F * std::fabs(head_oracle),
                    "Gemma3 offset head differs from independent dequantized reference");
            }
        }
        for (std::uint32_t column = 0; column < shape.columns; ++column) {
            const std::array<std::uint64_t, 2> guard {shape.rows, column};
            expect(std::bit_cast<std::uint32_t>(read_f32(actual, guard)) == 0x7fc00000U
                    && std::bit_cast<std::uint32_t>(read_f32(actual_head, guard)) == 0x7fc00000U,
                "Gemma3 parallel projection overwrote its explicit row boundary");
        }
    }
}

void test_rejects_explicit_metal_matmul_rows_beyond_input_before_encoding() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    constexpr auto input_rows = std::uint64_t {3U};
    constexpr auto requested_rows = std::uint32_t {4U};
    constexpr auto shared = std::uint64_t {256U};
    constexpr auto columns = std::uint64_t {5U};
    constexpr auto poison = std::uint32_t {0x7fc00000U};
    const auto input_values = tensor_f32_projection_values(input_rows, shared);
    const auto input = tensor_f32_matrix(input_rows, shared, input_values);

    for (const auto dtype : {DType::q4_k, DType::q6_k}) {
        const auto weight = dtype == DType::q4_k
            ? tensor_q4_k_stress_gguf_weight_matrix(shared, columns, requested_rows)
            : tensor_q6_k_stress_gguf_weight_matrix(shared, columns, requested_rows);
        const auto input_buffer = copy_tensor_to_backend(input, BackendKind::metal);
        const auto weight_buffer = copy_tensor_to_backend(weight, BackendKind::metal);
        const auto output = allocate_backend_buffer(
            TensorDescriptor::from(Shape::from({requested_rows, columns}), DType::f32),
            BackendKind::metal
        );
        fill_shared_metal_buffer_u32(output, poison);

        MetalCommandContext command("invalid explicit K-quant matmul row count");
        try {
            command.encode_matmul(
                input_buffer,
                weight_buffer,
                output,
                requested_rows
            );
            throw TestFailure("Metal matmul should reject a row count beyond the input");
        } catch (const std::invalid_argument& error) {
            expect(
                std::string_view(error.what())
                    == "Metal resident execution matmul explicit row count exceeds input rows",
                "Metal matmul excessive-row diagnostic mismatch"
            );
        }

        // Submitting the otherwise-empty command buffer exposes whether the
        // rejected operation created an encoder or dispatched any GPU work.
        const auto stats = command.commit_and_wait();
        expect(stats.compute_encoder_count == 0U, "invalid Metal matmul encoded GPU work");
        expect(stats.compute_dispatch_count == 0U, "invalid Metal matmul dispatched GPU work");
        const auto untouched = copy_tensor_from_backend(output);
        for (std::uint64_t row = 0; row < requested_rows; ++row) {
            for (std::uint64_t column = 0; column < columns; ++column) {
                const std::array<std::uint64_t, 2> index {row, column};
                expect(
                    std::isnan(read_f32(untouched, index)),
                    "invalid Metal matmul modified its output"
                );
            }
        }
    }
}

void test_force_injected_unavailable_metal_device_paths() {
#if !defined(LOCAL_GGUF_ENGINE_TESTING)
    throw TestFailure("Metal unavailable-device fault injection is absent from the test build");
#else
    class ScopedUnavailableMetalDevice final {
    public:
        ScopedUnavailableMetalDevice() {
            set_metal_device_unavailable_for_testing(true);
        }
        ~ScopedUnavailableMetalDevice() {
            set_metal_device_unavailable_for_testing(false);
        }
        ScopedUnavailableMetalDevice(const ScopedUnavailableMetalDevice&) = delete;
        ScopedUnavailableMetalDevice& operator=(const ScopedUnavailableMetalDevice&) = delete;
    };

    const auto availability_before_injection = tensor_backend_is_available(BackendKind::metal);
    {
        const ScopedUnavailableMetalDevice unavailable;
        expect(
            !tensor_backend_is_available(BackendKind::metal),
            "injected unavailable Metal device still reported available"
        );
        expect(
            tensor_backend_unavailable_reason(BackendKind::metal)
                == "Metal backend unavailable: no system default Metal device",
            "injected unavailable Metal reason mismatch"
        );

        const auto valid_descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
        auto invalid_descriptor = valid_descriptor;
        invalid_descriptor.byte_count = std::numeric_limits<std::size_t>::max();
        try {
            (void)allocate_backend_buffer(invalid_descriptor, BackendKind::metal);
            throw TestFailure("malformed Metal descriptor should fail before device lookup");
        } catch (const std::invalid_argument&) {
        }
        try {
            (void)allocate_backend_buffer(valid_descriptor, BackendKind::metal);
            throw TestFailure("valid Metal allocation should report the unavailable device");
        } catch (const BackendError& error) {
            expect(error.code() == BackendErrorCode::unavailable_metal_device,
                "valid Metal allocation changed unavailable-device behavior");
        }

        constexpr auto shared = std::uint64_t {256U};
        constexpr auto columns = std::uint64_t {3U};
        for (const auto rows : {std::uint64_t {3U}, std::uint64_t {8U}}) {
            const auto input_values = tensor_f32_projection_values(rows, shared);
            const auto input = tensor_f32_matrix(rows, shared, input_values);
            for (const auto dtype : {DType::q4_k, DType::q6_k}) {
                const auto weight = dtype == DType::q4_k
                    ? tensor_q4_k_stress_gguf_weight_matrix(shared, columns, rows)
                    : tensor_q6_k_stress_gguf_weight_matrix(shared, columns, rows);
                try {
                    static_cast<void>(matmul_f32_quantized_gguf_weight(
                        input,
                        weight,
                        BackendKind::metal
                    ));
                    throw TestFailure(
                        "forced-unavailable Metal small-batch matmul should fail closed"
                    );
                } catch (const BackendError& error) {
                    expect(
                        error.code() == BackendErrorCode::unavailable_metal_device,
                        "forced-unavailable Metal small-batch error code mismatch"
                    );
                    expect(
                        error.backend() == BackendKind::metal,
                        "forced-unavailable Metal small-batch backend mismatch"
                    );
                }
            }
        }
    }
    expect(
        tensor_backend_is_available(BackendKind::metal) == availability_before_injection,
        "Metal availability was not restored after fault injection"
    );
#endif
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

void test_metal_buffer_views_validate_descriptors_when_available() {
#if !defined(LOCAL_GGUF_ENGINE_TESTING)
    throw TestFailure("Metal buffer view fault helper is absent from the test build");
#else
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto parent = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({8}), DType::f32), BackendKind::metal);
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);
    const auto view = local_gguf_engine::tensor::backends::metal::shared_metal_buffer_view_for_testing(
        parent, descriptor, sizeof(float));
    expect(view.descriptor() == descriptor, "valid Metal subview descriptor mismatch");
    for (const auto invalid : {0U, 1U, 2U, 3U}) {
        auto forged = descriptor;
        switch (invalid) {
            case 0U: ++forged.element_count; break;
            case 1U: forged.byte_count = sizeof(float); break;
            case 2U: forged.shape = Shape::from({4}); break;
            case 3U: forged.dtype = DType::f16; break;
        }
        try {
            (void)local_gguf_engine::tensor::backends::metal::shared_metal_buffer_view_for_testing(
                parent, forged, sizeof(float));
            throw TestFailure("Metal subview accepted an inconsistent descriptor");
        } catch (const std::invalid_argument&) {
        }
    }
#endif
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

local_gguf_engine::runtime::OwnedReadOnlyBytes owned_mapped_test_tensor(const TensorStorage& tensor) {
    const auto page_result = ::sysconf(_SC_PAGESIZE);
    expect(page_result > 0, "mapped test requires a VM page size");
    const auto page = static_cast<std::size_t>(page_result);
    const auto extent = page * 2U;
    void* bytes = ::mmap(nullptr, extent, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    expect(bytes != MAP_FAILED, "mapped test allocation failed");
    std::shared_ptr<const void> owner(bytes, [extent](const void* pointer) {
        static_cast<void>(::munmap(const_cast<void*>(pointer), extent));
    });
    std::memset(bytes, 0xA5, extent);
    // The matrix straddles two pages. Neither registration nor binding may
    // assume that its borrowed GGUF-style bytes begin at the parent's base.
    expect(tensor.bytes().size() <= page, "mapped test payload must fit one page");
    const auto before_boundary = (tensor.bytes().size() / 2U + 3U) / 4U * 4U;
    const auto offset = page - before_boundary;
    std::memcpy(static_cast<std::uint8_t*>(bytes) + offset, tensor.bytes().data(), tensor.bytes().size());
    expect(::mprotect(bytes, extent, PROT_READ) == 0, "mapped test read-only protection failed");
    return {std::move(owner),
        {static_cast<const std::uint8_t*>(bytes), offset + tensor.bytes().size()}, extent};
}

local_gguf_engine::runtime::OwnedReadOnlyBytes owned_mapped_test_matrix() {
    return owned_mapped_test_tensor(tensor_f32_matrix(2U, 2U,
        std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}));
}

local_gguf_engine::tensor::TensorView mapped_test_matrix_view(
    const local_gguf_engine::runtime::OwnedReadOnlyBytes& backing) {
    return local_gguf_engine::tensor::TensorView::from_bytes(
        TensorDescriptor::from(Shape::from({2U, 2U}), DType::f32),
        backing.bytes().last(4U * sizeof(float)));
}

void test_maps_immutable_matrix_offsets_and_rejects_writes_when_available() {
    if (!local_gguf_engine::tensor::backends::metal::supports_immutable_metal_mappings()) return;
    namespace metal = local_gguf_engine::tensor::backends::metal;
    const auto backing = owned_mapped_test_matrix();
    const auto view = mapped_test_matrix_view(backing);
    const auto mapped = metal::map_immutable_tensor_view_to_metal_backend(view, backing);
    expect(metal::metal_backend_buffer_allocation_byte_count(mapped) == backing.accessible_byte_count(),
        "mapped matrix must register exactly its covering pages");
    expect(metal::mapped_immutable_tensor_byte_count(view, backing) == backing.accessible_byte_count(),
        "mapped pre-admission extent must match its real parent");
    const auto copied = copy_tensor_from_backend(mapped);
    expect(std::equal(view.bytes().begin(), view.bytes().end(), copied.bytes().begin()),
        "mapped tensor readback must apply its offset and preserve every byte");
    const auto expected = tensor_f32_matrix(2U, 2U, std::array<float, 4>{2.0F, 4.0F, 6.0F, 8.0F});
    expect_f32_matrix_close(copy_tensor_from_backend(add_f32_to_backend(mapped, mapped)), expected, 0.0F,
        "generic Metal inputs must honor mapped tensor offsets");
    const auto child = metal::shared_metal_buffer_view_for_testing(mapped,
        TensorDescriptor::from(Shape::from({1U, 2U}), DType::f32), 2U * sizeof(float));
    for (const auto* destination : {&mapped, &child}) {
        try {
            fill_shared_metal_buffer_u32(*destination, 0U);
            throw TestFailure("mapped matrix CPU fill must reject writes");
        } catch (const std::invalid_argument&) {}
        MetalCommandContext command("immutable matrix write rejection");
        try {
            command.write_u32(*destination, 0U);
            throw TestFailure("mapped matrix CPU staging must reject writes");
        } catch (const std::invalid_argument&) {}
        try {
            command.encode_softcap(*destination, *destination, 1.0F);
            throw TestFailure("mapped matrix GPU destination must reject writes");
        } catch (const std::invalid_argument&) {}
    }
    const auto unchanged = copy_tensor_from_backend(mapped);
    expect(std::equal(view.bytes().begin(), view.bytes().end(), unchanged.bytes().begin()),
        "rejected writes must leave mapped source bytes intact");
    const auto borrowed = tensor_f32_matrix(2U, 2U, std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F});
    try {
        static_cast<void>(metal::mapped_immutable_tensor_byte_count(borrowed.view(), backing));
        throw TestFailure("borrowed data outside the owned mapping must not become a no-copy tensor");
    } catch (const std::invalid_argument&) {}
    const auto padding = local_gguf_engine::tensor::TensorView::from_bytes(view.descriptor(),
        {backing.bytes().data() + backing.bytes().size(), view.bytes().size()});
    try {
        static_cast<void>(metal::mapped_immutable_tensor_byte_count(padding, backing));
        throw TestFailure("readable VM padding must not count as logical tensor source bytes");
    } catch (const std::invalid_argument&) {}
}

void test_generic_composed_metal_operators_preserve_mapped_inputs_when_available() {
    namespace metal = local_gguf_engine::tensor::backends::metal;
    if (!metal::supports_immutable_metal_mappings()) return;
    const auto input = tensor_f32_matrix(2U, 2U, std::array<float, 4>{0.25F, -0.5F, 0.75F, 0.125F});
    const auto weight = tensor_f16_gguf_matrix(2U, 2U,
        std::array<float, 4>{0.5F, 0.25F, -0.25F, 0.5F});
    const auto input_backing = owned_mapped_test_tensor(input);
    const auto weight_backing = owned_mapped_test_tensor(weight);
    const auto mapped_input = metal::map_immutable_tensor_view_to_metal_backend(
        local_gguf_engine::tensor::TensorView::from_bytes(input.descriptor(),
            input_backing.bytes().last(input.bytes().size())), input_backing);
    const auto mapped_weight = metal::map_immutable_tensor_view_to_metal_backend(
        local_gguf_engine::tensor::TensorView::from_bytes(weight.descriptor(),
            weight_backing.bytes().last(weight.bytes().size())), weight_backing);
    const auto expected_product = matmul_f32_f16_gguf_weight(input, weight, BackendKind::cpu);
    const auto triple = matmul_f32_f16_gguf_weight_triple_to_backend(
        mapped_input, mapped_weight, mapped_weight, mapped_weight);
    for (const auto* result : {&triple.first, &triple.second, &triple.third})
        expect_f32_matrix_close(copy_tensor_from_backend(*result), expected_product, 1.0e-3F,
            "generic triple Metal projection must retain all mapped source offsets");
    const auto expected_gated = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(expected_product), expected_product), weight, BackendKind::cpu);
    const auto gated = gated_silu_projection_f32_f16_gguf_weight_to_backend(
        mapped_input, mapped_weight, mapped_weight, mapped_weight);
    expect_f32_matrix_close(copy_tensor_from_backend(gated), expected_gated, 1.0e-3F,
        "generic gated Metal projections must retain both mapped inputs and weights");
    const auto norm = tensor_f32_vector(std::array<float, 2>{0.25F, -0.5F});
    const auto mapped_norm = metal::shared_metal_buffer_view_for_testing(mapped_input, norm.descriptor(), 0U);
    const auto residual = add_f32(input, expected_product);
    const auto normalized = rms_norm_f32(residual, norm, 0.25F);
    const auto gate = matmul_f32_f16_gguf_weight(normalized, weight, BackendKind::cpu);
    const auto projected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), gate), weight, BackendKind::cpu);
    const auto expected_residual = add_f32(residual, projected);
    const auto actual_residual = residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
        mapped_input, mapped_input, mapped_weight, mapped_norm,
        mapped_weight, mapped_weight, mapped_weight, 0.25F);
    expect_f32_matrix_close(copy_tensor_from_backend(actual_residual), expected_residual, 1.0e-3F,
        "generic residual Metal projections must retain original input/norm/weight offsets");
}

void test_pending_metal_commands_retain_mapped_matrix_backing_when_available() {
    if (!local_gguf_engine::tensor::backends::metal::supports_immutable_metal_mappings()) return;
    namespace metal = local_gguf_engine::tensor::backends::metal;
    std::weak_ptr<const void> lifetime;
    const auto descriptor = TensorDescriptor::from(Shape::from({1U, 2U}), DType::f32);
    const auto output = allocate_backend_buffer(descriptor, BackendKind::metal);
    {
        MetalCommandContext command("mapped owner after all C++ facades are gone");
        {
            const auto backing = owned_mapped_test_matrix();
            lifetime = backing.owner();
            const auto mapped = metal::map_immutable_tensor_view_to_metal_backend(
                mapped_test_matrix_view(backing), backing);
            const auto child = metal::shared_metal_buffer_view_for_testing(mapped,
                descriptor, 2U * sizeof(float));
            command.encode_add(child, child, output);
        }
        expect(!lifetime.expired(), "encoded Metal resource must retain mapping independently of C++ buffers");
        static_cast<void>(command.commit_and_wait());
        const auto expected = tensor_f32_matrix(1U, 2U, std::array<float, 2>{6.0F, 8.0F});
        expect_f32_matrix_close(copy_tensor_from_backend(output), expected, 0.0F,
            "pending mapped command must remain valid after source/view destruction");
    }
    expect(lifetime.expired(), "completed destroyed commands must release the final mapped owner");
}

void test_initializes_final_metal_storage_and_cleans_up_failure_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    namespace metal = local_gguf_engine::tensor::backends::metal;
    const auto source = tensor_f32_matrix(2U, 2U, std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F});
    bool initialized = false;
    const auto buffer = metal::initialize_metal_backend_buffer(source.descriptor(),
        [&](std::span<std::uint8_t> destination) {
            expect(destination.size() == source.bytes().size(), "final initializer extent mismatch");
            std::copy(source.bytes().begin(), source.bytes().end(), destination.begin());
            initialized = true;
        });
    expect(initialized, "final buffer initializer must run before publication");
    const auto copied = copy_tensor_from_backend(buffer);
    expect(std::equal(source.bytes().begin(), source.bytes().end(), copied.bytes().begin()),
        "final initialized storage must preserve source bytes");
    try {
        static_cast<void>(metal::initialize_metal_backend_buffer(source.descriptor(),
            [](std::span<std::uint8_t> destination) {
                expect(metal::initialized_metal_buffer_is_alive_for_testing(),
                    "failed initializer allocation must be observed during the callback");
                destination[0] = 0xA5U;
                throw std::runtime_error("intentional initializer failure");
            }));
        throw TestFailure("failed initializer must propagate its error");
    } catch (const std::runtime_error& error) {
        expect(std::string_view(error.what()) == "intentional initializer failure", "initializer error was changed");
    }
    expect(!metal::initialized_metal_buffer_is_alive_for_testing(),
        "failed final initializer must release its allocation");
}

void test_metal_construction_and_mapping_report_unavailable_device() {
    namespace metal = local_gguf_engine::tensor::backends::metal;
    struct RestoreDevice {
        ~RestoreDevice() { set_metal_device_unavailable_for_testing(false); }
    } restore;
    const auto backing = owned_mapped_test_matrix();
    const auto view = mapped_test_matrix_view(backing);
    set_metal_device_unavailable_for_testing(true);
    expect(!metal::supports_immutable_metal_mappings(), "unavailable Metal device must not admit mappings");
    bool callback_ran = false;
    for (const auto operation : {0U, 1U}) {
        try {
            if (operation == 0U) {
                static_cast<void>(metal::initialize_metal_backend_buffer(view.descriptor(),
                    [&](std::span<std::uint8_t>) { callback_ran = true; }));
            } else {
                static_cast<void>(metal::map_immutable_tensor_view_to_metal_backend(view, backing));
            }
            throw TestFailure("new Metal factories must reject unavailable devices");
        } catch (const BackendError& error) {
            expect(error.code() == BackendErrorCode::unavailable_metal_device,
                "new Metal factory unavailable-device error mismatch");
        }
    }
    expect(!callback_ran, "failed device admission must happen before final-buffer initialization");
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
    {"generic composed Metal operators preserve mapped input offsets when available",
        test_generic_composed_metal_operators_preserve_mapped_inputs_when_available},
    {"Metal construction and mapping report unavailable devices",
        test_metal_construction_and_mapping_report_unavailable_device},
    {"mapped immutable matrices preserve offsets and reject writes when available",
        test_maps_immutable_matrix_offsets_and_rejects_writes_when_available},
    {"pending Metal commands retain mapped backing after facade destruction when available",
        test_pending_metal_commands_retain_mapped_matrix_backing_when_available},
    {"final Metal initialization is byte exact and cleans up failure when available",
        test_initializes_final_metal_storage_and_cleans_up_failure_when_available},
    {"Metal buffer views validate descriptors when available", test_metal_buffer_views_validate_descriptors_when_available},
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
        "Gemma3 parallel projections preserve scalar bits for native shapes and tails",
        test_gemma3_parallel_projection_preserves_scalar_bits_for_native_shapes,
    },
    {
        "rejects explicit Metal matmul rows beyond input before encoding",
        test_rejects_explicit_metal_matmul_rows_beyond_input_before_encoding,
    },
    {
        "force-injected unavailable Metal validates allocations and rejects K-quant paths",
        test_force_injected_unavailable_metal_device_paths,
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
