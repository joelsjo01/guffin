#include "tensor_backend_tests.hpp"

#include "tensor_test_support.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::copy_tensor_from_backend;
using local_gguf_engine::tensor::copy_tensor_to_backend;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::tensor_backend_unavailable_reason;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::llama_cached_attention_f32_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_triple_to_backend;
using local_gguf_engine::tensor::ops::multiply_f32;
using local_gguf_engine::tensor::ops::residual_gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::silu_f32;
using local_gguf_engine::tensor::ops::silu_multiply_f32_to_backend;
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

const TestCase tensor_backend_tests[] {
    {
        "multiplies F32 by F16 GGUF weight matrix on CPU backend",
        test_multiplies_f32_by_f16_gguf_weight_matrix_on_cpu_backend,
    },
    {"copies tensor to and from CPU backend buffer", test_copies_tensor_to_and_from_cpu_backend_buffer},
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
    {"runs Llama cached attention to CPU backend buffer", test_runs_llama_cached_attention_to_cpu_backend_buffer},
    {
        "matches Llama cached attention to Metal backend buffer when available",
        test_matches_llama_cached_attention_to_metal_backend_buffer_when_available,
    },
    {
        "matches one-token grouped Llama cached attention to Metal backend buffer when available",
        test_matches_llama_one_token_grouped_cached_attention_to_metal_backend_buffer_when_available,
    },
    {
        "reports unavailable Metal backend for GGUF matmul when missing",
        test_reports_unavailable_metal_backend_for_gguf_matmul_when_missing,
    },
    {
        "reports unavailable Metal backend buffer copy when missing",
        test_reports_unavailable_metal_backend_buffer_copy_when_missing,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> tensor_backend_test_cases() {
    return std::span<const TestCase>(tensor_backend_tests);
}

}  // namespace local_gguf_engine::tests
