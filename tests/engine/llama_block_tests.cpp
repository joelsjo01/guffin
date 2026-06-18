#include "llama_block_tests.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <sstream>
#include <string_view>

namespace {

using local_gguf_engine::engine::LlamaAttentionLayerWeights;
using local_gguf_engine::engine::LlamaBlockWeights;
using local_gguf_engine::engine::LlamaFeedForwardLayerWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::run_llama_block_f32;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;

LlamaModelConfig tiny_block_config() {
    return LlamaModelConfig {
        .context_length = 16,
        .embedding_length = 2,
        .block_count = 1,
        .feed_forward_length = 2,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .attention_key_length = 2,
        .attention_value_length = 2,
        .rope_dimension_count = 2,
        .vocab_size = 8,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 3.5F,
    };
}

void test_runs_llama_block_with_f16_weights() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaBlockWeights weights {
        .attention = LlamaAttentionLayerWeights {
            .attention_norm_weight = attention_norm_weight,
            .query_weight = query_weight,
            .key_weight = key_weight,
            .value_weight = value_weight,
            .output_weight = output_weight,
        },
        .feed_forward = LlamaFeedForwardLayerWeights {
            .feed_forward_norm_weight = feed_forward_norm_weight,
            .gate_weight = gate_weight,
            .up_weight = up_weight,
            .down_weight = down_weight,
        },
    };

    const auto output = run_llama_block_f32(input, weights, tiny_block_config(), 0);
    const auto attention_first = 3.75;
    const auto attention_second = 5.0;
    const auto block_mean_square = ((attention_first * attention_first) + (attention_second * attention_second)) / 2.0;
    const auto inverse_rms = 1.0 / std::sqrt(block_mean_square + 3.5);
    const auto first_norm = attention_first * inverse_rms;
    const auto second_norm = attention_second * inverse_rms;
    const auto first_expected = attention_first + first_norm * (first_norm / (1.0 + std::exp(-first_norm)));
    const auto second_expected = attention_second + second_norm * (second_norm / (1.0 + std::exp(-second_norm)));

    expect(output.descriptor().shape == Shape::from({1, 2}), "Llama block output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama block output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(first_expected),
        0.000001F,
        "Llama block output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(second_expected),
        0.000001F,
        "Llama block output value 1 mismatch"
    );
}

void test_rejects_llama_block_attention_projection_shape_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 2> narrow_values {1.0F, 0.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 1, std::span<const float>(narrow_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaBlockWeights weights {
        .attention = LlamaAttentionLayerWeights {
            .attention_norm_weight = attention_norm_weight,
            .query_weight = query_weight,
            .key_weight = key_weight,
            .value_weight = value_weight,
            .output_weight = output_weight,
        },
        .feed_forward = LlamaFeedForwardLayerWeights {
            .feed_forward_norm_weight = feed_forward_norm_weight,
            .gate_weight = gate_weight,
            .up_weight = up_weight,
            .down_weight = down_weight,
        },
    };

    try {
        (void)run_llama_block_f32(input, weights, tiny_block_config(), 0);
        throw TestFailure("Llama block should reject attention projection shape mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Llama block shape error mismatch");
        expect(error.expected_value() == 2, "Llama block expected query width mismatch");
        expect(error.actual_value() == 1, "Llama block actual query width mismatch");
    }
}

const TestCase kLlamaBlockPrimaryTests[] = {
    {"runs Llama block with F16 weights", test_runs_llama_block_with_f16_weights},
};

const TestCase kLlamaBlockValidationTests[] = {
    {
        "rejects Llama block attention projection shape mismatch",
        test_rejects_llama_block_attention_projection_shape_mismatch,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_block_primary_test_cases() {
    return kLlamaBlockPrimaryTests;
}

std::span<const TestCase> llama_block_validation_test_cases() {
    return kLlamaBlockValidationTests;
}

}  // namespace local_gguf_engine::tests
