#include "llama_feed_forward_layer_tests.hpp"
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
#include <stdexcept>
#include <string_view>

namespace {

using local_gguf_engine::engine::LlamaFeedForwardLayerWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::run_llama_feed_forward_layer_f32;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;

LlamaModelConfig tiny_feed_forward_layer_config() {
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

void test_runs_llama_feed_forward_layer_with_f16_weights() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    const auto output = run_llama_feed_forward_layer_f32(input, weights, tiny_feed_forward_layer_config());
    const auto first_norm = 0.75;
    const auto second_norm = 1.0;
    const auto first_expected = 3.0 + first_norm * (first_norm / (1.0 + std::exp(-first_norm)));
    const auto second_expected = 4.0 + second_norm * (second_norm / (1.0 + std::exp(-second_norm)));

    expect(output.descriptor().shape == Shape::from({1, 2}), "Llama FFN layer output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama FFN layer output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(first_expected),
        0.000001F,
        "Llama FFN layer output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(second_expected),
        0.000001F,
        "Llama FFN layer output value 1 mismatch"
    );
}

void test_rejects_llama_feed_forward_layer_projection_shape_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 2> narrow_values {1.0F, 0.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 1, std::span<const float>(narrow_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    try {
        (void)run_llama_feed_forward_layer_f32(input, weights, tiny_feed_forward_layer_config());
        throw TestFailure("Llama FFN layer should reject gate projection shape mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Llama FFN shape error mismatch");
        expect(error.expected_value() == 2, "Llama FFN expected gate width mismatch");
        expect(error.actual_value() == 1, "Llama FFN actual gate width mismatch");
    }
}

void test_rejects_llama_feed_forward_layer_projection_dtype_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f32_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    try {
        (void)run_llama_feed_forward_layer_f32(input, weights, tiny_feed_forward_layer_config());
        throw TestFailure("Llama FFN layer should reject F32 gate projection weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "Llama FFN projection dtype error mismatch"
        );
        expect(error.expected_value() == 1, "Llama FFN expected projection dtype mismatch");
        expect(error.actual_value() == 0, "Llama FFN actual projection dtype mismatch");
    }
}

void test_rejects_llama_feed_forward_layer_invalid_config() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_feed_forward_layer_config();
    config.feed_forward_length = 0;
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    try {
        (void)run_llama_feed_forward_layer_f32(input, weights, config);
        throw TestFailure("Llama FFN layer should reject zero feed-forward length");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama layer config dimensions must be positive",
            "Llama FFN invalid config error mismatch"
        );
    }
}

const TestCase kLlamaFeedForwardLayerPrimaryTests[] = {
    {"runs Llama FFN layer with F16 weights", test_runs_llama_feed_forward_layer_with_f16_weights},
};

const TestCase kLlamaFeedForwardLayerValidationTests[] = {
    {
        "rejects Llama FFN layer projection shape mismatch",
        test_rejects_llama_feed_forward_layer_projection_shape_mismatch,
    },
    {
        "rejects Llama FFN layer projection dtype mismatch",
        test_rejects_llama_feed_forward_layer_projection_dtype_mismatch,
    },
    {"rejects Llama FFN layer invalid config", test_rejects_llama_feed_forward_layer_invalid_config},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_feed_forward_layer_primary_test_cases() {
    return kLlamaFeedForwardLayerPrimaryTests;
}

std::span<const TestCase> llama_feed_forward_layer_validation_test_cases() {
    return kLlamaFeedForwardLayerValidationTests;
}

}  // namespace local_gguf_engine::tests
