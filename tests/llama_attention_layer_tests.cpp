#include "llama_attention_layer_tests.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
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

using local_gguf_engine::engine::create_llama_kv_cache;
using local_gguf_engine::engine::LlamaAttentionLayerWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::run_llama_attention_layer_f32;
using local_gguf_engine::engine::run_llama_attention_layer_with_kv_cache_f32;
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

LlamaModelConfig tiny_attention_layer_config() {
    return LlamaModelConfig {
        .context_length = 16,
        .embedding_length = 2,
        .block_count = 1,
        .feed_forward_length = 8,
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

void test_runs_llama_attention_layer_with_f16_weights() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    const auto output = run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);

    expect(output.descriptor().shape == Shape::from({1, 2}), "Llama attention layer output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama attention layer output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        3.75F,
        0.000001F,
        "Llama attention layer output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        5.0F,
        0.000001F,
        "Llama attention layer output value 1 mismatch"
    );
}

void test_runs_llama_attention_layer_with_causal_mask() {
    const std::array<float, 4> input_values {
        3.0F, 4.0F,
        30.0F, 40.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> zero_values {
        0.0F, 0.0F,
        0.0F, 0.0F,
    };
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(2, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    const auto output = run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 1);

    expect(output.descriptor().shape == Shape::from({2, 2}), "Llama causal attention output shape mismatch");
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const auto row1_inverse_rms = 1.0 / std::sqrt(1250.0 + 3.5);
    const auto row1_norm0 = 30.0 * row1_inverse_rms;
    const auto row1_norm1 = 40.0 * row1_inverse_rms;

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        3.75F,
        0.000001F,
        "Llama causal attention row 0 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col1)),
        5.0F,
        0.000001F,
        "Llama causal attention row 0 value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        static_cast<float>(30.0 + (0.75 + row1_norm0) / 2.0),
        0.000001F,
        "Llama causal attention row 1 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        static_cast<float>(40.0 + (1.0 + row1_norm1) / 2.0),
        0.000001F,
        "Llama causal attention row 1 value 1 mismatch"
    );
}

void test_runs_llama_attention_layer_with_kv_cache_matches_no_cache() {
    const std::array<float, 4> full_input_values {
        3.0F, 4.0F,
        30.0F, 40.0F,
    };
    const std::array<float, 2> first_input_values {3.0F, 4.0F};
    const std::array<float, 2> second_input_values {30.0F, 40.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> zero_values {
        0.0F, 0.0F,
        0.0F, 0.0F,
    };
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const auto full_input = tensor_f32_matrix(2, 2, std::span<const float>(full_input_values));
    const auto first_input = tensor_f32_matrix(1, 2, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 2, std::span<const float>(second_input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());

    const auto no_cache = run_llama_attention_layer_f32(full_input, weights, tiny_attention_layer_config(), 1);
    (void)run_llama_attention_layer_with_kv_cache_f32(first_input, weights, tiny_attention_layer_config(), 0, cache.layers[0]);
    const auto cached = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        weights,
        tiny_attention_layer_config(),
        1,
        cache.layers[0]
    );

    expect(cache.layers[0].filled_token_count == 2, "Llama attention KV cache fill mismatch");
    expect(cached.descriptor().shape == Shape::from({1, 2}), "Llama cached attention output shape mismatch");
    const std::array<std::uint64_t, 2> no_cache_row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> no_cache_row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> cached_row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> cached_row0_col1 {0, 1};
    expect_close(
        read_f32(cached, std::span<const std::uint64_t>(cached_row0_col0)),
        read_f32(no_cache, std::span<const std::uint64_t>(no_cache_row1_col0)),
        0.000001F,
        "Llama cached attention value 0 should match no-cache row 1"
    );
    expect_close(
        read_f32(cached, std::span<const std::uint64_t>(cached_row0_col1)),
        read_f32(no_cache, std::span<const std::uint64_t>(no_cache_row1_col1)),
        0.000001F,
        "Llama cached attention value 1 should match no-cache row 1"
    );
}

void test_runs_grouped_llama_attention_layer_with_kv_cache_matches_no_cache() {
    const std::array<float, 12> full_input_values {
        1.0F, 2.0F, 3.0F, 4.0F,
        -2.0F, 1.0F, 0.5F, 3.0F,
        4.0F, -1.0F, 2.0F, 0.0F,
    };
    const std::array<float, 8> prefill_input_values {
        1.0F, 2.0F, 3.0F, 4.0F,
        -2.0F, 1.0F, 0.5F, 3.0F,
    };
    const std::array<float, 4> decode_input_values {
        4.0F, -1.0F, 2.0F, 0.0F,
    };
    const std::array<float, 4> norm_values {1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 32> query_weight_values {};
    std::array<float, 16> key_weight_values {};
    std::array<float, 16> value_weight_values {};
    std::array<float, 32> output_weight_values {};

    for (std::uint64_t row = 0; row < 4; ++row) {
        for (std::uint64_t column = 0; column < 8; ++column) {
            const auto pattern = static_cast<int>((row + 1) * (column + 2)) % 7 - 3;
            query_weight_values[static_cast<std::size_t>(row * 8 + column)] = static_cast<float>(pattern) * 0.125F;
        }
    }

    for (std::uint64_t row = 0; row < 4; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            const auto key_pattern = static_cast<int>((row + 2) * (column + 1)) % 5 - 2;
            const auto value_pattern = static_cast<int>((row + 3) * (column + 2)) % 7 - 3;
            key_weight_values[static_cast<std::size_t>(row * 4 + column)] =
                static_cast<float>(key_pattern) * 0.25F;
            value_weight_values[static_cast<std::size_t>(row * 4 + column)] =
                static_cast<float>(value_pattern) * 0.125F;
        }
    }

    for (std::uint64_t row = 0; row < 8; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            const auto pattern = static_cast<int>((row + 1) + (column * 2)) % 7 - 3;
            output_weight_values[static_cast<std::size_t>(row * 4 + column)] =
                static_cast<float>(pattern) * 0.125F;
        }
    }

    const auto full_input = tensor_f32_matrix(3, 4, std::span<const float>(full_input_values));
    const auto prefill_input = tensor_f32_matrix(2, 4, std::span<const float>(prefill_input_values));
    const auto decode_input = tensor_f32_matrix(1, 4, std::span<const float>(decode_input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(4, 8, std::span<const float>(query_weight_values));
    auto key_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(key_weight_values));
    auto value_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(value_weight_values));
    auto output_weight = tensor_f16_gguf_matrix(8, 4, std::span<const float>(output_weight_values));
    auto config = tiny_attention_layer_config();
    config.embedding_length = 4;
    config.attention_head_count = 4;
    config.attention_head_count_kv = 2;
    config.attention_key_length = 2;
    config.attention_value_length = 2;
    config.rope_freq_base = 2.0F;
    config.rms_norm_epsilon = 0.25F;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(config);

    const auto no_cache = run_llama_attention_layer_f32(full_input, weights, config, 2);
    (void)run_llama_attention_layer_with_kv_cache_f32(prefill_input, weights, config, 1, cache.layers[0]);
    const auto cached = run_llama_attention_layer_with_kv_cache_f32(
        decode_input,
        weights,
        config,
        2,
        cache.layers[0]
    );

    expect(cache.layers[0].filled_token_count == 3, "grouped Llama attention KV cache fill mismatch");
    expect(cached.descriptor().shape == Shape::from({1, 4}), "grouped Llama cached attention output shape mismatch");
    for (std::uint64_t column = 0; column < 4; ++column) {
        const std::array<std::uint64_t, 2> no_cache_index {2, column};
        const std::array<std::uint64_t, 2> cached_index {0, column};
        expect_close(
            read_f32(cached, std::span<const std::uint64_t>(cached_index)),
            read_f32(no_cache, std::span<const std::uint64_t>(no_cache_index)),
            0.00001F,
            "grouped Llama cached attention should match no-cache final row"
        );
    }
}

void test_runs_llama_attention_layer_with_kv_cache_appends_rotated_key_heads() {
    const std::array<float, 8> input_values {
        1.0F, 0.0F, 0.0F, 1.0F,
        0.0F, 1.0F, 1.0F, 0.0F,
    };
    const std::array<float, 4> norm_values {1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 16> identity_values {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    const auto input = tensor_f32_matrix(2, 4, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.embedding_length = 4;
    config.attention_head_count = 2;
    config.attention_head_count_kv = 2;
    config.attention_key_length = 2;
    config.attention_value_length = 2;
    config.rope_freq_base = 1.0F;
    config.rms_norm_epsilon = 0.5F;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(config);

    (void)run_llama_attention_layer_with_kv_cache_f32(input, weights, config, 1, cache.layers[0]);

    const auto sine = static_cast<float>(std::sin(1.0));
    const auto cosine = static_cast<float>(std::cos(1.0));
    const std::array<float, 8> expected_key_values {
        1.0F, 0.0F, 0.0F, 1.0F,
        -sine, cosine, cosine, sine,
    };
    const std::array<float, 4> expected_value_row1 {
        0.0F, 1.0F, 1.0F, 0.0F,
    };

    expect(cache.layers[0].filled_token_count == 2, "Llama cached attention rotated key fill mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(cache.layers[0].key, std::span<const std::uint64_t>(index)),
                expected_key_values[static_cast<std::size_t>(row * 4 + column)],
                0.000001F,
                "Llama cached rotated key value mismatch"
            );
        }
    }

    for (std::uint64_t column = 0; column < 4; ++column) {
        const std::array<std::uint64_t, 2> index {1, column};
        expect_close(
            read_f32(cache.layers[0].value, std::span<const std::uint64_t>(index)),
            expected_value_row1[static_cast<std::size_t>(column)],
            0.000001F,
            "Llama cached appended value mismatch"
        );
    }
}

void test_rejects_llama_attention_layer_with_kv_cache_invalid_rope_dimension_count() {
    const std::array<float, 2> input_values {1.0F, 0.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.rope_dimension_count = 1;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(config);

    try {
        (void)run_llama_attention_layer_with_kv_cache_f32(input, weights, config, 0, cache.layers[0]);
        throw TestFailure("cached Llama attention should reject invalid RoPE dimension count");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_shape_dimension,
            "cached Llama attention RoPE dimension error mismatch"
        );
        expect(error.expected_value() == 2, "cached Llama attention expected RoPE dimension mismatch");
        expect(error.actual_value() == 1, "cached Llama attention actual RoPE dimension mismatch");
        expect(cache.layers[0].filled_token_count == 0, "cached Llama attention should not append invalid RoPE rows");
    }
}

void test_runs_llama_attention_layer_with_rowwise_rope_positions() {
    const std::array<float, 4> input_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(2, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.rope_freq_base = 1.0F;
    config.rms_norm_epsilon = 0.5F;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    const auto output = run_llama_attention_layer_f32(input, weights, config, 1);

    const auto score_scale = 1.0 / std::sqrt(2.0);
    const auto score0 = -std::sin(1.0) * score_scale;
    const auto score1 = score_scale;
    const auto score_max = score0 > score1 ? score0 : score1;
    const auto weight0_numerator = std::exp(score0 - score_max);
    const auto weight1_numerator = std::exp(score1 - score_max);
    const auto weight_sum = weight0_numerator + weight1_numerator;
    const auto row1_weight0 = static_cast<float>(weight0_numerator / weight_sum);
    const auto row1_weight1 = static_cast<float>(weight1_numerator / weight_sum);
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        2.0F,
        0.000001F,
        "Llama rowwise RoPE row 0 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col1)),
        0.0F,
        0.000001F,
        "Llama rowwise RoPE row 0 value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        row1_weight0,
        0.000001F,
        "Llama rowwise RoPE row 1 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        1.0F + row1_weight1,
        0.000001F,
        "Llama rowwise RoPE row 1 value 1 mismatch"
    );
}

void test_rejects_llama_attention_layer_position_before_input_window() {
    const std::array<float, 4> input_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(2, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);
        throw TestFailure("Llama attention layer should reject a final position before the input window");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama attention position must cover input rows",
            "Llama attention layer input-window position error mismatch"
        );
    }
}

void test_rejects_llama_attention_layer_projection_shape_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 2> narrow_values {1.0F, 0.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 1, std::span<const float>(narrow_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);
        throw TestFailure("Llama attention layer should reject query projection shape mismatch");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_shape_mismatch,
            "Llama attention layer projection shape error mismatch"
        );
        expect(error.expected_value() == 2, "Llama attention layer expected query width mismatch");
        expect(error.actual_value() == 1, "Llama attention layer actual query width mismatch");
    }
}

void test_rejects_llama_attention_layer_projection_dtype_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f32_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);
        throw TestFailure("Llama attention layer should reject F32 query projection weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "Llama attention layer projection dtype error mismatch"
        );
        expect(error.expected_value() == 1, "Llama attention layer expected projection dtype mismatch");
        expect(error.actual_value() == 0, "Llama attention layer actual projection dtype mismatch");
    }
}

void test_rejects_llama_attention_layer_invalid_head_layout() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.attention_head_count = 3;
    config.attention_head_count_kv = 2;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, config, 0);
        throw TestFailure("Llama attention layer should reject non-divisible head layout");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama attention head count must be divisible by KV head count",
            "Llama attention layer invalid head layout error mismatch"
        );
    }
}

const TestCase kLlamaAttentionLayerPrimaryTests[] = {
    {"runs Llama attention layer with F16 weights", test_runs_llama_attention_layer_with_f16_weights},
};

const TestCase kLlamaAttentionLayerValidationTests[] = {
    {"runs Llama attention layer with causal mask", test_runs_llama_attention_layer_with_causal_mask},
    {
        "runs Llama attention layer with KV cache matches no-cache",
        test_runs_llama_attention_layer_with_kv_cache_matches_no_cache,
    },
    {
        "runs grouped Llama attention layer with KV cache matches no-cache",
        test_runs_grouped_llama_attention_layer_with_kv_cache_matches_no_cache,
    },
    {
        "runs Llama attention layer with KV cache appends rotated key heads",
        test_runs_llama_attention_layer_with_kv_cache_appends_rotated_key_heads,
    },
    {
        "rejects Llama attention layer with KV cache invalid RoPE dimension count",
        test_rejects_llama_attention_layer_with_kv_cache_invalid_rope_dimension_count,
    },
    {
        "runs Llama attention layer with rowwise RoPE positions",
        test_runs_llama_attention_layer_with_rowwise_rope_positions,
    },
    {
        "rejects Llama attention layer position before input window",
        test_rejects_llama_attention_layer_position_before_input_window,
    },
    {
        "rejects Llama attention layer projection shape mismatch",
        test_rejects_llama_attention_layer_projection_shape_mismatch,
    },
    {
        "rejects Llama attention layer projection dtype mismatch",
        test_rejects_llama_attention_layer_projection_dtype_mismatch,
    },
    {
        "rejects Llama attention layer invalid head layout",
        test_rejects_llama_attention_layer_invalid_head_layout,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_attention_layer_primary_test_cases() {
    return kLlamaAttentionLayerPrimaryTests;
}

std::span<const TestCase> llama_attention_layer_validation_test_cases() {
    return kLlamaAttentionLayerValidationTests;
}

}  // namespace local_gguf_engine::tests
