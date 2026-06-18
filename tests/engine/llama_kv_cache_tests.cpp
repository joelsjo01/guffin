#include "llama_kv_cache_tests.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {

using local_gguf_engine::engine::append_llama_kv_cache_layer_f32;
using local_gguf_engine::engine::create_llama_kv_cache;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::llama_kv_cache_layer_key_prefix_f32;
using local_gguf_engine::engine::llama_kv_cache_layer_value_prefix_f32;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::tensor_f32_matrix;
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

void test_creates_llama_kv_cache() {
    auto config = tiny_attention_layer_config();
    config.block_count = 2;

    const auto cache = create_llama_kv_cache(config);

    expect(cache.layers.size() == 2, "Llama KV cache layer count mismatch");
    expect(cache.context_length == 16, "Llama KV cache context length mismatch");
    expect(cache.key_width == 2, "Llama KV cache key width mismatch");
    expect(cache.value_width == 2, "Llama KV cache value width mismatch");
    expect(cache.layers[0].filled_token_count == 0, "Llama KV cache initial fill mismatch");
    expect(cache.layers[0].key.descriptor().shape == Shape::from({16, 2}), "Llama KV cache key shape mismatch");
    expect(cache.layers[0].key.descriptor().dtype == DType::f32, "Llama KV cache key dtype mismatch");
    expect(cache.layers[0].value.descriptor().shape == Shape::from({16, 2}), "Llama KV cache value shape mismatch");
    expect(cache.layers[0].value.descriptor().dtype == DType::f32, "Llama KV cache value dtype mismatch");
}

void test_appends_and_reads_llama_kv_cache_layer() {
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());
    const std::array<float, 4> key_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> value_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
    };
    const auto key_rows = tensor_f32_matrix(2, 2, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(2, 2, std::span<const float>(value_values));

    append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 0);
    const auto key_prefix = llama_kv_cache_layer_key_prefix_f32(cache.layers[0]);
    const auto value_prefix = llama_kv_cache_layer_value_prefix_f32(cache.layers[0]);

    expect(cache.layers[0].filled_token_count == 2, "Llama KV cache filled token count mismatch");
    expect(key_prefix.descriptor().shape == Shape::from({2, 2}), "Llama KV cache key prefix shape mismatch");
    expect(value_prefix.descriptor().shape == Shape::from({2, 2}), "Llama KV cache value prefix shape mismatch");
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    expect(read_f32(key_prefix, std::span<const std::uint64_t>(row1_col0)) == 3.0F, "Llama KV key prefix value mismatch");
    expect(read_f32(value_prefix, std::span<const std::uint64_t>(row1_col1)) == 40.0F, "Llama KV value prefix value mismatch");
}

void test_rejects_llama_kv_cache_append_position_mismatch() {
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());
    const std::array<float, 2> key_values {1.0F, 2.0F};
    const std::array<float, 2> value_values {10.0F, 20.0F};
    const auto key_rows = tensor_f32_matrix(1, 2, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(1, 2, std::span<const float>(value_values));

    try {
        append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 1);
        throw TestFailure("Llama KV cache should reject non-contiguous append position");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama KV cache append position must match filled token count",
            "Llama KV cache position mismatch error text mismatch"
        );
    }
}

void test_rejects_llama_kv_cache_context_overflow() {
    auto config = tiny_attention_layer_config();
    config.context_length = 1;
    auto cache = create_llama_kv_cache(config);
    const std::array<float, 4> key_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> value_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
    };
    const auto key_rows = tensor_f32_matrix(2, 2, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(2, 2, std::span<const float>(value_values));

    try {
        append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 0);
        throw TestFailure("Llama KV cache should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama KV cache append would exceed context length",
            "Llama KV cache overflow error text mismatch"
        );
    }
}

void test_rejects_llama_kv_cache_key_shape_mismatch() {
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());
    const std::array<float, 1> key_values {1.0F};
    const std::array<float, 2> value_values {10.0F, 20.0F};
    const auto key_rows = tensor_f32_matrix(1, 1, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(1, 2, std::span<const float>(value_values));

    try {
        append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 0);
        throw TestFailure("Llama KV cache should reject key width mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Llama KV cache key shape error mismatch");
        expect(error.expected_value() == 2, "Llama KV cache expected key width mismatch");
        expect(error.actual_value() == 1, "Llama KV cache actual key width mismatch");
    }
}

const TestCase kLlamaKVCacheTests[] = {
    {"creates Llama KV cache", test_creates_llama_kv_cache},
    {"appends and reads Llama KV cache layer", test_appends_and_reads_llama_kv_cache_layer},
    {
        "rejects Llama KV cache append position mismatch",
        test_rejects_llama_kv_cache_append_position_mismatch,
    },
    {"rejects Llama KV cache context overflow", test_rejects_llama_kv_cache_context_overflow},
    {"rejects Llama KV cache key shape mismatch", test_rejects_llama_kv_cache_key_shape_mismatch},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_kv_cache_test_cases() {
    return kLlamaKVCacheTests;
}

}  // namespace local_gguf_engine::tests
