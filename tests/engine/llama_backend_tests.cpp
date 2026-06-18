#include "llama_backend_tests.hpp"

#include "llama_test_support.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
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
using local_gguf_engine::engine::create_llama_backend_kv_cache;
using local_gguf_engine::engine::create_llama_kv_cache;
using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::LlamaBlockBackendWeights;
using local_gguf_engine::engine::LlamaBlockWeights;
using local_gguf_engine::engine::LlamaKVCache;
using local_gguf_engine::engine::LlamaModelAttentionBackendWeights;
using local_gguf_engine::engine::LlamaModelBlockBackendWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::LlamaModelFeedForwardBackendWeights;
using local_gguf_engine::engine::LlamaModelLogitsBackendWeights;
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
using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::add_tiny_llama_block_tensors;
using local_gguf_engine::tests::add_tiny_llama_top_level_tensors;
using local_gguf_engine::tests::engine_tensor;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::llama_block_tensor_name;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::tiny_bound_llama_model;
using local_gguf_engine::tests::tiny_feed_forward_layer_config;

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

const TestCase llama_backend_attention_tests[] {
    {
        "runs Llama attention layer with CPU backend weight buffers",
        test_runs_llama_attention_layer_with_cpu_backend_weight_buffers,
    },
    {
        "runs Llama attention layer with KV cache and CPU backend weight buffers",
        test_runs_llama_attention_layer_with_kv_cache_and_cpu_backend_weight_buffers,
    },
};

const TestCase llama_backend_feed_forward_tests[] {
    {
        "runs Llama FFN layer with CPU backend weight buffers",
        test_runs_llama_feed_forward_layer_with_cpu_backend_weight_buffers,
    },
};

const TestCase llama_backend_block_tests[] {
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

const TestCase llama_backend_model_tests[] {
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

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_backend_attention_test_cases() {
    return std::span<const TestCase>(llama_backend_attention_tests);
}

std::span<const TestCase> llama_backend_feed_forward_test_cases() {
    return std::span<const TestCase>(llama_backend_feed_forward_tests);
}

std::span<const TestCase> llama_backend_block_test_cases() {
    return std::span<const TestCase>(llama_backend_block_tests);
}

std::span<const TestCase> llama_backend_model_test_cases() {
    return std::span<const TestCase>(llama_backend_model_tests);
}

}  // namespace local_gguf_engine::tests
