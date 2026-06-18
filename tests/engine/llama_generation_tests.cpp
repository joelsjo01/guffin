#include "llama_generation_tests.hpp"

#include "llama_test_support.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::bind_llama_model_weights;
using local_gguf_engine::engine::copy_llama_model_logits_weights_to_backend;
using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::generate_llama_argmax_tokens_f32;
using local_gguf_engine::engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::generate_llama_sampled_tokens_f32;
using local_gguf_engine::engine::generate_llama_sampled_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::LlamaBlockBackendWeights;
using local_gguf_engine::engine::LlamaBlockWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::LlamaModelLogitsBackendWeights;
using local_gguf_engine::engine::LlamaModelWeights;
using local_gguf_engine::engine::run_llama_model_next_token_argmax_f32;
using local_gguf_engine::engine::run_llama_model_next_token_sampled_f32;
using local_gguf_engine::engine::SamplingConfig;
using local_gguf_engine::engine::SamplingState;
using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::add_tiny_llama_block_tensors;
using local_gguf_engine::tests::engine_tensor;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::tiny_bound_llama_model;
using local_gguf_engine::tests::tiny_logits_config;

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

const TestCase llama_generation_tests[] {
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

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_generation_test_cases() {
    return std::span<const TestCase>(llama_generation_tests);
}

}  // namespace local_gguf_engine::tests
