#include "llama_model_tests.hpp"

#include "llama_test_support.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::bind_llama_model_weights;
using local_gguf_engine::engine::create_llama_kv_cache;
using local_gguf_engine::engine::embed_llama_token_ids_f32;
using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::LlamaKVCache;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::LlamaModelWeights;
using local_gguf_engine::engine::project_llama_logits_f32;
using local_gguf_engine::engine::run_llama_block_f32;
using local_gguf_engine::engine::run_llama_model_blocks_f32;
using local_gguf_engine::engine::run_llama_model_logits_f32;
using local_gguf_engine::engine::run_llama_model_logits_with_kv_cache_f32;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tests::add_tiny_llama_block_tensors;
using local_gguf_engine::tests::add_tiny_llama_top_level_tensors;
using local_gguf_engine::tests::engine_tensor;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::tiny_bound_llama_model;
using local_gguf_engine::tests::tiny_feed_forward_layer_config;
using local_gguf_engine::tests::tiny_logits_config;

void test_binds_and_runs_llama_model_blocks_from_named_tensors() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 2;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    add_tiny_llama_block_tensors(tensors, 1, 0.5F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);

    expect(weights.blocks.size() == 2, "bound Llama model block count mismatch");
    const auto* block_one_down = model.find_tensor("blk.1.ffn_down.weight");
    expect(block_one_down != nullptr, "bound Llama model test tensor lookup failed");
    expect(
        &weights.blocks[1].feed_forward.down_weight == &block_one_down->storage,
        "bound Llama model should borrow block 1 down weight"
    );

    const std::array<float, 2> input_values {3.0F, 4.0F};
    const auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    const auto output = run_llama_model_blocks_f32(input, weights, model.llama_config(), 0);
    const auto first_block = run_llama_block_f32(input, weights.blocks[0], model.llama_config(), 0);
    const auto expected = run_llama_block_f32(first_block, weights.blocks[1], model.llama_config(), 0);

    expect(output.descriptor().shape == Shape::from({1, 2}), "bound Llama model output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "bound Llama model output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        read_f32(expected, std::span<const std::uint64_t>(index_0)),
        0.000001F,
        "bound Llama model output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        read_f32(expected, std::span<const std::uint64_t>(index_1)),
        0.000001F,
        "bound Llama model output value 1 mismatch"
    );
}

void test_rejects_llama_model_weight_binding_missing_tensor() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 1;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, false);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));

    try {
        (void)bind_llama_model_weights(model);
        throw TestFailure("Llama model weight binding should reject missing named tensors");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_tensor, "Llama model missing tensor error mismatch");
        expect(error.detail() == "blk.0.ffn_down.weight", "Llama model missing tensor detail mismatch");
    }
}

EngineModel tiny_tied_output_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

void test_binds_llama_model_missing_output_weight_to_token_embedding() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_tied_output_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto* token_embedding = model.find_tensor("token_embd.weight");

    expect(token_embedding != nullptr, "tied-output Llama model should have token embedding");
    expect(model.find_tensor("output.weight") == nullptr, "tied-output Llama model should omit output weight");
    expect(
        &weights.output_weight == &token_embedding->storage,
        "tied-output Llama model should reuse token embedding as output weight"
    );

    const std::array<std::uint32_t, 1> prompt_tokens {0};
    const auto tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );

    expect(tokens.size() == 3, "tied-output Llama generation token count mismatch");
    expect(tokens[0] == 0, "tied-output Llama generation prompt token mismatch");
    expect(tokens[1] == 2, "tied-output Llama generation first token mismatch");
    expect(tokens[2] == 2, "tied-output Llama generation second token mismatch");
}

void test_rejects_llama_model_blocks_projection_dtype_mismatch() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 1;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, true, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);

    const std::array<float, 2> input_values {3.0F, 4.0F};
    const auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    try {
        (void)run_llama_model_blocks_f32(input, weights, model.llama_config(), 0);
        throw TestFailure("Llama model block execution should reject F32 query projection weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "Llama model projection dtype error mismatch"
        );
        expect(error.expected_value() == 1, "Llama model expected projection dtype mismatch");
        expect(error.actual_value() == 0, "Llama model actual projection dtype mismatch");
    }
}

void test_embeds_llama_token_ids_from_f16_weight() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };
    const std::array<std::uint32_t, 2> token_ids {2, 0};

    const auto output = embed_llama_token_ids_f32(std::span<const std::uint32_t>(token_ids), weights, tiny_logits_config());

    expect(output.descriptor().shape == Shape::from({2, 2}), "Llama token embedding output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama token embedding output dtype mismatch");
    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 3.0F, "Llama token embedding value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 30.0F, "Llama token embedding value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 1.0F, "Llama token embedding value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 10.0F, "Llama token embedding value 1,1 mismatch");
}

void test_rejects_llama_token_embedding_id_out_of_range() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };
    const std::array<std::uint32_t, 1> token_ids {3};

    try {
        (void)embed_llama_token_ids_f32(std::span<const std::uint32_t>(token_ids), weights, tiny_logits_config());
        throw TestFailure("Llama token embedding should reject token id above vocab");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_index_out_of_range, "Llama token id error mismatch");
        expect(error.expected_value() == 3, "Llama token id expected vocab mismatch");
        expect(error.actual_value() == 3, "Llama token id actual mismatch");
    }
}

void test_projects_llama_logits_with_final_norm() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const std::array<float, 2> hidden_values {3.0F, 4.0F};
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values));
    const auto hidden_states = tensor_f32_matrix(1, 2, std::span<const float>(hidden_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };

    const auto logits = project_llama_logits_f32(hidden_states, weights, tiny_logits_config());

    expect(logits.descriptor().shape == Shape::from({1, 3}), "Llama logits shape mismatch");
    expect(logits.descriptor().dtype == DType::f32, "Llama logits dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    const std::array<std::uint64_t, 2> index_2 {0, 2};
    expect_close(read_f32(logits, std::span<const std::uint64_t>(index_0)), 0.75F, 0.000001F, "Llama logits value 0 mismatch");
    expect_close(read_f32(logits, std::span<const std::uint64_t>(index_1)), 1.0F, 0.000001F, "Llama logits value 1 mismatch");
    expect_close(read_f32(logits, std::span<const std::uint64_t>(index_2)), 4.5F, 0.000001F, "Llama logits value 2 mismatch");
}

void test_runs_llama_model_logits_from_token_ids() {
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
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto logits = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );
    const auto embeddings = embed_llama_token_ids_f32(std::span<const std::uint32_t>(token_ids), weights, model.llama_config());
    const auto hidden_states = run_llama_model_blocks_f32(embeddings, weights, model.llama_config(), 0);
    const auto expected = project_llama_logits_f32(hidden_states, weights, model.llama_config());

    expect(logits.descriptor().shape == Shape::from({1, 3}), "Llama model logits shape mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    const std::array<std::uint64_t, 2> index_2 {0, 2};
    expect_close(
        read_f32(logits, std::span<const std::uint64_t>(index_0)),
        read_f32(expected, std::span<const std::uint64_t>(index_0)),
        0.000001F,
        "Llama model logits value 0 mismatch"
    );
    expect_close(
        read_f32(logits, std::span<const std::uint64_t>(index_1)),
        read_f32(expected, std::span<const std::uint64_t>(index_1)),
        0.000001F,
        "Llama model logits value 1 mismatch"
    );
    expect_close(
        read_f32(logits, std::span<const std::uint64_t>(index_2)),
        read_f32(expected, std::span<const std::uint64_t>(index_2)),
        0.000001F,
        "Llama model logits value 2 mismatch"
    );
}

void test_runs_llama_model_logits_with_kv_cache_matches_no_cache() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.vocab_size = 8;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 2> full_token_ids {0, 1};
    const std::array<std::uint32_t, 1> first_token_id {0};
    const std::array<std::uint32_t, 1> second_token_id {1};
    auto cache = create_llama_kv_cache(model.llama_config());

    const auto no_cache = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(full_token_ids),
        weights,
        model.llama_config(),
        1
    );
    (void)run_llama_model_logits_with_kv_cache_f32(
        std::span<const std::uint32_t>(first_token_id),
        weights,
        model.llama_config(),
        0,
        cache
    );
    const auto cached = run_llama_model_logits_with_kv_cache_f32(
        std::span<const std::uint32_t>(second_token_id),
        weights,
        model.llama_config(),
        1,
        cache
    );

    expect(cache.layers[0].filled_token_count == 2, "Llama model KV cache fill mismatch");
    expect(cached.descriptor().shape == Shape::from({1, 8}), "Llama cached logits shape mismatch");
    for (std::uint64_t token = 0; token < 8; ++token) {
        const std::array<std::uint64_t, 2> no_cache_index {1, token};
        const std::array<std::uint64_t, 2> cached_index {0, token};
        expect_close(
            read_f32(cached, std::span<const std::uint64_t>(cached_index)),
            read_f32(no_cache, std::span<const std::uint64_t>(no_cache_index)),
            0.00001F,
            "Llama cached logits should match no-cache final row"
        );
    }
}

void test_rejects_llama_model_kv_cache_layer_count_mismatch() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.vocab_size = 8;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> token_ids {0};
    LlamaKVCache cache {
        .layers = {},
        .context_length = 0,
        .key_width = 0,
        .value_width = 0,
    };

    try {
        (void)run_llama_model_logits_with_kv_cache_f32(
            std::span<const std::uint32_t>(token_ids),
            weights,
            model.llama_config(),
            0,
            cache
        );
        throw TestFailure("Llama model cached logits should reject cache layer count mismatch");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama KV cache layer count must match model block count",
            "Llama KV cache layer count error text mismatch"
        );
    }
}

void test_rejects_llama_logits_output_weight_dtype_mismatch() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const std::array<float, 2> hidden_values {3.0F, 4.0F};
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f32_matrix(2, 3, std::span<const float>(output_values));
    const auto hidden_states = tensor_f32_matrix(1, 2, std::span<const float>(hidden_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };

    try {
        (void)project_llama_logits_f32(hidden_states, weights, tiny_logits_config());
        throw TestFailure("Llama logits projection should reject F32 output weight");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "Llama logits dtype error mismatch");
        expect(error.expected_value() == 1, "Llama logits expected output dtype mismatch");
        expect(error.actual_value() == 0, "Llama logits actual output dtype mismatch");
    }
}

const TestCase kLlamaModelBlockTests[] = {
    {"binds and runs Llama model blocks from named tensors", test_binds_and_runs_llama_model_blocks_from_named_tensors},
};

const TestCase kLlamaModelLogitsTests[] = {
    {
        "rejects Llama model weight binding missing tensor",
        test_rejects_llama_model_weight_binding_missing_tensor,
    },
    {
        "binds Llama model missing output weight to token embedding",
        test_binds_llama_model_missing_output_weight_to_token_embedding,
    },
    {
        "rejects Llama model blocks projection dtype mismatch",
        test_rejects_llama_model_blocks_projection_dtype_mismatch,
    },
    {"embeds Llama token IDs from F16 weight", test_embeds_llama_token_ids_from_f16_weight},
    {
        "rejects Llama token embedding ID out of range",
        test_rejects_llama_token_embedding_id_out_of_range,
    },
    {"projects Llama logits with final norm", test_projects_llama_logits_with_final_norm},
    {"runs Llama model logits from token IDs", test_runs_llama_model_logits_from_token_ids},
    {
        "runs Llama model logits with KV cache matches no-cache",
        test_runs_llama_model_logits_with_kv_cache_matches_no_cache,
    },
    {
        "rejects Llama model KV cache layer count mismatch",
        test_rejects_llama_model_kv_cache_layer_count_mismatch,
    },
    {
        "rejects Llama logits output weight dtype mismatch",
        test_rejects_llama_logits_output_weight_dtype_mismatch,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_model_block_test_cases() {
    return kLlamaModelBlockTests;
}

std::span<const TestCase> llama_model_logits_test_cases() {
    return kLlamaModelLogitsTests;
}

}  // namespace local_gguf_engine::tests
