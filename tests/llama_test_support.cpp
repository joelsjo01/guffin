#include "llama_test_support.hpp"

#include "tensor_test_support.hpp"

#include <array>
#include <span>
#include <utility>

namespace local_gguf_engine::tests {

using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::ModelArchitecture;
using local_gguf_engine::tensor::TensorStorage;

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

LlamaModelConfig tiny_logits_config() {
    auto config = tiny_feed_forward_layer_config();
    config.vocab_size = 3;
    return config;
}

EngineTensor engine_tensor(std::string name, TensorStorage storage) {
    return EngineTensor {
        .name = std::move(name),
        .storage = std::move(storage),
    };
}

std::string llama_block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

void add_tiny_llama_block_tensors(
    std::vector<EngineTensor>& tensors,
    std::uint32_t block_index,
    float projection_scale,
    bool query_as_f32,
    bool include_down_weight
) {
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> projection_values {
        projection_scale, 0.0F,
        0.0F, projection_scale,
    };

    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    if (query_as_f32) {
        tensors.push_back(engine_tensor(
            llama_block_tensor_name(block_index, "attn_q.weight"),
            tensor_f32_matrix(2, 2, std::span<const float>(projection_values))
        ));
    } else {
        tensors.push_back(engine_tensor(
            llama_block_tensor_name(block_index, "attn_q.weight"),
            tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
        ));
    }
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_k.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_v.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_output.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_norm.weight"),
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_gate.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_up.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    if (include_down_weight) {
        tensors.push_back(engine_tensor(
            llama_block_tensor_name(block_index, "ffn_down.weight"),
            tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
        ));
    }
}

void add_tiny_llama_top_level_tensors(std::vector<EngineTensor>& tensors) {
    const std::array<float, 16> token_embedding_values {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
        10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F, 70.0F, 80.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 16> output_values {
        1.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };

    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 8, std::span<const float>(token_embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 8, std::span<const float>(output_values))
    ));
}

EngineModel tiny_bound_llama_model(LlamaModelConfig config, std::vector<EngineTensor> tensors) {
    return EngineModel(ModelArchitecture::llama, {}, config, std::move(tensors));
}

}  // namespace local_gguf_engine::tests
