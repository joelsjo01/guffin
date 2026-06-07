#include "local_gguf_engine/engine/llama_model.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

[[nodiscard]] std::string block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

[[nodiscard]] const tensor::TensorStorage& require_tensor(const EngineModel& model, std::string name) {
    const auto* tensor = model.find_tensor(name);
    if (tensor == nullptr) {
        throw EngineModelError(EngineModelErrorCode::missing_tensor, std::move(name));
    }

    return tensor->storage;
}

[[nodiscard]] LlamaBlockWeights bind_llama_block_weights(const EngineModel& model, std::uint32_t block_index) {
    return LlamaBlockWeights {
        .attention = LlamaAttentionLayerWeights {
            .attention_norm_weight = require_tensor(model, block_tensor_name(block_index, "attn_norm.weight")),
            .query_weight = require_tensor(model, block_tensor_name(block_index, "attn_q.weight")),
            .key_weight = require_tensor(model, block_tensor_name(block_index, "attn_k.weight")),
            .value_weight = require_tensor(model, block_tensor_name(block_index, "attn_v.weight")),
            .output_weight = require_tensor(model, block_tensor_name(block_index, "attn_output.weight")),
        },
        .feed_forward = LlamaFeedForwardLayerWeights {
            .feed_forward_norm_weight = require_tensor(model, block_tensor_name(block_index, "ffn_norm.weight")),
            .gate_weight = require_tensor(model, block_tensor_name(block_index, "ffn_gate.weight")),
            .up_weight = require_tensor(model, block_tensor_name(block_index, "ffn_up.weight")),
            .down_weight = require_tensor(model, block_tensor_name(block_index, "ffn_down.weight")),
        },
    };
}

}  // namespace

LlamaModelWeights bind_llama_model_weights(const EngineModel& model) {
    if (model.architecture() != ModelArchitecture::llama) {
        throw EngineModelError(
            EngineModelErrorCode::unsupported_architecture,
            std::string(model_architecture_name(model.architecture()))
        );
    }

    LlamaModelWeights weights;
    weights.blocks.reserve(model.llama_config().block_count);
    for (std::uint32_t block_index = 0; block_index < model.llama_config().block_count; ++block_index) {
        weights.blocks.push_back(bind_llama_block_weights(model, block_index));
    }

    return weights;
}

tensor::TensorStorage run_llama_model_blocks_f32(
    const tensor::TensorStorage& input,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    if (weights.blocks.empty()) {
        throw std::invalid_argument("Llama model must contain at least one block");
    }

    if (weights.blocks.size() != config.block_count) {
        throw std::invalid_argument("Llama model bound block count must match config block count");
    }

    auto output = run_llama_block_f32(input, weights.blocks[0], config, position);
    for (std::size_t block_index = 1; block_index < weights.blocks.size(); ++block_index) {
        output = run_llama_block_f32(output, weights.blocks[block_index], config, position);
    }

    return output;
}

}  // namespace local_gguf_engine::engine
