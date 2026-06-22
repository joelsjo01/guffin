#include "local_gguf_engine/engine/gemma2_model.hpp"

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

[[nodiscard]] const tensor::TensorStorage& output_weight_or_tied_embedding(
    const EngineModel& model,
    const tensor::TensorStorage& token_embedding_weight
) {
    const auto* output_weight = model.find_tensor("output.weight");
    if (output_weight == nullptr) {
        return token_embedding_weight;
    }

    return output_weight->storage;
}

[[nodiscard]] Gemma2BlockWeights bind_gemma2_block_weights(const EngineModel& model, std::uint32_t block_index) {
    return Gemma2BlockWeights {
        .attention = Gemma2AttentionWeights {
            .attention_norm_weight = require_tensor(model, block_tensor_name(block_index, "attn_norm.weight")),
            .post_attention_norm_weight = require_tensor(
                model,
                block_tensor_name(block_index, "post_attention_norm.weight")
            ),
            .query_weight = require_tensor(model, block_tensor_name(block_index, "attn_q.weight")),
            .key_weight = require_tensor(model, block_tensor_name(block_index, "attn_k.weight")),
            .value_weight = require_tensor(model, block_tensor_name(block_index, "attn_v.weight")),
            .output_weight = require_tensor(model, block_tensor_name(block_index, "attn_output.weight")),
        },
        .feed_forward = Gemma2FeedForwardWeights {
            .feed_forward_norm_weight = require_tensor(model, block_tensor_name(block_index, "ffn_norm.weight")),
            .post_feed_forward_norm_weight = require_tensor(
                model,
                block_tensor_name(block_index, "post_ffw_norm.weight")
            ),
            .gate_weight = require_tensor(model, block_tensor_name(block_index, "ffn_gate.weight")),
            .up_weight = require_tensor(model, block_tensor_name(block_index, "ffn_up.weight")),
            .down_weight = require_tensor(model, block_tensor_name(block_index, "ffn_down.weight")),
        },
    };
}

}  // namespace

Gemma2ModelWeights bind_gemma2_model_weights(const EngineModel& model) {
    if (model.architecture() != ModelArchitecture::gemma2) {
        throw EngineModelError(
            EngineModelErrorCode::unsupported_architecture,
            std::string(model_architecture_name(model.architecture()))
        );
    }

    std::vector<Gemma2BlockWeights> blocks;
    blocks.reserve(model.gemma2_config().block_count);
    for (std::uint32_t block_index = 0; block_index < model.gemma2_config().block_count; ++block_index) {
        blocks.push_back(bind_gemma2_block_weights(model, block_index));
    }

    const auto& token_embedding_weight = require_tensor(model, "token_embd.weight");
    return Gemma2ModelWeights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = require_tensor(model, "output_norm.weight"),
        .output_weight = output_weight_or_tied_embedding(model, token_embedding_weight),
        .blocks = std::move(blocks),
    };
}

}  // namespace local_gguf_engine::engine
