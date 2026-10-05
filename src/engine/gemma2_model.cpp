#include "local_gguf_engine/engine/gemma2_model.hpp"

#include <string>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

[[nodiscard]] std::string block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

[[nodiscard]] tensor::TensorView require_tensor_view(const EngineModel& model, std::string name) {
    const auto* tensor_view = model.find_tensor_view(name);
    if (tensor_view != nullptr) {
        return tensor_view->view;
    }

    throw EngineModelError(EngineModelErrorCode::missing_tensor, std::move(name));
}

[[nodiscard]] tensor::TensorView output_weight_or_tied_embedding(
    const EngineModel& model,
    const tensor::TensorView& token_embedding_weight
) {
    const auto* output_weight_view = model.find_tensor_view("output.weight");
    if (output_weight_view != nullptr) {
        return output_weight_view->view;
    }

    return token_embedding_weight;
}

[[nodiscard]] Gemma2BlockWeights bind_gemma2_block_weights(const EngineModel& model, std::uint32_t block_index) {
    return Gemma2BlockWeights {
        .attention = Gemma2AttentionWeights {
            .attention_norm_weight = require_tensor_view(model, block_tensor_name(block_index, "attn_norm.weight")),
            .post_attention_norm_weight = require_tensor_view(
                model,
                block_tensor_name(block_index, "post_attention_norm.weight")
            ),
            .query_weight = require_tensor_view(model, block_tensor_name(block_index, "attn_q.weight")),
            .key_weight = require_tensor_view(model, block_tensor_name(block_index, "attn_k.weight")),
            .value_weight = require_tensor_view(model, block_tensor_name(block_index, "attn_v.weight")),
            .output_weight = require_tensor_view(model, block_tensor_name(block_index, "attn_output.weight")),
            .query_norm_weight = model.gemma2_config().uses_qk_norm
                ? std::optional(require_tensor_view(model, block_tensor_name(block_index, "attn_q_norm.weight")))
                : std::nullopt,
            .key_norm_weight = model.gemma2_config().uses_qk_norm
                ? std::optional(require_tensor_view(model, block_tensor_name(block_index, "attn_k_norm.weight")))
                : std::nullopt,
        },
        .feed_forward = Gemma2FeedForwardWeights {
            .feed_forward_norm_weight = require_tensor_view(model, block_tensor_name(block_index, "ffn_norm.weight")),
            .post_feed_forward_norm_weight = require_tensor_view(
                model,
                block_tensor_name(block_index, "post_ffw_norm.weight")
            ),
            .gate_weight = require_tensor_view(model, block_tensor_name(block_index, "ffn_gate.weight")),
            .up_weight = require_tensor_view(model, block_tensor_name(block_index, "ffn_up.weight")),
            .down_weight = require_tensor_view(model, block_tensor_name(block_index, "ffn_down.weight")),
        },
    };
}

}  // namespace

Gemma2ModelWeights bind_gemma2_model_weights(const EngineModel& model) {
    if (model.architecture() != ModelArchitecture::gemma2 && model.architecture() != ModelArchitecture::gemma3) {
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

    const auto token_embedding_weight = require_tensor_view(model, "token_embd.weight");
    auto weights = Gemma2ModelWeights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = require_tensor_view(model, "output_norm.weight"),
        .output_weight = output_weight_or_tied_embedding(model, token_embedding_weight),
        .blocks = std::move(blocks),
    };
    if (model.architecture() == ModelArchitecture::gemma3) {
        validate_gemma3_model_weights(model.gemma2_config(), weights);
    }
    return weights;
}

void validate_gemma3_model_weights(const Gemma2ModelConfig& config, const Gemma2ModelWeights& weights) {
    using tensor::DType;
    const auto invalid = [](std::string detail) {
        throw EngineModelError(EngineModelErrorCode::invalid_tensor, std::move(detail));
    };
    if (config.architecture != ModelArchitecture::gemma3 || !config.uses_qk_norm
        || config.attention_global_layer_period != 6U || config.embedding_length == 0U
        || config.feed_forward_length == 0U || config.block_count == 0U
        || weights.blocks.size() != config.block_count || config.vocab_size == 0U
        || config.attention_head_count == 0U || config.attention_head_count_kv == 0U
        || config.attention_head_count % config.attention_head_count_kv != 0U
        || config.attention_key_length == 0U || config.attention_key_length % 2U != 0U
        || config.attention_key_length != config.attention_value_length
        || !std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0F
        || config.attention_logit_softcap != 0.0F || config.final_logit_softcap != 0.0F
        || !std::isfinite(config.rope_freq_base) || config.rope_freq_base <= 0.0F
        || !std::isfinite(config.local_rope_freq_base) || config.local_rope_freq_base <= 0.0F
        || !std::isfinite(config.rope_freq_scale) || config.rope_freq_scale <= 0.0F) {
        invalid("Gemma 3 execution configuration");
    }
    if (config.attention_key_length != 256U
        || std::uint64_t(config.attention_head_count) != 2U * std::uint64_t(config.attention_head_count_kv)
        || config.block_count > 64U) {
        invalid("Gemma 3 target requires 256-wide paired GQA heads and at most 64 layers");
    }
    if (config.context_length < config.sliding_window) {
        invalid("Gemma 3 runtime context must cover the local attention window");
    }
    const auto query_width = std::uint64_t(config.attention_head_count) * config.attention_key_length;
    const auto kv_width = std::uint64_t(config.attention_head_count_kv) * config.attention_key_length;
    if (query_width > std::numeric_limits<std::uint32_t>::max()
        || kv_width > std::numeric_limits<std::uint32_t>::max()) {
        invalid("Gemma 3 attention width exceeds Metal indexing range");
    }
    const auto norm = [&](tensor::TensorView view, std::uint64_t width, const std::string& name) {
        if (view.descriptor().packing != tensor::TensorPacking::canonical
            || view.descriptor().shape != tensor::Shape::from({width}) || view.descriptor().dtype != DType::f32) {
            invalid(name + ": expected F32 norm with width " + std::to_string(width));
        }
    };
    const auto matrix = [&](tensor::TensorView view, std::uint64_t input, std::uint64_t output,
                            const std::string& name, bool head = false) {
        const auto type = view.descriptor().dtype;
        if (view.descriptor().packing != tensor::TensorPacking::canonical
            || view.descriptor().shape != tensor::Shape::from({input, output})
            || (head ? type != DType::q6_k : type != DType::q4_k && type != DType::q6_k)
            || input % 256U != 0U) {
            invalid(name + ": expected native " + (head ? "Q6_K" : "Q4_K/Q6_K")
                + " GGUF matrix [" + std::to_string(input) + ", " + std::to_string(output) + "]");
        }
    };
    matrix(weights.token_embedding_weight, config.embedding_length, config.vocab_size, "token_embd.weight", true);
    matrix(weights.output_weight, config.embedding_length, config.vocab_size, "output.weight", true);
    if (config.uses_tied_output_weight
        && weights.token_embedding_weight.bytes().data() != weights.output_weight.bytes().data()) {
        invalid("Gemma 3 tied output must alias token_embd.weight");
    }
    norm(weights.output_norm_weight, config.embedding_length, "output_norm.weight");
    for (std::uint32_t i = 0; i < config.block_count; ++i) {
        const auto& block = weights.blocks[i];
        const auto name = [&](std::string_view suffix) { return block_tensor_name(i, suffix); };
        norm(block.attention.attention_norm_weight, config.embedding_length, name("attn_norm.weight"));
        norm(block.attention.post_attention_norm_weight, config.embedding_length, name("post_attention_norm.weight"));
        if (!block.attention.query_norm_weight || !block.attention.key_norm_weight) {
            invalid(name("attention requires Q/K norm weights"));
        }
        norm(*block.attention.query_norm_weight, config.attention_key_length, name("attn_q_norm.weight"));
        norm(*block.attention.key_norm_weight, config.attention_key_length, name("attn_k_norm.weight"));
        matrix(block.attention.query_weight, config.embedding_length, query_width, name("attn_q.weight"));
        matrix(block.attention.key_weight, config.embedding_length, kv_width, name("attn_k.weight"));
        matrix(block.attention.value_weight, config.embedding_length, kv_width, name("attn_v.weight"));
        matrix(block.attention.output_weight, query_width, config.embedding_length, name("attn_output.weight"));
        norm(block.feed_forward.feed_forward_norm_weight, config.embedding_length, name("ffn_norm.weight"));
        norm(block.feed_forward.post_feed_forward_norm_weight, config.embedding_length, name("post_ffw_norm.weight"));
        matrix(block.feed_forward.gate_weight, config.embedding_length, config.feed_forward_length, name("ffn_gate.weight"));
        matrix(block.feed_forward.up_weight, config.embedding_length, config.feed_forward_length, name("ffn_up.weight"));
        matrix(block.feed_forward.down_weight, config.feed_forward_length, config.embedding_length, name("ffn_down.weight"));
    }
}

}  // namespace local_gguf_engine::engine
