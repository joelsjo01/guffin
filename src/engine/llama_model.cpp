#include "local_gguf_engine/engine/llama_model.hpp"

#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorError;
using tensor::TensorErrorCode;
using tensor::TensorStorage;
using tensor::ops::matmul_f32_f16;
using tensor::ops::rms_norm_f32;

constexpr std::size_t vector_rank = 1;
constexpr std::size_t matrix_rank = 2;

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

[[nodiscard]] std::uint64_t dtype_value(DType dtype) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype));
}

void require_dtype(const TensorDescriptor& descriptor, DType expected) {
    if (descriptor.dtype != expected) {
        throw TensorError(TensorErrorCode::unsupported_operator_dtype, dtype_value(expected), dtype_value(descriptor.dtype));
    }
}

void require_rank(const TensorDescriptor& descriptor, std::size_t expected_rank) {
    if (descriptor.shape.rank() != expected_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(expected_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_dimension(
    const TensorDescriptor& descriptor,
    std::size_t dimension_index,
    std::uint64_t expected_dimension
) {
    const auto actual_dimension = descriptor.shape.dimension(dimension_index);
    if (actual_dimension != expected_dimension) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_dimension, actual_dimension);
    }
}

void require_f16_matrix_shape(const TensorStorage& storage, std::uint64_t rows, std::uint64_t columns) {
    require_dtype(storage.descriptor(), DType::f16);
    require_rank(storage.descriptor(), matrix_rank);
    require_dimension(storage.descriptor(), 0, rows);
    require_dimension(storage.descriptor(), 1, columns);
}

void require_f32_matrix_hidden(const TensorStorage& storage, std::uint64_t hidden) {
    require_dtype(storage.descriptor(), DType::f32);
    require_rank(storage.descriptor(), matrix_rank);
    require_dimension(storage.descriptor(), 1, hidden);
}

void require_f32_vector_shape(const TensorStorage& storage, std::uint64_t length) {
    require_dtype(storage.descriptor(), DType::f32);
    require_rank(storage.descriptor(), vector_rank);
    require_dimension(storage.descriptor(), 0, length);
}

void require_positive_config_dimension(std::uint32_t value) {
    if (value == 0) {
        throw std::invalid_argument("Llama model config dimensions must be positive");
    }
}

void require_generation_fits_context(std::size_t prompt_tokens, std::size_t max_new_tokens, std::uint32_t context_length) {
    if (context_length == 0) {
        throw std::invalid_argument("Llama model config dimensions must be positive");
    }

    if (prompt_tokens > std::numeric_limits<std::size_t>::max() - max_new_tokens) {
        throw std::invalid_argument("Llama generation token count overflow");
    }

    const auto total_tokens = prompt_tokens + max_new_tokens;
    if (total_tokens > static_cast<std::size_t>(context_length)) {
        throw std::invalid_argument("Llama generation would exceed context length");
    }
}

[[nodiscard]] float read_f16_linear_as_f32(const TensorStorage& storage, std::uint64_t linear_index) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(std::uint16_t));
    const auto bytes = storage.bytes();
    const auto bits = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[byte_offset])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[byte_offset + 1]) << 8U)
    );
    return tensor::f16_bits_to_f32(bits);
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
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

    std::vector<LlamaBlockWeights> blocks;
    blocks.reserve(model.llama_config().block_count);
    for (std::uint32_t block_index = 0; block_index < model.llama_config().block_count; ++block_index) {
        blocks.push_back(bind_llama_block_weights(model, block_index));
    }

    return LlamaModelWeights {
        .token_embedding_weight = require_tensor(model, "token_embd.weight"),
        .output_norm_weight = require_tensor(model, "output_norm.weight"),
        .output_weight = require_tensor(model, "output.weight"),
        .blocks = std::move(blocks),
    };
}

TensorStorage embed_llama_token_ids_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
) {
    if (token_ids.empty()) {
        throw std::invalid_argument("Llama token id span must not be empty");
    }

    require_positive_config_dimension(config.embedding_length);
    require_positive_config_dimension(config.vocab_size);

    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto vocab = static_cast<std::uint64_t>(config.vocab_size);
    const auto token_count = static_cast<std::uint64_t>(token_ids.size());
    require_f16_matrix_shape(weights.token_embedding_weight, hidden, vocab);

    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({token_count, hidden}), DType::f32));
    for (std::uint64_t token_index = 0; token_index < token_count; ++token_index) {
        const auto token_id = static_cast<std::uint64_t>(token_ids[static_cast<std::size_t>(token_index)]);
        if (token_id >= vocab) {
            throw TensorError(TensorErrorCode::tensor_index_out_of_range, vocab, token_id);
        }

        for (std::uint64_t hidden_index = 0; hidden_index < hidden; ++hidden_index) {
            write_f32_linear(
                output,
                token_index * hidden + hidden_index,
                read_f16_linear_as_f32(weights.token_embedding_weight, hidden_index * vocab + token_id)
            );
        }
    }

    return output;
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

TensorStorage project_llama_logits_f32(
    const TensorStorage& hidden_states,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
) {
    require_positive_config_dimension(config.embedding_length);
    require_positive_config_dimension(config.vocab_size);

    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto vocab = static_cast<std::uint64_t>(config.vocab_size);
    require_f32_matrix_hidden(hidden_states, hidden);
    require_f32_vector_shape(weights.output_norm_weight, hidden);
    require_f16_matrix_shape(weights.output_weight, hidden, vocab);

    const auto normalized = rms_norm_f32(hidden_states, weights.output_norm_weight, config.rms_norm_epsilon);
    return matmul_f32_f16(normalized, weights.output_weight);
}

TensorStorage run_llama_model_logits_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    const auto embeddings = embed_llama_token_ids_f32(token_ids, weights, config);
    const auto hidden_states = run_llama_model_blocks_f32(embeddings, weights, config, position);
    return project_llama_logits_f32(hidden_states, weights, config);
}

std::uint32_t run_llama_model_next_token_argmax_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    const auto logits = run_llama_model_logits_f32(token_ids, weights, config, position);
    return sample_argmax_token_f32(logits);
}

std::vector<std::uint32_t> generate_llama_argmax_tokens_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
) {
    if (max_new_tokens == 0) {
        throw std::invalid_argument("Llama generation max_new_tokens must be positive");
    }

    if (prompt_token_ids.empty()) {
        throw std::invalid_argument("Llama token id span must not be empty");
    }

    require_generation_fits_context(prompt_token_ids.size(), max_new_tokens, config.context_length);

    std::vector<std::uint32_t> tokens(prompt_token_ids.begin(), prompt_token_ids.end());
    tokens.reserve(prompt_token_ids.size() + max_new_tokens);
    for (std::size_t generated = 0; generated < max_new_tokens; ++generated) {
        const auto position = static_cast<std::uint64_t>(tokens.size() - 1);
        const auto next_token = run_llama_model_next_token_argmax_f32(
            std::span<const std::uint32_t>(tokens.data(), tokens.size()),
            weights,
            config,
            position
        );
        tokens.push_back(next_token);
    }

    return tokens;
}

}  // namespace local_gguf_engine::engine
