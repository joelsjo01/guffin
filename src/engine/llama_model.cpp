#include "local_gguf_engine/engine/llama_model.hpp"

#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
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
using tensor::copy_tensor_to_backend;
using tensor::TensorDescriptor;
using tensor::TensorError;
using tensor::TensorErrorCode;
using tensor::TensorStorage;
using tensor::ops::matmul_f32_f16_gguf_weight;
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

[[nodiscard]] const tensor::TensorStorage& output_weight_or_tied_embedding(
    const EngineModel& model,
    const tensor::TensorStorage& token_embedding_weight
) {
    const auto* output_weight = model.find_tensor("output.weight");
    if (output_weight == nullptr) {
        // Tied-output GGUFs omit lm_head/output.weight and reuse token_embd.weight.
        return token_embedding_weight;
    }

    return output_weight->storage;
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

void require_f16_matrix_shape(const TensorDescriptor& descriptor, std::uint64_t rows, std::uint64_t columns) {
    require_dtype(descriptor, DType::f16);
    require_rank(descriptor, matrix_rank);
    require_dimension(descriptor, 0, rows);
    require_dimension(descriptor, 1, columns);
}

void require_f16_matrix_shape(const TensorStorage& storage, std::uint64_t rows, std::uint64_t columns) {
    require_f16_matrix_shape(storage.descriptor(), rows, columns);
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

[[nodiscard]] const TensorDescriptor& output_weight_descriptor(const TensorStorage& storage) noexcept {
    return storage.descriptor();
}

[[nodiscard]] const TensorDescriptor& output_weight_descriptor(const tensor::BackendBuffer& buffer) noexcept {
    return buffer.descriptor();
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

    const auto& token_embedding_weight = require_tensor(model, "token_embd.weight");

    return LlamaModelWeights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = require_tensor(model, "output_norm.weight"),
        .output_weight = output_weight_or_tied_embedding(model, token_embedding_weight),
        .blocks = std::move(blocks),
    };
}

LlamaModelAttentionBackendWeights copy_llama_model_attention_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
) {
    std::vector<LlamaAttentionLayerBackendWeights> blocks;
    blocks.reserve(weights.blocks.size());
    for (const auto& block : weights.blocks) {
        blocks.push_back(copy_llama_attention_layer_weights_to_backend(block.attention, backend));
    }

    return LlamaModelAttentionBackendWeights {
        .blocks = std::move(blocks),
    };
}

LlamaModelFeedForwardBackendWeights copy_llama_model_feed_forward_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
) {
    std::vector<LlamaFeedForwardLayerBackendWeights> blocks;
    blocks.reserve(weights.blocks.size());
    for (const auto& block : weights.blocks) {
        blocks.push_back(copy_llama_feed_forward_layer_weights_to_backend(block.feed_forward, backend));
    }

    return LlamaModelFeedForwardBackendWeights {
        .blocks = std::move(blocks),
    };
}

LlamaModelBlockBackendWeights copy_llama_model_block_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
) {
    std::vector<LlamaBlockBackendWeights> blocks;
    blocks.reserve(weights.blocks.size());
    for (const auto& block : weights.blocks) {
        blocks.push_back(copy_llama_block_weights_to_backend(block, backend));
    }

    return LlamaModelBlockBackendWeights {
        .blocks = std::move(blocks),
    };
}

LlamaModelLogitsBackendWeights copy_llama_model_logits_weights_to_backend(
    const LlamaModelWeights& weights,
    runtime::BackendKind backend
) {
    auto block_weights = copy_llama_model_block_weights_to_backend(weights, backend);
    return LlamaModelLogitsBackendWeights {
        .token_embedding_weight = weights.token_embedding_weight,
        .output_norm_weight = weights.output_norm_weight,
        .output_weight = copy_tensor_to_backend(weights.output_weight, backend),
        .blocks = std::move(block_weights.blocks),
    };
}

template <typename Weights>
TensorStorage embed_llama_token_ids_f32_impl(
    std::span<const std::uint32_t> token_ids,
    const Weights& weights,
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
                read_f16_linear_as_f32(weights.token_embedding_weight, token_id * hidden + hidden_index)
            );
        }
    }

    return output;
}

TensorStorage embed_llama_token_ids_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
) {
    return embed_llama_token_ids_f32_impl(token_ids, weights, config);
}

void require_llama_model_block_count(
    std::size_t block_count,
    std::uint32_t expected_block_count
) {
    if (block_count == 0) {
        throw std::invalid_argument("Llama model must contain at least one block");
    }

    if (block_count != expected_block_count) {
        throw std::invalid_argument("Llama model bound block count must match config block count");
    }
}

tensor::TensorStorage run_llama_model_blocks_f32(
    const tensor::TensorStorage& input,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    require_llama_model_block_count(weights.blocks.size(), config.block_count);

    auto output = run_llama_block_f32(input, weights.blocks[0], config, position);
    for (std::size_t block_index = 1; block_index < weights.blocks.size(); ++block_index) {
        output = run_llama_block_f32(output, weights.blocks[block_index], config, position);
    }

    return output;
}

tensor::TensorStorage run_llama_model_blocks_f32(
    const tensor::TensorStorage& input,
    const LlamaModelBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    require_llama_model_block_count(weights.blocks.size(), config.block_count);

    auto output = run_llama_block_f32(input, weights.blocks[0], config, position);
    for (std::size_t block_index = 1; block_index < weights.blocks.size(); ++block_index) {
        output = run_llama_block_f32(output, weights.blocks[block_index], config, position);
    }

    return output;
}

tensor::TensorStorage run_llama_model_blocks_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
) {
    require_llama_model_block_count(weights.blocks.size(), config.block_count);

    if (cache.layers.size() != weights.blocks.size()) {
        throw std::invalid_argument("Llama KV cache layer count must match model block count");
    }

    auto output = run_llama_block_with_kv_cache_f32(input, weights.blocks[0], config, position, cache.layers[0]);
    for (std::size_t block_index = 1; block_index < weights.blocks.size(); ++block_index) {
        output = run_llama_block_with_kv_cache_f32(
            output,
            weights.blocks[block_index],
            config,
            position,
            cache.layers[block_index]
        );
    }

    return output;
}

tensor::TensorStorage run_llama_model_blocks_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaModelBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
) {
    require_llama_model_block_count(weights.blocks.size(), config.block_count);

    if (cache.layers.size() != weights.blocks.size()) {
        throw std::invalid_argument("Llama KV cache layer count must match model block count");
    }

    auto output = run_llama_block_with_kv_cache_f32(input, weights.blocks[0], config, position, cache.layers[0]);
    for (std::size_t block_index = 1; block_index < weights.blocks.size(); ++block_index) {
        output = run_llama_block_with_kv_cache_f32(
            output,
            weights.blocks[block_index],
            config,
            position,
            cache.layers[block_index]
        );
    }

    return output;
}

tensor::TensorStorage run_llama_model_blocks_with_kv_cache_f32(
    const tensor::TensorStorage& input,
    const LlamaModelBlockBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaBackendKVCache& cache
) {
    require_llama_model_block_count(weights.blocks.size(), config.block_count);

    if (cache.layers.size() != weights.blocks.size()) {
        throw std::invalid_argument("Llama KV cache layer count must match model block count");
    }

    auto output = run_llama_block_with_kv_cache_f32(input, weights.blocks[0], config, position, cache.layers[0]);
    for (std::size_t block_index = 1; block_index < weights.blocks.size(); ++block_index) {
        output = run_llama_block_with_kv_cache_f32(
            output,
            weights.blocks[block_index],
            config,
            position,
            cache.layers[block_index]
        );
    }

    return output;
}

template <typename Weights>
TensorStorage project_llama_logits_f32_impl(
    const TensorStorage& hidden_states,
    const Weights& weights,
    const LlamaModelConfig& config
) {
    require_positive_config_dimension(config.embedding_length);
    require_positive_config_dimension(config.vocab_size);

    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto vocab = static_cast<std::uint64_t>(config.vocab_size);
    require_f32_matrix_hidden(hidden_states, hidden);
    require_f32_vector_shape(weights.output_norm_weight, hidden);
    require_f16_matrix_shape(output_weight_descriptor(weights.output_weight), hidden, vocab);

    const auto normalized = rms_norm_f32(hidden_states, weights.output_norm_weight, config.rms_norm_epsilon);
    return matmul_f32_f16_gguf_weight(normalized, weights.output_weight);
}

TensorStorage project_llama_logits_f32(
    const TensorStorage& hidden_states,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config
) {
    return project_llama_logits_f32_impl(hidden_states, weights, config);
}

TensorStorage project_llama_logits_f32(
    const TensorStorage& hidden_states,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config
) {
    return project_llama_logits_f32_impl(hidden_states, weights, config);
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

TensorStorage run_llama_model_logits_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    const auto embeddings = embed_llama_token_ids_f32_impl(token_ids, weights, config);
    const auto hidden_states = run_llama_model_blocks_f32(embeddings, LlamaModelBlockBackendWeights {
        .blocks = weights.blocks,
    }, config, position);
    return project_llama_logits_f32(hidden_states, weights, config);
}

TensorStorage run_llama_model_logits_with_kv_cache_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
) {
    const auto embeddings = embed_llama_token_ids_f32(token_ids, weights, config);
    const auto hidden_states = run_llama_model_blocks_with_kv_cache_f32(
        embeddings,
        weights,
        config,
        position,
        cache
    );
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

std::uint32_t run_llama_model_next_token_argmax_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position
) {
    const auto logits = run_llama_model_logits_f32(token_ids, weights, config, position);
    return sample_argmax_token_f32(logits);
}

std::uint32_t run_llama_model_next_token_argmax_with_kv_cache_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache
) {
    const auto logits = run_llama_model_logits_with_kv_cache_f32(token_ids, weights, config, position, cache);
    return sample_argmax_token_f32(logits);
}

std::uint32_t run_llama_model_next_token_sampled_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    const SamplingConfig& sampling,
    SamplingState& sampling_state,
    std::span<const std::uint32_t> repetition_token_ids
) {
    const auto logits = run_llama_model_logits_f32(token_ids, weights, config, position);
    return sample_token_f32(logits, sampling, sampling_state, repetition_token_ids);
}

std::uint32_t run_llama_model_next_token_sampled_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelLogitsBackendWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    const SamplingConfig& sampling,
    SamplingState& sampling_state,
    std::span<const std::uint32_t> repetition_token_ids
) {
    const auto logits = run_llama_model_logits_f32(token_ids, weights, config, position);
    return sample_token_f32(logits, sampling, sampling_state, repetition_token_ids);
}

std::uint32_t run_llama_model_next_token_sampled_with_kv_cache_f32(
    std::span<const std::uint32_t> token_ids,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::uint64_t position,
    LlamaKVCache& cache,
    const SamplingConfig& sampling,
    SamplingState& sampling_state,
    std::span<const std::uint32_t> repetition_token_ids
) {
    const auto logits = run_llama_model_logits_with_kv_cache_f32(token_ids, weights, config, position, cache);
    return sample_token_f32(logits, sampling, sampling_state, repetition_token_ids);
}

std::vector<std::uint32_t> generate_llama_argmax_tokens_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::optional<std::uint32_t> stop_token_id
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
        if (stop_token_id == next_token) {
            break;
        }
    }

    return tokens;
}

std::vector<std::uint32_t> generate_llama_argmax_tokens_with_kv_cache_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    std::optional<std::uint32_t> stop_token_id
) {
    if (max_new_tokens == 0) {
        throw std::invalid_argument("Llama generation max_new_tokens must be positive");
    }

    if (prompt_token_ids.empty()) {
        throw std::invalid_argument("Llama token id span must not be empty");
    }

    require_generation_fits_context(prompt_token_ids.size(), max_new_tokens, config.context_length);

    auto cache = create_llama_kv_cache(config);
    std::vector<std::uint32_t> tokens(prompt_token_ids.begin(), prompt_token_ids.end());
    tokens.reserve(prompt_token_ids.size() + max_new_tokens);

    const auto prompt_final_position = static_cast<std::uint64_t>(prompt_token_ids.size() - 1);
    auto next_token = run_llama_model_next_token_argmax_with_kv_cache_f32(
        prompt_token_ids,
        weights,
        config,
        prompt_final_position,
        cache
    );
    tokens.push_back(next_token);
    if (stop_token_id == next_token) {
        return tokens;
    }

    for (std::size_t generated = 1; generated < max_new_tokens; ++generated) {
        const auto position = static_cast<std::uint64_t>(tokens.size() - 1);
        next_token = run_llama_model_next_token_argmax_with_kv_cache_f32(
            std::span<const std::uint32_t>(&tokens.back(), 1),
            weights,
            config,
            position,
            cache
        );
        tokens.push_back(next_token);
        if (stop_token_id == next_token) {
            break;
        }
    }

    return tokens;
}

std::vector<std::uint32_t> generate_llama_sampled_tokens_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    const SamplingConfig& sampling,
    std::optional<std::uint32_t> stop_token_id
) {
    if (max_new_tokens == 0) {
        throw std::invalid_argument("Llama generation max_new_tokens must be positive");
    }

    if (prompt_token_ids.empty()) {
        throw std::invalid_argument("Llama token id span must not be empty");
    }

    require_generation_fits_context(prompt_token_ids.size(), max_new_tokens, config.context_length);

    SamplingState sampling_state(sampling.seed);
    std::vector<std::uint32_t> tokens(prompt_token_ids.begin(), prompt_token_ids.end());
    tokens.reserve(prompt_token_ids.size() + max_new_tokens);
    for (std::size_t generated = 0; generated < max_new_tokens; ++generated) {
        const auto position = static_cast<std::uint64_t>(tokens.size() - 1);
        const auto next_token = run_llama_model_next_token_sampled_f32(
            std::span<const std::uint32_t>(tokens.data(), tokens.size()),
            weights,
            config,
            position,
            sampling,
            sampling_state,
            std::span<const std::uint32_t>(tokens.data(), tokens.size())
        );
        tokens.push_back(next_token);
        if (stop_token_id == next_token) {
            break;
        }
    }

    return tokens;
}

std::vector<std::uint32_t> generate_llama_sampled_tokens_with_kv_cache_f32(
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    const SamplingConfig& sampling,
    std::optional<std::uint32_t> stop_token_id
) {
    if (max_new_tokens == 0) {
        throw std::invalid_argument("Llama generation max_new_tokens must be positive");
    }

    if (prompt_token_ids.empty()) {
        throw std::invalid_argument("Llama token id span must not be empty");
    }

    require_generation_fits_context(prompt_token_ids.size(), max_new_tokens, config.context_length);

    SamplingState sampling_state(sampling.seed);
    auto cache = create_llama_kv_cache(config);
    std::vector<std::uint32_t> tokens(prompt_token_ids.begin(), prompt_token_ids.end());
    tokens.reserve(prompt_token_ids.size() + max_new_tokens);

    const auto prompt_final_position = static_cast<std::uint64_t>(prompt_token_ids.size() - 1);
    auto next_token = run_llama_model_next_token_sampled_with_kv_cache_f32(
        prompt_token_ids,
        weights,
        config,
        prompt_final_position,
        cache,
        sampling,
        sampling_state,
        prompt_token_ids
    );
    tokens.push_back(next_token);
    if (stop_token_id == next_token) {
        return tokens;
    }

    for (std::size_t generated = 1; generated < max_new_tokens; ++generated) {
        const auto position = static_cast<std::uint64_t>(tokens.size() - 1);
        next_token = run_llama_model_next_token_sampled_with_kv_cache_f32(
            std::span<const std::uint32_t>(&tokens.back(), 1),
            weights,
            config,
            position,
            cache,
            sampling,
            sampling_state,
            std::span<const std::uint32_t>(tokens.data(), tokens.size())
        );
        tokens.push_back(next_token);
        if (stop_token_id == next_token) {
            break;
        }
    }

    return tokens;
}

}  // namespace local_gguf_engine::engine
