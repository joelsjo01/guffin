#include "gemma2_reference_test_support.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace local_gguf_engine::tests {
namespace {

using tensor::DType;
using tensor::TensorStorage;
using runtime::BackendKind;

TensorStorage copy_view(tensor::TensorView view) {
    auto storage = TensorStorage::allocate(view.descriptor());
    std::copy(view.bytes().begin(), view.bytes().end(), storage.mutable_bytes().begin());
    return storage;
}

TensorStorage weight_matmul(const TensorStorage& input, tensor::TensorView weight) {
    const auto storage = copy_view(weight);
    return weight.descriptor().dtype == DType::f16
        ? tensor::ops::matmul_f32_f16_gguf_weight(input, storage)
        : tensor::ops::matmul_f32_quantized_gguf_weight(input, storage);
}

TensorStorage round_through_f16(const TensorStorage& input) {
    auto rounded = TensorStorage::allocate(input.descriptor());
    for (std::uint64_t index = 0; index < input.descriptor().element_count; ++index) {
        float value = 0.0F;
        const auto offset = static_cast<std::size_t>(index * sizeof(float));
        std::memcpy(&value, input.bytes().data() + offset, sizeof(value));
        const auto rounded_value = tensor::f16_bits_to_f32(tensor::f32_to_f16_bits(value));
        std::memcpy(
            rounded.mutable_bytes().data() + offset,
            &rounded_value,
            sizeof(rounded_value)
        );
    }
    return rounded;
}

void write_cache_row(TensorStorage& cache, const TensorStorage& row, std::uint32_t position) {
    const auto row_bytes = row.descriptor().byte_count;
    const auto offset = static_cast<std::size_t>(position) * row_bytes;
    if (offset > cache.mutable_bytes().size() || row_bytes > cache.mutable_bytes().size() - offset) {
        throw std::invalid_argument("reference decode KV row exceeds logical cache");
    }
    std::memcpy(cache.mutable_bytes().data() + offset, row.bytes().data(), row_bytes);
}

std::vector<engine::TokenLogit> sorted_candidates(const TensorStorage& logits) {
    std::vector<engine::TokenLogit> candidates;
    candidates.reserve(static_cast<std::size_t>(logits.descriptor().element_count));
    for (std::uint64_t token = 0; token < logits.descriptor().element_count; ++token) {
        const std::array<std::uint64_t, 2> indices {0U, token};
        candidates.push_back({
            .token = static_cast<std::uint32_t>(token),
            .logit = tensor::read_f32(logits, indices),
        });
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.logit > right.logit || (left.logit == right.logit && left.token < right.token);
    });
    return candidates;
}

}  // namespace

ReferenceDecodeResult reference_decode(
    const engine::Gemma2ModelConfig& config,
    const engine::Gemma2ModelWeights& weights,
    std::uint32_t token_id,
    std::uint32_t cache_position,
    std::vector<TensorStorage> key_cache,
    std::vector<TensorStorage> value_cache
) {
    if (key_cache.size() != config.block_count || value_cache.size() != config.block_count) {
        throw std::invalid_argument("reference decode cache layer count mismatch");
    }
    const auto embedding_weight = tensor::copy_tensor_view_to_backend(
        weights.token_embedding_weight, BackendKind::cpu
    );
    const std::array<std::uint32_t, 1> token_ids {token_id};
    auto hidden = tensor::copy_tensor_from_backend(
        tensor::ops::embed_f16_gguf_weight_token_ids_to_backend(
            std::span<const std::uint32_t>(token_ids),
            embedding_weight,
            std::sqrt(static_cast<float>(config.embedding_length))
        )
    );
    const auto dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(dimension));
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto& block = weights.blocks[layer];
        const auto normalized_attention = tensor::ops::rms_norm_f32(
            hidden, copy_view(block.attention.attention_norm_weight), config.rms_norm_epsilon
        );
        const auto raw_query = weight_matmul(normalized_attention, block.attention.query_weight);
        const auto raw_key = weight_matmul(normalized_attention, block.attention.key_weight);
        const auto value = weight_matmul(normalized_attention, block.attention.value_weight);
        const auto query = tensor::ops::rope_neox_f32_by_heads_with_position_start(
            raw_query, cache_position, config.attention_head_count, dimension, dimension, 10000.0F
        );
        const auto key = round_through_f16(tensor::ops::rope_neox_f32_by_heads_with_position_start(
            raw_key, cache_position, config.attention_head_count_kv, dimension, dimension, 10000.0F
        ));
        const auto rounded_value = round_through_f16(value);
        write_cache_row(key_cache[layer], key, cache_position);
        write_cache_row(value_cache[layer], rounded_value, cache_position);
        const auto total_tokens = cache_position + 1U;
        const auto attention_tokens = engine::gemma2_layer_uses_sliding_window(layer)
            ? std::min(total_tokens, config.sliding_window) : total_tokens;
        const auto scores = tensor::ops::causal_grouped_attention_scores_f32(
            query, key_cache[layer], cache_position, config.attention_head_count,
            config.attention_head_count_kv, dimension, attention_tokens,
            attention_scale, config.attention_logit_softcap
        );
        const auto probabilities = tensor::ops::softmax_f32(scores);
        const auto attention_output = tensor::ops::causal_grouped_attention_values_f32(
            probabilities, value_cache[layer], cache_position, config.attention_head_count,
            config.attention_head_count_kv, config.attention_value_length, attention_tokens
        );
        const auto projected_attention = weight_matmul(attention_output, block.attention.output_weight);
        const auto post_attention = tensor::ops::rms_norm_f32(
            projected_attention, copy_view(block.attention.post_attention_norm_weight),
            config.rms_norm_epsilon
        );
        const auto attention_residual = tensor::ops::add_f32(hidden, post_attention);
        const auto normalized_ffn = tensor::ops::rms_norm_f32(
            attention_residual, copy_view(block.feed_forward.feed_forward_norm_weight),
            config.rms_norm_epsilon
        );
        const auto gate = weight_matmul(normalized_ffn, block.feed_forward.gate_weight);
        const auto up = weight_matmul(normalized_ffn, block.feed_forward.up_weight);
        const auto gated = tensor::ops::multiply_f32(tensor::ops::gelu_tanh_f32(gate), up);
        const auto projected_ffn = weight_matmul(gated, block.feed_forward.down_weight);
        const auto post_ffn = tensor::ops::rms_norm_f32(
            projected_ffn, copy_view(block.feed_forward.post_feed_forward_norm_weight),
            config.rms_norm_epsilon
        );
        hidden = tensor::ops::add_f32(attention_residual, post_ffn);
    }
    const auto normalized_output = tensor::ops::rms_norm_f32(
        hidden, copy_view(weights.output_norm_weight), config.rms_norm_epsilon
    );
    auto logits = weight_matmul(normalized_output, weights.output_weight);
    logits = tensor::ops::softcap_f32(logits, config.final_logit_softcap);
    return {
        .logits = std::move(logits),
        .key_cache = std::move(key_cache),
        .value_cache = std::move(value_cache),
    };
}

ReferenceDecodeTrace reference_decode_trace(
    const engine::Gemma2ModelConfig& config,
    const engine::Gemma2ModelWeights& weights,
    std::span<const std::uint32_t> token_ids,
    std::uint32_t initial_cache_position,
    std::vector<TensorStorage> key_cache,
    std::vector<TensorStorage> value_cache
) {
    if (token_ids.empty()) throw std::invalid_argument("reference decode trace requires tokens");
    if (initial_cache_position > config.context_length
        || token_ids.size() > config.context_length - initial_cache_position) {
        throw std::invalid_argument("reference decode trace exceeds context");
    }
    ReferenceDecodeTrace trace {.initial_cache_position = initial_cache_position};
    trace.steps.reserve(token_ids.size());
    auto position = initial_cache_position;
    for (const auto token_id : token_ids) {
        auto result = reference_decode(config, weights, token_id, position, key_cache, value_cache);
        key_cache = result.key_cache;
        value_cache = result.value_cache;
        trace.steps.push_back({
            .input_token_id = token_id,
            .cache_position = position,
            .logits = result.logits,
            .candidates = sorted_candidates(result.logits),
            .key_cache = result.key_cache,
            .value_cache = result.value_cache,
        });
        ++position;
    }
    trace.final_cache_position = position;
    return trace;
}

}  // namespace local_gguf_engine::tests
