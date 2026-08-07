#pragma once

#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::tests {

struct ReferenceDecodeResult {
    tensor::TensorStorage logits;
    std::vector<tensor::TensorStorage> key_cache;
    std::vector<tensor::TensorStorage> value_cache;
};

struct ReferenceDecodeTraceStep {
    std::uint32_t input_token_id = 0;
    std::uint32_t cache_position = 0;
    tensor::TensorStorage logits;
    std::vector<engine::TokenLogit> candidates;
    std::vector<tensor::TensorStorage> key_cache;
    std::vector<tensor::TensorStorage> value_cache;
};

struct ReferenceDecodeTrace {
    std::uint32_t initial_cache_position = 0;
    std::uint32_t final_cache_position = 0;
    std::vector<ReferenceDecodeTraceStep> steps;
};

[[nodiscard]] ReferenceDecodeResult reference_decode(
    const engine::Gemma2ModelConfig& config,
    const engine::Gemma2ModelWeights& weights,
    std::uint32_t token_id,
    std::uint32_t cache_position,
    std::vector<tensor::TensorStorage> key_cache,
    std::vector<tensor::TensorStorage> value_cache
);

[[nodiscard]] ReferenceDecodeTrace reference_decode_trace(
    const engine::Gemma2ModelConfig& config,
    const engine::Gemma2ModelWeights& weights,
    std::span<const std::uint32_t> token_ids,
    std::uint32_t initial_cache_position,
    std::vector<tensor::TensorStorage> key_cache,
    std::vector<tensor::TensorStorage> value_cache
);

}  // namespace local_gguf_engine::tests
