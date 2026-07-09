#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace local_gguf_engine::engine {

struct TokenLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

[[nodiscard]] std::vector<TokenLogit> top_logit_tokens_f32(
    const tensor::TensorStorage& logits,
    std::size_t max_tokens
);

}  // namespace local_gguf_engine::engine
