#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

struct TokenLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

struct TokenSamplerOptions {
    float temperature = 0.0F;
    std::size_t top_k = 40;
    float top_p = 0.95F;
    float repetition_penalty = 1.0F;
    std::uint64_t seed = 0;
};

class TokenSampler final {
public:
    explicit TokenSampler(TokenSamplerOptions options = {});

    [[nodiscard]] bool is_greedy() const noexcept;
    [[nodiscard]] std::uint32_t sample(const tensor::TensorStorage& logits);
    void observe_tokens(std::span<const std::uint32_t> token_ids);
    void reset_history(std::span<const std::uint32_t> token_ids = {});

private:
    TokenSamplerOptions options_;
    std::mt19937_64 rng_;
    std::vector<std::uint32_t> token_history_;
};

[[nodiscard]] std::vector<TokenLogit> top_logit_tokens_f32(
    const tensor::TensorStorage& logits,
    std::size_t max_tokens
);

}  // namespace local_gguf_engine::engine
