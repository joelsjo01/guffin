#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

struct SamplingConfig {
    float temperature = 0.0F;
    std::uint32_t top_k = 0;
    float top_p = 1.0F;
    float repetition_penalty = 1.0F;
    std::uint64_t seed = 0;
};

struct TokenLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

class SamplingState final {
public:
    explicit SamplingState(std::uint64_t seed) noexcept;

    [[nodiscard]] std::uint64_t next_u64() noexcept;

private:
    std::uint64_t state_;
};

[[nodiscard]] std::uint32_t sample_argmax_token_f32(const tensor::TensorStorage& logits);

[[nodiscard]] std::vector<TokenLogit> top_logit_tokens_f32(
    const tensor::TensorStorage& logits,
    std::size_t max_tokens
);

[[nodiscard]] std::uint32_t sample_token_f32(
    const tensor::TensorStorage& logits,
    const SamplingConfig& config,
    SamplingState& state,
    std::span<const std::uint32_t> repetition_token_ids = {}
);

}  // namespace local_gguf_engine::engine
