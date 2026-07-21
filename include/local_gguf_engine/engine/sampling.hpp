#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::size_t token_sampler_max_top_k = 256;
inline constexpr std::size_t token_sampler_max_repetition_window = 64;
inline constexpr std::size_t token_sampler_max_base_candidates =
    token_sampler_max_top_k + token_sampler_max_repetition_window;
inline constexpr std::size_t token_sampler_max_candidates =
    token_sampler_max_base_candidates + token_sampler_max_repetition_window;

struct TokenLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

struct TokenSelectionRequest {
    std::uint32_t top_k = 1;
    std::uint32_t base_candidate_count = 1;
    std::array<std::uint32_t, token_sampler_max_repetition_window> history_token_ids {};
    std::uint32_t history_token_count = 0;
};

struct CompactTokenCandidates {
    std::array<TokenLogit, token_sampler_max_candidates> values {};
    std::uint32_t count = 0;
    std::uint32_t readback_bytes = 0;
};

struct TokenSamplerOptions {
    static constexpr std::uint64_t random_seed = std::numeric_limits<std::uint64_t>::max();

    float temperature = 0.0F;
    std::size_t top_k = 40;
    float top_p = 0.95F;
    float min_p = 0.05F;
    float repetition_penalty = 1.0F;
    std::uint64_t seed = random_seed;
    std::size_t repetition_window = 64;
};

class TokenSampler final {
public:
    explicit TokenSampler(TokenSamplerOptions options = {});

    [[nodiscard]] bool is_greedy() const noexcept;
    [[nodiscard]] TokenSelectionRequest selection_request() const noexcept;
    [[nodiscard]] std::uint32_t sample(std::span<const TokenLogit> compact_candidates);
    void observe_tokens(std::span<const std::uint32_t> token_ids);
    void reset_history(std::span<const std::uint32_t> token_ids = {});

private:
    TokenSamplerOptions options_;
    std::mt19937_64 rng_;
    std::vector<std::uint32_t> token_history_;
};

}  // namespace local_gguf_engine::engine
