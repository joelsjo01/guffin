#include "sampling_tests.hpp"
#include "test_support.hpp"

#include "local_gguf_engine/engine/sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using local_gguf_engine::engine::TokenLogit;
using local_gguf_engine::engine::TokenSampler;
using local_gguf_engine::engine::TokenSamplerOptions;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

[[nodiscard]] bool better(const TokenLogit& left, const TokenLogit& right) {
    return left.logit > right.logit || (left.logit == right.logit && left.token < right.token);
}

void test_compact_sampler_stable_ties_and_repetition() {
    const std::array<TokenLogit, 4> candidates {{{2, 3.0F}, {1, 3.0F}, {0, 2.9F}, {3, -1.0F}}};
    TokenSampler tied(TokenSamplerOptions {.temperature = 0.0F, .top_k = 4});
    expect(tied.sample(candidates) == 1U, "equal logits must prefer lower token id");

    TokenSampler penalized(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 4, .repetition_penalty = 2.0F, .repetition_window = 2,
    });
    const std::array<std::uint32_t, 3> history {2U, 2U, 1U};
    penalized.observe_tokens(history);
    const auto request = penalized.selection_request();
    expect(request.history_token_count == 2U, "selection request must deduplicate history");
    expect(request.base_candidate_count == 3U, "greedy superset must be K plus unique history");
    expect(penalized.sample(candidates) == 0U, "positive repeated logits must be divided");

    const std::array<TokenLogit, 3> negative {{{0, -1.0F}, {1, -1.1F}, {2, -2.0F}}};
    TokenSampler negative_penalty(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 3, .repetition_penalty = 2.0F, .repetition_window = 1,
    });
    const std::array<std::uint32_t, 1> repeated {0U};
    negative_penalty.observe_tokens(repeated);
    expect(negative_penalty.sample(negative) == 1U, "negative repeated logits must be multiplied");
}

void test_compact_sampler_seed_snapshot_and_filter_order() {
    const std::array<TokenLogit, 4> candidates {{{0, 0.0F}, {1, -1.0F}, {2, -4.0F}, {3, -8.0F}}};
    const auto options = TokenSamplerOptions {
        .temperature = 2.0F, .top_k = 4, .top_p = 0.7F, .min_p = 0.05F,
        .repetition_penalty = 1.1F, .seed = 123,
    };
    TokenSampler first(options);
    const std::array<std::uint32_t, 2> history {3U, 3U};
    first.observe_tokens(history);
    auto second = first;
    for (std::size_t index = 0; index < 32U; ++index) {
        const auto a = first.sample(candidates);
        const auto b = second.sample(candidates);
        expect(a == b, "copied sampler must replay RNG and history");
        expect(a == 0U, "top-p must filter before temperature and min-p");
    }
    first.reset_history();
    expect(first.selection_request().history_token_count == 0U, "reset must clear sampler history");
}

void test_repetition_window_eviction_and_promotion() {
    const std::array<TokenLogit, 3> candidates {{{0, 1.0F}, {1, 0.75F}, {2, 0.5F}}};
    TokenSampler sampler(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 3, .repetition_penalty = 2.0F, .repetition_window = 2,
    });
    const std::array<std::uint32_t, 3> old {0U, 2U, 2U};
    sampler.observe_tokens(old);
    expect(sampler.sample(candidates) == 0U, "evicted token must not be penalized");
    const std::array<std::uint32_t, 1> recent {0U};
    sampler.observe_tokens(recent);
    expect(sampler.sample(candidates) == 1U, "recent token must be penalized");

    TokenSampler promoting(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 3, .repetition_penalty = 0.5F, .repetition_window = 1,
    });
    const std::array<std::uint32_t, 1> promote {1U};
    promoting.observe_tokens(promote);
    expect(promoting.sample(candidates) == 1U, "penalty below one must promote a positive history logit");
}

void test_sparse_change_candidate_superset_property() {
    std::mt19937 rng(77U);
    std::uniform_real_distribution<float> values(-5.0F, 5.0F);
    for (const std::size_t k : {1U, 40U, 256U}) {
        const auto vocab = std::max<std::size_t>(320U, k + 64U);
        for (const std::size_t history_count : {0U, 1U, 64U}) {
            for (const float penalty : {0.5F, 1.0F, 1.7F}) {
                std::vector<TokenLogit> base(vocab);
                for (std::size_t token = 0; token < vocab; ++token) {
                    base[token] = TokenLogit {
                        .token = static_cast<std::uint32_t>(token),
                        .logit = token % 17U == 0U ? 1.0F : values(rng),
                    };
                }
                std::vector<std::uint32_t> history;
                for (std::size_t index = 0; index < history_count; ++index) {
                    history.push_back(static_cast<std::uint32_t>((index * 37U + vocab / 2U) % vocab));
                }
                auto base_sorted = base;
                std::sort(base_sorted.begin(), base_sorted.end(), better);
                std::vector<std::uint32_t> union_tokens;
                for (std::size_t index = 0; index < std::min(vocab, k + history_count); ++index) {
                    union_tokens.push_back(base_sorted[index].token);
                }
                union_tokens.insert(union_tokens.end(), history.begin(), history.end());

                auto adjusted = base;
                for (auto& candidate : adjusted) {
                    if (std::find(history.begin(), history.end(), candidate.token) != history.end()) {
                        candidate.logit = candidate.logit < 0.0F
                            ? candidate.logit * penalty : candidate.logit / penalty;
                    }
                }
                std::sort(adjusted.begin(), adjusted.end(), better);
                for (std::size_t index = 0; index < k; ++index) {
                    expect(
                        std::find(union_tokens.begin(), union_tokens.end(), adjusted[index].token)
                            != union_tokens.end(),
                        "sparse-change top-K escaped the K+U union"
                    );
                }
            }
        }
    }
}

void test_rejects_invalid_compact_sampler_inputs() {
    for (const auto options : {
        TokenSamplerOptions {.top_k = 0},
        TokenSamplerOptions {.top_k = 257},
        TokenSamplerOptions {.repetition_window = 65},
        TokenSamplerOptions {.temperature = -1.0F},
    }) {
        bool rejected = false;
        try { static_cast<void>(TokenSampler(options)); } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "invalid sampler product bounds must fail");
    }
    TokenSampler sampler;
    const std::array<TokenLogit, 1> invalid {{{0, std::numeric_limits<float>::quiet_NaN()}}};
    bool rejected = false;
    try { static_cast<void>(sampler.sample(invalid)); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "non-finite compact candidate must fail");
}

const TestCase kSamplingTests[] = {
    {"compact sampler stable ties and repetition", test_compact_sampler_stable_ties_and_repetition},
    {"compact sampler seed snapshot and filter order", test_compact_sampler_seed_snapshot_and_filter_order},
    {"repetition window eviction and promotion", test_repetition_window_eviction_and_promotion},
    {"sparse-change candidate superset property", test_sparse_change_candidate_superset_property},
    {"rejects invalid compact sampler inputs", test_rejects_invalid_compact_sampler_inputs},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> sampling_test_cases() { return kSamplingTests; }

}  // namespace local_gguf_engine::tests
