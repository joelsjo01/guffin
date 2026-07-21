#include "local_gguf_engine/engine/sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>

namespace local_gguf_engine::engine {

namespace {

[[nodiscard]] bool better_logit_candidate(const TokenLogit& left, const TokenLogit& right) noexcept {
    if (left.logit != right.logit) {
        return left.logit > right.logit;
    }
    return left.token < right.token;
}

[[nodiscard]] std::mt19937_64 make_sampler_rng(std::uint64_t seed) {
    if (seed != TokenSamplerOptions::random_seed) {
        return std::mt19937_64(seed);
    }

    std::random_device entropy;
    std::seed_seq random_seed {
        entropy(), entropy(), entropy(), entropy(),
        entropy(), entropy(), entropy(), entropy(),
    };
    return std::mt19937_64(random_seed);
}

}  // namespace

TokenSampler::TokenSampler(TokenSamplerOptions options)
    : options_(options), rng_(make_sampler_rng(options.seed)) {
    if (!std::isfinite(options_.temperature) || options_.temperature < 0.0F) {
        throw std::invalid_argument("sampling temperature must be finite and nonnegative");
    }
    if (options_.top_k == 0U || options_.top_k > token_sampler_max_top_k) {
        throw std::invalid_argument("sampling top_k must be in [1, 256]");
    }
    if (!std::isfinite(options_.top_p) || options_.top_p <= 0.0F || options_.top_p > 1.0F) {
        throw std::invalid_argument("sampling top_p must be in (0, 1]");
    }
    if (!std::isfinite(options_.min_p) || options_.min_p < 0.0F || options_.min_p > 1.0F) {
        throw std::invalid_argument("sampling min_p must be in [0, 1]");
    }
    if (!std::isfinite(options_.repetition_penalty) || options_.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("sampling repetition_penalty must be finite and positive");
    }
    if (options_.repetition_window > token_sampler_max_repetition_window) {
        throw std::invalid_argument("sampling repetition_window must be at most 64");
    }
    token_history_.reserve(token_sampler_max_repetition_window);
}

bool TokenSampler::is_greedy() const noexcept {
    return options_.temperature == 0.0F;
}

TokenSelectionRequest TokenSampler::selection_request() const noexcept {
    TokenSelectionRequest request;
    request.top_k = static_cast<std::uint32_t>(is_greedy() ? 1U : options_.top_k);

    if (options_.repetition_penalty != 1.0F) {
        for (const auto token : token_history_) {
            const auto begin = request.history_token_ids.begin();
            const auto end = begin + request.history_token_count;
            if (std::find(begin, end, token) == end) {
                request.history_token_ids[request.history_token_count++] = token;
            }
        }
    }
    request.base_candidate_count = request.top_k + request.history_token_count;
    return request;
}

std::uint32_t TokenSampler::sample(std::span<const TokenLogit> compact_candidates) {
    if (compact_candidates.empty() || compact_candidates.size() > token_sampler_max_candidates) {
        throw std::invalid_argument("compact sampling candidates must contain between 1 and 384 entries");
    }

    std::array<TokenLogit, token_sampler_max_candidates> candidates {};
    std::size_t candidate_count = 0;
    for (const auto candidate : compact_candidates) {
        if (!std::isfinite(candidate.logit)) {
            throw std::invalid_argument("compact sampling candidate logits must be finite");
        }
        auto existing = std::find_if(
            candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count),
            [&](const TokenLogit& value) { return value.token == candidate.token; }
        );
        if (existing == candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count)) {
            candidates[candidate_count++] = candidate;
        } else if (candidate.logit != existing->logit) {
            throw std::invalid_argument("duplicate compact sampling candidates have different logits");
        }
    }

    const auto request = selection_request();
    if (candidate_count < request.top_k) {
        throw std::invalid_argument("compact sampling candidate set is smaller than top_k");
    }
    for (std::size_t index = 0; index < candidate_count; ++index) {
        auto& candidate = candidates[index];
        if (std::find(
                request.history_token_ids.begin(),
                request.history_token_ids.begin() + request.history_token_count,
                candidate.token
            ) != request.history_token_ids.begin() + request.history_token_count) {
            candidate.logit = candidate.logit < 0.0F
                ? candidate.logit * options_.repetition_penalty
                : candidate.logit / options_.repetition_penalty;
        }
    }

    std::sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count), better_logit_candidate);
    candidate_count = std::min(candidate_count, static_cast<std::size_t>(request.top_k));
    if (is_greedy()) {
        return candidates.front().token;
    }

    std::array<double, token_sampler_max_top_k> weights {};
    const auto max_logit = candidates.front().logit;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        weights[index] = std::exp(static_cast<double>(candidates[index].logit - max_logit));
    }

    const auto total_weight = std::accumulate(weights.begin(), weights.begin() + candidate_count, 0.0);
    double retained_weight = 0.0;
    std::size_t retained_count = 0;
    do {
        retained_weight += weights[retained_count++];
    } while (retained_count < candidate_count
        && retained_weight / total_weight < static_cast<double>(options_.top_p));
    candidate_count = retained_count;

    if (options_.min_p > 0.0F) {
        std::size_t min_p_count = 1;
        while (min_p_count < candidate_count
            && weights[min_p_count] >= static_cast<double>(options_.min_p)) {
            ++min_p_count;
        }
        candidate_count = min_p_count;
    }

    for (std::size_t index = 0; index < candidate_count; ++index) {
        weights[index] = std::exp(
            static_cast<double>(candidates[index].logit - max_logit)
                / static_cast<double>(options_.temperature)
        );
    }
    const auto sampling_weight = std::accumulate(weights.begin(), weights.begin() + candidate_count, 0.0);
    std::uniform_real_distribution<double> distribution(0.0, sampling_weight);
    const auto draw = distribution(rng_);
    double cumulative = 0.0;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        cumulative += weights[index];
        if (draw < cumulative) {
            return candidates[index].token;
        }
    }
    return candidates[candidate_count - 1U].token;
}

void TokenSampler::observe_tokens(std::span<const std::uint32_t> token_ids) {
    if (options_.repetition_window == 0U) {
        token_history_.clear();
        return;
    }
    if (token_ids.size() >= options_.repetition_window) {
        token_history_.assign(
            token_ids.end() - static_cast<std::ptrdiff_t>(options_.repetition_window), token_ids.end()
        );
        return;
    }

    const auto retained_history = options_.repetition_window - token_ids.size();
    if (token_history_.size() > retained_history) {
        token_history_.erase(
            token_history_.begin(),
            token_history_.end() - static_cast<std::ptrdiff_t>(retained_history)
        );
    }
    token_history_.insert(token_history_.end(), token_ids.begin(), token_ids.end());
}

void TokenSampler::reset_history(std::span<const std::uint32_t> token_ids) {
    token_history_.clear();
    observe_tokens(token_ids);
}

}  // namespace local_gguf_engine::engine
