#include "local_gguf_engine/engine/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::engine {

namespace {

using tensor::DType;
using tensor::TensorDescriptor;
using tensor::TensorError;
using tensor::TensorErrorCode;
using tensor::TensorStorage;

constexpr std::size_t vector_rank = 1;
constexpr std::size_t matrix_rank = 2;

[[nodiscard]] std::uint64_t dtype_value(DType dtype) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype));
}

void require_f32_logits(const TensorDescriptor& descriptor) {
    if (descriptor.dtype != DType::f32) {
        throw TensorError(TensorErrorCode::unsupported_operator_dtype, dtype_value(DType::f32), dtype_value(descriptor.dtype));
    }
}

[[nodiscard]] std::uint64_t require_supported_logits_rank(const TensorDescriptor& descriptor) {
    const auto rank = descriptor.shape.rank();
    if (rank != vector_rank && rank != matrix_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(matrix_rank),
            static_cast<std::uint64_t>(rank)
        );
    }

    return static_cast<std::uint64_t>(rank);
}

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

[[nodiscard]] std::uint64_t logits_vocab_size(const TensorStorage& logits, std::uint64_t rank) {
    return rank == vector_rank
        ? logits.descriptor().shape.dimension(0)
        : logits.descriptor().shape.dimension(1);
}

[[nodiscard]] std::uint64_t logits_last_row_offset(const TensorStorage& logits, std::uint64_t rank, std::uint64_t vocab) {
    return rank == vector_rank
        ? 0
        : (logits.descriptor().shape.dimension(0) - 1) * vocab;
}

[[nodiscard]] bool better_logit_candidate(const TokenLogit& left, const TokenLogit& right) noexcept {
    if (left.logit != right.logit) {
        return left.logit > right.logit;
    }

    return left.token < right.token;
}

[[nodiscard]] std::vector<TokenLogit> collect_last_row_logits(const TensorStorage& logits) {
    require_f32_logits(logits.descriptor());
    const auto rank = require_supported_logits_rank(logits.descriptor());
    const auto vocab = logits_vocab_size(logits, rank);
    if (vocab > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("logit vocabulary is too large for uint32 token ids");
    }

    const auto row_offset = logits_last_row_offset(logits, rank, vocab);
    std::vector<TokenLogit> candidates;
    candidates.reserve(static_cast<std::size_t>(vocab));
    for (std::uint64_t token = 0; token < vocab; ++token) {
        const auto logit = read_f32_linear(logits, row_offset + token);
        if (!std::isfinite(logit)) {
            throw std::invalid_argument(
                "sampling logit for token " + std::to_string(token) + " must be finite"
            );
        }

        candidates.push_back(TokenLogit {
            .token = static_cast<std::uint32_t>(token),
            .logit = logit,
        });
    }

    return candidates;
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
    if (!std::isfinite(options_.top_p) || options_.top_p <= 0.0F || options_.top_p > 1.0F) {
        throw std::invalid_argument("sampling top_p must be in (0, 1]");
    }
    if (!std::isfinite(options_.min_p) || options_.min_p < 0.0F || options_.min_p > 1.0F) {
        throw std::invalid_argument("sampling min_p must be in [0, 1]");
    }
    if (
        !std::isfinite(options_.repetition_penalty)
        || options_.repetition_penalty <= 0.0F
    ) {
        throw std::invalid_argument("sampling repetition_penalty must be finite and positive");
    }
}

bool TokenSampler::is_greedy() const noexcept {
    return options_.temperature == 0.0F;
}

std::uint32_t TokenSampler::sample(const TensorStorage& logits) {
    auto candidates = collect_last_row_logits(logits);
    if (candidates.empty()) {
        throw std::invalid_argument("sampling logits vocabulary must not be empty");
    }

    for (auto& candidate : candidates) {
        if (
            options_.repetition_penalty != 1.0F
            && std::find(token_history_.begin(), token_history_.end(), candidate.token)
                != token_history_.end()
        ) {
            candidate.logit = candidate.logit < 0.0F
                ? candidate.logit * options_.repetition_penalty
                : candidate.logit / options_.repetition_penalty;
        }
    }
    std::sort(candidates.begin(), candidates.end(), better_logit_candidate);
    if (is_greedy()) {
        return candidates.front().token;
    }

    if (options_.top_k > 0U && candidates.size() > options_.top_k) {
        candidates.resize(options_.top_k);
    }

    const auto max_logit = candidates.front().logit;
    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        weights.push_back(std::exp(static_cast<double>(candidate.logit - max_logit)));
    }

    // These filters are deliberately non-commutative. Apply them to the
    // untempered distribution in the runtime's public transform order:
    // top-k, top-p, min-p, then temperature and random selection.
    const auto total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    double retained_weight = 0.0;
    std::size_t retained_count = 0;
    do {
        retained_weight += weights[retained_count];
        ++retained_count;
    } while (
        retained_count < weights.size()
        && retained_weight / total_weight < static_cast<double>(options_.top_p)
    );
    candidates.resize(retained_count);
    weights.resize(retained_count);

    const auto min_weight = static_cast<double>(options_.min_p);
    if (min_weight > 0.0) {
        auto min_p_count = std::size_t {1};
        while (min_p_count < weights.size() && weights[min_p_count] >= min_weight) {
            ++min_p_count;
        }
        candidates.resize(min_p_count);
        weights.resize(min_p_count);
    }

    weights.clear();
    weights.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        weights.push_back(std::exp(
            static_cast<double>(candidate.logit - max_logit)
                / static_cast<double>(options_.temperature)
        ));
    }
    const auto sampling_weight = std::accumulate(weights.begin(), weights.end(), 0.0);

    std::uniform_real_distribution<double> distribution(0.0, sampling_weight);
    const auto draw = distribution(rng_);
    double cumulative = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        cumulative += weights[index];
        if (draw < cumulative) {
            return candidates[index].token;
        }
    }
    return candidates.back().token;
}

void TokenSampler::observe_tokens(std::span<const std::uint32_t> token_ids) {
    if (options_.repetition_window == 0U) {
        token_history_.clear();
        return;
    }
    if (token_ids.size() >= options_.repetition_window) {
        token_history_.assign(
            token_ids.end() - static_cast<std::ptrdiff_t>(options_.repetition_window),
            token_ids.end()
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

std::vector<TokenLogit> top_logit_tokens_f32(const TensorStorage& logits, std::size_t max_tokens) {
    auto candidates = collect_last_row_logits(logits);
    if (max_tokens >= candidates.size()) {
        std::sort(candidates.begin(), candidates.end(), better_logit_candidate);
        return candidates;
    }

    const auto selected_end = candidates.begin() + static_cast<std::ptrdiff_t>(max_tokens);
    std::partial_sort(candidates.begin(), selected_end, candidates.end(), better_logit_candidate);
    candidates.resize(max_tokens);
    return candidates;
}

}  // namespace local_gguf_engine::engine
