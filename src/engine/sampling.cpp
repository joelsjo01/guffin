#include "local_gguf_engine/engine/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
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
constexpr std::uint64_t splitmix64_increment = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix64_multiplier_0 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix64_multiplier_1 = 0x94D049BB133111EBULL;
constexpr double unit_float_scale = 1.0 / 9007199254740992.0;

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

void require_sampling_config(const SamplingConfig& config) {
    if (config.temperature < 0.0F || !std::isfinite(config.temperature)) {
        throw std::invalid_argument("sampling temperature must be zero or a positive finite value");
    }
}

[[nodiscard]] double next_unit_float(SamplingState& state) noexcept {
    return static_cast<double>(state.next_u64() >> 11U) * unit_float_scale;
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
            throw std::invalid_argument("sampling logits must be finite");
        }

        candidates.push_back(TokenLogit {
            .token = static_cast<std::uint32_t>(token),
            .logit = logit,
        });
    }

    return candidates;
}

}  // namespace

SamplingState::SamplingState(std::uint64_t seed) noexcept : state_(seed) {}

std::uint64_t SamplingState::next_u64() noexcept {
    state_ += splitmix64_increment;
    auto value = state_;
    value = (value ^ (value >> 30U)) * splitmix64_multiplier_0;
    value = (value ^ (value >> 27U)) * splitmix64_multiplier_1;
    return value ^ (value >> 31U);
}

std::uint32_t sample_argmax_token_f32(const TensorStorage& logits) {
    require_f32_logits(logits.descriptor());
    const auto rank = require_supported_logits_rank(logits.descriptor());
    const auto vocab = logits_vocab_size(logits, rank);

    if (vocab > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("logit vocabulary is too large for uint32 token ids");
    }

    const auto row_offset = logits_last_row_offset(logits, rank, vocab);
    std::uint32_t best_token = 0;
    auto best_value = read_f32_linear(logits, row_offset);
    for (std::uint64_t token = 1; token < vocab; ++token) {
        const auto value = read_f32_linear(logits, row_offset + token);
        if (value > best_value) {
            best_value = value;
            best_token = static_cast<std::uint32_t>(token);
        }
    }

    return best_token;
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

std::uint32_t sample_token_f32(
    const TensorStorage& logits,
    const SamplingConfig& config,
    SamplingState& state
) {
    require_sampling_config(config);
    if (config.temperature == 0.0F) {
        return sample_argmax_token_f32(logits);
    }

    auto candidates = collect_last_row_logits(logits);

    if (config.top_k > 0 && config.top_k < candidates.size()) {
        const auto selected_begin = candidates.begin();
        const auto selected_end = candidates.begin() + static_cast<std::ptrdiff_t>(config.top_k);
        std::partial_sort(selected_begin, selected_end, candidates.end(), better_logit_candidate);
        candidates.resize(static_cast<std::size_t>(config.top_k));
    }

    auto max_logit = candidates.front().logit;
    for (const auto& candidate : candidates) {
        max_logit = std::max(max_logit, candidate.logit);
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total_weight = 0.0;
    // Subtracting max_logit keeps temperature softmax finite even for large raw logits.
    for (const auto& candidate : candidates) {
        const auto weight = std::exp(static_cast<double>(candidate.logit - max_logit) / config.temperature);
        weights.push_back(weight);
        total_weight += weight;
    }

    const auto draw = next_unit_float(state) * total_weight;
    double cumulative_weight = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        cumulative_weight += weights[index];
        if (draw < cumulative_weight) {
            return candidates[index].token;
        }
    }

    return candidates.back().token;
}

}  // namespace local_gguf_engine::engine
