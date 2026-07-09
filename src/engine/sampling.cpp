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
