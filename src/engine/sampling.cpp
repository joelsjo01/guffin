#include "local_gguf_engine/engine/sampling.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

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

}  // namespace

std::uint32_t sample_argmax_token_f32(const TensorStorage& logits) {
    require_f32_logits(logits.descriptor());
    const auto rank = require_supported_logits_rank(logits.descriptor());
    const auto vocab = rank == vector_rank
        ? logits.descriptor().shape.dimension(0)
        : logits.descriptor().shape.dimension(1);

    if (vocab > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("logit vocabulary is too large for uint32 token ids");
    }

    const auto row_offset = rank == vector_rank
        ? 0
        : (logits.descriptor().shape.dimension(0) - 1) * vocab;
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

}  // namespace local_gguf_engine::engine
