#include "local_gguf_engine/tensor/ops/reduction.hpp"

#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "validation.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::tensor::ops {

namespace {

[[nodiscard]] float read_f32_byte_offset(const TensorStorage& storage, std::size_t byte_offset) {
    float value = 0.0F;
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

[[nodiscard]] std::size_t require_supported_logits_rank(const TensorDescriptor& descriptor) {
    const auto rank = descriptor.shape.rank();
    if (rank < 1 || rank > 3) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            3,
            static_cast<std::uint64_t>(rank)
        );
    }

    return rank;
}

[[nodiscard]] std::uint64_t logits_vocab_size(
    const TensorStorage& logits,
    std::size_t rank
) {
    const auto vocab = logits.descriptor().shape.dimension(rank - 1U);
    if (vocab > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("F32 top-logit vocabulary is too large for uint32 token ids");
    }

    return vocab;
}

[[nodiscard]] std::uint64_t logits_last_row_offset(
    const TensorStorage& logits,
    std::size_t rank,
    std::uint64_t vocab
) {
    if (rank == 1U) {
        return 0;
    }

    const auto rows = logits.descriptor().element_count / vocab;
    return (rows - 1U) * vocab;
}

}  // namespace

float sum_f32(const TensorStorage& input) {
    require_f32_operand(input);

    float sum = 0.0F;
    for (std::uint64_t index = 0; index < input.descriptor().element_count; ++index) {
        sum += read_f32_byte_offset(input, static_cast<std::size_t>(index * sizeof(float)));
    }

    return sum;
}

float sum_squares_f32(const TensorStorage& input) {
    require_f32_operand(input);

    float sum = 0.0F;
    for (std::uint64_t index = 0; index < input.descriptor().element_count; ++index) {
        const auto value = read_f32_byte_offset(input, static_cast<std::size_t>(index * sizeof(float)));
        sum += value * value;
    }

    return sum;
}

float mean_square_f32(const TensorStorage& input) {
    require_f32_operand(input);
    return sum_squares_f32(input) / static_cast<float>(input.descriptor().element_count);
}

float rms_f32(const TensorStorage& input) {
    return std::sqrt(mean_square_f32(input));
}

TopLogit top_logit_f32(const TensorStorage& logits) {
    require_f32_operand(logits);
    const auto rank = require_supported_logits_rank(logits.descriptor());
    const auto vocab = logits_vocab_size(logits, rank);
    const auto row_offset = logits_last_row_offset(logits, rank, vocab);

    TopLogit best {
        .token = 0,
        .logit = read_f32_byte_offset(logits, static_cast<std::size_t>(row_offset * sizeof(float))),
    };
    for (std::uint64_t token = 1; token < vocab; ++token) {
        const auto value = read_f32_byte_offset(
            logits,
            static_cast<std::size_t>((row_offset + token) * sizeof(float))
        );
        if (value > best.logit) {
            best = TopLogit {
                .token = static_cast<std::uint32_t>(token),
                .logit = value,
            };
        }
    }

    return best;
}

}  // namespace local_gguf_engine::tensor::ops
