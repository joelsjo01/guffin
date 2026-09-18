#pragma once

#include "local_gguf_engine/runtime/sha256.hpp"

#include <array>
#include <cstdint>

namespace local_gguf_engine::engine {

// This payload describes the registered direct-token gated residual model.
// Its recurrent state is not a transformer K/V cache. Other architecture
// families require their own payload and state ownership.
enum class RecurrentTensor : std::uint32_t {
    fusion, fusion_norm, token_projection, gate, gate_bias,
    candidate, candidate_bias, target_projection, confidence, confidence_bias,
    count,
};
inline constexpr std::size_t recurrent_tensor_count = 10U;
inline constexpr std::size_t recurrent_provenance_count = 12U;
// Provenance order: checkpoint, training receipt, registry, run contract,
// training code, trace corpus, split receipt, environment, feature contract,
// tokenizer, chat template, exporter source.
struct RecurrentWeightTensor {
    std::uint32_t rows = 0U;
    std::uint32_t columns = 0U;
    std::uint64_t offset = 0U;
    std::uint64_t bytes = 0U;
    runtime::Sha256Digest digest;
    constexpr bool operator==(const RecurrentWeightTensor&) const = default;
};

struct LearnedRecurrentSpec {
    std::uint32_t version = 0U;
    // 1 = weights only; no fitted routing or calibration policy exists yet.
    std::uint32_t role = 1U;
    // 1 = F16 captured final tap, F32 master weights, BF16 linear operands/results and recurrent
    // elementwise rounding, F32 Gemma norm and F16 full-vocabulary table.
    std::uint32_t numeric_contract = 1U;
    std::uint32_t width = 0U;
    std::uint32_t horizon = 0U;
    std::uint32_t top_k = 0U;
    float temperature = 0.0F;
    float top_p = 0.0F;
    float min_p = 0.0F;
    std::array<runtime::Sha256Digest, recurrent_provenance_count> provenance {};
    std::array<RecurrentWeightTensor, recurrent_tensor_count> tensors {};
    constexpr bool operator==(const LearnedRecurrentSpec&) const = default;
};

}  // namespace local_gguf_engine::engine
