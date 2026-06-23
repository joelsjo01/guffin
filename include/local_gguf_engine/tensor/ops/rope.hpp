#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage rope_f32(
    const TensorStorage& input,
    std::uint64_t position,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

[[nodiscard]] TensorStorage rope_f32_with_position_start(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

[[nodiscard]] TensorStorage rope_f32_by_heads_with_position_start(
    const TensorStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

[[nodiscard]] BackendBuffer rope_f32_by_heads_with_position_start_to_backend(
    const BackendBuffer& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

}  // namespace local_gguf_engine::tensor::ops
