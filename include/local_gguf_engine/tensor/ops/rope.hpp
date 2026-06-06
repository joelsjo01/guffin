#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage rope_f32(
    const TensorStorage& input,
    std::uint64_t position,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

}  // namespace local_gguf_engine::tensor::ops
