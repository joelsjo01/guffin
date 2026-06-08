#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor::ops {

struct F32MatrixColumnView {
    const TensorStorage& storage;
    std::uint64_t row_count = 0;
    std::uint64_t first_column = 0;
    std::uint64_t column_count = 0;
};

[[nodiscard]] TensorStorage scaled_dot_product_attention_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    const TensorStorage& value,
    float scale
);

[[nodiscard]] TensorStorage causal_scaled_dot_product_attention_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    const TensorStorage& value,
    float scale,
    std::uint64_t query_position_start
);

[[nodiscard]] TensorStorage causal_scaled_dot_product_attention_f32(
    const TensorStorage& query,
    F32MatrixColumnView key,
    F32MatrixColumnView value,
    float scale,
    std::uint64_t query_position_start
);

}  // namespace local_gguf_engine::tensor::ops
