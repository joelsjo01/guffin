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

struct F32RoPEMatrixColumnView {
    F32MatrixColumnView columns;
    std::uint64_t rope_dimension_count = 0;
    float frequency_base = 0.0F;
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

[[nodiscard]] TensorStorage causal_scaled_dot_product_attention_f32(
    F32RoPEMatrixColumnView query,
    F32MatrixColumnView key,
    F32MatrixColumnView value,
    float scale,
    std::uint64_t query_position_start
);

}  // namespace local_gguf_engine::tensor::ops
