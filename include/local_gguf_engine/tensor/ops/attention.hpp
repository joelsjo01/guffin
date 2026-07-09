#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
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

[[nodiscard]] TensorStorage causal_grouped_attention_scores_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
);

[[nodiscard]] BackendBuffer causal_grouped_attention_scores_f32_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
);

[[nodiscard]] BackendBuffer causal_grouped_attention_scores_f32_with_kv_cache_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& cache_key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
);

[[nodiscard]] TensorStorage causal_grouped_attention_values_f32(
    const TensorStorage& probabilities,
    const TensorStorage& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
);

[[nodiscard]] BackendBuffer causal_grouped_attention_values_f32_to_backend(
    const BackendBuffer& probabilities,
    const BackendBuffer& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
);

[[nodiscard]] BackendBuffer causal_grouped_attention_values_f32_with_kv_cache_to_backend(
    const BackendBuffer& probabilities,
    const BackendBuffer& cache_value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
);

void write_kv_cache_f32_to_backend(
    const BackendBuffer& key,
    const BackendBuffer& value,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length
);

void write_prefill_kv_cache_f32_to_backend(
    const BackendBuffer& key,
    const BackendBuffer& value,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length
);

}  // namespace local_gguf_engine::tensor::ops
