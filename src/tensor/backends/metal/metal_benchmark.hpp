#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <array>
#include <cstdint>

namespace local_gguf_engine::tensor::backends::metal {

void benchmark_noop_dispatch();

[[nodiscard]] BackendBuffer benchmark_llama_cached_decode_attention_norm_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_norm_weight,
    float rms_norm_epsilon
);

[[nodiscard]] std::array<BackendBuffer, 3> benchmark_llama_cached_decode_qkv_projection_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& query_weight,
    const BackendBuffer& key_weight,
    const BackendBuffer& value_weight
);

void benchmark_llama_cached_decode_kv_append_to_backend(
    const BackendBuffer& key,
    const BackendBuffer& value,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base
);

[[nodiscard]] std::array<BackendBuffer, 3> benchmark_llama_cached_decode_attention_frontend_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_norm_weight,
    const BackendBuffer& query_weight,
    const BackendBuffer& key_weight,
    const BackendBuffer& value_weight,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float rms_norm_epsilon
);

[[nodiscard]] BackendBuffer benchmark_llama_cached_decode_attention_output_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& cache_key,
    const BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
);

[[nodiscard]] BackendBuffer benchmark_llama_cached_decode_block_tail_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    float rms_norm_epsilon
);

[[nodiscard]] BackendBuffer benchmark_llama_cached_decode_attention_residual_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight
);

[[nodiscard]] BackendBuffer benchmark_llama_cached_decode_ffn_gate_up_to_backend(
    const BackendBuffer& attention_residual,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    float rms_norm_epsilon
);

[[nodiscard]] BackendBuffer benchmark_llama_cached_decode_ffn_down_to_backend(
    const BackendBuffer& gated,
    const BackendBuffer& attention_residual,
    const BackendBuffer& down_weight
);

}  // namespace local_gguf_engine::tensor::backends::metal
