#pragma once

#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace local_gguf_engine::engine {
struct LlamaAttentionLayerBackendWeights;
struct LlamaAttentionLayerWeights;
struct LlamaBackendKVCache;
struct LlamaBlockBackendWeights;
struct LlamaBlockWeights;
struct LlamaFeedForwardLayerBackendWeights;
struct LlamaFeedForwardLayerWeights;
struct LlamaKVCache;
struct LlamaModelBlockBackendWeights;
struct LlamaModelLogitsBackendWeights;
struct LlamaModelWeights;
} // namespace local_gguf_engine::engine

namespace local_gguf_engine::tensor {
class BackendBuffer;
} // namespace local_gguf_engine::tensor

namespace local_gguf_engine::benchmarks {

struct IterationMeasurement {
    std::size_t bytes_processed = 0;
    std::uint64_t checksum = 0;
};

[[nodiscard]] std::uint64_t checksum_byte_span(std::span<const std::uint8_t> bytes) noexcept;
[[nodiscard]] std::uint64_t checksum_f32(float value) noexcept;
[[nodiscard]] std::size_t backend_buffer_byte_count(const tensor::BackendBuffer &buffer) noexcept;
[[nodiscard]] std::size_t
llama_attention_weight_bytes(const engine::LlamaAttentionLayerWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_attention_weight_bytes(const engine::LlamaAttentionLayerBackendWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_feed_forward_weight_bytes(const engine::LlamaFeedForwardLayerWeights &weights) noexcept;
[[nodiscard]] std::size_t llama_feed_forward_weight_bytes(
    const engine::LlamaFeedForwardLayerBackendWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_block_weight_bytes(const engine::LlamaBlockWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_block_weight_bytes(const engine::LlamaBlockBackendWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_model_weight_bytes(const engine::LlamaModelWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_model_block_weight_bytes(const engine::LlamaModelWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_model_block_weight_bytes(const engine::LlamaModelBlockBackendWeights &weights) noexcept;
[[nodiscard]] std::size_t
llama_model_weight_bytes(const engine::LlamaModelLogitsBackendWeights &weights) noexcept;
[[nodiscard]] std::size_t llama_kv_cache_bytes(const engine::LlamaKVCache &cache) noexcept;
[[nodiscard]] std::size_t
llama_backend_kv_cache_bytes(const engine::LlamaBackendKVCache &cache) noexcept;
[[nodiscard]] std::uint64_t
llama_kv_cache_fill_checksum(const engine::LlamaKVCache &cache) noexcept;
[[nodiscard]] std::uint64_t
llama_backend_kv_cache_fill_checksum(const engine::LlamaBackendKVCache &cache) noexcept;
[[nodiscard]] tensor::TensorStorage make_f32_storage(std::uint64_t element_count,
                                                     float start_value);
[[nodiscard]] tensor::TensorStorage make_f32_matrix(std::uint64_t rows, std::uint64_t columns,
                                                    float scale);
[[nodiscard]] tensor::TensorStorage make_f16_matrix(std::uint64_t rows, std::uint64_t columns,
                                                    float scale);
[[nodiscard]] tensor::TensorStorage make_f16_gguf_weight_matrix(std::uint64_t rows,
                                                                std::uint64_t columns, float scale);
[[nodiscard]] std::vector<std::uint8_t> make_repeated_payload(std::size_t byte_count);
[[nodiscard]] model_format::GGUFMetadataSection make_synthetic_tokenizer_metadata();

template <typename Callable>
[[nodiscard]] BenchmarkMeasurement measure(std::string name, std::size_t iterations,
                                           std::string detail, Callable callable) {
    BenchmarkMeasurement measurement{
        .name = std::move(name),
        .iterations = iterations,
        .total_ms = 0.0,
        .min_ms = std::numeric_limits<double>::max(),
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = std::move(detail),
    };

    for (std::size_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        const auto iteration = callable();
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        measurement.total_ms += elapsed_ms;
        measurement.min_ms = std::min(measurement.min_ms, elapsed_ms);
        measurement.max_ms = std::max(measurement.max_ms, elapsed_ms);
        measurement.bytes_processed += iteration.bytes_processed;
        measurement.checksum ^= iteration.checksum + index;
    }

    if (iterations == 0) {
        measurement.min_ms = 0.0;
    }

    return measurement;
}

} // namespace local_gguf_engine::benchmarks
