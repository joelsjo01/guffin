#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace local_gguf_engine::tensor::backends::metal {

void benchmark_noop_dispatch();

std::array<local_gguf_engine::tensor::BackendBuffer, 3>
benchmark_llama_cached_decode_attention_frontend_to_backend(
    const local_gguf_engine::tensor::BackendBuffer& input,
    const local_gguf_engine::tensor::BackendBuffer& attention_norm_weight,
    const local_gguf_engine::tensor::BackendBuffer& query_weight,
    const local_gguf_engine::tensor::BackendBuffer& key_weight,
    const local_gguf_engine::tensor::BackendBuffer& value_weight,
    local_gguf_engine::tensor::BackendBuffer& cache_key,
    local_gguf_engine::tensor::BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float rms_norm_epsilon
);

local_gguf_engine::tensor::BackendBuffer benchmark_llama_cached_decode_attention_output_to_backend(
    const local_gguf_engine::tensor::BackendBuffer& query,
    const local_gguf_engine::tensor::BackendBuffer& cache_key,
    const local_gguf_engine::tensor::BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
);

local_gguf_engine::tensor::BackendBuffer benchmark_llama_cached_decode_block_tail_to_backend(
    const local_gguf_engine::tensor::BackendBuffer& input,
    const local_gguf_engine::tensor::BackendBuffer& attention_output,
    const local_gguf_engine::tensor::BackendBuffer& attention_output_weight,
    const local_gguf_engine::tensor::BackendBuffer& feed_forward_norm_weight,
    const local_gguf_engine::tensor::BackendBuffer& gate_weight,
    const local_gguf_engine::tensor::BackendBuffer& up_weight,
    const local_gguf_engine::tensor::BackendBuffer& down_weight,
    float rms_norm_epsilon
);

}  // namespace local_gguf_engine::tensor::backends::metal

namespace local_gguf_engine::benchmarks {

namespace {

using Clock = std::chrono::steady_clock;
using engine::copy_llama_model_block_weights_to_backend;
using engine::copy_llama_model_logits_weights_to_backend;
using engine::create_llama_backend_kv_cache;
using engine::create_llama_kv_cache;
using engine::generate_llama_argmax_tokens_f32;
using engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using engine::LlamaAttentionLayerBackendWeights;
using engine::LlamaAttentionLayerWeights;
using engine::LlamaBlockBackendWeights;
using engine::LlamaBlockWeights;
using engine::LlamaFeedForwardLayerBackendWeights;
using engine::LlamaFeedForwardLayerWeights;
using engine::LlamaBackendKVCache;
using engine::LlamaKVCache;
using engine::LlamaModelBlockBackendWeights;
using engine::LlamaModelConfig;
using engine::LlamaModelLogitsBackendWeights;
using engine::LlamaModelWeights;
using engine::load_engine_model_from_gguf;
using engine::run_llama_attention_layer_f32;
using engine::run_llama_block_f32;
using engine::run_llama_feed_forward_layer_f32;
using engine::run_llama_model_blocks_f32;
using engine::run_llama_model_blocks_with_kv_cache_f32;
using engine::run_llama_model_logits_f32;
using engine::run_llama_model_logits_with_kv_cache_f32;
using engine::run_llama_model_next_token_argmax_f32;
using engine::run_llama_model_next_token_argmax_with_kv_cache_f32;
using engine::run_llama_model_next_token_sampled_f32;
using engine::SamplingConfig;
using engine::SamplingState;
using model_format::load_gguf_tensor_payloads;
using model_format::GGUFHeader;
using model_format::GGUFMetadataArray;
using model_format::GGUFMetadataEntry;
using model_format::GGUFMetadataSection;
using model_format::GGUFMetadataValue;
using model_format::GGUFMetadataValueType;
using model_format::read_gguf_model_view;
using runtime::BackendKind;
using tensor::BackendBuffer;
using tensor::copy_tensor_to_backend;
using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorStorage;
using tensor::f32_to_f16_bits;
using tensor::tensor_backend_is_available;
using tensor::tensor_backend_unavailable_reason;
using tensor::write_f16_from_f32;
using tensor::write_f32;
using tensor::ops::add_f32;
using tensor::ops::causal_scaled_dot_product_attention_f32;
using tensor::ops::matmul_f32;
using tensor::ops::matmul_f32_f16;
using tensor::ops::matmul_f32_f16_gguf_weight;
using tensor::ops::multiply_f32;
using tensor::ops::rms_norm_f32;
using tensor::ops::rope_f32;
using tensor::ops::rope_f32_with_position_start;
using tensor::ops::scaled_dot_product_attention_f32;
using tensor::ops::silu_f32;
using tensor::ops::softmax_f32;
using tensor::ops::sum_f32;
using tokenizer::load_tokenizer_from_gguf_metadata;
using tokenizer::TokenizerDecodeOptions;
using tokenizer::TokenizerEncodeOptions;

struct IterationMeasurement {
    std::size_t bytes_processed = 0;
    std::uint64_t checksum = 0;
};

[[nodiscard]] std::uint64_t checksum_byte_span(std::span<const std::uint8_t> bytes) noexcept {
    if (bytes.empty()) {
        return 0;
    }

    return static_cast<std::uint64_t>(bytes.size())
        ^ static_cast<std::uint64_t>(bytes.front())
        ^ (static_cast<std::uint64_t>(bytes.back()) << 8);
}

[[nodiscard]] std::uint64_t checksum_f32(float value) noexcept {
    return static_cast<std::uint64_t>(value >= 0.0F ? value * 1000.0F : -value * 1000.0F);
}

template <typename Callable>
[[nodiscard]] BenchmarkMeasurement measure(
    std::string name,
    std::size_t iterations,
    std::string detail,
    Callable callable
) {
    BenchmarkMeasurement measurement {
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
        const auto start = Clock::now();
        const auto iteration = callable();
        const auto end = Clock::now();
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

[[nodiscard]] TensorStorage make_f32_storage(std::uint64_t element_count, float start_value) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({element_count}), DType::f32));
    for (std::uint64_t index = 0; index < element_count; ++index) {
        const std::array<std::uint64_t, 1> tensor_index {index};
        write_f32(storage, std::span<const std::uint64_t>(tensor_index), start_value + static_cast<float>(index));
    }

    return storage;
}

[[nodiscard]] TensorStorage make_f32_matrix(std::uint64_t rows, std::uint64_t columns, float scale) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            const auto value = scale * static_cast<float>((row + 1) + (column % 7));
            write_f32(storage, std::span<const std::uint64_t>(index), value);
        }
    }

    return storage;
}

[[nodiscard]] TensorStorage make_f16_matrix(std::uint64_t rows, std::uint64_t columns, float scale) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            const auto value = scale * static_cast<float>((row + 1) + (column % 7));
            write_f16_from_f32(storage, std::span<const std::uint64_t>(index), value);
        }
    }

    return storage;
}

[[nodiscard]] TensorStorage make_f16_gguf_weight_matrix(
    std::uint64_t rows,
    std::uint64_t columns,
    float scale
) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    auto bytes = storage.mutable_bytes();
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto value = scale * static_cast<float>((row % 17) + 1 + (column % 11));
            const auto bits = f32_to_f16_bits(value);
            const auto byte_offset = static_cast<std::size_t>((column * rows + row) * sizeof(std::uint16_t));
            bytes[byte_offset] = static_cast<std::uint8_t>(bits & 0xFFU);
            bytes[byte_offset + 1] = static_cast<std::uint8_t>(bits >> 8U);
        }
    }

    return storage;
}

[[nodiscard]] std::vector<std::uint8_t> make_repeated_payload(std::size_t byte_count) {
    std::vector<std::uint8_t> bytes(byte_count);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(index % 251);
    }

    return bytes;
}

[[nodiscard]] GGUFMetadataEntry metadata_string(std::string key, std::string value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::string,
            .data = std::move(value),
        },
    };
}

[[nodiscard]] GGUFMetadataEntry metadata_u32(std::string key, std::uint32_t value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::uint32,
            .data = value,
        },
    };
}

[[nodiscard]] GGUFMetadataEntry metadata_bool(std::string key, bool value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::boolean,
            .data = value,
        },
    };
}

[[nodiscard]] GGUFMetadataEntry metadata_string_array(std::string key, std::vector<std::string> values) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::array,
            .data = GGUFMetadataArray {
                .element_type = GGUFMetadataValueType::string,
                .values = std::move(values),
            },
        },
    };
}

[[nodiscard]] GGUFMetadataEntry metadata_f32_array(std::string key, std::vector<float> values) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::array,
            .data = GGUFMetadataArray {
                .element_type = GGUFMetadataValueType::float32,
                .values = std::move(values),
            },
        },
    };
}

[[nodiscard]] GGUFMetadataEntry metadata_i32_array(std::string key, std::vector<std::int32_t> values) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::array,
            .data = GGUFMetadataArray {
                .element_type = GGUFMetadataValueType::int32,
                .values = std::move(values),
            },
        },
    };
}

[[nodiscard]] GGUFMetadataSection make_synthetic_tokenizer_metadata() {
    std::vector<GGUFMetadataEntry> entries {
        metadata_string("general.architecture", "llama"),
        metadata_u32("llama.vocab_size", 7),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array(
            "tokenizer.ggml.tokens",
            {"<s>", "</s>", "Hello", " world", "!", "Hello world", "<unk>"}
        ),
        metadata_f32_array(
            "tokenizer.ggml.scores",
            {0.0F, 0.0F, -1.0F, -1.1F, -1.2F, -0.2F, -10.0F}
        ),
        metadata_i32_array("tokenizer.ggml.token_type", {3, 3, 1, 1, 1, 1, 2}),
        metadata_u32("tokenizer.ggml.bos_token_id", 0),
        metadata_u32("tokenizer.ggml.eos_token_id", 1),
        metadata_u32("tokenizer.ggml.unknown_token_id", 6),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
    };
    const auto metadata_count = static_cast<std::uint64_t>(entries.size());

    return GGUFMetadataSection {
        .header = GGUFHeader {
            .version = 3,
            .tensor_count = 0,
            .metadata_kv_count = metadata_count,
        },
        .entries = std::move(entries),
        .encoded_byte_count = 0,
    };
}

[[nodiscard]] std::string projection_benchmark_detail(
    const TensorStorage& left,
    const TensorStorage& right,
    std::string_view backend,
    std::string_view cost_scope
) {
    std::ostringstream stream;
    stream << "left=[" << left.descriptor().shape.dimension(0)
           << "," << left.descriptor().shape.dimension(1)
           << "] right=[" << right.descriptor().shape.dimension(0)
           << "," << right.descriptor().shape.dimension(1)
           << "] dtype=F32/F16 layout=gguf-weight backend=" << backend
           << " cost=" << cost_scope;
    return stream.str();
}

void add_projection_backend_benchmarks(
    BenchmarkReport& report,
    std::string_view label,
    const TensorStorage& left,
    const TensorStorage& right,
    bool metal_available
) {
    const auto cpu_right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);
    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.cpu." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(left, right, "cpu", "compute+allocation"),
        [&left, &right]() {
            const auto output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.cpu_buffered." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(left, right, "cpu", "compute+allocation preloaded-right"),
        [&left, cpu_right_buffer]() {
            const auto output = matmul_f32_f16_gguf_weight(left, cpu_right_buffer);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + cpu_right_buffer.descriptor().byte_count
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    if (!metal_available) {
        return;
    }

    const auto metal_right_buffer = copy_tensor_to_backend(right, BackendKind::metal);
    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.metal." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(
            left,
            right,
            "metal",
            "input-copy+dispatch+output-copy cached-pipeline"
        ),
        [&left, &right]() {
            const auto output = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.metal_buffered." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(
            left,
            right,
            "metal",
            "input-copy+dispatch+output-copy preloaded-right cached-pipeline"
        ),
        [&left, metal_right_buffer]() {
            const auto output = matmul_f32_f16_gguf_weight(left, metal_right_buffer);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + metal_right_buffer.descriptor().byte_count
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));
}

void add_unavailable_metal_projection_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "tensor.matmul_f32_f16_gguf_weight.metal.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_model_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_model_logits_next_token_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_cached_model_block_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_model_blocks_with_kv_cache_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_backend_kv_cached_model_block_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_model_blocks_with_backend_kv_cache_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_attention_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_attention_layer_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_feed_forward_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_feed_forward_layer_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

[[nodiscard]] std::size_t backend_buffer_byte_count(const tensor::BackendBuffer& buffer) noexcept {
    return buffer.descriptor().byte_count;
}

[[nodiscard]] std::size_t llama_attention_weight_bytes(const LlamaAttentionLayerWeights& weights) noexcept {
    return weights.attention_norm_weight.bytes().size()
        + weights.query_weight.bytes().size()
        + weights.key_weight.bytes().size()
        + weights.value_weight.bytes().size()
        + weights.output_weight.bytes().size();
}

[[nodiscard]] std::size_t llama_attention_weight_bytes(const LlamaAttentionLayerBackendWeights& weights) noexcept {
    return backend_buffer_byte_count(weights.attention_norm_weight_buffer)
        + backend_buffer_byte_count(weights.query_weight)
        + backend_buffer_byte_count(weights.key_weight)
        + backend_buffer_byte_count(weights.value_weight)
        + backend_buffer_byte_count(weights.output_weight);
}

[[nodiscard]] std::size_t llama_feed_forward_weight_bytes(const LlamaFeedForwardLayerWeights& weights) noexcept {
    return weights.feed_forward_norm_weight.bytes().size()
        + weights.gate_weight.bytes().size()
        + weights.up_weight.bytes().size()
        + weights.down_weight.bytes().size();
}

[[nodiscard]] std::size_t llama_feed_forward_weight_bytes(
    const LlamaFeedForwardLayerBackendWeights& weights
) noexcept {
    return backend_buffer_byte_count(weights.feed_forward_norm_weight_buffer)
        + backend_buffer_byte_count(weights.gate_weight)
        + backend_buffer_byte_count(weights.up_weight)
        + backend_buffer_byte_count(weights.down_weight);
}

[[nodiscard]] std::size_t llama_block_weight_bytes(const LlamaBlockWeights& weights) noexcept {
    return llama_attention_weight_bytes(weights.attention)
        + llama_feed_forward_weight_bytes(weights.feed_forward);
}

[[nodiscard]] std::size_t llama_block_weight_bytes(const LlamaBlockBackendWeights& weights) noexcept {
    return llama_attention_weight_bytes(weights.attention)
        + llama_feed_forward_weight_bytes(weights.feed_forward);
}

[[nodiscard]] std::size_t llama_model_weight_bytes(const LlamaModelWeights& weights) noexcept {
    std::size_t bytes = weights.token_embedding_weight.bytes().size()
        + weights.output_norm_weight.bytes().size()
        + weights.output_weight.bytes().size();
    for (const auto& block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

[[nodiscard]] std::size_t llama_model_block_weight_bytes(const LlamaModelWeights& weights) noexcept {
    std::size_t bytes = 0;
    for (const auto& block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

[[nodiscard]] std::size_t llama_model_block_weight_bytes(const LlamaModelBlockBackendWeights& weights) noexcept {
    std::size_t bytes = 0;
    for (const auto& block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

[[nodiscard]] std::size_t llama_model_weight_bytes(const LlamaModelLogitsBackendWeights& weights) noexcept {
    std::size_t bytes = weights.token_embedding_weight.bytes().size()
        + weights.output_norm_weight.bytes().size()
        + backend_buffer_byte_count(weights.output_weight);
    for (const auto& block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

[[nodiscard]] std::size_t llama_kv_cache_bytes(const LlamaKVCache& cache) noexcept {
    std::size_t bytes = 0;
    for (const auto& layer : cache.layers) {
        bytes += layer.key.bytes().size() + layer.value.bytes().size();
    }
    return bytes;
}

[[nodiscard]] std::size_t llama_backend_kv_cache_bytes(const LlamaBackendKVCache& cache) noexcept {
    std::size_t bytes = 0;
    for (const auto& layer : cache.layers) {
        bytes += backend_buffer_byte_count(layer.key) + backend_buffer_byte_count(layer.value);
    }
    return bytes;
}

[[nodiscard]] std::uint64_t llama_kv_cache_fill_checksum(const LlamaKVCache& cache) noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t layer_index = 0; layer_index < cache.layers.size(); ++layer_index) {
        checksum ^= cache.layers[layer_index].filled_token_count << (layer_index % 8U);
    }
    return checksum;
}

[[nodiscard]] std::uint64_t llama_backend_kv_cache_fill_checksum(const LlamaBackendKVCache& cache) noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t layer_index = 0; layer_index < cache.layers.size(); ++layer_index) {
        checksum ^= cache.layers[layer_index].filled_token_count << (layer_index % 8U);
    }
    return checksum;
}

void warm_up_metal_projection_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto left = make_f32_matrix(1, 4, 0.01F);
    const auto right = make_f16_gguf_weight_matrix(4, 4, 0.02F);
    (void)matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
}

[[nodiscard]] std::string focused_resident_kv_decode_detail(
    std::string_view stage,
    std::string_view backend,
    std::uint64_t cache_length,
    std::uint64_t blocks,
    std::string_view cost_scope
) {
    std::ostringstream stream;
    stream << "input=[1,576] blocks=" << blocks
           << " heads=9 kv_heads=3 key=64 value=64 ffn=1536 backend=" << backend
           << " resident-kv cache_prefilled=" << cache_length
           << " stage=" << stage
           << " cost=" << cost_scope;
    return stream.str();
}

[[nodiscard]] LlamaBackendKVCache make_prefilled_focused_backend_kv_cache(
    const LlamaModelConfig& config,
    BackendKind backend,
    std::uint64_t filled_token_count,
    const TensorStorage& key_storage,
    const TensorStorage& value_storage
) {
    auto cache = create_llama_backend_kv_cache(config, backend);
    for (auto& layer : cache.layers) {
        layer.key = copy_tensor_to_backend(key_storage, backend);
        layer.value = copy_tensor_to_backend(value_storage, backend);
        layer.filled_token_count = filled_token_count;
    }
    return cache;
}

void add_unavailable_metal_focused_resident_kv_decode_stage_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_cached_decode_stage_sweep.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_focused_resident_kv_decode_stage_benchmarks(BenchmarkReport& report, bool metal_available) {
    constexpr std::uint64_t hidden = 576;
    constexpr std::uint64_t feed_forward = 1536;
    constexpr std::uint64_t head_count = 9;
    constexpr std::uint64_t kv_head_count = 3;
    constexpr std::uint64_t key_length = 64;
    constexpr std::uint64_t value_length = 64;
    constexpr std::uint64_t context_length = 4096;
    constexpr std::uint64_t vocab_size = 16;
    constexpr float rms_norm_epsilon = 0.00001F;
    constexpr float rope_freq_base = 10000.0F;
    constexpr float attention_scale = 0.125F;
    const std::array<std::uint64_t, 5> cache_lengths {7, 64, 256, 1024, 2048};

    const auto decode_input = make_f32_matrix(1, hidden, 0.0005F);
    const auto attention_norm_weight = make_f32_storage(hidden, 1.0F);
    const auto query_weight = make_f16_gguf_weight_matrix(hidden, head_count * key_length, 0.00012F);
    const auto key_weight = make_f16_gguf_weight_matrix(hidden, kv_head_count * key_length, 0.00014F);
    const auto value_weight = make_f16_gguf_weight_matrix(hidden, kv_head_count * value_length, 0.00016F);
    const auto output_weight = make_f16_gguf_weight_matrix(head_count * value_length, hidden, 0.00011F);
    const auto feed_forward_norm_weight = make_f32_storage(hidden, 1.0F);
    const auto gate_weight = make_f16_gguf_weight_matrix(hidden, feed_forward, 0.00007F);
    const auto up_weight = make_f16_gguf_weight_matrix(hidden, feed_forward, 0.00008F);
    const auto down_weight = make_f16_gguf_weight_matrix(feed_forward, hidden, 0.00006F);
    const auto token_embedding_weight = make_f16_gguf_weight_matrix(hidden, vocab_size, 0.00003F);
    const auto final_norm_weight = make_f32_storage(hidden, 1.0F);
    const auto logits_weight = make_f16_gguf_weight_matrix(hidden, vocab_size, 0.00004F);
    const auto query_fixture = make_f32_matrix(1, head_count * key_length, 0.00023F);
    const auto attention_output_fixture = make_f32_matrix(1, head_count * value_length, 0.00019F);
    const auto cache_key_storage = make_f32_matrix(context_length, kv_head_count * key_length, 0.00009F);
    const auto cache_value_storage = make_f32_matrix(context_length, kv_head_count * value_length, 0.00007F);

    LlamaModelConfig focused_config {
        .context_length = static_cast<std::uint32_t>(context_length),
        .embedding_length = static_cast<std::uint32_t>(hidden),
        .block_count = 2,
        .feed_forward_length = static_cast<std::uint32_t>(feed_forward),
        .attention_head_count = static_cast<std::uint32_t>(head_count),
        .attention_head_count_kv = static_cast<std::uint32_t>(kv_head_count),
        .attention_key_length = static_cast<std::uint32_t>(key_length),
        .attention_value_length = static_cast<std::uint32_t>(value_length),
        .rope_dimension_count = static_cast<std::uint32_t>(key_length),
        .vocab_size = static_cast<std::uint32_t>(vocab_size),
        .rope_freq_base = rope_freq_base,
        .rms_norm_epsilon = rms_norm_epsilon,
    };
    const LlamaAttentionLayerWeights attention_weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    const LlamaFeedForwardLayerWeights feed_forward_weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };
    const LlamaBlockWeights block_weights {
        .attention = attention_weights,
        .feed_forward = feed_forward_weights,
    };
    const LlamaModelWeights model_weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = final_norm_weight,
        .output_weight = logits_weight,
        .blocks = {block_weights, block_weights},
    };
    const auto cpu_backend_weights = copy_llama_model_block_weights_to_backend(model_weights, BackendKind::cpu);

    for (const auto cache_length : cache_lengths) {
        std::vector<LlamaBackendKVCache> cpu_decode_caches;
        cpu_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            cpu_decode_caches.push_back(make_prefilled_focused_backend_kv_cache(
                focused_config,
                BackendKind::cpu,
                cache_length,
                cache_key_storage,
                cache_value_storage
            ));
        }

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.cpu_buffered.focused_cache_"
                + std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "full_model_decode",
                "cpu",
                cache_length,
                focused_config.block_count,
                "model-resident fused-cached-block one-readback"
            ),
            [
                &decode_input,
                &cpu_backend_weights,
                &focused_config,
                &cpu_decode_caches,
                cache_index = std::size_t {0},
                cache_length
            ]() mutable {
                auto& cache = cpu_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    decode_input,
                    cpu_backend_weights,
                    focused_config,
                    cache_length,
                    cache
                );
                return IterationMeasurement {
                    .bytes_processed = decode_input.bytes().size()
                        + llama_backend_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(cpu_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }
        ));
    }

    if (!metal_available) {
        add_unavailable_metal_focused_resident_kv_decode_stage_benchmark(report);
        return;
    }

    const auto metal_backend_weights = copy_llama_model_block_weights_to_backend(model_weights, BackendKind::metal);
    const auto input_buffer = copy_tensor_to_backend(decode_input, BackendKind::metal);
    const auto query_buffer = copy_tensor_to_backend(query_fixture, BackendKind::metal);
    const auto attention_output_buffer = copy_tensor_to_backend(attention_output_fixture, BackendKind::metal);

    report.measurements.push_back(measure(
        "tensor.metal_noop_dispatch",
        report.config.iterations,
        "backend=metal stage=noop cost=command-buffer+one-empty-dispatch",
        []() {
            tensor::backends::metal::benchmark_noop_dispatch();
            return IterationMeasurement {
                .bytes_processed = 0,
                .checksum = 1,
            };
        }
    ));

    for (const auto cache_length : cache_lengths) {
        std::vector<LlamaBackendKVCache> metal_decode_caches;
        metal_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            metal_decode_caches.push_back(make_prefilled_focused_backend_kv_cache(
                focused_config,
                BackendKind::metal,
                cache_length,
                cache_key_storage,
                cache_value_storage
            ));
        }
        auto front_end_cache = make_prefilled_focused_backend_kv_cache(
            focused_config,
            BackendKind::metal,
            cache_length,
            cache_key_storage,
            cache_value_storage
        );
        auto attention_cache = make_prefilled_focused_backend_kv_cache(
            focused_config,
            BackendKind::metal,
            cache_length + 1,
            cache_key_storage,
            cache_value_storage
        );

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.metal_buffered.focused_cache_"
                + std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "full_model_decode",
                "metal",
                cache_length,
                focused_config.block_count,
                "model-resident fused-cached-block one-readback"
            ),
            [
                &decode_input,
                &metal_backend_weights,
                &focused_config,
                &metal_decode_caches,
                cache_index = std::size_t {0},
                cache_length
            ]() mutable {
                auto& cache = metal_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    decode_input,
                    metal_backend_weights,
                    focused_config,
                    cache_length,
                    cache
                );
                return IterationMeasurement {
                    .bytes_processed = decode_input.bytes().size()
                        + llama_backend_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_cached_decode_attention_frontend_f32.metal_buffered.focused_cache_"
                + std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "attention_frontend",
                "metal",
                cache_length,
                1,
                "RMSNorm+QKV+KV-append one-command-buffer no-readback"
            ),
            [
                &input_buffer,
                &metal_backend_weights,
                &front_end_cache,
                cache_length
            ]() {
                const auto& block = metal_backend_weights.blocks[0];
                auto outputs = tensor::backends::metal::benchmark_llama_cached_decode_attention_frontend_to_backend(
                    input_buffer,
                    block.attention.attention_norm_weight_buffer,
                    block.attention.query_weight,
                    block.attention.key_weight,
                    block.attention.value_weight,
                    front_end_cache.layers[0].key,
                    front_end_cache.layers[0].value,
                    cache_length,
                    head_count,
                    kv_head_count,
                    key_length,
                    value_length,
                    key_length,
                    rope_freq_base,
                    rms_norm_epsilon
                );
                front_end_cache.layers[0].filled_token_count = cache_length + 1;
                return IterationMeasurement {
                    .bytes_processed = backend_buffer_byte_count(input_buffer)
                        + llama_attention_weight_bytes(block.attention)
                        + llama_backend_kv_cache_bytes(front_end_cache)
                        + outputs[0].descriptor().byte_count
                        + outputs[1].descriptor().byte_count
                        + outputs[2].descriptor().byte_count,
                    .checksum = outputs[0].descriptor().byte_count
                        ^ (outputs[1].descriptor().byte_count << 1U)
                        ^ (outputs[2].descriptor().byte_count << 2U)
                        ^ llama_backend_kv_cache_fill_checksum(front_end_cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_cached_decode_attention_output_f32.metal_buffered.focused_cache_"
                + std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "cached_attention_output",
                "metal",
                cache_length,
                1,
                "attention-output-only no-append no-readback"
            ),
            [
                &query_buffer,
                &attention_cache,
                cache_length
            ]() {
                const auto output = tensor::backends::metal::benchmark_llama_cached_decode_attention_output_to_backend(
                    query_buffer,
                    attention_cache.layers[0].key,
                    attention_cache.layers[0].value,
                    cache_length,
                    head_count,
                    kv_head_count,
                    key_length,
                    value_length,
                    key_length,
                    rope_freq_base,
                    attention_scale
                );
                return IterationMeasurement {
                    .bytes_processed = backend_buffer_byte_count(query_buffer)
                        + llama_backend_kv_cache_bytes(attention_cache)
                        + backend_buffer_byte_count(output),
                    .checksum = output.descriptor().byte_count
                        ^ llama_backend_kv_cache_fill_checksum(attention_cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_cached_decode_block_tail_f32.metal_buffered.focused_cache_"
                + std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "block_tail",
                "metal",
                cache_length,
                1,
                "attention-output-projection+residual+FFN no-readback"
            ),
            [
                &input_buffer,
                &attention_output_buffer,
                &metal_backend_weights
            ]() {
                const auto& block = metal_backend_weights.blocks[0];
                const auto output = tensor::backends::metal::benchmark_llama_cached_decode_block_tail_to_backend(
                    input_buffer,
                    attention_output_buffer,
                    block.attention.output_weight,
                    block.feed_forward.feed_forward_norm_weight_buffer,
                    block.feed_forward.gate_weight,
                    block.feed_forward.up_weight,
                    block.feed_forward.down_weight,
                    rms_norm_epsilon
                );
                return IterationMeasurement {
                    .bytes_processed = backend_buffer_byte_count(input_buffer)
                        + backend_buffer_byte_count(attention_output_buffer)
                        + llama_feed_forward_weight_bytes(block.feed_forward)
                        + backend_buffer_byte_count(block.attention.output_weight)
                        + backend_buffer_byte_count(output),
                    .checksum = output.descriptor().byte_count,
                };
            }
        ));
    }
}

void add_synthetic_tensor_benchmarks(BenchmarkReport& report) {
    constexpr std::uint64_t add_element_count = 4096;
    constexpr std::uint64_t copy_element_count = 16 * 1024;
    constexpr std::uint64_t matmul_size = 64;
    constexpr std::uint64_t focused_hidden = 576;
    constexpr std::uint64_t focused_ffn = 1536;

    const auto add_descriptor = TensorDescriptor::from(Shape::from({add_element_count}), DType::f32);
    const auto copy_descriptor = TensorDescriptor::from(Shape::from({copy_element_count}), DType::f32);
    const auto copy_payload = make_repeated_payload(copy_descriptor.byte_count);
    const auto add_left = make_f32_storage(add_element_count, 1.0F);
    const auto add_right = make_f32_storage(add_element_count, -0.5F);
    const auto matmul_left = make_f32_matrix(matmul_size, matmul_size, 0.01F);
    const auto matmul_right = make_f32_matrix(matmul_size, matmul_size, 0.02F);
    const auto matmul_right_f16 = make_f16_matrix(matmul_size, matmul_size, 0.02F);
    const auto projection_small_left = make_f32_matrix(8, matmul_size, 0.01F);
    const auto projection_small_weight = make_f16_gguf_weight_matrix(matmul_size, matmul_size, 0.02F);
    const auto projection_next_token_left = make_f32_matrix(1, focused_hidden, 0.001F);
    const auto projection_hidden_weight =
        make_f16_gguf_weight_matrix(focused_hidden, focused_hidden, 0.0005F);
    const auto projection_prefill_left = make_f32_matrix(8, focused_hidden, 0.001F);
    const auto projection_ffn_weight = make_f16_gguf_weight_matrix(focused_hidden, focused_ffn, 0.00025F);
    const auto rms_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto attention_query = make_f32_matrix(matmul_size, matmul_size, 0.01F);
    const auto attention_key = make_f32_matrix(matmul_size, matmul_size, 0.02F);
    const auto attention_value = make_f32_matrix(matmul_size, matmul_size, 0.03F);
    const auto llama_layer_input = make_f32_matrix(8, matmul_size, 0.01F);
    const auto llama_cached_model_block_prefill_input = make_f32_matrix(7, matmul_size, 0.011F);
    const auto llama_cached_model_block_decode_input = make_f32_matrix(1, matmul_size, 0.017F);
    const auto llama_attention_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto llama_query_weight = make_f16_matrix(matmul_size, matmul_size, 0.01F);
    const auto llama_key_weight = make_f16_matrix(matmul_size, 32, 0.02F);
    const auto llama_value_weight = make_f16_matrix(matmul_size, 32, 0.03F);
    const auto llama_output_weight = make_f16_matrix(matmul_size, matmul_size, 0.04F);
    const auto llama_feed_forward_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto llama_gate_weight = make_f16_matrix(matmul_size, 128, 0.001F);
    const auto llama_up_weight = make_f16_matrix(matmul_size, 128, 0.0015F);
    const auto llama_down_weight = make_f16_matrix(128, matmul_size, 0.002F);
    const auto llama_token_embedding_weight = make_f16_matrix(matmul_size, 1024, 0.0005F);
    const auto llama_final_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto llama_logits_weight = make_f16_matrix(matmul_size, 1024, 0.00075F);
    const std::array<std::uint32_t, 8> llama_token_ids {1, 5, 9, 13, 17, 21, 25, 29};
    const auto tokenizer_metadata = make_synthetic_tokenizer_metadata();
    const auto synthetic_tokenizer = load_tokenizer_from_gguf_metadata(tokenizer_metadata);
    const std::string tokenizer_text = "Hello world!Hello world!";
    const auto tokenizer_ids = synthetic_tokenizer.encode(
        tokenizer_text,
        TokenizerEncodeOptions {.add_bos = true, .add_eos = true}
    );
    const LlamaModelConfig llama_layer_config {
        .context_length = 128,
        .embedding_length = 64,
        .block_count = 1,
        .feed_forward_length = 128,
        .attention_head_count = 4,
        .attention_head_count_kv = 2,
        .attention_key_length = 16,
        .attention_value_length = 16,
        .rope_dimension_count = 16,
        .vocab_size = 1024,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 0.00001F,
    };
    const LlamaAttentionLayerWeights llama_layer_weights {
        .attention_norm_weight = llama_attention_norm_weight,
        .query_weight = llama_query_weight,
        .key_weight = llama_key_weight,
        .value_weight = llama_value_weight,
        .output_weight = llama_output_weight,
    };
    const LlamaFeedForwardLayerWeights llama_feed_forward_weights {
        .feed_forward_norm_weight = llama_feed_forward_norm_weight,
        .gate_weight = llama_gate_weight,
        .up_weight = llama_up_weight,
        .down_weight = llama_down_weight,
    };
    const LlamaBlockWeights llama_block_weights {
        .attention = llama_layer_weights,
        .feed_forward = llama_feed_forward_weights,
    };
    auto llama_model_blocks_config = llama_layer_config;
    llama_model_blocks_config.block_count = 2;
    const LlamaModelWeights llama_model_weights {
        .token_embedding_weight = llama_token_embedding_weight,
        .output_norm_weight = llama_final_norm_weight,
        .output_weight = llama_logits_weight,
        .blocks = {llama_block_weights, llama_block_weights},
    };
    const auto llama_model_cpu_block_backend_weights = copy_llama_model_block_weights_to_backend(
        llama_model_weights,
        BackendKind::cpu
    );
    const auto llama_attention_cpu_backend_weights = copy_llama_attention_layer_weights_to_backend(
        llama_layer_weights,
        BackendKind::cpu
    );
    const auto llama_feed_forward_cpu_backend_weights = copy_llama_feed_forward_layer_weights_to_backend(
        llama_feed_forward_weights,
        BackendKind::cpu
    );
    const auto llama_model_cpu_backend_weights = copy_llama_model_logits_weights_to_backend(
        llama_model_weights,
        BackendKind::cpu
    );
    const SamplingConfig llama_sampling {
        .temperature = 0.7F,
        .top_k = 40,
        .top_p = 0.9F,
        .repetition_penalty = 1.1F,
        .seed = 1,
    };
    std::vector<LlamaKVCache> cached_model_block_decode_caches;
    cached_model_block_decode_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_kv_cache(llama_model_blocks_config);
        (void)run_llama_model_blocks_with_kv_cache_f32(
            llama_cached_model_block_prefill_input,
            llama_model_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_model_block_decode_caches.push_back(std::move(cache));
    }
    std::vector<LlamaKVCache> cached_model_block_cpu_backend_decode_caches;
    cached_model_block_cpu_backend_decode_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_kv_cache(llama_model_blocks_config);
        (void)run_llama_model_blocks_with_kv_cache_f32(
            llama_cached_model_block_prefill_input,
            llama_model_cpu_block_backend_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_model_block_cpu_backend_decode_caches.push_back(std::move(cache));
    }
    std::vector<LlamaBackendKVCache> cached_model_block_cpu_backend_resident_decode_caches;
    cached_model_block_cpu_backend_resident_decode_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::cpu);
        (void)run_llama_model_blocks_with_kv_cache_f32(
            llama_cached_model_block_prefill_input,
            llama_model_cpu_block_backend_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_model_block_cpu_backend_resident_decode_caches.push_back(std::move(cache));
    }
    std::vector<LlamaKVCache> cached_next_token_caches;
    cached_next_token_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_kv_cache(llama_model_blocks_config);
        (void)run_llama_model_logits_with_kv_cache_f32(
            std::span<const std::uint32_t>(llama_token_ids.data(), llama_token_ids.size() - 1),
            llama_model_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_next_token_caches.push_back(std::move(cache));
    }
    const auto metal_available = tensor_backend_is_available(BackendKind::metal);
    if (metal_available) {
        warm_up_metal_projection_backend();
    }

    report.measurements.push_back(measure(
        "tensor.allocate_f32",
        report.config.iterations,
        "shape=[4096] dtype=F32",
        [add_descriptor]() {
            const auto storage = TensorStorage::allocate(add_descriptor);
            return IterationMeasurement {
                .bytes_processed = storage.bytes().size(),
                .checksum = checksum_byte_span(storage.bytes()),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.copy_f32",
        report.config.iterations,
        "shape=[16384] dtype=F32",
        [copy_descriptor, copy_payload]() {
            const auto storage = TensorStorage::from_bytes(copy_descriptor, std::span<const std::uint8_t>(copy_payload));
            return IterationMeasurement {
                .bytes_processed = storage.bytes().size(),
                .checksum = checksum_byte_span(storage.bytes()),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.add_f32",
        report.config.iterations,
        "shape=[4096] dtype=F32",
        [&add_left, &add_right]() {
            const auto output = add_f32(add_left, add_right);
            return IterationMeasurement {
                .bytes_processed = add_left.bytes().size() + add_right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.multiply_f32",
        report.config.iterations,
        "shape=[4096] dtype=F32",
        [&add_left, &add_right]() {
            const auto output = multiply_f32(add_left, add_right);
            return IterationMeasurement {
                .bytes_processed = add_left.bytes().size() + add_right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.silu_f32",
        report.config.iterations,
        "shape=[4096] dtype=F32",
        [&add_left]() {
            const auto output = silu_f32(add_left);
            return IterationMeasurement {
                .bytes_processed = add_left.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.matmul_f32",
        report.config.iterations,
        "left=[64,64] right=[64,64] dtype=F32",
        [&matmul_left, &matmul_right]() {
            const auto output = matmul_f32(matmul_left, matmul_right);
            return IterationMeasurement {
                .bytes_processed = matmul_left.bytes().size() + matmul_right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16",
        report.config.iterations,
        "left=[64,64] dtype=F32 right=[64,64] dtype=F16",
        [&matmul_left, &matmul_right_f16]() {
            const auto output = matmul_f32_f16(matmul_left, matmul_right_f16);
            return IterationMeasurement {
                .bytes_processed = matmul_left.bytes().size() + matmul_right_f16.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    add_projection_backend_benchmarks(
        report,
        "small_prefill",
        projection_small_left,
        projection_small_weight,
        metal_available
    );
    add_projection_backend_benchmarks(
        report,
        "next_token_hidden",
        projection_next_token_left,
        projection_hidden_weight,
        metal_available
    );
    add_projection_backend_benchmarks(
        report,
        "prefill_hidden",
        projection_prefill_left,
        projection_hidden_weight,
        metal_available
    );
    add_projection_backend_benchmarks(
        report,
        "next_token_ffn",
        projection_next_token_left,
        projection_ffn_weight,
        metal_available
    );
    if (!metal_available) {
        add_unavailable_metal_projection_benchmark(report);
    }

    report.measurements.push_back(measure(
        "tensor.rms_norm_f32",
        report.config.iterations,
        "input=[64,64] weight=[64] dtype=F32 epsilon=1e-5",
        [&matmul_left, &rms_norm_weight]() {
            const auto output = rms_norm_f32(matmul_left, rms_norm_weight, 0.00001F);
            return IterationMeasurement {
                .bytes_processed = matmul_left.bytes().size() + rms_norm_weight.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.rope_f32",
        report.config.iterations,
        "input=[64,64] position=1 rope_dim=64 freq_base=10000 dtype=F32",
        [&matmul_left]() {
            const auto output = rope_f32(matmul_left, 1, 64, 10000.0F);
            return IterationMeasurement {
                .bytes_processed = matmul_left.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.rope_f32_with_position_start",
        report.config.iterations,
        "input=[64,64] position_start=0 rope_dim=64 freq_base=10000 dtype=F32",
        [&matmul_left]() {
            const auto output = rope_f32_with_position_start(matmul_left, 0, 64, 10000.0F);
            return IterationMeasurement {
                .bytes_processed = matmul_left.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.softmax_f32",
        report.config.iterations,
        "input=[64,64] dtype=F32 rowwise",
        [&matmul_left]() {
            const auto output = softmax_f32(matmul_left);
            return IterationMeasurement {
                .bytes_processed = matmul_left.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tokenizer.encode_greedy",
        report.config.iterations,
        "model=gpt2 vocab=7 text_bytes=24 bos/eos",
        [&synthetic_tokenizer, &tokenizer_text]() {
            const auto token_ids = synthetic_tokenizer.encode(
                tokenizer_text,
                TokenizerEncodeOptions {.add_bos = true, .add_eos = true}
            );
            return IterationMeasurement {
                .bytes_processed = tokenizer_text.size() + token_ids.size() * sizeof(std::uint32_t),
                .checksum = token_ids.empty() ? 0 : static_cast<std::uint64_t>(token_ids.back() + token_ids.size()),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tokenizer.decode",
        report.config.iterations,
        "model=gpt2 tokens=6 skip_special=true",
        [&synthetic_tokenizer, &tokenizer_ids]() {
            const auto text = synthetic_tokenizer.decode(
                std::span<const std::uint32_t>(tokenizer_ids),
                TokenizerDecodeOptions {.skip_special_tokens = true}
            );
            return IterationMeasurement {
                .bytes_processed = tokenizer_ids.size() * sizeof(std::uint32_t) + text.size(),
                .checksum = checksum_byte_span(std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(text.data()),
                    text.size()
                )),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.scaled_dot_product_attention_f32",
        report.config.iterations,
        "query=[64,64] key=[64,64] value=[64,64] dtype=F32 scale=0.125",
        [&attention_query, &attention_key, &attention_value]() {
            const auto output = scaled_dot_product_attention_f32(
                attention_query,
                attention_key,
                attention_value,
                0.125F
            );
            return IterationMeasurement {
                .bytes_processed = attention_query.bytes().size()
                    + attention_key.bytes().size()
                    + attention_value.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.causal_scaled_dot_product_attention_f32",
        report.config.iterations,
        "query=[64,64] key=[64,64] value=[64,64] dtype=F32 scale=0.125 start=0",
        [&attention_query, &attention_key, &attention_value]() {
            const auto output = causal_scaled_dot_product_attention_f32(
                attention_query,
                attention_key,
                attention_value,
                0.125F,
                0
            );
            return IterationMeasurement {
                .bytes_processed = attention_query.bytes().size()
                    + attention_key.bytes().size()
                    + attention_value.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_attention_layer_f32",
        report.config.iterations,
        "input=[8,64] heads=4 kv_heads=2 key=16 value=16 dtype=F32/F16",
        [&llama_layer_input, &llama_layer_weights, &llama_layer_config]() {
            const auto output = run_llama_attention_layer_f32(
                llama_layer_input,
                llama_layer_weights,
                llama_layer_config,
                7
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_layer_weights.attention_norm_weight.bytes().size()
                    + llama_layer_weights.query_weight.bytes().size()
                    + llama_layer_weights.key_weight.bytes().size()
                    + llama_layer_weights.value_weight.bytes().size()
                    + llama_layer_weights.output_weight.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_attention_layer_f32.cpu_buffered",
        report.config.iterations,
        "input=[8,64] heads=4 kv_heads=2 key=16 value=16 backend=cpu preloaded-projections fused-qkv",
        [&llama_layer_input, &llama_attention_cpu_backend_weights, &llama_layer_config]() {
            const auto output = run_llama_attention_layer_f32(
                llama_layer_input,
                llama_attention_cpu_backend_weights,
                llama_layer_config,
                7
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_attention_weight_bytes(llama_attention_cpu_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_feed_forward_layer_f32",
        report.config.iterations,
        "input=[8,64] hidden=64 ffn=128 dtype=F32/F16",
        [&llama_layer_input, &llama_feed_forward_weights, &llama_layer_config]() {
            const auto output = run_llama_feed_forward_layer_f32(
                llama_layer_input,
                llama_feed_forward_weights,
                llama_layer_config
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_feed_forward_weights.feed_forward_norm_weight.bytes().size()
                    + llama_feed_forward_weights.gate_weight.bytes().size()
                    + llama_feed_forward_weights.up_weight.bytes().size()
                    + llama_feed_forward_weights.down_weight.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_feed_forward_layer_f32.cpu_buffered",
        report.config.iterations,
        "input=[8,64] hidden=64 ffn=128 backend=cpu preloaded-projections fused-ffn",
        [&llama_layer_input, &llama_feed_forward_cpu_backend_weights, &llama_layer_config]() {
            const auto output = run_llama_feed_forward_layer_f32(
                llama_layer_input,
                llama_feed_forward_cpu_backend_weights,
                llama_layer_config
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_feed_forward_weight_bytes(llama_feed_forward_cpu_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_block_f32",
        report.config.iterations,
        "input=[8,64] heads=4 kv_heads=2 ffn=128 dtype=F32/F16",
        [&llama_layer_input, &llama_block_weights, &llama_layer_config]() {
            const auto output = run_llama_block_f32(
                llama_layer_input,
                llama_block_weights,
                llama_layer_config,
                7
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_block_weights.attention.attention_norm_weight.bytes().size()
                    + llama_block_weights.attention.query_weight.bytes().size()
                    + llama_block_weights.attention.key_weight.bytes().size()
                    + llama_block_weights.attention.value_weight.bytes().size()
                    + llama_block_weights.attention.output_weight.bytes().size()
                    + llama_block_weights.feed_forward.feed_forward_norm_weight.bytes().size()
                    + llama_block_weights.feed_forward.gate_weight.bytes().size()
                    + llama_block_weights.feed_forward.up_weight.bytes().size()
                    + llama_block_weights.feed_forward.down_weight.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_f32",
        report.config.iterations,
        "input=[8,64] blocks=2 heads=4 kv_heads=2 ffn=128 dtype=F32/F16",
        [&llama_layer_input, &llama_model_weights, &llama_model_blocks_config]() {
            const auto output = run_llama_model_blocks_f32(
                llama_layer_input,
                llama_model_weights,
                llama_model_blocks_config,
                7
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_f32",
        report.config.iterations,
        "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu cache_prefill",
        [&llama_cached_model_block_prefill_input, &llama_model_weights, &llama_model_blocks_config]() {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_weights,
                llama_model_blocks_config,
                6,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_f32.cpu_buffered",
        report.config.iterations,
        "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections fused-qkv fused-block-tail cache_prefill",
        [
            &llama_cached_model_block_prefill_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config
        ]() {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_decode_f32",
        report.config.iterations,
        "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu cache_prefilled=7",
        [
            &llama_cached_model_block_decode_input,
            &llama_model_weights,
            &llama_model_blocks_config,
            &cached_model_block_decode_caches,
            cache_index = std::size_t {0}
        ]() mutable {
            auto& cache = cached_model_block_decode_caches[cache_index];
            ++cache_index;
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_decode_input,
                llama_model_weights,
                llama_model_blocks_config,
                7,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_decode_f32.cpu_buffered",
        report.config.iterations,
        "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections fused-qkv fused-block-tail cache_prefilled=7",
        [
            &llama_cached_model_block_decode_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config,
            &cached_model_block_cpu_backend_decode_caches,
            cache_index = std::size_t {0}
        ]() mutable {
            auto& cache = cached_model_block_cpu_backend_decode_caches[cache_index];
            ++cache_index;
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_decode_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                7,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_backend_kv_cache_f32.cpu_buffered",
        report.config.iterations,
        "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections resident-kv fused-cached-block model-resident cache_prefill",
        [
            &llama_cached_model_block_prefill_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config
        ]() {
            auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::cpu);
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                    + llama_backend_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.cpu_buffered",
        report.config.iterations,
        "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections resident-kv fused-cached-block model-resident cache_prefilled=7",
        [
            &llama_cached_model_block_decode_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config,
            &cached_model_block_cpu_backend_resident_decode_caches,
            cache_index = std::size_t {0}
        ]() mutable {
            auto& cache = cached_model_block_cpu_backend_resident_decode_caches[cache_index];
            ++cache_index;
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_decode_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                7,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                    + llama_backend_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_logits_f32",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 dtype=F32/F16",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            const auto output = run_llama_model_logits_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_weights,
                llama_model_blocks_config,
                7
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_logits_f32.cpu_buffered",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 dtype=F32/F16 backend=cpu preloaded-projections fused-qkv fused-block-tail",
        [&llama_token_ids, &llama_model_cpu_backend_weights, &llama_model_blocks_config]() {
            const auto output = run_llama_model_logits_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_cpu_backend_weights,
                llama_model_blocks_config,
                7
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_cpu_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_argmax_f32",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 sampler=argmax",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            const auto token = run_llama_model_next_token_argmax_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_weights,
                llama_model_blocks_config,
                7
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = token,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_argmax_f32.cpu_buffered",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 backend=cpu preloaded-projections fused-qkv fused-block-tail sampler=argmax",
        [&llama_token_ids, &llama_model_cpu_backend_weights, &llama_model_blocks_config]() {
            const auto token = run_llama_model_next_token_argmax_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_cpu_backend_weights,
                llama_model_blocks_config,
                7
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_cpu_backend_weights),
                .checksum = token,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_sampled_f32",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 sampler=temp0.7/top-k40/top-p0.9/repeat1.1",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config, &llama_sampling]() {
            SamplingState sampling_state(llama_sampling.seed);
            const auto token = run_llama_model_next_token_sampled_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_weights,
                llama_model_blocks_config,
                7,
                llama_sampling,
                sampling_state,
                std::span<const std::uint32_t>(llama_token_ids)
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_weights),
                .checksum = token,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_sampled_f32.cpu_buffered",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 backend=cpu preloaded-projections fused-qkv fused-block-tail sampler=temp0.7/top-k40/top-p0.9/repeat1.1",
        [&llama_token_ids, &llama_model_cpu_backend_weights, &llama_model_blocks_config, &llama_sampling]() {
            SamplingState sampling_state(llama_sampling.seed);
            const auto token = run_llama_model_next_token_sampled_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_cpu_backend_weights,
                llama_model_blocks_config,
                7,
                llama_sampling,
                sampling_state,
                std::span<const std::uint32_t>(llama_token_ids)
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_cpu_backend_weights),
                .checksum = token,
            };
        }
    ));

    if (metal_available) {
        const auto llama_attention_metal_backend_weights = copy_llama_attention_layer_weights_to_backend(
            llama_layer_weights,
            BackendKind::metal
        );
        const auto llama_feed_forward_metal_backend_weights = copy_llama_feed_forward_layer_weights_to_backend(
            llama_feed_forward_weights,
            BackendKind::metal
        );
        const auto llama_model_metal_block_backend_weights = copy_llama_model_block_weights_to_backend(
            llama_model_weights,
            BackendKind::metal
        );
        const auto llama_model_metal_backend_weights = copy_llama_model_logits_weights_to_backend(
            llama_model_weights,
            BackendKind::metal
        );
        std::vector<LlamaKVCache> cached_model_block_metal_backend_decode_caches;
        cached_model_block_metal_backend_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            (void)run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_metal_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );
            cached_model_block_metal_backend_decode_caches.push_back(std::move(cache));
        }
        std::vector<LlamaBackendKVCache> cached_model_block_metal_backend_resident_decode_caches;
        cached_model_block_metal_backend_resident_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::metal);
            (void)run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_metal_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );
            cached_model_block_metal_backend_resident_decode_caches.push_back(std::move(cache));
        }

        report.measurements.push_back(measure(
            "engine.llama_attention_layer_f32.metal_buffered",
            report.config.iterations,
            "input=[8,64] heads=4 kv_heads=2 key=16 value=16 backend=metal preloaded-projections fused-qkv",
            [&llama_layer_input, &llama_attention_metal_backend_weights, &llama_layer_config]() {
                const auto output = run_llama_attention_layer_f32(
                    llama_layer_input,
                    llama_attention_metal_backend_weights,
                    llama_layer_config,
                    7
                );
                return IterationMeasurement {
                    .bytes_processed = llama_layer_input.bytes().size()
                        + llama_attention_weight_bytes(llama_attention_metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_feed_forward_layer_f32.metal_buffered",
            report.config.iterations,
            "input=[8,64] hidden=64 ffn=128 backend=metal preloaded-projections fused-ffn one-command-buffer",
            [&llama_layer_input, &llama_feed_forward_metal_backend_weights, &llama_layer_config]() {
                const auto output = run_llama_feed_forward_layer_f32(
                    llama_layer_input,
                    llama_feed_forward_metal_backend_weights,
                    llama_layer_config
                );
                return IterationMeasurement {
                    .bytes_processed = llama_layer_input.bytes().size()
                        + llama_feed_forward_weight_bytes(llama_feed_forward_metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_kv_cache_f32.metal_buffered",
            report.config.iterations,
            "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections fused-qkv fused-block-tail cache_prefill",
            [
                &llama_cached_model_block_prefill_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config
            ]() {
                auto cache = create_llama_kv_cache(llama_model_blocks_config);
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_prefill_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    6,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                        + llama_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_kv_cache_decode_f32.metal_buffered",
            report.config.iterations,
            "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections fused-qkv fused-block-tail cache_prefilled=7",
            [
                &llama_cached_model_block_decode_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config,
                &cached_model_block_metal_backend_decode_caches,
                cache_index = std::size_t {0}
            ]() mutable {
                auto& cache = cached_model_block_metal_backend_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_decode_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    7,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                        + llama_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_f32.metal_buffered",
            report.config.iterations,
            "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections resident-kv fused-cached-block model-resident one-readback cache_prefill",
            [
                &llama_cached_model_block_prefill_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config
            ]() {
                auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::metal);
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_prefill_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    6,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                        + llama_backend_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.metal_buffered",
            report.config.iterations,
            "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections resident-kv fused-cached-block model-resident one-readback cache_prefilled=7",
            [
                &llama_cached_model_block_decode_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config,
                &cached_model_block_metal_backend_resident_decode_caches,
                cache_index = std::size_t {0}
            ]() mutable {
                auto& cache = cached_model_block_metal_backend_resident_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_decode_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    7,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                        + llama_backend_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_logits_f32.metal_buffered",
            report.config.iterations,
            "tokens=8 hidden=64 vocab=1024 blocks=2 dtype=F32/F16 backend=metal preloaded-projections fused-qkv fused-block-tail",
            [&llama_token_ids, &llama_model_metal_backend_weights, &llama_model_blocks_config]() {
                const auto output = run_llama_model_logits_f32(
                    std::span<const std::uint32_t>(llama_token_ids),
                    llama_model_metal_backend_weights,
                    llama_model_blocks_config,
                    7
                );

                return IterationMeasurement {
                    .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                        + llama_model_weight_bytes(llama_model_metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_next_token_argmax_f32.metal_buffered",
            report.config.iterations,
            "tokens=8 hidden=64 vocab=1024 blocks=2 backend=metal preloaded-projections fused-qkv fused-block-tail sampler=argmax",
            [&llama_token_ids, &llama_model_metal_backend_weights, &llama_model_blocks_config]() {
                const auto token = run_llama_model_next_token_argmax_f32(
                    std::span<const std::uint32_t>(llama_token_ids),
                    llama_model_metal_backend_weights,
                    llama_model_blocks_config,
                    7
                );

                return IterationMeasurement {
                    .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                        + llama_model_weight_bytes(llama_model_metal_backend_weights),
                    .checksum = token,
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_next_token_sampled_f32.metal_buffered",
            report.config.iterations,
            "tokens=8 hidden=64 vocab=1024 blocks=2 backend=metal preloaded-projections fused-qkv fused-block-tail sampler=temp0.7/top-k40/top-p0.9/repeat1.1",
            [&llama_token_ids, &llama_model_metal_backend_weights, &llama_model_blocks_config, &llama_sampling]() {
                SamplingState sampling_state(llama_sampling.seed);
                const auto token = run_llama_model_next_token_sampled_f32(
                    std::span<const std::uint32_t>(llama_token_ids),
                    llama_model_metal_backend_weights,
                    llama_model_blocks_config,
                    7,
                    llama_sampling,
                    sampling_state,
                    std::span<const std::uint32_t>(llama_token_ids)
                );

                return IterationMeasurement {
                    .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                        + llama_model_weight_bytes(llama_model_metal_backend_weights),
                    .checksum = token,
                };
            }
        ));
    } else {
        add_unavailable_metal_attention_backend_benchmark(report);
        add_unavailable_metal_feed_forward_backend_benchmark(report);
        add_unavailable_metal_model_backend_benchmark(report);
        add_unavailable_metal_cached_model_block_backend_benchmark(report);
        add_unavailable_metal_backend_kv_cached_model_block_backend_benchmark(report);
    }

    add_focused_resident_kv_decode_stage_benchmarks(report, metal_available);

    report.measurements.push_back(measure(
        "engine.llama_kv_cache_prefill_f32",
        report.config.iterations,
        "prompt_tokens=7 hidden=64 vocab=1024 blocks=2",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            const auto logits = run_llama_model_logits_with_kv_cache_f32(
                std::span<const std::uint32_t>(llama_token_ids.data(), llama_token_ids.size() - 1),
                llama_model_weights,
                llama_model_blocks_config,
                6,
                cache
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = (llama_token_ids.size() - 1) * sizeof(std::uint32_t)
                    + cache.layers.size() * (
                        cache.layers[0].key.bytes().size()
                        + cache.layers[0].value.bytes().size()
                    )
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = checksum_f32(sum_f32(logits))
                    ^ static_cast<std::uint64_t>(cache.layers[0].filled_token_count),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_cached_next_token_argmax_f32",
        report.config.iterations,
        "next_tokens=1 hidden=64 vocab=1024 blocks=2 cache_prefilled=7",
        [&llama_token_ids,
         &llama_model_weights,
         &llama_model_blocks_config,
         &cached_next_token_caches,
         cache_index = std::size_t {0}]() mutable {
            auto& cache = cached_next_token_caches[cache_index];
            ++cache_index;
            const auto token = run_llama_model_next_token_argmax_with_kv_cache_f32(
                std::span<const std::uint32_t>(llama_token_ids.data() + (llama_token_ids.size() - 1), 1),
                llama_model_weights,
                llama_model_blocks_config,
                7,
                cache
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = sizeof(std::uint32_t)
                    + cache.layers.size() * (
                        cache.layers[0].key.bytes().size()
                        + cache.layers[0].value.bytes().size()
                    )
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = token ^ static_cast<std::uint64_t>(cache.layers[0].filled_token_count),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_generate_argmax_f32",
        report.config.iterations,
        "prompt_tokens=8 max_new_tokens=2 hidden=64 vocab=1024 blocks=2",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            constexpr std::size_t max_new_tokens = 2;
            const auto tokens = generate_llama_argmax_tokens_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                max_new_tokens,
                llama_model_weights,
                llama_model_blocks_config
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = tokens.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + max_new_tokens * llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = tokens.empty() ? 0 : tokens.back(),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_generate_argmax_with_kv_cache_f32",
        report.config.iterations,
        "prompt_tokens=8 max_new_tokens=2 hidden=64 vocab=1024 blocks=2",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            constexpr std::size_t max_new_tokens = 2;
            const auto tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                max_new_tokens,
                llama_model_weights,
                llama_model_blocks_config
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = tokens.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + max_new_tokens * llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = tokens.empty() ? 0 : tokens.back(),
            };
        }
    ));
}

[[nodiscard]] std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::error_code error_code;
    const auto file_size = std::filesystem::file_size(path, error_code);
    if (error_code) {
        throw std::runtime_error("could not read '" + path.string() + "': " + error_code.message());
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open '" + path.string() + "' for reading");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw std::runtime_error("could not read entire file '" + path.string() + "'");
        }
    }

    return bytes;
}

void add_model_load_benchmarks(BenchmarkReport& report, const std::filesystem::path& model_path) {
    report.measurements.push_back(measure(
        "gguf.read_file",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model_path]() {
            const auto bytes = read_binary_file(model_path);
            return IterationMeasurement {
                .bytes_processed = bytes.size(),
                .checksum = checksum_byte_span(bytes),
            };
        }
    ));

    const auto bytes = read_binary_file(model_path);
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));

    report.measurements.push_back(measure(
        "gguf.parse_model_view",
        report.config.iterations,
        "model=" + model_path.string(),
        [&bytes]() {
            const auto parsed = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
            return IterationMeasurement {
                .bytes_processed = bytes.size(),
                .checksum = parsed.directory.metadata.header.tensor_count
                    ^ parsed.directory.metadata.header.metadata_kv_count,
            };
        }
    ));

    report.measurements.push_back(measure(
        "gguf.load_tensor_payloads",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model]() {
            const auto loaded = load_gguf_tensor_payloads(model);
            std::size_t bytes_processed = 0;
            std::uint64_t checksum = 0;
            for (const auto& tensor : loaded) {
                bytes_processed += tensor.storage.bytes().size();
                checksum ^= checksum_byte_span(tensor.storage.bytes());
            }

            return IterationMeasurement {
                .bytes_processed = bytes_processed,
                .checksum = checksum,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.load_model",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model]() {
            const auto loaded = load_engine_model_from_gguf(model);
            std::size_t bytes_processed = 0;
            std::uint64_t checksum = loaded.llama_config().block_count
                ^ loaded.llama_config().embedding_length
                ^ static_cast<std::uint64_t>(loaded.tensors().size());
            for (const auto& tensor : loaded.tensors()) {
                bytes_processed += tensor.storage.bytes().size();
                checksum ^= checksum_byte_span(tensor.storage.bytes());
            }

            return IterationMeasurement {
                .bytes_processed = bytes_processed,
                .checksum = checksum,
            };
        }
    ));
}

[[nodiscard]] std::size_t parse_iteration_count(std::string_view text) {
    std::size_t parsed_characters = 0;
    const auto value = std::stoull(std::string(text), &parsed_characters, 10);
    if (parsed_characters != text.size() || value == 0) {
        throw std::invalid_argument("invalid iteration count");
    }

    if (value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::out_of_range("iteration count is too large");
    }

    return static_cast<std::size_t>(value);
}

[[nodiscard]] BenchmarkArgParseResult parse_error(std::string message) {
    BenchmarkArgParseResult result;
    result.config.output_path = default_benchmark_output_path();
    result.should_exit = true;
    result.exit_code = EXIT_FAILURE;
    result.message = "error: " + std::move(message) + "\n\n" + benchmark_usage();
    return result;
}

[[nodiscard]] bool path_has_prefix(
    const std::filesystem::path& path,
    const std::filesystem::path& prefix
) {
    auto path_iterator = path.begin();
    auto prefix_iterator = prefix.begin();

    for (; prefix_iterator != prefix.end(); ++prefix_iterator, ++path_iterator) {
        if (path_iterator == path.end() || *path_iterator != *prefix_iterator) {
            return false;
        }
    }

    return path_iterator != path.end();
}

}  // namespace

double BenchmarkMeasurement::average_ms() const noexcept {
    if (iterations == 0) {
        return 0.0;
    }

    return total_ms / static_cast<double>(iterations);
}

std::filesystem::path default_benchmark_output_path() {
    return std::filesystem::path("benchmarks") / "results" / "latest.txt";
}

bool benchmark_output_path_is_under_results(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) {
        return false;
    }

    return path_has_prefix(
        path.lexically_normal(),
        (std::filesystem::path("benchmarks") / "results").lexically_normal()
    );
}

std::string benchmark_usage() {
    return "usage: local-gguf-bench [--model <model.gguf>] "
           "[--output benchmarks/results/<name>.txt] [--iterations <count>]\n";
}

BenchmarkArgParseResult parse_benchmark_args(int argc, const char* const* argv) {
    BenchmarkArgParseResult result;
    result.config.output_path = default_benchmark_output_path();

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];

        if (argument == "--help" || argument == "-h") {
            result.should_exit = true;
            result.exit_code = EXIT_SUCCESS;
            result.message = benchmark_usage();
            return result;
        }

        if (argument == "--model") {
            if (index + 1 >= argc) {
                return parse_error("--model requires a path");
            }
            result.config.model_path = std::filesystem::path(argv[++index]);
            continue;
        }

        if (argument == "--output") {
            if (index + 1 >= argc) {
                return parse_error("--output requires a path under benchmarks/results");
            }
            result.config.output_path = std::filesystem::path(argv[++index]);
            continue;
        }

        if (argument == "--iterations") {
            if (index + 1 >= argc) {
                return parse_error("--iterations requires a positive count");
            }
            try {
                result.config.iterations = parse_iteration_count(argv[++index]);
            } catch (const std::exception&) {
                return parse_error("--iterations requires a positive count");
            }
            continue;
        }

        if (!argument.empty() && argument.front() == '-') {
            return parse_error("unknown option '" + std::string(argument) + "'");
        }

        if (result.config.model_path.has_value()) {
            return parse_error("expected at most one model path");
        }
        result.config.model_path = std::filesystem::path(argument);
    }

    if (!benchmark_output_path_is_under_results(result.config.output_path)) {
        return parse_error("benchmark output path must be under benchmarks/results");
    }

    return result;
}

BenchmarkReport run_benchmarks(const BenchmarkConfig& config) {
    BenchmarkReport report {
        .config = config,
        .measurements = {},
    };

    add_synthetic_tensor_benchmarks(report);

    if (config.model_path.has_value()) {
        add_model_load_benchmarks(report, *config.model_path);
    }

    return report;
}

std::string format_benchmark_report(const BenchmarkReport& report) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "local GGUF benchmark report\n";
    output << "iterations: " << report.config.iterations << '\n';
    output << "output: " << report.config.output_path.string() << '\n';
    output << "model: "
           << (report.config.model_path.has_value() ? report.config.model_path->string() : "<none>")
           << '\n';
    output << "benchmarks:\n";

    for (const auto& measurement : report.measurements) {
        output << "- " << measurement.name << '\n';
        output << "  iterations: " << measurement.iterations << '\n';
        output << "  total_ms: " << measurement.total_ms << '\n';
        output << "  avg_ms: " << measurement.average_ms() << '\n';
        output << "  min_ms: " << measurement.min_ms << '\n';
        output << "  max_ms: " << measurement.max_ms << '\n';
        output << "  bytes_processed: " << measurement.bytes_processed << '\n';
        output << "  checksum: " << measurement.checksum << '\n';
        output << "  detail: " << measurement.detail << '\n';
    }

    return output.str();
}

int run_benchmark_cli(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
    const auto parsed = parse_benchmark_args(argc, argv);
    if (parsed.should_exit) {
        auto& stream = parsed.exit_code == EXIT_SUCCESS ? output : error;
        stream << parsed.message;
        return parsed.exit_code;
    }

    try {
        const auto report = run_benchmarks(parsed.config);
        const auto report_text = format_benchmark_report(report);
        std::filesystem::create_directories(parsed.config.output_path.parent_path());

        std::ofstream file(parsed.config.output_path, std::ios::trunc);
        if (!file) {
            throw std::runtime_error("could not open benchmark output '" + parsed.config.output_path.string() + "'");
        }
        file << report_text;
        if (!file) {
            throw std::runtime_error("could not write benchmark output '" + parsed.config.output_path.string() + "'");
        }

        output << report_text;
        output << "wrote benchmark report: " << parsed.config.output_path.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        error << "error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}

}  // namespace local_gguf_engine::benchmarks
