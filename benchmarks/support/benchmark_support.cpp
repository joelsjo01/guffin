#include "benchmark_support.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace local_gguf_engine::benchmarks {

namespace {

[[nodiscard]] model_format::GGUFMetadataEntry metadata_string(std::string key, std::string value) {
    return model_format::GGUFMetadataEntry{
        .key = std::move(key),
        .value =
            model_format::GGUFMetadataValue{
                .type = model_format::GGUFMetadataValueType::string,
                .data = std::move(value),
            },
    };
}

[[nodiscard]] model_format::GGUFMetadataEntry metadata_u32(std::string key, std::uint32_t value) {
    return model_format::GGUFMetadataEntry{
        .key = std::move(key),
        .value =
            model_format::GGUFMetadataValue{
                .type = model_format::GGUFMetadataValueType::uint32,
                .data = value,
            },
    };
}

[[nodiscard]] model_format::GGUFMetadataEntry metadata_bool(std::string key, bool value) {
    return model_format::GGUFMetadataEntry{
        .key = std::move(key),
        .value =
            model_format::GGUFMetadataValue{
                .type = model_format::GGUFMetadataValueType::boolean,
                .data = value,
            },
    };
}

[[nodiscard]] model_format::GGUFMetadataEntry
metadata_string_array(std::string key, std::vector<std::string> values) {
    return model_format::GGUFMetadataEntry{
        .key = std::move(key),
        .value =
            model_format::GGUFMetadataValue{
                .type = model_format::GGUFMetadataValueType::array,
                .data =
                    model_format::GGUFMetadataArray{
                        .element_type = model_format::GGUFMetadataValueType::string,
                        .values = std::move(values),
                    },
            },
    };
}

[[nodiscard]] model_format::GGUFMetadataEntry metadata_f32_array(std::string key,
                                                                 std::vector<float> values) {
    return model_format::GGUFMetadataEntry{
        .key = std::move(key),
        .value =
            model_format::GGUFMetadataValue{
                .type = model_format::GGUFMetadataValueType::array,
                .data =
                    model_format::GGUFMetadataArray{
                        .element_type = model_format::GGUFMetadataValueType::float32,
                        .values = std::move(values),
                    },
            },
    };
}

[[nodiscard]] model_format::GGUFMetadataEntry metadata_i32_array(std::string key,
                                                                 std::vector<std::int32_t> values) {
    return model_format::GGUFMetadataEntry{
        .key = std::move(key),
        .value =
            model_format::GGUFMetadataValue{
                .type = model_format::GGUFMetadataValueType::array,
                .data =
                    model_format::GGUFMetadataArray{
                        .element_type = model_format::GGUFMetadataValueType::int32,
                        .values = std::move(values),
                    },
            },
    };
}

} // namespace

std::uint64_t checksum_byte_span(std::span<const std::uint8_t> bytes) noexcept {
    if (bytes.empty()) {
        return 0;
    }

    return static_cast<std::uint64_t>(bytes.size()) ^ static_cast<std::uint64_t>(bytes.front()) ^
           (static_cast<std::uint64_t>(bytes.back()) << 8);
}

std::uint64_t checksum_f32(float value) noexcept {
    return static_cast<std::uint64_t>(value >= 0.0F ? value * 1000.0F : -value * 1000.0F);
}

std::size_t backend_buffer_byte_count(const tensor::BackendBuffer &buffer) noexcept {
    return buffer.descriptor().byte_count;
}

std::size_t
llama_attention_weight_bytes(const engine::LlamaAttentionLayerWeights &weights) noexcept {
    return weights.attention_norm_weight.bytes().size() + weights.query_weight.bytes().size() +
           weights.key_weight.bytes().size() + weights.value_weight.bytes().size() +
           weights.output_weight.bytes().size();
}

std::size_t
llama_attention_weight_bytes(const engine::LlamaAttentionLayerBackendWeights &weights) noexcept {
    return backend_buffer_byte_count(weights.attention_norm_weight_buffer) +
           backend_buffer_byte_count(weights.query_weight) +
           backend_buffer_byte_count(weights.key_weight) +
           backend_buffer_byte_count(weights.value_weight) +
           backend_buffer_byte_count(weights.output_weight);
}

std::size_t
llama_feed_forward_weight_bytes(const engine::LlamaFeedForwardLayerWeights &weights) noexcept {
    return weights.feed_forward_norm_weight.bytes().size() + weights.gate_weight.bytes().size() +
           weights.up_weight.bytes().size() + weights.down_weight.bytes().size();
}

std::size_t llama_feed_forward_weight_bytes(
    const engine::LlamaFeedForwardLayerBackendWeights &weights) noexcept {
    return backend_buffer_byte_count(weights.feed_forward_norm_weight_buffer) +
           backend_buffer_byte_count(weights.gate_weight) +
           backend_buffer_byte_count(weights.up_weight) +
           backend_buffer_byte_count(weights.down_weight);
}

std::size_t llama_block_weight_bytes(const engine::LlamaBlockWeights &weights) noexcept {
    return llama_attention_weight_bytes(weights.attention) +
           llama_feed_forward_weight_bytes(weights.feed_forward);
}

std::size_t llama_block_weight_bytes(const engine::LlamaBlockBackendWeights &weights) noexcept {
    return llama_attention_weight_bytes(weights.attention) +
           llama_feed_forward_weight_bytes(weights.feed_forward);
}

std::size_t llama_model_weight_bytes(const engine::LlamaModelWeights &weights) noexcept {
    std::size_t bytes = weights.token_embedding_weight.bytes().size() +
                        weights.output_norm_weight.bytes().size() +
                        weights.output_weight.bytes().size();
    for (const auto &block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

std::size_t llama_model_block_weight_bytes(const engine::LlamaModelWeights &weights) noexcept {
    std::size_t bytes = 0;
    for (const auto &block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

std::size_t
llama_model_block_weight_bytes(const engine::LlamaModelBlockBackendWeights &weights) noexcept {
    std::size_t bytes = 0;
    for (const auto &block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

std::size_t
llama_model_weight_bytes(const engine::LlamaModelLogitsBackendWeights &weights) noexcept {
    std::size_t bytes = weights.token_embedding_weight.bytes().size() +
                        weights.output_norm_weight.bytes().size() +
                        backend_buffer_byte_count(weights.output_weight);
    for (const auto &block : weights.blocks) {
        bytes += llama_block_weight_bytes(block);
    }
    return bytes;
}

std::size_t llama_kv_cache_bytes(const engine::LlamaKVCache &cache) noexcept {
    std::size_t bytes = 0;
    for (const auto &layer : cache.layers) {
        bytes += layer.key.bytes().size() + layer.value.bytes().size();
    }
    return bytes;
}

std::size_t llama_backend_kv_cache_bytes(const engine::LlamaBackendKVCache &cache) noexcept {
    std::size_t bytes = 0;
    for (const auto &layer : cache.layers) {
        bytes += backend_buffer_byte_count(layer.key) + backend_buffer_byte_count(layer.value);
    }
    return bytes;
}

std::uint64_t llama_kv_cache_fill_checksum(const engine::LlamaKVCache &cache) noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t layer_index = 0; layer_index < cache.layers.size(); ++layer_index) {
        checksum ^= cache.layers[layer_index].filled_token_count << (layer_index % 8U);
    }
    return checksum;
}

std::uint64_t
llama_backend_kv_cache_fill_checksum(const engine::LlamaBackendKVCache &cache) noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t layer_index = 0; layer_index < cache.layers.size(); ++layer_index) {
        checksum ^= cache.layers[layer_index].filled_token_count << (layer_index % 8U);
    }
    return checksum;
}

tensor::TensorStorage make_f32_storage(std::uint64_t element_count, float start_value) {
    auto storage = tensor::TensorStorage::allocate(
        tensor::TensorDescriptor::from(tensor::Shape::from({element_count}), tensor::DType::f32));
    for (std::uint64_t index = 0; index < element_count; ++index) {
        const std::array<std::uint64_t, 1> tensor_index{index};
        tensor::write_f32(storage, std::span<const std::uint64_t>(tensor_index),
                          start_value + static_cast<float>(index));
    }

    return storage;
}

tensor::TensorStorage make_f32_matrix(std::uint64_t rows, std::uint64_t columns, float scale) {
    auto storage = tensor::TensorStorage::allocate(
        tensor::TensorDescriptor::from(tensor::Shape::from({rows, columns}), tensor::DType::f32));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index{row, column};
            const auto value = scale * static_cast<float>((row + 1) + (column % 7));
            tensor::write_f32(storage, std::span<const std::uint64_t>(index), value);
        }
    }

    return storage;
}

tensor::TensorStorage make_f16_matrix(std::uint64_t rows, std::uint64_t columns, float scale) {
    auto storage = tensor::TensorStorage::allocate(
        tensor::TensorDescriptor::from(tensor::Shape::from({rows, columns}), tensor::DType::f16));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index{row, column};
            const auto value = scale * static_cast<float>((row + 1) + (column % 7));
            tensor::write_f16_from_f32(storage, std::span<const std::uint64_t>(index), value);
        }
    }

    return storage;
}

tensor::TensorStorage make_f16_gguf_weight_matrix(std::uint64_t rows, std::uint64_t columns,
                                                  float scale) {
    auto storage = tensor::TensorStorage::allocate(
        tensor::TensorDescriptor::from(tensor::Shape::from({rows, columns}), tensor::DType::f16));
    auto bytes = storage.mutable_bytes();
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto value = scale * static_cast<float>((row % 17) + 1 + (column % 11));
            const auto bits = tensor::f32_to_f16_bits(value);
            const auto byte_offset =
                static_cast<std::size_t>((column * rows + row) * sizeof(std::uint16_t));
            bytes[byte_offset] = static_cast<std::uint8_t>(bits & 0xFFU);
            bytes[byte_offset + 1] = static_cast<std::uint8_t>(bits >> 8U);
        }
    }

    return storage;
}

std::vector<std::uint8_t> make_repeated_payload(std::size_t byte_count) {
    std::vector<std::uint8_t> bytes(byte_count);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(index % 251);
    }

    return bytes;
}

model_format::GGUFMetadataSection make_synthetic_tokenizer_metadata() {
    std::vector<model_format::GGUFMetadataEntry> entries{
        metadata_string("general.architecture", "llama"),
        metadata_u32("llama.vocab_size", 7),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array("tokenizer.ggml.tokens",
                              {"<s>", "</s>", "Hello", " world", "!", "Hello world", "<unk>"}),
        metadata_f32_array("tokenizer.ggml.scores",
                           {0.0F, 0.0F, -1.0F, -1.1F, -1.2F, -0.2F, -10.0F}),
        metadata_i32_array("tokenizer.ggml.token_type", {3, 3, 1, 1, 1, 1, 2}),
        metadata_u32("tokenizer.ggml.bos_token_id", 0),
        metadata_u32("tokenizer.ggml.eos_token_id", 1),
        metadata_u32("tokenizer.ggml.unknown_token_id", 6),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
    };
    const auto metadata_count = static_cast<std::uint64_t>(entries.size());

    return model_format::GGUFMetadataSection{
        .header =
            model_format::GGUFHeader{
                .version = 3,
                .tensor_count = 0,
                .metadata_kv_count = metadata_count,
            },
        .entries = std::move(entries),
        .encoded_byte_count = 0,
    };
}

} // namespace local_gguf_engine::benchmarks
