#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"
#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/frontends/cli/generate.hpp"
#include "local_gguf_engine/frontends/cli/gguf_inspect.hpp"
#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using local_gguf_engine::benchmarks::BenchmarkConfig;
using local_gguf_engine::benchmarks::BenchmarkMeasurement;
using local_gguf_engine::benchmarks::BenchmarkReport;
using local_gguf_engine::benchmarks::benchmark_output_path_is_under_results;
using local_gguf_engine::benchmarks::default_benchmark_output_path;
using local_gguf_engine::benchmarks::format_benchmark_report;
using local_gguf_engine::benchmarks::parse_benchmark_args;
using local_gguf_engine::engine::append_llama_kv_cache_layer_f32;
using local_gguf_engine::engine::bind_llama_model_weights;
using local_gguf_engine::engine::copy_llama_backend_kv_cache_to_cpu;
using local_gguf_engine::engine::copy_llama_attention_layer_weights_to_backend;
using local_gguf_engine::engine::copy_llama_block_weights_to_backend;
using local_gguf_engine::engine::copy_llama_feed_forward_layer_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_attention_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_block_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_feed_forward_weights_to_backend;
using local_gguf_engine::engine::copy_llama_model_logits_weights_to_backend;
using local_gguf_engine::engine::create_llama_kv_cache;
using local_gguf_engine::engine::create_llama_backend_kv_cache;
using local_gguf_engine::engine::embed_llama_token_ids_f32;
using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::EngineTensor;
using local_gguf_engine::engine::generate_llama_argmax_tokens_f32;
using local_gguf_engine::engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::generate_llama_sampled_tokens_f32;
using local_gguf_engine::engine::generate_llama_sampled_tokens_with_kv_cache_f32;
using local_gguf_engine::engine::LlamaAttentionLayerBackendWeights;
using local_gguf_engine::engine::LlamaAttentionLayerWeights;
using local_gguf_engine::engine::LlamaBlockBackendWeights;
using local_gguf_engine::engine::LlamaBlockWeights;
using local_gguf_engine::engine::LlamaFeedForwardLayerWeights;
using local_gguf_engine::engine::LlamaKVCache;
using local_gguf_engine::engine::LlamaKVCacheLayer;
using local_gguf_engine::engine::llama_kv_cache_layer_key_prefix_f32;
using local_gguf_engine::engine::llama_kv_cache_layer_value_prefix_f32;
using local_gguf_engine::engine::LlamaModelAttentionBackendWeights;
using local_gguf_engine::engine::LlamaModelBlockBackendWeights;
using local_gguf_engine::engine::LlamaModelFeedForwardBackendWeights;
using local_gguf_engine::engine::LlamaModelLogitsBackendWeights;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::LlamaModelWeights;
using local_gguf_engine::engine::ModelArchitecture;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::engine::model_architecture_name;
using local_gguf_engine::engine::project_llama_logits_f32;
using local_gguf_engine::engine::run_llama_attention_layer_f32;
using local_gguf_engine::engine::run_llama_attention_layer_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_block_f32;
using local_gguf_engine::engine::run_llama_block_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_feed_forward_layer_f32;
using local_gguf_engine::engine::run_llama_model_blocks_f32;
using local_gguf_engine::engine::run_llama_model_blocks_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_model_logits_f32;
using local_gguf_engine::engine::run_llama_model_logits_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_model_next_token_argmax_f32;
using local_gguf_engine::engine::run_llama_model_next_token_argmax_with_kv_cache_f32;
using local_gguf_engine::engine::run_llama_model_next_token_sampled_f32;
using local_gguf_engine::engine::sample_argmax_token_f32;
using local_gguf_engine::engine::sample_token_f32;
using local_gguf_engine::engine::SamplingConfig;
using local_gguf_engine::engine::SamplingState;
using local_gguf_engine::engine::top_logit_tokens_f32;
using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::model_format::GGUFHeader;
using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::model_format::GGUFMetadataArray;
using local_gguf_engine::model_format::GGUFMetadataSection;
using local_gguf_engine::model_format::GGUFMetadataValue;
using local_gguf_engine::model_format::GGUFMetadataValueType;
using local_gguf_engine::model_format::GGUFModelView;
using local_gguf_engine::model_format::GGUFReadError;
using local_gguf_engine::model_format::GGUFReadErrorCode;
using local_gguf_engine::model_format::GGUFTensorDirectory;
using local_gguf_engine::model_format::GGUFTensorInfo;
using local_gguf_engine::model_format::GGUFTensorPayloadView;
using local_gguf_engine::model_format::load_gguf_tensor_payload;
using local_gguf_engine::model_format::load_gguf_tensor_payloads;
using local_gguf_engine::model_format::read_gguf_header;
using local_gguf_engine::model_format::read_gguf_metadata_section;
using local_gguf_engine::model_format::read_gguf_model_view;
using local_gguf_engine::model_format::read_gguf_tensor_directory;
using local_gguf_engine::frontends::cli::GenerateOptions;
using local_gguf_engine::frontends::cli::generate_text_from_gguf_bytes;
using local_gguf_engine::frontends::cli::inspect_gguf_bytes;
using local_gguf_engine::frontends::cli::run_generate;
using local_gguf_engine::frontends::cli::run_gguf_inspect;
using local_gguf_engine::frontends::cli::write_gguf_inspection_summary;
using local_gguf_engine::tensor::byte_size;
using local_gguf_engine::tensor::copy_tensor_from_backend;
using local_gguf_engine::tensor::copy_tensor_to_backend;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::element_byte_offset;
using local_gguf_engine::tensor::element_bytes;
using local_gguf_engine::tensor::element_count;
using local_gguf_engine::tensor::linear_index;
using local_gguf_engine::tensor::mutable_element_bytes;
using local_gguf_engine::tensor::dequantize_q8_0_block;
using local_gguf_engine::tensor::f32_to_f16_bits;
using local_gguf_engine::tensor::q8_0_block_count;
using local_gguf_engine::tensor::q8_0_block_view;
using local_gguf_engine::tensor::Q8_0BlockView;
using local_gguf_engine::tensor::read_f16_as_f32;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::tensor_backend_unavailable_reason;
using local_gguf_engine::tensor::write_f16_from_f32;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::causal_scaled_dot_product_attention_f32;
using local_gguf_engine::tensor::ops::copy_f32;
using local_gguf_engine::tensor::ops::F32MatrixColumnView;
using local_gguf_engine::tensor::ops::F32RoPEMatrixColumnView;
using local_gguf_engine::tensor::ops::fill_f32;
using local_gguf_engine::tensor::ops::gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::llama_cached_attention_f32_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32;
using local_gguf_engine::tensor::ops::matmul_f32_f16;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight_triple_to_backend;
using local_gguf_engine::tensor::ops::mean_square_f32;
using local_gguf_engine::tensor::ops::multiply_f32;
using local_gguf_engine::tensor::ops::residual_gated_silu_projection_f32_f16_gguf_weight_to_backend;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rms_f32;
using local_gguf_engine::tensor::ops::rope_f32;
using local_gguf_engine::tensor::ops::rope_f32_with_position_start;
using local_gguf_engine::tensor::ops::scaled_dot_product_attention_f32;
using local_gguf_engine::tensor::ops::silu_f32;
using local_gguf_engine::tensor::ops::silu_multiply_f32_to_backend;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::sum_f32;
using local_gguf_engine::tensor::ops::sum_squares_f32;
using local_gguf_engine::tokenizer::load_tokenizer_from_gguf_metadata;
using local_gguf_engine::tokenizer::format_generate_prompt_from_gguf_metadata;
using local_gguf_engine::tokenizer::TokenizerDecodeOptions;
using local_gguf_engine::tokenizer::TokenizerEncodeOptions;
using local_gguf_engine::tokenizer::TokenizerError;
using local_gguf_engine::tokenizer::TokenizerErrorCode;
using local_gguf_engine::tokenizer::TokenizerModel;
using local_gguf_engine::tokenizer::tokenizer_model_name;

class TestFailure final : public std::exception {
public:
    explicit TestFailure(std::string message) : message_(std::move(message)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

void expect_close(float actual, float expected, float tolerance, std::string_view message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream stream;
        stream << message << ": expected " << expected << ", got " << actual;
        throw TestFailure(stream.str());
    }
}

void expect_f32_matrix_close(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    expect(actual.descriptor().dtype == DType::f32, std::string(message) + " dtype mismatch");
    expect(actual.descriptor().shape.rank() == 2, std::string(message) + " rank mismatch");

    const auto rows = actual.descriptor().shape.dimension(0);
    const auto columns = actual.descriptor().shape.dimension(1);
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(actual, std::span<const std::uint64_t>(index)),
                read_f32(expected, std::span<const std::uint64_t>(index)),
                tolerance,
                message
            );
        }
    }
}

void expect_llama_kv_cache_close(
    const LlamaKVCache& actual,
    const LlamaKVCache& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.layers.size() == expected.layers.size(), std::string(message) + " layer count mismatch");
    expect(actual.context_length == expected.context_length, std::string(message) + " context length mismatch");
    expect(actual.key_width == expected.key_width, std::string(message) + " key width mismatch");
    expect(actual.value_width == expected.value_width, std::string(message) + " value width mismatch");

    for (std::size_t layer_index = 0; layer_index < actual.layers.size(); ++layer_index) {
        const auto prefix = std::string(message) + " layer " + std::to_string(layer_index);
        expect(
            actual.layers[layer_index].filled_token_count == expected.layers[layer_index].filled_token_count,
            prefix + " filled token count mismatch"
        );
        expect_f32_matrix_close(
            actual.layers[layer_index].key,
            expected.layers[layer_index].key,
            tolerance,
            prefix + " key cache"
        );
        expect_f32_matrix_close(
            actual.layers[layer_index].value,
            expected.layers[layer_index].value,
            tolerance,
            prefix + " value cache"
        );
    }
}

std::vector<std::uint8_t> load_hex_fixture(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw TestFailure("fixture missing: " + path.string());
    }

    std::vector<std::uint8_t> bytes;
    std::string token;
    while (input >> token) {
        const auto value = std::stoul(token, nullptr, 16);
        if (value > 0xFFU) {
            throw TestFailure("fixture byte is out of range: " + token);
        }
        bytes.push_back(static_cast<std::uint8_t>(value));
    }

    return bytes;
}

std::vector<std::uint8_t> fixture(std::string_view name) {
    return load_hex_fixture(std::filesystem::path("tests") / "fixtures" / (std::string(name) + ".hex"));
}

class TempBinaryFixture final {
public:
    explicit TempBinaryFixture(std::string_view name)
        : path_(std::filesystem::temp_directory_path() / ("local-gguf-engine-" + std::string(name) + ".gguf")) {
        const auto bytes = fixture(name);
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw TestFailure("could not create temp binary fixture: " + path_.string());
        }

        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!output) {
            throw TestFailure("could not write temp binary fixture: " + path_.string());
        }
    }

    ~TempBinaryFixture() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

class TempBinaryFile final {
public:
    TempBinaryFile(std::string_view name, std::span<const std::uint8_t> bytes)
        : path_(std::filesystem::temp_directory_path() / ("local-gguf-engine-" + std::string(name) + ".gguf")) {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw TestFailure("could not create temp binary file: " + path_.string());
        }

        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!output) {
            throw TestFailure("could not write temp binary file: " + path_.string());
        }
    }

    ~TempBinaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

int run_gguf_inspect_for_path(
    const std::filesystem::path& path,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "gguf-inspect";
    const std::string path_string = path.string();
    const char* argv[] = {program.c_str(), path_string.c_str()};
    return run_gguf_inspect(2, argv, output, error);
}

int run_generate_for_path(
    const std::filesystem::path& path,
    std::string_view prompt,
    std::string_view max_new_tokens,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string prompt_string(prompt);
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        prompt_string.c_str(),
        max_new_tokens_string.c_str(),
    };
    return run_generate(4, argv, output, error);
}

template <typename Value>
const Value& metadata_data(
    const GGUFMetadataValue& metadata_value,
    GGUFMetadataValueType expected_type,
    std::string_view message
) {
    expect(metadata_value.type == expected_type, message);
    expect(std::holds_alternative<Value>(metadata_value.data), "metadata value storage type mismatch");
    return std::get<Value>(metadata_value.data);
}

GGUFMetadataEntry metadata_string(std::string key, std::string value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::string,
            .data = std::move(value),
        },
    };
}

GGUFMetadataEntry metadata_u32(std::string key, std::uint32_t value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::uint32,
            .data = value,
        },
    };
}

GGUFMetadataEntry metadata_f32(std::string key, float value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::float32,
            .data = value,
        },
    };
}

GGUFMetadataEntry metadata_bool(std::string key, bool value) {
    return GGUFMetadataEntry {
        .key = std::move(key),
        .value = GGUFMetadataValue {
            .type = GGUFMetadataValueType::boolean,
            .data = value,
        },
    };
}

GGUFMetadataEntry metadata_string_array(std::string key, std::vector<std::string> values) {
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

GGUFMetadataEntry metadata_f32_array(std::string key, std::vector<float> values) {
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

GGUFMetadataEntry metadata_i32_array(std::string key, std::vector<std::int32_t> values) {
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

void append_u8(std::vector<std::uint8_t>& bytes, std::uint8_t value) {
    bytes.push_back(value);
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (std::uint32_t shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_i32(std::vector<std::uint8_t>& bytes, std::int32_t value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void append_f32(std::vector<std::uint8_t>& bytes, float value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void append_gguf_string(std::vector<std::uint8_t>& bytes, std::string_view value) {
    append_u64(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t gguf_metadata_type_id(GGUFMetadataValueType type) {
    return static_cast<std::uint32_t>(type);
}

void append_metadata_array(std::vector<std::uint8_t>& bytes, const GGUFMetadataArray& array) {
    append_u32(bytes, gguf_metadata_type_id(array.element_type));

    switch (array.element_type) {
        case GGUFMetadataValueType::string: {
            const auto& values = std::get<std::vector<std::string>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto& value : values) {
                append_gguf_string(bytes, value);
            }
            return;
        }
        case GGUFMetadataValueType::float32: {
            const auto& values = std::get<std::vector<float>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_f32(bytes, value);
            }
            return;
        }
        case GGUFMetadataValueType::int32: {
            const auto& values = std::get<std::vector<std::int32_t>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_i32(bytes, value);
            }
            return;
        }
        default:
            throw TestFailure("test GGUF writer unsupported metadata array type");
    }
}

void append_metadata_value(std::vector<std::uint8_t>& bytes, const GGUFMetadataValue& value) {
    append_u32(bytes, gguf_metadata_type_id(value.type));

    switch (value.type) {
        case GGUFMetadataValueType::uint32:
            append_u32(bytes, std::get<std::uint32_t>(value.data));
            return;
        case GGUFMetadataValueType::float32:
            append_f32(bytes, std::get<float>(value.data));
            return;
        case GGUFMetadataValueType::boolean:
            append_u8(bytes, static_cast<std::uint8_t>(std::get<bool>(value.data) ? 1 : 0));
            return;
        case GGUFMetadataValueType::string:
            append_gguf_string(bytes, std::get<std::string>(value.data));
            return;
        case GGUFMetadataValueType::array:
            append_metadata_array(bytes, std::get<GGUFMetadataArray>(value.data));
            return;
        default:
            throw TestFailure("test GGUF writer unsupported metadata value type");
    }
}

void append_metadata_entry(std::vector<std::uint8_t>& bytes, const GGUFMetadataEntry& entry) {
    append_gguf_string(bytes, entry.key);
    append_metadata_value(bytes, entry.value);
}

std::uint32_t gguf_type_id_for_dtype(DType dtype) {
    switch (dtype) {
        case DType::f32:
            return 0;
        case DType::f16:
            return 1;
        case DType::q8_0:
            return 8;
    }

    throw TestFailure("test GGUF writer unsupported tensor dtype");
}

struct TinyGGUFTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type_id = 0;
    std::vector<std::uint8_t> bytes;
};

std::size_t align_size(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }

    return value + alignment - remainder;
}

std::vector<TinyGGUFTensor> tiny_gguf_tensors_from_engine_model(const EngineModel& model) {
    std::vector<TinyGGUFTensor> tensors;
    tensors.reserve(model.tensors().size());

    for (const auto& tensor : model.tensors()) {
        const auto dimensions = tensor.storage.descriptor().shape.dimensions();
        tensors.push_back(TinyGGUFTensor {
            .name = tensor.name,
            .dimensions = std::vector<std::uint64_t>(dimensions.begin(), dimensions.end()),
            .type_id = gguf_type_id_for_dtype(tensor.storage.descriptor().dtype),
            .bytes = std::vector<std::uint8_t>(tensor.storage.bytes().begin(), tensor.storage.bytes().end()),
        });
    }

    return tensors;
}

std::vector<std::uint8_t> write_tiny_gguf_bytes(
    const std::vector<GGUFMetadataEntry>& metadata,
    const std::vector<TinyGGUFTensor>& tensors
) {
    constexpr std::size_t alignment = 32;
    std::vector<std::uint64_t> tensor_offsets;
    tensor_offsets.reserve(tensors.size());

    std::size_t payload_offset = 0;
    for (const auto& tensor : tensors) {
        tensor_offsets.push_back(static_cast<std::uint64_t>(payload_offset));
        payload_offset = align_size(payload_offset + tensor.bytes.size(), alignment);
    }

    std::vector<std::uint8_t> bytes;
    bytes.push_back(0x47);
    bytes.push_back(0x47);
    bytes.push_back(0x55);
    bytes.push_back(0x46);
    append_u32(bytes, 3);
    append_u64(bytes, static_cast<std::uint64_t>(tensors.size()));
    append_u64(bytes, static_cast<std::uint64_t>(metadata.size()));

    for (const auto& entry : metadata) {
        append_metadata_entry(bytes, entry);
    }

    for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto& tensor = tensors[tensor_index];
        append_gguf_string(bytes, tensor.name);
        append_u32(bytes, static_cast<std::uint32_t>(tensor.dimensions.size()));
        for (const auto dimension : tensor.dimensions) {
            append_u64(bytes, dimension);
        }
        append_u32(bytes, tensor.type_id);
        append_u64(bytes, tensor_offsets[tensor_index]);
    }

    const auto tensor_data_start = align_size(bytes.size(), alignment);
    bytes.resize(tensor_data_start, 0);

    for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto& tensor = tensors[tensor_index];
        const auto write_offset = tensor_data_start + static_cast<std::size_t>(tensor_offsets[tensor_index]);
        if (bytes.size() < write_offset) {
            bytes.resize(write_offset, 0);
        }
        bytes.insert(bytes.end(), tensor.bytes.begin(), tensor.bytes.end());
    }

    return bytes;
}

std::vector<GGUFMetadataEntry> tiny_llama_metadata() {
    return {
        metadata_string("general.architecture", "llama"),
        metadata_string("general.name", "Tiny Llama Fixture"),
        metadata_u32("llama.context_length", 16),
        metadata_u32("llama.embedding_length", 8),
        metadata_u32("llama.block_count", 2),
        metadata_u32("llama.feed_forward_length", 32),
        metadata_u32("llama.attention.head_count", 4),
        metadata_u32("llama.attention.head_count_kv", 2),
        metadata_u32("llama.attention.key_length", 2),
        metadata_u32("llama.attention.value_length", 2),
        metadata_u32("llama.rope.dimension_count", 2),
        metadata_u32("llama.vocab_size", 64),
        metadata_f32("llama.rope.freq_base", 10000.0F),
        metadata_f32("llama.attention.layer_norm_rms_epsilon", 0.00001F),
    };
}

std::vector<GGUFMetadataEntry> tiny_llama_tokenizer_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 7);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {"<s>", "</s>", "Hello", " world", "!", "Hello world", "<unk>"}
    ));
    metadata.push_back(metadata_f32_array(
        "tokenizer.ggml.scores",
        {0.0F, 0.0F, -1.0F, -1.1F, -1.2F, -0.2F, -10.0F}
    ));
    metadata.push_back(metadata_i32_array(
        "tokenizer.ggml.token_type",
        {3, 3, 1, 1, 1, 1, 2}
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 0));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 6));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", true));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    return metadata;
}

std::vector<GGUFMetadataEntry> tiny_gpt2_byte_decoder_tokenizer_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 8);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {"<s>", "</s>", "Hello", "\xC4\xA0world", "!", "\xC4\x8A", "\xC3\x83\xC2\xA5", "<unk>"}
    ));
    metadata.push_back(metadata_f32_array(
        "tokenizer.ggml.scores",
        {0.0F, 0.0F, -1.0F, -1.1F, -1.2F, -1.3F, -1.4F, -10.0F}
    ));
    metadata.push_back(metadata_i32_array(
        "tokenizer.ggml.token_type",
        {3, 3, 1, 1, 1, 1, 1, 2}
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 0));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 7));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", true));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    return metadata;
}

std::vector<GGUFMetadataEntry> tiny_gpt2_bpe_tokenizer_metadata(bool add_space_prefix) {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 25);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {
            "<s>", "</s>", "H", "e", "l", "o", "\xC4\xA0", "w", "r", "d", "!",
            "\xC3\x83", "\xC2\xA5", "He", "Hel", "Hell", "Hello",
            "\xC4\xA0w", "\xC4\xA0wo", "\xC4\xA0wor", "\xC4\xA0worl", "\xC4\xA0world",
            "\xC3\x83\xC2\xA5", "\xC4\xA0Hello", "<unk>",
        }
    ));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.merges",
        {
            "H e", "He l", "Hel l", "Hell o",
            "\xC4\xA0 w", "\xC4\xA0w o", "\xC4\xA0wo r", "\xC4\xA0wor l", "\xC4\xA0worl d",
            "\xC3\x83 \xC2\xA5", "\xC4\xA0 Hello",
        }
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 0));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 24));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", true));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_space_prefix", add_space_prefix));
    return metadata;
}

std::string smollm2_chat_template_fixture() {
    return "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n"
        "' + message['content'] + '<|im_end|>' + '\n"
        "'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n"
        "' }}{% endif %}";
}

std::string smollm2_formatted_prompt_fixture(std::string_view prompt) {
    std::string formatted;
    formatted += "<|im_start|>system\n";
    formatted += "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
    formatted += "<|im_end|>\n<|im_start|>user\n";
    formatted += prompt;
    formatted += "<|im_end|>\n<|im_start|>assistant\n";
    return formatted;
}

std::vector<GGUFMetadataEntry> tiny_gpt2_special_tokenizer_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata[11] = metadata_u32("llama.vocab_size", 4);
    metadata.push_back(metadata_string("tokenizer.ggml.model", "gpt2"));
    metadata.push_back(metadata_string_array(
        "tokenizer.ggml.tokens",
        {"<unk>", "<|im_start|>", "<|im_end|>", "Hello"}
    ));
    metadata.push_back(metadata_i32_array(
        "tokenizer.ggml.token_type",
        {2, 3, 3, 1}
    ));
    metadata.push_back(metadata_u32("tokenizer.ggml.bos_token_id", 1));
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 2));
    metadata.push_back(metadata_u32("tokenizer.ggml.unknown_token_id", 0));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_bos_token", false));
    metadata.push_back(metadata_bool("tokenizer.ggml.add_eos_token", false));
    return metadata;
}

std::vector<GGUFMetadataEntry> tiny_generate_cli_metadata(const LlamaModelConfig& config) {
    return {
        metadata_string("general.architecture", "llama"),
        metadata_string("general.name", "Tiny Generate Fixture"),
        metadata_u32("llama.context_length", config.context_length),
        metadata_u32("llama.embedding_length", config.embedding_length),
        metadata_u32("llama.block_count", config.block_count),
        metadata_u32("llama.feed_forward_length", config.feed_forward_length),
        metadata_u32("llama.attention.head_count", config.attention_head_count),
        metadata_u32("llama.attention.head_count_kv", config.attention_head_count_kv),
        metadata_u32("llama.attention.key_length", config.attention_key_length),
        metadata_u32("llama.attention.value_length", config.attention_value_length),
        metadata_u32("llama.rope.dimension_count", config.rope_dimension_count),
        metadata_u32("llama.vocab_size", config.vocab_size),
        metadata_f32("llama.rope.freq_base", config.rope_freq_base),
        metadata_f32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_epsilon),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array("tokenizer.ggml.tokens", {"<s>", "Hello", "!"}),
        metadata_f32_array("tokenizer.ggml.scores", {0.0F, -1.0F, -0.5F}),
        metadata_i32_array("tokenizer.ggml.token_type", {3, 1, 1}),
        metadata_u32("tokenizer.ggml.bos_token_id", 0),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
    };
}

std::vector<GGUFMetadataEntry> tiny_generate_cli_chat_metadata(const LlamaModelConfig& config) {
    return {
        metadata_string("general.architecture", "llama"),
        metadata_string("general.name", "Tiny Chat Generate Fixture"),
        metadata_u32("llama.context_length", config.context_length),
        metadata_u32("llama.embedding_length", config.embedding_length),
        metadata_u32("llama.block_count", config.block_count),
        metadata_u32("llama.feed_forward_length", config.feed_forward_length),
        metadata_u32("llama.attention.head_count", config.attention_head_count),
        metadata_u32("llama.attention.head_count_kv", config.attention_head_count_kv),
        metadata_u32("llama.attention.key_length", config.attention_key_length),
        metadata_u32("llama.attention.value_length", config.attention_value_length),
        metadata_u32("llama.rope.dimension_count", config.rope_dimension_count),
        metadata_u32("llama.vocab_size", config.vocab_size),
        metadata_f32("llama.rope.freq_base", config.rope_freq_base),
        metadata_f32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_epsilon),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array(
            "tokenizer.ggml.tokens",
            {
                "<pad>",
                "<|im_start|>",
                "<|im_end|>",
                "system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face",
                "\n",
                "user\nHello",
                "assistant\n",
                "!",
            }
        ),
        metadata_i32_array(
            "tokenizer.ggml.token_type",
            {1, 3, 3, 1, 1, 1, 1, 1}
        ),
        metadata_u32("tokenizer.ggml.bos_token_id", 1),
        metadata_u32("tokenizer.ggml.eos_token_id", 2),
        metadata_bool("tokenizer.ggml.add_bos_token", false),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
        metadata_string("tokenizer.chat_template", smollm2_chat_template_fixture()),
    };
}

std::vector<GGUFMetadataEntry> tiny_generate_cli_eos_metadata(const LlamaModelConfig& config) {
    auto metadata = tiny_generate_cli_metadata(config);
    metadata.push_back(metadata_u32("tokenizer.ggml.eos_token_id", 1));
    return metadata;
}

GGUFMetadataSection metadata_section(std::vector<GGUFMetadataEntry> entries) {
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

TensorStorage tensor_f32_vector(std::span<const float> values) {
    auto storage = TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({static_cast<std::uint64_t>(values.size())}), DType::f32)
    );
    for (std::uint64_t index = 0; index < values.size(); ++index) {
        const std::array<std::uint64_t, 1> tensor_index {index};
        write_f32(storage, std::span<const std::uint64_t>(tensor_index), values[static_cast<std::size_t>(index)]);
    }

    return storage;
}

TensorStorage tensor_f32_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F32 matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> tensor_index {row, column};
            write_f32(
                storage,
                std::span<const std::uint64_t>(tensor_index),
                values[static_cast<std::size_t>(row * columns + column)]
            );
        }
    }

    return storage;
}

TensorStorage tensor_f16_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F16 matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const std::array<std::uint64_t, 2> tensor_index {row, column};
            write_f16_from_f32(
                storage,
                std::span<const std::uint64_t>(tensor_index),
                values[static_cast<std::size_t>(row * columns + column)]
            );
        }
    }

    return storage;
}

TensorStorage tensor_f16_gguf_matrix(std::uint64_t rows, std::uint64_t columns, std::span<const float> values) {
    if (values.size() != rows * columns) {
        throw TestFailure("F16 GGUF matrix fixture value count mismatch");
    }

    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f16));
    auto bytes = storage.mutable_bytes();
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto bits = f32_to_f16_bits(values[static_cast<std::size_t>(row * columns + column)]);
            const auto byte_offset = static_cast<std::size_t>((column * rows + row) * sizeof(std::uint16_t));
            bytes[byte_offset] = static_cast<std::uint8_t>(bits & 0xFFU);
            bytes[byte_offset + 1] = static_cast<std::uint8_t>(bits >> 8U);
        }
    }

    return storage;
}

GGUFModelView tiny_model_view(
    std::vector<GGUFMetadataEntry> metadata_entries,
    std::vector<GGUFTensorPayloadView> payloads
) {
    const auto metadata_count = static_cast<std::uint64_t>(metadata_entries.size());
    const auto tensor_count = static_cast<std::uint64_t>(payloads.size());

    return GGUFModelView {
        .directory = GGUFTensorDirectory {
            .metadata = GGUFMetadataSection {
                .header = GGUFHeader {
                    .version = 3,
                    .tensor_count = tensor_count,
                    .metadata_kv_count = metadata_count,
                },
                .entries = std::move(metadata_entries),
                .encoded_byte_count = 0,
            },
            .tensors = {},
            .alignment = 32,
            .tensor_infos_end_offset = 0,
            .padding_byte_count = 0,
            .tensor_data_start_offset = 0,
            .encoded_byte_count = 0,
        },
        .tensor_payloads = std::move(payloads),
    };
}

GGUFTensorPayloadView tensor_payload_view(
    std::string name,
    std::span<const std::uint8_t> bytes,
    std::uint32_t type_id = 0
) {
    return GGUFTensorPayloadView {
        .tensor = GGUFTensorInfo {
            .name = std::move(name),
            .dimensions = {2},
            .type_id = type_id,
            .offset = 0,
            .file_offset = 0,
        },
        .bytes = bytes,
    };
}

void test_default_backend_starts_on_cpu() {
    expect(
        local_gguf_engine::runtime::default_backend()
            == local_gguf_engine::runtime::BackendKind::cpu,
        "default backend should start on CPU"
    );
}

void test_tensor_shape_accepts_valid_shapes() {
    const auto vector = Shape::from({128});
    expect(vector.rank() == 1, "1D tensor shape rank mismatch");
    expect(vector.dimension(0) == 128, "1D tensor dimension mismatch");
    expect(element_count(vector) == 128, "1D tensor element count mismatch");

    const auto matrix = Shape::from({2, 3});
    expect(matrix.rank() == 2, "2D tensor shape rank mismatch");
    expect(matrix.dimensions().size() == 2, "2D tensor dimensions span size mismatch");
    expect(matrix.dimensions()[0] == 2, "2D tensor dimension 0 mismatch");
    expect(matrix.dimensions()[1] == 3, "2D tensor dimension 1 mismatch");
    expect(element_count(matrix) == 6, "2D tensor element count mismatch");

    const auto rank_four = Shape::from({2, 3, 4, 5});
    expect(rank_four.rank() == Shape::max_rank, "4D tensor shape rank mismatch");
    expect(rank_four.dimension(3) == 5, "4D tensor final dimension mismatch");
    expect(element_count(rank_four) == 120, "4D tensor element count mismatch");
    expect(sizeof(Shape) <= 40, "tensor shape should stay compact");
}

void test_rejects_tensor_shape_rank_over_four() {
    const std::array<std::uint64_t, 5> dimensions {1, 2, 3, 4, 5};

    try {
        (void)Shape::from(std::span<const std::uint64_t>(dimensions));
        throw TestFailure("rank above 4 should throw");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "error code should be invalid_shape_rank");
        expect(error.expected_value() == Shape::max_rank, "expected shape rank mismatch");
        expect(error.actual_value() == 5, "actual shape rank mismatch");
    }
}

void test_rejects_tensor_shape_element_count_overflow() {
    const auto shape = Shape::from({std::numeric_limits<std::uint64_t>::max(), 2});

    try {
        (void)element_count(shape);
        throw TestFailure("element count overflow should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::element_count_overflow,
            "error code should be element_count_overflow"
        );
    }
}

void test_calculates_tensor_byte_sizes() {
    expect(byte_size(Shape::from({2, 3}), DType::f32) == 24, "F32 tensor byte size mismatch");
    expect(byte_size(Shape::from({2, 3}), DType::f16) == 12, "F16 tensor byte size mismatch");
    expect(byte_size(Shape::from({32}), DType::q8_0) == 34, "Q8_0 single-block byte size mismatch");
    expect(byte_size(Shape::from({2, 32}), DType::q8_0) == 68, "Q8_0 multi-block byte size mismatch");
}

void test_rejects_unsupported_tensor_dtype() {
    try {
        (void)byte_size(Shape::from({1}), static_cast<DType>(255));
        throw TestFailure("unsupported dtype should throw");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_dtype, "error code should be unsupported_dtype");
        expect(error.actual_value() == 255, "unsupported dtype value mismatch");
    }
}

void test_rejects_invalid_quantized_tensor_element_count() {
    try {
        (void)byte_size(Shape::from({31}), DType::q8_0);
        throw TestFailure("invalid Q8_0 element count should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "error code should be invalid_quantized_element_count"
        );
        expect(error.expected_value() == 32, "expected Q8_0 block element count mismatch");
        expect(error.actual_value() == 31, "actual Q8_0 element count mismatch");
    }
}

void test_rejects_tensor_byte_size_overflow() {
    const auto shape = Shape::from({std::numeric_limits<std::uint64_t>::max() / 4 + 1});

    try {
        (void)byte_size(shape, DType::f32);
        throw TestFailure("byte size overflow should throw");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::byte_size_overflow, "error code should be byte_size_overflow");
    }
}

void test_creates_tensor_descriptor_from_shape_and_dtype() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f16);

    expect(descriptor.shape == Shape::from({2, 3}), "tensor descriptor shape mismatch");
    expect(descriptor.dtype == DType::f16, "tensor descriptor dtype mismatch");
    expect(descriptor.element_count == 6, "tensor descriptor element count mismatch");
    expect(descriptor.byte_count == 12, "tensor descriptor byte count mismatch");
}

void test_allocates_zero_initialized_tensor_storage() {
    const auto descriptor = TensorDescriptor::from(Shape::from({4}), DType::f32);
    auto storage = TensorStorage::allocate(descriptor);

    expect(storage.descriptor() == descriptor, "allocated tensor descriptor mismatch");
    expect(storage.bytes().size() == 16, "allocated tensor byte count mismatch");

    for (const auto byte : storage.bytes()) {
        expect(byte == 0, "allocated tensor storage should be zero initialized");
    }

    storage.mutable_bytes()[0] = 0x7F;
    expect(storage.bytes()[0] == 0x7F, "mutable tensor storage byte should be visible through const view");
}

void test_copies_tensor_storage_from_bytes() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, 34> bytes {
        0x00, 0x3C, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
        0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
        0x1F, 0x20,
    };

    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    expect(storage.descriptor() == descriptor, "copied tensor descriptor mismatch");
    expect(storage.bytes().size() == bytes.size(), "copied tensor byte count mismatch");
    expect(storage.bytes().data() != bytes.data(), "tensor storage should own a byte copy");
    expect(storage.bytes()[0] == 0x00, "copied tensor byte 0 mismatch");
    expect(storage.bytes()[1] == 0x3C, "copied tensor byte 1 mismatch");
    expect(storage.bytes()[33] == 0x20, "copied tensor final byte mismatch");
}

void test_rejects_tensor_storage_byte_count_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const std::array<std::uint8_t, 23> bytes {};

    try {
        (void)TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));
        throw TestFailure("storage byte count mismatch should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_storage_byte_count,
            "error code should be invalid_storage_byte_count"
        );
        expect(error.expected_value() == 24, "expected storage byte count mismatch");
        expect(error.actual_value() == 23, "actual storage byte count mismatch");
    }
}

void test_rejects_tensor_descriptor_with_invalid_shape_dtype_combination() {
    try {
        (void)TensorDescriptor::from(Shape::from({2, 15}), DType::q8_0);
        throw TestFailure("invalid Q8_0 descriptor shape should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "error code should be invalid_quantized_element_count"
        );
        expect(error.expected_value() == 32, "expected Q8_0 descriptor block size mismatch");
        expect(error.actual_value() == 30, "actual Q8_0 descriptor element count mismatch");
    }
}

void test_decodes_q8_0_block_values() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, Q8_0BlockView::byte_count> bytes {
        0x00, 0x38,
        0x00, 0x02, 0xFE, 0x7F, 0x80, 0x04, 0x06, 0x08,
        0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18,
        0x1A, 0x1C, 0x1E, 0x20, 0x22, 0x24, 0x26, 0x28,
        0x2A, 0x2C, 0x2E, 0x30, 0x32, 0x34, 0x36, 0x38,
    };
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    expect(q8_0_block_count(descriptor) == 1, "Q8_0 block count mismatch");

    const auto block = q8_0_block_view(storage, 0);
    expect(block.scale_bits() == 0x3800U, "Q8_0 scale bits mismatch");
    expect(block.scale() == 0.5F, "Q8_0 scale decode mismatch");
    expect(block.quantized_bytes().size() == Q8_0BlockView::element_count, "Q8_0 quantized byte count mismatch");
    expect(block.quantized_value(0) == 0, "Q8_0 quantized value 0 mismatch");
    expect(block.quantized_value(1) == 2, "Q8_0 quantized value 1 mismatch");
    expect(block.quantized_value(2) == static_cast<std::int8_t>(-2), "Q8_0 quantized value 2 mismatch");
    expect(block.quantized_value(3) == 127, "Q8_0 quantized value 3 mismatch");
    expect(block.quantized_value(4) == static_cast<std::int8_t>(-128), "Q8_0 quantized value 4 mismatch");

    const auto values = dequantize_q8_0_block(storage, 0);
    expect(values[0] == 0.0F, "Q8_0 dequantized value 0 mismatch");
    expect(values[1] == 1.0F, "Q8_0 dequantized value 1 mismatch");
    expect(values[2] == -1.0F, "Q8_0 dequantized value 2 mismatch");
    expect(values[3] == 63.5F, "Q8_0 dequantized value 3 mismatch");
    expect(values[4] == -64.0F, "Q8_0 dequantized value 4 mismatch");
}

void test_views_loaded_gguf_q8_0_block() {
    const auto bytes = fixture("valid-q8-0-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(q8_0_block_count(loaded.storage.descriptor()) == 1, "loaded Q8_0 block count mismatch");

    const auto block = q8_0_block_view(loaded.storage, 0);
    expect(block.scale_bits() == 0x0100U, "loaded Q8_0 scale bits mismatch");
    expect(block.quantized_value(0) == 2, "loaded Q8_0 quantized value 0 mismatch");
    expect(block.quantized_value(31) == 33, "loaded Q8_0 quantized final value mismatch");
}

void test_rejects_q8_0_block_access_for_unquantized_dtype() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);

    try {
        (void)q8_0_block_count(descriptor);
        throw TestFailure("Q8_0 block count should reject F32 descriptor");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_quantized_dtype,
            "error code should be unsupported_quantized_dtype"
        );
        expect(error.expected_value() == 8, "expected Q8_0 dtype id mismatch");
        expect(error.actual_value() == 0, "actual F32 dtype id mismatch");
    }
}

void test_rejects_q8_0_block_index_out_of_range() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, Q8_0BlockView::byte_count> bytes {};
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    try {
        (void)q8_0_block_view(storage, 1);
        throw TestFailure("Q8_0 block index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::quantized_block_index_out_of_range,
            "error code should be quantized_block_index_out_of_range"
        );
        expect(error.expected_value() == 1, "expected Q8_0 block count mismatch");
        expect(error.actual_value() == 1, "actual Q8_0 block index mismatch");
    }
}

void test_rejects_q8_0_block_element_index_out_of_range() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint8_t, Q8_0BlockView::byte_count> bytes {};
    const auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));
    const auto block = q8_0_block_view(storage, 0);

    try {
        (void)block.quantized_value(Q8_0BlockView::element_count);
        throw TestFailure("Q8_0 block element index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::quantized_block_element_index_out_of_range,
            "error code should be quantized_block_element_index_out_of_range"
        );
        expect(error.expected_value() == Q8_0BlockView::element_count, "expected Q8_0 element count mismatch");
        expect(error.actual_value() == Q8_0BlockView::element_count, "actual Q8_0 element index mismatch");
    }
}

void test_calculates_row_major_linear_indices() {
    const auto matrix = Shape::from({2, 3});
    const std::array<std::uint64_t, 2> matrix_index {1, 2};
    expect(
        linear_index(matrix, std::span<const std::uint64_t>(matrix_index)) == 5,
        "2D row-major linear index mismatch"
    );

    const auto rank_three = Shape::from({2, 3, 4});
    const std::array<std::uint64_t, 3> rank_three_index {1, 2, 3};
    expect(
        linear_index(rank_three, std::span<const std::uint64_t>(rank_three_index)) == 23,
        "3D row-major linear index mismatch"
    );
}

void test_calculates_unquantized_element_byte_offsets() {
    const auto f32_descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const std::array<std::uint64_t, 2> f32_index {1, 2};
    expect(
        element_byte_offset(f32_descriptor, std::span<const std::uint64_t>(f32_index)) == 20,
        "F32 element byte offset mismatch"
    );

    const auto f16_descriptor = TensorDescriptor::from(Shape::from({2, 3, 4}), DType::f16);
    const std::array<std::uint64_t, 3> f16_index {1, 0, 2};
    expect(
        element_byte_offset(f16_descriptor, std::span<const std::uint64_t>(f16_index)) == 28,
        "F16 element byte offset mismatch"
    );
}

void test_reads_and_mutates_unquantized_element_bytes() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    const std::array<std::uint8_t, 16> bytes {
        0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B,
        0x0C, 0x0D, 0x0E, 0x0F,
    };
    auto storage = TensorStorage::from_bytes(descriptor, std::span<const std::uint8_t>(bytes));

    const std::array<std::uint64_t, 2> read_index {1, 0};
    const auto read_bytes = element_bytes(storage, std::span<const std::uint64_t>(read_index));
    expect(read_bytes.size() == 4, "F32 element byte view size mismatch");
    expect(read_bytes[0] == 0x08, "F32 element byte view byte 0 mismatch");
    expect(read_bytes[3] == 0x0B, "F32 element byte view byte 3 mismatch");

    const std::array<std::uint64_t, 2> write_index {0, 1};
    auto write_bytes = mutable_element_bytes(storage, std::span<const std::uint64_t>(write_index));
    write_bytes[0] = 0xAA;
    write_bytes[3] = 0xDD;
    expect(storage.bytes()[4] == 0xAA, "mutable F32 element byte 0 mismatch");
    expect(storage.bytes()[7] == 0xDD, "mutable F32 element byte 3 mismatch");
}

void test_rejects_tensor_index_rank_mismatch() {
    const auto shape = Shape::from({2, 3});
    const std::array<std::uint64_t, 1> index {1};

    try {
        (void)linear_index(shape, std::span<const std::uint64_t>(index));
        throw TestFailure("tensor index rank mismatch should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_tensor_index_rank,
            "error code should be invalid_tensor_index_rank"
        );
        expect(error.expected_value() == 2, "expected tensor index rank mismatch");
        expect(error.actual_value() == 1, "actual tensor index rank mismatch");
    }
}

void test_rejects_tensor_index_out_of_range() {
    const auto shape = Shape::from({2, 3});
    const std::array<std::uint64_t, 2> index {2, 0};

    try {
        (void)linear_index(shape, std::span<const std::uint64_t>(index));
        throw TestFailure("tensor index out of range should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_index_out_of_range,
            "error code should be tensor_index_out_of_range"
        );
        expect(error.expected_value() == 2, "expected tensor index dimension size mismatch");
        expect(error.actual_value() == 2, "actual tensor index mismatch");
    }
}

void test_rejects_quantized_tensor_element_indexing() {
    const auto descriptor = TensorDescriptor::from(Shape::from({32}), DType::q8_0);
    const std::array<std::uint64_t, 1> index {0};

    try {
        (void)element_byte_offset(descriptor, std::span<const std::uint64_t>(index));
        throw TestFailure("quantized tensor element indexing should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_indexed_dtype,
            "error code should be unsupported_indexed_dtype"
        );
        expect(error.actual_value() == 8, "unsupported indexed dtype value mismatch");
    }
}

void test_reads_and_writes_f32_scalars_by_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto storage = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 2> first_index {0, 1};
    const std::array<std::uint64_t, 2> second_index {1, 0};
    write_f32(storage, std::span<const std::uint64_t>(first_index), 1.5F);
    write_f32(storage, std::span<const std::uint64_t>(second_index), -2.25F);

    expect(read_f32(storage, std::span<const std::uint64_t>(first_index)) == 1.5F, "F32 read/write value mismatch");
    expect(read_f32(storage, std::span<const std::uint64_t>(second_index)) == -2.25F, "F32 read/write negative value mismatch");

    const auto first_bytes = element_bytes(storage, std::span<const std::uint64_t>(first_index));
    expect(first_bytes.size() == sizeof(float), "F32 typed accessor byte size mismatch");
}

void test_reads_f32_scalars_from_loaded_gguf_payload() {
    const auto bytes = fixture("valid-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    const std::array<std::uint64_t, 1> first_index {0};
    const std::array<std::uint64_t, 1> second_index {1};
    expect(read_f32(loaded.storage, std::span<const std::uint64_t>(first_index)) == 1.0F, "loaded GGUF F32 scalar 0 mismatch");
    expect(read_f32(loaded.storage, std::span<const std::uint64_t>(second_index)) == 2.0F, "loaded GGUF F32 scalar 1 mismatch");
}

void test_rejects_f32_accessor_dtype_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f16);
    auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {0};

    try {
        (void)read_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F32 read accessor should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "error code should be unsupported_typed_accessor_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 dtype id mismatch");
    }

    try {
        write_f32(storage, std::span<const std::uint64_t>(index), 1.0F);
        throw TestFailure("F32 write accessor should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "write error code should be unsupported_typed_accessor_dtype"
        );
    }
}

void test_rejects_f32_accessor_invalid_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);
    const auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {2};

    try {
        (void)read_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F32 accessor invalid index should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_index_out_of_range,
            "error code should be tensor_index_out_of_range"
        );
        expect(error.expected_value() == 2, "expected F32 accessor dimension size mismatch");
        expect(error.actual_value() == 2, "actual F32 accessor index mismatch");
    }
}

void test_reads_and_writes_f16_scalars_by_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({3}), DType::f16);
    auto storage = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 1> first_index {0};
    const std::array<std::uint64_t, 1> second_index {1};
    const std::array<std::uint64_t, 1> third_index {2};
    write_f16_from_f32(storage, std::span<const std::uint64_t>(first_index), 1.0F);
    write_f16_from_f32(storage, std::span<const std::uint64_t>(second_index), -2.0F);
    write_f16_from_f32(storage, std::span<const std::uint64_t>(third_index), 0.5F);

    expect(read_f16_as_f32(storage, std::span<const std::uint64_t>(first_index)) == 1.0F, "F16 read/write value 0 mismatch");
    expect(read_f16_as_f32(storage, std::span<const std::uint64_t>(second_index)) == -2.0F, "F16 read/write value 1 mismatch");
    expect(read_f16_as_f32(storage, std::span<const std::uint64_t>(third_index)) == 0.5F, "F16 read/write value 2 mismatch");

    expect(storage.bytes()[0] == 0x00, "F16 value 0 low byte mismatch");
    expect(storage.bytes()[1] == 0x3C, "F16 value 0 high byte mismatch");
    expect(storage.bytes()[2] == 0x00, "F16 value 1 low byte mismatch");
    expect(storage.bytes()[3] == 0xC0, "F16 value 1 high byte mismatch");
    expect(storage.bytes()[4] == 0x00, "F16 value 2 low byte mismatch");
    expect(storage.bytes()[5] == 0x38, "F16 value 2 high byte mismatch");
}

void test_reads_f16_scalars_from_loaded_gguf_payload() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    const std::array<std::uint64_t, 1> first_index {0};
    const std::array<std::uint64_t, 1> second_index {1};
    expect(read_f16_as_f32(loaded.storage, std::span<const std::uint64_t>(first_index)) == 1.0F, "loaded GGUF F16 scalar 0 mismatch");
    expect(read_f16_as_f32(loaded.storage, std::span<const std::uint64_t>(second_index)) == 2.0F, "loaded GGUF F16 scalar 1 mismatch");
}

void test_rejects_f16_accessor_dtype_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f32);
    auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {0};

    try {
        (void)read_f16_as_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F16 read accessor should reject F32 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "error code should be unsupported_typed_accessor_dtype"
        );
        expect(error.expected_value() == 1, "expected F16 dtype id mismatch");
        expect(error.actual_value() == 0, "actual F32 dtype id mismatch");
    }

    try {
        write_f16_from_f32(storage, std::span<const std::uint64_t>(index), 1.0F);
        throw TestFailure("F16 write accessor should reject F32 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_typed_accessor_dtype,
            "write error code should be unsupported_typed_accessor_dtype"
        );
    }
}

void test_rejects_f16_accessor_invalid_index() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f16);
    const auto storage = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index {2};

    try {
        (void)read_f16_as_f32(storage, std::span<const std::uint64_t>(index));
        throw TestFailure("F16 accessor invalid index should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_index_out_of_range,
            "error code should be tensor_index_out_of_range"
        );
        expect(error.expected_value() == 2, "expected F16 accessor dimension size mismatch");
        expect(error.actual_value() == 2, "actual F16 accessor index mismatch");
    }
}

void test_adds_f32_tensors_elementwise() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto left = TensorStorage::allocate(descriptor);
    auto right = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};

    write_f32(left, std::span<const std::uint64_t>(index_00), 1.0F);
    write_f32(left, std::span<const std::uint64_t>(index_01), -2.0F);
    write_f32(left, std::span<const std::uint64_t>(index_10), 3.5F);
    write_f32(left, std::span<const std::uint64_t>(index_11), 4.0F);

    write_f32(right, std::span<const std::uint64_t>(index_00), 0.5F);
    write_f32(right, std::span<const std::uint64_t>(index_01), 2.0F);
    write_f32(right, std::span<const std::uint64_t>(index_10), -1.5F);
    write_f32(right, std::span<const std::uint64_t>(index_11), 8.0F);

    const auto output = add_f32(left, right);

    expect(output.descriptor() == descriptor, "F32 add output descriptor mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 1.5F, "F32 add value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 0.0F, "F32 add value 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 2.0F, "F32 add value 2 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 12.0F, "F32 add value 3 mismatch");
    expect(read_f32(left, std::span<const std::uint64_t>(index_00)) == 1.0F, "F32 add should not mutate left input");
    expect(read_f32(right, std::span<const std::uint64_t>(index_11)) == 8.0F, "F32 add should not mutate right input");
}

void test_rejects_f32_add_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)add_f32(left, right);
        throw TestFailure("F32 add should reject F16 operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 operator dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 operator dtype id mismatch");
    }
}

void test_rejects_f32_add_shape_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)add_f32(left, right);
        throw TestFailure("F32 add should reject mismatched shapes");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "error code should be tensor_shape_mismatch");
        expect(error.expected_value() == 3, "expected shape dimension mismatch");
        expect(error.actual_value() == 2, "actual shape dimension mismatch");
    }
}

void test_multiplies_f32_tensors_elementwise() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto left = TensorStorage::allocate(descriptor);
    auto right = TensorStorage::allocate(descriptor);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};

    write_f32(left, std::span<const std::uint64_t>(index_00), 1.0F);
    write_f32(left, std::span<const std::uint64_t>(index_01), -2.0F);
    write_f32(left, std::span<const std::uint64_t>(index_10), 3.5F);
    write_f32(left, std::span<const std::uint64_t>(index_11), 4.0F);

    write_f32(right, std::span<const std::uint64_t>(index_00), 0.5F);
    write_f32(right, std::span<const std::uint64_t>(index_01), 2.0F);
    write_f32(right, std::span<const std::uint64_t>(index_10), -1.5F);
    write_f32(right, std::span<const std::uint64_t>(index_11), 8.0F);

    const auto output = multiply_f32(left, right);

    expect(output.descriptor() == descriptor, "F32 multiply output descriptor mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 0.5F, "F32 multiply value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == -4.0F, "F32 multiply value 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == -5.25F, "F32 multiply value 2 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 32.0F, "F32 multiply value 3 mismatch");
}

void test_rejects_f32_multiply_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)multiply_f32(left, right);
        throw TestFailure("F32 multiply should reject F16 operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 multiply dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 multiply expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 multiply actual dtype mismatch");
    }
}

void test_rejects_f32_multiply_shape_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)multiply_f32(left, right);
        throw TestFailure("F32 multiply should reject mismatched shapes");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 multiply shape error code mismatch");
        expect(error.expected_value() == 3, "F32 multiply expected shape dimension mismatch");
        expect(error.actual_value() == 2, "F32 multiply actual shape dimension mismatch");
    }
}

void test_applies_f32_silu() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    write_f32(input, std::span<const std::uint64_t>(index_0), -1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 2.0F);

    const auto output = silu_f32(input);

    expect(output.descriptor() == input.descriptor(), "F32 SiLU output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 SiLU output should not alias input");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(-1.0 / (1.0 + std::exp(1.0))),
        0.000001F,
        "F32 SiLU value 0 mismatch"
    );
    expect(read_f32(output, std::span<const std::uint64_t>(index_1)) == 0.0F, "F32 SiLU value 1 mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_2)),
        static_cast<float>(2.0 / (1.0 + std::exp(-2.0))),
        0.000001F,
        "F32 SiLU value 2 mismatch"
    );
}

void test_rejects_f32_silu_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)silu_f32(input);
        throw TestFailure("F32 SiLU should reject F16 input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 SiLU dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 SiLU expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 SiLU actual dtype mismatch");
    }
}

void test_fills_f32_tensor_with_scalar() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const auto output = fill_f32(descriptor, -3.25F);

    expect(output.descriptor() == descriptor, "F32 fill output descriptor mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect(read_f32(output, std::span<const std::uint64_t>(index)) == -3.25F, "F32 fill value mismatch");
        }
    }
}

void test_rejects_f32_fill_dtype_mismatch() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2}), DType::f16);

    try {
        (void)fill_f32(descriptor, 1.0F);
        throw TestFailure("F32 fill should reject F16 descriptor");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 fill dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 fill dtype id mismatch");
    }
}

void test_copies_f32_tensor_storage() {
    const auto descriptor = TensorDescriptor::from(Shape::from({3}), DType::f32);
    auto input = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};

    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 2.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 3.0F);

    auto output = copy_f32(input);

    expect(output.descriptor() == descriptor, "F32 copy output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 copy should own distinct storage");
    expect(read_f32(output, std::span<const std::uint64_t>(index_0)) == 1.0F, "F32 copy value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_1)) == 2.0F, "F32 copy value 1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_2)) == 3.0F, "F32 copy value 2 mismatch");

    write_f32(output, std::span<const std::uint64_t>(index_0), 9.0F);
    expect(read_f32(input, std::span<const std::uint64_t>(index_0)) == 1.0F, "F32 copy should not alias input");
    expect(read_f32(output, std::span<const std::uint64_t>(index_0)) == 9.0F, "F32 copy mutation mismatch");
}

void test_rejects_f32_copy_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({32}), DType::q8_0));

    try {
        (void)copy_f32(input);
        throw TestFailure("F32 copy should reject Q8_0 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 copy dtype id mismatch");
        expect(error.actual_value() == 8, "actual Q8_0 copy dtype id mismatch");
    }
}

void test_reduces_f32_tensor_sum_and_sum_squares() {
    const auto descriptor = TensorDescriptor::from(Shape::from({2, 2}), DType::f32);
    auto input = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};

    write_f32(input, std::span<const std::uint64_t>(index_00), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_01), -2.0F);
    write_f32(input, std::span<const std::uint64_t>(index_10), 3.5F);
    write_f32(input, std::span<const std::uint64_t>(index_11), 0.5F);

    expect(sum_f32(input) == 3.0F, "F32 sum mismatch");
    expect(sum_squares_f32(input) == 17.5F, "F32 sum squares mismatch");
}

void test_reduces_loaded_gguf_f32_payload() {
    const auto bytes = fixture("valid-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(sum_f32(loaded.storage) == 3.0F, "loaded GGUF F32 sum mismatch");
    expect(sum_squares_f32(loaded.storage) == 5.0F, "loaded GGUF F32 sum squares mismatch");
}

void test_rejects_f32_reduction_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)sum_f32(input);
        throw TestFailure("F32 sum should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "sum error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 sum dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 sum dtype id mismatch");
    }

    try {
        (void)sum_squares_f32(input);
        throw TestFailure("F32 sum squares should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "sum squares error code should be unsupported_operator_dtype"
        );
    }
}

void test_calculates_f32_mean_square_and_rms() {
    const auto descriptor = TensorDescriptor::from(Shape::from({4}), DType::f32);
    auto input = TensorStorage::allocate(descriptor);
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    const std::array<std::uint64_t, 1> index_3 {3};

    write_f32(input, std::span<const std::uint64_t>(index_0), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 4.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_3), 0.0F);

    expect(mean_square_f32(input) == 6.25F, "F32 mean square mismatch");
    expect(rms_f32(input) == 2.5F, "F32 RMS mismatch");
}

void test_rejects_f32_rms_helpers_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)mean_square_f32(input);
        throw TestFailure("F32 mean square should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "mean square error code should be unsupported_operator_dtype"
        );
        expect(error.expected_value() == 0, "expected F32 mean square dtype id mismatch");
        expect(error.actual_value() == 1, "actual F16 mean square dtype id mismatch");
    }

    try {
        (void)rms_f32(input);
        throw TestFailure("F32 RMS should reject F16 storage");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "RMS error code should be unsupported_operator_dtype"
        );
    }
}

void test_applies_f32_rms_norm_to_vector() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    write_f32(input, std::span<const std::uint64_t>(index_0), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 4.0F);
    write_f32(weight, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(weight, std::span<const std::uint64_t>(index_1), 2.0F);

    const auto output = rms_norm_f32(input, weight, 3.5F);

    expect(output.descriptor() == input.descriptor(), "F32 RMSNorm vector output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 RMSNorm vector output should not alias input");
    expect(read_f32(output, std::span<const std::uint64_t>(index_0)) == 0.75F, "F32 RMSNorm vector value 0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_1)) == 2.0F, "F32 RMSNorm vector value 1 mismatch");
}

void test_applies_f32_rms_norm_to_batched_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 4.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), -3.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 4.0F);

    const std::array<std::uint64_t, 1> weight0 {0};
    const std::array<std::uint64_t, 1> weight1 {1};
    write_f32(weight, std::span<const std::uint64_t>(weight0), 1.0F);
    write_f32(weight, std::span<const std::uint64_t>(weight1), 2.0F);

    const auto output = rms_norm_f32(input, weight, 3.5F);

    expect(output.descriptor() == input.descriptor(), "F32 RMSNorm batched output descriptor mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row0_col0)) == 0.75F, "F32 RMSNorm batch value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row0_col1)) == 2.0F, "F32 RMSNorm batch value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row1_col0)) == -0.75F, "F32 RMSNorm batch value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(row1_col1)) == 2.0F, "F32 RMSNorm batch value 1,1 mismatch");
}

void test_rejects_f32_rms_norm_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f16));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject F16 weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 RMSNorm dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 RMSNorm expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 RMSNorm actual dtype mismatch");
    }
}

void test_rejects_f32_rms_norm_input_rank_above_two() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2, 2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject rank-three input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 RMSNorm input rank error mismatch");
        expect(error.expected_value() == 2, "F32 RMSNorm expected input rank mismatch");
        expect(error.actual_value() == 3, "F32 RMSNorm actual input rank mismatch");
    }
}

void test_rejects_f32_rms_norm_weight_rank_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject rank-two weight");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 RMSNorm weight rank error mismatch");
        expect(error.expected_value() == 1, "F32 RMSNorm expected weight rank mismatch");
        expect(error.actual_value() == 2, "F32 RMSNorm actual weight rank mismatch");
    }
}

void test_rejects_f32_rms_norm_hidden_dimension_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.00001F);
        throw TestFailure("F32 RMSNorm should reject hidden dimension mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 RMSNorm hidden shape error mismatch");
        expect(error.expected_value() == 3, "F32 RMSNorm expected hidden dimension mismatch");
        expect(error.actual_value() == 2, "F32 RMSNorm actual hidden dimension mismatch");
    }
}

void test_rejects_f32_rms_norm_nonpositive_epsilon() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto weight = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));

    try {
        (void)rms_norm_f32(input, weight, 0.0F);
        throw TestFailure("F32 RMSNorm should reject nonpositive epsilon");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RMSNorm epsilon must be positive",
            "F32 RMSNorm epsilon error mismatch"
        );
    }

    try {
        (void)rms_norm_f32(input, weight, std::numeric_limits<float>::quiet_NaN());
        throw TestFailure("F32 RMSNorm should reject NaN epsilon");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RMSNorm epsilon must be positive",
            "F32 RMSNorm NaN epsilon error mismatch"
        );
    }
}

void test_applies_f32_rope_to_vector() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({5}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    const std::array<std::uint64_t, 1> index_3 {3};
    const std::array<std::uint64_t, 1> index_4 {4};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_3), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_4), 9.0F);

    const auto output = rope_f32(input, 1, 4, 1.0F);
    const auto cosine = static_cast<float>(std::cos(1.0));
    const auto sine = static_cast<float>(std::sin(1.0));

    expect(output.descriptor() == input.descriptor(), "F32 RoPE vector output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 RoPE vector output should not alias input");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_0)), cosine, 0.000001F, "F32 RoPE vector value 0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_1)), sine, 0.000001F, "F32 RoPE vector value 1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_2)), -sine, 0.000001F, "F32 RoPE vector value 2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_3)), cosine, 0.000001F, "F32 RoPE vector value 3 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_4)) == 9.0F, "F32 RoPE trailing value mismatch");
}

void test_applies_f32_rope_to_batched_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 4}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_col2 {0, 2};
    const std::array<std::uint64_t, 2> row0_col3 {0, 3};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_col2 {1, 2};
    const std::array<std::uint64_t, 2> row1_col3 {1, 3};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col3), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col2), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col3), 0.0F);

    const auto output = rope_f32(input, 1, 4, 1.0F);
    const auto cosine = static_cast<float>(std::cos(1.0));
    const auto sine = static_cast<float>(std::sin(1.0));

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col0)), cosine, 0.000001F, "F32 RoPE batch value 0,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col1)), sine, 0.000001F, "F32 RoPE batch value 0,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col2)), -sine, 0.000001F, "F32 RoPE batch value 0,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col3)), cosine, 0.000001F, "F32 RoPE batch value 0,3 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col0)), -sine, 0.000001F, "F32 RoPE batch value 1,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col1)), cosine, 0.000001F, "F32 RoPE batch value 1,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col2)), cosine, 0.000001F, "F32 RoPE batch value 1,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col3)), sine, 0.000001F, "F32 RoPE batch value 1,3 mismatch");
}

void test_applies_f32_rope_to_batched_rows_with_position_start() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 4}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_col2 {0, 2};
    const std::array<std::uint64_t, 2> row0_col3 {0, 3};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_col2 {1, 2};
    const std::array<std::uint64_t, 2> row1_col3 {1, 3};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col3), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col2), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col3), 0.0F);

    const auto output = rope_f32_with_position_start(input, 1, 4, 1.0F);
    const auto row0_cosine = static_cast<float>(std::cos(1.0));
    const auto row0_sine = static_cast<float>(std::sin(1.0));
    const auto row1_cosine = static_cast<float>(std::cos(2.0));
    const auto row1_sine = static_cast<float>(std::sin(2.0));

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col0)), row0_cosine, 0.000001F, "F32 rowwise RoPE batch value 0,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col1)), row0_sine, 0.000001F, "F32 rowwise RoPE batch value 0,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col2)), -row0_sine, 0.000001F, "F32 rowwise RoPE batch value 0,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col3)), row0_cosine, 0.000001F, "F32 rowwise RoPE batch value 0,3 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col0)), -row1_sine, 0.000001F, "F32 rowwise RoPE batch value 1,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col1)), row1_cosine, 0.000001F, "F32 rowwise RoPE batch value 1,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col2)), row1_cosine, 0.000001F, "F32 rowwise RoPE batch value 1,2 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row1_col3)), row1_sine, 0.000001F, "F32 rowwise RoPE batch value 1,3 mismatch");
}

void test_applies_f32_rope_frequency_base_by_pair() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    const std::array<std::uint64_t, 1> index_3 {3};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_3), 0.0F);

    const auto output = rope_f32(input, 1, 4, 10000.0F);
    const auto first_angle = 1.0;
    const auto second_angle = 0.01;

    expect_close(read_f32(output, std::span<const std::uint64_t>(index_0)), static_cast<float>(std::cos(first_angle)), 0.000001F, "F32 RoPE first pair cosine mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_1)), static_cast<float>(std::sin(first_angle)), 0.000001F, "F32 RoPE first pair sine mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_2)), static_cast<float>(std::cos(second_angle)), 0.000001F, "F32 RoPE second pair cosine mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(index_3)), static_cast<float>(std::sin(second_angle)), 0.000001F, "F32 RoPE second pair sine mismatch");
}

void test_rejects_f32_rope_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f16));

    try {
        (void)rope_f32(input, 1, 4, 10000.0F);
        throw TestFailure("F32 RoPE should reject F16 input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 RoPE dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 RoPE expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 RoPE actual dtype mismatch");
    }
}

void test_rejects_f32_rope_input_rank_above_two() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2, 2}), DType::f32));

    try {
        (void)rope_f32(input, 1, 2, 10000.0F);
        throw TestFailure("F32 RoPE should reject rank-three input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 RoPE rank error code mismatch");
        expect(error.expected_value() == 2, "F32 RoPE expected rank mismatch");
        expect(error.actual_value() == 3, "F32 RoPE actual rank mismatch");
    }
}

void test_rejects_f32_rope_invalid_dimension_count() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));

    try {
        (void)rope_f32(input, 1, 3, 10000.0F);
        throw TestFailure("F32 RoPE should reject odd dimension count");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_dimension, "F32 RoPE odd dimension error mismatch");
        expect(error.expected_value() == 2, "F32 RoPE expected dimension multiple mismatch");
        expect(error.actual_value() == 3, "F32 RoPE actual odd dimension mismatch");
    }
}

void test_rejects_f32_rope_dimension_count_above_hidden() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));

    try {
        (void)rope_f32(input, 1, 6, 10000.0F);
        throw TestFailure("F32 RoPE should reject dimension count above hidden size");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 RoPE dimension shape error mismatch");
        expect(error.expected_value() == 4, "F32 RoPE expected hidden dimension mismatch");
        expect(error.actual_value() == 6, "F32 RoPE actual rope dimension mismatch");
    }
}

void test_rejects_f32_rope_nonpositive_frequency_base() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({4}), DType::f32));

    try {
        (void)rope_f32(input, 1, 4, 0.0F);
        throw TestFailure("F32 RoPE should reject nonpositive frequency base");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RoPE frequency base must be positive",
            "F32 RoPE frequency base error mismatch"
        );
    }

    try {
        (void)rope_f32(input, 1, 4, std::numeric_limits<float>::quiet_NaN());
        throw TestFailure("F32 RoPE should reject NaN frequency base");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RoPE frequency base must be positive",
            "F32 RoPE NaN frequency base error mismatch"
        );
    }
}

void test_rejects_f32_rope_row_position_overflow() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)rope_f32_with_position_start(input, std::numeric_limits<std::uint64_t>::max(), 2, 1.0F);
        throw TestFailure("F32 rowwise RoPE should reject overflowing row positions");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "RoPE row positions overflow",
            "F32 rowwise RoPE overflow error mismatch"
        );
    }
}

void test_applies_f32_softmax_to_vector() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 2.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 3.0F);

    const auto output = softmax_f32(input);
    const auto denominator = std::exp(-2.0) + std::exp(-1.0) + 1.0;

    expect(output.descriptor() == input.descriptor(), "F32 softmax vector output descriptor mismatch");
    expect(output.bytes().data() != input.bytes().data(), "F32 softmax vector output should not alias input");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(std::exp(-2.0) / denominator),
        0.000001F,
        "F32 softmax vector value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(std::exp(-1.0) / denominator),
        0.000001F,
        "F32 softmax vector value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_2)),
        static_cast<float>(1.0 / denominator),
        0.000001F,
        "F32 softmax vector value 2 mismatch"
    );
}

void test_applies_f32_softmax_to_batched_rows() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row0_col2 {0, 2};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> row1_col2 {1, 2};
    write_f32(input, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col1), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row0_col2), 0.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col0), 1.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col1), 3.0F);
    write_f32(input, std::span<const std::uint64_t>(row1_col2), 3.0F);

    const auto output = softmax_f32(input);
    const auto row1_denominator = std::exp(-2.0) + 2.0;

    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col0)), 1.0F / 3.0F, 0.000001F, "F32 softmax row 0,0 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col1)), 1.0F / 3.0F, 0.000001F, "F32 softmax row 0,1 mismatch");
    expect_close(read_f32(output, std::span<const std::uint64_t>(row0_col2)), 1.0F / 3.0F, 0.000001F, "F32 softmax row 0,2 mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        static_cast<float>(std::exp(-2.0) / row1_denominator),
        0.000001F,
        "F32 softmax row 1,0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        static_cast<float>(1.0 / row1_denominator),
        0.000001F,
        "F32 softmax row 1,1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col2)),
        static_cast<float>(1.0 / row1_denominator),
        0.000001F,
        "F32 softmax row 1,2 mismatch"
    );
}

void test_applies_f32_softmax_to_large_values() {
    auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f32));
    const std::array<std::uint64_t, 1> index_0 {0};
    const std::array<std::uint64_t, 1> index_1 {1};
    const std::array<std::uint64_t, 1> index_2 {2};
    write_f32(input, std::span<const std::uint64_t>(index_0), 1000.0F);
    write_f32(input, std::span<const std::uint64_t>(index_1), 1001.0F);
    write_f32(input, std::span<const std::uint64_t>(index_2), 1002.0F);

    const auto output = softmax_f32(input);
    const auto denominator = std::exp(-2.0) + std::exp(-1.0) + 1.0;

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(std::exp(-2.0) / denominator),
        0.000001F,
        "F32 softmax stable value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(std::exp(-1.0) / denominator),
        0.000001F,
        "F32 softmax stable value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_2)),
        static_cast<float>(1.0 / denominator),
        0.000001F,
        "F32 softmax stable value 2 mismatch"
    );
}

void test_rejects_f32_softmax_dtype_mismatch() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3}), DType::f16));

    try {
        (void)softmax_f32(input);
        throw TestFailure("F32 softmax should reject F16 input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 softmax dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 softmax expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 softmax actual dtype mismatch");
    }
}

void test_rejects_f32_softmax_input_rank_above_two() {
    const auto input = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2, 2}), DType::f32));

    try {
        (void)softmax_f32(input);
        throw TestFailure("F32 softmax should reject rank-three input");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 softmax rank error code mismatch");
        expect(error.expected_value() == 2, "F32 softmax expected rank mismatch");
        expect(error.actual_value() == 3, "F32 softmax actual rank mismatch");
    }
}

void test_applies_f32_scaled_dot_product_attention() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2}), DType::f32));

    const float query_values[2][2] {
        {1.0F, 0.0F},
        {0.0F, 1.0F},
    };
    const float key_values[3][2] {
        {1.0F, 0.0F},
        {0.0F, 1.0F},
        {1.0F, 1.0F},
    };
    const float value_values[3][2] {
        {10.0F, 0.0F},
        {0.0F, 20.0F},
        {10.0F, 20.0F},
    };

    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(query, std::span<const std::uint64_t>(index), query_values[row][column]);
        }
    }

    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(key, std::span<const std::uint64_t>(index), key_values[row][column]);
            write_f32(value, std::span<const std::uint64_t>(index), value_values[row][column]);
        }
    }

    const auto output = scaled_dot_product_attention_f32(query, key, value, 0.5F);
    const auto denominator = 2.0 + std::exp(-0.5);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 attention output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "F32 attention output dtype mismatch");
    expect(output.bytes().data() != query.bytes().data(), "F32 attention output should not alias query");
    expect(output.bytes().data() != key.bytes().data(), "F32 attention output should not alias key");
    expect(output.bytes().data() != value.bytes().data(), "F32 attention output should not alias value");

    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        static_cast<float>(20.0 / denominator),
        0.000001F,
        "F32 attention value 0,0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col1)),
        static_cast<float>(20.0 * (1.0 + std::exp(-0.5)) / denominator),
        0.000001F,
        "F32 attention value 0,1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        static_cast<float>(10.0 * (1.0 + std::exp(-0.5)) / denominator),
        0.000001F,
        "F32 attention value 1,0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        static_cast<float>(40.0 / denominator),
        0.000001F,
        "F32 attention value 1,1 mismatch"
    );
}

void test_applies_f32_scaled_dot_product_attention_with_distinct_value_width() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const std::array<std::uint64_t, 2> query_0_0 {0, 0};
    const std::array<std::uint64_t, 2> query_0_1 {0, 1};
    const std::array<std::uint64_t, 2> key_0_0 {0, 0};
    const std::array<std::uint64_t, 2> key_0_1 {0, 1};
    const std::array<std::uint64_t, 2> key_1_0 {1, 0};
    const std::array<std::uint64_t, 2> key_1_1 {1, 1};
    const std::array<std::uint64_t, 2> value_0_0 {0, 0};
    const std::array<std::uint64_t, 2> value_1_0 {1, 0};
    write_f32(query, std::span<const std::uint64_t>(query_0_0), 1.0F);
    write_f32(query, std::span<const std::uint64_t>(query_0_1), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(key_0_0), 1.0F);
    write_f32(key, std::span<const std::uint64_t>(key_0_1), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(key_1_0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(key_1_1), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(value_0_0), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(value_1_0), 10.0F);

    const auto output = scaled_dot_product_attention_f32(query, key, value, 2.0F);
    const auto denominator = 1.0 + std::exp(-2.0);
    const std::array<std::uint64_t, 2> output_0_0 {0, 0};

    expect(output.descriptor().shape == Shape::from({1, 1}), "F32 attention distinct value width shape mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(output_0_0)),
        static_cast<float>(10.0 * std::exp(-2.0) / denominator),
        0.000001F,
        "F32 attention explicit scale value mismatch"
    );
}

void test_applies_f32_causal_scaled_dot_product_attention() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    write_f32(query, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(query, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(row0_col0), 10.0F);
    write_f32(value, std::span<const std::uint64_t>(row1_col0), 100.0F);

    const auto output = causal_scaled_dot_product_attention_f32(query, key, value, 1.0F, 0);

    expect(output.descriptor().shape == Shape::from({2, 1}), "F32 causal attention output shape mismatch");
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        10.0F,
        0.000001F,
        "F32 causal attention row 0 should not see future value"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        55.0F,
        0.000001F,
        "F32 causal attention row 1 value mismatch"
    );
}

void test_applies_f32_causal_attention_with_query_position_offset() {
    auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 1}), DType::f32));
    auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 1}), DType::f32));
    auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 1}), DType::f32));
    const std::array<std::uint64_t, 2> query_index {0, 0};
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row2_col0 {2, 0};
    write_f32(query, std::span<const std::uint64_t>(query_index), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row0_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row1_col0), 0.0F);
    write_f32(key, std::span<const std::uint64_t>(row2_col0), 0.0F);
    write_f32(value, std::span<const std::uint64_t>(row0_col0), 10.0F);
    write_f32(value, std::span<const std::uint64_t>(row1_col0), 20.0F);
    write_f32(value, std::span<const std::uint64_t>(row2_col0), 300.0F);

    const auto output = causal_scaled_dot_product_attention_f32(query, key, value, 1.0F, 1);

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(query_index)),
        15.0F,
        0.000001F,
        "F32 causal attention offset should mask keys after query position"
    );
}

void test_applies_f32_causal_attention_with_column_views() {
    const std::array<float, 4> query_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 4> key_reference_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 4> value_reference_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
    };
    const std::array<float, 12> key_cache_values {
        99.0F, 1.0F, 0.0F, -99.0F,
        88.0F, 0.0F, 1.0F, -88.0F,
        77.0F, 5.0F, 5.0F, -77.0F,
    };
    const std::array<float, 12> value_cache_values {
        -1.0F, -2.0F, 10.0F, 20.0F,
        -3.0F, -4.0F, 30.0F, 40.0F,
        -5.0F, -6.0F, 50.0F, 60.0F,
    };
    const auto query = tensor_f32_matrix(2, 2, std::span<const float>(query_values));
    const auto key_reference = tensor_f32_matrix(2, 2, std::span<const float>(key_reference_values));
    const auto value_reference = tensor_f32_matrix(2, 2, std::span<const float>(value_reference_values));
    const auto key_cache = tensor_f32_matrix(3, 4, std::span<const float>(key_cache_values));
    const auto value_cache = tensor_f32_matrix(3, 4, std::span<const float>(value_cache_values));

    const auto reference = causal_scaled_dot_product_attention_f32(
        query,
        key_reference,
        value_reference,
        1.0F,
        0
    );
    const auto output = causal_scaled_dot_product_attention_f32(
        query,
        F32MatrixColumnView {
            .storage = key_cache,
            .row_count = 2,
            .first_column = 1,
            .column_count = 2,
        },
        F32MatrixColumnView {
            .storage = value_cache,
            .row_count = 2,
            .first_column = 2,
            .column_count = 2,
        },
        1.0F,
        0
    );

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 causal attention view output shape mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(output, std::span<const std::uint64_t>(index)),
                read_f32(reference, std::span<const std::uint64_t>(index)),
                0.000001F,
                "F32 causal attention view parity mismatch"
            );
        }
    }
}

void test_applies_f32_causal_attention_with_rope_query_column_view() {
    const auto row0_sine = static_cast<float>(std::sin(1.0));
    const auto row0_cosine = static_cast<float>(std::cos(1.0));
    const auto row1_sine = static_cast<float>(std::sin(2.0));
    const auto row1_cosine = static_cast<float>(std::cos(2.0));
    const std::array<float, 8> query_source_values {
        99.0F, 1.0F, 0.0F, -99.0F,
        88.0F, 0.0F, 1.0F, -88.0F,
    };
    const std::array<float, 4> query_reference_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 6> key_reference_values {
        1.0F, 0.0F,
        row0_cosine, row0_sine,
        -row1_sine, row1_cosine,
    };
    const std::array<float, 12> key_cache_values {
        99.0F, 1.0F, 0.0F, -99.0F,
        88.0F, row0_cosine, row0_sine, -88.0F,
        77.0F, -row1_sine, row1_cosine, -77.0F,
    };
    const std::array<float, 6> value_reference_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
        50.0F, 60.0F,
    };
    const std::array<float, 12> value_cache_values {
        -1.0F, 10.0F, 20.0F, -2.0F,
        -3.0F, 30.0F, 40.0F, -4.0F,
        -5.0F, 50.0F, 60.0F, -6.0F,
    };
    const auto query_source = tensor_f32_matrix(2, 4, std::span<const float>(query_source_values));
    const auto query_reference = tensor_f32_matrix(2, 2, std::span<const float>(query_reference_values));
    const auto rotated_query = rope_f32_with_position_start(query_reference, 1, 2, 1.0F);
    const auto key_reference = tensor_f32_matrix(3, 2, std::span<const float>(key_reference_values));
    const auto key_cache = tensor_f32_matrix(3, 4, std::span<const float>(key_cache_values));
    const auto value_reference = tensor_f32_matrix(3, 2, std::span<const float>(value_reference_values));
    const auto value_cache = tensor_f32_matrix(3, 4, std::span<const float>(value_cache_values));

    const auto reference = causal_scaled_dot_product_attention_f32(
        rotated_query,
        key_reference,
        value_reference,
        1.0F,
        1
    );
    const auto output = causal_scaled_dot_product_attention_f32(
        F32RoPEMatrixColumnView {
            .columns = F32MatrixColumnView {
                .storage = query_source,
                .row_count = 2,
                .first_column = 1,
                .column_count = 2,
            },
            .rope_dimension_count = 2,
            .frequency_base = 1.0F,
        },
        F32MatrixColumnView {
            .storage = key_cache,
            .row_count = 3,
            .first_column = 1,
            .column_count = 2,
        },
        F32MatrixColumnView {
            .storage = value_cache,
            .row_count = 3,
            .first_column = 1,
            .column_count = 2,
        },
        1.0F,
        1
    );

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 causal attention RoPE query view shape mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(output, std::span<const std::uint64_t>(index)),
                read_f32(reference, std::span<const std::uint64_t>(index)),
                0.000001F,
                "F32 causal attention RoPE query view parity mismatch"
            );
        }
    }
}

void test_rejects_f32_causal_attention_column_view_row_mismatch() {
    const std::array<float, 2> query_values {1.0F, 0.0F};
    const std::array<float, 6> key_cache_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
        1.0F, 1.0F,
    };
    const std::array<float, 6> value_cache_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
        50.0F, 60.0F,
    };
    const auto query = tensor_f32_matrix(1, 2, std::span<const float>(query_values));
    const auto key_cache = tensor_f32_matrix(3, 2, std::span<const float>(key_cache_values));
    const auto value_cache = tensor_f32_matrix(3, 2, std::span<const float>(value_cache_values));

    try {
        (void)causal_scaled_dot_product_attention_f32(
            query,
            F32MatrixColumnView {
                .storage = key_cache,
                .row_count = 2,
                .first_column = 0,
                .column_count = 2,
            },
            F32MatrixColumnView {
                .storage = value_cache,
                .row_count = 1,
                .first_column = 0,
                .column_count = 2,
            },
            1.0F,
            0
        );
        throw TestFailure("F32 causal attention should reject mismatched column view rows");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 causal attention view row error mismatch");
        expect(error.expected_value() == 2, "F32 causal attention view expected row mismatch");
        expect(error.actual_value() == 1, "F32 causal attention view actual row mismatch");
    }
}

void test_rejects_f32_causal_attention_rope_query_view_invalid_dimension_count() {
    const std::array<float, 2> query_values {1.0F, 0.0F};
    const std::array<float, 2> key_values {1.0F, 0.0F};
    const std::array<float, 1> value_values {10.0F};
    const auto query = tensor_f32_matrix(1, 2, std::span<const float>(query_values));
    const auto key = tensor_f32_matrix(1, 2, std::span<const float>(key_values));
    const auto value = tensor_f32_matrix(1, 1, std::span<const float>(value_values));

    try {
        (void)causal_scaled_dot_product_attention_f32(
            F32RoPEMatrixColumnView {
                .columns = F32MatrixColumnView {
                    .storage = query,
                    .row_count = 1,
                    .first_column = 0,
                    .column_count = 2,
                },
                .rope_dimension_count = 1,
                .frequency_base = 1.0F,
            },
            F32MatrixColumnView {
                .storage = key,
                .row_count = 1,
                .first_column = 0,
                .column_count = 2,
            },
            F32MatrixColumnView {
                .storage = value,
                .row_count = 1,
                .first_column = 0,
                .column_count = 1,
            },
            1.0F,
            0
        );
        throw TestFailure("F32 causal attention should reject invalid RoPE query dimension count");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_shape_dimension,
            "F32 causal attention RoPE query dimension error mismatch"
        );
        expect(error.expected_value() == 2, "F32 causal attention expected RoPE dimension mismatch");
        expect(error.actual_value() == 1, "F32 causal attention actual RoPE dimension mismatch");
    }
}

void test_rejects_f32_causal_attention_position_window_overflow() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 1}), DType::f32));

    try {
        (void)causal_scaled_dot_product_attention_f32(query, key, value, 1.0F, 1);
        throw TestFailure("F32 causal attention should reject query positions beyond key rows");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 causal attention window error mismatch");
        expect(error.expected_value() == 2, "F32 causal attention expected key rows mismatch");
        expect(error.actual_value() == 3, "F32 causal attention actual position end mismatch");
    }
}

void test_rejects_f32_attention_dtype_mismatch() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f16));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject F16 key");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "F32 attention dtype error code mismatch");
        expect(error.expected_value() == 0, "F32 attention expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 attention actual dtype mismatch");
    }
}

void test_rejects_f32_attention_non_matrix_rank() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject rank-one query");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 attention rank error code mismatch");
        expect(error.expected_value() == 2, "F32 attention expected rank mismatch");
        expect(error.actual_value() == 1, "F32 attention actual rank mismatch");
    }
}

void test_rejects_f32_attention_query_key_hidden_mismatch() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 3}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject query/key hidden mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 attention hidden error code mismatch");
        expect(error.expected_value() == 2, "F32 attention expected hidden mismatch");
        expect(error.actual_value() == 3, "F32 attention actual hidden mismatch");
    }
}

void test_rejects_f32_attention_key_value_row_mismatch() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({3, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 1.0F);
        throw TestFailure("F32 attention should reject key/value row mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 attention row error code mismatch");
        expect(error.expected_value() == 2, "F32 attention expected key row mismatch");
        expect(error.actual_value() == 3, "F32 attention actual value row mismatch");
    }
}

void test_rejects_f32_attention_nonpositive_scale() {
    const auto query = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto key = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));
    const auto value = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 2}), DType::f32));

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, 0.0F);
        throw TestFailure("F32 attention should reject nonpositive scale");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "attention scale must be positive and finite",
            "F32 attention scale error mismatch"
        );
    }

    try {
        (void)scaled_dot_product_attention_f32(query, key, value, std::numeric_limits<float>::quiet_NaN());
        throw TestFailure("F32 attention should reject NaN scale");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "attention scale must be positive and finite",
            "F32 attention NaN scale error mismatch"
        );
    }
}

void test_multiplies_f32_matrices() {
    const auto left_descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const auto right_descriptor = TensorDescriptor::from(Shape::from({3, 2}), DType::f32);
    auto left = TensorStorage::allocate(left_descriptor);
    auto right = TensorStorage::allocate(right_descriptor);

    const float left_values[2][3] {
        {1.0F, 2.0F, 3.0F},
        {4.0F, 5.0F, 6.0F},
    };
    const float right_values[3][2] {
        {7.0F, 8.0F},
        {9.0F, 10.0F},
        {11.0F, 12.0F},
    };

    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(left, std::span<const std::uint64_t>(index), left_values[row][column]);
        }
    }

    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(right, std::span<const std::uint64_t>(index), right_values[row][column]);
        }
    }

    const auto output = matmul_f32(left, right);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32 matmul output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "F32 matmul output dtype mismatch");
    expect(output.bytes().data() != left.bytes().data(), "F32 matmul output should not alias left");
    expect(output.bytes().data() != right.bytes().data(), "F32 matmul output should not alias right");

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 58.0F, "F32 matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 64.0F, "F32 matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 139.0F, "F32 matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 154.0F, "F32 matmul value 1,1 mismatch");
}

void test_rejects_f32_matmul_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f16));

    try {
        (void)matmul_f32(left, right);
        throw TestFailure("F32 matmul should reject F16 right operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32 matmul dtype error code mismatch"
        );
        expect(error.expected_value() == 0, "F32 matmul expected dtype mismatch");
        expect(error.actual_value() == 1, "F32 matmul actual dtype mismatch");
    }
}

void test_rejects_f32_matmul_non_matrix_rank() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)matmul_f32(left, right);
        throw TestFailure("F32 matmul should reject rank-one operands");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "F32 matmul rank error code mismatch");
        expect(error.expected_value() == 2, "F32 matmul expected rank mismatch");
        expect(error.actual_value() == 1, "F32 matmul actual rank mismatch");
    }
}

void test_rejects_f32_matmul_inner_dimension_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 3}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)matmul_f32(left, right);
        throw TestFailure("F32 matmul should reject mismatched inner dimensions");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "F32 matmul shape error code mismatch");
        expect(error.expected_value() == 3, "F32 matmul expected inner dimension mismatch");
        expect(error.actual_value() == 2, "F32 matmul actual inner dimension mismatch");
    }
}

void test_multiplies_f32_by_f16_matrices_into_f32() {
    const auto left_descriptor = TensorDescriptor::from(Shape::from({2, 3}), DType::f32);
    const auto right_descriptor = TensorDescriptor::from(Shape::from({3, 2}), DType::f16);
    auto left = TensorStorage::allocate(left_descriptor);
    auto right = TensorStorage::allocate(right_descriptor);

    const float left_values[2][3] {
        {1.0F, 2.0F, 3.0F},
        {4.0F, 5.0F, 6.0F},
    };
    const float right_values[3][2] {
        {7.0F, 8.0F},
        {9.0F, 10.0F},
        {11.0F, 12.0F},
    };

    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f32(left, std::span<const std::uint64_t>(index), left_values[row][column]);
        }
    }

    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 2; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            write_f16_from_f32(right, std::span<const std::uint64_t>(index), right_values[row][column]);
        }
    }

    const auto output = matmul_f32_f16(left, right);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32xF16 matmul output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "F32xF16 matmul output dtype mismatch");

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 58.0F, "F32xF16 matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 64.0F, "F32xF16 matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 139.0F, "F32xF16 matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 154.0F, "F32xF16 matmul value 1,1 mismatch");
}

void test_multiplies_f32_by_f16_gguf_weight_matrix() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_matrix(3, 2, std::span<const float>(right_values));

    const auto output = matmul_f32_f16_gguf_weight(left, right);

    expect(output.descriptor().shape == Shape::from({2, 2}), "F32xF16 GGUF weight matmul output shape mismatch");
    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 6.0F, "F32xF16 GGUF matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 60.0F, "F32xF16 GGUF matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 5.0F, "F32xF16 GGUF matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 50.0F, "F32xF16 GGUF matmul value 1,1 mismatch");
}

void test_multiplies_f32_by_f16_gguf_weight_matrix_on_cpu_backend() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_matrix(3, 2, std::span<const float>(right_values));

    const auto output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 6.0F, "CPU backend GGUF matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 60.0F, "CPU backend GGUF matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 5.0F, "CPU backend GGUF matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 50.0F, "CPU backend GGUF matmul value 1,1 mismatch");
}

void test_copies_tensor_to_and_from_cpu_backend_buffer() {
    const std::array<float, 4> values {
        1.25F, -2.5F,
        3.75F, -4.5F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    const auto buffer = copy_tensor_to_backend(storage, BackendKind::cpu);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::cpu, "CPU backend buffer kind mismatch");
    expect(buffer.descriptor() == storage.descriptor(), "CPU backend buffer descriptor mismatch");
    expect(roundtrip.descriptor() == storage.descriptor(), "CPU backend buffer roundtrip descriptor mismatch");
    expect(roundtrip.bytes().data() != storage.bytes().data(), "CPU backend buffer should not alias original storage");
    expect(
        std::equal(storage.bytes().begin(), storage.bytes().end(), roundtrip.bytes().begin()),
        "CPU backend buffer roundtrip bytes mismatch"
    );
}

void test_multiplies_f32_by_cpu_backend_f16_gguf_weight_buffer() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_matrix(3, 2, std::span<const float>(right_values));
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);

    const auto output = matmul_f32_f16_gguf_weight(left, right_buffer);

    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 6.0F, "CPU buffered GGUF matmul value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 60.0F, "CPU buffered GGUF matmul value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 5.0F, "CPU buffered GGUF matmul value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 50.0F, "CPU buffered GGUF matmul value 1,1 mismatch");
}

void test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_cpu_backend_buffer() {
    const std::array<float, 6> left_values {
        1.0F, 1.0F, 1.0F,
        2.0F, 0.0F, 1.0F,
    };
    const std::array<float, 6> right_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(3, 2, std::span<const float>(right_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);

    const auto output_buffer = matmul_f32_f16_gguf_weight_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU backend-buffer GGUF matmul output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer GGUF matmul output");
}

void test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_cpu_backend_buffers() {
    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 6> first_values {
        0.25F, -0.5F,
        1.0F, 0.75F,
        -1.25F, 0.5F,
    };
    const std::array<float, 12> second_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 3> third_values {
        1.5F,
        -0.25F,
        0.375F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto first_right = tensor_f16_gguf_matrix(3, 2, std::span<const float>(first_values));
    const auto second_right = tensor_f16_gguf_matrix(3, 4, std::span<const float>(second_values));
    const auto third_right = tensor_f16_gguf_matrix(3, 1, std::span<const float>(third_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto first_buffer = copy_tensor_to_backend(first_right, BackendKind::cpu);
    const auto second_buffer = copy_tensor_to_backend(second_right, BackendKind::cpu);
    const auto third_buffer = copy_tensor_to_backend(third_right, BackendKind::cpu);

    const auto outputs = matmul_f32_f16_gguf_weight_triple_to_backend(
        left_buffer,
        first_buffer,
        second_buffer,
        third_buffer
    );
    const auto first_output = copy_tensor_from_backend(outputs.first);
    const auto second_output = copy_tensor_from_backend(outputs.second);
    const auto third_output = copy_tensor_from_backend(outputs.third);

    expect(outputs.first.backend() == BackendKind::cpu, "CPU triple matmul first output backend mismatch");
    expect(outputs.second.backend() == BackendKind::cpu, "CPU triple matmul second output backend mismatch");
    expect(outputs.third.backend() == BackendKind::cpu, "CPU triple matmul third output backend mismatch");
    expect_f32_matrix_close(
        first_output,
        matmul_f32_f16_gguf_weight(left, first_right, BackendKind::cpu),
        0.0F,
        "CPU triple matmul first output"
    );
    expect_f32_matrix_close(
        second_output,
        matmul_f32_f16_gguf_weight(left, second_right, BackendKind::cpu),
        0.0F,
        "CPU triple matmul second output"
    );
    expect_f32_matrix_close(
        third_output,
        matmul_f32_f16_gguf_weight(left, third_right, BackendKind::cpu),
        0.0F,
        "CPU triple matmul third output"
    );
}

void test_applies_silu_multiply_to_cpu_backend_buffers() {
    const std::array<float, 6> gate_values {
        -2.0F, -1.0F, 0.0F,
        0.5F, 1.0F, 2.0F,
    };
    const std::array<float, 6> up_values {
        0.25F, -0.5F, 1.0F,
        1.5F, -2.0F, 0.75F,
    };
    const auto gate = tensor_f32_matrix(2, 3, std::span<const float>(gate_values));
    const auto up = tensor_f32_matrix(2, 3, std::span<const float>(up_values));
    const auto gate_buffer = copy_tensor_to_backend(gate, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up, BackendKind::cpu);

    const auto output_buffer = silu_multiply_f32_to_backend(gate_buffer, up_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = multiply_f32(silu_f32(gate), up);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU SiLU multiply output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer SiLU multiply output");
}

void test_runs_gated_silu_projection_to_cpu_backend_buffer() {
    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 8> down_values {
        0.5F, -0.25F,
        -0.5F, 1.0F,
        0.25F, 0.375F,
        1.5F, -1.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 2, std::span<const float>(down_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::cpu);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::cpu);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::cpu);

    const auto output_buffer = gated_silu_projection_f32_f16_gguf_weight_to_backend(
        left_buffer,
        gate_buffer,
        up_buffer,
        down_buffer
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto gate = matmul_f32_f16_gguf_weight(left, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(left, up_weight, BackendKind::cpu);
    const auto expected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );

    expect(output_buffer.backend() == BackendKind::cpu, "CPU gated SiLU projection output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer gated SiLU projection output");
}

void test_runs_residual_gated_silu_projection_to_cpu_backend_buffer() {
    const std::array<float, 6> input_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 4> attention_values {
        1.25F, -0.5F,
        -0.75F, 0.875F,
    };
    const std::array<float, 6> attention_output_weight_values {
        0.25F, -0.5F, 0.75F,
        1.0F, -0.25F, 0.5F,
    };
    const std::array<float, 3> norm_values {1.0F, 0.75F, 1.25F};
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 12> down_values {
        0.5F, -0.25F, 0.125F,
        -0.5F, 1.0F, 0.625F,
        0.25F, 0.375F, -0.75F,
        1.5F, -1.0F, 0.5F,
    };
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto attention_output = tensor_f32_matrix(2, 2, std::span<const float>(attention_values));
    const auto attention_output_weight =
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(attention_output_weight_values));
    const auto norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 3, std::span<const float>(down_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::cpu);
    const auto attention_output_buffer = copy_tensor_to_backend(attention_output, BackendKind::cpu);
    const auto attention_output_weight_buffer = copy_tensor_to_backend(attention_output_weight, BackendKind::cpu);
    const auto norm_buffer = copy_tensor_to_backend(norm_weight, BackendKind::cpu);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::cpu);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::cpu);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::cpu);

    const auto output_buffer = residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
        input_buffer,
        attention_output_buffer,
        attention_output_weight_buffer,
        norm_buffer,
        gate_buffer,
        up_buffer,
        down_buffer,
        0.25F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto attention_projected = matmul_f32_f16_gguf_weight(
        attention_output,
        attention_output_weight,
        BackendKind::cpu
    );
    const auto attention_residual = add_f32(input, attention_projected);
    const auto normalized = rms_norm_f32(attention_residual, norm_weight, 0.25F);
    const auto gate = matmul_f32_f16_gguf_weight(normalized, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(normalized, up_weight, BackendKind::cpu);
    const auto projected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );
    const auto expected = add_f32(attention_residual, projected);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU residual gated SiLU output backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "CPU backend-buffer residual gated SiLU output");
}

void test_matches_cpu_f32_by_f16_gguf_weight_matmul_on_metal_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> left_values {
        0.5F, -1.0F, 2.0F, 3.5F,
        -2.0F, 0.25F, 1.5F, -0.75F,
        4.0F, -3.0F, 0.125F, 2.25F,
    };
    const std::array<float, 20> right_values {
        0.25F, -0.5F, 1.0F, 2.0F, -1.5F,
        1.25F, 0.75F, -2.0F, 0.5F, 3.0F,
        -0.25F, 1.5F, 0.125F, -1.0F, 2.25F,
        2.5F, -1.25F, 0.625F, 1.75F, -0.875F,
    };
    const auto left = tensor_f32_matrix(3, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 5, std::span<const float>(right_values));

    const auto cpu_output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
    const auto metal_output = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);

    expect(metal_output.descriptor() == cpu_output.descriptor(), "Metal GGUF matmul descriptor mismatch");
    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 5; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(metal_output, std::span<const std::uint64_t>(index)),
                read_f32(cpu_output, std::span<const std::uint64_t>(index)),
                1.0e-3F,
                "Metal GGUF matmul parity mismatch"
            );
        }
    }
}

void test_reuses_metal_f32_by_f16_gguf_weight_matmul_state() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 8> left_values {
        1.0F, -2.0F, 0.5F, 3.0F,
        -1.5F, 0.25F, 2.0F, -0.75F,
    };
    const std::array<float, 12> right_values {
        0.5F, -1.0F, 2.0F,
        1.5F, 0.25F, -0.5F,
        -2.5F, 0.75F, 1.25F,
        0.125F, -1.75F, 2.25F,
    };
    const auto left = tensor_f32_matrix(2, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 3, std::span<const float>(right_values));

    const auto first = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
    const auto second = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);

    expect(second.descriptor() == first.descriptor(), "reused Metal GGUF matmul descriptor mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 3; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(second, std::span<const std::uint64_t>(index)),
                read_f32(first, std::span<const std::uint64_t>(index)),
                0.0F,
                "reused Metal GGUF matmul value mismatch"
            );
        }
    }
}

void test_copies_tensor_to_and_from_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> values {
        1.25F, -2.5F,
        3.75F, -4.5F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    const auto buffer = copy_tensor_to_backend(storage, BackendKind::metal);
    const auto roundtrip = copy_tensor_from_backend(buffer);

    expect(buffer.backend() == BackendKind::metal, "Metal backend buffer kind mismatch");
    expect(buffer.descriptor() == storage.descriptor(), "Metal backend buffer descriptor mismatch");
    expect(roundtrip.descriptor() == storage.descriptor(), "Metal backend buffer roundtrip descriptor mismatch");
    expect(
        std::equal(storage.bytes().begin(), storage.bytes().end(), roundtrip.bytes().begin()),
        "Metal backend buffer roundtrip bytes mismatch"
    );
}

void test_multiplies_f32_by_metal_backend_f16_gguf_weight_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> left_values {
        0.5F, -1.0F, 2.0F, 3.5F,
        -2.0F, 0.25F, 1.5F, -0.75F,
        4.0F, -3.0F, 0.125F, 2.25F,
    };
    const std::array<float, 20> right_values {
        0.25F, -0.5F, 1.0F, 2.0F, -1.5F,
        1.25F, 0.75F, -2.0F, 0.5F, 3.0F,
        -0.25F, 1.5F, 0.125F, -1.0F, 2.25F,
        2.5F, -1.25F, 0.625F, 1.75F, -0.875F,
    };
    const auto left = tensor_f32_matrix(3, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 5, std::span<const float>(right_values));
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);

    const auto cpu_output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
    const auto metal_output = matmul_f32_f16_gguf_weight(left, right_buffer);

    expect(metal_output.descriptor() == cpu_output.descriptor(), "Metal buffered GGUF matmul descriptor mismatch");
    for (std::uint64_t row = 0; row < 3; ++row) {
        for (std::uint64_t column = 0; column < 5; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(metal_output, std::span<const std::uint64_t>(index)),
                read_f32(cpu_output, std::span<const std::uint64_t>(index)),
                1.0e-3F,
                "Metal buffered GGUF matmul parity mismatch"
            );
        }
    }
}

void test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 12> left_values {
        0.5F, -1.0F, 2.0F, 3.5F,
        -2.0F, 0.25F, 1.5F, -0.75F,
        4.0F, -3.0F, 0.125F, 2.25F,
    };
    const std::array<float, 20> right_values {
        0.25F, -0.5F, 1.0F, 2.0F, -1.5F,
        1.25F, 0.75F, -2.0F, 0.5F, 3.0F,
        -0.25F, 1.5F, 0.125F, -1.0F, 2.25F,
        2.5F, -1.25F, 0.625F, 1.75F, -0.875F,
    };
    const auto left = tensor_f32_matrix(3, 4, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(4, 5, std::span<const float>(right_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto right_buffer = copy_tensor_to_backend(right, BackendKind::metal);

    const auto output_buffer = matmul_f32_f16_gguf_weight_to_backend(left_buffer, right_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);

    expect(output_buffer.backend() == BackendKind::metal, "Metal backend-buffer GGUF matmul output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer GGUF matmul output");
}

void test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_metal_backend_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 6> first_values {
        0.25F, -0.5F,
        1.0F, 0.75F,
        -1.25F, 0.5F,
    };
    const std::array<float, 12> second_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 3> third_values {
        1.5F,
        -0.25F,
        0.375F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto first_right = tensor_f16_gguf_matrix(3, 2, std::span<const float>(first_values));
    const auto second_right = tensor_f16_gguf_matrix(3, 4, std::span<const float>(second_values));
    const auto third_right = tensor_f16_gguf_matrix(3, 1, std::span<const float>(third_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto first_buffer = copy_tensor_to_backend(first_right, BackendKind::metal);
    const auto second_buffer = copy_tensor_to_backend(second_right, BackendKind::metal);
    const auto third_buffer = copy_tensor_to_backend(third_right, BackendKind::metal);

    const auto outputs = matmul_f32_f16_gguf_weight_triple_to_backend(
        left_buffer,
        first_buffer,
        second_buffer,
        third_buffer
    );
    const auto first_output = copy_tensor_from_backend(outputs.first);
    const auto second_output = copy_tensor_from_backend(outputs.second);
    const auto third_output = copy_tensor_from_backend(outputs.third);

    expect(outputs.first.backend() == BackendKind::metal, "Metal triple matmul first output backend mismatch");
    expect(outputs.second.backend() == BackendKind::metal, "Metal triple matmul second output backend mismatch");
    expect(outputs.third.backend() == BackendKind::metal, "Metal triple matmul third output backend mismatch");
    expect_f32_matrix_close(
        first_output,
        matmul_f32_f16_gguf_weight(left, first_right, BackendKind::cpu),
        1.0e-3F,
        "Metal triple matmul first output"
    );
    expect_f32_matrix_close(
        second_output,
        matmul_f32_f16_gguf_weight(left, second_right, BackendKind::cpu),
        1.0e-3F,
        "Metal triple matmul second output"
    );
    expect_f32_matrix_close(
        third_output,
        matmul_f32_f16_gguf_weight(left, third_right, BackendKind::cpu),
        1.0e-3F,
        "Metal triple matmul third output"
    );
}

void test_applies_silu_multiply_to_metal_backend_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> gate_values {
        -2.0F, -1.0F, 0.0F,
        0.5F, 1.0F, 2.0F,
    };
    const std::array<float, 6> up_values {
        0.25F, -0.5F, 1.0F,
        1.5F, -2.0F, 0.75F,
    };
    const auto gate = tensor_f32_matrix(2, 3, std::span<const float>(gate_values));
    const auto up = tensor_f32_matrix(2, 3, std::span<const float>(up_values));
    const auto gate_buffer = copy_tensor_to_backend(gate, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up, BackendKind::metal);

    const auto output_buffer = silu_multiply_f32_to_backend(gate_buffer, up_buffer);
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto expected = multiply_f32(silu_f32(gate), up);

    expect(output_buffer.backend() == BackendKind::metal, "Metal SiLU multiply output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-5F, "Metal backend-buffer SiLU multiply output");
}

void test_runs_gated_silu_projection_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> left_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 8> down_values {
        0.5F, -0.25F,
        -0.5F, 1.0F,
        0.25F, 0.375F,
        1.5F, -1.0F,
    };
    const auto left = tensor_f32_matrix(2, 3, std::span<const float>(left_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 2, std::span<const float>(down_values));
    const auto left_buffer = copy_tensor_to_backend(left, BackendKind::metal);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::metal);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::metal);

    const auto output_buffer = gated_silu_projection_f32_f16_gguf_weight_to_backend(
        left_buffer,
        gate_buffer,
        up_buffer,
        down_buffer
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto gate = matmul_f32_f16_gguf_weight(left, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(left, up_weight, BackendKind::cpu);
    const auto expected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );

    expect(output_buffer.backend() == BackendKind::metal, "Metal gated SiLU projection output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer gated SiLU projection output");
}

void test_runs_residual_gated_silu_projection_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 6> input_values {
        0.5F, -1.0F, 2.0F,
        -0.75F, 1.5F, -0.25F,
    };
    const std::array<float, 4> attention_values {
        1.25F, -0.5F,
        -0.75F, 0.875F,
    };
    const std::array<float, 6> attention_output_weight_values {
        0.25F, -0.5F, 0.75F,
        1.0F, -0.25F, 0.5F,
    };
    const std::array<float, 3> norm_values {1.0F, 0.75F, 1.25F};
    const std::array<float, 12> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F,
        -1.0F, 0.125F, 0.375F, -0.625F,
        0.5F, 1.25F, -0.75F, 0.25F,
    };
    const std::array<float, 12> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F,
        0.75F, -0.25F, 1.5F, 0.125F,
        1.0F, 0.25F, -1.25F, 0.5F,
    };
    const std::array<float, 12> down_values {
        0.5F, -0.25F, 0.125F,
        -0.5F, 1.0F, 0.625F,
        0.25F, 0.375F, -0.75F,
        1.5F, -1.0F, 0.5F,
    };
    const auto input = tensor_f32_matrix(2, 3, std::span<const float>(input_values));
    const auto attention_output = tensor_f32_matrix(2, 2, std::span<const float>(attention_values));
    const auto attention_output_weight =
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(attention_output_weight_values));
    const auto norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto gate_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(gate_values));
    const auto up_weight = tensor_f16_gguf_matrix(3, 4, std::span<const float>(up_values));
    const auto down_weight = tensor_f16_gguf_matrix(4, 3, std::span<const float>(down_values));
    const auto input_buffer = copy_tensor_to_backend(input, BackendKind::metal);
    const auto attention_output_buffer = copy_tensor_to_backend(attention_output, BackendKind::metal);
    const auto attention_output_weight_buffer = copy_tensor_to_backend(attention_output_weight, BackendKind::metal);
    const auto norm_buffer = copy_tensor_to_backend(norm_weight, BackendKind::metal);
    const auto gate_buffer = copy_tensor_to_backend(gate_weight, BackendKind::metal);
    const auto up_buffer = copy_tensor_to_backend(up_weight, BackendKind::metal);
    const auto down_buffer = copy_tensor_to_backend(down_weight, BackendKind::metal);

    const auto output_buffer = residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
        input_buffer,
        attention_output_buffer,
        attention_output_weight_buffer,
        norm_buffer,
        gate_buffer,
        up_buffer,
        down_buffer,
        0.25F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto attention_projected = matmul_f32_f16_gguf_weight(
        attention_output,
        attention_output_weight,
        BackendKind::cpu
    );
    const auto attention_residual = add_f32(input, attention_projected);
    const auto normalized = rms_norm_f32(attention_residual, norm_weight, 0.25F);
    const auto gate = matmul_f32_f16_gguf_weight(normalized, gate_weight, BackendKind::cpu);
    const auto up = matmul_f32_f16_gguf_weight(normalized, up_weight, BackendKind::cpu);
    const auto projected = matmul_f32_f16_gguf_weight(
        multiply_f32(silu_f32(gate), up),
        down_weight,
        BackendKind::cpu
    );
    const auto expected = add_f32(attention_residual, projected);

    expect(output_buffer.backend() == BackendKind::metal, "Metal residual gated SiLU output backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Metal backend-buffer residual gated SiLU output");
}

void test_runs_llama_cached_attention_to_cpu_backend_buffer() {
    const std::array<float, 2> query_values {1.0F, 0.0F};
    const std::array<float, 2> key_values {0.5F, -1.0F};
    const std::array<float, 2> value_values {2.0F, -3.0F};
    const auto query = tensor_f32_matrix(1, 2, std::span<const float>(query_values));
    const auto key = tensor_f32_matrix(1, 2, std::span<const float>(key_values));
    const auto value = tensor_f32_matrix(1, 2, std::span<const float>(value_values));
    const auto cache_descriptor = TensorDescriptor::from(Shape::from({4, 2}), DType::f32);
    auto cache_key_buffer = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);
    auto cache_value_buffer = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);

    const auto output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query, BackendKind::cpu),
        copy_tensor_to_backend(key, BackendKind::cpu),
        copy_tensor_to_backend(value, BackendKind::cpu),
        cache_key_buffer,
        cache_value_buffer,
        0,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    const auto output = copy_tensor_from_backend(output_buffer);
    const auto cache_key = copy_tensor_from_backend(cache_key_buffer);
    const auto cache_value = copy_tensor_from_backend(cache_value_buffer);
    auto expected_cache_key = TensorStorage::allocate(cache_descriptor);
    auto expected_cache_value = TensorStorage::allocate(cache_descriptor);
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    write_f32(expected_cache_key, std::span<const std::uint64_t>(row0_col0), 0.5F);
    write_f32(expected_cache_key, std::span<const std::uint64_t>(row0_col1), -1.0F);
    write_f32(expected_cache_value, std::span<const std::uint64_t>(row0_col0), 2.0F);
    write_f32(expected_cache_value, std::span<const std::uint64_t>(row0_col1), -3.0F);

    expect(output_buffer.backend() == BackendKind::cpu, "CPU Llama cached attention output backend mismatch");
    expect_f32_matrix_close(output, value, 1.0e-6F, "CPU backend Llama cached attention output");
    expect_f32_matrix_close(cache_key, expected_cache_key, 1.0e-6F, "CPU backend Llama cached attention key cache");
    expect_f32_matrix_close(cache_value, expected_cache_value, 1.0e-6F, "CPU backend Llama cached attention value cache");
}

void test_matches_llama_cached_attention_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 2> query0_values {0.5F, -0.25F};
    const std::array<float, 2> key0_values {1.0F, 0.5F};
    const std::array<float, 2> value0_values {2.0F, -1.0F};
    const std::array<float, 2> query1_values {0.25F, 0.75F};
    const std::array<float, 2> key1_values {-0.5F, 1.25F};
    const std::array<float, 2> value1_values {0.5F, 3.0F};
    const auto query0 = tensor_f32_matrix(1, 2, std::span<const float>(query0_values));
    const auto key0 = tensor_f32_matrix(1, 2, std::span<const float>(key0_values));
    const auto value0 = tensor_f32_matrix(1, 2, std::span<const float>(value0_values));
    const auto query1 = tensor_f32_matrix(1, 2, std::span<const float>(query1_values));
    const auto key1 = tensor_f32_matrix(1, 2, std::span<const float>(key1_values));
    const auto value1 = tensor_f32_matrix(1, 2, std::span<const float>(value1_values));
    const auto cache_descriptor = TensorDescriptor::from(Shape::from({4, 2}), DType::f32);
    auto cpu_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);
    auto cpu_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::cpu);
    auto metal_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::metal);
    auto metal_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_descriptor), BackendKind::metal);

    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query0, BackendKind::cpu),
        copy_tensor_to_backend(key0, BackendKind::cpu),
        copy_tensor_to_backend(value0, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        0,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    const auto cpu_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query1, BackendKind::cpu),
        copy_tensor_to_backend(key1, BackendKind::cpu),
        copy_tensor_to_backend(value1, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        1,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query0, BackendKind::metal),
        copy_tensor_to_backend(key0, BackendKind::metal),
        copy_tensor_to_backend(value0, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        0,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );
    const auto metal_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(query1, BackendKind::metal),
        copy_tensor_to_backend(key1, BackendKind::metal),
        copy_tensor_to_backend(value1, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        1,
        1,
        1,
        2,
        2,
        2,
        10000.0F,
        1.0F
    );

    expect(metal_output_buffer.backend() == BackendKind::metal, "Metal Llama cached attention output backend mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-4F,
        "Metal backend Llama cached attention output"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_key),
        copy_tensor_from_backend(cpu_cache_key),
        1.0e-4F,
        "Metal backend Llama cached attention key cache"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_value),
        copy_tensor_from_backend(cpu_cache_value),
        1.0e-4F,
        "Metal backend Llama cached attention value cache"
    );
}

void test_matches_llama_one_token_grouped_cached_attention_to_metal_backend_buffer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint64_t prefill_rows = 3;
    constexpr std::uint64_t decode_rows = 1;
    constexpr std::uint64_t head_count = 4;
    constexpr std::uint64_t kv_head_count = 2;
    constexpr std::uint64_t key_length = 4;
    constexpr std::uint64_t value_length = 3;
    constexpr std::uint64_t cache_capacity = 8;
    constexpr std::uint64_t query_width = head_count * key_length;
    constexpr std::uint64_t key_width = kv_head_count * key_length;
    constexpr std::uint64_t value_width = kv_head_count * value_length;
    constexpr std::uint64_t rope_dimension_count = key_length;
    constexpr float frequency_base = 10000.0F;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(key_length));
    auto make_values = [](std::size_t count, float scale, float bias) {
        std::vector<float> values;
        values.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto centered = static_cast<float>(static_cast<int>((index * 7U) % 17U) - 8);
            values.push_back(bias + (centered * scale) + (0.0003F * static_cast<float>(index)));
        }
        return values;
    };

    const auto prefill_query_values = make_values(prefill_rows * query_width, 0.013F, 0.002F);
    const auto prefill_key_values = make_values(prefill_rows * key_width, 0.011F, -0.003F);
    const auto prefill_value_values = make_values(prefill_rows * value_width, 0.017F, 0.005F);
    const auto decode_query_values = make_values(decode_rows * query_width, 0.019F, -0.004F);
    const auto decode_key_values = make_values(decode_rows * key_width, 0.007F, 0.006F);
    const auto decode_value_values = make_values(decode_rows * value_width, 0.023F, -0.001F);
    const auto prefill_query = tensor_f32_matrix(
        prefill_rows,
        query_width,
        std::span<const float>(prefill_query_values.data(), prefill_query_values.size())
    );
    const auto prefill_key = tensor_f32_matrix(
        prefill_rows,
        key_width,
        std::span<const float>(prefill_key_values.data(), prefill_key_values.size())
    );
    const auto prefill_value = tensor_f32_matrix(
        prefill_rows,
        value_width,
        std::span<const float>(prefill_value_values.data(), prefill_value_values.size())
    );
    const auto decode_query = tensor_f32_matrix(
        decode_rows,
        query_width,
        std::span<const float>(decode_query_values.data(), decode_query_values.size())
    );
    const auto decode_key = tensor_f32_matrix(
        decode_rows,
        key_width,
        std::span<const float>(decode_key_values.data(), decode_key_values.size())
    );
    const auto decode_value = tensor_f32_matrix(
        decode_rows,
        value_width,
        std::span<const float>(decode_value_values.data(), decode_value_values.size())
    );
    const auto cache_key_descriptor = TensorDescriptor::from(Shape::from({cache_capacity, key_width}), DType::f32);
    const auto cache_value_descriptor = TensorDescriptor::from(Shape::from({cache_capacity, value_width}), DType::f32);
    auto cpu_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_key_descriptor), BackendKind::cpu);
    auto cpu_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_value_descriptor), BackendKind::cpu);
    auto metal_cache_key = copy_tensor_to_backend(TensorStorage::allocate(cache_key_descriptor), BackendKind::metal);
    auto metal_cache_value = copy_tensor_to_backend(TensorStorage::allocate(cache_value_descriptor), BackendKind::metal);

    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(prefill_query, BackendKind::cpu),
        copy_tensor_to_backend(prefill_key, BackendKind::cpu),
        copy_tensor_to_backend(prefill_value, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        0,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );
    const auto cpu_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(decode_query, BackendKind::cpu),
        copy_tensor_to_backend(decode_key, BackendKind::cpu),
        copy_tensor_to_backend(decode_value, BackendKind::cpu),
        cpu_cache_key,
        cpu_cache_value,
        prefill_rows,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );

    (void)llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(prefill_query, BackendKind::metal),
        copy_tensor_to_backend(prefill_key, BackendKind::metal),
        copy_tensor_to_backend(prefill_value, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        0,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );
    const auto metal_output_buffer = llama_cached_attention_f32_to_backend(
        copy_tensor_to_backend(decode_query, BackendKind::metal),
        copy_tensor_to_backend(decode_key, BackendKind::metal),
        copy_tensor_to_backend(decode_value, BackendKind::metal),
        metal_cache_key,
        metal_cache_value,
        prefill_rows,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );

    expect(
        metal_output_buffer.backend() == BackendKind::metal,
        "Metal grouped one-token Llama cached attention output backend mismatch"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_output_buffer),
        copy_tensor_from_backend(cpu_output_buffer),
        1.0e-4F,
        "Metal grouped one-token Llama cached attention output"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_key),
        copy_tensor_from_backend(cpu_cache_key),
        1.0e-4F,
        "Metal grouped one-token Llama cached attention key cache"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(metal_cache_value),
        copy_tensor_from_backend(cpu_cache_value),
        1.0e-4F,
        "Metal grouped one-token Llama cached attention value cache"
    );
}

void test_reports_unavailable_metal_backend_for_gguf_matmul_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> left_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> right_values {
        5.0F, 6.0F,
        7.0F, 8.0F,
    };
    const auto left = tensor_f32_matrix(2, 2, std::span<const float>(left_values));
    const auto right = tensor_f16_gguf_matrix(2, 2, std::span<const float>(right_values));

    try {
        (void)matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
        throw TestFailure("Metal GGUF matmul should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal GGUF matmul unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal GGUF matmul unavailable backend mismatch");
        expect(
            !tensor_backend_unavailable_reason(BackendKind::metal).empty(),
            "Metal backend unavailable reason should be non-empty"
        );
    }
}

void test_reports_unavailable_metal_backend_buffer_copy_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const std::array<float, 4> values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const auto storage = tensor_f32_matrix(2, 2, std::span<const float>(values));

    try {
        (void)copy_tensor_to_backend(storage, BackendKind::metal);
        throw TestFailure("Metal backend buffer copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Metal backend buffer unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Metal backend buffer unavailable backend mismatch");
    }
}

void test_rejects_f32_f16_matmul_right_dtype_mismatch() {
    const auto left = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));
    const auto right = TensorStorage::allocate(TensorDescriptor::from(Shape::from({2, 2}), DType::f32));

    try {
        (void)matmul_f32_f16(left, right);
        throw TestFailure("F32xF16 matmul should reject F32 right operand");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "F32xF16 matmul dtype error code mismatch"
        );
        expect(error.expected_value() == 1, "F32xF16 matmul expected dtype mismatch");
        expect(error.actual_value() == 0, "F32xF16 matmul actual dtype mismatch");
    }
}

LlamaModelConfig tiny_attention_layer_config() {
    return LlamaModelConfig {
        .context_length = 16,
        .embedding_length = 2,
        .block_count = 1,
        .feed_forward_length = 8,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .attention_key_length = 2,
        .attention_value_length = 2,
        .rope_dimension_count = 2,
        .vocab_size = 8,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 3.5F,
    };
}

void test_creates_llama_kv_cache() {
    auto config = tiny_attention_layer_config();
    config.block_count = 2;

    const auto cache = create_llama_kv_cache(config);

    expect(cache.layers.size() == 2, "Llama KV cache layer count mismatch");
    expect(cache.context_length == 16, "Llama KV cache context length mismatch");
    expect(cache.key_width == 2, "Llama KV cache key width mismatch");
    expect(cache.value_width == 2, "Llama KV cache value width mismatch");
    expect(cache.layers[0].filled_token_count == 0, "Llama KV cache initial fill mismatch");
    expect(cache.layers[0].key.descriptor().shape == Shape::from({16, 2}), "Llama KV cache key shape mismatch");
    expect(cache.layers[0].key.descriptor().dtype == DType::f32, "Llama KV cache key dtype mismatch");
    expect(cache.layers[0].value.descriptor().shape == Shape::from({16, 2}), "Llama KV cache value shape mismatch");
    expect(cache.layers[0].value.descriptor().dtype == DType::f32, "Llama KV cache value dtype mismatch");
}

void test_appends_and_reads_llama_kv_cache_layer() {
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());
    const std::array<float, 4> key_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> value_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
    };
    const auto key_rows = tensor_f32_matrix(2, 2, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(2, 2, std::span<const float>(value_values));

    append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 0);
    const auto key_prefix = llama_kv_cache_layer_key_prefix_f32(cache.layers[0]);
    const auto value_prefix = llama_kv_cache_layer_value_prefix_f32(cache.layers[0]);

    expect(cache.layers[0].filled_token_count == 2, "Llama KV cache filled token count mismatch");
    expect(key_prefix.descriptor().shape == Shape::from({2, 2}), "Llama KV cache key prefix shape mismatch");
    expect(value_prefix.descriptor().shape == Shape::from({2, 2}), "Llama KV cache value prefix shape mismatch");
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    expect(read_f32(key_prefix, std::span<const std::uint64_t>(row1_col0)) == 3.0F, "Llama KV key prefix value mismatch");
    expect(read_f32(value_prefix, std::span<const std::uint64_t>(row1_col1)) == 40.0F, "Llama KV value prefix value mismatch");
}

void test_rejects_llama_kv_cache_append_position_mismatch() {
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());
    const std::array<float, 2> key_values {1.0F, 2.0F};
    const std::array<float, 2> value_values {10.0F, 20.0F};
    const auto key_rows = tensor_f32_matrix(1, 2, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(1, 2, std::span<const float>(value_values));

    try {
        append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 1);
        throw TestFailure("Llama KV cache should reject non-contiguous append position");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama KV cache append position must match filled token count",
            "Llama KV cache position mismatch error text mismatch"
        );
    }
}

void test_rejects_llama_kv_cache_context_overflow() {
    auto config = tiny_attention_layer_config();
    config.context_length = 1;
    auto cache = create_llama_kv_cache(config);
    const std::array<float, 4> key_values {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::array<float, 4> value_values {
        10.0F, 20.0F,
        30.0F, 40.0F,
    };
    const auto key_rows = tensor_f32_matrix(2, 2, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(2, 2, std::span<const float>(value_values));

    try {
        append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 0);
        throw TestFailure("Llama KV cache should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama KV cache append would exceed context length",
            "Llama KV cache overflow error text mismatch"
        );
    }
}

void test_rejects_llama_kv_cache_key_shape_mismatch() {
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());
    const std::array<float, 1> key_values {1.0F};
    const std::array<float, 2> value_values {10.0F, 20.0F};
    const auto key_rows = tensor_f32_matrix(1, 1, std::span<const float>(key_values));
    const auto value_rows = tensor_f32_matrix(1, 2, std::span<const float>(value_values));

    try {
        append_llama_kv_cache_layer_f32(cache.layers[0], key_rows, value_rows, 0);
        throw TestFailure("Llama KV cache should reject key width mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Llama KV cache key shape error mismatch");
        expect(error.expected_value() == 2, "Llama KV cache expected key width mismatch");
        expect(error.actual_value() == 1, "Llama KV cache actual key width mismatch");
    }
}

void test_runs_llama_attention_layer_with_f16_weights() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    const auto output = run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);

    expect(output.descriptor().shape == Shape::from({1, 2}), "Llama attention layer output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama attention layer output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        3.75F,
        0.000001F,
        "Llama attention layer output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        5.0F,
        0.000001F,
        "Llama attention layer output value 1 mismatch"
    );
}

void test_runs_llama_attention_layer_with_causal_mask() {
    const std::array<float, 4> input_values {
        3.0F, 4.0F,
        30.0F, 40.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> zero_values {
        0.0F, 0.0F,
        0.0F, 0.0F,
    };
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(2, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    const auto output = run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 1);

    expect(output.descriptor().shape == Shape::from({2, 2}), "Llama causal attention output shape mismatch");
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};
    const auto row1_inverse_rms = 1.0 / std::sqrt(1250.0 + 3.5);
    const auto row1_norm0 = 30.0 * row1_inverse_rms;
    const auto row1_norm1 = 40.0 * row1_inverse_rms;

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        3.75F,
        0.000001F,
        "Llama causal attention row 0 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col1)),
        5.0F,
        0.000001F,
        "Llama causal attention row 0 value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        static_cast<float>(30.0 + (0.75 + row1_norm0) / 2.0),
        0.000001F,
        "Llama causal attention row 1 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        static_cast<float>(40.0 + (1.0 + row1_norm1) / 2.0),
        0.000001F,
        "Llama causal attention row 1 value 1 mismatch"
    );
}

void test_runs_llama_attention_layer_with_kv_cache_matches_no_cache() {
    const std::array<float, 4> full_input_values {
        3.0F, 4.0F,
        30.0F, 40.0F,
    };
    const std::array<float, 2> first_input_values {3.0F, 4.0F};
    const std::array<float, 2> second_input_values {30.0F, 40.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> zero_values {
        0.0F, 0.0F,
        0.0F, 0.0F,
    };
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const auto full_input = tensor_f32_matrix(2, 2, std::span<const float>(full_input_values));
    const auto first_input = tensor_f32_matrix(1, 2, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 2, std::span<const float>(second_input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(zero_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(tiny_attention_layer_config());

    const auto no_cache = run_llama_attention_layer_f32(full_input, weights, tiny_attention_layer_config(), 1);
    (void)run_llama_attention_layer_with_kv_cache_f32(first_input, weights, tiny_attention_layer_config(), 0, cache.layers[0]);
    const auto cached = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        weights,
        tiny_attention_layer_config(),
        1,
        cache.layers[0]
    );

    expect(cache.layers[0].filled_token_count == 2, "Llama attention KV cache fill mismatch");
    expect(cached.descriptor().shape == Shape::from({1, 2}), "Llama cached attention output shape mismatch");
    const std::array<std::uint64_t, 2> no_cache_row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> no_cache_row1_col1 {1, 1};
    const std::array<std::uint64_t, 2> cached_row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> cached_row0_col1 {0, 1};
    expect_close(
        read_f32(cached, std::span<const std::uint64_t>(cached_row0_col0)),
        read_f32(no_cache, std::span<const std::uint64_t>(no_cache_row1_col0)),
        0.000001F,
        "Llama cached attention value 0 should match no-cache row 1"
    );
    expect_close(
        read_f32(cached, std::span<const std::uint64_t>(cached_row0_col1)),
        read_f32(no_cache, std::span<const std::uint64_t>(no_cache_row1_col1)),
        0.000001F,
        "Llama cached attention value 1 should match no-cache row 1"
    );
}

void test_runs_grouped_llama_attention_layer_with_kv_cache_matches_no_cache() {
    const std::array<float, 12> full_input_values {
        1.0F, 2.0F, 3.0F, 4.0F,
        -2.0F, 1.0F, 0.5F, 3.0F,
        4.0F, -1.0F, 2.0F, 0.0F,
    };
    const std::array<float, 8> prefill_input_values {
        1.0F, 2.0F, 3.0F, 4.0F,
        -2.0F, 1.0F, 0.5F, 3.0F,
    };
    const std::array<float, 4> decode_input_values {
        4.0F, -1.0F, 2.0F, 0.0F,
    };
    const std::array<float, 4> norm_values {1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 32> query_weight_values {};
    std::array<float, 16> key_weight_values {};
    std::array<float, 16> value_weight_values {};
    std::array<float, 32> output_weight_values {};

    for (std::uint64_t row = 0; row < 4; ++row) {
        for (std::uint64_t column = 0; column < 8; ++column) {
            const auto pattern = static_cast<int>((row + 1) * (column + 2)) % 7 - 3;
            query_weight_values[static_cast<std::size_t>(row * 8 + column)] = static_cast<float>(pattern) * 0.125F;
        }
    }

    for (std::uint64_t row = 0; row < 4; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            const auto key_pattern = static_cast<int>((row + 2) * (column + 1)) % 5 - 2;
            const auto value_pattern = static_cast<int>((row + 3) * (column + 2)) % 7 - 3;
            key_weight_values[static_cast<std::size_t>(row * 4 + column)] =
                static_cast<float>(key_pattern) * 0.25F;
            value_weight_values[static_cast<std::size_t>(row * 4 + column)] =
                static_cast<float>(value_pattern) * 0.125F;
        }
    }

    for (std::uint64_t row = 0; row < 8; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            const auto pattern = static_cast<int>((row + 1) + (column * 2)) % 7 - 3;
            output_weight_values[static_cast<std::size_t>(row * 4 + column)] =
                static_cast<float>(pattern) * 0.125F;
        }
    }

    const auto full_input = tensor_f32_matrix(3, 4, std::span<const float>(full_input_values));
    const auto prefill_input = tensor_f32_matrix(2, 4, std::span<const float>(prefill_input_values));
    const auto decode_input = tensor_f32_matrix(1, 4, std::span<const float>(decode_input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(4, 8, std::span<const float>(query_weight_values));
    auto key_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(key_weight_values));
    auto value_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(value_weight_values));
    auto output_weight = tensor_f16_gguf_matrix(8, 4, std::span<const float>(output_weight_values));
    auto config = tiny_attention_layer_config();
    config.embedding_length = 4;
    config.attention_head_count = 4;
    config.attention_head_count_kv = 2;
    config.attention_key_length = 2;
    config.attention_value_length = 2;
    config.rope_freq_base = 2.0F;
    config.rms_norm_epsilon = 0.25F;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(config);

    const auto no_cache = run_llama_attention_layer_f32(full_input, weights, config, 2);
    (void)run_llama_attention_layer_with_kv_cache_f32(prefill_input, weights, config, 1, cache.layers[0]);
    const auto cached = run_llama_attention_layer_with_kv_cache_f32(
        decode_input,
        weights,
        config,
        2,
        cache.layers[0]
    );

    expect(cache.layers[0].filled_token_count == 3, "grouped Llama attention KV cache fill mismatch");
    expect(cached.descriptor().shape == Shape::from({1, 4}), "grouped Llama cached attention output shape mismatch");
    for (std::uint64_t column = 0; column < 4; ++column) {
        const std::array<std::uint64_t, 2> no_cache_index {2, column};
        const std::array<std::uint64_t, 2> cached_index {0, column};
        expect_close(
            read_f32(cached, std::span<const std::uint64_t>(cached_index)),
            read_f32(no_cache, std::span<const std::uint64_t>(no_cache_index)),
            0.00001F,
            "grouped Llama cached attention should match no-cache final row"
        );
    }
}

void test_runs_llama_attention_layer_with_kv_cache_appends_rotated_key_heads() {
    const std::array<float, 8> input_values {
        1.0F, 0.0F, 0.0F, 1.0F,
        0.0F, 1.0F, 1.0F, 0.0F,
    };
    const std::array<float, 4> norm_values {1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 16> identity_values {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    const auto input = tensor_f32_matrix(2, 4, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(4, 4, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.embedding_length = 4;
    config.attention_head_count = 2;
    config.attention_head_count_kv = 2;
    config.attention_key_length = 2;
    config.attention_value_length = 2;
    config.rope_freq_base = 1.0F;
    config.rms_norm_epsilon = 0.5F;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(config);

    (void)run_llama_attention_layer_with_kv_cache_f32(input, weights, config, 1, cache.layers[0]);

    const auto sine = static_cast<float>(std::sin(1.0));
    const auto cosine = static_cast<float>(std::cos(1.0));
    const std::array<float, 8> expected_key_values {
        1.0F, 0.0F, 0.0F, 1.0F,
        -sine, cosine, cosine, sine,
    };
    const std::array<float, 4> expected_value_row1 {
        0.0F, 1.0F, 1.0F, 0.0F,
    };

    expect(cache.layers[0].filled_token_count == 2, "Llama cached attention rotated key fill mismatch");
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            const std::array<std::uint64_t, 2> index {row, column};
            expect_close(
                read_f32(cache.layers[0].key, std::span<const std::uint64_t>(index)),
                expected_key_values[static_cast<std::size_t>(row * 4 + column)],
                0.000001F,
                "Llama cached rotated key value mismatch"
            );
        }
    }

    for (std::uint64_t column = 0; column < 4; ++column) {
        const std::array<std::uint64_t, 2> index {1, column};
        expect_close(
            read_f32(cache.layers[0].value, std::span<const std::uint64_t>(index)),
            expected_value_row1[static_cast<std::size_t>(column)],
            0.000001F,
            "Llama cached appended value mismatch"
        );
    }
}

void test_rejects_llama_attention_layer_with_kv_cache_invalid_rope_dimension_count() {
    const std::array<float, 2> input_values {1.0F, 0.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.rope_dimension_count = 1;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    auto cache = create_llama_kv_cache(config);

    try {
        (void)run_llama_attention_layer_with_kv_cache_f32(input, weights, config, 0, cache.layers[0]);
        throw TestFailure("cached Llama attention should reject invalid RoPE dimension count");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_shape_dimension,
            "cached Llama attention RoPE dimension error mismatch"
        );
        expect(error.expected_value() == 2, "cached Llama attention expected RoPE dimension mismatch");
        expect(error.actual_value() == 1, "cached Llama attention actual RoPE dimension mismatch");
        expect(cache.layers[0].filled_token_count == 0, "cached Llama attention should not append invalid RoPE rows");
    }
}

void test_runs_llama_attention_layer_with_rowwise_rope_positions() {
    const std::array<float, 4> input_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(2, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.rope_freq_base = 1.0F;
    config.rms_norm_epsilon = 0.5F;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    const auto output = run_llama_attention_layer_f32(input, weights, config, 1);

    const auto score_scale = 1.0 / std::sqrt(2.0);
    const auto score0 = -std::sin(1.0) * score_scale;
    const auto score1 = score_scale;
    const auto score_max = score0 > score1 ? score0 : score1;
    const auto weight0_numerator = std::exp(score0 - score_max);
    const auto weight1_numerator = std::exp(score1 - score_max);
    const auto weight_sum = weight0_numerator + weight1_numerator;
    const auto row1_weight0 = static_cast<float>(weight0_numerator / weight_sum);
    const auto row1_weight1 = static_cast<float>(weight1_numerator / weight_sum);
    const std::array<std::uint64_t, 2> row0_col0 {0, 0};
    const std::array<std::uint64_t, 2> row0_col1 {0, 1};
    const std::array<std::uint64_t, 2> row1_col0 {1, 0};
    const std::array<std::uint64_t, 2> row1_col1 {1, 1};

    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col0)),
        2.0F,
        0.000001F,
        "Llama rowwise RoPE row 0 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row0_col1)),
        0.0F,
        0.000001F,
        "Llama rowwise RoPE row 0 value 1 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col0)),
        row1_weight0,
        0.000001F,
        "Llama rowwise RoPE row 1 value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(row1_col1)),
        1.0F + row1_weight1,
        0.000001F,
        "Llama rowwise RoPE row 1 value 1 mismatch"
    );
}

void test_rejects_llama_attention_layer_position_before_input_window() {
    const std::array<float, 4> input_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(2, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);
        throw TestFailure("Llama attention layer should reject a final position before the input window");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama attention position must cover input rows",
            "Llama attention layer input-window position error mismatch"
        );
    }
}

void test_rejects_llama_attention_layer_projection_shape_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 2> narrow_values {1.0F, 0.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 1, std::span<const float>(narrow_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);
        throw TestFailure("Llama attention layer should reject query projection shape mismatch");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::tensor_shape_mismatch,
            "Llama attention layer projection shape error mismatch"
        );
        expect(error.expected_value() == 2, "Llama attention layer expected query width mismatch");
        expect(error.actual_value() == 1, "Llama attention layer actual query width mismatch");
    }
}

void test_rejects_llama_attention_layer_projection_dtype_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f32_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, tiny_attention_layer_config(), 0);
        throw TestFailure("Llama attention layer should reject F32 query projection weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "Llama attention layer projection dtype error mismatch"
        );
        expect(error.expected_value() == 1, "Llama attention layer expected projection dtype mismatch");
        expect(error.actual_value() == 0, "Llama attention layer actual projection dtype mismatch");
    }
}

void test_rejects_llama_attention_layer_invalid_head_layout() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_attention_layer_config();
    config.attention_head_count = 3;
    config.attention_head_count_kv = 2;
    const LlamaAttentionLayerWeights weights {
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };

    try {
        (void)run_llama_attention_layer_f32(input, weights, config, 0);
        throw TestFailure("Llama attention layer should reject non-divisible head layout");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama attention head count must be divisible by KV head count",
            "Llama attention layer invalid head layout error mismatch"
        );
    }
}

LlamaModelConfig tiny_feed_forward_layer_config() {
    auto config = tiny_attention_layer_config();
    config.feed_forward_length = 2;
    return config;
}

void test_runs_llama_feed_forward_layer_with_f16_weights() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    const auto output = run_llama_feed_forward_layer_f32(input, weights, tiny_feed_forward_layer_config());
    const auto first_norm = 0.75;
    const auto second_norm = 1.0;
    const auto first_expected = 3.0 + first_norm * (first_norm / (1.0 + std::exp(-first_norm)));
    const auto second_expected = 4.0 + second_norm * (second_norm / (1.0 + std::exp(-second_norm)));

    expect(output.descriptor().shape == Shape::from({1, 2}), "Llama FFN layer output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama FFN layer output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(first_expected),
        0.000001F,
        "Llama FFN layer output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(second_expected),
        0.000001F,
        "Llama FFN layer output value 1 mismatch"
    );
}

void test_rejects_llama_feed_forward_layer_projection_shape_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 2> narrow_values {1.0F, 0.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 1, std::span<const float>(narrow_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    try {
        (void)run_llama_feed_forward_layer_f32(input, weights, tiny_feed_forward_layer_config());
        throw TestFailure("Llama FFN layer should reject gate projection shape mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Llama FFN shape error mismatch");
        expect(error.expected_value() == 2, "Llama FFN expected gate width mismatch");
        expect(error.actual_value() == 1, "Llama FFN actual gate width mismatch");
    }
}

void test_rejects_llama_feed_forward_layer_projection_dtype_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f32_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    try {
        (void)run_llama_feed_forward_layer_f32(input, weights, tiny_feed_forward_layer_config());
        throw TestFailure("Llama FFN layer should reject F32 gate projection weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "Llama FFN projection dtype error mismatch"
        );
        expect(error.expected_value() == 1, "Llama FFN expected projection dtype mismatch");
        expect(error.actual_value() == 0, "Llama FFN actual projection dtype mismatch");
    }
}

void test_rejects_llama_feed_forward_layer_invalid_config() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto config = tiny_feed_forward_layer_config();
    config.feed_forward_length = 0;
    const LlamaFeedForwardLayerWeights weights {
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };

    try {
        (void)run_llama_feed_forward_layer_f32(input, weights, config);
        throw TestFailure("Llama FFN layer should reject zero feed-forward length");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama layer config dimensions must be positive",
            "Llama FFN invalid config error mismatch"
        );
    }
}

void test_runs_llama_block_with_f16_weights() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaBlockWeights weights {
        .attention = LlamaAttentionLayerWeights {
            .attention_norm_weight = attention_norm_weight,
            .query_weight = query_weight,
            .key_weight = key_weight,
            .value_weight = value_weight,
            .output_weight = output_weight,
        },
        .feed_forward = LlamaFeedForwardLayerWeights {
            .feed_forward_norm_weight = feed_forward_norm_weight,
            .gate_weight = gate_weight,
            .up_weight = up_weight,
            .down_weight = down_weight,
        },
    };

    const auto output = run_llama_block_f32(input, weights, tiny_feed_forward_layer_config(), 0);
    const auto attention_first = 3.75;
    const auto attention_second = 5.0;
    const auto block_mean_square = ((attention_first * attention_first) + (attention_second * attention_second)) / 2.0;
    const auto inverse_rms = 1.0 / std::sqrt(block_mean_square + 3.5);
    const auto first_norm = attention_first * inverse_rms;
    const auto second_norm = attention_second * inverse_rms;
    const auto first_expected = attention_first + first_norm * (first_norm / (1.0 + std::exp(-first_norm)));
    const auto second_expected = attention_second + second_norm * (second_norm / (1.0 + std::exp(-second_norm)));

    expect(output.descriptor().shape == Shape::from({1, 2}), "Llama block output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama block output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        static_cast<float>(first_expected),
        0.000001F,
        "Llama block output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        static_cast<float>(second_expected),
        0.000001F,
        "Llama block output value 1 mismatch"
    );
}

void test_rejects_llama_block_attention_projection_shape_mismatch() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 2> narrow_values {1.0F, 0.0F};
    const std::array<float, 4> identity_values {
        1.0F, 0.0F,
        0.0F, 1.0F,
    };
    auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    auto attention_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto feed_forward_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    auto query_weight = tensor_f16_gguf_matrix(2, 1, std::span<const float>(narrow_values));
    auto key_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto value_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto output_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto gate_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto up_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    auto down_weight = tensor_f16_gguf_matrix(2, 2, std::span<const float>(identity_values));
    const LlamaBlockWeights weights {
        .attention = LlamaAttentionLayerWeights {
            .attention_norm_weight = attention_norm_weight,
            .query_weight = query_weight,
            .key_weight = key_weight,
            .value_weight = value_weight,
            .output_weight = output_weight,
        },
        .feed_forward = LlamaFeedForwardLayerWeights {
            .feed_forward_norm_weight = feed_forward_norm_weight,
            .gate_weight = gate_weight,
            .up_weight = up_weight,
            .down_weight = down_weight,
        },
    };

    try {
        (void)run_llama_block_f32(input, weights, tiny_feed_forward_layer_config(), 0);
        throw TestFailure("Llama block should reject attention projection shape mismatch");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_shape_mismatch, "Llama block shape error mismatch");
        expect(error.expected_value() == 2, "Llama block expected query width mismatch");
        expect(error.actual_value() == 1, "Llama block actual query width mismatch");
    }
}

EngineTensor engine_tensor(std::string name, TensorStorage storage) {
    return EngineTensor {
        .name = std::move(name),
        .storage = std::move(storage),
    };
}

[[nodiscard]] std::string llama_block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

void add_tiny_llama_block_tensors(
    std::vector<EngineTensor>& tensors,
    std::uint32_t block_index,
    float projection_scale,
    bool query_as_f32,
    bool include_down_weight
) {
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 4> projection_values {
        projection_scale, 0.0F,
        0.0F, projection_scale,
    };

    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    if (query_as_f32) {
        tensors.push_back(engine_tensor(
            llama_block_tensor_name(block_index, "attn_q.weight"),
            tensor_f32_matrix(2, 2, std::span<const float>(projection_values))
        ));
    } else {
        tensors.push_back(engine_tensor(
            llama_block_tensor_name(block_index, "attn_q.weight"),
            tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
        ));
    }
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_k.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_v.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "attn_output.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_norm.weight"),
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_gate.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(block_index, "ffn_up.weight"),
        tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
    ));
    if (include_down_weight) {
        tensors.push_back(engine_tensor(
            llama_block_tensor_name(block_index, "ffn_down.weight"),
            tensor_f16_gguf_matrix(2, 2, std::span<const float>(projection_values))
        ));
    }
}

void add_tiny_llama_top_level_tensors(std::vector<EngineTensor>& tensors) {
    const std::array<float, 16> token_embedding_values {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
        10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F, 70.0F, 80.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 16> output_values {
        1.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };

    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 8, std::span<const float>(token_embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 8, std::span<const float>(output_values))
    ));
}

[[nodiscard]] EngineModel tiny_bound_llama_model(LlamaModelConfig config, std::vector<EngineTensor> tensors) {
    return EngineModel(ModelArchitecture::llama, {}, config, std::move(tensors));
}

LlamaModelConfig backend_feed_forward_fixture_config() {
    return LlamaModelConfig {
        .context_length = 8,
        .embedding_length = 4,
        .block_count = 1,
        .feed_forward_length = 6,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .attention_key_length = 4,
        .attention_value_length = 4,
        .rope_dimension_count = 4,
        .vocab_size = 5,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 0.25F,
    };
}

void add_backend_feed_forward_fixture_tensors(std::vector<EngineTensor>& tensors) {
    const std::array<float, 20> token_embedding_values {
        0.5F, -1.0F, 1.5F, 0.25F, -0.75F,
        1.0F, 0.5F, -0.25F, 1.25F, -1.5F,
        -0.5F, 1.75F, 0.75F, -1.25F, 0.0F,
        1.5F, -0.75F, 0.25F, 0.5F, 1.0F,
    };
    const std::array<float, 20> output_values {
        0.25F, -0.5F, 1.0F, 0.75F, -1.25F,
        1.5F, 0.375F, -0.875F, 0.125F, 0.5F,
        -0.75F, 1.25F, 0.625F, -0.25F, 1.0F,
        0.875F, -1.5F, 0.5F, 1.125F, -0.375F,
    };
    const std::array<float, 4> attention_norm_values {1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 4> feed_forward_norm_values {1.0F, 0.75F, 1.25F, 0.5F};
    const std::array<float, 16> query_values {
        0.5F, -0.25F, 1.0F, 0.75F,
        -0.5F, 0.875F, 0.25F, -1.25F,
        1.5F, 0.0F, -0.75F, 0.5F,
        0.125F, 1.25F, -1.0F, 0.375F,
    };
    const std::array<float, 16> key_values {
        -0.75F, 0.5F, 0.25F, 1.25F,
        1.0F, -1.5F, 0.625F, -0.125F,
        0.375F, 0.875F, -0.5F, 0.75F,
        -1.25F, 0.25F, 1.5F, -0.375F,
    };
    const std::array<float, 16> value_values {
        1.25F, -0.5F, 0.875F, 0.0F,
        -0.25F, 1.5F, -1.0F, 0.625F,
        0.75F, 0.125F, 0.5F, -1.25F,
        -1.5F, 0.375F, 1.0F, 0.25F,
    };
    const std::array<float, 16> attention_output_values {
        0.625F, -0.875F, 0.25F, 1.0F,
        1.25F, 0.5F, -1.5F, 0.125F,
        -0.375F, 1.125F, 0.75F, -0.5F,
        0.0F, -0.25F, 1.5F, 0.875F,
    };
    const std::array<float, 24> gate_values {
        0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.5F,
        -1.0F, 0.125F, 0.375F, -0.625F, 0.875F, 0.25F,
        0.5F, 1.25F, -0.75F, 0.0F, 0.625F, -0.375F,
        1.5F, -1.25F, 0.25F, 0.5F, -0.875F, 0.125F,
    };
    const std::array<float, 24> up_values {
        -0.125F, 0.625F, 0.5F, -1.0F, 0.375F, 1.25F,
        0.75F, -0.25F, 1.5F, 0.125F, -0.5F, 0.875F,
        1.0F, 0.25F, -1.25F, 0.5F, 0.75F, -0.625F,
        -0.75F, 1.125F, 0.0F, -0.375F, 1.375F, 0.25F,
    };
    const std::array<float, 24> down_values {
        0.5F, -0.25F, 0.125F, 0.75F,
        -0.5F, 1.0F, 0.625F, -0.125F,
        0.25F, 0.375F, -0.75F, 1.25F,
        1.5F, -1.0F, 0.5F, 0.0F,
        -0.875F, 0.25F, 1.125F, -0.5F,
        0.625F, 0.75F, -1.25F, 0.375F,
    };

    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(4, 5, std::span<const float>(token_embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(attention_norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(4, 5, std::span<const float>(output_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(attention_norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_q.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(query_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_k.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(key_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_v.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(value_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "attn_output.weight"),
        tensor_f16_gguf_matrix(4, 4, std::span<const float>(attention_output_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_norm.weight"),
        tensor_f32_vector(std::span<const float>(feed_forward_norm_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_gate.weight"),
        tensor_f16_gguf_matrix(4, 6, std::span<const float>(gate_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_up.weight"),
        tensor_f16_gguf_matrix(4, 6, std::span<const float>(up_values))
    ));
    tensors.push_back(engine_tensor(
        llama_block_tensor_name(0, "ffn_down.weight"),
        tensor_f16_gguf_matrix(6, 4, std::span<const float>(down_values))
    ));
}

EngineModel backend_feed_forward_fixture_model() {
    std::vector<EngineTensor> tensors;
    add_backend_feed_forward_fixture_tensors(tensors);
    return tiny_bound_llama_model(backend_feed_forward_fixture_config(), std::move(tensors));
}

TensorStorage backend_feed_forward_fixture_input() {
    const std::array<float, 8> input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    return tensor_f32_matrix(2, 4, std::span<const float>(input_values));
}

EngineModel backend_cached_model_blocks_fixture_model() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 2;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    add_tiny_llama_block_tensors(tensors, 1, 0.5F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

TensorStorage backend_cached_model_blocks_first_input() {
    const std::array<float, 2> input_values {3.0F, 4.0F};
    return tensor_f32_matrix(1, 2, std::span<const float>(input_values));
}

TensorStorage backend_cached_model_blocks_second_input() {
    const std::array<float, 2> input_values {30.0F, 40.0F};
    return tensor_f32_matrix(1, 2, std::span<const float>(input_values));
}

void expect_model_attention_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelAttentionBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model attention backend weight block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        const auto& expected = weights.blocks[block_index].attention;
        const auto& actual = backend_weights.blocks[block_index];
        expect(actual.attention_norm_weight_buffer.backend() == backend, "Llama model attention norm backend mismatch");
        expect(actual.query_weight.backend() == backend, "Llama model attention query backend mismatch");
        expect(actual.key_weight.backend() == backend, "Llama model attention key backend mismatch");
        expect(actual.value_weight.backend() == backend, "Llama model attention value backend mismatch");
        expect(actual.output_weight.backend() == backend, "Llama model attention output backend mismatch");
        expect(
            actual.attention_norm_weight_buffer.descriptor() == expected.attention_norm_weight.descriptor(),
            "Llama model attention norm backend descriptor mismatch"
        );
        expect(
            actual.query_weight.descriptor() == expected.query_weight.descriptor(),
            "Llama model attention query backend descriptor mismatch"
        );
        expect(
            actual.key_weight.descriptor() == expected.key_weight.descriptor(),
            "Llama model attention key backend descriptor mismatch"
        );
        expect(
            actual.value_weight.descriptor() == expected.value_weight.descriptor(),
            "Llama model attention value backend descriptor mismatch"
        );
        expect(
            actual.output_weight.descriptor() == expected.output_weight.descriptor(),
            "Llama model attention output backend descriptor mismatch"
        );
    }
}

void expect_model_feed_forward_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelFeedForwardBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model FFN backend weight block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        const auto& expected = weights.blocks[block_index].feed_forward;
        const auto& actual = backend_weights.blocks[block_index];
        expect(actual.feed_forward_norm_weight_buffer.backend() == backend, "Llama model FFN norm backend mismatch");
        expect(actual.gate_weight.backend() == backend, "Llama model FFN gate backend mismatch");
        expect(actual.up_weight.backend() == backend, "Llama model FFN up backend mismatch");
        expect(actual.down_weight.backend() == backend, "Llama model FFN down backend mismatch");
        expect(
            actual.feed_forward_norm_weight_buffer.descriptor() == expected.feed_forward_norm_weight.descriptor(),
            "Llama model FFN norm backend descriptor mismatch"
        );
        expect(
            actual.gate_weight.descriptor() == expected.gate_weight.descriptor(),
            "Llama model FFN gate backend descriptor mismatch"
        );
        expect(
            actual.up_weight.descriptor() == expected.up_weight.descriptor(),
            "Llama model FFN up backend descriptor mismatch"
        );
        expect(
            actual.down_weight.descriptor() == expected.down_weight.descriptor(),
            "Llama model FFN down backend descriptor mismatch"
        );
    }
}

void expect_block_backend_weights(
    const LlamaBlockWeights& expected,
    const LlamaBlockBackendWeights& actual,
    BackendKind backend
) {
    expect(actual.attention.attention_norm_weight_buffer.backend() == backend, "Llama block attention norm backend mismatch");
    expect(actual.attention.query_weight.backend() == backend, "Llama block attention query backend mismatch");
    expect(actual.attention.key_weight.backend() == backend, "Llama block attention key backend mismatch");
    expect(actual.attention.value_weight.backend() == backend, "Llama block attention value backend mismatch");
    expect(actual.attention.output_weight.backend() == backend, "Llama block attention output backend mismatch");
    expect(actual.feed_forward.feed_forward_norm_weight_buffer.backend() == backend, "Llama block FFN norm backend mismatch");
    expect(actual.feed_forward.gate_weight.backend() == backend, "Llama block FFN gate backend mismatch");
    expect(actual.feed_forward.up_weight.backend() == backend, "Llama block FFN up backend mismatch");
    expect(actual.feed_forward.down_weight.backend() == backend, "Llama block FFN down backend mismatch");
    expect(
        actual.attention.attention_norm_weight_buffer.descriptor()
            == expected.attention.attention_norm_weight.descriptor(),
        "Llama block attention norm backend descriptor mismatch"
    );
    expect(
        actual.attention.query_weight.descriptor() == expected.attention.query_weight.descriptor(),
        "Llama block attention query backend descriptor mismatch"
    );
    expect(
        actual.attention.key_weight.descriptor() == expected.attention.key_weight.descriptor(),
        "Llama block attention key backend descriptor mismatch"
    );
    expect(
        actual.attention.value_weight.descriptor() == expected.attention.value_weight.descriptor(),
        "Llama block attention value backend descriptor mismatch"
    );
    expect(
        actual.attention.output_weight.descriptor() == expected.attention.output_weight.descriptor(),
        "Llama block attention output backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.feed_forward_norm_weight_buffer.descriptor()
            == expected.feed_forward.feed_forward_norm_weight.descriptor(),
        "Llama block FFN norm backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.gate_weight.descriptor() == expected.feed_forward.gate_weight.descriptor(),
        "Llama block FFN gate backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.up_weight.descriptor() == expected.feed_forward.up_weight.descriptor(),
        "Llama block FFN up backend descriptor mismatch"
    );
    expect(
        actual.feed_forward.down_weight.descriptor() == expected.feed_forward.down_weight.descriptor(),
        "Llama block FFN down backend descriptor mismatch"
    );
}

void expect_model_block_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelBlockBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model block backend weight block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        expect_block_backend_weights(weights.blocks[block_index], backend_weights.blocks[block_index], backend);
    }
}

void expect_model_logits_backend_weights(
    const LlamaModelWeights& weights,
    const LlamaModelLogitsBackendWeights& backend_weights,
    BackendKind backend
) {
    expect(
        &backend_weights.token_embedding_weight == &weights.token_embedding_weight,
        "Llama model logits backend should borrow token embedding weight"
    );
    expect(
        &backend_weights.output_norm_weight == &weights.output_norm_weight,
        "Llama model logits backend should borrow output norm weight"
    );
    expect(backend_weights.output_weight.backend() == backend, "Llama model logits output backend mismatch");
    expect(
        backend_weights.output_weight.descriptor() == weights.output_weight.descriptor(),
        "Llama model logits output backend descriptor mismatch"
    );
    expect(
        backend_weights.blocks.size() == weights.blocks.size(),
        "Llama model logits backend block count mismatch"
    );
    for (std::size_t block_index = 0; block_index < weights.blocks.size(); ++block_index) {
        expect_block_backend_weights(weights.blocks[block_index], backend_weights.blocks[block_index], backend);
    }
}

void test_runs_llama_attention_layer_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_attention_layer_weights_to_backend(
        weights.blocks[0].attention,
        BackendKind::cpu
    );
    const auto input = backend_feed_forward_fixture_input();

    const auto expected = run_llama_attention_layer_f32(input, weights.blocks[0].attention, model.llama_config(), 1);
    const auto output = run_llama_attention_layer_f32(input, backend_weights, model.llama_config(), 1);

    expect(backend_weights.query_weight.backend() == BackendKind::cpu, "Llama attention CPU query buffer backend mismatch");
    expect(backend_weights.key_weight.backend() == BackendKind::cpu, "Llama attention CPU key buffer backend mismatch");
    expect(backend_weights.value_weight.backend() == BackendKind::cpu, "Llama attention CPU value buffer backend mismatch");
    expect(backend_weights.output_weight.backend() == BackendKind::cpu, "Llama attention CPU output buffer backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "Llama attention CPU backend weight buffer output");
}

void test_runs_llama_attention_layer_with_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_attention_layer_weights_to_backend(
        weights.blocks[0].attention,
        BackendKind::cpu
    );
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        weights.blocks[0].attention,
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        weights.blocks[0].attention,
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect(backend_cache.layers[0].filled_token_count == 2, "Llama CPU backend cached attention fill mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-5F, "Llama CPU backend cached attention output");
    expect_f32_matrix_close(
        backend_cache.layers[0].key,
        expected_cache.layers[0].key,
        1.0e-5F,
        "Llama CPU backend cached attention key cache"
    );
    expect_f32_matrix_close(
        backend_cache.layers[0].value,
        expected_cache.layers[0].value,
        1.0e-5F,
        "Llama CPU backend cached attention value cache"
    );
}

void test_runs_llama_feed_forward_layer_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_feed_forward_layer_weights_to_backend(
        weights.blocks[0].feed_forward,
        BackendKind::cpu
    );
    const auto input = backend_feed_forward_fixture_input();

    const auto expected = run_llama_feed_forward_layer_f32(
        input,
        weights.blocks[0].feed_forward,
        model.llama_config()
    );
    const auto output = run_llama_feed_forward_layer_f32(input, backend_weights, model.llama_config());

    expect(backend_weights.gate_weight.backend() == BackendKind::cpu, "Llama FFN CPU gate buffer backend mismatch");
    expect(backend_weights.up_weight.backend() == BackendKind::cpu, "Llama FFN CPU up buffer backend mismatch");
    expect(backend_weights.down_weight.backend() == BackendKind::cpu, "Llama FFN CPU down buffer backend mismatch");
    expect_f32_matrix_close(output, expected, 0.0F, "Llama FFN CPU backend weight buffer output");
}

void test_runs_llama_block_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();

    const auto expected = run_llama_block_f32(input, weights.blocks[0], model.llama_config(), 1);
    const auto output = run_llama_block_f32(input, backend_weights, model.llama_config(), 1);

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama block CPU backend weight buffer output");
}

void test_runs_llama_block_with_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::cpu);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama CPU backend cached block output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 1.0e-5F, "Llama CPU backend cached block cache");
}

void test_matches_llama_block_with_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::metal);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::metal);
    expect_f32_matrix_close(output, expected, 2.0e-2F, "Llama Metal backend cached block output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 1.0e-2F, "Llama Metal backend cached block cache");
}

void test_runs_llama_block_with_backend_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::cpu);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::cpu);

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::cpu);
    expect(backend_cache.backend == BackendKind::cpu, "Llama block backend KV cache CPU backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama CPU backend resident-KV cached block output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        1.0e-5F,
        "Llama CPU backend resident-KV cached block cache"
    );
}

void test_matches_llama_block_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_block_weights_to_backend(weights.blocks[0], BackendKind::metal);
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::metal);

    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        weights.blocks[0],
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_block_with_kv_cache_f32(
        second_input,
        weights.blocks[0],
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_block_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_block_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect_block_backend_weights(weights.blocks[0], backend_weights, BackendKind::metal);
    expect(backend_cache.backend == BackendKind::metal, "Llama block backend KV cache Metal backend mismatch");
    expect_f32_matrix_close(output, expected, 5.0e-2F, "Llama Metal backend resident-KV cached block output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        2.0e-2F,
        "Llama Metal backend resident-KV cached block cache"
    );
}

void test_binds_and_runs_llama_model_blocks_from_named_tensors() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 2;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    add_tiny_llama_block_tensors(tensors, 1, 0.5F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);

    expect(weights.blocks.size() == 2, "bound Llama model block count mismatch");
    const auto* block_one_down = model.find_tensor("blk.1.ffn_down.weight");
    expect(block_one_down != nullptr, "bound Llama model test tensor lookup failed");
    expect(
        &weights.blocks[1].feed_forward.down_weight == &block_one_down->storage,
        "bound Llama model should borrow block 1 down weight"
    );

    const std::array<float, 2> input_values {3.0F, 4.0F};
    const auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    const auto output = run_llama_model_blocks_f32(input, weights, model.llama_config(), 0);
    const auto first_block = run_llama_block_f32(input, weights.blocks[0], model.llama_config(), 0);
    const auto expected = run_llama_block_f32(first_block, weights.blocks[1], model.llama_config(), 0);

    expect(output.descriptor().shape == Shape::from({1, 2}), "bound Llama model output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "bound Llama model output dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_0)),
        read_f32(expected, std::span<const std::uint64_t>(index_0)),
        0.000001F,
        "bound Llama model output value 0 mismatch"
    );
    expect_close(
        read_f32(output, std::span<const std::uint64_t>(index_1)),
        read_f32(expected, std::span<const std::uint64_t>(index_1)),
        0.000001F,
        "bound Llama model output value 1 mismatch"
    );
}

void test_copies_llama_model_attention_weights_to_cpu_backend_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_attention_weights_to_backend(weights, BackendKind::cpu);

    expect_model_attention_backend_weights(weights, backend_weights, BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_attention_layer_f32(
        input,
        weights.blocks[0].attention,
        model.llama_config(),
        1
    );
    const auto output = run_llama_attention_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config(),
        1
    );
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU attention backend weight output");
}

void test_matches_llama_attention_layer_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_attention_weights_to_backend(weights, BackendKind::metal);

    expect_model_attention_backend_weights(weights, backend_weights, BackendKind::metal);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_attention_layer_f32(
        input,
        weights.blocks[0].attention,
        model.llama_config(),
        1
    );
    const auto output = run_llama_attention_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config(),
        1
    );
    expect_f32_matrix_close(output, expected, 2.0e-3F, "Llama model Metal attention backend weight output");
}

void test_matches_llama_attention_layer_with_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_attention_layer_weights_to_backend(
        weights.blocks[0].attention,
        BackendKind::metal
    );
    const std::array<float, 4> first_input_values {
        0.5F, -1.0F, 2.0F, 0.25F,
    };
    const std::array<float, 4> second_input_values {
        -0.75F, 1.5F, -0.5F, 2.25F,
    };
    const auto first_input = tensor_f32_matrix(1, 4, std::span<const float>(first_input_values));
    const auto second_input = tensor_f32_matrix(1, 4, std::span<const float>(second_input_values));
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        weights.blocks[0].attention,
        model.llama_config(),
        0,
        expected_cache.layers[0]
    );
    const auto expected = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        weights.blocks[0].attention,
        model.llama_config(),
        1,
        expected_cache.layers[0]
    );
    (void)run_llama_attention_layer_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache.layers[0]
    );
    const auto output = run_llama_attention_layer_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache.layers[0]
    );

    expect(backend_cache.layers[0].filled_token_count == 2, "Llama Metal backend cached attention fill mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-2F, "Llama Metal backend cached attention output");
    expect_f32_matrix_close(
        backend_cache.layers[0].key,
        expected_cache.layers[0].key,
        1.0e-2F,
        "Llama Metal backend cached attention key cache"
    );
    expect_f32_matrix_close(
        backend_cache.layers[0].value,
        expected_cache.layers[0].value,
        1.0e-2F,
        "Llama Metal backend cached attention value cache"
    );
}

void test_reports_unavailable_llama_model_attention_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_attention_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal attention backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal attention backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal attention backend weight copy backend mismatch");
    }
}

void test_copies_llama_model_feed_forward_weights_to_cpu_backend_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_feed_forward_weights_to_backend(weights, BackendKind::cpu);

    expect_model_feed_forward_backend_weights(weights, backend_weights, BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_feed_forward_layer_f32(
        input,
        weights.blocks[0].feed_forward,
        model.llama_config()
    );
    const auto output = run_llama_feed_forward_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config()
    );
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU FFN backend weight output");
}

void test_matches_llama_feed_forward_layer_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_feed_forward_weights_to_backend(weights, BackendKind::metal);

    expect_model_feed_forward_backend_weights(weights, backend_weights, BackendKind::metal);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_feed_forward_layer_f32(
        input,
        weights.blocks[0].feed_forward,
        model.llama_config()
    );
    const auto output = run_llama_feed_forward_layer_f32(
        input,
        backend_weights.blocks[0],
        model.llama_config()
    );
    expect_f32_matrix_close(output, expected, 1.0e-3F, "Llama model Metal FFN backend weight output");
}

void test_copies_llama_model_block_weights_to_cpu_backend_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::cpu);

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::cpu);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_model_blocks_f32(input, weights, model.llama_config(), 1);
    const auto output = run_llama_model_blocks_f32(input, backend_weights, model.llama_config(), 1);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU block backend weight output");
}

void test_matches_llama_model_blocks_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::metal);
    const auto input = backend_feed_forward_fixture_input();
    const auto expected = run_llama_model_blocks_f32(input, weights, model.llama_config(), 1);
    const auto output = run_llama_model_blocks_f32(input, backend_weights, model.llama_config(), 1);
    expect_f32_matrix_close(output, expected, 4.0e-3F, "Llama model Metal block backend weight output");
}

void test_runs_llama_model_blocks_with_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::cpu);
    const auto first_input = backend_cached_model_blocks_first_input();
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        0,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama model CPU backend cached blocks output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 1.0e-5F, "Llama model CPU backend cached blocks cache");
}

void test_matches_llama_model_blocks_with_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
    const auto first_input = backend_cached_model_blocks_first_input();
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_kv_cache(model.llama_config());

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        0,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::metal);
    expect_f32_matrix_close(output, expected, 5.0e-2F, "Llama model Metal backend cached blocks output");
    expect_llama_kv_cache_close(backend_cache, expected_cache, 2.0e-2F, "Llama model Metal backend cached blocks cache");
}

void test_runs_llama_model_blocks_with_backend_kv_cache_and_cpu_backend_weight_buffers() {
    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::cpu);
    const auto first_input = backend_cached_model_blocks_first_input();
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::cpu);

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        0,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        0,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect(backend_cache.backend == BackendKind::cpu, "Llama backend KV cache CPU backend mismatch");
    expect_f32_matrix_close(output, expected, 1.0e-4F, "Llama model CPU backend resident-KV cached blocks output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        1.0e-5F,
        "Llama model CPU backend resident-KV cached blocks cache"
    );
}

void test_matches_llama_model_blocks_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
    const std::array<float, 4> prefill_input_values {
        3.0F, 4.0F,
        5.0F, 6.0F,
    };
    const auto first_input = tensor_f32_matrix(2, 2, std::span<const float>(prefill_input_values));
    const auto second_input = backend_cached_model_blocks_second_input();
    auto expected_cache = create_llama_kv_cache(model.llama_config());
    auto backend_cache = create_llama_backend_kv_cache(model.llama_config(), BackendKind::metal);

    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        weights,
        model.llama_config(),
        1,
        expected_cache
    );
    const auto expected = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        weights,
        model.llama_config(),
        2,
        expected_cache
    );
    (void)run_llama_model_blocks_with_kv_cache_f32(
        first_input,
        backend_weights,
        model.llama_config(),
        1,
        backend_cache
    );
    const auto output = run_llama_model_blocks_with_kv_cache_f32(
        second_input,
        backend_weights,
        model.llama_config(),
        2,
        backend_cache
    );

    expect_model_block_backend_weights(weights, backend_weights, BackendKind::metal);
    expect(backend_cache.backend == BackendKind::metal, "Llama backend KV cache Metal backend mismatch");
    expect_f32_matrix_close(output, expected, 5.0e-2F, "Llama model Metal backend resident-KV cached blocks output");
    expect_llama_kv_cache_close(
        copy_llama_backend_kv_cache_to_cpu(backend_cache),
        expected_cache,
        2.0e-2F,
        "Llama model Metal backend resident-KV cached blocks cache"
    );
}

void test_reports_unavailable_llama_backend_kv_cache_creation_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    try {
        (void)create_llama_backend_kv_cache(backend_cached_model_blocks_fixture_model().llama_config(), BackendKind::metal);
        throw TestFailure("Llama backend KV cache creation should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama backend KV cache creation unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama backend KV cache creation backend mismatch");
    }
}

void test_reports_unavailable_llama_model_blocks_with_kv_cache_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_cached_model_blocks_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model cached-block Metal backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model cached-block Metal backend weight copy unavailable error code mismatch"
        );
        expect(
            error.backend() == BackendKind::metal,
            "Llama model cached-block Metal backend weight copy backend mismatch"
        );
    }
}

void test_reports_unavailable_llama_model_block_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_block_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal block backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal block backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal block backend weight copy backend mismatch");
    }
}

void test_projects_llama_logits_with_cpu_backend_output_weight_buffer() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const auto hidden_states = backend_feed_forward_fixture_input();

    const auto expected = project_llama_logits_f32(hidden_states, weights, model.llama_config());
    const auto output = project_llama_logits_f32(hidden_states, backend_weights, model.llama_config());

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama CPU logits backend output projection");
}

void test_runs_llama_model_logits_with_cpu_backend_weight_buffers() {
    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const std::array<std::uint32_t, 2> token_ids {0, 3};

    const auto expected = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        1
    );
    const auto output = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        1
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect_f32_matrix_close(output, expected, 0.0F, "Llama model CPU logits backend output");
}

void test_matches_llama_model_logits_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
    const std::array<std::uint32_t, 2> token_ids {0, 3};

    const auto expected = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        1
    );
    const auto output = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        1
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::metal);
    expect_f32_matrix_close(output, expected, 1.0e-2F, "Llama model Metal logits backend output");
}

void test_reports_unavailable_llama_model_logits_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal logits backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal logits backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal logits backend weight copy backend mismatch");
    }
}

void test_reports_unavailable_llama_model_feed_forward_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto model = backend_feed_forward_fixture_model();
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_feed_forward_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama model Metal FFN backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama model Metal FFN backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama model Metal FFN backend weight copy backend mismatch");
    }
}

void test_rejects_llama_model_weight_binding_missing_tensor() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 1;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, false);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));

    try {
        (void)bind_llama_model_weights(model);
        throw TestFailure("Llama model weight binding should reject missing named tensors");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_tensor, "Llama model missing tensor error mismatch");
        expect(error.detail() == "blk.0.ffn_down.weight", "Llama model missing tensor detail mismatch");
    }
}

void test_rejects_llama_model_blocks_projection_dtype_mismatch() {
    auto config = tiny_feed_forward_layer_config();
    config.block_count = 1;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, true, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);

    const std::array<float, 2> input_values {3.0F, 4.0F};
    const auto input = tensor_f32_matrix(1, 2, std::span<const float>(input_values));
    try {
        (void)run_llama_model_blocks_f32(input, weights, model.llama_config(), 0);
        throw TestFailure("Llama model block execution should reject F32 query projection weight");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::unsupported_operator_dtype,
            "Llama model projection dtype error mismatch"
        );
        expect(error.expected_value() == 1, "Llama model expected projection dtype mismatch");
        expect(error.actual_value() == 0, "Llama model actual projection dtype mismatch");
    }
}

LlamaModelConfig tiny_logits_config() {
    auto config = tiny_feed_forward_layer_config();
    config.vocab_size = 3;
    return config;
}

void test_embeds_llama_token_ids_from_f16_weight() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };
    const std::array<std::uint32_t, 2> token_ids {2, 0};

    const auto output = embed_llama_token_ids_f32(std::span<const std::uint32_t>(token_ids), weights, tiny_logits_config());

    expect(output.descriptor().shape == Shape::from({2, 2}), "Llama token embedding output shape mismatch");
    expect(output.descriptor().dtype == DType::f32, "Llama token embedding output dtype mismatch");
    const std::array<std::uint64_t, 2> index_00 {0, 0};
    const std::array<std::uint64_t, 2> index_01 {0, 1};
    const std::array<std::uint64_t, 2> index_10 {1, 0};
    const std::array<std::uint64_t, 2> index_11 {1, 1};
    expect(read_f32(output, std::span<const std::uint64_t>(index_00)) == 3.0F, "Llama token embedding value 0,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_01)) == 30.0F, "Llama token embedding value 0,1 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_10)) == 1.0F, "Llama token embedding value 1,0 mismatch");
    expect(read_f32(output, std::span<const std::uint64_t>(index_11)) == 10.0F, "Llama token embedding value 1,1 mismatch");
}

void test_rejects_llama_token_embedding_id_out_of_range() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };
    const std::array<std::uint32_t, 1> token_ids {3};

    try {
        (void)embed_llama_token_ids_f32(std::span<const std::uint32_t>(token_ids), weights, tiny_logits_config());
        throw TestFailure("Llama token embedding should reject token id above vocab");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::tensor_index_out_of_range, "Llama token id error mismatch");
        expect(error.expected_value() == 3, "Llama token id expected vocab mismatch");
        expect(error.actual_value() == 3, "Llama token id actual mismatch");
    }
}

void test_projects_llama_logits_with_final_norm() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const std::array<float, 2> hidden_values {3.0F, 4.0F};
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values));
    const auto hidden_states = tensor_f32_matrix(1, 2, std::span<const float>(hidden_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };

    const auto logits = project_llama_logits_f32(hidden_states, weights, tiny_logits_config());

    expect(logits.descriptor().shape == Shape::from({1, 3}), "Llama logits shape mismatch");
    expect(logits.descriptor().dtype == DType::f32, "Llama logits dtype mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    const std::array<std::uint64_t, 2> index_2 {0, 2};
    expect_close(read_f32(logits, std::span<const std::uint64_t>(index_0)), 0.75F, 0.000001F, "Llama logits value 0 mismatch");
    expect_close(read_f32(logits, std::span<const std::uint64_t>(index_1)), 1.0F, 0.000001F, "Llama logits value 1 mismatch");
    expect_close(read_f32(logits, std::span<const std::uint64_t>(index_2)), 4.5F, 0.000001F, "Llama logits value 2 mismatch");
}

void test_runs_llama_model_logits_from_token_ids() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto logits = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );
    const auto embeddings = embed_llama_token_ids_f32(std::span<const std::uint32_t>(token_ids), weights, model.llama_config());
    const auto hidden_states = run_llama_model_blocks_f32(embeddings, weights, model.llama_config(), 0);
    const auto expected = project_llama_logits_f32(hidden_states, weights, model.llama_config());

    expect(logits.descriptor().shape == Shape::from({1, 3}), "Llama model logits shape mismatch");
    const std::array<std::uint64_t, 2> index_0 {0, 0};
    const std::array<std::uint64_t, 2> index_1 {0, 1};
    const std::array<std::uint64_t, 2> index_2 {0, 2};
    expect_close(
        read_f32(logits, std::span<const std::uint64_t>(index_0)),
        read_f32(expected, std::span<const std::uint64_t>(index_0)),
        0.000001F,
        "Llama model logits value 0 mismatch"
    );
    expect_close(
        read_f32(logits, std::span<const std::uint64_t>(index_1)),
        read_f32(expected, std::span<const std::uint64_t>(index_1)),
        0.000001F,
        "Llama model logits value 1 mismatch"
    );
    expect_close(
        read_f32(logits, std::span<const std::uint64_t>(index_2)),
        read_f32(expected, std::span<const std::uint64_t>(index_2)),
        0.000001F,
        "Llama model logits value 2 mismatch"
    );
}

void test_runs_llama_model_logits_with_kv_cache_matches_no_cache() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.vocab_size = 8;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 2> full_token_ids {0, 1};
    const std::array<std::uint32_t, 1> first_token_id {0};
    const std::array<std::uint32_t, 1> second_token_id {1};
    auto cache = create_llama_kv_cache(model.llama_config());

    const auto no_cache = run_llama_model_logits_f32(
        std::span<const std::uint32_t>(full_token_ids),
        weights,
        model.llama_config(),
        1
    );
    (void)run_llama_model_logits_with_kv_cache_f32(
        std::span<const std::uint32_t>(first_token_id),
        weights,
        model.llama_config(),
        0,
        cache
    );
    const auto cached = run_llama_model_logits_with_kv_cache_f32(
        std::span<const std::uint32_t>(second_token_id),
        weights,
        model.llama_config(),
        1,
        cache
    );

    expect(cache.layers[0].filled_token_count == 2, "Llama model KV cache fill mismatch");
    expect(cached.descriptor().shape == Shape::from({1, 8}), "Llama cached logits shape mismatch");
    for (std::uint64_t token = 0; token < 8; ++token) {
        const std::array<std::uint64_t, 2> no_cache_index {1, token};
        const std::array<std::uint64_t, 2> cached_index {0, token};
        expect_close(
            read_f32(cached, std::span<const std::uint64_t>(cached_index)),
            read_f32(no_cache, std::span<const std::uint64_t>(no_cache_index)),
            0.00001F,
            "Llama cached logits should match no-cache final row"
        );
    }
}

void test_rejects_llama_model_kv_cache_layer_count_mismatch() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.vocab_size = 8;
    std::vector<EngineTensor> tensors;
    add_tiny_llama_top_level_tensors(tensors);
    add_tiny_llama_block_tensors(tensors, 0, 1.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> token_ids {0};
    LlamaKVCache cache {
        .layers = {},
        .context_length = 0,
        .key_width = 0,
        .value_width = 0,
    };

    try {
        (void)run_llama_model_logits_with_kv_cache_f32(
            std::span<const std::uint32_t>(token_ids),
            weights,
            model.llama_config(),
            0,
            cache
        );
        throw TestFailure("Llama model cached logits should reject cache layer count mismatch");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama KV cache layer count must match model block count",
            "Llama KV cache layer count error text mismatch"
        );
    }
}

void test_rejects_llama_logits_output_weight_dtype_mismatch() {
    const std::array<float, 6> embedding_values {
        1.0F, 2.0F, 3.0F,
        10.0F, 20.0F, 30.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    const std::array<float, 2> hidden_values {3.0F, 4.0F};
    const auto token_embedding_weight = tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values));
    const auto output_norm_weight = tensor_f32_vector(std::span<const float>(norm_values));
    const auto output_weight = tensor_f32_matrix(2, 3, std::span<const float>(output_values));
    const auto hidden_states = tensor_f32_matrix(1, 2, std::span<const float>(hidden_values));
    const LlamaModelWeights weights {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = output_norm_weight,
        .output_weight = output_weight,
        .blocks = {},
    };

    try {
        (void)project_llama_logits_f32(hidden_states, weights, tiny_logits_config());
        throw TestFailure("Llama logits projection should reject F32 output weight");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "Llama logits dtype error mismatch");
        expect(error.expected_value() == 1, "Llama logits expected output dtype mismatch");
        expect(error.actual_value() == 0, "Llama logits actual output dtype mismatch");
    }
}

void test_samples_argmax_token_from_f32_logits_vector() {
    const std::array<float, 4> logits_values {0.5F, 3.0F, 3.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));

    const auto token = sample_argmax_token_f32(logits);

    expect(token == 1, "argmax sampler should pick the first max token");
}

void test_samples_argmax_token_from_last_logits_row() {
    const std::array<float, 6> logits_values {
        0.0F, 1.0F, 9.0F,
        4.0F, 3.0F, 4.0F,
    };
    const auto logits = tensor_f32_matrix(2, 3, std::span<const float>(logits_values));

    const auto token = sample_argmax_token_f32(logits);

    expect(token == 0, "argmax sampler should use the last logits row and first-max tie break");
}

void test_selects_top_logit_tokens_from_last_row() {
    const std::array<float, 8> logits_values {
        9.0F, 0.0F, 1.0F, 2.0F,
        4.0F, 5.0F, 5.0F, 1.0F,
    };
    const auto logits = tensor_f32_matrix(2, 4, std::span<const float>(logits_values));

    const auto top_logits = top_logit_tokens_f32(logits, 3);

    expect(top_logits.size() == 3, "top logits count mismatch");
    expect(top_logits[0].token == 1, "top logits first token mismatch");
    expect(top_logits[1].token == 2, "top logits tie should prefer lower token id");
    expect(top_logits[2].token == 0, "top logits third token mismatch");
    expect_close(top_logits[0].logit, 5.0F, 0.000001F, "top logits first value mismatch");
}

void test_rejects_argmax_sampler_dtype_mismatch() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f16_gguf_matrix(1, 3, std::span<const float>(logits_values));

    try {
        (void)sample_argmax_token_f32(logits);
        throw TestFailure("argmax sampler should reject F16 logits");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "argmax sampler dtype error mismatch");
        expect(error.expected_value() == 0, "argmax sampler expected dtype mismatch");
        expect(error.actual_value() == 1, "argmax sampler actual dtype mismatch");
    }
}

void test_rejects_argmax_sampler_rank_above_two() {
    const auto logits = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 1, 2}), DType::f32));

    try {
        (void)sample_argmax_token_f32(logits);
        throw TestFailure("argmax sampler should reject logits rank above two");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "argmax sampler rank error mismatch");
        expect(error.expected_value() == 2, "argmax sampler expected rank mismatch");
        expect(error.actual_value() == 3, "argmax sampler actual rank mismatch");
    }
}

void test_samples_temperature_top_k_token_from_f32_logits_vector() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState first_state(sampling.seed);
    SamplingState second_state(sampling.seed);

    const auto first_token = sample_token_f32(logits, sampling, first_state);
    const auto second_token = sample_token_f32(logits, sampling, second_state);

    expect(first_token == 1, "temperature top-k sampler should draw the second top-k candidate for seed 0");
    expect(second_token == first_token, "temperature top-k sampler should be deterministic for the same seed");
}

void test_samples_temperature_top_k_token_from_last_logits_row() {
    const std::array<float, 6> logits_values {
        100.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 2.0F,
    };
    const auto logits = tensor_f32_matrix(2, 3, std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state);

    expect(token == 1, "temperature top-k sampler should use the final logits row");
}

void test_samples_temperature_top_p_token_from_f32_logits_vector() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.5F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state);

    expect(token == 2, "temperature top-p sampler should keep at least the highest-probability token");
}

void test_samples_temperature_repetition_penalty_token_from_f32_logits_vector() {
    const std::array<float, 3> logits_values {0.0F, 4.0F, 5.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const std::array<std::uint32_t, 1> repetition_tokens {2};
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.01F,
        .repetition_penalty = 2.0F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state, std::span<const std::uint32_t>(repetition_tokens));

    expect(token == 1, "repetition penalty should lower a repeated positive-logit token before top-p filtering");
}

void test_sampling_temperature_zero_keeps_argmax_path() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 0.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state);

    expect(token == 2, "zero-temperature sampler should preserve argmax behavior");
}

void test_rejects_sampling_negative_temperature() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = -1.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    try {
        (void)sample_token_f32(logits, sampling, state);
        throw TestFailure("temperature sampler should reject negative temperature");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling temperature must be zero or a positive finite value",
            "temperature sampler error text mismatch"
        );
    }
}

void test_rejects_sampling_invalid_top_p() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.0F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    try {
        (void)sample_token_f32(logits, sampling, state);
        throw TestFailure("temperature sampler should reject invalid top-p");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling top-p must be in (0, 1]",
            "top-p sampler error text mismatch"
        );
    }
}

void test_rejects_sampling_invalid_repetition_penalty() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .repetition_penalty = 0.5F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    try {
        (void)sample_token_f32(logits, sampling, state);
        throw TestFailure("temperature sampler should reject invalid repetition penalty");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling repetition penalty must be at least 1",
            "repetition-penalty sampler error text mismatch"
        );
    }
}

EngineModel tiny_argmax_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

EngineModel tiny_tied_output_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

EngineModel tiny_eos_then_visible_generation_model(LlamaModelConfig config) {
    const std::array<float, 6> embedding_values {
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        0.0F, 2.0F, 0.0F,
        0.0F, 0.0F, 2.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

EngineModel tiny_chat_template_generation_model(LlamaModelConfig config) {
    std::vector<float> embedding_values(static_cast<std::size_t>(config.embedding_length) * config.vocab_size, 0.0F);
    std::vector<float> output_values(static_cast<std::size_t>(config.embedding_length) * config.vocab_size, 0.0F);
    for (std::uint32_t token = 0; token < config.vocab_size; ++token) {
        embedding_values[token] = 1.0F;
    }
    output_values[7] = 2.0F;

    const std::array<float, 2> norm_values {1.0F, 1.0F};
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(config.embedding_length, config.vocab_size, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(config.embedding_length, config.vocab_size, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    return tiny_bound_llama_model(config, std::move(tensors));
}

std::vector<std::uint8_t> tiny_generate_cli_gguf_bytes() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

std::vector<std::uint8_t> tiny_tied_output_generate_cli_gguf_bytes() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_tied_output_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

std::vector<std::uint8_t> tiny_eos_generate_cli_gguf_bytes() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_eos_then_visible_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_eos_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

std::vector<std::uint8_t> tiny_chat_template_generate_cli_gguf_bytes() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 16;
    config.vocab_size = 8;
    const auto model = tiny_chat_template_generation_model(config);
    return write_tiny_gguf_bytes(
        tiny_generate_cli_chat_metadata(config),
        tiny_gguf_tensors_from_engine_model(model)
    );
}

void test_binds_llama_model_missing_output_weight_to_token_embedding() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_tied_output_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto* token_embedding = model.find_tensor("token_embd.weight");

    expect(token_embedding != nullptr, "tied-output Llama model should have token embedding");
    expect(model.find_tensor("output.weight") == nullptr, "tied-output Llama model should omit output weight");
    expect(
        &weights.output_weight == &token_embedding->storage,
        "tied-output Llama model should reuse token embedding as output weight"
    );

    const std::array<std::uint32_t, 1> prompt_tokens {0};
    const auto tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );

    expect(tokens.size() == 3, "tied-output Llama generation token count mismatch");
    expect(tokens[0] == 0, "tied-output Llama generation prompt token mismatch");
    expect(tokens[1] == 2, "tied-output Llama generation first token mismatch");
    expect(tokens[2] == 2, "tied-output Llama generation second token mismatch");
}

void test_runs_llama_next_token_argmax_from_token_ids() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto token = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );

    expect(token == 2, "Llama next-token argmax mismatch");
}

void test_rejects_llama_next_token_empty_token_ids() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 0> token_ids {};

    try {
        (void)run_llama_model_next_token_argmax_f32(
            std::span<const std::uint32_t>(token_ids),
            weights,
            model.llama_config(),
            0
        );
        throw TestFailure("Llama next-token helper should reject empty token IDs");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama token id span must not be empty",
            "Llama next-token empty input error mismatch"
        );
    }
}

void test_runs_llama_next_token_argmax_with_cpu_backend_weight_buffers() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto expected = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );
    const auto token = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect(token == expected, "Llama CPU backend next-token argmax should match CPU reference");
    expect(token == 2, "Llama CPU backend next-token argmax token mismatch");
}

void test_runs_llama_next_token_sampled_with_cpu_backend_weight_buffers() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::cpu);
    const std::array<std::uint32_t, 1> token_ids {0};
    const SamplingConfig sampling {
        .temperature = 10.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState expected_state(sampling.seed);
    SamplingState backend_state(sampling.seed);

    const auto expected = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0,
        sampling,
        expected_state,
        std::span<const std::uint32_t>(token_ids)
    );
    const auto token = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0,
        sampling,
        backend_state,
        std::span<const std::uint32_t>(token_ids)
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::cpu);
    expect(token == expected, "Llama CPU backend next-token sampled output should match CPU reference");
    expect(token == 1, "Llama CPU backend next-token sampled token mismatch");
}

void test_matches_llama_next_token_argmax_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
    const std::array<std::uint32_t, 1> token_ids {0};

    const auto expected = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0
    );
    const auto token = run_llama_model_next_token_argmax_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::metal);
    expect(token == expected, "Llama Metal backend next-token argmax should match CPU reference");
}

void test_matches_llama_next_token_sampled_with_metal_backend_weight_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const auto backend_weights = copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
    const std::array<std::uint32_t, 1> token_ids {0};
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.01F,
        .seed = 17,
    };
    SamplingState expected_state(sampling.seed);
    SamplingState backend_state(sampling.seed);

    const auto expected = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        weights,
        model.llama_config(),
        0,
        sampling,
        expected_state,
        std::span<const std::uint32_t>(token_ids)
    );
    const auto token = run_llama_model_next_token_sampled_f32(
        std::span<const std::uint32_t>(token_ids),
        backend_weights,
        model.llama_config(),
        0,
        sampling,
        backend_state,
        std::span<const std::uint32_t>(token_ids)
    );

    expect_model_logits_backend_weights(weights, backend_weights, BackendKind::metal);
    expect(token == expected, "Llama Metal backend next-token sampled output should match CPU reference");
}

void test_reports_unavailable_llama_next_token_backend_weight_copy_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);

    try {
        (void)copy_llama_model_logits_weights_to_backend(weights, BackendKind::metal);
        throw TestFailure("Llama next-token Metal backend weight copy should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Llama next-token Metal backend weight copy unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Llama next-token Metal backend weight copy backend mismatch");
    }
}

void test_generates_llama_argmax_tokens() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto tokens = generate_llama_argmax_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );

    expect(tokens.size() == 3, "Llama argmax generation token count mismatch");
    expect(tokens[0] == 0, "Llama argmax generation prompt token mismatch");
    expect(tokens[1] == 2, "Llama argmax generation first token mismatch");
    expect(tokens[2] == 0, "Llama argmax generation second token mismatch");
}

void test_generates_llama_argmax_tokens_stops_at_token() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto tokens = generate_llama_argmax_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        4,
        weights,
        model.llama_config(),
        std::optional<std::uint32_t>(2)
    );

    expect(tokens.size() == 2, "Llama argmax stop-token generation count mismatch");
    expect(tokens[0] == 0, "Llama argmax stop-token prompt mismatch");
    expect(tokens[1] == 2, "Llama argmax stop-token generated token mismatch");
}

void test_generates_llama_argmax_tokens_with_kv_cache_matches_no_cache() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto cached_tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );
    const auto no_cache_tokens = generate_llama_argmax_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config()
    );

    expect(cached_tokens == no_cache_tokens, "cached Llama argmax generation should match no-cache generation");
    expect(cached_tokens.size() == 3, "cached Llama argmax generation token count mismatch");
    expect(cached_tokens[0] == 0, "cached Llama argmax generation prompt token mismatch");
    expect(cached_tokens[1] == 2, "cached Llama argmax generation first token mismatch");
    expect(cached_tokens[2] == 0, "cached Llama argmax generation second token mismatch");
}

void test_generates_llama_argmax_tokens_with_kv_cache_stops_at_token() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    const auto tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        4,
        weights,
        model.llama_config(),
        std::optional<std::uint32_t>(2)
    );

    expect(tokens.size() == 2, "cached Llama argmax stop-token generation count mismatch");
    expect(tokens[0] == 0, "cached Llama argmax stop-token prompt mismatch");
    expect(tokens[1] == 2, "cached Llama argmax stop-token generated token mismatch");
}

void test_generates_llama_sampled_tokens_with_kv_cache_matches_no_cache() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 8;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};
    const SamplingConfig sampling {
        .temperature = 10.0F,
        .top_k = 2,
        .seed = 0,
    };

    const auto cached_tokens = generate_llama_sampled_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config(),
        sampling
    );
    const auto no_cache_tokens = generate_llama_sampled_tokens_f32(
        std::span<const std::uint32_t>(prompt_tokens),
        2,
        weights,
        model.llama_config(),
        sampling
    );

    expect(cached_tokens == no_cache_tokens, "cached Llama sampled generation should match no-cache generation");
    expect(cached_tokens.size() == 3, "cached Llama sampled generation token count mismatch");
    expect(cached_tokens[0] == 0, "cached Llama sampled generation prompt token mismatch");
    expect(cached_tokens[1] == 1, "cached Llama sampled generation first token mismatch");
    expect(cached_tokens[2] == 0, "cached Llama sampled generation second token mismatch");
}

void test_rejects_llama_cached_generation_zero_new_tokens() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_with_kv_cache_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            0,
            weights,
            model.llama_config()
        );
        throw TestFailure("cached Llama generation should reject zero new tokens");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation max_new_tokens must be positive",
            "cached Llama generation zero-token error mismatch"
        );
    }
}

void test_rejects_llama_cached_generation_empty_prompt() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 0> prompt_tokens {};

    try {
        (void)generate_llama_argmax_tokens_with_kv_cache_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            1,
            weights,
            model.llama_config()
        );
        throw TestFailure("cached Llama generation should reject empty prompt");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama token id span must not be empty",
            "cached Llama generation empty-prompt error mismatch"
        );
    }
}

void test_rejects_llama_cached_generation_context_overflow() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 2;
    const auto model = tiny_argmax_generation_model(config);
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_with_kv_cache_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            2,
            weights,
            model.llama_config()
        );
        throw TestFailure("cached Llama generation should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation would exceed context length",
            "cached Llama generation context overflow error mismatch"
        );
    }
}

void test_rejects_llama_generation_zero_new_tokens() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            0,
            weights,
            model.llama_config()
        );
        throw TestFailure("Llama generation should reject zero new tokens");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation max_new_tokens must be positive",
            "Llama generation zero-token error mismatch"
        );
    }
}

void test_rejects_llama_generation_context_overflow() {
    auto config = tiny_logits_config();
    config.block_count = 1;
    config.context_length = 2;
    const std::array<float, 6> embedding_values {
        3.0F, 0.0F, 0.0F,
        4.0F, 0.0F, 0.0F,
    };
    const std::array<float, 2> norm_values {1.0F, 1.0F};
    const std::array<float, 6> output_values {
        1.0F, 0.0F, 2.0F,
        0.0F, 1.0F, 3.0F,
    };
    std::vector<EngineTensor> tensors;
    tensors.push_back(engine_tensor(
        "token_embd.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(embedding_values))
    ));
    tensors.push_back(engine_tensor(
        "output_norm.weight",
        tensor_f32_vector(std::span<const float>(norm_values))
    ));
    tensors.push_back(engine_tensor(
        "output.weight",
        tensor_f16_gguf_matrix(2, 3, std::span<const float>(output_values))
    ));
    add_tiny_llama_block_tensors(tensors, 0, 0.0F, false, true);
    const auto model = tiny_bound_llama_model(config, std::move(tensors));
    const auto weights = bind_llama_model_weights(model);
    const std::array<std::uint32_t, 1> prompt_tokens {0};

    try {
        (void)generate_llama_argmax_tokens_f32(
            std::span<const std::uint32_t>(prompt_tokens),
            2,
            weights,
            model.llama_config()
        );
        throw TestFailure("Llama generation should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Llama generation would exceed context length",
            "Llama generation context overflow error mismatch"
        );
    }
}

void test_generates_text_from_gguf_bytes_with_cached_argmax() {
    const auto bytes = tiny_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2);

    expect(text == "!", "generate text should decode newly generated non-special tokens");
}

void test_generates_text_from_gguf_bytes_with_tied_output_weight() {
    const auto bytes = tiny_tied_output_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2);

    expect(text == "!!", "generate text should support tied output weights");
}

void test_generates_text_from_gguf_bytes_stops_at_tokenizer_eos() {
    const auto bytes = tiny_eos_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 2);

    expect(text.empty(), "generate text should stop before visible token after EOS");
}

void test_generates_text_from_gguf_bytes_with_chat_template_prompt() {
    const auto bytes = tiny_chat_template_generate_cli_gguf_bytes();

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "Hello", 1);

    expect(text == "!", "generate text should apply tokenizer chat template before encoding");
}

void test_generates_text_from_gguf_bytes_with_sampling_options() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const GenerateOptions options {
        .sampling = SamplingConfig {
            .temperature = 10.0F,
            .top_k = 2,
            .seed = 0,
        },
    };

    const auto text = generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), "", 1, options);

    expect(text == "Hello", "generate text should use seeded temperature/top-k sampling options");
}

void test_runs_generate_success_path() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "", "2", output, error);

    expect(exit_code == EXIT_SUCCESS, "generate success exit code mismatch");
    expect(output.str() == "!!\n", "generate success output mismatch");
    expect(error.str().empty(), "generate success should not write stderr");
}

void test_runs_generate_with_greedy_option() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-greedy", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "2",
        "--greedy",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(5, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate greedy success exit code mismatch");
    expect(output.str() == "!\n", "generate greedy output mismatch");
    expect(error.str().empty(), "generate greedy should not write stderr");
}

void test_runs_generate_with_sampling_options() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-sampling", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--temperature",
        "10",
        "--top-k",
        "2",
        "--top-p",
        "1",
        "--repeat-penalty",
        "1",
        "--seed",
        "0",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(14, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate sampling success exit code mismatch");
    expect(output.str() == "Hello\n", "generate sampling success output mismatch");
    expect(error.str().empty(), "generate sampling success should not write stderr");
}

void test_runs_generate_with_top_p_option() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-top-p", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--temperature",
        "10",
        "--top-p",
        "0.01",
        "--seed",
        "0",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(10, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate top-p success exit code mismatch");
    expect(output.str() == "!\n", "generate top-p output mismatch");
    expect(error.str().empty(), "generate top-p should not write stderr");
}

void test_runs_generate_with_first_token_debug() {
    const auto bytes = tiny_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-generate-debug", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--debug-first-token",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(5, argv, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate debug success exit code mismatch");
    expect(output.str() == "!\n", "generate debug output mismatch");
    expect(error.str().find("debug: formatted_prompt=\"\"") != std::string::npos, "generate debug prompt missing");
    expect(
        error.str().find("debug: encode_special_tokens=false") != std::string::npos,
        "generate debug special-token flag missing"
    );
    expect(error.str().find("debug: prompt_token_ids=[0]") != std::string::npos, "generate debug token IDs missing");
    expect(
        error.str().find("debug:   token=2") != std::string::npos,
        "generate debug top token missing"
    );
    expect(
        error.str().find("text=\"!\"") != std::string::npos,
        "generate debug decoded top token missing"
    );
}

void test_rejects_generate_missing_arguments() {
    const char* argv[] = {"local-gguf-generate"};
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(1, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate missing-args exit code mismatch");
    expect(output.str().empty(), "generate missing-args should not write stdout");
    expect(
        error.str().find("usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>") != std::string::npos,
        "generate missing-args usage missing"
    );
    expect(error.str().find("missing required argument") != std::string::npos, "generate missing-args error missing");
}

void test_rejects_generate_invalid_max_new_tokens() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(path, "", "0", output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-max exit code mismatch");
    expect(output.str().empty(), "generate invalid-max should not write stdout");
    expect(
        error.str().find("max-new-tokens must be a positive integer") != std::string::npos,
        "generate invalid-max error missing"
    );
}

void test_rejects_generate_invalid_temperature_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--temperature",
        "-1",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-temperature exit code mismatch");
    expect(output.str().empty(), "generate invalid-temperature should not write stdout");
    expect(
        error.str().find("--temperature must be a nonnegative finite number") != std::string::npos,
        "generate invalid-temperature error missing"
    );
}

void test_rejects_generate_invalid_top_p_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--top-p",
        "0",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-top-p exit code mismatch");
    expect(output.str().empty(), "generate invalid-top-p should not write stdout");
    expect(
        error.str().find("--top-p must be in (0, 1]") != std::string::npos,
        "generate invalid-top-p error missing"
    );
}

void test_rejects_generate_invalid_repeat_penalty_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--repeat-penalty",
        "0.5",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-repeat-penalty exit code mismatch");
    expect(output.str().empty(), "generate invalid-repeat-penalty should not write stdout");
    expect(
        error.str().find("--repeat-penalty must be at least 1") != std::string::npos,
        "generate invalid-repeat-penalty error missing"
    );
}

void test_rejects_generate_greedy_sampling_conflict() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--greedy",
        "--seed",
        "1",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate greedy-conflict exit code mismatch");
    expect(output.str().empty(), "generate greedy-conflict should not write stdout");
    expect(
        error.str().find("--greedy cannot be combined with sampling options") != std::string::npos,
        "generate greedy-conflict error missing"
    );
}

void test_converts_gguf_payload_view_to_tensor_storage() {
    const auto bytes = fixture("valid-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded = load_gguf_tensor_payload(model.tensor_payloads[0]);

    expect(loaded.name == "ab", "loaded GGUF tensor name mismatch");
    expect(loaded.storage.descriptor().shape == Shape::from({2}), "loaded GGUF tensor shape mismatch");
    expect(loaded.storage.descriptor().dtype == DType::f32, "loaded GGUF tensor dtype mismatch");
    expect(loaded.storage.descriptor().element_count == 2, "loaded GGUF tensor element count mismatch");
    expect(loaded.storage.descriptor().byte_count == 8, "loaded GGUF tensor byte count mismatch");
    expect(loaded.storage.bytes().size() == 8, "loaded GGUF tensor storage byte count mismatch");
    expect(
        loaded.storage.bytes().data() != model.tensor_payloads[0].bytes.data(),
        "loaded GGUF tensor storage should own a byte copy"
    );
    expect(loaded.storage.bytes()[0] == 0x00, "loaded GGUF tensor byte 0 mismatch");
    expect(loaded.storage.bytes()[7] == 0x40, "loaded GGUF tensor byte 7 mismatch");
}

void test_converts_q8_0_gguf_payload_view_to_tensor_storage() {
    const auto bytes = fixture("valid-q8-0-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
    const auto loaded_tensors = load_gguf_tensor_payloads(model);

    expect(loaded_tensors.size() == 1, "loaded Q8_0 GGUF tensor count mismatch");
    expect(loaded_tensors[0].name == "q8", "loaded Q8_0 GGUF tensor name mismatch");
    expect(loaded_tensors[0].storage.descriptor().shape == Shape::from({32}), "loaded Q8_0 shape mismatch");
    expect(loaded_tensors[0].storage.descriptor().dtype == DType::q8_0, "loaded Q8_0 dtype mismatch");
    expect(loaded_tensors[0].storage.descriptor().element_count == 32, "loaded Q8_0 element count mismatch");
    expect(loaded_tensors[0].storage.descriptor().byte_count == 34, "loaded Q8_0 byte count mismatch");
    expect(loaded_tensors[0].storage.bytes()[33] == 0x21, "loaded Q8_0 final byte mismatch");
}

void test_rejects_unsupported_gguf_tensor_type_in_bridge() {
    const GGUFTensorPayloadView payload {
        .tensor = GGUFTensorInfo {
            .name = "bad",
            .dimensions = std::vector<std::uint64_t> {1},
            .type_id = 255,
            .offset = 0,
            .file_offset = 0,
        },
        .bytes = {},
    };

    try {
        (void)load_gguf_tensor_payload(payload);
        throw TestFailure("unsupported GGUF tensor bridge type should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::unsupported_tensor_type, "error code should be unsupported_tensor_type");
        expect(error.actual_byte_count() == 255, "unsupported GGUF tensor type id mismatch");
    }
}

void test_rejects_invalid_quantized_gguf_tensor_shape_in_bridge() {
    const std::array<std::uint8_t, 34> bytes {};
    const GGUFTensorPayloadView payload {
        .tensor = GGUFTensorInfo {
            .name = "q8_bad",
            .dimensions = std::vector<std::uint64_t> {31},
            .type_id = 8,
            .offset = 0,
            .file_offset = 0,
        },
        .bytes = std::span<const std::uint8_t>(bytes),
    };

    try {
        (void)load_gguf_tensor_payload(payload);
        throw TestFailure("invalid Q8_0 GGUF bridge shape should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_quantized_element_count,
            "error code should be invalid_quantized_element_count"
        );
        expect(error.expected_value() == 32, "expected Q8_0 bridge block size mismatch");
        expect(error.actual_value() == 31, "actual Q8_0 bridge element count mismatch");
    }
}

void test_rejects_gguf_tensor_payload_size_mismatch_in_bridge() {
    const std::array<std::uint8_t, 7> bytes {};
    const GGUFTensorPayloadView payload {
        .tensor = GGUFTensorInfo {
            .name = "short",
            .dimensions = std::vector<std::uint64_t> {2},
            .type_id = 0,
            .offset = 0,
            .file_offset = 0,
        },
        .bytes = std::span<const std::uint8_t>(bytes),
    };

    try {
        (void)load_gguf_tensor_payload(payload);
        throw TestFailure("GGUF bridge payload size mismatch should throw");
    } catch (const TensorError& error) {
        expect(
            error.code() == TensorErrorCode::invalid_storage_byte_count,
            "error code should be invalid_storage_byte_count"
        );
        expect(error.expected_value() == 8, "expected GGUF bridge storage byte count mismatch");
        expect(error.actual_value() == 7, "actual GGUF bridge storage byte count mismatch");
    }
}

void test_reads_valid_gguf_fixed_header_fixture() {
    const auto bytes = fixture("valid-header-v3");
    const auto header = read_gguf_header(std::span<const std::uint8_t>(bytes));

    expect(
        header == GGUFHeader {.version = 3, .tensor_count = 2, .metadata_kv_count = 5},
        "GGUF header should parse magic, version, tensor count, and metadata kv count"
    );
}

void test_reads_little_endian_header_counts() {
    const std::array<std::uint8_t, GGUFHeader::encoded_byte_count> bytes {
        0x47, 0x47, 0x55, 0x46,
        0x03, 0x00, 0x00, 0x00,
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
    };

    const auto header = read_gguf_header(std::span<const std::uint8_t>(bytes));

    expect(header.version == 3, "GGUF header version should be 3");
    expect(header.tensor_count == 0x0102030405060708ULL, "tensor count should parse as little-endian uint64");
    expect(
        header.metadata_kv_count == 0x1122334455667788ULL,
        "metadata kv count should parse as little-endian uint64"
    );
}

void test_rejects_invalid_gguf_magic() {
    const auto bytes = fixture("invalid-magic");

    try {
        (void)read_gguf_header(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid magic should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_magic, "error code should be invalid_magic");
        expect(error.actual_magic() == std::array<std::uint8_t, 4> {0x42, 0x41, 0x44, 0x21}, "actual magic mismatch");
    }
}

void test_rejects_truncated_gguf_header() {
    const auto bytes = fixture("truncated-header");

    try {
        (void)read_gguf_header(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated header should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::truncated_header, "error code should be truncated_header");
        expect(error.expected_byte_count() == GGUFHeader::encoded_byte_count, "expected byte count mismatch");
        expect(error.actual_byte_count() == 3, "actual byte count mismatch");
    }
}

void test_rejects_truncated_count_data() {
    const auto bytes = fixture("truncated-count-data");

    try {
        (void)read_gguf_header(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated count data should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::truncated_header, "error code should be truncated_header");
        expect(error.expected_byte_count() == GGUFHeader::encoded_byte_count, "expected byte count mismatch");
        expect(error.actual_byte_count() == 16, "actual byte count mismatch");
    }
}

void test_reads_scalar_metadata_entries() {
    const auto bytes = fixture("valid-metadata-scalars");
    const auto section = read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));

    expect(
        section.header == GGUFHeader {.version = 3, .tensor_count = 0, .metadata_kv_count = 4},
        "GGUF metadata section header mismatch"
    );
    expect(section.entries.size() == 4, "metadata entry count mismatch");
    expect(section.encoded_byte_count == bytes.size(), "metadata byte count should end after parsed entries");

    expect(section.entries[0].key == "general.architecture", "metadata key 0 mismatch");
    expect(
        metadata_data<std::string>(
            section.entries[0].value,
            GGUFMetadataValueType::string,
            "general.architecture should be a string"
        ) == "llama",
        "general.architecture value mismatch"
    );

    expect(section.entries[1].key == "general.alignment", "metadata key 1 mismatch");
    expect(
        metadata_data<std::uint32_t>(
            section.entries[1].value,
            GGUFMetadataValueType::uint32,
            "general.alignment should be a uint32"
        ) == 32,
        "general.alignment value mismatch"
    );

    expect(section.entries[2].key == "llama.context_length", "metadata key 2 mismatch");
    expect(
        metadata_data<std::uint64_t>(
            section.entries[2].value,
            GGUFMetadataValueType::uint64,
            "llama.context_length should be a uint64"
        ) == 4096,
        "llama.context_length value mismatch"
    );

    expect(section.entries[3].key == "llama.use_parallel_residual", "metadata key 3 mismatch");
    expect(
        metadata_data<bool>(
            section.entries[3].value,
            GGUFMetadataValueType::boolean,
            "llama.use_parallel_residual should be a bool"
        ),
        "llama.use_parallel_residual value mismatch"
    );
}

void test_reads_all_scalar_metadata_value_types() {
    const auto bytes = fixture("valid-metadata-all-scalars");
    const auto section = read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));

    expect(section.entries.size() == 12, "metadata scalar type entry count mismatch");
    expect(section.encoded_byte_count == bytes.size(), "metadata scalar type byte count mismatch");

    expect(section.entries[0].key == "test.u8", "uint8 metadata key mismatch");
    expect(
        metadata_data<std::uint8_t>(section.entries[0].value, GGUFMetadataValueType::uint8, "test.u8 type mismatch")
            == 0x7FU,
        "uint8 metadata value mismatch"
    );

    expect(section.entries[1].key == "test.i8", "int8 metadata key mismatch");
    expect(
        metadata_data<std::int8_t>(section.entries[1].value, GGUFMetadataValueType::int8, "test.i8 type mismatch")
            == static_cast<std::int8_t>(-1),
        "int8 metadata value mismatch"
    );

    expect(section.entries[2].key == "test.u16", "uint16 metadata key mismatch");
    expect(
        metadata_data<std::uint16_t>(
            section.entries[2].value,
            GGUFMetadataValueType::uint16,
            "test.u16 type mismatch"
        ) == 0x1234U,
        "uint16 metadata value mismatch"
    );

    expect(section.entries[3].key == "test.i16", "int16 metadata key mismatch");
    expect(
        metadata_data<std::int16_t>(section.entries[3].value, GGUFMetadataValueType::int16, "test.i16 type mismatch")
            == static_cast<std::int16_t>(-2),
        "int16 metadata value mismatch"
    );

    expect(section.entries[4].key == "test.u32", "uint32 metadata key mismatch");
    expect(
        metadata_data<std::uint32_t>(
            section.entries[4].value,
            GGUFMetadataValueType::uint32,
            "test.u32 type mismatch"
        ) == 0x12345678U,
        "uint32 metadata value mismatch"
    );

    expect(section.entries[5].key == "test.i32", "int32 metadata key mismatch");
    expect(
        metadata_data<std::int32_t>(section.entries[5].value, GGUFMetadataValueType::int32, "test.i32 type mismatch")
            == static_cast<std::int32_t>(-1),
        "int32 metadata value mismatch"
    );

    expect(section.entries[6].key == "test.f32", "float32 metadata key mismatch");
    expect(
        metadata_data<float>(section.entries[6].value, GGUFMetadataValueType::float32, "test.f32 type mismatch")
            == 3.5F,
        "float32 metadata value mismatch"
    );

    expect(section.entries[7].key == "test.bool", "bool metadata key mismatch");
    expect(
        metadata_data<bool>(section.entries[7].value, GGUFMetadataValueType::boolean, "test.bool type mismatch"),
        "bool metadata value mismatch"
    );

    expect(section.entries[8].key == "test.string", "string metadata key mismatch");
    expect(
        metadata_data<std::string>(
            section.entries[8].value,
            GGUFMetadataValueType::string,
            "test.string type mismatch"
        ) == "ok",
        "string metadata value mismatch"
    );

    expect(section.entries[9].key == "test.u64", "uint64 metadata key mismatch");
    expect(
        metadata_data<std::uint64_t>(
            section.entries[9].value,
            GGUFMetadataValueType::uint64,
            "test.u64 type mismatch"
        ) == 0x0102030405060708ULL,
        "uint64 metadata value mismatch"
    );

    expect(section.entries[10].key == "test.i64", "int64 metadata key mismatch");
    expect(
        metadata_data<std::int64_t>(
            section.entries[10].value,
            GGUFMetadataValueType::int64,
            "test.i64 type mismatch"
        ) == static_cast<std::int64_t>(-2),
        "int64 metadata value mismatch"
    );

    expect(section.entries[11].key == "test.f64", "float64 metadata key mismatch");
    expect(
        metadata_data<double>(section.entries[11].value, GGUFMetadataValueType::float64, "test.f64 type mismatch")
            == 3.5,
        "float64 metadata value mismatch"
    );
}

void test_reads_string_array_metadata_entry() {
    const auto bytes = fixture("valid-metadata-string-array");
    const auto section = read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));

    expect(
        section.header == GGUFHeader {.version = 3, .tensor_count = 0, .metadata_kv_count = 1},
        "GGUF metadata section header mismatch"
    );
    expect(section.entries.size() == 1, "metadata array entry count mismatch");
    expect(section.encoded_byte_count == bytes.size(), "metadata byte count should end after array entry");
    expect(section.entries[0].key == "tokenizer.ggml.tokens", "metadata array key mismatch");

    const auto& array = metadata_data<GGUFMetadataArray>(
        section.entries[0].value,
        GGUFMetadataValueType::array,
        "tokenizer.ggml.tokens should be an array"
    );
    expect(array.element_type == GGUFMetadataValueType::string, "metadata array element type mismatch");

    const auto& tokens = std::get<std::vector<std::string>>(array.values);
    expect(tokens.size() == 2, "metadata string array size mismatch");
    expect(tokens[0] == "<s>", "metadata string array token 0 mismatch");
    expect(tokens[1] == "</s>", "metadata string array token 1 mismatch");
}

void test_rejects_invalid_metadata_value_type() {
    const auto bytes = fixture("invalid-metadata-type");

    try {
        (void)read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid metadata value type should throw");
    } catch (const GGUFReadError& error) {
        expect(
            error.code() == GGUFReadErrorCode::invalid_metadata_value_type,
            "error code should be invalid_metadata_value_type"
        );
    }
}

void test_rejects_invalid_metadata_bool() {
    const auto bytes = fixture("invalid-metadata-bool");

    try {
        (void)read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid metadata bool should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_metadata_value, "error code should be invalid_metadata_value");
    }
}

void test_rejects_nested_metadata_array() {
    const auto bytes = fixture("unsupported-nested-metadata-array");

    try {
        (void)read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));
        throw TestFailure("nested metadata array should throw until supported");
    } catch (const GGUFReadError& error) {
        expect(
            error.code() == GGUFReadErrorCode::unsupported_metadata_value_type,
            "error code should be unsupported_metadata_value_type"
        );
    }
}

void test_rejects_truncated_metadata_value() {
    const auto bytes = fixture("truncated-metadata-value");

    try {
        (void)read_gguf_metadata_section(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated metadata value should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::truncated_metadata, "error code should be truncated_metadata");
    }
}

void test_reads_tensor_directory_entries() {
    const auto bytes = fixture("valid-tensor-directory");
    const auto directory = read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));

    expect(
        directory.metadata.header == GGUFHeader {.version = 3, .tensor_count = 2, .metadata_kv_count = 1},
        "GGUF tensor directory header mismatch"
    );
    expect(directory.metadata.entries.size() == 1, "metadata entry count before tensor directory mismatch");
    expect(directory.metadata.entries[0].key == "general.alignment", "metadata key before tensor directory mismatch");
    expect(
        metadata_data<std::uint32_t>(
            directory.metadata.entries[0].value,
            GGUFMetadataValueType::uint32,
            "general.alignment should be a uint32"
        ) == 32,
        "metadata value before tensor directory mismatch"
    );

    expect(directory.tensors.size() == 2, "tensor directory entry count mismatch");
    expect(directory.alignment == 32, "tensor directory alignment mismatch");
    expect(directory.tensor_infos_end_offset == 171, "tensor info end offset mismatch");
    expect(directory.padding_byte_count == 21, "tensor padding byte count mismatch");
    expect(directory.tensor_data_start_offset == 192, "tensor data start offset mismatch");
    expect(directory.encoded_byte_count == bytes.size(), "tensor directory byte count should include padding");

    expect(directory.tensors[0].name == "tok_embeddings.weight", "tensor 0 name mismatch");
    expect(
        directory.tensors[0].dimensions == std::vector<std::uint64_t> {4, 8},
        "tensor 0 dimensions mismatch"
    );
    expect(directory.tensors[0].type_id == 0, "tensor 0 type id mismatch");
    expect(directory.tensors[0].offset == 0, "tensor 0 data-relative offset mismatch");
    expect(directory.tensors[0].file_offset == 192, "tensor 0 file-relative offset mismatch");

    expect(directory.tensors[1].name == "output.weight", "tensor 1 name mismatch");
    expect(
        directory.tensors[1].dimensions == std::vector<std::uint64_t> {8, 4},
        "tensor 1 dimensions mismatch"
    );
    expect(directory.tensors[1].type_id == 8, "tensor 1 type id mismatch");
    expect(directory.tensors[1].offset == 128, "tensor 1 data-relative offset mismatch");
    expect(directory.tensors[1].file_offset == 320, "tensor 1 file-relative offset mismatch");
}

void test_reads_unpadded_tensor_directory_with_default_alignment() {
    const auto bytes = fixture("valid-tensor-directory-default-alignment");
    const auto directory = read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));

    expect(
        directory.metadata.header == GGUFHeader {.version = 3, .tensor_count = 1, .metadata_kv_count = 0},
        "default alignment tensor directory header mismatch"
    );
    expect(directory.alignment == 32, "default tensor directory alignment mismatch");
    expect(directory.tensor_infos_end_offset == 64, "default tensor info end offset mismatch");
    expect(directory.padding_byte_count == 0, "default tensor padding byte count mismatch");
    expect(directory.tensor_data_start_offset == 64, "default tensor data start offset mismatch");
    expect(directory.encoded_byte_count == bytes.size(), "default tensor directory byte count mismatch");
    expect(directory.tensors.size() == 1, "default alignment tensor count mismatch");
    expect(directory.tensors[0].name == "abcdefgh", "default alignment tensor name mismatch");
    expect(directory.tensors[0].dimensions == std::vector<std::uint64_t> {8}, "default tensor dimensions mismatch");
    expect(directory.tensors[0].offset == 0, "default tensor data-relative offset mismatch");
    expect(directory.tensors[0].file_offset == 64, "default tensor file-relative offset mismatch");
}

void test_rejects_invalid_tensor_dimension_count() {
    const auto bytes = fixture("invalid-tensor-dimension-count");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid tensor dimension count should throw");
    } catch (const GGUFReadError& error) {
        expect(
            error.code() == GGUFReadErrorCode::invalid_tensor_dimension_count,
            "error code should be invalid_tensor_dimension_count"
        );
        expect(error.expected_byte_count() == 4, "expected dimension count mismatch");
        expect(error.actual_byte_count() == 5, "actual dimension count mismatch");
    }
}

void test_rejects_invalid_tensor_name() {
    const auto bytes = fixture("invalid-tensor-name");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid tensor name should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_tensor_name, "error code should be invalid_tensor_name");
        expect(error.expected_byte_count() == 64, "expected tensor name byte count mismatch");
        expect(error.actual_byte_count() == 65, "actual tensor name byte count mismatch");
    }
}

void test_rejects_invalid_alignment_value() {
    const auto bytes = fixture("invalid-alignment-value");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid alignment value should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_alignment, "error code should be invalid_alignment");
        expect(error.expected_byte_count() == 8, "expected alignment byte count mismatch");
        expect(error.actual_byte_count() == 12, "actual alignment value mismatch");
    }
}

void test_rejects_invalid_alignment_type() {
    const auto bytes = fixture("invalid-alignment-type");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid alignment type should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_alignment, "error code should be invalid_alignment");
    }
}

void test_rejects_misaligned_tensor_offset() {
    const auto bytes = fixture("invalid-tensor-offset-alignment");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("misaligned tensor offset should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_tensor_offset, "error code should be invalid_tensor_offset");
        expect(error.expected_byte_count() == 32, "expected tensor offset alignment mismatch");
        expect(error.actual_byte_count() == 16, "actual tensor offset mismatch");
    }
}

void test_rejects_invalid_tensor_padding() {
    const auto bytes = fixture("invalid-tensor-padding");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid tensor padding should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_tensor_padding, "error code should be invalid_tensor_padding");
    }
}

void test_rejects_truncated_tensor_directory() {
    const auto bytes = fixture("truncated-tensor-directory");

    try {
        (void)read_gguf_tensor_directory(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated tensor directory should throw");
    } catch (const GGUFReadError& error) {
        expect(
            error.code() == GGUFReadErrorCode::truncated_tensor_directory,
            "error code should be truncated_tensor_directory"
        );
    }
}

void test_reads_tensor_payload_views_without_copying() {
    const auto bytes = fixture("valid-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));

    expect(model.directory.tensors.size() == 1, "payload view tensor count mismatch");
    expect(model.directory.tensor_data_start_offset == 64, "payload view tensor data start mismatch");
    expect(model.tensor_payloads.size() == 1, "payload view count mismatch");
    expect(model.tensor_payloads[0].tensor.name == "ab", "payload view tensor name mismatch");
    expect(model.tensor_payloads[0].tensor.type_id == 0, "payload view tensor type mismatch");
    expect(model.tensor_payloads[0].tensor.file_offset == 64, "payload view tensor file offset mismatch");
    expect(model.tensor_payloads[0].bytes.size() == 8, "payload view byte count mismatch");
    expect(model.tensor_payloads[0].bytes.data() == bytes.data() + 64, "payload view should point into input bytes");
    expect(model.tensor_payloads[0].bytes[0] == 0x00, "payload byte 0 mismatch");
    expect(model.tensor_payloads[0].bytes[2] == 0x80, "payload byte 2 mismatch");
    expect(model.tensor_payloads[0].bytes[6] == 0x00, "payload byte 6 mismatch");
    expect(model.tensor_payloads[0].bytes[7] == 0x40, "payload byte 7 mismatch");
}

void test_reads_q8_0_tensor_payload_view() {
    const auto bytes = fixture("valid-q8-0-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));

    expect(model.directory.tensors.size() == 1, "Q8_0 payload view tensor count mismatch");
    expect(model.directory.tensors[0].name == "q8", "Q8_0 payload tensor name mismatch");
    expect(model.directory.tensors[0].dimensions == std::vector<std::uint64_t> {32}, "Q8_0 dimensions mismatch");
    expect(model.directory.tensors[0].type_id == 8, "Q8_0 tensor type mismatch");
    expect(model.directory.tensors[0].file_offset == 64, "Q8_0 file offset mismatch");
    expect(model.tensor_payloads.size() == 1, "Q8_0 payload view count mismatch");
    expect(model.tensor_payloads[0].bytes.size() == 34, "Q8_0 payload byte count mismatch");
    expect(model.tensor_payloads[0].bytes.data() == bytes.data() + 64, "Q8_0 payload should point into input bytes");
    expect(model.tensor_payloads[0].bytes[0] == 0x00, "Q8_0 payload byte 0 mismatch");
    expect(model.tensor_payloads[0].bytes[33] == 0x21, "Q8_0 payload byte 33 mismatch");
}

void test_reads_f16_tensor_payload_view() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));

    expect(model.directory.tensors.size() == 1, "F16 payload view tensor count mismatch");
    expect(model.directory.tensors[0].name == "h2", "F16 payload tensor name mismatch");
    expect(model.directory.tensors[0].dimensions == std::vector<std::uint64_t> {2}, "F16 dimensions mismatch");
    expect(model.directory.tensors[0].type_id == 1, "F16 tensor type mismatch");
    expect(model.directory.tensors[0].file_offset == 64, "F16 file offset mismatch");
    expect(model.tensor_payloads.size() == 1, "F16 payload view count mismatch");
    expect(model.tensor_payloads[0].bytes.size() == 4, "F16 payload byte count mismatch");
    expect(model.tensor_payloads[0].bytes.data() == bytes.data() + 64, "F16 payload should point into input bytes");
    expect(model.tensor_payloads[0].bytes[0] == 0x00, "F16 payload byte 0 mismatch");
    expect(model.tensor_payloads[0].bytes[3] == 0x40, "F16 payload byte 3 mismatch");
}

void test_rejects_truncated_tensor_payload() {
    const auto bytes = fixture("truncated-tensor-payload");

    try {
        (void)read_gguf_model_view(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated tensor payload should throw");
    } catch (const GGUFReadError& error) {
        expect(
            error.code() == GGUFReadErrorCode::truncated_tensor_payload,
            "error code should be truncated_tensor_payload"
        );
    }
}

void test_rejects_unsupported_tensor_payload_type() {
    const auto bytes = fixture("unsupported-tensor-payload-type");

    try {
        (void)read_gguf_model_view(std::span<const std::uint8_t>(bytes));
        throw TestFailure("unsupported tensor type should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::unsupported_tensor_type, "error code should be unsupported_tensor_type");
    }
}

void test_rejects_invalid_quantized_tensor_payload_size() {
    const auto bytes = fixture("invalid-quantized-tensor-payload-size");

    try {
        (void)read_gguf_model_view(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid quantized tensor payload size should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_tensor_size, "error code should be invalid_tensor_size");
    }
}

void test_inspects_gguf_summary_from_f16_payload() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto summary = inspect_gguf_bytes(std::span<const std::uint8_t>(bytes), 3);

    expect(summary.version == 3, "GGUF inspect version mismatch");
    expect(summary.tensor_count == 1, "GGUF inspect tensor count mismatch");
    expect(summary.metadata_kv_count == 0, "GGUF inspect metadata count mismatch");
    expect(!summary.architecture.has_value(), "GGUF inspect architecture should be absent");
    expect(summary.alignment == 32, "GGUF inspect alignment mismatch");
    expect(summary.tensor_data_start_offset == 64, "GGUF inspect tensor data start mismatch");
    expect(summary.dtype_counts.f32 == 0, "GGUF inspect F32 count mismatch");
    expect(summary.dtype_counts.f16 == 1, "GGUF inspect F16 count mismatch");
    expect(summary.dtype_counts.q8_0 == 0, "GGUF inspect Q8_0 count mismatch");
    expect(summary.dtype_counts.unsupported == 0, "GGUF inspect unsupported count mismatch");
    expect(summary.total_supported_payload_bytes == 4, "GGUF inspect supported payload byte mismatch");
    expect(summary.first_tensors.size() == 1, "GGUF inspect first tensor count mismatch");
    expect(summary.first_tensors[0].name == "h2", "GGUF inspect first tensor name mismatch");
    expect(summary.first_tensors[0].shape == std::vector<std::uint64_t> {2}, "GGUF inspect shape mismatch");
    expect(summary.first_tensors[0].dtype == "F16", "GGUF inspect first tensor dtype mismatch");
}

void test_formats_gguf_inspection_summary() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto summary = inspect_gguf_bytes(std::span<const std::uint8_t>(bytes), 1);

    std::ostringstream output;
    write_gguf_inspection_summary(summary, "fixture.gguf", output);
    const auto text = output.str();

    expect(text.find("GGUF summary\n") == 0, "GGUF inspect formatted heading mismatch");
    expect(text.find("version: 3\n") != std::string::npos, "GGUF inspect formatted version missing");
    expect(text.find("tensor count: 1\n") != std::string::npos, "GGUF inspect formatted tensor count missing");
    expect(text.find("metadata kv count: 0\n") != std::string::npos, "GGUF inspect formatted metadata count missing");
    expect(text.find("architecture: <absent>\n") != std::string::npos, "GGUF inspect formatted architecture missing");
    expect(text.find("alignment: 32\n") != std::string::npos, "GGUF inspect formatted alignment missing");
    expect(
        text.find("tensor data start offset: 64\n") != std::string::npos,
        "GGUF inspect formatted tensor data start missing"
    );
    expect(text.find("  F16: 1\n") != std::string::npos, "GGUF inspect formatted F16 count missing");
    expect(
        text.find("total supported payload bytes: 4\n") != std::string::npos,
        "GGUF inspect formatted payload byte count missing"
    );
    expect(
        text.find("  - h2 shape=[2] dtype=F16\n") != std::string::npos,
        "GGUF inspect formatted tensor line missing"
    );
}

void test_runs_gguf_inspect_success() {
    const TempBinaryFixture model("valid-f16-tensor-payload-view");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_SUCCESS, "gguf-inspect success exit code mismatch");
    expect(error.str().empty(), "gguf-inspect success should not write stderr");
    expect(output.str().find("GGUF summary\n") == 0, "gguf-inspect success output missing summary");
    expect(
        output.str().find("total supported payload bytes: 4\n") != std::string::npos,
        "gguf-inspect success output missing payload byte count"
    );
}

void test_rejects_gguf_inspect_missing_path() {
    const char* argv[] = {"gguf-inspect"};
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect(1, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect missing path exit code mismatch");
    expect(output.str().empty(), "gguf-inspect missing path should not write stdout");
    expect(error.str().find("usage: gguf-inspect <model.gguf>") != std::string::npos, "gguf-inspect usage missing");
    expect(error.str().find("missing GGUF path") != std::string::npos, "gguf-inspect missing path error missing");
}

void test_rejects_gguf_inspect_unreadable_path() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-missing-inspect.gguf";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(path, output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect unreadable path exit code mismatch");
    expect(output.str().empty(), "gguf-inspect unreadable path should not write stdout");
    expect(error.str().find("could not read") != std::string::npos, "gguf-inspect unreadable path error missing");
}

void test_reports_invalid_gguf_in_gguf_inspect() {
    const TempBinaryFixture model("invalid-magic");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect invalid GGUF exit code mismatch");
    expect(output.str().empty(), "gguf-inspect invalid GGUF should not write stdout");
    expect(
        error.str().find("invalid GGUF: invalid GGUF magic") != std::string::npos,
        "gguf-inspect invalid GGUF error missing"
    );
}

void test_reports_unsupported_tensor_type_in_gguf_inspect() {
    const TempBinaryFixture model("unsupported-tensor-payload-type");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect unsupported tensor exit code mismatch");
    expect(output.str().empty(), "gguf-inspect unsupported tensor should not write stdout");
    expect(
        error.str().find("unsupported tensor type during payload conversion") != std::string::npos,
        "gguf-inspect unsupported tensor error missing"
    );
    expect(error.str().find("GGUF tensor type 39") != std::string::npos, "gguf-inspect unsupported type id missing");
}

void test_reports_malformed_tensor_payload_in_gguf_inspect() {
    const TempBinaryFixture model("truncated-tensor-payload");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect malformed payload exit code mismatch");
    expect(output.str().empty(), "gguf-inspect malformed payload should not write stdout");
    expect(
        error.str().find("malformed tensor payload: truncated GGUF tensor payload") != std::string::npos,
        "gguf-inspect malformed payload error missing"
    );
}

void test_loads_engine_model_from_llama_gguf_view() {
    const std::array<std::uint8_t, 8> tensor_bytes {
        0x00, 0x00, 0x80, 0x3F,
        0x00, 0x00, 0x00, 0x40,
    };
    auto view = tiny_model_view(
        tiny_llama_metadata(),
        {tensor_payload_view("token_embd.weight", std::span<const std::uint8_t>(tensor_bytes))}
    );

    const auto model = load_engine_model_from_gguf(view);

    expect(model.architecture() == ModelArchitecture::llama, "engine model architecture mismatch");
    expect(model_architecture_name(model.architecture()) == "llama", "engine model architecture name mismatch");
    expect(model.name().has_value(), "engine model name should be present");
    expect(*model.name() == "Tiny Llama Fixture", "engine model name mismatch");
    expect(model.llama_config().context_length == 16, "engine model context length mismatch");
    expect(model.llama_config().embedding_length == 8, "engine model embedding length mismatch");
    expect(model.llama_config().block_count == 2, "engine model block count mismatch");
    expect(model.llama_config().feed_forward_length == 32, "engine model feed-forward length mismatch");
    expect(model.llama_config().attention_head_count == 4, "engine model head count mismatch");
    expect(model.llama_config().attention_head_count_kv == 2, "engine model KV head count mismatch");
    expect(model.llama_config().attention_key_length == 2, "engine model key length mismatch");
    expect(model.llama_config().attention_value_length == 2, "engine model value length mismatch");
    expect(model.llama_config().rope_dimension_count == 2, "engine model rope dimension mismatch");
    expect(model.llama_config().vocab_size == 64, "engine model vocab size mismatch");
    expect(model.llama_config().rope_freq_base == 10000.0F, "engine model rope base mismatch");
    expect(model.llama_config().rms_norm_epsilon == 0.00001F, "engine model RMS epsilon mismatch");
    expect(model.tensors().size() == 1, "engine model tensor count mismatch");

    const auto* tensor = model.find_tensor("token_embd.weight");
    expect(tensor != nullptr, "engine model tensor lookup failed");
    expect(tensor->storage.descriptor().dtype == DType::f32, "engine model tensor dtype mismatch");
    expect(tensor->storage.bytes().size() == tensor_bytes.size(), "engine model tensor byte count mismatch");
    expect(tensor->storage.bytes().data() != tensor_bytes.data(), "engine model tensor storage should own bytes");
    expect(tensor->storage.bytes()[3] == 0x3F, "engine model tensor copied byte mismatch");
    expect(model.find_tensor("missing.weight") == nullptr, "engine model missing tensor lookup should return null");
}

void test_rejects_engine_model_without_architecture_metadata() {
    std::array<std::uint8_t, 8> tensor_bytes {};
    auto metadata = tiny_llama_metadata();
    metadata.erase(metadata.begin());
    auto view = tiny_model_view(
        std::move(metadata),
        {tensor_payload_view("token_embd.weight", std::span<const std::uint8_t>(tensor_bytes))}
    );

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject missing architecture metadata");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_metadata, "engine model missing metadata error mismatch");
        expect(error.detail() == "general.architecture", "engine model missing metadata detail mismatch");
    }
}

void test_rejects_engine_model_unsupported_architecture() {
    auto metadata = tiny_llama_metadata();
    metadata[0] = metadata_string("general.architecture", "gptneox");
    auto view = tiny_model_view(std::move(metadata), {});

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject unsupported architecture");
    } catch (const EngineModelError& error) {
        expect(
            error.code() == EngineModelErrorCode::unsupported_architecture,
            "engine model unsupported architecture error mismatch"
        );
        expect(error.detail() == "gptneox", "engine model unsupported architecture detail mismatch");
    }
}

void test_rejects_engine_model_missing_llama_metadata() {
    auto metadata = tiny_llama_metadata();
    metadata.erase(metadata.begin() + 2);
    auto view = tiny_model_view(std::move(metadata), {});

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject missing Llama metadata");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_metadata, "engine model missing Llama metadata mismatch");
        expect(error.detail() == "llama.context_length", "engine model missing Llama metadata detail mismatch");
    }
}

void test_rejects_engine_model_invalid_metadata_type() {
    auto metadata = tiny_llama_metadata();
    metadata[4] = metadata_string("llama.block_count", "two");
    auto view = tiny_model_view(std::move(metadata), {});

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject invalid metadata type");
    } catch (const EngineModelError& error) {
        expect(
            error.code() == EngineModelErrorCode::invalid_metadata_type,
            "engine model invalid metadata type mismatch"
        );
        expect(error.detail() == "llama.block_count", "engine model invalid metadata detail mismatch");
    }
}

void test_rejects_engine_model_duplicate_tensor_names() {
    const std::array<std::uint8_t, 8> first_tensor_bytes {
        0x00, 0x00, 0x80, 0x3F,
        0x00, 0x00, 0x00, 0x40,
    };
    const std::array<std::uint8_t, 8> second_tensor_bytes {
        0x00, 0x00, 0x40, 0x40,
        0x00, 0x00, 0x80, 0x40,
    };
    auto view = tiny_model_view(
        tiny_llama_metadata(),
        {
            tensor_payload_view("token_embd.weight", std::span<const std::uint8_t>(first_tensor_bytes)),
            tensor_payload_view("token_embd.weight", std::span<const std::uint8_t>(second_tensor_bytes)),
        }
    );

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject duplicate tensor names");
    } catch (const EngineModelError& error) {
        expect(
            error.code() == EngineModelErrorCode::duplicate_tensor_name,
            "engine model duplicate tensor error mismatch"
        );
        expect(error.detail() == "token_embd.weight", "engine model duplicate tensor detail mismatch");
    }
}

void test_loads_llama_tokenizer_from_metadata() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));

    expect(tokenizer.model() == TokenizerModel::gpt2, "tokenizer model enum mismatch");
    expect(tokenizer.model_name() == "gpt2", "tokenizer model name mismatch");
    expect(tokenizer_model_name(tokenizer.model()) == "gpt2", "tokenizer model helper mismatch");
    expect(tokenizer.vocabulary().size() == 7, "tokenizer vocabulary size mismatch");
    expect(tokenizer.vocabulary()[0].text == "<s>", "tokenizer BOS text mismatch");
    expect(tokenizer.vocabulary()[2].text == "Hello", "tokenizer normal token text mismatch");
    expect(tokenizer.vocabulary()[5].score.has_value(), "tokenizer score should load");
    expect(*tokenizer.vocabulary()[5].score == -0.2F, "tokenizer score mismatch");
    expect(tokenizer.vocabulary()[0].type.has_value(), "tokenizer token type should load");
    expect(*tokenizer.vocabulary()[0].type == 3, "tokenizer token type mismatch");
    expect(tokenizer.special_tokens().bos_token_id == 0, "tokenizer BOS id mismatch");
    expect(tokenizer.special_tokens().eos_token_id == 1, "tokenizer EOS id mismatch");
    expect(tokenizer.special_tokens().unknown_token_id == 6, "tokenizer unknown id mismatch");

    const auto options = tokenizer.default_encode_options();
    expect(options.add_bos, "tokenizer default should add BOS from metadata");
    expect(!options.add_eos, "tokenizer default should not add EOS from metadata");
}

void test_encodes_llama_tokenizer_known_ids_with_special_tokens() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));

    const auto default_ids = tokenizer.encode("Hello world!");

    expect(default_ids.size() == 3, "tokenizer default encoded count mismatch");
    expect(default_ids[0] == 0, "tokenizer default BOS id mismatch");
    expect(default_ids[1] == 5, "tokenizer should prefer longest matching token");
    expect(default_ids[2] == 4, "tokenizer punctuation id mismatch");

    const auto eos_ids = tokenizer.encode("Hello world!", TokenizerEncodeOptions {.add_bos = false, .add_eos = true});
    expect(eos_ids.size() == 3, "tokenizer explicit EOS count mismatch");
    expect(eos_ids[0] == 5, "tokenizer explicit first token mismatch");
    expect(eos_ids[1] == 4, "tokenizer explicit punctuation mismatch");
    expect(eos_ids[2] == 1, "tokenizer explicit EOS id mismatch");
}

void test_encodes_gpt2_bpe_prompt_pieces() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_bpe_tokenizer_metadata(false))
    );

    const auto prompt = std::string("Hello world!") + "\xC3\xA5";
    const auto ids = tokenizer.encode(prompt, TokenizerEncodeOptions {.add_bos = false, .add_eos = true});

    expect(ids.size() == 5, "GPT-2 BPE encoded count mismatch");
    expect(ids[0] == 16, "GPT-2 BPE should merge Hello");
    expect(ids[1] == 21, "GPT-2 BPE should merge space-prefixed world");
    expect(ids[2] == 10, "GPT-2 BPE punctuation token mismatch");
    expect(ids[3] == 22, "GPT-2 BPE UTF-8 byte merge mismatch");
    expect(ids[4] == 1, "GPT-2 BPE EOS id mismatch");

    const auto decoded = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );
    expect(decoded == prompt, "GPT-2 BPE encode/decode round-trip mismatch");
}

void test_encodes_gpt2_bpe_with_space_prefix() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_bpe_tokenizer_metadata(true))
    );

    const auto ids = tokenizer.encode("Hello", TokenizerEncodeOptions {.add_bos = false, .add_eos = false});

    expect(ids.size() == 1, "GPT-2 BPE space-prefix encoded count mismatch");
    expect(ids[0] == 23, "GPT-2 BPE should use space-prefixed token");
    expect(
        tokenizer.decode(std::span<const std::uint32_t>(ids)) == " Hello",
        "GPT-2 BPE space-prefix decode mismatch"
    );
}

void test_encodes_tokenizer_special_tokens_when_enabled() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_special_tokenizer_metadata())
    );

    const auto ids = tokenizer.encode(
        "<|im_start|>Hello<|im_end|>",
        TokenizerEncodeOptions {.add_bos = false, .add_eos = false, .encode_special_tokens = true}
    );
    const auto ordinary_ids = tokenizer.encode(
        "<|im_start|>",
        TokenizerEncodeOptions {.add_bos = false, .add_eos = false, .encode_special_tokens = false}
    );

    expect(ids.size() == 3, "special-token encode count mismatch");
    expect(ids[0] == 1, "special-token encode im_start mismatch");
    expect(ids[1] == 3, "special-token encode text mismatch");
    expect(ids[2] == 2, "special-token encode im_end mismatch");
    expect(ordinary_ids.size() > 1, "ordinary encode should not collapse control tokens");
    expect(ordinary_ids[0] == 0, "ordinary encode should route literal control text through unknown token");
}

void test_formats_smollm2_generate_prompt_from_chat_template() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.push_back(metadata_string("tokenizer.chat_template", smollm2_chat_template_fixture()));

    const auto formatted = format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");

    expect(formatted.encode_special_tokens, "chat-formatted prompt should enable special-token encoding");
    expect(
        formatted.text == smollm2_formatted_prompt_fixture("Hello"),
        "SmolLM2 chat-formatted prompt mismatch"
    );
}

void test_keeps_raw_generate_prompt_without_chat_template() {
    const auto formatted = format_generate_prompt_from_gguf_metadata(
        metadata_section(tiny_llama_tokenizer_metadata()),
        "Hello"
    );

    expect(!formatted.encode_special_tokens, "raw prompt should not enable special-token encoding");
    expect(formatted.text == "Hello", "raw prompt formatting mismatch");
}

void test_rejects_unsupported_chat_template() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.push_back(metadata_string("tokenizer.chat_template", "{{ messages }}"));

    try {
        (void)format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");
        throw TestFailure("prompt formatter should reject unsupported chat template");
    } catch (const TokenizerError& error) {
        expect(
            error.code() == TokenizerErrorCode::unsupported_chat_template,
            "unsupported chat-template error mismatch"
        );
        expect(error.detail() == "tokenizer.chat_template", "unsupported chat-template detail mismatch");
    }
}

void test_rejects_chat_template_invalid_type() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.push_back(metadata_bool("tokenizer.chat_template", true));

    try {
        (void)format_generate_prompt_from_gguf_metadata(metadata_section(std::move(metadata)), "Hello");
        throw TestFailure("prompt formatter should reject invalid chat-template metadata type");
    } catch (const TokenizerError& error) {
        expect(
            error.code() == TokenizerErrorCode::invalid_metadata_type,
            "invalid chat-template metadata type error mismatch"
        );
        expect(error.detail() == "tokenizer.chat_template", "invalid chat-template metadata detail mismatch");
    }
}

void test_decodes_llama_tokenizer_known_ids_and_skips_special_tokens() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));
    const std::array<std::uint32_t, 5> ids {0, 2, 3, 4, 1};

    const auto decoded = tokenizer.decode(std::span<const std::uint32_t>(ids));
    const auto skipped = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );

    expect(decoded == "<s>Hello world!</s>", "tokenizer decoded text mismatch");
    expect(skipped == "Hello world!", "tokenizer skip-special decoded text mismatch");
}

void test_decodes_gpt2_byte_level_markers() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(
        metadata_section(tiny_gpt2_byte_decoder_tokenizer_metadata())
    );
    const std::array<std::uint32_t, 7> ids {0, 2, 3, 4, 5, 6, 1};

    const auto decoded = tokenizer.decode(std::span<const std::uint32_t>(ids));
    const auto skipped = tokenizer.decode(
        std::span<const std::uint32_t>(ids),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );

    const auto expected = std::string("<s>Hello world!\n") + "\xC3\xA5" + "</s>";
    const auto expected_skipped = std::string("Hello world!\n") + "\xC3\xA5";

    expect(decoded == expected, "GPT-2 byte-level decoded text mismatch");
    expect(skipped == expected_skipped, "GPT-2 byte-level skip-special decoded text mismatch");
}

void test_rejects_tokenizer_decode_id_out_of_range() {
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(tiny_llama_tokenizer_metadata()));
    const std::array<std::uint32_t, 1> ids {7};

    try {
        (void)tokenizer.decode(std::span<const std::uint32_t>(ids));
        throw TestFailure("tokenizer decode should reject invalid token id");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::token_id_out_of_range, "tokenizer invalid id error mismatch");
        expect(error.detail() == "7", "tokenizer invalid id detail mismatch");
    }
}

void test_rejects_tokenizer_malformed_token_metadata() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata[17] = metadata_i32_array("tokenizer.ggml.token_type", {3, 3});

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject token type length mismatch");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_vocabulary, "tokenizer malformed metadata error mismatch");
        expect(error.detail() == "tokenizer.ggml.token_type", "tokenizer malformed metadata detail mismatch");
    }
}

void test_rejects_tokenizer_malformed_bpe_merges() {
    auto metadata = tiny_gpt2_bpe_tokenizer_metadata(false);
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.ggml.merges") {
            entry = metadata_string_array("tokenizer.ggml.merges", {"bad-merge"});
            break;
        }
    }

    try {
        (void)load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));
        throw TestFailure("tokenizer should reject malformed BPE merge metadata");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::invalid_vocabulary, "tokenizer malformed BPE error mismatch");
        expect(error.detail() == "tokenizer.ggml.merges", "tokenizer malformed BPE detail mismatch");
    }
}

void test_rejects_tokenizer_without_matching_token() {
    auto metadata = tiny_llama_tokenizer_metadata();
    metadata.erase(metadata.begin() + 20);
    const auto tokenizer = load_tokenizer_from_gguf_metadata(metadata_section(std::move(metadata)));

    try {
        (void)tokenizer.encode("Hello?");
        throw TestFailure("tokenizer should reject unmatched text without unknown token");
    } catch (const TokenizerError& error) {
        expect(error.code() == TokenizerErrorCode::no_matching_token, "tokenizer no-match error mismatch");
        expect(error.detail() == "byte offset 5", "tokenizer no-match detail mismatch");
    }
}

void test_parses_default_benchmark_args() {
    const char* argv[] = {"local-gguf-bench"};
    const auto parsed = parse_benchmark_args(1, argv);

    expect(!parsed.should_exit, "benchmark default args should not exit");
    expect(parsed.config.output_path == default_benchmark_output_path(), "benchmark default output path mismatch");
    expect(!parsed.config.model_path.has_value(), "benchmark default model path should be absent");
    expect(parsed.config.iterations == 10, "benchmark default iteration count mismatch");
}

void test_parses_benchmark_model_output_and_iterations() {
    const char* argv[] = {
        "local-gguf-bench",
        "--model",
        "models/example.gguf",
        "--output",
        "benchmarks/results/example.txt",
        "--iterations",
        "3",
    };
    const auto parsed = parse_benchmark_args(7, argv);

    expect(!parsed.should_exit, "benchmark explicit args should not exit");
    expect(parsed.config.model_path.has_value(), "benchmark model path should be present");
    expect(*parsed.config.model_path == std::filesystem::path("models/example.gguf"), "benchmark model path mismatch");
    expect(
        parsed.config.output_path == std::filesystem::path("benchmarks/results/example.txt"),
        "benchmark output path mismatch"
    );
    expect(parsed.config.iterations == 3, "benchmark explicit iteration count mismatch");
}

void test_rejects_benchmark_output_outside_results() {
    const char* argv[] = {"local-gguf-bench", "--output", "benchmark-output.txt"};
    const auto parsed = parse_benchmark_args(3, argv);

    expect(parsed.should_exit, "benchmark output outside results should exit");
    expect(parsed.exit_code == EXIT_FAILURE, "benchmark output outside results exit code mismatch");
    expect(
        parsed.message.find("benchmarks/results") != std::string::npos,
        "benchmark output outside results error missing"
    );
}

void test_rejects_invalid_benchmark_iterations() {
    const char* argv[] = {"local-gguf-bench", "--iterations", "0"};
    const auto parsed = parse_benchmark_args(3, argv);

    expect(parsed.should_exit, "benchmark zero iterations should exit");
    expect(parsed.exit_code == EXIT_FAILURE, "benchmark zero iterations exit code mismatch");
    expect(
        parsed.message.find("--iterations requires a positive count") != std::string::npos,
        "benchmark zero iterations error missing"
    );
}

void test_checks_benchmark_output_path_scope() {
    expect(
        benchmark_output_path_is_under_results("benchmarks/results/latest.txt"),
        "benchmark output path under results should be accepted"
    );
    expect(
        benchmark_output_path_is_under_results("benchmarks/results/nested/run.txt"),
        "benchmark nested output path under results should be accepted"
    );
    expect(
        !benchmark_output_path_is_under_results("benchmarks/results/../latest.txt"),
        "benchmark normalized path outside results should be rejected"
    );
    expect(
        !benchmark_output_path_is_under_results("/tmp/local-gguf-bench.txt"),
        "benchmark absolute output path should be rejected"
    );
}

void test_formats_benchmark_report() {
    BenchmarkReport report {
        .config = BenchmarkConfig {
            .output_path = std::filesystem::path("benchmarks/results/test.txt"),
            .model_path = std::filesystem::path("models/example.gguf"),
            .iterations = 2,
        },
        .measurements = {
            BenchmarkMeasurement {
                .name = "tensor.add_f32",
                .iterations = 2,
                .total_ms = 3.0,
                .min_ms = 1.0,
                .max_ms = 2.0,
                .bytes_processed = 96,
                .checksum = 42,
                .detail = "shape=[8] dtype=F32",
            },
        },
    };

    const auto text = format_benchmark_report(report);

    expect(text.find("local GGUF benchmark report\n") == 0, "benchmark report heading mismatch");
    expect(text.find("iterations: 2\n") != std::string::npos, "benchmark report iterations missing");
    expect(
        text.find("output: benchmarks/results/test.txt\n") != std::string::npos,
        "benchmark report output path missing"
    );
    expect(text.find("model: models/example.gguf\n") != std::string::npos, "benchmark report model missing");
    expect(text.find("- tensor.add_f32\n") != std::string::npos, "benchmark report measurement missing");
    expect(text.find("  avg_ms: 1.500\n") != std::string::npos, "benchmark report average missing");
    expect(text.find("  bytes_processed: 96\n") != std::string::npos, "benchmark report bytes missing");
    expect(text.find("  checksum: 42\n") != std::string::npos, "benchmark report checksum missing");
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"default backend starts on CPU", test_default_backend_starts_on_cpu},
        {"tensor shape accepts valid shapes", test_tensor_shape_accepts_valid_shapes},
        {"rejects tensor shape rank over four", test_rejects_tensor_shape_rank_over_four},
        {"rejects tensor shape element count overflow", test_rejects_tensor_shape_element_count_overflow},
        {"calculates tensor byte sizes", test_calculates_tensor_byte_sizes},
        {"rejects unsupported tensor dtype", test_rejects_unsupported_tensor_dtype},
        {"rejects invalid quantized tensor element count", test_rejects_invalid_quantized_tensor_element_count},
        {"rejects tensor byte size overflow", test_rejects_tensor_byte_size_overflow},
        {"creates tensor descriptor from shape and dtype", test_creates_tensor_descriptor_from_shape_and_dtype},
        {"allocates zero initialized tensor storage", test_allocates_zero_initialized_tensor_storage},
        {"copies tensor storage from bytes", test_copies_tensor_storage_from_bytes},
        {"rejects tensor storage byte count mismatch", test_rejects_tensor_storage_byte_count_mismatch},
        {
            "rejects tensor descriptor with invalid shape dtype combination",
            test_rejects_tensor_descriptor_with_invalid_shape_dtype_combination,
        },
        {"decodes Q8_0 block values", test_decodes_q8_0_block_values},
        {"views loaded GGUF Q8_0 block", test_views_loaded_gguf_q8_0_block},
        {
            "rejects Q8_0 block access for unquantized dtype",
            test_rejects_q8_0_block_access_for_unquantized_dtype,
        },
        {"rejects Q8_0 block index out of range", test_rejects_q8_0_block_index_out_of_range},
        {
            "rejects Q8_0 block element index out of range",
            test_rejects_q8_0_block_element_index_out_of_range,
        },
        {"calculates row-major linear indices", test_calculates_row_major_linear_indices},
        {"calculates unquantized element byte offsets", test_calculates_unquantized_element_byte_offsets},
        {"reads and mutates unquantized element bytes", test_reads_and_mutates_unquantized_element_bytes},
        {"rejects tensor index rank mismatch", test_rejects_tensor_index_rank_mismatch},
        {"rejects tensor index out of range", test_rejects_tensor_index_out_of_range},
        {"rejects quantized tensor element indexing", test_rejects_quantized_tensor_element_indexing},
        {"reads and writes F32 scalars by index", test_reads_and_writes_f32_scalars_by_index},
        {"reads F32 scalars from loaded GGUF payload", test_reads_f32_scalars_from_loaded_gguf_payload},
        {"rejects F32 accessor dtype mismatch", test_rejects_f32_accessor_dtype_mismatch},
        {"rejects F32 accessor invalid index", test_rejects_f32_accessor_invalid_index},
        {"reads and writes F16 scalars by index", test_reads_and_writes_f16_scalars_by_index},
        {"reads F16 scalars from loaded GGUF payload", test_reads_f16_scalars_from_loaded_gguf_payload},
        {"rejects F16 accessor dtype mismatch", test_rejects_f16_accessor_dtype_mismatch},
        {"rejects F16 accessor invalid index", test_rejects_f16_accessor_invalid_index},
        {"adds F32 tensors elementwise", test_adds_f32_tensors_elementwise},
        {"rejects F32 add dtype mismatch", test_rejects_f32_add_dtype_mismatch},
        {"rejects F32 add shape mismatch", test_rejects_f32_add_shape_mismatch},
        {"multiplies F32 tensors elementwise", test_multiplies_f32_tensors_elementwise},
        {"rejects F32 multiply dtype mismatch", test_rejects_f32_multiply_dtype_mismatch},
        {"rejects F32 multiply shape mismatch", test_rejects_f32_multiply_shape_mismatch},
        {"applies F32 SiLU", test_applies_f32_silu},
        {"rejects F32 SiLU dtype mismatch", test_rejects_f32_silu_dtype_mismatch},
        {"fills F32 tensor with scalar", test_fills_f32_tensor_with_scalar},
        {"rejects F32 fill dtype mismatch", test_rejects_f32_fill_dtype_mismatch},
        {"copies F32 tensor storage", test_copies_f32_tensor_storage},
        {"rejects F32 copy dtype mismatch", test_rejects_f32_copy_dtype_mismatch},
        {"reduces F32 tensor sum and sum squares", test_reduces_f32_tensor_sum_and_sum_squares},
        {"reduces loaded GGUF F32 payload", test_reduces_loaded_gguf_f32_payload},
        {"rejects F32 reduction dtype mismatch", test_rejects_f32_reduction_dtype_mismatch},
        {"calculates F32 mean square and RMS", test_calculates_f32_mean_square_and_rms},
        {"rejects F32 RMS helpers dtype mismatch", test_rejects_f32_rms_helpers_dtype_mismatch},
        {"applies F32 RMSNorm to vector", test_applies_f32_rms_norm_to_vector},
        {"applies F32 RMSNorm to batched rows", test_applies_f32_rms_norm_to_batched_rows},
        {"rejects F32 RMSNorm dtype mismatch", test_rejects_f32_rms_norm_dtype_mismatch},
        {"rejects F32 RMSNorm input rank above two", test_rejects_f32_rms_norm_input_rank_above_two},
        {"rejects F32 RMSNorm weight rank mismatch", test_rejects_f32_rms_norm_weight_rank_mismatch},
        {
            "rejects F32 RMSNorm hidden dimension mismatch",
            test_rejects_f32_rms_norm_hidden_dimension_mismatch,
        },
        {"rejects F32 RMSNorm nonpositive epsilon", test_rejects_f32_rms_norm_nonpositive_epsilon},
        {"applies F32 RoPE to vector", test_applies_f32_rope_to_vector},
        {"applies F32 RoPE to batched rows", test_applies_f32_rope_to_batched_rows},
        {
            "applies F32 RoPE to batched rows with position start",
            test_applies_f32_rope_to_batched_rows_with_position_start,
        },
        {"applies F32 RoPE frequency base by pair", test_applies_f32_rope_frequency_base_by_pair},
        {"rejects F32 RoPE dtype mismatch", test_rejects_f32_rope_dtype_mismatch},
        {"rejects F32 RoPE input rank above two", test_rejects_f32_rope_input_rank_above_two},
        {"rejects F32 RoPE invalid dimension count", test_rejects_f32_rope_invalid_dimension_count},
        {"rejects F32 RoPE dimension count above hidden", test_rejects_f32_rope_dimension_count_above_hidden},
        {"rejects F32 RoPE nonpositive frequency base", test_rejects_f32_rope_nonpositive_frequency_base},
        {"rejects F32 RoPE row position overflow", test_rejects_f32_rope_row_position_overflow},
        {"applies F32 softmax to vector", test_applies_f32_softmax_to_vector},
        {"applies F32 softmax to batched rows", test_applies_f32_softmax_to_batched_rows},
        {"applies F32 softmax to large values", test_applies_f32_softmax_to_large_values},
        {"rejects F32 softmax dtype mismatch", test_rejects_f32_softmax_dtype_mismatch},
        {"rejects F32 softmax input rank above two", test_rejects_f32_softmax_input_rank_above_two},
        {"applies F32 scaled dot product attention", test_applies_f32_scaled_dot_product_attention},
        {
            "applies F32 scaled dot product attention with distinct value width",
            test_applies_f32_scaled_dot_product_attention_with_distinct_value_width,
        },
        {
            "applies F32 causal scaled dot product attention",
            test_applies_f32_causal_scaled_dot_product_attention,
        },
        {
            "applies F32 causal attention with query position offset",
            test_applies_f32_causal_attention_with_query_position_offset,
        },
        {
            "applies F32 causal attention with column views",
            test_applies_f32_causal_attention_with_column_views,
        },
        {
            "applies F32 causal attention with RoPE query column view",
            test_applies_f32_causal_attention_with_rope_query_column_view,
        },
        {
            "rejects F32 causal attention column view row mismatch",
            test_rejects_f32_causal_attention_column_view_row_mismatch,
        },
        {
            "rejects F32 causal attention RoPE query view invalid dimension count",
            test_rejects_f32_causal_attention_rope_query_view_invalid_dimension_count,
        },
        {
            "rejects F32 causal attention position window overflow",
            test_rejects_f32_causal_attention_position_window_overflow,
        },
        {"rejects F32 attention dtype mismatch", test_rejects_f32_attention_dtype_mismatch},
        {"rejects F32 attention non-matrix rank", test_rejects_f32_attention_non_matrix_rank},
        {
            "rejects F32 attention query/key hidden mismatch",
            test_rejects_f32_attention_query_key_hidden_mismatch,
        },
        {
            "rejects F32 attention key/value row mismatch",
            test_rejects_f32_attention_key_value_row_mismatch,
        },
        {"rejects F32 attention nonpositive scale", test_rejects_f32_attention_nonpositive_scale},
        {"multiplies F32 matrices", test_multiplies_f32_matrices},
        {"rejects F32 matmul dtype mismatch", test_rejects_f32_matmul_dtype_mismatch},
        {"rejects F32 matmul non-matrix rank", test_rejects_f32_matmul_non_matrix_rank},
        {"rejects F32 matmul inner dimension mismatch", test_rejects_f32_matmul_inner_dimension_mismatch},
        {"multiplies F32 by F16 matrices into F32", test_multiplies_f32_by_f16_matrices_into_f32},
        {"multiplies F32 by F16 GGUF weight matrix", test_multiplies_f32_by_f16_gguf_weight_matrix},
        {
            "multiplies F32 by F16 GGUF weight matrix on CPU backend",
            test_multiplies_f32_by_f16_gguf_weight_matrix_on_cpu_backend,
        },
        {
            "copies tensor to and from CPU backend buffer",
            test_copies_tensor_to_and_from_cpu_backend_buffer,
        },
        {
            "multiplies F32 by CPU backend F16 GGUF weight buffer",
            test_multiplies_f32_by_cpu_backend_f16_gguf_weight_buffer,
        },
        {
            "multiplies backend F32 by backend F16 GGUF weight to CPU backend buffer",
            test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_cpu_backend_buffer,
        },
        {
            "multiplies backend F32 by three backend F16 GGUF weights to CPU backend buffers",
            test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_cpu_backend_buffers,
        },
        {
            "applies SiLU multiply to CPU backend buffers",
            test_applies_silu_multiply_to_cpu_backend_buffers,
        },
        {
            "runs gated SiLU projection to CPU backend buffer",
            test_runs_gated_silu_projection_to_cpu_backend_buffer,
        },
        {
            "runs residual gated SiLU projection to CPU backend buffer",
            test_runs_residual_gated_silu_projection_to_cpu_backend_buffer,
        },
        {
            "matches CPU F32 by F16 GGUF weight matmul on Metal backend",
            test_matches_cpu_f32_by_f16_gguf_weight_matmul_on_metal_backend,
        },
        {
            "copies tensor to and from Metal backend buffer when available",
            test_copies_tensor_to_and_from_metal_backend_buffer_when_available,
        },
        {
            "multiplies F32 by Metal backend F16 GGUF weight buffer when available",
            test_multiplies_f32_by_metal_backend_f16_gguf_weight_buffer_when_available,
        },
        {
            "multiplies backend F32 by backend F16 GGUF weight to Metal backend buffer when available",
            test_multiplies_backend_f32_by_backend_f16_gguf_weight_to_metal_backend_buffer_when_available,
        },
        {
            "multiplies backend F32 by three backend F16 GGUF weights to Metal backend buffers when available",
            test_multiplies_backend_f32_by_three_backend_f16_gguf_weights_to_metal_backend_buffers_when_available,
        },
        {
            "applies SiLU multiply to Metal backend buffers when available",
            test_applies_silu_multiply_to_metal_backend_buffers_when_available,
        },
        {
            "runs gated SiLU projection to Metal backend buffer when available",
            test_runs_gated_silu_projection_to_metal_backend_buffer_when_available,
        },
        {
            "runs residual gated SiLU projection to Metal backend buffer when available",
            test_runs_residual_gated_silu_projection_to_metal_backend_buffer_when_available,
        },
        {
            "runs Llama cached attention to CPU backend buffer",
            test_runs_llama_cached_attention_to_cpu_backend_buffer,
        },
        {
            "matches Llama cached attention to Metal backend buffer when available",
            test_matches_llama_cached_attention_to_metal_backend_buffer_when_available,
        },
        {
            "matches one-token grouped Llama cached attention to Metal backend buffer when available",
            test_matches_llama_one_token_grouped_cached_attention_to_metal_backend_buffer_when_available,
        },
        {
            "reuses Metal F32 by F16 GGUF weight matmul state",
            test_reuses_metal_f32_by_f16_gguf_weight_matmul_state,
        },
        {
            "reports unavailable Metal backend for GGUF matmul when missing",
            test_reports_unavailable_metal_backend_for_gguf_matmul_when_missing,
        },
        {
            "reports unavailable Metal backend buffer copy when missing",
            test_reports_unavailable_metal_backend_buffer_copy_when_missing,
        },
        {"rejects F32xF16 matmul right dtype mismatch", test_rejects_f32_f16_matmul_right_dtype_mismatch},
        {"creates Llama KV cache", test_creates_llama_kv_cache},
        {"appends and reads Llama KV cache layer", test_appends_and_reads_llama_kv_cache_layer},
        {
            "rejects Llama KV cache append position mismatch",
            test_rejects_llama_kv_cache_append_position_mismatch,
        },
        {"rejects Llama KV cache context overflow", test_rejects_llama_kv_cache_context_overflow},
        {"rejects Llama KV cache key shape mismatch", test_rejects_llama_kv_cache_key_shape_mismatch},
        {"runs Llama attention layer with F16 weights", test_runs_llama_attention_layer_with_f16_weights},
        {
            "runs Llama attention layer with CPU backend weight buffers",
            test_runs_llama_attention_layer_with_cpu_backend_weight_buffers,
        },
        {
            "runs Llama attention layer with KV cache and CPU backend weight buffers",
            test_runs_llama_attention_layer_with_kv_cache_and_cpu_backend_weight_buffers,
        },
        {"runs Llama attention layer with causal mask", test_runs_llama_attention_layer_with_causal_mask},
        {
            "runs Llama attention layer with KV cache matches no-cache",
            test_runs_llama_attention_layer_with_kv_cache_matches_no_cache,
        },
        {
            "runs grouped Llama attention layer with KV cache matches no-cache",
            test_runs_grouped_llama_attention_layer_with_kv_cache_matches_no_cache,
        },
        {
            "runs Llama attention layer with KV cache appends rotated key heads",
            test_runs_llama_attention_layer_with_kv_cache_appends_rotated_key_heads,
        },
        {
            "rejects Llama attention layer with KV cache invalid RoPE dimension count",
            test_rejects_llama_attention_layer_with_kv_cache_invalid_rope_dimension_count,
        },
        {
            "runs Llama attention layer with rowwise RoPE positions",
            test_runs_llama_attention_layer_with_rowwise_rope_positions,
        },
        {
            "rejects Llama attention layer position before input window",
            test_rejects_llama_attention_layer_position_before_input_window,
        },
        {
            "rejects Llama attention layer projection shape mismatch",
            test_rejects_llama_attention_layer_projection_shape_mismatch,
        },
        {
            "rejects Llama attention layer projection dtype mismatch",
            test_rejects_llama_attention_layer_projection_dtype_mismatch,
        },
        {
            "rejects Llama attention layer invalid head layout",
            test_rejects_llama_attention_layer_invalid_head_layout,
        },
        {"runs Llama FFN layer with F16 weights", test_runs_llama_feed_forward_layer_with_f16_weights},
        {
            "runs Llama FFN layer with CPU backend weight buffers",
            test_runs_llama_feed_forward_layer_with_cpu_backend_weight_buffers,
        },
        {
            "rejects Llama FFN layer projection shape mismatch",
            test_rejects_llama_feed_forward_layer_projection_shape_mismatch,
        },
        {
            "rejects Llama FFN layer projection dtype mismatch",
            test_rejects_llama_feed_forward_layer_projection_dtype_mismatch,
        },
        {"rejects Llama FFN layer invalid config", test_rejects_llama_feed_forward_layer_invalid_config},
        {"runs Llama block with F16 weights", test_runs_llama_block_with_f16_weights},
        {
            "runs Llama block with CPU backend weight buffers",
            test_runs_llama_block_with_cpu_backend_weight_buffers,
        },
        {
            "runs Llama block with KV cache and CPU backend weight buffers",
            test_runs_llama_block_with_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama block with KV cache and Metal backend weight buffers when available",
            test_matches_llama_block_with_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "runs Llama block with backend KV cache and CPU backend weight buffers",
            test_runs_llama_block_with_backend_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama block with backend KV cache and Metal backend weight buffers when available",
            test_matches_llama_block_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "rejects Llama block attention projection shape mismatch",
            test_rejects_llama_block_attention_projection_shape_mismatch,
        },
        {
            "binds and runs Llama model blocks from named tensors",
            test_binds_and_runs_llama_model_blocks_from_named_tensors,
        },
        {
            "copies Llama model attention weights to CPU backend buffers",
            test_copies_llama_model_attention_weights_to_cpu_backend_buffers,
        },
        {
            "matches Llama attention layer with Metal backend weight buffers when available",
            test_matches_llama_attention_layer_with_metal_backend_weight_buffers_when_available,
        },
        {
            "matches Llama attention layer with KV cache and Metal backend weight buffers when available",
            test_matches_llama_attention_layer_with_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama model attention backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_attention_backend_weight_copy_when_metal_missing,
        },
        {
            "copies Llama model FFN weights to CPU backend buffers",
            test_copies_llama_model_feed_forward_weights_to_cpu_backend_buffers,
        },
        {
            "matches Llama FFN layer with Metal backend weight buffers when available",
            test_matches_llama_feed_forward_layer_with_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama model FFN backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_feed_forward_backend_weight_copy_when_metal_missing,
        },
        {
            "copies Llama model block weights to CPU backend buffers",
            test_copies_llama_model_block_weights_to_cpu_backend_buffers,
        },
        {
            "matches Llama model blocks with Metal backend weight buffers when available",
            test_matches_llama_model_blocks_with_metal_backend_weight_buffers_when_available,
        },
        {
            "runs Llama model blocks with KV cache and CPU backend weight buffers",
            test_runs_llama_model_blocks_with_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama model blocks with KV cache and Metal backend weight buffers when available",
            test_matches_llama_model_blocks_with_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "runs Llama model blocks with backend KV cache and CPU backend weight buffers",
            test_runs_llama_model_blocks_with_backend_kv_cache_and_cpu_backend_weight_buffers,
        },
        {
            "matches Llama model blocks with backend KV cache and Metal backend weight buffers when available",
            test_matches_llama_model_blocks_with_backend_kv_cache_and_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama backend KV cache creation when Metal is missing",
            test_reports_unavailable_llama_backend_kv_cache_creation_when_metal_missing,
        },
        {
            "reports unavailable Llama model blocks with KV cache backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_blocks_with_kv_cache_backend_weight_copy_when_metal_missing,
        },
        {
            "reports unavailable Llama model block backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_block_backend_weight_copy_when_metal_missing,
        },
        {
            "projects Llama logits with CPU backend output weight buffer",
            test_projects_llama_logits_with_cpu_backend_output_weight_buffer,
        },
        {
            "runs Llama model logits with CPU backend weight buffers",
            test_runs_llama_model_logits_with_cpu_backend_weight_buffers,
        },
        {
            "matches Llama model logits with Metal backend weight buffers when available",
            test_matches_llama_model_logits_with_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama model logits backend weight copy when Metal is missing",
            test_reports_unavailable_llama_model_logits_backend_weight_copy_when_metal_missing,
        },
        {
            "rejects Llama model weight binding missing tensor",
            test_rejects_llama_model_weight_binding_missing_tensor,
        },
        {
            "binds Llama model missing output weight to token embedding",
            test_binds_llama_model_missing_output_weight_to_token_embedding,
        },
        {
            "rejects Llama model blocks projection dtype mismatch",
            test_rejects_llama_model_blocks_projection_dtype_mismatch,
        },
        {"embeds Llama token IDs from F16 weight", test_embeds_llama_token_ids_from_f16_weight},
        {
            "rejects Llama token embedding ID out of range",
            test_rejects_llama_token_embedding_id_out_of_range,
        },
        {"projects Llama logits with final norm", test_projects_llama_logits_with_final_norm},
        {"runs Llama model logits from token IDs", test_runs_llama_model_logits_from_token_ids},
        {
            "runs Llama model logits with KV cache matches no-cache",
            test_runs_llama_model_logits_with_kv_cache_matches_no_cache,
        },
        {
            "rejects Llama model KV cache layer count mismatch",
            test_rejects_llama_model_kv_cache_layer_count_mismatch,
        },
        {
            "rejects Llama logits output weight dtype mismatch",
            test_rejects_llama_logits_output_weight_dtype_mismatch,
        },
        {"samples argmax token from F32 logits vector", test_samples_argmax_token_from_f32_logits_vector},
        {"samples argmax token from last logits row", test_samples_argmax_token_from_last_logits_row},
        {"selects top logit tokens from last row", test_selects_top_logit_tokens_from_last_row},
        {"rejects argmax sampler dtype mismatch", test_rejects_argmax_sampler_dtype_mismatch},
        {"rejects argmax sampler rank above two", test_rejects_argmax_sampler_rank_above_two},
        {
            "samples temperature top-k token from F32 logits vector",
            test_samples_temperature_top_k_token_from_f32_logits_vector,
        },
        {
            "samples temperature top-k token from last logits row",
            test_samples_temperature_top_k_token_from_last_logits_row,
        },
        {
            "samples temperature top-p token from F32 logits vector",
            test_samples_temperature_top_p_token_from_f32_logits_vector,
        },
        {
            "samples temperature repetition-penalty token from F32 logits vector",
            test_samples_temperature_repetition_penalty_token_from_f32_logits_vector,
        },
        {"sampling temperature zero keeps argmax path", test_sampling_temperature_zero_keeps_argmax_path},
        {"rejects sampling negative temperature", test_rejects_sampling_negative_temperature},
        {"rejects sampling invalid top-p", test_rejects_sampling_invalid_top_p},
        {"rejects sampling invalid repetition penalty", test_rejects_sampling_invalid_repetition_penalty},
        {"runs Llama next-token argmax from token IDs", test_runs_llama_next_token_argmax_from_token_ids},
        {"rejects Llama next-token empty token IDs", test_rejects_llama_next_token_empty_token_ids},
        {
            "runs Llama next-token argmax with CPU backend weight buffers",
            test_runs_llama_next_token_argmax_with_cpu_backend_weight_buffers,
        },
        {
            "runs Llama next-token sampled with CPU backend weight buffers",
            test_runs_llama_next_token_sampled_with_cpu_backend_weight_buffers,
        },
        {
            "matches Llama next-token argmax with Metal backend weight buffers when available",
            test_matches_llama_next_token_argmax_with_metal_backend_weight_buffers_when_available,
        },
        {
            "matches Llama next-token sampled with Metal backend weight buffers when available",
            test_matches_llama_next_token_sampled_with_metal_backend_weight_buffers_when_available,
        },
        {
            "reports unavailable Llama next-token backend weight copy when Metal is missing",
            test_reports_unavailable_llama_next_token_backend_weight_copy_when_metal_missing,
        },
        {"generates Llama argmax tokens", test_generates_llama_argmax_tokens},
        {"generates Llama argmax tokens stops at token", test_generates_llama_argmax_tokens_stops_at_token},
        {
            "generates Llama argmax tokens with KV cache matches no-cache",
            test_generates_llama_argmax_tokens_with_kv_cache_matches_no_cache,
        },
        {
            "generates Llama argmax tokens with KV cache stops at token",
            test_generates_llama_argmax_tokens_with_kv_cache_stops_at_token,
        },
        {
            "generates Llama sampled tokens with KV cache matches no-cache",
            test_generates_llama_sampled_tokens_with_kv_cache_matches_no_cache,
        },
        {
            "rejects Llama cached generation zero new tokens",
            test_rejects_llama_cached_generation_zero_new_tokens,
        },
        {
            "rejects Llama cached generation empty prompt",
            test_rejects_llama_cached_generation_empty_prompt,
        },
        {
            "rejects Llama cached generation context overflow",
            test_rejects_llama_cached_generation_context_overflow,
        },
        {"rejects Llama generation zero new tokens", test_rejects_llama_generation_zero_new_tokens},
        {"rejects Llama generation context overflow", test_rejects_llama_generation_context_overflow},
        {
            "generates text from GGUF bytes with cached argmax",
            test_generates_text_from_gguf_bytes_with_cached_argmax,
        },
        {
            "generates text from GGUF bytes with tied output weight",
            test_generates_text_from_gguf_bytes_with_tied_output_weight,
        },
        {
            "generates text from GGUF bytes stops at tokenizer EOS",
            test_generates_text_from_gguf_bytes_stops_at_tokenizer_eos,
        },
        {
            "generates text from GGUF bytes with chat-template prompt",
            test_generates_text_from_gguf_bytes_with_chat_template_prompt,
        },
        {
            "generates text from GGUF bytes with sampling options",
            test_generates_text_from_gguf_bytes_with_sampling_options,
        },
        {"runs generate success path", test_runs_generate_success_path},
        {"runs generate with greedy option", test_runs_generate_with_greedy_option},
        {"runs generate with sampling options", test_runs_generate_with_sampling_options},
        {"runs generate with top-p option", test_runs_generate_with_top_p_option},
        {"runs generate with first-token debug", test_runs_generate_with_first_token_debug},
        {"rejects generate missing arguments", test_rejects_generate_missing_arguments},
        {"rejects generate invalid max new tokens", test_rejects_generate_invalid_max_new_tokens},
        {"rejects generate invalid temperature option", test_rejects_generate_invalid_temperature_option},
        {"rejects generate invalid top-p option", test_rejects_generate_invalid_top_p_option},
        {
            "rejects generate invalid repeat-penalty option",
            test_rejects_generate_invalid_repeat_penalty_option,
        },
        {"rejects generate greedy sampling conflict", test_rejects_generate_greedy_sampling_conflict},
        {"converts GGUF payload view to tensor storage", test_converts_gguf_payload_view_to_tensor_storage},
        {
            "converts Q8_0 GGUF payload view to tensor storage",
            test_converts_q8_0_gguf_payload_view_to_tensor_storage,
        },
        {"rejects unsupported GGUF tensor type in bridge", test_rejects_unsupported_gguf_tensor_type_in_bridge},
        {
            "rejects invalid quantized GGUF tensor shape in bridge",
            test_rejects_invalid_quantized_gguf_tensor_shape_in_bridge,
        },
        {
            "rejects GGUF tensor payload size mismatch in bridge",
            test_rejects_gguf_tensor_payload_size_mismatch_in_bridge,
        },
        {"reads valid GGUF fixed header fixture", test_reads_valid_gguf_fixed_header_fixture},
        {"reads little-endian GGUF header counts", test_reads_little_endian_header_counts},
        {"rejects invalid GGUF magic", test_rejects_invalid_gguf_magic},
        {"rejects truncated GGUF header", test_rejects_truncated_gguf_header},
        {"rejects truncated GGUF count data", test_rejects_truncated_count_data},
        {"reads scalar GGUF metadata entries", test_reads_scalar_metadata_entries},
        {"reads all scalar GGUF metadata value types", test_reads_all_scalar_metadata_value_types},
        {"reads string array GGUF metadata entry", test_reads_string_array_metadata_entry},
        {"rejects invalid GGUF metadata value type", test_rejects_invalid_metadata_value_type},
        {"rejects invalid GGUF metadata bool", test_rejects_invalid_metadata_bool},
        {"rejects nested GGUF metadata array", test_rejects_nested_metadata_array},
        {"rejects truncated GGUF metadata value", test_rejects_truncated_metadata_value},
        {"reads GGUF tensor directory entries", test_reads_tensor_directory_entries},
        {"reads unpadded GGUF tensor directory with default alignment", test_reads_unpadded_tensor_directory_with_default_alignment},
        {"rejects invalid GGUF tensor dimension count", test_rejects_invalid_tensor_dimension_count},
        {"rejects invalid GGUF tensor name", test_rejects_invalid_tensor_name},
        {"rejects invalid GGUF alignment value", test_rejects_invalid_alignment_value},
        {"rejects invalid GGUF alignment type", test_rejects_invalid_alignment_type},
        {"rejects misaligned GGUF tensor offset", test_rejects_misaligned_tensor_offset},
        {"rejects invalid GGUF tensor padding", test_rejects_invalid_tensor_padding},
        {"rejects truncated GGUF tensor directory", test_rejects_truncated_tensor_directory},
        {"reads GGUF tensor payload views without copying", test_reads_tensor_payload_views_without_copying},
        {"reads Q8_0 GGUF tensor payload view", test_reads_q8_0_tensor_payload_view},
        {"reads F16 GGUF tensor payload view", test_reads_f16_tensor_payload_view},
        {"rejects truncated GGUF tensor payload", test_rejects_truncated_tensor_payload},
        {"rejects unsupported GGUF tensor payload type", test_rejects_unsupported_tensor_payload_type},
        {"rejects invalid quantized GGUF tensor payload size", test_rejects_invalid_quantized_tensor_payload_size},
        {"inspects GGUF summary from F16 payload", test_inspects_gguf_summary_from_f16_payload},
        {"formats GGUF inspection summary", test_formats_gguf_inspection_summary},
        {"runs gguf-inspect success path", test_runs_gguf_inspect_success},
        {"rejects gguf-inspect missing path", test_rejects_gguf_inspect_missing_path},
        {"rejects gguf-inspect unreadable path", test_rejects_gguf_inspect_unreadable_path},
        {"reports invalid GGUF in gguf-inspect", test_reports_invalid_gguf_in_gguf_inspect},
        {
            "reports unsupported tensor type in gguf-inspect",
            test_reports_unsupported_tensor_type_in_gguf_inspect,
        },
        {"reports malformed tensor payload in gguf-inspect", test_reports_malformed_tensor_payload_in_gguf_inspect},
        {"loads engine model from Llama GGUF view", test_loads_engine_model_from_llama_gguf_view},
        {
            "rejects engine model without architecture metadata",
            test_rejects_engine_model_without_architecture_metadata,
        },
        {"rejects engine model unsupported architecture", test_rejects_engine_model_unsupported_architecture},
        {"rejects engine model missing Llama metadata", test_rejects_engine_model_missing_llama_metadata},
        {"rejects engine model invalid metadata type", test_rejects_engine_model_invalid_metadata_type},
        {"rejects engine model duplicate tensor names", test_rejects_engine_model_duplicate_tensor_names},
        {"loads Llama tokenizer from metadata", test_loads_llama_tokenizer_from_metadata},
        {
            "encodes Llama tokenizer known IDs with special tokens",
            test_encodes_llama_tokenizer_known_ids_with_special_tokens,
        },
        {"encodes GPT-2 BPE prompt pieces", test_encodes_gpt2_bpe_prompt_pieces},
        {"encodes GPT-2 BPE with space prefix", test_encodes_gpt2_bpe_with_space_prefix},
        {"encodes tokenizer special tokens when enabled", test_encodes_tokenizer_special_tokens_when_enabled},
        {
            "formats SmolLM2 generate prompt from chat template",
            test_formats_smollm2_generate_prompt_from_chat_template,
        },
        {"keeps raw generate prompt without chat template", test_keeps_raw_generate_prompt_without_chat_template},
        {"rejects unsupported chat template", test_rejects_unsupported_chat_template},
        {"rejects chat template invalid type", test_rejects_chat_template_invalid_type},
        {
            "decodes Llama tokenizer known IDs and skips special tokens",
            test_decodes_llama_tokenizer_known_ids_and_skips_special_tokens,
        },
        {"decodes GPT-2 byte-level markers", test_decodes_gpt2_byte_level_markers},
        {"rejects tokenizer decode ID out of range", test_rejects_tokenizer_decode_id_out_of_range},
        {
            "rejects tokenizer malformed token metadata",
            test_rejects_tokenizer_malformed_token_metadata,
        },
        {"rejects tokenizer malformed BPE merges", test_rejects_tokenizer_malformed_bpe_merges},
        {"rejects tokenizer without matching token", test_rejects_tokenizer_without_matching_token},
        {"parses default benchmark args", test_parses_default_benchmark_args},
        {
            "parses benchmark model output and iterations",
            test_parses_benchmark_model_output_and_iterations,
        },
        {"rejects benchmark output outside results", test_rejects_benchmark_output_outside_results},
        {"rejects invalid benchmark iterations", test_rejects_invalid_benchmark_iterations},
        {"checks benchmark output path scope", test_checks_benchmark_output_path_scope},
        {"formats benchmark report", test_formats_benchmark_report},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << std::size(tests) << " test(s) passed\n";
    return EXIT_SUCCESS;
}
