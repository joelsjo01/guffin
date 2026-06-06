#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/frontends/cli/gguf_inspect.hpp"
#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::ModelArchitecture;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::engine::model_architecture_name;
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
using local_gguf_engine::frontends::cli::inspect_gguf_bytes;
using local_gguf_engine::frontends::cli::run_gguf_inspect;
using local_gguf_engine::frontends::cli::write_gguf_inspection_summary;
using local_gguf_engine::tensor::byte_size;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::element_byte_offset;
using local_gguf_engine::tensor::element_bytes;
using local_gguf_engine::tensor::element_count;
using local_gguf_engine::tensor::linear_index;
using local_gguf_engine::tensor::mutable_element_bytes;
using local_gguf_engine::tensor::dequantize_q8_0_block;
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
using local_gguf_engine::tensor::write_f16_from_f32;
using local_gguf_engine::tensor::write_f32;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::copy_f32;
using local_gguf_engine::tensor::ops::fill_f32;
using local_gguf_engine::tensor::ops::matmul_f32;
using local_gguf_engine::tensor::ops::matmul_f32_f16;
using local_gguf_engine::tensor::ops::mean_square_f32;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rms_f32;
using local_gguf_engine::tensor::ops::rope_f32;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::sum_f32;
using local_gguf_engine::tensor::ops::sum_squares_f32;

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
        {"applies F32 RoPE frequency base by pair", test_applies_f32_rope_frequency_base_by_pair},
        {"rejects F32 RoPE dtype mismatch", test_rejects_f32_rope_dtype_mismatch},
        {"rejects F32 RoPE input rank above two", test_rejects_f32_rope_input_rank_above_two},
        {"rejects F32 RoPE invalid dimension count", test_rejects_f32_rope_invalid_dimension_count},
        {"rejects F32 RoPE dimension count above hidden", test_rejects_f32_rope_dimension_count_above_hidden},
        {"rejects F32 RoPE nonpositive frequency base", test_rejects_f32_rope_nonpositive_frequency_base},
        {"applies F32 softmax to vector", test_applies_f32_softmax_to_vector},
        {"applies F32 softmax to batched rows", test_applies_f32_softmax_to_batched_rows},
        {"applies F32 softmax to large values", test_applies_f32_softmax_to_large_values},
        {"rejects F32 softmax dtype mismatch", test_rejects_f32_softmax_dtype_mismatch},
        {"rejects F32 softmax input rank above two", test_rejects_f32_softmax_input_rank_above_two},
        {"multiplies F32 matrices", test_multiplies_f32_matrices},
        {"rejects F32 matmul dtype mismatch", test_rejects_f32_matmul_dtype_mismatch},
        {"rejects F32 matmul non-matrix rank", test_rejects_f32_matmul_non_matrix_rank},
        {"rejects F32 matmul inner dimension mismatch", test_rejects_f32_matmul_inner_dimension_mismatch},
        {"multiplies F32 by F16 matrices into F32", test_multiplies_f32_by_f16_matrices_into_f32},
        {"rejects F32xF16 matmul right dtype mismatch", test_rejects_f32_f16_matmul_right_dtype_mismatch},
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
