#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using local_gguf_engine::model_format::GGUFHeader;
using local_gguf_engine::model_format::GGUFMetadataArray;
using local_gguf_engine::model_format::GGUFMetadataValue;
using local_gguf_engine::model_format::GGUFMetadataValueType;
using local_gguf_engine::model_format::GGUFReadError;
using local_gguf_engine::model_format::GGUFReadErrorCode;
using local_gguf_engine::model_format::GGUFTensorInfo;
using local_gguf_engine::model_format::GGUFTensorPayloadView;
using local_gguf_engine::model_format::load_gguf_tensor_payload;
using local_gguf_engine::model_format::load_gguf_tensor_payloads;
using local_gguf_engine::model_format::read_gguf_header;
using local_gguf_engine::model_format::read_gguf_metadata_section;
using local_gguf_engine::model_format::read_gguf_model_view;
using local_gguf_engine::model_format::read_gguf_tensor_directory;
using local_gguf_engine::tensor::byte_size;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::element_byte_offset;
using local_gguf_engine::tensor::element_bytes;
using local_gguf_engine::tensor::element_count;
using local_gguf_engine::tensor::linear_index;
using local_gguf_engine::tensor::mutable_element_bytes;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::write_f32;

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
