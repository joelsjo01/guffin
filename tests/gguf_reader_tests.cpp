#include "gguf_reader_tests.hpp"

#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"

#include <array>
#include <cstdint>
#include <span>
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
using local_gguf_engine::model_format::read_gguf_header;
using local_gguf_engine::model_format::read_gguf_metadata_section;
using local_gguf_engine::model_format::read_gguf_model_view;
using local_gguf_engine::model_format::read_gguf_tensor_directory;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::fixture;

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

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gguf_reader_test_cases() {
    static constexpr TestCase tests[] = {
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

    return tests;
}

}  // namespace local_gguf_engine::tests
