#include "gguf_tensor_bridge_tests.hpp"

#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace {

using local_gguf_engine::model_format::GGUFReadError;
using local_gguf_engine::model_format::GGUFReadErrorCode;
using local_gguf_engine::model_format::GGUFTensorInfo;
using local_gguf_engine::model_format::GGUFTensorPayloadView;
using local_gguf_engine::model_format::load_gguf_tensor_payload;
using local_gguf_engine::model_format::load_gguf_tensor_payloads;
using local_gguf_engine::model_format::read_gguf_model_view;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::fixture;
using local_gguf_engine::tests::TestFailure;

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

const local_gguf_engine::tests::TestCase kGGUFTensorBridgeTests[] = {
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
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gguf_tensor_bridge_test_cases() {
    return kGGUFTensorBridgeTests;
}

}  // namespace local_gguf_engine::tests
