#include "engine_model_tests.hpp"

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/tensor/layout.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::engine::ModelArchitecture;
using local_gguf_engine::engine::model_architecture_name;
using local_gguf_engine::model_format::GGUFHeader;
using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::model_format::GGUFMetadataSection;
using local_gguf_engine::model_format::GGUFModelView;
using local_gguf_engine::model_format::GGUFTensorDirectory;
using local_gguf_engine::model_format::GGUFTensorInfo;
using local_gguf_engine::model_format::GGUFTensorPayloadView;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::tiny_llama_metadata;

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

const local_gguf_engine::tests::TestCase kEngineModelTests[] = {
    {"loads engine model from Llama GGUF view", test_loads_engine_model_from_llama_gguf_view},
    {
        "rejects engine model without architecture metadata",
        test_rejects_engine_model_without_architecture_metadata,
    },
    {"rejects engine model unsupported architecture", test_rejects_engine_model_unsupported_architecture},
    {"rejects engine model missing Llama metadata", test_rejects_engine_model_missing_llama_metadata},
    {"rejects engine model invalid metadata type", test_rejects_engine_model_invalid_metadata_type},
    {"rejects engine model duplicate tensor names", test_rejects_engine_model_duplicate_tensor_names},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> engine_model_test_cases() {
    return kEngineModelTests;
}

}  // namespace local_gguf_engine::tests
