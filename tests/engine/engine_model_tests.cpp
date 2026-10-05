#include "engine_model_tests.hpp"

#include "local_gguf_engine/engine/gemma_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/tensor/layout.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::GemmaModelConfig;
using local_gguf_engine::engine::bind_gemma_model_weights;
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
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::TestFailure;

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
    std::vector<std::uint64_t> dimensions,
    std::uint32_t type_id,
    std::span<const std::uint8_t> bytes
) {
    return GGUFTensorPayloadView {
        .tensor = GGUFTensorInfo {
            .name = std::move(name),
            .dimensions = std::move(dimensions),
            .type_id = type_id,
            .offset = 0,
            .file_offset = 0,
        },
        .bytes = bytes,
    };
}

std::uint32_t gguf_type_id_for_dtype(DType dtype) {
    switch (dtype) {
        case DType::f32:
            return 0;
        case DType::f16:
            return 1;
        case DType::q8_0:
            return 8;
        case DType::q4_k:
            return 12;
        case DType::q6_k:
            return 14;
    }

    throw TestFailure("engine model test unsupported dtype");
}

std::vector<std::uint8_t> zero_tensor_bytes(std::span<const std::uint64_t> dimensions, DType dtype) {
    return std::vector<std::uint8_t>(
        TensorDescriptor::from(Shape::from(dimensions), dtype).byte_count
    );
}

std::vector<std::uint8_t> zero_tensor_bytes(std::initializer_list<std::uint64_t> dimensions, DType dtype) {
    return zero_tensor_bytes(std::span<const std::uint64_t>(dimensions.begin(), dimensions.size()), dtype);
}

std::vector<GGUFMetadataEntry> tiny_gemma2_metadata() {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Fixture"),
        metadata_u32("gemma2.context_length", 16),
        metadata_u32("gemma2.embedding_length", 256),
        metadata_u32("gemma2.block_count", 1),
        metadata_u32("gemma2.feed_forward_length", 256),
        metadata_u32("gemma2.attention.head_count", 1),
        metadata_u32("gemma2.attention.head_count_kv", 1),
        metadata_u32("gemma2.attention.key_length", 256),
        metadata_u32("gemma2.attention.value_length", 256),
        metadata_u32("gemma2.attention.sliding_window", 8),
        metadata_f32("gemma2.attention.layer_norm_rms_epsilon", 0.000001F),
        metadata_f32("gemma2.attn_logit_softcapping", 50.0F),
        metadata_f32("gemma2.final_logit_softcapping", 30.0F),
    };
}

struct TinyGemma2PayloadStore {
    std::vector<std::vector<std::uint8_t>> bytes;
    std::vector<GGUFTensorPayloadView> payloads;
};

void add_payload(
    TinyGemma2PayloadStore& store,
    std::string name,
    std::vector<std::uint64_t> dimensions,
    DType dtype
) {
    store.bytes.push_back(zero_tensor_bytes(std::span<const std::uint64_t>(dimensions), dtype));
    store.payloads.push_back(tensor_payload_view(
        std::move(name),
        std::move(dimensions),
        gguf_type_id_for_dtype(dtype),
        std::span<const std::uint8_t>(store.bytes.back())
    ));
}

std::string gemma2_block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

TinyGemma2PayloadStore tiny_gemma2_payload_store(bool include_post_ffw_norm = true) {
    TinyGemma2PayloadStore store;
    store.bytes.reserve(16);
    store.payloads.reserve(16);

    add_payload(store, "token_embd.weight", {256, 4}, DType::f16);
    add_payload(store, "output_norm.weight", {256}, DType::f32);
    add_payload(store, gemma2_block_tensor_name(0, "attn_norm.weight"), {256}, DType::f32);
    add_payload(store, gemma2_block_tensor_name(0, "post_attention_norm.weight"), {256}, DType::f32);
    add_payload(store, gemma2_block_tensor_name(0, "attn_q.weight"), {256, 256}, DType::q4_k);
    add_payload(store, gemma2_block_tensor_name(0, "attn_k.weight"), {256, 256}, DType::q4_k);
    add_payload(store, gemma2_block_tensor_name(0, "attn_v.weight"), {256, 256}, DType::q6_k);
    add_payload(store, gemma2_block_tensor_name(0, "attn_output.weight"), {256, 256}, DType::q4_k);
    add_payload(store, gemma2_block_tensor_name(0, "ffn_norm.weight"), {256}, DType::f32);
    if (include_post_ffw_norm) {
        add_payload(store, gemma2_block_tensor_name(0, "post_ffw_norm.weight"), {256}, DType::f32);
    }
    add_payload(store, gemma2_block_tensor_name(0, "ffn_gate.weight"), {256, 256}, DType::q4_k);
    add_payload(store, gemma2_block_tensor_name(0, "ffn_up.weight"), {256, 256}, DType::q4_k);
    add_payload(store, gemma2_block_tensor_name(0, "ffn_down.weight"), {256, 256}, DType::q6_k);

    return store;
}

void test_loads_engine_model_from_gemma2_gguf_view() {
    auto payload_store = tiny_gemma2_payload_store();
    const auto* token_embedding_bytes = payload_store.payloads[0].bytes.data();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));

    const auto model = load_engine_model_from_gguf(view);

    expect(model.architecture() == ModelArchitecture::gemma2, "Gemma2 engine model architecture mismatch");
    expect(model_architecture_name(model.architecture()) == "gemma2", "Gemma2 architecture name mismatch");
    expect(model.name().has_value(), "Gemma2 engine model name should be present");
    expect(*model.name() == "Tiny Gemma 2 Fixture", "Gemma2 engine model name mismatch");

    const auto expected_config = GemmaModelConfig {
        .context_length = 16,
        .embedding_length = 256,
        .block_count = 1,
        .feed_forward_length = 256,
        .attention_head_count = 1,
        .attention_head_count_kv = 1,
        .attention_key_length = 256,
        .attention_value_length = 256,
        .sliding_window = 8,
        .vocab_size = 4,
        .rms_norm_epsilon = 0.000001F,
        .attention_logit_softcap = 50.0F,
        .final_logit_softcap = 30.0F,
        .uses_tied_output_weight = true,
    };
    expect(model.gemma_config() == expected_config, "Gemma2 config mismatch");
    expect(model.tensor_views().size() == 13, "Gemma2 tensor view count mismatch");

    const auto* token_embedding = model.find_tensor_view("token_embd.weight");
    expect(token_embedding != nullptr, "Gemma2 token embedding view lookup failed");
    expect(
        token_embedding->view.bytes().data() == token_embedding_bytes,
        "Gemma2 token embedding view should point at GGUF payload bytes"
    );
}

void test_binds_gemma2_model_weights_with_mixed_k_quant_tensors() {
    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    const auto weights = bind_gemma_model_weights(model);

    expect(weights.blocks.size() == 1, "Gemma2 bound block count mismatch");
    expect(
        weights.output_weight.bytes().data() == weights.token_embedding_weight.bytes().data(),
        "Gemma2 missing output.weight should tie output to token embedding"
    );
    expect(weights.token_embedding_weight.descriptor().dtype == DType::f16, "Gemma2 token embedding dtype mismatch");
    expect(weights.output_norm_weight.descriptor().dtype == DType::f32, "Gemma2 output norm dtype mismatch");
    expect(
        weights.blocks[0].attention.query_weight.descriptor().dtype == DType::q4_k,
        "Gemma2 query dtype mismatch"
    );
    expect(
        weights.blocks[0].attention.value_weight.descriptor().dtype == DType::q6_k,
        "Gemma2 value dtype mismatch"
    );
    expect(
        weights.blocks[0].feed_forward.down_weight.descriptor().dtype == DType::q6_k,
        "Gemma2 FFN down dtype mismatch"
    );
    expect(
        weights.blocks[0].attention.post_attention_norm_weight.descriptor().dtype == DType::f32,
        "Gemma2 post-attention norm dtype mismatch"
    );
    expect(
        weights.blocks[0].feed_forward.post_feed_forward_norm_weight.descriptor().dtype == DType::f32,
        "Gemma2 post-FFN norm dtype mismatch"
    );
}

void test_rejects_engine_model_without_architecture_metadata() {
    auto metadata = tiny_gemma2_metadata();
    metadata.erase(metadata.begin());
    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(
        std::move(metadata),
        std::move(payload_store.payloads)
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
    auto metadata = tiny_gemma2_metadata();
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

void test_rejects_engine_model_missing_gemma2_metadata() {
    auto metadata = tiny_gemma2_metadata();
    metadata.erase(metadata.begin() + 2);
    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(std::move(metadata), std::move(payload_store.payloads));

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject missing Gemma2 metadata");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_metadata, "engine model missing Gemma2 metadata mismatch");
        expect(error.detail() == "gemma2.context_length", "engine model missing Gemma2 metadata detail mismatch");
    }
}

void test_rejects_engine_model_invalid_gemma2_metadata_type() {
    auto metadata = tiny_gemma2_metadata();
    metadata[4] = metadata_string("gemma2.block_count", "one");
    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(std::move(metadata), std::move(payload_store.payloads));

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject invalid Gemma2 metadata type");
    } catch (const EngineModelError& error) {
        expect(
            error.code() == EngineModelErrorCode::invalid_metadata_type,
            "engine model invalid Gemma2 metadata type mismatch"
        );
        expect(error.detail() == "gemma2.block_count", "engine model invalid Gemma2 metadata detail mismatch");
    }
}

void test_rejects_engine_model_invalid_gemma2_token_embedding_shape() {
    auto payload_store = tiny_gemma2_payload_store();
    payload_store.bytes[0] = zero_tensor_bytes({128, 4}, DType::f16);
    payload_store.payloads[0] = tensor_payload_view(
        "token_embd.weight",
        {128, 4},
        gguf_type_id_for_dtype(DType::f16),
        std::span<const std::uint8_t>(payload_store.bytes[0])
    );
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));

    try {
        (void)load_engine_model_from_gguf(view);
        throw TestFailure("engine model should reject invalid Gemma2 token embedding shape");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::invalid_tensor, "engine model invalid Gemma2 tensor mismatch");
        expect(error.detail() == "token_embd.weight", "engine model invalid Gemma2 tensor detail mismatch");
    }
}

void test_rejects_engine_model_duplicate_tensor_names() {
    auto first_tensor_bytes = zero_tensor_bytes({256, 4}, DType::f16);
    auto second_tensor_bytes = zero_tensor_bytes({256, 4}, DType::f16);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(),
        {
            tensor_payload_view(
                "token_embd.weight",
                {256, 4},
                gguf_type_id_for_dtype(DType::f16),
                std::span<const std::uint8_t>(first_tensor_bytes)
            ),
            tensor_payload_view(
                "token_embd.weight",
                {256, 4},
                gguf_type_id_for_dtype(DType::f16),
                std::span<const std::uint8_t>(second_tensor_bytes)
            ),
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

void test_rejects_gemma2_weight_binding_missing_tensor() {
    auto payload_store = tiny_gemma2_payload_store(false);
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    try {
        (void)bind_gemma_model_weights(model);
        throw TestFailure("Gemma2 weight binding should reject missing post-FFN norm");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_tensor, "Gemma2 missing tensor error mismatch");
        expect(
            error.detail() == "blk.0.post_ffw_norm.weight",
            "Gemma2 missing tensor detail mismatch"
        );
    }
}

const local_gguf_engine::tests::TestCase kEngineModelTests[] = {
    {
        "loads engine model from Gemma2 GGUF view without eager owned tensor copies",
        test_loads_engine_model_from_gemma2_gguf_view,
    },
    {
        "binds Gemma2 model weights with mixed K-quant tensors",
        test_binds_gemma2_model_weights_with_mixed_k_quant_tensors,
    },
    {
        "rejects engine model without architecture metadata",
        test_rejects_engine_model_without_architecture_metadata,
    },
    {"rejects engine model unsupported architecture", test_rejects_engine_model_unsupported_architecture},
    {"rejects engine model missing Gemma2 metadata", test_rejects_engine_model_missing_gemma2_metadata},
    {
        "rejects engine model invalid Gemma2 metadata type",
        test_rejects_engine_model_invalid_gemma2_metadata_type,
    },
    {
        "rejects engine model invalid Gemma2 token embedding shape",
        test_rejects_engine_model_invalid_gemma2_token_embedding_shape,
    },
    {"rejects engine model duplicate tensor names", test_rejects_engine_model_duplicate_tensor_names},
    {"rejects Gemma2 weight binding missing tensor", test_rejects_gemma2_weight_binding_missing_tensor},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> engine_model_test_cases() {
    return kEngineModelTests;
}

}  // namespace local_gguf_engine::tests
