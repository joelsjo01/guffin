#include "gemma2_metal_plan_tests.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "tensor_test_support.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::EngineModel;
using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::Gemma2MetalSchedulePhase;
using local_gguf_engine::engine::Gemma2MetalScratchOwnership;
using local_gguf_engine::engine::Gemma2MetalModelPlan;
using local_gguf_engine::engine::Gemma2ModelConfig;
using local_gguf_engine::engine::LlamaModelConfig;
using local_gguf_engine::engine::ModelArchitecture;
using local_gguf_engine::engine::create_gemma2_metal_decode_schedule;
using local_gguf_engine::engine::create_gemma2_metal_generation_runtime_shell;
using local_gguf_engine::engine::create_gemma2_metal_model_plan;
using local_gguf_engine::engine::create_gemma2_metal_prefill_schedule;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::model_format::GGUFHeader;
using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::model_format::GGUFMetadataSection;
using local_gguf_engine::model_format::GGUFModelView;
using local_gguf_engine::model_format::GGUFTensorDirectory;
using local_gguf_engine::model_format::GGUFTensorInfo;
using local_gguf_engine::model_format::GGUFTensorPayloadView;
using local_gguf_engine::runtime::BackendError;
using local_gguf_engine::runtime::BackendErrorCode;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::backend_buffer_storage;
using local_gguf_engine::tensor::copy_tensor_from_backend;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_vector;

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

    throw TestFailure("Gemma2 Metal plan test unsupported dtype");
}

std::vector<std::uint8_t> zero_tensor_bytes(std::span<const std::uint64_t> dimensions, DType dtype) {
    return std::vector<std::uint8_t>(
        TensorDescriptor::from(Shape::from(dimensions), dtype).byte_count
    );
}

GGUFTensorPayloadView tensor_payload_view(
    std::string name,
    std::vector<std::uint64_t> dimensions,
    DType dtype,
    std::span<const std::uint8_t> bytes
) {
    return GGUFTensorPayloadView {
        .tensor = GGUFTensorInfo {
            .name = std::move(name),
            .dimensions = std::move(dimensions),
            .type_id = gguf_type_id_for_dtype(dtype),
            .offset = 0,
            .file_offset = 0,
        },
        .bytes = bytes,
    };
}

std::vector<GGUFMetadataEntry> tiny_gemma2_metadata(std::uint32_t block_count = 1) {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Metal Plan Fixture"),
        metadata_u32("gemma2.context_length", 16),
        metadata_u32("gemma2.embedding_length", 256),
        metadata_u32("gemma2.block_count", block_count),
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

std::string block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

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
        dtype,
        std::span<const std::uint8_t>(store.bytes.back())
    ));
}

TinyGemma2PayloadStore tiny_gemma2_payload_store(
    bool include_post_ffw_norm = true,
    std::uint32_t block_count = 1
) {
    TinyGemma2PayloadStore store;
    store.bytes.reserve(static_cast<std::size_t>(2 + block_count * 11));
    store.payloads.reserve(static_cast<std::size_t>(2 + block_count * 11));

    add_payload(store, "token_embd.weight", {256, 4}, DType::f16);
    add_payload(store, "output_norm.weight", {256}, DType::f32);

    for (std::uint32_t block_index = 0; block_index < block_count; ++block_index) {
        add_payload(store, block_tensor_name(block_index, "attn_norm.weight"), {256}, DType::f32);
        add_payload(store, block_tensor_name(block_index, "post_attention_norm.weight"), {256}, DType::f32);
        add_payload(store, block_tensor_name(block_index, "attn_q.weight"), {256, 256}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "attn_k.weight"), {256, 256}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "attn_v.weight"), {256, 256}, DType::q6_k);
        add_payload(store, block_tensor_name(block_index, "attn_output.weight"), {256, 256}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "ffn_norm.weight"), {256}, DType::f32);
        if (include_post_ffw_norm) {
            add_payload(store, block_tensor_name(block_index, "post_ffw_norm.weight"), {256}, DType::f32);
        }
        add_payload(store, block_tensor_name(block_index, "ffn_gate.weight"), {256, 256}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "ffn_up.weight"), {256, 256}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "ffn_down.weight"), {256, 256}, DType::q6_k);
    }

    return store;
}

std::size_t payload_byte_count(std::span<const GGUFTensorPayloadView> payloads) {
    std::size_t total = 0;
    for (const auto& payload : payloads) {
        total += payload.bytes.size();
    }

    return total;
}

std::size_t backend_buffer_byte_count(const local_gguf_engine::tensor::BackendBuffer& buffer) {
    return buffer.descriptor().byte_count;
}

void expect_f32_matrix_close(
    const local_gguf_engine::tensor::TensorStorage& actual,
    const local_gguf_engine::tensor::TensorStorage& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    expect(actual.descriptor().dtype == DType::f32, std::string(message) + " dtype mismatch");

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

std::size_t expected_kv_cache_byte_count(const Gemma2ModelConfig& config) {
    const auto key_descriptor = TensorDescriptor::from(
        Shape::from({config.context_length, config.attention_head_count_kv, config.attention_key_length}),
        DType::f32
    );
    const auto value_descriptor = TensorDescriptor::from(
        Shape::from({config.context_length, config.attention_head_count_kv, config.attention_value_length}),
        DType::f32
    );

    return static_cast<std::size_t>(config.block_count) * (key_descriptor.byte_count + value_descriptor.byte_count);
}

std::size_t expected_scratch_byte_count(const Gemma2MetalModelPlan& plan) {
    const auto& scratch = plan.decode_scratch();
    return backend_buffer_byte_count(scratch.normalized_hidden)
        + backend_buffer_byte_count(scratch.query)
        + backend_buffer_byte_count(scratch.key)
        + backend_buffer_byte_count(scratch.value)
        + backend_buffer_byte_count(scratch.attention_scores)
        + backend_buffer_byte_count(scratch.attention_output)
        + backend_buffer_byte_count(scratch.feed_forward_gate)
        + backend_buffer_byte_count(scratch.feed_forward_up)
        + backend_buffer_byte_count(scratch.feed_forward_gated)
        + backend_buffer_byte_count(scratch.logits);
}

std::size_t schedule_scratch_byte_count(const local_gguf_engine::engine::Gemma2MetalScheduleScratch& scratch) {
    return scratch.normalized_hidden.byte_count
        + scratch.query.byte_count
        + scratch.key.byte_count
        + scratch.value.byte_count
        + scratch.attention_scores.byte_count
        + scratch.attention_output.byte_count
        + scratch.feed_forward_gate.byte_count
        + scratch.feed_forward_up.byte_count
        + scratch.feed_forward_gated.byte_count
        + scratch.logits.byte_count;
}

bool shares_backend_storage(
    const local_gguf_engine::tensor::BackendBuffer& left,
    const local_gguf_engine::tensor::BackendBuffer& right
) {
    return &backend_buffer_storage(left) == &backend_buffer_storage(right);
}

EngineModel tiny_llama_model() {
    return EngineModel(
        ModelArchitecture::llama,
        {},
        LlamaModelConfig {},
        {}
    );
}

void test_creates_gemma2_metal_plan_from_borrowed_weight_views_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    const auto expected_weight_bytes = payload_byte_count(std::span<const GGUFTensorPayloadView>(payload_store.payloads));
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    const auto plan = create_gemma2_metal_model_plan(model);

    expect(plan.backend() == BackendKind::metal, "Gemma2 Metal plan backend mismatch");
    expect(model.tensors().empty(), "Gemma2 Metal plan should not require owned host tensors");
    expect(plan.config() == model.gemma2_config(), "Gemma2 Metal plan config mismatch");
    expect(plan.uploaded_weight_byte_count() == expected_weight_bytes, "Gemma2 Metal uploaded byte count mismatch");
    expect(plan.kv_cache_byte_count() == expected_kv_cache_byte_count(plan.config()), "Gemma2 Metal KV byte count mismatch");
    expect(plan.scratch_byte_count() == expected_scratch_byte_count(plan), "Gemma2 Metal scratch byte count mismatch");

    expect(plan.weights().blocks.size() == 1, "Gemma2 Metal block weight count mismatch");
    expect(plan.kv_cache().size() == 1, "Gemma2 Metal KV layer count mismatch");
    expect(plan.layers().size() == 1, "Gemma2 Metal layer state count mismatch");
    expect(plan.layers()[0].layer_index == 0, "Gemma2 Metal layer state index mismatch");
    expect(plan.layers()[0].uses_sliding_window, "Gemma2 Metal layer 0 should use sliding window");

    expect(
        plan.weights().token_embedding_weight.descriptor().dtype == DType::f16,
        "Gemma2 Metal token embedding dtype mismatch"
    );
    expect(
        plan.weights().blocks[0].attention.query_weight.descriptor().dtype == DType::q4_k,
        "Gemma2 Metal query weight dtype mismatch"
    );
    expect(
        plan.weights().blocks[0].feed_forward.down_weight.descriptor().dtype == DType::q6_k,
        "Gemma2 Metal down weight dtype mismatch"
    );
    expect(
        plan.kv_cache()[0].key.descriptor().shape == Shape::from({16, 1, 256}),
        "Gemma2 Metal KV key shape mismatch"
    );
    expect(
        plan.decode_scratch().logits.descriptor().shape == Shape::from({1, 4}),
        "Gemma2 Metal logits scratch shape mismatch"
    );
}

void test_gemma2_metal_plan_roundtrips_uploaded_borrowed_weight_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    const auto expected_embedding_bytes = payload_store.payloads[0].bytes;
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    const auto plan = create_gemma2_metal_model_plan(model);
    const auto roundtrip = copy_tensor_from_backend(plan.weights().token_embedding_weight);

    expect(roundtrip.descriptor() == plan.weights().token_embedding_weight.descriptor(), "Gemma2 Metal roundtrip descriptor mismatch");
    expect(
        std::equal(expected_embedding_bytes.begin(), expected_embedding_bytes.end(), roundtrip.bytes().begin()),
        "Gemma2 Metal uploaded token embedding bytes mismatch"
    );
}

void test_creates_distinct_gemma2_metal_prefill_and_decode_schedules_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 2);
    auto view = tiny_model_view(tiny_gemma2_metadata(2), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);

    const auto prefill = create_gemma2_metal_prefill_schedule(plan, 12);
    const auto decode = create_gemma2_metal_decode_schedule(plan, 12);

    expect(prefill.backend() == BackendKind::metal, "Gemma2 prefill schedule backend mismatch");
    expect(decode.backend() == BackendKind::metal, "Gemma2 decode schedule backend mismatch");
    expect(prefill.config() == plan.config(), "Gemma2 prefill schedule config mismatch");
    expect(decode.config() == plan.config(), "Gemma2 decode schedule config mismatch");
    expect(prefill.phase() == Gemma2MetalSchedulePhase::prefill, "Gemma2 prefill schedule phase mismatch");
    expect(decode.phase() == Gemma2MetalSchedulePhase::decode, "Gemma2 decode schedule phase mismatch");

    expect(prefill.token_count() == 12, "Gemma2 prefill token count mismatch");
    expect(prefill.cache_position() == 0, "Gemma2 prefill cache position mismatch");
    expect(prefill.total_kv_token_count() == 12, "Gemma2 prefill total KV token count mismatch");
    expect(decode.token_count() == 1, "Gemma2 decode token count mismatch");
    expect(decode.cache_position() == 12, "Gemma2 decode cache position mismatch");
    expect(decode.total_kv_token_count() == 13, "Gemma2 decode total KV token count mismatch");

    expect(
        prefill.scratch_ownership() == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 prefill scratch ownership mismatch"
    );
    expect(
        decode.scratch_ownership() == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 decode scratch ownership mismatch"
    );
    expect(
        prefill.scratch().normalized_hidden.shape == Shape::from({12, 256}),
        "Gemma2 prefill normalized hidden scratch shape mismatch"
    );
    expect(
        prefill.scratch().attention_scores.shape == Shape::from({12, 1, 12}),
        "Gemma2 prefill attention score scratch shape mismatch"
    );
    expect(
        decode.scratch().normalized_hidden.shape == Shape::from({1, 256}),
        "Gemma2 decode normalized hidden scratch shape mismatch"
    );
    expect(
        decode.scratch().attention_scores.shape == Shape::from({1, 1, 16}),
        "Gemma2 decode resident attention score scratch shape mismatch"
    );
    expect(
        prefill.phase_scratch_byte_count() == schedule_scratch_byte_count(prefill.scratch()),
        "Gemma2 prefill schedule scratch byte count mismatch"
    );
    expect(
        decode.phase_scratch_byte_count() == plan.scratch_byte_count(),
        "Gemma2 decode schedule scratch byte count mismatch"
    );

    expect(prefill.layers().size() == 2, "Gemma2 prefill layer schedule count mismatch");
    expect(decode.layers().size() == 2, "Gemma2 decode layer schedule count mismatch");
    expect(prefill.layers()[0].layer_index == 0, "Gemma2 prefill layer 0 index mismatch");
    expect(prefill.layers()[0].uses_sliding_window, "Gemma2 prefill layer 0 should use sliding attention");
    expect(prefill.layers()[0].attention_token_count == 8, "Gemma2 prefill sliding attention count mismatch");
    expect(prefill.layers()[1].layer_index == 1, "Gemma2 prefill layer 1 index mismatch");
    expect(!prefill.layers()[1].uses_sliding_window, "Gemma2 prefill layer 1 should use full attention");
    expect(prefill.layers()[1].attention_token_count == 12, "Gemma2 prefill full attention count mismatch");
    expect(decode.layers()[0].attention_token_count == 8, "Gemma2 decode sliding attention count mismatch");
    expect(decode.layers()[1].attention_token_count == 13, "Gemma2 decode full attention count mismatch");
}

void test_gemma2_metal_schedules_share_resident_buffers_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);

    const auto prefill = create_gemma2_metal_prefill_schedule(plan, 4);
    const auto decode = create_gemma2_metal_decode_schedule(plan, 4);

    expect(
        shares_backend_storage(prefill.weights().token_embedding_weight, plan.weights().token_embedding_weight),
        "Gemma2 prefill schedule should share token embedding buffer"
    );
    expect(
        shares_backend_storage(decode.weights().blocks[0].attention.query_weight, plan.weights().blocks[0].attention.query_weight),
        "Gemma2 decode schedule should share query weight buffer"
    );
    expect(
        shares_backend_storage(prefill.kv_cache()[0].key, plan.kv_cache()[0].key),
        "Gemma2 prefill schedule should share KV key buffer"
    );
    expect(
        shares_backend_storage(decode.kv_cache()[0].value, plan.kv_cache()[0].value),
        "Gemma2 decode schedule should share KV value buffer"
    );
    expect(
        shares_backend_storage(decode.resident_decode_scratch().logits, plan.decode_scratch().logits),
        "Gemma2 decode schedule should share resident logits scratch buffer"
    );
    expect(
        prefill.resident_weight_byte_count() == plan.uploaded_weight_byte_count(),
        "Gemma2 prefill resident weight byte count mismatch"
    );
    expect(
        decode.resident_kv_cache_byte_count() == plan.kv_cache_byte_count(),
        "Gemma2 decode resident KV byte count mismatch"
    );
    expect(
        decode.resident_decode_scratch_byte_count() == plan.scratch_byte_count(),
        "Gemma2 decode resident scratch byte count mismatch"
    );
}

void test_gemma2_metal_schedule_retains_buffers_after_plan_lifetime_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto schedule = [] {
        auto payload_store = tiny_gemma2_payload_store();
        auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
        const auto model = load_engine_model_from_gguf(view);
        const auto plan = create_gemma2_metal_model_plan(model);

        return create_gemma2_metal_decode_schedule(plan, 4);
    }();

    const auto roundtrip = copy_tensor_from_backend(schedule.weights().token_embedding_weight);
    expect(
        roundtrip.descriptor() == schedule.weights().token_embedding_weight.descriptor(),
        "Gemma2 Metal schedule retained weight descriptor mismatch"
    );
    expect(schedule.kv_cache().size() == 1, "Gemma2 Metal schedule retained KV cache count mismatch");
    expect(
        schedule.resident_decode_scratch().logits.descriptor().shape == Shape::from({1, 4}),
        "Gemma2 Metal schedule retained logits scratch shape mismatch"
    );
}

void test_gemma2_metal_runtime_shell_embeds_prefill_prompt_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    std::vector<float> embedding_values(256U * 4U, 0.0F);
    embedding_values[0] = 1.0F;
    embedding_values[1] = 2.0F;
    embedding_values[2] = 3.0F;
    embedding_values[3] = 4.0F;
    embedding_values[4] = 10.0F;
    embedding_values[5] = 20.0F;
    embedding_values[6] = 30.0F;
    embedding_values[7] = 40.0F;
    const auto embedding = tensor_f16_gguf_matrix(256, 4, std::span<const float>(embedding_values));
    std::copy(embedding.bytes().begin(), embedding.bytes().end(), payload_store.bytes[0].begin());
    std::vector<float> attention_norm_values(256U, 1.0F);
    attention_norm_values[1] = 0.5F;
    const auto attention_norm = tensor_f32_vector(std::span<const float>(attention_norm_values));
    std::copy(attention_norm.bytes().begin(), attention_norm.bytes().end(), payload_store.bytes[2].begin());

    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};

    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );
    const auto hidden_states = copy_tensor_from_backend(shell.prefill_input.hidden_states);
    const auto normalized_attention_input =
        copy_tensor_from_backend(shell.prefill_first_layer_input.normalized_attention_input);
    const auto expected_normalized = rms_norm_f32(
        hidden_states,
        attention_norm,
        shell.plan.config().rms_norm_epsilon
    );

    expect(shell.prefill_input.token_ids == std::vector<std::uint32_t>({3, 1}), "Gemma2 shell token ids mismatch");
    expect(shell.prefill_input.embedding_scale == 16.0F, "Gemma2 shell embedding scale mismatch");
    expect(shell.prefill_input.hidden_states.backend() == BackendKind::metal, "Gemma2 shell input backend mismatch");
    expect(
        shell.prefill_input.hidden_states.descriptor().shape == Shape::from({2, 256}),
        "Gemma2 shell input hidden state shape mismatch"
    );
    expect(
        shell.prefill_input.hidden_states.descriptor().dtype == DType::f32,
        "Gemma2 shell input hidden state dtype mismatch"
    );
    expect(hidden_states.descriptor() == shell.prefill_input.hidden_states.descriptor(), "Gemma2 shell input readback descriptor mismatch");
    expect(shell.prefill_first_layer_input.layer_index == 0, "Gemma2 shell first layer input index mismatch");
    expect(
        shell.prefill_first_layer_input.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer input phase mismatch"
    );
    expect(
        shell.prefill_first_layer_input.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer input scratch ownership mismatch"
    );
    expect(
        shell.prefill_first_layer_input.normalized_attention_input.backend() == BackendKind::metal,
        "Gemma2 shell first layer input backend mismatch"
    );
    expect(
        shell.prefill_first_layer_input.normalized_attention_input.descriptor()
            == shell.prefill.scratch().normalized_hidden,
        "Gemma2 shell first layer input scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            shell.prefill_first_layer_input.normalized_attention_input,
            shell.prefill_input.hidden_states
        ),
        "Gemma2 shell first layer input should be an explicit output buffer"
    );

    const std::array<std::uint64_t, 2> token_0_hidden_0 {0, 0};
    const std::array<std::uint64_t, 2> token_0_hidden_1 {0, 1};
    const std::array<std::uint64_t, 2> token_1_hidden_0 {1, 0};
    const std::array<std::uint64_t, 2> token_1_hidden_1 {1, 1};
    expect_close(
        read_f32(hidden_states, std::span<const std::uint64_t>(token_0_hidden_0)),
        64.0F,
        0.0F,
        "Gemma2 shell token 0 hidden 0 mismatch"
    );
    expect_close(
        read_f32(hidden_states, std::span<const std::uint64_t>(token_0_hidden_1)),
        640.0F,
        0.0F,
        "Gemma2 shell token 0 hidden 1 mismatch"
    );
    expect_close(
        read_f32(hidden_states, std::span<const std::uint64_t>(token_1_hidden_0)),
        32.0F,
        0.0F,
        "Gemma2 shell token 1 hidden 0 mismatch"
    );
    expect_close(
        read_f32(hidden_states, std::span<const std::uint64_t>(token_1_hidden_1)),
        320.0F,
        0.0F,
        "Gemma2 shell token 1 hidden 1 mismatch"
    );
    expect_f32_matrix_close(
        normalized_attention_input,
        expected_normalized,
        1.0e-5F,
        "Gemma2 shell first layer normalized attention input"
    );
}

void test_rejects_invalid_gemma2_metal_schedule_shapes_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);

    try {
        (void)create_gemma2_metal_prefill_schedule(plan, 0);
        throw TestFailure("Gemma2 Metal prefill schedule should reject empty prompts");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 Metal prefill schedule token count must be positive",
            "Gemma2 Metal empty prefill error mismatch"
        );
    }

    try {
        (void)create_gemma2_metal_prefill_schedule(plan, 17);
        throw TestFailure("Gemma2 Metal prefill schedule should reject prompts past context");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 Metal prefill schedule token count exceeds context length",
            "Gemma2 Metal oversized prefill error mismatch"
        );
    }

    try {
        (void)create_gemma2_metal_decode_schedule(plan, 0);
        throw TestFailure("Gemma2 Metal decode schedule should reject empty cache");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 Metal decode schedule cached token count must be positive",
            "Gemma2 Metal empty decode cache error mismatch"
        );
    }

    try {
        (void)create_gemma2_metal_decode_schedule(plan, 16);
        throw TestFailure("Gemma2 Metal decode schedule should reject full cache");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 Metal decode schedule cached token count must leave room for one token",
            "Gemma2 Metal full decode cache error mismatch"
        );
    }
}

void test_rejects_gemma2_metal_plan_for_non_gemma_model() {
    try {
        (void)create_gemma2_metal_model_plan(tiny_llama_model());
        throw TestFailure("Gemma2 Metal plan should reject non-Gemma model");
    } catch (const EngineModelError& error) {
        expect(
            error.code() == EngineModelErrorCode::unsupported_architecture,
            "Gemma2 Metal plan unsupported architecture error mismatch"
        );
    }
}

void test_rejects_gemma2_metal_plan_missing_required_weight_before_upload() {
    auto payload_store = tiny_gemma2_payload_store(false);
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    try {
        (void)create_gemma2_metal_model_plan(model);
        throw TestFailure("Gemma2 Metal plan should reject missing required weight");
    } catch (const EngineModelError& error) {
        expect(error.code() == EngineModelErrorCode::missing_tensor, "Gemma2 Metal missing weight error mismatch");
        expect(
            error.detail() == "blk.0.post_ffw_norm.weight",
            "Gemma2 Metal missing weight detail mismatch"
        );
    }
}

void test_reports_gemma2_metal_plan_unavailable_when_metal_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    try {
        (void)create_gemma2_metal_model_plan(model);
        throw TestFailure("Gemma2 Metal plan should report unavailable Metal device");
    } catch (const BackendError& error) {
        expect(
            error.code() == BackendErrorCode::unavailable_metal_device,
            "Gemma2 Metal unavailable error code mismatch"
        );
        expect(error.backend() == BackendKind::metal, "Gemma2 Metal unavailable backend mismatch");
    }
}

const TestCase kGemma2MetalPlanTests[] = {
    {
        "creates Gemma2 Metal plan from borrowed weight views when available",
        test_creates_gemma2_metal_plan_from_borrowed_weight_views_when_available,
    },
    {
        "roundtrips Gemma2 Metal uploaded borrowed weight when available",
        test_gemma2_metal_plan_roundtrips_uploaded_borrowed_weight_when_available,
    },
    {
        "creates distinct Gemma2 Metal prefill and decode schedules when available",
        test_creates_distinct_gemma2_metal_prefill_and_decode_schedules_when_available,
    },
    {
        "Gemma2 Metal schedules share resident buffers when available",
        test_gemma2_metal_schedules_share_resident_buffers_when_available,
    },
    {
        "Gemma2 Metal schedule retains buffers after plan lifetime when available",
        test_gemma2_metal_schedule_retains_buffers_after_plan_lifetime_when_available,
    },
    {
        "Gemma2 Metal runtime shell embeds prefill prompt when available",
        test_gemma2_metal_runtime_shell_embeds_prefill_prompt_when_available,
    },
    {
        "rejects invalid Gemma2 Metal schedule shapes when available",
        test_rejects_invalid_gemma2_metal_schedule_shapes_when_available,
    },
    {"rejects Gemma2 Metal plan for non-Gemma model", test_rejects_gemma2_metal_plan_for_non_gemma_model},
    {
        "rejects Gemma2 Metal plan missing required weight before upload",
        test_rejects_gemma2_metal_plan_missing_required_weight_before_upload,
    },
    {
        "reports Gemma2 Metal plan unavailable when Metal missing",
        test_reports_gemma2_metal_plan_unavailable_when_metal_missing,
    },
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gemma2_metal_plan_test_cases() {
    return kGemma2MetalPlanTests;
}

}  // namespace local_gguf_engine::tests
