#include "gemma2_metal_plan_tests.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "tensor_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using local_gguf_engine::engine::EngineModelError;
using local_gguf_engine::engine::EngineModelErrorCode;
using local_gguf_engine::engine::Gemma2MetalSchedulePhase;
using local_gguf_engine::engine::Gemma2MetalScratchOwnership;
using local_gguf_engine::engine::Gemma2MetalModelPlan;
using local_gguf_engine::engine::Gemma2ModelConfig;
using local_gguf_engine::engine::create_gemma2_metal_decode_schedule;
using local_gguf_engine::engine::create_gemma2_metal_model_plan;
using local_gguf_engine::engine::create_gemma2_metal_prefill_schedule;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens_with_plan;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::engine::run_gemma2_metal_decode_step;
using local_gguf_engine::engine::run_gemma2_metal_resident_prefill;
using local_gguf_engine::engine::run_gemma2_metal_resident_decode_step;
using local_gguf_engine::engine::top_logit_tokens_f32;
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
using local_gguf_engine::tensor::copy_tensor_to_backend;
using local_gguf_engine::tensor::copy_tensor_from_backend;
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32;
using local_gguf_engine::tensor::ops::matmul_f32_f16_gguf_weight;
using local_gguf_engine::tensor::ops::matmul_f32_quantized_gguf_weight;
using local_gguf_engine::tensor::ops::multiply_f32;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rope_neox_f32_by_heads_with_position_start;
using local_gguf_engine::tensor::ops::gelu_tanh_f32;
using local_gguf_engine::tensor::ops::softcap_f32;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::tensor_q4_k_gguf_weight_matrix;
using local_gguf_engine::tests::tensor_q6_k_gguf_weight_matrix;

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

std::vector<GGUFMetadataEntry> tiny_gemma2_metadata(
    std::uint32_t block_count = 1,
    std::uint32_t attention_head_count = 1,
    std::uint32_t attention_head_count_kv = 1,
    std::uint32_t attention_value_length = 256,
    std::uint32_t context_length = 16
) {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Metal Plan Fixture"),
        metadata_u32("gemma2.context_length", context_length),
        metadata_u32("gemma2.embedding_length", 256),
        metadata_u32("gemma2.block_count", block_count),
        metadata_u32("gemma2.feed_forward_length", 256),
        metadata_u32("gemma2.attention.head_count", attention_head_count),
        metadata_u32("gemma2.attention.head_count_kv", attention_head_count_kv),
        metadata_u32("gemma2.attention.key_length", 256),
        metadata_u32("gemma2.attention.value_length", attention_value_length),
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
    const auto& prefill = plan.prefill_scratch();
    const auto& decode = plan.decode_scratch();
    return backend_buffer_byte_count(prefill.token_id_staging)
        + backend_buffer_byte_count(prefill.normalized_hidden)
        + backend_buffer_byte_count(prefill.raw_query)
        + backend_buffer_byte_count(prefill.raw_key)
        + backend_buffer_byte_count(prefill.query)
        + backend_buffer_byte_count(prefill.key)
        + backend_buffer_byte_count(prefill.value)
        + backend_buffer_byte_count(prefill.attention_scores)
        + backend_buffer_byte_count(prefill.attention_probabilities)
        + backend_buffer_byte_count(prefill.attention_output)
        + backend_buffer_byte_count(prefill.projected_attention_output)
        + backend_buffer_byte_count(prefill.post_attention_residual)
        + backend_buffer_byte_count(prefill.normalized_feed_forward_input)
        + backend_buffer_byte_count(prefill.feed_forward_gate)
        + backend_buffer_byte_count(prefill.feed_forward_up)
        + backend_buffer_byte_count(prefill.feed_forward_gated)
        + backend_buffer_byte_count(prefill.projected_feed_forward_output)
        + backend_buffer_byte_count(prefill.feed_forward_residual)
        + backend_buffer_byte_count(prefill.final_normalized_hidden)
        + backend_buffer_byte_count(prefill.logits)
        + backend_buffer_byte_count(prefill.top_logit_staging)
        + backend_buffer_byte_count(decode.token_id_staging)
        + backend_buffer_byte_count(decode.normalized_hidden)
        + backend_buffer_byte_count(decode.raw_query)
        + backend_buffer_byte_count(decode.raw_key)
        + backend_buffer_byte_count(decode.query)
        + backend_buffer_byte_count(decode.key)
        + backend_buffer_byte_count(decode.value)
        + backend_buffer_byte_count(decode.attention_scores)
        + backend_buffer_byte_count(decode.attention_probabilities)
        + backend_buffer_byte_count(decode.attention_output)
        + backend_buffer_byte_count(decode.projected_attention_output)
        + backend_buffer_byte_count(decode.post_attention_residual)
        + backend_buffer_byte_count(decode.normalized_feed_forward_input)
        + backend_buffer_byte_count(decode.feed_forward_gate)
        + backend_buffer_byte_count(decode.feed_forward_up)
        + backend_buffer_byte_count(decode.feed_forward_gated)
        + backend_buffer_byte_count(decode.projected_feed_forward_output)
        + backend_buffer_byte_count(decode.feed_forward_residual)
        + backend_buffer_byte_count(decode.layer_handoff_normalized_attention_input)
        + backend_buffer_byte_count(decode.logits)
        + backend_buffer_byte_count(decode.top_logit_staging);
}

std::size_t schedule_scratch_byte_count(const local_gguf_engine::engine::Gemma2MetalScheduleScratch& scratch) {
    return scratch.normalized_hidden.byte_count
        + scratch.query.byte_count
        + scratch.key.byte_count
        + scratch.value.byte_count
        + scratch.attention_scores.byte_count
        + scratch.attention_probabilities.byte_count
        + scratch.attention_output.byte_count
        + scratch.projected_attention_output.byte_count
        + scratch.post_attention_residual.byte_count
        + scratch.normalized_feed_forward_input.byte_count
        + scratch.feed_forward_gate.byte_count
        + scratch.feed_forward_up.byte_count
        + scratch.feed_forward_gated.byte_count
        + scratch.projected_feed_forward_output.byte_count
        + scratch.feed_forward_residual.byte_count
        + scratch.layer_handoff_normalized_attention_input.byte_count
        + scratch.logits.byte_count;
}

bool shares_backend_storage(
    const local_gguf_engine::tensor::BackendBuffer& left,
    const local_gguf_engine::tensor::BackendBuffer& right
) {
    return &backend_buffer_storage(left) == &backend_buffer_storage(right);
}

void test_creates_gemma2_metal_plan_from_borrowed_weight_views_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    const auto expected_weight_bytes = payload_byte_count(std::span<const GGUFTensorPayloadView>(payload_store.payloads));
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);

    auto plan = create_gemma2_metal_model_plan(model);

    expect(plan.backend() == BackendKind::metal, "Gemma2 Metal plan backend mismatch");
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
    expect(
        plan.decode_scratch().projected_attention_output.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal projected attention output scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().post_attention_residual.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal post-attention residual scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().normalized_feed_forward_input.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal normalized FFN input scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().feed_forward_gated.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal gated FFN scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().projected_feed_forward_output.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal projected FFN output scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().feed_forward_residual.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal FFN residual scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().layer_handoff_normalized_attention_input.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal layer handoff normalized attention input scratch shape mismatch"
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
        prefill.scratch_ownership() == Gemma2MetalScratchOwnership::resident_prefill,
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
        prefill.scratch().attention_probabilities.shape == Shape::from({12, 1, 12}),
        "Gemma2 prefill attention probability scratch shape mismatch"
    );
    expect(
        prefill.scratch().projected_attention_output.shape == Shape::from({12, 256}),
        "Gemma2 prefill projected attention output scratch shape mismatch"
    );
    expect(
        prefill.scratch().post_attention_residual.shape == Shape::from({12, 256}),
        "Gemma2 prefill post-attention residual scratch shape mismatch"
    );
    expect(
        prefill.scratch().normalized_feed_forward_input.shape == Shape::from({12, 256}),
        "Gemma2 prefill normalized FFN input scratch shape mismatch"
    );
    expect(
        prefill.scratch().feed_forward_gate.shape == Shape::from({12, 256}),
        "Gemma2 prefill FFN gate scratch shape mismatch"
    );
    expect(
        prefill.scratch().feed_forward_up.shape == Shape::from({12, 256}),
        "Gemma2 prefill FFN up scratch shape mismatch"
    );
    expect(
        prefill.scratch().feed_forward_gated.shape == Shape::from({12, 256}),
        "Gemma2 prefill gated FFN scratch shape mismatch"
    );
    expect(
        prefill.scratch().projected_feed_forward_output.shape == Shape::from({12, 256}),
        "Gemma2 prefill projected FFN output scratch shape mismatch"
    );
    expect(
        prefill.scratch().feed_forward_residual.shape == Shape::from({12, 256}),
        "Gemma2 prefill FFN residual scratch shape mismatch"
    );
    expect(
        prefill.scratch().layer_handoff_normalized_attention_input.shape == Shape::from({12, 256}),
        "Gemma2 prefill layer handoff normalized attention input scratch shape mismatch"
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
        decode.scratch().attention_probabilities.shape == Shape::from({1, 1, 16}),
        "Gemma2 decode resident attention probability scratch shape mismatch"
    );
    expect(
        decode.scratch().normalized_feed_forward_input.shape == Shape::from({1, 256}),
        "Gemma2 decode resident normalized FFN input scratch shape mismatch"
    );
    expect(
        decode.scratch().projected_feed_forward_output.shape == Shape::from({1, 256}),
        "Gemma2 decode resident projected FFN output scratch shape mismatch"
    );
    expect(
        decode.scratch().feed_forward_residual.shape == Shape::from({1, 256}),
        "Gemma2 decode resident FFN residual scratch shape mismatch"
    );
    expect(
        decode.scratch().layer_handoff_normalized_attention_input.shape == Shape::from({1, 256}),
        "Gemma2 decode resident layer handoff normalized attention input scratch shape mismatch"
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

void test_creates_gemma2_metal_append_prefill_schedule_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 2);
    auto view = tiny_model_view(tiny_gemma2_metadata(2), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);

    const auto prefill = create_gemma2_metal_prefill_schedule(plan, 3, 7);

    expect(prefill.phase() == Gemma2MetalSchedulePhase::prefill, "Gemma2 append prefill phase mismatch");
    expect(prefill.token_count() == 3, "Gemma2 append prefill token count mismatch");
    expect(prefill.cache_position() == 7, "Gemma2 append prefill cache position mismatch");
    expect(prefill.total_kv_token_count() == 10, "Gemma2 append prefill total KV count mismatch");
    expect(
        prefill.scratch().normalized_hidden.shape == Shape::from({3, 256}),
        "Gemma2 append prefill normalized hidden scratch shape mismatch"
    );
    expect(
        prefill.scratch().attention_scores.shape == Shape::from({3, 1, 10}),
        "Gemma2 append prefill attention scores scratch shape mismatch"
    );
    expect(prefill.layers()[0].attention_token_count == 8, "Gemma2 append sliding attention count mismatch");
    expect(prefill.layers()[1].attention_token_count == 10, "Gemma2 append full attention count mismatch");
}

void test_gemma2_metal_plan_tracks_projected_attention_output_shape_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    auto metadata = tiny_gemma2_metadata(1, 2, 1, 256);
    auto view = tiny_model_view(std::move(metadata), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);

    const auto prefill = create_gemma2_metal_prefill_schedule(plan, 3);
    const auto decode = create_gemma2_metal_decode_schedule(plan, 3);

    expect(
        prefill.scratch().attention_output.shape == Shape::from({3, 512}),
        "Gemma2 prefill attention output scratch should follow query heads times value length"
    );
    expect(
        prefill.scratch().projected_attention_output.shape == Shape::from({3, 256}),
        "Gemma2 prefill projected attention output scratch should return to hidden width"
    );
    expect(
        prefill.scratch().post_attention_residual.shape == Shape::from({3, 256}),
        "Gemma2 prefill post-attention residual scratch should stay at hidden width"
    );
    expect(
        prefill.scratch().normalized_feed_forward_input.shape == Shape::from({3, 256}),
        "Gemma2 prefill normalized FFN input scratch should stay at hidden width"
    );
    expect(
        prefill.scratch().feed_forward_gate.shape == Shape::from({3, 256}),
        "Gemma2 prefill FFN gate scratch should follow feed-forward width"
    );
    expect(
        prefill.scratch().feed_forward_up.shape == Shape::from({3, 256}),
        "Gemma2 prefill FFN up scratch should follow feed-forward width"
    );
    expect(
        prefill.scratch().feed_forward_gated.shape == Shape::from({3, 256}),
        "Gemma2 prefill gated FFN scratch should follow feed-forward width"
    );
    expect(
        prefill.scratch().projected_feed_forward_output.shape == Shape::from({3, 256}),
        "Gemma2 prefill projected FFN output scratch should return to hidden width"
    );
    expect(
        prefill.scratch().feed_forward_residual.shape == Shape::from({3, 256}),
        "Gemma2 prefill FFN residual scratch should stay at hidden width"
    );
    expect(
        prefill.scratch().layer_handoff_normalized_attention_input.shape == Shape::from({3, 256}),
        "Gemma2 prefill layer handoff normalized attention input scratch should stay at hidden width"
    );
    expect(
        decode.scratch().attention_output.shape == Shape::from({1, 512}),
        "Gemma2 decode attention output scratch should follow query heads times value length"
    );
    expect(
        decode.scratch().projected_attention_output.shape == Shape::from({1, 256}),
        "Gemma2 decode projected attention output scratch should return to hidden width"
    );
    expect(
        decode.scratch().post_attention_residual.shape == Shape::from({1, 256}),
        "Gemma2 decode post-attention residual scratch should stay at hidden width"
    );
    expect(
        decode.scratch().normalized_feed_forward_input.shape == Shape::from({1, 256}),
        "Gemma2 decode normalized FFN input scratch should stay at hidden width"
    );
    expect(
        decode.scratch().projected_feed_forward_output.shape == Shape::from({1, 256}),
        "Gemma2 decode projected FFN output scratch should return to hidden width"
    );
    expect(
        decode.scratch().feed_forward_residual.shape == Shape::from({1, 256}),
        "Gemma2 decode FFN residual scratch should stay at hidden width"
    );
    expect(
        decode.scratch().layer_handoff_normalized_attention_input.shape == Shape::from({1, 256}),
        "Gemma2 decode layer handoff normalized attention input scratch should stay at hidden width"
    );
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
        shares_backend_storage(
            decode.resident_decode_scratch().projected_attention_output,
            plan.decode_scratch().projected_attention_output
        ),
        "Gemma2 decode schedule should share resident projected attention output scratch buffer"
    );
    expect(
        shares_backend_storage(
            decode.resident_decode_scratch().post_attention_residual,
            plan.decode_scratch().post_attention_residual
        ),
        "Gemma2 decode schedule should share resident post-attention residual scratch buffer"
    );
    expect(
        shares_backend_storage(
            decode.resident_decode_scratch().normalized_feed_forward_input,
            plan.decode_scratch().normalized_feed_forward_input
        ),
        "Gemma2 decode schedule should share resident normalized FFN input scratch buffer"
    );
    expect(
        shares_backend_storage(
            decode.resident_decode_scratch().projected_feed_forward_output,
            plan.decode_scratch().projected_feed_forward_output
        ),
        "Gemma2 decode schedule should share resident projected FFN output scratch buffer"
    );
    expect(
        shares_backend_storage(
            decode.resident_decode_scratch().feed_forward_residual,
            plan.decode_scratch().feed_forward_residual
        ),
        "Gemma2 decode schedule should share resident FFN residual scratch buffer"
    );
    expect(
        shares_backend_storage(
            decode.resident_decode_scratch().layer_handoff_normalized_attention_input,
            plan.decode_scratch().layer_handoff_normalized_attention_input
        ),
        "Gemma2 decode schedule should share resident layer handoff normalized attention input scratch buffer"
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
    expect(
        schedule.resident_decode_scratch().projected_attention_output.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal schedule retained projected attention output scratch shape mismatch"
    );
    expect(
        schedule.resident_decode_scratch().post_attention_residual.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal schedule retained post-attention residual scratch shape mismatch"
    );
    expect(
        schedule.resident_decode_scratch().normalized_feed_forward_input.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal schedule retained normalized FFN input scratch shape mismatch"
    );
    expect(
        schedule.resident_decode_scratch().projected_feed_forward_output.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal schedule retained projected FFN output scratch shape mismatch"
    );
    expect(
        schedule.resident_decode_scratch().feed_forward_residual.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal schedule retained FFN residual scratch shape mismatch"
    );
    expect(
        schedule.resident_decode_scratch().layer_handoff_normalized_attention_input.descriptor().shape
            == Shape::from({1, 256}),
        "Gemma2 Metal schedule retained layer handoff normalized attention input scratch shape mismatch"
    );
}

void test_gemma2_metal_resident_decode_is_one_submission_and_matches_diagnostics_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 2);
    auto view = tiny_model_view(tiny_gemma2_metadata(2), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {1, 2};
    auto diagnostic_plan = create_gemma2_metal_model_plan(model);
    auto resident_plan = create_gemma2_metal_model_plan(model);
    const auto embedding_scale = std::sqrt(
        static_cast<float>(diagnostic_plan.config().embedding_length)
    );
    const auto diagnostic_prefill = run_gemma2_metal_resident_prefill(
        diagnostic_plan,
        std::span<const std::uint32_t>(prompt_token_ids),
        0,
        embedding_scale,
        true
    );
    const auto resident_prefill = run_gemma2_metal_resident_prefill(
        resident_plan,
        std::span<const std::uint32_t>(prompt_token_ids),
        0,
        embedding_scale,
        true
    );
    expect(diagnostic_prefill.next_token_selection.has_value(), "diagnostic prefill token missing");
    expect(resident_prefill.next_token_selection.has_value(), "resident prefill token missing");
    const auto token_id = diagnostic_prefill.next_token_selection->token_id;
    const auto diagnostic_schedule = create_gemma2_metal_decode_schedule(
        diagnostic_plan,
        static_cast<std::uint32_t>(prompt_token_ids.size())
    );
    const auto resident_schedule = create_gemma2_metal_decode_schedule(
        resident_plan,
        static_cast<std::uint32_t>(prompt_token_ids.size())
    );

    const auto diagnostic = run_gemma2_metal_decode_step(
        diagnostic_plan,
        diagnostic_schedule,
        token_id,
        embedding_scale
    );
    const auto resident = run_gemma2_metal_resident_decode_step(
        resident_plan,
        resident_schedule,
        token_id,
        embedding_scale
    );

    expect(resident.execution.command_buffer_count == 1U, "resident decode command-buffer count mismatch");
    expect(resident.execution.commit_count == 1U, "resident decode commit count mismatch");
    expect(resident.execution.synchronous_wait_count == 1U, "resident decode wait count mismatch");
    expect(
        resident.execution.transient_buffer_allocation_count == 0U,
        "resident decode transient allocation count mismatch"
    );
    expect(resident.execution.compute_encoder_count == 43U, "resident decode compute encoder count mismatch");
    expect(resident.execution.blit_encoder_count == 2U, "resident decode blit encoder count mismatch");
    expect(resident.next_token_selection.has_value(), "resident decode top-logit result missing");

    const auto diagnostic_logits = copy_tensor_from_backend(diagnostic.logits.logits);
    const auto resident_logits = copy_tensor_from_backend(resident.logits);
    expect(
        diagnostic_logits.descriptor() == resident_logits.descriptor(),
        "resident decode logits descriptor mismatch"
    );
    for (std::uint64_t token = 0; token < diagnostic_logits.descriptor().element_count; ++token) {
        float diagnostic_value = 0.0F;
        float resident_value = 0.0F;
        const auto byte_offset = static_cast<std::size_t>(token * sizeof(float));
        std::memcpy(&diagnostic_value, diagnostic_logits.bytes().data() + byte_offset, sizeof(float));
        std::memcpy(&resident_value, resident_logits.bytes().data() + byte_offset, sizeof(float));
        expect_close(diagnostic_value, resident_value, 1.0e-5F, "resident decode logits parity mismatch");
    }
    expect(
        resident.next_token_selection->token_id == diagnostic.next_token_selection.token_id,
        "resident decode selected token mismatch"
    );
    expect_close(
        resident.next_token_selection->logit,
        diagnostic.next_token_selection.logit,
        1.0e-5F,
        "resident decode selected logit mismatch"
    );

    for (std::size_t layer = 0; layer < resident_plan.kv_cache().size(); ++layer) {
        const auto diagnostic_key = copy_tensor_from_backend(diagnostic_plan.kv_cache()[layer].key);
        const auto resident_key = copy_tensor_from_backend(resident_plan.kv_cache()[layer].key);
        const auto diagnostic_value = copy_tensor_from_backend(diagnostic_plan.kv_cache()[layer].value);
        const auto resident_value = copy_tensor_from_backend(resident_plan.kv_cache()[layer].value);
        expect(
            std::equal(diagnostic_key.bytes().begin(), diagnostic_key.bytes().end(), resident_key.bytes().begin()),
            "resident decode key-cache parity mismatch"
        );
        expect(
            std::equal(diagnostic_value.bytes().begin(), diagnostic_value.bytes().end(), resident_value.bytes().begin()),
            "resident decode value-cache parity mismatch"
        );
    }
}

void test_gemma2_metal_resident_prefill_chunks_once_waits_once_and_matches_incremental_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 2);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(2, 1, 1, 256, 32),
        std::move(payload_store.payloads)
    );
    const auto model = load_engine_model_from_gguf(view);
    auto chunked_plan = create_gemma2_metal_model_plan(model);
    auto incremental_plan = create_gemma2_metal_model_plan(model);
    std::vector<std::uint32_t> prompt_token_ids(17);
    for (std::size_t index = 0; index < prompt_token_ids.size(); ++index) {
        prompt_token_ids[index] = static_cast<std::uint32_t>(index % 4U);
    }
    const auto embedding_scale = std::sqrt(
        static_cast<float>(chunked_plan.config().embedding_length)
    );

    const auto chunked = run_gemma2_metal_resident_prefill(
        chunked_plan,
        std::span<const std::uint32_t>(prompt_token_ids),
        0,
        embedding_scale,
        true
    );
    const auto first = run_gemma2_metal_resident_prefill(
        incremental_plan,
        std::span<const std::uint32_t>(prompt_token_ids).first(16),
        0,
        embedding_scale,
        false
    );
    const auto last = run_gemma2_metal_resident_prefill(
        incremental_plan,
        std::span<const std::uint32_t>(prompt_token_ids).last(1),
        16,
        embedding_scale,
        true
    );

    expect(chunked.execution.chunk_count == 2U, "resident prefill chunk count mismatch");
    expect(chunked.execution.command_buffer_count == 2U, "resident prefill command-buffer count mismatch");
    expect(chunked.execution.commit_count == 2U, "resident prefill commit count mismatch");
    expect(chunked.execution.synchronous_wait_count == 1U, "resident prefill wait count mismatch");
    expect(
        chunked.execution.transient_buffer_allocation_count == 0U,
        "resident prefill transient allocation count mismatch"
    );
    expect(
        chunked.execution.scheduled_layer_execution_count == 4U,
        "resident prefill scheduled layer execution count mismatch"
    );
    expect(chunked.execution.kv_write_count == 4U, "resident prefill KV write count mismatch");
    expect(first.execution.chunk_count == 1U, "exact-size resident prefill should use one chunk");
    expect(first.execution.synchronous_wait_count == 1U, "exact-size resident prefill wait mismatch");
    expect(chunked.next_token_selection.has_value(), "resident prefill selected token missing");
    expect(last.next_token_selection.has_value(), "incremental resident prefill selected token missing");

    const auto chunked_logits = copy_tensor_from_backend(chunked.logits);
    const auto incremental_logits = copy_tensor_from_backend(last.logits);
    expect_f32_matrix_close(
        chunked_logits,
        incremental_logits,
        1.0e-5F,
        "resident prefill chunked logits parity"
    );
    expect(
        chunked.next_token_selection->token_id == last.next_token_selection->token_id,
        "resident prefill chunked selected token mismatch"
    );
    expect_close(
        chunked.next_token_selection->logit,
        last.next_token_selection->logit,
        1.0e-5F,
        "resident prefill chunked selected logit mismatch"
    );

    for (std::size_t layer = 0; layer < chunked_plan.kv_cache().size(); ++layer) {
        const auto chunked_key = copy_tensor_from_backend(chunked_plan.kv_cache()[layer].key);
        const auto incremental_key = copy_tensor_from_backend(incremental_plan.kv_cache()[layer].key);
        const auto chunked_value = copy_tensor_from_backend(chunked_plan.kv_cache()[layer].value);
        const auto incremental_value = copy_tensor_from_backend(incremental_plan.kv_cache()[layer].value);
        expect(
            std::equal(chunked_key.bytes().begin(), chunked_key.bytes().end(), incremental_key.bytes().begin()),
            "resident prefill chunked key-cache parity mismatch"
        );
        expect(
            std::equal(
                chunked_value.bytes().begin(),
                chunked_value.bytes().end(),
                incremental_value.bytes().begin()
            ),
            "resident prefill chunked value-cache parity mismatch"
        );
    }

    try {
        (void)run_gemma2_metal_resident_prefill(
            chunked_plan,
            std::span<const std::uint32_t>(),
            0,
            embedding_scale
        );
        throw TestFailure("resident prefill should reject an empty token span");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 resident Metal prefill token ids must not be empty",
            "resident prefill empty-token error mismatch"
        );
    }
    const std::array<std::uint32_t, 2> overflow_tokens {0, 1};
    try {
        (void)run_gemma2_metal_resident_prefill(
            chunked_plan,
            std::span<const std::uint32_t>(overflow_tokens),
            31,
            embedding_scale
        );
        throw TestFailure("resident prefill should reject a context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 resident Metal prefill range exceeds context length",
            "resident prefill context-overflow error mismatch"
        );
    }
    const std::array<std::uint32_t, 1> invalid_token {4};
    try {
        (void)run_gemma2_metal_resident_prefill(
            chunked_plan,
            std::span<const std::uint32_t>(invalid_token),
            0,
            embedding_scale
        );
        throw TestFailure("resident prefill should reject a token outside vocabulary");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 resident Metal prefill token is outside vocabulary",
            "resident prefill vocabulary error mismatch"
        );
    }
}

void test_gemma2_metal_plan_generation_append_matches_full_replay_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    auto plan = create_gemma2_metal_model_plan(model);
    const std::vector<std::uint32_t> prefix_token_ids {0, 1};
    const std::vector<std::uint32_t> suffix_token_ids {2};
    const std::vector<std::uint32_t> stop_token_ids;

    const auto first = generate_gemma2_metal_argmax_tokens_with_plan(
        plan,
        std::span<const std::uint32_t>(prefix_token_ids),
        0,
        1,
        std::span<const std::uint32_t>(stop_token_ids)
    );
    std::vector<std::uint32_t> full_prompt(prefix_token_ids.begin(), prefix_token_ids.end());
    full_prompt.insert(full_prompt.end(), first.generated_token_ids.begin(), first.generated_token_ids.end());
    const auto append_cache_position = static_cast<std::uint32_t>(full_prompt.size());
    full_prompt.insert(full_prompt.end(), suffix_token_ids.begin(), suffix_token_ids.end());

    const auto appended = generate_gemma2_metal_argmax_tokens_with_plan(
        plan,
        std::span<const std::uint32_t>(suffix_token_ids),
        append_cache_position,
        1,
        std::span<const std::uint32_t>(stop_token_ids)
    );
    const auto replay = generate_gemma2_metal_argmax_tokens(
        model,
        std::span<const std::uint32_t>(full_prompt),
        1
    );

    expect(appended.prefill_cache_position == append_cache_position, "Gemma2 append replay cache position mismatch");
    expect(appended.generated_token_ids.size() == 1, "Gemma2 append replay generated count mismatch");
    expect(
        appended.generated_token_ids.front() == replay.back(),
        "Gemma2 append generation should match full replay token"
    );
}

void test_gemma2_metal_plan_generation_ignores_stale_kv_rows_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    auto plan = create_gemma2_metal_model_plan(model);
    const std::vector<std::uint32_t> dirty_prefix_token_ids {0, 1, 2};
    const std::vector<std::uint32_t> suffix_token_ids {3};
    const std::vector<std::uint32_t> stop_token_ids;

    (void)generate_gemma2_metal_argmax_tokens_with_plan(
        plan,
        std::span<const std::uint32_t>(dirty_prefix_token_ids),
        0,
        1,
        std::span<const std::uint32_t>(stop_token_ids)
    );

    const auto appended = generate_gemma2_metal_argmax_tokens_with_plan(
        plan,
        std::span<const std::uint32_t>(suffix_token_ids),
        1,
        1,
        std::span<const std::uint32_t>(stop_token_ids)
    );
    const std::vector<std::uint32_t> active_full_prompt {0, 3};
    const auto replay = generate_gemma2_metal_argmax_tokens(
        model,
        std::span<const std::uint32_t>(active_full_prompt),
        1
    );

    expect(appended.prefill_cache_position == 1, "Gemma2 stale-KV append cache position mismatch");
    expect(appended.generated_token_ids.size() == 1, "Gemma2 stale-KV generated count mismatch");
    expect(
        appended.generated_token_ids.front() == replay.back(),
        "Gemma2 append generation should ignore stale rows outside active context"
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
        (void)create_gemma2_metal_prefill_schedule(plan, 4, 13);
        throw TestFailure("Gemma2 Metal append prefill schedule should reject context overflow");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 Metal prefill schedule token count exceeds context length",
            "Gemma2 Metal append prefill overflow error mismatch"
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
        "creates Gemma2 Metal append prefill schedule when available",
        test_creates_gemma2_metal_append_prefill_schedule_when_available,
    },
    {
        "Gemma2 Metal plan tracks projected attention output shape when available",
        test_gemma2_metal_plan_tracks_projected_attention_output_shape_when_available,
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
        "Gemma2 Metal resident decode uses one submission and matches diagnostics when available",
        test_gemma2_metal_resident_decode_is_one_submission_and_matches_diagnostics_when_available,
    },
    {
        "Gemma2 Metal resident prefill chunks once waits once and matches incremental when available",
        test_gemma2_metal_resident_prefill_chunks_once_waits_once_and_matches_incremental_when_available,
    },
    {
        "Gemma2 Metal plan generation append matches full replay when available",
        test_gemma2_metal_plan_generation_append_matches_full_replay_when_available,
    },
    {
        "Gemma2 Metal plan generation ignores stale KV rows when available",
        test_gemma2_metal_plan_generation_ignores_stale_kv_rows_when_available,
    },
    {
        "rejects invalid Gemma2 Metal schedule shapes when available",
        test_rejects_invalid_gemma2_metal_schedule_shapes_when_available,
    },
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
