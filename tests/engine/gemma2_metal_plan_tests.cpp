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
using local_gguf_engine::engine::create_gemma2_metal_generated_token_loop;
using local_gguf_engine::engine::create_gemma2_metal_generation_runtime_shell;
using local_gguf_engine::engine::create_gemma2_metal_model_plan;
using local_gguf_engine::engine::create_gemma2_metal_prefill_schedule;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::engine::run_gemma2_metal_scheduled_prefill_block;
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
using local_gguf_engine::tensor::ops::rope_f32_by_heads_with_position_start;
using local_gguf_engine::tensor::ops::silu_f32;
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
    std::uint32_t attention_value_length = 256
) {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Metal Plan Fixture"),
        metadata_u32("gemma2.context_length", 16),
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

void overwrite_payload_bytes(
    TinyGemma2PayloadStore& store,
    std::string_view name,
    const local_gguf_engine::tensor::TensorStorage& storage
) {
    for (std::size_t index = 0; index < store.payloads.size(); ++index) {
        if (store.payloads[index].tensor.name != name) {
            continue;
        }

        expect(
            store.bytes[index].size() == storage.bytes().size(),
            std::string("Gemma2 payload fixture byte count mismatch for ") + std::string(name)
        );
        std::copy(storage.bytes().begin(), storage.bytes().end(), store.bytes[index].begin());
        return;
    }

    throw TestFailure(std::string("Gemma2 payload fixture missing tensor ") + std::string(name));
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

void expect_f32_rank3_close(
    const local_gguf_engine::tensor::TensorStorage& actual,
    const local_gguf_engine::tensor::TensorStorage& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    expect(actual.descriptor().dtype == DType::f32, std::string(message) + " dtype mismatch");
    expect(actual.descriptor().shape.rank() == 3, std::string(message) + " rank mismatch");

    const auto dim0 = actual.descriptor().shape.dimension(0);
    const auto dim1 = actual.descriptor().shape.dimension(1);
    const auto dim2 = actual.descriptor().shape.dimension(2);
    for (std::uint64_t index0 = 0; index0 < dim0; ++index0) {
        for (std::uint64_t index1 = 0; index1 < dim1; ++index1) {
            for (std::uint64_t index2 = 0; index2 < dim2; ++index2) {
                const std::array<std::uint64_t, 3> index {index0, index1, index2};
                const auto actual_value = read_f32(actual, std::span<const std::uint64_t>(index));
                const auto expected_value = read_f32(expected, std::span<const std::uint64_t>(index));
                if (std::isinf(actual_value) || std::isinf(expected_value)) {
                    expect(
                        std::isinf(actual_value)
                            && std::isinf(expected_value)
                            && std::signbit(actual_value) == std::signbit(expected_value),
                        std::string(message) + " infinity mismatch"
                    );
                    continue;
                }

                expect_close(actual_value, expected_value, tolerance, message);
            }
        }
    }
}

void expect_cache_window_matches_flat_rows(
    const local_gguf_engine::tensor::TensorStorage& cache,
    const local_gguf_engine::tensor::TensorStorage& rows,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t head_length,
    std::string_view message
) {
    expect(cache.descriptor().dtype == DType::f32, std::string(message) + " cache dtype mismatch");
    expect(rows.descriptor().dtype == DType::f32, std::string(message) + " rows dtype mismatch");
    expect(cache.descriptor().shape.rank() == 3, std::string(message) + " cache rank mismatch");
    expect(rows.descriptor().shape.rank() == 2, std::string(message) + " rows rank mismatch");
    expect(cache.descriptor().shape.dimension(1) == head_count, std::string(message) + " cache head count mismatch");
    expect(cache.descriptor().shape.dimension(2) == head_length, std::string(message) + " cache head length mismatch");
    expect(rows.descriptor().shape.dimension(1) == head_count * head_length, std::string(message) + " row width mismatch");

    const auto token_count = rows.descriptor().shape.dimension(0);
    for (std::uint64_t row = 0; row < token_count; ++row) {
        for (std::uint64_t head = 0; head < head_count; ++head) {
            for (std::uint64_t column = 0; column < head_length; ++column) {
                const std::array<std::uint64_t, 3> cache_index {cache_position + row, head, column};
                const std::array<std::uint64_t, 2> row_index {row, head * head_length + column};
                expect_close(
                    read_f32(cache, std::span<const std::uint64_t>(cache_index)),
                    read_f32(rows, std::span<const std::uint64_t>(row_index)),
                    0.0F,
                    message
                );
            }
        }
    }
}

void expect_cache_row_zero(
    const local_gguf_engine::tensor::TensorStorage& cache,
    std::uint64_t row,
    std::string_view message
) {
    const auto head_count = cache.descriptor().shape.dimension(1);
    const auto head_length = cache.descriptor().shape.dimension(2);
    for (std::uint64_t head = 0; head < head_count; ++head) {
        for (std::uint64_t column = 0; column < head_length; ++column) {
            const std::array<std::uint64_t, 3> index {row, head, column};
            expect_close(read_f32(cache, std::span<const std::uint64_t>(index)), 0.0F, 0.0F, message);
        }
    }
}

local_gguf_engine::tensor::TensorStorage flatten_f32_cache_rows(
    const local_gguf_engine::tensor::TensorStorage& cache,
    std::uint64_t head_count,
    std::uint64_t head_length
) {
    expect(cache.descriptor().dtype == DType::f32, "Gemma2 cache flatten dtype mismatch");
    expect(cache.descriptor().shape.rank() == 3, "Gemma2 cache flatten rank mismatch");
    expect(cache.descriptor().shape.dimension(1) == head_count, "Gemma2 cache flatten head count mismatch");
    expect(cache.descriptor().shape.dimension(2) == head_length, "Gemma2 cache flatten head length mismatch");

    return local_gguf_engine::tensor::TensorStorage::from_bytes(
        TensorDescriptor::from(
            Shape::from({cache.descriptor().shape.dimension(0), head_count * head_length}),
            DType::f32
        ),
        cache.bytes()
    );
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
        + backend_buffer_byte_count(scratch.attention_probabilities)
        + backend_buffer_byte_count(scratch.attention_output)
        + backend_buffer_byte_count(scratch.projected_attention_output)
        + backend_buffer_byte_count(scratch.post_attention_residual)
        + backend_buffer_byte_count(scratch.normalized_feed_forward_input)
        + backend_buffer_byte_count(scratch.feed_forward_gate)
        + backend_buffer_byte_count(scratch.feed_forward_up)
        + backend_buffer_byte_count(scratch.feed_forward_gated)
        + backend_buffer_byte_count(scratch.projected_feed_forward_output)
        + backend_buffer_byte_count(scratch.feed_forward_residual)
        + backend_buffer_byte_count(scratch.layer_handoff_normalized_attention_input)
        + backend_buffer_byte_count(scratch.logits);
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

    const auto plan = create_gemma2_metal_model_plan(model);

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
    expect(
        shell.prefill_reusable_layer_inputs.empty(),
        "Gemma2 shell one-layer fixture should not prepare later prefill layer inputs"
    );
    expect(
        shell.prefill_reusable_blocks.empty(),
        "Gemma2 shell one-layer fixture should not run later prefill blocks"
    );
    expect(
        shell.prefill_output_norm.schedule_layer_offset == 0U,
        "Gemma2 shell one-layer output norm should select first layer terminal offset"
    );
    expect(
        shell.prefill_output_norm.layer_index == 0U,
        "Gemma2 shell one-layer output norm should select first layer"
    );
    expect(
        shell.prefill_output_norm.normalized_hidden_states.descriptor() == shell.prefill.scratch().normalized_hidden,
        "Gemma2 shell one-layer output norm descriptor mismatch"
    );
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

void test_gemma2_metal_runtime_shell_runs_first_prefill_block_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr float metal_rope_shell_tolerance = 2.0e-3F;
    auto payload_store = tiny_gemma2_payload_store(true, 2);

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
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> attention_norm_values(256U, 1.0F);
    attention_norm_values[1] = 0.5F;
    const auto attention_norm = tensor_f32_vector(std::span<const float>(attention_norm_values));
    std::vector<float> next_attention_norm_values(256U, 1.0F);
    next_attention_norm_values[0] = 1.5F;
    next_attention_norm_values[1] = 0.25F;
    const auto next_attention_norm = tensor_f32_vector(std::span<const float>(next_attention_norm_values));
    std::vector<float> post_attention_norm_values(256U, 1.0F);
    post_attention_norm_values[0] = 0.25F;
    post_attention_norm_values[1] = 0.75F;
    const auto post_attention_norm = tensor_f32_vector(std::span<const float>(post_attention_norm_values));
    std::vector<float> feed_forward_norm_values(256U, 1.0F);
    feed_forward_norm_values[0] = 1.25F;
    feed_forward_norm_values[1] = 0.5F;
    const auto feed_forward_norm = tensor_f32_vector(std::span<const float>(feed_forward_norm_values));
    std::vector<float> post_feed_forward_norm_values(256U, 1.0F);
    post_feed_forward_norm_values[0] = 0.5F;
    post_feed_forward_norm_values[1] = 1.5F;
    const auto post_feed_forward_norm = tensor_f32_vector(
        std::span<const float>(post_feed_forward_norm_values)
    );

    const auto query_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 1);
    const auto key_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 2);
    const auto value_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 3);
    const auto output_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 4);
    const auto gate_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 5);
    const auto up_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 6);
    const auto down_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 7);

    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_norm.weight"), attention_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "post_attention_norm.weight"), post_attention_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_q.weight"), query_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_k.weight"), key_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_v.weight"), value_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_output.weight"), output_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "ffn_norm.weight"), feed_forward_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "post_ffw_norm.weight"), post_feed_forward_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "ffn_gate.weight"), gate_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "ffn_up.weight"), up_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "ffn_down.weight"), down_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(1, "attn_norm.weight"), next_attention_norm);

    auto view = tiny_model_view(tiny_gemma2_metadata(2), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};
    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );

    expect(
        shell.prefill_reusable_layer_inputs.size() == 1U,
        "Gemma2 first prefill block fixture should prepare one later input"
    );
    expect(
        shell.prefill_reusable_blocks.size() == 1U,
        "Gemma2 first prefill block fixture should run one later block"
    );
    expect(
        shell.prefill_reusable_layer_inputs.front().layer_index == 1U,
        "Gemma2 first prefill block next layer input mismatch"
    );
    expect(
        shell.prefill_reusable_blocks.front().schedule_layer_offset == 1U,
        "Gemma2 first prefill block reusable offset mismatch"
    );
    expect(
        shell.prefill_output_norm.layer_index == 1U,
        "Gemma2 first prefill block fixture should terminate on the reusable layer"
    );

    const auto hidden_states = copy_tensor_from_backend(shell.prefill_input.hidden_states);
    const auto normalized_attention_input =
        copy_tensor_from_backend(shell.prefill_first_layer_input.normalized_attention_input);
    const auto query = copy_tensor_from_backend(shell.prefill_first_layer_attention_projection.query);
    const auto key = copy_tensor_from_backend(shell.prefill_first_layer_attention_projection.key);
    const auto value = copy_tensor_from_backend(shell.prefill_first_layer_attention_projection.value);
    const auto positioned_query = copy_tensor_from_backend(shell.prefill_first_layer_attention_position.query);
    const auto positioned_key = copy_tensor_from_backend(shell.prefill_first_layer_attention_position.key);
    const auto scores = copy_tensor_from_backend(shell.prefill_first_layer_attention_scores.scores);
    const auto probabilities = copy_tensor_from_backend(
        shell.prefill_first_layer_attention_probabilities.probabilities
    );
    const auto attention_output = copy_tensor_from_backend(
        shell.prefill_first_layer_attention_values.attention_output
    );
    const auto projected_attention_output = copy_tensor_from_backend(
        shell.prefill_first_layer_attention_output_projection.projected_attention_output
    );
    const auto post_attention_residual = copy_tensor_from_backend(
        shell.prefill_first_layer_post_attention_residual.post_attention_residual
    );
    const auto normalized_feed_forward_input = copy_tensor_from_backend(
        shell.prefill_first_layer_feed_forward_gate_up.normalized_feed_forward_input
    );
    const auto gate = copy_tensor_from_backend(shell.prefill_first_layer_feed_forward_gate_up.gate);
    const auto up = copy_tensor_from_backend(shell.prefill_first_layer_feed_forward_gate_up.up);
    const auto gated = copy_tensor_from_backend(
        shell.prefill_first_layer_feed_forward_output.feed_forward_gated
    );
    const auto projected_feed_forward_output = copy_tensor_from_backend(
        shell.prefill_first_layer_feed_forward_output.projected_feed_forward_output
    );
    const auto feed_forward_residual = copy_tensor_from_backend(
        shell.prefill_first_layer_feed_forward_output.feed_forward_residual
    );
    const auto next_layer_input = copy_tensor_from_backend(
        shell.prefill_reusable_layer_inputs.front().normalized_attention_input
    );

    const auto expected_normalized_attention_input = rms_norm_f32(
        hidden_states,
        attention_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto expected_query = matmul_f32_quantized_gguf_weight(
        normalized_attention_input,
        query_weight,
        BackendKind::cpu
    );
    const auto expected_key = matmul_f32_quantized_gguf_weight(
        normalized_attention_input,
        key_weight,
        BackendKind::cpu
    );
    const auto expected_value = matmul_f32_quantized_gguf_weight(
        normalized_attention_input,
        value_weight,
        BackendKind::cpu
    );
    const auto expected_positioned_query = rope_f32_by_heads_with_position_start(
        query,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_key_length,
        shell.plan.config().attention_key_length,
        10000.0F
    );
    const auto expected_positioned_key = rope_f32_by_heads_with_position_start(
        key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.plan.config().attention_key_length,
        10000.0F
    );
    const auto expected_scores = causal_grouped_attention_scores_f32(
        positioned_query,
        positioned_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.prefill.layers().front().attention_token_count,
        shell.prefill_first_layer_attention_scores.scale,
        shell.plan.config().attention_logit_softcap
    );
    const auto expected_probabilities = softmax_f32(scores);
    const auto expected_attention_output = causal_grouped_attention_values_f32(
        probabilities,
        value,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        shell.prefill.layers().front().attention_token_count
    );
    const auto expected_projected_attention_output = matmul_f32_quantized_gguf_weight(
        attention_output,
        output_weight,
        BackendKind::cpu
    );
    const auto expected_post_attention_residual = add_f32(
        hidden_states,
        rms_norm_f32(
            projected_attention_output,
            post_attention_norm,
            shell.plan.config().rms_norm_epsilon
        )
    );
    const auto expected_normalized_feed_forward_input = rms_norm_f32(
        post_attention_residual,
        feed_forward_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto expected_gate = matmul_f32_quantized_gguf_weight(
        normalized_feed_forward_input,
        gate_weight,
        BackendKind::cpu
    );
    const auto expected_up = matmul_f32_quantized_gguf_weight(
        normalized_feed_forward_input,
        up_weight,
        BackendKind::cpu
    );
    const auto expected_gated = multiply_f32(silu_f32(gate), up);
    const auto expected_projected_feed_forward_output = matmul_f32_quantized_gguf_weight(
        gated,
        down_weight,
        BackendKind::cpu
    );
    const auto expected_feed_forward_residual = add_f32(
        post_attention_residual,
        rms_norm_f32(
            projected_feed_forward_output,
            post_feed_forward_norm,
            shell.plan.config().rms_norm_epsilon
        )
    );
    const auto expected_next_layer_input = rms_norm_f32(
        feed_forward_residual,
        next_attention_norm,
        shell.plan.config().rms_norm_epsilon
    );

    expect(
        shell.prefill_first_layer_feed_forward_output.layer_index == 0U,
        "Gemma2 first prefill block FFN residual layer mismatch"
    );
    expect(
        shell.prefill_first_layer_feed_forward_output.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 first prefill block FFN residual phase mismatch"
    );
    expect(
        shell.prefill_first_layer_feed_forward_output.feed_forward_residual.descriptor()
            == shell.prefill.scratch().feed_forward_residual,
        "Gemma2 first prefill block FFN residual descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            shell.prefill_first_layer_feed_forward_output.feed_forward_residual,
            shell.prefill_input.hidden_states
        ),
        "Gemma2 first prefill block FFN residual should not alias prompt embeddings"
    );

    expect_f32_matrix_close(
        normalized_attention_input,
        expected_normalized_attention_input,
        1.0e-5F,
        "Gemma2 first prefill block normalized input"
    );
    expect_f32_matrix_close(query, expected_query, 1.0e-2F, "Gemma2 first prefill block query");
    expect_f32_matrix_close(key, expected_key, 1.0e-2F, "Gemma2 first prefill block key");
    expect_f32_matrix_close(value, expected_value, 1.0e-2F, "Gemma2 first prefill block value");
    expect_f32_matrix_close(
        positioned_query,
        expected_positioned_query,
        metal_rope_shell_tolerance,
        "Gemma2 first prefill block positioned query"
    );
    expect_f32_matrix_close(
        positioned_key,
        expected_positioned_key,
        metal_rope_shell_tolerance,
        "Gemma2 first prefill block positioned key"
    );
    expect_f32_rank3_close(scores, expected_scores, 1.0e-3F, "Gemma2 first prefill block scores");
    expect_f32_rank3_close(
        probabilities,
        expected_probabilities,
        1.0e-4F,
        "Gemma2 first prefill block probabilities"
    );
    expect_f32_matrix_close(
        attention_output,
        expected_attention_output,
        1.0e-3F,
        "Gemma2 first prefill block attention output"
    );
    expect_f32_matrix_close(
        projected_attention_output,
        expected_projected_attention_output,
        1.0e-2F,
        "Gemma2 first prefill block projected attention"
    );
    expect_f32_matrix_close(
        post_attention_residual,
        expected_post_attention_residual,
        1.0e-2F,
        "Gemma2 first prefill block post-attention residual"
    );
    expect_f32_matrix_close(
        normalized_feed_forward_input,
        expected_normalized_feed_forward_input,
        1.0e-5F,
        "Gemma2 first prefill block normalized FFN input"
    );
    expect_f32_matrix_close(gate, expected_gate, 1.0e-2F, "Gemma2 first prefill block gate");
    expect_f32_matrix_close(up, expected_up, 1.0e-2F, "Gemma2 first prefill block up");
    expect_f32_matrix_close(gated, expected_gated, 1.0e-2F, "Gemma2 first prefill block gated FFN");
    expect_f32_matrix_close(
        projected_feed_forward_output,
        expected_projected_feed_forward_output,
        1.0e-2F,
        "Gemma2 first prefill block projected FFN"
    );
    expect_f32_matrix_close(
        feed_forward_residual,
        expected_feed_forward_residual,
        1.0e-2F,
        "Gemma2 first prefill block FFN residual"
    );
    expect_f32_matrix_close(
        next_layer_input,
        expected_next_layer_input,
        1.0e-5F,
        "Gemma2 first prefill block next reusable input"
    );
}

void test_gemma2_metal_prefill_block_runner_executes_scheduled_layer_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr float metal_rope_shell_tolerance = 2.0e-3F;
    auto payload_store = tiny_gemma2_payload_store(true, 4);
    const auto query_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 31);
    const auto key_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 32);
    const auto value_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 33);
    const auto output_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 34);
    const auto gate_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 35);
    const auto up_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 36);
    const auto down_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 37);
    std::vector<float> post_attention_norm_values(256U, 1.0F);
    post_attention_norm_values[0] = 0.5F;
    post_attention_norm_values[1] = 1.25F;
    const auto post_attention_norm = tensor_f32_vector(std::span<const float>(post_attention_norm_values));
    std::vector<float> feed_forward_norm_values(256U, 1.0F);
    feed_forward_norm_values[0] = 1.5F;
    feed_forward_norm_values[1] = 0.75F;
    const auto feed_forward_norm = tensor_f32_vector(std::span<const float>(feed_forward_norm_values));
    std::vector<float> post_feed_forward_norm_values(256U, 1.0F);
    post_feed_forward_norm_values[0] = 0.25F;
    post_feed_forward_norm_values[1] = 1.75F;
    const auto post_feed_forward_norm = tensor_f32_vector(
        std::span<const float>(post_feed_forward_norm_values)
    );

    overwrite_payload_bytes(payload_store, block_tensor_name(3, "post_attention_norm.weight"), post_attention_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_q.weight"), query_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_k.weight"), key_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_v.weight"), value_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_output.weight"), output_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "ffn_norm.weight"), feed_forward_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "post_ffw_norm.weight"), post_feed_forward_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "ffn_gate.weight"), gate_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "ffn_up.weight"), up_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "ffn_down.weight"), down_weight);

    auto view = tiny_model_view(tiny_gemma2_metadata(4), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);
    const auto prefill = create_gemma2_metal_prefill_schedule(plan, 2);

    std::vector<float> normalized_values(2U * 256U, 0.0F);
    std::vector<float> residual_values(2U * 256U, 0.0F);
    for (std::size_t column = 0; column < 8U; ++column) {
        normalized_values[column] = 0.125F * static_cast<float>(column + 1U);
        normalized_values[256U + column] = column % 2U == 0U ? -0.2F : 0.3F;
        residual_values[column] = 0.05F * static_cast<float>(column + 1U);
        residual_values[256U + column] = -0.03F * static_cast<float>(column + 1U);
    }
    const auto normalized_input = tensor_f32_matrix(2, 256, std::span<const float>(normalized_values));
    const auto residual_input = tensor_f32_matrix(2, 256, std::span<const float>(residual_values));
    const auto normalized_input_buffer = copy_tensor_to_backend(normalized_input, BackendKind::metal);
    const auto residual_input_buffer = copy_tensor_to_backend(residual_input, BackendKind::metal);

    const auto block = run_gemma2_metal_scheduled_prefill_block(
        plan,
        prefill,
        3,
        normalized_input_buffer,
        residual_input_buffer
    );

    const auto query = copy_tensor_from_backend(block.attention_projection.query);
    const auto expected_query = matmul_f32_quantized_gguf_weight(normalized_input, query_weight, BackendKind::cpu);
    const auto key = copy_tensor_from_backend(block.attention_projection.key);
    const auto expected_key = matmul_f32_quantized_gguf_weight(normalized_input, key_weight, BackendKind::cpu);
    const auto value = copy_tensor_from_backend(block.attention_projection.value);
    const auto expected_value = matmul_f32_quantized_gguf_weight(normalized_input, value_weight, BackendKind::cpu);
    const auto positioned_query = copy_tensor_from_backend(block.attention_position.query);
    const auto expected_positioned_query = rope_f32_by_heads_with_position_start(
        expected_query,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_key_length,
        plan.config().attention_key_length,
        10000.0F
    );
    const auto positioned_key = copy_tensor_from_backend(block.attention_position.key);
    const auto expected_positioned_key = rope_f32_by_heads_with_position_start(
        expected_key,
        prefill.cache_position(),
        plan.config().attention_head_count_kv,
        plan.config().attention_key_length,
        plan.config().attention_key_length,
        10000.0F
    );
    const auto scores = copy_tensor_from_backend(block.attention_scores.scores);
    const auto expected_scores = causal_grouped_attention_scores_f32(
        expected_positioned_query,
        expected_positioned_key,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        plan.config().attention_key_length,
        prefill.layers()[3].attention_token_count,
        block.attention_scores.scale,
        plan.config().attention_logit_softcap
    );
    const auto probabilities = copy_tensor_from_backend(block.attention_probabilities.probabilities);
    const auto expected_probabilities = softmax_f32(expected_scores);
    const auto attention_output = copy_tensor_from_backend(block.attention_values.attention_output);
    const auto expected_attention_output = causal_grouped_attention_values_f32(
        expected_probabilities,
        expected_value,
        prefill.cache_position(),
        plan.config().attention_head_count,
        plan.config().attention_head_count_kv,
        plan.config().attention_value_length,
        prefill.layers()[3].attention_token_count
    );
    const auto projected_attention_output = copy_tensor_from_backend(
        block.attention_output_projection.projected_attention_output
    );
    const auto expected_projected_attention_output = matmul_f32_quantized_gguf_weight(
        expected_attention_output,
        output_weight,
        BackendKind::cpu
    );
    const auto post_attention_residual = copy_tensor_from_backend(
        block.post_attention_residual.post_attention_residual
    );
    const auto expected_post_attention_residual = add_f32(
        residual_input,
        rms_norm_f32(
            expected_projected_attention_output,
            post_attention_norm,
            plan.config().rms_norm_epsilon
        )
    );
    const auto normalized_feed_forward_input = copy_tensor_from_backend(
        block.feed_forward_gate_up.normalized_feed_forward_input
    );
    const auto expected_normalized_feed_forward_input = rms_norm_f32(
        expected_post_attention_residual,
        feed_forward_norm,
        plan.config().rms_norm_epsilon
    );
    const auto gate = copy_tensor_from_backend(block.feed_forward_gate_up.gate);
    const auto expected_gate = matmul_f32_quantized_gguf_weight(
        expected_normalized_feed_forward_input,
        gate_weight,
        BackendKind::cpu
    );
    const auto up = copy_tensor_from_backend(block.feed_forward_gate_up.up);
    const auto expected_up = matmul_f32_quantized_gguf_weight(
        expected_normalized_feed_forward_input,
        up_weight,
        BackendKind::cpu
    );
    const auto gated = copy_tensor_from_backend(block.feed_forward_output.feed_forward_gated);
    const auto expected_gated = multiply_f32(silu_f32(expected_gate), expected_up);
    const auto projected_feed_forward_output = copy_tensor_from_backend(
        block.feed_forward_output.projected_feed_forward_output
    );
    const auto expected_projected_feed_forward_output = matmul_f32_quantized_gguf_weight(
        expected_gated,
        down_weight,
        BackendKind::cpu
    );
    const auto feed_forward_residual = copy_tensor_from_backend(
        block.feed_forward_output.feed_forward_residual
    );
    const auto expected_feed_forward_residual = add_f32(
        expected_post_attention_residual,
        rms_norm_f32(
            expected_projected_feed_forward_output,
            post_feed_forward_norm,
            plan.config().rms_norm_epsilon
        )
    );

    expect(block.schedule_layer_offset == 3, "Gemma2 prefill block runner schedule offset mismatch");
    expect(block.layer_index == prefill.layers()[3].layer_index, "Gemma2 prefill block runner layer mismatch");
    expect(block.phase == Gemma2MetalSchedulePhase::prefill, "Gemma2 prefill block runner phase mismatch");
    expect(
        block.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 prefill block runner scratch ownership mismatch"
    );
    expect(
        block.attention_projection.query.descriptor() == prefill.scratch().query,
        "Gemma2 prefill block runner query descriptor mismatch"
    );
    expect(
        block.attention_scores.scores.descriptor() == prefill.scratch().attention_scores,
        "Gemma2 prefill block runner score descriptor mismatch"
    );
    expect(
        block.feed_forward_output.feed_forward_residual.descriptor() == prefill.scratch().feed_forward_residual,
        "Gemma2 prefill block runner FFN residual descriptor mismatch"
    );
    expect(
        block.feed_forward_output.feed_forward_residual.backend() == BackendKind::metal,
        "Gemma2 prefill block runner FFN residual backend mismatch"
    );
    expect(
        !shares_backend_storage(block.attention_projection.query, normalized_input_buffer),
        "Gemma2 prefill block runner query should not alias normalized input"
    );
    expect(
        !shares_backend_storage(block.attention_projection.query, plan.weights().blocks[3].attention.query_weight),
        "Gemma2 prefill block runner query should not alias query weight"
    );
    expect(
        !shares_backend_storage(block.post_attention_residual.post_attention_residual, residual_input_buffer),
        "Gemma2 prefill block runner post-attention residual should not alias residual input"
    );
    expect(
        !shares_backend_storage(
            block.feed_forward_output.feed_forward_residual,
            block.post_attention_residual.post_attention_residual
        ),
        "Gemma2 prefill block runner FFN residual should not alias post-attention residual"
    );
    expect(
        !shares_backend_storage(
            block.feed_forward_output.feed_forward_residual,
            plan.weights().blocks[3].feed_forward.post_feed_forward_norm_weight
        ),
        "Gemma2 prefill block runner FFN residual should not alias post-FFN norm weight"
    );

    expect_f32_matrix_close(query, expected_query, 1.0e-2F, "Gemma2 prefill block runner query");
    expect_f32_matrix_close(key, expected_key, 1.0e-2F, "Gemma2 prefill block runner key");
    expect_f32_matrix_close(value, expected_value, 1.0e-2F, "Gemma2 prefill block runner value");
    expect_f32_matrix_close(
        positioned_query,
        expected_positioned_query,
        metal_rope_shell_tolerance,
        "Gemma2 prefill block runner positioned query"
    );
    expect_f32_matrix_close(
        positioned_key,
        expected_positioned_key,
        metal_rope_shell_tolerance,
        "Gemma2 prefill block runner positioned key"
    );
    expect_f32_rank3_close(scores, expected_scores, 1.0e-3F, "Gemma2 prefill block runner scores");
    expect_f32_rank3_close(
        probabilities,
        expected_probabilities,
        1.0e-5F,
        "Gemma2 prefill block runner probabilities"
    );
    expect_f32_matrix_close(
        attention_output,
        expected_attention_output,
        1.0e-3F,
        "Gemma2 prefill block runner attention output"
    );
    expect_f32_matrix_close(
        projected_attention_output,
        expected_projected_attention_output,
        1.0e-2F,
        "Gemma2 prefill block runner projected attention"
    );
    expect_f32_matrix_close(
        post_attention_residual,
        expected_post_attention_residual,
        1.0e-2F,
        "Gemma2 prefill block runner post-attention residual"
    );
    expect_f32_matrix_close(
        normalized_feed_forward_input,
        expected_normalized_feed_forward_input,
        1.0e-5F,
        "Gemma2 prefill block runner normalized FFN input"
    );
    expect_f32_matrix_close(gate, expected_gate, 1.0e-2F, "Gemma2 prefill block runner gate");
    expect_f32_matrix_close(up, expected_up, 1.0e-2F, "Gemma2 prefill block runner up");
    expect_f32_matrix_close(gated, expected_gated, 1.0e-2F, "Gemma2 prefill block runner gated FFN");
    expect_f32_matrix_close(
        projected_feed_forward_output,
        expected_projected_feed_forward_output,
        1.0e-2F,
        "Gemma2 prefill block runner projected FFN"
    );
    expect_f32_matrix_close(
        feed_forward_residual,
        expected_feed_forward_residual,
        1.0e-2F,
        "Gemma2 prefill block runner FFN residual"
    );

    std::vector<float> wrong_shape_values(256U, 0.0F);
    const auto wrong_shape_input = tensor_f32_matrix(1, 256, std::span<const float>(wrong_shape_values));
    const auto wrong_shape_buffer = copy_tensor_to_backend(wrong_shape_input, BackendKind::metal);
    try {
        (void)run_gemma2_metal_scheduled_prefill_block(
            plan,
            prefill,
            3,
            wrong_shape_buffer,
            residual_input_buffer
        );
        throw TestFailure("Gemma2 prefill block runner should reject wrong normalized input shape");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what())
                == "Gemma2 Metal prefill block normalized attention input must match scheduled input shape",
            "Gemma2 prefill block runner wrong input shape error mismatch"
        );
    }

    try {
        (void)run_gemma2_metal_scheduled_prefill_block(
            plan,
            prefill,
            4,
            normalized_input_buffer,
            residual_input_buffer
        );
        throw TestFailure("Gemma2 prefill block runner should reject unscheduled layer offset");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 Metal prefill block runner layer offset is not scheduled",
            "Gemma2 prefill block runner unscheduled layer error mismatch"
        );
    }
}

void test_gemma2_metal_runtime_shell_loops_over_later_reusable_blocks_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr float metal_rope_shell_tolerance = 2.0e-3F;
    auto payload_store = tiny_gemma2_payload_store(true, 6);
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
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> output_norm_values(256U, 1.0F);
    output_norm_values[0] = 0.25F;
    output_norm_values[1] = 1.75F;
    const auto output_norm = tensor_f32_vector(std::span<const float>(output_norm_values));
    overwrite_payload_bytes(payload_store, "output_norm.weight", output_norm);

    std::vector<float> sixth_attention_norm_values(256U, 1.0F);
    sixth_attention_norm_values[0] = 1.5F;
    sixth_attention_norm_values[1] = 0.25F;
    const auto sixth_attention_norm = tensor_f32_vector(std::span<const float>(sixth_attention_norm_values));
    std::vector<float> sixth_post_attention_norm_values(256U, 1.0F);
    sixth_post_attention_norm_values[0] = 0.625F;
    sixth_post_attention_norm_values[1] = 1.375F;
    const auto sixth_post_attention_norm = tensor_f32_vector(
        std::span<const float>(sixth_post_attention_norm_values)
    );
    std::vector<float> sixth_feed_forward_norm_values(256U, 1.0F);
    sixth_feed_forward_norm_values[0] = 1.75F;
    sixth_feed_forward_norm_values[1] = 0.5F;
    const auto sixth_feed_forward_norm = tensor_f32_vector(
        std::span<const float>(sixth_feed_forward_norm_values)
    );
    std::vector<float> sixth_post_feed_forward_norm_values(256U, 1.0F);
    sixth_post_feed_forward_norm_values[0] = 0.875F;
    sixth_post_feed_forward_norm_values[1] = 1.625F;
    const auto sixth_post_feed_forward_norm = tensor_f32_vector(
        std::span<const float>(sixth_post_feed_forward_norm_values)
    );
    const auto sixth_query_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 51);
    const auto sixth_key_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 52);
    const auto sixth_value_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 53);
    const auto sixth_output_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 54);
    const auto sixth_gate_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 55);
    const auto sixth_up_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 56);
    const auto sixth_down_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 57);

    overwrite_payload_bytes(payload_store, block_tensor_name(5, "attn_norm.weight"), sixth_attention_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "post_attention_norm.weight"), sixth_post_attention_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "attn_q.weight"), sixth_query_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "attn_k.weight"), sixth_key_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "attn_v.weight"), sixth_value_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "attn_output.weight"), sixth_output_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "ffn_norm.weight"), sixth_feed_forward_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "post_ffw_norm.weight"), sixth_post_feed_forward_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "ffn_gate.weight"), sixth_gate_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "ffn_up.weight"), sixth_up_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(5, "ffn_down.weight"), sixth_down_weight);

    auto view = tiny_model_view(tiny_gemma2_metadata(6), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {0, 1};

    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );

    expect(
        shell.prefill_reusable_layer_inputs.size() == 5U,
        "Gemma2 shell six-layer fixture should prepare five later prefill layer inputs"
    );
    expect(
        shell.prefill_reusable_blocks.size() == 5U,
        "Gemma2 shell six-layer fixture should run five later prefill blocks"
    );
    const auto& fourth_reusable_layer_input = shell.prefill_reusable_layer_inputs[3];
    const auto& fifth_reusable_layer_input = shell.prefill_reusable_layer_inputs[4];
    const auto& fourth_reusable_block = shell.prefill_reusable_blocks[3];
    const auto& fifth_reusable_block = shell.prefill_reusable_blocks[4];

    expect(fourth_reusable_layer_input.layer_index == 4U, "Gemma2 shell fifth layer input index mismatch");
    expect(fifth_reusable_layer_input.layer_index == 5U, "Gemma2 shell sixth layer input index mismatch");
    expect(
        fourth_reusable_layer_input.normalized_attention_input.descriptor()
            == shell.prefill.scratch().layer_handoff_normalized_attention_input,
        "Gemma2 shell fifth layer input descriptor mismatch"
    );
    expect(
        fifth_reusable_layer_input.normalized_attention_input.descriptor()
            == shell.prefill.scratch().layer_handoff_normalized_attention_input,
        "Gemma2 shell sixth layer input descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            fourth_reusable_layer_input.normalized_attention_input,
            fourth_reusable_block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 shell fifth layer input should not alias fifth layer residual"
    );
    expect(
        !shares_backend_storage(
            fifth_reusable_layer_input.normalized_attention_input,
            fifth_reusable_block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 shell sixth layer input should not alias sixth layer residual"
    );
    expect(
        !shares_backend_storage(
            fifth_reusable_layer_input.normalized_attention_input,
            fourth_reusable_layer_input.normalized_attention_input
        ),
        "Gemma2 shell reusable layer inputs should not alias each other"
    );
    expect(fourth_reusable_block.schedule_layer_offset == 4U, "Gemma2 shell fifth layer block offset mismatch");
    expect(fifth_reusable_block.schedule_layer_offset == 5U, "Gemma2 shell sixth layer block offset mismatch");
    expect(fourth_reusable_block.layer_index == 4U, "Gemma2 shell fifth layer block layer mismatch");
    expect(fifth_reusable_block.layer_index == 5U, "Gemma2 shell sixth layer block layer mismatch");
    expect(
        fourth_reusable_block.feed_forward_output.feed_forward_residual.descriptor()
            == shell.prefill.scratch().feed_forward_residual,
        "Gemma2 shell fifth layer block FFN residual descriptor mismatch"
    );
    expect(
        fifth_reusable_block.feed_forward_output.feed_forward_residual.descriptor()
            == shell.prefill.scratch().feed_forward_residual,
        "Gemma2 shell sixth layer block FFN residual descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            fourth_reusable_block.feed_forward_output.feed_forward_residual,
            shell.prefill_reusable_blocks[2].feed_forward_output.feed_forward_residual
        ),
        "Gemma2 shell fifth layer block FFN residual should not alias previous block"
    );
    expect(
        !shares_backend_storage(
            fifth_reusable_block.feed_forward_output.feed_forward_residual,
            fourth_reusable_block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 shell sixth layer block FFN residual should not alias previous block"
    );

    const auto previous_reusable_residual = copy_tensor_from_backend(
        fourth_reusable_block.feed_forward_output.feed_forward_residual
    );
    const auto actual_sixth_layer_input = copy_tensor_from_backend(
        fifth_reusable_layer_input.normalized_attention_input
    );
    const auto expected_sixth_layer_input = rms_norm_f32(
        previous_reusable_residual,
        sixth_attention_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto sixth_query = matmul_f32_quantized_gguf_weight(
        expected_sixth_layer_input,
        sixth_query_weight,
        BackendKind::cpu
    );
    const auto sixth_key = matmul_f32_quantized_gguf_weight(
        expected_sixth_layer_input,
        sixth_key_weight,
        BackendKind::cpu
    );
    const auto sixth_value = matmul_f32_quantized_gguf_weight(
        expected_sixth_layer_input,
        sixth_value_weight,
        BackendKind::cpu
    );
    const auto sixth_positioned_query = rope_f32_by_heads_with_position_start(
        sixth_query,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_key_length,
        shell.plan.config().attention_key_length,
        10000.0F
    );
    const auto sixth_positioned_key = rope_f32_by_heads_with_position_start(
        sixth_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.plan.config().attention_key_length,
        10000.0F
    );
    const auto sixth_scores = causal_grouped_attention_scores_f32(
        sixth_positioned_query,
        sixth_positioned_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.prefill.layers()[5].attention_token_count,
        fifth_reusable_block.attention_scores.scale,
        shell.plan.config().attention_logit_softcap
    );
    const auto sixth_probabilities = softmax_f32(sixth_scores);
    const auto sixth_attention_output = causal_grouped_attention_values_f32(
        sixth_probabilities,
        sixth_value,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        shell.prefill.layers()[5].attention_token_count
    );
    const auto sixth_projected_attention = matmul_f32_quantized_gguf_weight(
        sixth_attention_output,
        sixth_output_weight,
        BackendKind::cpu
    );
    const auto sixth_post_attention_residual = add_f32(
        previous_reusable_residual,
        rms_norm_f32(
            sixth_projected_attention,
            sixth_post_attention_norm,
            shell.plan.config().rms_norm_epsilon
        )
    );
    const auto sixth_normalized_feed_forward = rms_norm_f32(
        sixth_post_attention_residual,
        sixth_feed_forward_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto sixth_gate = matmul_f32_quantized_gguf_weight(
        sixth_normalized_feed_forward,
        sixth_gate_weight,
        BackendKind::cpu
    );
    const auto sixth_up = matmul_f32_quantized_gguf_weight(
        sixth_normalized_feed_forward,
        sixth_up_weight,
        BackendKind::cpu
    );
    const auto sixth_gated = multiply_f32(silu_f32(sixth_gate), sixth_up);
    const auto sixth_projected_feed_forward = matmul_f32_quantized_gguf_weight(
        sixth_gated,
        sixth_down_weight,
        BackendKind::cpu
    );
    const auto expected_sixth_feed_forward_residual = add_f32(
        sixth_post_attention_residual,
        rms_norm_f32(
            sixth_projected_feed_forward,
            sixth_post_feed_forward_norm,
            shell.plan.config().rms_norm_epsilon
        )
    );
    const auto expected_prefill_output_norm = rms_norm_f32(
        expected_sixth_feed_forward_residual,
        output_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto expected_prefill_raw_logits = matmul_f32_f16_gguf_weight(
        expected_prefill_output_norm,
        embedding,
        BackendKind::cpu
    );
    const auto expected_prefill_logits = softcap_f32(
        expected_prefill_raw_logits,
        shell.plan.config().final_logit_softcap
    );
    const auto expected_next_token = top_logit_tokens_f32(expected_prefill_logits, 1).front();
    const auto actual_sixth_positioned_query = copy_tensor_from_backend(
        fifth_reusable_block.attention_position.query
    );
    const auto actual_sixth_feed_forward_residual = copy_tensor_from_backend(
        fifth_reusable_block.feed_forward_output.feed_forward_residual
    );
    const auto actual_prefill_output_norm = copy_tensor_from_backend(
        shell.prefill_output_norm.normalized_hidden_states
    );
    const auto actual_prefill_raw_logits = copy_tensor_from_backend(shell.prefill_logits.raw_logits);
    const auto actual_prefill_logits = copy_tensor_from_backend(shell.prefill_logits.logits);

    expect_f32_matrix_close(
        actual_sixth_layer_input,
        expected_sixth_layer_input,
        1.0e-5F,
        "Gemma2 shell loop sixth normalized attention input"
    );
    expect_f32_matrix_close(
        actual_sixth_positioned_query,
        sixth_positioned_query,
        metal_rope_shell_tolerance,
        "Gemma2 shell loop sixth positioned query"
    );
    expect_f32_matrix_close(
        actual_sixth_feed_forward_residual,
        expected_sixth_feed_forward_residual,
        1.0e-2F,
        "Gemma2 shell loop sixth FFN residual"
    );
    expect(
        shell.prefill_output_norm.schedule_layer_offset == 5U,
        "Gemma2 shell output norm terminal schedule offset mismatch"
    );
    expect(shell.prefill_output_norm.layer_index == 5U, "Gemma2 shell output norm terminal layer mismatch");
    expect(
        shell.prefill_output_norm.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell output norm phase mismatch"
    );
    expect(
        shell.prefill_output_norm.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell output norm scratch ownership mismatch"
    );
    expect(
        shell.prefill_output_norm.hidden_width == shell.plan.config().embedding_length,
        "Gemma2 shell output norm hidden width mismatch"
    );
    expect(
        shell.prefill_output_norm.normalized_hidden_states.backend() == BackendKind::metal,
        "Gemma2 shell output norm backend mismatch"
    );
    expect(
        shell.prefill_output_norm.normalized_hidden_states.descriptor()
            == shell.prefill.scratch().normalized_hidden,
        "Gemma2 shell output norm descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            shell.prefill_output_norm.normalized_hidden_states,
            fifth_reusable_block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 shell output norm should not alias terminal residual"
    );
    expect(
        !shares_backend_storage(
            shell.prefill_output_norm.normalized_hidden_states,
            shell.plan.weights().output_norm_weight
        ),
        "Gemma2 shell output norm should not alias output norm weight"
    );
    expect_f32_matrix_close(
        actual_prefill_output_norm,
        expected_prefill_output_norm,
        1.0e-5F,
        "Gemma2 shell output norm normalized hidden states"
    );
    expect(
        shell.prefill_logits.schedule_layer_offset == 5U,
        "Gemma2 shell logits terminal schedule offset mismatch"
    );
    expect(shell.prefill_logits.layer_index == 5U, "Gemma2 shell logits terminal layer mismatch");
    expect(
        shell.prefill_logits.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell logits phase mismatch"
    );
    expect(
        shell.prefill_logits.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell logits scratch ownership mismatch"
    );
    expect(
        shell.prefill_logits.hidden_width == shell.plan.config().embedding_length,
        "Gemma2 shell logits hidden width mismatch"
    );
    expect(
        shell.prefill_logits.vocab_size == shell.plan.config().vocab_size,
        "Gemma2 shell logits vocab size mismatch"
    );
    expect(
        shell.prefill_logits.final_logit_softcap == shell.plan.config().final_logit_softcap,
        "Gemma2 shell logits final softcap mismatch"
    );
    expect(shell.prefill_logits.raw_logits.backend() == BackendKind::metal, "Gemma2 shell raw logits backend mismatch");
    expect(shell.prefill_logits.logits.backend() == BackendKind::metal, "Gemma2 shell logits backend mismatch");
    expect(
        shell.prefill_logits.raw_logits.descriptor() == shell.prefill.scratch().logits,
        "Gemma2 shell raw logits descriptor mismatch"
    );
    expect(
        shell.prefill_logits.logits.descriptor() == shell.prefill.scratch().logits,
        "Gemma2 shell logits descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            shell.prefill_logits.raw_logits,
            shell.prefill_output_norm.normalized_hidden_states
        ),
        "Gemma2 shell raw logits should not alias output norm"
    );
    expect(
        !shares_backend_storage(shell.prefill_logits.raw_logits, shell.plan.weights().output_weight),
        "Gemma2 shell raw logits should not alias output weight"
    );
    expect(
        !shares_backend_storage(shell.prefill_logits.logits, shell.prefill_logits.raw_logits),
        "Gemma2 shell softcapped logits should not alias raw logits"
    );
    expect_f32_matrix_close(
        actual_prefill_raw_logits,
        expected_prefill_raw_logits,
        1.0e-2F,
        "Gemma2 shell prefill raw logits"
    );
    expect_f32_matrix_close(
        actual_prefill_logits,
        expected_prefill_logits,
        1.0e-2F,
        "Gemma2 shell prefill softcapped logits"
    );
    expect(
        shell.prefill_next_token_selection.schedule_layer_offset == 5U,
        "Gemma2 shell next-token selection terminal schedule offset mismatch"
    );
    expect(
        shell.prefill_next_token_selection.layer_index == 5U,
        "Gemma2 shell next-token selection terminal layer mismatch"
    );
    expect(
        shell.prefill_next_token_selection.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell next-token selection phase mismatch"
    );
    expect(
        shell.prefill_next_token_selection.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell next-token selection scratch ownership mismatch"
    );
    expect(
        shell.prefill_next_token_selection.logits_row_index == prompt_token_ids.size() - 1U,
        "Gemma2 shell next-token selection logits row mismatch"
    );
    expect(
        shell.prefill_next_token_selection.vocab_size == shell.plan.config().vocab_size,
        "Gemma2 shell next-token selection vocab size mismatch"
    );
    expect(
        shell.prefill_next_token_selection.token_id == expected_next_token.token,
        "Gemma2 shell next-token selection token mismatch"
    );
    expect_close(
        shell.prefill_next_token_selection.logit,
        expected_next_token.logit,
        1.0e-2F,
        "Gemma2 shell next-token selection logit mismatch"
    );
}

void test_gemma2_metal_runtime_shell_writes_prefill_kv_cache_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 4);
    std::vector<float> embedding_values(256U * 4U, 0.0F);
    for (std::size_t column = 0; column < 8U; ++column) {
        embedding_values[3U * 256U + column] = 0.25F * static_cast<float>(column + 1U);
        embedding_values[1U * 256U + column] = -0.125F * static_cast<float>(column + 1U);
    }
    const auto embedding = tensor_f16_gguf_matrix(256, 4, std::span<const float>(embedding_values));
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> first_attention_norm_values(256U, 1.0F);
    first_attention_norm_values[0] = 1.25F;
    first_attention_norm_values[1] = 0.5F;
    overwrite_payload_bytes(
        payload_store,
        block_tensor_name(0, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(first_attention_norm_values))
    );
    std::vector<float> fourth_attention_norm_values(256U, 1.0F);
    fourth_attention_norm_values[0] = 0.75F;
    fourth_attention_norm_values[1] = 1.5F;
    overwrite_payload_bytes(
        payload_store,
        block_tensor_name(3, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(fourth_attention_norm_values))
    );

    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_k.weight"), tensor_q4_k_gguf_weight_matrix(256, 256, 12));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_v.weight"), tensor_q6_k_gguf_weight_matrix(256, 256, 13));
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_k.weight"), tensor_q4_k_gguf_weight_matrix(256, 256, 42));
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_v.weight"), tensor_q6_k_gguf_weight_matrix(256, 256, 43));

    auto view = tiny_model_view(tiny_gemma2_metadata(4), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};
    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );

    expect(shell.prefill_reusable_blocks.size() == 3U, "Gemma2 KV write fixture should run three later blocks");
    expect(
        shell.prefill_kv_cache_writes.size() == shell.prefill.layers().size(),
        "Gemma2 shell should record one prefill KV write per scheduled layer"
    );

    for (std::size_t offset = 0; offset < shell.prefill_kv_cache_writes.size(); ++offset) {
        const auto& write = shell.prefill_kv_cache_writes[offset];
        expect(write.schedule_layer_offset == offset, "Gemma2 prefill KV write schedule offset mismatch");
        expect(write.layer_index == shell.prefill.layers()[offset].layer_index, "Gemma2 prefill KV write layer mismatch");
        expect(write.phase == Gemma2MetalSchedulePhase::prefill, "Gemma2 prefill KV write phase mismatch");
        expect(
            write.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
            "Gemma2 prefill KV write scratch ownership mismatch"
        );
        expect(write.cache_position == shell.prefill.cache_position(), "Gemma2 prefill KV write position mismatch");
        expect(write.token_count == prompt_token_ids.size(), "Gemma2 prefill KV write token count mismatch");
        expect(
            write.key_head_count == shell.plan.config().attention_head_count_kv,
            "Gemma2 prefill KV write key head count mismatch"
        );
        expect(
            write.key_length == shell.plan.config().attention_key_length,
            "Gemma2 prefill KV write key length mismatch"
        );
        expect(
            write.value_head_count == shell.plan.config().attention_head_count_kv,
            "Gemma2 prefill KV write value head count mismatch"
        );
        expect(
            write.value_length == shell.plan.config().attention_value_length,
            "Gemma2 prefill KV write value length mismatch"
        );
        expect(write.key_cache.backend() == BackendKind::metal, "Gemma2 prefill KV key cache backend mismatch");
        expect(write.value_cache.backend() == BackendKind::metal, "Gemma2 prefill KV value cache backend mismatch");
        expect(
            write.key_cache.descriptor() == shell.plan.kv_cache()[offset].key.descriptor(),
            "Gemma2 prefill KV key cache descriptor mismatch"
        );
        expect(
            write.value_cache.descriptor() == shell.plan.kv_cache()[offset].value.descriptor(),
            "Gemma2 prefill KV value cache descriptor mismatch"
        );
        expect(
            shares_backend_storage(write.key_cache, shell.plan.kv_cache()[offset].key),
            "Gemma2 prefill KV key write should share resident plan cache"
        );
        expect(
            shares_backend_storage(write.value_cache, shell.first_decode.kv_cache()[offset].value),
            "Gemma2 prefill KV value write should share decode schedule cache"
        );
    }

    const auto first_layer_positioned_key = copy_tensor_from_backend(shell.prefill_first_layer_attention_position.key);
    const auto first_layer_value = copy_tensor_from_backend(shell.prefill_first_layer_attention_projection.value);
    const auto first_layer_key_cache = copy_tensor_from_backend(shell.plan.kv_cache()[0].key);
    const auto first_layer_value_cache = copy_tensor_from_backend(shell.plan.kv_cache()[0].value);
    const auto reusable_positioned_key =
        copy_tensor_from_backend(shell.prefill_reusable_blocks.back().attention_position.key);
    const auto reusable_value =
        copy_tensor_from_backend(shell.prefill_reusable_blocks.back().attention_projection.value);
    const auto reusable_key_cache = copy_tensor_from_backend(shell.first_decode.kv_cache()[3].key);
    const auto reusable_value_cache = copy_tensor_from_backend(shell.first_decode.kv_cache()[3].value);
    const std::array<std::uint64_t, 2> first_cell {0, 0};
    expect(
        read_f32(first_layer_positioned_key, std::span<const std::uint64_t>(first_cell)) != 0.0F,
        "Gemma2 first-layer positioned key fixture should be nonzero"
    );
    expect(
        read_f32(reusable_value, std::span<const std::uint64_t>(first_cell)) != 0.0F,
        "Gemma2 reusable value fixture should be nonzero"
    );

    expect_cache_window_matches_flat_rows(
        first_layer_key_cache,
        first_layer_positioned_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        "Gemma2 first-layer prefill key cache"
    );
    expect_cache_window_matches_flat_rows(
        first_layer_value_cache,
        first_layer_value,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        "Gemma2 first-layer prefill value cache"
    );
    expect_cache_window_matches_flat_rows(
        reusable_key_cache,
        reusable_positioned_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        "Gemma2 reusable block prefill key cache"
    );
    expect_cache_window_matches_flat_rows(
        reusable_value_cache,
        reusable_value,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        "Gemma2 reusable block prefill value cache"
    );
    expect_cache_row_zero(first_layer_key_cache, prompt_token_ids.size() + 1U, "Gemma2 first-layer key cache unwritten row");
    expect_cache_row_zero(reusable_value_cache, prompt_token_ids.size() + 1U, "Gemma2 reusable value cache unwritten row");
}

void test_gemma2_metal_runtime_shell_runs_first_decode_block_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true);
    std::vector<float> embedding_values(256U * 4U, 0.0F);
    for (std::size_t column = 0; column < 8U; ++column) {
        embedding_values[column] = 0.0625F * static_cast<float>(column + 1U);
        embedding_values[3U * 256U + column] = 0.25F * static_cast<float>(column + 1U);
        embedding_values[1U * 256U + column] = -0.125F * static_cast<float>(column + 1U);
    }
    const auto embedding = tensor_f16_gguf_matrix(256, 4, std::span<const float>(embedding_values));
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> attention_norm_values(256U, 1.0F);
    overwrite_payload_bytes(
        payload_store,
        block_tensor_name(0, "attn_norm.weight"),
        tensor_f32_vector(std::span<const float>(attention_norm_values))
    );
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_q.weight"), tensor_q4_k_gguf_weight_matrix(256, 256, 11));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_k.weight"), tensor_q4_k_gguf_weight_matrix(256, 256, 12));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_v.weight"), tensor_q6_k_gguf_weight_matrix(256, 256, 13));

    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};
    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );
    const auto& block = shell.first_decode_block;

    expect(shell.first_decode_input.token_ids.size() == 1U, "Gemma2 first decode input token count mismatch");
    expect(
        shell.first_decode_input.token_ids.front() == shell.prefill_next_token_selection.token_id,
        "Gemma2 first decode input should consume selected prefill token"
    );
    expect(
        shell.first_decode_input.hidden_states.descriptor() == shell.first_decode.scratch().normalized_hidden,
        "Gemma2 first decode hidden state shape mismatch"
    );
    expect(
        shell.first_decode_first_layer_input.normalized_attention_input.descriptor()
            == shell.first_decode.scratch().normalized_hidden,
        "Gemma2 first decode normalized input shape mismatch"
    );
    expect(block.schedule_layer_offset == 0U, "Gemma2 first decode block schedule offset mismatch");
    expect(block.layer_index == shell.first_decode.layers().front().layer_index, "Gemma2 first decode block layer mismatch");
    expect(block.phase == Gemma2MetalSchedulePhase::decode, "Gemma2 first decode block phase mismatch");
    expect(
        block.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 first decode block scratch ownership mismatch"
    );
    expect(
        block.attention_scores.attention_token_count == shell.first_decode.layers().front().attention_token_count,
        "Gemma2 first decode block attention token count mismatch"
    );
    expect(
        block.attention_scores.scores.descriptor().shape == Shape::from({
            1,
            shell.plan.config().attention_head_count,
            shell.first_decode.layers().front().attention_token_count,
        }),
        "Gemma2 first decode block score shape mismatch"
    );
    expect(
        block.feed_forward_output.feed_forward_residual.descriptor()
            == shell.first_decode.scratch().feed_forward_residual,
        "Gemma2 first decode block FFN residual shape mismatch"
    );
    expect(
        shell.first_decode_reusable_layer_inputs.empty(),
        "Gemma2 one-layer decode fixture should not prepare reusable decode layer inputs"
    );
    expect(
        shell.first_decode_reusable_blocks.empty(),
        "Gemma2 one-layer decode fixture should not run reusable decode blocks"
    );
    expect(
        shares_backend_storage(
            shell.first_decode_terminal_feed_forward_output.feed_forward_residual,
            block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 one-layer decode terminal residual should select first block"
    );
    expect(block.kv_cache_write.schedule_layer_offset == 0U, "Gemma2 first decode KV write offset mismatch");
    expect(
        block.kv_cache_write.phase == Gemma2MetalSchedulePhase::decode,
        "Gemma2 first decode KV write phase mismatch"
    );
    expect(
        block.kv_cache_write.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 first decode KV write scratch ownership mismatch"
    );
    expect(
        block.kv_cache_write.cache_position == shell.first_decode.cache_position(),
        "Gemma2 first decode KV write position mismatch"
    );
    expect(block.kv_cache_write.token_count == 1U, "Gemma2 first decode KV write token count mismatch");
    expect(
        block.kv_cache_write.total_kv_token_count == shell.first_decode.total_kv_token_count(),
        "Gemma2 first decode KV write total KV count mismatch"
    );
    expect(
        shares_backend_storage(block.kv_cache_write.key_cache, shell.plan.kv_cache()[0].key),
        "Gemma2 first decode KV key write should share resident plan cache"
    );
    expect(
        shares_backend_storage(block.kv_cache_write.value_cache, shell.first_decode.kv_cache()[0].value),
        "Gemma2 first decode KV value write should share decode schedule cache"
    );

    const auto key_cache = copy_tensor_from_backend(shell.plan.kv_cache()[0].key);
    const auto value_cache = copy_tensor_from_backend(shell.plan.kv_cache()[0].value);
    const auto decode_positioned_key = copy_tensor_from_backend(block.attention_position.key);
    const auto decode_value = copy_tensor_from_backend(block.attention_projection.value);
    expect_cache_window_matches_flat_rows(
        key_cache,
        decode_positioned_key,
        shell.first_decode.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        "Gemma2 first decode key cache append"
    );
    expect_cache_window_matches_flat_rows(
        value_cache,
        decode_value,
        shell.first_decode.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        "Gemma2 first decode value cache append"
    );
    expect_cache_row_zero(
        key_cache,
        shell.first_decode.cache_position() + 1U,
        "Gemma2 first decode key cache next row should remain unwritten"
    );

    const auto flat_key_cache = flatten_f32_cache_rows(
        key_cache,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length
    );
    const auto flat_value_cache = flatten_f32_cache_rows(
        value_cache,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length
    );
    const auto positioned_query = copy_tensor_from_backend(block.attention_position.query);
    const auto expected_scores = causal_grouped_attention_scores_f32(
        positioned_query,
        flat_key_cache,
        shell.first_decode.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.first_decode.layers().front().attention_token_count,
        block.attention_scores.scale,
        shell.plan.config().attention_logit_softcap
    );
    const auto expected_attention_values = causal_grouped_attention_values_f32(
        softmax_f32(expected_scores),
        flat_value_cache,
        shell.first_decode.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        shell.first_decode.layers().front().attention_token_count
    );
    expect_f32_rank3_close(
        copy_tensor_from_backend(block.attention_scores.scores),
        expected_scores,
        1.0e-2F,
        "Gemma2 first decode cache attention scores"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(block.attention_values.attention_output),
        expected_attention_values,
        1.0e-2F,
        "Gemma2 first decode cache attention values"
    );
}

void test_gemma2_metal_runtime_shell_loops_over_reusable_decode_blocks_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 4);
    std::vector<float> embedding_values(256U * 4U, 0.0F);
    for (std::size_t token = 0; token < 4U; ++token) {
        for (std::size_t column = 0; column < 8U; ++column) {
            embedding_values[token * 256U + column] =
                0.03125F * static_cast<float>((token + 1U) * (column + 1U));
        }
    }
    const auto embedding = tensor_f16_gguf_matrix(256, 4, std::span<const float>(embedding_values));
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);
    std::vector<float> output_norm_values(256U, 1.0F);
    output_norm_values[0] = 0.625F;
    output_norm_values[1] = 1.375F;
    const auto output_norm = tensor_f32_vector(std::span<const float>(output_norm_values));
    overwrite_payload_bytes(payload_store, "output_norm.weight", output_norm);

    auto attention_norm_for_block = [](float first, float second) {
        std::vector<float> values(256U, 1.0F);
        values[0] = first;
        values[1] = second;
        return tensor_f32_vector(std::span<const float>(values));
    };
    const auto second_attention_norm = attention_norm_for_block(1.5F, 0.25F);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_norm.weight"), attention_norm_for_block(1.0F, 0.75F));
    overwrite_payload_bytes(payload_store, block_tensor_name(1, "attn_norm.weight"), second_attention_norm);
    overwrite_payload_bytes(payload_store, block_tensor_name(2, "attn_norm.weight"), attention_norm_for_block(0.625F, 1.25F));
    overwrite_payload_bytes(payload_store, block_tensor_name(3, "attn_norm.weight"), attention_norm_for_block(1.875F, 0.5F));

    auto overwrite_block_weights = [&](std::uint32_t block_index, std::uint64_t seed) {
        std::vector<float> post_attention_norm_values(256U, 1.0F);
        post_attention_norm_values[0] = 0.5F + 0.125F * static_cast<float>(block_index);
        post_attention_norm_values[1] = 1.25F - 0.125F * static_cast<float>(block_index);
        overwrite_payload_bytes(
            payload_store,
            block_tensor_name(block_index, "post_attention_norm.weight"),
            tensor_f32_vector(std::span<const float>(post_attention_norm_values))
        );

        std::vector<float> feed_forward_norm_values(256U, 1.0F);
        feed_forward_norm_values[0] = 0.75F + 0.125F * static_cast<float>(block_index);
        feed_forward_norm_values[1] = 1.5F - 0.125F * static_cast<float>(block_index);
        overwrite_payload_bytes(
            payload_store,
            block_tensor_name(block_index, "ffn_norm.weight"),
            tensor_f32_vector(std::span<const float>(feed_forward_norm_values))
        );

        std::vector<float> post_feed_forward_norm_values(256U, 1.0F);
        post_feed_forward_norm_values[0] = 1.25F - 0.125F * static_cast<float>(block_index);
        post_feed_forward_norm_values[1] = 0.625F + 0.125F * static_cast<float>(block_index);
        overwrite_payload_bytes(
            payload_store,
            block_tensor_name(block_index, "post_ffw_norm.weight"),
            tensor_f32_vector(std::span<const float>(post_feed_forward_norm_values))
        );

        overwrite_payload_bytes(
            payload_store,
            block_tensor_name(block_index, "attn_q.weight"),
            tensor_q4_k_gguf_weight_matrix(256, 256, seed)
        );
        overwrite_payload_bytes(
            payload_store,
            block_tensor_name(block_index, "attn_k.weight"),
            tensor_q4_k_gguf_weight_matrix(256, 256, seed + 1U)
        );
        overwrite_payload_bytes(
            payload_store,
            block_tensor_name(block_index, "attn_v.weight"),
            tensor_q6_k_gguf_weight_matrix(256, 256, seed + 2U)
        );
        // This fixture targets decode-loop handoff, resident K/V attention, and token selection.
        // Leaving output/FFN projections at their zero fixture values prevents random
        // quantized FFN chains from overflowing before the argmax boundary.
    };
    for (std::uint32_t block_index = 0; block_index < 4U; ++block_index) {
        overwrite_block_weights(block_index, 31U + static_cast<std::uint64_t>(block_index) * 9U);
    }

    auto view = tiny_model_view(tiny_gemma2_metadata(4), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};
    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );

    expect(shell.first_decode.layers().size() == 4U, "Gemma2 decode loop fixture layer count mismatch");
    expect(
        shell.first_decode_reusable_layer_inputs.size() == 3U,
        "Gemma2 decode loop should prepare one reusable input per later decode layer"
    );
    expect(
        shell.first_decode_reusable_blocks.size() == 3U,
        "Gemma2 decode loop should run one reusable block per later decode layer"
    );

    const auto previous_decode_residual = copy_tensor_from_backend(
        shell.first_decode_block.feed_forward_output.feed_forward_residual
    );
    const auto expected_second_decode_input = rms_norm_f32(
        previous_decode_residual,
        second_attention_norm,
        shell.plan.config().rms_norm_epsilon
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(shell.first_decode_reusable_layer_inputs.front().normalized_attention_input),
        expected_second_decode_input,
        1.0e-5F,
        "Gemma2 decode loop second-layer normalized input"
    );

    for (std::size_t reusable_index = 0; reusable_index < shell.first_decode_reusable_blocks.size(); ++reusable_index) {
        const auto schedule_layer_offset = static_cast<std::uint32_t>(reusable_index + 1U);
        const auto& layer_input = shell.first_decode_reusable_layer_inputs[reusable_index];
        const auto& block = shell.first_decode_reusable_blocks[reusable_index];
        const auto& scheduled_layer = shell.first_decode.layers()[schedule_layer_offset];

        expect(
            layer_input.layer_index == scheduled_layer.layer_index,
            "Gemma2 decode loop reusable input layer mismatch"
        );
        expect(layer_input.phase == Gemma2MetalSchedulePhase::decode, "Gemma2 decode loop input phase mismatch");
        expect(
            layer_input.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
            "Gemma2 decode loop input scratch ownership mismatch"
        );
        expect(
            layer_input.normalized_attention_input.descriptor()
                == shell.first_decode.scratch().layer_handoff_normalized_attention_input,
            "Gemma2 decode loop input descriptor mismatch"
        );
        expect(
            block.schedule_layer_offset == schedule_layer_offset,
            "Gemma2 decode loop block offset mismatch"
        );
        expect(block.layer_index == scheduled_layer.layer_index, "Gemma2 decode loop block layer mismatch");
        expect(block.phase == Gemma2MetalSchedulePhase::decode, "Gemma2 decode loop block phase mismatch");
        expect(
            block.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
            "Gemma2 decode loop block scratch ownership mismatch"
        );
        expect(
            block.feed_forward_output.feed_forward_residual.descriptor()
                == shell.first_decode.scratch().feed_forward_residual,
            "Gemma2 decode loop FFN residual descriptor mismatch"
        );
        expect(
            block.kv_cache_write.schedule_layer_offset == schedule_layer_offset,
            "Gemma2 decode loop KV write offset mismatch"
        );
        expect(
            block.kv_cache_write.cache_position == shell.first_decode.cache_position(),
            "Gemma2 decode loop KV write position mismatch"
        );
        expect(block.kv_cache_write.token_count == 1U, "Gemma2 decode loop KV write token count mismatch");
        expect(
            block.kv_cache_write.total_kv_token_count == shell.first_decode.total_kv_token_count(),
            "Gemma2 decode loop KV write total count mismatch"
        );
        expect(
            shares_backend_storage(block.kv_cache_write.key_cache, shell.plan.kv_cache()[block.layer_index].key),
            "Gemma2 decode loop KV key write should share resident plan cache"
        );
        expect(
            shares_backend_storage(block.kv_cache_write.value_cache, shell.first_decode.kv_cache()[block.layer_index].value),
            "Gemma2 decode loop KV value write should share decode schedule cache"
        );

        const auto key_cache = copy_tensor_from_backend(shell.plan.kv_cache()[block.layer_index].key);
        const auto value_cache = copy_tensor_from_backend(shell.plan.kv_cache()[block.layer_index].value);
        const auto positioned_key = copy_tensor_from_backend(block.attention_position.key);
        const auto decode_value = copy_tensor_from_backend(block.attention_projection.value);
        const auto key_append_message =
            std::string("Gemma2 decode loop key cache append offset ") + std::to_string(schedule_layer_offset);
        const auto value_append_message =
            std::string("Gemma2 decode loop value cache append offset ") + std::to_string(schedule_layer_offset);
        expect_cache_window_matches_flat_rows(
            key_cache,
            positioned_key,
            shell.first_decode.cache_position(),
            shell.plan.config().attention_head_count_kv,
            shell.plan.config().attention_key_length,
            key_append_message
        );
        expect_cache_window_matches_flat_rows(
            value_cache,
            decode_value,
            shell.first_decode.cache_position(),
            shell.plan.config().attention_head_count_kv,
            shell.plan.config().attention_value_length,
            value_append_message
        );
        expect_cache_row_zero(
            key_cache,
            shell.first_decode.cache_position() + 1U,
            "Gemma2 decode loop key cache next row should remain unwritten"
        );
    }

    const auto& parity_block = shell.first_decode_reusable_blocks.front();
    const auto parity_key_cache = copy_tensor_from_backend(shell.plan.kv_cache()[parity_block.layer_index].key);
    const auto parity_value_cache = copy_tensor_from_backend(shell.plan.kv_cache()[parity_block.layer_index].value);
    const auto flat_key_cache = flatten_f32_cache_rows(
        parity_key_cache,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length
    );
    const auto flat_value_cache = flatten_f32_cache_rows(
        parity_value_cache,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length
    );
    const auto positioned_query = copy_tensor_from_backend(parity_block.attention_position.query);
    const auto expected_scores = causal_grouped_attention_scores_f32(
        positioned_query,
        flat_key_cache,
        shell.first_decode.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        parity_block.attention_scores.attention_token_count,
        parity_block.attention_scores.scale,
        shell.plan.config().attention_logit_softcap
    );
    const auto expected_attention_values = causal_grouped_attention_values_f32(
        softmax_f32(expected_scores),
        flat_value_cache,
        shell.first_decode.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        parity_block.attention_values.attention_token_count
    );
    expect_f32_rank3_close(
        copy_tensor_from_backend(parity_block.attention_scores.scores),
        expected_scores,
        1.0e-2F,
        "Gemma2 decode loop resident-cache attention scores"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(parity_block.attention_values.attention_output),
        expected_attention_values,
        1.0e-2F,
        "Gemma2 decode loop resident-cache attention values"
    );

    const auto& terminal_block = shell.first_decode_reusable_blocks.back();
    expect(
        shell.first_decode_terminal_feed_forward_output.layer_index == terminal_block.layer_index,
        "Gemma2 decode loop terminal FFN residual layer mismatch"
    );
    expect(
        shell.first_decode_terminal_feed_forward_output.phase == Gemma2MetalSchedulePhase::decode,
        "Gemma2 decode loop terminal FFN residual phase mismatch"
    );
    expect(
        shell.first_decode_terminal_feed_forward_output.scratch_ownership
            == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 decode loop terminal FFN residual scratch ownership mismatch"
    );
    expect(
        shares_backend_storage(
            shell.first_decode_terminal_feed_forward_output.feed_forward_residual,
            terminal_block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 decode loop terminal FFN residual should select final reusable block"
    );

    const auto terminal_decode_residual = copy_tensor_from_backend(
        shell.first_decode_terminal_feed_forward_output.feed_forward_residual
    );
    const auto expected_decode_output_norm = rms_norm_f32(
        terminal_decode_residual,
        output_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto expected_decode_raw_logits = matmul_f32_f16_gguf_weight(
        expected_decode_output_norm,
        embedding,
        BackendKind::cpu
    );
    const auto expected_decode_logits = softcap_f32(
        expected_decode_raw_logits,
        shell.plan.config().final_logit_softcap
    );

    expect(
        shell.first_decode_output_norm.schedule_layer_offset == shell.first_decode.layers().size() - 1U,
        "Gemma2 decode output norm terminal schedule offset mismatch"
    );
    expect(shell.first_decode_output_norm.layer_index == terminal_block.layer_index, "Gemma2 decode output norm layer mismatch");
    expect(
        shell.first_decode_output_norm.phase == Gemma2MetalSchedulePhase::decode,
        "Gemma2 decode output norm phase mismatch"
    );
    expect(
        shell.first_decode_output_norm.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 decode output norm scratch ownership mismatch"
    );
    expect(
        shell.first_decode_output_norm.hidden_width == shell.plan.config().embedding_length,
        "Gemma2 decode output norm hidden width mismatch"
    );
    expect(
        shell.first_decode_output_norm.normalized_hidden_states.backend() == BackendKind::metal,
        "Gemma2 decode output norm backend mismatch"
    );
    expect(
        shell.first_decode_output_norm.normalized_hidden_states.descriptor()
            == shell.first_decode.scratch().normalized_hidden,
        "Gemma2 decode output norm descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            shell.first_decode_output_norm.normalized_hidden_states,
            shell.first_decode_terminal_feed_forward_output.feed_forward_residual
        ),
        "Gemma2 decode output norm should not alias terminal residual"
    );
    expect(
        !shares_backend_storage(
            shell.first_decode_output_norm.normalized_hidden_states,
            shell.plan.weights().output_norm_weight
        ),
        "Gemma2 decode output norm should not alias output norm weight"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(shell.first_decode_output_norm.normalized_hidden_states),
        expected_decode_output_norm,
        1.0e-5F,
        "Gemma2 decode output norm normalized hidden states"
    );

    expect(
        shell.first_decode_logits.schedule_layer_offset == shell.first_decode.layers().size() - 1U,
        "Gemma2 decode logits terminal schedule offset mismatch"
    );
    expect(shell.first_decode_logits.layer_index == terminal_block.layer_index, "Gemma2 decode logits layer mismatch");
    expect(shell.first_decode_logits.phase == Gemma2MetalSchedulePhase::decode, "Gemma2 decode logits phase mismatch");
    expect(
        shell.first_decode_logits.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 decode logits scratch ownership mismatch"
    );
    expect(
        shell.first_decode_logits.hidden_width == shell.plan.config().embedding_length,
        "Gemma2 decode logits hidden width mismatch"
    );
    expect(
        shell.first_decode_logits.vocab_size == shell.plan.config().vocab_size,
        "Gemma2 decode logits vocab size mismatch"
    );
    expect(
        shell.first_decode_logits.final_logit_softcap == shell.plan.config().final_logit_softcap,
        "Gemma2 decode logits final softcap mismatch"
    );
    expect(shell.first_decode_logits.raw_logits.backend() == BackendKind::metal, "Gemma2 decode raw logits backend mismatch");
    expect(shell.first_decode_logits.logits.backend() == BackendKind::metal, "Gemma2 decode logits backend mismatch");
    expect(
        shell.first_decode_logits.raw_logits.descriptor() == shell.first_decode.scratch().logits,
        "Gemma2 decode raw logits descriptor mismatch"
    );
    expect(
        shell.first_decode_logits.logits.descriptor() == shell.first_decode.scratch().logits,
        "Gemma2 decode logits descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            shell.first_decode_logits.raw_logits,
            shell.first_decode_output_norm.normalized_hidden_states
        ),
        "Gemma2 decode raw logits should not alias output norm"
    );
    expect(
        !shares_backend_storage(shell.first_decode_logits.raw_logits, shell.plan.weights().output_weight),
        "Gemma2 decode raw logits should not alias output weight"
    );
    expect(
        !shares_backend_storage(shell.first_decode_logits.logits, shell.first_decode_logits.raw_logits),
        "Gemma2 decode softcapped logits should not alias raw logits"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(shell.first_decode_logits.raw_logits),
        expected_decode_raw_logits,
        1.0e-2F,
        "Gemma2 decode raw logits"
    );
    expect_f32_matrix_close(
        copy_tensor_from_backend(shell.first_decode_logits.logits),
        expected_decode_logits,
        1.0e-2F,
        "Gemma2 decode softcapped logits"
    );
}

void test_gemma2_metal_runtime_shell_selects_first_decode_next_token_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 1);
    std::vector<float> embedding_values(256U * 4U, 0.0F);
    for (std::size_t token = 0; token < 4U; ++token) {
        for (std::size_t column = 0; column < 6U; ++column) {
            embedding_values[token * 256U + column] =
                0.0625F * static_cast<float>((token + 1U) * (column + 2U));
        }
    }
    const auto embedding = tensor_f16_gguf_matrix(256, 4, std::span<const float>(embedding_values));
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> output_norm_values(256U, 1.0F);
    output_norm_values[0] = 1.25F;
    output_norm_values[1] = 0.75F;
    const auto output_norm = tensor_f32_vector(std::span<const float>(output_norm_values));
    overwrite_payload_bytes(payload_store, "output_norm.weight", output_norm);

    auto view = tiny_model_view(tiny_gemma2_metadata(1), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {1, 2};
    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );
    expect(shell.first_decode.layers().size() == 1U, "Gemma2 decode selection fixture layer count mismatch");

    const auto terminal_decode_residual = copy_tensor_from_backend(
        shell.first_decode_terminal_feed_forward_output.feed_forward_residual
    );
    const auto expected_decode_output_norm = rms_norm_f32(
        terminal_decode_residual,
        output_norm,
        shell.plan.config().rms_norm_epsilon
    );
    const auto expected_decode_raw_logits = matmul_f32_f16_gguf_weight(
        expected_decode_output_norm,
        embedding,
        BackendKind::cpu
    );
    const auto expected_decode_logits = softcap_f32(
        expected_decode_raw_logits,
        shell.plan.config().final_logit_softcap
    );
    const auto expected_decode_next_token = top_logit_tokens_f32(expected_decode_logits, 1).front();

    expect(
        shell.first_decode_next_token_selection.schedule_layer_offset == shell.first_decode.layers().size() - 1U,
        "Gemma2 decode next-token selection terminal schedule offset mismatch"
    );
    expect(
        shell.first_decode_next_token_selection.layer_index == shell.first_decode.layers().back().layer_index,
        "Gemma2 decode next-token selection terminal layer mismatch"
    );
    expect(
        shell.first_decode_next_token_selection.phase == Gemma2MetalSchedulePhase::decode,
        "Gemma2 decode next-token selection phase mismatch"
    );
    expect(
        shell.first_decode_next_token_selection.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 decode next-token selection scratch ownership mismatch"
    );
    expect(
        shell.first_decode_next_token_selection.logits_row_index == 0U,
        "Gemma2 decode next-token selection logits row mismatch"
    );
    expect(
        shell.first_decode_next_token_selection.vocab_size == shell.plan.config().vocab_size,
        "Gemma2 decode next-token selection vocab size mismatch"
    );
    expect(
        shell.first_decode_next_token_selection.token_id < shell.plan.config().vocab_size,
        "Gemma2 decode next-token selection token should be within vocab"
    );
    expect(
        shell.first_decode_next_token_selection.token_id == expected_decode_next_token.token,
        "Gemma2 decode next-token selection token mismatch"
    );
    expect_close(
        shell.first_decode_next_token_selection.logit,
        expected_decode_next_token.logit,
        1.0e-2F,
        "Gemma2 decode next-token selection logit mismatch"
    );
}

void test_gemma2_metal_generated_token_loop_repeats_decode_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 1);
    auto view = tiny_model_view(tiny_gemma2_metadata(1), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {1, 2};

    const auto loop = create_gemma2_metal_generated_token_loop(
        model,
        std::span<const std::uint32_t>(prompt_token_ids),
        3
    );

    expect(loop.generated_token_ids.size() == 3U, "Gemma2 generated loop token count mismatch");
    expect(loop.decode_steps.size() == 2U, "Gemma2 generated loop decode step count mismatch");
    expect(
        loop.generated_token_ids[0] == loop.runtime_shell.prefill_next_token_selection.token_id,
        "Gemma2 generated loop first token should come from prefill selection"
    );
    expect(
        loop.generated_token_ids[1] == loop.runtime_shell.first_decode_next_token_selection.token_id,
        "Gemma2 generated loop second token should come from first decode selection"
    );

    const auto& first_step = loop.decode_steps[0];
    expect(first_step.decode.cache_position() == prompt_token_ids.size(), "Gemma2 generated loop first decode position mismatch");
    expect(first_step.input.token_ids.size() == 1U, "Gemma2 generated loop first decode token count mismatch");
    expect(
        first_step.input.token_ids.front() == loop.generated_token_ids[0],
        "Gemma2 generated loop first decode input token mismatch"
    );
    expect(
        first_step.next_token_selection.token_id == loop.generated_token_ids[1],
        "Gemma2 generated loop first decode selected token mismatch"
    );
    expect(
        shares_backend_storage(
            first_step.first_block.kv_cache_write.key_cache,
            loop.runtime_shell.plan.kv_cache()[0].key
        ),
        "Gemma2 generated loop first decode key write should share resident cache"
    );
    expect(
        shares_backend_storage(
            first_step.first_block.kv_cache_write.value_cache,
            loop.runtime_shell.plan.kv_cache()[0].value
        ),
        "Gemma2 generated loop first decode value write should share resident cache"
    );

    const auto& second_step = loop.decode_steps[1];
    expect(
        second_step.decode.cache_position() == prompt_token_ids.size() + 1U,
        "Gemma2 generated loop second decode position mismatch"
    );
    expect(second_step.decode.token_count() == 1U, "Gemma2 generated loop second decode token count mismatch");
    expect(
        second_step.decode.total_kv_token_count() == prompt_token_ids.size() + 2U,
        "Gemma2 generated loop second decode total KV count mismatch"
    );
    expect(second_step.input.token_ids.size() == 1U, "Gemma2 generated loop second decode input size mismatch");
    expect(
        second_step.input.token_ids.front() == loop.generated_token_ids[1],
        "Gemma2 generated loop second decode input token mismatch"
    );
    expect(
        second_step.first_layer_input.phase == Gemma2MetalSchedulePhase::decode,
        "Gemma2 generated loop second decode layer input phase mismatch"
    );
    expect(
        second_step.first_layer_input.scratch_ownership == Gemma2MetalScratchOwnership::resident_decode,
        "Gemma2 generated loop second decode layer input scratch ownership mismatch"
    );
    expect(
        second_step.first_layer_input.normalized_attention_input.descriptor()
            == second_step.decode.scratch().normalized_hidden,
        "Gemma2 generated loop second decode layer input descriptor mismatch"
    );
    expect(second_step.first_block.layer_index == 0U, "Gemma2 generated loop second decode block layer mismatch");
    expect(
        second_step.first_block.kv_cache_write.cache_position == second_step.decode.cache_position(),
        "Gemma2 generated loop second decode KV write position mismatch"
    );
    expect(
        second_step.first_block.kv_cache_write.total_kv_token_count == second_step.decode.total_kv_token_count(),
        "Gemma2 generated loop second decode KV write total count mismatch"
    );
    expect(
        shares_backend_storage(
            second_step.first_block.kv_cache_write.key_cache,
            loop.runtime_shell.plan.kv_cache()[0].key
        ),
        "Gemma2 generated loop second decode key write should share resident cache"
    );
    expect(
        shares_backend_storage(
            second_step.first_block.kv_cache_write.value_cache,
            loop.runtime_shell.plan.kv_cache()[0].value
        ),
        "Gemma2 generated loop second decode value write should share resident cache"
    );
    expect(
        shares_backend_storage(
            second_step.terminal_feed_forward_output.feed_forward_residual,
            second_step.first_block.feed_forward_output.feed_forward_residual
        ),
        "Gemma2 generated loop single-layer terminal residual should use first block"
    );

    const auto key_cache = copy_tensor_from_backend(loop.runtime_shell.plan.kv_cache()[0].key);
    const auto value_cache = copy_tensor_from_backend(loop.runtime_shell.plan.kv_cache()[0].value);
    expect_cache_window_matches_flat_rows(
        key_cache,
        copy_tensor_from_backend(second_step.first_block.attention_position.key),
        second_step.decode.cache_position(),
        loop.runtime_shell.plan.config().attention_head_count_kv,
        loop.runtime_shell.plan.config().attention_key_length,
        "Gemma2 generated loop second decode key cache append"
    );
    expect_cache_window_matches_flat_rows(
        value_cache,
        copy_tensor_from_backend(second_step.first_block.attention_projection.value),
        second_step.decode.cache_position(),
        loop.runtime_shell.plan.config().attention_head_count_kv,
        loop.runtime_shell.plan.config().attention_value_length,
        "Gemma2 generated loop second decode value cache append"
    );
    expect_cache_row_zero(
        key_cache,
        second_step.decode.cache_position() + 1U,
        "Gemma2 generated loop next key cache row should remain unwritten"
    );

    const auto expected_next_token = top_logit_tokens_f32(
        copy_tensor_from_backend(second_step.logits.logits),
        1
    ).front();
    expect(
        second_step.next_token_selection.token_id == expected_next_token.token,
        "Gemma2 generated loop second decode next token mismatch"
    );
    expect(
        second_step.next_token_selection.token_id == loop.generated_token_ids[2],
        "Gemma2 generated loop third token should come from second decode selection"
    );
    expect_close(
        second_step.next_token_selection.logit,
        expected_next_token.logit,
        1.0e-6F,
        "Gemma2 generated loop second decode selected logit mismatch"
    );

    const auto output_token_ids = generate_gemma2_metal_argmax_tokens(
        model,
        std::span<const std::uint32_t>(prompt_token_ids),
        3
    );
    expect(
        output_token_ids.size() == prompt_token_ids.size() + loop.generated_token_ids.size(),
        "Gemma2 generated loop output token count mismatch"
    );
    expect(output_token_ids[0] == prompt_token_ids[0], "Gemma2 generated loop output prompt token 0 mismatch");
    expect(output_token_ids[1] == prompt_token_ids[1], "Gemma2 generated loop output prompt token 1 mismatch");
    for (std::size_t index = 0; index < loop.generated_token_ids.size(); ++index) {
        expect(
            output_token_ids[prompt_token_ids.size() + index] == loop.generated_token_ids[index],
            "Gemma2 generated loop output generated token mismatch"
        );
    }
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
        "Gemma2 Metal runtime shell embeds prefill prompt when available",
        test_gemma2_metal_runtime_shell_embeds_prefill_prompt_when_available,
    },
    {
        "Gemma2 Metal runtime shell runs first prefill block when available",
        test_gemma2_metal_runtime_shell_runs_first_prefill_block_when_available,
    },
    {
        "Gemma2 Metal prefill block runner executes scheduled layer when available",
        test_gemma2_metal_prefill_block_runner_executes_scheduled_layer_when_available,
    },
    {
        "Gemma2 Metal runtime shell loops over later reusable blocks when available",
        test_gemma2_metal_runtime_shell_loops_over_later_reusable_blocks_when_available,
    },
    {
        "Gemma2 Metal runtime shell writes prefill KV cache when available",
        test_gemma2_metal_runtime_shell_writes_prefill_kv_cache_when_available,
    },
    {
        "Gemma2 Metal runtime shell runs first decode block when available",
        test_gemma2_metal_runtime_shell_runs_first_decode_block_when_available,
    },
    {
        "Gemma2 Metal runtime shell loops over reusable decode blocks when available",
        test_gemma2_metal_runtime_shell_loops_over_reusable_decode_blocks_when_available,
    },
    {
        "Gemma2 Metal runtime shell selects first decode next token when available",
        test_gemma2_metal_runtime_shell_selects_first_decode_next_token_when_available,
    },
    {
        "Gemma2 Metal generated-token loop repeats decode when available",
        test_gemma2_metal_generated_token_loop_repeats_decode_when_available,
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
