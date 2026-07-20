#include "gemma2_metal_plan_tests.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_model.hpp"
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
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "tensor_test_support.hpp"
#include "../../src/tensor/backends/metal/metal_command_context.hpp"

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
using local_gguf_engine::engine::Gemma2ModelWeights;
using local_gguf_engine::engine::Gemma2ModelConfig;
using local_gguf_engine::engine::create_gemma2_metal_decode_schedule;
using local_gguf_engine::engine::create_gemma2_metal_model_plan;
using local_gguf_engine::engine::create_gemma2_metal_prefill_schedule;
using local_gguf_engine::engine::bind_gemma2_model_weights;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens_with_plan;
using local_gguf_engine::engine::load_engine_model_from_gguf;
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
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tensor::backend_buffer_storage;
using local_gguf_engine::tensor::allocate_backend_buffer;
using local_gguf_engine::tensor::copy_tensor_to_backend;
using local_gguf_engine::tensor::copy_tensor_view_to_backend;
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
using local_gguf_engine::tensor::ops::embed_f16_gguf_weight_token_ids_to_backend;
using local_gguf_engine::tensor::ops::softcap_f32;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::ops::write_kv_cache_f32_to_backend;
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::backends::metal::MetalCommandContext;
using local_gguf_engine::tensor::backends::metal::gemma2_decode_attention_partition_count;
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
    std::uint32_t attention_head_count = 2,
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
        add_payload(store, block_tensor_name(block_index, "attn_q.weight"), {256, 512}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "attn_k.weight"), {256, 256}, DType::q4_k);
        add_payload(store, block_tensor_name(block_index, "attn_v.weight"), {256, 256}, DType::q6_k);
        add_payload(store, block_tensor_name(block_index, "attn_output.weight"), {512, 256}, DType::q4_k);
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

void add_payload_storage(
    TinyGemma2PayloadStore& store,
    std::string name,
    std::vector<std::uint64_t> dimensions,
    const local_gguf_engine::tensor::TensorStorage& storage
) {
    store.bytes.emplace_back(storage.bytes().begin(), storage.bytes().end());
    store.payloads.push_back(tensor_payload_view(
        std::move(name),
        std::move(dimensions),
        storage.descriptor().dtype,
        std::span<const std::uint8_t>(store.bytes.back())
    ));
}

TinyGemma2PayloadStore nonzero_tiny_gemma2_payload_store(std::uint32_t block_count = 2) {
    TinyGemma2PayloadStore store;
    store.bytes.reserve(static_cast<std::size_t>(2 + block_count * 11));
    store.payloads.reserve(static_cast<std::size_t>(2 + block_count * 11));

    std::vector<float> embedding_values(256U * 4U);
    for (std::size_t index = 0; index < embedding_values.size(); ++index) {
        embedding_values[index] = static_cast<float>(static_cast<int>(index % 19U) - 9) * 0.0125F;
    }
    add_payload_storage(
        store,
        "token_embd.weight",
        {256, 4},
        tensor_f16_gguf_matrix(256, 4, std::span<const float>(embedding_values))
    );

    std::vector<float> norm_values(256U);
    for (std::size_t index = 0; index < norm_values.size(); ++index) {
        norm_values[index] = 0.9F + static_cast<float>(index % 11U) * 0.02F;
    }
    add_payload_storage(store, "output_norm.weight", {256}, tensor_f32_vector(norm_values));

    for (std::uint32_t block = 0; block < block_count; ++block) {
        const auto seed = static_cast<std::uint64_t>(31U + block * 17U);
        add_payload_storage(store, block_tensor_name(block, "attn_norm.weight"), {256}, tensor_f32_vector(norm_values));
        add_payload_storage(store, block_tensor_name(block, "post_attention_norm.weight"), {256}, tensor_f32_vector(norm_values));
        add_payload_storage(store, block_tensor_name(block, "attn_q.weight"), {256, 512}, tensor_q4_k_gguf_weight_matrix(256, 512, seed));
        add_payload_storage(store, block_tensor_name(block, "attn_k.weight"), {256, 256}, tensor_q4_k_gguf_weight_matrix(256, 256, seed + 1U));
        add_payload_storage(store, block_tensor_name(block, "attn_v.weight"), {256, 256}, tensor_q6_k_gguf_weight_matrix(256, 256, seed + 2U));
        add_payload_storage(store, block_tensor_name(block, "attn_output.weight"), {512, 256}, tensor_q4_k_gguf_weight_matrix(512, 256, seed + 3U));
        add_payload_storage(store, block_tensor_name(block, "ffn_norm.weight"), {256}, tensor_f32_vector(norm_values));
        add_payload_storage(store, block_tensor_name(block, "post_ffw_norm.weight"), {256}, tensor_f32_vector(norm_values));
        add_payload_storage(store, block_tensor_name(block, "ffn_gate.weight"), {256, 256}, tensor_q4_k_gguf_weight_matrix(256, 256, seed + 4U));
        add_payload_storage(store, block_tensor_name(block, "ffn_up.weight"), {256, 256}, tensor_q4_k_gguf_weight_matrix(256, 256, seed + 5U));
        add_payload_storage(store, block_tensor_name(block, "ffn_down.weight"), {256, 256}, tensor_q6_k_gguf_weight_matrix(256, 256, seed + 6U));
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

void expect_f32_storage_close(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
    for (std::uint64_t index = 0; index < actual.descriptor().element_count; ++index) {
        float actual_value = 0.0F;
        float expected_value = 0.0F;
        const auto offset = static_cast<std::size_t>(index * sizeof(float));
        std::memcpy(&actual_value, actual.bytes().data() + offset, sizeof(float));
        std::memcpy(&expected_value, expected.bytes().data() + offset, sizeof(float));
        expect_close(actual_value, expected_value, tolerance, message);
    }
}

TensorStorage copy_tensor_view(const local_gguf_engine::tensor::TensorView& view) {
    return TensorStorage::from_bytes(view.descriptor(), view.bytes());
}

TensorStorage reference_weight_matmul(
    const TensorStorage& input,
    const local_gguf_engine::tensor::TensorView& weight
) {
    const auto storage = copy_tensor_view(weight);
    if (weight.descriptor().dtype == DType::f16) {
        return matmul_f32_f16_gguf_weight(input, storage);
    }
    return matmul_f32_quantized_gguf_weight(input, storage);
}

void write_cache_row(TensorStorage& cache, const TensorStorage& row, std::uint32_t position) {
    const auto row_bytes = row.descriptor().byte_count;
    const auto offset = static_cast<std::size_t>(position) * row_bytes;
    expect(offset + row_bytes <= cache.mutable_bytes().size(), "reference KV row range mismatch");
    std::memcpy(cache.mutable_bytes().data() + offset, row.bytes().data(), row_bytes);
}

struct ReferenceDecodeResult {
    TensorStorage logits;
    std::vector<TensorStorage> key_cache;
    std::vector<TensorStorage> value_cache;
};

ReferenceDecodeResult reference_decode(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights,
    std::uint32_t token_id,
    std::uint32_t cache_position,
    std::vector<TensorStorage> key_cache,
    std::vector<TensorStorage> value_cache
) {
    const auto embedding_weight = copy_tensor_view_to_backend(
        weights.token_embedding_weight,
        BackendKind::cpu
    );
    const std::array<std::uint32_t, 1> token_ids {token_id};
    auto hidden = copy_tensor_from_backend(embed_f16_gguf_weight_token_ids_to_backend(
        std::span<const std::uint32_t>(token_ids),
        embedding_weight,
        std::sqrt(static_cast<float>(config.embedding_length))
    ));
    const auto head_dimension = config.attention_key_length;
    const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));

    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto& block = weights.blocks[layer];
        const auto normalized_attention = rms_norm_f32(
            hidden,
            copy_tensor_view(block.attention.attention_norm_weight),
            config.rms_norm_epsilon
        );
        const auto raw_query = reference_weight_matmul(normalized_attention, block.attention.query_weight);
        const auto raw_key = reference_weight_matmul(normalized_attention, block.attention.key_weight);
        const auto value = reference_weight_matmul(normalized_attention, block.attention.value_weight);
        const auto query = rope_neox_f32_by_heads_with_position_start(
            raw_query,
            cache_position,
            config.attention_head_count,
            head_dimension,
            head_dimension,
            10000.0F
        );
        const auto key = rope_neox_f32_by_heads_with_position_start(
            raw_key,
            cache_position,
            config.attention_head_count_kv,
            head_dimension,
            head_dimension,
            10000.0F
        );
        write_cache_row(key_cache[layer], key, cache_position);
        write_cache_row(value_cache[layer], value, cache_position);

        const auto total_tokens = cache_position + 1U;
        const auto attention_tokens = layer % 2U == 0U
            ? std::min(total_tokens, config.sliding_window)
            : total_tokens;
        const auto scores = causal_grouped_attention_scores_f32(
            query,
            key_cache[layer],
            cache_position,
            config.attention_head_count,
            config.attention_head_count_kv,
            head_dimension,
            attention_tokens,
            attention_scale,
            config.attention_logit_softcap
        );
        const auto probabilities = softmax_f32(scores);
        const auto attention_output = causal_grouped_attention_values_f32(
            probabilities,
            value_cache[layer],
            cache_position,
            config.attention_head_count,
            config.attention_head_count_kv,
            config.attention_value_length,
            attention_tokens
        );
        const auto projected_attention = reference_weight_matmul(
            attention_output,
            block.attention.output_weight
        );
        const auto post_attention = rms_norm_f32(
            projected_attention,
            copy_tensor_view(block.attention.post_attention_norm_weight),
            config.rms_norm_epsilon
        );
        const auto attention_residual = add_f32(hidden, post_attention);
        const auto normalized_ffn = rms_norm_f32(
            attention_residual,
            copy_tensor_view(block.feed_forward.feed_forward_norm_weight),
            config.rms_norm_epsilon
        );
        const auto gate = reference_weight_matmul(normalized_ffn, block.feed_forward.gate_weight);
        const auto up = reference_weight_matmul(normalized_ffn, block.feed_forward.up_weight);
        const auto gated = multiply_f32(gelu_tanh_f32(gate), up);
        const auto projected_ffn = reference_weight_matmul(gated, block.feed_forward.down_weight);
        const auto post_ffn = rms_norm_f32(
            projected_ffn,
            copy_tensor_view(block.feed_forward.post_feed_forward_norm_weight),
            config.rms_norm_epsilon
        );
        hidden = add_f32(attention_residual, post_ffn);
    }

    const auto normalized_output = rms_norm_f32(
        hidden,
        copy_tensor_view(weights.output_norm_weight),
        config.rms_norm_epsilon
    );
    auto logits = reference_weight_matmul(normalized_output, weights.output_weight);
    logits = softcap_f32(logits, config.final_logit_softcap);
    return ReferenceDecodeResult {
        .logits = std::move(logits),
        .key_cache = std::move(key_cache),
        .value_cache = std::move(value_cache),
    };
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
        + backend_buffer_byte_count(decode.attention_partial_output)
        + backend_buffer_byte_count(decode.attention_partial_maxima)
        + backend_buffer_byte_count(decode.attention_partial_denominators)
        + backend_buffer_byte_count(decode.attention_output)
        + backend_buffer_byte_count(decode.projected_attention_output)
        + backend_buffer_byte_count(decode.post_attention_residual)
        + backend_buffer_byte_count(decode.normalized_feed_forward_input)
        + backend_buffer_byte_count(decode.feed_forward_gate)
        + backend_buffer_byte_count(decode.feed_forward_up)
        + backend_buffer_byte_count(decode.feed_forward_gated)
        + backend_buffer_byte_count(decode.projected_feed_forward_output)
        + backend_buffer_byte_count(decode.feed_forward_residual)
        + backend_buffer_byte_count(decode.logits)
        + backend_buffer_byte_count(decode.top_logit_staging);
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

void test_creates_gemma2_metal_schedules_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 2);
    auto view = tiny_model_view(tiny_gemma2_metadata(2), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);
    const auto prefill = create_gemma2_metal_prefill_schedule(plan, 3, 4);
    const auto decode = create_gemma2_metal_decode_schedule(plan, 9);

    expect(prefill.phase() == Gemma2MetalSchedulePhase::prefill, "prefill phase mismatch");
    expect(prefill.scratch_ownership() == Gemma2MetalScratchOwnership::resident_prefill, "prefill ownership mismatch");
    expect(prefill.token_count() == 3U && prefill.cache_position() == 4U, "prefill range mismatch");
    expect(prefill.total_kv_token_count() == 7U, "prefill KV count mismatch");
    expect(decode.phase() == Gemma2MetalSchedulePhase::decode, "decode phase mismatch");
    expect(decode.scratch_ownership() == Gemma2MetalScratchOwnership::resident_decode, "decode ownership mismatch");
    expect(decode.token_count() == 1U && decode.cache_position() == 9U, "decode range mismatch");
    expect(decode.layers().size() == 2U, "decode layer count mismatch");
    expect(decode.layers()[0].attention_token_count == 8U, "sliding attention count mismatch");
    expect(decode.layers()[1].attention_token_count == 10U, "global attention count mismatch");
}

void test_gemma2_fused_decode_attention_matches_exact_shape_cpu_reference_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint32_t query_heads = 16U;
    constexpr std::uint32_t kv_heads = 8U;
    constexpr std::uint32_t dimension = 256U;
    constexpr std::uint32_t cache_rows = 8192U;
    constexpr float scale = 1.0F / 16.0F;
    constexpr float softcap = 50.0F;

    std::vector<float> query_values(query_heads * dimension);
    for (std::uint32_t head = 0; head < query_heads; ++head) {
        for (std::uint32_t column = 0; column < dimension; ++column) {
            query_values[head * dimension + column] =
                1.25F + static_cast<float>(head) * 0.03125F
                + static_cast<float>(static_cast<int>(column % 13U) - 6) * 0.015625F;
        }
    }
    std::vector<float> key_values(static_cast<std::size_t>(cache_rows) * kv_heads * dimension);
    std::vector<float> value_values(key_values.size());
    for (std::uint32_t position = 0; position < cache_rows; ++position) {
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            for (std::uint32_t column = 0; column < dimension; ++column) {
                const auto index = (static_cast<std::size_t>(position) * kv_heads + head) * dimension + column;
                key_values[index] =
                    1.5F + static_cast<float>(head) * 0.0625F
                    + static_cast<float>(static_cast<int>((position + column) % 17U) - 8) * 0.0078125F;
                value_values[index] =
                    static_cast<float>(static_cast<int>(position % 29U) - 14) * 0.003F
                    + static_cast<float>(head) * 0.021F
                    + static_cast<float>(static_cast<int>(column % 11U) - 5) * 0.002F;
            }
        }
    }

    const auto query_cpu = tensor_f32_matrix(1, query_heads * dimension, query_values);
    const auto key_cpu = tensor_f32_matrix(cache_rows, kv_heads * dimension, key_values);
    const auto value_cpu = tensor_f32_matrix(cache_rows, kv_heads * dimension, value_values);
    const auto query = copy_tensor_to_backend(query_cpu, BackendKind::metal);
    const auto key = copy_tensor_to_backend(
        TensorStorage::from_bytes(
            TensorDescriptor::from(Shape::from({cache_rows, kv_heads, dimension}), DType::f32),
            key_cpu.bytes()
        ),
        BackendKind::metal
    );
    const auto value = copy_tensor_to_backend(
        TensorStorage::from_bytes(
            TensorDescriptor::from(Shape::from({cache_rows, kv_heads, dimension}), DType::f32),
            value_cpu.bytes()
        ),
        BackendKind::metal
    );
    const auto output = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U, query_heads * dimension}), DType::f32),
        BackendKind::metal
    );
    const auto partial_output = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({32U, query_heads, dimension}), DType::f32),
        BackendKind::metal
    );
    const auto partial_maxima = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({32U, query_heads}), DType::f32),
        BackendKind::metal
    );
    const auto partial_denominators = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({32U, query_heads}), DType::f32),
        BackendKind::metal
    );

    constexpr std::array<std::uint32_t, 8> token_counts {1U, 31U, 32U, 33U, 256U, 257U, 4096U, 8192U};
    for (const auto token_count : token_counts) {
        MetalCommandContext command("Gemma2 fused-attention parity", true);
        command.encode_gemma2_decode_attention(
            query,
            key,
            value,
            output,
            partial_output,
            partial_maxima,
            partial_denominators,
            cache_rows - 1U,
            token_count,
            scale,
            softcap
        );
        const auto stats = command.commit_and_wait();
        const auto partitions = gemma2_decode_attention_partition_count(token_count);
        expect(stats.compute_dispatch_count == (partitions == 1U ? 1U : 2U), "fused attention dispatch mismatch");
        expect(stats.memory_barrier_count == (partitions == 1U ? 0U : 1U), "fused attention barrier mismatch");
        expect(stats.transient_buffer_allocation_count == 0U, "fused attention transient allocation mismatch");

        const auto scores = causal_grouped_attention_scores_f32(
            query_cpu,
            key_cpu,
            cache_rows - 1U,
            query_heads,
            kv_heads,
            dimension,
            token_count,
            scale,
            softcap
        );
        const auto probabilities = softmax_f32(scores);
        const auto expected = causal_grouped_attention_values_f32(
            probabilities,
            value_cpu,
            cache_rows - 1U,
            query_heads,
            kv_heads,
            dimension,
            token_count
        );
        expect_f32_matrix_close(
            copy_tensor_from_backend(output),
            expected,
            4.0e-4F,
            "exact-shape fused attention CPU parity"
        );
    }

    try {
        MetalCommandContext command("Gemma2 fused-attention invalid window", true);
        command.encode_gemma2_decode_attention(
            query,
            key,
            value,
            output,
            partial_output,
            partial_maxima,
            partial_denominators,
            0U,
            0U,
            scale,
            softcap
        );
        throw TestFailure("fused attention should reject an empty window");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Metal Gemma 2 decode attention token count must not be zero",
            "fused attention empty-window error mismatch"
        );
    }

    try {
        MetalCommandContext command("Gemma2 fused-attention invalid cache position", true);
        command.encode_gemma2_decode_attention(
            query, key, value, output, partial_output, partial_maxima, partial_denominators,
            cache_rows, 1U, scale, softcap
        );
        throw TestFailure("fused attention should reject a cache position outside capacity");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what())
                == "Metal Gemma 2 decode attention cache position is outside capacity",
            "fused attention cache-position error mismatch"
        );
    }

    const auto undersized_partials = allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({1U}), DType::f32),
        BackendKind::metal
    );
    try {
        MetalCommandContext command("Gemma2 fused-attention undersized partials", true);
        command.encode_gemma2_decode_attention(
            query, key, value, output, undersized_partials, partial_maxima, partial_denominators,
            0U, 1U, scale, softcap
        );
        throw TestFailure("fused attention should reject undersized partial output");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what())
                == "Metal Gemma 2 decode attention partial-output buffer is too small",
            "fused attention partial-output error mismatch"
        );
    }
}

void test_gemma2_metal_decode_matches_nonzero_cpu_reference_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = nonzero_tiny_gemma2_payload_store();
    auto view = tiny_model_view(tiny_gemma2_metadata(2), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const auto weights = bind_gemma2_model_weights(model);
    auto plan = create_gemma2_metal_model_plan(model);
    auto no_selection_plan = create_gemma2_metal_model_plan(model);
    const auto& config = plan.config();
    constexpr std::uint32_t cache_position = 9U;

    std::vector<TensorStorage> key_cache;
    std::vector<TensorStorage> value_cache;
    key_cache.reserve(config.block_count);
    value_cache.reserve(config.block_count);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        auto keys = TensorStorage::allocate(TensorDescriptor::from(
            Shape::from({config.context_length, config.attention_head_count_kv * config.attention_key_length}),
            DType::f32
        ));
        auto values = TensorStorage::allocate(TensorDescriptor::from(
            Shape::from({config.context_length, config.attention_head_count_kv * config.attention_value_length}),
            DType::f32
        ));
        for (std::uint32_t position = 0; position < cache_position; ++position) {
            std::vector<float> key_row(config.attention_key_length);
            std::vector<float> value_row(config.attention_value_length);
            for (std::size_t column = 0; column < key_row.size(); ++column) {
                key_row[column] = static_cast<float>((layer + 1U) * (position + 1U))
                    * static_cast<float>(static_cast<int>(column % 7U) - 3) * 0.001F;
            }
            for (std::size_t column = 0; column < value_row.size(); ++column) {
                value_row[column] = static_cast<float>((layer + 2U) * (position + 1U))
                    * static_cast<float>(static_cast<int>(column % 5U) - 2) * 0.0015F;
            }
            write_cache_row(keys, tensor_f32_matrix(1, config.attention_key_length, key_row), position);
            write_cache_row(values, tensor_f32_matrix(1, config.attention_value_length, value_row), position);
        }
        key_cache.push_back(std::move(keys));
        value_cache.push_back(std::move(values));
    }

    for (std::size_t layer = 0; layer < key_cache.size(); ++layer) {
        const auto metal_keys = TensorStorage::from_bytes(
            plan.kv_cache()[layer].key.descriptor(),
            key_cache[layer].bytes()
        );
        const auto metal_values = TensorStorage::from_bytes(
            plan.kv_cache()[layer].value.descriptor(),
            value_cache[layer].bytes()
        );
        plan.kv_cache()[layer].key = copy_tensor_to_backend(metal_keys, BackendKind::metal);
        plan.kv_cache()[layer].value = copy_tensor_to_backend(metal_values, BackendKind::metal);
        no_selection_plan.kv_cache()[layer].key = copy_tensor_to_backend(metal_keys, BackendKind::metal);
        no_selection_plan.kv_cache()[layer].value = copy_tensor_to_backend(metal_values, BackendKind::metal);
    }

    constexpr std::uint32_t token_id = 3U;
    ReferenceDecodeResult expected = [&] {
        try {
            return reference_decode(
                config,
                weights,
                token_id,
                cache_position,
                key_cache,
                value_cache
            );
        } catch (const std::exception& error) {
            throw TestFailure(std::string("CPU reference failed: ") + error.what());
        }
    }();
    const auto schedule = create_gemma2_metal_decode_schedule(plan, cache_position);
    const auto result = run_gemma2_metal_resident_decode_step(
        plan,
        schedule,
        token_id,
        std::sqrt(static_cast<float>(config.embedding_length)),
        true
    );

    expect(result.execution.command_buffer_count == 1U, "decode command-buffer count mismatch");
    expect(result.execution.commit_count == 1U, "decode commit count mismatch");
    expect(result.execution.synchronous_wait_count == 1U, "decode wait count mismatch");
    expect(result.execution.compute_encoder_count == 1U, "decode compute encoder count mismatch");
    expect(result.execution.compute_dispatch_count == 35U, "decode dispatch count mismatch");
    expect(result.execution.blit_encoder_count == 0U, "decode blit encoder count mismatch");
    expect(result.execution.memory_barrier_count == 26U, "decode barrier count mismatch");
    expect(result.execution.transient_buffer_allocation_count == 0U, "decode transient allocation mismatch");
    expect(1U + 15U * 42U + 4U == 635U, "real decode dispatch formula mismatch");
    expect(11U * 42U + 4U == 466U, "real decode barrier formula mismatch");

    const auto actual_logits = copy_tensor_from_backend(result.logits);
    expect_f32_matrix_close(actual_logits, expected.logits, 3.0e-3F, "decode CPU-reference logits");
    expect(result.next_token_selection.has_value(), "decode selected token missing");
    const auto expected_top = top_logit_tokens_f32(expected.logits, 1).front();
    expect(result.next_token_selection->token_id == expected_top.token, "decode selected token mismatch");
    expect_close(result.next_token_selection->logit, expected_top.logit, 3.0e-3F, "decode selected logit mismatch");

    for (std::size_t layer = 0; layer < key_cache.size(); ++layer) {
        expect_f32_storage_close(
            copy_tensor_from_backend(plan.kv_cache()[layer].key),
            TensorStorage::from_bytes(
                plan.kv_cache()[layer].key.descriptor(),
                expected.key_cache[layer].bytes()
            ),
            5.0e-3F,
            "decode direct key-cache write"
        );
        expect_f32_storage_close(
            copy_tensor_from_backend(plan.kv_cache()[layer].value),
            TensorStorage::from_bytes(
                plan.kv_cache()[layer].value.descriptor(),
                expected.value_cache[layer].bytes()
            ),
            3.0e-3F,
            "decode direct value-cache write"
        );
    }

    const auto no_selection_schedule = create_gemma2_metal_decode_schedule(no_selection_plan, cache_position);
    const auto no_selection = run_gemma2_metal_resident_decode_step(
        no_selection_plan,
        no_selection_schedule,
        token_id,
        std::sqrt(static_cast<float>(config.embedding_length)),
        false
    );
    expect(!no_selection.next_token_selection.has_value(), "decode should omit top-logit result");
    expect(no_selection.execution.compute_encoder_count == 1U, "no-selection encoder count mismatch");
    expect(no_selection.execution.compute_dispatch_count == 34U, "no-selection dispatch count mismatch");
    expect(no_selection.execution.blit_encoder_count == 0U, "no-selection blit count mismatch");
    expect(no_selection.execution.memory_barrier_count == 25U, "no-selection barrier count mismatch");
    expect_f32_matrix_close(
        copy_tensor_from_backend(no_selection.logits),
        expected.logits,
        3.0e-3F,
        "no-selection decode logits"
    );

    try {
        (void)run_gemma2_metal_resident_decode_step(
            plan,
            schedule,
            config.vocab_size,
            1.0F
        );
        throw TestFailure("decode should reject invalid token");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 resident Metal decode token is outside vocabulary",
            "decode invalid-token error mismatch"
        );
    }
}

void test_gemma2_fused_rms_norm_add_matches_decode_width_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint32_t hidden = 3584U;
    constexpr float epsilon = 1.0e-6F;
    std::vector<float> input_values(hidden);
    std::vector<float> weight_values(hidden);
    std::vector<float> residual_values(hidden);
    for (std::uint32_t index = 0; index < hidden; ++index) {
        input_values[index] = static_cast<float>(static_cast<int>(index % 37U) - 18) * 0.03125F;
        weight_values[index] = 0.75F + static_cast<float>(index % 11U) * 0.015625F;
        residual_values[index] = static_cast<float>(static_cast<int>(index % 23U) - 11) * 0.0625F;
    }

    const auto input_cpu = tensor_f32_matrix(1U, hidden, input_values);
    const auto weight_cpu = tensor_f32_vector(weight_values);
    const auto residual_cpu = tensor_f32_matrix(1U, hidden, residual_values);
    const auto normalized = rms_norm_f32(input_cpu, weight_cpu, epsilon);
    const auto expected = add_f32(normalized, residual_cpu);
    const auto input = copy_tensor_to_backend(input_cpu, BackendKind::metal);
    const auto weight = copy_tensor_to_backend(weight_cpu, BackendKind::metal);
    const auto residual = copy_tensor_to_backend(residual_cpu, BackendKind::metal);
    const auto output = allocate_backend_buffer(input.descriptor(), BackendKind::metal);

    MetalCommandContext command("Gemma2 fused RMSNorm-add parity", true);
    command.encode_rms_norm_add(input, weight, residual, output, epsilon);
    const auto stats = command.commit_and_wait();
    expect(stats.compute_dispatch_count == 1U, "fused RMSNorm-add dispatch mismatch");
    expect(stats.memory_barrier_count == 0U, "fused RMSNorm-add barrier mismatch");
    expect(stats.transient_buffer_allocation_count == 0U, "fused RMSNorm-add transient allocation mismatch");
    expect_f32_matrix_close(copy_tensor_from_backend(output), expected, 2.0e-5F, "fused RMSNorm-add parity");
}

void test_gemma2_metal_resident_prefill_chunks_once_waits_once_and_matches_incremental_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = tiny_gemma2_payload_store(true, 2);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(2, 2, 1, 256, 32),
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
        "creates Gemma2 Metal schedules when available",
        test_creates_gemma2_metal_schedules_when_available,
    },
    {
        "Gemma2 fused decode attention matches exact-shape CPU reference when available",
        test_gemma2_fused_decode_attention_matches_exact_shape_cpu_reference_when_available,
    },
    {
        "Gemma2 fused RMSNorm-add matches decode width when available",
        test_gemma2_fused_rms_norm_add_matches_decode_width_when_available,
    },
    {
        "Gemma2 Metal decode matches nonzero CPU reference when available",
        test_gemma2_metal_decode_matches_nonzero_cpu_reference_when_available,
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
