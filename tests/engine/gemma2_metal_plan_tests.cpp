#include "gemma2_metal_plan_tests.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/retrieval_proposer.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "tensor_test_support.hpp"
#include "gemma2_reference_test_support.hpp"
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
using local_gguf_engine::engine::Gemma2MetalDecodeOutputMode;
using local_gguf_engine::engine::Gemma2MetalModelPlan;
using local_gguf_engine::engine::Gemma2OutputHeadPolicy;
using local_gguf_engine::engine::Gemma2ModelWeights;
using local_gguf_engine::engine::Gemma2ModelConfig;
using local_gguf_engine::engine::create_gemma2_metal_decode_schedule;
using local_gguf_engine::engine::create_gemma2_metal_model_plan;
using local_gguf_engine::engine::create_gemma2_metal_prefill_schedule;
using local_gguf_engine::engine::create_gemma2_metal_verifier_schedule;
using local_gguf_engine::engine::bind_gemma2_model_weights;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens;
using local_gguf_engine::engine::generate_gemma2_metal_argmax_tokens_with_plan;
using local_gguf_engine::engine::generate_gemma2_metal_sampled_tokens_with_plan;
using local_gguf_engine::engine::load_engine_model_from_gguf;
using local_gguf_engine::engine::run_gemma2_metal_resident_prefill;
using local_gguf_engine::engine::run_gemma2_metal_resident_decode_step;
using local_gguf_engine::engine::run_gemma2_metal_width_two_verifier;
using local_gguf_engine::engine::commit_gemma2_metal_width_two_verification;
using local_gguf_engine::engine::provisionally_commit_gemma2_metal_width_two_verification;
using local_gguf_engine::engine::finalize_gemma2_metal_width_two_verification;
using local_gguf_engine::engine::discard_gemma2_metal_width_two_verification;
using local_gguf_engine::engine::warm_up_gemma2_metal_plan;
using local_gguf_engine::engine::TokenSampler;
using local_gguf_engine::engine::TokenSamplerOptions;
using local_gguf_engine::engine::TokenSelectionRequest;
using local_gguf_engine::engine::TokenLogit;
using local_gguf_engine::engine::RetrievalProposer;
using local_gguf_engine::engine::Gemma2MetalSelectionWorkspace;
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
using local_gguf_engine::tensor::f16_bits_to_f32;
using local_gguf_engine::tensor::f32_to_f16_bits;
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
using local_gguf_engine::tensor::read_f32;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tensor::backends::metal::MetalCommandContext;
using local_gguf_engine::tensor::backends::metal::fill_shared_metal_buffer_u32;
using local_gguf_engine::tensor::backends::metal::gemma2_decode_attention_partition_count;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::ReferenceDecodeResult;
using local_gguf_engine::tests::reference_decode;
using local_gguf_engine::tests::reference_decode_trace;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f16_matrix;
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
    std::uint32_t context_length = 16,
    std::uint32_t sliding_window = 8
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
        metadata_u32("gemma2.attention.sliding_window", sliding_window),
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

void expect_f16_storage_close_absolute_relative(
    const TensorStorage& actual,
    const TensorStorage& expected,
    float absolute_tolerance,
    float relative_tolerance,
    std::string_view message
) {
    expect(actual.descriptor() == expected.descriptor(),
        std::string(message) + " descriptor mismatch");
    expect(actual.descriptor().dtype == DType::f16,
        std::string(message) + " dtype mismatch");
    auto maximum_absolute_error = 0.0F;
    auto maximum_relative_error = 0.0F;
    auto within_bound = true;
    for (std::size_t offset = 0U; offset < actual.bytes().size();
         offset += sizeof(std::uint16_t)) {
        std::uint16_t actual_bits = 0U;
        std::uint16_t expected_bits = 0U;
        std::memcpy(&actual_bits, actual.bytes().data() + offset, sizeof(actual_bits));
        std::memcpy(&expected_bits, expected.bytes().data() + offset, sizeof(expected_bits));
        const auto actual_value = f16_bits_to_f32(actual_bits);
        const auto expected_value = f16_bits_to_f32(expected_bits);
        if (!std::isfinite(actual_value) || !std::isfinite(expected_value)) {
            within_bound = false;
            continue;
        }
        const auto absolute_error = std::fabs(actual_value - expected_value);
        const auto scale = std::max(std::fabs(actual_value), std::fabs(expected_value));
        const auto relative_error = scale == 0.0F ? 0.0F : absolute_error / scale;
        maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
        maximum_relative_error = std::max(maximum_relative_error, relative_error);
        if (absolute_error > absolute_tolerance + relative_tolerance * scale) {
            within_bound = false;
        }
    }
    expect(
        within_bound,
        std::string(message) + ": max_abs=" + std::to_string(maximum_absolute_error)
            + ", max_rel=" + std::to_string(maximum_relative_error)
    );
}

void write_cache_row(TensorStorage& cache, const TensorStorage& row, std::uint32_t position) {
    const auto row_bytes = row.descriptor().byte_count;
    const auto offset = static_cast<std::size_t>(position) * row_bytes;
    expect(offset + row_bytes <= cache.mutable_bytes().size(), "reference KV row range mismatch");
    std::memcpy(cache.mutable_bytes().data() + offset, row.bytes().data(), row_bytes);
}

TensorStorage round_f32_storage_through_f16(const TensorStorage& input) {
    auto rounded = TensorStorage::allocate(input.descriptor());
    for (std::uint64_t index = 0; index < input.descriptor().element_count; ++index) {
        float value = 0.0F;
        const auto offset = static_cast<std::size_t>(index * sizeof(float));
        std::memcpy(&value, input.bytes().data() + offset, sizeof(value));
        value = f16_bits_to_f32(f32_to_f16_bits(value));
        std::memcpy(rounded.mutable_bytes().data() + offset, &value, sizeof(value));
    }
    return rounded;
}

TensorStorage f16_ring_cache_from_f32(
    const TensorStorage& logical_cache,
    const TensorDescriptor& ring_descriptor,
    std::uint32_t populated_rows,
    std::uint32_t row_mask
) {
    auto ring = TensorStorage::allocate(ring_descriptor);
    const auto width = logical_cache.descriptor().shape.dimension(1);
    for (std::uint32_t logical_row = 0; logical_row < populated_rows; ++logical_row) {
        const auto physical_row = logical_row & row_mask;
        for (std::uint64_t column = 0; column < width; ++column) {
            float value = 0.0F;
            const auto source_offset = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(logical_row) * width + column) * sizeof(float)
            );
            std::memcpy(&value, logical_cache.bytes().data() + source_offset, sizeof(value));
            const auto bits = f32_to_f16_bits(value);
            const auto destination_offset = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(physical_row) * width + column) * sizeof(std::uint16_t)
            );
            std::memcpy(ring.mutable_bytes().data() + destination_offset, &bits, sizeof(bits));
        }
    }
    return ring;
}

void expect_f16_ring_matches_f32(
    const TensorStorage& actual_ring,
    const TensorStorage& expected_logical,
    std::uint32_t first_logical_row,
    std::uint32_t logical_row_end,
    std::uint32_t row_mask,
    float tolerance,
    std::string_view message
) {
    const auto width = expected_logical.descriptor().shape.dimension(1);
    for (std::uint32_t logical_row = first_logical_row; logical_row < logical_row_end; ++logical_row) {
        const auto physical_row = logical_row & row_mask;
        for (std::uint64_t column = 0; column < width; ++column) {
            float expected_value = 0.0F;
            const auto expected_offset = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(logical_row) * width + column) * sizeof(float)
            );
            std::memcpy(&expected_value, expected_logical.bytes().data() + expected_offset, sizeof(expected_value));
            const auto expected_bits = f32_to_f16_bits(expected_value);
            std::uint16_t actual_bits = 0;
            const auto actual_offset = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(physical_row) * width + column) * sizeof(std::uint16_t)
            );
            std::memcpy(&actual_bits, actual_ring.bytes().data() + actual_offset, sizeof(actual_bits));
            if (tolerance == 0.0F) {
                expect(actual_bits == expected_bits, message);
            } else {
                expect_close(
                    f16_bits_to_f32(actual_bits),
                    f16_bits_to_f32(expected_bits),
                    std::max(
                        tolerance,
                        std::abs(f16_bits_to_f32(expected_bits)) * 0.001F
                    ),
                    message
                );
            }
        }
    }
}

std::size_t expected_kv_cache_byte_count(const Gemma2ModelConfig& config) {
    std::size_t bytes = 0;
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto rows = local_gguf_engine::engine::gemma2_layer_uses_sliding_window(layer)
            ? config.sliding_window : config.context_length;
        bytes += static_cast<std::size_t>(rows) * config.attention_head_count_kv
            * (config.attention_key_length + config.attention_value_length) * sizeof(std::uint16_t);
    }
    return bytes;
}

std::vector<float> read_backend_f32_row(
    const local_gguf_engine::tensor::BackendBuffer& buffer,
    std::uint32_t row
) {
    const auto storage = copy_tensor_from_backend(buffer);
    expect(storage.descriptor().dtype == DType::f32
            && storage.descriptor().shape.rank() == 2U
            && row < storage.descriptor().shape.dimension(0U),
        "backend F32 row descriptor mismatch");
    const auto width = storage.descriptor().shape.dimension(1U);
    std::vector<float> values(static_cast<std::size_t>(width));
    for (std::uint64_t column = 0U; column < width; ++column) {
        const std::array<std::uint64_t, 2> index {row, column};
        values[static_cast<std::size_t>(column)] = read_f32(storage, index);
    }
    return values;
}

void expect_f32_rows_close(
    std::span<const float> actual,
    std::span<const float> expected,
    float tolerance,
    std::string_view message
) {
    expect(actual.size() == expected.size(), std::string(message) + " size mismatch");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        expect_close(actual[index], expected[index], tolerance, message);
    }
}

std::size_t expected_scratch_byte_count(const Gemma2MetalModelPlan& plan) {
    const auto& prefill = plan.prefill_scratch();
    const auto& decode = plan.decode_scratch();
    const auto& selection = plan.selection_workspace();
    const auto& verifier = plan.verifier_scratch();
    const auto logical_bytes = backend_buffer_byte_count(prefill.token_id_staging)
        + backend_buffer_byte_count(prefill.normalized_hidden)
        + backend_buffer_byte_count(prefill.raw_query)
        + backend_buffer_byte_count(prefill.raw_key)
        + backend_buffer_byte_count(prefill.query)
        + backend_buffer_byte_count(prefill.value)
        + backend_buffer_byte_count(prefill.current_chunk_key)
        + backend_buffer_byte_count(prefill.current_chunk_value)
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
        + backend_buffer_byte_count(decode.token_id_staging)
        + backend_buffer_byte_count(decode.normalized_hidden)
        + backend_buffer_byte_count(decode.raw_query)
        + backend_buffer_byte_count(decode.raw_key)
        + backend_buffer_byte_count(decode.raw_value)
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
        + backend_buffer_byte_count(verifier.token_id_staging)
        + backend_buffer_byte_count(verifier.normalized_hidden)
        + backend_buffer_byte_count(verifier.raw_query)
        + backend_buffer_byte_count(verifier.raw_key)
        + backend_buffer_byte_count(verifier.raw_value)
        + backend_buffer_byte_count(verifier.query)
        + backend_buffer_byte_count(verifier.attention_partial_output)
        + backend_buffer_byte_count(verifier.attention_partial_maxima)
        + backend_buffer_byte_count(verifier.attention_partial_denominators)
        + backend_buffer_byte_count(verifier.attention_output)
        + backend_buffer_byte_count(verifier.projected_attention_output)
        + backend_buffer_byte_count(verifier.post_attention_residual)
        + backend_buffer_byte_count(verifier.normalized_feed_forward_input)
        + backend_buffer_byte_count(verifier.feed_forward_gate)
        + backend_buffer_byte_count(verifier.feed_forward_up)
        + backend_buffer_byte_count(verifier.feed_forward_gated)
        + backend_buffer_byte_count(verifier.projected_feed_forward_output)
        + backend_buffer_byte_count(verifier.feed_forward_residual)
        + backend_buffer_byte_count(verifier.logits)
        + backend_buffer_byte_count(selection.candidate_runs_a)
        + backend_buffer_byte_count(selection.candidate_runs_b)
        + backend_buffer_byte_count(selection.compact_result)
        + backend_buffer_byte_count(selection.history_token_staging);
    auto verifier_kv_bytes = std::size_t {0};
    for (const auto& layer : verifier.kv_backups) {
        verifier_kv_bytes += backend_buffer_byte_count(layer.backup_key)
            + backend_buffer_byte_count(layer.backup_value);
    }
    return logical_bytes + verifier_kv_bytes
        - backend_buffer_byte_count(prefill.attention_probabilities)
        - backend_buffer_byte_count(prefill.projected_attention_output)
        - backend_buffer_byte_count(prefill.normalized_feed_forward_input)
        - backend_buffer_byte_count(prefill.projected_feed_forward_output)
        - backend_buffer_byte_count(prefill.feed_forward_gated);
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
    constexpr std::size_t target_kv_bytes =
        (21ULL * 8192ULL + 21ULL * 4096ULL) * 8ULL * 256ULL * 2ULL * sizeof(std::uint16_t);
    expect(target_kv_bytes == 2'113'929'216ULL, "Gemma2 target KV byte formula mismatch");
    constexpr std::size_t target_prefill_stage_bytes =
        128ULL * 8ULL * (256ULL + 256ULL) * sizeof(std::uint16_t);
    expect(target_prefill_stage_bytes == 1'048'576ULL,
        "Gemma2 target wrapped-prefill stage byte formula mismatch");
    expect(plan.scratch_byte_count() == expected_scratch_byte_count(plan), "Gemma2 Metal scratch byte count mismatch");
    const auto& prefill = plan.prefill_scratch();
    const auto same_storage = [](const auto& left, const auto& right) {
        return &backend_buffer_storage(left) == &backend_buffer_storage(right);
    };
    expect(same_storage(prefill.attention_scores, prefill.attention_probabilities),
        "prefill scores and probabilities must share storage");
    expect(same_storage(prefill.normalized_hidden, prefill.projected_attention_output)
            && same_storage(prefill.normalized_hidden, prefill.normalized_feed_forward_input)
            && same_storage(prefill.normalized_hidden, prefill.projected_feed_forward_output),
        "prefill hidden-width liveness aliases mismatch");
    expect(same_storage(prefill.feed_forward_gate, prefill.feed_forward_gated),
        "prefill gate/GELU liveness alias mismatch");
    expect(prefill.current_chunk_key.descriptor().shape == Shape::from({128U, 1U, 256U})
            && prefill.current_chunk_value.descriptor().shape == Shape::from({128U, 1U, 256U})
            && prefill.current_chunk_key.descriptor().dtype == DType::f16
            && prefill.current_chunk_value.descriptor().dtype == DType::f16,
        "prefill wrapped-current-chunk stage layout mismatch");
    expect(!same_storage(prefill.current_chunk_key, prefill.current_chunk_value)
            && !same_storage(prefill.current_chunk_key, plan.kv_cache()[0].key)
            && !same_storage(prefill.current_chunk_value, plan.kv_cache()[0].value),
        "prefill wrapped-current-chunk stage must own distinct resident storage");

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
        plan.kv_cache()[0].key.descriptor().shape == Shape::from({8, 1, 256}),
        "Gemma2 Metal KV key shape mismatch"
    );
    expect(plan.kv_cache()[0].key.descriptor().dtype == DType::f16, "Gemma2 Metal KV dtype mismatch");
    expect(
        plan.kv_cache()[0].row_capacity == 8U && plan.kv_cache()[0].row_mask == 7U,
        "Gemma2 Metal KV ring layout mismatch"
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
    const auto& bytes = plan.decode_scheduled_weight_bytes();
    expect(
        bytes.embedding + bytes.attention_projections + bytes.feed_forward_gate_up
            + bytes.feed_forward_down + bytes.output_head + bytes.norms == bytes.total,
        "Gemma2 scheduled decode weight categories do not sum"
    );
    expect(bytes.embedding == 256U * sizeof(std::uint16_t),
        "Gemma2 scheduled embedding must count one F16 row");
    expect(bytes.output_head == plan.weights().output_weight.descriptor().byte_count,
        "Gemma2 scheduled output head byte count mismatch");
}

void test_gemma2_q8_output_policy_preserves_f16_embedding_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    auto payload_store = tiny_gemma2_payload_store();
    const auto source_weight_bytes = payload_byte_count(
        std::span<const GGUFTensorPayloadView>(payload_store.payloads)
    );
    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    auto plan = create_gemma2_metal_model_plan(model, Gemma2OutputHeadPolicy::q8_0);

    const auto expected_q8_bytes = TensorDescriptor::from_quantized_gguf_weight_matrix(
        Shape::from({256U, 4U}), DType::q8_0
    ).byte_count;
    expect(plan.output_head_policy() == Gemma2OutputHeadPolicy::q8_0, "Gemma2 Q8 output policy mismatch");
    expect(plan.weights().token_embedding_weight.descriptor().dtype == DType::f16,
        "Gemma2 Q8 policy changed embedding dtype");
    expect(plan.weights().output_weight.descriptor().dtype == DType::q8_0,
        "Gemma2 Q8 output dtype mismatch");
    expect(&backend_buffer_storage(plan.weights().token_embedding_weight)
            != &backend_buffer_storage(plan.weights().output_weight),
        "Gemma2 Q8 output must not alias F16 embedding");
    expect(plan.uploaded_weight_byte_count() == source_weight_bytes + expected_q8_bytes,
        "Gemma2 Q8 resident weight accounting mismatch");
    expect(plan.decode_scheduled_weight_bytes().output_head == expected_q8_bytes,
        "Gemma2 Q8 scheduled output bytes mismatch");
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
    const auto verifier = create_gemma2_metal_verifier_schedule(plan, 9);

    expect(prefill.phase() == Gemma2MetalSchedulePhase::prefill, "prefill phase mismatch");
    expect(prefill.scratch_ownership() == Gemma2MetalScratchOwnership::resident_prefill, "prefill ownership mismatch");
    expect(prefill.token_count() == 3U && prefill.cache_position() == 4U, "prefill range mismatch");
    expect(prefill.total_kv_token_count() == 7U, "prefill KV count mismatch");
    expect(decode.phase() == Gemma2MetalSchedulePhase::decode, "decode phase mismatch");
    expect(decode.scratch_ownership() == Gemma2MetalScratchOwnership::resident_decode, "decode ownership mismatch");
    expect(decode.token_count() == 1U && decode.cache_position() == 9U, "decode range mismatch");
    expect(verifier.phase() == Gemma2MetalSchedulePhase::verify, "verifier phase mismatch");
    expect(verifier.scratch_ownership() == Gemma2MetalScratchOwnership::resident_verifier,
        "verifier ownership mismatch");
    expect(verifier.token_count() == 2U && verifier.cache_position() == 9U,
        "verifier range mismatch");
    expect(verifier.layers()[0].attention_token_count == 8U
            && verifier.layers()[0].second_attention_token_count == 8U,
        "local verifier attention bounds mismatch");
    expect(verifier.layers()[1].attention_token_count == 10U
            && verifier.layers()[1].second_attention_token_count == 11U,
        "global verifier attention bounds mismatch");
    expect(decode.layers().size() == 2U, "decode layer count mismatch");
    expect(decode.layers()[0].attention_token_count == 8U, "sliding attention count mismatch");
    expect(decode.layers()[1].attention_token_count == 10U, "global attention count mismatch");
    expect(
        plan.kv_cache()[0].row_capacity == 8U && plan.kv_cache()[0].row_mask == 7U,
        "sliding KV ring layout mismatch"
    );
    expect(
        plan.kv_cache()[1].row_capacity == 16U && plan.kv_cache()[1].row_mask == 15U,
        "global KV layout mismatch"
    );
}

void test_gemma2_metal_schedule_preserves_target_4k_8k_populated_bounds_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    auto payload_store = tiny_gemma2_payload_store(true, 2U);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(2U, 2U, 1U, 256U, 8192U, 4096U),
        std::move(payload_store.payloads)
    );
    const auto model = load_engine_model_from_gguf(view);
    const auto plan = create_gemma2_metal_model_plan(model);

    const auto before_boundary = create_gemma2_metal_decode_schedule(plan, 4095U);
    expect(before_boundary.layers()[0].attention_token_count == 4096U
            && before_boundary.layers()[1].attention_token_count == 4096U,
        "target schedule 4K boundary mismatch");
    const auto after_boundary = create_gemma2_metal_decode_schedule(plan, 4096U);
    expect(after_boundary.layers()[0].attention_token_count == 4096U
            && after_boundary.layers()[1].attention_token_count == 4097U,
        "target schedule local/global growth mismatch");
    const auto full = create_gemma2_metal_decode_schedule(plan, 8191U);
    expect(full.layers()[0].attention_token_count == 4096U
            && full.layers()[1].attention_token_count == 8192U,
        "target schedule 8K boundary mismatch");
}

void test_gemma2_metal_warmup_is_state_neutral_and_covers_production_routes_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = nonzero_tiny_gemma2_payload_store(2U);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(2U, 2U, 1U, 256U, 32U, 32U),
        std::move(payload_store.payloads)
    );
    const auto model = load_engine_model_from_gguf(view);
    auto warmed_plan = create_gemma2_metal_model_plan(model);
    auto reference_plan = create_gemma2_metal_model_plan(model);
    const auto identities = warmed_plan.allocation_identities();
    std::array<std::uint32_t, 16> warmup_tokens {};
    warmup_tokens.fill(1U);
    const TokenSamplerOptions sampler_options {
        .temperature = 0.8F,
        .top_k = 4U,
        .top_p = 0.95F,
        .min_p = 0.05F,
        .repetition_penalty = 1.1F,
        .seed = 42U,
    };
    auto sampler = TokenSampler(sampler_options);
    auto sampler_control = TokenSampler(sampler_options);

    const auto stats = warm_up_gemma2_metal_plan(
        warmed_plan, std::span<const std::uint32_t, 16>(warmup_tokens), sampler
    );

    expect(stats.prefill_token_count == 16U, "warm-up prefill token count mismatch");
    expect(stats.kv_clear_pass_count == 0U && stats.host_cleared_kv_bytes == 0U,
        "warm-up must not materialize the KV allocation with host clears");
    expect(stats.prefill.command_buffer_count == 1U && stats.prefill.synchronous_wait_count == 1U,
        "warm-up prefill must submit and synchronize exactly once");
    expect(stats.prefill.scheduled_layer_execution_count == 2U && stats.prefill.kv_write_count == 2U,
        "warm-up prefill layer/KV coverage mismatch");
    expect(stats.prefill.q4_exact_16_projection_count > 0U,
        "warm-up did not execute exact-16 Q4 projection");
    expect(stats.prefill.q6_exact_16_projection_count > 0U,
        "warm-up did not execute exact-16 Q6 projection");
    expect(stats.prefill.f16_projection_count > 0U && stats.prefill.selector_dispatch_count > 0U,
        "warm-up prefill did not execute output projection and selection");
    expect(stats.decode.command_buffer_count == 1U && stats.decode.synchronous_wait_count == 1U,
        "warm-up decode must submit and synchronize exactly once");
    expect(stats.decode.q4_decode_projection_count > 0U && stats.decode.q6_decode_projection_count > 0U,
        "warm-up did not execute decode K-quant projections");
    expect(stats.decode.f16_projection_count > 0U && stats.decode.decode_attention_dispatch_count > 0U,
        "warm-up did not execute decode output projection and attention");
    expect(stats.decode.selector_dispatch_count > 0U && stats.decode.compact_readback_bytes > 0U,
        "warm-up decode did not execute compact selection and readback");
    expect(stats.verifier.command_buffer_count == 1U
            && stats.verifier.synchronous_wait_count == 1U
            && stats.verifier.q4_width_two_projection_count > 0U
            && stats.verifier.q6_width_two_projection_count > 0U
            && stats.verifier.selector_dispatch_count > 0U
            && stats.verifier.compact_readback_bytes > 0U,
        "warm-up did not execute the width-two verifier and selector");
    expect(warmed_plan.allocation_identities() == identities,
        "warm-up changed resident allocation identities");

    const auto stale_warmup_key = copy_tensor_from_backend(warmed_plan.kv_cache().front().key);
    expect(
        std::any_of(stale_warmup_key.bytes().begin(), stale_warmup_key.bytes().end(), [](std::uint8_t byte) {
            return byte != 0U;
        }),
        "warm-up test must retain stale physical KV so parity cannot depend on zero fill"
    );

    const std::array<TokenLogit, 4> sampler_candidates {{
        {.token = 0U, .logit = 0.1F},
        {.token = 1U, .logit = 0.2F},
        {.token = 2U, .logit = 0.3F},
        {.token = 3U, .logit = 0.4F},
    }};
    expect(
        sampler.sample(sampler_candidates) == sampler_control.sample(sampler_candidates),
        "warm-up advanced the caller sampler"
    );

    constexpr std::array<std::uint32_t, 3> prompt {1U, 2U, 3U};
    const auto warmed = generate_gemma2_metal_argmax_tokens_with_plan(
        warmed_plan, prompt, 0U, 2U, {}
    );
    const auto reference = generate_gemma2_metal_argmax_tokens_with_plan(
        reference_plan, prompt, 0U, 2U, {}
    );
    expect(warmed.selected_token_ids == reference.selected_token_ids,
        "warm-up changed first production token selection");
    for (std::size_t layer = 0; layer < warmed_plan.kv_cache().size(); ++layer) {
        const auto valid_kv_bytes = 5U
            * warmed_plan.kv_cache()[layer].key.descriptor().byte_count
            / warmed_plan.kv_cache()[layer].row_capacity;
        const auto warmed_key = copy_tensor_from_backend(warmed_plan.kv_cache()[layer].key);
        const auto reference_key = copy_tensor_from_backend(reference_plan.kv_cache()[layer].key);
        expect(
            std::equal(
                warmed_key.bytes().begin(),
                warmed_key.bytes().begin() + static_cast<std::ptrdiff_t>(valid_kv_bytes),
                reference_key.bytes().begin()
            ),
            "warm-up changed production key KV contents"
        );
        const auto warmed_value = copy_tensor_from_backend(warmed_plan.kv_cache()[layer].value);
        const auto reference_value = copy_tensor_from_backend(reference_plan.kv_cache()[layer].value);
        expect(
            std::equal(
                warmed_value.bytes().begin(),
                warmed_value.bytes().begin() + static_cast<std::ptrdiff_t>(valid_kv_bytes),
                reference_value.bytes().begin()
            ),
            "warm-up changed production value KV contents"
        );
    }
}

void test_gemma2_f16_kv_store_rounds_values_and_wraps_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    constexpr std::uint32_t rows = 2U;
    constexpr std::uint32_t heads = 1U;
    constexpr std::uint32_t dimension = 256U;
    constexpr std::uint32_t cache_rows = 8U;
    std::vector<float> raw_key(rows * dimension);
    std::vector<float> raw_value(rows * dimension);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < dimension; ++column) {
            const auto index = row * dimension + column;
            raw_key[index] = static_cast<float>(static_cast<int>(column % 19U) - 9) * 0.03125F
                + static_cast<float>(row) * 0.0078125F;
            raw_value[index] = static_cast<float>(static_cast<int>(column % 23U) - 11) * 0.015625F
                - static_cast<float>(row) * 0.00390625F;
        }
    }
    raw_value[0] = 65504.0F;
    raw_value[1] = -65504.0F;
    raw_value[2] = 0.00006103515625F;

    const auto raw_key_cpu = tensor_f32_matrix(rows, dimension, raw_key);
    const auto raw_value_cpu = tensor_f32_matrix(rows, dimension, raw_value);
    const auto expected_key = rope_neox_f32_by_heads_with_position_start(
        raw_key_cpu, 7U, heads, dimension, dimension, 10000.0F
    );
    const auto cache_descriptor = TensorDescriptor::from(
        Shape::from({cache_rows, heads, dimension}), DType::f16
    );
    const auto key_cache = allocate_backend_buffer(cache_descriptor, BackendKind::metal);
    const auto value_cache = allocate_backend_buffer(cache_descriptor, BackendKind::metal);

    MetalCommandContext command("Gemma2 exact F16 KV store");
    command.encode_gemma2_kv_cache_store(
        copy_tensor_to_backend(raw_key_cpu, BackendKind::metal),
        copy_tensor_to_backend(raw_value_cpu, BackendKind::metal),
        key_cache,
        value_cache,
        7U,
        heads,
        dimension,
        dimension,
        10000.0F,
        cache_rows - 1U,
        rows
    );
    const auto stats = command.commit_and_wait();
    expect(stats.compute_dispatch_count == 1U, "F16 KV store dispatch count mismatch");
    expect(stats.blit_encoder_count == 0U, "F16 KV store must not use a blit encoder");

    const auto actual_key = copy_tensor_from_backend(key_cache);
    const auto actual_value = copy_tensor_from_backend(value_cache);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const auto physical_row = (7U + row) & (cache_rows - 1U);
        for (std::uint32_t column = 0; column < dimension; ++column) {
            const std::array<std::uint64_t, 2> source_index {row, column};
            const auto key_bits = f32_to_f16_bits(read_f32(expected_key, source_index));
            const auto value_bits = f32_to_f16_bits(raw_value[row * dimension + column]);
            const auto byte_offset = static_cast<std::size_t>(physical_row * dimension + column)
                * sizeof(std::uint16_t);
            std::uint16_t actual_key_bits = 0;
            std::uint16_t actual_value_bits = 0;
            std::memcpy(&actual_key_bits, actual_key.bytes().data() + byte_offset, sizeof(actual_key_bits));
            std::memcpy(&actual_value_bits, actual_value.bytes().data() + byte_offset, sizeof(actual_value_bits));
            // Metal and host transcendental implementations may land on
            // opposite sides of one F16 boundary; the narrowing itself is
            // proved bit-exact by V, which has no transcendental operation.
            expect_close(
                f16_bits_to_f32(actual_key_bits),
                f16_bits_to_f32(key_bits),
                5.0e-4F,
                "F16 KV store rotated key mismatch"
            );
            expect(actual_value_bits == value_bits, "F16 KV store value bits mismatch");
        }
    }
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

    for (auto& value : key_values) value = f16_bits_to_f32(f32_to_f16_bits(value));
    for (auto& value : value_values) value = f16_bits_to_f32(f32_to_f16_bits(value));

    const auto query_cpu = tensor_f32_matrix(1, query_heads * dimension, query_values);
    const auto key_cpu = tensor_f32_matrix(cache_rows, kv_heads * dimension, key_values);
    const auto value_cpu = tensor_f32_matrix(cache_rows, kv_heads * dimension, value_values);
    const auto query = copy_tensor_to_backend(query_cpu, BackendKind::metal);
    const auto key_f16 = tensor_f16_matrix(cache_rows, kv_heads * dimension, key_values);
    const auto value_f16 = tensor_f16_matrix(cache_rows, kv_heads * dimension, value_values);
    const auto key = copy_tensor_to_backend(
        TensorStorage::from_bytes(
            TensorDescriptor::from(Shape::from({cache_rows, kv_heads, dimension}), DType::f16),
            key_f16.bytes()
        ),
        BackendKind::metal
    );
    const auto value = copy_tensor_to_backend(
        TensorStorage::from_bytes(
            TensorDescriptor::from(Shape::from({cache_rows, kv_heads, dimension}), DType::f16),
            value_f16.bytes()
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
            softcap,
            cache_rows - 1U
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
            softcap,
            cache_rows - 1U
        );
        throw TestFailure("fused attention should reject an empty window");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Metal Gemma 2 decode attention token count must not be zero",
            "fused attention empty-window error mismatch"
        );
    }

    try {
        MetalCommandContext command("Gemma2 fused-attention invalid cache mask", true);
        command.encode_gemma2_decode_attention(
            query, key, value, output, partial_output, partial_maxima, partial_denominators,
            cache_rows, 1U, scale, softcap, cache_rows
        );
        throw TestFailure("fused attention should reject an invalid cache mask");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what())
                == "Metal Gemma 2 decode attention cache mask is invalid",
            "fused attention cache-mask error mismatch"
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
            0U, 1U, scale, softcap, cache_rows - 1U
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
    auto kv_only_plan = create_gemma2_metal_model_plan(model);
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
            write_cache_row(
                keys,
                round_f32_storage_through_f16(
                    tensor_f32_matrix(1, config.attention_key_length, key_row)
                ),
                position
            );
            write_cache_row(
                values,
                round_f32_storage_through_f16(
                    tensor_f32_matrix(1, config.attention_value_length, value_row)
                ),
                position
            );
        }
        key_cache.push_back(std::move(keys));
        value_cache.push_back(std::move(values));
    }

    for (std::size_t layer = 0; layer < key_cache.size(); ++layer) {
        const auto metal_keys = f16_ring_cache_from_f32(
            key_cache[layer],
            plan.kv_cache()[layer].key.descriptor(),
            cache_position,
            plan.kv_cache()[layer].row_mask
        );
        const auto metal_values = f16_ring_cache_from_f32(
            value_cache[layer],
            plan.kv_cache()[layer].value.descriptor(),
            cache_position,
            plan.kv_cache()[layer].row_mask
        );
        plan.kv_cache()[layer].key = copy_tensor_to_backend(metal_keys, BackendKind::metal);
        plan.kv_cache()[layer].value = copy_tensor_to_backend(metal_values, BackendKind::metal);
        no_selection_plan.kv_cache()[layer].key = copy_tensor_to_backend(metal_keys, BackendKind::metal);
        no_selection_plan.kv_cache()[layer].value = copy_tensor_to_backend(metal_values, BackendKind::metal);
        kv_only_plan.kv_cache()[layer].key = copy_tensor_to_backend(metal_keys, BackendKind::metal);
        kv_only_plan.kv_cache()[layer].value = copy_tensor_to_backend(metal_values, BackendKind::metal);
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
        TokenSelectionRequest {}
    );

    expect(result.execution.command_buffer_count == 1U, "decode command-buffer count mismatch");
    expect(result.execution.commit_count == 1U, "decode commit count mismatch");
    expect(result.execution.synchronous_wait_count == 1U, "decode wait count mismatch");
    expect(result.execution.compute_encoder_count == 1U, "decode compute encoder count mismatch");
    expect(result.execution.compute_dispatch_count == 34U, "decode dispatch count mismatch");
    expect(result.execution.blit_encoder_count == 0U, "decode blit encoder count mismatch");
    expect(result.execution.memory_barrier_count == 25U, "decode barrier count mismatch");
    expect(result.execution.transient_buffer_allocation_count == 0U, "decode transient allocation mismatch");
    expect(result.execution.f16_projection_count > 0U, "decode F16 projection family missing");
    expect(result.execution.decode_attention_dispatch_count == 2U,
        "decode attention family count mismatch");
    expect(result.execution.decode_attention_partition_count == 2U,
        "decode attention partition count mismatch");
    expect(1U + 15U * 42U + 4U == 635U, "real decode dispatch formula mismatch");
    expect(11U * 42U + 4U == 466U, "real decode barrier formula mismatch");

    const auto actual_logits = copy_tensor_from_backend(result.logits);
    expect_f32_matrix_close(
        softcap_f32(actual_logits, config.final_logit_softcap),
        expected.logits,
        3.0e-3F,
        "decode CPU-reference logits"
    );
    expect(result.candidates.has_value(), "decode candidates missing");
    const auto expected_top = local_gguf_engine::tensor::ops::top_logit_f32(expected.logits);
    expect(result.candidates->values[0].token == expected_top.token, "decode selected token mismatch");
    expect_close(result.candidates->values[0].logit, expected_top.logit, 3.0e-3F, "decode selected logit mismatch");
    expect(result.execution.compact_readback_bytes == 8U, "decode compact readback mismatch");

    for (std::size_t layer = 0; layer < key_cache.size(); ++layer) {
        const auto first_live_row = plan.layers()[layer].uses_sliding_window
            ? cache_position + 1U - std::min(cache_position + 1U, config.sliding_window)
            : 0U;
        expect_f16_ring_matches_f32(
            copy_tensor_from_backend(plan.kv_cache()[layer].key),
            expected.key_cache[layer],
            first_live_row,
            cache_position + 1U,
            plan.kv_cache()[layer].row_mask,
            5.0e-4F,
            "decode direct key-cache write"
        );
        expect_f16_ring_matches_f32(
            copy_tensor_from_backend(plan.kv_cache()[layer].value),
            expected.value_cache[layer],
            first_live_row,
            cache_position + 1U,
            plan.kv_cache()[layer].row_mask,
            5.0e-4F,
            "decode direct value-cache write"
        );
    }

    const auto no_selection_schedule = create_gemma2_metal_decode_schedule(no_selection_plan, cache_position);
    const auto no_selection = run_gemma2_metal_resident_decode_step(
        no_selection_plan,
        no_selection_schedule,
        token_id,
        std::sqrt(static_cast<float>(config.embedding_length)),
        std::nullopt
    );
    expect(!no_selection.candidates.has_value(), "decode should omit candidates");
    expect(no_selection.execution.compute_encoder_count == 1U, "no-selection encoder count mismatch");
    expect(no_selection.execution.compute_dispatch_count == 33U, "no-selection dispatch count mismatch");
    expect(no_selection.execution.blit_encoder_count == 0U, "no-selection blit count mismatch");
    expect(no_selection.execution.memory_barrier_count == 25U, "no-selection barrier count mismatch");
    expect_f32_matrix_close(
        softcap_f32(copy_tensor_from_backend(no_selection.logits), config.final_logit_softcap),
        expected.logits,
        3.0e-3F,
        "no-selection decode logits"
    );

    const auto kv_only_schedule = create_gemma2_metal_decode_schedule(kv_only_plan, cache_position);
    const auto kv_only = run_gemma2_metal_resident_decode_step(
        kv_only_plan,
        kv_only_schedule,
        token_id,
        std::sqrt(static_cast<float>(config.embedding_length)),
        std::nullopt,
        Gemma2MetalDecodeOutputMode::kv_only
    );
    expect(!kv_only.produced_logits && !kv_only.candidates.has_value(),
        "KV-only decode must omit logits and candidates");
    expect(kv_only.execution.compute_dispatch_count + 2U == no_selection.execution.compute_dispatch_count,
        "KV-only decode must elide final norm and output-head dispatches");
    expect(kv_only.execution.memory_barrier_count + 2U == no_selection.execution.memory_barrier_count,
        "KV-only decode must elide final norm and output-head barriers");
    expect(kv_only.execution.f16_projection_count + 1U == no_selection.execution.f16_projection_count,
        "KV-only decode must elide one F16 output projection");
    expect(kv_only.execution.output_head_elision_count == 1U,
        "KV-only decode elision accounting mismatch");
    expect(
        kv_only.execution.output_elided_weight_bytes
            == kv_only_plan.weights().output_norm_weight.descriptor().byte_count
                + kv_only_plan.weights().output_weight.descriptor().byte_count,
        "KV-only decode elided weight-byte accounting mismatch"
    );
    for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
        const auto kv_only_key = copy_tensor_from_backend(kv_only_plan.kv_cache()[layer].key);
        const auto full_key = copy_tensor_from_backend(plan.kv_cache()[layer].key);
        expect(
            std::equal(kv_only_key.bytes().begin(), kv_only_key.bytes().end(), full_key.bytes().begin()),
            "KV-only decode key cache differs from full-output decode"
        );
        const auto kv_only_value = copy_tensor_from_backend(kv_only_plan.kv_cache()[layer].value);
        const auto full_value = copy_tensor_from_backend(plan.kv_cache()[layer].value);
        expect(
            std::equal(kv_only_value.bytes().begin(), kv_only_value.bytes().end(), full_value.bytes().begin()),
            "KV-only decode value cache differs from full-output decode"
        );
    }

    try {
        (void)run_gemma2_metal_resident_decode_step(
            kv_only_plan,
            kv_only_schedule,
            token_id,
            1.0F,
            TokenSelectionRequest {},
            Gemma2MetalDecodeOutputMode::kv_only
        );
        throw TestFailure("KV-only decode should reject token selection");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "Gemma2 KV-only decode cannot request token selection",
            "KV-only selection error mismatch"
        );
    }

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
    constexpr std::uint32_t rows = 2U;
    constexpr float epsilon = 1.0e-6F;
    std::vector<float> input_values(rows * hidden);
    std::vector<float> weight_values(hidden);
    std::vector<float> residual_values(rows * hidden);
    for (std::uint32_t index = 0; index < rows * hidden; ++index) {
        input_values[index] = static_cast<float>(static_cast<int>(index % 37U) - 18) * 0.03125F;
        residual_values[index] = static_cast<float>(static_cast<int>(index % 23U) - 11) * 0.0625F;
    }
    for (std::uint32_t index = 0; index < hidden; ++index) {
        weight_values[index] = 0.75F + static_cast<float>(index % 11U) * 0.015625F;
    }

    const auto input_cpu = tensor_f32_matrix(rows, hidden, input_values);
    const auto weight_cpu = tensor_f32_vector(weight_values);
    const auto residual_cpu = tensor_f32_matrix(rows, hidden, residual_values);
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

void test_gemma2_metal_resident_prefill_stages_wrapped_sliding_chunks_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    auto payload_store = nonzero_tiny_gemma2_payload_store(2U);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(2U, 2U, 1U, 256U, 32U, 8U),
        std::move(payload_store.payloads)
    );
    const auto model = load_engine_model_from_gguf(view);
    std::array<std::uint32_t, 13U> tokens {};
    for (std::size_t index = 0U; index < tokens.size(); ++index) {
        tokens[index] = static_cast<std::uint32_t>((index * 3U + 1U) % 4U);
    }
    constexpr auto cache_pattern = std::uint32_t {0x35553555U};
    constexpr auto stage_pattern = std::uint32_t {0x2aaa2aaaU};
    const auto expect_u32_pattern = [](const auto& buffer, std::uint32_t pattern,
                                       std::string_view message) {
        const auto storage = copy_tensor_from_backend(buffer);
        expect(storage.bytes().size() % sizeof(std::uint32_t) == 0U,
            std::string(message) + " byte layout mismatch");
        for (std::size_t offset = 0U; offset < storage.bytes().size();
             offset += sizeof(std::uint32_t)) {
            std::uint32_t actual = 0U;
            std::memcpy(&actual, storage.bytes().data() + offset, sizeof(actual));
            if (actual != pattern) {
                throw TestFailure(std::string(message) + " was modified");
            }
        }
    };

    const auto run_case = [&](std::uint32_t initial_token_count, std::string_view label) {
        constexpr auto wrapped_token_count = std::uint32_t {5U};
        auto candidate = create_gemma2_metal_model_plan(model);
        auto scalar = create_gemma2_metal_model_plan(model);
        const auto embedding_scale = std::sqrt(
            static_cast<float>(candidate.config().embedding_length)
        );
        for (auto* plan : {&candidate, &scalar}) {
            for (const auto& cache : plan->kv_cache()) {
                fill_shared_metal_buffer_u32(cache.key, cache_pattern);
                fill_shared_metal_buffer_u32(cache.value, cache_pattern);
            }
        }
        const auto* stage_key_identity =
            &backend_buffer_storage(candidate.prefill_scratch().current_chunk_key);
        const auto* stage_value_identity =
            &backend_buffer_storage(candidate.prefill_scratch().current_chunk_value);
        fill_shared_metal_buffer_u32(candidate.prefill_scratch().current_chunk_key, stage_pattern);
        fill_shared_metal_buffer_u32(candidate.prefill_scratch().current_chunk_value, stage_pattern);

        const auto prefix = std::span<const std::uint32_t>(tokens).first(initial_token_count);
        const auto candidate_prefix = run_gemma2_metal_resident_prefill(
            candidate, prefix, 0U, embedding_scale
        );
        (void)run_gemma2_metal_resident_prefill(scalar, prefix, 0U, embedding_scale);
        expect(candidate_prefix.execution.chunk_count == 1U
                && candidate_prefix.execution.command_buffer_count == 1U
                && candidate_prefix.execution.synchronous_wait_count == 1U
                && candidate_prefix.execution.staged_kv_copy_dispatch_count == 0U
                && candidate_prefix.execution.staged_kv_copy_logical_bytes == 0U,
            std::string(label) + " pre-window path topology changed");
        expect_u32_pattern(candidate.prefill_scratch().current_chunk_key, stage_pattern,
            std::string(label) + " pre-window key stage");
        expect_u32_pattern(candidate.prefill_scratch().current_chunk_value, stage_pattern,
            std::string(label) + " pre-window value stage");

        const auto wrapped = std::span<const std::uint32_t>(tokens)
            .subspan(initial_token_count, wrapped_token_count);
        const auto candidate_result = run_gemma2_metal_resident_prefill(
            candidate, wrapped, initial_token_count, embedding_scale
        );
        const auto candidate_rows = copy_tensor_from_backend(
            candidate.prefill_scratch().feed_forward_residual
        );
        std::vector<std::vector<float>> scalar_rows;
        scalar_rows.reserve(wrapped_token_count);
        auto scalar_logits = std::optional<TensorStorage> {};
        for (std::uint32_t row = 0U; row < wrapped_token_count; ++row) {
            const auto result = run_gemma2_metal_resident_decode_step(
                scalar,
                create_gemma2_metal_decode_schedule(scalar, initial_token_count + row),
                wrapped[row],
                embedding_scale
            );
            scalar_rows.push_back(read_backend_f32_row(
                scalar.decode_scratch().feed_forward_residual, 0U
            ));
            scalar_logits = copy_tensor_from_backend(result.logits);
        }

        expect(candidate_result.execution.chunk_count == 1U
                && candidate_result.execution.command_buffer_count == 1U
                && candidate_result.execution.commit_count == 1U
                && candidate_result.execution.synchronous_wait_count == 1U
                && candidate_result.execution.compute_encoder_count == 1U,
            std::string(label) + " wrapped prefill must use one submit and wait");
        expect(candidate_result.execution.scheduled_layer_execution_count == 2U
                && candidate_result.execution.kv_write_count == 2U
                && candidate_result.execution.staged_kv_copy_dispatch_count == 1U,
            std::string(label) + " wrapped prefill layer/copy accounting mismatch");
        const auto expected_copy_bytes = static_cast<std::uint64_t>(wrapped_token_count)
            * candidate.config().attention_head_count_kv
            * (candidate.config().attention_key_length
                + candidate.config().attention_value_length)
            * sizeof(std::uint16_t) * 2U;
        expect(candidate_result.execution.staged_kv_copy_logical_bytes == expected_copy_bytes
                && candidate_result.execution.compute_dispatch_count
                    == candidate_prefix.execution.compute_dispatch_count + 1U
                && candidate_result.execution.blit_encoder_count == 0U
                && candidate_result.execution.transient_buffer_allocation_count == 0U,
            std::string(label) + " wrapped prefill dispatch/byte accounting mismatch");
        expect(&backend_buffer_storage(candidate.prefill_scratch().current_chunk_key)
                    == stage_key_identity
                && &backend_buffer_storage(candidate.prefill_scratch().current_chunk_value)
                    == stage_value_identity,
            std::string(label) + " wrapped prefill replaced its resident stage");

        for (std::uint32_t row = 0U; row < wrapped_token_count; ++row) {
            std::vector<float> candidate_row(candidate.config().embedding_length);
            for (std::uint32_t column = 0U;
                 column < candidate.config().embedding_length; ++column) {
                candidate_row[column] = read_f32(
                    candidate_rows,
                    std::array<std::uint64_t, 2> {row, column}
                );
            }
            expect_f32_rows_close(candidate_row, scalar_rows[row], 3.0e-2F,
                std::string(label) + " wrapped query row");
        }
        expect_f32_matrix_close(copy_tensor_from_backend(candidate_result.logits),
            *scalar_logits, 3.0e-2F, std::string(label) + " final logits");
        for (std::size_t layer = 0U; layer < candidate.kv_cache().size(); ++layer) {
            expect_f16_storage_close_absolute_relative(
                copy_tensor_from_backend(candidate.kv_cache()[layer].key),
                copy_tensor_from_backend(scalar.kv_cache()[layer].key),
                3.0e-2F, 2.0e-3F, std::string(label) + " key-cache parity"
            );
            expect_f16_storage_close_absolute_relative(
                copy_tensor_from_backend(candidate.kv_cache()[layer].value),
                copy_tensor_from_backend(scalar.kv_cache()[layer].value),
                3.0e-2F, 2.0e-3F, std::string(label) + " value-cache parity"
            );
        }
    };

    run_case(6U, "first-wrap crossing");
    run_case(8U, "start-after-wrap");
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

void test_gemma2_width_two_verifier_matches_sequential_decode_and_commits_transactionally_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;

    auto make_plan = []() {
        auto payload_store = nonzero_tiny_gemma2_payload_store(2U);
        auto view = tiny_model_view(
            tiny_gemma2_metadata(2U, 2U, 1U, 256U, 16U, 8U),
            std::move(payload_store.payloads)
        );
        const auto model = load_engine_model_from_gguf(view);
        return create_gemma2_metal_model_plan(model, Gemma2OutputHeadPolicy::q8_0);
    };
    auto candidate = make_plan();
    auto control = make_plan();
    const auto embedding_scale = std::sqrt(static_cast<float>(candidate.config().embedding_length));
    const std::array<std::uint32_t, 2> prefix {0U, 1U};
    (void)run_gemma2_metal_resident_prefill(candidate, prefix, 0U, embedding_scale);
    (void)run_gemma2_metal_resident_prefill(control, prefix, 0U, embedding_scale);

    const std::array<std::uint32_t, 2> proposed {2U, 3U};
    const auto verifier = create_gemma2_metal_verifier_schedule(candidate, 2U);
    const auto verified = run_gemma2_metal_width_two_verifier(
        candidate, verifier, proposed, embedding_scale
    );
    expect(candidate.has_pending_verification(), "width-two verifier transaction missing");
    expect(verified.execution.command_buffer_count == 1U
            && verified.execution.commit_count == 1U
            && verified.execution.synchronous_wait_count == 1U,
        "width-two verifier must use one submission and wait");
    expect(verified.execution.q4_small_batch_projection_count != 0U
            && verified.execution.q6_small_batch_projection_count != 0U,
        "width-two verifier did not select K-quant small-batch projections");
    expect(verified.execution.q4_width_two_projection_count
            == verified.execution.q4_small_batch_projection_count
            && verified.execution.q6_width_two_projection_count
                == verified.execution.q6_small_batch_projection_count,
        "width-two verifier K-quant family accounting mismatch");
    expect(verified.execution.q8_decode_projection_count == 1U,
        "width-two verifier Q8 output projection count mismatch");
    expect(verified.execution.q8_width_two_projection_count == 1U,
        "width-two verifier Q8 family accounting mismatch");

    const auto first = run_gemma2_metal_resident_decode_step(
        control, create_gemma2_metal_decode_schedule(control, 2U), proposed[0], embedding_scale
    );
    const auto first_logits = copy_tensor_from_backend(first.logits);
    const auto second = run_gemma2_metal_resident_decode_step(
        control, create_gemma2_metal_decode_schedule(control, 3U), proposed[1], embedding_scale
    );
    const auto second_logits = copy_tensor_from_backend(second.logits);
    const auto verifier_logits = copy_tensor_from_backend(verified.logits);
    for (std::uint64_t token = 0; token < candidate.config().vocab_size; ++token) {
        const std::array<std::uint64_t, 2> actual_first {0U, token};
        const std::array<std::uint64_t, 2> actual_second {1U, token};
        const std::array<std::uint64_t, 2> expected {0U, token};
        expect_close(read_f32(verifier_logits, actual_first), read_f32(first_logits, expected),
            2.0e-2F, "width-two first logit parity");
        expect_close(read_f32(verifier_logits, actual_second), read_f32(second_logits, expected),
            2.0e-2F, "width-two second logit parity");
    }

    try {
        (void)run_gemma2_metal_resident_decode_step(
            candidate,
            create_gemma2_metal_decode_schedule(candidate, 2U),
            proposed[0],
            embedding_scale
        );
        throw TestFailure("decode should reject a pending verifier transaction");
    } catch (const std::logic_error&) {
    }

    const auto committed = commit_gemma2_metal_width_two_verification(candidate, 2U);
    expect(committed.accepted_token_count == 2U && !candidate.has_pending_verification()
            && committed.execution.command_buffer_count == 0U,
        "width-two two-row zero-dispatch commit state mismatch");
    const auto expect_f16_close = [](const TensorStorage& actual, const TensorStorage& expected,
                                     std::string_view message) {
        expect(actual.descriptor() == expected.descriptor(), std::string(message) + " descriptor mismatch");
        for (std::size_t offset = 0; offset < actual.bytes().size(); offset += sizeof(std::uint16_t)) {
            std::uint16_t actual_bits = 0;
            std::uint16_t expected_bits = 0;
            std::memcpy(&actual_bits, actual.bytes().data() + offset, sizeof(actual_bits));
            std::memcpy(&expected_bits, expected.bytes().data() + offset, sizeof(expected_bits));
            expect_close(f16_bits_to_f32(actual_bits), f16_bits_to_f32(expected_bits), 2.0e-2F, message);
        }
    };
    for (std::size_t layer = 0; layer < candidate.kv_cache().size(); ++layer) {
        expect_f16_close(
            copy_tensor_from_backend(candidate.kv_cache()[layer].key),
            copy_tensor_from_backend(control.kv_cache()[layer].key),
            "width-two committed key parity"
        );
        expect_f16_close(
            copy_tensor_from_backend(candidate.kv_cache()[layer].value),
            copy_tensor_from_backend(control.kv_cache()[layer].value),
            "width-two committed value parity"
        );
    }

    const std::array<std::uint32_t, 2> partially_accepted_tokens {0U, 1U};
    (void)run_gemma2_metal_width_two_verifier(
        candidate, create_gemma2_metal_verifier_schedule(candidate, 7U),
        partially_accepted_tokens, embedding_scale
    );
    (void)run_gemma2_metal_resident_decode_step(
        control, create_gemma2_metal_decode_schedule(control, 7U),
        partially_accepted_tokens[0], embedding_scale
    );
    const auto partially_committed = commit_gemma2_metal_width_two_verification(candidate, 1U);
    expect(partially_committed.accepted_token_count == 1U && !candidate.has_pending_verification(),
        "width-two one-row commit state mismatch");
    for (std::size_t layer = 0; layer < candidate.kv_cache().size(); ++layer) {
        expect_f16_close(
            copy_tensor_from_backend(candidate.kv_cache()[layer].key),
            copy_tensor_from_backend(control.kv_cache()[layer].key),
            "width-two partially committed wrapped key parity"
        );
        expect_f16_close(
            copy_tensor_from_backend(candidate.kv_cache()[layer].value),
            copy_tensor_from_backend(control.kv_cache()[layer].value),
            "width-two partially committed wrapped value parity"
        );
    }

    const auto discard_schedule = create_gemma2_metal_verifier_schedule(candidate, 8U);
    const std::array<std::uint32_t, 2> discarded_tokens {0U, 1U};
    (void)run_gemma2_metal_width_two_verifier(
        candidate, discard_schedule, discarded_tokens, embedding_scale
    );
    try {
        (void)run_gemma2_metal_resident_prefill(
            candidate, std::span<const std::uint32_t>(discarded_tokens).first(1U),
            8U, embedding_scale
        );
        throw TestFailure("prefill should reject a pending verifier transaction");
    } catch (const std::logic_error&) {
    }
    try {
        (void)commit_gemma2_metal_width_two_verification(candidate, 3U);
        throw TestFailure("width-two commit should reject more than two rows");
    } catch (const std::invalid_argument&) {
    }
    expect(candidate.has_pending_verification(),
        "invalid width-two commit must preserve pending transaction");
    const auto discarded = commit_gemma2_metal_width_two_verification(candidate, 0U);
    expect(discarded.accepted_token_count == 0U && !candidate.has_pending_verification(),
        "width-two zero-row discard state mismatch");
    discard_gemma2_metal_width_two_verification(candidate);

    const std::array<std::uint32_t, 2> provisional_tokens {1U, 0U};
    (void)run_gemma2_metal_width_two_verifier(
        candidate, create_gemma2_metal_verifier_schedule(candidate, 8U),
        provisional_tokens, embedding_scale
    );
    const auto provisional = provisionally_commit_gemma2_metal_width_two_verification(candidate, 2U);
    expect(provisional.accepted_token_count == 2U && candidate.has_pending_verification()
            && candidate.provisionally_committed_token_count() == 2U,
        "provisional width-two commit state mismatch");
    (void)run_gemma2_metal_resident_decode_step(
        control, create_gemma2_metal_decode_schedule(control, 8U),
        provisional_tokens[0], embedding_scale
    );
    const auto finalized = finalize_gemma2_metal_width_two_verification(candidate, 1U);
    expect(finalized.accepted_token_count == 1U && !candidate.has_pending_verification(),
        "provisional width-two suffix rollback state mismatch");
    for (std::size_t layer = 0; layer < candidate.kv_cache().size(); ++layer) {
        expect_f16_close(copy_tensor_from_backend(candidate.kv_cache()[layer].key),
            copy_tensor_from_backend(control.kv_cache()[layer].key),
            "provisional width-two key rollback parity");
        expect_f16_close(copy_tensor_from_backend(candidate.kv_cache()[layer].value),
            copy_tensor_from_backend(control.kv_cache()[layer].value),
            "provisional width-two value rollback parity");
    }
}

void test_gemma2_speculative_controller_matches_seeded_baseline_and_rolls_back_callback_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto make_plan = []() {
        auto payload_store = nonzero_tiny_gemma2_payload_store(2U);
        auto view = tiny_model_view(
            tiny_gemma2_metadata(2U, 2U, 1U, 256U, 16U, 8U),
            std::move(payload_store.payloads)
        );
        return create_gemma2_metal_model_plan(
            load_engine_model_from_gguf(view), Gemma2OutputHeadPolicy::q8_0
        );
    };
    const std::array<std::uint32_t, 2> prompt {0U, 1U};
    const auto options = TokenSamplerOptions {
        .temperature = 0.8F, .top_k = 2U, .top_p = 0.95F, .min_p = 0.0F,
        .repetition_penalty = 1.0F, .seed = 42U,
    };
    auto baseline_plan = make_plan();
    TokenSampler baseline_sampler(options);
    const auto baseline = generate_gemma2_metal_sampled_tokens_with_plan(
        baseline_plan, prompt, 0U, 4U, {}, baseline_sampler
    );
    expect(baseline.selected_token_ids.size() == 4U,
        "speculative baseline token count mismatch");

    const auto first = baseline.selected_token_ids[0];
    const auto second = baseline.selected_token_ids[1];
    const std::array<std::uint32_t, 7> proposal_history {
        0U, first, second, 0U, first, second, 0U,
    };
    RetrievalProposer proposer;
    proposer.rebuild(proposal_history);
    auto candidate_plan = make_plan();
    TokenSampler candidate_sampler(options);
    const auto candidate = generate_gemma2_metal_sampled_tokens_with_plan(
        candidate_plan, prompt, 0U, 4U, {}, candidate_sampler, {}, {}, &proposer
    );
    expect(candidate.selected_token_ids == baseline.selected_token_ids,
        "speculative seeded token sequence mismatch");
    expect(candidate.metrics.selected_token_hash == baseline.metrics.selected_token_hash
            && candidate.metrics.selected_token_hash != 0U,
        "speculative selected-token hash mismatch");
    expect(candidate.metrics.speculative_cycle_count >= 1U
            && candidate.metrics.speculative_second_token_match_count >= 1U,
        "speculative controller did not exercise an accepted pair");
    std::uint32_t bucket_cycles = 0U;
    std::uint32_t bucket_matches = 0U;
    for (const auto& bucket : candidate.metrics.speculative_proposal_buckets) {
        bucket_cycles += bucket.cycle_count;
        bucket_matches += bucket.match_count;
    }
    expect(bucket_cycles == candidate.metrics.speculative_cycle_count
            && bucket_matches == candidate.metrics.speculative_second_token_match_count,
        "speculative proposal bucket totals mismatch");

    RetrievalProposer cancelling_proposer;
    cancelling_proposer.rebuild(proposal_history);
    auto cancelling_plan = make_plan();
    TokenSampler cancelling_sampler(options);
    const auto cancelled = generate_gemma2_metal_sampled_tokens_with_plan(
        cancelling_plan, prompt, 0U, 4U, {}, cancelling_sampler,
        [](const local_gguf_engine::engine::Gemma2MetalGeneratedTokenEvent&) {
            return local_gguf_engine::engine::Gemma2MetalTokenStreamDecision::stop_generation;
        }, {}, &cancelling_proposer
    );
    expect(cancelled.selected_token_ids.size() == 1U
            && cancelled.selected_token_ids[0] == first,
        "speculative callback cancellation retained the preview token");
    expect(cancelled.cached_token_count == prompt.size() + 1U
            && cancelled.metrics.speculative_rollback_count == 1U,
        "speculative callback cancellation did not roll back row two");

    auto one_token_plan = make_plan();
    TokenSampler one_token_sampler(options);
    const auto one_token = generate_gemma2_metal_sampled_tokens_with_plan(
        one_token_plan, prompt, 0U, 1U, {}, one_token_sampler
    );
    expect(one_token.selected_token_ids == cancelled.selected_token_ids,
        "speculative cancellation differs from one-token baseline");
    for (std::size_t layer = 0; layer < cancelling_plan.kv_cache().size(); ++layer) {
        const auto cancelling_key = copy_tensor_from_backend(cancelling_plan.kv_cache()[layer].key);
        const auto baseline_key = copy_tensor_from_backend(one_token_plan.kv_cache()[layer].key);
        const auto cancelling_value = copy_tensor_from_backend(cancelling_plan.kv_cache()[layer].value);
        const auto baseline_value = copy_tensor_from_backend(one_token_plan.kv_cache()[layer].value);
        expect(std::equal(cancelling_key.bytes().begin(), cancelling_key.bytes().end(),
                          baseline_key.bytes().begin()),
            "speculative callback key cache mismatch");
        expect(std::equal(cancelling_value.bytes().begin(), cancelling_value.bytes().end(),
                          baseline_value.bytes().begin()),
            "speculative callback value cache mismatch");
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

    for (const auto cached_token_count : {0U, 15U, 16U}) {
        try {
            (void)create_gemma2_metal_verifier_schedule(plan, cached_token_count);
            throw TestFailure("Gemma2 Metal verifier schedule should reject invalid cache range");
        } catch (const std::invalid_argument& error) {
            expect(
                std::string_view(error.what())
                    == "Gemma2 Metal verifier schedule requires room for two tokens",
                "Gemma2 Metal verifier cache-range error mismatch"
            );
        }
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

void test_gemma2_compact_metal_candidates_match_cpu_sort_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }
    constexpr float softcap = 30.0F;
    for (const std::uint32_t vocab : {1U, 31U, 33U, 511U, 512U, 513U, 4097U, 256000U}) {
        std::vector<float> logits(vocab);
        for (std::uint32_t token = 0; token < vocab; ++token) {
            logits[token] = -static_cast<float>(token % 97U);
        }
        const auto top_k = std::min<std::uint32_t>(vocab, vocab > 128U ? 40U : 1U);
        const auto history_count = vocab > 64U ? 3U : 0U;
        TokenSelectionRequest request {
            .top_k = top_k,
            .base_candidate_count = top_k + history_count,
            .history_token_count = history_count,
        };
        if (history_count != 0U) {
            request.history_token_ids[0] = vocab - 1U;
            request.history_token_ids[1] = vocab / 2U;
            request.history_token_ids[2] = 0U;
        }

        const auto tile_count = (vocab + 511U) / 512U;
        const auto allocate_f32 = [](std::uint64_t elements) {
            return allocate_backend_buffer(
                TensorDescriptor::from(Shape::from({elements}), DType::f32), BackendKind::metal
            );
        };
        Gemma2MetalSelectionWorkspace workspace {
            .candidate_runs_a = allocate_f32(static_cast<std::uint64_t>(tile_count) * 320U * 2U),
            .candidate_runs_b = allocate_f32(static_cast<std::uint64_t>(tile_count) * 320U * 2U),
            .compact_result = allocate_f32(2U + 384U * 2U),
            .history_token_staging = allocate_f32(64U),
        };
        const auto input = copy_tensor_to_backend(
            tensor_f32_vector(std::span<const float>(logits)), BackendKind::metal
        );
        MetalCommandContext command("compact candidate parity");
        const auto dispatches = command.encode_token_candidates(input, workspace, request, softcap);
        const auto stats = command.commit_and_wait();
        const auto actual = command.read_token_candidates(workspace);

        std::vector<TokenLogit> expected;
        expected.reserve(vocab);
        for (std::uint32_t token = 0; token < vocab; ++token) {
            expected.push_back(TokenLogit {
                .token = token,
                .logit = local_gguf_engine::engine::gemma2_softcap_value(logits[token], softcap),
            });
        }
        std::sort(expected.begin(), expected.end(), [](const TokenLogit& left, const TokenLogit& right) {
            return left.logit > right.logit || (left.logit == right.logit && left.token < right.token);
        });
        const auto retained = std::min<std::uint32_t>(vocab, request.base_candidate_count);
        expect(actual.count == retained + history_count, "compact candidate count mismatch");
        expect(actual.readback_bytes == actual.count * 8U, "compact candidate byte count mismatch");
        expect(stats.command_buffer_count == 1U && stats.commit_count == 1U
            && stats.synchronous_wait_count == 1U, "compact candidates must use one submission and wait");
        expect(stats.transient_buffer_allocation_count == 0U, "compact candidates allocated transient buffers");
        expect(dispatches == stats.compute_dispatch_count, "compact selector dispatch accounting mismatch");
        for (std::uint32_t index = 0; index < retained; ++index) {
            expect(
                actual.values[index].token == expected[index].token,
                "compact candidate token mismatch at vocab " + std::to_string(vocab)
                    + " index " + std::to_string(index)
                    + ": expected " + std::to_string(expected[index].token)
                    + ", got " + std::to_string(actual.values[index].token)
            );
            expect_close(actual.values[index].logit, expected[index].logit, 2.0e-5F, "compact candidate logit mismatch");
        }
        for (std::uint32_t index = 0; index < history_count; ++index) {
            const auto token = request.history_token_ids[index];
            expect(actual.values[retained + index].token == token, "compact history token mismatch");
            expect_close(
                actual.values[retained + index].logit,
                local_gguf_engine::engine::gemma2_softcap_value(logits[token], softcap),
                2.0e-5F,
                "compact history logit mismatch"
            );
        }
    }
}

void test_gemma2_width_two_compact_candidates_match_independent_rows_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    constexpr std::uint32_t vocab = 1025U;
    constexpr float softcap = 30.0F;
    std::vector<float> logits(2U * vocab);
    for (std::uint32_t token = 0; token < vocab; ++token) {
        logits[token] = -static_cast<float>((token * 7U) % 101U);
        logits[vocab + token] = -static_cast<float>((token * 11U + 3U) % 103U);
    }
    std::array<TokenSelectionRequest, 2> requests {
        TokenSelectionRequest {.top_k = 4U, .base_candidate_count = 5U, .history_token_ids = {1024U}, .history_token_count = 1U},
        TokenSelectionRequest {.top_k = 3U, .base_candidate_count = 5U, .history_token_ids = {1000U, 999U}, .history_token_count = 2U},
    };
    const auto allocate_f32 = [](std::uint64_t elements) {
        return allocate_backend_buffer(
            TensorDescriptor::from(Shape::from({elements}), DType::f32), BackendKind::metal
        );
    };
    const auto tile_count = (vocab + 511U) / 512U;
    Gemma2MetalSelectionWorkspace workspace {
        .candidate_runs_a = allocate_f32(static_cast<std::uint64_t>(2U * tile_count) * 320U * 2U),
        .candidate_runs_b = allocate_f32(static_cast<std::uint64_t>(2U * tile_count) * 320U * 2U),
        .compact_result = allocate_f32(2U * (2U + 384U * 2U)),
        .history_token_staging = allocate_f32(2U * 64U),
    };
    const auto input = copy_tensor_to_backend(
        tensor_f32_matrix(2U, vocab, logits), BackendKind::metal
    );
    MetalCommandContext command("width-two compact candidate parity");
    const auto dispatches = command.encode_token_candidates_width_two(
        input, workspace, requests, softcap
    );
    const auto stats = command.commit_and_wait();
    const auto actual = command.read_token_candidates_width_two(workspace);
    expect(dispatches == stats.compute_dispatch_count,
        "width-two compact selector dispatch accounting mismatch");
    for (std::size_t row = 0; row < 2U; ++row) {
        std::vector<TokenLogit> expected;
        expected.reserve(vocab);
        for (std::uint32_t token = 0; token < vocab; ++token) {
            expected.push_back(TokenLogit {
                .token = token,
                .logit = local_gguf_engine::engine::gemma2_softcap_value(
                    logits[row * vocab + token], softcap
                ),
            });
        }
        std::sort(expected.begin(), expected.end(), [](const TokenLogit& left, const TokenLogit& right) {
            return left.logit > right.logit || (left.logit == right.logit && left.token < right.token);
        });
        expect(actual[row].count
                == requests[row].base_candidate_count + requests[row].history_token_count,
            "width-two compact candidate count mismatch");
        for (std::uint32_t index = 0; index < requests[row].top_k; ++index) {
            expect(actual[row].values[index].token == expected[index].token,
                "width-two compact candidate token mismatch");
            expect_close(actual[row].values[index].logit, expected[index].logit, 2.0e-5F,
                "width-two compact candidate logit mismatch");
        }
        for (std::uint32_t index = 0; index < requests[row].history_token_count; ++index) {
            const auto token = requests[row].history_token_ids[index];
            const auto result_index = requests[row].base_candidate_count + index;
            expect(actual[row].values[result_index].token == token,
                "width-two compact history token mismatch");
            expect_close(actual[row].values[result_index].logit,
                local_gguf_engine::engine::gemma2_softcap_value(logits[row * vocab + token], softcap),
                2.0e-5F, "width-two compact history logit mismatch");
        }
    }
}

void test_gemma2_sequential_decode_trace_matches_each_metal_row_and_kv_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    auto payload_store = nonzero_tiny_gemma2_payload_store(2U);
    auto view = tiny_model_view(
        tiny_gemma2_metadata(2U, 2U, 1U, 256U, 16U, 8U),
        std::move(payload_store.payloads)
    );
    const auto model = load_engine_model_from_gguf(view);
    const auto weights = bind_gemma2_model_weights(model);
    const auto& config = model.gemma2_config();
    auto plan = create_gemma2_metal_model_plan(model);
    constexpr std::uint32_t initial_position = 7U;
    std::vector<TensorStorage> keys;
    std::vector<TensorStorage> values;
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        keys.push_back(TensorStorage::allocate(TensorDescriptor::from(
            Shape::from({config.context_length, config.attention_head_count_kv * config.attention_key_length}),
            DType::f32
        )));
        values.push_back(TensorStorage::allocate(TensorDescriptor::from(
            Shape::from({config.context_length, config.attention_head_count_kv * config.attention_value_length}),
            DType::f32
        )));
    }
    const std::array<std::uint32_t, 3> tokens {1U, 2U, 3U};
    const auto trace = reference_decode_trace(
        config, weights, tokens, initial_position, keys, values
    );
    expect(trace.steps.size() == tokens.size() && trace.final_cache_position == 10U,
        "sequential reference trace positions mismatch");

    std::uint64_t total_attention_tokens = 0U;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto cache_position = initial_position + static_cast<std::uint32_t>(index);
        const auto schedule = create_gemma2_metal_decode_schedule(plan, cache_position);
        const auto result = run_gemma2_metal_resident_decode_step(
            plan,
            schedule,
            tokens[index],
            std::sqrt(static_cast<float>(config.embedding_length)),
            TokenSelectionRequest {.top_k = 4U, .base_candidate_count = 4U}
        );
        const auto actual_logits = copy_tensor_from_backend(result.logits);
        expect_f32_matrix_close(
            softcap_f32(actual_logits, config.final_logit_softcap),
            trace.steps[index].logits,
            3.0e-3F,
            "sequential decode per-row logits"
        );
        expect(result.candidates.has_value() && result.candidates->count == 4U,
            "sequential decode candidate set mismatch");
        for (std::uint32_t candidate = 0; candidate < 4U; ++candidate) {
            expect(result.candidates->values[candidate].token
                    == trace.steps[index].candidates[candidate].token,
                "sequential decode candidate token mismatch");
            expect_close(
                result.candidates->values[candidate].logit,
                trace.steps[index].candidates[candidate].logit,
                3.0e-3F,
                "sequential decode candidate logit mismatch"
            );
        }
        std::uint64_t expected_attention_tokens = 0U;
        for (const auto& layer : schedule.layers()) expected_attention_tokens += layer.attention_token_count;
        expect(result.execution.decode_attention_token_count == expected_attention_tokens,
            "sequential decode attention-token accounting mismatch");
        total_attention_tokens += result.execution.decode_attention_token_count;
        for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
            const auto row_end = cache_position + 1U;
            const auto first_row = plan.layers()[layer].uses_sliding_window
                ? row_end - std::min(row_end, config.sliding_window) : 0U;
            expect_f16_ring_matches_f32(
                copy_tensor_from_backend(plan.kv_cache()[layer].key),
                trace.steps[index].key_cache[layer],
                first_row,
                row_end,
                plan.kv_cache()[layer].row_mask,
                3.0e-3F,
                "sequential decode key transition"
            );
            expect_f16_ring_matches_f32(
                copy_tensor_from_backend(plan.kv_cache()[layer].value),
                trace.steps[index].value_cache[layer],
                first_row,
                row_end,
                plan.kv_cache()[layer].row_mask,
                3.0e-3F,
                "sequential decode value transition"
            );
        }
    }
    expect(total_attention_tokens == 51U,
        "sequential trace must cross the sliding-window boundary while global attention grows");
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
        "Gemma2 Q8 output policy preserves F16 embedding when available",
        test_gemma2_q8_output_policy_preserves_f16_embedding_when_available,
    },
    {
        "creates Gemma2 Metal schedules when available",
        test_creates_gemma2_metal_schedules_when_available,
    },
    {
        "Gemma2 Metal schedule preserves target 4K/8K populated bounds when available",
        test_gemma2_metal_schedule_preserves_target_4k_8k_populated_bounds_when_available,
    },
    {
        "Gemma2 Metal warm-up is state neutral and covers production routes when available",
        test_gemma2_metal_warmup_is_state_neutral_and_covers_production_routes_when_available,
    },
    {
        "Gemma2 F16 KV store rounds values and wraps when available",
        test_gemma2_f16_kv_store_rounds_values_and_wraps_when_available,
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
        "Gemma2 Metal resident prefill stages wrapped sliding chunks when available",
        test_gemma2_metal_resident_prefill_stages_wrapped_sliding_chunks_when_available,
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
        "Gemma2 width-two verifier matches sequential decode and commits transactionally when available",
        test_gemma2_width_two_verifier_matches_sequential_decode_and_commits_transactionally_when_available,
    },
    {
        "Gemma2 speculative controller matches seeded baseline and rolls back callback when available",
        test_gemma2_speculative_controller_matches_seeded_baseline_and_rolls_back_callback_when_available,
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
        "Gemma2 compact Metal candidates match CPU sort when available",
        test_gemma2_compact_metal_candidates_match_cpu_sort_when_available,
    },
    {
        "Gemma2 width-two compact Metal candidates match independent rows when available",
        test_gemma2_width_two_compact_candidates_match_independent_rows_when_available,
    },
    {
        "Gemma2 sequential decode trace matches each Metal row and KV when available",
        test_gemma2_sequential_decode_trace_matches_each_metal_row_and_kv_when_available,
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
