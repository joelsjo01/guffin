#include "gemma2_metal_plan_tests.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
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
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include "tensor_test_support.hpp"

#include <algorithm>
#include <array>
#include <bit>
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
using local_gguf_engine::tensor::ops::add_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_scores_f32;
using local_gguf_engine::tensor::ops::causal_grouped_attention_values_f32;
using local_gguf_engine::tensor::ops::matmul_f32_quantized_gguf_weight;
using local_gguf_engine::tensor::ops::rms_norm_f32;
using local_gguf_engine::tensor::ops::rope_f32_by_heads_with_position_start;
using local_gguf_engine::tensor::ops::softmax_f32;
using local_gguf_engine::tensor::Q4_KBlockView;
using local_gguf_engine::tensor::Q6_KBlockView;
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

void write_little_endian_u16(std::span<std::uint8_t> bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[offset + 1U] = static_cast<std::uint8_t>(value >> 8U);
}

void set_q4_k_scale_min(
    std::span<std::uint8_t> block_bytes,
    std::size_t group_index,
    std::uint8_t scale,
    std::uint8_t min
) {
    auto scale_min = block_bytes.subspan(4, Q4_KBlockView::scale_min_byte_count);
    scale = static_cast<std::uint8_t>(scale & 0x3FU);
    min = static_cast<std::uint8_t>(min & 0x3FU);

    if (group_index < 4U) {
        scale_min[group_index] = static_cast<std::uint8_t>((scale_min[group_index] & 0xC0U) | scale);
        scale_min[group_index + 4U] = static_cast<std::uint8_t>((scale_min[group_index + 4U] & 0xC0U) | min);
        return;
    }

    const auto scale_high_bits = static_cast<std::uint8_t>((scale & 0x30U) << 2U);
    const auto min_high_bits = static_cast<std::uint8_t>((min & 0x30U) << 2U);
    scale_min[group_index - 4U] =
        static_cast<std::uint8_t>((scale_min[group_index - 4U] & 0x3FU) | scale_high_bits);
    scale_min[group_index] = static_cast<std::uint8_t>((scale_min[group_index] & 0x3FU) | min_high_bits);
    scale_min[group_index + 4U] = static_cast<std::uint8_t>((scale & 0x0FU) | ((min & 0x0FU) << 4U));
}

void set_q4_k_quantized_nibble(
    std::span<std::uint8_t> block_bytes,
    std::size_t element_index,
    std::uint8_t value
) {
    auto quantized = block_bytes.subspan(16, Q4_KBlockView::quantized_byte_count);
    const auto super_group_base = (element_index / 64U) * 32U;
    const auto index_in_super_group = element_index % 64U;
    const auto byte_index = super_group_base + (index_in_super_group % 32U);
    const auto shift = index_in_super_group < 32U ? 0U : 4U;
    const auto mask = static_cast<std::uint8_t>(0x0FU << shift);
    quantized[byte_index] = static_cast<std::uint8_t>(
        (quantized[byte_index] & ~mask) | ((value & 0x0FU) << shift)
    );
}

void set_q6_k_scale(std::span<std::uint8_t> block_bytes, std::size_t group_index, std::int8_t value) {
    block_bytes[Q6_KBlockView::low_bit_byte_count + Q6_KBlockView::high_bit_byte_count + group_index] =
        std::bit_cast<std::uint8_t>(value);
}

void set_q6_k_quantized_value(
    std::span<std::uint8_t> block_bytes,
    std::size_t element_index,
    std::int8_t value
) {
    const auto encoded = static_cast<std::uint8_t>(static_cast<std::int16_t>(value) + 32);
    const auto low = static_cast<std::uint8_t>(encoded & 0x0FU);
    const auto high = static_cast<std::uint8_t>((encoded >> 4U) & 0x03U);
    const auto half_block_index = element_index / 128U;
    const auto index_in_half_block = element_index % 128U;
    const auto lane_index = index_in_half_block % 32U;

    const auto low_base = half_block_index * 64U;
    const auto low_plane_offset = ((index_in_half_block / 32U) % 2U) * 32U;
    const auto low_shift = index_in_half_block < 64U ? 0U : 4U;
    const auto low_mask = static_cast<std::uint8_t>(0x0FU << low_shift);
    auto& low_byte = block_bytes[low_base + low_plane_offset + lane_index];
    low_byte = static_cast<std::uint8_t>((low_byte & ~low_mask) | (low << low_shift));

    const auto high_base = Q6_KBlockView::low_bit_byte_count + half_block_index * 32U;
    const auto high_shift = (index_in_half_block / 32U) * 2U;
    const auto high_mask = static_cast<std::uint8_t>(0x03U << high_shift);
    auto& high_byte = block_bytes[high_base + lane_index];
    high_byte = static_cast<std::uint8_t>((high_byte & ~high_mask) | (high << high_shift));
}

local_gguf_engine::tensor::TensorStorage tensor_q4_k_gguf_weight_matrix(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint64_t seed
) {
    auto storage = local_gguf_engine::tensor::TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({shared, columns}), DType::q4_k)
    );
    auto bytes = storage.mutable_bytes();
    const auto blocks_per_column = shared / Q4_KBlockView::element_count;

    for (std::uint64_t column = 0; column < columns; ++column) {
        for (std::uint64_t column_block = 0; column_block < blocks_per_column; ++column_block) {
            const auto block_index = column * blocks_per_column + column_block;
            auto block = bytes.subspan(
                static_cast<std::size_t>(block_index * Q4_KBlockView::byte_count),
                Q4_KBlockView::byte_count
            );
            write_little_endian_u16(block, 0, 0x3C00U);
            write_little_endian_u16(block, 2, 0x0000U);

            for (std::size_t group = 0; group < 8U; ++group) {
                const auto scale = static_cast<std::uint8_t>(1U + ((seed + column + column_block + group) % 5U));
                set_q4_k_scale_min(block, group, scale, 0);
            }
            for (std::size_t element = 0; element < Q4_KBlockView::element_count; ++element) {
                const auto nibble = static_cast<std::uint8_t>(
                    (seed + column + (3U * column_block) + (5U * element)) % 16U
                );
                set_q4_k_quantized_nibble(block, element, nibble);
            }
        }
    }

    return storage;
}

local_gguf_engine::tensor::TensorStorage tensor_q6_k_gguf_weight_matrix(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint64_t seed
) {
    auto storage = local_gguf_engine::tensor::TensorStorage::allocate(
        TensorDescriptor::from(Shape::from({shared, columns}), DType::q6_k)
    );
    auto bytes = storage.mutable_bytes();
    const auto blocks_per_column = shared / Q6_KBlockView::element_count;

    for (std::uint64_t column = 0; column < columns; ++column) {
        for (std::uint64_t column_block = 0; column_block < blocks_per_column; ++column_block) {
            const auto block_index = column * blocks_per_column + column_block;
            auto block = bytes.subspan(
                static_cast<std::size_t>(block_index * Q6_KBlockView::byte_count),
                Q6_KBlockView::byte_count
            );
            write_little_endian_u16(block, Q6_KBlockView::byte_count - 2U, 0x3800U);

            for (std::size_t group = 0; group < Q6_KBlockView::scale_byte_count; ++group) {
                const auto raw = static_cast<int>((seed + column + column_block + group) % 7U) - 3;
                const auto scale = static_cast<std::int8_t>(raw == 0 ? 2 : raw);
                set_q6_k_scale(block, group, scale);
            }
            for (std::size_t element = 0; element < Q6_KBlockView::element_count; ++element) {
                const auto quantized = static_cast<std::int8_t>(
                    static_cast<int>((seed + column + (5U * column_block) + (7U * element)) % 31U) - 15
                );
                set_q6_k_quantized_value(block, element, quantized);
            }
        }
    }

    return storage;
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
        + scratch.attention_probabilities.byte_count
        + scratch.attention_output.byte_count
        + scratch.projected_attention_output.byte_count
        + scratch.post_attention_residual.byte_count
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
    expect(
        plan.decode_scratch().projected_attention_output.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal projected attention output scratch shape mismatch"
    );
    expect(
        plan.decode_scratch().post_attention_residual.descriptor().shape == Shape::from({1, 256}),
        "Gemma2 Metal post-attention residual scratch shape mismatch"
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

void test_gemma2_metal_runtime_shell_projects_first_layer_attention_when_available() {
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
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> attention_norm_values(256U, 1.0F);
    attention_norm_values[1] = 0.5F;
    const auto attention_norm = tensor_f32_vector(std::span<const float>(attention_norm_values));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_norm.weight"), attention_norm);

    const auto query_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 1);
    const auto key_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 2);
    const auto value_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 3);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_q.weight"), query_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_k.weight"), key_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_v.weight"), value_weight);

    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};

    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );
    const auto& projection = shell.prefill_first_layer_attention_projection;
    const auto normalized_attention_input =
        copy_tensor_from_backend(shell.prefill_first_layer_input.normalized_attention_input);
    const auto query = copy_tensor_from_backend(projection.query);
    const auto key = copy_tensor_from_backend(projection.key);
    const auto value = copy_tensor_from_backend(projection.value);
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

    expect(projection.layer_index == 0, "Gemma2 shell first layer projection index mismatch");
    expect(
        projection.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer projection phase mismatch"
    );
    expect(
        projection.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer projection scratch ownership mismatch"
    );
    expect(projection.query.backend() == BackendKind::metal, "Gemma2 shell query projection backend mismatch");
    expect(projection.key.backend() == BackendKind::metal, "Gemma2 shell key projection backend mismatch");
    expect(projection.value.backend() == BackendKind::metal, "Gemma2 shell value projection backend mismatch");
    expect(
        projection.query.descriptor() == shell.prefill.scratch().query,
        "Gemma2 shell query projection scheduled descriptor mismatch"
    );
    expect(
        projection.key.descriptor() == shell.prefill.scratch().key,
        "Gemma2 shell key projection scheduled descriptor mismatch"
    );
    expect(
        projection.value.descriptor() == shell.prefill.scratch().value,
        "Gemma2 shell value projection scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(projection.query, shell.prefill_first_layer_input.normalized_attention_input),
        "Gemma2 shell query projection should be an explicit output buffer"
    );
    expect(
        !shares_backend_storage(projection.key, shell.prefill_first_layer_input.normalized_attention_input),
        "Gemma2 shell key projection should be an explicit output buffer"
    );
    expect(
        !shares_backend_storage(projection.value, shell.prefill_first_layer_input.normalized_attention_input),
        "Gemma2 shell value projection should be an explicit output buffer"
    );
    expect(
        !shares_backend_storage(projection.query, shell.plan.weights().blocks[0].attention.query_weight),
        "Gemma2 shell query projection should not alias the query weight"
    );
    expect(
        !shares_backend_storage(projection.key, shell.plan.weights().blocks[0].attention.key_weight),
        "Gemma2 shell key projection should not alias the key weight"
    );
    expect(
        !shares_backend_storage(projection.value, shell.plan.weights().blocks[0].attention.value_weight),
        "Gemma2 shell value projection should not alias the value weight"
    );

    const std::array<std::uint64_t, 2> first_projection_value {0, 0};
    expect(
        read_f32(expected_query, std::span<const std::uint64_t>(first_projection_value)) != 0.0F,
        "Gemma2 shell query projection fixture should be nonzero"
    );
    expect_f32_matrix_close(query, expected_query, 1.0e-2F, "Gemma2 shell query projection");
    expect_f32_matrix_close(key, expected_key, 1.0e-2F, "Gemma2 shell key projection");
    expect_f32_matrix_close(value, expected_value, 1.0e-2F, "Gemma2 shell value projection");
}

void test_gemma2_metal_runtime_shell_positions_first_layer_attention_when_available() {
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
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> attention_norm_values(256U, 1.0F);
    attention_norm_values[1] = 0.5F;
    const auto attention_norm = tensor_f32_vector(std::span<const float>(attention_norm_values));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_norm.weight"), attention_norm);

    const auto query_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 1);
    const auto key_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 2);
    const auto value_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 3);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_q.weight"), query_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_k.weight"), key_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_v.weight"), value_weight);

    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};

    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );
    const auto& projection = shell.prefill_first_layer_attention_projection;
    const auto& position = shell.prefill_first_layer_attention_position;
    const auto projection_query = copy_tensor_from_backend(projection.query);
    const auto projection_key = copy_tensor_from_backend(projection.key);
    const auto positioned_query = copy_tensor_from_backend(position.query);
    const auto positioned_key = copy_tensor_from_backend(position.key);
    const auto expected_query = rope_f32_by_heads_with_position_start(
        projection_query,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_key_length,
        shell.plan.config().attention_key_length,
        10000.0F
    );
    const auto expected_key = rope_f32_by_heads_with_position_start(
        projection_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.plan.config().attention_key_length,
        10000.0F
    );

    expect(position.layer_index == 0, "Gemma2 shell first layer attention position index mismatch");
    expect(
        position.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer attention position phase mismatch"
    );
    expect(
        position.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer attention position scratch ownership mismatch"
    );
    expect(
        position.position_start == shell.prefill.cache_position(),
        "Gemma2 shell attention position start mismatch"
    );
    expect(
        position.query_head_count == shell.plan.config().attention_head_count,
        "Gemma2 shell attention position query head count mismatch"
    );
    expect(
        position.key_head_count == shell.plan.config().attention_head_count_kv,
        "Gemma2 shell attention position key head count mismatch"
    );
    expect(
        position.head_dimension == shell.plan.config().attention_key_length,
        "Gemma2 shell attention position head dimension mismatch"
    );
    expect(
        position.rope_dimension_count == shell.plan.config().attention_key_length,
        "Gemma2 shell attention position RoPE dimension mismatch"
    );
    expect_close(
        position.rope_frequency_base,
        10000.0F,
        0.0F,
        "Gemma2 shell attention position RoPE frequency base mismatch"
    );
    expect(position.query.backend() == BackendKind::metal, "Gemma2 shell positioned query backend mismatch");
    expect(position.key.backend() == BackendKind::metal, "Gemma2 shell positioned key backend mismatch");
    expect(
        position.query.descriptor() == shell.prefill.scratch().query,
        "Gemma2 shell positioned query scheduled descriptor mismatch"
    );
    expect(
        position.key.descriptor() == shell.prefill.scratch().key,
        "Gemma2 shell positioned key scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(position.query, projection.query),
        "Gemma2 shell positioned query should be an explicit output buffer"
    );
    expect(
        !shares_backend_storage(position.key, projection.key),
        "Gemma2 shell positioned key should be an explicit output buffer"
    );
    expect(
        !shares_backend_storage(position.query, position.key),
        "Gemma2 shell positioned query and key should be distinct buffers"
    );

    constexpr float metal_rope_shell_tolerance = 2.0e-3F;
    expect_f32_matrix_close(positioned_query, expected_query, metal_rope_shell_tolerance, "Gemma2 shell positioned query");
    expect_f32_matrix_close(positioned_key, expected_key, metal_rope_shell_tolerance, "Gemma2 shell positioned key");
}

void test_gemma2_metal_runtime_shell_applies_first_layer_post_attention_residual_when_available() {
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
    overwrite_payload_bytes(payload_store, "token_embd.weight", embedding);

    std::vector<float> attention_norm_values(256U, 1.0F);
    attention_norm_values[1] = 0.5F;
    const auto attention_norm = tensor_f32_vector(std::span<const float>(attention_norm_values));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_norm.weight"), attention_norm);

    std::vector<float> post_attention_norm_values(256U, 1.0F);
    post_attention_norm_values[0] = 0.25F;
    post_attention_norm_values[1] = 0.75F;
    const auto post_attention_norm = tensor_f32_vector(std::span<const float>(post_attention_norm_values));
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "post_attention_norm.weight"), post_attention_norm);

    const auto query_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 1);
    const auto key_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 2);
    const auto value_weight = tensor_q6_k_gguf_weight_matrix(256, 256, 3);
    const auto output_weight = tensor_q4_k_gguf_weight_matrix(256, 256, 4);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_q.weight"), query_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_k.weight"), key_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_v.weight"), value_weight);
    overwrite_payload_bytes(payload_store, block_tensor_name(0, "attn_output.weight"), output_weight);

    auto view = tiny_model_view(tiny_gemma2_metadata(), std::move(payload_store.payloads));
    const auto model = load_engine_model_from_gguf(view);
    const std::array<std::uint32_t, 2> prompt_token_ids {3, 1};

    const auto shell = create_gemma2_metal_generation_runtime_shell(
        model,
        std::span<const std::uint32_t>(prompt_token_ids)
    );
    const auto& projection = shell.prefill_first_layer_attention_projection;
    const auto& position = shell.prefill_first_layer_attention_position;
    const auto& scores = shell.prefill_first_layer_attention_scores;
    const auto& probabilities = shell.prefill_first_layer_attention_probabilities;
    const auto& values = shell.prefill_first_layer_attention_values;
    const auto& output_projection = shell.prefill_first_layer_attention_output_projection;
    const auto& post_attention_residual = shell.prefill_first_layer_post_attention_residual;
    const auto residual_hidden_states = copy_tensor_from_backend(shell.prefill_input.hidden_states);
    const auto projected_value = copy_tensor_from_backend(projection.value);
    const auto positioned_query = copy_tensor_from_backend(position.query);
    const auto positioned_key = copy_tensor_from_backend(position.key);
    const auto score_output = copy_tensor_from_backend(scores.scores);
    const auto expected_scores = causal_grouped_attention_scores_f32(
        positioned_query,
        positioned_key,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_key_length,
        shell.prefill.layers().front().attention_token_count,
        scores.scale,
        shell.plan.config().attention_logit_softcap
    );
    const auto probability_output = copy_tensor_from_backend(probabilities.probabilities);
    const auto expected_probabilities = softmax_f32(expected_scores);
    const auto attention_value_output = copy_tensor_from_backend(values.attention_output);
    const auto expected_attention_values = causal_grouped_attention_values_f32(
        probability_output,
        projected_value,
        shell.prefill.cache_position(),
        shell.plan.config().attention_head_count,
        shell.plan.config().attention_head_count_kv,
        shell.plan.config().attention_value_length,
        shell.prefill.layers().front().attention_token_count
    );
    const auto projected_attention_output = copy_tensor_from_backend(
        output_projection.projected_attention_output
    );
    const auto expected_projected_attention_output = matmul_f32_quantized_gguf_weight(
        attention_value_output,
        output_weight,
        BackendKind::cpu
    );
    const auto post_attention_residual_output = copy_tensor_from_backend(
        post_attention_residual.post_attention_residual
    );
    const auto expected_post_attention_residual = add_f32(
        residual_hidden_states,
        rms_norm_f32(
            projected_attention_output,
            post_attention_norm,
            shell.plan.config().rms_norm_epsilon
        )
    );

    expect(scores.layer_index == 0, "Gemma2 shell first layer attention score index mismatch");
    expect(
        scores.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer attention score phase mismatch"
    );
    expect(
        scores.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer attention score scratch ownership mismatch"
    );
    expect(
        scores.query_position_start == shell.prefill.cache_position(),
        "Gemma2 shell attention score query position start mismatch"
    );
    expect(
        scores.attention_token_count == shell.prefill.layers().front().attention_token_count,
        "Gemma2 shell attention score token count mismatch"
    );
    expect(
        scores.query_head_count == shell.plan.config().attention_head_count,
        "Gemma2 shell attention score query head count mismatch"
    );
    expect(
        scores.key_head_count == shell.plan.config().attention_head_count_kv,
        "Gemma2 shell attention score key head count mismatch"
    );
    expect(
        scores.key_length == shell.plan.config().attention_key_length,
        "Gemma2 shell attention score key length mismatch"
    );
    expect_close(scores.scale, 1.0F / 16.0F, 0.0F, "Gemma2 shell attention score scale mismatch");
    expect_close(
        scores.logit_softcap,
        shell.plan.config().attention_logit_softcap,
        0.0F,
        "Gemma2 shell attention score softcap mismatch"
    );
    expect(scores.scores.backend() == BackendKind::metal, "Gemma2 shell attention score backend mismatch");
    expect(
        scores.scores.descriptor() == shell.prefill.scratch().attention_scores,
        "Gemma2 shell attention score scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(scores.scores, position.query),
        "Gemma2 shell attention scores should not alias positioned query"
    );
    expect(
        !shares_backend_storage(scores.scores, position.key),
        "Gemma2 shell attention scores should not alias positioned key"
    );

    const std::array<std::uint64_t, 3> masked_score {0, 0, 1};
    expect(
        std::isinf(read_f32(expected_scores, std::span<const std::uint64_t>(masked_score))),
        "Gemma2 shell attention score fixture should include causal mask"
    );
    expect_f32_rank3_close(score_output, expected_scores, 1.0e-3F, "Gemma2 shell attention scores");

    expect(probabilities.layer_index == 0, "Gemma2 shell first layer attention probability index mismatch");
    expect(
        probabilities.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer attention probability phase mismatch"
    );
    expect(
        probabilities.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer attention probability scratch ownership mismatch"
    );
    expect(
        probabilities.attention_token_count == shell.prefill.layers().front().attention_token_count,
        "Gemma2 shell attention probability token count mismatch"
    );
    expect(
        probabilities.query_head_count == shell.plan.config().attention_head_count,
        "Gemma2 shell attention probability query head count mismatch"
    );
    expect(
        probabilities.probabilities.backend() == BackendKind::metal,
        "Gemma2 shell attention probability backend mismatch"
    );
    expect(
        probabilities.probabilities.descriptor() == shell.prefill.scratch().attention_probabilities,
        "Gemma2 shell attention probability scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(probabilities.probabilities, scores.scores),
        "Gemma2 shell attention probabilities should not alias scores"
    );
    expect(
        !shares_backend_storage(probabilities.probabilities, position.query),
        "Gemma2 shell attention probabilities should not alias positioned query"
    );
    expect(
        !shares_backend_storage(probabilities.probabilities, position.key),
        "Gemma2 shell attention probabilities should not alias positioned key"
    );
    expect_f32_rank3_close(
        probability_output,
        expected_probabilities,
        1.0e-4F,
        "Gemma2 shell attention probabilities"
    );

    expect(values.layer_index == 0, "Gemma2 shell first layer attention value index mismatch");
    expect(
        values.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer attention value phase mismatch"
    );
    expect(
        values.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer attention value scratch ownership mismatch"
    );
    expect(
        values.query_position_start == shell.prefill.cache_position(),
        "Gemma2 shell attention value query position start mismatch"
    );
    expect(
        values.attention_token_count == shell.prefill.layers().front().attention_token_count,
        "Gemma2 shell attention value token count mismatch"
    );
    expect(
        values.query_head_count == shell.plan.config().attention_head_count,
        "Gemma2 shell attention value query head count mismatch"
    );
    expect(
        values.value_head_count == shell.plan.config().attention_head_count_kv,
        "Gemma2 shell attention value KV head count mismatch"
    );
    expect(
        values.value_length == shell.plan.config().attention_value_length,
        "Gemma2 shell attention value length mismatch"
    );
    expect(values.attention_output.backend() == BackendKind::metal, "Gemma2 shell attention value backend mismatch");
    expect(
        values.attention_output.descriptor() == shell.prefill.scratch().attention_output,
        "Gemma2 shell attention value scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(values.attention_output, probabilities.probabilities),
        "Gemma2 shell attention value output should not alias probabilities"
    );
    expect(
        !shares_backend_storage(values.attention_output, projection.value),
        "Gemma2 shell attention value output should not alias value projection"
    );
    expect_f32_matrix_close(
        attention_value_output,
        expected_attention_values,
        1.0e-3F,
        "Gemma2 shell attention values"
    );

    expect(
        output_projection.layer_index == 0,
        "Gemma2 shell first layer attention output projection index mismatch"
    );
    expect(
        output_projection.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer attention output projection phase mismatch"
    );
    expect(
        output_projection.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer attention output projection scratch ownership mismatch"
    );
    expect(
        output_projection.attention_output_width
            == shell.plan.config().attention_head_count * shell.plan.config().attention_value_length,
        "Gemma2 shell first layer attention output projection input width mismatch"
    );
    expect(
        output_projection.hidden_width == shell.plan.config().embedding_length,
        "Gemma2 shell first layer attention output projection hidden width mismatch"
    );
    expect(
        output_projection.projected_attention_output.backend() == BackendKind::metal,
        "Gemma2 shell projected attention output backend mismatch"
    );
    expect(
        output_projection.projected_attention_output.descriptor()
            == shell.prefill.scratch().projected_attention_output,
        "Gemma2 shell projected attention output scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(output_projection.projected_attention_output, values.attention_output),
        "Gemma2 shell projected attention output should not alias attention value output"
    );
    expect(
        !shares_backend_storage(
            output_projection.projected_attention_output,
            shell.plan.weights().blocks[0].attention.output_weight
        ),
        "Gemma2 shell projected attention output should not alias attention output weight"
    );
    expect_f32_matrix_close(
        projected_attention_output,
        expected_projected_attention_output,
        1.0e-2F,
        "Gemma2 shell projected attention output"
    );

    expect(
        post_attention_residual.layer_index == 0,
        "Gemma2 shell first layer post-attention residual index mismatch"
    );
    expect(
        post_attention_residual.phase == Gemma2MetalSchedulePhase::prefill,
        "Gemma2 shell first layer post-attention residual phase mismatch"
    );
    expect(
        post_attention_residual.scratch_ownership == Gemma2MetalScratchOwnership::phase_local,
        "Gemma2 shell first layer post-attention residual scratch ownership mismatch"
    );
    expect(
        post_attention_residual.hidden_width == shell.plan.config().embedding_length,
        "Gemma2 shell first layer post-attention residual hidden width mismatch"
    );
    expect(
        post_attention_residual.post_attention_residual.backend() == BackendKind::metal,
        "Gemma2 shell post-attention residual backend mismatch"
    );
    expect(
        post_attention_residual.post_attention_residual.descriptor()
            == shell.prefill.scratch().post_attention_residual,
        "Gemma2 shell post-attention residual scheduled descriptor mismatch"
    );
    expect(
        !shares_backend_storage(
            post_attention_residual.post_attention_residual,
            output_projection.projected_attention_output
        ),
        "Gemma2 shell post-attention residual should not alias projected attention output"
    );
    expect(
        !shares_backend_storage(post_attention_residual.post_attention_residual, shell.prefill_input.hidden_states),
        "Gemma2 shell post-attention residual should not alias residual input"
    );
    expect(
        !shares_backend_storage(
            post_attention_residual.post_attention_residual,
            shell.plan.weights().blocks[0].attention.post_attention_norm_weight
        ),
        "Gemma2 shell post-attention residual should not alias post-attention norm weight"
    );
    expect_f32_matrix_close(
        post_attention_residual_output,
        expected_post_attention_residual,
        1.0e-2F,
        "Gemma2 shell post-attention residual"
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
        "Gemma2 Metal runtime shell projects first layer attention when available",
        test_gemma2_metal_runtime_shell_projects_first_layer_attention_when_available,
    },
    {
        "Gemma2 Metal runtime shell positions first layer attention when available",
        test_gemma2_metal_runtime_shell_positions_first_layer_attention_when_available,
    },
    {
        "Gemma2 Metal runtime shell applies first layer post-attention residual when available",
        test_gemma2_metal_runtime_shell_applies_first_layer_post_attention_residual_when_available,
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
