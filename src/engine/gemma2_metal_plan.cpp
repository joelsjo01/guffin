#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"

#include "local_gguf_engine/engine/target_verifier.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/kquant_packing.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "../tensor/backends/metal_backend.hpp"

namespace local_gguf_engine::engine {

namespace {

using runtime::BackendKind;
using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;

constexpr auto gemma_backend = BackendKind::metal;

class ResidentPayloadHasher final {
public:
    explicit ResidentPayloadHasher(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            constexpr std::string_view domain =
                "local-gguf-engine/resident-target-payload/v1";
            hash_.update(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(domain.data()), domain.size()
            ));
        }
    }

    void add(std::string_view label, tensor::TensorView view) {
        if (!enabled_) return;
        const auto& descriptor = view.descriptor();
        if (descriptor.packing == tensor::TensorPacking::kquant_columns8) {
            // Canonical receipts retain their original byte stream. An
            // impossible label length frames the execution-layout version
            // before the physical payload it authenticates.
            constexpr std::string_view layout =
                "local-gguf-engine/kquant-columns8/v1";
            add_u64(std::numeric_limits<std::uint64_t>::max());
            add_u64(layout.size());
            hash_.update(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(layout.data()), layout.size()
            ));
        } else if (descriptor.packing != tensor::TensorPacking::canonical) {
            throw std::invalid_argument("resident target receipt has an unknown tensor packing");
        }
        add_u64(label.size());
        hash_.update(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(label.data()), label.size()
        ));
        add_u32(static_cast<std::uint32_t>(descriptor.dtype));
        add_u32(static_cast<std::uint32_t>(descriptor.shape.rank()));
        for (std::size_t index = 0U; index < descriptor.shape.rank(); ++index) {
            add_u64(descriptor.shape.dimension(index));
        }
        add_u64(descriptor.byte_count);
        hash_.update(view.bytes());
        if (descriptor.byte_count > std::numeric_limits<std::uint64_t>::max() - bytes_) {
            throw std::overflow_error("resident target receipt byte count overflow");
        }
        bytes_ += descriptor.byte_count;
        if (allocations_ == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("resident target receipt allocation count overflow");
        }
        ++allocations_;
    }

    [[nodiscard]] ResidentTargetPayloadReceipt finish() {
        if (!enabled_ || allocations_ == 0U) {
            throw std::logic_error("resident target payload receipt was not enabled or is empty");
        }
        return ResidentTargetPayloadReceipt {
            .allocation_count = allocations_,
            .byte_count = bytes_,
            .bytes = hash_.finalize(),
        };
    }

private:
    void add_u32(std::uint32_t value) {
        const std::array<std::uint8_t, 4> bytes {
            static_cast<std::uint8_t>(value),
            static_cast<std::uint8_t>(value >> 8U),
            static_cast<std::uint8_t>(value >> 16U),
            static_cast<std::uint8_t>(value >> 24U),
        };
        hash_.update(bytes);
    }

    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes {};
        for (std::size_t index = 0U; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
        }
        hash_.update(bytes);
    }

    bool enabled_ = false;
    runtime::Sha256 hash_;
    std::uint32_t allocations_ = 0U;
    std::uint64_t bytes_ = 0U;
};

void require_metal_available() {
    if (tensor::tensor_backend_is_available(gemma_backend)) {
        return;
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unavailable_metal_device,
        gemma_backend,
        tensor::tensor_backend_unavailable_reason(gemma_backend)
    );
}

void checked_add_bytes(std::size_t& total, std::size_t byte_count) {
    if (byte_count > std::numeric_limits<std::size_t>::max() - total) {
        throw runtime::BackendError(
            runtime::BackendErrorCode::metal_execution_failed,
            gemma_backend,
            "Gemma2 Metal plan byte count overflow"
        );
    }

    total += byte_count;
}

void checked_add_decode_bytes(std::uint64_t& total, std::size_t byte_count) {
    if (byte_count > std::numeric_limits<std::uint64_t>::max() - total) {
        throw runtime::BackendError(
            runtime::BackendErrorCode::metal_execution_failed,
            gemma_backend,
            "Gemma2 scheduled decode weight byte count overflow"
        );
    }
    total += static_cast<std::uint64_t>(byte_count);
}

[[nodiscard]] Gemma2DecodeScheduledWeightBytes decode_scheduled_weight_bytes(
    const Gemma2MetalModelWeightBuffers& weights,
    const GemmaModelConfig& config
) {
    Gemma2DecodeScheduledWeightBytes bytes;
    const auto& embedding = weights.token_embedding_weight.descriptor();
    if (embedding.shape.rank() != 2U || embedding.shape.dimension(0) != config.embedding_length
        || embedding.shape.dimension(1) != config.vocab_size) {
        throw std::invalid_argument("Gemma2 token embedding layout is invalid for decode accounting");
    }
    const auto row_elements = embedding.shape.dimension(0);
    const auto dtype = tensor::dtype_layout(embedding.dtype);
    if (row_elements % dtype.block_element_count != 0U) {
        throw std::invalid_argument("Gemma2 token embedding row is not dtype-block aligned");
    }
    bytes.embedding = row_elements / dtype.block_element_count * dtype.block_byte_count;

    for (const auto& block : weights.blocks) {
        for (const auto* projection : {
                 &block.attention.query_weight, &block.attention.key_weight,
                 &block.attention.value_weight, &block.attention.output_weight,
             }) {
            checked_add_decode_bytes(bytes.attention_projections, projection->descriptor().byte_count);
        }
        checked_add_decode_bytes(bytes.feed_forward_gate_up, block.feed_forward.gate_weight.descriptor().byte_count);
        checked_add_decode_bytes(bytes.feed_forward_gate_up, block.feed_forward.up_weight.descriptor().byte_count);
        checked_add_decode_bytes(bytes.feed_forward_down, block.feed_forward.down_weight.descriptor().byte_count);
        for (const auto* norm : {
                 &block.attention.attention_norm_weight,
                 &block.attention.post_attention_norm_weight,
                 &block.feed_forward.feed_forward_norm_weight,
                 &block.feed_forward.post_feed_forward_norm_weight,
             }) {
            checked_add_decode_bytes(bytes.norms, norm->descriptor().byte_count);
        }
        if (block.attention.query_norm_weight) {
            checked_add_decode_bytes(bytes.norms, block.attention.query_norm_weight->descriptor().byte_count);
            checked_add_decode_bytes(bytes.norms, block.attention.key_norm_weight->descriptor().byte_count);
        }
    }
    bytes.output_head = weights.output_weight.descriptor().byte_count;
    checked_add_decode_bytes(bytes.norms, weights.output_norm_weight.descriptor().byte_count);
    for (const auto value : {
             bytes.embedding, bytes.attention_projections, bytes.feed_forward_gate_up,
             bytes.feed_forward_down, bytes.output_head, bytes.norms,
         }) {
        if (value > std::numeric_limits<std::uint64_t>::max() - bytes.total) {
            throw runtime::BackendError(
                runtime::BackendErrorCode::metal_execution_failed,
                gemma_backend,
                "Gemma2 scheduled decode weight total overflow"
            );
        }
        bytes.total += value;
    }
    return bytes;
}

[[nodiscard]] tensor::BackendBuffer upload_weight(
    tensor::TensorView view,
    std::size_t& uploaded_byte_count,
    ResidentPayloadHasher& hasher,
    std::string_view label
) {
    checked_add_bytes(uploaded_byte_count, view.bytes().size());
    hasher.add(label, view);
    return tensor::copy_tensor_view_to_backend(view, gemma_backend);
}

[[nodiscard]] TensorDescriptor gemma3_projection_descriptor(tensor::TensorView view) {
    const auto& descriptor = view.descriptor();
    if (descriptor.dtype == DType::q4_k || descriptor.dtype == DType::q6_k) {
        return TensorDescriptor::from_interleaved_kquant_weight_matrix(
            descriptor.shape, descriptor.dtype);
    }
    return descriptor;
}

[[nodiscard]] std::size_t gemma3_resident_weight_bytes(
    const GemmaModelConfig& config, const GemmaModelWeights& weights
) {
    std::size_t bytes = 0U;
    const auto native = [&](tensor::TensorView view) {
        checked_add_bytes(bytes, view.descriptor().byte_count);
    };
    native(weights.token_embedding_weight);
    native(weights.output_norm_weight);
    if (!config.uses_tied_output_weight) native(weights.output_weight);
    for (const auto& block : weights.blocks) {
        for (const auto* projection : {
                 &block.attention.query_weight, &block.attention.key_weight,
                 &block.attention.value_weight, &block.attention.output_weight,
                 &block.feed_forward.gate_weight, &block.feed_forward.up_weight,
                 &block.feed_forward.down_weight,
             }) {
            checked_add_bytes(bytes, gemma3_projection_descriptor(*projection).byte_count);
        }
        for (const auto* norm : {
                 &block.attention.attention_norm_weight,
                 &block.attention.post_attention_norm_weight,
                 &block.feed_forward.feed_forward_norm_weight,
                 &block.feed_forward.post_feed_forward_norm_weight,
             }) native(*norm);
        native(block.attention.query_norm_weight.value());
        native(block.attention.key_norm_weight.value());
    }
    return bytes;
}

enum class ProjectionPacking { canonical, q4_columns8, kquant_columns8 };

[[nodiscard]] tensor::BackendBuffer upload_projection_weight(
    tensor::TensorView view, std::size_t& uploaded_byte_count,
    ResidentPayloadHasher& hasher, std::string_view label, ProjectionPacking packing
) {
    if ((packing != ProjectionPacking::canonical && view.descriptor().dtype == DType::q4_k)
        || (packing == ProjectionPacking::kquant_columns8 && view.descriptor().dtype == DType::q6_k)) {
        // Only this tensor's packed staging storage is live during upload;
        // the resident allocation replaces its GGUF representation.
        const auto packed = tensor::repack_kquant_weight_columns8(view);
        return upload_weight(packed.view(), uploaded_byte_count, hasher, label);
    }
    return upload_weight(view, uploaded_byte_count, hasher, label);
}

[[nodiscard]] tensor::BackendBuffer allocate_buffer(
    const TensorDescriptor& descriptor,
    std::size_t& allocated_byte_count
) {
    checked_add_bytes(allocated_byte_count, descriptor.byte_count);
    return tensor::allocate_backend_buffer(descriptor, gemma_backend);
}

[[nodiscard]] TensorDescriptor f32_descriptor(std::initializer_list<std::uint64_t> dimensions) {
    return TensorDescriptor::from(Shape::from(dimensions), DType::f32);
}

[[nodiscard]] TensorDescriptor kv_descriptor(
    const GemmaModelConfig& config, std::initializer_list<std::uint64_t> dimensions
) {
    return TensorDescriptor::from(Shape::from(dimensions), gemma_kv_dtype(config));
}

[[nodiscard]] bool is_power_of_two(std::uint32_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

void require_power_of_two_cache_layout(const GemmaModelConfig& config) {
    if (!is_power_of_two(config.context_length) || !is_power_of_two(config.sliding_window)) {
        throw std::invalid_argument(
            "Gemma2 Metal KV cache requires power-of-two context and sliding-window lengths"
        );
    }
}

[[nodiscard]] std::uint64_t attention_query_width(const GemmaModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count) * config.attention_key_length;
}

[[nodiscard]] std::uint64_t attention_key_width(const GemmaModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count_kv) * config.attention_key_length;
}

[[nodiscard]] std::uint64_t attention_value_width(const GemmaModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count_kv) * config.attention_value_length;
}

[[nodiscard]] std::uint64_t attention_output_width(const GemmaModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count) * config.attention_value_length;
}

[[nodiscard]] bool is_retained_gemma2_9b_target(
    const GemmaModelConfig& config,
    const GemmaModelWeights& weights
);

[[nodiscard]] Gemma2MetalAttentionWeightBuffers upload_attention_weights(
    const GemmaAttentionWeights& weights,
    std::size_t& uploaded_byte_count,
    ResidentPayloadHasher& hasher,
    std::uint32_t block_index,
    ProjectionPacking packing
) {
    const auto prefix = "blk." + std::to_string(block_index) + ".";
    return Gemma2MetalAttentionWeightBuffers {
        .attention_norm_weight = upload_weight(weights.attention_norm_weight, uploaded_byte_count, hasher, prefix + "attn_norm.weight"),
        .post_attention_norm_weight = upload_weight(weights.post_attention_norm_weight, uploaded_byte_count, hasher, prefix + "post_attention_norm.weight"),
        .query_weight = upload_projection_weight(weights.query_weight, uploaded_byte_count, hasher, prefix + "attn_q.weight", packing),
        .key_weight = upload_projection_weight(weights.key_weight, uploaded_byte_count, hasher, prefix + "attn_k.weight", packing),
        .value_weight = upload_projection_weight(weights.value_weight, uploaded_byte_count, hasher, prefix + "attn_v.weight", packing),
        .output_weight = upload_projection_weight(weights.output_weight, uploaded_byte_count, hasher, prefix + "attn_output.weight", packing),
        .query_norm_weight = weights.query_norm_weight
            ? std::optional(upload_weight(*weights.query_norm_weight, uploaded_byte_count, hasher, prefix + "attn_q_norm.weight"))
            : std::nullopt,
        .key_norm_weight = weights.key_norm_weight
            ? std::optional(upload_weight(*weights.key_norm_weight, uploaded_byte_count, hasher, prefix + "attn_k_norm.weight"))
            : std::nullopt,
    };
}

[[nodiscard]] Gemma2MetalFeedForwardWeightBuffers upload_feed_forward_weights(
    const GemmaFeedForwardWeights& weights,
    std::size_t& uploaded_byte_count,
    bool requantize_q6_down,
    ResidentPayloadHasher& hasher,
    std::uint32_t block_index,
    ProjectionPacking packing
) {
    const auto prefix = "blk." + std::to_string(block_index) + ".";
    auto down_weight = [&]() {
        if (requantize_q6_down && weights.down_weight.descriptor().dtype == DType::q6_k) {
            auto quantized = tensor::requantize_q6_k_gguf_weight_q4_k(weights.down_weight);
            return upload_projection_weight(quantized.view(), uploaded_byte_count,
                hasher, prefix + "ffn_down.weight:q6_k->q4_k", packing);
        }
        return upload_projection_weight(
            weights.down_weight, uploaded_byte_count, hasher, prefix + "ffn_down.weight", packing
        );
    }();
    return Gemma2MetalFeedForwardWeightBuffers {
        .feed_forward_norm_weight = upload_weight(weights.feed_forward_norm_weight, uploaded_byte_count, hasher, prefix + "ffn_norm.weight"),
        .post_feed_forward_norm_weight = upload_weight(weights.post_feed_forward_norm_weight, uploaded_byte_count, hasher, prefix + "post_ffw_norm.weight"),
        .gate_weight = upload_projection_weight(weights.gate_weight, uploaded_byte_count, hasher, prefix + "ffn_gate.weight", packing),
        .up_weight = upload_projection_weight(weights.up_weight, uploaded_byte_count, hasher, prefix + "ffn_up.weight", packing),
        .down_weight = std::move(down_weight),
    };
}

[[nodiscard]] std::vector<Gemma2MetalBlockWeightBuffers> upload_block_weights(
    const GemmaModelWeights& weights,
    std::size_t& uploaded_byte_count,
    bool requantize_q6_down,
    ResidentPayloadHasher& hasher,
    ProjectionPacking packing
) {
    std::vector<Gemma2MetalBlockWeightBuffers> blocks;
    blocks.reserve(weights.blocks.size());

    for (std::size_t block_index = 0U; block_index < weights.blocks.size(); ++block_index) {
        const auto& block = weights.blocks[block_index];
        blocks.push_back(Gemma2MetalBlockWeightBuffers {
            .attention = upload_attention_weights(
                block.attention, uploaded_byte_count, hasher,
                static_cast<std::uint32_t>(block_index), packing
            ),
            .feed_forward = upload_feed_forward_weights(
                block.feed_forward, uploaded_byte_count, requantize_q6_down,
                hasher, static_cast<std::uint32_t>(block_index), packing
            ),
        });
    }

    return blocks;
}

[[nodiscard]] Gemma2MetalModelWeightBuffers upload_model_weights(
    const GemmaModelWeights& weights,
    std::size_t& uploaded_byte_count,
    GemmaOutputHeadPolicy output_head_policy,
    bool retained_target,
    ResidentPayloadHasher& hasher,
    ProjectionPacking packing
) {
    auto token_embedding_weight = upload_weight(
        weights.token_embedding_weight, uploaded_byte_count, hasher, "token_embd.weight"
    );
    auto output_weight = [&]() {
        if (output_head_policy == GemmaOutputHeadPolicy::q8_0) {
            auto quantized = tensor::quantize_f16_gguf_weight_q8_0(weights.output_weight);
            checked_add_bytes(uploaded_byte_count, quantized.descriptor().byte_count);
            hasher.add("output.weight:f16->q8_0", quantized.view());
            return tensor::copy_tensor_to_backend(quantized, gemma_backend);
        }
        return weights.output_weight.bytes().data() == weights.token_embedding_weight.bytes().data()
            ? token_embedding_weight
            : upload_weight(
                weights.output_weight, uploaded_byte_count, hasher, "output.weight"
            );
    }();

    return Gemma2MetalModelWeightBuffers {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = upload_weight(
            weights.output_norm_weight, uploaded_byte_count, hasher, "output_norm.weight"
        ),
        .output_weight = std::move(output_weight),
        .blocks = upload_block_weights(
            weights, uploaded_byte_count, retained_target, hasher, packing
        ),
    };
}

[[nodiscard]] bool is_retained_gemma2_9b_target(
    const GemmaModelConfig& config,
    const GemmaModelWeights& weights
) {
    const auto& embedding = weights.token_embedding_weight;
    const auto& output = weights.output_weight;
    return config.architecture == ModelArchitecture::gemma2 && config.uses_tied_output_weight
        && config.embedding_length == 3584U
        && config.block_count == 42U
        && config.feed_forward_length == 14336U
        && config.vocab_size == 256000U
        && weights.blocks.size() == 42U
        && embedding.descriptor().dtype == DType::f16
        && output.descriptor().dtype == DType::f16
        && embedding.descriptor().shape == Shape::from({3584U, 256000U})
        && output.descriptor().shape == embedding.descriptor().shape
        && output.bytes().data() == embedding.bytes().data();
}

[[nodiscard]] std::vector<Gemma2MetalKVCacheLayer> allocate_kv_cache(
    const GemmaModelConfig& config,
    std::size_t& allocated_byte_count
) {
    std::vector<Gemma2MetalKVCacheLayer> layers;
    layers.reserve(config.block_count);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto rows = gemma_layer_uses_sliding_window(config, layer)
            ? config.sliding_window
            : config.context_length;
        layers.push_back(Gemma2MetalKVCacheLayer {
            .key = allocate_buffer(kv_descriptor(config, {
                rows, config.attention_head_count_kv, config.attention_key_length,
            }), allocated_byte_count),
            .value = allocate_buffer(kv_descriptor(config, {
                rows, config.attention_head_count_kv, config.attention_value_length,
            }), allocated_byte_count),
            .row_capacity = rows,
            .row_mask = rows - 1U,
        });
    }

    return layers;
}

[[nodiscard]] Gemma2MetalDecodeScratch allocate_decode_scratch(
    const GemmaModelConfig& config,
    std::size_t& allocated_byte_count
) {
    return Gemma2MetalDecodeScratch {
        .token_id_staging = allocate_buffer(f32_descriptor({1}), allocated_byte_count),
        .normalized_hidden = allocate_buffer(f32_descriptor({1, config.embedding_length}), allocated_byte_count),
        .raw_query = allocate_buffer(f32_descriptor({1, attention_query_width(config)}), allocated_byte_count),
        .raw_key = allocate_buffer(f32_descriptor({1, attention_key_width(config)}), allocated_byte_count),
        .raw_value = allocate_buffer(f32_descriptor({1, attention_value_width(config)}), allocated_byte_count),
        .query = allocate_buffer(f32_descriptor({1, attention_query_width(config)}), allocated_byte_count),
        .attention_partial_output = allocate_buffer(
            f32_descriptor({
                gemma2_decode_attention_max_partitions,
                gemma2_decode_attention_query_heads,
                gemma2_decode_attention_head_dimension,
            }),
            allocated_byte_count
        ),
        .attention_partial_maxima = allocate_buffer(
            f32_descriptor({gemma2_decode_attention_max_partitions, gemma2_decode_attention_query_heads}),
            allocated_byte_count
        ),
        .attention_partial_denominators = allocate_buffer(
            f32_descriptor({gemma2_decode_attention_max_partitions, gemma2_decode_attention_query_heads}),
            allocated_byte_count
        ),
        .attention_output = allocate_buffer(f32_descriptor({1, attention_output_width(config)}), allocated_byte_count),
        .projected_attention_output = allocate_buffer(
            f32_descriptor({1, config.embedding_length}),
            allocated_byte_count
        ),
        .post_attention_residual = allocate_buffer(
            f32_descriptor({1, config.embedding_length}),
            allocated_byte_count
        ),
        .normalized_feed_forward_input = allocate_buffer(
            f32_descriptor({1, config.embedding_length}),
            allocated_byte_count
        ),
        .feed_forward_gate = allocate_buffer(f32_descriptor({1, config.feed_forward_length}), allocated_byte_count),
        .feed_forward_up = allocate_buffer(f32_descriptor({1, config.feed_forward_length}), allocated_byte_count),
        .feed_forward_gated = allocate_buffer(f32_descriptor({1, config.feed_forward_length}), allocated_byte_count),
        .projected_feed_forward_output = allocate_buffer(
            f32_descriptor({1, config.embedding_length}),
            allocated_byte_count
        ),
        .feed_forward_residual = allocate_buffer(
            f32_descriptor({1, config.embedding_length}),
            allocated_byte_count
        ),
        .logits = allocate_buffer(f32_descriptor({1, config.vocab_size}), allocated_byte_count),
    };
}

[[nodiscard]] Gemma2MetalVerifierScratch allocate_verifier_scratch(
    const GemmaModelConfig& config,
    std::size_t& allocated_byte_count
) {
    constexpr std::uint64_t rows = 2U;
    const auto staged_kv_descriptor = kv_descriptor(config, {
        rows, config.attention_head_count_kv, config.attention_key_length,
    });
    std::vector<Gemma2MetalVerifierKVBackupLayer> kv_backups;
    kv_backups.reserve(config.block_count);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        kv_backups.push_back(Gemma2MetalVerifierKVBackupLayer {
            .backup_key = allocate_buffer(staged_kv_descriptor, allocated_byte_count),
            .backup_value = allocate_buffer(staged_kv_descriptor, allocated_byte_count),
        });
    }
    return Gemma2MetalVerifierScratch {
        .token_id_staging = allocate_buffer(f32_descriptor({rows}), allocated_byte_count),
        .normalized_hidden = allocate_buffer(f32_descriptor({rows, config.embedding_length}), allocated_byte_count),
        .raw_query = allocate_buffer(f32_descriptor({rows, attention_query_width(config)}), allocated_byte_count),
        .raw_key = allocate_buffer(f32_descriptor({rows, attention_key_width(config)}), allocated_byte_count),
        .raw_value = allocate_buffer(f32_descriptor({rows, attention_value_width(config)}), allocated_byte_count),
        .query = allocate_buffer(f32_descriptor({rows, attention_query_width(config)}), allocated_byte_count),
        .attention_partial_output = allocate_buffer(f32_descriptor({
            gemma2_decode_attention_max_partitions,
            gemma2_decode_attention_query_heads,
            gemma2_decode_attention_head_dimension,
        }), allocated_byte_count),
        .attention_partial_maxima = allocate_buffer(f32_descriptor({
            gemma2_decode_attention_max_partitions, gemma2_decode_attention_query_heads,
        }), allocated_byte_count),
        .attention_partial_denominators = allocate_buffer(f32_descriptor({
            gemma2_decode_attention_max_partitions, gemma2_decode_attention_query_heads,
        }), allocated_byte_count),
        .attention_output = allocate_buffer(f32_descriptor({rows, attention_output_width(config)}), allocated_byte_count),
        .projected_attention_output = allocate_buffer(f32_descriptor({rows, config.embedding_length}), allocated_byte_count),
        .post_attention_residual = allocate_buffer(f32_descriptor({rows, config.embedding_length}), allocated_byte_count),
        .normalized_feed_forward_input = allocate_buffer(f32_descriptor({rows, config.embedding_length}), allocated_byte_count),
        .feed_forward_gate = allocate_buffer(f32_descriptor({rows, config.feed_forward_length}), allocated_byte_count),
        .feed_forward_up = allocate_buffer(f32_descriptor({rows, config.feed_forward_length}), allocated_byte_count),
        .feed_forward_gated = allocate_buffer(f32_descriptor({rows, config.feed_forward_length}), allocated_byte_count),
        .projected_feed_forward_output = allocate_buffer(f32_descriptor({rows, config.embedding_length}), allocated_byte_count),
        .feed_forward_residual = allocate_buffer(f32_descriptor({rows, config.embedding_length}), allocated_byte_count),
        .logits = allocate_buffer(f32_descriptor({rows, config.vocab_size}), allocated_byte_count),
        .kv_backups = std::move(kv_backups),
    };
}

[[nodiscard]] Gemma2MetalPrefillScratch allocate_prefill_scratch(
    const GemmaModelConfig& config,
    std::size_t& allocated_byte_count
) {
    const auto tokens = gemma2_metal_prefill_chunk_size;
    // Prefill is encoded serially. These aliases follow producer/last-consumer
    // boundaries: scores become probabilities in place; the hidden workspace
    // advances from attention norm through both residual branches; and GELU
    // replaces its dead gate input. All aliases retain identical descriptors.
    const auto hidden_workspace = allocate_buffer(
        f32_descriptor({tokens, config.embedding_length}), allocated_byte_count
    );
    const auto attention_score_workspace = config.architecture == ModelArchitecture::gemma3
        ? hidden_workspace : allocate_buffer(
        f32_descriptor({tokens, config.attention_head_count, config.context_length}),
        allocated_byte_count
    );
    const auto feed_forward_gate_workspace = allocate_buffer(
        f32_descriptor({tokens, config.feed_forward_length}), allocated_byte_count
    );
    return Gemma2MetalPrefillScratch {
        .token_id_staging = allocate_buffer(f32_descriptor({config.context_length}), allocated_byte_count),
        .normalized_hidden = hidden_workspace,
        .raw_query = allocate_buffer(f32_descriptor({tokens, attention_query_width(config)}), allocated_byte_count),
        .raw_key = allocate_buffer(f32_descriptor({tokens, attention_key_width(config)}), allocated_byte_count),
        .query = allocate_buffer(f32_descriptor({tokens, attention_query_width(config)}), allocated_byte_count),
        .value = allocate_buffer(f32_descriptor({tokens, attention_value_width(config)}), allocated_byte_count),
        .current_chunk_key = allocate_buffer(kv_descriptor(config, {
            tokens, config.attention_head_count_kv, config.attention_key_length,
        }), allocated_byte_count),
        .current_chunk_value = allocate_buffer(kv_descriptor(config, {
            tokens, config.attention_head_count_kv, config.attention_value_length,
        }), allocated_byte_count),
        .attention_scores = attention_score_workspace,
        .attention_probabilities = attention_score_workspace,
        .attention_output = allocate_buffer(
            f32_descriptor({tokens, attention_output_width(config)}),
            allocated_byte_count
        ),
        .projected_attention_output = hidden_workspace,
        .post_attention_residual =
            allocate_buffer(f32_descriptor({tokens, config.embedding_length}), allocated_byte_count),
        .normalized_feed_forward_input = hidden_workspace,
        .feed_forward_gate = feed_forward_gate_workspace,
        .feed_forward_up =
            allocate_buffer(f32_descriptor({tokens, config.feed_forward_length}), allocated_byte_count),
        .feed_forward_gated = feed_forward_gate_workspace,
        .projected_feed_forward_output = hidden_workspace,
        .feed_forward_residual =
            allocate_buffer(f32_descriptor({tokens, config.embedding_length}), allocated_byte_count),
        .final_normalized_hidden =
            allocate_buffer(f32_descriptor({1, config.embedding_length}), allocated_byte_count),
        .logits = allocate_buffer(f32_descriptor({1, config.vocab_size}), allocated_byte_count),
    };
}

[[nodiscard]] Gemma2MetalSelectionWorkspace allocate_selection_workspace(
    const GemmaModelConfig& config,
    std::size_t& allocated_byte_count
) {
    constexpr std::uint64_t candidate_f32_elements = sizeof(TokenLogit) / sizeof(float);
    const auto tile_count = (
        static_cast<std::uint64_t>(config.vocab_size) + gemma2_metal_selection_tile_size - 1U
    ) / gemma2_metal_selection_tile_size;
    constexpr std::uint64_t selection_rows = 2U;
    const auto run_elements = selection_rows * tile_count
        * token_sampler_max_base_candidates * candidate_f32_elements;
    constexpr auto result_header_f32_elements = std::uint64_t {2};
    constexpr auto result_elements = result_header_f32_elements
        + token_sampler_max_candidates * candidate_f32_elements;
    return Gemma2MetalSelectionWorkspace {
        .candidate_runs_a = allocate_buffer(f32_descriptor({run_elements}), allocated_byte_count),
        .candidate_runs_b = allocate_buffer(f32_descriptor({run_elements}), allocated_byte_count),
        .compact_result = allocate_buffer(
            f32_descriptor({selection_rows * result_elements}), allocated_byte_count
        ),
        .history_token_staging = allocate_buffer(
            f32_descriptor({selection_rows * token_sampler_max_repetition_window}), allocated_byte_count
        ),
    };
}

[[nodiscard]] std::vector<Gemma2MetalLayerState> make_layer_states(const GemmaModelConfig& config) {
    std::vector<Gemma2MetalLayerState> layers;
    layers.reserve(config.block_count);
    for (std::uint32_t layer_index = 0; layer_index < config.block_count; ++layer_index) {
        layers.push_back(Gemma2MetalLayerState {
            .layer_index = layer_index,
            .uses_sliding_window = gemma_layer_uses_sliding_window(config, layer_index),
        });
    }

    return layers;
}

[[nodiscard]] std::uint32_t layer_attention_token_count(
    const GemmaModelConfig& config,
    const Gemma2MetalLayerState& layer,
    std::uint32_t total_kv_token_count
) {
    // KV storage may contain stale rows after warm-up, rollback, or a logical
    // session reset. Only this populated logical prefix/window is addressable;
    // the store overwrites a row before that row joins the readable range.
    if (layer.uses_sliding_window && config.sliding_window > 0) {
        return std::min(total_kv_token_count, config.sliding_window);
    }

    return total_kv_token_count;
}

[[nodiscard]] std::vector<Gemma2MetalLayerSchedule> make_layer_schedules(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t total_kv_token_count
) {
    std::vector<Gemma2MetalLayerSchedule> layers;
    layers.reserve(plan.layers().size());
    for (const auto& layer : plan.layers()) {
        layers.push_back(Gemma2MetalLayerSchedule {
            .layer_index = layer.layer_index,
            .uses_sliding_window = layer.uses_sliding_window,
            .attention_token_count = layer_attention_token_count(plan.config(), layer, total_kv_token_count),
        });
    }

    return layers;
}

void require_prefill_range(
    const GemmaModelConfig& config,
    std::uint32_t token_count,
    std::uint32_t cache_position
) {
    if (token_count == 0) {
        throw std::invalid_argument("Gemma2 Metal prefill schedule token count must be positive");
    }
    if (cache_position > config.context_length) {
        throw std::invalid_argument("Gemma2 Metal prefill schedule cache position exceeds context length");
    }
    if (token_count > config.context_length - cache_position) {
        throw std::invalid_argument("Gemma2 Metal prefill schedule token count exceeds context length");
    }
}

void require_decode_cached_token_count(const GemmaModelConfig& config, std::uint32_t cached_token_count) {
    if (cached_token_count == 0) {
        throw std::invalid_argument("Gemma2 Metal decode schedule cached token count must be positive");
    }
    if (cached_token_count >= config.context_length) {
        throw std::invalid_argument("Gemma2 Metal decode schedule cached token count must leave room for one token");
    }
}

}  // namespace

const char* gemma_output_head_policy_name(GemmaOutputHeadPolicy policy) noexcept {
    switch (policy) {
        case GemmaOutputHeadPolicy::automatic: return "automatic";
        case GemmaOutputHeadPolicy::f16: return "f16";
        case GemmaOutputHeadPolicy::q8_0: return "q8_0";
        case GemmaOutputHeadPolicy::q6_k: return "q6_k";
    }
    return "unknown";
}

GemmaOutputHeadPolicy resolve_gemma_output_head_policy(
    const GemmaModelConfig& config,
    const GemmaModelWeights& weights,
    GemmaOutputHeadPolicy requested_policy
) {
    if (config.architecture == ModelArchitecture::gemma3) {
        if (requested_policy != GemmaOutputHeadPolicy::automatic && requested_policy != GemmaOutputHeadPolicy::q6_k) {
            throw std::invalid_argument("Gemma 3 target requires its native Q6_K output head");
        }
        if (weights.output_weight.descriptor().dtype != DType::q6_k) {
            throw std::invalid_argument("Gemma 3 target output weight must be Q6_K");
        }
        return GemmaOutputHeadPolicy::q6_k;
    }
    switch (requested_policy) {
        case GemmaOutputHeadPolicy::f16:
        case GemmaOutputHeadPolicy::q8_0:
            return requested_policy;
        case GemmaOutputHeadPolicy::automatic:
            return is_retained_gemma2_9b_target(config, weights)
                ? GemmaOutputHeadPolicy::q8_0
                : GemmaOutputHeadPolicy::f16;
        case GemmaOutputHeadPolicy::q6_k:
            throw std::invalid_argument("Q6_K output-head policy requires Gemma 3");
    }

    // Policy values cross CLI/configuration and evidence-artifact boundaries.
    // Reject unknown identities instead of silently treating them as explicit.
    throw std::invalid_argument("unknown Gemma2 output-head policy");
}

Gemma2MetalModelPlan::Gemma2MetalModelPlan(
    GemmaModelConfig config,
    Gemma2MetalModelWeightBuffers weights,
    std::vector<Gemma2MetalKVCacheLayer> kv_cache,
    Gemma2MetalPrefillScratch prefill_scratch,
    Gemma2MetalDecodeScratch decode_scratch,
    Gemma2MetalVerifierScratch verifier_scratch,
    Gemma2MetalSelectionWorkspace selection_workspace,
    std::vector<Gemma2MetalLayerState> layers,
    GemmaOutputHeadPolicy output_head_policy,
    Gemma2DecodeScheduledWeightBytes decode_scheduled_weight_bytes,
    std::size_t uploaded_weight_byte_count,
    std::size_t kv_cache_byte_count,
    std::size_t scratch_byte_count
)
    : storage_(std::make_shared<Storage>(Storage {
          .config_ = config,
          .weights_ = std::move(weights),
          .kv_cache_ = std::move(kv_cache),
          .prefill_scratch_ = std::move(prefill_scratch),
          .decode_scratch_ = std::move(decode_scratch),
          .verifier_scratch_ = std::move(verifier_scratch),
          .selection_workspace_ = std::move(selection_workspace),
          .layers_ = std::move(layers),
          .output_head_policy_ = output_head_policy,
          .decode_scheduled_weight_bytes_ = decode_scheduled_weight_bytes,
          .uploaded_weight_byte_count_ = uploaded_weight_byte_count,
          .kv_cache_byte_count_ = kv_cache_byte_count,
          .scratch_byte_count_ = scratch_byte_count,
      })) {}

std::size_t Gemma2MetalModelPlan::host_storage_byte_count() noexcept {
    return sizeof(Storage);
}

runtime::BackendKind Gemma2MetalModelPlan::backend() const noexcept {
    return gemma_backend;
}

const GemmaModelConfig& Gemma2MetalModelPlan::config() const noexcept {
    return storage_->config_;
}

GemmaOutputHeadPolicy Gemma2MetalModelPlan::output_head_policy() const noexcept {
    return storage_->output_head_policy_;
}

const Gemma2MetalModelWeightBuffers& Gemma2MetalModelPlan::weights() const noexcept {
    return storage_->weights_;
}

std::vector<Gemma2MetalKVCacheLayer>& Gemma2MetalModelPlan::kv_cache() noexcept {
    return storage_->kv_cache_;
}

const std::vector<Gemma2MetalKVCacheLayer>& Gemma2MetalModelPlan::kv_cache() const noexcept {
    return storage_->kv_cache_;
}

const Gemma2MetalPrefillScratch& Gemma2MetalModelPlan::prefill_scratch() const noexcept {
    return storage_->prefill_scratch_;
}

const Gemma2MetalDecodeScratch& Gemma2MetalModelPlan::decode_scratch() const noexcept {
    return storage_->decode_scratch_;
}

const Gemma2MetalVerifierScratch& Gemma2MetalModelPlan::verifier_scratch() const noexcept {
    return storage_->verifier_scratch_;
}

bool Gemma2MetalModelPlan::has_pending_verification() const noexcept {
    return storage_ && storage_->verification_state_ != VerifierTransactionState::idle;
}

bool Gemma2MetalModelPlan::pending_verification_completed() const noexcept {
    return storage_ && (storage_->verification_state_ == VerifierTransactionState::completed
        || storage_->verification_state_ == VerifierTransactionState::provisional);
}

VerifierTransactionState Gemma2MetalModelPlan::verification_state() const noexcept {
    return storage_ ? storage_->verification_state_ : VerifierTransactionState::idle;
}

bool Gemma2MetalModelPlan::verification_poisoned() const noexcept {
    return storage_ && storage_->verification_state_ == VerifierTransactionState::poisoned;
}

std::uint32_t Gemma2MetalModelPlan::pending_verification_position() const {
    if (!has_pending_verification()) {
        throw std::logic_error("Gemma2 Metal plan has no pending verification");
    }
    return storage_->pending_verification_position_;
}

std::uint32_t Gemma2MetalModelPlan::pending_verification_row_count() const {
    if (!has_pending_verification()) {
        throw std::logic_error("Gemma2 Metal plan has no pending verification");
    }
    return storage_->pending_verification_row_count_;
}

std::uint32_t Gemma2MetalModelPlan::provisionally_committed_token_count() const noexcept {
    return storage_->provisionally_committed_token_count_;
}

void Gemma2MetalModelPlan::Storage::set_pending_verification(
    std::uint32_t cache_position,
    std::uint32_t row_count,
    VerifierTransactionState state
) {
    if (verification_state_ != VerifierTransactionState::idle) {
        throw std::logic_error("Gemma2 Metal plan already has a pending verification");
    }
    if (row_count == 0U || row_count > gemma2_metal_verifier_row_max
        || (state != VerifierTransactionState::encoded
            && state != VerifierTransactionState::submitted)) {
        throw std::invalid_argument("Gemma2 Metal pending verification extent or state is invalid");
    }
    verification_state_ = state;
    pending_verification_position_ = cache_position;
    pending_verification_row_count_ = row_count;
    provisionally_committed_token_count_ = 0U;
}

void Gemma2MetalModelPlan::Storage::mark_pending_verification_submitted() {
    if (verification_state_ != VerifierTransactionState::encoded) {
        throw std::logic_error("Gemma2 Metal encoded verification submit state is invalid");
    }
    verification_state_ = VerifierTransactionState::submitted;
}

void Gemma2MetalModelPlan::Storage::mark_pending_verification_completed() {
    if (verification_state_ != VerifierTransactionState::submitted) {
        throw std::logic_error("Gemma2 Metal pending verification completion state is invalid");
    }
    verification_state_ = VerifierTransactionState::completed;
}

void Gemma2MetalModelPlan::Storage::set_provisionally_committed_token_count(std::uint32_t token_count) {
    if (verification_state_ != VerifierTransactionState::completed
        || provisionally_committed_token_count_ != 0U
        || token_count == 0U || token_count > pending_verification_row_count_) {
        throw std::logic_error("Gemma2 Metal provisional verification state is invalid");
    }
    provisionally_committed_token_count_ = token_count;
    verification_state_ = VerifierTransactionState::provisional;
}

void Gemma2MetalModelPlan::Storage::clear_pending_verification() noexcept {
    // A submitted verifier whose wait did not complete has uncertain K/V and
    // backup contents. It is intentionally impossible to clear through the
    // ordinary transaction API; destruction is the only safe recovery.
    if (verification_state_ == VerifierTransactionState::submitted
        || verification_state_ == VerifierTransactionState::poisoned) return;
    verification_state_ = VerifierTransactionState::idle;
    pending_verification_position_ = 0U;
    pending_verification_row_count_ = 0U;
    provisionally_committed_token_count_ = 0U;
}

void Gemma2MetalModelPlan::Storage::poison_pending_verification() noexcept {
    verification_state_ = VerifierTransactionState::poisoned;
}

const Gemma2MetalSelectionWorkspace& Gemma2MetalModelPlan::selection_workspace() const noexcept {
    return storage_->selection_workspace_;
}

const std::vector<Gemma2MetalLayerState>& Gemma2MetalModelPlan::layers() const noexcept {
    return storage_->layers_;
}

const Gemma2DecodeScheduledWeightBytes& Gemma2MetalModelPlan::decode_scheduled_weight_bytes() const noexcept {
    return storage_->decode_scheduled_weight_bytes_;
}

std::size_t Gemma2MetalModelPlan::uploaded_weight_byte_count() const noexcept {
    return storage_->uploaded_weight_byte_count_;
}

std::size_t Gemma2MetalModelPlan::kv_cache_byte_count() const noexcept {
    return storage_->kv_cache_byte_count_;
}

std::size_t Gemma2MetalModelPlan::scratch_byte_count() const noexcept {
    return storage_->scratch_byte_count_ + (storage_->target_execution_workspace_
        ? storage_->target_execution_workspace_->allocated_byte_count() : 0U);
}

std::shared_ptr<Gemma2MetalTargetVerifierPlan>& Gemma2MetalModelPlan::target_execution_workspace() noexcept {
    return storage_->target_execution_workspace_;
}

std::array<std::uintptr_t, 4> Gemma2MetalModelPlan::allocation_identities() const noexcept {
    return {
        reinterpret_cast<std::uintptr_t>(&tensor::backend_buffer_storage(storage_->weights_.token_embedding_weight)),
        storage_->kv_cache_.empty() ? 0U : reinterpret_cast<std::uintptr_t>(
            &tensor::backend_buffer_storage(storage_->kv_cache_.front().key)),
        reinterpret_cast<std::uintptr_t>(&tensor::backend_buffer_storage(storage_->prefill_scratch_.normalized_hidden)),
        reinterpret_cast<std::uintptr_t>(&tensor::backend_buffer_storage(storage_->decode_scratch_.normalized_hidden)),
    };
}

Gemma2MetalSchedule::Gemma2MetalSchedule(
    std::vector<Gemma2MetalLayerSchedule> layers,
    Gemma2MetalSchedulePhase phase,
    Gemma2MetalScratchOwnership scratch_ownership,
    std::uint32_t token_count,
    std::uint32_t cache_position,
    std::uint32_t total_kv_token_count
)
    : layers_(std::move(layers)),
      phase_(phase),
      scratch_ownership_(scratch_ownership),
      token_count_(token_count),
      cache_position_(cache_position),
      total_kv_token_count_(total_kv_token_count) {}

runtime::BackendKind Gemma2MetalSchedule::backend() const noexcept {
    return gemma_backend;
}

Gemma2MetalSchedulePhase Gemma2MetalSchedule::phase() const noexcept {
    return phase_;
}

std::uint32_t Gemma2MetalSchedule::token_count() const noexcept {
    return token_count_;
}

std::uint32_t Gemma2MetalSchedule::cache_position() const noexcept {
    return cache_position_;
}

std::uint32_t Gemma2MetalSchedule::total_kv_token_count() const noexcept {
    return total_kv_token_count_;
}

Gemma2MetalScratchOwnership Gemma2MetalSchedule::scratch_ownership() const noexcept {
    return scratch_ownership_;
}

const std::vector<Gemma2MetalLayerSchedule>& Gemma2MetalSchedule::layers() const noexcept {
    return layers_;
}

namespace {

[[nodiscard]] Gemma2MetalModelPlan create_gemma2_metal_model_plan_impl(
    GemmaModelConfig config,
    const GemmaModelWeights& weights,
    GemmaOutputHeadPolicy policy,
    ResidentTargetPayloadReceipt* receipt
) {
    config.kv_precision = resolve_gemma_kv_precision(config.architecture, config.kv_precision);
    if (config.architecture == ModelArchitecture::gemma3) {
        validate_gemma3_model_weights(config, weights);
    }
    require_power_of_two_cache_layout(config);
    if (config.model_context_length != 0U && config.context_length > config.model_context_length) {
        throw std::invalid_argument("Gemma runtime context exceeds model context length");
    }
    require_metal_available();

    if (config.architecture == ModelArchitecture::gemma3) {
        // Fail before uploading weights when even immutable weights and durable
        // cache exceed this device's resident-work budget. Scratch and other
        // processes still require measured headroom; this is not a free-RAM test.
        std::size_t minimum_bytes = gemma3_resident_weight_bytes(config, weights);
        for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
            const auto rows = gemma_layer_uses_sliding_window(config, layer)
                ? config.sliding_window : config.context_length;
            const auto cache = kv_descriptor(config,
                {rows, config.attention_head_count_kv, config.attention_key_length});
            checked_add_bytes(minimum_bytes, cache.byte_count);
            checked_add_bytes(minimum_bytes, cache.byte_count);
        }
        if (const auto memory = tensor::backends::metal::memory_info()) {
            const auto budget = std::min(memory->recommended_working_set_bytes,
                memory->physical_memory_bytes);
            if (minimum_bytes > budget || memory->current_allocated_bytes > budget - minimum_bytes) {
                throw std::invalid_argument("Gemma 3 context capacity "
                    + std::to_string(config.context_length)
                    + " requires at least " + std::to_string(minimum_bytes)
                    + " resident bytes before scratch, exceeding the available engine Metal budget "
                    + std::to_string(budget > memory->current_allocated_bytes
                        ? budget - memory->current_allocated_bytes : 0U)
                    + "; select a smaller --context-size");
            }
        }
    }

    std::size_t uploaded_weight_byte_count = 0;
    std::size_t kv_cache_byte_count = 0;
    std::size_t scratch_byte_count = 0;

    const auto retained_target = is_retained_gemma2_9b_target(config, weights);
    const auto resolved_policy = resolve_gemma_output_head_policy(config, weights, policy);
    ResidentPayloadHasher resident_hasher(receipt != nullptr);
    // Gemma 2's Q6 V projections retain their original worker/reduction order.
    // Packing only Q4 preserves that target while sharing its dominant body
    // projections across R1/R2; each tensor still has one resident payload.
    auto resident_weights = upload_model_weights(
        weights, uploaded_weight_byte_count, resolved_policy, retained_target,
        resident_hasher, config.architecture == ModelArchitecture::gemma3
            ? ProjectionPacking::kquant_columns8
            : retained_target && resolved_policy == GemmaOutputHeadPolicy::q8_0
                ? ProjectionPacking::q4_columns8 : ProjectionPacking::canonical
    );
    auto kv_cache = allocate_kv_cache(config, kv_cache_byte_count);
    auto prefill_scratch = allocate_prefill_scratch(config, scratch_byte_count);
    auto decode_scratch = allocate_decode_scratch(config, scratch_byte_count);
    auto verifier_scratch = allocate_verifier_scratch(config, scratch_byte_count);
    auto selection_workspace = allocate_selection_workspace(config, scratch_byte_count);
    auto layers = make_layer_states(config);
    const auto scheduled_weight_bytes = decode_scheduled_weight_bytes(resident_weights, config);

    auto plan = Gemma2MetalModelPlan(
        config,
        std::move(resident_weights),
        std::move(kv_cache),
        std::move(prefill_scratch),
        std::move(decode_scratch),
        std::move(verifier_scratch),
        std::move(selection_workspace),
        std::move(layers),
        resolved_policy,
        scheduled_weight_bytes,
        uploaded_weight_byte_count,
        kv_cache_byte_count,
        scratch_byte_count
    );
    if (receipt != nullptr) *receipt = resident_hasher.finish();
    return plan;
}

}  // namespace

Gemma3MetalMemoryEstimate estimate_gemma3_metal_memory(
    const GemmaModelConfig& requested_config, const GemmaModelWeights& weights) {
    auto config = requested_config;
    config.kv_precision = resolve_gemma_kv_precision(config.architecture, config.kv_precision);
    if (config.architecture != ModelArchitecture::gemma3)
        throw std::invalid_argument("Gemma 3 memory accounting requires a Gemma 3 target");
    validate_gemma3_model_weights(config, weights);
    require_power_of_two_cache_layout(config);
    if (config.model_context_length && config.context_length > config.model_context_length)
        throw std::invalid_argument("Gemma runtime context exceeds model context length");
    Gemma3MetalMemoryEstimate result;
    result.weights = gemma3_resident_weight_bytes(config, weights);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto rows = gemma_layer_uses_sliding_window(config, layer) ? config.sliding_window : config.context_length;
        checked_add_bytes(result.kv_cache, kv_descriptor(config,
            {rows, config.attention_head_count_kv, config.attention_key_length}).byte_count);
        checked_add_bytes(result.kv_cache, kv_descriptor(config,
            {rows, config.attention_head_count_kv, config.attention_value_length}).byte_count);
    }
    const auto add = [&](std::initializer_list<std::uint64_t> dimensions, DType dtype = DType::f32) {
        checked_add_bytes(result.scratch, TensorDescriptor::from(Shape::from(dimensions), dtype).byte_count);
    };
    constexpr auto prompt = gemma2_metal_prefill_chunk_size;
    // Unique ordinary prefill allocations: aliases are deliberately counted once.
    add({config.context_length});
    add({3U, prompt, config.embedding_length});
    add({2U, prompt, config.attention_head_count, config.attention_key_length});
    add({prompt, config.attention_head_count_kv, config.attention_key_length});
    add({prompt, config.attention_head_count_kv, config.attention_value_length});
    add({prompt, config.attention_head_count_kv, config.attention_key_length}, gemma_kv_dtype(config));
    add({prompt, config.attention_head_count_kv, config.attention_value_length}, gemma_kv_dtype(config));
    add({prompt, config.attention_head_count, config.attention_value_length});
    add({2U, prompt, config.feed_forward_length});
    add({config.embedding_length});
    add({config.vocab_size});
    // The existing scalar and width-two workspaces own separate intermediates.
    for (const auto rows : {1U, 2U}) {
        add({rows});
        add({6U, rows, config.embedding_length});
        add({2U, rows, config.attention_head_count, config.attention_key_length});
        add({rows, config.attention_head_count_kv, config.attention_key_length});
        add({rows, config.attention_head_count_kv, config.attention_value_length});
        add({gemma2_decode_attention_max_partitions, gemma2_decode_attention_query_heads,
            gemma2_decode_attention_head_dimension + 2U});
        add({rows, config.attention_head_count, config.attention_value_length});
        add({3U, rows, config.feed_forward_length});
        add({rows, config.vocab_size});
    }
    add({config.block_count, 2U, config.attention_head_count_kv,
        std::uint64_t(config.attention_key_length) + config.attention_value_length}, gemma_kv_dtype(config));
    const auto tiles = (std::uint64_t(config.vocab_size) + gemma2_metal_selection_tile_size - 1U)
        / gemma2_metal_selection_tile_size;
    add({4U, tiles, token_sampler_max_base_candidates, sizeof(TokenLogit) / sizeof(float)});
    add({2U, 2U + token_sampler_max_candidates * sizeof(TokenLogit) / sizeof(float)});
    add({2U, token_sampler_max_repetition_window});
    checked_add_bytes(result.scratch, estimate_target_verifier_bytes(config, prompt));
    return result;
}

Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const EngineModel& model,
    GemmaOutputHeadPolicy policy,
    std::uint32_t context_length,
    GemmaKVPrecision kv_precision
) {
    const auto weights = bind_gemma_model_weights(model);
    auto config = model.gemma_config();
    config.kv_precision = resolve_gemma_kv_precision(config.architecture, kv_precision);
    if (context_length != 0U) {
        if (context_length > (config.model_context_length == 0U ? config.context_length : config.model_context_length)) {
            throw std::invalid_argument("Gemma runtime context exceeds model context length");
        }
        config.context_length = context_length;
    }
    return create_gemma2_metal_model_plan_impl(
        config, weights, policy, nullptr
    );
}

Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const EngineModel& model,
    GemmaOutputHeadPolicy policy,
    ResidentTargetPayloadReceipt& resident_receipt,
    std::uint32_t context_length,
    GemmaKVPrecision kv_precision
) {
    const auto weights = bind_gemma_model_weights(model);
    auto config = model.gemma_config();
    config.kv_precision = resolve_gemma_kv_precision(config.architecture, kv_precision);
    if (context_length != 0U) {
        if (context_length > (config.model_context_length == 0U ? config.context_length : config.model_context_length)) {
            throw std::invalid_argument("Gemma runtime context exceeds model context length");
        }
        config.context_length = context_length;
    }
    auto receipt = ResidentTargetPayloadReceipt {};
    auto plan = create_gemma2_metal_model_plan_impl(
        config, weights, policy, &receipt
    );
    resident_receipt = receipt;
    return plan;
}

Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const GemmaModelConfig& config,
    const GemmaModelWeights& weights,
    GemmaOutputHeadPolicy policy
) {
    return create_gemma2_metal_model_plan_impl(config, weights, policy, nullptr);
}

Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t prompt_token_count
) {
    return create_gemma2_metal_prefill_schedule(plan, prompt_token_count, 0);
}

Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t token_count,
    std::uint32_t cache_position
) {
    require_prefill_range(plan.config(), token_count, cache_position);

    const auto total_kv_token_count = cache_position + token_count;
    auto layers = make_layer_schedules(plan, total_kv_token_count);

    return Gemma2MetalSchedule(
        std::move(layers),
        Gemma2MetalSchedulePhase::prefill,
        Gemma2MetalScratchOwnership::resident_prefill,
        token_count,
        cache_position,
        total_kv_token_count
    );
}

Gemma2MetalSchedule create_gemma2_metal_decode_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t cached_token_count
) {
    require_decode_cached_token_count(plan.config(), cached_token_count);

    const auto token_count = 1U;
    const auto total_kv_token_count = cached_token_count + token_count;
    auto layers = make_layer_schedules(plan, total_kv_token_count);

    return Gemma2MetalSchedule(
        std::move(layers),
        Gemma2MetalSchedulePhase::decode,
        Gemma2MetalScratchOwnership::resident_decode,
        token_count,
        cached_token_count,
        total_kv_token_count
    );
}

Gemma2MetalSchedule create_gemma2_metal_verifier_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t cached_token_count
) {
    if (plan.config().context_length < 2U || cached_token_count == 0U
        || cached_token_count > plan.config().context_length - 2U) {
        throw std::invalid_argument("Gemma2 Metal verifier schedule requires room for two tokens");
    }
    auto layers = make_layer_schedules(plan, cached_token_count + 1U);
    for (std::size_t index = 0; index < layers.size(); ++index) {
        layers[index].second_attention_token_count = layer_attention_token_count(
            plan.config(), plan.layers()[index], cached_token_count + 2U
        );
    }
    return Gemma2MetalSchedule(
        std::move(layers),
        Gemma2MetalSchedulePhase::verify,
        Gemma2MetalScratchOwnership::resident_verifier,
        2U,
        cached_token_count,
        cached_token_count + 2U
    );
}

}  // namespace local_gguf_engine::engine
