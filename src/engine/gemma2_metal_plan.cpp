#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/shape.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

using runtime::BackendKind;
using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;

constexpr auto gemma_backend = BackendKind::metal;

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
    const Gemma2ModelConfig& config
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

[[nodiscard]] tensor::BackendBuffer upload_weight(tensor::TensorView view, std::size_t& uploaded_byte_count) {
    checked_add_bytes(uploaded_byte_count, view.bytes().size());
    return tensor::copy_tensor_view_to_backend(view, gemma_backend);
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

[[nodiscard]] TensorDescriptor f16_descriptor(std::initializer_list<std::uint64_t> dimensions) {
    return TensorDescriptor::from(Shape::from(dimensions), DType::f16);
}

[[nodiscard]] bool is_power_of_two(std::uint32_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

void require_power_of_two_cache_layout(const Gemma2ModelConfig& config) {
    if (!is_power_of_two(config.context_length) || !is_power_of_two(config.sliding_window)) {
        throw std::invalid_argument(
            "Gemma2 Metal KV cache requires power-of-two context and sliding-window lengths"
        );
    }
}

[[nodiscard]] std::uint64_t attention_query_width(const Gemma2ModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count) * config.attention_key_length;
}

[[nodiscard]] std::uint64_t attention_key_width(const Gemma2ModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count_kv) * config.attention_key_length;
}

[[nodiscard]] std::uint64_t attention_value_width(const Gemma2ModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count_kv) * config.attention_value_length;
}

[[nodiscard]] std::uint64_t attention_output_width(const Gemma2ModelConfig& config) {
    return static_cast<std::uint64_t>(config.attention_head_count) * config.attention_value_length;
}

[[nodiscard]] bool is_retained_gemma2_9b_target(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights
);

[[nodiscard]] Gemma2MetalAttentionWeightBuffers upload_attention_weights(
    const Gemma2AttentionWeights& weights,
    std::size_t& uploaded_byte_count
) {
    return Gemma2MetalAttentionWeightBuffers {
        .attention_norm_weight = upload_weight(weights.attention_norm_weight, uploaded_byte_count),
        .post_attention_norm_weight = upload_weight(weights.post_attention_norm_weight, uploaded_byte_count),
        .query_weight = upload_weight(weights.query_weight, uploaded_byte_count),
        .key_weight = upload_weight(weights.key_weight, uploaded_byte_count),
        .value_weight = upload_weight(weights.value_weight, uploaded_byte_count),
        .output_weight = upload_weight(weights.output_weight, uploaded_byte_count),
    };
}

[[nodiscard]] Gemma2MetalFeedForwardWeightBuffers upload_feed_forward_weights(
    const Gemma2FeedForwardWeights& weights,
    std::size_t& uploaded_byte_count,
    bool requantize_q6_down
) {
    auto down_weight = [&]() {
        if (requantize_q6_down && weights.down_weight.descriptor().dtype == DType::q6_k) {
            auto quantized = tensor::requantize_q6_k_gguf_weight_q4_k(weights.down_weight);
            checked_add_bytes(uploaded_byte_count, quantized.descriptor().byte_count);
            return tensor::copy_tensor_to_backend(quantized, gemma_backend);
        }
        return upload_weight(weights.down_weight, uploaded_byte_count);
    }();
    return Gemma2MetalFeedForwardWeightBuffers {
        .feed_forward_norm_weight = upload_weight(weights.feed_forward_norm_weight, uploaded_byte_count),
        .post_feed_forward_norm_weight = upload_weight(weights.post_feed_forward_norm_weight, uploaded_byte_count),
        .gate_weight = upload_weight(weights.gate_weight, uploaded_byte_count),
        .up_weight = upload_weight(weights.up_weight, uploaded_byte_count),
        .down_weight = std::move(down_weight),
    };
}

[[nodiscard]] std::vector<Gemma2MetalBlockWeightBuffers> upload_block_weights(
    const Gemma2ModelWeights& weights,
    std::size_t& uploaded_byte_count,
    bool requantize_q6_down
) {
    std::vector<Gemma2MetalBlockWeightBuffers> blocks;
    blocks.reserve(weights.blocks.size());

    for (const auto& block : weights.blocks) {
        blocks.push_back(Gemma2MetalBlockWeightBuffers {
            .attention = upload_attention_weights(block.attention, uploaded_byte_count),
            .feed_forward = upload_feed_forward_weights(
                block.feed_forward, uploaded_byte_count, requantize_q6_down
            ),
        });
    }

    return blocks;
}

[[nodiscard]] Gemma2MetalModelWeightBuffers upload_model_weights(
    const Gemma2ModelWeights& weights,
    std::size_t& uploaded_byte_count,
    Gemma2OutputHeadPolicy output_head_policy,
    bool retained_target
) {
    auto token_embedding_weight = upload_weight(weights.token_embedding_weight, uploaded_byte_count);
    auto output_weight = [&]() {
        if (output_head_policy == Gemma2OutputHeadPolicy::q8_0) {
            auto quantized = tensor::quantize_f16_gguf_weight_q8_0(weights.output_weight);
            checked_add_bytes(uploaded_byte_count, quantized.descriptor().byte_count);
            return tensor::copy_tensor_to_backend(quantized, gemma_backend);
        }
        return weights.output_weight.bytes().data() == weights.token_embedding_weight.bytes().data()
            ? token_embedding_weight
            : upload_weight(weights.output_weight, uploaded_byte_count);
    }();

    return Gemma2MetalModelWeightBuffers {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = upload_weight(weights.output_norm_weight, uploaded_byte_count),
        .output_weight = std::move(output_weight),
        .blocks = upload_block_weights(weights, uploaded_byte_count, retained_target),
    };
}

[[nodiscard]] bool is_retained_gemma2_9b_target(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights
) {
    const auto& embedding = weights.token_embedding_weight;
    const auto& output = weights.output_weight;
    return config.uses_tied_output_weight
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
    const Gemma2ModelConfig& config,
    std::size_t& allocated_byte_count
) {
    std::vector<Gemma2MetalKVCacheLayer> layers;
    layers.reserve(config.block_count);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        const auto rows = gemma2_layer_uses_sliding_window(layer)
            ? config.sliding_window
            : config.context_length;
        layers.push_back(Gemma2MetalKVCacheLayer {
            .key = allocate_buffer(f16_descriptor({
                rows, config.attention_head_count_kv, config.attention_key_length,
            }), allocated_byte_count),
            .value = allocate_buffer(f16_descriptor({
                rows, config.attention_head_count_kv, config.attention_value_length,
            }), allocated_byte_count),
            .row_capacity = rows,
            .row_mask = rows - 1U,
        });
    }

    return layers;
}

[[nodiscard]] Gemma2MetalDecodeScratch allocate_decode_scratch(
    const Gemma2ModelConfig& config,
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

[[nodiscard]] Gemma2MetalPrefillScratch allocate_prefill_scratch(
    const Gemma2ModelConfig& config,
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
    const auto attention_score_workspace = allocate_buffer(
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
    const Gemma2ModelConfig& config,
    std::size_t& allocated_byte_count
) {
    constexpr std::uint64_t candidate_f32_elements = sizeof(TokenLogit) / sizeof(float);
    const auto tile_count = (
        static_cast<std::uint64_t>(config.vocab_size) + gemma2_metal_selection_tile_size - 1U
    ) / gemma2_metal_selection_tile_size;
    const auto run_elements = tile_count * token_sampler_max_base_candidates * candidate_f32_elements;
    constexpr auto result_header_f32_elements = std::uint64_t {2};
    constexpr auto result_elements = result_header_f32_elements
        + token_sampler_max_candidates * candidate_f32_elements;
    return Gemma2MetalSelectionWorkspace {
        .candidate_runs_a = allocate_buffer(f32_descriptor({run_elements}), allocated_byte_count),
        .candidate_runs_b = allocate_buffer(f32_descriptor({run_elements}), allocated_byte_count),
        .compact_result = allocate_buffer(f32_descriptor({result_elements}), allocated_byte_count),
        .history_token_staging = allocate_buffer(
            f32_descriptor({token_sampler_max_repetition_window}), allocated_byte_count
        ),
    };
}

[[nodiscard]] std::vector<Gemma2MetalLayerState> make_layer_states(const Gemma2ModelConfig& config) {
    std::vector<Gemma2MetalLayerState> layers;
    layers.reserve(config.block_count);
    for (std::uint32_t layer_index = 0; layer_index < config.block_count; ++layer_index) {
        layers.push_back(Gemma2MetalLayerState {
            .layer_index = layer_index,
            .uses_sliding_window = gemma2_layer_uses_sliding_window(layer_index),
        });
    }

    return layers;
}

[[nodiscard]] std::uint32_t layer_attention_token_count(
    const Gemma2ModelConfig& config,
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
    const Gemma2ModelConfig& config,
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

void require_decode_cached_token_count(const Gemma2ModelConfig& config, std::uint32_t cached_token_count) {
    if (cached_token_count == 0) {
        throw std::invalid_argument("Gemma2 Metal decode schedule cached token count must be positive");
    }
    if (cached_token_count >= config.context_length) {
        throw std::invalid_argument("Gemma2 Metal decode schedule cached token count must leave room for one token");
    }
}

}  // namespace

const char* gemma2_output_head_policy_name(Gemma2OutputHeadPolicy policy) noexcept {
    switch (policy) {
        case Gemma2OutputHeadPolicy::automatic: return "automatic";
        case Gemma2OutputHeadPolicy::f16: return "f16";
        case Gemma2OutputHeadPolicy::q8_0: return "q8_0";
    }
    return "unknown";
}

Gemma2MetalModelPlan::Gemma2MetalModelPlan(
    Gemma2ModelConfig config,
    Gemma2MetalModelWeightBuffers weights,
    std::vector<Gemma2MetalKVCacheLayer> kv_cache,
    Gemma2MetalPrefillScratch prefill_scratch,
    Gemma2MetalDecodeScratch decode_scratch,
    Gemma2MetalSelectionWorkspace selection_workspace,
    std::vector<Gemma2MetalLayerState> layers,
    Gemma2OutputHeadPolicy output_head_policy,
    Gemma2DecodeScheduledWeightBytes decode_scheduled_weight_bytes,
    std::size_t uploaded_weight_byte_count,
    std::size_t kv_cache_byte_count,
    std::size_t scratch_byte_count
)
    : config_(config),
      weights_(std::move(weights)),
      kv_cache_(std::move(kv_cache)),
      prefill_scratch_(std::move(prefill_scratch)),
      decode_scratch_(std::move(decode_scratch)),
      selection_workspace_(std::move(selection_workspace)),
      layers_(std::move(layers)),
      output_head_policy_(output_head_policy),
      decode_scheduled_weight_bytes_(decode_scheduled_weight_bytes),
      uploaded_weight_byte_count_(uploaded_weight_byte_count),
      kv_cache_byte_count_(kv_cache_byte_count),
      scratch_byte_count_(scratch_byte_count) {}

runtime::BackendKind Gemma2MetalModelPlan::backend() const noexcept {
    return gemma_backend;
}

const Gemma2ModelConfig& Gemma2MetalModelPlan::config() const noexcept {
    return config_;
}

Gemma2OutputHeadPolicy Gemma2MetalModelPlan::output_head_policy() const noexcept {
    return output_head_policy_;
}

const Gemma2MetalModelWeightBuffers& Gemma2MetalModelPlan::weights() const noexcept {
    return weights_;
}

std::vector<Gemma2MetalKVCacheLayer>& Gemma2MetalModelPlan::kv_cache() noexcept {
    return kv_cache_;
}

const std::vector<Gemma2MetalKVCacheLayer>& Gemma2MetalModelPlan::kv_cache() const noexcept {
    return kv_cache_;
}

const Gemma2MetalPrefillScratch& Gemma2MetalModelPlan::prefill_scratch() const noexcept {
    return prefill_scratch_;
}

const Gemma2MetalDecodeScratch& Gemma2MetalModelPlan::decode_scratch() const noexcept {
    return decode_scratch_;
}

const Gemma2MetalSelectionWorkspace& Gemma2MetalModelPlan::selection_workspace() const noexcept {
    return selection_workspace_;
}

const std::vector<Gemma2MetalLayerState>& Gemma2MetalModelPlan::layers() const noexcept {
    return layers_;
}

const Gemma2DecodeScheduledWeightBytes& Gemma2MetalModelPlan::decode_scheduled_weight_bytes() const noexcept {
    return decode_scheduled_weight_bytes_;
}

std::size_t Gemma2MetalModelPlan::uploaded_weight_byte_count() const noexcept {
    return uploaded_weight_byte_count_;
}

std::size_t Gemma2MetalModelPlan::kv_cache_byte_count() const noexcept {
    return kv_cache_byte_count_;
}

std::size_t Gemma2MetalModelPlan::scratch_byte_count() const noexcept {
    return scratch_byte_count_;
}

std::array<std::uintptr_t, 4> Gemma2MetalModelPlan::allocation_identities() const noexcept {
    return {
        reinterpret_cast<std::uintptr_t>(&tensor::backend_buffer_storage(weights_.token_embedding_weight)),
        kv_cache_.empty() ? 0U : reinterpret_cast<std::uintptr_t>(
            &tensor::backend_buffer_storage(kv_cache_.front().key)),
        reinterpret_cast<std::uintptr_t>(&tensor::backend_buffer_storage(prefill_scratch_.normalized_hidden)),
        reinterpret_cast<std::uintptr_t>(&tensor::backend_buffer_storage(decode_scratch_.normalized_hidden)),
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

Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const EngineModel& model,
    Gemma2OutputHeadPolicy policy
) {
    const auto weights = bind_gemma2_model_weights(model);
    return create_gemma2_metal_model_plan(model.gemma2_config(), weights, policy);
}

Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights,
    Gemma2OutputHeadPolicy policy
) {
    require_metal_available();
    require_power_of_two_cache_layout(config);

    std::size_t uploaded_weight_byte_count = 0;
    std::size_t kv_cache_byte_count = 0;
    std::size_t scratch_byte_count = 0;

    const auto retained_target = is_retained_gemma2_9b_target(config, weights);
    const auto resolved_policy = policy == Gemma2OutputHeadPolicy::automatic
        ? (retained_target
            ? Gemma2OutputHeadPolicy::q8_0
            : Gemma2OutputHeadPolicy::f16)
        : policy;
    auto resident_weights = upload_model_weights(
        weights, uploaded_weight_byte_count, resolved_policy, retained_target
    );
    auto kv_cache = allocate_kv_cache(config, kv_cache_byte_count);
    auto prefill_scratch = allocate_prefill_scratch(config, scratch_byte_count);
    auto decode_scratch = allocate_decode_scratch(config, scratch_byte_count);
    auto selection_workspace = allocate_selection_workspace(config, scratch_byte_count);
    auto layers = make_layer_states(config);
    const auto scheduled_weight_bytes = decode_scheduled_weight_bytes(resident_weights, config);

    return Gemma2MetalModelPlan(
        config,
        std::move(resident_weights),
        std::move(kv_cache),
        std::move(prefill_scratch),
        std::move(decode_scratch),
        std::move(selection_workspace),
        std::move(layers),
        resolved_policy,
        scheduled_weight_bytes,
        uploaded_weight_byte_count,
        kv_cache_byte_count,
        scratch_byte_count
    );
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

}  // namespace local_gguf_engine::engine
