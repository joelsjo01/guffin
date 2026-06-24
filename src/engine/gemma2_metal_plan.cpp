#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
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
    std::size_t& uploaded_byte_count
) {
    return Gemma2MetalFeedForwardWeightBuffers {
        .feed_forward_norm_weight = upload_weight(weights.feed_forward_norm_weight, uploaded_byte_count),
        .post_feed_forward_norm_weight = upload_weight(weights.post_feed_forward_norm_weight, uploaded_byte_count),
        .gate_weight = upload_weight(weights.gate_weight, uploaded_byte_count),
        .up_weight = upload_weight(weights.up_weight, uploaded_byte_count),
        .down_weight = upload_weight(weights.down_weight, uploaded_byte_count),
    };
}

[[nodiscard]] std::vector<Gemma2MetalBlockWeightBuffers> upload_block_weights(
    const Gemma2ModelWeights& weights,
    std::size_t& uploaded_byte_count
) {
    std::vector<Gemma2MetalBlockWeightBuffers> blocks;
    blocks.reserve(weights.blocks.size());

    for (const auto& block : weights.blocks) {
        blocks.push_back(Gemma2MetalBlockWeightBuffers {
            .attention = upload_attention_weights(block.attention, uploaded_byte_count),
            .feed_forward = upload_feed_forward_weights(block.feed_forward, uploaded_byte_count),
        });
    }

    return blocks;
}

[[nodiscard]] Gemma2MetalModelWeightBuffers upload_model_weights(
    const Gemma2ModelWeights& weights,
    std::size_t& uploaded_byte_count
) {
    auto token_embedding_weight = upload_weight(weights.token_embedding_weight, uploaded_byte_count);
    auto output_weight = weights.output_weight.bytes().data() == weights.token_embedding_weight.bytes().data()
        ? token_embedding_weight
        : upload_weight(weights.output_weight, uploaded_byte_count);

    return Gemma2MetalModelWeightBuffers {
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = upload_weight(weights.output_norm_weight, uploaded_byte_count),
        .output_weight = std::move(output_weight),
        .blocks = upload_block_weights(weights, uploaded_byte_count),
    };
}

[[nodiscard]] std::vector<Gemma2MetalKVCacheLayer> allocate_kv_cache(
    const Gemma2ModelConfig& config,
    std::size_t& allocated_byte_count
) {
    const auto key_descriptor = f32_descriptor({
        config.context_length,
        config.attention_head_count_kv,
        config.attention_key_length,
    });
    const auto value_descriptor = f32_descriptor({
        config.context_length,
        config.attention_head_count_kv,
        config.attention_value_length,
    });

    std::vector<Gemma2MetalKVCacheLayer> layers;
    layers.reserve(config.block_count);
    for (std::uint32_t layer = 0; layer < config.block_count; ++layer) {
        layers.push_back(Gemma2MetalKVCacheLayer {
            .key = allocate_buffer(key_descriptor, allocated_byte_count),
            .value = allocate_buffer(value_descriptor, allocated_byte_count),
        });
    }

    return layers;
}

[[nodiscard]] Gemma2MetalDecodeScratch allocate_decode_scratch(
    const Gemma2ModelConfig& config,
    std::size_t& allocated_byte_count
) {
    return Gemma2MetalDecodeScratch {
        .normalized_hidden = allocate_buffer(f32_descriptor({1, config.embedding_length}), allocated_byte_count),
        .query = allocate_buffer(f32_descriptor({1, attention_query_width(config)}), allocated_byte_count),
        .key = allocate_buffer(f32_descriptor({1, attention_key_width(config)}), allocated_byte_count),
        .value = allocate_buffer(f32_descriptor({1, attention_value_width(config)}), allocated_byte_count),
        .attention_scores = allocate_buffer(
            f32_descriptor({1, config.attention_head_count, config.context_length}),
            allocated_byte_count
        ),
        .attention_probabilities = allocate_buffer(
            f32_descriptor({1, config.attention_head_count, config.context_length}),
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
        .logits = allocate_buffer(f32_descriptor({1, config.vocab_size}), allocated_byte_count),
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

[[nodiscard]] Gemma2MetalScheduleScratch schedule_scratch_from_decode_scratch(
    const Gemma2MetalDecodeScratch& scratch
) {
    return Gemma2MetalScheduleScratch {
        .normalized_hidden = scratch.normalized_hidden.descriptor(),
        .query = scratch.query.descriptor(),
        .key = scratch.key.descriptor(),
        .value = scratch.value.descriptor(),
        .attention_scores = scratch.attention_scores.descriptor(),
        .attention_probabilities = scratch.attention_probabilities.descriptor(),
        .attention_output = scratch.attention_output.descriptor(),
        .projected_attention_output = scratch.projected_attention_output.descriptor(),
        .post_attention_residual = scratch.post_attention_residual.descriptor(),
        .normalized_feed_forward_input = scratch.normalized_feed_forward_input.descriptor(),
        .feed_forward_gate = scratch.feed_forward_gate.descriptor(),
        .feed_forward_up = scratch.feed_forward_up.descriptor(),
        .feed_forward_gated = scratch.feed_forward_gated.descriptor(),
        .logits = scratch.logits.descriptor(),
    };
}

[[nodiscard]] std::uint32_t layer_attention_token_count(
    const Gemma2ModelConfig& config,
    const Gemma2MetalLayerState& layer,
    std::uint32_t total_kv_token_count
) {
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

[[nodiscard]] std::uint32_t max_attention_token_count(std::span<const Gemma2MetalLayerSchedule> layers) {
    std::uint32_t max_tokens = 0;
    for (const auto& layer : layers) {
        max_tokens = std::max(max_tokens, layer.attention_token_count);
    }

    return max_tokens;
}

[[nodiscard]] Gemma2MetalScheduleScratch make_phase_local_scratch(
    const Gemma2ModelConfig& config,
    std::uint32_t token_count,
    std::uint32_t attention_token_count
) {
    return Gemma2MetalScheduleScratch {
        .normalized_hidden = f32_descriptor({token_count, config.embedding_length}),
        .query = f32_descriptor({token_count, attention_query_width(config)}),
        .key = f32_descriptor({token_count, attention_key_width(config)}),
        .value = f32_descriptor({token_count, attention_value_width(config)}),
        .attention_scores = f32_descriptor({token_count, config.attention_head_count, attention_token_count}),
        .attention_probabilities = f32_descriptor({token_count, config.attention_head_count, attention_token_count}),
        .attention_output = f32_descriptor({token_count, attention_output_width(config)}),
        .projected_attention_output = f32_descriptor({token_count, config.embedding_length}),
        .post_attention_residual = f32_descriptor({token_count, config.embedding_length}),
        .normalized_feed_forward_input = f32_descriptor({token_count, config.embedding_length}),
        .feed_forward_gate = f32_descriptor({token_count, config.feed_forward_length}),
        .feed_forward_up = f32_descriptor({token_count, config.feed_forward_length}),
        .feed_forward_gated = f32_descriptor({token_count, config.feed_forward_length}),
        .logits = f32_descriptor({token_count, config.vocab_size}),
    };
}

[[nodiscard]] std::size_t schedule_scratch_byte_count(const Gemma2MetalScheduleScratch& scratch) {
    std::size_t total = 0;
    checked_add_bytes(total, scratch.normalized_hidden.byte_count);
    checked_add_bytes(total, scratch.query.byte_count);
    checked_add_bytes(total, scratch.key.byte_count);
    checked_add_bytes(total, scratch.value.byte_count);
    checked_add_bytes(total, scratch.attention_scores.byte_count);
    checked_add_bytes(total, scratch.attention_probabilities.byte_count);
    checked_add_bytes(total, scratch.attention_output.byte_count);
    checked_add_bytes(total, scratch.projected_attention_output.byte_count);
    checked_add_bytes(total, scratch.post_attention_residual.byte_count);
    checked_add_bytes(total, scratch.normalized_feed_forward_input.byte_count);
    checked_add_bytes(total, scratch.feed_forward_gate.byte_count);
    checked_add_bytes(total, scratch.feed_forward_up.byte_count);
    checked_add_bytes(total, scratch.feed_forward_gated.byte_count);
    checked_add_bytes(total, scratch.logits.byte_count);

    return total;
}

void require_prefill_token_count(const Gemma2ModelConfig& config, std::uint32_t prompt_token_count) {
    if (prompt_token_count == 0) {
        throw std::invalid_argument("Gemma2 Metal prefill schedule token count must be positive");
    }
    if (prompt_token_count > config.context_length) {
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

Gemma2MetalModelPlan::Gemma2MetalModelPlan(
    Gemma2ModelConfig config,
    Gemma2MetalModelWeightBuffers weights,
    std::vector<Gemma2MetalKVCacheLayer> kv_cache,
    Gemma2MetalDecodeScratch decode_scratch,
    std::vector<Gemma2MetalLayerState> layers,
    std::size_t uploaded_weight_byte_count,
    std::size_t kv_cache_byte_count,
    std::size_t scratch_byte_count
)
    : config_(config),
      weights_(std::move(weights)),
      kv_cache_(std::move(kv_cache)),
      decode_scratch_(std::move(decode_scratch)),
      layers_(std::move(layers)),
      uploaded_weight_byte_count_(uploaded_weight_byte_count),
      kv_cache_byte_count_(kv_cache_byte_count),
      scratch_byte_count_(scratch_byte_count) {}

runtime::BackendKind Gemma2MetalModelPlan::backend() const noexcept {
    return gemma_backend;
}

const Gemma2ModelConfig& Gemma2MetalModelPlan::config() const noexcept {
    return config_;
}

const Gemma2MetalModelWeightBuffers& Gemma2MetalModelPlan::weights() const noexcept {
    return weights_;
}

const std::vector<Gemma2MetalKVCacheLayer>& Gemma2MetalModelPlan::kv_cache() const noexcept {
    return kv_cache_;
}

const Gemma2MetalDecodeScratch& Gemma2MetalModelPlan::decode_scratch() const noexcept {
    return decode_scratch_;
}

const std::vector<Gemma2MetalLayerState>& Gemma2MetalModelPlan::layers() const noexcept {
    return layers_;
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

Gemma2MetalSchedule::Gemma2MetalSchedule(
    Gemma2ModelConfig config,
    Gemma2MetalModelWeightBuffers weights,
    std::vector<Gemma2MetalKVCacheLayer> kv_cache,
    Gemma2MetalDecodeScratch resident_decode_scratch,
    std::vector<Gemma2MetalLayerSchedule> layers,
    Gemma2MetalSchedulePhase phase,
    Gemma2MetalScratchOwnership scratch_ownership,
    Gemma2MetalScheduleScratch scratch,
    std::uint32_t token_count,
    std::uint32_t cache_position,
    std::uint32_t total_kv_token_count,
    std::size_t resident_weight_byte_count,
    std::size_t resident_kv_cache_byte_count,
    std::size_t resident_decode_scratch_byte_count,
    std::size_t phase_scratch_byte_count
)
    : config_(config),
      weights_(std::move(weights)),
      kv_cache_(std::move(kv_cache)),
      resident_decode_scratch_(std::move(resident_decode_scratch)),
      layers_(std::move(layers)),
      phase_(phase),
      scratch_ownership_(scratch_ownership),
      scratch_(std::move(scratch)),
      token_count_(token_count),
      cache_position_(cache_position),
      total_kv_token_count_(total_kv_token_count),
      resident_weight_byte_count_(resident_weight_byte_count),
      resident_kv_cache_byte_count_(resident_kv_cache_byte_count),
      resident_decode_scratch_byte_count_(resident_decode_scratch_byte_count),
      phase_scratch_byte_count_(phase_scratch_byte_count) {}

runtime::BackendKind Gemma2MetalSchedule::backend() const noexcept {
    return gemma_backend;
}

const Gemma2ModelConfig& Gemma2MetalSchedule::config() const noexcept {
    return config_;
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

const Gemma2MetalScheduleScratch& Gemma2MetalSchedule::scratch() const noexcept {
    return scratch_;
}

const Gemma2MetalModelWeightBuffers& Gemma2MetalSchedule::weights() const noexcept {
    return weights_;
}

const std::vector<Gemma2MetalKVCacheLayer>& Gemma2MetalSchedule::kv_cache() const noexcept {
    return kv_cache_;
}

const Gemma2MetalDecodeScratch& Gemma2MetalSchedule::resident_decode_scratch() const noexcept {
    return resident_decode_scratch_;
}

const std::vector<Gemma2MetalLayerSchedule>& Gemma2MetalSchedule::layers() const noexcept {
    return layers_;
}

std::size_t Gemma2MetalSchedule::resident_weight_byte_count() const noexcept {
    return resident_weight_byte_count_;
}

std::size_t Gemma2MetalSchedule::resident_kv_cache_byte_count() const noexcept {
    return resident_kv_cache_byte_count_;
}

std::size_t Gemma2MetalSchedule::resident_decode_scratch_byte_count() const noexcept {
    return resident_decode_scratch_byte_count_;
}

std::size_t Gemma2MetalSchedule::phase_scratch_byte_count() const noexcept {
    return phase_scratch_byte_count_;
}

Gemma2MetalModelPlan create_gemma2_metal_model_plan(const EngineModel& model) {
    const auto weights = bind_gemma2_model_weights(model);
    return create_gemma2_metal_model_plan(model.gemma2_config(), weights);
}

Gemma2MetalModelPlan create_gemma2_metal_model_plan(
    const Gemma2ModelConfig& config,
    const Gemma2ModelWeights& weights
) {
    require_metal_available();

    std::size_t uploaded_weight_byte_count = 0;
    std::size_t kv_cache_byte_count = 0;
    std::size_t scratch_byte_count = 0;

    auto resident_weights = upload_model_weights(weights, uploaded_weight_byte_count);
    auto kv_cache = allocate_kv_cache(config, kv_cache_byte_count);
    auto decode_scratch = allocate_decode_scratch(config, scratch_byte_count);
    auto layers = make_layer_states(config);

    return Gemma2MetalModelPlan(
        config,
        std::move(resident_weights),
        std::move(kv_cache),
        std::move(decode_scratch),
        std::move(layers),
        uploaded_weight_byte_count,
        kv_cache_byte_count,
        scratch_byte_count
    );
}

Gemma2MetalSchedule create_gemma2_metal_prefill_schedule(
    const Gemma2MetalModelPlan& plan,
    std::uint32_t prompt_token_count
) {
    require_prefill_token_count(plan.config(), prompt_token_count);

    auto layers = make_layer_schedules(plan, prompt_token_count);
    auto scratch = make_phase_local_scratch(
        plan.config(),
        prompt_token_count,
        max_attention_token_count(std::span<const Gemma2MetalLayerSchedule>(layers))
    );
    const auto phase_scratch_bytes = schedule_scratch_byte_count(scratch);

    return Gemma2MetalSchedule(
        plan.config(),
        plan.weights(),
        plan.kv_cache(),
        plan.decode_scratch(),
        std::move(layers),
        Gemma2MetalSchedulePhase::prefill,
        Gemma2MetalScratchOwnership::phase_local,
        std::move(scratch),
        prompt_token_count,
        0,
        prompt_token_count,
        plan.uploaded_weight_byte_count(),
        plan.kv_cache_byte_count(),
        plan.scratch_byte_count(),
        phase_scratch_bytes
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
    auto scratch = schedule_scratch_from_decode_scratch(plan.decode_scratch());
    const auto phase_scratch_bytes = schedule_scratch_byte_count(scratch);

    return Gemma2MetalSchedule(
        plan.config(),
        plan.weights(),
        plan.kv_cache(),
        plan.decode_scratch(),
        std::move(layers),
        Gemma2MetalSchedulePhase::decode,
        Gemma2MetalScratchOwnership::resident_decode,
        std::move(scratch),
        token_count,
        cached_token_count,
        total_kv_token_count,
        plan.uploaded_weight_byte_count(),
        plan.kv_cache_byte_count(),
        plan.scratch_byte_count(),
        phase_scratch_bytes
    );
}

}  // namespace local_gguf_engine::engine
