#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/shape.hpp"

#include <initializer_list>
#include <limits>
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
        .attention_output = allocate_buffer(f32_descriptor({1, attention_output_width(config)}), allocated_byte_count),
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

}  // namespace local_gguf_engine::engine
