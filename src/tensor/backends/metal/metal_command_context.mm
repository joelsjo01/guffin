#include "metal_command_context.hpp"

#include "metal_buffer.hpp"
#include "metal_context.hpp"

#include "local_gguf_engine/tensor/dtype.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

struct MatmulDimensions {
    std::uint32_t rows;
    std::uint32_t shared;
    std::uint32_t columns;
};

struct EmbeddingDimensions {
    std::uint32_t tokens;
    std::uint32_t hidden;
    std::uint32_t vocab;
    std::uint32_t token_offset;
    float scale;
};

struct ElementwiseDimensions {
    std::uint32_t elements;
};

struct SoftcapDimensions {
    std::uint32_t elements;
    float softcap;
};

struct RMSNormDimensions {
    std::uint32_t rows;
    std::uint32_t hidden;
    std::uint32_t input_row_start;
    float epsilon;
};

struct RoPEByHeadsDimensions {
    std::uint32_t rows;
    std::uint32_t head_count;
    std::uint32_t head_dimension;
    std::uint32_t rope_dimension_count;
    std::uint32_t position_start;
    float frequency_base;
};

struct Gemma2KVCacheStoreDimensions {
    std::uint32_t rows;
    std::uint32_t head_count;
    std::uint32_t head_dimension;
    std::uint32_t rope_dimension_count;
    std::uint32_t position_start;
    std::uint32_t cache_row_mask;
    float frequency_base;
};

struct GroupedAttentionScoreDimensions {
    std::uint32_t query_rows;
    std::uint32_t key_rows;
    std::uint32_t head_count;
    std::uint32_t kv_head_count;
    std::uint32_t key_length;
    std::uint32_t attention_token_count;
    std::uint32_t query_position_start;
    std::uint32_t cache_row_mask;
    float scale;
    float logit_softcap;
};

struct GroupedAttentionValueDimensions {
    std::uint32_t query_rows;
    std::uint32_t value_rows;
    std::uint32_t head_count;
    std::uint32_t kv_head_count;
    std::uint32_t value_length;
    std::uint32_t attention_token_count;
    std::uint32_t query_position_start;
    std::uint32_t cache_row_mask;
};

struct Gemma2DecodeAttentionDimensions {
    std::uint32_t cache_row_mask;
    std::uint32_t first_position;
    std::uint32_t attention_token_count;
    std::uint32_t partition_count;
    std::uint32_t query_head_count;
    std::uint32_t kv_head_count;
    std::uint32_t key_dimension;
    std::uint32_t value_dimension;
    float scale;
    float logit_softcap;
};

static_assert(sizeof(Gemma2DecodeAttentionDimensions) == 40U);

struct SoftmaxDimensions {
    std::uint32_t rows;
    std::uint32_t columns;
};

struct TokenCandidateSelectionDimensions {
    std::uint32_t columns;
    std::uint32_t retain_count;
    std::uint32_t run_count;
    std::uint32_t history_count;
    float softcap;
};

struct TokenCandidateMergeDimensions {
    std::uint32_t retain_count;
    std::uint32_t input_run_count;
};

struct TokenCandidateResultHeader {
    std::uint32_t status;
    std::uint32_t count;
};

[[nodiscard]] const MetalBackendBufferStorage& storage(const BackendBuffer& buffer) {
    return require_metal_backend_buffer(backend_buffer_storage(buffer));
}

[[nodiscard]] std::uint32_t u32(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("Metal resident execution ") + name + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

void require_capacity(const BackendBuffer& buffer, std::uint64_t bytes, const char* name) {
    if (bytes > buffer.descriptor().byte_count) {
        throw std::invalid_argument(std::string("Metal resident execution ") + name + " buffer is too small");
    }
}

[[nodiscard]] id<MTLComputePipelineState> f16_matmul_pipeline(
    MatmulPipelineState& state,
    DType dtype
) {
    switch (dtype) {
        case DType::f16:
            return state.matmul_f16_simd_pipeline;
        default:
            throw std::invalid_argument("Metal resident execution matmul weight dtype is unsupported");
    }
}

}  // namespace

std::uint32_t gemma2_decode_attention_partition_count(
    std::uint32_t attention_token_count
) noexcept {
    // Keep the direct path through eight 32-token blocks; beyond that, doubling
    // partitions with context bounds per-workgroup streaming while keeping the
    // reduction fan-out within the 32 persistent partials. Medium/long real-path
    // measurements own any future threshold adjustment.
    if (attention_token_count <= 256U) return 1U;
    if (attention_token_count <= 512U) return 2U;
    if (attention_token_count <= 1024U) return 4U;
    if (attention_token_count <= 2048U) return 8U;
    if (attention_token_count <= 4096U) return 16U;
    return 32U;
}

struct MetalCommandContext::Impl {
    MatmulPipelineState* state = nullptr;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> active_compute_encoder = nil;
    MetalCommandStats stats;
    bool concurrent_compute = false;
    bool submitted = false;
    bool waited = false;

    void end_compute_encoding() noexcept {
        if (active_compute_encoder != nil) {
            [active_compute_encoder endEncoding];
            active_compute_encoder = nil;
        }
    }

    [[nodiscard]] id<MTLComputeCommandEncoder> compute_encoder(const char* label) {
        if (submitted) {
            throw std::logic_error("Metal resident execution command buffer was already submitted");
        }
        if (active_compute_encoder != nil) {
            return active_compute_encoder;
        }
        active_compute_encoder = concurrent_compute
            ? [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
            : [command_buffer computeCommandEncoder];
        if (active_compute_encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal resident execution compute encoder creation failed: ") + label
            );
        }
        active_compute_encoder.label = [NSString stringWithUTF8String:label];
        ++stats.compute_encoder_count;
        return active_compute_encoder;
    }

    ~Impl() {
        end_compute_encoding();
    }
};

MetalCommandContext::MetalCommandContext(std::string_view label, bool concurrent_compute)
    : impl_(std::make_unique<Impl>()) {
    impl_->concurrent_compute = concurrent_compute;
    impl_->state = &shared_matmul_pipeline_state();
    impl_->command_buffer = [impl_->state->queue commandBuffer];
    if (impl_->command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal resident execution command buffer creation failed"
        );
    }
    const std::string owned_label(label);
    impl_->command_buffer.label = [NSString stringWithUTF8String:owned_label.c_str()];
    impl_->stats.command_buffer_count = 1;
}

MetalCommandContext::~MetalCommandContext() = default;
MetalCommandContext::MetalCommandContext(MetalCommandContext&&) noexcept = default;
MetalCommandContext& MetalCommandContext::operator=(MetalCommandContext&&) noexcept = default;

void MetalCommandContext::write_u32(const BackendBuffer& destination, std::uint32_t value) {
    require_capacity(destination, sizeof(value), "u32 staging");
    std::memcpy([storage(destination).buffer() contents], &value, sizeof(value));
}

void MetalCommandContext::write_u32s(
    const BackendBuffer& destination,
    std::uint32_t element_offset,
    std::span<const std::uint32_t> values
) {
    const auto byte_offset = static_cast<std::uint64_t>(element_offset) * sizeof(std::uint32_t);
    const auto byte_count = static_cast<std::uint64_t>(values.size()) * sizeof(std::uint32_t);
    require_capacity(destination, byte_offset + byte_count, "u32 staging");
    std::memcpy(
        static_cast<std::byte*>([storage(destination).buffer() contents]) + byte_offset,
        values.data(),
        static_cast<std::size_t>(byte_count)
    );
}

void MetalCommandContext::encode_embedding(
    const BackendBuffer& token_ids,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    float scale,
    std::uint32_t token_offset,
    std::uint32_t token_count
) {
    const auto& weight_storage = storage(weight);
    const auto hidden = weight.descriptor().shape.dimension(0);
    const auto vocab = weight.descriptor().shape.dimension(1);
    const auto tokens = token_count == 0U ? output.descriptor().shape.dimension(0) : token_count;
    require_capacity(token_ids, (static_cast<std::uint64_t>(token_offset) + tokens) * sizeof(std::uint32_t), "embedding tokens");
    require_capacity(output, tokens * hidden * sizeof(float), "embedding output");
    const EmbeddingDimensions dims {
        .tokens = u32(tokens, "embedding tokens"),
        .hidden = u32(hidden, "embedding hidden"),
        .vocab = u32(vocab, "embedding vocab"),
        .token_offset = token_offset,
        .scale = scale,
    };
    auto encoder = impl_->compute_encoder("Gemma2 embedding");
    [encoder setComputePipelineState:impl_->state->embedding_pipeline];
    [encoder setBuffer:storage(token_ids).buffer() offset:0 atIndex:0];
    [encoder setBuffer:weight_storage.buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((hidden + 15U) / 16U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_matmul(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t row_count,
    std::uint64_t output_byte_offset
) {
    const auto rows = row_count == 0U ? input.descriptor().shape.dimension(0) : row_count;
    const auto shared = input.descriptor().shape.dimension(1);
    const auto columns = weight.descriptor().shape.dimension(1);
    if (shared != weight.descriptor().shape.dimension(0)) {
        throw std::invalid_argument("Metal resident execution matmul inner dimensions do not match");
    }
    const auto output_bytes = rows * columns * sizeof(float);
    if (output_byte_offset > std::numeric_limits<std::uint64_t>::max() - output_bytes) {
        throw std::invalid_argument("Metal resident execution matmul output byte range overflows");
    }
    require_capacity(output, output_byte_offset + output_bytes, "matmul output");
    const MatmulDimensions dims {
        .rows = u32(rows, "matmul rows"),
        .shared = u32(shared, "matmul shared"),
        .columns = u32(columns, "matmul columns"),
    };
    auto encoder = impl_->compute_encoder("Gemma2 matmul");
    const bool quantized = weight.descriptor().dtype == DType::q4_k
        || weight.descriptor().dtype == DType::q6_k;
    QuantizedMatmulDispatch quantized_dispatch;
    F16MatmulDispatch f16_dispatch;
    id<MTLComputePipelineState> pipeline = nil;
    if (quantized) {
        quantized_dispatch = select_quantized_matmul_dispatch(
            *impl_->state, weight.descriptor().dtype, rows, shared, columns
        );
        pipeline = quantized_dispatch.pipeline;
    } else {
        pipeline = f16_matmul_pipeline(*impl_->state, weight.descriptor().dtype);
        f16_dispatch = select_f16_matmul_dispatch(*impl_->state, rows, columns);
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(weight).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:static_cast<NSUInteger>(output_byte_offset) atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    if (quantized) {
        [encoder dispatchThreadgroups:quantized_dispatch.threadgroups
                  threadsPerThreadgroup:quantized_dispatch.threads_per_threadgroup];
    } else {
        [encoder dispatchThreadgroups:f16_dispatch.threadgroups
                  threadsPerThreadgroup:f16_dispatch.threads_per_threadgroup];
    }
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_rms_norm(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    float epsilon,
    std::uint32_t row_count,
    std::uint32_t input_row_start
) {
    const auto rows = row_count == 0U ? input.descriptor().shape.dimension(0) : row_count;
    const auto hidden = input.descriptor().shape.dimension(1);
    if (static_cast<std::uint64_t>(input_row_start) + rows > input.descriptor().shape.dimension(0)) {
        throw std::invalid_argument("Metal resident execution RMSNorm input row range is invalid");
    }
    require_capacity(output, rows * hidden * sizeof(float), "RMSNorm output");
    const RMSNormDimensions dims {
        .rows = u32(rows, "RMSNorm rows"),
        .hidden = u32(hidden, "RMSNorm hidden"),
        .input_row_start = input_row_start,
        .epsilon = epsilon,
    };
    auto encoder = impl_->compute_encoder("Gemma2 RMSNorm");
    [encoder setComputePipelineState:impl_->state->rms_norm_pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(weight).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_rms_norm_add(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& residual,
    const BackendBuffer& output,
    float epsilon
) {
    const auto hidden = input.descriptor().shape.dimension(1);
    if ((hidden & 3U) != 0U) {
        throw std::invalid_argument("Metal fused RMSNorm-add hidden width must be divisible by four");
    }
    require_capacity(weight, hidden * sizeof(float), "fused RMSNorm-add weight");
    require_capacity(residual, hidden * sizeof(float), "fused RMSNorm-add residual");
    require_capacity(output, hidden * sizeof(float), "fused RMSNorm-add output");
    const RMSNormDimensions dims {
        .rows = 1U,
        .hidden = u32(hidden, "fused RMSNorm-add hidden"),
        .input_row_start = 0U,
        .epsilon = epsilon,
    };
    auto encoder = impl_->compute_encoder("Gemma2 fused RMSNorm-add");
    [encoder setComputePipelineState:impl_->state->rms_norm_add_pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(weight).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(residual).buffer() offset:0 atIndex:2];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:3];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_rope(
    const BackendBuffer& input,
    const BackendBuffer& output,
    std::uint32_t position_start,
    std::uint32_t head_count,
    std::uint32_t head_dimension,
    std::uint32_t rope_dimension_count,
    float frequency_base,
    std::uint32_t row_count,
    std::uint64_t output_byte_offset
) {
    const auto rows = row_count == 0U ? input.descriptor().shape.dimension(0) : row_count;
    const auto columns = static_cast<std::uint64_t>(head_count) * head_dimension;
    const auto output_bytes = rows * columns * sizeof(float);
    if (output_byte_offset > std::numeric_limits<std::uint64_t>::max() - output_bytes) {
        throw std::invalid_argument("Metal resident execution RoPE output byte range overflows");
    }
    require_capacity(output, output_byte_offset + output_bytes, "RoPE output");
    const RoPEByHeadsDimensions dims {
        .rows = u32(rows, "RoPE rows"),
        .head_count = head_count,
        .head_dimension = head_dimension,
        .rope_dimension_count = rope_dimension_count,
        .position_start = position_start,
        .frequency_base = frequency_base,
    };
    auto encoder = impl_->compute_encoder("Gemma2 RoPE");
    [encoder setComputePipelineState:impl_->state->rope_neox_by_heads_pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(output).buffer() offset:static_cast<NSUInteger>(output_byte_offset) atIndex:1];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((columns + 15U) / 16U),
                                      static_cast<NSUInteger>((rows + 15U) / 16U),
                                      1
                                  )
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_gemma2_kv_cache_store(
    const BackendBuffer& raw_key,
    const BackendBuffer& raw_value,
    const BackendBuffer& cache_key,
    const BackendBuffer& cache_value,
    std::uint32_t position_start,
    std::uint32_t head_count,
    std::uint32_t head_dimension,
    std::uint32_t rope_dimension_count,
    float frequency_base,
    std::uint32_t cache_row_mask,
    std::uint32_t token_count
) {
    const auto rows = token_count == 0U ? raw_key.descriptor().shape.dimension(0) : token_count;
    const auto width = static_cast<std::uint64_t>(head_count) * head_dimension;
    const auto require_raw = [&](const BackendBuffer& buffer, const char* name) {
        const auto& descriptor = buffer.descriptor();
        if (descriptor.dtype != DType::f32 || descriptor.shape.rank() != 2U
            || descriptor.shape.dimension(0) < rows || descriptor.shape.dimension(1) != width) {
            throw std::invalid_argument(std::string("Metal Gemma 2 KV store ") + name + " layout is invalid");
        }
        (void)storage(buffer);
    };
    const auto require_cache = [&](const BackendBuffer& buffer, const char* name) {
        const auto& descriptor = buffer.descriptor();
        if (descriptor.dtype != DType::f16 || descriptor.shape.rank() != 3U
            || descriptor.shape.dimension(0) != static_cast<std::uint64_t>(cache_row_mask) + 1U
            || descriptor.shape.dimension(1) != head_count
            || descriptor.shape.dimension(2) != head_dimension) {
            throw std::invalid_argument(std::string("Metal Gemma 2 KV store ") + name + " layout is invalid");
        }
        (void)storage(buffer);
    };
    require_raw(raw_key, "raw-key");
    require_raw(raw_value, "raw-value");
    require_cache(cache_key, "key-cache");
    require_cache(cache_value, "value-cache");
    if (rows == 0U || (cache_row_mask & (cache_row_mask + 1U)) != 0U
        || rope_dimension_count == 0U || rope_dimension_count > head_dimension
        || (rope_dimension_count & 1U) != 0U || !std::isfinite(frequency_base) || frequency_base <= 0.0F) {
        throw std::invalid_argument("Metal Gemma 2 KV store dimensions are invalid");
    }
    const Gemma2KVCacheStoreDimensions dims {
        .rows = u32(rows, "KV store rows"),
        .head_count = head_count,
        .head_dimension = head_dimension,
        .rope_dimension_count = rope_dimension_count,
        .position_start = position_start,
        .cache_row_mask = cache_row_mask,
        .frequency_base = frequency_base,
    };
    auto encoder = impl_->compute_encoder("Gemma2 F16 KV store");
    [encoder setComputePipelineState:impl_->state->gemma2_kv_cache_store_pipeline];
    [encoder setBuffer:storage(raw_key).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(raw_value).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(cache_key).buffer() offset:0 atIndex:2];
    [encoder setBuffer:storage(cache_value).buffer() offset:0 atIndex:3];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((width + 255U) / 256U),
                                      static_cast<NSUInteger>(rows),
                                      1
                                  )
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_attention_scores(
    const BackendBuffer& query,
    const BackendBuffer& key_cache,
    const BackendBuffer& output,
    std::uint32_t query_position_start,
    std::uint32_t head_count,
    std::uint32_t kv_head_count,
    std::uint32_t key_length,
    std::uint32_t attention_token_count,
    float scale,
    float logit_softcap,
    std::uint32_t cache_row_mask,
    std::uint32_t query_row_count
) {
    const auto query_rows =
        query_row_count == 0U ? query.descriptor().shape.dimension(0) : query_row_count;
    const auto key_rows = key_cache.descriptor().shape.dimension(0);
    if (key_cache.descriptor().dtype != DType::f16
        || key_cache.descriptor().shape.rank() != 3U
        || key_rows != static_cast<std::uint64_t>(cache_row_mask) + 1U) {
        throw std::invalid_argument("Metal Gemma 2 attention key-cache layout is invalid");
    }
    (void)storage(key_cache);
    const auto output_elements = query_rows * head_count * attention_token_count;
    require_capacity(output, output_elements * sizeof(float), "attention score output");
    const GroupedAttentionScoreDimensions dims {
        .query_rows = u32(query_rows, "attention query rows"),
        .key_rows = u32(key_rows, "attention key rows"),
        .head_count = head_count,
        .kv_head_count = kv_head_count,
        .key_length = key_length,
        .attention_token_count = attention_token_count,
        .query_position_start = query_position_start,
        .cache_row_mask = cache_row_mask,
        .scale = scale,
        .logit_softcap = logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma2 attention scores");
    [encoder setComputePipelineState:impl_->state->gemma2_prefill_attention_scores_pipeline];
    [encoder setBuffer:storage(query).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(key_cache).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((attention_token_count + 7U) / 8U),
                                      static_cast<NSUInteger>((head_count + 7U) / 8U),
                                      static_cast<NSUInteger>((query_rows + 1U) / 2U)
                                  )
              threadsPerThreadgroup:MTLSizeMake(8, 8, 2)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_softmax(
    const BackendBuffer& input,
    const BackendBuffer& output,
    std::uint32_t rows,
    std::uint32_t columns
) {
    require_capacity(input, static_cast<std::uint64_t>(rows) * columns * sizeof(float), "softmax input");
    require_capacity(output, static_cast<std::uint64_t>(rows) * columns * sizeof(float), "softmax output");
    const SoftmaxDimensions dims {.rows = rows, .columns = columns};
    auto encoder = impl_->compute_encoder("Gemma2 softmax");
    [encoder setComputePipelineState:impl_->state->softmax_pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:1];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_attention_values(
    const BackendBuffer& probabilities,
    const BackendBuffer& value_cache,
    const BackendBuffer& output,
    std::uint32_t query_position_start,
    std::uint32_t head_count,
    std::uint32_t kv_head_count,
    std::uint32_t value_length,
    std::uint32_t attention_token_count,
    std::uint32_t cache_row_mask,
    std::uint32_t query_row_count
) {
    const auto query_rows = query_row_count == 0U
        ? probabilities.descriptor().shape.dimension(0)
        : query_row_count;
    const auto value_rows = value_cache.descriptor().shape.dimension(0);
    if (value_cache.descriptor().dtype != DType::f16
        || value_cache.descriptor().shape.rank() != 3U
        || value_rows != static_cast<std::uint64_t>(cache_row_mask) + 1U) {
        throw std::invalid_argument("Metal Gemma 2 attention value-cache layout is invalid");
    }
    (void)storage(value_cache);
    require_capacity(
        output,
        static_cast<std::uint64_t>(query_rows) * head_count * value_length * sizeof(float),
        "attention value output"
    );
    const GroupedAttentionValueDimensions dims {
        .query_rows = u32(query_rows, "attention query rows"),
        .value_rows = u32(value_rows, "attention value rows"),
        .head_count = head_count,
        .kv_head_count = kv_head_count,
        .value_length = value_length,
        .attention_token_count = attention_token_count,
        .query_position_start = query_position_start,
        .cache_row_mask = cache_row_mask,
    };
    auto encoder = impl_->compute_encoder("Gemma2 attention values");
    [encoder setComputePipelineState:impl_->state->gemma2_prefill_attention_values_pipeline];
    [encoder setBuffer:storage(probabilities).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(value_cache).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((value_length + 7U) / 8U),
                                      static_cast<NSUInteger>((head_count + 7U) / 8U),
                                      static_cast<NSUInteger>((query_rows + 1U) / 2U)
                                  )
              threadsPerThreadgroup:MTLSizeMake(8, 8, 2)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_gemma2_decode_attention(
    const BackendBuffer& query,
    const BackendBuffer& key_cache,
    const BackendBuffer& value_cache,
    const BackendBuffer& output,
    const BackendBuffer& partial_output,
    const BackendBuffer& partial_maxima,
    const BackendBuffer& partial_denominators,
    std::uint32_t cache_position,
    std::uint32_t attention_token_count,
    float scale,
    float logit_softcap,
    std::uint32_t cache_row_mask
) {
    constexpr std::uint32_t dimension = 256U;
    constexpr std::uint32_t max_partitions = 32U;
    const auto require_f32_shape = [](const BackendBuffer& buffer,
                                      std::initializer_list<std::uint64_t> dimensions,
                                      const char* name) {
        const auto& descriptor = buffer.descriptor();
        const auto expected = tensor::TensorDescriptor::from(
            tensor::Shape::from(dimensions),
            DType::f32
        );
        if (descriptor != expected) {
            throw std::invalid_argument(
                std::string("Metal Gemma 2 decode attention ") + name + " layout is invalid"
            );
        }
        (void)storage(buffer);
    };

    const auto& query_descriptor = query.descriptor();
    if (query_descriptor.dtype != DType::f32
        || query_descriptor.shape.rank() != 2U
        || query_descriptor.shape.dimension(0) != 1U
        || query_descriptor.shape.dimension(1) % dimension != 0U) {
        throw std::invalid_argument("Metal Gemma 2 decode attention query layout is invalid");
    }
    const auto query_heads = u32(
        query_descriptor.shape.dimension(1) / dimension,
        "attention query heads"
    );
    const auto& key_descriptor = key_cache.descriptor();
    const auto& value_descriptor = value_cache.descriptor();
    if (key_descriptor.dtype != DType::f16
        || key_descriptor.shape.rank() != 3U
        || key_descriptor.shape.dimension(2) != dimension) {
        throw std::invalid_argument("Metal Gemma 2 decode attention key-cache layout is invalid");
    }
    const auto kv_heads = u32(key_descriptor.shape.dimension(1), "attention KV heads");
    if (query_heads == 0U || query_heads > 16U || kv_heads == 0U || query_heads != kv_heads * 2U) {
        throw std::invalid_argument("Metal Gemma 2 decode attention requires paired-query GQA");
    }
    if (value_descriptor.dtype != DType::f16
        || value_descriptor.shape.rank() != 3U
        || value_descriptor.shape.dimension(0) != key_descriptor.shape.dimension(0)
        || value_descriptor.shape.dimension(1) != kv_heads
        || value_descriptor.shape.dimension(2) != dimension) {
        throw std::invalid_argument("Metal Gemma 2 decode attention value-cache layout is invalid");
    }
    (void)storage(key_cache);
    (void)storage(value_cache);
    (void)storage(query);
    require_f32_shape(output, {1U, static_cast<std::uint64_t>(query_heads) * dimension}, "output");
    if (partial_output.descriptor().dtype != DType::f32
        || partial_output.descriptor().byte_count
            < static_cast<std::uint64_t>(max_partitions) * query_heads * dimension * sizeof(float)) {
        throw std::invalid_argument("Metal Gemma 2 decode attention partial-output buffer is too small");
    }
    if (partial_maxima.descriptor().dtype != DType::f32
        || partial_maxima.descriptor().byte_count
            < static_cast<std::uint64_t>(max_partitions) * query_heads * sizeof(float)) {
        throw std::invalid_argument("Metal Gemma 2 decode attention partial-maxima buffer is too small");
    }
    if (partial_denominators.descriptor().dtype != DType::f32
        || partial_denominators.descriptor().byte_count
            < static_cast<std::uint64_t>(max_partitions) * query_heads * sizeof(float)) {
        throw std::invalid_argument("Metal Gemma 2 decode attention partial-denominators buffer is too small");
    }
    (void)storage(partial_output);
    (void)storage(partial_maxima);
    (void)storage(partial_denominators);
    const auto cache_rows = u32(key_descriptor.shape.dimension(0), "attention cache rows");
    if (attention_token_count == 0U) {
        throw std::invalid_argument("Metal Gemma 2 decode attention token count must not be zero");
    }
    if (cache_rows != cache_row_mask + 1U
        || (cache_row_mask & (cache_row_mask + 1U)) != 0U
        || attention_token_count > cache_rows) {
        throw std::invalid_argument("Metal Gemma 2 decode attention cache mask is invalid");
    }
    if (attention_token_count > cache_position + 1U) {
        throw std::invalid_argument("Metal Gemma 2 decode attention window exceeds populated cache");
    }
    if (!std::isfinite(scale) || !std::isfinite(logit_softcap) || logit_softcap <= 0.0F) {
        throw std::invalid_argument("Metal Gemma 2 decode attention scale or softcap is invalid");
    }

    const auto partitions = gemma2_decode_attention_partition_count(attention_token_count);
    const Gemma2DecodeAttentionDimensions dims {
        .cache_row_mask = cache_row_mask,
        .first_position = cache_position + 1U - attention_token_count,
        .attention_token_count = attention_token_count,
        .partition_count = partitions,
        .query_head_count = query_heads,
        .kv_head_count = kv_heads,
        .key_dimension = dimension,
        .value_dimension = dimension,
        .scale = scale,
        .logit_softcap = logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma2 fused decode attention");
    [encoder setComputePipelineState:impl_->state->gemma2_decode_attention_pipeline];
    [encoder setBuffer:storage(query).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(key_cache).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(value_cache).buffer() offset:0 atIndex:2];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:3];
    [encoder setBuffer:storage(partial_output).buffer() offset:0 atIndex:4];
    [encoder setBuffer:storage(partial_maxima).buffer() offset:0 atIndex:5];
    [encoder setBuffer:storage(partial_denominators).buffer() offset:0 atIndex:6];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(kv_heads, partitions, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++impl_->stats.compute_dispatch_count;

    if (partitions > 1U) {
        encode_memory_barrier();
        encoder = impl_->compute_encoder("Gemma2 fused decode attention reduction");
        [encoder setComputePipelineState:impl_->state->gemma2_decode_attention_reduce_pipeline];
        [encoder setBuffer:storage(partial_output).buffer() offset:0 atIndex:0];
        [encoder setBuffer:storage(partial_maxima).buffer() offset:0 atIndex:1];
        [encoder setBuffer:storage(partial_denominators).buffer() offset:0 atIndex:2];
        [encoder setBuffer:storage(output).buffer() offset:0 atIndex:3];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(kv_heads, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
    }
}

void MetalCommandContext::encode_add(
    const BackendBuffer& left,
    const BackendBuffer& right,
    const BackendBuffer& output,
    std::uint64_t element_count
) {
    const auto elements = element_count == 0U ? left.descriptor().element_count : element_count;
    require_capacity(right, elements * sizeof(float), "add right");
    require_capacity(output, elements * sizeof(float), "add output");
    const ElementwiseDimensions dims {.elements = u32(elements, "add elements")};
    auto encoder = impl_->compute_encoder("Gemma2 add");
    [encoder setComputePipelineState:impl_->state->add_pipeline];
    [encoder setBuffer:storage(left).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(right).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((elements + 255U) / 256U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_silu_multiply(
    const BackendBuffer& gate,
    const BackendBuffer& up,
    const BackendBuffer& output,
    std::uint64_t element_count
) {
    const auto elements = element_count == 0U ? gate.descriptor().element_count : element_count;
    require_capacity(up, elements * sizeof(float), "SiLU up");
    require_capacity(output, elements * sizeof(float), "SiLU output");
    const ElementwiseDimensions dims {.elements = u32(elements, "SiLU elements")};
    auto encoder = impl_->compute_encoder("Gemma2 SiLU multiply");
    [encoder setComputePipelineState:impl_->state->silu_multiply_pipeline];
    [encoder setBuffer:storage(gate).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(up).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((elements + 255U) / 256U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_gelu_tanh_multiply(
    const BackendBuffer& gate,
    const BackendBuffer& up,
    const BackendBuffer& output,
    std::uint64_t element_count
) {
    const auto elements = element_count == 0U ? gate.descriptor().element_count : element_count;
    require_capacity(up, elements * sizeof(float), "GELU-tanh up");
    require_capacity(output, elements * sizeof(float), "GELU-tanh output");
    const ElementwiseDimensions dims {.elements = u32(elements, "GELU-tanh elements")};
    auto encoder = impl_->compute_encoder("Gemma2 GELU-tanh multiply");
    [encoder setComputePipelineState:impl_->state->gelu_tanh_multiply_pipeline];
    [encoder setBuffer:storage(gate).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(up).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((elements + 255U) / 256U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_softcap(
    const BackendBuffer& input,
    const BackendBuffer& output,
    float softcap,
    std::uint64_t element_count
) {
    const auto elements = element_count == 0U ? input.descriptor().element_count : element_count;
    require_capacity(output, elements * sizeof(float), "softcap output");
    const SoftcapDimensions dims {.elements = u32(elements, "softcap elements"), .softcap = softcap};
    auto encoder = impl_->compute_encoder("Gemma2 final softcap");
    [encoder setComputePipelineState:impl_->state->softcap_pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(output).buffer() offset:0 atIndex:1];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((elements + 255U) / 256U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

std::uint32_t MetalCommandContext::encode_token_candidates(
    const BackendBuffer& logits,
    const engine::Gemma2MetalSelectionWorkspace& workspace,
    const engine::TokenSelectionRequest& request,
    float softcap
) {
    if (request.top_k == 0U || request.top_k > engine::token_sampler_max_top_k
        || request.history_token_count > engine::token_sampler_max_repetition_window
        || request.base_candidate_count != request.top_k + request.history_token_count) {
        throw std::invalid_argument("Metal token-candidate request is outside product bounds");
    }
    if (!std::isfinite(softcap) || softcap <= 0.0F) {
        throw std::invalid_argument("Metal token-candidate softcap must be finite and positive");
    }
    const auto columns64 = logits.descriptor().shape.dimension(logits.descriptor().shape.rank() - 1U);
    const auto columns = u32(columns64, "token-candidate columns");
    const auto retain_count = std::min(columns, request.base_candidate_count);
    const auto run_count = (columns + engine::gemma2_metal_selection_tile_size - 1U)
        / engine::gemma2_metal_selection_tile_size;
    const TokenCandidateSelectionDimensions dims {
        .columns = columns,
        .retain_count = retain_count,
        .run_count = run_count,
        .history_count = request.history_token_count,
        .softcap = softcap,
    };
    const std::array<std::uint32_t, 2> empty_header {0U, 0U};
    write_u32s(workspace.compact_result, 0U, empty_header);
    if (request.history_token_count > 0U) {
        write_u32s(
            workspace.history_token_staging,
            0U,
            std::span<const std::uint32_t>(request.history_token_ids).first(request.history_token_count)
        );
    }

    if (retain_count == 1U && request.history_token_count == 0U) {
        auto encoder = impl_->compute_encoder("Gemma2 compact token top-1");
        [encoder setComputePipelineState:impl_->state->token_candidate_top1_pipeline];
        [encoder setBuffer:storage(logits).buffer() offset:0 atIndex:0];
        [encoder setBuffer:storage(workspace.compact_result).buffer() offset:0 atIndex:1];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        return 1U;
    }

    auto encoder = impl_->compute_encoder("Gemma2 compact token candidates");
    [encoder setComputePipelineState:impl_->state->token_candidate_tile_pipeline];
    [encoder setBuffer:storage(logits).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(workspace.candidate_runs_a).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(workspace.compact_result).buffer() offset:0 atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(run_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
    std::uint32_t selector_dispatches = 1U;

    auto input_is_a = true;
    auto input_run_count = run_count;
    while (input_run_count > 1U) {
        encode_memory_barrier();
        const auto output_run_count = (input_run_count + 1U) / 2U;
        const TokenCandidateMergeDimensions merge_dims {
            .retain_count = retain_count,
            .input_run_count = input_run_count,
        };
        encoder = impl_->compute_encoder("Gemma2 compact token merge");
        [encoder setComputePipelineState:impl_->state->token_candidate_merge_pipeline];
        [encoder setBuffer:storage(input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b).buffer()
                    offset:0 atIndex:0];
        [encoder setBuffer:storage(input_is_a ? workspace.candidate_runs_b : workspace.candidate_runs_a).buffer()
                    offset:0 atIndex:1];
        [encoder setBytes:&merge_dims length:sizeof(merge_dims) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(output_run_count, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        ++selector_dispatches;
        input_run_count = output_run_count;
        input_is_a = !input_is_a;
    }

    encode_memory_barrier();
    encoder = impl_->compute_encoder("Gemma2 compact token result");
    [encoder setComputePipelineState:impl_->state->token_candidate_finalize_pipeline];
    [encoder setBuffer:storage(logits).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b).buffer()
                offset:0 atIndex:1];
    [encoder setBuffer:storage(workspace.history_token_staging).buffer() offset:0 atIndex:2];
    [encoder setBuffer:storage(workspace.compact_result).buffer() offset:0 atIndex:3];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(retain_count + request.history_token_count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
    return selector_dispatches + 1U;
}

void MetalCommandContext::encode_memory_barrier() {
    auto encoder = impl_->compute_encoder("Gemma2 dependency barrier");
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++impl_->stats.memory_barrier_count;
}

MetalCommandStats MetalCommandContext::commit() {
    if (impl_->submitted) {
        throw std::logic_error("Metal resident command buffer was already submitted");
    }
    impl_->end_compute_encoding();
    impl_->submitted = true;
    [impl_->command_buffer commit];
    ++impl_->stats.commit_count;
    return impl_->stats;
}

MetalCommandStats MetalCommandContext::wait_until_completed() {
    if (!impl_->submitted) {
        throw std::logic_error("Metal resident command buffer must be committed before waiting");
    }
    if (impl_->waited) {
        throw std::logic_error("Metal resident command buffer was already waited");
    }
    impl_->waited = true;
    [impl_->command_buffer waitUntilCompleted];
    ++impl_->stats.synchronous_wait_count;
    return completion_stats();
}

MetalCommandStats MetalCommandContext::completion_stats() const {
    if (!impl_->submitted || [impl_->command_buffer status] < MTLCommandBufferStatusCompleted) {
        throw std::logic_error("Metal resident command buffer has not completed");
    }
    if ([impl_->command_buffer status] == MTLCommandBufferStatusError) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal resident execution command failed: "
                + ns_error_message([impl_->command_buffer error], "unknown error")
        );
    }

    const CFTimeInterval start = impl_->command_buffer.GPUStartTime;
    const CFTimeInterval end = impl_->command_buffer.GPUEndTime;
    if (end >= start && start > 0.0) {
        constexpr double ns_per_second = 1'000'000'000.0;
        impl_->stats.gpu_duration_ns = static_cast<std::uint64_t>((end - start) * ns_per_second);
    }
    return impl_->stats;
}

MetalCommandStats MetalCommandContext::commit_and_wait() {
    static_cast<void>(commit());
    return wait_until_completed();
}

engine::CompactTokenCandidates MetalCommandContext::read_token_candidates(
    const engine::Gemma2MetalSelectionWorkspace& workspace
) const {
    if (!impl_->submitted) {
        throw std::logic_error("Metal token-candidate result cannot be read before completion");
    }
    const auto* bytes = static_cast<const std::byte*>([storage(workspace.compact_result).buffer() contents]);
    TokenCandidateResultHeader header {};
    std::memcpy(&header, bytes, sizeof(header));
    if (header.status != 0U) {
        throw std::invalid_argument(
            (header.status & 2U) != 0U
                ? "Metal token-candidate history token is outside vocabulary"
                : "Metal token-candidate logit must be finite after softcap"
        );
    }
    if (header.count == 0U || header.count > engine::token_sampler_max_candidates) {
        throw std::runtime_error("Metal token-candidate result count is invalid");
    }
    engine::CompactTokenCandidates result;
    result.count = header.count;
    result.readback_bytes = header.count * sizeof(engine::TokenLogit);
    std::memcpy(result.values.data(), bytes + sizeof(header), result.readback_bytes);
    return result;
}

}  // namespace local_gguf_engine::tensor::backends::metal
