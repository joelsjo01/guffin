#include "metal_command_context.hpp"

#include "metal_buffer.hpp"
#include "metal_context.hpp"

#include "local_gguf_engine/tensor/dtype.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
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

struct GroupedAttentionScoreDimensions {
    std::uint32_t query_rows;
    std::uint32_t key_rows;
    std::uint32_t head_count;
    std::uint32_t kv_head_count;
    std::uint32_t key_length;
    std::uint32_t attention_token_count;
    std::uint32_t query_position_start;
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
};

struct SoftmaxDimensions {
    std::uint32_t rows;
    std::uint32_t columns;
};

struct TopLogitDimensions {
    std::uint32_t row_offset;
    std::uint32_t columns;
};

struct TopLogitMetalResult {
    std::uint32_t token;
    float logit;
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

[[nodiscard]] id<MTLComputePipelineState> matmul_pipeline(
    MatmulPipelineState& state,
    DType dtype
) {
    switch (dtype) {
        case DType::f16:
            return state.matmul_pipeline;
        case DType::q4_k:
            return state.matmul_q4_k_pipeline;
        case DType::q6_k:
            return state.matmul_q6_k_pipeline;
        default:
            throw std::invalid_argument("Metal resident execution matmul weight dtype is unsupported");
    }
}

}  // namespace

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

    [[nodiscard]] id<MTLBlitCommandEncoder> blit_encoder(const char* label) {
        if (submitted) {
            throw std::logic_error("Metal resident execution command buffer was already submitted");
        }
        end_compute_encoding();
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal resident execution blit encoder creation failed: ") + label
            );
        }
        encoder.label = [NSString stringWithUTF8String:label];
        ++stats.blit_encoder_count;
        return encoder;
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
    const bool use_q4_decode_matvec =
        rows == 1
        && weight.descriptor().dtype == DType::q4_k
        && shared % 256 == 0
        && impl_->state->matvec_q4_k_pipeline.threadExecutionWidth == 32
        && impl_->state->matvec_q4_k_pipeline.maxTotalThreadsPerThreadgroup >= 64;
    const bool use_q6_decode_matvec =
        rows == 1
        && weight.descriptor().dtype == DType::q6_k
        && shared % 256 == 0
        && impl_->state->matvec_q6_k_pipeline.threadExecutionWidth == 32
        && impl_->state->matvec_q6_k_pipeline.maxTotalThreadsPerThreadgroup >= 64;
    id<MTLComputePipelineState> pipeline = matmul_pipeline(*impl_->state, weight.descriptor().dtype);
    if (use_q4_decode_matvec) {
        pipeline = impl_->state->matvec_q4_k_pipeline;
    } else if (use_q6_decode_matvec) {
        pipeline = impl_->state->matvec_q6_k_pipeline;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:storage(input).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(weight).buffer() offset:0 atIndex:1];
    [encoder setBuffer:storage(output).buffer() offset:static_cast<NSUInteger>(output_byte_offset) atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    if (use_q4_decode_matvec || use_q6_decode_matvec) {
        constexpr std::uint64_t outputs_per_threadgroup = 4;
        [encoder dispatchThreadgroups:MTLSizeMake(
                                          static_cast<NSUInteger>(
                                              (columns + outputs_per_threadgroup - 1) / outputs_per_threadgroup
                                          ),
                                          1,
                                          1
                                      )
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
        [encoder dispatchThreadgroups:MTLSizeMake(
                                          static_cast<NSUInteger>((columns + 15U) / 16U),
                                          static_cast<NSUInteger>((rows + 15U) / 16U),
                                          1
                                      )
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
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
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
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

void MetalCommandContext::encode_kv_cache_write(
    const BackendBuffer& key,
    const BackendBuffer& value,
    const BackendBuffer& cache_key,
    const BackendBuffer& cache_value,
    std::uint32_t cache_position,
    std::uint32_t token_count
) {
    const auto rows = token_count == 0U ? key.descriptor().shape.dimension(0) : token_count;
    const auto key_row_bytes = key.descriptor().shape.dimension(1) * sizeof(float);
    const auto value_row_bytes = value.descriptor().shape.dimension(1) * sizeof(float);
    const auto key_bytes = rows * key_row_bytes;
    const auto value_bytes = rows * value_row_bytes;
    const auto key_offset = static_cast<std::uint64_t>(cache_position) * key_row_bytes;
    const auto value_offset = static_cast<std::uint64_t>(cache_position) * value_row_bytes;
    require_capacity(cache_key, key_offset + key_bytes, "key cache");
    require_capacity(cache_value, value_offset + value_bytes, "value cache");
    auto encoder = impl_->blit_encoder("Gemma2 KV write");
    [encoder copyFromBuffer:storage(key).buffer()
               sourceOffset:0
                   toBuffer:storage(cache_key).buffer()
          destinationOffset:static_cast<NSUInteger>(key_offset)
                       size:static_cast<NSUInteger>(key_bytes)];
    [encoder copyFromBuffer:storage(value).buffer()
               sourceOffset:0
                   toBuffer:storage(cache_value).buffer()
          destinationOffset:static_cast<NSUInteger>(value_offset)
                       size:static_cast<NSUInteger>(value_bytes)];
    [encoder endEncoding];
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
    std::uint32_t query_row_count
) {
    const auto query_rows =
        query_row_count == 0U ? query.descriptor().shape.dimension(0) : query_row_count;
    const auto key_rows = key_cache.descriptor().shape.dimension(0);
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
        .scale = scale,
        .logit_softcap = logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma2 attention scores");
    [encoder setComputePipelineState:impl_->state->causal_grouped_attention_scores_pipeline];
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
    std::uint32_t query_row_count
) {
    const auto query_rows = query_row_count == 0U
        ? probabilities.descriptor().shape.dimension(0)
        : query_row_count;
    const auto value_rows = value_cache.descriptor().shape.dimension(0);
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
    };
    auto encoder = impl_->compute_encoder("Gemma2 attention values");
    [encoder setComputePipelineState:impl_->state->causal_grouped_attention_values_pipeline];
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

void MetalCommandContext::encode_top_logit(
    const BackendBuffer& logits,
    const BackendBuffer& result
) {
    static_assert(sizeof(TopLogitMetalResult) == 8);
    require_capacity(result, sizeof(TopLogitMetalResult), "top-logit result");
    const auto columns = logits.descriptor().shape.dimension(logits.descriptor().shape.rank() - 1U);
    const auto rows = logits.descriptor().element_count / columns;
    const TopLogitDimensions dims {
        .row_offset = u32((rows - 1U) * columns, "top-logit row offset"),
        .columns = u32(columns, "top-logit columns"),
    };
    auto encoder = impl_->compute_encoder("Gemma2 top logit");
    [encoder setComputePipelineState:impl_->state->top_logit_pipeline];
    [encoder setBuffer:storage(logits).buffer() offset:0 atIndex:0];
    [encoder setBuffer:storage(result).buffer() offset:0 atIndex:1];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
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

ops::TopLogit MetalCommandContext::read_top_logit(const BackendBuffer& result) const {
    if (!impl_->submitted) {
        throw std::logic_error("Metal top-logit result cannot be read before completion");
    }
    TopLogitMetalResult raw {};
    std::memcpy(&raw, [storage(result).buffer() contents], sizeof(raw));
    return ops::TopLogit {.token = raw.token, .logit = raw.logit};
}

}  // namespace local_gguf_engine::tensor::backends::metal
