#include "metal_command_context.hpp"

#include "metal_buffer.hpp"
#include "metal_context.hpp"

#include "local_gguf_engine/tensor/dtype.hpp"
#include "local_gguf_engine/tensor/metal_trace_capture.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

struct TraceCaptureState {
    MetalTraceCaptureStatus status = MetalTraceCaptureStatus::idle;
    std::string output_path;
};

TraceCaptureState& trace_capture_state() {
    static TraceCaptureState state;
    return state;
}

[[nodiscard]] bool begin_trace_capture_if_armed(
    std::string_view label,
    id<MTLCommandQueue> queue
) {
    auto& state = trace_capture_state();
    if (state.status != MetalTraceCaptureStatus::armed || label != "Gemma2 resident decode") {
        return false;
    }
    auto* manager = [MTLCaptureManager sharedCaptureManager];
    if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        state.status = MetalTraceCaptureStatus::idle;
        throw std::runtime_error("Metal GPU trace document capture is unsupported on this system");
    }
    auto* descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.captureObject = queue;
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:state.output_path.c_str()]];
    NSError* error = nil;
    if (![manager startCaptureWithDescriptor:descriptor error:&error]) {
        state.status = MetalTraceCaptureStatus::idle;
        throw std::runtime_error(
            "Metal GPU trace capture could not start: "
            + ns_error_message(error, "unknown capture error")
        );
    }
    state.status = MetalTraceCaptureStatus::active;
    return true;
}

void finish_trace_capture(bool completed) noexcept {
    auto& state = trace_capture_state();
    if (state.status != MetalTraceCaptureStatus::active) return;
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
    state.status = completed ? MetalTraceCaptureStatus::completed : MetalTraceCaptureStatus::idle;
}

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

struct Gemma2KVCacheCopyDimensions {
    std::uint32_t rows;
    std::uint32_t row_width;
    std::uint32_t position_start;
    std::uint32_t source_row_mask;
    std::uint32_t destination_row_mask;
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

struct Gemma2WrappedPrefillAttentionScoreDimensions {
    std::uint32_t query_rows;
    std::uint32_t head_count;
    std::uint32_t kv_head_count;
    std::uint32_t key_length;
    std::uint32_t attention_token_count;
    std::uint32_t query_position_start;
    std::uint32_t cache_row_mask;
    std::uint32_t current_chunk_row_mask;
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

struct Gemma2WrappedPrefillAttentionValueDimensions {
    std::uint32_t query_rows;
    std::uint32_t head_count;
    std::uint32_t kv_head_count;
    std::uint32_t value_length;
    std::uint32_t attention_token_count;
    std::uint32_t query_position_start;
    std::uint32_t cache_row_mask;
    std::uint32_t current_chunk_row_mask;
};

static_assert(sizeof(Gemma2WrappedPrefillAttentionScoreDimensions) == 40U);
static_assert(sizeof(Gemma2WrappedPrefillAttentionValueDimensions) == 32U);

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

struct TokenCandidateBatchSelectionDimensions {
    std::array<TokenCandidateSelectionDimensions, 2> rows;
    std::uint32_t run_stride;
};

struct TokenCandidateBatchMergeDimensions {
    std::array<std::uint32_t, 2> retain_count;
    std::uint32_t input_run_count;
    std::uint32_t run_stride;
};

struct TokenCandidateResultHeader {
    std::uint32_t status;
    std::uint32_t count;
};

struct VerifierResolvedSelectionRow {
    std::uint32_t top_k;
    std::uint32_t base_candidate_count;
    std::uint32_t history_count;
    std::uint32_t reserved;
    std::array<std::uint32_t, engine::token_sampler_max_repetition_window> history_token_ids;
};

struct VerifierRequestResolveDimensions {
    std::uint32_t rows;
    std::uint32_t vocabulary_size;
    std::uint32_t top_k;
    std::uint32_t repetition_enabled;
};

struct VerifierInputValidationDimensions {
    std::uint32_t rows;
    std::uint32_t vocabulary_size;
};

struct VerifierRoPEDimensions {
    std::uint32_t rows;
    std::uint32_t head_count;
    std::uint32_t head_dimension;
    std::uint32_t rope_dimension_count;
    float frequency_base;
};

struct VerifierKVStoreDimensions {
    std::uint32_t rows;
    std::uint32_t head_count;
    std::uint32_t head_dimension;
    std::uint32_t rope_dimension_count;
    float frequency_base;
};

struct VerifierAttentionDimensions {
    std::uint32_t rows;
    std::uint32_t cache_row_mask;
    std::uint32_t query_head_count;
    std::uint32_t kv_head_count;
    std::uint32_t key_dimension;
    std::uint32_t value_dimension;
    float scale;
    float logit_softcap;
};

struct VerifierKVScatterDimensions {
    std::uint32_t rows;
    std::uint32_t row_width;
    std::uint32_t destination_position;
    std::uint32_t destination_row_mask;
};

struct FeatureTapCopyDimensions {
    std::uint32_t rows;
    std::uint32_t input_columns;
    std::uint32_t output_columns;
    std::uint32_t input_row_start;
    std::uint32_t output_column_start;
};

struct FeatureScatterDimensions {
    std::uint32_t rows;
    std::uint32_t width;
    std::uint32_t source_row_capacity;
    std::uint32_t destination_row_capacity;
};

struct TargetCandidateBatchDimensions {
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t run_count;
    std::uint32_t run_stride;
    float softcap;
};

static_assert(sizeof(VerifierResolvedSelectionRow) == 272U);
static_assert(sizeof(engine::VerifierDeviceSelectionRow) == 364U);

[[nodiscard]] const MetalBackendBufferStorage& storage(const BackendBuffer& buffer) {
    return require_metal_backend_buffer(backend_buffer_storage(buffer));
}

void bind_buffer(
    id<MTLComputeCommandEncoder> encoder,
    const BackendBuffer& buffer,
    NSUInteger index,
    std::uint64_t operation_byte_offset = 0
) {
    const auto& value = storage(buffer);
    [encoder setBuffer:value.buffer()
                offset:metal_buffer_binding_offset(value, operation_byte_offset)
               atIndex:index];
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
    std::string command_label;
    MatmulPipelineState* state = nullptr;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> active_compute_encoder = nil;
    MetalCommandStats stats;
    bool concurrent_compute = false;
    bool submitted = false;
    bool waited = false;
    bool owns_trace_capture = false;
    std::uint32_t debug_group_depth = 0;

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
        if (owns_trace_capture) finish_trace_capture(false);
    }
};

void fill_shared_metal_buffer_u32(
    const BackendBuffer& buffer,
    std::uint32_t pattern
) {
    const auto byte_count = buffer.descriptor().byte_count;
    if (byte_count % sizeof(std::uint32_t) != 0U
        || byte_count / sizeof(std::uint32_t) > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "Metal diagnostic buffer must contain a whole size_t-addressable u32 range"
        );
    }
    auto* words = static_cast<std::uint32_t*>(
        metal_buffer_contents(storage(buffer))
    );
    std::fill_n(
        words,
        static_cast<std::size_t>(byte_count / sizeof(std::uint32_t)),
        pattern
    );
}

void read_shared_metal_buffer_bytes(
    const BackendBuffer& buffer,
    std::uint64_t byte_offset,
    std::span<std::uint8_t> destination
) {
    if (byte_offset > buffer.descriptor().byte_count
        || destination.size() > buffer.descriptor().byte_count - byte_offset) {
        throw std::out_of_range("Metal shared-buffer read exceeds its buffer");
    }
    std::memcpy(
        destination.data(), metal_buffer_contents(storage(buffer), byte_offset),
        destination.size()
    );
}

MetalCommandContext::MetalCommandContext(std::string_view label, bool concurrent_compute)
    : impl_(std::make_unique<Impl>()) {
    impl_->command_label = label;
    impl_->concurrent_compute = concurrent_compute;
    impl_->state = &shared_matmul_pipeline_state();
    impl_->owns_trace_capture = begin_trace_capture_if_armed(label, impl_->state->queue);
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

void MetalCommandContext::restart() {
    if (impl_->submitted && (!impl_->waited
        || [impl_->command_buffer status] != MTLCommandBufferStatusCompleted)) {
        throw std::logic_error("cannot reuse a pending or failed Metal command context");
    }
    impl_->end_compute_encoding();
    if (impl_->owns_trace_capture) finish_trace_capture(false);
    auto label = impl_->command_buffer.label;
    impl_->command_buffer = [impl_->state->queue commandBuffer];
    if (!impl_->command_buffer) throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
        "Metal resident command buffer renewal failed");
    impl_->command_buffer.label = label;
    impl_->stats = {};
    impl_->stats.command_buffer_count = 1;
    impl_->submitted = false;
    impl_->waited = false;
    impl_->debug_group_depth = 0;
    impl_->owns_trace_capture = begin_trace_capture_if_armed(impl_->command_label, impl_->state->queue);
}

MetalCommandContext::~MetalCommandContext() = default;
MetalCommandContext::MetalCommandContext(MetalCommandContext&&) noexcept = default;
MetalCommandContext& MetalCommandContext::operator=(MetalCommandContext&&) noexcept = default;

void MetalCommandContext::write_u32(const BackendBuffer& destination, std::uint32_t value) {
    require_capacity(destination, sizeof(value), "u32 staging");
    std::memcpy(metal_buffer_contents(storage(destination)), &value, sizeof(value));
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
        metal_buffer_contents(storage(destination), byte_offset),
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
    bind_buffer(encoder, token_ids, 0);
    [encoder setBuffer:weight_storage.buffer()
                offset:metal_buffer_binding_offset(weight_storage)
               atIndex:1];
    bind_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((hidden + 15U) / 16U),
                                      static_cast<NSUInteger>((tokens + 15U) / 16U),
                                      1
                                  )
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
    const auto input_rows = input.descriptor().shape.dimension(0);
    const auto rows = row_count == 0U ? input_rows : row_count;
    if (rows > input_rows) {
        throw std::invalid_argument(
            "Metal resident execution matmul explicit row count exceeds input rows"
        );
    }
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
    const bool quantized = weight.descriptor().dtype == DType::q8_0
        || weight.descriptor().dtype == DType::q4_k
        || weight.descriptor().dtype == DType::q6_k;
    QuantizedMatmulDispatch quantized_dispatch;
    F16MatmulDispatch f16_dispatch;
    id<MTLComputePipelineState> pipeline = nil;
    if (quantized) {
        quantized_dispatch = select_quantized_matmul_dispatch(
            *impl_->state, weight.descriptor().dtype, rows, shared, columns
        );
        pipeline = quantized_dispatch.pipeline;
        if (weight.descriptor().dtype == DType::q8_0) {
            ++impl_->stats.q8_decode_projection_count;
            if (rows == 2U) ++impl_->stats.q8_width_two_projection_count;
        }
        auto* decode = weight.descriptor().dtype == DType::q4_k
            ? &impl_->stats.q4_decode_projection_count : &impl_->stats.q6_decode_projection_count;
        auto* small = weight.descriptor().dtype == DType::q4_k
            ? &impl_->stats.q4_small_batch_projection_count : &impl_->stats.q6_small_batch_projection_count;
        auto* exact_16 = weight.descriptor().dtype == DType::q4_k
            ? &impl_->stats.q4_exact_16_projection_count : &impl_->stats.q6_exact_16_projection_count;
        auto* generic = weight.descriptor().dtype == DType::q4_k
            ? &impl_->stats.q4_generic_tiled_projection_count : &impl_->stats.q6_generic_tiled_projection_count;
        if (weight.descriptor().dtype == DType::q8_0) {
            // Q8_0 is output-only and counted separately above.
        } else if (rows == 1U) ++*decode;
        else if (rows <= 8U) {
            ++*small;
            if (rows == 2U) {
                if (weight.descriptor().dtype == DType::q4_k) {
                    ++impl_->stats.q4_width_two_projection_count;
                } else {
                    ++impl_->stats.q6_width_two_projection_count;
                }
            }
        }
        else if (rows == 16U && columns % 64U == 0U) ++*exact_16;
        else ++*generic;
    } else {
        pipeline = f16_matmul_pipeline(*impl_->state, weight.descriptor().dtype);
        f16_dispatch = select_f16_matmul_dispatch(*impl_->state, rows, columns);
        ++impl_->stats.f16_projection_count;
    }
    [encoder setComputePipelineState:pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, weight, 1);
    bind_buffer(encoder, output, 2, output_byte_offset);
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

void MetalCommandContext::encode_verifier_matmul(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t row_count
) {
    if (row_count == 0U || row_count > input.descriptor().shape.dimension(0)) {
        throw std::invalid_argument("Metal verifier matmul row count is invalid");
    }
    const auto shared = input.descriptor().shape.dimension(1);
    const auto columns = weight.descriptor().shape.dimension(1);
    if (shared != weight.descriptor().shape.dimension(0)) {
        throw std::invalid_argument("Metal verifier matmul inner dimensions do not match");
    }
    require_capacity(output,
        static_cast<std::uint64_t>(row_count) * columns * sizeof(float),
        "verifier matmul output");
    const MatmulDimensions dims {
        .rows = row_count,
        .shared = u32(shared, "verifier matmul shared"),
        .columns = u32(columns, "verifier matmul columns"),
    };
    auto encoder = impl_->compute_encoder("Gemma2 target-verifier matmul");
    id<MTLComputePipelineState> pipeline = nil;
    if (weight.descriptor().dtype == DType::q4_k
        || weight.descriptor().dtype == DType::q6_k) {
        if (shared % 256U != 0U) {
            throw std::invalid_argument("Metal verifier K-quant shared width must be 256-aligned");
        }
        const auto blocks = shared / 256U;
        if (weight.descriptor().dtype == DType::q4_k) {
            pipeline = blocks == 14U ? impl_->state->matmul_q4_k_packed_row_batch_14_pipeline
                : blocks == 16U ? impl_->state->matmul_q4_k_packed_row_batch_16_pipeline
                : impl_->state->matmul_q4_k_packed_row_batch_pipeline;
            if (row_count == 1U) ++impl_->stats.q4_decode_projection_count;
            else if (row_count <= 8U) ++impl_->stats.q4_small_batch_projection_count;
            else ++impl_->stats.q4_generic_tiled_projection_count;
        } else {
            pipeline = blocks == 14U ? impl_->state->matmul_q6_k_packed_row_batch_14_pipeline
                : blocks == 56U ? impl_->state->matmul_q6_k_packed_row_batch_56_pipeline
                : impl_->state->matmul_q6_k_packed_row_batch_pipeline;
            if (row_count == 1U) ++impl_->stats.q6_decode_projection_count;
            else if (row_count <= 8U) ++impl_->stats.q6_small_batch_projection_count;
            else ++impl_->stats.q6_generic_tiled_projection_count;
        }
        [encoder setComputePipelineState:pipeline];
        bind_buffer(encoder, input, 0);
        bind_buffer(encoder, weight, 1);
        bind_buffer(encoder, output, 2);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(
                                          static_cast<NSUInteger>((columns + 3U) / 4U),
                                          static_cast<NSUInteger>((row_count + 1U) / 2U),
                                          1U
                                      )
                  threadsPerThreadgroup:MTLSizeMake(64U, 1U, 1U)];
    } else {
        pipeline = f16_matmul_pipeline(*impl_->state, weight.descriptor().dtype);
        const auto dispatch = select_f16_matmul_dispatch(*impl_->state, row_count, columns);
        ++impl_->stats.f16_projection_count;
        [encoder setComputePipelineState:pipeline];
        bind_buffer(encoder, input, 0);
        bind_buffer(encoder, weight, 1);
        bind_buffer(encoder, output, 2);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        [encoder dispatchThreadgroups:dispatch.threadgroups
                  threadsPerThreadgroup:dispatch.threads_per_threadgroup];
    }
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_target_head_q8_rows(
    const BackendBuffer& input,
    std::uint32_t input_row_start,
    const BackendBuffer& weight,
    const BackendBuffer& logits_tile,
    std::uint32_t row_count
) {
    if (weight.descriptor().dtype != DType::q8_0 || row_count == 0U
        || row_count > engine::external_speculation_max_horizon
        || input_row_start > input.descriptor().shape.dimension(0)
        || row_count > input.descriptor().shape.dimension(0) - input_row_start) {
        throw std::invalid_argument("Metal target-head row tile is invalid");
    }
    const auto shared = input.descriptor().shape.dimension(1);
    const auto columns = weight.descriptor().shape.dimension(1);
    if (shared != weight.descriptor().shape.dimension(0) || shared % 32U != 0U) {
        throw std::invalid_argument("Metal target-head Q8 layout is invalid");
    }
    require_capacity(logits_tile,
        static_cast<std::uint64_t>(row_count) * columns * sizeof(float),
        "target-head logits tile");
    const MatmulDimensions dims {
        .rows = row_count,
        .shared = u32(shared, "target-head shared"),
        .columns = u32(columns, "target-head columns"),
    };
    auto encoder = impl_->compute_encoder("Gemma2 Q8 target-head row tile");
    [encoder setComputePipelineState:shared == 3584U
        ? impl_->state->target_head_q8_row_parallel_112_pipeline
        : impl_->state->target_head_q8_row_parallel_pipeline];
    bind_buffer(encoder, input, 0,
        static_cast<std::uint64_t>(input_row_start) * shared * sizeof(float));
    bind_buffer(encoder, weight, 1);
    bind_buffer(encoder, logits_tile, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((columns + 3U) / 4U),
                                      static_cast<NSUInteger>(row_count),
                                      1U
                                  )
              threadsPerThreadgroup:MTLSizeMake(64U, 1U, 1U)];
    ++impl_->stats.q8_decode_projection_count;
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_feature_tap_copy(
    const BackendBuffer& input,
    std::uint32_t input_row_start,
    const BackendBuffer& fused_output,
    std::uint32_t output_column_start,
    std::uint32_t row_count
) {
    if (input.descriptor().dtype != DType::f32
        || fused_output.descriptor().dtype != DType::f32
        || input.descriptor().shape.rank() != 2U
        || fused_output.descriptor().shape.rank() != 2U
        || row_count == 0U
        || input_row_start > input.descriptor().shape.dimension(0)
        || row_count > input.descriptor().shape.dimension(0) - input_row_start) {
        throw std::invalid_argument("Metal feature tap copy input is invalid");
    }
    const auto input_columns = input.descriptor().shape.dimension(1);
    const auto output_columns = fused_output.descriptor().shape.dimension(1);
    if (row_count > fused_output.descriptor().shape.dimension(0)
        || output_column_start > output_columns
        || input_columns > output_columns - output_column_start) {
        throw std::invalid_argument("Metal feature tap copy output range is invalid");
    }
    const FeatureTapCopyDimensions dims {
        .rows = row_count,
        .input_columns = u32(input_columns, "feature tap input width"),
        .output_columns = u32(output_columns, "feature tap fused width"),
        .input_row_start = input_row_start,
        .output_column_start = output_column_start,
    };
    auto encoder = impl_->compute_encoder("Target feature tap copy");
    [encoder setComputePipelineState:impl_->state->feature_tap_copy_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, fused_output, 1);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
    const auto elements = static_cast<std::uint64_t>(row_count) * input_columns;
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(elements), 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_feature_tap_store_f16(
    const BackendBuffer& input,
    std::uint32_t input_row_start,
    const BackendBuffer& output,
    std::uint32_t output_column_start,
    std::uint32_t row_count
) {
    if (input.descriptor().dtype != DType::f32
        || output.descriptor().dtype != DType::f16
        || input.descriptor().shape.rank() != 2U
        || output.descriptor().shape.rank() != 2U
        || row_count == 0U
        || input_row_start > input.descriptor().shape.dimension(0)
        || row_count > input.descriptor().shape.dimension(0) - input_row_start) {
        throw std::invalid_argument("Metal trace feature tap input is invalid");
    }
    const auto input_columns = input.descriptor().shape.dimension(1);
    const auto output_columns = output.descriptor().shape.dimension(1);
    if (row_count > output.descriptor().shape.dimension(0)
        || output_column_start > output_columns
        || input_columns > output_columns - output_column_start) {
        throw std::invalid_argument("Metal trace feature tap output range is invalid");
    }
    const FeatureTapCopyDimensions dims {
        .rows = row_count,
        .input_columns = u32(input_columns, "trace feature input width"),
        .output_columns = u32(output_columns, "trace feature output width"),
        .input_row_start = input_row_start,
        .output_column_start = output_column_start,
    };
    auto encoder = impl_->compute_encoder("Trace target feature tap store");
    [encoder setComputePipelineState:impl_->state->feature_tap_store_f16_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, output, 1);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
    const auto elements = static_cast<std::uint64_t>(row_count) * input_columns;
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(elements), 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_feature_scatter(
    const BackendBuffer& source_f32,
    const BackendBuffer& source_rows_u32,
    const BackendBuffer& destination_rows_u32,
    const BackendBuffer& destination,
    const BackendBuffer& status_u32,
    std::uint32_t row_count,
    std::uint32_t width
) {
    if (source_f32.descriptor().dtype != DType::f32
        || source_f32.descriptor().shape.rank() != 2U
        || destination.descriptor().shape.rank() != 2U
        || (destination.descriptor().dtype != DType::f32
            && destination.descriptor().dtype != DType::f16)
        || source_f32.descriptor().shape.dimension(1) != width
        || destination.descriptor().shape.dimension(1) != width
        || row_count == 0U
        || row_count > source_rows_u32.descriptor().element_count
        || row_count > destination_rows_u32.descriptor().element_count
        || status_u32.descriptor().byte_count < sizeof(std::uint32_t)) {
        throw std::invalid_argument("Metal feature scatter shape is invalid");
    }
    const FeatureScatterDimensions dims {
        .rows = row_count,
        .width = width,
        .source_row_capacity = u32(
            source_f32.descriptor().shape.dimension(0), "feature source row capacity"
        ),
        .destination_row_capacity = u32(
            destination.descriptor().shape.dimension(0),
            "feature destination row capacity"
        ),
    };
    auto encoder = impl_->compute_encoder("Target feature scatter");
    [encoder setComputePipelineState:destination.descriptor().dtype == DType::f16
        ? impl_->state->feature_scatter_f16_pipeline
        : impl_->state->feature_scatter_f32_pipeline];
    bind_buffer(encoder, source_f32, 0);
    bind_buffer(encoder, source_rows_u32, 1);
    bind_buffer(encoder, destination_rows_u32, 2);
    bind_buffer(encoder, destination, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    bind_buffer(encoder, status_u32, 5);
    const auto elements = static_cast<std::uint64_t>(row_count) * width;
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(elements), 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_f16_row_scatter(
    const BackendBuffer& source,
    const BackendBuffer& source_rows_u32,
    const BackendBuffer& destination_rows_u32,
    const BackendBuffer& destination,
    const BackendBuffer& status_u32,
    std::uint32_t row_count
) {
    if (source.descriptor().dtype != DType::f16
        || destination.descriptor().dtype != DType::f16
        || source.descriptor().shape.rank() != 2U
        || destination.descriptor().shape.rank() != 2U
        || source.descriptor().shape.dimension(1)
            != destination.descriptor().shape.dimension(1)
        || row_count == 0U
        || row_count > source_rows_u32.descriptor().element_count
        || row_count > destination_rows_u32.descriptor().element_count
        || status_u32.descriptor().byte_count < sizeof(std::uint32_t)) {
        throw std::invalid_argument("Metal F16 row scatter shape is invalid");
    }
    const auto width = u32(source.descriptor().shape.dimension(1), "F16 row width");
    const FeatureScatterDimensions dims {
        .rows = row_count,
        .width = width,
        .source_row_capacity = u32(
            source.descriptor().shape.dimension(0), "F16 source row capacity"
        ),
        .destination_row_capacity = u32(
            destination.descriptor().shape.dimension(0), "F16 destination row capacity"
        ),
    };
    auto encoder = impl_->compute_encoder("F16 row scatter");
    [encoder setComputePipelineState:impl_->state->f16_row_scatter_pipeline];
    bind_buffer(encoder, source, 0);
    bind_buffer(encoder, source_rows_u32, 1);
    bind_buffer(encoder, destination_rows_u32, 2);
    bind_buffer(encoder, destination, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    bind_buffer(encoder, status_u32, 5);
    const auto elements = static_cast<std::uint64_t>(row_count) * width;
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(elements), 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
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
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, weight, 1);
    bind_buffer(encoder, output, 2);
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
    float epsilon,
    std::uint32_t row_count
) {
    const auto rows = row_count == 0U ? input.descriptor().shape.dimension(0) : row_count;
    const auto hidden = input.descriptor().shape.dimension(1);
    if ((hidden & 3U) != 0U) {
        throw std::invalid_argument("Metal fused RMSNorm-add hidden width must be divisible by four");
    }
    require_capacity(weight, hidden * sizeof(float), "fused RMSNorm-add weight");
    require_capacity(input, rows * hidden * sizeof(float), "fused RMSNorm-add input");
    require_capacity(residual, rows * hidden * sizeof(float), "fused RMSNorm-add residual");
    require_capacity(output, rows * hidden * sizeof(float), "fused RMSNorm-add output");
    const RMSNormDimensions dims {
        .rows = u32(rows, "fused RMSNorm-add rows"),
        .hidden = u32(hidden, "fused RMSNorm-add hidden"),
        .input_row_start = 0U,
        .epsilon = epsilon,
    };
    auto encoder = impl_->compute_encoder("Gemma2 fused RMSNorm-add");
    [encoder setComputePipelineState:impl_->state->rms_norm_add_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, weight, 1);
    bind_buffer(encoder, residual, 2);
    bind_buffer(encoder, output, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
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
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, output, 1, output_byte_offset);
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
    std::uint32_t token_count,
    std::uint32_t source_row_start
) {
    const auto rows = token_count == 0U ? raw_key.descriptor().shape.dimension(0) : token_count;
    const auto width = static_cast<std::uint64_t>(head_count) * head_dimension;
    const auto require_raw = [&](const BackendBuffer& buffer, const char* name) {
        const auto& descriptor = buffer.descriptor();
        if (descriptor.dtype != DType::f32 || descriptor.shape.rank() != 2U
            || descriptor.shape.dimension(0) < static_cast<std::uint64_t>(source_row_start) + rows
            || descriptor.shape.dimension(1) != width) {
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
    const auto source_byte_offset = static_cast<std::uint64_t>(source_row_start) * width * sizeof(float);
    bind_buffer(encoder, raw_key, 0, source_byte_offset);
    bind_buffer(encoder, raw_value, 1, source_byte_offset);
    bind_buffer(encoder, cache_key, 2);
    bind_buffer(encoder, cache_value, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((width + 255U) / 256U),
                                      static_cast<NSUInteger>(rows),
                                      1
                                  )
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_gemma2_kv_cache_copy(
    const BackendBuffer& source_key,
    const BackendBuffer& source_value,
    const BackendBuffer& destination_key,
    const BackendBuffer& destination_value,
    std::uint32_t position_start,
    std::uint32_t token_count,
    std::uint32_t source_row_mask,
    std::uint32_t destination_row_mask
) {
    const auto& source = source_key.descriptor();
    const auto& source_v = source_value.descriptor();
    const auto& destination = destination_key.descriptor();
    const auto& destination_v = destination_value.descriptor();
    if (token_count == 0U || source.dtype != DType::f16 || destination.dtype != DType::f16
        || source.shape.rank() != 3U || destination.shape.rank() != 3U
        || source.shape.dimension(0) != static_cast<std::uint64_t>(source_row_mask) + 1U
        || destination.shape.dimension(0) != static_cast<std::uint64_t>(destination_row_mask) + 1U
        || source.shape.dimension(1) != destination.shape.dimension(1)
        || source.shape.dimension(2) != destination.shape.dimension(2)
        || source_v != source || destination_v != destination
        || (source_row_mask & (source_row_mask + 1U)) != 0U
        || (destination_row_mask & (destination_row_mask + 1U)) != 0U) {
        throw std::invalid_argument("Metal Gemma 2 KV-cache copy layout is invalid");
    }
    const auto row_width = source.shape.dimension(1) * source.shape.dimension(2);
    const Gemma2KVCacheCopyDimensions dims {
        .rows = token_count,
        .row_width = u32(row_width, "KV-cache copy row width"),
        .position_start = position_start,
        .source_row_mask = source_row_mask,
        .destination_row_mask = destination_row_mask,
    };
    auto encoder = impl_->compute_encoder("Gemma2 F16 KV-cache copy");
    [encoder setComputePipelineState:impl_->state->gemma2_kv_cache_copy_pipeline];
    bind_buffer(encoder, source_key, 0);
    bind_buffer(encoder, source_value, 1);
    bind_buffer(encoder, destination_key, 2);
    bind_buffer(encoder, destination_value, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((row_width + 255U) / 256U),
                                      token_count,
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
    bind_buffer(encoder, query, 0);
    bind_buffer(encoder, key_cache, 1);
    bind_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((attention_token_count + 7U) / 8U),
                                      static_cast<NSUInteger>((head_count + 7U) / 8U),
                                      static_cast<NSUInteger>((query_rows + 1U) / 2U)
                                  )
              threadsPerThreadgroup:MTLSizeMake(8, 8, 2)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_wrapped_prefill_attention_scores(
    const BackendBuffer& query,
    const BackendBuffer& key_cache,
    const BackendBuffer& current_chunk_key,
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
    const auto query_rows = query_row_count == 0U
        ? query.descriptor().shape.dimension(0) : query_row_count;
    const auto& cache = key_cache.descriptor();
    const auto& current = current_chunk_key.descriptor();
    if (cache.dtype != DType::f16 || current.dtype != DType::f16
        || cache.shape.rank() != 3U || current.shape.rank() != 3U
        || cache.shape.dimension(0) != static_cast<std::uint64_t>(cache_row_mask) + 1U
        || current.shape.dimension(0) != engine::gemma2_metal_prefill_chunk_size
        || current.shape.dimension(1) != kv_head_count
        || current.shape.dimension(2) != key_length
        || cache.shape.dimension(1) != current.shape.dimension(1)
        || cache.shape.dimension(2) != current.shape.dimension(2)
        || query_rows == 0U || query_rows > current.shape.dimension(0)
        || kv_head_count == 0U || head_count == 0U || head_count % kv_head_count != 0U
        || attention_token_count == 0U) {
        throw std::invalid_argument(
            "Metal Gemma 2 wrapped-prefill attention key layout is invalid"
        );
    }
    (void)storage(key_cache);
    (void)storage(current_chunk_key);
    require_capacity(
        query,
        query_rows * head_count * key_length * sizeof(float),
        "wrapped-prefill attention query"
    );
    require_capacity(
        output,
        query_rows * head_count * attention_token_count * sizeof(float),
        "wrapped-prefill attention score output"
    );
    const Gemma2WrappedPrefillAttentionScoreDimensions dims {
        .query_rows = u32(query_rows, "wrapped-prefill attention query rows"),
        .head_count = head_count,
        .kv_head_count = kv_head_count,
        .key_length = key_length,
        .attention_token_count = attention_token_count,
        .query_position_start = query_position_start,
        .cache_row_mask = cache_row_mask,
        .current_chunk_row_mask = engine::gemma2_metal_prefill_chunk_size - 1U,
        .scale = scale,
        .logit_softcap = logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma2 wrapped prefill attention scores");
    [encoder setComputePipelineState:impl_->state->gemma2_wrapped_prefill_attention_scores_pipeline];
    bind_buffer(encoder, query, 0);
    bind_buffer(encoder, key_cache, 1);
    bind_buffer(encoder, current_chunk_key, 2);
    bind_buffer(encoder, output, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
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
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, output, 1);
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
    bind_buffer(encoder, probabilities, 0);
    bind_buffer(encoder, value_cache, 1);
    bind_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((value_length + 7U) / 8U),
                                      static_cast<NSUInteger>((head_count + 7U) / 8U),
                                      static_cast<NSUInteger>((query_rows + 1U) / 2U)
                                  )
              threadsPerThreadgroup:MTLSizeMake(8, 8, 2)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_wrapped_prefill_attention_values(
    const BackendBuffer& probabilities,
    const BackendBuffer& value_cache,
    const BackendBuffer& current_chunk_value,
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
        ? probabilities.descriptor().shape.dimension(0) : query_row_count;
    const auto& cache = value_cache.descriptor();
    const auto& current = current_chunk_value.descriptor();
    if (cache.dtype != DType::f16 || current.dtype != DType::f16
        || cache.shape.rank() != 3U || current.shape.rank() != 3U
        || cache.shape.dimension(0) != static_cast<std::uint64_t>(cache_row_mask) + 1U
        || current.shape.dimension(0) != engine::gemma2_metal_prefill_chunk_size
        || current.shape.dimension(1) != kv_head_count
        || current.shape.dimension(2) != value_length
        || cache.shape.dimension(1) != current.shape.dimension(1)
        || cache.shape.dimension(2) != current.shape.dimension(2)
        || query_rows == 0U || query_rows > current.shape.dimension(0)
        || kv_head_count == 0U || head_count == 0U || head_count % kv_head_count != 0U
        || attention_token_count == 0U) {
        throw std::invalid_argument(
            "Metal Gemma 2 wrapped-prefill attention value layout is invalid"
        );
    }
    (void)storage(value_cache);
    (void)storage(current_chunk_value);
    require_capacity(
        probabilities,
        query_rows * head_count * attention_token_count * sizeof(float),
        "wrapped-prefill attention probabilities"
    );
    require_capacity(
        output,
        query_rows * head_count * value_length * sizeof(float),
        "wrapped-prefill attention value output"
    );
    const Gemma2WrappedPrefillAttentionValueDimensions dims {
        .query_rows = u32(query_rows, "wrapped-prefill attention query rows"),
        .head_count = head_count,
        .kv_head_count = kv_head_count,
        .value_length = value_length,
        .attention_token_count = attention_token_count,
        .query_position_start = query_position_start,
        .cache_row_mask = cache_row_mask,
        .current_chunk_row_mask = engine::gemma2_metal_prefill_chunk_size - 1U,
    };
    auto encoder = impl_->compute_encoder("Gemma2 wrapped prefill attention values");
    [encoder setComputePipelineState:impl_->state->gemma2_wrapped_prefill_attention_values_pipeline];
    bind_buffer(encoder, probabilities, 0);
    bind_buffer(encoder, value_cache, 1);
    bind_buffer(encoder, current_chunk_value, 2);
    bind_buffer(encoder, output, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
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
    std::uint32_t cache_row_mask,
    std::uint32_t query_row,
    std::uint32_t output_row
) {
    constexpr std::uint32_t dimension = 256U;
    constexpr std::uint32_t max_partitions = 32U;
    const auto& query_descriptor = query.descriptor();
    if (query_descriptor.dtype != DType::f32
        || query_descriptor.shape.rank() != 2U
        || query_descriptor.shape.dimension(0) <= query_row
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
    if (output.descriptor().dtype != DType::f32 || output.descriptor().shape.rank() != 2U
        || output.descriptor().shape.dimension(0) <= output_row
        || output.descriptor().shape.dimension(1)
            != static_cast<std::uint64_t>(query_heads) * dimension) {
        throw std::invalid_argument("Metal Gemma 2 decode attention output layout is invalid");
    }
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
    ++impl_->stats.decode_attention_dispatch_count;
    impl_->stats.decode_attention_partition_count += partitions;
    if (attention_token_count > std::numeric_limits<std::uint64_t>::max()
            - impl_->stats.decode_attention_token_count) {
        throw std::overflow_error("Metal decode attention token accounting overflow");
    }
    impl_->stats.decode_attention_token_count += attention_token_count;
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
    const auto row_bytes = static_cast<std::uint64_t>(query_heads) * dimension * sizeof(float);
    bind_buffer(encoder, query, 0, static_cast<std::uint64_t>(query_row) * row_bytes);
    bind_buffer(encoder, key_cache, 1);
    bind_buffer(encoder, value_cache, 2);
    bind_buffer(encoder, output, 3, static_cast<std::uint64_t>(output_row) * row_bytes);
    bind_buffer(encoder, partial_output, 4);
    bind_buffer(encoder, partial_maxima, 5);
    bind_buffer(encoder, partial_denominators, 6);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(kv_heads, partitions, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++impl_->stats.compute_dispatch_count;

    if (partitions > 1U) {
        encode_memory_barrier();
        encoder = impl_->compute_encoder("Gemma2 fused decode attention reduction");
        [encoder setComputePipelineState:impl_->state->gemma2_decode_attention_reduce_pipeline];
        bind_buffer(encoder, partial_output, 0);
        bind_buffer(encoder, partial_maxima, 1);
        bind_buffer(encoder, partial_denominators, 2);
        bind_buffer(encoder, output, 3, static_cast<std::uint64_t>(output_row) * row_bytes);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(kv_heads, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
    }
}

void MetalCommandContext::encode_verifier_input_validation(
    const BackendBuffer& actual_token_ids,
    std::uint64_t actual_byte_offset,
    const BackendBuffer& expected_token_ids,
    const BackendBuffer& status,
    std::uint32_t row_count,
    std::uint32_t vocabulary_size
) {
    if (row_count == 0U || vocabulary_size == 0U) {
        throw std::invalid_argument("Metal verifier input validation extent is invalid");
    }
    require_capacity(actual_token_ids,
        actual_byte_offset + static_cast<std::uint64_t>(row_count) * sizeof(std::uint32_t),
        "verifier actual token IDs");
    require_capacity(expected_token_ids,
        static_cast<std::uint64_t>(row_count) * sizeof(std::uint32_t),
        "verifier expected token IDs");
    require_capacity(status, sizeof(std::uint32_t), "verifier input status");
    const VerifierInputValidationDimensions dims {row_count, vocabulary_size};
    auto encoder = impl_->compute_encoder("Gemma2 verifier input validation");
    [encoder setComputePipelineState:impl_->state->verifier_input_validate_pipeline];
    bind_buffer(encoder, actual_token_ids, 0, actual_byte_offset);
    bind_buffer(encoder, expected_token_ids, 1);
    bind_buffer(encoder, status, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(row_count, 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(std::min<std::uint32_t>(row_count, 256U), 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_verifier_selection_requests(
    const BackendBuffer& raw_requests,
    std::uint64_t raw_byte_offset,
    const BackendBuffer& resolved_requests,
    const BackendBuffer& histories,
    const BackendBuffer& status,
    std::uint32_t row_count,
    const engine::TokenSamplerOptions& options,
    std::uint32_t vocabulary_size
) {
    if (row_count == 0U || row_count > engine::VerifierRowMax
        || options.top_k == 0U || options.top_k > engine::token_sampler_max_top_k
        || options.repetition_window > engine::token_sampler_max_repetition_window
        || !std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("Metal verifier selection-request parameters are invalid");
    }
    require_capacity(raw_requests,
        raw_byte_offset + static_cast<std::uint64_t>(row_count)
            * sizeof(engine::VerifierDeviceSelectionRow),
        "verifier raw selection requests");
    require_capacity(resolved_requests,
        static_cast<std::uint64_t>(row_count) * sizeof(VerifierResolvedSelectionRow),
        "verifier resolved selection requests");
    require_capacity(histories,
        static_cast<std::uint64_t>(row_count)
            * engine::token_sampler_max_repetition_window * sizeof(std::uint32_t),
        "verifier selection histories");
    const VerifierRequestResolveDimensions dims {
        .rows = row_count,
        .vocabulary_size = vocabulary_size,
        .top_k = static_cast<std::uint32_t>(options.temperature == 0.0F ? 1U : options.top_k),
        .repetition_enabled = options.repetition_penalty == 1.0F ? 0U : 1U,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier request resolve");
    [encoder setComputePipelineState:impl_->state->verifier_request_resolve_pipeline];
    bind_buffer(encoder, raw_requests, 0, raw_byte_offset);
    bind_buffer(encoder, resolved_requests, 1);
    bind_buffer(encoder, histories, 2);
    bind_buffer(encoder, status, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(row_count, 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(std::min<std::uint32_t>(row_count, 256U), 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_verifier_rope(
    const BackendBuffer& input,
    const BackendBuffer& positions,
    const BackendBuffer& output,
    std::uint32_t row_count,
    std::uint32_t head_count,
    std::uint32_t head_dimension,
    std::uint32_t rope_dimension_count,
    float frequency_base
) {
    if (row_count == 0U || head_count == 0U || head_dimension == 0U
        || rope_dimension_count == 0U || rope_dimension_count > head_dimension
        || (rope_dimension_count & 1U) != 0U || !std::isfinite(frequency_base)
        || frequency_base <= 0.0F) {
        throw std::invalid_argument("Metal verifier RoPE parameters are invalid");
    }
    const auto row_width = static_cast<std::uint64_t>(head_count) * head_dimension;
    require_capacity(input, static_cast<std::uint64_t>(row_count) * row_width * sizeof(float),
        "verifier RoPE input");
    require_capacity(output, static_cast<std::uint64_t>(row_count) * row_width * sizeof(float),
        "verifier RoPE output");
    require_capacity(positions, static_cast<std::uint64_t>(row_count) * sizeof(std::uint32_t),
        "verifier RoPE positions");
    const VerifierRoPEDimensions dims {
        row_count, head_count, head_dimension, rope_dimension_count, frequency_base,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier query RoPE");
    [encoder setComputePipelineState:impl_->state->verifier_rope_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, positions, 1);
    bind_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(row_width), row_count, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_verifier_kv_store(
    const BackendBuffer& raw_key,
    const BackendBuffer& raw_value,
    const BackendBuffer& positions,
    const BackendBuffer& private_key,
    const BackendBuffer& private_value,
    std::uint32_t row_count,
    std::uint32_t head_count,
    std::uint32_t head_dimension,
    std::uint32_t rope_dimension_count,
    float frequency_base
) {
    const auto row_width = static_cast<std::uint64_t>(head_count) * head_dimension;
    if (row_count == 0U || row_width == 0U || rope_dimension_count > head_dimension
        || (rope_dimension_count & 1U) != 0U) {
        throw std::invalid_argument("Metal verifier private K/V parameters are invalid");
    }
    require_capacity(raw_key, static_cast<std::uint64_t>(row_count) * row_width * sizeof(float),
        "verifier raw key");
    require_capacity(raw_value, static_cast<std::uint64_t>(row_count) * row_width * sizeof(float),
        "verifier raw value");
    require_capacity(private_key, static_cast<std::uint64_t>(row_count) * row_width * sizeof(std::uint16_t),
        "verifier private key");
    require_capacity(private_value, static_cast<std::uint64_t>(row_count) * row_width * sizeof(std::uint16_t),
        "verifier private value");
    const VerifierKVStoreDimensions dims {
        row_count, head_count, head_dimension, rope_dimension_count, frequency_base,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier private K/V store");
    [encoder setComputePipelineState:impl_->state->verifier_kv_store_pipeline];
    bind_buffer(encoder, raw_key, 0);
    bind_buffer(encoder, raw_value, 1);
    bind_buffer(encoder, positions, 2);
    bind_buffer(encoder, private_key, 3);
    bind_buffer(encoder, private_value, 4);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(row_width), row_count, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_verifier_attention(
    const BackendBuffer& query,
    const BackendBuffer& resident_key,
    const BackendBuffer& resident_value,
    const BackendBuffer& private_key,
    const BackendBuffer& private_value,
    const BackendBuffer& schedules,
    std::uint64_t schedule_byte_offset,
    const BackendBuffer& ancestors,
    const BackendBuffer& output,
    const BackendBuffer& partial_output,
    const BackendBuffer& partial_maxima,
    const BackendBuffer& partial_denominators,
    std::uint32_t row_count,
    std::uint32_t cache_row_mask,
    std::uint32_t query_head_count,
    std::uint32_t kv_head_count,
    std::uint32_t key_dimension,
    std::uint32_t value_dimension,
    float scale,
    float logit_softcap
) {
    if (row_count == 0U || kv_head_count == 0U
        || query_head_count != 2U * kv_head_count
        || key_dimension != value_dimension || key_dimension != 256U
        || !std::isfinite(scale) || scale <= 0.0F
        || !std::isfinite(logit_softcap) || logit_softcap <= 0.0F) {
        throw std::invalid_argument("Metal verifier attention supports the exact Gemma2 grouped-head layout");
    }
    require_capacity(schedules,
        schedule_byte_offset + static_cast<std::uint64_t>(row_count)
            * sizeof(engine::VerifierRowAttention),
        "verifier attention schedules");
    const VerifierAttentionDimensions dims {
        row_count, cache_row_mask, query_head_count, kv_head_count,
        key_dimension, value_dimension, scale, logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier tree attention");
    [encoder setComputePipelineState:impl_->state->verifier_attention_pipeline];
    bind_buffer(encoder, query, 0);
    bind_buffer(encoder, resident_key, 1);
    bind_buffer(encoder, resident_value, 2);
    bind_buffer(encoder, private_key, 3);
    bind_buffer(encoder, private_value, 4);
    bind_buffer(encoder, schedules, 5, schedule_byte_offset);
    bind_buffer(encoder, ancestors, 6);
    bind_buffer(encoder, output, 7);
    bind_buffer(encoder, partial_output, 8);
    bind_buffer(encoder, partial_maxima, 9);
    bind_buffer(encoder, partial_denominators, 10);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(kv_head_count,
                                      engine::gemma2_decode_attention_max_partitions,
                                      row_count)
              threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
    encode_memory_barrier();
    encoder = impl_->compute_encoder("Gemma2 verifier tree attention reduction");
    [encoder setComputePipelineState:impl_->state->verifier_attention_reduce_pipeline];
    bind_buffer(encoder, partial_output, 0);
    bind_buffer(encoder, partial_maxima, 1);
    bind_buffer(encoder, partial_denominators, 2);
    bind_buffer(encoder, schedules, 3, schedule_byte_offset);
    bind_buffer(encoder, output, 4);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(kv_head_count, row_count, 1U)
              threadsPerThreadgroup:MTLSizeMake(32U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_verifier_kv_scatter(
    const BackendBuffer& private_key,
    const BackendBuffer& private_value,
    const BackendBuffer& selected_nodes,
    const BackendBuffer& resident_key,
    const BackendBuffer& resident_value,
    std::uint32_t destination_position,
    std::uint32_t row_count,
    std::uint32_t row_width,
    std::uint32_t destination_row_mask
) {
    if (row_count == 0U || row_count > engine::external_speculation_max_horizon + 1U
        || row_width == 0U) {
        throw std::invalid_argument("Metal verifier accepted-path scatter extent is invalid");
    }
    const VerifierKVScatterDimensions dims {
        row_count, row_width, destination_position, destination_row_mask,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier accepted-path K/V scatter");
    [encoder setComputePipelineState:impl_->state->verifier_kv_scatter_pipeline];
    bind_buffer(encoder, private_key, 0);
    bind_buffer(encoder, private_value, 1);
    bind_buffer(encoder, selected_nodes, 2);
    bind_buffer(encoder, resident_key, 3);
    bind_buffer(encoder, resident_value, 4);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(row_width, row_count, 1U)
          threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
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
    bind_buffer(encoder, left, 0);
    bind_buffer(encoder, right, 1);
    bind_buffer(encoder, output, 2);
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
    bind_buffer(encoder, gate, 0);
    bind_buffer(encoder, up, 1);
    bind_buffer(encoder, output, 2);
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
    bind_buffer(encoder, gate, 0);
    bind_buffer(encoder, up, 1);
    bind_buffer(encoder, output, 2);
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
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, output, 1);
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
        bind_buffer(encoder, logits, 0);
        bind_buffer(encoder, workspace.compact_result, 1);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        return 1U;
    }

    auto encoder = impl_->compute_encoder("Gemma2 compact token candidates");
    [encoder setComputePipelineState:impl_->state->token_candidate_tile_pipeline];
    bind_buffer(encoder, logits, 0);
    bind_buffer(encoder, workspace.candidate_runs_a, 1);
    bind_buffer(encoder, workspace.compact_result, 2);
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
        bind_buffer(encoder, input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b, 0);
        bind_buffer(encoder, input_is_a ? workspace.candidate_runs_b : workspace.candidate_runs_a, 1);
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
    bind_buffer(encoder, logits, 0);
    bind_buffer(encoder, input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b, 1);
    bind_buffer(encoder, workspace.history_token_staging, 2);
    bind_buffer(encoder, workspace.compact_result, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(retain_count + request.history_token_count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
    return selector_dispatches + 1U;
}

std::uint32_t MetalCommandContext::encode_token_candidates_width_two(
    const BackendBuffer& logits,
    const engine::Gemma2MetalSelectionWorkspace& workspace,
    const std::array<engine::TokenSelectionRequest, 2>& requests,
    float softcap
) {
    if (!std::isfinite(softcap) || softcap <= 0.0F) {
        throw std::invalid_argument("Metal batched token-candidate softcap must be finite and positive");
    }
    const auto columns = u32(
        logits.descriptor().shape.dimension(logits.descriptor().shape.rank() - 1U),
        "batched token-candidate columns"
    );
    if (logits.descriptor().shape.rank() < 2U
        || logits.descriptor().shape.dimension(logits.descriptor().shape.rank() - 2U) < 2U) {
        throw std::invalid_argument("Metal batched token-candidate logits require two rows");
    }
    const auto run_count = (columns + engine::gemma2_metal_selection_tile_size - 1U)
        / engine::gemma2_metal_selection_tile_size;
    const auto run_stride = run_count * static_cast<std::uint32_t>(engine::token_sampler_max_base_candidates);
    TokenCandidateBatchSelectionDimensions batch {
        .rows = {},
        .run_stride = run_stride,
    };
    constexpr auto result_u32_stride = std::uint32_t {2U + engine::token_sampler_max_candidates * 2U};
    constexpr std::array<std::uint32_t, 2> empty_header {0U, 0U};
    for (std::size_t row = 0; row < requests.size(); ++row) {
        const auto& request = requests[row];
        if (request.top_k == 0U || request.top_k > engine::token_sampler_max_top_k
            || request.history_token_count > engine::token_sampler_max_repetition_window
            || request.base_candidate_count != request.top_k + request.history_token_count) {
            throw std::invalid_argument("Metal batched token-candidate request is outside product bounds");
        }
        batch.rows[row] = TokenCandidateSelectionDimensions {
            .columns = columns,
            .retain_count = std::min(columns, request.base_candidate_count),
            .run_count = run_count,
            .history_count = request.history_token_count,
            .softcap = softcap,
        };
        write_u32s(workspace.compact_result, static_cast<std::uint32_t>(row) * result_u32_stride, empty_header);
        if (request.history_token_count > 0U) {
            write_u32s(
                workspace.history_token_staging,
                static_cast<std::uint32_t>(row) * static_cast<std::uint32_t>(engine::token_sampler_max_repetition_window),
                std::span<const std::uint32_t>(request.history_token_ids).first(request.history_token_count)
            );
        }
    }

    const bool top1 = batch.rows[0].retain_count == 1U && batch.rows[1].retain_count == 1U
        && batch.rows[0].history_count == 0U && batch.rows[1].history_count == 0U;
    if (top1) {
        auto encoder = impl_->compute_encoder("Gemma2 batched compact token top-1");
        [encoder setComputePipelineState:impl_->state->token_candidate_batch_top1_pipeline];
        bind_buffer(encoder, logits, 0);
        bind_buffer(encoder, workspace.compact_result, 1);
        [encoder setBytes:&batch length:sizeof(batch) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        return 1U;
    }

    auto encoder = impl_->compute_encoder("Gemma2 batched compact token candidates");
    [encoder setComputePipelineState:impl_->state->token_candidate_batch_tile_pipeline];
    bind_buffer(encoder, logits, 0);
    bind_buffer(encoder, workspace.candidate_runs_a, 1);
    bind_buffer(encoder, workspace.compact_result, 2);
    [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(run_count, 2, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
    std::uint32_t selector_dispatches = 1U;
    auto input_is_a = true;
    auto input_run_count = run_count;
    while (input_run_count > 1U) {
        encode_memory_barrier();
        const auto output_run_count = (input_run_count + 1U) / 2U;
        const TokenCandidateBatchMergeDimensions merge {
            .retain_count = {batch.rows[0].retain_count, batch.rows[1].retain_count},
            .input_run_count = input_run_count,
            .run_stride = run_stride,
        };
        encoder = impl_->compute_encoder("Gemma2 batched compact token merge");
        [encoder setComputePipelineState:impl_->state->token_candidate_batch_merge_pipeline];
        bind_buffer(encoder, input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b, 0);
        bind_buffer(encoder, input_is_a ? workspace.candidate_runs_b : workspace.candidate_runs_a, 1);
        [encoder setBytes:&merge length:sizeof(merge) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(output_run_count, 2, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        ++selector_dispatches;
        input_run_count = output_run_count;
        input_is_a = !input_is_a;
    }
    encode_memory_barrier();
    const auto finalize_count = std::max(
        batch.rows[0].retain_count + batch.rows[0].history_count,
        batch.rows[1].retain_count + batch.rows[1].history_count
    );
    encoder = impl_->compute_encoder("Gemma2 batched compact token result");
    [encoder setComputePipelineState:impl_->state->token_candidate_batch_finalize_pipeline];
    bind_buffer(encoder, logits, 0);
    bind_buffer(encoder, input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b, 1);
    bind_buffer(encoder, workspace.history_token_staging, 2);
    bind_buffer(encoder, workspace.compact_result, 3);
    [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(finalize_count, 2, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
    return selector_dispatches + 1U;
}

std::uint32_t MetalCommandContext::encode_target_token_candidates(
    const BackendBuffer& logits_tile,
    const engine::TargetSelectionWorkspace& workspace,
    std::uint32_t first_request_row,
    std::uint32_t row_count,
    float softcap,
    bool greedy_without_history
) {
    if (row_count == 0U || row_count > workspace.batch_row_capacity
        || row_count > engine::target_selection_batch_rows
        || first_request_row > workspace.row_capacity
        || row_count > workspace.row_capacity - first_request_row
        || !std::isfinite(softcap) || softcap <= 0.0F) {
        throw std::invalid_argument("Metal target-candidate row tile is invalid");
    }
    const auto columns = u32(
        logits_tile.descriptor().shape.dimension(logits_tile.descriptor().shape.rank() - 1U),
        "target-candidate columns"
    );
    const auto run_count = (columns + engine::gemma2_metal_selection_tile_size - 1U)
        / engine::gemma2_metal_selection_tile_size;
    const auto run_stride = run_count
        * static_cast<std::uint32_t>(engine::token_sampler_max_base_candidates);
    const TargetCandidateBatchDimensions dims {
        row_count, columns, run_count, run_stride, softcap,
    };
    constexpr auto result_u32_stride = std::uint32_t {
        2U + engine::token_sampler_max_candidates * 2U
    };
    constexpr std::array<std::uint32_t, 2> empty_header {0U, 0U};
    for (std::uint32_t row = 0; row < row_count; ++row) {
        write_u32s(workspace.compact_results,
            (first_request_row + row) * result_u32_stride, empty_header);
    }
    const auto request_offset = static_cast<std::uint64_t>(first_request_row)
        * sizeof(VerifierResolvedSelectionRow);
    const auto history_offset = static_cast<std::uint64_t>(first_request_row)
        * engine::token_sampler_max_repetition_window * sizeof(std::uint32_t);
    const auto result_offset = static_cast<std::uint64_t>(first_request_row)
        * result_u32_stride * sizeof(std::uint32_t);
    if (greedy_without_history) {
        auto encoder = impl_->compute_encoder("Gemma2 target-candidate top-1");
        [encoder setComputePipelineState:impl_->state->target_candidate_batch_top1_pipeline];
        bind_buffer(encoder, logits_tile, 0);
        bind_buffer(encoder, workspace.compact_results, 1, result_offset);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(row_count, 1U, 1U)
                  threadsPerThreadgroup:MTLSizeMake(512U, 1U, 1U)];
        ++impl_->stats.compute_dispatch_count;
        return 1U;
    }

    auto encoder = impl_->compute_encoder("Gemma2 target-candidate tile");
    [encoder setComputePipelineState:impl_->state->target_candidate_batch_tile_pipeline];
    bind_buffer(encoder, logits_tile, 0);
    bind_buffer(encoder, workspace.resolved_request_staging, 1, request_offset);
    bind_buffer(encoder, workspace.candidate_runs_a, 2);
    bind_buffer(encoder, workspace.compact_results, 3, result_offset);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(run_count, row_count, 1U)
              threadsPerThreadgroup:MTLSizeMake(512U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
    auto dispatches = std::uint32_t {1U};
    auto input_is_a = true;
    auto input_runs = run_count;
    while (input_runs > 1U) {
        encode_memory_barrier();
        const auto output_runs = (input_runs + 1U) / 2U;
        encoder = impl_->compute_encoder("Gemma2 target-candidate merge");
        [encoder setComputePipelineState:impl_->state->target_candidate_batch_merge_pipeline];
        bind_buffer(encoder,
            input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b, 0);
        bind_buffer(encoder, workspace.resolved_request_staging, 1, request_offset);
        bind_buffer(encoder,
            input_is_a ? workspace.candidate_runs_b : workspace.candidate_runs_a, 2);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        [encoder setBytes:&input_runs length:sizeof(input_runs) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(output_runs, row_count, 1U)
                  threadsPerThreadgroup:MTLSizeMake(1U, 1U, 1U)];
        ++impl_->stats.compute_dispatch_count;
        ++dispatches;
        input_runs = output_runs;
        input_is_a = !input_is_a;
    }
    encode_memory_barrier();
    encoder = impl_->compute_encoder("Gemma2 target-candidate finalize");
    [encoder setComputePipelineState:impl_->state->target_candidate_batch_finalize_pipeline];
    bind_buffer(encoder, logits_tile, 0);
    bind_buffer(encoder,
        input_is_a ? workspace.candidate_runs_a : workspace.candidate_runs_b, 1);
    bind_buffer(encoder, workspace.resolved_request_staging, 2, request_offset);
    bind_buffer(encoder, workspace.history_token_staging, 3, history_offset);
    bind_buffer(encoder, workspace.compact_results, 4, result_offset);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(engine::token_sampler_max_candidates, row_count, 1U)
              threadsPerThreadgroup:MTLSizeMake(1U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
    return dispatches + 1U;
}

void MetalCommandContext::encode_memory_barrier() {
    auto encoder = impl_->compute_encoder("Gemma2 dependency barrier");
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++impl_->stats.memory_barrier_count;
}

void MetalCommandContext::push_debug_group(std::string_view label) {
    if (label.empty()) throw std::invalid_argument("Metal debug group label must not be empty");
    const std::string owned_label(label);
    auto encoder = impl_->compute_encoder("Gemma2 diagnostic groups");
    [encoder pushDebugGroup:[NSString stringWithUTF8String:owned_label.c_str()]];
    ++impl_->debug_group_depth;
}

void MetalCommandContext::pop_debug_group() {
    if (impl_->debug_group_depth == 0U) {
        throw std::logic_error("Metal debug group stack is empty");
    }
    [impl_->active_compute_encoder popDebugGroup];
    --impl_->debug_group_depth;
}

MetalCommandStats MetalCommandContext::commit() {
    if (impl_->submitted) {
        throw std::logic_error("Metal resident command buffer was already submitted");
    }
    if (impl_->debug_group_depth != 0U) {
        throw std::logic_error("Metal command buffer has unclosed debug groups");
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
    if (impl_->owns_trace_capture) {
        finish_trace_capture(true);
        impl_->owns_trace_capture = false;
    }
    return impl_->stats;
}

void arm_next_gemma2_decode_trace(std::string output_path) {
    auto& state = trace_capture_state();
    if (state.status != MetalTraceCaptureStatus::idle) {
        throw std::logic_error("a Metal GPU trace capture is already armed or completed");
    }
    if (output_path.empty()) throw std::invalid_argument("Metal GPU trace path must not be empty");
    const char* enabled = std::getenv("MTL_CAPTURE_ENABLED");
    if (enabled == nullptr || std::string_view(enabled) != "1") {
        throw std::runtime_error("Metal GPU trace capture requires MTL_CAPTURE_ENABLED=1");
    }
    const std::filesystem::path path(output_path);
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("Metal GPU trace path already exists: " + output_path);
    }
    if (!path.parent_path().empty() && !std::filesystem::exists(path.parent_path())) {
        throw std::runtime_error("Metal GPU trace parent directory does not exist: " + output_path);
    }
    state.output_path = std::move(output_path);
    state.status = MetalTraceCaptureStatus::armed;
}

MetalTraceCaptureStatus metal_trace_capture_status() noexcept {
    return trace_capture_state().status;
}

void reset_completed_metal_trace_capture() {
    auto& state = trace_capture_state();
    if (state.status != MetalTraceCaptureStatus::completed) {
        throw std::logic_error("Metal GPU trace capture did not complete");
    }
    state = TraceCaptureState {};
}

void cancel_armed_metal_trace_capture() {
    auto& state = trace_capture_state();
    if (state.status != MetalTraceCaptureStatus::armed) {
        throw std::logic_error("Metal GPU trace capture is not armed");
    }
    state = TraceCaptureState {};
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
    const auto* bytes = static_cast<const std::byte*>(
        metal_buffer_contents(storage(workspace.compact_result))
    );
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

std::array<engine::CompactTokenCandidates, 2>
MetalCommandContext::read_token_candidates_width_two(
    const engine::Gemma2MetalSelectionWorkspace& workspace
) const {
    if (!impl_->submitted) {
        throw std::logic_error("Metal batched token-candidate result cannot be read before completion");
    }
    constexpr auto result_byte_stride = sizeof(TokenCandidateResultHeader)
        + engine::token_sampler_max_candidates * sizeof(engine::TokenLogit);
    const auto* bytes = static_cast<const std::byte*>(metal_buffer_contents(storage(workspace.compact_result)));
    std::array<engine::CompactTokenCandidates, 2> output;
    for (std::size_t row = 0; row < output.size(); ++row) {
        TokenCandidateResultHeader header {};
        const auto* row_bytes = bytes + row * result_byte_stride;
        std::memcpy(&header, row_bytes, sizeof(header));
        if (header.status != 0U) {
            throw std::invalid_argument(
                (header.status & 2U) != 0U
                    ? "Metal batched token-candidate history token is outside vocabulary"
                    : "Metal batched token-candidate logit must be finite after softcap"
            );
        }
        if (header.count == 0U || header.count > engine::token_sampler_max_candidates) {
            throw std::runtime_error("Metal batched token-candidate result count is invalid");
        }
        output[row].count = header.count;
        output[row].readback_bytes = header.count * sizeof(engine::TokenLogit);
        std::memcpy(output[row].values.data(), row_bytes + sizeof(header), output[row].readback_bytes);
    }
    return output;
}

std::uint32_t MetalCommandContext::read_verifier_status(const BackendBuffer& status) const {
    if (!impl_->submitted || [impl_->command_buffer status] < MTLCommandBufferStatusCompleted) {
        throw std::logic_error("Metal verifier status cannot be read before completion");
    }
    const auto* bytes = static_cast<const std::byte*>(metal_buffer_contents(storage(status)));
    std::uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

void MetalCommandContext::read_target_token_candidates(
    const engine::TargetSelectionWorkspace& workspace,
    std::uint32_t first_row,
    std::span<engine::CompactTokenCandidates> output
) const {
    if (!impl_->submitted || [impl_->command_buffer status] < MTLCommandBufferStatusCompleted) {
        throw std::logic_error("Metal target candidates cannot be read before completion");
    }
    if (first_row > workspace.row_capacity || output.size() > workspace.row_capacity - first_row) {
        throw std::invalid_argument("Metal target candidate readback extent is invalid");
    }
    constexpr auto result_byte_stride = sizeof(TokenCandidateResultHeader)
        + engine::token_sampler_max_candidates * sizeof(engine::TokenLogit);
    const auto* bytes = static_cast<const std::byte*>(
        metal_buffer_contents(storage(workspace.compact_results))
    );
    for (std::size_t row = 0; row < output.size(); ++row) {
        const auto* row_bytes = bytes + (first_row + row) * result_byte_stride;
        TokenCandidateResultHeader header {};
        std::memcpy(&header, row_bytes, sizeof(header));
        if (header.status != 0U) {
            const auto detail = (header.status & 2U) != 0U
                ? "history token is outside vocabulary"
                : "logit must be finite after softcap";
            throw std::invalid_argument(
                "Metal target-candidate row " + std::to_string(first_row + row)
                    + " " + detail
            );
        }
        if (header.count == 0U || header.count > engine::token_sampler_max_candidates) {
            throw std::runtime_error("Metal target-candidate result count is invalid");
        }
        output[row].count = header.count;
        output[row].readback_bytes = header.count * sizeof(engine::TokenLogit);
        std::memcpy(output[row].values.data(), row_bytes + sizeof(header),
            output[row].readback_bytes);
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal
