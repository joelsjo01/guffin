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

struct HeadRMSNormDimensions {
    std::uint32_t rows;
    std::uint32_t head_count;
    std::uint32_t head_dimension;
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

// Field order and size match Gemma2PrefillAttentionDimensions in the embedded shader.
struct F16PrefillAttentionDimensions {
    std::uint32_t query_rows;
    std::uint32_t head_count;
    std::uint32_t kv_head_count;
    std::uint32_t key_dimension;
    std::uint32_t value_dimension;
    std::uint32_t attention_token_count;
    std::uint32_t query_position_start;
    std::uint32_t cache_row_mask;
    std::uint32_t current_chunk_row_mask;
    std::uint32_t uses_staged_chunk;
    std::uint32_t current_chunk_position_origin;
    float scale;
    float logit_softcap;
};

static_assert(sizeof(F16PrefillAttentionDimensions) == 52U);

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
    float frequency_scale;
};

struct VerifierKVStoreDimensions {
    std::uint32_t rows;
    std::uint32_t head_count;
    std::uint32_t head_dimension;
    std::uint32_t rope_dimension_count;
    float frequency_base;
    float frequency_scale;
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

void bind_output_buffer(id<MTLComputeCommandEncoder> encoder, const BackendBuffer& buffer,
    NSUInteger index, std::uint64_t operation_byte_offset = 0) {
    const auto& value = storage(buffer);
    require_writable_metal_buffer(value);
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

void require_disjoint_ranges(
    const BackendBuffer& first, std::uint64_t first_bytes,
    const BackendBuffer& second, std::uint64_t second_bytes,
    const char* name
) {
    const auto& first_storage = storage(first);
    const auto& second_storage = storage(second);
    if (first_storage.buffer() != second_storage.buffer()) return;
    const auto first_offset = metal_buffer_binding_offset(first_storage, 0U);
    const auto second_offset = metal_buffer_binding_offset(second_storage, 0U);
    if (first_offset < second_offset + second_bytes
        && second_offset < first_offset + first_bytes) {
        throw std::invalid_argument(std::string("Metal resident execution ") + name + " buffers overlap");
    }
}

[[nodiscard]] std::uint64_t q4_input_element_count(
    const BackendBuffer& input, std::uint32_t row_count
) {
    const auto& descriptor = input.descriptor();
    if (descriptor.dtype != DType::f32 || descriptor.packing != TensorPacking::canonical
        || descriptor.shape.rank() != 2U || row_count == 0U
        || row_count > descriptor.shape.dimension(0)
        || descriptor.shape.dimension(1) == 0U
        || descriptor.shape.dimension(1) % 256U != 0U) {
        throw std::invalid_argument("Metal Q4 input sums require canonical F32 rows and a 256-aligned width");
    }
    const auto shared = u32(descriptor.shape.dimension(1), "Q4 input-sum shared width");
    const auto elements = static_cast<std::uint64_t>(row_count) * shared;
    static_cast<void>(u32(elements, "Q4 input-sum element addressing"));
    require_capacity(input, elements * sizeof(float), "Q4 input-sum input");
    return elements;
}

void require_q4_input_sum_capacity(const BackendBuffer& sums, std::uint64_t input_elements) {
    if (sums.descriptor().dtype != DType::f32
        || sums.descriptor().packing != TensorPacking::canonical) {
        throw std::invalid_argument("Metal Q4 input sums require canonical F32 storage");
    }
    require_capacity(sums, input_elements / 8U * sizeof(float), "Q4 input sums");
}

void require_f32_storage(const BackendBuffer& buffer, std::uint64_t elements, const char* name) {
    if (buffer.descriptor().dtype != DType::f32
        || buffer.descriptor().packing != TensorPacking::canonical)
        throw std::invalid_argument(std::string("Metal Q4 sum producer requires canonical F32 ") + name);
    static_cast<void>(u32(elements, name));
    require_capacity(buffer, elements * sizeof(float), name);
}

struct PackedRowDispatch {
    id<MTLComputePipelineState> pipeline;
    std::uint32_t rows_per_group;
    std::uint32_t columns_per_group = 4U;
    std::uint32_t threads_per_group = 64U;
};

[[nodiscard]] PackedRowDispatch packed_row_dispatch(
    MatmulPipelineState& state, const TensorDescriptor& weight,
    std::uint64_t shared, std::uint32_t rows, bool allow_prefill_matrix = false,
    bool prepared_q4 = false
) {
    const auto dtype = weight.dtype;
    const auto packing = weight.packing;
    if (packing == TensorPacking::kquant_columns8) {
        // Packed shaders form byte offsets with uint arithmetic. Bound the
        // physical payload, including padded columns, before selecting them.
        if (weight.byte_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("Metal columns8 weight payload exceeds 32-bit shader addressing");
        }
        if ((dtype == DType::q4_k ? state.columns8_q4_rows[0] : state.columns8_q6_rows[0]) == nil) {
            throw std::invalid_argument("Columns8 weights require prepared Metal pipelines");
        }
        if (allow_prefill_matrix && rows > 8U && state.columns8_gemma2_prompt[0] != nil) {
            const bool exact16 = rows == 16U && weight.shape.dimension(1) % 64U == 0U;
            return {state.columns8_gemma2_prompt[exact16 ? 1U : 0U],
                exact16 ? 16U : 32U, 64U, exact16 ? 64U : 128U};
        }
        if (allow_prefill_matrix && rows >= 4U && state.gemma3_prompt_matrix[1][0][0] != nil) {
            const auto tile = rows >= 32U ? 1U : 0U;
            return {state.gemma3_prompt_matrix[1][dtype == DType::q4_k ? 0U : 1U][tile],
                tile == 1U ? 32U : 8U, tile == 1U ? 64U : 32U, 128U};
        }
        const auto tile = rows == 1U ? 1U : 2U;
        return {dtype == DType::q4_k ? (prepared_q4 ? state.columns8_q4_prepared_rows[tile - 1U]
            : state.columns8_q4_rows[tile - 1U])
            : state.columns8_q6_rows[tile - 1U], tile,
            dtype == DType::q4_k ? 8U : 16U, 128U};
    }
    if (packing != TensorPacking::canonical) {
        throw std::invalid_argument("Metal packed projection has an unknown weight packing");
    }
    if (allow_prefill_matrix && rows >= 4U && state.gemma3_prompt_matrix[0][0][0] != nil) {
        const auto tile = rows >= 32U ? 1U : 0U;
        return {state.gemma3_prompt_matrix[0][dtype == DType::q4_k ? 0U : 1U][tile],
            tile == 1U ? 32U : 8U, tile == 1U ? 64U : 32U, 128U};
    }
    // Keep the retained Q4 scalar kernel; the shared two-row kernel preserves
    // its exact per-worker arithmetic. Two-row tiles bound register use for
    // larger batches. Q6 uses one common factored core for ordinary and verifier.
    if (state.gemma3_q4_shared_two_pipeline != nil && (dtype == DType::q6_k || rows > 1U)) {
        const auto tile = rows == 1U ? 1U : 2U;
        return {dtype == DType::q4_k ? state.gemma3_q4_shared_two_pipeline
            : (tile == 1U ? state.gemma3_q6_shared_one_pipeline : state.gemma3_q6_shared_two_pipeline),
            tile, dtype == DType::q4_k ? 8U : 16U, 128U};
    }
    const auto blocks = shared / 256U;
    if (dtype == DType::q4_k) {
        return {blocks == 14U ? state.matmul_q4_k_packed_row_batch_14_pipeline
            : blocks == 16U ? state.matmul_q4_k_packed_row_batch_16_pipeline
            : state.matmul_q4_k_packed_row_batch_pipeline, 2U};
    }
    return {blocks == 14U ? state.matmul_q6_k_packed_row_batch_14_pipeline
        : blocks == 56U ? state.matmul_q6_k_packed_row_batch_56_pipeline
        : state.matmul_q6_k_packed_row_batch_pipeline, 2U};
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

#if defined(LOCAL_GGUF_ENGINE_TESTING)
BackendBuffer shared_metal_buffer_view_for_testing(
    const BackendBuffer& parent, TensorDescriptor descriptor, std::size_t byte_offset
) {
    const auto& owner = storage(parent);
    return make_metal_backend_buffer_view(std::move(descriptor), owner.buffer(),
        metal_buffer_binding_offset(owner, byte_offset), owner.read_only());
}
#endif

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
        @autoreleasepool {
            if (active_compute_encoder != nil) {
                [active_compute_encoder endEncoding];
                active_compute_encoder = nil;
            }
        }
    }

    void create_command_buffer() {
        // Metal's convenience constructors return autoreleased objects. Keep
        // the ARC-owned reference while draining that temporary ownership here;
        // callers are C++ and need not run an Objective-C event-loop pool.
        // Otherwise discarded commands retain queue slots and model buffers.
        @autoreleasepool {
            command_buffer = [state->queue commandBuffer];
            if (command_buffer == nil) {
                throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                    "Metal resident execution command buffer creation failed");
            }
            command_buffer.label = [NSString stringWithUTF8String:command_label.c_str()];
        }
    }

    [[nodiscard]] id<MTLComputeCommandEncoder> const& compute_encoder(const char* label) {
        if (submitted) {
            throw std::logic_error("Metal resident execution command buffer was already submitted");
        }
        if (active_compute_encoder == nil) {
            @autoreleasepool {
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
            }
        }
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
        mutable_metal_buffer_contents(storage(buffer))
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

MetalCommandContext::MetalCommandContext(
    std::string_view label, bool concurrent_compute, bool precise_math
)
    : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->command_label = label;
        impl_->concurrent_compute = concurrent_compute;
        impl_->state = precise_math ? &shared_precise_matmul_pipeline_state()
            : &shared_matmul_pipeline_state();
        impl_->owns_trace_capture = begin_trace_capture_if_armed(label, impl_->state->queue);
        impl_->create_command_buffer();
        impl_->stats.command_buffer_count = 1;
    }
}

void MetalCommandContext::restart() {
    if (impl_->submitted && (!impl_->waited
        || [impl_->command_buffer status] != MTLCommandBufferStatusCompleted)) {
        throw std::logic_error("cannot reuse a pending or failed Metal command context");
    }
    impl_->end_compute_encoding();
    if (impl_->owns_trace_capture) finish_trace_capture(false);
    // Release a discarded command's queue slot before requesting its replacement.
    impl_->command_buffer = nil;
    impl_->create_command_buffer();
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
    std::memcpy(mutable_metal_buffer_contents(storage(destination)), &value, sizeof(value));
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
        mutable_metal_buffer_contents(storage(destination), byte_offset),
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
    const auto dtype = weight.descriptor().dtype;
    if (weight.descriptor().packing != TensorPacking::canonical) {
        throw std::invalid_argument("Metal embedding requires canonical weight packing");
    }
    if (dtype != DType::f16 && dtype != DType::q6_k) {
        throw std::invalid_argument("Metal embedding requires F16 or native Q6_K weights");
    }
    if (dtype == DType::q6_k && weight.descriptor().shape.dimension(0) % 256U != 0U) {
        throw std::invalid_argument("Metal Q6_K embedding width must be 256-aligned");
    }
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
    [encoder setComputePipelineState:dtype == DType::q6_k
        ? impl_->state->embedding_q6_pipeline : impl_->state->embedding_pipeline];
    bind_buffer(encoder, token_ids, 0);
    [encoder setBuffer:weight_storage.buffer()
                offset:metal_buffer_binding_offset(weight_storage)
               atIndex:1];
    bind_output_buffer(encoder, output, 2);
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
    const bool packed_prompt = weight.descriptor().packing == TensorPacking::kquant_columns8
        && weight.descriptor().dtype == DType::q4_k
        && impl_->state->columns8_gemma2_prompt[0] != nil;
    if (weight.descriptor().packing != TensorPacking::canonical && !packed_prompt) {
        throw std::invalid_argument("Generic Metal matmul requires canonical weight packing");
    }
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
        if (packed_prompt) {
            // Gemma 2 prefill retains its original operation graph, row-offset
            // stores and half-operand matrix arithmetic with a packed source.
            const auto dispatch = packed_row_dispatch(*impl_->state, weight.descriptor(),
                shared, u32(rows, "packed prompt rows"), true);
            quantized_dispatch = {dispatch.pipeline,
                MTLSizeMake((columns + dispatch.columns_per_group - 1U) / dispatch.columns_per_group,
                    (rows + dispatch.rows_per_group - 1U) / dispatch.rows_per_group, 1U),
                MTLSizeMake(dispatch.threads_per_group, 1U, 1U)};
        } else {
            quantized_dispatch = select_quantized_matmul_dispatch(
                *impl_->state, weight.descriptor().dtype, rows, shared, columns
            );
        }
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
    bind_output_buffer(encoder, output, 2, output_byte_offset);
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
    std::uint32_t row_count,
    bool allow_prefill_matrix,
    const BackendBuffer* q4_input_sums
) {
    if (q4_input_sums != nullptr && input.descriptor().shape.rank() != 2U)
        throw std::invalid_argument("Metal Q4 sum execution requires matrix input rows");
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
    if (q4_input_sums != nullptr) {
        if (allow_prefill_matrix || weight.descriptor().dtype != DType::q4_k
            || weight.descriptor().packing != TensorPacking::kquant_columns8
            || impl_->state->columns8_q4_prepared_rows[0] == nil
            || row_count > 2U
            || output.descriptor().dtype != DType::f32
            || output.descriptor().packing != TensorPacking::canonical) {
            throw std::invalid_argument("Metal prepared Q4 projection requires compact Q4 execution without prefill");
        }
        const auto input_elements = q4_input_element_count(input, row_count);
        require_q4_input_sum_capacity(*q4_input_sums, input_elements);
        const auto output_elements = static_cast<std::uint64_t>(row_count)
            * u32(columns, "prepared Q4 output width");
        static_cast<void>(u32(output_elements, "prepared Q4 output element addressing"));
        require_disjoint_ranges(input, input_elements * sizeof(float), *q4_input_sums,
            input_elements / 8U * sizeof(float), "prepared Q4 input/sums");
        require_disjoint_ranges(output, output_elements * sizeof(float), *q4_input_sums,
            input_elements / 8U * sizeof(float), "prepared Q4 output/sums");
        require_disjoint_ranges(weight, weight.descriptor().byte_count, *q4_input_sums,
            input_elements / 8U * sizeof(float), "prepared Q4 weight/sums");
        require_disjoint_ranges(input, input_elements * sizeof(float), output,
            output_elements * sizeof(float), "prepared Q4 input/output");
    }
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
        const auto dispatch = packed_row_dispatch(*impl_->state, weight.descriptor(),
            shared, row_count, allow_prefill_matrix, q4_input_sums != nullptr);
        pipeline = dispatch.pipeline;
        if (weight.descriptor().dtype == DType::q4_k) {
            if (row_count == 1U) ++impl_->stats.q4_decode_projection_count;
            else if (row_count <= 8U) ++impl_->stats.q4_small_batch_projection_count;
            else ++impl_->stats.q4_generic_tiled_projection_count;
        } else {
            if (row_count == 1U) ++impl_->stats.q6_decode_projection_count;
            else if (row_count <= 8U) ++impl_->stats.q6_small_batch_projection_count;
            else ++impl_->stats.q6_generic_tiled_projection_count;
        }
        [encoder setComputePipelineState:pipeline];
        bind_buffer(encoder, input, 0);
        bind_buffer(encoder, weight, 1);
        bind_output_buffer(encoder, output, 2);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        if (q4_input_sums != nullptr) bind_buffer(encoder, *q4_input_sums, 4U);
        [encoder dispatchThreadgroups:MTLSizeMake(
                                          static_cast<NSUInteger>((columns + dispatch.columns_per_group - 1U) / dispatch.columns_per_group),
                                          static_cast<NSUInteger>((row_count + dispatch.rows_per_group - 1U) / dispatch.rows_per_group),
                                          1U
                                      )
                  threadsPerThreadgroup:MTLSizeMake(dispatch.threads_per_group, 1U, 1U)];
    } else {
        pipeline = f16_matmul_pipeline(*impl_->state, weight.descriptor().dtype);
        const auto dispatch = select_f16_matmul_dispatch(*impl_->state, row_count, columns);
        ++impl_->stats.f16_projection_count;
        [encoder setComputePipelineState:pipeline];
        bind_buffer(encoder, input, 0);
        bind_buffer(encoder, weight, 1);
        bind_output_buffer(encoder, output, 2);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        [encoder dispatchThreadgroups:dispatch.threadgroups
                  threadsPerThreadgroup:dispatch.threads_per_threadgroup];
    }
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_target_head_rows(
    const BackendBuffer& input,
    std::uint32_t input_row_start,
    const BackendBuffer& weight,
    const BackendBuffer& logits_tile,
    std::uint32_t row_count
) {
    if (weight.descriptor().dtype == DType::q8_0) {
        encode_target_head_q8_rows(input, input_row_start, weight, logits_tile, row_count);
        return;
    }
    if ((weight.descriptor().dtype != DType::q4_k && weight.descriptor().dtype != DType::q6_k)
        || row_count == 0U || input.descriptor().dtype != DType::f32
        || logits_tile.descriptor().dtype != DType::f32
        || input_row_start > input.descriptor().shape.dimension(0)
        || row_count > input.descriptor().shape.dimension(0) - input_row_start) {
        throw std::invalid_argument("Metal native target-head row tile is invalid");
    }
    const auto shared = input.descriptor().shape.dimension(1);
    const auto columns = weight.descriptor().shape.dimension(1);
    if (shared != weight.descriptor().shape.dimension(0) || shared % 256U != 0U) {
        throw std::invalid_argument("Metal native K-quant target-head layout is invalid");
    }
    require_capacity(logits_tile,
        static_cast<std::uint64_t>(row_count) * columns * sizeof(float), "target-head logits tile");
    const MatmulDimensions dims {row_count, u32(shared, "head shared"), u32(columns, "head columns")};
    auto encoder = impl_->compute_encoder("Native K-quant target-head row tile");
    const auto dispatch = packed_row_dispatch(*impl_->state, weight.descriptor(),
        shared, row_count);
    [encoder setComputePipelineState:dispatch.pipeline];
    bind_buffer(encoder, input, 0, static_cast<std::uint64_t>(input_row_start) * shared * sizeof(float));
    bind_buffer(encoder, weight, 1);
    bind_output_buffer(encoder, logits_tile, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((columns + dispatch.columns_per_group - 1U) / dispatch.columns_per_group),
                                      (row_count + dispatch.rows_per_group - 1U) / dispatch.rows_per_group, 1U)
              threadsPerThreadgroup:MTLSizeMake(dispatch.threads_per_group, 1U, 1U)];
    if (weight.descriptor().dtype == DType::q4_k) ++impl_->stats.q4_small_batch_projection_count;
    else ++impl_->stats.q6_small_batch_projection_count;
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
    bind_output_buffer(encoder, logits_tile, 2);
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

void MetalCommandContext::encode_head_rms_norm(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    float epsilon,
    std::uint32_t row_count,
    std::uint32_t head_count,
    std::uint32_t head_dimension
) {
    if (row_count == 0U || head_count == 0U || head_dimension != 256U
        || !std::isfinite(epsilon) || epsilon <= 0.0F
        || input.descriptor().dtype != DType::f32 || weight.descriptor().dtype != DType::f32
        || output.descriptor().dtype != DType::f32) {
        throw std::invalid_argument("Metal per-head RMSNorm requires F32 256-wide heads and positive epsilon");
    }
    const auto bytes = static_cast<std::uint64_t>(row_count) * head_count * head_dimension * sizeof(float);
    require_capacity(input, bytes, "head RMSNorm input");
    require_capacity(output, bytes, "head RMSNorm output");
    require_capacity(weight, head_dimension * sizeof(float), "head RMSNorm weight");
    const HeadRMSNormDimensions dims {row_count, head_count, head_dimension, epsilon};
    auto encoder = impl_->compute_encoder("Per-head RMSNorm");
    [encoder setComputePipelineState:impl_->state->head_rms_norm_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, weight, 1);
    bind_output_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(head_count, row_count, 1U)
              threadsPerThreadgroup:MTLSizeMake(32U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_rms_norm(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    float epsilon,
    std::uint32_t row_count,
    std::uint32_t input_row_start,
    const BackendBuffer* q4_input_sums
) {
    if (q4_input_sums != nullptr && input.descriptor().shape.rank() != 2U)
        throw std::invalid_argument("Metal Q4 sum execution requires matrix input rows");
    const auto rows = row_count == 0U ? input.descriptor().shape.dimension(0) : row_count;
    const auto hidden = input.descriptor().shape.dimension(1);
    if (static_cast<std::uint64_t>(input_row_start) + rows > input.descriptor().shape.dimension(0)) {
        throw std::invalid_argument("Metal resident execution RMSNorm input row range is invalid");
    }
    require_capacity(output, rows * hidden * sizeof(float), "RMSNorm output");
    if (q4_input_sums != nullptr) {
        if (impl_->state->rms_norm_q4_sums_pipeline == nil)
            throw std::invalid_argument("Metal Q4 RMSNorm sums require prepared target pipelines");
        const auto input_elements = q4_input_element_count(input,
            u32(std::uint64_t(input_row_start) + rows, "Q4 RMSNorm input rows"));
        const auto output_elements = rows * hidden;
        require_f32_storage(weight, hidden, "Q4 RMSNorm weight");
        require_f32_storage(output, output_elements, "Q4 RMSNorm output");
        require_q4_input_sum_capacity(*q4_input_sums, output_elements);
        const auto sum_bytes = output_elements / 8U * sizeof(float);
        require_disjoint_ranges(input, input_elements * sizeof(float), *q4_input_sums,
            sum_bytes, "Q4 RMSNorm input/sums");
        require_disjoint_ranges(weight, hidden * sizeof(float), *q4_input_sums,
            sum_bytes, "Q4 RMSNorm weight/sums");
        require_disjoint_ranges(output, output_elements * sizeof(float), *q4_input_sums,
            sum_bytes, "Q4 RMSNorm output/sums");
        require_disjoint_ranges(input, input_elements * sizeof(float), output,
            output_elements * sizeof(float), "Q4 RMSNorm input/output");
        require_disjoint_ranges(weight, hidden * sizeof(float), output,
            output_elements * sizeof(float), "Q4 RMSNorm weight/output");
    }
    const RMSNormDimensions dims {
        .rows = u32(rows, "RMSNorm rows"),
        .hidden = u32(hidden, "RMSNorm hidden"),
        .input_row_start = input_row_start,
        .epsilon = epsilon,
    };
    auto encoder = impl_->compute_encoder("Gemma2 RMSNorm");
    [encoder setComputePipelineState:q4_input_sums != nullptr
        ? impl_->state->rms_norm_q4_sums_pipeline : impl_->state->rms_norm_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, weight, 1);
    bind_output_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    if (q4_input_sums != nullptr) bind_output_buffer(encoder, *q4_input_sums, 4U);
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
    bind_output_buffer(encoder, output, 3);
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
    bind_output_buffer(encoder, output, 1, output_byte_offset);
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
    bind_output_buffer(encoder, cache_key, 2);
    bind_output_buffer(encoder, cache_value, 3);
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
    if (token_count == 0U || (source.dtype != DType::f16 && source.dtype != DType::f32)
        || destination.dtype != source.dtype
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
    [encoder setComputePipelineState:source.dtype == DType::f32
        ? impl_->state->kv_cache_copy_f32_pipeline : impl_->state->gemma2_kv_cache_copy_pipeline];
    bind_buffer(encoder, source_key, 0);
    bind_buffer(encoder, source_value, 1);
    bind_output_buffer(encoder, destination_key, 2);
    bind_output_buffer(encoder, destination_value, 3);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      static_cast<NSUInteger>((row_width + 255U) / 256U),
                                      token_count,
                                      1
                                  )
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_f16_prefill_attention(
    const BackendBuffer& query,
    const BackendBuffer& key_cache,
    const BackendBuffer& value_cache,
    const BackendBuffer& current_chunk_key,
    const BackendBuffer& current_chunk_value,
    const BackendBuffer& output,
    std::uint32_t query_position_start,
    std::uint32_t head_count,
    std::uint32_t kv_head_count,
    std::uint32_t key_length,
    std::uint32_t value_length,
    std::uint32_t attention_token_count,
    float scale,
    float logit_softcap,
    std::uint32_t cache_row_mask,
    std::uint32_t query_row_count,
    bool uses_staged_chunk,
    std::uint32_t current_chunk_position_origin
) {
    const auto position_end = static_cast<std::uint64_t>(query_position_start) + query_row_count;
    if (query_row_count == 0U || head_count == 0U || kv_head_count == 0U
        || head_count % kv_head_count != 0U || key_length == 0U || value_length == 0U
        || attention_token_count == 0U || attention_token_count > position_end
        || position_end > std::numeric_limits<std::uint32_t>::max()
        || !std::isfinite(scale) || scale <= 0.0F
        || !std::isfinite(logit_softcap) || logit_softcap < 0.0F
        || current_chunk_position_origin > query_position_start
        || (!uses_staged_chunk && current_chunk_position_origin != 0U)) {
        throw std::invalid_argument("Metal F16 prefill attention dimensions or range are invalid");
    }
    const auto cache_rows = static_cast<std::uint64_t>(cache_row_mask) + 1U;
    const auto stage_rank = current_chunk_key.descriptor().shape.rank();
    const auto stage_rows = stage_rank == 2U || stage_rank == 3U
        ? current_chunk_key.descriptor().shape.dimension(0U) : 0U;
    if ((cache_rows & (cache_rows - 1U)) != 0U || attention_token_count > cache_rows
        || stage_rows == 0U || (stage_rows & (stage_rows - 1U)) != 0U
        || query_row_count > stage_rows) {
        throw std::invalid_argument("Metal F16 prefill attention cache capacity is invalid");
    }
    // A resident ring can hold the union of all query windows only when that
    // union fits its physical capacity. Otherwise the old prefix must stay
    // separate from the current chunk until every query has consumed it.
    const auto required_union_rows = std::min(static_cast<std::uint64_t>(query_position_start) + 1U,
        static_cast<std::uint64_t>(attention_token_count)) + query_row_count - 1U;
    if (!uses_staged_chunk && required_union_rows > cache_rows)
        throw std::invalid_argument("Metal F16 prefill attention requires staged rows for a wrapped query block");
    const auto require_cache = [&](const BackendBuffer& buffer,
                                   std::uint64_t rows, std::uint32_t width) {
        const auto& descriptor = buffer.descriptor();
        const auto row_elements = u32(static_cast<std::uint64_t>(kv_head_count) * width,
            "prefill attention cache row addressing");
        const auto& shape = descriptor.shape;
        const bool matching_rows = (shape.rank() == 3U && shape.dimension(0U) == rows
            && shape.dimension(1U) == kv_head_count && shape.dimension(2U) == width)
            || (shape.rank() == 2U && shape.dimension(0U) == rows
                && shape.dimension(1U) == row_elements);
        if (descriptor.dtype != DType::f16 || descriptor.packing != TensorPacking::canonical
            || !matching_rows) {
            throw std::invalid_argument("Metal F16 prefill attention requires canonical F16 KV storage");
        }
        if (rows > std::numeric_limits<std::uint32_t>::max() / row_elements)
            throw std::invalid_argument("Metal F16 prefill attention cache exceeds uint32 addressing");
        const auto elements = rows * row_elements;
        require_capacity(buffer, elements * sizeof(std::uint16_t), "prefill attention cache");
        (void)storage(buffer);
    };
    require_cache(key_cache, cache_rows, key_length);
    require_cache(value_cache, cache_rows, value_length);
    require_cache(current_chunk_key, stage_rows, key_length);
    require_cache(current_chunk_value, stage_rows, value_length);
    if (uses_staged_chunk) {
        for (const auto* staged : {&current_chunk_key, &current_chunk_value}) {
            for (const auto* resident : {&key_cache, &value_cache}) {
                require_disjoint_ranges(*staged, staged->descriptor().byte_count,
                    *resident, resident->descriptor().byte_count, "prefill attention staged/resident KV");
            }
        }
    }
    const auto require_rows = [&](const BackendBuffer& buffer, std::uint32_t width) {
        const auto& descriptor = buffer.descriptor();
        const auto row_width = u32(static_cast<std::uint64_t>(head_count) * width,
            "prefill attention row width");
        if (descriptor.dtype != DType::f32 || descriptor.packing != TensorPacking::canonical
            || descriptor.shape.rank() != 2U
            || descriptor.shape.dimension(0U) < query_row_count
            || descriptor.shape.dimension(1U) != row_width) {
            throw std::invalid_argument("Metal F16 prefill attention query/output layout is invalid");
        }
        const auto elements = static_cast<std::uint64_t>(query_row_count) * row_width;
        static_cast<void>(u32(elements, "prefill attention row addressing"));
        require_capacity(buffer, elements * sizeof(float), "prefill attention rows");
        (void)storage(buffer);
    };
    require_rows(query, key_length);
    require_rows(output, value_length);
    const auto output_bytes = static_cast<std::uint64_t>(query_row_count)
        * head_count * value_length * sizeof(float);
    for (const auto* source : {&query, &key_cache, &value_cache,
                              &current_chunk_key, &current_chunk_value}) {
        require_disjoint_ranges(*source, source->descriptor().byte_count,
            output, output_bytes, "prefill attention source/output");
    }
    const bool use_matrix = key_length == 256U && value_length == 256U;
    const auto generic_scratch_bytes = (static_cast<std::uint64_t>(key_length)
        + value_length) * sizeof(float);
    const auto aligned_scratch_bytes = (generic_scratch_bytes + 15U) & ~std::uint64_t {15U};
    if (!use_matrix && aligned_scratch_bytes > impl_->state->device.maxThreadgroupMemoryLength)
        throw std::invalid_argument("Metal F16 prefill attention head widths exceed device threadgroup capacity");
    const F16PrefillAttentionDimensions dims {
        query_row_count, head_count, kv_head_count, key_length, value_length,
        attention_token_count, query_position_start, cache_row_mask,
        u32(stage_rows - 1U, "prefill attention stage mask"),
        uses_staged_chunk ? 1U : 0U, current_chunk_position_origin, scale, logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma fused prefill attention");
    [encoder setComputePipelineState:use_matrix
        ? impl_->state->gemma2_prefill_attention_pipeline
        : impl_->state->gemma2_prefill_attention_generic_pipeline];
    bind_buffer(encoder, query, 0U);
    bind_buffer(encoder, key_cache, 1U);
    bind_buffer(encoder, value_cache, 2U);
    bind_buffer(encoder, current_chunk_key, 3U);
    bind_buffer(encoder, current_chunk_value, 4U);
    bind_output_buffer(encoder, output, 5U);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:6U];
    if (!use_matrix)
        [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(aligned_scratch_bytes) atIndex:0U];
    [encoder dispatchThreadgroups:MTLSizeMake(use_matrix
        ? (query_row_count + 7U) / 8U : query_row_count, head_count, 1U)
        threadsPerThreadgroup:MTLSizeMake(use_matrix ? 128U : 32U, 1U, 1U)];
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
    bind_output_buffer(encoder, output, 3, static_cast<std::uint64_t>(output_row) * row_bytes);
    bind_output_buffer(encoder, partial_output, 4);
    bind_output_buffer(encoder, partial_maxima, 5);
    bind_output_buffer(encoder, partial_denominators, 6);
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
        bind_output_buffer(encoder, output, 3, static_cast<std::uint64_t>(output_row) * row_bytes);
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
    bind_output_buffer(encoder, status, 2);
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
    bind_output_buffer(encoder, resolved_requests, 1);
    bind_output_buffer(encoder, histories, 2);
    bind_output_buffer(encoder, status, 3);
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
    float frequency_base,
    float frequency_scale
) {
    if (row_count == 0U || head_count == 0U || head_dimension == 0U
        || rope_dimension_count == 0U || rope_dimension_count > head_dimension
        || (rope_dimension_count & 1U) != 0U || !std::isfinite(frequency_base)
        || frequency_base <= 0.0F || !std::isfinite(frequency_scale) || frequency_scale <= 0.0F) {
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
        row_count, head_count, head_dimension, rope_dimension_count, frequency_base, frequency_scale,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier query RoPE");
    [encoder setComputePipelineState:impl_->state->verifier_rope_pipeline];
    bind_buffer(encoder, input, 0);
    bind_buffer(encoder, positions, 1);
    bind_output_buffer(encoder, output, 2);
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
    float frequency_base,
    float frequency_scale,
    const BackendBuffer* numerical_status
) {
    const auto cache_dtype = private_key.descriptor().dtype;
    if ((cache_dtype != DType::f16 && cache_dtype != DType::f32)
        || private_value.descriptor().dtype != cache_dtype
        || raw_key.descriptor().dtype != DType::f32 || raw_value.descriptor().dtype != DType::f32
        || !std::isfinite(frequency_base) || frequency_base <= 0.0F
        || !std::isfinite(frequency_scale) || frequency_scale <= 0.0F) {
        throw std::invalid_argument("Metal private K/V dtype or RoPE frequency is invalid");
    }
    const std::uint64_t cache_element_bytes = cache_dtype == DType::f32 ? sizeof(float) : sizeof(std::uint16_t);
    require_capacity(positions, static_cast<std::uint64_t>(row_count) * sizeof(std::uint32_t), "private K/V positions");
    const auto row_width = static_cast<std::uint64_t>(head_count) * head_dimension;
    if (row_count == 0U || row_width == 0U || rope_dimension_count > head_dimension
        || (rope_dimension_count & 1U) != 0U) {
        throw std::invalid_argument("Metal verifier private K/V parameters are invalid");
    }
    require_capacity(raw_key, static_cast<std::uint64_t>(row_count) * row_width * sizeof(float),
        "verifier raw key");
    require_capacity(raw_value, static_cast<std::uint64_t>(row_count) * row_width * sizeof(float),
        "verifier raw value");
    require_capacity(private_key, static_cast<std::uint64_t>(row_count) * row_width * cache_element_bytes,
        "verifier private key");
    require_capacity(private_value, static_cast<std::uint64_t>(row_count) * row_width * cache_element_bytes,
        "verifier private value");
    if (numerical_status != nullptr) {
        if (cache_dtype != DType::f16 || impl_->state->verifier_kv_store_f16_checked_pipeline == nil)
            throw std::invalid_argument("Metal checked K/V store requires precise F16 cache storage");
        require_f32_storage(*numerical_status, 1U, "verifier K/V numerical status");
        for (const auto* source : {&raw_key, &raw_value, &positions, &private_key, &private_value})
            require_disjoint_ranges(*source, source->descriptor().byte_count, *numerical_status,
                sizeof(std::uint32_t), "K/V numerical status and storage");
    }
    const VerifierKVStoreDimensions dims {
        row_count, head_count, head_dimension, rope_dimension_count, frequency_base, frequency_scale,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier private K/V store");
    [encoder setComputePipelineState:numerical_status != nullptr
        ? impl_->state->verifier_kv_store_f16_checked_pipeline
        : (cache_dtype == DType::f32
            ? impl_->state->verifier_kv_store_f32_pipeline : impl_->state->verifier_kv_store_pipeline)];
    bind_buffer(encoder, raw_key, 0);
    bind_buffer(encoder, raw_value, 1);
    bind_buffer(encoder, positions, 2);
    bind_output_buffer(encoder, private_key, 3);
    bind_output_buffer(encoder, private_value, 4);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    if (numerical_status != nullptr) bind_output_buffer(encoder, *numerical_status, 6U);
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
    float logit_softcap,
    const BackendBuffer* q4_input_sums
) {
    if (row_count == 0U || kv_head_count == 0U
        || query_head_count != 2U * kv_head_count
        || key_dimension != value_dimension || key_dimension != 256U
        || !std::isfinite(scale) || scale <= 0.0F
        || !std::isfinite(logit_softcap) || logit_softcap < 0.0F) {
        throw std::invalid_argument("Metal verifier attention supports the two query heads per KV head and head width 256");
    }
    const auto cache_dtype = resident_key.descriptor().dtype;
    if ((cache_dtype != DType::f16 && cache_dtype != DType::f32)
        || resident_value.descriptor().dtype != cache_dtype
        || private_key.descriptor().dtype != cache_dtype || private_value.descriptor().dtype != cache_dtype) {
        throw std::invalid_argument("Metal verifier attention K/V storage dtypes must agree");
    }
    const auto row_elements = static_cast<std::uint64_t>(row_count) * query_head_count;
    const auto fused_attention_pipeline = cache_dtype == DType::f32
        ? impl_->state->verifier_attention_q4_sums_pipeline
        : impl_->state->verifier_attention_q4_sums_f16_pipeline;
    if (q4_input_sums != nullptr) {
        if (fused_attention_pipeline == nil)
            throw std::invalid_argument("Metal Q4 attention sums require prepared target attention");
        const auto elements = std::uint64_t(u32(row_elements, "Q4 attention row heads")) * value_dimension;
        require_f32_storage(query, elements, "Q4 attention query");
        require_f32_storage(output, elements, "Q4 attention output");
        require_q4_input_sum_capacity(*q4_input_sums, elements);
        for (const auto* source : {&query, &resident_key, &resident_value, &private_key, &private_value,
                 &schedules, &ancestors, &output, &partial_output, &partial_maxima, &partial_denominators})
            require_disjoint_ranges(*source, source->descriptor().byte_count, *q4_input_sums,
                elements / 8U * sizeof(float), "Q4 attention storage/sums");
    }
    require_capacity(query, row_elements * key_dimension * sizeof(float), "verifier query");
    require_capacity(output, row_elements * value_dimension * sizeof(float), "verifier attention output");
    require_capacity(partial_output, row_elements * engine::gemma2_decode_attention_max_partitions
        * value_dimension * sizeof(float), "verifier partial output");
    require_capacity(partial_maxima, row_elements * engine::gemma2_decode_attention_max_partitions
        * sizeof(float), "verifier partial maxima");
    require_capacity(partial_denominators, row_elements * engine::gemma2_decode_attention_max_partitions
        * sizeof(float), "verifier partial denominators");
    static_assert(sizeof(engine::VerifierRowAttention) == 4U * sizeof(std::uint32_t));
    const auto schedule_bytes = static_cast<std::uint64_t>(row_count)
        * sizeof(engine::VerifierRowAttention);
    if (schedule_byte_offset % alignof(std::uint32_t) != 0U
        || schedule_byte_offset > schedules.descriptor().byte_count
        || schedule_bytes > schedules.descriptor().byte_count - schedule_byte_offset) {
        throw std::invalid_argument("Metal verifier attention schedule extent is invalid");
    }
    // Both ordinary and graph execution stage these immutable rows on the CPU.
    // Reading shared staging here avoids an extra schedule representation or a
    // GPU readback. Buffer-view and operation offsets are both applied below.
    const auto* schedule_data = static_cast<const std::byte*>(
        metal_buffer_contents(storage(schedules), schedule_byte_offset));
    std::uint32_t maximum_tokens = 0U;
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        std::array<std::uint32_t, 4> packed;
        std::memcpy(packed.data(), schedule_data + row * sizeof(packed), sizeof(packed));
        const auto tokens = static_cast<std::uint64_t>(packed[1]) + packed[3];
        if (tokens == 0U || tokens > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("Metal verifier attention token count is invalid");
        }
        maximum_tokens = std::max(maximum_tokens, static_cast<std::uint32_t>(tokens));
    }
    const auto maximum_partitions = gemma2_decode_attention_partition_count(maximum_tokens);
    const VerifierAttentionDimensions dims {
        row_count, cache_row_mask, query_head_count, kv_head_count,
        key_dimension, value_dimension, scale, logit_softcap,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier tree attention");
    [encoder setComputePipelineState:q4_input_sums != nullptr
        ? fused_attention_pipeline
        : (cache_dtype == DType::f32 ? impl_->state->verifier_attention_f32_pipeline
            : impl_->state->verifier_attention_pipeline)];
    bind_buffer(encoder, query, 0);
    bind_buffer(encoder, resident_key, 1);
    bind_buffer(encoder, resident_value, 2);
    bind_buffer(encoder, private_key, 3);
    bind_buffer(encoder, private_value, 4);
    bind_buffer(encoder, schedules, 5, schedule_byte_offset);
    bind_buffer(encoder, ancestors, 6);
    bind_output_buffer(encoder, output, 7);
    bind_output_buffer(encoder, partial_output, 8);
    bind_output_buffer(encoder, partial_maxima, 9);
    bind_output_buffer(encoder, partial_denominators, 10);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:11];
    if (q4_input_sums != nullptr) bind_output_buffer(encoder, *q4_input_sums, 12U);
    [encoder dispatchThreadgroups:MTLSizeMake(kv_head_count,
                                      maximum_partitions,
                                      row_count)
              threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
    ++impl_->stats.compute_dispatch_count;
    // One-partition rows already write final output. Keep the fixed partial
    // stride and each shader row's partition/reduction order for every other case.
    if (maximum_partitions == 1U) return;
    encode_memory_barrier();
    encoder = impl_->compute_encoder("Gemma2 verifier tree attention reduction");
    [encoder setComputePipelineState:q4_input_sums != nullptr
        ? impl_->state->verifier_attention_reduce_q4_sums_pipeline : impl_->state->verifier_attention_reduce_pipeline];
    bind_buffer(encoder, partial_output, 0);
    bind_buffer(encoder, partial_maxima, 1);
    bind_buffer(encoder, partial_denominators, 2);
    bind_buffer(encoder, schedules, 3, schedule_byte_offset);
    bind_output_buffer(encoder, output, 4);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    if (q4_input_sums != nullptr) bind_output_buffer(encoder, *q4_input_sums, 6U);
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
    if (row_count == 0U || row_count > engine::VerifierRowMax
        || row_width == 0U) {
        throw std::invalid_argument("Metal verifier accepted-path scatter extent is invalid");
    }
    const auto dtype = private_key.descriptor().dtype;
    if ((dtype != DType::f16 && dtype != DType::f32)
        || private_value.descriptor().dtype != dtype || resident_key.descriptor().dtype != dtype
        || resident_value.descriptor().dtype != dtype) {
        throw std::invalid_argument("Metal K/V scatter storage dtypes must agree");
    }
    require_capacity(selected_nodes, static_cast<std::uint64_t>(row_count) * sizeof(std::uint32_t), "K/V scatter selected rows");
    const VerifierKVScatterDimensions dims {
        row_count, row_width, destination_position, destination_row_mask,
    };
    auto encoder = impl_->compute_encoder("Gemma2 verifier accepted-path K/V scatter");
    [encoder setComputePipelineState:dtype == DType::f32
        ? impl_->state->verifier_kv_scatter_f32_pipeline : impl_->state->verifier_kv_scatter_pipeline];
    bind_buffer(encoder, private_key, 0);
    bind_buffer(encoder, private_value, 1);
    bind_buffer(encoder, selected_nodes, 2);
    bind_output_buffer(encoder, resident_key, 3);
    bind_output_buffer(encoder, resident_value, 4);
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
    bind_output_buffer(encoder, output, 2);
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
    bind_output_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((elements + 255U) / 256U), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++impl_->stats.compute_dispatch_count;
}

void MetalCommandContext::encode_gelu_tanh_multiply(
    const BackendBuffer& gate,
    const BackendBuffer& up,
    const BackendBuffer& output,
    std::uint64_t element_count,
    const BackendBuffer* q4_input_sums
) {
    const auto elements = element_count == 0U ? gate.descriptor().element_count : element_count;
    require_capacity(up, elements * sizeof(float), "GELU-tanh up");
    require_capacity(output, elements * sizeof(float), "GELU-tanh output");
    if (q4_input_sums != nullptr) {
        if (impl_->state->gelu_tanh_multiply_q4_sums_pipeline == nil
            || gate.descriptor().shape.rank() != 2U)
            throw std::invalid_argument("Metal Q4 GELU sums require prepared target matrix rows");
        const auto width = gate.descriptor().shape.dimension(1);
        if (width == 0U || width % 256U != 0U || elements == 0U || elements % width != 0U)
            throw std::invalid_argument("Metal Q4 GELU sum extent requires complete 256-aligned rows");
        static_cast<void>(q4_input_element_count(gate, u32(elements / width, "Q4 GELU rows")));
        require_f32_storage(up, elements, "Q4 GELU up");
        require_f32_storage(output, elements, "Q4 GELU output");
        require_q4_input_sum_capacity(*q4_input_sums, elements);
        const auto sum_bytes = elements / 8U * sizeof(float);
        for (const auto* source : {&gate, &up, &output})
            require_disjoint_ranges(*source, elements * sizeof(float), *q4_input_sums,
                sum_bytes, "Q4 GELU values/sums");
        require_disjoint_ranges(gate, elements * sizeof(float), output,
            elements * sizeof(float), "Q4 GELU gate/output");
        require_disjoint_ranges(up, elements * sizeof(float), output,
            elements * sizeof(float), "Q4 GELU up/output");
    }
    const ElementwiseDimensions dims {.elements = u32(elements, "GELU-tanh elements")};
    auto encoder = impl_->compute_encoder("Gemma2 GELU-tanh multiply");
    [encoder setComputePipelineState:q4_input_sums != nullptr
        ? impl_->state->gelu_tanh_multiply_q4_sums_pipeline : impl_->state->gelu_tanh_multiply_pipeline];
    bind_buffer(encoder, gate, 0);
    bind_buffer(encoder, up, 1);
    bind_output_buffer(encoder, output, 2);
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    if (q4_input_sums != nullptr) bind_output_buffer(encoder, *q4_input_sums, 4U);
    const auto threads = q4_input_sums != nullptr ? elements / 8U : elements;
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((threads + 255U) / 256U), 1, 1)
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
    bind_output_buffer(encoder, output, 1);
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
    if (!std::isfinite(softcap) || softcap < 0.0F) {
        throw std::invalid_argument("Metal token-candidate softcap must be finite and nonnegative");
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
        bind_output_buffer(encoder, workspace.compact_result, 1);
        [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        return 1U;
    }

    auto encoder = impl_->compute_encoder("Gemma2 compact token candidates");
    [encoder setComputePipelineState:impl_->state->token_candidate_tile_pipeline];
    bind_buffer(encoder, logits, 0);
    bind_output_buffer(encoder, workspace.candidate_runs_a, 1);
    bind_output_buffer(encoder, workspace.compact_result, 2);
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
        bind_output_buffer(encoder, input_is_a ? workspace.candidate_runs_b : workspace.candidate_runs_a, 1);
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
    bind_output_buffer(encoder, workspace.compact_result, 3);
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
    if (!std::isfinite(softcap) || softcap < 0.0F) {
        throw std::invalid_argument("Metal batched token-candidate softcap must be finite and nonnegative");
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
        bind_output_buffer(encoder, workspace.compact_result, 1);
        [encoder setBytes:&batch length:sizeof(batch) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        ++impl_->stats.compute_dispatch_count;
        return 1U;
    }

    auto encoder = impl_->compute_encoder("Gemma2 batched compact token candidates");
    [encoder setComputePipelineState:impl_->state->token_candidate_batch_tile_pipeline];
    bind_buffer(encoder, logits, 0);
    bind_output_buffer(encoder, workspace.candidate_runs_a, 1);
    bind_output_buffer(encoder, workspace.compact_result, 2);
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
        bind_output_buffer(encoder, input_is_a ? workspace.candidate_runs_b : workspace.candidate_runs_a, 1);
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
    bind_output_buffer(encoder, workspace.compact_result, 3);
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
        || !std::isfinite(softcap) || softcap < 0.0F) {
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
        bind_output_buffer(encoder, workspace.compact_results, 1, result_offset);
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
    bind_output_buffer(encoder, workspace.candidate_runs_a, 2);
    bind_output_buffer(encoder, workspace.compact_results, 3, result_offset);
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
        bind_output_buffer(encoder,
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
    bind_output_buffer(encoder, workspace.compact_results, 4, result_offset);
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
    @autoreleasepool {
        [encoder pushDebugGroup:[NSString stringWithUTF8String:owned_label.c_str()]];
    }
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
    keep_metal_buffers_resident();
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
