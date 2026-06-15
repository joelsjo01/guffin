#include "metal_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "../ops/validation.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

constexpr const char* matmul_f32_f16_gguf_weight_source = R"metal(
#include <metal_stdlib>
using namespace metal;

struct MatmulDimensions {
    uint rows;
    uint shared;
    uint columns;
};

kernel void matmul_f32_f16_gguf_weight_kernel(
    const device float* left [[buffer(0)]],
    const device half* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    if (row >= dims.rows || column >= dims.columns) {
        return;
    }

    float sum = 0.0f;
    for (uint inner = 0; inner < dims.shared; ++inner) {
        sum += left[row * dims.shared + inner] * float(right[column * dims.shared + inner]);
    }

    output[row * dims.columns + column] = sum;
}

kernel void matmul_f32_f16_gguf_weight_add_kernel(
    const device float* left [[buffer(0)]],
    const device half* right [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant MatmulDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    if (row >= dims.rows || column >= dims.columns) {
        return;
    }

    float sum = 0.0f;
    for (uint inner = 0; inner < dims.shared; ++inner) {
        sum += left[row * dims.shared + inner] * float(right[column * dims.shared + inner]);
    }

    const uint output_index = row * dims.columns + column;
    output[output_index] = residual[output_index] + sum;
}

struct ElementwiseDimensions {
    uint elements;
};

kernel void silu_multiply_f32_kernel(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseDimensions& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.elements) {
        return;
    }

    const float value = gate[gid];
    output[gid] = (value / (1.0f + exp(-value))) * up[gid];
}

kernel void add_f32_kernel(
    const device float* left [[buffer(0)]],
    const device float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseDimensions& dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.elements) {
        return;
    }

    output[gid] = left[gid] + right[gid];
}

struct RMSNormDimensions {
    uint rows;
    uint hidden;
    float epsilon;
};

kernel void rms_norm_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormDimensions& dims [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= dims.rows) {
        return;
    }

    float sum_squares = 0.0f;
    const uint row_offset = row * dims.hidden;
    for (uint column = 0; column < dims.hidden; ++column) {
        const float value = input[row_offset + column];
        sum_squares += value * value;
    }

    const float inverse_rms = rsqrt((sum_squares / float(dims.hidden)) + dims.epsilon);
    for (uint column = 0; column < dims.hidden; ++column) {
        output[row_offset + column] = input[row_offset + column] * inverse_rms * weight[column];
    }
}

struct LlamaCachedAttentionDimensions {
    uint token_count;
    uint cache_position;
    uint cached_token_count;
    uint head_count;
    uint kv_head_count;
    uint key_length;
    uint value_length;
    uint rope_dimension_count;
    float frequency_base;
    float scale;
};

inline float llama_rope_angle(
    uint position,
    uint pair_offset,
    uint rope_dimension_count,
    float frequency_base
) {
    return float(position) / pow(frequency_base, float(pair_offset) / float(rope_dimension_count));
}

kernel void llama_cached_kv_append_f32_kernel(
    const device float* key [[buffer(0)]],
    const device float* value [[buffer(1)]],
    device float* cache_key [[buffer(2)]],
    device float* cache_value [[buffer(3)]],
    constant LlamaCachedAttentionDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    if (row >= dims.token_count) {
        return;
    }

    const uint key_width = dims.kv_head_count * dims.key_length;
    const uint value_width = dims.kv_head_count * dims.value_length;
    const uint cache_row = dims.cache_position + row;

    if (column < key_width) {
        const uint head_column = column % dims.key_length;
        float output_value = key[row * key_width + column];

        if (head_column < dims.rope_dimension_count) {
            const uint pair_column = head_column - (head_column % 2);
            const uint head_first_column = column - head_column;
            const uint first_index = row * key_width + head_first_column + pair_column;
            const float first = key[first_index];
            const float second = key[first_index + 1];
            const float angle = llama_rope_angle(
                cache_row,
                pair_column,
                dims.rope_dimension_count,
                dims.frequency_base
            );
            const float angle_cos = cos(angle);
            const float angle_sin = sin(angle);

            output_value = head_column == pair_column
                ? first * angle_cos - second * angle_sin
                : first * angle_sin + second * angle_cos;
        }

        cache_key[cache_row * key_width + column] = output_value;
    }

    if (column < value_width) {
        cache_value[cache_row * value_width + column] = value[row * value_width + column];
    }
}

kernel void llama_cached_attention_output_f32_kernel(
    const device float* query [[buffer(0)]],
    const device float* cache_key [[buffer(1)]],
    const device float* cache_value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant LlamaCachedAttentionDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint output_column = gid.x;
    const uint query_row = gid.y;
    const uint output_width = dims.head_count * dims.value_length;
    if (query_row >= dims.token_count || output_column >= output_width) {
        return;
    }

    const uint key_width = dims.kv_head_count * dims.key_length;
    const uint value_width = dims.kv_head_count * dims.value_length;
    const uint query_width = dims.head_count * dims.key_length;
    const uint kv_group_size = dims.head_count / dims.kv_head_count;
    const uint head = output_column / dims.value_length;
    const uint value_column = output_column % dims.value_length;
    const uint kv_head = head / kv_group_size;
    const uint query_first_column = head * dims.key_length;
    const uint key_first_column = kv_head * dims.key_length;
    const uint value_first_column = kv_head * dims.value_length;
    const uint query_position = dims.cache_position + query_row;

    float max_score = -3.402823466e+38f;
    uint has_score = 0;
    float denominator = 0.0f;
    float accumulator = 0.0f;

    for (uint key_row = 0; key_row < dims.cached_token_count; ++key_row) {
        if (key_row > query_position) {
            continue;
        }

        float dot = 0.0f;
        for (uint column = 0; column < dims.key_length; ++column) {
            float query_value = query[query_row * query_width + query_first_column + column];
            if (column < dims.rope_dimension_count) {
                const uint pair_column = column - (column % 2);
                const uint first_index = query_row * query_width + query_first_column + pair_column;
                const float first = query[first_index];
                const float second = query[first_index + 1];
                const float angle = llama_rope_angle(
                    query_position,
                    pair_column,
                    dims.rope_dimension_count,
                    dims.frequency_base
                );
                const float angle_cos = cos(angle);
                const float angle_sin = sin(angle);
                query_value = column == pair_column
                    ? first * angle_cos - second * angle_sin
                    : first * angle_sin + second * angle_cos;
            }

            dot += query_value * cache_key[key_row * key_width + key_first_column + column];
        }

        const float score = dot * dims.scale;
        if (score > max_score) {
            const float rescale = has_score == 0 ? 0.0f : exp(max_score - score);
            denominator *= rescale;
            accumulator *= rescale;
            max_score = score;
            has_score = 1;
        }

        const float weight = exp(score - max_score);
        denominator += weight;
        accumulator += weight * cache_value[key_row * value_width + value_first_column + value_column];
    }

    output[query_row * output_width + output_column] = accumulator / denominator;
}
)metal";

struct MatmulDimensions {
    std::uint32_t rows = 0;
    std::uint32_t shared = 0;
    std::uint32_t columns = 0;
};

struct ElementwiseDimensions {
    std::uint32_t elements = 0;
};

struct RMSNormDimensions {
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
    float epsilon = 0.0F;
};

struct LlamaCachedAttentionDimensions {
    std::uint32_t token_count = 0;
    std::uint32_t cache_position = 0;
    std::uint32_t cached_token_count = 0;
    std::uint32_t head_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t key_length = 0;
    std::uint32_t value_length = 0;
    std::uint32_t rope_dimension_count = 0;
    float frequency_base = 0.0F;
    float scale = 0.0F;
};

struct LlamaCachedBlockScratchBuffers {
    id<MTLBuffer> attention_normalized = nil;
    id<MTLBuffer> query = nil;
    id<MTLBuffer> key = nil;
    id<MTLBuffer> value = nil;
    id<MTLBuffer> attention_output = nil;
    id<MTLBuffer> attention_residual = nil;
    id<MTLBuffer> feed_forward_normalized = nil;
    id<MTLBuffer> gate = nil;
    id<MTLBuffer> up = nil;
    id<MTLBuffer> gated = nil;
};

struct MatmulPipelineState {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> matmul_add_pipeline = nil;
    id<MTLComputePipelineState> silu_multiply_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_kv_append_pipeline = nil;
    id<MTLComputePipelineState> llama_cached_attention_output_pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    std::mutex llama_cached_block_scratch_mutex;
    LlamaCachedBlockScratchBuffers llama_cached_block_scratch;

    MatmulPipelineState();
};

class CpuBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    explicit CpuBackendBufferStorage(const TensorStorage& storage)
        : storage_(TensorStorage::from_bytes(storage.descriptor(), storage.bytes())) {}

    [[nodiscard]] runtime::BackendKind backend() const noexcept override {
        return runtime::BackendKind::cpu;
    }

    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override {
        return storage_.descriptor();
    }

    [[nodiscard]] const TensorStorage& storage() const noexcept {
        return storage_;
    }

private:
    TensorStorage storage_;
};

class MetalBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    MetalBackendBufferStorage(const TensorStorage& storage, id<MTLBuffer> buffer)
        : MetalBackendBufferStorage(storage.descriptor(), buffer) {}

    MetalBackendBufferStorage(TensorDescriptor descriptor, id<MTLBuffer> buffer)
        : descriptor_(std::move(descriptor)),
          buffer_(buffer) {}

    [[nodiscard]] runtime::BackendKind backend() const noexcept override {
        return runtime::BackendKind::metal;
    }

    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }

    [[nodiscard]] id<MTLBuffer> buffer() const noexcept {
        return buffer_;
    }

private:
    TensorDescriptor descriptor_;
    id<MTLBuffer> buffer_ = nil;
};

[[nodiscard]] std::string ns_error_message(NSError* error, const char* fallback) {
    if (error == nil) {
        return fallback;
    }

    NSString* description = [error localizedDescription];
    if (description == nil) {
        return fallback;
    }

    const char* utf8 = [description UTF8String];
    if (utf8 == nullptr) {
        return fallback;
    }

    return std::string(utf8);
}

[[nodiscard]] runtime::BackendError metal_error(
    runtime::BackendErrorCode code,
    const std::string& detail
) {
    return runtime::BackendError(code, runtime::BackendKind::metal, detail);
}

[[nodiscard]] id<MTLBuffer> reusable_scratch_buffer(
    id<MTLDevice> device,
    __strong id<MTLBuffer>& buffer,
    std::uint64_t byte_count,
    const char* label
) {
    const auto length = static_cast<NSUInteger>(byte_count);
    if (buffer == nil || [buffer length] < length) {
        buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                std::string("Metal Llama cached block scratch buffer allocation failed: ") + label
            );
        }
    }

    return buffer;
}

[[nodiscard]] id<MTLDevice> metal_device_or_throw() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw metal_error(
            runtime::BackendErrorCode::unavailable_metal_device,
            "Metal backend unavailable: no system default Metal device"
        );
    }

    return device;
}

[[nodiscard]] std::uint32_t require_u32_dimension(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            std::string("Metal F32xF16 GGUF weight matmul dimension is too large: ") + name
        );
    }

    return static_cast<std::uint32_t>(value);
}

void require_matrix_descriptor(const TensorDescriptor& descriptor) {
    constexpr std::size_t matrix_rank = 2;
    if (descriptor.shape.rank() != matrix_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(matrix_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_matmul_inner_dimensions(const TensorDescriptor& left, const TensorDescriptor& right) {
    const auto left_columns = left.shape.dimension(1);
    const auto right_rows = right.shape.dimension(0);
    if (left_columns != right_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_columns, right_rows);
    }
}

void require_f32_f16_matmul_gguf_weight_operands(
    const TensorStorage& left,
    const TensorDescriptor& right
) {
    ops::require_f32_descriptor(left.descriptor());
    ops::require_f16_descriptor(right);
    require_matrix_descriptor(left.descriptor());
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left.descriptor(), right);
}

void require_f32_f16_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
) {
    ops::require_f32_descriptor(left);
    ops::require_f16_descriptor(right);
    require_matrix_descriptor(left);
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left, right);
}

void require_same_backend(
    runtime::BackendKind left,
    runtime::BackendKind right,
    const char* operation
) {
    if (left != right) {
        throw std::invalid_argument(std::string(operation) + " backend buffers must use the same backend");
    }
}

void require_silu_multiply_f32_operands(
    const TensorDescriptor& gate,
    const TensorDescriptor& up
) {
    ops::require_f32_descriptor(gate);
    ops::require_f32_descriptor(up);
    ops::require_same_shape(gate, up);
}

void require_gated_silu_projection_f32_f16_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight
) {
    require_f32_f16_matmul_gguf_weight_operands(left, gate_weight);
    require_f32_f16_matmul_gguf_weight_operands(left, up_weight);
    ops::require_f16_descriptor(down_weight);
    require_matrix_descriptor(down_weight);

    const auto feed_forward = gate_weight.shape.dimension(1);
    const auto up_feed_forward = up_weight.shape.dimension(1);
    if (feed_forward != up_feed_forward) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, feed_forward, up_feed_forward);
    }

    const auto down_rows = down_weight.shape.dimension(0);
    if (down_rows != feed_forward) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, feed_forward, down_rows);
    }
}

void require_f32_vector_descriptor(const TensorDescriptor& descriptor) {
    constexpr std::size_t vector_rank = 1;
    ops::require_f32_descriptor(descriptor);
    if (descriptor.shape.rank() != vector_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(vector_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_positive_epsilon(float epsilon) {
    if (!(epsilon > 0.0F)) {
        throw std::invalid_argument("RMSNorm epsilon must be positive");
    }
}

void require_positive_finite_scale(float scale) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("attention scale must be positive and finite");
    }
}

void require_positive_frequency_base(float frequency_base) {
    if (!(frequency_base > 0.0F)) {
        throw std::invalid_argument("RoPE frequency base must be positive");
    }
}

void require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& attention_output,
    const TensorDescriptor& attention_output_weight,
    const TensorDescriptor& feed_forward_norm_weight,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight,
    float epsilon
) {
    ops::require_f32_descriptor(input);
    ops::require_f32_descriptor(attention_output);
    require_matrix_descriptor(input);
    require_matrix_descriptor(attention_output);

    const auto token_count = input.shape.dimension(0);
    const auto attention_token_count = attention_output.shape.dimension(0);
    if (token_count != attention_token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, token_count, attention_token_count);
    }

    require_f32_f16_matmul_gguf_weight_operands(attention_output, attention_output_weight);
    const auto hidden = input.shape.dimension(1);
    const auto attention_output_columns = attention_output_weight.shape.dimension(1);
    if (hidden != attention_output_columns) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, attention_output_columns);
    }

    require_f32_vector_descriptor(feed_forward_norm_weight);
    const auto norm_hidden = feed_forward_norm_weight.shape.dimension(0);
    if (hidden != norm_hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, norm_hidden);
    }

    require_gated_silu_projection_f32_f16_gguf_weight_operands(
        input,
        gate_weight,
        up_weight,
        down_weight
    );
    const auto down_columns = down_weight.shape.dimension(1);
    if (hidden != down_columns) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, down_columns);
    }

    require_positive_epsilon(epsilon);
}

void require_llama_cached_attention_f32_operands(
    const TensorDescriptor& query,
    const TensorDescriptor& key,
    const TensorDescriptor& value,
    const TensorDescriptor& cache_key,
    const TensorDescriptor& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    ops::require_f32_descriptor(query);
    ops::require_f32_descriptor(key);
    ops::require_f32_descriptor(value);
    ops::require_f32_descriptor(cache_key);
    ops::require_f32_descriptor(cache_value);
    require_matrix_descriptor(query);
    require_matrix_descriptor(key);
    require_matrix_descriptor(value);
    require_matrix_descriptor(cache_key);
    require_matrix_descriptor(cache_value);

    if (head_count == 0 || kv_head_count == 0 || key_length == 0 || value_length == 0) {
        throw std::invalid_argument("Llama cached attention dimensions must be positive");
    }
    if (head_count % kv_head_count != 0) {
        throw std::invalid_argument("Llama attention head count must be divisible by KV head count");
    }
    if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 2, rope_dimension_count);
    }
    if (rope_dimension_count > key_length) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_length, rope_dimension_count);
    }
    require_positive_frequency_base(frequency_base);
    require_positive_finite_scale(scale);

    const auto token_count = query.shape.dimension(0);
    if (token_count == 0) {
        throw std::invalid_argument("Llama cached attention input rows must be positive");
    }
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto key_rows = key.shape.dimension(0);
    const auto value_rows = value.shape.dimension(0);
    if (key_rows != token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, token_count, key_rows);
    }
    if (value_rows != token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, token_count, value_rows);
    }
    if (query.shape.dimension(1) != query_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_width, query.shape.dimension(1));
    }
    if (key.shape.dimension(1) != key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, key.shape.dimension(1));
    }
    if (value.shape.dimension(1) != value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, value.shape.dimension(1));
    }
    if (cache_key.shape.dimension(1) != key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, cache_key.shape.dimension(1));
    }
    if (cache_value.shape.dimension(1) != value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, cache_value.shape.dimension(1));
    }

    const auto cache_capacity = cache_key.shape.dimension(0);
    if (cache_value.shape.dimension(0) != cache_capacity) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, cache_capacity, cache_value.shape.dimension(0));
    }
    if (cache_position > cache_capacity || token_count > cache_capacity - cache_position) {
        throw std::invalid_argument("Llama backend KV cache append would exceed context length");
    }
}

void require_llama_cached_block_f32_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& attention_norm_weight,
    const TensorDescriptor& query_weight,
    const TensorDescriptor& key_weight,
    const TensorDescriptor& value_weight,
    const TensorDescriptor& cache_key,
    const TensorDescriptor& cache_value,
    const TensorDescriptor& attention_output_weight,
    const TensorDescriptor& feed_forward_norm_weight,
    const TensorDescriptor& gate_weight,
    const TensorDescriptor& up_weight,
    const TensorDescriptor& down_weight,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float attention_scale,
    float rms_norm_epsilon
) {
    ops::require_f32_descriptor(input);
    require_matrix_descriptor(input);
    const auto token_count = input.shape.dimension(0);
    const auto hidden = input.shape.dimension(1);
    if (token_count == 0) {
        throw std::invalid_argument("Llama cached block input rows must be positive");
    }

    require_f32_vector_descriptor(attention_norm_weight);
    if (attention_norm_weight.shape.dimension(0) != hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, attention_norm_weight.shape.dimension(0));
    }

    if (head_count == 0 || kv_head_count == 0 || key_length == 0 || value_length == 0) {
        throw std::invalid_argument("Llama cached block attention dimensions must be positive");
    }
    if (head_count % kv_head_count != 0) {
        throw std::invalid_argument("Llama attention head count must be divisible by KV head count");
    }

    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto attention_output_width = head_count * value_length;
    require_f32_f16_matmul_gguf_weight_operands(input, query_weight);
    require_f32_f16_matmul_gguf_weight_operands(input, key_weight);
    require_f32_f16_matmul_gguf_weight_operands(input, value_weight);
    if (query_weight.shape.dimension(1) != query_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, query_width, query_weight.shape.dimension(1));
    }
    if (key_weight.shape.dimension(1) != key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_width, key_weight.shape.dimension(1));
    }
    if (value_weight.shape.dimension(1) != value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, value_width, value_weight.shape.dimension(1));
    }

    const auto query_descriptor = TensorDescriptor::from(Shape::from({token_count, query_width}), DType::f32);
    const auto key_descriptor = TensorDescriptor::from(Shape::from({token_count, key_width}), DType::f32);
    const auto value_descriptor = TensorDescriptor::from(Shape::from({token_count, value_width}), DType::f32);
    require_llama_cached_attention_f32_operands(
        query_descriptor,
        key_descriptor,
        value_descriptor,
        cache_key,
        cache_value,
        cache_position,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale
    );

    const auto attention_output_descriptor =
        TensorDescriptor::from(Shape::from({token_count, attention_output_width}), DType::f32);
    require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
        input,
        attention_output_descriptor,
        attention_output_weight,
        feed_forward_norm_weight,
        gate_weight,
        up_weight,
        down_weight,
        rms_norm_epsilon
    );
}

void require_f32_f16_matmul_gguf_weight_triple_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& first_right,
    const TensorDescriptor& second_right,
    const TensorDescriptor& third_right
) {
    require_f32_f16_matmul_gguf_weight_operands(left, first_right);
    require_f32_f16_matmul_gguf_weight_operands(left, second_right);
    require_f32_f16_matmul_gguf_weight_operands(left, third_right);
}

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
}

[[nodiscard]] const CpuBackendBufferStorage& require_cpu_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const CpuBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain CPU storage"
        );
    }

    return *typed_storage;
}

[[nodiscard]] const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const MetalBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain Metal storage"
        );
    }

    return *typed_storage;
}

MatmulPipelineState::MatmulPipelineState() {
    device = metal_device_or_throw();

    NSError* library_error = nil;
    NSString* source = [NSString stringWithUTF8String:matmul_f32_f16_gguf_weight_source];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&library_error];
    if (library == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul library compilation failed: " + ns_error_message(library_error, "unknown error")
        );
    }

    id<MTLFunction> matmul_function = [library newFunctionWithName:@"matmul_f32_f16_gguf_weight_kernel"];
    if (matmul_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul kernel function was not found"
        );
    }

    NSError* pipeline_error = nil;
    matmul_pipeline = [device newComputePipelineStateWithFunction:matmul_function error:&pipeline_error];
    if (matmul_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul pipeline creation failed: " + ns_error_message(pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> matmul_add_function =
        [library newFunctionWithName:@"matmul_f32_f16_gguf_weight_add_kernel"];
    if (matmul_add_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul-add kernel function was not found"
        );
    }

    NSError* matmul_add_pipeline_error = nil;
    matmul_add_pipeline = [device newComputePipelineStateWithFunction:matmul_add_function
                                                                error:&matmul_add_pipeline_error];
    if (matmul_add_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul-add pipeline creation failed: "
                + ns_error_message(matmul_add_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> silu_multiply_function = [library newFunctionWithName:@"silu_multiply_f32_kernel"];
    if (silu_multiply_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply kernel function was not found"
        );
    }

    NSError* silu_multiply_pipeline_error = nil;
    silu_multiply_pipeline = [device newComputePipelineStateWithFunction:silu_multiply_function
                                                                   error:&silu_multiply_pipeline_error];
    if (silu_multiply_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply pipeline creation failed: "
                + ns_error_message(silu_multiply_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> add_function = [library newFunctionWithName:@"add_f32_kernel"];
    if (add_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add kernel function was not found"
        );
    }

    NSError* add_pipeline_error = nil;
    add_pipeline = [device newComputePipelineStateWithFunction:add_function error:&add_pipeline_error];
    if (add_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add pipeline creation failed: " + ns_error_message(add_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> rms_norm_function = [library newFunctionWithName:@"rms_norm_f32_kernel"];
    if (rms_norm_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm kernel function was not found"
        );
    }

    NSError* rms_norm_pipeline_error = nil;
    rms_norm_pipeline = [device newComputePipelineStateWithFunction:rms_norm_function
                                                             error:&rms_norm_pipeline_error];
    if (rms_norm_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm pipeline creation failed: " + ns_error_message(rms_norm_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_kv_append_function =
        [library newFunctionWithName:@"llama_cached_kv_append_f32_kernel"];
    if (llama_cached_kv_append_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached KV append kernel function was not found"
        );
    }

    NSError* llama_cached_kv_append_pipeline_error = nil;
    llama_cached_kv_append_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_kv_append_function
                                              error:&llama_cached_kv_append_pipeline_error];
    if (llama_cached_kv_append_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached KV append pipeline creation failed: "
                + ns_error_message(llama_cached_kv_append_pipeline_error, "unknown error")
        );
    }

    id<MTLFunction> llama_cached_attention_output_function =
        [library newFunctionWithName:@"llama_cached_attention_output_f32_kernel"];
    if (llama_cached_attention_output_function == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention output kernel function was not found"
        );
    }

    NSError* llama_cached_attention_output_pipeline_error = nil;
    llama_cached_attention_output_pipeline =
        [device newComputePipelineStateWithFunction:llama_cached_attention_output_function
                                              error:&llama_cached_attention_output_pipeline_error];
    if (llama_cached_attention_output_pipeline == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention output pipeline creation failed: "
                + ns_error_message(llama_cached_attention_output_pipeline_error, "unknown error")
        );
    }

    queue = [device newCommandQueue];
    if (queue == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul command queue creation failed"
        );
    }
}

[[nodiscard]] MatmulPipelineState& shared_matmul_pipeline_state() {
    static MatmulPipelineState state;
    return state;
}

[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight_with_metal_right_buffer(
    const TensorStorage& left,
    const MetalBackendBufferStorage& right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    const MatmulDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(columns, "columns"),
    };
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

    id<MTLBuffer> left_buffer = [state.device newBufferWithBytes:left.bytes().data()
                                                           length:static_cast<NSUInteger>(left.bytes().size())
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output.bytes().size())
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(MatmulDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (left_buffer == nil || output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.matmul_pipeline];
    [encoder setBuffer:left_buffer offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
    const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal matmul command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    std::memcpy(
        output.mutable_bytes().data(),
        [output_buffer contents],
        output.bytes().size()
    );
    return output;
}

[[nodiscard]] BackendBuffer matmul_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    const MatmulDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(columns, "columns"),
    };
    const auto output_descriptor = TensorDescriptor::from(Shape::from({rows, columns}), DType::f32);

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(MatmulDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer matmul command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.matmul_pipeline];
    [encoder setBuffer:left.buffer() offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
    const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer matmul command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] ops::F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& first_right,
    const MetalBackendBufferStorage& second_right,
    const MetalBackendBufferStorage& third_right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto first_columns = first_right.descriptor().shape.dimension(1);
    const auto second_columns = second_right.descriptor().shape.dimension(1);
    const auto third_columns = third_right.descriptor().shape.dimension(1);
    const MatmulDimensions first_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(first_columns, "first_columns"),
    };
    const MatmulDimensions second_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(second_columns, "second_columns"),
    };
    const MatmulDimensions third_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(shared, "shared"),
        .columns = require_u32_dimension(third_columns, "third_columns"),
    };
    const auto first_descriptor = TensorDescriptor::from(Shape::from({rows, first_columns}), DType::f32);
    const auto second_descriptor = TensorDescriptor::from(Shape::from({rows, second_columns}), DType::f32);
    const auto third_descriptor = TensorDescriptor::from(Shape::from({rows, third_columns}), DType::f32);

    id<MTLBuffer> first_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                     first_descriptor.byte_count
                                  )
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> second_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      second_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> third_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                     third_descriptor.byte_count
                                  )
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> first_dims_buffer = [state.device newBufferWithBytes:&first_dims
                                                                length:sizeof(MatmulDimensions)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> second_dims_buffer = [state.device newBufferWithBytes:&second_dims
                                                                 length:sizeof(MatmulDimensions)
                                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> third_dims_buffer = [state.device newBufferWithBytes:&third_dims
                                                                length:sizeof(MatmulDimensions)
                                                               options:MTLResourceStorageModeShared];
    if (first_buffer == nil || second_buffer == nil || third_buffer == nil
        || first_dims_buffer == nil || second_dims_buffer == nil || third_dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal triple matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal triple matmul command buffer creation failed"
        );
    }

    auto encode_matmul = [&](
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> output_buffer,
        id<MTLBuffer> dims_buffer,
        std::uint64_t output_columns
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal triple matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left.buffer() offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(first_right.buffer(), first_buffer, first_dims_buffer, first_columns);
    encode_matmul(second_right.buffer(), second_buffer, second_dims_buffer, second_columns);
    encode_matmul(third_right.buffer(), third_buffer, third_dims_buffer, third_columns);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal triple matmul command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return ops::F32F16GGUFWeightTripleMatmulBackendBuffers {
        .first = detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(first_descriptor, first_buffer)
        ),
        .second = detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(second_descriptor, second_buffer)
        ),
        .third = detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(third_descriptor, third_buffer)
        ),
    };
}

[[nodiscard]] BackendBuffer silu_multiply_f32_with_metal_buffers(
    const MetalBackendBufferStorage& gate,
    const MetalBackendBufferStorage& up
) {
    auto& state = shared_matmul_pipeline_state();

    const auto element_count = gate.descriptor().element_count;
    const ElementwiseDimensions dims {
        .elements = require_u32_dimension(element_count, "elements"),
    };
    const auto output_descriptor = gate.descriptor();

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(ElementwiseDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.silu_multiply_pipeline];
    [encoder setBuffer:gate.buffer() offset:0 atIndex:0];
    [encoder setBuffer:up.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto threadgroup_count = static_cast<NSUInteger>((element_count + 255U) / 256U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_count, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal SiLU multiply command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = left.descriptor().shape.dimension(0);
    const auto hidden = left.descriptor().shape.dimension(1);
    const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
    const auto output_columns = down_weight.descriptor().shape.dimension(1);
    const MatmulDimensions gate_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(feed_forward, "feed_forward"),
    };
    const MatmulDimensions down_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(feed_forward, "feed_forward"),
        .columns = require_u32_dimension(output_columns, "output_columns"),
    };
    const auto intermediate_descriptor = TensorDescriptor::from(
        Shape::from({rows, feed_forward}),
        DType::f32
    );
    const auto output_descriptor = TensorDescriptor::from(
        Shape::from({rows, output_columns}),
        DType::f32
    );
    const ElementwiseDimensions elementwise_dims {
        .elements = require_u32_dimension(intermediate_descriptor.element_count, "elements"),
    };

    id<MTLBuffer> gate_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                    intermediate_descriptor.byte_count
                                 )
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> up_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                  intermediate_descriptor.byte_count
                               )
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                     intermediate_descriptor.byte_count
                                  )
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_dims_buffer = [state.device newBufferWithBytes:&gate_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> down_dims_buffer = [state.device newBufferWithBytes:&down_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> elementwise_dims_buffer = [state.device newBufferWithBytes:&elementwise_dims
                                                                      length:sizeof(ElementwiseDimensions)
                                                                     options:MTLResourceStorageModeShared];
    if (gate_buffer == nil || up_buffer == nil || gated_buffer == nil || output_buffer == nil
        || gate_dims_buffer == nil || down_dims_buffer == nil || elementwise_dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection command buffer creation failed"
        );
    }

    auto encode_matmul = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> matmul_output_buffer,
        id<MTLBuffer> dims_buffer,
        std::uint64_t output_rows,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal gated SiLU projection matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((output_rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(left.buffer(), gate_weight.buffer(), gate_buffer, gate_dims_buffer, rows, feed_forward);
    encode_matmul(left.buffer(), up_weight.buffer(), up_buffer, gate_dims_buffer, rows, feed_forward);

    id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
    if (silu_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection activation command encoding failed"
        );
    }
    [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
    [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
    [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
    [silu_encoder setBuffer:elementwise_dims_buffer offset:0 atIndex:3];

    const MTLSize activation_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto activation_threadgroup_count = static_cast<NSUInteger>(
        (intermediate_descriptor.element_count + 255U) / 256U
    );
    const MTLSize activation_threadgroups = MTLSizeMake(activation_threadgroup_count, 1, 1);
    [silu_encoder dispatchThreadgroups:activation_threadgroups
                 threadsPerThreadgroup:activation_threads_per_threadgroup];
    [silu_encoder endEncoding];

    encode_matmul(gated_buffer, down_weight.buffer(), output_buffer, down_dims_buffer, rows, output_columns);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal gated SiLU projection command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& attention_output,
    const MetalBackendBufferStorage& attention_output_weight,
    const MetalBackendBufferStorage& feed_forward_norm_weight,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight,
    float epsilon
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = input.descriptor().shape.dimension(0);
    const auto hidden = input.descriptor().shape.dimension(1);
    const auto attention_output_width = attention_output.descriptor().shape.dimension(1);
    const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
    const MatmulDimensions attention_output_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(attention_output_width, "attention_output_width"),
        .columns = require_u32_dimension(hidden, "hidden"),
    };
    const MatmulDimensions gate_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(feed_forward, "feed_forward"),
    };
    const MatmulDimensions down_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(feed_forward, "feed_forward"),
        .columns = require_u32_dimension(hidden, "hidden"),
    };
    const auto hidden_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
    const auto feed_forward_descriptor = TensorDescriptor::from(Shape::from({rows, feed_forward}), DType::f32);
    const ElementwiseDimensions hidden_elementwise_dims {
        .elements = require_u32_dimension(hidden_descriptor.element_count, "hidden_elements"),
    };
    const ElementwiseDimensions feed_forward_elementwise_dims {
        .elements = require_u32_dimension(feed_forward_descriptor.element_count, "feed_forward_elements"),
    };
    const RMSNormDimensions rms_norm_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .epsilon = epsilon,
    };

    id<MTLBuffer> attention_projected_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_residual_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> normalized_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(feed_forward_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> up_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(feed_forward_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> gated_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(feed_forward_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> feed_forward_projected_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> attention_output_dims_buffer = [state.device newBufferWithBytes:&attention_output_dims
                                                                           length:sizeof(MatmulDimensions)
                                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_dims_buffer = [state.device newBufferWithBytes:&gate_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> down_dims_buffer = [state.device newBufferWithBytes:&down_dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden_elementwise_dims_buffer = [state.device newBufferWithBytes:&hidden_elementwise_dims
                                                                             length:sizeof(ElementwiseDimensions)
                                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> feed_forward_elementwise_dims_buffer =
        [state.device newBufferWithBytes:&feed_forward_elementwise_dims
                                  length:sizeof(ElementwiseDimensions)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> rms_norm_dims_buffer = [state.device newBufferWithBytes:&rms_norm_dims
                                                                   length:sizeof(RMSNormDimensions)
                                                                  options:MTLResourceStorageModeShared];
    if (attention_projected_buffer == nil || attention_residual_buffer == nil || normalized_buffer == nil
        || gate_buffer == nil || up_buffer == nil || gated_buffer == nil || feed_forward_projected_buffer == nil
        || output_buffer == nil || attention_output_dims_buffer == nil || gate_dims_buffer == nil
        || down_dims_buffer == nil || hidden_elementwise_dims_buffer == nil
        || feed_forward_elementwise_dims_buffer == nil || rms_norm_dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection command buffer creation failed"
        );
    }

    auto encode_matmul = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> matmul_output_buffer,
        id<MTLBuffer> dims_buffer,
        std::uint64_t output_rows,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal residual gated SiLU projection matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((output_rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    auto encode_add = [&](id<MTLBuffer> left_buffer, id<MTLBuffer> right_buffer, id<MTLBuffer> add_output_buffer) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal residual gated SiLU projection add command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.add_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:add_output_buffer offset:0 atIndex:2];
        [encoder setBuffer:hidden_elementwise_dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
        const auto threadgroup_count = static_cast<NSUInteger>((hidden_descriptor.element_count + 255U) / 256U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_count, 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_matmul(
        attention_output.buffer(),
        attention_output_weight.buffer(),
        attention_projected_buffer,
        attention_output_dims_buffer,
        rows,
        hidden
    );
    encode_add(input.buffer(), attention_projected_buffer, attention_residual_buffer);

    id<MTLComputeCommandEncoder> rms_norm_encoder = [command_buffer computeCommandEncoder];
    if (rms_norm_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection RMSNorm command encoding failed"
        );
    }
    [rms_norm_encoder setComputePipelineState:state.rms_norm_pipeline];
    [rms_norm_encoder setBuffer:attention_residual_buffer offset:0 atIndex:0];
    [rms_norm_encoder setBuffer:feed_forward_norm_weight.buffer() offset:0 atIndex:1];
    [rms_norm_encoder setBuffer:normalized_buffer offset:0 atIndex:2];
    [rms_norm_encoder setBuffer:rms_norm_dims_buffer offset:0 atIndex:3];

    const MTLSize rms_norm_threads_per_threadgroup = MTLSizeMake(64, 1, 1);
    const auto rms_norm_threadgroup_count = static_cast<NSUInteger>((rows + 63U) / 64U);
    const MTLSize rms_norm_threadgroups = MTLSizeMake(rms_norm_threadgroup_count, 1, 1);
    [rms_norm_encoder dispatchThreadgroups:rms_norm_threadgroups
                     threadsPerThreadgroup:rms_norm_threads_per_threadgroup];
    [rms_norm_encoder endEncoding];

    encode_matmul(normalized_buffer, gate_weight.buffer(), gate_buffer, gate_dims_buffer, rows, feed_forward);
    encode_matmul(normalized_buffer, up_weight.buffer(), up_buffer, gate_dims_buffer, rows, feed_forward);

    id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
    if (silu_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection activation command encoding failed"
        );
    }
    [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
    [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
    [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
    [silu_encoder setBuffer:feed_forward_elementwise_dims_buffer offset:0 atIndex:3];

    const MTLSize activation_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto activation_threadgroup_count = static_cast<NSUInteger>(
        (feed_forward_descriptor.element_count + 255U) / 256U
    );
    const MTLSize activation_threadgroups = MTLSizeMake(activation_threadgroup_count, 1, 1);
    [silu_encoder dispatchThreadgroups:activation_threadgroups
                 threadsPerThreadgroup:activation_threads_per_threadgroup];
    [silu_encoder endEncoding];

    encode_matmul(gated_buffer, down_weight.buffer(), feed_forward_projected_buffer, down_dims_buffer, rows, hidden);
    encode_add(attention_residual_buffer, feed_forward_projected_buffer, output_buffer);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal residual gated SiLU projection command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(hidden_descriptor, output_buffer)
    );
}

[[nodiscard]] double llama_rope_angle(
    std::uint64_t position,
    std::uint64_t pair_offset,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto exponent = static_cast<double>(pair_offset) / static_cast<double>(rope_dimension_count);
    return static_cast<double>(position) / std::pow(static_cast<double>(frequency_base), exponent);
}

[[nodiscard]] float read_rope_f32_columns(
    const TensorStorage& storage,
    std::uint64_t row,
    std::uint64_t first_column,
    std::uint64_t column,
    std::uint64_t position,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto columns = storage.descriptor().shape.dimension(1);
    if (column >= rope_dimension_count) {
        return read_f32_linear(storage, row * columns + first_column + column);
    }

    const auto pair_offset = column - (column % 2);
    const auto angle = llama_rope_angle(position, pair_offset, rope_dimension_count, frequency_base);
    const auto angle_cos = static_cast<float>(std::cos(angle));
    const auto angle_sin = static_cast<float>(std::sin(angle));
    const auto first = read_f32_linear(storage, row * columns + first_column + pair_offset);
    const auto second = read_f32_linear(storage, row * columns + first_column + pair_offset + 1);

    if (column == pair_offset) {
        return first * angle_cos - second * angle_sin;
    }

    return first * angle_sin + second * angle_cos;
}

void append_llama_cached_kv_to_tensors_f32(
    TensorStorage& cache_key,
    TensorStorage& cache_value,
    const TensorStorage& key,
    const TensorStorage& value,
    std::uint64_t cache_position,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    const auto token_count = key.descriptor().shape.dimension(0);
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;

    for (std::uint64_t row = 0; row < token_count; ++row) {
        const auto cache_row = cache_position + row;
        for (std::uint64_t kv_head = 0; kv_head < kv_head_count; ++kv_head) {
            const auto first_column = kv_head * key_length;
            for (std::uint64_t column = 0; column < key_length; ++column) {
                write_f32_linear(
                    cache_key,
                    cache_row * key_width + first_column + column,
                    read_rope_f32_columns(
                        key,
                        row,
                        first_column,
                        column,
                        cache_row,
                        rope_dimension_count,
                        frequency_base
                    )
                );
            }
        }

        for (std::uint64_t column = 0; column < value_width; ++column) {
            write_f32_linear(
                cache_value,
                cache_row * value_width + column,
                read_f32_linear(value, row * value_width + column)
            );
        }
    }
}

[[nodiscard]] TensorStorage materialize_llama_cached_attention_output_f32(
    const TensorStorage& query,
    const TensorStorage& cache_key,
    const TensorStorage& cache_value,
    std::uint64_t cache_position,
    std::uint64_t cached_token_count,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    const auto token_count = query.descriptor().shape.dimension(0);
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto output_width = head_count * value_length;
    const auto kv_group_size = head_count / kv_head_count;
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({token_count, output_width}), DType::f32));

    for (std::uint64_t head = 0; head < head_count; ++head) {
        const auto kv_head = head / kv_group_size;
        const auto query_first_column = head * key_length;
        const auto key_first_column = kv_head * key_length;
        const auto value_first_column = kv_head * value_length;
        const auto output_first_column = head * value_length;

        for (std::uint64_t query_row = 0; query_row < token_count; ++query_row) {
            const auto query_position = cache_position + query_row;
            for (std::uint64_t value_column = 0; value_column < value_length; ++value_column) {
                float max_score = -std::numeric_limits<float>::infinity();
                float denominator = 0.0F;
                float accumulator = 0.0F;

                for (std::uint64_t key_row = 0; key_row < cached_token_count; ++key_row) {
                    if (key_row > query_position) {
                        continue;
                    }

                    float dot = 0.0F;
                    for (std::uint64_t column = 0; column < key_length; ++column) {
                        dot += read_rope_f32_columns(
                            query,
                            query_row,
                            query_first_column,
                            column,
                            query_position,
                            rope_dimension_count,
                            frequency_base
                        ) * read_f32_linear(cache_key, key_row * key_width + key_first_column + column);
                    }

                    const auto score = dot * scale;
                    if (score > max_score) {
                        const auto rescale = std::isfinite(max_score) ? std::exp(max_score - score) : 0.0F;
                        denominator *= rescale;
                        accumulator *= rescale;
                        max_score = score;
                    }

                    const auto weight = std::exp(score - max_score);
                    denominator += weight;
                    accumulator += weight * read_f32_linear(
                        cache_value,
                        key_row * value_width + value_first_column + value_column
                    );
                }

                if (!(denominator > 0.0F)) {
                    throw std::invalid_argument("Llama cached attention row has no visible keys");
                }
                write_f32_linear(
                    output,
                    query_row * output_width + output_first_column + value_column,
                    accumulator / denominator
                );
            }
        }
    }

    (void)query_width;
    return output;
}

[[nodiscard]] BackendBuffer llama_cached_attention_f32_with_metal_buffers(
    const MetalBackendBufferStorage& query,
    const MetalBackendBufferStorage& key,
    const MetalBackendBufferStorage& value,
    const MetalBackendBufferStorage& cache_key,
    const MetalBackendBufferStorage& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    auto& state = shared_matmul_pipeline_state();

    const auto token_count = query.descriptor().shape.dimension(0);
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto output_width = head_count * value_length;
    const auto cached_token_count = cache_position + token_count;
    const LlamaCachedAttentionDimensions dims {
        .token_count = require_u32_dimension(token_count, "token_count"),
        .cache_position = require_u32_dimension(cache_position, "cache_position"),
        .cached_token_count = require_u32_dimension(cached_token_count, "cached_token_count"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
        .key_length = require_u32_dimension(key_length, "key_length"),
        .value_length = require_u32_dimension(value_length, "value_length"),
        .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
        .frequency_base = frequency_base,
        .scale = scale,
    };
    const auto output_descriptor = TensorDescriptor::from(Shape::from({token_count, output_width}), DType::f32);

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(LlamaCachedAttentionDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention command buffer creation failed"
        );
    }

    id<MTLComputeCommandEncoder> append_encoder = [command_buffer computeCommandEncoder];
    if (append_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached KV append command encoding failed"
        );
    }
    [append_encoder setComputePipelineState:state.llama_cached_kv_append_pipeline];
    [append_encoder setBuffer:key.buffer() offset:0 atIndex:0];
    [append_encoder setBuffer:value.buffer() offset:0 atIndex:1];
    [append_encoder setBuffer:cache_key.buffer() offset:0 atIndex:2];
    [append_encoder setBuffer:cache_value.buffer() offset:0 atIndex:3];
    [append_encoder setBuffer:dims_buffer offset:0 atIndex:4];

    const auto append_columns = key_width > value_width ? key_width : value_width;
    const MTLSize append_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize append_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((append_columns + 15U) / 16U),
        static_cast<NSUInteger>((token_count + 15U) / 16U),
        1
    );
    [append_encoder dispatchThreadgroups:append_threadgroups threadsPerThreadgroup:append_threads_per_threadgroup];
    [append_encoder endEncoding];

    id<MTLComputeCommandEncoder> attention_encoder = [command_buffer computeCommandEncoder];
    if (attention_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention output command encoding failed"
        );
    }
    [attention_encoder setComputePipelineState:state.llama_cached_attention_output_pipeline];
    [attention_encoder setBuffer:query.buffer() offset:0 atIndex:0];
    [attention_encoder setBuffer:cache_key.buffer() offset:0 atIndex:1];
    [attention_encoder setBuffer:cache_value.buffer() offset:0 atIndex:2];
    [attention_encoder setBuffer:output_buffer offset:0 atIndex:3];
    [attention_encoder setBuffer:dims_buffer offset:0 atIndex:4];

    const MTLSize attention_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize attention_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((output_width + 15U) / 16U),
        static_cast<NSUInteger>((token_count + 15U) / 16U),
        1
    );
    [attention_encoder dispatchThreadgroups:attention_threadgroups
                      threadsPerThreadgroup:attention_threads_per_threadgroup];
    [attention_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached attention command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(output_descriptor, output_buffer)
    );
}

[[nodiscard]] BackendBuffer llama_cached_block_f32_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& attention_norm_weight,
    const MetalBackendBufferStorage& query_weight,
    const MetalBackendBufferStorage& key_weight,
    const MetalBackendBufferStorage& value_weight,
    const MetalBackendBufferStorage& cache_key,
    const MetalBackendBufferStorage& cache_value,
    const MetalBackendBufferStorage& attention_output_weight,
    const MetalBackendBufferStorage& feed_forward_norm_weight,
    const MetalBackendBufferStorage& gate_weight,
    const MetalBackendBufferStorage& up_weight,
    const MetalBackendBufferStorage& down_weight,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float attention_scale,
    float rms_norm_epsilon
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = input.descriptor().shape.dimension(0);
    const auto hidden = input.descriptor().shape.dimension(1);
    const auto query_width = head_count * key_length;
    const auto key_width = kv_head_count * key_length;
    const auto value_width = kv_head_count * value_length;
    const auto attention_output_width = head_count * value_length;
    const auto feed_forward = gate_weight.descriptor().shape.dimension(1);
    const auto cached_token_count = cache_position + rows;
    const auto hidden_descriptor = TensorDescriptor::from(Shape::from({rows, hidden}), DType::f32);
    const auto query_descriptor = TensorDescriptor::from(Shape::from({rows, query_width}), DType::f32);
    const auto key_descriptor = TensorDescriptor::from(Shape::from({rows, key_width}), DType::f32);
    const auto value_descriptor = TensorDescriptor::from(Shape::from({rows, value_width}), DType::f32);
    const auto attention_output_descriptor =
        TensorDescriptor::from(Shape::from({rows, attention_output_width}), DType::f32);
    const auto feed_forward_descriptor = TensorDescriptor::from(Shape::from({rows, feed_forward}), DType::f32);
    const RMSNormDimensions rms_norm_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .epsilon = rms_norm_epsilon,
    };
    const MatmulDimensions query_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(query_width, "query_width"),
    };
    const MatmulDimensions key_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(key_width, "key_width"),
    };
    const MatmulDimensions value_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(value_width, "value_width"),
    };
    const LlamaCachedAttentionDimensions attention_dims {
        .token_count = require_u32_dimension(rows, "token_count"),
        .cache_position = require_u32_dimension(cache_position, "cache_position"),
        .cached_token_count = require_u32_dimension(cached_token_count, "cached_token_count"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
        .key_length = require_u32_dimension(key_length, "key_length"),
        .value_length = require_u32_dimension(value_length, "value_length"),
        .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
        .frequency_base = frequency_base,
        .scale = attention_scale,
    };
    const MatmulDimensions attention_output_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(attention_output_width, "attention_output_width"),
        .columns = require_u32_dimension(hidden, "hidden"),
    };
    const MatmulDimensions gate_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(hidden, "hidden"),
        .columns = require_u32_dimension(feed_forward, "feed_forward"),
    };
    const MatmulDimensions down_dims {
        .rows = require_u32_dimension(rows, "rows"),
        .shared = require_u32_dimension(feed_forward, "feed_forward"),
        .columns = require_u32_dimension(hidden, "hidden"),
    };
    const ElementwiseDimensions feed_forward_elementwise_dims {
        .elements = require_u32_dimension(feed_forward_descriptor.element_count, "feed_forward_elements"),
    };

    std::lock_guard<std::mutex> scratch_lock(state.llama_cached_block_scratch_mutex);
    auto& scratch = state.llama_cached_block_scratch;
    // The fused block waits before returning, so intra-block work buffers can be
    // reused across calls. The returned output buffer stays per-call owned
    // because it may become the next layer's input.
    id<MTLBuffer> attention_normalized_buffer =
        reusable_scratch_buffer(state.device, scratch.attention_normalized, hidden_descriptor.byte_count, "attention_norm");
    id<MTLBuffer> query_buffer =
        reusable_scratch_buffer(state.device, scratch.query, query_descriptor.byte_count, "query");
    id<MTLBuffer> key_buffer =
        reusable_scratch_buffer(state.device, scratch.key, key_descriptor.byte_count, "key");
    id<MTLBuffer> value_buffer =
        reusable_scratch_buffer(state.device, scratch.value, value_descriptor.byte_count, "value");
    id<MTLBuffer> attention_output_buffer =
        reusable_scratch_buffer(state.device, scratch.attention_output, attention_output_descriptor.byte_count, "attention_output");
    id<MTLBuffer> attention_residual_buffer =
        reusable_scratch_buffer(state.device, scratch.attention_residual, hidden_descriptor.byte_count, "attention_residual");
    id<MTLBuffer> feed_forward_normalized_buffer =
        reusable_scratch_buffer(state.device, scratch.feed_forward_normalized, hidden_descriptor.byte_count, "ffn_norm");
    id<MTLBuffer> gate_buffer =
        reusable_scratch_buffer(state.device, scratch.gate, feed_forward_descriptor.byte_count, "gate");
    id<MTLBuffer> up_buffer =
        reusable_scratch_buffer(state.device, scratch.up, feed_forward_descriptor.byte_count, "up");
    id<MTLBuffer> gated_buffer =
        reusable_scratch_buffer(state.device, scratch.gated, feed_forward_descriptor.byte_count, "gated");
    id<MTLBuffer> output_buffer =
        [state.device newBufferWithLength:static_cast<NSUInteger>(hidden_descriptor.byte_count)
                                  options:MTLResourceStorageModeShared];
    if (output_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block command buffer creation failed"
        );
    }

    auto encode_rms_norm = [&](id<MTLBuffer> source, id<MTLBuffer> weight, id<MTLBuffer> output) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal Llama cached block RMSNorm command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.rms_norm_pipeline];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:weight offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&rms_norm_dims length:sizeof(RMSNormDimensions) atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(64, 1, 1);
        const MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>((rows + 63U) / 64U), 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    auto encode_matmul = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> matmul_output_buffer,
        const MatmulDimensions& dims,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal Llama cached block matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:2];
        [encoder setBytes:&dims length:sizeof(MatmulDimensions) atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    auto encode_matmul_add = [&](
        id<MTLBuffer> left_buffer,
        id<MTLBuffer> right_buffer,
        id<MTLBuffer> residual_buffer,
        id<MTLBuffer> matmul_output_buffer,
        const MatmulDimensions& dims,
        std::uint64_t output_columns_for_dispatch
    ) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal Llama cached block matmul-add command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_add_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:residual_buffer offset:0 atIndex:2];
        [encoder setBuffer:matmul_output_buffer offset:0 atIndex:3];
        [encoder setBytes:&dims length:sizeof(MatmulDimensions) atIndex:4];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((output_columns_for_dispatch + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
    };

    encode_rms_norm(input.buffer(), attention_norm_weight.buffer(), attention_normalized_buffer);
    encode_matmul(attention_normalized_buffer, query_weight.buffer(), query_buffer, query_dims, query_width);
    encode_matmul(attention_normalized_buffer, key_weight.buffer(), key_buffer, key_dims, key_width);
    encode_matmul(attention_normalized_buffer, value_weight.buffer(), value_buffer, value_dims, value_width);

    id<MTLComputeCommandEncoder> append_encoder = [command_buffer computeCommandEncoder];
    if (append_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block KV append command encoding failed"
        );
    }
    [append_encoder setComputePipelineState:state.llama_cached_kv_append_pipeline];
    [append_encoder setBuffer:key_buffer offset:0 atIndex:0];
    [append_encoder setBuffer:value_buffer offset:0 atIndex:1];
    [append_encoder setBuffer:cache_key.buffer() offset:0 atIndex:2];
    [append_encoder setBuffer:cache_value.buffer() offset:0 atIndex:3];
    [append_encoder setBytes:&attention_dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];

    const auto append_columns = key_width > value_width ? key_width : value_width;
    const MTLSize append_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize append_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((append_columns + 15U) / 16U),
        static_cast<NSUInteger>((rows + 15U) / 16U),
        1
    );
    [append_encoder dispatchThreadgroups:append_threadgroups threadsPerThreadgroup:append_threads_per_threadgroup];
    [append_encoder endEncoding];

    id<MTLComputeCommandEncoder> attention_encoder = [command_buffer computeCommandEncoder];
    if (attention_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block attention command encoding failed"
        );
    }
    [attention_encoder setComputePipelineState:state.llama_cached_attention_output_pipeline];
    [attention_encoder setBuffer:query_buffer offset:0 atIndex:0];
    [attention_encoder setBuffer:cache_key.buffer() offset:0 atIndex:1];
    [attention_encoder setBuffer:cache_value.buffer() offset:0 atIndex:2];
    [attention_encoder setBuffer:attention_output_buffer offset:0 atIndex:3];
    [attention_encoder setBytes:&attention_dims length:sizeof(LlamaCachedAttentionDimensions) atIndex:4];

    const MTLSize attention_threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize attention_threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((attention_output_width + 15U) / 16U),
        static_cast<NSUInteger>((rows + 15U) / 16U),
        1
    );
    [attention_encoder dispatchThreadgroups:attention_threadgroups
                      threadsPerThreadgroup:attention_threads_per_threadgroup];
    [attention_encoder endEncoding];

    encode_matmul_add(
        attention_output_buffer,
        attention_output_weight.buffer(),
        input.buffer(),
        attention_residual_buffer,
        attention_output_dims,
        hidden
    );
    encode_rms_norm(
        attention_residual_buffer,
        feed_forward_norm_weight.buffer(),
        feed_forward_normalized_buffer
    );
    encode_matmul(feed_forward_normalized_buffer, gate_weight.buffer(), gate_buffer, gate_dims, feed_forward);
    encode_matmul(feed_forward_normalized_buffer, up_weight.buffer(), up_buffer, gate_dims, feed_forward);

    id<MTLComputeCommandEncoder> silu_encoder = [command_buffer computeCommandEncoder];
    if (silu_encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block activation command encoding failed"
        );
    }
    [silu_encoder setComputePipelineState:state.silu_multiply_pipeline];
    [silu_encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [silu_encoder setBuffer:up_buffer offset:0 atIndex:1];
    [silu_encoder setBuffer:gated_buffer offset:0 atIndex:2];
    [silu_encoder setBytes:&feed_forward_elementwise_dims length:sizeof(ElementwiseDimensions) atIndex:3];

    const MTLSize activation_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto activation_threadgroup_count = static_cast<NSUInteger>(
        (feed_forward_descriptor.element_count + 255U) / 256U
    );
    const MTLSize activation_threadgroups = MTLSizeMake(activation_threadgroup_count, 1, 1);
    [silu_encoder dispatchThreadgroups:activation_threadgroups
                 threadsPerThreadgroup:activation_threads_per_threadgroup];
    [silu_encoder endEncoding];

    encode_matmul_add(gated_buffer, down_weight.buffer(), attention_residual_buffer, output_buffer, down_dims, hidden);

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal Llama cached block command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(hidden_descriptor, output_buffer)
    );
}

}  // namespace

bool is_available() noexcept {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

std::string unavailable_reason() {
    @autoreleasepool {
        if (MTLCreateSystemDefaultDevice() != nil) {
            return {};
        }

        return "Metal backend unavailable: no system default Metal device";
    }
}

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const TensorStorage& right) {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();

        const auto rows = left.descriptor().shape.dimension(0);
        const auto shared = left.descriptor().shape.dimension(1);
        const auto columns = right.descriptor().shape.dimension(1);
        const MatmulDimensions dims {
            .rows = require_u32_dimension(rows, "rows"),
            .shared = require_u32_dimension(shared, "shared"),
            .columns = require_u32_dimension(columns, "columns"),
        };
        auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

        id<MTLBuffer> left_buffer = [state.device newBufferWithBytes:left.bytes().data()
                                                               length:static_cast<NSUInteger>(left.bytes().size())
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> right_buffer = [state.device newBufferWithBytes:right.bytes().data()
                                                                length:static_cast<NSUInteger>(right.bytes().size())
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output.bytes().size())
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                               length:sizeof(MatmulDimensions)
                                                              options:MTLResourceStorageModeShared];
        if (left_buffer == nil || right_buffer == nil || output_buffer == nil || dims_buffer == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal matmul buffer allocation failed"
            );
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal matmul command encoding failed"
            );
        }

        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:left_buffer offset:0 atIndex:0];
        [encoder setBuffer:right_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBuffer:dims_buffer offset:0 atIndex:3];

        const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
        const auto threadgroup_columns = static_cast<NSUInteger>((columns + 15U) / 16U);
        const auto threadgroup_rows = static_cast<NSUInteger>((rows + 15U) / 16U);
        const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_rows, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSError* execution_error = [command_buffer error];
            throw metal_error(
                runtime::BackendErrorCode::metal_execution_failed,
                "Metal matmul command failed: " + ns_error_message(execution_error, "unknown error")
            );
        }

        std::memcpy(
            output.mutable_bytes().data(),
            [output_buffer contents],
            output.bytes().size()
        );
        return output;
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal

namespace local_gguf_engine::tensor::detail {

BackendBufferStorage::~BackendBufferStorage() = default;

BackendBuffer make_backend_buffer(std::shared_ptr<BackendBufferStorage> storage) {
    return BackendBuffer(std::move(storage));
}

}  // namespace local_gguf_engine::tensor::detail

namespace local_gguf_engine::tensor {

BackendBuffer::BackendBuffer(std::shared_ptr<detail::BackendBufferStorage> storage)
    : storage_(std::move(storage)) {}

runtime::BackendKind BackendBuffer::backend() const noexcept {
    return storage_->backend();
}

const TensorDescriptor& BackendBuffer::descriptor() const noexcept {
    return storage_->descriptor();
}

BackendBuffer copy_tensor_to_backend(const TensorStorage& storage, runtime::BackendKind backend) {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return BackendBuffer(std::make_shared<backends::metal::CpuBackendBufferStorage>(storage));
        case runtime::BackendKind::metal: {
            @autoreleasepool {
                auto& state = backends::metal::shared_matmul_pipeline_state();
                id<MTLBuffer> buffer = [state.device newBufferWithBytes:storage.bytes().data()
                                                                  length:static_cast<NSUInteger>(storage.bytes().size())
                                                                 options:MTLResourceStorageModeShared];
                if (buffer == nil) {
                    throw runtime::BackendError(
                        runtime::BackendErrorCode::metal_execution_failed,
                        runtime::BackendKind::metal,
                        "Metal backend buffer allocation failed"
                    );
                }

                return BackendBuffer(
                    std::make_shared<backends::metal::MetalBackendBufferStorage>(storage, buffer)
                );
            }
        }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for tensor buffer copy"
    );
}

TensorStorage copy_tensor_from_backend(const BackendBuffer& buffer) {
    const auto& storage = backend_buffer_storage(buffer);
    switch (storage.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& cpu_storage = backends::metal::require_cpu_backend_buffer(storage).storage();
            return TensorStorage::from_bytes(cpu_storage.descriptor(), cpu_storage.bytes());
        }
        case runtime::BackendKind::metal: {
            @autoreleasepool {
                const auto& metal_storage = backends::metal::require_metal_backend_buffer(storage);
                auto output = TensorStorage::allocate(metal_storage.descriptor());
                std::memcpy(
                    output.mutable_bytes().data(),
                    [metal_storage.buffer() contents],
                    output.bytes().size()
                );
                return output;
            }
        }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        storage.backend(),
        "unsupported backend for tensor buffer readback"
    );
}

const detail::BackendBufferStorage& backend_buffer_storage(const BackendBuffer& buffer) noexcept {
    return *buffer.storage_;
}

}  // namespace local_gguf_engine::tensor

namespace local_gguf_engine::tensor::ops {

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const BackendBuffer& right) {
    backends::metal::require_f32_f16_matmul_gguf_weight_operands(left, right.descriptor());

    switch (right.backend()) {
        case runtime::BackendKind::cpu:
            return matmul_f32_f16_gguf_weight(
                left,
                backends::metal::require_cpu_backend_buffer(backend_buffer_storage(right)).storage(),
                runtime::BackendKind::cpu
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_f16_gguf_weight_with_metal_right_buffer(
                    left,
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        right.backend(),
        "unsupported backend buffer for F32xF16 GGUF weight matmul"
    );
}

BackendBuffer matmul_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
) {
    backends::metal::require_f32_f16_matmul_gguf_weight_operands(left.descriptor(), right.descriptor());
    backends::metal::require_same_backend(
        left.backend(),
        right.backend(),
        "F32xF16 GGUF weight matmul"
    );

    switch (right.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& left_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(left)
            ).storage();
            const auto& right_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(right)
            ).storage();
            const auto output = matmul_f32_f16_gguf_weight(
                left_storage,
                right_storage,
                runtime::BackendKind::cpu
            );
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_f16_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        right.backend(),
        "unsupported backend buffers for F32xF16 GGUF weight matmul"
    );
}

F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& first_right,
    const BackendBuffer& second_right,
    const BackendBuffer& third_right
) {
    backends::metal::require_f32_f16_matmul_gguf_weight_triple_operands(
        left.descriptor(),
        first_right.descriptor(),
        second_right.descriptor(),
        third_right.descriptor()
    );
    backends::metal::require_same_backend(left.backend(), first_right.backend(), "F32xF16 GGUF triple matmul");
    backends::metal::require_same_backend(left.backend(), second_right.backend(), "F32xF16 GGUF triple matmul");
    backends::metal::require_same_backend(left.backend(), third_right.backend(), "F32xF16 GGUF triple matmul");

    switch (left.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& left_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(left)
            ).storage();
            const auto& first_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(first_right)
            ).storage();
            const auto& second_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(second_right)
            ).storage();
            const auto& third_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(third_right)
            ).storage();
            const auto first_output = matmul_f32_f16_gguf_weight(
                left_storage,
                first_storage,
                runtime::BackendKind::cpu
            );
            const auto second_output = matmul_f32_f16_gguf_weight(
                left_storage,
                second_storage,
                runtime::BackendKind::cpu
            );
            const auto third_output = matmul_f32_f16_gguf_weight(
                left_storage,
                third_storage,
                runtime::BackendKind::cpu
            );
            return F32F16GGUFWeightTripleMatmulBackendBuffers {
                .first = copy_tensor_to_backend(first_output, runtime::BackendKind::cpu),
                .second = copy_tensor_to_backend(second_output, runtime::BackendKind::cpu),
                .third = copy_tensor_to_backend(third_output, runtime::BackendKind::cpu),
            };
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(first_right)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(second_right)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(third_right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for F32xF16 GGUF triple matmul"
    );
}

BackendBuffer silu_multiply_f32_to_backend(
    const BackendBuffer& gate,
    const BackendBuffer& up
) {
    backends::metal::require_silu_multiply_f32_operands(gate.descriptor(), up.descriptor());
    backends::metal::require_same_backend(gate.backend(), up.backend(), "SiLU multiply");

    switch (gate.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& gate_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(gate)
            ).storage();
            const auto& up_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(up)
            ).storage();
            const auto output = multiply_f32(silu_f32(gate_storage), up_storage);
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::silu_multiply_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        gate.backend(),
        "unsupported backend buffers for SiLU multiply"
    );
}

BackendBuffer gated_silu_projection_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight
) {
    backends::metal::require_gated_silu_projection_f32_f16_gguf_weight_operands(
        left.descriptor(),
        gate_weight.descriptor(),
        up_weight.descriptor(),
        down_weight.descriptor()
    );
    backends::metal::require_same_backend(
        left.backend(),
        gate_weight.backend(),
        "gated SiLU projection"
    );
    backends::metal::require_same_backend(
        left.backend(),
        up_weight.backend(),
        "gated SiLU projection"
    );
    backends::metal::require_same_backend(
        left.backend(),
        down_weight.backend(),
        "gated SiLU projection"
    );

    switch (left.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& left_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(left)
            ).storage();
            const auto& gate_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(gate_weight)
            ).storage();
            const auto& up_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(up_weight)
            ).storage();
            const auto& down_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(down_weight)
            ).storage();
            const auto gate = matmul_f32_f16_gguf_weight(
                left_storage,
                gate_storage,
                runtime::BackendKind::cpu
            );
            const auto up = matmul_f32_f16_gguf_weight(
                left_storage,
                up_storage,
                runtime::BackendKind::cpu
            );
            const auto gated = multiply_f32(silu_f32(gate), up);
            const auto output = matmul_f32_f16_gguf_weight(
                gated,
                down_storage,
                runtime::BackendKind::cpu
            );
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(down_weight))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for gated SiLU projection"
    );
}

BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_output,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    float epsilon
) {
    backends::metal::require_residual_gated_silu_projection_f32_f16_gguf_weight_operands(
        input.descriptor(),
        attention_output.descriptor(),
        attention_output_weight.descriptor(),
        feed_forward_norm_weight.descriptor(),
        gate_weight.descriptor(),
        up_weight.descriptor(),
        down_weight.descriptor(),
        epsilon
    );
    backends::metal::require_same_backend(input.backend(), attention_output.backend(), "residual gated SiLU projection");
    backends::metal::require_same_backend(
        input.backend(),
        attention_output_weight.backend(),
        "residual gated SiLU projection"
    );
    backends::metal::require_same_backend(
        input.backend(),
        feed_forward_norm_weight.backend(),
        "residual gated SiLU projection"
    );
    backends::metal::require_same_backend(input.backend(), gate_weight.backend(), "residual gated SiLU projection");
    backends::metal::require_same_backend(input.backend(), up_weight.backend(), "residual gated SiLU projection");
    backends::metal::require_same_backend(input.backend(), down_weight.backend(), "residual gated SiLU projection");

    switch (input.backend()) {
        case runtime::BackendKind::cpu: {
            const auto& input_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(input)
            ).storage();
            const auto& attention_output_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(attention_output)
            ).storage();
            const auto& attention_output_weight_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(attention_output_weight)
            ).storage();
            const auto& feed_forward_norm_weight_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(feed_forward_norm_weight)
            ).storage();
            const auto& gate_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(gate_weight)
            ).storage();
            const auto& up_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(up_weight)
            ).storage();
            const auto& down_storage = backends::metal::require_cpu_backend_buffer(
                backend_buffer_storage(down_weight)
            ).storage();

            const auto attention_projected = matmul_f32_f16_gguf_weight(
                attention_output_storage,
                attention_output_weight_storage,
                runtime::BackendKind::cpu
            );
            const auto attention_residual = add_f32(input_storage, attention_projected);
            const auto normalized = rms_norm_f32(
                attention_residual,
                feed_forward_norm_weight_storage,
                epsilon
            );
            const auto gate = matmul_f32_f16_gguf_weight(
                normalized,
                gate_storage,
                runtime::BackendKind::cpu
            );
            const auto up = matmul_f32_f16_gguf_weight(
                normalized,
                up_storage,
                runtime::BackendKind::cpu
            );
            const auto gated = multiply_f32(silu_f32(gate), up);
            const auto feed_forward_projected = matmul_f32_f16_gguf_weight(
                gated,
                down_storage,
                runtime::BackendKind::cpu
            );
            const auto output = add_f32(attention_residual, feed_forward_projected);
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_output)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_output_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(down_weight)),
                    epsilon
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffers for residual gated SiLU projection"
    );
}

BackendBuffer llama_cached_attention_f32_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& key,
    const BackendBuffer& value,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float scale
) {
    backends::metal::require_llama_cached_attention_f32_operands(
        query.descriptor(),
        key.descriptor(),
        value.descriptor(),
        cache_key.descriptor(),
        cache_value.descriptor(),
        cache_position,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        scale
    );
    backends::metal::require_same_backend(query.backend(), key.backend(), "Llama cached attention");
    backends::metal::require_same_backend(query.backend(), value.backend(), "Llama cached attention");
    backends::metal::require_same_backend(query.backend(), cache_key.backend(), "Llama cached attention");
    backends::metal::require_same_backend(query.backend(), cache_value.backend(), "Llama cached attention");

    switch (query.backend()) {
        case runtime::BackendKind::cpu: {
            const auto query_storage = copy_tensor_from_backend(query);
            const auto key_storage = copy_tensor_from_backend(key);
            const auto value_storage = copy_tensor_from_backend(value);
            auto cache_key_storage = copy_tensor_from_backend(cache_key);
            auto cache_value_storage = copy_tensor_from_backend(cache_value);

            backends::metal::append_llama_cached_kv_to_tensors_f32(
                cache_key_storage,
                cache_value_storage,
                key_storage,
                value_storage,
                cache_position,
                kv_head_count,
                key_length,
                value_length,
                rope_dimension_count,
                frequency_base
            );
            const auto output = backends::metal::materialize_llama_cached_attention_output_f32(
                query_storage,
                cache_key_storage,
                cache_value_storage,
                cache_position,
                cache_position + query_storage.descriptor().shape.dimension(0),
                head_count,
                kv_head_count,
                key_length,
                value_length,
                rope_dimension_count,
                frequency_base,
                scale
            );

            cache_key = copy_tensor_to_backend(cache_key_storage, runtime::BackendKind::cpu);
            cache_value = copy_tensor_to_backend(cache_value_storage, runtime::BackendKind::cpu);
            return copy_tensor_to_backend(output, runtime::BackendKind::cpu);
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::llama_cached_attention_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(query)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(key)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(value)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_key)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_value)),
                    cache_position,
                    head_count,
                    kv_head_count,
                    key_length,
                    value_length,
                    rope_dimension_count,
                    frequency_base,
                    scale
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        query.backend(),
        "unsupported backend buffers for Llama cached attention"
    );
}

BackendBuffer llama_cached_block_f32_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& attention_norm_weight,
    const BackendBuffer& query_weight,
    const BackendBuffer& key_weight,
    const BackendBuffer& value_weight,
    BackendBuffer& cache_key,
    BackendBuffer& cache_value,
    const BackendBuffer& attention_output_weight,
    const BackendBuffer& feed_forward_norm_weight,
    const BackendBuffer& gate_weight,
    const BackendBuffer& up_weight,
    const BackendBuffer& down_weight,
    std::uint64_t cache_position,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t value_length,
    std::uint64_t rope_dimension_count,
    float frequency_base,
    float attention_scale,
    float rms_norm_epsilon
) {
    backends::metal::require_llama_cached_block_f32_operands(
        input.descriptor(),
        attention_norm_weight.descriptor(),
        query_weight.descriptor(),
        key_weight.descriptor(),
        value_weight.descriptor(),
        cache_key.descriptor(),
        cache_value.descriptor(),
        attention_output_weight.descriptor(),
        feed_forward_norm_weight.descriptor(),
        gate_weight.descriptor(),
        up_weight.descriptor(),
        down_weight.descriptor(),
        cache_position,
        head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        frequency_base,
        attention_scale,
        rms_norm_epsilon
    );
    backends::metal::require_same_backend(input.backend(), attention_norm_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), query_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), key_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), value_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), cache_key.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), cache_value.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), attention_output_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), feed_forward_norm_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), gate_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), up_weight.backend(), "Llama cached block");
    backends::metal::require_same_backend(input.backend(), down_weight.backend(), "Llama cached block");

    switch (input.backend()) {
        case runtime::BackendKind::cpu: {
            const auto input_storage = copy_tensor_from_backend(input);
            const auto attention_norm_storage = copy_tensor_from_backend(attention_norm_weight);
            const auto normalized = rms_norm_f32(input_storage, attention_norm_storage, rms_norm_epsilon);
            const auto normalized_buffer = copy_tensor_to_backend(normalized, runtime::BackendKind::cpu);
            const auto projections = matmul_f32_f16_gguf_weight_triple_to_backend(
                normalized_buffer,
                query_weight,
                key_weight,
                value_weight
            );
            const auto attention_output = llama_cached_attention_f32_to_backend(
                projections.first,
                projections.second,
                projections.third,
                cache_key,
                cache_value,
                cache_position,
                head_count,
                kv_head_count,
                key_length,
                value_length,
                rope_dimension_count,
                frequency_base,
                attention_scale
            );
            return residual_gated_silu_projection_f32_f16_gguf_weight_to_backend(
                input,
                attention_output,
                attention_output_weight,
                feed_forward_norm_weight,
                gate_weight,
                up_weight,
                down_weight,
                rms_norm_epsilon
            );
        }
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::llama_cached_block_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_norm_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(query_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(key_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(value_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_key)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(cache_value)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(attention_output_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up_weight)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(down_weight)),
                    cache_position,
                    head_count,
                    kv_head_count,
                    key_length,
                    value_length,
                    rope_dimension_count,
                    frequency_base,
                    attention_scale,
                    rms_norm_epsilon
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffers for Llama cached block"
    );
}

}  // namespace local_gguf_engine::tensor::ops
