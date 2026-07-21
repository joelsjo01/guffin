#include "metal_tensor_ops.hpp"

#include "../cpu_backend.hpp"
#include "../metal_backend.hpp"
#include "metal_context.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "../../ops/validation.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

struct MatmulDimensions {
    std::uint32_t rows = 0;
    std::uint32_t shared = 0;
    std::uint32_t columns = 0;
};

struct EmbeddingDimensions {
    std::uint32_t tokens = 0;
    std::uint32_t hidden = 0;
    std::uint32_t vocab = 0;
    std::uint32_t token_offset = 0;
    float scale = 1.0F;
};

struct ElementwiseDimensions {
    std::uint32_t elements = 0;
};

struct SoftcapDimensions {
    std::uint32_t elements = 0;
    float softcap = 0.0F;
};

struct RMSNormDimensions {
    std::uint32_t rows = 0;
    std::uint32_t hidden = 0;
    std::uint32_t input_row_start = 0;
    float epsilon = 0.0F;
};

struct RoPEByHeadsDimensions {
    std::uint32_t rows = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_dimension = 0;
    std::uint32_t rope_dimension_count = 0;
    std::uint32_t position_start = 0;
    float frequency_base = 0.0F;
};

struct GroupedAttentionScoreDimensions {
    std::uint32_t query_rows = 0;
    std::uint32_t key_rows = 0;
    std::uint32_t head_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t key_length = 0;
    std::uint32_t attention_token_count = 0;
    std::uint32_t query_position_start = 0;
    float scale = 0.0F;
    float logit_softcap = 0.0F;
};

struct GroupedAttentionValueDimensions {
    std::uint32_t query_rows = 0;
    std::uint32_t value_rows = 0;
    std::uint32_t head_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t value_length = 0;
    std::uint32_t attention_token_count = 0;
    std::uint32_t query_position_start = 0;
};

struct SoftmaxDimensions {
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
};

struct TopLogitDimensions {
    std::uint32_t row_offset = 0;
    std::uint32_t columns = 0;
};

struct TopLogitMetalResult {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

void require_matmul_inner_dimensions(const TensorDescriptor& left, const TensorDescriptor& right) {
    const auto left_columns = left.shape.dimension(1);
    const auto right_rows = right.shape.dimension(0);
    if (left_columns != right_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_columns, right_rows);
    }
}

void require_quantized_gguf_weight_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.dtype == DType::q4_k || descriptor.dtype == DType::q6_k) {
        return;
    }

    throw TensorError(
        TensorErrorCode::unsupported_operator_dtype,
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::q4_k)),
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
    );
}

void require_quantized_gguf_weight_inner_block_alignment(const TensorDescriptor& descriptor) {
    constexpr std::uint64_t k_quant_block_elements = 256;
    const auto inner_width = descriptor.shape.dimension(0);
    if (inner_width % k_quant_block_elements != 0) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            k_quant_block_elements,
            inner_width
        );
    }
}

void require_valid_rope_dimension_count(std::uint64_t rope_dimension_count) {
    if (rope_dimension_count == 0 || rope_dimension_count % 2 != 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 2, rope_dimension_count);
    }
}

void require_positive_head_layout(std::uint64_t head_count, std::uint64_t head_dimension) {
    if (head_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, head_count);
    }
    if (head_dimension == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, head_dimension);
    }
}

void require_head_width(
    const TensorDescriptor& descriptor,
    std::uint64_t head_count,
    std::uint64_t head_dimension
) {
    if (head_count > std::numeric_limits<std::uint64_t>::max() / head_dimension) {
        throw TensorError(TensorErrorCode::element_count_overflow, head_count, head_dimension);
    }

    const auto expected_width = head_count * head_dimension;
    const auto actual_width = descriptor.shape.dimension(1);
    if (actual_width != expected_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_width, actual_width);
    }
}

void require_rope_dimension_within_head(std::uint64_t head_dimension, std::uint64_t rope_dimension_count) {
    if (rope_dimension_count > head_dimension) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, head_dimension, rope_dimension_count);
    }
}

void require_row_position_window(std::uint64_t row_count, std::uint64_t position_start) {
    if (row_count == 0) {
        return;
    }

    if (position_start > std::numeric_limits<std::uint64_t>::max() - (row_count - 1)) {
        throw std::invalid_argument("RoPE row positions overflow");
    }
}

void require_causal_position_window(
    std::uint64_t query_rows,
    std::uint64_t key_rows,
    std::uint64_t query_position_start
) {
    if (query_position_start > std::numeric_limits<std::uint64_t>::max() - query_rows) {
        throw std::invalid_argument("attention causal query positions overflow");
    }

    const auto query_position_end = query_position_start + query_rows;
    if (query_position_end > key_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, key_rows, query_position_end);
    }
}

void require_grouped_attention_layout(
    const TensorDescriptor& query,
    const TensorDescriptor& key,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count
) {
    if (head_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, head_count);
    }
    if (kv_head_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, kv_head_count);
    }
    if (key_length == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, key_length);
    }
    if (attention_token_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, attention_token_count);
    }
    if (head_count % kv_head_count != 0) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, kv_head_count, head_count);
    }
    if (head_count > std::numeric_limits<std::uint64_t>::max() / key_length) {
        throw TensorError(TensorErrorCode::element_count_overflow, head_count, key_length);
    }
    if (key_length != 0 && kv_head_count > std::numeric_limits<std::uint64_t>::max() / key_length) {
        throw TensorError(TensorErrorCode::element_count_overflow, kv_head_count, key_length);
    }

    const auto expected_query_width = head_count * key_length;
    const auto expected_key_width = kv_head_count * key_length;
    const auto actual_query_width = query.shape.dimension(1);
    const auto actual_key_width = key.shape.dimension(1);
    if (actual_query_width != expected_query_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_query_width, actual_query_width);
    }
    if (actual_key_width != expected_key_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_key_width, actual_key_width);
    }
}

void require_grouped_attention_value_layout(
    const TensorDescriptor& probabilities,
    const TensorDescriptor& value,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
) {
    if (probabilities.shape.rank() != 3) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            3,
            static_cast<std::uint64_t>(probabilities.shape.rank())
        );
    }
    require_matrix_descriptor(value);
    if (head_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, head_count);
    }
    if (kv_head_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, kv_head_count);
    }
    if (value_length == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, value_length);
    }
    if (attention_token_count == 0) {
        throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, attention_token_count);
    }
    if (head_count % kv_head_count != 0) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, kv_head_count, head_count);
    }
    if (head_count > std::numeric_limits<std::uint64_t>::max() / value_length) {
        throw TensorError(TensorErrorCode::element_count_overflow, head_count, value_length);
    }
    if (value_length != 0 && kv_head_count > std::numeric_limits<std::uint64_t>::max() / value_length) {
        throw TensorError(TensorErrorCode::element_count_overflow, kv_head_count, value_length);
    }

    const auto probability_heads = probabilities.shape.dimension(1);
    const auto probability_columns = probabilities.shape.dimension(2);
    if (probability_heads != head_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, head_count, probability_heads);
    }
    if (probability_columns != attention_token_count) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, attention_token_count, probability_columns);
    }

    const auto expected_value_width = kv_head_count * value_length;
    const auto actual_value_width = value.shape.dimension(1);
    if (actual_value_width != expected_value_width) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, expected_value_width, actual_value_width);
    }
}

}  // namespace

std::uint32_t require_u32_dimension(std::uint64_t value, const char* name) {
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

void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorStorage& left,
    const TensorDescriptor& right
) {
    require_f32_quantized_matmul_gguf_weight_operands(left.descriptor(), right);
}

void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
) {
    ops::require_f32_descriptor(left);
    require_quantized_gguf_weight_descriptor(right);
    require_matrix_descriptor(left);
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left, right);
    require_quantized_gguf_weight_inner_block_alignment(right);
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

void require_rms_norm_f32_operands(
    const TensorDescriptor& input,
    const TensorDescriptor& weight,
    float epsilon
) {
    ops::require_f32_descriptor(input);
    require_matrix_descriptor(input);
    require_f32_vector_descriptor(weight);
    const auto hidden = input.shape.dimension(1);
    const auto weight_hidden = weight.shape.dimension(0);
    if (hidden != weight_hidden) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, hidden, weight_hidden);
    }
    require_positive_epsilon(epsilon);
}

void require_rope_f32_by_heads_operands(
    const TensorDescriptor& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    ops::require_f32_descriptor(input);
    require_matrix_descriptor(input);
    require_positive_head_layout(head_count, head_dimension);
    require_head_width(input, head_count, head_dimension);
    require_valid_rope_dimension_count(rope_dimension_count);
    require_rope_dimension_within_head(head_dimension, rope_dimension_count);
    if (!(frequency_base > 0.0F)) {
        throw std::invalid_argument("RoPE frequency base must be positive");
    }
    require_row_position_window(input.shape.dimension(0), position_start);
}

void require_positive_finite_scale(float scale) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("attention scale must be positive and finite");
    }
}

void require_positive_finite_logit_softcap(float logit_softcap) {
    if (!(logit_softcap > 0.0F) || !std::isfinite(logit_softcap)) {
        throw std::invalid_argument("attention logit softcap must be positive and finite");
    }
}

void require_causal_grouped_attention_scores_f32_operands(
    const TensorDescriptor& query,
    const TensorDescriptor& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
) {
    ops::require_f32_descriptor(query);
    ops::require_f32_descriptor(key);
    require_matrix_descriptor(query);
    require_matrix_descriptor(key);
    require_grouped_attention_layout(query, key, head_count, kv_head_count, key_length, attention_token_count);
    require_causal_position_window(query.shape.dimension(0), key.shape.dimension(0), query_position_start);
    require_positive_finite_scale(scale);
    require_positive_finite_logit_softcap(logit_softcap);
}

void require_causal_grouped_attention_values_f32_operands(
    const TensorDescriptor& probabilities,
    const TensorDescriptor& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
) {
    ops::require_f32_descriptor(probabilities);
    ops::require_f32_descriptor(value);
    require_grouped_attention_value_layout(
        probabilities,
        value,
        head_count,
        kv_head_count,
        value_length,
        attention_token_count
    );
    require_causal_position_window(probabilities.shape.dimension(0), value.shape.dimension(0), query_position_start);
}

void require_softmax_f32_operand(const TensorDescriptor& input) {
    ops::require_f32_descriptor(input);
    const auto rank = input.shape.rank();
    if (rank < 1 || rank > 3) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            3,
            static_cast<std::uint64_t>(rank)
        );
    }
}

void require_softcap_f32_operand(const TensorDescriptor& input, float softcap) {
    ops::require_f32_descriptor(input);
    if (!(softcap > 0.0F) || !std::isfinite(softcap)) {
        throw std::invalid_argument("F32 softcap must be positive and finite");
    }
}

void require_top_logit_f32_operand(const TensorDescriptor& input) {
    ops::require_f32_descriptor(input);
    const auto rank = input.shape.rank();
    if (rank < 1 || rank > 3) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            3,
            static_cast<std::uint64_t>(rank)
        );
    }
    if (input.shape.dimension(rank - 1U) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("F32 top-logit vocabulary is too large for uint32 token ids");
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

TensorStorage matmul_f32_f16_gguf_weight_with_metal_right_buffer(
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

BackendBuffer matmul_f32_f16_gguf_weight_with_metal_buffers(
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

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer rope_neox_f32_by_heads_with_metal_buffer(
    const MetalBackendBufferStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    auto& state = shared_matmul_pipeline_state();
    const auto rows = input.descriptor().shape.dimension(0);
    const auto columns = input.descriptor().shape.dimension(1);
    const RoPEByHeadsDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .head_dimension = require_u32_dimension(head_dimension, "head_dimension"),
        .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
        .position_start = require_u32_dimension(position_start, "position_start"),
        .frequency_base = frequency_base,
    };
    const auto output_descriptor = input.descriptor();
    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(output_descriptor.byte_count)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(RoPEByHeadsDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(runtime::BackendErrorCode::metal_execution_failed, "Metal NeoX RoPE buffer allocation failed");
    }
    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(runtime::BackendErrorCode::metal_execution_failed, "Metal NeoX RoPE encoding failed");
    }
    [encoder setComputePipelineState:state.rope_neox_by_heads_pipeline];
    [encoder setBuffer:input.buffer() offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    [encoder setBuffer:dims_buffer offset:0 atIndex:2];
    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const MTLSize threadgroups = MTLSizeMake(
        static_cast<NSUInteger>((columns + 15U) / 16U),
        static_cast<NSUInteger>((rows + 15U) / 16U),
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal NeoX RoPE command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }
    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer embed_f16_gguf_weight_token_ids_with_metal_buffer(
    std::span<const std::uint32_t> token_ids,
    const MetalBackendBufferStorage& embedding_weight,
    float scale
) {
    auto& state = shared_matmul_pipeline_state();

    const auto hidden = embedding_weight.descriptor().shape.dimension(0);
    const auto vocab = embedding_weight.descriptor().shape.dimension(1);
    const EmbeddingDimensions dims {
        .tokens = require_u32_dimension(token_ids.size(), "tokens"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .vocab = require_u32_dimension(vocab, "vocab"),
        .token_offset = 0,
        .scale = scale,
    };
    const auto output_descriptor = TensorDescriptor::from(
        Shape::from({static_cast<std::uint64_t>(token_ids.size()), hidden}),
        DType::f32
    );

    id<MTLBuffer> token_ids_buffer = [state.device newBufferWithBytes:token_ids.data()
                                                               length:static_cast<NSUInteger>(token_ids.size_bytes())
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(EmbeddingDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (token_ids_buffer == nil || output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal F16 GGUF token embedding buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal F16 GGUF token embedding command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.embedding_pipeline];
    [encoder setBuffer:token_ids_buffer offset:0 atIndex:0];
    [encoder setBuffer:embedding_weight.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(16, 16, 1);
    const auto threadgroup_hidden = static_cast<NSUInteger>((hidden + 15U) / 16U);
    const auto threadgroup_tokens = static_cast<NSUInteger>((token_ids.size() + 15U) / 16U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_hidden, threadgroup_tokens, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal F16 GGUF token embedding command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer rms_norm_f32_with_metal_buffers(
    const MetalBackendBufferStorage& input,
    const MetalBackendBufferStorage& weight,
    float epsilon
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = input.descriptor().shape.dimension(0);
    const auto hidden = input.descriptor().shape.dimension(1);
    const RMSNormDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .hidden = require_u32_dimension(hidden, "hidden"),
        .input_row_start = 0,
        .epsilon = epsilon,
    };
    const auto output_descriptor = input.descriptor();

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(RMSNormDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.rms_norm_pipeline];
    [encoder setBuffer:input.buffer() offset:0 atIndex:0];
    [encoder setBuffer:weight.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto threadgroup_count = static_cast<NSUInteger>(rows);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_count, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RMSNorm command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer rope_f32_by_heads_with_metal_buffer(
    const MetalBackendBufferStorage& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    auto& state = shared_matmul_pipeline_state();

    const auto rows = input.descriptor().shape.dimension(0);
    const auto columns = input.descriptor().shape.dimension(1);
    const RoPEByHeadsDimensions dims {
        .rows = require_u32_dimension(rows, "rows"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .head_dimension = require_u32_dimension(head_dimension, "head_dimension"),
        .rope_dimension_count = require_u32_dimension(rope_dimension_count, "rope_dimension_count"),
        .position_start = require_u32_dimension(position_start, "position_start"),
        .frequency_base = frequency_base,
    };
    const auto output_descriptor = input.descriptor();

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(RoPEByHeadsDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RoPE-by-heads buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal RoPE-by-heads command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.rope_by_heads_pipeline];
    [encoder setBuffer:input.buffer() offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    [encoder setBuffer:dims_buffer offset:0 atIndex:2];

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
            "Metal RoPE-by-heads command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer causal_grouped_attention_scores_f32_with_metal_buffers(
    const MetalBackendBufferStorage& query,
    const MetalBackendBufferStorage& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
) {
    auto& state = shared_matmul_pipeline_state();

    const auto query_rows = query.descriptor().shape.dimension(0);
    const auto key_rows = key.descriptor().shape.dimension(0);
    const GroupedAttentionScoreDimensions dims {
        .query_rows = require_u32_dimension(query_rows, "query_rows"),
        .key_rows = require_u32_dimension(key_rows, "key_rows"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
        .key_length = require_u32_dimension(key_length, "key_length"),
        .attention_token_count = require_u32_dimension(attention_token_count, "attention_token_count"),
        .query_position_start = require_u32_dimension(query_position_start, "query_position_start"),
        .scale = scale,
        .logit_softcap = logit_softcap,
    };
    const auto output_descriptor = TensorDescriptor::from(
        Shape::from({query_rows, head_count, attention_token_count}),
        DType::f32
    );

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(GroupedAttentionScoreDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention score buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention score command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.causal_grouped_attention_scores_pipeline];
    [encoder setBuffer:query.buffer() offset:0 atIndex:0];
    [encoder setBuffer:key.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(8, 8, 2);
    const auto threadgroup_columns = static_cast<NSUInteger>((attention_token_count + 7U) / 8U);
    const auto threadgroup_heads = static_cast<NSUInteger>((head_count + 7U) / 8U);
    const auto threadgroup_rows = static_cast<NSUInteger>((query_rows + 1U) / 2U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_heads, threadgroup_rows);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention score command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer causal_grouped_attention_values_f32_with_metal_buffers(
    const MetalBackendBufferStorage& probabilities,
    const MetalBackendBufferStorage& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
) {
    auto& state = shared_matmul_pipeline_state();

    const auto query_rows = probabilities.descriptor().shape.dimension(0);
    const auto value_rows = value.descriptor().shape.dimension(0);
    const GroupedAttentionValueDimensions dims {
        .query_rows = require_u32_dimension(query_rows, "query_rows"),
        .value_rows = require_u32_dimension(value_rows, "value_rows"),
        .head_count = require_u32_dimension(head_count, "head_count"),
        .kv_head_count = require_u32_dimension(kv_head_count, "kv_head_count"),
        .value_length = require_u32_dimension(value_length, "value_length"),
        .attention_token_count = require_u32_dimension(attention_token_count, "attention_token_count"),
        .query_position_start = require_u32_dimension(query_position_start, "query_position_start"),
    };
    const auto output_descriptor = TensorDescriptor::from(
        Shape::from({query_rows, head_count * value_length}),
        DType::f32
    );

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(GroupedAttentionValueDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention value buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention value command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.causal_grouped_attention_values_pipeline];
    [encoder setBuffer:probabilities.buffer() offset:0 atIndex:0];
    [encoder setBuffer:value.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    const MTLSize threads_per_threadgroup = MTLSizeMake(8, 8, 2);
    const auto threadgroup_columns = static_cast<NSUInteger>((value_length + 7U) / 8U);
    const auto threadgroup_heads = static_cast<NSUInteger>((head_count + 7U) / 8U);
    const auto threadgroup_rows = static_cast<NSUInteger>((query_rows + 1U) / 2U);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_columns, threadgroup_heads, threadgroup_rows);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal causal grouped attention value command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer softmax_f32_with_metal_buffer(const MetalBackendBufferStorage& input) {
    auto& state = shared_matmul_pipeline_state();

    const auto& input_descriptor = input.descriptor();
    const auto columns = input_descriptor.shape.dimension(input_descriptor.shape.rank() - 1U);
    const auto rows = input_descriptor.element_count / columns;
    const SoftmaxDimensions dims {
        .rows = require_u32_dimension(rows, "softmax rows"),
        .columns = require_u32_dimension(columns, "softmax columns"),
    };

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      input_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(SoftmaxDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softmax buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softmax command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.softmax_pipeline];
    [encoder setBuffer:input.buffer() offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    [encoder setBuffer:dims_buffer offset:0 atIndex:2];

    constexpr NSUInteger threads_per_threadgroup_count = 64U;
    const MTLSize threads_per_threadgroup = MTLSizeMake(threads_per_threadgroup_count, 1, 1);
    const auto threadgroup_rows = static_cast<NSUInteger>((rows + threads_per_threadgroup_count - 1U)
        / threads_per_threadgroup_count);
    const MTLSize threadgroups = MTLSizeMake(threadgroup_rows, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softmax command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(input_descriptor, output_buffer);
}

BackendBuffer softcap_f32_with_metal_buffer(
    const MetalBackendBufferStorage& input,
    float softcap
) {
    auto& state = shared_matmul_pipeline_state();

    const auto& input_descriptor = input.descriptor();
    const auto element_count = input_descriptor.element_count;
    const SoftcapDimensions dims {
        .elements = require_u32_dimension(element_count, "softcap elements"),
        .softcap = softcap,
    };

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      input_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(SoftcapDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softcap buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softcap command buffer creation failed"
        );
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal softcap command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.softcap_pipeline];
    [encoder setBuffer:input.buffer() offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    [encoder setBuffer:dims_buffer offset:0 atIndex:2];

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
            "Metal softcap command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(input_descriptor, output_buffer);
}

ops::TopLogit top_logit_f32_with_metal_buffer(
    const MetalBackendBufferStorage& input
) {
    static_assert(sizeof(TopLogitMetalResult) == 8);

    auto& state = shared_matmul_pipeline_state();
    const auto& input_descriptor = input.descriptor();
    const auto rank = input_descriptor.shape.rank();
    const auto columns = input_descriptor.shape.dimension(rank - 1U);
    const auto rows = input_descriptor.element_count / columns;
    const TopLogitDimensions dims {
        .row_offset = require_u32_dimension((rows - 1U) * columns, "top-logit row offset"),
        .columns = require_u32_dimension(columns, "top-logit columns"),
    };

    id<MTLBuffer> result_buffer = [state.device newBufferWithLength:sizeof(TopLogitMetalResult)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(TopLogitDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (result_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal top-logit buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal top-logit command buffer creation failed"
        );
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal top-logit command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.top_logit_pipeline];
    [encoder setBuffer:input.buffer() offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    [encoder setBuffer:dims_buffer offset:0 atIndex:2];

    const MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const MTLSize threadgroups = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal top-logit command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    TopLogitMetalResult result {};
    std::memcpy(&result, [result_buffer contents], sizeof(TopLogitMetalResult));
    return ops::TopLogit {
        .token = result.token,
        .logit = result.logit,
    };
}

BackendBuffer matmul_f32_quantized_gguf_weight_with_metal_buffers(
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
            "Metal backend-buffer K-quant matmul buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer K-quant matmul command encoding failed"
        );
    }

    const auto dispatch = select_quantized_matmul_dispatch(
        state, right.descriptor().dtype, rows, shared, columns
    );
    [encoder setComputePipelineState:dispatch.pipeline];
    [encoder setBuffer:left.buffer() offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBuffer:dims_buffer offset:0 atIndex:3];

    [encoder dispatchThreadgroups:dispatch.threadgroups
              threadsPerThreadgroup:dispatch.threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* execution_error = [command_buffer error];
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal backend-buffer K-quant matmul command failed: "
                + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

ops::F32F16GGUFWeightTripleMatmulBackendBuffers matmul_f32_f16_gguf_weight_triple_with_metal_buffers(
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
        .first = make_metal_backend_buffer(first_descriptor, first_buffer),
        .second = make_metal_backend_buffer(second_descriptor, second_buffer),
        .third = make_metal_backend_buffer(third_descriptor, third_buffer),
    };
}

BackendBuffer silu_multiply_f32_with_metal_buffers(
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

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer gelu_tanh_multiply_f32_with_metal_buffers(
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
            "Metal GELU-tanh multiply buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal GELU-tanh multiply command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.gelu_tanh_multiply_pipeline];
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
            "Metal GELU-tanh multiply command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer add_f32_with_metal_buffers(
    const MetalBackendBufferStorage& left,
    const MetalBackendBufferStorage& right
) {
    auto& state = shared_matmul_pipeline_state();

    const auto element_count = left.descriptor().element_count;
    const ElementwiseDimensions dims {
        .elements = require_u32_dimension(element_count, "elements"),
    };
    const auto output_descriptor = left.descriptor();

    id<MTLBuffer> output_buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(
                                      output_descriptor.byte_count
                                   )
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [state.device newBufferWithBytes:&dims
                                                          length:sizeof(ElementwiseDimensions)
                                                         options:MTLResourceStorageModeShared];
    if (output_buffer == nil || dims_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add buffer allocation failed"
        );
    }

    id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
    if (command_buffer == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add command buffer creation failed"
        );
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        throw metal_error(
            runtime::BackendErrorCode::metal_execution_failed,
            "Metal add command encoding failed"
        );
    }

    [encoder setComputePipelineState:state.add_pipeline];
    [encoder setBuffer:left.buffer() offset:0 atIndex:0];
    [encoder setBuffer:right.buffer() offset:0 atIndex:1];
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
            "Metal add command failed: " + ns_error_message(execution_error, "unknown error")
        );
    }

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
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

    return make_metal_backend_buffer(output_descriptor, output_buffer);
}

BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_metal_buffers(
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

    const MTLSize rms_norm_threads_per_threadgroup = MTLSizeMake(256, 1, 1);
    const auto rms_norm_threadgroup_count = static_cast<NSUInteger>(rows);
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

    return make_metal_backend_buffer(hidden_descriptor, output_buffer);
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

TensorStorage matmul_f32_quantized_gguf_weight(const TensorStorage& left, const TensorStorage& right) {
    @autoreleasepool {
        const auto left_buffer = copy_tensor_to_metal_backend(left);
        const auto right_buffer = copy_tensor_to_metal_backend(right);
        const auto output_buffer = matmul_f32_quantized_gguf_weight_with_metal_buffers(
            require_metal_backend_buffer(backend_buffer_storage(left_buffer)),
            require_metal_backend_buffer(backend_buffer_storage(right_buffer))
        );
        return copy_tensor_from_metal_backend(backend_buffer_storage(output_buffer));
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal

namespace local_gguf_engine::tensor::ops {

namespace {

void require_embedding_weight_descriptor(const TensorDescriptor& descriptor) {
    backends::metal::require_matrix_descriptor(descriptor);
    require_f16_descriptor(descriptor);
}

void require_token_ids(
    std::span<const std::uint32_t> token_ids,
    std::uint64_t vocab
) {
    if (token_ids.empty()) {
        throw std::invalid_argument("token embedding token ids must not be empty");
    }

    for (const auto token_id : token_ids) {
        if (static_cast<std::uint64_t>(token_id) >= vocab) {
            throw TensorError(TensorErrorCode::tensor_index_out_of_range, vocab, token_id);
        }
    }
}

void require_token_embedding_scale(float scale) {
    if (!(scale > 0.0F) || !std::isfinite(scale)) {
        throw std::invalid_argument("token embedding scale must be positive and finite");
    }
}

void require_add_f32_backend_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
) {
    require_f32_descriptor(left);
    require_f32_descriptor(right);
    require_same_shape(left, right);
}

}  // namespace

BackendBuffer embed_f16_gguf_weight_token_ids_to_backend(
    std::span<const std::uint32_t> token_ids,
    const BackendBuffer& embedding_weight,
    float scale
) {
    require_embedding_weight_descriptor(embedding_weight.descriptor());
    require_token_ids(token_ids, embedding_weight.descriptor().shape.dimension(1));
    require_token_embedding_scale(scale);

    switch (embedding_weight.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::embed_f16_gguf_weight_token_ids_with_cpu_buffer(
                token_ids,
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(embedding_weight)),
                scale
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::embed_f16_gguf_weight_token_ids_with_metal_buffer(
                    token_ids,
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(embedding_weight)),
                    scale
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        embedding_weight.backend(),
        "unsupported backend buffer for F16 GGUF token embedding"
    );
}

BackendBuffer rms_norm_f32_to_backend(
    const BackendBuffer& input,
    const BackendBuffer& weight,
    float epsilon
) {
    backends::metal::require_rms_norm_f32_operands(input.descriptor(), weight.descriptor(), epsilon);
    backends::metal::require_same_backend(input.backend(), weight.backend(), "RMSNorm");

    switch (input.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::rms_norm_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(weight)),
                epsilon
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::rms_norm_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(weight)),
                    epsilon
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffers for RMSNorm"
    );
}

BackendBuffer softmax_f32_to_backend(const BackendBuffer& input) {
    backends::metal::require_softmax_f32_operand(input.descriptor());

    switch (input.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::softmax_f32_with_cpu_buffer(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::softmax_f32_with_metal_buffer(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffer for softmax"
    );
}

BackendBuffer softcap_f32_to_backend(const BackendBuffer& input, float softcap) {
    backends::metal::require_softcap_f32_operand(input.descriptor(), softcap);

    switch (input.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::softcap_f32_with_cpu_buffer(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input)),
                softcap
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::softcap_f32_with_metal_buffer(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    softcap
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffer for softcap"
    );
}

TopLogit top_logit_f32_from_backend(const BackendBuffer& logits) {
    backends::metal::require_top_logit_f32_operand(logits.descriptor());

    switch (logits.backend()) {
        case runtime::BackendKind::cpu:
            return top_logit_f32(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(logits)).storage()
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::top_logit_f32_with_metal_buffer(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(logits))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        logits.backend(),
        "unsupported backend buffer for top-logit"
    );
}

BackendBuffer rope_f32_by_heads_with_position_start_to_backend(
    const BackendBuffer& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    backends::metal::require_rope_f32_by_heads_operands(
        input.descriptor(),
        position_start,
        head_count,
        head_dimension,
        rope_dimension_count,
        frequency_base
    );

    switch (input.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::rope_f32_by_heads_with_cpu_buffer(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input)),
                position_start,
                head_count,
                head_dimension,
                rope_dimension_count,
                frequency_base
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::rope_f32_by_heads_with_metal_buffer(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    position_start,
                    head_count,
                    head_dimension,
                    rope_dimension_count,
                    frequency_base
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffer for RoPE-by-heads"
    );
}

BackendBuffer rope_neox_f32_by_heads_with_position_start_to_backend(
    const BackendBuffer& input,
    std::uint64_t position_start,
    std::uint64_t head_count,
    std::uint64_t head_dimension,
    std::uint64_t rope_dimension_count,
    float frequency_base
) {
    backends::metal::require_rope_f32_by_heads_operands(
        input.descriptor(),
        position_start,
        head_count,
        head_dimension,
        rope_dimension_count,
        frequency_base
    );
    switch (input.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::rope_neox_f32_by_heads_with_cpu_buffer(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input)),
                position_start,
                head_count,
                head_dimension,
                rope_dimension_count,
                frequency_base
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::rope_neox_f32_by_heads_with_metal_buffer(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(input)),
                    position_start,
                    head_count,
                    head_dimension,
                    rope_dimension_count,
                    frequency_base
                );
            }
    }
    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        input.backend(),
        "unsupported backend buffer for NeoX RoPE-by-heads"
    );
}

BackendBuffer causal_grouped_attention_scores_f32_to_backend(
    const BackendBuffer& query,
    const BackendBuffer& key,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t key_length,
    std::uint64_t attention_token_count,
    float scale,
    float logit_softcap
) {
    backends::metal::require_causal_grouped_attention_scores_f32_operands(
        query.descriptor(),
        key.descriptor(),
        query_position_start,
        head_count,
        kv_head_count,
        key_length,
        attention_token_count,
        scale,
        logit_softcap
    );
    backends::metal::require_same_backend(query.backend(), key.backend(), "causal grouped attention scores");

    switch (query.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::causal_grouped_attention_scores_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(query)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(key)),
                query_position_start,
                head_count,
                kv_head_count,
                key_length,
                attention_token_count,
                scale,
                logit_softcap
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::causal_grouped_attention_scores_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(query)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(key)),
                    query_position_start,
                    head_count,
                    kv_head_count,
                    key_length,
                    attention_token_count,
                    scale,
                    logit_softcap
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        query.backend(),
        "unsupported backend buffers for causal grouped attention scores"
    );
}

BackendBuffer causal_grouped_attention_values_f32_to_backend(
    const BackendBuffer& probabilities,
    const BackendBuffer& value,
    std::uint64_t query_position_start,
    std::uint64_t head_count,
    std::uint64_t kv_head_count,
    std::uint64_t value_length,
    std::uint64_t attention_token_count
) {
    backends::metal::require_causal_grouped_attention_values_f32_operands(
        probabilities.descriptor(),
        value.descriptor(),
        query_position_start,
        head_count,
        kv_head_count,
        value_length,
        attention_token_count
    );
    backends::metal::require_same_backend(probabilities.backend(), value.backend(), "causal grouped attention values");

    switch (probabilities.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::causal_grouped_attention_values_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(probabilities)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(value)),
                query_position_start,
                head_count,
                kv_head_count,
                value_length,
                attention_token_count
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::causal_grouped_attention_values_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(probabilities)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(value)),
                    query_position_start,
                    head_count,
                    kv_head_count,
                    value_length,
                    attention_token_count
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        probabilities.backend(),
        "unsupported backend buffers for causal grouped attention values"
    );
}

BackendBuffer add_f32_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
) {
    require_add_f32_backend_operands(left.descriptor(), right.descriptor());
    backends::metal::require_same_backend(left.backend(), right.backend(), "F32 add");

    switch (left.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::add_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::add_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for F32 add"
    );
}

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const BackendBuffer& right) {
    backends::metal::require_f32_f16_matmul_gguf_weight_operands(left, right.descriptor());

    switch (right.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight_with_cpu_right_buffer(
                left,
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
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
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
            );
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

BackendBuffer matmul_f32_quantized_gguf_weight_to_backend(
    const BackendBuffer& left,
    const BackendBuffer& right
) {
    backends::metal::require_f32_quantized_matmul_gguf_weight_operands(left.descriptor(), right.descriptor());
    backends::metal::require_same_backend(
        left.backend(),
        right.backend(),
        "F32xK-quant GGUF weight matmul"
    );

    switch (left.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_quantized_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(right))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::matmul_f32_quantized_gguf_weight_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(left)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(right))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        left.backend(),
        "unsupported backend buffers for F32xK-quant GGUF weight matmul"
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
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight_triple_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(first_right)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(second_right)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(third_right))
            );
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
        case runtime::BackendKind::cpu:
            return backends::cpu::silu_multiply_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up))
            );
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

BackendBuffer gelu_tanh_multiply_f32_to_backend(
    const BackendBuffer& gate,
    const BackendBuffer& up
) {
    backends::metal::require_silu_multiply_f32_operands(gate.descriptor(), up.descriptor());
    backends::metal::require_same_backend(gate.backend(), up.backend(), "GELU-tanh multiply");

    switch (gate.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::gelu_tanh_multiply_f32_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up))
            );
        case runtime::BackendKind::metal:
            @autoreleasepool {
                return backends::metal::gelu_tanh_multiply_f32_with_metal_buffers(
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(gate)),
                    backends::metal::require_metal_backend_buffer(backend_buffer_storage(up))
                );
            }
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        gate.backend(),
        "unsupported backend buffers for GELU-tanh multiply"
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
        case runtime::BackendKind::cpu:
            return backends::cpu::gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(left)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(down_weight))
            );
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
        case runtime::BackendKind::cpu:
            return backends::cpu::residual_gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(input)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(attention_output)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(attention_output_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(feed_forward_norm_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(gate_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(up_weight)),
                backends::cpu::require_cpu_backend_buffer(backend_buffer_storage(down_weight)),
                epsilon
            );
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

}  // namespace local_gguf_engine::tensor::ops
