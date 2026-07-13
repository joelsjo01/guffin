#include "metal_backend.hpp"

#include "metal/metal_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

constexpr const char* matmul_f32_f16_gguf_weight_source = R"metal(
#include <metal_stdlib>
using namespace metal;

constant uint top_logit_thread_count = 256;

struct MatmulDimensions {
    uint rows;
    uint shared;
    uint columns;
};

struct EmbeddingDimensions {
    uint tokens;
    uint hidden;
    uint vocab;
    float scale;
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

kernel void embed_f16_gguf_weight_token_ids_kernel(
    const device uint* token_ids [[buffer(0)]],
    const device half* embedding [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint hidden_index = gid.x;
    const uint token_index = gid.y;
    if (token_index >= dims.tokens || hidden_index >= dims.hidden) {
        return;
    }

    const uint token_id = token_ids[token_index];
    if (token_id >= dims.vocab) {
        return;
    }

    output[token_index * dims.hidden + hidden_index] =
        float(embedding[token_id * dims.hidden + hidden_index]) * dims.scale;
}

inline ushort read_little_endian_u16(const device uchar* bytes, uint offset) {
    return ushort(bytes[offset]) | ushort(ushort(bytes[offset + 1]) << 8);
}

inline float f16_bits_to_float(ushort bits) {
    return float(as_type<half>(bits));
}

inline uint q4_k_scale_min_value(
    const device uchar* scale_min,
    uint group_index,
    bool min_value
) {
    if (group_index < 4) {
        return uint(scale_min[group_index + (min_value ? 4 : 0)] & 0x3f);
    }

    const uchar low_source = scale_min[group_index + 4];
    if (min_value) {
        const uint high_bits = uint((scale_min[group_index] & 0xc0) >> 2);
        return uint(low_source >> 4) | high_bits;
    }

    const uint high_bits = uint((scale_min[group_index - 4] & 0xc0) >> 2);
    return uint(low_source & 0x0f) | high_bits;
}

inline float q4_k_value(const device uchar* block, uint element_index) {
    const float d = f16_bits_to_float(read_little_endian_u16(block, 0));
    const float dmin = f16_bits_to_float(read_little_endian_u16(block, 2));
    const device uchar* scale_min = block + 4;
    const device uchar* quantized = block + 16;

    const uint group_index = element_index / 32;
    const uint super_group_base = (element_index / 64) * 32;
    const uint index_in_super_group = element_index % 64;
    const uchar packed = quantized[super_group_base + (index_in_super_group % 32)];
    const uint shift = index_in_super_group < 32 ? 0 : 4;
    const uint nibble = uint((packed >> shift) & 0x0f);

    return d * float(q4_k_scale_min_value(scale_min, group_index, false)) * float(nibble)
        - dmin * float(q4_k_scale_min_value(scale_min, group_index, true));
}

// GGUF/GGML projection weights are shaped [input_width, output_columns], while
// packed K-quant blocks are stored contiguously per output column.
kernel void matmul_f32_q4_k_gguf_weight_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    if (row >= dims.rows || column >= dims.columns) {
        return;
    }

    const uint blocks_per_column = dims.shared / 256;
    float sum = 0.0f;
    for (uint inner = 0; inner < dims.shared; ++inner) {
        const uint block_index = column * blocks_per_column + (inner / 256);
        const device uchar* block = right + block_index * 144;
        sum += left[row * dims.shared + inner] * q4_k_value(block, inner % 256);
    }

    output[row * dims.columns + column] = sum;
}

// Decode has a single activation row. Four 8-lane subgroups divide four
// packed blocks while the full SIMD group reduces their independent partial
// sums. This keeps packed reads contiguous inside each subgroup and avoids
// staging either activations or dequantized weights in threadgroup memory.
kernel void matvec_f32_q4_k_gguf_weight_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first_column = (threadgroup_index * 4) + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const uint subgroup = uint(lane) >> 3;
    const uint subgroup_lane = uint(lane) & 7;
    const uint blocks_per_column = dims.shared / 256;
    float first_sum = 0.0f;
    float second_sum = 0.0f;

    if (first_column < dims.columns) {
        for (uint column_block = subgroup; column_block < blocks_per_column; column_block += 4) {
            const device uchar* first_block =
                right + (first_column * blocks_per_column + column_block) * 144;
            const device uchar* second_block = second_column < dims.columns
                ? right + (second_column * blocks_per_column + column_block) * 144
                : first_block;
            const float first_d = f16_bits_to_float(read_little_endian_u16(first_block, 0));
            const float first_dmin = f16_bits_to_float(read_little_endian_u16(first_block, 2));
            const device uchar* first_scale_min = first_block + 4;
            const device uchar* first_quantized = first_block + 16;
            const float second_d = second_column < dims.columns
                ? f16_bits_to_float(read_little_endian_u16(second_block, 0))
                : 0.0f;
            const float second_dmin = second_column < dims.columns
                ? f16_bits_to_float(read_little_endian_u16(second_block, 2))
                : 0.0f;
            const device uchar* second_scale_min = second_block + 4;
            const device uchar* second_quantized = second_block + 16;

            for (uint group = 0; group < 8; ++group) {
                const uint element_base = group * 32 + subgroup_lane * 4;
                const uint quantized_base = (group >> 1) * 32 + subgroup_lane * 4;
                const uint shift = (group & 1) * 4;
                float activation_sum = 0.0f;
                float first_quantized_dot = 0.0f;
                float second_quantized_dot = 0.0f;
                for (uint item = 0; item < 4; ++item) {
                    const float activation = left[column_block * 256 + element_base + item];
                    activation_sum += activation;
                    first_quantized_dot += activation
                        * float((first_quantized[quantized_base + item] >> shift) & 0x0f);
                    if (second_column < dims.columns) {
                        second_quantized_dot += activation
                            * float((second_quantized[quantized_base + item] >> shift) & 0x0f);
                    }
                }
                first_sum += first_d
                        * float(q4_k_scale_min_value(first_scale_min, group, false))
                        * first_quantized_dot
                    - first_dmin
                        * float(q4_k_scale_min_value(first_scale_min, group, true))
                        * activation_sum;
                if (second_column < dims.columns) {
                    second_sum += second_d
                            * float(q4_k_scale_min_value(second_scale_min, group, false))
                            * second_quantized_dot
                        - second_dmin
                            * float(q4_k_scale_min_value(second_scale_min, group, true))
                            * activation_sum;
                }
            }
        }
    }

    first_sum = simd_sum(first_sum);
    second_sum = simd_sum(second_sum);
    if (lane == 0) {
        if (first_column < dims.columns) {
            output[first_column] = first_sum;
        }
        if (second_column < dims.columns) {
            output[second_column] = second_sum;
        }
    }
}

inline int signed_i8(uchar value) {
    return value >= 128 ? int(value) - 256 : int(value);
}

inline float q6_k_value(const device uchar* block, uint element_index) {
    const device uchar* low_bits = block;
    const device uchar* high_bits = block + 128;
    const device uchar* scales = block + 192;
    const float d = f16_bits_to_float(read_little_endian_u16(block, 208));

    const uint half_block_index = element_index / 128;
    const uint index_in_half_block = element_index % 128;
    const uint lane_index = index_in_half_block % 32;
    const uint low_base = half_block_index * 64;
    const uint low_plane_offset = ((index_in_half_block / 32) % 2) * 32;
    const uint low_shift = index_in_half_block < 64 ? 0 : 4;
    const uint low = uint((low_bits[low_base + low_plane_offset + lane_index] >> low_shift) & 0x0f);

    const uint high_base = half_block_index * 32;
    const uint high_shift = (index_in_half_block / 32) * 2;
    const uint high = uint((high_bits[high_base + lane_index] >> high_shift) & 0x03);
    const int quantized = int(low | (high << 4)) - 32;
    const int scale = signed_i8(scales[element_index / 16]);

    return d * float(scale) * float(quantized);
}

// Keep this separate from Q4_K: the block sizes and bit-plane layout differ.
kernel void matmul_f32_q6_k_gguf_weight_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    if (row >= dims.rows || column >= dims.columns) {
        return;
    }

    const uint blocks_per_column = dims.shared / 256;
    float sum = 0.0f;
    for (uint inner = 0; inner < dims.shared; ++inner) {
        const uint block_index = column * blocks_per_column + (inner / 256);
        const device uchar* block = right + block_index * 210;
        sum += left[row * dims.shared + inner] * q6_k_value(block, inner % 256);
    }

    output[row * dims.columns + column] = sum;
}

// Q6_K uses two interleaved bit planes, so pairs of SIMD lanes walk two
// blocks concurrently. Sixteen lanes cover one block while activations are
// reused across two output columns; the full SIMD reduction combines all
// block contributions without threadgroup memory.
kernel void matvec_f32_q6_k_gguf_weight_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first_column = (threadgroup_index * 4) + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const uint block_parity = uint(lane) & 1;
    const uint block_lane = uint(lane) >> 1;
    const uint half_block = block_lane >> 3;
    const uint lane_in_half = block_lane & 7;
    const uint lane_offset = lane_in_half * 4;
    const uint activation_base = half_block * 128 + lane_offset;
    const uint low_base = half_block * 64 + lane_offset;
    const uint high_base = half_block * 32 + lane_offset;
    const uint scale_base = half_block * 8 + lane_offset / 16;
    const uint blocks_per_column = dims.shared / 256;
    float first_sum = 0.0f;
    float second_sum = 0.0f;

    if (first_column < dims.columns) {
        for (uint column_block = block_parity; column_block < blocks_per_column; column_block += 2) {
            const device uchar* first_block =
                right + (first_column * blocks_per_column + column_block) * 210;
            const device uchar* second_block = second_column < dims.columns
                ? right + (second_column * blocks_per_column + column_block) * 210
                : first_block;
            const device uchar* first_low = first_block;
            const device uchar* first_high = first_block + 128;
            const device uchar* first_scales = first_block + 192;
            const device uchar* second_low = second_block;
            const device uchar* second_high = second_block + 128;
            const device uchar* second_scales = second_block + 192;
            float first_block_sum = 0.0f;
            float second_block_sum = 0.0f;

            for (uint item = 0; item < 4; ++item) {
                const float activation0 = left[column_block * 256 + activation_base + item];
                const float activation1 = left[column_block * 256 + activation_base + 32 + item];
                const float activation2 = left[column_block * 256 + activation_base + 64 + item];
                const float activation3 = left[column_block * 256 + activation_base + 96 + item];

                const uchar first_low0 = first_low[low_base + item];
                const uchar first_low1 = first_low[low_base + 32 + item];
                const uchar first_high_value = first_high[high_base + item];
                const int first_q0 = int((first_low0 & 0x0f) | ((first_high_value & 0x03) << 4)) - 32;
                const int first_q1 = int((first_low1 & 0x0f) | ((first_high_value & 0x0c) << 2)) - 32;
                const int first_q2 = int((first_low0 >> 4) | (first_high_value & 0x30)) - 32;
                const int first_q3 = int((first_low1 >> 4) | ((first_high_value & 0xc0) >> 2)) - 32;
                first_block_sum += activation0 * float(first_q0 * signed_i8(first_scales[scale_base]))
                    + activation1 * float(first_q1 * signed_i8(first_scales[scale_base + 2]))
                    + activation2 * float(first_q2 * signed_i8(first_scales[scale_base + 4]))
                    + activation3 * float(first_q3 * signed_i8(first_scales[scale_base + 6]));

                if (second_column < dims.columns) {
                    const uchar second_low0 = second_low[low_base + item];
                    const uchar second_low1 = second_low[low_base + 32 + item];
                    const uchar second_high_value = second_high[high_base + item];
                    const int second_q0 = int(
                        (second_low0 & 0x0f) | ((second_high_value & 0x03) << 4)
                    ) - 32;
                    const int second_q1 = int(
                        (second_low1 & 0x0f) | ((second_high_value & 0x0c) << 2)
                    ) - 32;
                    const int second_q2 = int((second_low0 >> 4) | (second_high_value & 0x30)) - 32;
                    const int second_q3 = int(
                        (second_low1 >> 4) | ((second_high_value & 0xc0) >> 2)
                    ) - 32;
                    second_block_sum +=
                        activation0 * float(second_q0 * signed_i8(second_scales[scale_base]))
                        + activation1 * float(second_q1 * signed_i8(second_scales[scale_base + 2]))
                        + activation2 * float(second_q2 * signed_i8(second_scales[scale_base + 4]))
                        + activation3 * float(second_q3 * signed_i8(second_scales[scale_base + 6]));
                }
            }

            first_sum += f16_bits_to_float(read_little_endian_u16(first_block, 208)) * first_block_sum;
            if (second_column < dims.columns) {
                second_sum += f16_bits_to_float(read_little_endian_u16(second_block, 208))
                    * second_block_sum;
            }
        }
    }

    first_sum = simd_sum(first_sum);
    second_sum = simd_sum(second_sum);
    if (lane == 0) {
        if (first_column < dims.columns) {
            output[first_column] = first_sum;
        }
        if (second_column < dims.columns) {
            output[second_column] = second_sum;
        }
    }
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

struct SoftcapDimensions {
    uint elements;
    float softcap;
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

kernel void softcap_f32_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftcapDimensions& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.elements) {
        return;
    }

    output[gid] = tanh(input[gid] / dims.softcap) * dims.softcap;
}

kernel void noop_kernel(uint gid [[thread_position_in_grid]]) {
    (void)gid;
}

struct RMSNormDimensions {
    uint rows;
    uint hidden;
    float epsilon;
};

struct RoPEByHeadsDimensions {
    uint rows;
    uint head_count;
    uint head_dimension;
    uint rope_dimension_count;
    uint position_start;
    float frequency_base;
};

struct GroupedAttentionScoreDimensions {
    uint query_rows;
    uint key_rows;
    uint head_count;
    uint kv_head_count;
    uint key_length;
    uint attention_token_count;
    uint query_position_start;
    float scale;
    float logit_softcap;
};

struct GroupedAttentionValueDimensions {
    uint query_rows;
    uint value_rows;
    uint head_count;
    uint kv_head_count;
    uint value_length;
    uint attention_token_count;
    uint query_position_start;
};

struct SoftmaxDimensions {
    uint rows;
    uint columns;
};

struct TopLogitDimensions {
    uint row_offset;
    uint columns;
};

struct TopLogitResult {
    uint token;
    float logit;
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

inline float rope_angle(
    uint position,
    uint pair_offset,
    uint rope_dimension_count,
    float frequency_base
) {
    return float(position) / pow(frequency_base, float(pair_offset) / float(rope_dimension_count));
}

kernel void rope_f32_by_heads_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RoPEByHeadsDimensions& dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    const uint row_width = dims.head_count * dims.head_dimension;
    if (row >= dims.rows || column >= row_width) {
        return;
    }

    const uint row_offset = row * row_width;
    const uint head_column = column % dims.head_dimension;
    const uint input_index = row_offset + column;
    if (head_column >= dims.rope_dimension_count) {
        output[input_index] = input[input_index];
        return;
    }

    const uint pair_column = head_column - (head_column % 2);
    const uint head_first_column = column - head_column;
    const uint first_index = row_offset + head_first_column + pair_column;
    const float first = input[first_index];
    const float second = input[first_index + 1];
    const float angle = rope_angle(
        dims.position_start + row,
        pair_column,
        dims.rope_dimension_count,
        dims.frequency_base
    );
    const float angle_cos = cos(angle);
    const float angle_sin = sin(angle);

    output[input_index] = head_column == pair_column
        ? first * angle_cos - second * angle_sin
        : first * angle_sin + second * angle_cos;
}

kernel void causal_grouped_attention_scores_f32_kernel(
    const device float* query [[buffer(0)]],
    const device float* key [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GroupedAttentionScoreDimensions& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint score_column = gid.x;
    const uint head = gid.y;
    const uint query_row = gid.z;
    if (query_row >= dims.query_rows || head >= dims.head_count || score_column >= dims.attention_token_count) {
        return;
    }

    const uint query_position = dims.query_position_start + query_row;
    const uint first_key_position = query_position + 1 > dims.attention_token_count
        ? query_position + 1 - dims.attention_token_count
        : 0;
    const uint key_row = first_key_position + score_column;
    const uint output_index =
        (query_row * dims.head_count + head) * dims.attention_token_count + score_column;
    if (key_row > query_position || key_row >= dims.key_rows) {
        output[output_index] = -INFINITY;
        return;
    }

    const uint query_width = dims.head_count * dims.key_length;
    const uint key_width = dims.kv_head_count * dims.key_length;
    const uint kv_group_size = dims.head_count / dims.kv_head_count;
    const uint kv_head = head / kv_group_size;
    const uint query_first_column = head * dims.key_length;
    const uint key_first_column = kv_head * dims.key_length;

    float dot = 0.0f;
    for (uint column = 0; column < dims.key_length; ++column) {
        dot += query[query_row * query_width + query_first_column + column]
            * key[key_row * key_width + key_first_column + column];
    }

    const float scaled = dot * dims.scale;
    output[output_index] = tanh(scaled / dims.logit_softcap) * dims.logit_softcap;
}

kernel void causal_grouped_attention_values_f32_kernel(
    const device float* probabilities [[buffer(0)]],
    const device float* value [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GroupedAttentionValueDimensions& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint value_column = gid.x;
    const uint head = gid.y;
    const uint query_row = gid.z;
    if (query_row >= dims.query_rows || head >= dims.head_count || value_column >= dims.value_length) {
        return;
    }

    const uint query_position = dims.query_position_start + query_row;
    const uint first_value_position = query_position + 1 > dims.attention_token_count
        ? query_position + 1 - dims.attention_token_count
        : 0;
    const uint value_width = dims.kv_head_count * dims.value_length;
    const uint kv_group_size = dims.head_count / dims.kv_head_count;
    const uint kv_head = head / kv_group_size;
    const uint value_first_column = kv_head * dims.value_length;
    const uint output_width = dims.head_count * dims.value_length;
    const uint output_index = query_row * output_width + head * dims.value_length + value_column;

    float sum = 0.0f;
    for (uint score_column = 0; score_column < dims.attention_token_count; ++score_column) {
        const uint value_row = first_value_position + score_column;
        if (value_row > query_position || value_row >= dims.value_rows) {
            continue;
        }

        const uint probability_index =
            (query_row * dims.head_count + head) * dims.attention_token_count + score_column;
        sum += probabilities[probability_index]
            * value[value_row * value_width + value_first_column + value_column];
    }

    output[output_index] = sum;
}

kernel void softmax_f32_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxDimensions& dims [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= dims.rows) {
        return;
    }

    const uint row_offset = row * dims.columns;
    float max_value = input[row_offset];
    for (uint column = 1; column < dims.columns; ++column) {
        max_value = max(max_value, input[row_offset + column]);
    }

    float exp_sum = 0.0f;
    for (uint column = 0; column < dims.columns; ++column) {
        const float exp_value = exp(input[row_offset + column] - max_value);
        exp_sum += exp_value;
        output[row_offset + column] = exp_value;
    }

    for (uint column = 0; column < dims.columns; ++column) {
        output[row_offset + column] = output[row_offset + column] / exp_sum;
    }
}

kernel void top_logit_f32_kernel(
    const device float* input [[buffer(0)]],
    device TopLogitResult* result [[buffer(1)]],
    constant TopLogitDimensions& dims [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    threadgroup float best_values[top_logit_thread_count];
    threadgroup uint best_tokens[top_logit_thread_count];

    uint best_token = uint(-1);
    float best_value = 0.0f;
    for (uint token = thread_index; token < dims.columns; token += top_logit_thread_count) {
        const float value = input[dims.row_offset + token];
        if (best_token == uint(-1) || value > best_value) {
            best_token = token;
            best_value = value;
        }
    }

    best_values[thread_index] = best_value;
    best_tokens[thread_index] = best_token;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = top_logit_thread_count / 2; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            const uint other_token = best_tokens[thread_index + stride];
            if (other_token != uint(-1)) {
                const float other_value = best_values[thread_index + stride];
                const bool replace =
                    best_tokens[thread_index] == uint(-1)
                    || other_value > best_values[thread_index]
                    || (other_value == best_values[thread_index] && other_token < best_tokens[thread_index]);
                if (replace) {
                    best_values[thread_index] = other_value;
                    best_tokens[thread_index] = other_token;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        result[0].token = best_tokens[0];
        result[0].logit = best_values[0];
    }
}

)metal";

}  // namespace

const char* metal_kernel_source() noexcept {
    return matmul_f32_f16_gguf_weight_source;
}

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

}  // namespace local_gguf_engine::tensor::backends::metal
