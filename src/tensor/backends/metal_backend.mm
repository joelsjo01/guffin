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
    uint token_offset;
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

    const uint token_id = token_ids[dims.token_offset + token_index];
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

// One SIMD group owns one output column. Each lane decodes eight values from a
// native 256-value block once, then reuses them across a compile-time row tile.
// Four SIMD groups provide enough independent columns for occupancy without
// duplicating packed-weight reads between activation rows.
template <ushort row_tile>
inline void matmul_f32_q4_k_rows(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index,
    ushort simdgroup_index,
    ushort lane
) {
    const uint column = threadgroup_index.x * 4 + uint(simdgroup_index);
    const uint first_row = threadgroup_index.y * uint(row_tile);
    const uint blocks_per_column = dims.shared / 256;
    float sums[row_tile];
    for (ushort row = 0; row < row_tile; ++row) {
        sums[row] = 0.0f;
    }

    if (column < dims.columns) {
        for (uint column_block = 0; column_block < blocks_per_column; ++column_block) {
            const device uchar* block = right + (column * blocks_per_column + column_block) * 144;
            const float d = f16_bits_to_float(read_little_endian_u16(block, 0));
            const float dmin = f16_bits_to_float(read_little_endian_u16(block, 2));
            const device uchar* scale_min = block + 4;
            const device uchar* quantized = block + 16;

            for (uint group = 0; group < 8; ++group) {
                const uint inner = column_block * 256 + group * 32 + uint(lane);
                const uint packed_index = (group >> 1) * 32 + uint(lane);
                const uint shift = (group & 1) * 4;
                const float quantized_value = float((quantized[packed_index] >> shift) & 0x0f);
                const float weight = d
                        * float(q4_k_scale_min_value(scale_min, group, false))
                        * quantized_value
                    - dmin * float(q4_k_scale_min_value(scale_min, group, true));
                for (ushort row = 0; row < row_tile; ++row) {
                    const uint output_row = first_row + uint(row);
                    if (output_row < dims.rows) {
                        sums[row] += left[output_row * dims.shared + inner] * weight;
                    }
                }
            }
        }
    }

    for (ushort row = 0; row < row_tile; ++row) {
        sums[row] = simd_sum(sums[row]);
        const uint output_row = first_row + uint(row);
        if (lane == 0 && column < dims.columns && output_row < dims.rows) {
            output[output_row * dims.columns + column] = sums[row];
        }
    }
}

#define DEFINE_Q4_K_ROW_KERNEL(rows) \
kernel void matmul_f32_q4_k_rows_##rows##_kernel( \
    const device float* left [[buffer(0)]], \
    const device uchar* right [[buffer(1)]], \
    device float* output [[buffer(2)]], \
    constant MatmulDimensions& dims [[buffer(3)]], \
    uint2 threadgroup_index [[threadgroup_position_in_grid]], \
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]] \
) { \
    matmul_f32_q4_k_rows<rows>( \
        left, right, output, dims, threadgroup_index, simdgroup_index, lane \
    ); \
}

DEFINE_Q4_K_ROW_KERNEL(2)
DEFINE_Q4_K_ROW_KERNEL(4)
DEFINE_Q4_K_ROW_KERNEL(8)
DEFINE_Q4_K_ROW_KERNEL(16)

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

// Q6_K uses the same row-reuse ownership as Q4_K but retains its native
// interleaved low/high bit planes and signed per-16-value scales.
template <ushort row_tile>
inline void matmul_f32_q6_k_rows(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index,
    ushort simdgroup_index,
    ushort lane
) {
    const uint column = threadgroup_index.x * 4 + uint(simdgroup_index);
    const uint first_row = threadgroup_index.y * uint(row_tile);
    const uint blocks_per_column = dims.shared / 256;
    float sums[row_tile];
    for (ushort row = 0; row < row_tile; ++row) {
        sums[row] = 0.0f;
    }

    if (column < dims.columns) {
        for (uint column_block = 0; column_block < blocks_per_column; ++column_block) {
            const device uchar* block = right + (column * blocks_per_column + column_block) * 210;
            const device uchar* low_bits = block;
            const device uchar* high_bits = block + 128;
            const device uchar* scales = block + 192;
            const float d = f16_bits_to_float(read_little_endian_u16(block, 208));

            for (uint group = 0; group < 8; ++group) {
                const uint half_block = group >> 2;
                const uint group_in_half = group & 3;
                const uint low_index = half_block * 64 + (group_in_half & 1) * 32 + uint(lane);
                const uint low_shift = group_in_half < 2 ? 0 : 4;
                const uint low = uint((low_bits[low_index] >> low_shift) & 0x0f);
                const uint high_index = half_block * 32 + uint(lane);
                const uint high_shift = group_in_half * 2;
                const uint high = uint((high_bits[high_index] >> high_shift) & 0x03);
                const int quantized = int(low | (high << 4)) - 32;
                const int scale = signed_i8(scales[group * 2 + uint(lane) / 16]);
                const float weight = d * float(scale * quantized);
                const uint inner = column_block * 256 + group * 32 + uint(lane);
                for (ushort row = 0; row < row_tile; ++row) {
                    const uint output_row = first_row + uint(row);
                    if (output_row < dims.rows) {
                        sums[row] += left[output_row * dims.shared + inner] * weight;
                    }
                }
            }
        }
    }

    for (ushort row = 0; row < row_tile; ++row) {
        sums[row] = simd_sum(sums[row]);
        const uint output_row = first_row + uint(row);
        if (lane == 0 && column < dims.columns && output_row < dims.rows) {
            output[output_row * dims.columns + column] = sums[row];
        }
    }
}

#define DEFINE_Q6_K_ROW_KERNEL(rows) \
kernel void matmul_f32_q6_k_rows_##rows##_kernel( \
    const device float* left [[buffer(0)]], \
    const device uchar* right [[buffer(1)]], \
    device float* output [[buffer(2)]], \
    constant MatmulDimensions& dims [[buffer(3)]], \
    uint2 threadgroup_index [[threadgroup_position_in_grid]], \
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]] \
) { \
    matmul_f32_q6_k_rows<rows>( \
        left, right, output, dims, threadgroup_index, simdgroup_index, lane \
    ); \
}

DEFINE_Q6_K_ROW_KERNEL(2)
DEFINE_Q6_K_ROW_KERNEL(4)
DEFINE_Q6_K_ROW_KERNEL(8)
DEFINE_Q6_K_ROW_KERNEL(16)

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

kernel void gelu_tanh_multiply_f32_kernel(
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
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    constexpr float cubic_coefficient = 0.044715f;
    const float inner = sqrt_two_over_pi * (value + cubic_coefficient * value * value * value);
    output[gid] = (0.5f * value * (1.0f + precise::tanh(inner))) * up[gid];
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

    output[gid] = precise::tanh(input[gid] / dims.softcap) * dims.softcap;
}

kernel void noop_kernel(uint gid [[thread_position_in_grid]]) {
    (void)gid;
}

struct RMSNormDimensions {
    uint rows;
    uint hidden;
    uint input_row_start;
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

struct Gemma2KVCacheStoreDimensions {
    uint rows;
    uint head_count;
    uint head_dimension;
    uint rope_dimension_count;
    uint position_start;
    uint cache_row_mask;
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

struct Gemma2PrefillAttentionScoreDimensions {
    uint query_rows;
    uint key_rows;
    uint head_count;
    uint kv_head_count;
    uint key_length;
    uint attention_token_count;
    uint query_position_start;
    uint cache_row_mask;
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

struct Gemma2PrefillAttentionValueDimensions {
    uint query_rows;
    uint value_rows;
    uint head_count;
    uint kv_head_count;
    uint value_length;
    uint attention_token_count;
    uint query_position_start;
    uint cache_row_mask;
};

// Keep this layout byte-for-byte aligned with Gemma2DecodeAttentionDimensions
// in metal_command_context.mm. The production specialization deliberately
// carries the fixed shape as arguments so host validation and device indexing
// cannot silently disagree.
struct Gemma2DecodeAttentionDimensions {
    uint cache_row_mask;
    uint first_position;
    uint attention_token_count;
    uint partition_count;
    uint query_head_count;
    uint kv_head_count;
    uint key_dimension;
    uint value_dimension;
    float scale;
    float logit_softcap;
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
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroup_count [[simdgroups_per_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]
) {
    if (row >= dims.rows) {
        return;
    }

    threadgroup float partial_sums[32];
    const uint input_row_offset = (dims.input_row_start + row) * dims.hidden;
    const uint output_row_offset = row * dims.hidden;

    float sum_squares = 0.0f;
    if ((dims.hidden & 3u) == 0u) {
        const device float4* input4 = reinterpret_cast<const device float4*>(input + input_row_offset);
        const uint vector_count = dims.hidden / 4u;
        for (uint column = thread_index; column < vector_count; column += thread_count) {
            const float4 value = input4[column];
            sum_squares += dot(value, value);
        }
    } else {
        for (uint column = thread_index; column < dims.hidden; column += thread_count) {
            const float value = input[input_row_offset + column];
            sum_squares += value * value;
        }
    }

    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) {
        partial_sums[simdgroup_index] = sum_squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index == 0u) {
        sum_squares = simd_sum(lane < simdgroup_count ? partial_sums[lane] : 0.0f);
        if (lane == 0u) {
            partial_sums[0] = sum_squares;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = rsqrt((partial_sums[0] / float(dims.hidden)) + dims.epsilon);
    if ((dims.hidden & 3u) == 0u) {
        const device float4* input4 = reinterpret_cast<const device float4*>(input + input_row_offset);
        const device float4* weight4 = reinterpret_cast<const device float4*>(weight);
        device float4* output4 = reinterpret_cast<device float4*>(output + output_row_offset);
        const uint vector_count = dims.hidden / 4u;
        for (uint column = thread_index; column < vector_count; column += thread_count) {
            output4[column] = input4[column] * inverse_rms * weight4[column];
        }
    } else {
        for (uint column = thread_index; column < dims.hidden; column += thread_count) {
            output[output_row_offset + column] =
                input[input_row_offset + column] * inverse_rms * weight[column];
        }
    }
}

kernel void rms_norm_add_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant RMSNormDimensions& dims [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroup_count [[simdgroups_per_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[32];
    float sum_squares = 0.0f;
    const uint vector_count = dims.hidden / 4u;
    const device float4* input4 = reinterpret_cast<const device float4*>(input);
    for (uint column = thread_index; column < vector_count; column += thread_count) {
        const float4 value = input4[column];
        sum_squares += dot(value, value);
    }

    sum_squares = simd_sum(sum_squares);
    if (lane == 0u) {
        partial_sums[simdgroup_index] = sum_squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index == 0u) {
        sum_squares = simd_sum(lane < simdgroup_count ? partial_sums[lane] : 0.0f);
        if (lane == 0u) {
            partial_sums[0] = sum_squares;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = rsqrt((partial_sums[0] / float(dims.hidden)) + dims.epsilon);
    const device float4* weight4 = reinterpret_cast<const device float4*>(weight);
    const device float4* residual4 = reinterpret_cast<const device float4*>(residual);
    device float4* output4 = reinterpret_cast<device float4*>(output);
    for (uint column = thread_index; column < vector_count; column += thread_count) {
        output4[column] = input4[column] * inverse_rms * weight4[column] + residual4[column];
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

kernel void rope_neox_f32_by_heads_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RoPEByHeadsDimensions& dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    const uint row_width = dims.head_count * dims.head_dimension;
    if (row >= dims.rows || column >= row_width) return;

    const uint head_column = column % dims.head_dimension;
    const uint input_index = row * row_width + column;
    if (head_column >= dims.rope_dimension_count) {
        output[input_index] = input[input_index];
        return;
    }

    const uint half_dimension = dims.rope_dimension_count / 2;
    const uint pair_column = head_column % half_dimension;
    const uint head_first_column = column - head_column;
    const uint first_index = row * row_width + head_first_column + pair_column;
    const uint second_index = first_index + half_dimension;
    const float first = input[first_index];
    const float second = input[second_index];
    const float angle = rope_angle(
        dims.position_start + row,
        pair_column * 2,
        dims.rope_dimension_count,
        dims.frequency_base
    );
    const float angle_cos = cos(angle);
    const float angle_sin = sin(angle);
    output[input_index] = head_column < half_dimension
        ? first * angle_cos - second * angle_sin
        : first * angle_sin + second * angle_cos;
}

// K RoPE and both narrowing stores share a row mapping and one traversal. The
// exact power-of-two ring avoids a modulo in every cache producer and reader.
kernel void gemma2_kv_cache_store_f16_kernel(
    const device float* raw_key [[buffer(0)]],
    const device float* raw_value [[buffer(1)]],
    device half* key_cache [[buffer(2)]],
    device half* value_cache [[buffer(3)]],
    constant Gemma2KVCacheStoreDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    const uint row_width = dims.head_count * dims.head_dimension;
    if (row >= dims.rows || column >= row_width) return;

    const uint head_column = column % dims.head_dimension;
    const uint source_index = row * row_width + column;
    float rotated = raw_key[source_index];
    if (head_column < dims.rope_dimension_count) {
        const uint half_dimension = dims.rope_dimension_count / 2;
        const uint pair_column = head_column % half_dimension;
        const uint head_first_column = column - head_column;
        const uint first_index = row * row_width + head_first_column + pair_column;
        const uint second_index = first_index + half_dimension;
        const float first = raw_key[first_index];
        const float second = raw_key[second_index];
        const float angle = rope_angle(
            dims.position_start + row,
            pair_column * 2,
            dims.rope_dimension_count,
            dims.frequency_base
        );
        rotated = head_column < half_dimension
            ? first * cos(angle) - second * sin(angle)
            : first * sin(angle) + second * cos(angle);
    }
    const uint cache_row = (dims.position_start + row) & dims.cache_row_mask;
    const uint destination_index = cache_row * row_width + column;
    key_cache[destination_index] = half(rotated);
    value_cache[destination_index] = half(raw_value[source_index]);
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
    if (query_row >= dims.query_rows || head >= dims.head_count || score_column >= dims.attention_token_count) return;
    const uint query_position = dims.query_position_start + query_row;
    const uint first_key_position = query_position + 1 > dims.attention_token_count
        ? query_position + 1 - dims.attention_token_count : 0;
    const uint key_row = first_key_position + score_column;
    const uint output_index = (query_row * dims.head_count + head) * dims.attention_token_count + score_column;
    if (key_row > query_position || key_row >= dims.key_rows) {
        output[output_index] = -INFINITY;
        return;
    }
    const uint query_width = dims.head_count * dims.key_length;
    const uint key_width = dims.kv_head_count * dims.key_length;
    const uint kv_head = head / (dims.head_count / dims.kv_head_count);
    const uint query_first_column = head * dims.key_length;
    const uint key_first_column = kv_head * dims.key_length;
    float dot = 0.0f;
    for (uint column = 0; column < dims.key_length; ++column) {
        dot += query[query_row * query_width + query_first_column + column]
            * key[key_row * key_width + key_first_column + column];
    }
    const float scaled = dot * dims.scale;
    output[output_index] = precise::tanh(scaled / dims.logit_softcap) * dims.logit_softcap;
}

kernel void causal_grouped_attention_scores_f16_cache_kernel(
    const device float* query [[buffer(0)]],
    const device half* key [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Gemma2PrefillAttentionScoreDimensions& dims [[buffer(3)]],
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
    const uint logical_key_row = first_key_position + score_column;
    const uint output_index =
        (query_row * dims.head_count + head) * dims.attention_token_count + score_column;
    if (logical_key_row > query_position) {
        output[output_index] = -INFINITY;
        return;
    }

    const uint key_row = logical_key_row & dims.cache_row_mask;
    const uint query_width = dims.head_count * dims.key_length;
    const uint key_width = dims.kv_head_count * dims.key_length;
    const uint kv_group_size = dims.head_count / dims.kv_head_count;
    const uint kv_head = head / kv_group_size;
    const uint query_first_column = head * dims.key_length;
    const uint key_first_column = kv_head * dims.key_length;

    float dot = 0.0f;
    for (uint column = 0; column < dims.key_length; ++column) {
        dot += query[query_row * query_width + query_first_column + column]
            * float(key[key_row * key_width + key_first_column + column]);
    }

    const float scaled = dot * dims.scale;
    output[output_index] = precise::tanh(scaled / dims.logit_softcap) * dims.logit_softcap;
}

// Each workgroup owns one KV head and both query heads that share it. Four
// SIMD groups stream disjoint 32-token blocks; a lane owns eight adjacent head
// elements as two float4 values. K/V therefore enter the shader once per
// grouped pair, while both online-softmax states remain register resident.
kernel void gemma2_decode_attention_f16_cache_kernel(
    const device float4* query [[buffer(0)]],
    const device half4* key [[buffer(1)]],
    const device half4* value [[buffer(2)]],
    device float4* output [[buffer(3)]],
    device float4* partial_output [[buffer(4)]],
    device float* partial_maxima [[buffer(5)]],
    device float* partial_denominators [[buffer(6)]],
    constant Gemma2DecodeAttentionDimensions& dims [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint kv_head = group.x;
    const uint partition = group.y;
    const uint first_query_head = kv_head * 2;
    const uint second_query_head = first_query_head + 1;
    const uint vector_lane = uint(lane) * 2;
    const uint query_vectors_per_head = dims.key_dimension / 4;
    const uint cache_vectors_per_row = dims.kv_head_count * query_vectors_per_head;

    const float4 query_first_0 = query[first_query_head * query_vectors_per_head + vector_lane];
    const float4 query_first_1 = query[first_query_head * query_vectors_per_head + vector_lane + 1];
    const float4 query_second_0 = query[second_query_head * query_vectors_per_head + vector_lane];
    const float4 query_second_1 = query[second_query_head * query_vectors_per_head + vector_lane + 1];

    float maximum_first = -INFINITY;
    float maximum_second = -INFINITY;
    float denominator_first = 0.0f;
    float denominator_second = 0.0f;
    float4 accumulated_first_0 = 0.0f;
    float4 accumulated_first_1 = 0.0f;
    float4 accumulated_second_0 = 0.0f;
    float4 accumulated_second_1 = 0.0f;

    const uint first_block = partition * 4 + uint(simd_group);
    const uint block_stride = dims.partition_count * 4;
    const uint block_count = (dims.attention_token_count + 31) / 32;
    for (uint block = first_block; block < block_count; block += block_stride) {
        const uint token_begin = block * 32;
        const uint token_end = min(token_begin + 32, dims.attention_token_count);
        for (uint token = token_begin; token < token_end; ++token) {
            const uint cache_row = (dims.first_position + token) & dims.cache_row_mask;
            const uint cache_vector = cache_row * cache_vectors_per_row
                + kv_head * query_vectors_per_head + vector_lane;
            const float4 key_0 = float4(key[cache_vector]);
            const float4 key_1 = float4(key[cache_vector + 1]);
            float score_first = dot(query_first_0, key_0) + dot(query_first_1, key_1);
            float score_second = dot(query_second_0, key_0) + dot(query_second_1, key_1);
            score_first = simd_sum(score_first);
            score_second = simd_sum(score_second);
            score_first = precise::tanh((score_first * dims.scale) / dims.logit_softcap)
                * dims.logit_softcap;
            score_second = precise::tanh((score_second * dims.scale) / dims.logit_softcap)
                * dims.logit_softcap;

            const float new_maximum_first = max(maximum_first, score_first);
            const float new_maximum_second = max(maximum_second, score_second);
            const float old_scale_first = denominator_first == 0.0f
                ? 0.0f : exp(maximum_first - new_maximum_first);
            const float old_scale_second = denominator_second == 0.0f
                ? 0.0f : exp(maximum_second - new_maximum_second);
            const float new_scale_first = exp(score_first - new_maximum_first);
            const float new_scale_second = exp(score_second - new_maximum_second);
            const float4 value_0 = float4(value[cache_vector]);
            const float4 value_1 = float4(value[cache_vector + 1]);
            accumulated_first_0 = accumulated_first_0 * old_scale_first + value_0 * new_scale_first;
            accumulated_first_1 = accumulated_first_1 * old_scale_first + value_1 * new_scale_first;
            accumulated_second_0 = accumulated_second_0 * old_scale_second + value_0 * new_scale_second;
            accumulated_second_1 = accumulated_second_1 * old_scale_second + value_1 * new_scale_second;
            denominator_first = denominator_first * old_scale_first + new_scale_first;
            denominator_second = denominator_second * old_scale_second + new_scale_second;
            maximum_first = new_maximum_first;
            maximum_second = new_maximum_second;
        }
    }

    threadgroup float4 shared_output[8 * 64];
    threadgroup float shared_maxima[8];
    threadgroup float shared_denominators[8];
    const uint first_state = uint(simd_group) * 2;
    const uint second_state = first_state + 1;
    shared_output[first_state * 64 + vector_lane] = accumulated_first_0;
    shared_output[first_state * 64 + vector_lane + 1] = accumulated_first_1;
    shared_output[second_state * 64 + vector_lane] = accumulated_second_0;
    shared_output[second_state * 64 + vector_lane + 1] = accumulated_second_1;
    if (lane == 0) {
        shared_maxima[first_state] = maximum_first;
        shared_maxima[second_state] = maximum_second;
        shared_denominators[first_state] = denominator_first;
        shared_denominators[second_state] = denominator_second;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0) return;
    for (uint paired_head = 0; paired_head < 2; ++paired_head) {
        float merged_maximum = -INFINITY;
        float merged_denominator = 0.0f;
        float4 merged_0 = 0.0f;
        float4 merged_1 = 0.0f;
        for (uint source_simd = 0; source_simd < 4; ++source_simd) {
            const uint state = source_simd * 2 + paired_head;
            const float source_denominator = shared_denominators[state];
            if (source_denominator == 0.0f) continue;
            const float source_maximum = shared_maxima[state];
            const float4 source_0 = shared_output[state * 64 + vector_lane];
            const float4 source_1 = shared_output[state * 64 + vector_lane + 1];
            if (merged_denominator == 0.0f) {
                merged_maximum = source_maximum;
                merged_denominator = source_denominator;
                merged_0 = source_0;
                merged_1 = source_1;
                continue;
            }
            const float new_maximum = max(merged_maximum, source_maximum);
            const float old_scale = exp(merged_maximum - new_maximum);
            const float source_scale = exp(source_maximum - new_maximum);
            merged_0 = merged_0 * old_scale + source_0 * source_scale;
            merged_1 = merged_1 * old_scale + source_1 * source_scale;
            merged_denominator = merged_denominator * old_scale + source_denominator * source_scale;
            merged_maximum = new_maximum;
        }

        const uint query_head = first_query_head + paired_head;
        if (dims.partition_count == 1) {
            const uint output_vector = query_head * query_vectors_per_head + vector_lane;
            output[output_vector] = merged_0 / merged_denominator;
            output[output_vector + 1] = merged_1 / merged_denominator;
        } else {
            const uint partial_vector =
                (partition * dims.query_head_count + query_head) * query_vectors_per_head + vector_lane;
            partial_output[partial_vector] = merged_0;
            partial_output[partial_vector + 1] = merged_1;
            if (lane == 0) {
                const uint scalar = partition * dims.query_head_count + query_head;
                partial_maxima[scalar] = merged_maximum;
                partial_denominators[scalar] = merged_denominator;
            }
        }
    }
}

kernel void gemma2_decode_attention_reduce_f32_kernel(
    const device float4* partial_output [[buffer(0)]],
    const device float* partial_maxima [[buffer(1)]],
    const device float* partial_denominators [[buffer(2)]],
    device float4* output [[buffer(3)]],
    constant Gemma2DecodeAttentionDimensions& dims [[buffer(4)]],
    uint kv_head [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint vectors_per_head = dims.value_dimension / 4;
    const uint vector_lane = uint(lane) * 2;
    const uint first_query_head = kv_head * 2;
    for (uint paired_head = 0; paired_head < 2; ++paired_head) {
        const uint query_head = first_query_head + paired_head;
        float merged_maximum = -INFINITY;
        float merged_denominator = 0.0f;
        float4 merged_0 = 0.0f;
        float4 merged_1 = 0.0f;
        for (uint partition = 0; partition < dims.partition_count; ++partition) {
            const uint scalar = partition * dims.query_head_count + query_head;
            const float source_denominator = partial_denominators[scalar];
            if (source_denominator == 0.0f) continue;
            const uint source_vector = scalar * vectors_per_head + vector_lane;
            const float source_maximum = partial_maxima[scalar];
            const float4 source_0 = partial_output[source_vector];
            const float4 source_1 = partial_output[source_vector + 1];
            if (merged_denominator == 0.0f) {
                merged_maximum = source_maximum;
                merged_denominator = source_denominator;
                merged_0 = source_0;
                merged_1 = source_1;
                continue;
            }
            const float new_maximum = max(merged_maximum, source_maximum);
            const float old_scale = exp(merged_maximum - new_maximum);
            const float source_scale = exp(source_maximum - new_maximum);
            merged_0 = merged_0 * old_scale + source_0 * source_scale;
            merged_1 = merged_1 * old_scale + source_1 * source_scale;
            merged_denominator = merged_denominator * old_scale + source_denominator * source_scale;
            merged_maximum = new_maximum;
        }
        const uint output_vector = query_head * vectors_per_head + vector_lane;
        output[output_vector] = merged_0 / merged_denominator;
        output[output_vector + 1] = merged_1 / merged_denominator;
    }
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
    if (query_row >= dims.query_rows || head >= dims.head_count || value_column >= dims.value_length) return;
    const uint query_position = dims.query_position_start + query_row;
    const uint first_value_position = query_position + 1 > dims.attention_token_count
        ? query_position + 1 - dims.attention_token_count : 0;
    const uint value_width = dims.kv_head_count * dims.value_length;
    const uint kv_head = head / (dims.head_count / dims.kv_head_count);
    const uint value_first_column = kv_head * dims.value_length;
    const uint output_width = dims.head_count * dims.value_length;
    const uint output_index = query_row * output_width + head * dims.value_length + value_column;
    float sum = 0.0f;
    for (uint score_column = 0; score_column < dims.attention_token_count; ++score_column) {
        const uint value_row = first_value_position + score_column;
        if (value_row > query_position || value_row >= dims.value_rows) continue;
        const uint probability_index =
            (query_row * dims.head_count + head) * dims.attention_token_count + score_column;
        sum += probabilities[probability_index]
            * value[value_row * value_width + value_first_column + value_column];
    }
    output[output_index] = sum;
}

kernel void causal_grouped_attention_values_f16_cache_kernel(
    const device float* probabilities [[buffer(0)]],
    const device half* value [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Gemma2PrefillAttentionValueDimensions& dims [[buffer(3)]],
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
        const uint logical_value_row = first_value_position + score_column;
        if (logical_value_row > query_position) {
            continue;
        }
        const uint value_row = logical_value_row & dims.cache_row_mask;

        const uint probability_index =
            (query_row * dims.head_count + head) * dims.attention_token_count + score_column;
        sum += probabilities[probability_index]
            * float(value[value_row * value_width + value_first_column + value_column]);
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
