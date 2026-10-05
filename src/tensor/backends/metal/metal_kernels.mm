#include "metal_kernels.hpp"

#include <string>

namespace local_gguf_engine::tensor::backends::metal {

namespace {

constexpr const char* metal_kernel_source_head = R"metal(
#include <metal_stdlib>
using namespace metal;

constant uint token_candidate_tile_size = 512;
constant uint token_candidate_max_count = 384;

// Each Q4 minimum-correction component is the left-to-right sum of one
// contiguous eight-value chunk. Transpose only its address into the packed
// consumer's [block][worker][component] layout; do not change addition order.
inline void store_q4_input_sum(device float* sums, uint chunk, float4 first, float4 second) {
    float sum = 0.0f;
    sum += first.x; sum += first.y; sum += first.z; sum += first.w;
    sum += second.x; sum += second.y; sum += second.z; sum += second.w;
    const uint local = chunk & 31U;
    const uint worker = (local & 3U) + 4U * ((local >> 3U) & 1U);
    const uint component = ((local >> 2U) & 1U) + 2U * (local >> 4U);
    sums[(chunk / 32U) * 32U + worker * 4U + component] = sum;
}

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

struct TokenCandidate {
    uint token;
    float logit;
};

struct TokenCandidateSelectionDimensions {
    uint columns;
    uint retain_count;
    uint run_count;
    uint history_count;
    float softcap;
};

struct TokenCandidateMergeDimensions {
    uint retain_count;
    uint input_run_count;
};

struct TokenCandidateBatchSelectionDimensions {
    TokenCandidateSelectionDimensions rows[2];
    uint run_stride;
};

struct TokenCandidateBatchMergeDimensions {
    uint retain_count[2];
    uint input_run_count;
    uint run_stride;
};

struct TokenCandidateResult {
    atomic_uint status;
    uint count;
    TokenCandidate candidates[token_candidate_max_count];
};

constant uint verifier_history_capacity = 64;
constant uint verifier_suffix_capacity = 23;
constant uint verifier_max_partitions = 32;

struct VerifierDeviceSelectionRow {
    uint history_token_ids[64];
    uint suffix_token_ids[23];
    uint history_head;
    uint history_length;
    uint history_window_limit;
    uint suffix_count;
};

struct VerifierResolvedSelectionRow {
    uint top_k;
    uint base_candidate_count;
    uint history_count;
    uint reserved;
    uint history_token_ids[64];
};

struct VerifierRequestResolveDimensions {
    uint rows;
    uint vocabulary_size;
    uint top_k;
    uint repetition_enabled;
};

struct VerifierInputValidationDimensions {
    uint rows;
    uint vocabulary_size;
};

struct VerifierRoPEDimensions {
    uint rows;
    uint head_count;
    uint head_dimension;
    uint rope_dimension_count;
    float frequency_base;
    float frequency_scale;
};

struct VerifierKVStoreDimensions {
    uint rows;
    uint head_count;
    uint head_dimension;
    uint rope_dimension_count;
    float frequency_base;
    float frequency_scale;
};

struct VerifierAttentionRow {
    uint base_first_position;
    uint base_token_count;
    uint ancestor_offset;
    uint ancestor_count;
};

struct VerifierAttentionDimensions {
    uint rows;
    uint cache_row_mask;
    uint query_head_count;
    uint kv_head_count;
    uint key_dimension;
    uint value_dimension;
    float scale;
    float logit_softcap;
};

struct VerifierKVScatterDimensions {
    uint rows;
    uint row_width;
    uint destination_position;
    uint destination_row_mask;
};

struct TargetCandidateBatchDimensions {
    uint rows;
    uint columns;
    uint run_count;
    uint run_stride;
    float softcap;
};

constant ushort q4_k_decode_block_count [[function_constant(0)]];
constant ushort q6_k_decode_block_count [[function_constant(1)]];
constant ushort kquant_tiled_dtype [[function_constant(2)]];
constant bool kquant_tiled_exact_16 [[function_constant(3)]];
constant ushort q8_0_decode_block_count [[function_constant(4)]];
constant ushort packed_parallel_row_count [[function_constant(5)]];
constant ushort gemma3_shared_row_count [[function_constant(6)]];
constant bool kquant_columns8 [[function_constant(7)]];

inline bool token_candidate_better(TokenCandidate left, TokenCandidate right) {
    return left.logit > right.logit || (left.logit == right.logit && left.token < right.token);
}

// Two SIMD groups each own two output columns for one activation row. The
// production 3584-wide path uses aligned vector loads; the scalar branch keeps
// the public backend contract correct for arbitrary test and diagnostic shapes.
kernel void matmul_f32_f16_gguf_weight_simd_kernel(
    const device float* left [[buffer(0)]],
    const device half* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint row = threadgroup_index.y;
    const uint first_column = threadgroup_index.x * 4 + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    float first_sum = 0.0f;
    float second_sum = 0.0f;

    if (row < dims.rows && first_column < dims.columns) {
        const device float* activation = left + row * dims.shared;
        const device half* first_weight = right + first_column * dims.shared;
        const device half* second_weight = second_column < dims.columns
            ? right + second_column * dims.shared
            : first_weight;

        if ((dims.shared & 3U) == 0U) {
            const device float4* activation4 = reinterpret_cast<const device float4*>(activation);
            const device half4* first_weight4 = reinterpret_cast<const device half4*>(first_weight);
            const device half4* second_weight4 = reinterpret_cast<const device half4*>(second_weight);
            const uint vector_count = dims.shared / 4;
            for (uint vector_index = uint(lane); vector_index < vector_count; vector_index += 32) {
                const float4 values = activation4[vector_index];
                first_sum += dot(values, float4(first_weight4[vector_index]));
                if (second_column < dims.columns) {
                    second_sum += dot(values, float4(second_weight4[vector_index]));
                }
            }
        } else {
            for (uint inner = uint(lane); inner < dims.shared; inner += 32) {
                const float value = activation[inner];
                first_sum += value * float(first_weight[inner]);
                if (second_column < dims.columns) {
                    second_sum += value * float(second_weight[inner]);
                }
            }
        }
    }

    first_sum = simd_sum(first_sum);
    second_sum = simd_sum(second_sum);
    if (lane == 0 && row < dims.rows) {
        if (first_column < dims.columns) {
            output[row * dims.columns + first_column] = first_sum;
        }
        if (second_column < dims.columns) {
            output[row * dims.columns + second_column] = second_sum;
        }
    }
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

// Q8_0 blocks are an unpadded 2-byte F16 scale followed by 32 signed bytes.
// Two SIMD groups reuse each activation lane across four output columns.
kernel void matvec_f32_q8_0_packed_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first_column = threadgroup_index * 4 + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const uint blocks_per_column = q8_0_decode_block_count == 0
        ? dims.shared / 32
        : uint(q8_0_decode_block_count);
    const uint column_stride = blocks_per_column * 34;
    float first_sum = 0.0f;
    float second_sum = 0.0f;

    if (first_column < dims.columns) {
        const device uchar* first_bytes = right + first_column * column_stride;
        const device uchar* second_bytes = second_column < dims.columns
            ? right + second_column * column_stride
            : first_bytes;
        for (uint block = 0; block < blocks_per_column; ++block) {
            const float activation = left[block * 32 + uint(lane)];
            const uint block_offset = block * 34;
            const half first_scale = *reinterpret_cast<const device half*>(first_bytes + block_offset);
            const char first_quant = as_type<char>(first_bytes[block_offset + 2 + uint(lane)]);
            first_sum += activation * float(first_scale) * float(first_quant);
            if (second_column < dims.columns) {
                const half second_scale = *reinterpret_cast<const device half*>(second_bytes + block_offset);
                const char second_quant = as_type<char>(second_bytes[block_offset + 2 + uint(lane)]);
                second_sum += activation * float(second_scale) * float(second_quant);
            }
        }
    }

    first_sum = simd_sum(first_sum);
    second_sum = simd_sum(second_sum);
    if (lane == 0) {
        if (first_column < dims.columns) output[first_column] = first_sum;
        if (second_column < dims.columns) output[second_column] = second_sum;
    }
}

// Width-two decode keeps both activation rows live while each packed Q8 block
// is resident in the texture/cache hierarchy. The per-row traversal and SIMD
// reduction match the retained matvec; only the weight load is shared.
kernel void matmul_f32_q8_0_width_two_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first_column = threadgroup_index * 4 + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const uint blocks_per_column = q8_0_decode_block_count == 0
        ? dims.shared / 32
        : uint(q8_0_decode_block_count);
    const uint column_stride = blocks_per_column * 34;
    float first_sum_0 = 0.0f;
    float first_sum_1 = 0.0f;
    float second_sum_0 = 0.0f;
    float second_sum_1 = 0.0f;

    if (first_column < dims.columns) {
        const device uchar* first_bytes = right + first_column * column_stride;
        const device uchar* second_bytes = second_column < dims.columns
            ? right + second_column * column_stride
            : first_bytes;
        for (uint block = 0; block < blocks_per_column; ++block) {
            const uint activation_index = block * 32 + uint(lane);
            const float activation_0 = left[activation_index];
            const float activation_1 = left[dims.shared + activation_index];
            const uint block_offset = block * 34;
            const half first_scale = *reinterpret_cast<const device half*>(first_bytes + block_offset);
            const float first_weight = float(first_scale)
                * float(as_type<char>(first_bytes[block_offset + 2 + uint(lane)]));
            first_sum_0 += activation_0 * first_weight;
            first_sum_1 += activation_1 * first_weight;
            if (second_column < dims.columns) {
                const half second_scale = *reinterpret_cast<const device half*>(second_bytes + block_offset);
                const float second_weight = float(second_scale)
                    * float(as_type<char>(second_bytes[block_offset + 2 + uint(lane)]));
                second_sum_0 += activation_0 * second_weight;
                second_sum_1 += activation_1 * second_weight;
            }
        }
    }

    first_sum_0 = simd_sum(first_sum_0);
    first_sum_1 = simd_sum(first_sum_1);
    second_sum_0 = simd_sum(second_sum_0);
    second_sum_1 = simd_sum(second_sum_1);
    if (lane == 0) {
        if (first_column < dims.columns) {
            output[first_column] = first_sum_0;
            output[dims.columns + first_column] = first_sum_1;
        }
        if (second_column < dims.columns) {
            output[second_column] = second_sum_0;
            output[dims.columns + second_column] = second_sum_1;
        }
    }
}

// Each workgroup owns one target row and four vocabulary columns. All rows are
// dispatched together, so the verifier remains a single device-side full-V
// operation rather than a host-driven row loop. Keeping the inner expression,
// block traversal, and SIMD reduction literally identical to the Stage-4
// matvec is required for bitwise target-logit identity: sharing one decoded
// weight across several row accumulators lets the Metal compiler reassociate
// the products by a few ulps, which can change a boundary sampler decision.
// The output is still immediately compacted and reused, so a 1,025-row
// verifier never materializes rows-by-vocabulary.
kernel void matmul_f32_q8_0_target_rows_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint row = threadgroup_index.y;
    const uint first_column = threadgroup_index.x * 4 + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const uint blocks_per_column = q8_0_decode_block_count == 0
        ? dims.shared / 32 : uint(q8_0_decode_block_count);
    const uint column_stride = blocks_per_column * 34;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    if (row < dims.rows && first_column < dims.columns) {
        const device uchar* first_bytes = right + first_column * column_stride;
        const device uchar* second_bytes = second_column < dims.columns
            ? right + second_column * column_stride : first_bytes;
        for (uint block = 0; block < blocks_per_column; ++block) {
            const float activation = left[
                row * dims.shared + block * 32 + uint(lane)
            ];
            const uint block_offset = block * 34;
            const half first_scale =
                *reinterpret_cast<const device half*>(first_bytes + block_offset);
            const char first_quant =
                as_type<char>(first_bytes[block_offset + 2 + uint(lane)]);
            first_sum += activation * float(first_scale) * float(first_quant);
            if (second_column < dims.columns) {
                const half second_scale =
                    *reinterpret_cast<const device half*>(second_bytes + block_offset);
                const char second_quant =
                    as_type<char>(second_bytes[block_offset + 2 + uint(lane)]);
                second_sum += activation * float(second_scale) * float(second_quant);
            }
        }
    }

    first_sum = simd_sum(first_sum);
    second_sum = simd_sum(second_sum);
    if (lane == 0) {
        if (row < dims.rows && first_column < dims.columns) {
            output[row * dims.columns + first_column] = first_sum;
        }
        if (row < dims.rows && second_column < dims.columns) {
            output[row * dims.columns + second_column] = second_sum;
        }
    }
}

struct block_q4_k {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct block_q6_k {
    uchar low[128];
    uchar high[64];
    char scales[16];
    half d;
};

kernel void embed_q6_k_gguf_weight_token_ids_kernel(
    const device uint* token_ids [[buffer(0)]],
    const device block_q6_k* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= dims.hidden || gid.y >= dims.tokens) return;
    const uint token = token_ids[dims.token_offset + gid.y];
    if (token >= dims.vocab) { output[gid.y * dims.hidden + gid.x] = NAN; return; }
    const device block_q6_k& block = weight[token * (dims.hidden / 256) + gid.x / 256];
    const uint within = gid.x % 256;
    const uint half_block = within / 128;
    const uint quarter = (within % 128) / 32;
    const uint lane = within % 32;
    const uchar low = block.low[half_block * 64 + (quarter % 2) * 32 + lane];
    const uchar high = block.high[half_block * 32 + lane];
    const uint quant = ((uint(low) >> (quarter >= 2 ? 4 : 0)) & 15)
        | (((uint(high) >> (quarter * 2)) & 3) << 4);
    const float value = float(block.d) * float(block.scales[within / 16])
        * float(int(quant) - 32);
    output[gid.y * dims.hidden + gid.x] = value * dims.scale;
}

inline uchar2 q4_k_scale_min_pair(int group, int half_group, const device uchar* scales) {
    return group < 4
        ? uchar2(scales[group + half_group] & 63, scales[group + half_group + 4] & 63)
        : uchar2(
            (scales[group + half_group + 4] & 0x0f)
                | ((scales[group + half_group - 4] & 0xc0) >> 2),
            (scales[group + half_group + 4] >> 4)
                | ((scales[group + half_group] & 0xc0) >> 2)
        );
}

// Decode one consecutive 16-weight fragment. Keeping this layout identical to
// the native GGUF block lets the matrix path reuse resident weights directly.
template <typename matrix4x4>
inline void dequantize_q4_k_16(
    const device block_q4_k* block,
    short fragment_index,
    thread matrix4x4& values
) {
    const device uchar* quants = block->qs
        + (fragment_index / 4) * 32 + 16 * (fragment_index & 1);
    const short scale_group = (fragment_index / 4) * 2;
    fragment_index &= 3;
    const uchar2 scale_min = q4_k_scale_min_pair(
        scale_group, fragment_index / 2, block->scales
    );
    const float d = fragment_index < 2 ? float(block->d) : float(block->d) / 16.0f;
    const float scale = d * float(scale_min[0]);
    const float minimum = float(block->dmin) * float(scale_min[1]);
    const ushort mask = fragment_index < 2 ? 0x0f : 0xf0;
#pragma clang loop unroll(full)
    for (short item = 0; item < 16; ++item) {
        values[item / 4][item % 4] = scale * float(quants[item] & mask) - minimum;
    }
}

template <typename matrix4x4>
inline void dequantize_q6_k_16(
    const device block_q6_k* block,
    short fragment_index,
    thread matrix4x4& values
) {
    const device ushort* low = reinterpret_cast<const device ushort*>(block->low)
        + 32 * (fragment_index / 8) + 16 * ((fragment_index / 2) & 1)
        + 8 * (fragment_index & 1);
    const device ushort* high = reinterpret_cast<const device ushort*>(block->high)
        + 16 * (fragment_index / 8) + 8 * (fragment_index & 1);
    const float scale = float(block->scales[(fragment_index & 1) + 2 * (fragment_index / 2)]);
    fragment_index = (fragment_index / 2) & 3;
    const uint mask_high = fragment_index > 1
        ? (fragment_index > 2 ? 0xc0c0c0c0 : 0x30303030)
        : (fragment_index > 0 ? 0x0c0c0c0c : 0x03030303);
    const uint mask_low = fragment_index > 1 ? 0xf0f0f0f0 : 0x0f0f0f0f;
    const uint shift_high_right = fragment_index > 2 ? 2 : 0;
    const uint shift_high_left = fragment_index > 1 ? 0 : (fragment_index > 0 ? 2 : 4);
    const uint shift_low_right = fragment_index > 1 ? 4 : 0;
    const float factor = float(block->d) * scale;
    const float minimum = factor * 32.0f;
#pragma clang loop unroll(full)
    for (short item = 0; item < 4; ++item) {
        const uint packed_low = (uint(low[2 * item]) | (uint(low[2 * item + 1]) << 16)) & mask_low;
        const uint packed_high = (uint(high[2 * item]) | (uint(high[2 * item + 1]) << 16)) & mask_high;
        const uint quants = ((packed_high << shift_high_left) >> shift_high_right)
            | (packed_low >> shift_low_right);
        values[item][0] = factor * float(quants & 0xff) - minimum;
        values[item][1] = (factor / 256.0f) * float(quants & 0xff00) - minimum;
        values[item][2] = (factor / 65536.0f) * float(quants & 0xff0000) - minimum;
        values[item][3] = (factor / 16777216.0f) * float(quants & 0xff000000) - minimum;
    }
}

)metal";
constexpr const char* metal_kernel_source_projections = R"metal(
// Q4 shares packed quants and streamed activations across two rows while
// preserving the ordinary kernel's worker assignment and arithmetic order.
template <ushort rows_per_tile, bool columns8 = false, bool prepared_sums = false>
inline void gemma3_q4_stream_packed_rows(
    const device float* left,
    const device uchar* right,
    device float* output,
    constant MatmulDimensions& dims,
    uint2 group,
    ushort simd_group,
    ushort lane,
    threadgroup float* worker_sums = nullptr,
    const device float* input_sums = nullptr
) {
    // Interleaved weights put the four adjacent column pairs in adjacent
    // lanes. SIMD groups retain the four original block-worker partitions.
    const uint first_column = group.x * 8
        + (columns8 ? uint(lane & 3) * 2 : uint(simd_group) * 2);
    const uint first_row = group.y * uint(rows_per_tile);
    const bool valid = first_column < dims.columns && first_row < dims.rows;
    if constexpr (!columns8) {
        if (!valid) return;
    }
    const uint second_column = min(first_column + 1U, dims.columns - 1U);
    const ushort block_worker = columns8 ? simd_group : lane / 8;
    const ushort worker_lane = columns8 ? lane / 4 : lane % 8;
    const ushort quant_half = worker_lane / 4;
    const ushort pair_index = worker_lane % 4;
    const uint blocks_per_column = dims.shared / 256;
    const uint column_stride = blocks_per_column * 144;
    float sums[rows_per_tile][2];
#pragma clang loop unroll(full)
    for (ushort row = 0; row < rows_per_tile; ++row) {
        sums[row][0] = 0.0f;
        sums[row][1] = 0.0f;
    }
    for (uint block = uint(block_worker); block < blocks_per_column && valid; block += 4) {
        const device uchar* tile = right
            + ((first_column / 8) * blocks_per_column + block) * 1184;
        const device uchar* blocks[2] = {
            right + first_column * column_stride + block * 144,
            right + second_column * column_stride + block * 144
        };
        ushort4 packed_scales[2];
        float2 factors[2];
#pragma clang loop unroll(full)
        for (ushort column = 0; column < 2; ++column) {
            if constexpr (columns8) {
                const uint n = (first_column & 7) + uint(column);
                const device ushort* pairs = reinterpret_cast<const device ushort*>(tile + 1024) + n;
                const ushort p0 = pairs[(2 * quant_half) * 8];
                const ushort p1 = pairs[(2 * quant_half + 1) * 8];
                const ushort p2 = pairs[(2 * quant_half + 4) * 8];
                const ushort p3 = pairs[(2 * quant_half + 5) * 8];
                packed_scales[column] = ushort4(
                    (p0 & 255) | ((p1 & 255) << 8),
                    (p0 >> 8) | (p1 & 0xff00),
                    (p2 & 255) | ((p3 & 255) << 8),
                    (p2 >> 8) | (p3 & 0xff00));
                factors[column] = float2(reinterpret_cast<const device half2*>(tile + 1152)[n]);
            } else {
            const device ushort* scale_words =
                reinterpret_cast<const device ushort*>(blocks[column] + 4) + quant_half;
            packed_scales[column] = ushort4(
                scale_words[0] & 0x3f3f,
                scale_words[2] & 0x3f3f,
                ((scale_words[4] >> 0) & 0x0f0f) | ((scale_words[0] & 0xc0c0) >> 2),
                ((scale_words[4] >> 4) & 0x0f0f) | ((scale_words[2] & 0xc0c0) >> 2)
            );
            factors[column] = float2(*reinterpret_cast<const device half2*>(blocks[column]));
            }
        }
        float4 low_accumulator[rows_per_tile][2];
        float4 high_accumulator[rows_per_tile][2];
        float4 activation_sums[rows_per_tile];
#pragma clang loop unroll(full)
        for (ushort row = 0; row < rows_per_tile; ++row) {
            if constexpr (!prepared_sums) activation_sums[row] = float4(0.0f);
#pragma clang loop unroll(full)
            for (ushort column = 0; column < 2; ++column) {
                low_accumulator[row][column] = float4(0.0f);
                high_accumulator[row][column] = float4(0.0f);
            }
        }
#pragma clang loop unroll(full)
        for (ushort item = 0; item < 4; ++item) {
            float4 low_quants[2];
            float4 high_quants[2];
            uint pair_low = 0, pair_high = 0;
            if constexpr (columns8) {
                const uint index = (16 * uint(quant_half) + 4 * uint(pair_index) + uint(item)) * 8
                    + (first_column & 7);
                pair_low = *reinterpret_cast<const device uint*>(tile + index * 2);
                pair_high = *reinterpret_cast<const device uint*>(tile + (index + 32 * 8) * 2);
            }
#pragma clang loop unroll(full)
            for (ushort column = 0; column < 2; ++column) {
                const device ushort* quants =
                    reinterpret_cast<const device ushort*>(blocks[column] + 16)
                    + 16 * quant_half + 4 * pair_index;
                const ushort lo = columns8 ? ushort(pair_low >> (16 * column)) : quants[item];
                const ushort hi = columns8 ? ushort(pair_high >> (16 * column)) : quants[32 + item];
                low_quants[column] = float4(lo & 0x000f, lo & 0x0f00, lo & 0x00f0, lo & 0xf000);
                high_quants[column] = float4(hi & 0x000f, hi & 0x0f00, hi & 0x00f0, hi & 0xf000);
            }
#pragma clang loop unroll(full)
            for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
                const uint row = min(first_row + uint(row_offset), dims.rows - 1U);
                const device float* activation = left + row * dims.shared + block * 256
                    + 64 * uint(quant_half) + 8 * uint(pair_index) + 2 * uint(item);
                const float2 a0 = *reinterpret_cast<const device float2*>(activation);
                const float2 a1 = *reinterpret_cast<const device float2*>(activation + 32);
                const float2 a2 = *reinterpret_cast<const device float2*>(activation + 128);
                const float2 a3 = *reinterpret_cast<const device float2*>(activation + 160);
                const float4 low_values = float4(a0, a1);
                const float4 high_values = float4(a2, a3);
                // Preserve each old x-sum's sequential x[0], x[1], ... order.
                if constexpr (!prepared_sums) {
                    activation_sums[row_offset] += float4(a0.x, a1.x, a2.x, a3.x);
                    activation_sums[row_offset] += float4(a0.y, a1.y, a2.y, a3.y);
                }
#pragma clang loop unroll(full)
                for (ushort column = 0; column < 2; ++column) {
                    low_accumulator[row_offset][column] += low_values * low_quants[column];
                    high_accumulator[row_offset][column] += high_values * high_quants[column];
                }
            }
        }
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column = 0; column < 2; ++column) {
                const ushort4 packed = packed_scales[column];
                const float4 scale = float4(packed.x & 255, packed.x >> 8, packed.z & 255, packed.z >> 8);
                const float4 minimum = float4(packed.y & 255, packed.y >> 8, packed.w & 255, packed.w >> 8);
                const float4 lo = low_accumulator[row_offset][column];
                const float4 hi = high_accumulator[row_offset][column];
                float4 xs;
                if constexpr (prepared_sums) {
                    const uint row = min(first_row + uint(row_offset), dims.rows - 1U);
                    const uint index = ((row * blocks_per_column + block) * 8
                        + uint(worker_lane)) * 4;
                    xs = float4(input_sums[index], input_sums[index + 1],
                        input_sums[index + 2], input_sums[index + 3]);
                } else {
                    xs = activation_sums[row_offset];
                }
                sums[row_offset][column] += factors[column].x * (
                    (lo.x + lo.y / 256.0f) * scale.x
                    + (lo.z + lo.w / 256.0f) * scale.y / 16.0f
                    + (hi.x + hi.y / 256.0f) * scale.z
                    + (hi.z + hi.w / 256.0f) * scale.w / 16.0f
                ) - factors[column].y * (
                    xs.x * minimum.x + xs.y * minimum.y
                    + xs.z * minimum.z + xs.w * minimum.w
                );
            }
        }
    }
    if constexpr (columns8) {
        // Restore the original 32-lane ownership only for the final reduction.
        // Even padded columns publish zero partials and reach this barrier.
        const uint local_column = first_column - group.x * 8;
        const uint logical_worker = uint(block_worker) * 8 + uint(worker_lane);
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column = 0; column < 2; ++column) {
                worker_sums[(uint(row_offset) * 8 + local_column + uint(column)) * 32
                    + logical_worker] = sums[row_offset][column];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column = 0; column < 2; ++column) {
                const uint reduced_column = uint(simd_group) * 2 + uint(column);
                const float sum = simd_sum(worker_sums[
                    (uint(row_offset) * 8 + reduced_column) * 32 + uint(lane)]);
                const uint output_column = group.x * 8 + reduced_column;
                if (lane == 0 && first_row + uint(row_offset) < dims.rows
                    && output_column < dims.columns) {
                    output[(first_row + uint(row_offset)) * dims.columns + output_column] = sum;
                }
            }
        }
    } else {
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column = 0; column < 2; ++column) {
                const float sum = simd_sum(sums[row_offset][column]);
                if (lane == 0 && first_row + uint(row_offset) < dims.rows
                    && first_column + uint(column) < dims.columns) {
                    output[(first_row + uint(row_offset)) * dims.columns
                        + first_column + uint(column)] = sum;
                }
            }
        }
    }
}

// Q6 blocks have a 210-byte stride, so load through ushort rather than
// assuming four-byte block alignment. Scaling follows each 16-value dot.
inline float4 gemma3_q6_integer_4(
    const device block_q6_k* block,
    short fragment_index,
    short item
) {
    const device ushort* low = reinterpret_cast<const device ushort*>(block->low)
        + 32 * (fragment_index / 8) + 16 * ((fragment_index / 2) & 1)
        + 8 * (fragment_index & 1);
    const device ushort* high = reinterpret_cast<const device ushort*>(block->high)
        + 16 * (fragment_index / 8) + 8 * (fragment_index & 1);
    const short quarter = (fragment_index / 2) & 3;
    const uint mask_high = quarter > 1
        ? (quarter > 2 ? 0xc0c0c0c0 : 0x30303030)
        : (quarter > 0 ? 0x0c0c0c0c : 0x03030303);
    const uint mask_low = quarter > 1 ? 0xf0f0f0f0 : 0x0f0f0f0f;
    const uint shift_high_right = quarter > 2 ? 2 : 0;
    const uint shift_high_left = quarter > 1 ? 0 : (quarter > 0 ? 2 : 4);
    const uint shift_low_right = quarter > 1 ? 4 : 0;
    const uint packed_low = (uint(low[2 * item]) | (uint(low[2 * item + 1]) << 16)) & mask_low;
    const uint packed_high = (uint(high[2 * item]) | (uint(high[2 * item + 1]) << 16)) & mask_high;
    const uint packed = ((packed_high << shift_high_left) >> shift_high_right)
        | (packed_low >> shift_low_right);
    return float4(
        int(packed & 0xffU) - 32,
        int((packed >> 8) & 0xffU) - 32,
        int((packed >> 16) & 0xffU) - 32,
        int((packed >> 24) & 0xffU) - 32
    );
}

// Each 16-lane cluster owns two columns. Integer decoding is shared across
// rows; each activation float4 feeds both columns. Applying the native Q6 scale
// after its 16-value dot avoids expanding every weight. All row tiles use the
// same K partition and reduction, including ordinary decoding.
template <ushort rows_per_tile, bool columns8 = false>
inline void gemma3_q6_shared_rows(
    const device float* left,
    const device uchar* right,
    device float* output,
    constant MatmulDimensions& dims,
    uint2 group,
    ushort simd_group,
    ushort lane,
    threadgroup float* worker_sums = nullptr
) {
    constexpr ushort k_lanes = 16;
    const ushort k_lane = columns8 ? (simd_group & 1) * 8 + lane / 4 : lane % k_lanes;
    const uint first_column = columns8
        ? group.x * 16 + uint(simd_group / 2) * 8 + uint(lane & 3) * 2
        : (group.x * 8 + uint(simd_group) * 2 + uint(lane / k_lanes)) * 2;
    const uint first_row = group.y * uint(rows_per_tile);
    const bool valid = first_column < dims.columns && first_row < dims.rows;
    if constexpr (!columns8) {
        if (!valid) return;
    }
    const uint second_column = min(first_column + 1, dims.columns - 1);
    const uint blocks_per_column = dims.shared / 256;
    const device block_q6_k* weights = reinterpret_cast<const device block_q6_k*>(right);
    float sums[rows_per_tile][2] = {{0.0f, 0.0f}};
    for (uint scale_group = uint(k_lane); scale_group < dims.shared / 16 && valid;
         scale_group += uint(k_lanes)) {
        const uint inner = scale_group * 16;
        const uint block_index = inner / 256;
        const short fragment_index = short((inner % 256) / 16);
        const device block_q6_k* first_block = weights + first_column * blocks_per_column + block_index;
        const device block_q6_k* second_block = weights + second_column * blocks_per_column + block_index;
        const device uchar* tile = right
            + ((first_column / 8) * blocks_per_column + block_index) * 1680;
        const uint packed_column = first_column & 7;
        float scale[2];
        if constexpr (columns8) {
            const half2 d = *reinterpret_cast<const device half2*>(tile + 1664 + packed_column * 2);
            const char2 s = *reinterpret_cast<const device char2*>(tile + 1536
                + uint(fragment_index) * 8 + packed_column);
            scale[0] = float(d.x) * float(s.x);
            scale[1] = float(d.y) * float(s.y);
        } else {
            scale[0] = float(first_block->d) * float(first_block->scales[fragment_index]);
            scale[1] = float(second_block->d) * float(second_block->scales[fragment_index]);
        }
        float partial[rows_per_tile][2] = {{0.0f, 0.0f}};
#pragma clang loop unroll(full)
        for (short item = 0; item < 4; ++item) {
            float4 integer0, integer1;
            if constexpr (columns8) {
                const uint index = (uint(fragment_index) * 4 + uint(item)) * 8 + packed_column;
                const uint low = *reinterpret_cast<const device uint*>(tile + index * 2);
                const ushort high = *reinterpret_cast<const device ushort*>(tile + 1024 + index);
                integer0 = float4(int(low & 15) + int((high & 3) << 4) - 32,
                    int((low >> 4) & 15) + int(((high >> 2) & 3) << 4) - 32,
                    int((low >> 8) & 15) + int(((high >> 4) & 3) << 4) - 32,
                    int((low >> 12) & 15) + int(((high >> 6) & 3) << 4) - 32);
                integer1 = float4(int((low >> 16) & 15) + int(((high >> 8) & 3) << 4) - 32,
                    int((low >> 20) & 15) + int(((high >> 10) & 3) << 4) - 32,
                    int((low >> 24) & 15) + int(((high >> 12) & 3) << 4) - 32,
                    int((low >> 28) & 15) + int(((high >> 14) & 3) << 4) - 32);
            } else {
                integer0 = gemma3_q6_integer_4(first_block, fragment_index, item);
                integer1 = gemma3_q6_integer_4(second_block, fragment_index, item);
            }
#pragma clang loop unroll(full)
            for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
                const uint row = min(first_row + uint(row_offset), dims.rows - 1);
                const float4 activation = *reinterpret_cast<const device float4*>(
                    left + row * dims.shared + inner + uint(item) * 4);
                partial[row_offset][0] += dot(integer0, activation);
                partial[row_offset][1] += dot(integer1, activation);
            }
        }
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column_offset = 0; column_offset < 2; ++column_offset) {
                sums[row_offset][column_offset] += scale[column_offset] * partial[row_offset][column_offset];
            }
        }
    }
    if constexpr (columns8) {
        // Each adjacent SIMD-group pair covers one eight-column weight tile.
        // A final transpose restores the retained 16-lane reduction order.
        // Missing eight-column tiles skip all reads, but never this barrier.
        const uint local_column = first_column - group.x * 16;
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column_offset = 0; column_offset < 2; ++column_offset) {
                worker_sums[(uint(row_offset) * 16 + local_column + uint(column_offset)) * 16
                    + uint(k_lane)] = sums[row_offset][column_offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort reduction_lane = lane % k_lanes;
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column_offset = 0; column_offset < 2; ++column_offset) {
                const uint reduced_column = (uint(simd_group) * 2 + uint(lane / k_lanes)) * 2
                    + uint(column_offset);
                float sum = worker_sums[(uint(row_offset) * 16 + reduced_column) * 16
                    + uint(reduction_lane)];
                sum += simd_shuffle_down(sum, 8);
                sum += simd_shuffle_down(sum, 4);
                sum += simd_shuffle_down(sum, 2);
                sum += simd_shuffle_down(sum, 1);
                const uint column = group.x * 16 + reduced_column;
                if (reduction_lane == 0 && first_row + uint(row_offset) < dims.rows
                    && column < dims.columns) {
                    output[(first_row + uint(row_offset)) * dims.columns + column] = sum;
                }
            }
        }
    } else {
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < rows_per_tile; ++row_offset) {
#pragma clang loop unroll(full)
            for (ushort column_offset = 0; column_offset < 2; ++column_offset) {
                float sum = sums[row_offset][column_offset];
                sum += simd_shuffle_down(sum, 8);
                sum += simd_shuffle_down(sum, 4);
                sum += simd_shuffle_down(sum, 2);
                sum += simd_shuffle_down(sum, 1);
                const uint column = first_column + uint(column_offset);
                if (k_lane == 0 && first_row + uint(row_offset) < dims.rows && column < dims.columns) {
                    output[(first_row + uint(row_offset)) * dims.columns + column] = sum;
                }
            }
        }
    }
}

kernel void gemma3_matmul_f32_kquant_shared_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    if (kquant_tiled_dtype == 0) {
        gemma3_q4_stream_packed_rows<2>(left, right, output, dims, group, simd_group, lane);
    } else if (gemma3_shared_row_count == 1) {
        gemma3_q6_shared_rows<1>(left, right, output, dims, group, simd_group, lane);
    } else {
        gemma3_q6_shared_rows<2>(left, right, output, dims, group, simd_group, lane);
    }
}

// Columns8 coalesces adjacent output columns within each SIMD group, retaining
// each logical worker's K partition. One final threadgroup transpose restores
// the original float reduction; no decoded weight tile or per-block barrier is
// needed. Two rows each need 256 partials for either Q4 or Q6.
kernel void gemma3_matmul_f32_columns8_prepared_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    const device float* input_sums [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float worker_sums[512];
    if (gemma3_shared_row_count == 1)
        gemma3_q4_stream_packed_rows<1, true, true>(left, right, output, dims,
            group, simd_group, lane, worker_sums, input_sums);
    else
        gemma3_q4_stream_packed_rows<2, true, true>(left, right, output, dims,
            group, simd_group, lane, worker_sums, input_sums);
}

kernel void gemma3_matmul_f32_columns8_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float worker_sums[512];
    if (kquant_tiled_dtype == 0) {
        if (gemma3_shared_row_count == 1)
            gemma3_q4_stream_packed_rows<1, true>(left, right, output, dims, group, simd_group, lane, worker_sums);
        else
            gemma3_q4_stream_packed_rows<2, true>(left, right, output, dims, group, simd_group, lane, worker_sums);
    } else {
        if (gemma3_shared_row_count == 1)
            gemma3_q6_shared_rows<1, true>(left, right, output, dims, group, simd_group, lane, worker_sums);
        else
            gemma3_q6_shared_rows<2, true>(left, right, output, dims, group, simd_group, lane, worker_sums);
    }
}

template <typename matrix4x4>
inline void dequantize_columns8_16(
    const device uchar* right, uint column, uint blocks_per_column,
    uint absolute_fragment, thread matrix4x4& values
) {
    const uint fragment_index = absolute_fragment & 15;
    const uint n = column & 7;
    const device uchar* tile = right + ((column / 8) * blocks_per_column
        + absolute_fragment / 16) * (kquant_tiled_dtype == 0 ? 1184 : 1680);
    if (kquant_tiled_dtype == 0) {
        const half2 factors = reinterpret_cast<const device half2*>(tile + 1152)[n];
        const ushort pair = reinterpret_cast<const device ushort*>(tile + 1024)[(fragment_index / 2) * 8 + n];
        const bool high = (fragment_index & 3) >= 2;
        const float d = high ? float(factors.x) / 16.0f : float(factors.x);
        const float scale = d * float(pair & 255);
        const float minimum = float(factors.y) * float(pair >> 8);
        for (ushort item = 0; item < 16; ++item) {
            const uint word_index = (fragment_index / 4) * 16 + (fragment_index & 1) * 8 + uint(item) / 2;
            const ushort packed = reinterpret_cast<const device ushort*>(tile)[word_index * 8 + n];
            const uint byte = (uint(packed) >> ((item & 1) * 8)) & 255;
            values[item / 4][item % 4] = scale * float(byte & (high ? 240 : 15)) - minimum;
        }
    } else {
        const float factor = float(reinterpret_cast<const device half*>(tile + 1664)[n])
            * float(reinterpret_cast<const device char*>(tile + 1536)[fragment_index * 8 + n]);
        const float minimum = factor * 32.0f;
        for (ushort item = 0; item < 16; ++item) {
            const uint index = (fragment_index * 4 + uint(item) / 4) * 8 + n;
            const ushort low = reinterpret_cast<const device ushort*>(tile)[index];
            const uchar high = tile[1024 + index];
            const uint shift = item & 3;
            const uint quant = ((uint(low) >> (4 * shift)) & 15)
                | (((uint(high) >> (2 * shift)) & 3) << 4);
            values[item / 4][item % 4] = factor * float(quant) - minimum;
        }
    }
}

// Prompt rows share decoded weights while each 8x8 accumulator retains the
// same K8 sequence and F32 operands as the M8 tile. Decode/verifier rows use
// the canonical packed arithmetic above, independently of these prompt tiles.
template <uint rows_per_tile>
kernel void gemma3_matmul_f32_kquant_matrix_impl(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint row_groups = rows_per_tile / 8;
    constexpr uint columns_per_tile = 32;
    constexpr uint inner_per_tile = 64;
    threadgroup float scratch[(columns_per_tile + rows_per_tile) * inner_per_tile];
    threadgroup float* weights = scratch;
    threadgroup float* activations = scratch + columns_per_tile * inner_per_tile;
    const uint first_row = group.y * rows_per_tile;
    const uint first_column = group.x * columns_per_tile;
    const uint valid_rows = min(rows_per_tile, dims.rows - first_row);
    const uint valid_columns = min(columns_per_tile, dims.columns - first_column);
    const uint blocks_per_column = dims.shared / 256;
    simdgroup_float8x8 accumulators[row_groups];
#pragma clang loop unroll(full)
    for (ushort m = 0; m < row_groups; ++m)
        accumulators[m] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint inner = 0; inner < dims.shared; inner += inner_per_tile) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint column = uint(tid) / 4;
        const uint weight_fragment = uint(tid) & 3;
        const uint absolute_fragment = inner / 16 + weight_fragment;
        float4x4 decoded(0.0f);
        if (column < valid_columns) {
            if (kquant_columns8) {
                dequantize_columns8_16(right, first_column + column,
                    blocks_per_column, absolute_fragment, decoded);
            } else if (kquant_tiled_dtype == 0) {
                const device block_q4_k* blocks = reinterpret_cast<const device block_q4_k*>(right);
                dequantize_q4_k_16(blocks + (first_column + column) * blocks_per_column
                    + absolute_fragment / 16, short(absolute_fragment & 15), decoded);
            } else {
                const device block_q6_k* blocks = reinterpret_cast<const device block_q6_k*>(right);
                dequantize_q6_k_16(blocks + (first_column + column) * blocks_per_column
                    + absolute_fragment / 16, short(absolute_fragment & 15), decoded);
            }
        }
#pragma clang loop unroll(full)
        for (ushort item = 0; item < 16; ++item) {
            const uint k = weight_fragment * 16 + uint(item);
            weights[((k / 8) * 4 + column / 8) * 64
                + (k & 7) * 8 + (column & 7)] = decoded[item / 4][item % 4];
        }
#pragma clang loop unroll(full)
        for (ushort index = 0; index < rows_per_tile / 2; ++index) {
            const uint flat = uint(tid) + uint(index) * 128;
            const uint row = flat / inner_per_tile;
            const uint k = flat % inner_per_tile;
            activations[((k / 8) * row_groups + row / 8) * 64 + (row & 7) * 8 + (k & 7)] = row < valid_rows
                ? left[(first_row + row) * dims.shared + inner + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (ushort k = 0; k < 8; ++k) {
            simdgroup_float8x8 b;
            simdgroup_load(b, weights + (uint(k) * 4 + uint(simd_group)) * 64, 8, 0, false);
#pragma clang loop unroll(full)
            for (ushort m = 0; m < row_groups; ++m) {
                simdgroup_float8x8 a;
                simdgroup_load(a, activations + (uint(k) * row_groups + m) * 64, 8, 0, false);
                simdgroup_multiply_accumulate(accumulators[m], a, b, accumulators[m]);
            }
        }
    }
    if (valid_rows == rows_per_tile && valid_columns == columns_per_tile) {
#pragma clang loop unroll(full)
        for (ushort m = 0; m < row_groups; ++m)
            simdgroup_store(accumulators[m], output + (first_row + uint(m) * 8) * dims.columns
                + first_column + uint(simd_group) * 8, dims.columns, 0, false);
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort m = 0; m < row_groups; ++m)
        simdgroup_store(accumulators[m], scratch + uint(m) * 8 * columns_per_tile
            + uint(simd_group) * 8, columns_per_tile, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (ushort index = 0; index < rows_per_tile / 4; ++index) {
        const uint flat = uint(tid) + uint(index) * 128;
        const uint row = flat / columns_per_tile;
        const uint column = flat % columns_per_tile;
        if (row < valid_rows && column < valid_columns) {
            output[(first_row + row) * dims.columns + first_column + column] = scratch[flat];
        }
    }
}


template [[host_name("gemma3_matmul_f32_kquant_matrix_kernel")]]
kernel void gemma3_matmul_f32_kquant_matrix_impl<8>(
    const device float*, const device uchar*, device float*,
    constant MatmulDimensions&, uint2, ushort, ushort);

template [[host_name("gemma3_matmul_f32_kquant_matrix_m32_kernel")]]
kernel void gemma3_matmul_f32_kquant_matrix_impl<32>(
    const device float*, const device uchar*, device float*,
    constant MatmulDimensions&, uint2, ushort, ushort);

// A threadgroup forms a 64-column by 32-row output tile. Packed weights are
// dequantized once into an on-chip half tile and reused by four SIMD groups;
// F32 activations are rounded only when staged and all matrix accumulators are
// F32. The 8 KiB scratch also serves boundary stores for the public tail path.
kernel void matmul_f32_kquant_tiled_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    constexpr short output_tile = 64;
    const short row_tile = kquant_tiled_exact_16 ? 16 : 32;
    constexpr short inner_tile = 32;
    threadgroup uchar scratch[8192];
    threadgroup half* weights = reinterpret_cast<threadgroup half*>(scratch);
    threadgroup half* activations = weights + output_tile * inner_tile;

    const uint first_column = threadgroup_index.x * output_tile;
    const uint first_row = threadgroup_index.y * row_tile;
    const short valid_columns = short(min(uint(output_tile), dims.columns - first_column));
    const short valid_rows = short(min(uint(row_tile), dims.rows - first_row));
    const short activation_row = min(short(thread_index / 4), short(valid_rows - 1));
    const short activation_inner = 8 * (thread_index & 3);
    const uint blocks_per_column = dims.shared / 256;

    simdgroup_half8x8 weight_matrices[4];
    simdgroup_half8x8 activation_matrices[2];
    simdgroup_float8x8 accumulators[8];
#pragma clang loop unroll(full)
    for (short index = 0; index < 8; ++index) {
        accumulators[index] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint inner = 0; inner < dims.shared; inner += inner_tile) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const short loader_count = kquant_tiled_exact_16 ? 64 : 128;
        for (short load_index = thread_index; load_index < 128; load_index += loader_count) {
            const short weight_column = min(short(load_index / 2), short(valid_columns - 1));
            const short weight_fragment = load_index & 1;
            const uint fragment_index = inner / 16 + uint(weight_fragment);
            half4x4 decoded;
            if (kquant_columns8) {
                dequantize_columns8_16(right, first_column + uint(weight_column),
                    blocks_per_column, fragment_index, decoded);
            } else if (kquant_tiled_dtype == 0) {
                const device block_q4_k* blocks = reinterpret_cast<const device block_q4_k*>(right);
                const device block_q4_k* block = blocks
                    + (first_column + uint(weight_column)) * blocks_per_column
                    + fragment_index / 16;
                dequantize_q4_k_16(block, short(fragment_index & 15), decoded);
            } else {
                const device block_q6_k* blocks = reinterpret_cast<const device block_q6_k*>(right);
                const device block_q6_k* block = blocks
                    + (first_column + uint(weight_column)) * blocks_per_column
                    + fragment_index / 16;
                dequantize_q6_k_16(block, short(fragment_index & 15), decoded);
            }
#pragma clang loop unroll(full)
            for (short item = 0; item < 16; ++item) {
                const short sx = 2 * weight_fragment + item / 8;
                const short sy = (load_index / 2) / 8;
                const short lx = (load_index / 2) & 7;
                const short ly = item & 7;
                weights[64 * (8 * sx + sy) + 8 * ly + lx] = decoded[item / 4][item % 4];
            }
        }

        const device float* activation = left
            + (first_row + uint(activation_row)) * dims.shared + inner + activation_inner;
        const short sx = thread_index & 3;
        const short sy = (thread_index / 4) / 8;
        const short ly = (thread_index / 4) & 7;
        const short activation_block = 4 * sx + sy;
#pragma clang loop unroll(full)
        for (short item = 0; item < 8; ++item) {
            activations[64 * activation_block + 8 * ly + item] = half(activation[item]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* weight_tile = weights + 4 * 64 * (simdgroup_index & 1);
        threadgroup const half* activation_tile = activations
            + 2 * 64 * (kquant_tiled_exact_16 ? 0 : (simdgroup_index >> 1));
#pragma clang loop unroll(full)
        for (short inner_matrix = 0; inner_matrix < 4; ++inner_matrix) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
            for (short index = 0; index < 4; ++index) {
                simdgroup_load(weight_matrices[index], weight_tile + 64 * index, 8, 0, false);
            }
#pragma clang loop unroll(full)
            for (short index = 0; index < 2; ++index) {
                simdgroup_load(activation_matrices[index], activation_tile + 64 * index, 8, 0, false);
            }
#pragma clang loop unroll(full)
            for (short index = 0; index < 8; ++index) {
                simdgroup_multiply_accumulate(
                    accumulators[index],
                    activation_matrices[index / 4],
                    weight_matrices[index % 4],
                    accumulators[index]
                );
            }
            weight_tile += 8 * 64;
            activation_tile += 4 * 64;
        }
    }

    const bool complete_tile = valid_columns == output_tile && valid_rows == row_tile;
    if (complete_tile) {
        device float* destination = output + first_row * dims.columns + first_column
            + 32 * (simdgroup_index & 1)
            + 16 * (kquant_tiled_exact_16 ? 0 : (simdgroup_index >> 1)) * dims.columns;
#pragma clang loop unroll(full)
        for (short index = 0; index < 8; ++index) {
            simdgroup_store(
                accumulators[index],
                destination + 8 * (index % 4) + 8 * dims.columns * (index / 4),
                dims.columns,
                0,
                false
            );
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* tile = reinterpret_cast<threadgroup float*>(scratch)
        + 32 * (simdgroup_index & 1) + 16 * (simdgroup_index >> 1) * output_tile;
#pragma clang loop unroll(full)
    for (short index = 0; index < 8; ++index) {
        simdgroup_store(
            accumulators[index],
            tile + 8 * (index % 4) + 8 * output_tile * (index / 4),
            output_tile,
            0,
            false
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index == 0) {
        for (short row = thread_index; row < valid_rows; row += row_tile) {
            for (short column = 0; column < valid_columns; ++column) {
                output[(first_row + uint(row)) * dims.columns + first_column + uint(column)] =
                    reinterpret_cast<threadgroup float*>(scratch)[row * output_tile + column];
            }
        }
    }
}

// Each threadgroup owns two activation rows for a four-column output tile.
// Both rows traverse the same packed columns in one workgroup so those bytes
// remain cache-local. Every row keeps the retained scalar worker assignment,
// block traversal, arithmetic order, and SIMD reduction.
kernel void matmul_f32_q4_k_packed_row_batch_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const ushort row_tile = packed_parallel_row_count == 1 ? 2 : 1;
    const uint first_row = threadgroup_index.y * uint(row_tile) * uint(packed_parallel_row_count)
        + uint(simdgroup_index / 2);
    const uint first_column = (threadgroup_index.x * 4) + uint(simdgroup_index % 2) * 2;
    const uint second_column = first_column + 1;
    const ushort block_worker = lane / 8;
    const ushort worker_lane = lane % 8;
    const ushort quant_half = worker_lane / 4;
    const ushort pair_index = worker_lane % 4;
    const uint blocks_per_column = q4_k_decode_block_count == 0
        ? dims.shared / 256
        : uint(q4_k_decode_block_count);
    const uint column_stride = blocks_per_column * 144;
    const uint adjacent_stride = second_column < dims.columns ? column_stride : 0;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (first_row < dims.rows && first_column < dims.columns) {
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < row_tile; ++row_offset) {
            const uint row = first_row + uint(row_offset);
            if (row >= dims.rows) continue;
            const device uchar* first_column_bytes = right + first_column * column_stride;
            const device float* activation = left + row * dims.shared
                + uint(block_worker) * 256 + 64 * uint(quant_half) + 8 * uint(pair_index);
            float activation_low[16];
            float activation_high[16];
            ushort packed_scales[4];
            thread const uchar* scales =
                reinterpret_cast<thread const uchar*>(packed_scales);
            for (uint block = uint(block_worker); block < blocks_per_column; block += 4) {
                float4 activation_sums = float4(0.0f);
#pragma clang loop unroll(full)
                for (ushort item = 0; item < 8; ++item) {
                    activation_low[item] = activation[item];
                    activation_sums[0] += activation_low[item];
                    activation_low[item + 8] = activation[item + 32];
                    activation_sums[1] += activation_low[item + 8];
                    activation_high[item] = activation[item + 128];
                    activation_sums[2] += activation_high[item];
                    activation_high[item + 8] = activation[item + 160];
                    activation_sums[3] += activation_high[item + 8];
                }
                const device uchar* block_bytes = first_column_bytes + block * 144;
                const device ushort* scale_words =
                    reinterpret_cast<const device ushort*>(block_bytes + 4) + quant_half;
                const device ushort* quants =
                    reinterpret_cast<const device ushort*>(block_bytes + 16)
                    + 16 * quant_half + 4 * pair_index;
                const device half* factors = reinterpret_cast<const device half*>(block_bytes);
#pragma clang loop unroll(full)
                for (ushort output_index = 0; output_index < 2; ++output_index) {
                    packed_scales[0] = scale_words[0] & 0x3f3f;
                    packed_scales[1] = scale_words[2] & 0x3f3f;
                    packed_scales[2] = ((scale_words[4] >> 0) & 0x0f0f)
                        | ((scale_words[0] & 0xc0c0) >> 2);
                    packed_scales[3] = ((scale_words[4] >> 4) & 0x0f0f)
                        | ((scale_words[2] & 0xc0c0) >> 2);
                    const device ushort* high_quants = quants + 32;
                    float4 low_accumulator = float4(0.0f);
                    float4 high_accumulator = float4(0.0f);
#pragma clang loop unroll(full)
                    for (ushort item = 0; item < 4; ++item) {
                        low_accumulator[0] += activation_low[2 * item]
                            * (quants[item] & 0x000f);
                        low_accumulator[1] += activation_low[2 * item + 1]
                            * (quants[item] & 0x0f00);
                        low_accumulator[2] += activation_low[2 * item + 8]
                            * (quants[item] & 0x00f0);
                        low_accumulator[3] += activation_low[2 * item + 9]
                            * (quants[item] & 0xf000);
                        high_accumulator[0] += activation_high[2 * item]
                            * (high_quants[item] & 0x000f);
                        high_accumulator[1] += activation_high[2 * item + 1]
                            * (high_quants[item] & 0x0f00);
                        high_accumulator[2] += activation_high[2 * item + 8]
                            * (high_quants[item] & 0x00f0);
                        high_accumulator[3] += activation_high[2 * item + 9]
                            * (high_quants[item] & 0xf000);
                    }
                    sums[2 * row_offset + output_index] += float(factors[0]) * (
                            (low_accumulator[0] + low_accumulator[1] / 256.0f) * scales[0]
                            + (low_accumulator[2] + low_accumulator[3] / 256.0f)
                                * scales[1] / 16.0f
                            + (high_accumulator[0] + high_accumulator[1] / 256.0f)
                                * scales[4]
                            + (high_accumulator[2] + high_accumulator[3] / 256.0f)
                                * scales[5] / 16.0f
                        ) - float(factors[1]) * (
                            activation_sums[0] * scales[2]
                            + activation_sums[1] * scales[3]
                            + activation_sums[2] * scales[6]
                            + activation_sums[3] * scales[7]
                        );
                    quants += adjacent_stride / 2;
                    scale_words += adjacent_stride / 2;
                    factors += adjacent_stride / 2;
                }
                activation += 4 * 256;
            }
        }
    }

#pragma clang loop unroll(full)
    for (ushort index = 0; index < row_tile * 2; ++index) {
        sums[index] = simd_sum(sums[index]);
    }
    if (lane == 0) {
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < row_tile; ++row_offset) {
            const uint row = first_row + uint(row_offset);
            if (row >= dims.rows) continue;
            if (first_column < dims.columns) {
                output[row * dims.columns + first_column] = sums[2 * row_offset];
            }
            if (second_column < dims.columns) {
                output[row * dims.columns + second_column] = sums[2 * row_offset + 1];
            }
        }
    }
}

// Numerically equivalent width-two form of the retained packed decode kernel.
// Each row preserves the exact worker assignment and accumulation order. The
// second scan immediately reuses the same column's packed blocks from cache;
// this trades some peak reuse for stable deep-network parity.
kernel void matmul_f32_q4_k_packed_width_two_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first_column = threadgroup_index * 4 + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const ushort block_worker = lane / 8;
    const ushort worker_lane = lane % 8;
    const ushort quant_half = worker_lane / 4;
    const ushort pair_index = worker_lane % 4;
    const uint blocks_per_column = q4_k_decode_block_count == 0
        ? dims.shared / 256 : uint(q4_k_decode_block_count);
    const uint column_stride = blocks_per_column * 144;
    const uint adjacent_stride = second_column < dims.columns ? column_stride : 0;
    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (first_column < dims.columns) {
        for (ushort row = 0; row < 2; ++row) {
            const device uchar* first_column_bytes = right + first_column * column_stride;
            const device float* activation = left + uint(row) * dims.shared
                + uint(block_worker) * 256 + 64 * uint(quant_half) + 8 * uint(pair_index);
            float activation_low[16];
            float activation_high[16];
            ushort packed_scales[4];
            thread const uchar* scales = reinterpret_cast<thread const uchar*>(packed_scales);
            for (uint block = uint(block_worker); block < blocks_per_column; block += 4) {
                float4 activation_sums = float4(0.0f);
#pragma clang loop unroll(full)
                for (ushort item = 0; item < 8; ++item) {
                    activation_low[item] = activation[item];
                    activation_sums[0] += activation_low[item];
                    activation_low[item + 8] = activation[item + 32];
                    activation_sums[1] += activation_low[item + 8];
                    activation_high[item] = activation[item + 128];
                    activation_sums[2] += activation_high[item];
                    activation_high[item + 8] = activation[item + 160];
                    activation_sums[3] += activation_high[item + 8];
                }
                const device uchar* block_bytes = first_column_bytes + block * 144;
                const device ushort* scale_words =
                    reinterpret_cast<const device ushort*>(block_bytes + 4) + quant_half;
                const device ushort* quants = reinterpret_cast<const device ushort*>(block_bytes + 16)
                    + 16 * quant_half + 4 * pair_index;
                const device half* factors = reinterpret_cast<const device half*>(block_bytes);
#pragma clang loop unroll(full)
                for (ushort output_index = 0; output_index < 2; ++output_index) {
                    packed_scales[0] = scale_words[0] & 0x3f3f;
                    packed_scales[1] = scale_words[2] & 0x3f3f;
                    packed_scales[2] = ((scale_words[4] >> 0) & 0x0f0f)
                        | ((scale_words[0] & 0xc0c0) >> 2);
                    packed_scales[3] = ((scale_words[4] >> 4) & 0x0f0f)
                        | ((scale_words[2] & 0xc0c0) >> 2);
                    const device ushort* high_quants = quants + 32;
                    float4 low_accumulator = float4(0.0f);
                    float4 high_accumulator = float4(0.0f);
#pragma clang loop unroll(full)
                    for (ushort item = 0; item < 4; ++item) {
                        low_accumulator[0] += activation_low[2 * item] * (quants[item] & 0x000f);
                        low_accumulator[1] += activation_low[2 * item + 1] * (quants[item] & 0x0f00);
                        low_accumulator[2] += activation_low[2 * item + 8] * (quants[item] & 0x00f0);
                        low_accumulator[3] += activation_low[2 * item + 9] * (quants[item] & 0xf000);
                        high_accumulator[0] += activation_high[2 * item] * (high_quants[item] & 0x000f);
                        high_accumulator[1] += activation_high[2 * item + 1] * (high_quants[item] & 0x0f00);
                        high_accumulator[2] += activation_high[2 * item + 8] * (high_quants[item] & 0x00f0);
                        high_accumulator[3] += activation_high[2 * item + 9] * (high_quants[item] & 0xf000);
                    }
                    row_sums[2 * row + output_index] += float(factors[0]) * (
                            (low_accumulator[0] + low_accumulator[1] / 256.0f) * scales[0]
                            + (low_accumulator[2] + low_accumulator[3] / 256.0f) * scales[1] / 16.0f
                            + (high_accumulator[0] + high_accumulator[1] / 256.0f) * scales[4]
                            + (high_accumulator[2] + high_accumulator[3] / 256.0f) * scales[5] / 16.0f
                        ) - float(factors[1]) * (
                            activation_sums[0] * scales[2] + activation_sums[1] * scales[3]
                            + activation_sums[2] * scales[6] + activation_sums[3] * scales[7]
                        );
                    quants += adjacent_stride / 2;
                    scale_words += adjacent_stride / 2;
                    factors += adjacent_stride / 2;
                }
                activation += 4 * 256;
            }
        }
    }
#pragma clang loop unroll(full)
    for (ushort index = 0; index < 4; ++index) row_sums[index] = simd_sum(row_sums[index]);
    if (lane == 0) {
        for (ushort row = 0; row < 2; ++row) {
            if (first_column < dims.columns) output[uint(row) * dims.columns + first_column] = row_sums[2 * row];
            if (second_column < dims.columns) output[uint(row) * dims.columns + second_column] = row_sums[2 * row + 1];
        }
    }
}

)metal";
constexpr const char* metal_kernel_source_tail_a = R"metal(

// Q6_K uses the same two-row cache-local tile and exact per-row reduction
// contract as Q4_K above.
kernel void matmul_f32_q6_k_packed_row_batch_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint2 threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const ushort row_tile = packed_parallel_row_count == 1 ? 2 : 1;
    const uint first_row = threadgroup_index.y * uint(row_tile) * uint(packed_parallel_row_count)
        + uint(simdgroup_index / 2);
    const uint first_column = (threadgroup_index.x * 4) + uint(simdgroup_index % 2) * 2;
    const uint second_column = first_column + 1;
    const ushort block_parity = lane % 2;
    const ushort block_lane = lane / 2;
    const ushort half_block = block_lane / 8;
    const ushort lane_in_half = block_lane % 8;
    const ushort lane_offset = 4 * lane_in_half;
    const ushort scale_offset = 8 * half_block + lane_offset / 16;
    const uint blocks_per_column = q6_k_decode_block_count == 0
        ? dims.shared / 256
        : uint(q6_k_decode_block_count);
    const uint column_stride = blocks_per_column * 210;
    const uint adjacent_stride = second_column < dims.columns ? column_stride : 0;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (first_row < dims.rows && first_column < dims.columns) {
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < row_tile; ++row_offset) {
            const uint row = first_row + uint(row_offset);
            if (row >= dims.rows) continue;
            const device uchar* first_column_bytes = right + first_column * column_stride;
            for (uint block = uint(block_parity); block < blocks_per_column; block += 2) {
                const device uchar* block_bytes = first_column_bytes + block * 210;
                const device uchar* low0 = block_bytes + 64 * half_block + lane_offset;
                const device uchar* low1 = low0 + 32;
                const device uchar* high = block_bytes + 128 + 32 * half_block + lane_offset;
                const device int8_t* scales =
                    reinterpret_cast<const device int8_t*>(block_bytes + 192) + scale_offset;
                const device half* factor =
                    reinterpret_cast<const device half*>(block_bytes + 208);
                const device float* activation = left + row * dims.shared
                    + block * 256 + 128 * half_block + lane_offset;
                float activation_values[16];
#pragma clang loop unroll(full)
                for (ushort item = 0; item < 4; ++item) {
                    activation_values[4 * item] = activation[item];
                    activation_values[4 * item + 1] = activation[item + 32];
                    activation_values[4 * item + 2] = activation[item + 64];
                    activation_values[4 * item + 3] = activation[item + 96];
                }
#pragma clang loop unroll(full)
                for (ushort output_index = 0; output_index < 2; ++output_index) {
                    float4 accumulator = float4(0.0f);
#pragma clang loop unroll(full)
                    for (ushort item = 0; item < 4; ++item) {
                        const uchar high_value = high[item];
                        accumulator[0] += activation_values[4 * item]
                            * (int((low0[item] & 0x0f)
                                | ((high_value & 0x03) << 4)) - 32);
                        accumulator[1] += activation_values[4 * item + 1]
                            * (int((low1[item] & 0x0f)
                                | ((high_value & 0x0c) << 2)) - 32);
                        accumulator[2] += activation_values[4 * item + 2]
                            * (int((low0[item] >> 4) | (high_value & 0x30)) - 32);
                        accumulator[3] += activation_values[4 * item + 3]
                            * (int((low1[item] >> 4)
                                | ((high_value & 0xc0) >> 2)) - 32);
                    }
                    sums[2 * row_offset + output_index] += float(factor[0]) * (
                        accumulator[0] * float(scales[0])
                        + accumulator[1] * float(scales[2])
                        + accumulator[2] * float(scales[4])
                        + accumulator[3] * float(scales[6])
                    );
                    low0 += adjacent_stride;
                    low1 += adjacent_stride;
                    high += adjacent_stride;
                    scales += adjacent_stride;
                    factor += adjacent_stride / 2;
                }
            }
        }
    }

#pragma clang loop unroll(full)
    for (ushort index = 0; index < row_tile * 2; ++index) {
        sums[index] = simd_sum(sums[index]);
    }
    if (lane == 0) {
#pragma clang loop unroll(full)
        for (ushort row_offset = 0; row_offset < row_tile; ++row_offset) {
            const uint row = first_row + uint(row_offset);
            if (row >= dims.rows) continue;
            if (first_column < dims.columns) {
                output[row * dims.columns + first_column] = sums[2 * row_offset];
            }
            if (second_column < dims.columns) {
                output[row * dims.columns + second_column] = sums[2 * row_offset + 1];
            }
        }
    }
}

kernel void matmul_f32_q6_k_packed_width_two_kernel(
    const device float* left [[buffer(0)]],
    const device uchar* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MatmulDimensions& dims [[buffer(3)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first_column = threadgroup_index * 4 + uint(simdgroup_index) * 2;
    const uint second_column = first_column + 1;
    const ushort block_parity = lane % 2;
    const ushort block_lane = lane / 2;
    const ushort half_block = block_lane / 8;
    const ushort lane_in_half = block_lane % 8;
    const ushort lane_offset = 4 * lane_in_half;
    const ushort scale_offset = 8 * half_block + lane_offset / 16;
    const uint blocks_per_column = q6_k_decode_block_count == 0
        ? dims.shared / 256 : uint(q6_k_decode_block_count);
    const uint column_stride = blocks_per_column * 210;
    const uint adjacent_stride = second_column < dims.columns ? column_stride : 0;
    float row_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (first_column < dims.columns) {
        for (ushort row = 0; row < 2; ++row) {
            const device uchar* first_column_bytes = right + first_column * column_stride;
            for (uint block = uint(block_parity); block < blocks_per_column; block += 2) {
                const device uchar* block_bytes = first_column_bytes + block * 210;
                const device uchar* low0 = block_bytes + 64 * half_block + lane_offset;
                const device uchar* low1 = low0 + 32;
                const device uchar* high = block_bytes + 128 + 32 * half_block + lane_offset;
                const device int8_t* scales =
                    reinterpret_cast<const device int8_t*>(block_bytes + 192) + scale_offset;
                const device half* factor = reinterpret_cast<const device half*>(block_bytes + 208);
                const device float* activation = left + uint(row) * dims.shared
                    + block * 256 + 128 * half_block + lane_offset;
                float activation_values[16];
#pragma clang loop unroll(full)
                for (ushort item = 0; item < 4; ++item) {
                    activation_values[4 * item] = activation[item];
                    activation_values[4 * item + 1] = activation[item + 32];
                    activation_values[4 * item + 2] = activation[item + 64];
                    activation_values[4 * item + 3] = activation[item + 96];
                }
#pragma clang loop unroll(full)
                for (ushort output_index = 0; output_index < 2; ++output_index) {
                    float4 accumulator = float4(0.0f);
#pragma clang loop unroll(full)
                    for (ushort item = 0; item < 4; ++item) {
                        const uchar high_value = high[item];
                        accumulator[0] += activation_values[4 * item]
                            * (int((low0[item] & 0x0f) | ((high_value & 0x03) << 4)) - 32);
                        accumulator[1] += activation_values[4 * item + 1]
                            * (int((low1[item] & 0x0f) | ((high_value & 0x0c) << 2)) - 32);
                        accumulator[2] += activation_values[4 * item + 2]
                            * (int((low0[item] >> 4) | (high_value & 0x30)) - 32);
                        accumulator[3] += activation_values[4 * item + 3]
                            * (int((low1[item] >> 4) | ((high_value & 0xc0) >> 2)) - 32);
                    }
                    row_sums[2 * row + output_index] += float(factor[0]) * (
                        accumulator[0] * float(scales[0]) + accumulator[1] * float(scales[2])
                        + accumulator[2] * float(scales[4]) + accumulator[3] * float(scales[6])
                    );
                    low0 += adjacent_stride;
                    low1 += adjacent_stride;
                    high += adjacent_stride;
                    scales += adjacent_stride;
                    factor += adjacent_stride / 2;
                }
            }
        }
    }
#pragma clang loop unroll(full)
    for (ushort index = 0; index < 4; ++index) row_sums[index] = simd_sum(row_sums[index]);
    if (lane == 0) {
        for (ushort row = 0; row < 2; ++row) {
            if (first_column < dims.columns) output[uint(row) * dims.columns + first_column] = row_sums[2 * row];
            if (second_column < dims.columns) output[uint(row) * dims.columns + second_column] = row_sums[2 * row + 1];
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

kernel void gelu_tanh_multiply_q4_sums_f32_kernel(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseDimensions& dims [[buffer(3)]],
    device float* q4_sums [[buffer(4)]],
    uint chunk [[thread_position_in_grid]]
) {
    if (chunk >= dims.elements / 8U) return;
    float4 first(0.0f), second(0.0f);
#pragma clang loop unroll(full)
    for (uint item = 0U; item < 8U; ++item) {
        const uint index = chunk * 8U + item;
        const float value = gate[index];
        constexpr float sqrt_two_over_pi = 0.7978845608028654f;
        constexpr float cubic_coefficient = 0.044715f;
        const float inner = sqrt_two_over_pi * (value + cubic_coefficient * value * value * value);
        const float activated = (0.5f * value * (1.0f + precise::tanh(inner))) * up[index];
        output[index] = activated;
        if (item < 4U) first[item] = activated;
        else second[item - 4U] = activated;
    }
    store_q4_input_sum(q4_sums, chunk, first, second);
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

struct HeadRMSNormDimensions {
    uint rows;
    uint head_count;
    uint head_dimension;
    float epsilon;
};

kernel void head_rms_norm_f32_kernel(
    const device float4* input [[buffer(0)]],
    const device float4* weight [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant HeadRMSNormDimensions& dims [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint first = (group.y * dims.head_count + group.x) * 64 + uint(lane) * 2;
    const float4 a = input[first];
    const float4 b = input[first + 1];
    const float mean_square = simd_sum(dot(a, a) + dot(b, b)) / 256.0f;
    const float scale = rsqrt(mean_square + dims.epsilon);
    output[first] = a * scale * weight[uint(lane) * 2];
    output[first + 1] = b * scale * weight[uint(lane) * 2 + 1];
}

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

struct Gemma2KVCacheCopyDimensions {
    uint rows;
    uint row_width;
    uint position_start;
    uint source_row_mask;
    uint destination_row_mask;
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

struct Gemma2WrappedPrefillAttentionScoreDimensions {
    uint query_rows;
    uint head_count;
    uint kv_head_count;
    uint key_length;
    uint attention_token_count;
    uint query_position_start;
    uint cache_row_mask;
    uint current_chunk_row_mask;
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

struct Gemma2WrappedPrefillAttentionValueDimensions {
    uint query_rows;
    uint head_count;
    uint kv_head_count;
    uint value_length;
    uint attention_token_count;
    uint query_position_start;
    uint cache_row_mask;
    uint current_chunk_row_mask;
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

template <bool emit_q4_sums>
kernel void rms_norm_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormDimensions& dims [[buffer(3)]],
    device float* q4_sums [[buffer(4)]],
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
    if constexpr (emit_q4_sums) {
        // Only output ownership changes; the input reduction above, including
        // its 256-thread tree, is identical to the original RMSNorm.
        const device float4* input4 = reinterpret_cast<const device float4*>(input + input_row_offset);
        const device float4* weight4 = reinterpret_cast<const device float4*>(weight);
        device float4* output4 = reinterpret_cast<device float4*>(output + output_row_offset);
        for (uint chunk = thread_index; chunk < dims.hidden / 8U; chunk += thread_count) {
            const float4 first = input4[2U * chunk] * inverse_rms * weight4[2U * chunk];
            const float4 second = input4[2U * chunk + 1U] * inverse_rms * weight4[2U * chunk + 1U];
            output4[2U * chunk] = first;
            output4[2U * chunk + 1U] = second;
            store_q4_input_sum(q4_sums, output_row_offset / 8U + chunk, first, second);
        }
    } else if ((dims.hidden & 3u) == 0u) {
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
template [[host_name("rms_norm_f32_kernel")]] kernel void rms_norm_f32_kernel<false>(
    const device float*, const device float*, device float*,
    constant RMSNormDimensions&, device float*, uint, uint, uint, uint, uint, uint
);

template [[host_name("rms_norm_q4_sums_f32_kernel")]] kernel void rms_norm_f32_kernel<true>(
    const device float*, const device float*, device float*,
    constant RMSNormDimensions&, device float*, uint, uint, uint, uint, uint, uint
);


kernel void rms_norm_add_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant RMSNormDimensions& dims [[buffer(4)]],
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
    float sum_squares = 0.0f;
    const uint vector_count = dims.hidden / 4u;
    const uint row_vector_offset = row * vector_count;
    const device float4* input4 = reinterpret_cast<const device float4*>(input)
        + row_vector_offset;
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
    const device float4* residual4 = reinterpret_cast<const device float4*>(residual)
        + row_vector_offset;
    device float4* output4 = reinterpret_cast<device float4*>(output) + row_vector_offset;
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

template <typename Cache>
kernel void kv_cache_copy_kernel(
    const device Cache* source_key [[buffer(0)]],
    const device Cache* source_value [[buffer(1)]],
    device Cache* destination_key [[buffer(2)]],
    device Cache* destination_value [[buffer(3)]],
    constant Gemma2KVCacheCopyDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= dims.rows || gid.x >= dims.row_width) return;
    const uint position = dims.position_start + gid.y;
    const uint source_index = (position & dims.source_row_mask) * dims.row_width + gid.x;
    const uint destination_index = (position & dims.destination_row_mask) * dims.row_width + gid.x;
    destination_key[destination_index] = source_key[source_index];
    destination_value[destination_index] = source_value[source_index];
}

template [[host_name("gemma2_kv_cache_copy_f16_kernel")]] kernel void kv_cache_copy_kernel<half>(
const device half*,
    const device half*,
    device half*,
    device half*,
    constant Gemma2KVCacheCopyDimensions&,
    uint2
);
template [[host_name("kv_cache_copy_f32_kernel")]] kernel void kv_cache_copy_kernel<float>(
const device float*,
    const device float*,
    device float*,
    device float*,
    constant Gemma2KVCacheCopyDimensions&,
    uint2
);

kernel void verifier_input_validate_u32_kernel(
    const device uint* actual [[buffer(0)]],
    const device uint* expected [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant VerifierInputValidationDimensions& dims [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= dims.rows) return;
    if (actual[row] >= dims.vocabulary_size || actual[row] != expected[row]) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void verifier_selection_request_resolve_kernel(
    const device VerifierDeviceSelectionRow* raw [[buffer(0)]],
    device VerifierResolvedSelectionRow* resolved [[buffer(1)]],
    device uint* histories [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant VerifierRequestResolveDimensions& dims [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= dims.rows) return;
    const VerifierDeviceSelectionRow request = raw[row];
    VerifierResolvedSelectionRow output;
    output.top_k = dims.top_k;
    output.base_candidate_count = dims.top_k;
    output.history_count = 0;
    output.reserved = 0;
    if (request.history_window_limit > verifier_history_capacity
        || request.history_length > request.history_window_limit
        || request.suffix_count > verifier_suffix_capacity
        || (request.history_length == 0 && request.history_head != 0)
        || (request.history_length > 0
            && request.history_head >= request.history_window_limit)) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        resolved[row] = output;
        return;
    }

    uint chronological[64];
    uint length = request.history_length;
    for (uint index = 0; index < length; ++index) {
        const uint token = request.history_token_ids[
            (request.history_head + index) % request.history_window_limit
        ];
        chronological[index] = token;
        if (token >= dims.vocabulary_size) {
            atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        }
    }
    for (uint index = 0; index < request.suffix_count; ++index) {
        const uint token = request.suffix_token_ids[index];
        if (token >= dims.vocabulary_size) {
            atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        }
        if (request.history_window_limit == 0) continue;
        if (length < request.history_window_limit) chronological[length++] = token;
        else {
            for (uint move = 1; move < length; ++move) chronological[move - 1] = chronological[move];
            chronological[length - 1] = token;
        }
    }
    if (dims.repetition_enabled != 0) {
        for (uint index = 0; index < length; ++index) {
            const uint token = chronological[index];
            if (token >= dims.vocabulary_size) {
                atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
                continue;
            }
            bool duplicate = false;
            for (uint prior = 0; prior < output.history_count; ++prior) {
                duplicate = duplicate || output.history_token_ids[prior] == token;
            }
            if (!duplicate) output.history_token_ids[output.history_count++] = token;
        }
    }
    output.base_candidate_count = output.top_k + output.history_count;
    for (uint index = 0; index < output.history_count; ++index) {
        histories[row * verifier_history_capacity + index] = output.history_token_ids[index];
    }
    resolved[row] = output;
}

kernel void verifier_rope_f32_by_positions_kernel(
    const device float* input [[buffer(0)]],
    const device uint* positions [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant VerifierRoPEDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint column = gid.x;
    const uint row = gid.y;
    const uint row_width = dims.head_count * dims.head_dimension;
    if (row >= dims.rows || column >= row_width) return;
    const uint row_offset = row * row_width;
    const uint head_column = column % dims.head_dimension;
    const uint input_index = row_offset + column;
    if (head_column >= dims.rope_dimension_count) {
        output[input_index] = input[input_index];
        return;
    }
    const uint half_dimension = dims.rope_dimension_count / 2;
    const uint pair_column = head_column % half_dimension;
    const uint head_first_column = column - head_column;
    const uint first_index = row_offset + head_first_column + pair_column;
    const uint second_index = first_index + half_dimension;
    const float first = input[first_index];
    const float second = input[second_index];
    const float angle = rope_angle(
        positions[row], pair_column * 2, dims.rope_dimension_count, dims.frequency_base
    ) * dims.frequency_scale;
    const float angle_cos = cos(angle);
    const float angle_sin = sin(angle);
    output[input_index] = head_column < half_dimension
        ? first * angle_cos - second * angle_sin
        : first * angle_sin + second * angle_cos;
}

template <typename Cache, bool check_finite = false>
kernel void verifier_kv_store_kernel(
    const device float* raw_key [[buffer(0)]],
    const device float* raw_value [[buffer(1)]],
    const device uint* positions [[buffer(2)]],
    device Cache* private_key [[buffer(3)]],
    device Cache* private_value [[buffer(4)]],
    constant VerifierKVStoreDimensions& dims [[buffer(5)]],
    device atomic_uint* numerical_status [[buffer(6)]],
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
        const uint head_first = column - head_column;
        const uint first_index = row * row_width + head_first + pair_column;
        const uint second_index = first_index + half_dimension;
        const float first = raw_key[first_index];
        const float second = raw_key[second_index];
        const float angle = rope_angle(
            positions[row], pair_column * 2, dims.rope_dimension_count,
            dims.frequency_base
        ) * dims.frequency_scale;
        rotated = head_column < half_dimension
            ? first * cos(angle) - second * sin(angle)
            : first * sin(angle) + second * cos(angle);
    }
    const Cache key = Cache(rotated);
    const Cache value = Cache(raw_value[source_index]);
    private_key[source_index] = key;
    private_value[source_index] = value;
    if constexpr (check_finite) {
        // Bit 8 is verifier_kv_nonfinite_status in the host bridge. Existing
        // device-input bits must survive; ordinary/KV-only paths check this
        // same word at completion, with no per-layer dispatch or host wait.
        if (!isfinite(key) || !isfinite(value))
            atomic_fetch_or_explicit(numerical_status, 8u, memory_order_relaxed);
    }
}

template [[host_name("verifier_kv_store_f16_kernel")]] kernel void verifier_kv_store_kernel<half>(
const device float*,
    const device float*,
    const device uint*,
    device half*,
    device half*,
    constant VerifierKVStoreDimensions&,
    device atomic_uint*,
    uint2
);
template [[host_name("verifier_kv_store_f16_checked_kernel")]] kernel void verifier_kv_store_kernel<half, true>(
const device float*,
    const device float*,
    const device uint*,
    device half*,
    device half*,
    constant VerifierKVStoreDimensions&,
    device atomic_uint*,
    uint2
);
template [[host_name("verifier_kv_store_f32_kernel")]] kernel void verifier_kv_store_kernel<float>(
const device float*,
    const device float*,
    const device uint*,
    device float*,
    device float*,
    constant VerifierKVStoreDimensions&,
    device atomic_uint*,
    uint2
);

template <typename Cache>
kernel void verifier_kv_scatter_kernel(
    const device Cache* private_key [[buffer(0)]],
    const device Cache* private_value [[buffer(1)]],
    const device uint* selected_nodes [[buffer(2)]],
    device Cache* resident_key [[buffer(3)]],
    device Cache* resident_value [[buffer(4)]],
    constant VerifierKVScatterDimensions& dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= dims.rows || gid.x >= dims.row_width) return;
    // A prefill chunk can exceed a local ring's capacity. Only the newest
    // logical row for each slot publishes, so parallel stores never race.
    if (dims.rows - 1U - gid.y > dims.destination_row_mask) return;
    const uint source = selected_nodes[gid.y] * dims.row_width + gid.x;
    const uint destination = ((dims.destination_position + gid.y) & dims.destination_row_mask)
        * dims.row_width + gid.x;
    resident_key[destination] = private_key[source];
    resident_value[destination] = private_value[source];
}

template [[host_name("verifier_kv_scatter_f16_kernel")]] kernel void verifier_kv_scatter_kernel<half>(
const device half*,
    const device half*,
    const device uint*,
    device half*,
    device half*,
    constant VerifierKVScatterDimensions&,
    uint2
);
template [[host_name("verifier_kv_scatter_f32_kernel")]] kernel void verifier_kv_scatter_kernel<float>(
const device float*,
    const device float*,
    const device uint*,
    device float*,
    device float*,
    constant VerifierKVScatterDimensions&,
    uint2
);

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

// A wrapped multi-row prefill chunk must read old logical rows from the
// resident ring while reading rows produced by this chunk from private F16
// staging. The ring is published only after attention completes.
kernel void causal_grouped_attention_scores_f16_hybrid_kernel(
    const device float* query [[buffer(0)]],
    const device half* key_cache [[buffer(1)]],
    const device half* current_chunk_key [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant Gemma2WrappedPrefillAttentionScoreDimensions& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint score_column = gid.x;
    const uint head = gid.y;
    const uint query_row = gid.z;
    if (query_row >= dims.query_rows || head >= dims.head_count
        || score_column >= dims.attention_token_count) {
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

    const bool from_current_chunk = logical_key_row >= dims.query_position_start;
    const uint key_row = from_current_chunk
        ? logical_key_row & dims.current_chunk_row_mask
        : logical_key_row & dims.cache_row_mask;
    const device half* key = from_current_chunk ? current_chunk_key : key_cache;
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
)metal";
constexpr const char* metal_kernel_source_tail_b = R"metal(

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

inline uint verifier_partition_count_for_tokens(uint token_count) {
    return token_count <= 256 ? 1 : token_count <= 512 ? 2
        : token_count <= 1024 ? 4 : token_count <= 2048 ? 8
        : token_count <= 4096 ? 16 : 32;
}

// The loop and partition reduction intentionally mirror scalar decode. Only
// logical-token lookup differs: committed prefix rows come from the resident
// ring and the causal tree suffix comes from an explicit ancestor list.
template <typename CacheVector, bool emit_q4_sums = false>
kernel void verifier_attention_cache_kernel(
    const device float4* query [[buffer(0)]],
    const device CacheVector* resident_key [[buffer(1)]],
    const device CacheVector* resident_value [[buffer(2)]],
    const device CacheVector* private_key [[buffer(3)]],
    const device CacheVector* private_value [[buffer(4)]],
    const device VerifierAttentionRow* schedules [[buffer(5)]],
    const device uint* ancestors [[buffer(6)]],
    device float4* output [[buffer(7)]],
    device float4* partial_output [[buffer(8)]],
    device float* partial_maxima [[buffer(9)]],
    device float* partial_denominators [[buffer(10)]],
    constant VerifierAttentionDimensions& dims [[buffer(11)]],
    device float* q4_sums [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint kv_head = group.x;
    const uint partition = group.y;
    const uint row = group.z;
    if (row >= dims.rows) return;
    const VerifierAttentionRow schedule = schedules[row];
    const uint attention_count = schedule.base_token_count + schedule.ancestor_count;
    const uint partition_count = verifier_partition_count_for_tokens(attention_count);
    if (partition >= partition_count) return;
    const uint first_query_head = kv_head * 2;
    const uint second_query_head = first_query_head + 1;
    const uint vector_lane = uint(lane) * 2;
    const uint vectors_per_head = dims.key_dimension / 4;
    const uint cache_vectors_per_row = dims.kv_head_count * vectors_per_head;
    const uint query_row_base = row * dims.query_head_count * vectors_per_head;
    const float4 query_first_0 = query[query_row_base
        + first_query_head * vectors_per_head + vector_lane];
    const float4 query_first_1 = query[query_row_base
        + first_query_head * vectors_per_head + vector_lane + 1];
    const float4 query_second_0 = query[query_row_base
        + second_query_head * vectors_per_head + vector_lane];
    const float4 query_second_1 = query[query_row_base
        + second_query_head * vectors_per_head + vector_lane + 1];

    float maximum_first = -INFINITY;
    float maximum_second = -INFINITY;
    float denominator_first = 0.0f;
    float denominator_second = 0.0f;
    float4 accumulated_first_0 = 0.0f;
    float4 accumulated_first_1 = 0.0f;
    float4 accumulated_second_0 = 0.0f;
    float4 accumulated_second_1 = 0.0f;
    const uint first_block = partition * 4 + uint(simd_group);
    const uint block_stride = partition_count * 4;
    const uint block_count = (attention_count + 31) / 32;
    for (uint block = first_block; block < block_count; block += block_stride) {
        const uint token_begin = block * 32;
        const uint token_end = min(token_begin + 32, attention_count);
        for (uint token = token_begin; token < token_end; ++token) {
            bool from_private = token >= schedule.base_token_count;
            uint source_row = from_private
                ? ancestors[schedule.ancestor_offset + token - schedule.base_token_count]
                : (schedule.base_first_position + token) & dims.cache_row_mask;
            const device CacheVector* keys = from_private ? private_key : resident_key;
            const device CacheVector* values = from_private ? private_value : resident_value;
            const uint cache_vector = source_row * cache_vectors_per_row
                + kv_head * vectors_per_head + vector_lane;
            const float4 key_0 = float4(keys[cache_vector]);
            const float4 key_1 = float4(keys[cache_vector + 1]);
            float score_first = simd_sum(dot(query_first_0, key_0) + dot(query_first_1, key_1));
            float score_second = simd_sum(dot(query_second_0, key_0) + dot(query_second_1, key_1));
            score_first = dims.logit_softcap == 0.0f ? score_first * dims.scale
                : precise::tanh((score_first * dims.scale) / dims.logit_softcap) * dims.logit_softcap;
            score_second = dims.logit_softcap == 0.0f ? score_second * dims.scale
                : precise::tanh((score_second * dims.scale) / dims.logit_softcap) * dims.logit_softcap;
            const float new_maximum_first = max(maximum_first, score_first);
            const float new_maximum_second = max(maximum_second, score_second);
            const float old_scale_first = denominator_first == 0.0f
                ? 0.0f : exp(maximum_first - new_maximum_first);
            const float old_scale_second = denominator_second == 0.0f
                ? 0.0f : exp(maximum_second - new_maximum_second);
            const float new_scale_first = exp(score_first - new_maximum_first);
            const float new_scale_second = exp(score_second - new_maximum_second);
            const float4 value_0 = float4(values[cache_vector]);
            const float4 value_1 = float4(values[cache_vector + 1]);
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
            } else {
                const float new_maximum = max(merged_maximum, source_maximum);
                const float old_scale = exp(merged_maximum - new_maximum);
                const float source_scale = exp(source_maximum - new_maximum);
                merged_0 = merged_0 * old_scale + source_0 * source_scale;
                merged_1 = merged_1 * old_scale + source_1 * source_scale;
                merged_denominator = merged_denominator * old_scale
                    + source_denominator * source_scale;
                merged_maximum = new_maximum;
            }
        }
        const uint query_head = first_query_head + paired_head;
        const uint output_vector = (row * dims.query_head_count + query_head)
            * vectors_per_head + vector_lane;
        if (partition_count == 1) {
            const float4 first = merged_0 / merged_denominator;
            const float4 second = merged_1 / merged_denominator;
            output[output_vector] = first;
            output[output_vector + 1] = second;
            if constexpr (emit_q4_sums)
                store_q4_input_sum(q4_sums, output_vector / 2U, first, second);
        } else {
            const uint scalar = (row * verifier_max_partitions + partition)
                * dims.query_head_count + query_head;
            const uint partial_vector = scalar * vectors_per_head + vector_lane;
            partial_output[partial_vector] = merged_0;
            partial_output[partial_vector + 1] = merged_1;
            if (lane == 0) {
                partial_maxima[scalar] = merged_maximum;
                partial_denominators[scalar] = merged_denominator;
            }
        }
    }
}

template [[host_name("verifier_attention_f16_cache_kernel")]] kernel void verifier_attention_cache_kernel<half4>(
const device float4*,
    const device half4*,
    const device half4*,
    const device half4*,
    const device half4*,
    const device VerifierAttentionRow*,
    const device uint*,
    device float4*,
    device float4*,
    device float*,
    device float*,
    constant VerifierAttentionDimensions&,
    device float*,
    uint3,
    ushort,
    ushort
);
template [[host_name("verifier_attention_f32_cache_kernel")]] kernel void verifier_attention_cache_kernel<float4>(
const device float4*,
    const device float4*,
    const device float4*,
    const device float4*,
    const device float4*,
    const device VerifierAttentionRow*,
    const device uint*,
    device float4*,
    device float4*,
    device float*,
    device float*,
    constant VerifierAttentionDimensions&,
    device float*,
    uint3,
    ushort,
    ushort
);

template [[host_name("verifier_attention_q4_sums_f32_cache_kernel")]] kernel void verifier_attention_cache_kernel<float4, true>(
const device float4*,
    const device float4*,
    const device float4*,
    const device float4*,
    const device float4*,
    const device VerifierAttentionRow*,
    const device uint*,
    device float4*,
    device float4*,
    device float*,
    device float*,
    constant VerifierAttentionDimensions&,
    device float*,
    uint3,
    ushort,
    ushort
);

// Only cache loads narrow: queries, softmax state, output and the producer's
// K/8 receipt retain the same F32 arithmetic as the float-cache specialization.
template [[host_name("verifier_attention_q4_sums_f16_cache_kernel")]] kernel void verifier_attention_cache_kernel<half4, true>(
const device float4*,
    const device half4*,
    const device half4*,
    const device half4*,
    const device half4*,
    const device VerifierAttentionRow*,
    const device uint*,
    device float4*,
    device float4*,
    device float*,
    device float*,
    constant VerifierAttentionDimensions&,
    device float*,
    uint3,
    ushort,
    ushort
);

template <bool emit_q4_sums>
kernel void verifier_attention_reduce_f32_kernel(
    const device float4* partial_output [[buffer(0)]],
    const device float* partial_maxima [[buffer(1)]],
    const device float* partial_denominators [[buffer(2)]],
    const device VerifierAttentionRow* schedules [[buffer(3)]],
    device float4* output [[buffer(4)]],
    constant VerifierAttentionDimensions& dims [[buffer(5)]],
    device float* q4_sums [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint kv_head = group.x;
    const uint row = group.y;
    if (row >= dims.rows) return;
    const uint attention_count = schedules[row].base_token_count + schedules[row].ancestor_count;
    const uint partition_count = verifier_partition_count_for_tokens(attention_count);
    if (partition_count == 1) return;
    const uint vectors_per_head = dims.value_dimension / 4;
    const uint vector_lane = uint(lane) * 2;
    const uint first_query_head = kv_head * 2;
    for (uint paired_head = 0; paired_head < 2; ++paired_head) {
        const uint query_head = first_query_head + paired_head;
        float merged_maximum = -INFINITY;
        float merged_denominator = 0.0f;
        float4 merged_0 = 0.0f;
        float4 merged_1 = 0.0f;
        for (uint partition = 0; partition < partition_count; ++partition) {
            const uint scalar = (row * verifier_max_partitions + partition)
                * dims.query_head_count + query_head;
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
            } else {
                const float new_maximum = max(merged_maximum, source_maximum);
                const float old_scale = exp(merged_maximum - new_maximum);
                const float source_scale = exp(source_maximum - new_maximum);
                merged_0 = merged_0 * old_scale + source_0 * source_scale;
                merged_1 = merged_1 * old_scale + source_1 * source_scale;
                merged_denominator = merged_denominator * old_scale
                    + source_denominator * source_scale;
                merged_maximum = new_maximum;
            }
        }
        const uint output_vector = (row * dims.query_head_count + query_head)
            * vectors_per_head + vector_lane;
        const float4 first = merged_0 / merged_denominator;
        const float4 second = merged_1 / merged_denominator;
        output[output_vector] = first;
        output[output_vector + 1] = second;
        if constexpr (emit_q4_sums)
            store_q4_input_sum(q4_sums, output_vector / 2U, first, second);
    }
}
template [[host_name("verifier_attention_reduce_f32_kernel")]] kernel void verifier_attention_reduce_f32_kernel<false>(
    const device float4*, const device float*, const device float*,
    const device VerifierAttentionRow*, device float4*,
    constant VerifierAttentionDimensions&, device float*, uint2, ushort
);

template [[host_name("verifier_attention_reduce_q4_sums_f32_kernel")]] kernel void verifier_attention_reduce_f32_kernel<true>(
    const device float4*, const device float*, const device float*,
    const device VerifierAttentionRow*, device float4*,
    constant VerifierAttentionDimensions&, device float*, uint2, ushort
);


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

kernel void causal_grouped_attention_values_f16_hybrid_kernel(
    const device float* probabilities [[buffer(0)]],
    const device half* value_cache [[buffer(1)]],
    const device half* current_chunk_value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant Gemma2WrappedPrefillAttentionValueDimensions& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint value_column = gid.x;
    const uint head = gid.y;
    const uint query_row = gid.z;
    if (query_row >= dims.query_rows || head >= dims.head_count
        || value_column >= dims.value_length) {
        return;
    }

    const uint query_position = dims.query_position_start + query_row;
    const uint first_value_position = query_position + 1 > dims.attention_token_count
        ? query_position + 1 - dims.attention_token_count
        : 0;
    const uint valid_score_count = query_position - first_value_position + 1;
    const uint cached_score_count = dims.query_position_start > first_value_position
        ? min(valid_score_count, dims.query_position_start - first_value_position)
        : 0;
    const uint value_width = dims.kv_head_count * dims.value_length;
    const uint kv_group_size = dims.head_count / dims.kv_head_count;
    const uint kv_head = head / kv_group_size;
    const uint value_first_column = kv_head * dims.value_length;
    const uint output_width = dims.head_count * dims.value_length;
    const uint output_index = query_row * output_width
        + head * dims.value_length + value_column;
    const uint probability_offset =
        (query_row * dims.head_count + head) * dims.attention_token_count;

    float sum = 0.0f;
    for (uint score_column = 0; score_column < cached_score_count; ++score_column) {
        const uint logical_value_row = first_value_position + score_column;
        const uint value_row = logical_value_row & dims.cache_row_mask;
        sum += probabilities[probability_offset + score_column]
            * float(value_cache[value_row * value_width + value_first_column + value_column]);
    }
    for (uint score_column = cached_score_count; score_column < valid_score_count; ++score_column) {
        const uint logical_value_row = first_value_position + score_column;
        const uint value_row = logical_value_row & dims.current_chunk_row_mask;
        sum += probabilities[probability_offset + score_column]
            * float(current_chunk_value[
                value_row * value_width + value_first_column + value_column
            ]);
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

kernel void token_candidate_tile_f32_kernel(
    const device float* logits [[buffer(0)]],
    device TokenCandidate* runs [[buffer(1)]],
    device TokenCandidateResult* result [[buffer(2)]],
    constant TokenCandidateSelectionDimensions& dims [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup TokenCandidate tile[token_candidate_tile_size];
    const uint token = group * token_candidate_tile_size + tid;
    TokenCandidate candidate = {uint(-1), -INFINITY};
    if (token < dims.columns) {
        const float transformed = dims.softcap == 0.0f ? logits[token]
            : dims.softcap * tanh(logits[token] / dims.softcap);
        if (isfinite(transformed)) {
            candidate = {token, transformed};
        } else {
            atomic_fetch_or_explicit(&result->status, 1u, memory_order_relaxed);
        }
    }
    tile[tid] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A deterministic bitonic network orders best candidates first. The token
    // tie break is part of every compare-exchange, including padded tails.
    for (uint width = 2; width <= token_candidate_tile_size; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint other = tid ^ stride;
            if (other > tid) {
                const bool best_first = (tid & width) == 0;
                const bool other_better = token_candidate_better(tile[other], tile[tid]);
                const bool self_better = token_candidate_better(tile[tid], tile[other]);
                if ((best_first && other_better) || (!best_first && self_better)) {
                    const TokenCandidate temporary = tile[tid];
                    tile[tid] = tile[other];
                    tile[other] = temporary;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < dims.retain_count) {
        runs[group * dims.retain_count + tid] = tile[tid];
    }
}

kernel void token_candidate_merge_f32_kernel(
    const device TokenCandidate* input [[buffer(0)]],
    device TokenCandidate* output [[buffer(1)]],
    constant TokenCandidateMergeDimensions& dims [[buffer(2)]],
    uint output_run [[thread_position_in_grid]]
) {
    const uint left_run = output_run * 2;
    if (left_run >= dims.input_run_count) {
        return;
    }
    const uint right_run = left_run + 1;
    uint left = 0;
    uint right = 0;
    for (uint index = 0; index < dims.retain_count; ++index) {
        TokenCandidate selected = {uint(-1), -INFINITY};
        if (right_run < dims.input_run_count) {
            const TokenCandidate left_value = left < dims.retain_count
                ? input[left_run * dims.retain_count + left]
                : TokenCandidate{uint(-1), -INFINITY};
            const TokenCandidate right_value = right < dims.retain_count
                ? input[right_run * dims.retain_count + right]
                : TokenCandidate{uint(-1), -INFINITY};
            if (left >= dims.retain_count || (right < dims.retain_count && token_candidate_better(right_value, left_value))) {
                selected = right_value;
                ++right;
            } else {
                selected = left_value;
                ++left;
            }
        } else {
            selected = input[left_run * dims.retain_count + left];
            ++left;
        }
        output[output_run * dims.retain_count + index] = selected;
    }
}

kernel void token_candidate_finalize_f32_kernel(
    const device float* logits [[buffer(0)]],
    const device TokenCandidate* final_run [[buffer(1)]],
    const device uint* history [[buffer(2)]],
    device TokenCandidateResult* result [[buffer(3)]],
    constant TokenCandidateSelectionDimensions& dims [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    const uint count = dims.retain_count + dims.history_count;
    if (index == 0) {
        result->count = count;
    }
    if (index < dims.retain_count) {
        result->candidates[index] = final_run[index];
    } else if (index < count) {
        const uint token = history[index - dims.retain_count];
        if (token >= dims.columns) {
            atomic_fetch_or_explicit(&result->status, 2u, memory_order_relaxed);
            result->candidates[index] = {uint(-1), -INFINITY};
        } else {
            const float transformed = dims.softcap == 0.0f ? logits[token]
            : dims.softcap * tanh(logits[token] / dims.softcap);
            if (!isfinite(transformed)) {
                atomic_fetch_or_explicit(&result->status, 1u, memory_order_relaxed);
            }
            result->candidates[index] = {token, transformed};
        }
    }
}

kernel void token_candidate_top1_f32_kernel(
    const device float* logits [[buffer(0)]],
    device TokenCandidateResult* result [[buffer(1)]],
    constant TokenCandidateSelectionDimensions& dims [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup TokenCandidate best[token_candidate_tile_size];
    TokenCandidate local = {uint(-1), -INFINITY};
    for (uint token = tid; token < dims.columns; token += token_candidate_tile_size) {
        const float transformed = dims.softcap == 0.0f ? logits[token]
            : dims.softcap * tanh(logits[token] / dims.softcap);
        if (!isfinite(transformed)) {
            atomic_fetch_or_explicit(&result->status, 1u, memory_order_relaxed);
        } else {
            const TokenCandidate candidate = {token, transformed};
            if (token_candidate_better(candidate, local)) {
                local = candidate;
            }
        }
    }
    best[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = token_candidate_tile_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && token_candidate_better(best[tid + stride], best[tid])) {
            best[tid] = best[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        result->count = 1;
        result->candidates[0] = best[0];
    }
}

kernel void token_candidate_batch_tile_f32_kernel(
    const device float* logits [[buffer(0)]],
    device TokenCandidate* runs [[buffer(1)]],
    device TokenCandidateResult* results [[buffer(2)]],
    constant TokenCandidateBatchSelectionDimensions& batch [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint row = group.y;
    const TokenCandidateSelectionDimensions dims = batch.rows[row];
    threadgroup TokenCandidate tile[token_candidate_tile_size];
    const uint token = group.x * token_candidate_tile_size + tid;
    TokenCandidate candidate = {uint(-1), -INFINITY};
    if (token < dims.columns) {
        const float transformed = dims.softcap == 0.0f ? logits[row * dims.columns + token]
            : dims.softcap * tanh(logits[row * dims.columns + token] / dims.softcap);
        if (isfinite(transformed)) candidate = {token, transformed};
        else atomic_fetch_or_explicit(&results[row].status, 1u, memory_order_relaxed);
    }
    tile[tid] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= token_candidate_tile_size; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint other = tid ^ stride;
            if (other > tid) {
                const bool best_first = (tid & width) == 0;
                const bool other_better = token_candidate_better(tile[other], tile[tid]);
                const bool self_better = token_candidate_better(tile[tid], tile[other]);
                if ((best_first && other_better) || (!best_first && self_better)) {
                    const TokenCandidate temporary = tile[tid];
                    tile[tid] = tile[other];
                    tile[other] = temporary;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < dims.retain_count) {
        runs[row * batch.run_stride + group.x * dims.retain_count + tid] = tile[tid];
    }
}

kernel void token_candidate_batch_merge_f32_kernel(
    const device TokenCandidate* input [[buffer(0)]],
    device TokenCandidate* output [[buffer(1)]],
    constant TokenCandidateBatchMergeDimensions& dims [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint output_run = position.x;
    const uint row = position.y;
    const uint retain_count = dims.retain_count[row];
    const uint row_offset = row * dims.run_stride;
    const uint left_run = output_run * 2;
    if (left_run >= dims.input_run_count) return;
    const uint right_run = left_run + 1;
    uint left = 0;
    uint right = 0;
    for (uint index = 0; index < retain_count; ++index) {
        TokenCandidate selected = {uint(-1), -INFINITY};
        if (right_run < dims.input_run_count) {
            const TokenCandidate left_value = left < retain_count
                ? input[row_offset + left_run * retain_count + left]
                : TokenCandidate{uint(-1), -INFINITY};
            const TokenCandidate right_value = right < retain_count
                ? input[row_offset + right_run * retain_count + right]
                : TokenCandidate{uint(-1), -INFINITY};
            if (left >= retain_count || (right < retain_count && token_candidate_better(right_value, left_value))) {
                selected = right_value;
                ++right;
            } else {
                selected = left_value;
                ++left;
            }
        } else {
            selected = input[row_offset + left_run * retain_count + left];
            ++left;
        }
        output[row_offset + output_run * retain_count + index] = selected;
    }
}

kernel void token_candidate_batch_finalize_f32_kernel(
    const device float* logits [[buffer(0)]],
    const device TokenCandidate* final_runs [[buffer(1)]],
    const device uint* history [[buffer(2)]],
    device TokenCandidateResult* results [[buffer(3)]],
    constant TokenCandidateBatchSelectionDimensions& batch [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint index = position.x;
    const uint row = position.y;
    const TokenCandidateSelectionDimensions dims = batch.rows[row];
    const uint count = dims.retain_count + dims.history_count;
    if (index == 0) results[row].count = count;
    if (index < dims.retain_count) {
        results[row].candidates[index] = final_runs[row * batch.run_stride + index];
    } else if (index < count) {
        const uint token = history[row * 64 + index - dims.retain_count];
        if (token >= dims.columns) {
            atomic_fetch_or_explicit(&results[row].status, 2u, memory_order_relaxed);
            results[row].candidates[index] = {uint(-1), -INFINITY};
        } else {
            const float transformed = dims.softcap == 0.0f ? logits[row * dims.columns + token]
            : dims.softcap * tanh(logits[row * dims.columns + token] / dims.softcap);
            if (!isfinite(transformed)) atomic_fetch_or_explicit(&results[row].status, 1u, memory_order_relaxed);
            results[row].candidates[index] = {token, transformed};
        }
    }
}

kernel void token_candidate_batch_top1_f32_kernel(
    const device float* logits [[buffer(0)]],
    device TokenCandidateResult* results [[buffer(1)]],
    constant TokenCandidateBatchSelectionDimensions& batch [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const TokenCandidateSelectionDimensions dims = batch.rows[row];
    threadgroup TokenCandidate best[token_candidate_tile_size];
    TokenCandidate local = {uint(-1), -INFINITY};
    for (uint token = tid; token < dims.columns; token += token_candidate_tile_size) {
        const float transformed = dims.softcap == 0.0f ? logits[row * dims.columns + token]
            : dims.softcap * tanh(logits[row * dims.columns + token] / dims.softcap);
        if (!isfinite(transformed)) atomic_fetch_or_explicit(&results[row].status, 1u, memory_order_relaxed);
        else {
            const TokenCandidate candidate = {token, transformed};
            if (token_candidate_better(candidate, local)) local = candidate;
        }
    }
    best[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = token_candidate_tile_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && token_candidate_better(best[tid + stride], best[tid])) best[tid] = best[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        results[row].count = 1;
        results[row].candidates[0] = best[0];
    }
}

kernel void target_candidate_batch_tile_f32_kernel(
    const device float* logits [[buffer(0)]],
    const device VerifierResolvedSelectionRow* requests [[buffer(1)]],
    device TokenCandidate* runs [[buffer(2)]],
    device TokenCandidateResult* results [[buffer(3)]],
    constant TargetCandidateBatchDimensions& dims [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint row = group.y;
    if (row >= dims.rows) return;
    const uint retain_count = min(dims.columns, requests[row].base_candidate_count);
    threadgroup TokenCandidate tile[token_candidate_tile_size];
    const uint token = group.x * token_candidate_tile_size + tid;
    TokenCandidate candidate = {uint(-1), -INFINITY};
    if (token < dims.columns) {
        const float transformed = dims.softcap == 0.0f ? logits[row * dims.columns + token]
            : dims.softcap * tanh(logits[row * dims.columns + token] / dims.softcap);
        if (isfinite(transformed)) candidate = {token, transformed};
        else atomic_fetch_or_explicit(&results[row].status, 1u, memory_order_relaxed);
    }
    tile[tid] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= token_candidate_tile_size; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint other = tid ^ stride;
            if (other > tid) {
                const bool best_first = (tid & width) == 0;
                const bool other_better = token_candidate_better(tile[other], tile[tid]);
                const bool self_better = token_candidate_better(tile[tid], tile[other]);
                if ((best_first && other_better) || (!best_first && self_better)) {
                    const TokenCandidate temporary = tile[tid];
                    tile[tid] = tile[other];
                    tile[other] = temporary;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < retain_count) {
        runs[row * dims.run_stride + group.x * retain_count + tid] = tile[tid];
    }
}

kernel void target_candidate_batch_merge_f32_kernel(
    const device TokenCandidate* input [[buffer(0)]],
    const device VerifierResolvedSelectionRow* requests [[buffer(1)]],
    device TokenCandidate* output [[buffer(2)]],
    constant TargetCandidateBatchDimensions& dims [[buffer(3)]],
    constant uint& input_run_count [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint output_run = position.x;
    const uint row = position.y;
    if (row >= dims.rows) return;
    const uint retain_count = min(dims.columns, requests[row].base_candidate_count);
    const uint row_offset = row * dims.run_stride;
    const uint left_run = output_run * 2;
    if (left_run >= input_run_count) return;
    const uint right_run = left_run + 1;
    uint left = 0;
    uint right = 0;
    for (uint index = 0; index < retain_count; ++index) {
        TokenCandidate selected = {uint(-1), -INFINITY};
        if (right_run < input_run_count) {
            const TokenCandidate left_value = left < retain_count
                ? input[row_offset + left_run * retain_count + left]
                : TokenCandidate{uint(-1), -INFINITY};
            const TokenCandidate right_value = right < retain_count
                ? input[row_offset + right_run * retain_count + right]
                : TokenCandidate{uint(-1), -INFINITY};
            if (left >= retain_count
                || (right < retain_count && token_candidate_better(right_value, left_value))) {
                selected = right_value;
                ++right;
            } else {
                selected = left_value;
                ++left;
            }
        } else {
            selected = input[row_offset + left_run * retain_count + left++];
        }
        output[row_offset + output_run * retain_count + index] = selected;
    }
}

kernel void target_candidate_batch_finalize_f32_kernel(
    const device float* logits [[buffer(0)]],
    const device TokenCandidate* final_runs [[buffer(1)]],
    const device VerifierResolvedSelectionRow* requests [[buffer(2)]],
    const device uint* history [[buffer(3)]],
    device TokenCandidateResult* results [[buffer(4)]],
    constant TargetCandidateBatchDimensions& dims [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint index = position.x;
    const uint row = position.y;
    if (row >= dims.rows) return;
    const VerifierResolvedSelectionRow request = requests[row];
    const uint retain_count = min(dims.columns, request.base_candidate_count);
    const uint count = retain_count + request.history_count;
    if (index == 0) results[row].count = count;
    if (index < retain_count) {
        results[row].candidates[index] = final_runs[row * dims.run_stride + index];
    } else if (index < count) {
        const uint token = history[row * verifier_history_capacity + index - retain_count];
        if (token >= dims.columns) {
            atomic_fetch_or_explicit(&results[row].status, 2u, memory_order_relaxed);
            results[row].candidates[index] = {uint(-1), -INFINITY};
        } else {
            const float transformed = dims.softcap == 0.0f ? logits[row * dims.columns + token]
            : dims.softcap * tanh(logits[row * dims.columns + token] / dims.softcap);
            if (!isfinite(transformed)) {
                atomic_fetch_or_explicit(&results[row].status, 1u, memory_order_relaxed);
            }
            results[row].candidates[index] = {token, transformed};
        }
    }
}

kernel void target_candidate_batch_top1_f32_kernel(
    const device float* logits [[buffer(0)]],
    device TokenCandidateResult* results [[buffer(1)]],
    constant TargetCandidateBatchDimensions& dims [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row >= dims.rows) return;
    threadgroup TokenCandidate best[token_candidate_tile_size];
    TokenCandidate local = {uint(-1), -INFINITY};
    for (uint token = tid; token < dims.columns; token += token_candidate_tile_size) {
        const float transformed = dims.softcap == 0.0f ? logits[row * dims.columns + token]
            : dims.softcap * tanh(logits[row * dims.columns + token] / dims.softcap);
        if (!isfinite(transformed)) {
            atomic_fetch_or_explicit(&results[row].status, 1u, memory_order_relaxed);
        } else {
            const TokenCandidate candidate = {token, transformed};
            if (token_candidate_better(candidate, local)) local = candidate;
        }
    }
    best[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = token_candidate_tile_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && token_candidate_better(best[tid + stride], best[tid])) {
            best[tid] = best[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        results[row].count = 1;
        results[row].candidates[0] = best[0];
    }
}

)metal";
}  // namespace

const char* metal_kernel_source() noexcept {
    static const std::string source = std::string(metal_kernel_source_head)
        + metal_kernel_source_projections + metal_kernel_source_tail_a + metal_kernel_source_tail_b;
    return source.c_str();
}

}  // namespace local_gguf_engine::tensor::backends::metal
