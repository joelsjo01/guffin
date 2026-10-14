#include "local_gguf_engine/tensor/kquant_packing.hpp"

#include "kquant_packing_internal.hpp"
#include "weight_preparation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace local_gguf_engine::tensor {
namespace {

std::uint16_t read_u16(std::span<const std::uint8_t> bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset])
        | static_cast<std::uint16_t>(std::uint16_t(bytes[offset + 1U]) << 8U);
}

void write_u16(std::span<std::uint8_t> bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value);
    bytes[offset + 1U] = static_cast<std::uint8_t>(value >> 8U);
}

void validate_source(TensorView source, TensorPacking packing) {
    const auto& descriptor = source.descriptor();
    if (descriptor.packing != packing) {
        throw std::invalid_argument("K-quant conversion received the wrong source packing");
    }
    if (descriptor.dtype != DType::q4_k && descriptor.dtype != DType::q6_k) {
        throw std::invalid_argument("K-quant conversion requires Q4_K or Q6_K weights");
    }
    const auto expected = packing == TensorPacking::canonical
        ? TensorDescriptor::from_quantized_gguf_weight_matrix(descriptor.shape, descriptor.dtype)
        : TensorDescriptor::from_interleaved_kquant_weight_matrix(descriptor.shape, descriptor.dtype);
    if (descriptor != expected || source.bytes().size() != expected.byte_count) {
        throw std::invalid_argument("K-quant conversion received an inconsistent source descriptor");
    }
}

#if defined(__aarch64__)
// Raw byte loads/stores preserve GGUF's little-endian words without requiring
// aligned pointers or type-punning uint16_t access to caller-owned storage.
void pack_q4_k_columns8_tile(const std::array<const std::uint8_t*, 8>& native,
    std::span<std::uint8_t> tile) {
    for (std::size_t word = 0U; word < 64U; word += 8U) {
        std::array<uint16x8_t, 8> rows;
        for (std::size_t lane = 0U; lane < 8U; ++lane)
            rows[lane] = vreinterpretq_u16_u8(vld1q_u8(native[lane] + 16U + word * 2U));
        const auto a = vtrnq_u16(rows[0], rows[1]);
        const auto b = vtrnq_u16(rows[2], rows[3]);
        const auto c = vtrnq_u16(rows[4], rows[5]);
        const auto d = vtrnq_u16(rows[6], rows[7]);
        const std::array<uint32x4x2_t, 4> pairs {
            vtrnq_u32(vreinterpretq_u32_u16(a.val[0]), vreinterpretq_u32_u16(b.val[0])),
            vtrnq_u32(vreinterpretq_u32_u16(a.val[1]), vreinterpretq_u32_u16(b.val[1])),
            vtrnq_u32(vreinterpretq_u32_u16(c.val[0]), vreinterpretq_u32_u16(d.val[0])),
            vtrnq_u32(vreinterpretq_u32_u16(c.val[1]), vreinterpretq_u32_u16(d.val[1])),
        };
        for (std::size_t parity = 0U; parity < 2U; ++parity) {
            for (std::size_t pair = 0U; pair < 2U; ++pair) {
                const auto left = vreinterpretq_u64_u32(pairs[parity].val[pair]);
                const auto right = vreinterpretq_u64_u32(pairs[parity + 2U].val[pair]);
                const auto row = parity + pair * 2U;
                vst1q_u8(tile.data() + (word + row) * 16U,
                    vreinterpretq_u8_u64(vzip1q_u64(left, right)));
                vst1q_u8(tile.data() + (word + row + 4U) * 16U,
                    vreinterpretq_u8_u64(vzip2q_u64(left, right)));
            }
        }
    }
    for (std::size_t lane = 0U; lane < 8U; ++lane)
        detail::pack_q4_k_columns8_metadata(std::span<const std::uint8_t>(native[lane], 144U), tile, lane);
}
#endif

std::uint16_t q4_scale_min(std::span<const std::uint8_t> native, std::size_t group) {
    const auto packed = native.subspan(4U, 12U);
    const auto scale = group < 4U ? packed[group] & 63U
        : (packed[group + 4U] & 15U) | ((static_cast<unsigned>(packed[group - 4U]) >> 6U) << 4U);
    const auto minimum = group < 4U ? packed[group + 4U] & 63U
        : (static_cast<unsigned>(packed[group + 4U]) >> 4U)
            | ((static_cast<unsigned>(packed[group]) >> 6U) << 4U);
    return static_cast<std::uint16_t>(scale | (minimum << 8U));
}

void restore_q4_scale_min(std::span<std::uint8_t> native, std::size_t group,
    std::uint16_t value) {
    const auto scale = static_cast<std::uint8_t>(value & 255U);
    const auto minimum = static_cast<std::uint8_t>(value >> 8U);
    if (scale > 63U || minimum > 63U) {
        throw std::invalid_argument("interleaved Q4_K scale or minimum exceeds six bits");
    }
    auto packed = native.subspan(4U, 12U);
    if (group < 4U) {
        packed[group] |= scale;
        packed[group + 4U] |= minimum;
    } else {
        packed[group + 4U] = static_cast<std::uint8_t>((scale & 15U) | ((minimum & 15U) << 4U));
        packed[group - 4U] |= static_cast<std::uint8_t>((scale >> 4U) << 6U);
        packed[group] |= static_cast<std::uint8_t>((minimum >> 4U) << 6U);
    }
}

std::uint8_t q6_code(std::span<const std::uint8_t> native, std::size_t element) {
    const auto half = element / 128U;
    const auto quarter = (element % 128U) / 32U;
    const auto lane = element % 32U;
    const auto low = native[half * 64U + (quarter % 2U) * 32U + lane];
    const auto high = native[128U + half * 32U + lane];
    return static_cast<std::uint8_t>(((low >> (quarter >= 2U ? 4U : 0U)) & 15U)
        | (((high >> (quarter * 2U)) & 3U) << 4U));
}

// Scalar lanes for direct preparation's Q6 tiles and partial Q4 tails.
// The allocating wrapper retains its measured original compiler layout.
void pack_kquant_columns8_lane(std::span<const std::uint8_t> native,
    std::span<std::uint8_t> tile, std::size_t lane, bool q4) {
    if (q4) {
        detail::pack_q4_k_columns8_lane(native, tile, lane);
        return;
    }
    for (std::size_t word = 0U; word < 64U; ++word) {
        std::uint16_t low = 0U;
        std::uint8_t high = 0U;
        for (std::size_t item = 0U; item < 4U; ++item) {
            const auto code = q6_code(native, word * 4U + item);
            low |= static_cast<std::uint16_t>((code & 15U) << (item * 4U));
            high |= static_cast<std::uint8_t>((code >> 4U) << (item * 2U));
        }
        write_u16(tile, (word * 8U + lane) * 2U, low);
        tile[1024U + word * 8U + lane] = high;
    }
    for (std::size_t group = 0U; group < 16U; ++group)
        tile[1536U + group * 8U + lane] = native[192U + group];
    write_u16(tile, 1664U + lane * 2U, read_u16(native, 208U));
}

void restore_q6_code(std::span<std::uint8_t> native, std::size_t element, std::uint8_t code) {
    const auto half = element / 128U;
    const auto quarter = (element % 128U) / 32U;
    const auto lane = element % 32U;
    native[half * 64U + (quarter % 2U) * 32U + lane]
        |= static_cast<std::uint8_t>((code & 15U) << (quarter >= 2U ? 4U : 0U));
    native[128U + half * 32U + lane]
        |= static_cast<std::uint8_t>((code >> 4U) << (quarter * 2U));
}

}  // namespace

TensorStorage repack_kquant_weight_columns8(TensorView source) {
    validate_source(source, TensorPacking::canonical);
    const auto& descriptor = source.descriptor();
    auto output = TensorStorage::allocate(TensorDescriptor::from_interleaved_kquant_weight_matrix(
        descriptor.shape, descriptor.dtype));
    const auto blocks = static_cast<std::size_t>(descriptor.shape.dimension(0) / 256U);
    const auto columns = static_cast<std::size_t>(descriptor.shape.dimension(1));
    const bool q4 = descriptor.dtype == DType::q4_k;
    const auto native_bytes = q4 ? 144U : 210U;
    const auto tile_bytes = q4 ? kquant_q4_columns8_tile_bytes : kquant_q6_columns8_tile_bytes;
    // Keep each worker's native reads contiguous and preserve the halfword packing loops.
    // Eight-column boundaries give each worker complete, disjoint output tiles.
    const auto pack_columns = [&](std::size_t begin_column, std::size_t end_column) {
        for (std::size_t column = begin_column; column < end_column; ++column) {
            const auto lane = column % kquant_packing_columns;
            for (std::size_t block = 0U; block < blocks; ++block) {
                const auto native = source.bytes().subspan((column * blocks + block) * native_bytes, native_bytes);
                auto tile = output.mutable_bytes().subspan(
                    ((column / kquant_packing_columns) * blocks + block) * tile_bytes, tile_bytes);
                if (q4) {
                    for (std::size_t word = 0U; word < 64U; ++word)
                        write_u16(tile, (word * 8U + lane) * 2U, read_u16(native, 16U + word * 2U));
                    for (std::size_t group = 0U; group < 8U; ++group)
                        write_u16(tile, 1024U + (group * 8U + lane) * 2U, q4_scale_min(native, group));
                    std::copy_n(native.begin(), 4U, tile.subspan(1152U + lane * 4U).begin());
                } else {
                    for (std::size_t word = 0U; word < 64U; ++word) {
                        std::uint16_t low = 0U;
                        std::uint8_t high = 0U;
                        for (std::size_t item = 0U; item < 4U; ++item) {
                            const auto code = q6_code(native, word * 4U + item);
                            low |= static_cast<std::uint16_t>((code & 15U) << (item * 4U));
                            high |= static_cast<std::uint8_t>((code >> 4U) << (item * 2U));
                        }
                        write_u16(tile, (word * 8U + lane) * 2U, low);
                        tile[1024U + word * 8U + lane] = high;
                    }
                    for (std::size_t group = 0U; group < 16U; ++group)
                        tile[1536U + group * 8U + lane] = native[192U + group];
                    write_u16(tile, 1664U + lane * 2U, read_u16(native, 208U));
                }
            }
        }
    };
    const auto groups = columns / kquant_packing_columns
        + (columns % kquant_packing_columns != 0U ? 1U : 0U);
    if (source.bytes().size() < 2U * 1024U * 1024U || groups < 4U) {
        pack_columns(0U, columns);
    } else {
        detail::prepare_weight_ranges(groups, 4U, [&](std::size_t begin_group, std::size_t end_group) {
            pack_columns(begin_group * kquant_packing_columns,
                std::min(columns, end_group * kquant_packing_columns));
        });
    }
    return output;
}

void repack_kquant_weight_columns8_into(TensorView source,
    const TensorDescriptor& destination, std::span<std::uint8_t> bytes) {
    validate_source(source, TensorPacking::canonical);
    const auto& descriptor = source.descriptor();
    detail::validate_preparation_destination(source, destination,
        TensorDescriptor::from_interleaved_kquant_weight_matrix(descriptor.shape, descriptor.dtype), bytes);
    const auto blocks = static_cast<std::size_t>(descriptor.shape.dimension(0) / 256U);
    const auto columns = static_cast<std::size_t>(descriptor.shape.dimension(1));
    const bool q4 = descriptor.dtype == DType::q4_k;
    const auto native_bytes = q4 ? 144U : 210U;
    const auto tile_bytes = q4 ? kquant_q4_columns8_tile_bytes : kquant_q6_columns8_tile_bytes;
    const auto tile_count = bytes.size() / tile_bytes;
    detail::prepare_weight_ranges(tile_count, 4096U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const auto first_column = index / blocks * kquant_packing_columns;
            const auto block = index % blocks;
            const auto lanes = std::min(kquant_packing_columns, columns - first_column);
            auto tile = bytes.subspan(index * tile_bytes, tile_bytes);
            // Full tiles overwrite every byte. Only a partial final column tile
            // needs clearing when the caller supplies uninitialized storage.
            if (lanes < kquant_packing_columns) std::fill(tile.begin(), tile.end(), 0U);
#if defined(__aarch64__)
            if (q4 && lanes == kquant_packing_columns) {
                std::array<const std::uint8_t*, 8> native;
                for (std::size_t lane = 0U; lane < 8U; ++lane)
                    native[lane] = source.bytes().data() + ((first_column + lane) * blocks + block) * native_bytes;
                pack_q4_k_columns8_tile(native, tile);
                continue;
            }
#endif
            for (std::size_t lane = 0U; lane < lanes; ++lane) {
                const auto native = source.bytes().subspan(
                    ((first_column + lane) * blocks + block) * native_bytes, native_bytes);
                pack_kquant_columns8_lane(native, tile, lane, q4);
            }
        }
    });
}

TensorStorage restore_native_kquant_weight(TensorView source) {
    validate_source(source, TensorPacking::kquant_columns8);
    const auto& descriptor = source.descriptor();
    auto output = TensorStorage::allocate(TensorDescriptor::from_quantized_gguf_weight_matrix(
        descriptor.shape, descriptor.dtype));
    const auto blocks = static_cast<std::size_t>(descriptor.shape.dimension(0) / 256U);
    const auto columns = static_cast<std::size_t>(descriptor.shape.dimension(1));
    const bool q4 = descriptor.dtype == DType::q4_k;
    const auto native_bytes = q4 ? 144U : 210U;
    const auto tile_bytes = q4 ? kquant_q4_columns8_tile_bytes : kquant_q6_columns8_tile_bytes;
    for (std::size_t column = 0U; column < columns; ++column) {
        const auto lane = column % kquant_packing_columns;
        for (std::size_t block = 0U; block < blocks; ++block) {
            const auto tile = source.bytes().subspan(
                ((column / kquant_packing_columns) * blocks + block) * tile_bytes, tile_bytes);
            auto native = output.mutable_bytes().subspan((column * blocks + block) * native_bytes, native_bytes);
            if (q4) {
                for (std::size_t word = 0U; word < 64U; ++word)
                    write_u16(native, 16U + word * 2U, read_u16(tile, (word * 8U + lane) * 2U));
                for (std::size_t group = 0U; group < 8U; ++group)
                    restore_q4_scale_min(native, group, read_u16(tile, 1024U + (group * 8U + lane) * 2U));
                std::copy_n(tile.subspan(1152U + lane * 4U).begin(), 4U, native.begin());
            } else {
                for (std::size_t word = 0U; word < 64U; ++word) {
                    const auto low = read_u16(tile, (word * 8U + lane) * 2U);
                    const auto high = tile[1024U + word * 8U + lane];
                    for (std::size_t item = 0U; item < 4U; ++item) {
                        restore_q6_code(native, word * 4U + item, static_cast<std::uint8_t>(
                            ((low >> (item * 4U)) & 15U) | (((high >> (item * 2U)) & 3U) << 4U)));
                    }
                }
                for (std::size_t group = 0U; group < 16U; ++group)
                    native[192U + group] = tile[1536U + group * 8U + lane];
                write_u16(native, 208U, read_u16(tile, 1664U + lane * 2U));
            }
        }
    }
    return output;
}

}  // namespace local_gguf_engine::tensor
