#include "local_gguf_engine/tensor/kquant_packing.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

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
    for (std::size_t column = 0U; column < columns; ++column) {
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
    return output;
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
