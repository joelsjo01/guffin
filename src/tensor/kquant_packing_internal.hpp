#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor::detail {

// Shared scale/min and original F16 bytes for scalar and transposed Q4 tiles.
inline void pack_q4_k_columns8_metadata(std::span<const std::uint8_t> native,
    std::span<std::uint8_t> tile, std::size_t lane) {
    const auto packed = native.subspan(4U, 12U);
    for (std::size_t group = 0U; group < 8U; ++group) {
        const auto scale = group < 4U ? packed[group] & 63U
            : (packed[group + 4U] & 15U) | ((static_cast<unsigned>(packed[group - 4U]) >> 6U) << 4U);
        const auto minimum = group < 4U ? packed[group + 4U] & 63U
            : (static_cast<unsigned>(packed[group + 4U]) >> 4U)
                | ((static_cast<unsigned>(packed[group]) >> 6U) << 4U);
        const auto destination = 1024U + (group * 8U + lane) * 2U;
        tile[destination] = static_cast<std::uint8_t>(scale);
        tile[destination + 1U] = static_cast<std::uint8_t>(minimum);
    }
    std::copy_n(native.begin(), 4U, tile.subspan(1152U + lane * 4U).begin());
}

// Shared by lossless repacking and direct Q6_K-to-packed-Q4_K preparation.
// The caller owns the complete tile and supplies a valid lane and native block.
inline void pack_q4_k_columns8_lane(std::span<const std::uint8_t> native,
    std::span<std::uint8_t> tile, std::size_t lane) {
    for (std::size_t word = 0U; word < 64U; ++word) {
        const auto destination = (word * 8U + lane) * 2U;
        tile[destination] = native[16U + word * 2U];
        tile[destination + 1U] = native[17U + word * 2U];
    }
    pack_q4_k_columns8_metadata(native, tile, lane);
}

}  // namespace local_gguf_engine::tensor::detail
