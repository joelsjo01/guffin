#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstddef>

namespace local_gguf_engine::tensor {

inline constexpr std::size_t kquant_packing_columns = 8U;
inline constexpr std::size_t kquant_q4_columns8_tile_bytes = 1184U;
inline constexpr std::size_t kquant_q6_columns8_tile_bytes = 1680U;

// Shapes remain [K, N]. Tiles are ordered by N/8 then K/256; missing columns
// contain zero bytes. Half scale bits and every quantization code are lossless.
// Q4: ushort quants[64][8], ushort scale_min[8][8], half2 d_dmin[8].
// Q6: ushort low[64][8], uchar high[64][8], char scales[16][8], half d[8].
// Each Q6 low/high entry describes four consecutive values in a 16-value group.
[[nodiscard]] TensorStorage repack_kquant_weight_columns8(TensorView source);
[[nodiscard]] TensorStorage restore_native_kquant_weight(TensorView source);

}  // namespace local_gguf_engine::tensor
