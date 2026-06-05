#pragma once

#include <cstdint>

namespace local_gguf_engine::tensor {

[[nodiscard]] float f16_bits_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t f32_to_f16_bits(float value) noexcept;

}  // namespace local_gguf_engine::tensor
