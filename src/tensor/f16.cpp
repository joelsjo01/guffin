#include "local_gguf_engine/tensor/f16.hpp"

#include <bit>

namespace local_gguf_engine::tensor {

float f16_bits_to_f32(std::uint16_t bits) noexcept {
    const auto sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    auto exponent = static_cast<std::uint32_t>((bits >> 10U) & 0x1FU);
    auto fraction = static_cast<std::uint32_t>(bits & 0x03FFU);

    if (exponent == 0) {
        if (fraction == 0) {
            return std::bit_cast<float>(sign);
        }

        int normalized_exponent = -14;
        while ((fraction & 0x0400U) == 0) {
            fraction <<= 1U;
            --normalized_exponent;
        }

        fraction &= 0x03FFU;
        const auto f32_exponent = static_cast<std::uint32_t>(normalized_exponent + 127) << 23U;
        const auto f32_fraction = fraction << 13U;
        return std::bit_cast<float>(sign | f32_exponent | f32_fraction);
    }

    if (exponent == 0x1FU) {
        const auto f32_exponent = 0xFFU << 23U;
        const auto f32_fraction = fraction << 13U;
        return std::bit_cast<float>(sign | f32_exponent | f32_fraction);
    }

    exponent = exponent + 112U;
    return std::bit_cast<float>(sign | (exponent << 23U) | (fraction << 13U));
}

std::uint16_t f32_to_f16_bits(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    auto exponent = static_cast<int>((bits >> 23U) & 0xFFU);
    auto fraction = bits & 0x7FFFFFU;

    if (exponent == 0xFF) {
        if (fraction == 0) {
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
        return static_cast<std::uint16_t>(sign | 0x7E00U);
    }

    exponent = exponent - 127 + 15;

    if (exponent >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }

        fraction |= 0x800000U;
        const auto shift = static_cast<std::uint32_t>(14 - exponent);
        const auto half_fraction = static_cast<std::uint16_t>((fraction + (1U << (shift - 1U))) >> shift);
        return static_cast<std::uint16_t>(sign | half_fraction);
    }

    fraction += 0x00001000U;
    if ((fraction & 0x00800000U) != 0) {
        fraction = 0;
        ++exponent;
        if (exponent >= 0x1F) {
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint16_t>(exponent) << 10U)
        | static_cast<std::uint16_t>(fraction >> 13U)
    );
}

}  // namespace local_gguf_engine::tensor
