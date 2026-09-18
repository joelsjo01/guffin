#include "local_gguf_engine/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::runtime {
namespace {

constexpr std::array<std::uint32_t, 64> round_constants {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

[[nodiscard]] constexpr std::uint32_t rotate_right(
    std::uint32_t value,
    std::uint32_t count
) noexcept {
    return (value >> count) | (value << (32U - count));
}

[[nodiscard]] constexpr std::uint32_t load_be_u32(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint32_t>(bytes[0]) << 24U
        | static_cast<std::uint32_t>(bytes[1]) << 16U
        | static_cast<std::uint32_t>(bytes[2]) << 8U
        | static_cast<std::uint32_t>(bytes[3]);
}

void store_be_u32(std::uint8_t* bytes, std::uint32_t value) noexcept {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

[[nodiscard]] std::uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') return static_cast<std::uint8_t>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<std::uint8_t>(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return static_cast<std::uint8_t>(value - 'A' + 10);
    throw std::invalid_argument("SHA-256 text contains a non-hexadecimal character");
}

}  // namespace

bool Sha256Digest::is_zero() const noexcept {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t value) {
        return value == 0U;
    });
}

std::string Sha256Digest::hex() const {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string output;
    output.resize(bytes.size() * 2U);
    for (std::size_t index = 0U; index < bytes.size(); ++index) {
        output[index * 2U] = digits[bytes[index] >> 4U];
        output[index * 2U + 1U] = digits[bytes[index] & 0x0fU];
    }
    return output;
}

Sha256::Sha256()
    : state_ {
          0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
          0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
      } {}

void Sha256::transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words {};
    for (std::size_t index = 0U; index < 16U; ++index) {
        words[index] = load_be_u32(block + index * 4U);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
        const auto x = words[index - 15U];
        const auto y = words[index - 2U];
        const auto small0 = rotate_right(x, 7U) ^ rotate_right(x, 18U) ^ (x >> 3U);
        const auto small1 = rotate_right(y, 17U) ^ rotate_right(y, 19U) ^ (y >> 10U);
        words[index] = words[index - 16U] + small0 + words[index - 7U] + small1;
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0U; index < words.size(); ++index) {
        const auto big1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const auto choose = (e & f) ^ (~e & g);
        const auto first = h + big1 + choose + round_constants[index] + words[index];
        const auto big0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto second = big0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::span<const std::uint8_t> bytes) {
    if (finalized_) throw std::logic_error("SHA-256 update after finalize");
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - byte_count_) {
        throw std::overflow_error("SHA-256 input byte count overflow");
    }
    byte_count_ += static_cast<std::uint64_t>(bytes.size());
    while (!bytes.empty()) {
        const auto copied = std::min(bytes.size(), pending_.size() - pending_count_);
        std::copy_n(bytes.begin(), copied, pending_.begin() + pending_count_);
        pending_count_ += copied;
        bytes = bytes.subspan(copied);
        if (pending_count_ == pending_.size()) {
            transform(pending_.data());
            pending_count_ = 0U;
        }
    }
}

Sha256Digest Sha256::finalize() {
    if (finalized_) throw std::logic_error("SHA-256 finalize called twice");
    if (byte_count_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::overflow_error("SHA-256 bit count overflow");
    }
    const auto bit_count = byte_count_ * 8U;
    pending_[pending_count_++] = 0x80U;
    if (pending_count_ > 56U) {
        std::fill(pending_.begin() + pending_count_, pending_.end(), 0U);
        transform(pending_.data());
        pending_count_ = 0U;
    }
    std::fill(pending_.begin() + pending_count_, pending_.begin() + 56U, 0U);
    for (std::size_t index = 0U; index < 8U; ++index) {
        pending_[63U - index] = static_cast<std::uint8_t>(bit_count >> (index * 8U));
    }
    transform(pending_.data());
    finalized_ = true;

    Sha256Digest result;
    for (std::size_t index = 0U; index < state_.size(); ++index) {
        store_be_u32(result.bytes.data() + index * 4U, state_[index]);
    }
    return result;
}

Sha256Digest sha256(std::span<const std::uint8_t> bytes) {
    Sha256 state;
    state.update(bytes);
    return state.finalize();
}

Sha256Digest parse_sha256_hex(std::string_view text) {
    if (text.size() != 64U) {
        throw std::invalid_argument("SHA-256 text must contain exactly 64 hexadecimal characters");
    }
    Sha256Digest result;
    for (std::size_t index = 0U; index < result.bytes.size(); ++index) {
        result.bytes[index] = static_cast<std::uint8_t>(
            hex_nibble(text[index * 2U]) << 4U | hex_nibble(text[index * 2U + 1U])
        );
    }
    return result;
}

}  // namespace local_gguf_engine::runtime
