#include "local_gguf_engine/runtime/sha256.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::runtime {
namespace {

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

Sha256::Sha256() {
    if (CC_SHA256_Init(&state_) != 1) throw std::runtime_error("SHA-256 initialization failed");
}

void Sha256::update(std::span<const std::uint8_t> bytes) {
    if (finalized_) throw std::logic_error("SHA-256 update after finalize");
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - byte_count_) {
        throw std::overflow_error("SHA-256 input byte count overflow");
    }
    byte_count_ += static_cast<std::uint64_t>(bytes.size());
    // CommonCrypto is part of macOS libSystem, not an inference dependency.
    // CC_LONG is only 32 bits: a whole GGUF may exceed that in one update.
    // Keep every receipt's existing byte stream and hash it in bounded calls.
    while (!bytes.empty()) {
        const auto count = std::min(bytes.size(), std::size_t(std::numeric_limits<CC_LONG>::max()));
        if (CC_SHA256_Update(&state_, bytes.data(), static_cast<CC_LONG>(count)) != 1)
            throw std::runtime_error("SHA-256 update failed");
        bytes = bytes.subspan(count);
    }
}

Sha256Digest Sha256::finalize() {
    if (finalized_) throw std::logic_error("SHA-256 finalize called twice");
    if (byte_count_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::overflow_error("SHA-256 bit count overflow");
    }
    Sha256Digest result;
    if (CC_SHA256_Final(result.bytes.data(), &state_) != 1)
        throw std::runtime_error("SHA-256 finalization failed");
    finalized_ = true;
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
