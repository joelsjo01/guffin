#pragma once

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace local_gguf_engine::runtime {

struct Sha256Digest {
    std::array<std::uint8_t, 32> bytes {};

    constexpr bool operator==(const Sha256Digest&) const = default;
    [[nodiscard]] bool is_zero() const noexcept;
    [[nodiscard]] std::string hex() const;
};

class Sha256 final {
public:
    Sha256();

    void update(std::span<const std::uint8_t> bytes);
    [[nodiscard]] Sha256Digest finalize();

private:
    CC_SHA256_CTX state_ {};
    std::uint64_t byte_count_ = 0;
    bool finalized_ = false;
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> bytes);
[[nodiscard]] Sha256Digest parse_sha256_hex(std::string_view text);

}  // namespace local_gguf_engine::runtime
