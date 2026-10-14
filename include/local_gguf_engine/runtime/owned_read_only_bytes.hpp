#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::runtime {

// Explicit shared ownership of a readable allocation. The logical span excludes
// page padding; accessible_byte_count includes it for native buffer registration.
// Read-only access does not freeze external changes to a memory-mapped file.
class OwnedReadOnlyBytes final {
public:
    OwnedReadOnlyBytes() = default;
    OwnedReadOnlyBytes(std::shared_ptr<const void> owner,
        std::span<const std::uint8_t> bytes, std::size_t accessible_byte_count,
        std::function<bool(std::size_t, std::size_t)> read_ahead = {})
        : owner_(std::move(owner)), bytes_(bytes), accessible_byte_count_(accessible_byte_count),
          read_ahead_(std::move(read_ahead)) {
        if (!owner_ || bytes_.empty() || accessible_byte_count_ < bytes_.size())
            throw std::invalid_argument("owned read-only bytes require a nonempty owned allocation");
    }

    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
        return owner_ ? bytes_ : std::span<const std::uint8_t>{};
    }
    [[nodiscard]] std::size_t accessible_byte_count() const noexcept {
        return owner_ ? accessible_byte_count_ : 0U;
    }
    [[nodiscard]] const std::shared_ptr<const void>& owner() const noexcept { return owner_; }

    // Optional asynchronous advice does not copy bytes or change ownership.
    // Unsupported hints and invalid logical extents are harmless failures.
    [[nodiscard]] bool request_read_ahead(std::size_t offset, std::size_t length) const noexcept {
        if (!owner_ || !read_ahead_ || length == 0U || offset > bytes_.size()
            || length > bytes_.size() - offset) return false;
        try {
            return read_ahead_(offset, length);
        } catch (...) {
            return false;
        }
    }

private:
    std::shared_ptr<const void> owner_;
    std::span<const std::uint8_t> bytes_;
    std::size_t accessible_byte_count_ = 0;
    std::function<bool(std::size_t, std::size_t)> read_ahead_;
};

}  // namespace local_gguf_engine::runtime
