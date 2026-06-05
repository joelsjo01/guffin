#pragma once

#include "local_gguf_engine/tensor/dtype.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

namespace local_gguf_engine::tensor {

class Shape final {
public:
    using Dimension = std::uint64_t;

    static constexpr std::size_t max_rank = 4;

    [[nodiscard]] static Shape from(std::initializer_list<Dimension> dimensions);
    [[nodiscard]] static Shape from(std::span<const Dimension> dimensions);

    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] Dimension dimension(std::size_t index) const;
    [[nodiscard]] std::span<const Dimension> dimensions() const noexcept;

    constexpr bool operator==(const Shape&) const = default;

private:
    Shape() = default;

    std::array<Dimension, max_rank> dimensions_ {};
    std::uint8_t rank_ = 0;
};

[[nodiscard]] std::uint64_t element_count(const Shape& shape);
[[nodiscard]] std::size_t byte_size(const Shape& shape, DType dtype);

}  // namespace local_gguf_engine::tensor
