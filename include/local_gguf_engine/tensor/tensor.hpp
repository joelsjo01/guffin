#pragma once

#include "local_gguf_engine/tensor/shape.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::tensor {

struct TensorDescriptor {
    Shape shape;
    DType dtype = DType::f32;
    std::uint64_t element_count = 0;
    std::size_t byte_count = 0;

    [[nodiscard]] static TensorDescriptor from(const Shape& shape, DType dtype);
    [[nodiscard]] static TensorDescriptor from_quantized_gguf_weight_matrix(
        const Shape& shape,
        DType dtype
    );

    constexpr bool operator==(const TensorDescriptor&) const = default;

private:
    TensorDescriptor(Shape shape, DType dtype, std::uint64_t element_count, std::size_t byte_count);
};

class TensorView;

class TensorStorage final {
public:
    [[nodiscard]] static TensorStorage allocate(const TensorDescriptor& descriptor);
    [[nodiscard]] static TensorStorage from_bytes(
        const TensorDescriptor& descriptor,
        std::span<const std::uint8_t> bytes
    );

    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept;
    [[nodiscard]] std::span<std::uint8_t> mutable_bytes() noexcept;
    [[nodiscard]] TensorView view() const noexcept;

private:
    TensorStorage(TensorDescriptor descriptor, std::vector<std::uint8_t> bytes);

    TensorDescriptor descriptor_;
    std::vector<std::uint8_t> bytes_;
};

class TensorView final {
public:
    // Borrowed tensor bytes; the mapped file or storage backing the span must outlive this view.
    [[nodiscard]] static TensorView from_bytes(
        const TensorDescriptor& descriptor,
        std::span<const std::uint8_t> bytes
    );

    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept;

private:
    TensorView(TensorDescriptor descriptor, std::span<const std::uint8_t> bytes);

    TensorDescriptor descriptor_;
    std::span<const std::uint8_t> bytes_;
};

}  // namespace local_gguf_engine::tensor
