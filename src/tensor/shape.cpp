#include "local_gguf_engine/tensor/shape.hpp"

#include <limits>

namespace local_gguf_engine::tensor {

namespace {

[[nodiscard]] std::uint64_t checked_u64_multiply(
    std::uint64_t left,
    std::uint64_t right,
    TensorErrorCode error_code
) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw TensorError(error_code, std::numeric_limits<std::uint64_t>::max() / left, right);
    }

    return left * right;
}

[[nodiscard]] std::uint64_t max_size_as_u64() noexcept {
    return static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

void validate_quantized_row_element_count(const Shape& shape, DTypeLayout layout) {
    if (layout.block_element_count == 1) {
        return;
    }

    const auto row_element_count = shape.dimension(shape.rank() - 1);
    if (row_element_count % layout.block_element_count != 0) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            layout.block_element_count,
            row_element_count
        );
    }
}

}  // namespace

Shape Shape::from(std::initializer_list<Dimension> dimensions) {
    return from(std::span<const Dimension>(dimensions.begin(), dimensions.size()));
}

Shape Shape::from(std::span<const Dimension> dimensions) {
    if (dimensions.empty() || dimensions.size() > max_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(max_rank),
            static_cast<std::uint64_t>(dimensions.size())
        );
    }

    Shape shape;
    shape.rank_ = static_cast<std::uint8_t>(dimensions.size());

    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (dimensions[index] == 0) {
            throw TensorError(TensorErrorCode::invalid_shape_dimension, 1, 0);
        }

        shape.dimensions_[index] = dimensions[index];
    }

    return shape;
}

std::size_t Shape::rank() const noexcept {
    return rank_;
}

Shape::Dimension Shape::dimension(std::size_t index) const {
    if (index >= rank()) {
        throw TensorError(
            TensorErrorCode::dimension_index_out_of_range,
            static_cast<std::uint64_t>(rank()),
            static_cast<std::uint64_t>(index)
        );
    }

    return dimensions_[index];
}

std::span<const Shape::Dimension> Shape::dimensions() const noexcept {
    return std::span<const Dimension>(dimensions_.data(), rank());
}

std::uint64_t element_count(const Shape& shape) {
    std::uint64_t count = 1;
    for (const auto dimension : shape.dimensions()) {
        count = checked_u64_multiply(count, dimension, TensorErrorCode::element_count_overflow);
    }

    return count;
}

std::size_t byte_size(const Shape& shape, DType dtype) {
    const auto layout = dtype_layout(dtype);
    validate_quantized_row_element_count(shape, layout);

    const auto elements = element_count(shape);

    if (elements % layout.block_element_count != 0) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            layout.block_element_count,
            elements
        );
    }

    const auto block_count = elements / layout.block_element_count;
    const auto byte_count = checked_u64_multiply(
        block_count,
        layout.block_byte_count,
        TensorErrorCode::byte_size_overflow
    );

    if (byte_count > max_size_as_u64()) {
        throw TensorError(TensorErrorCode::byte_size_overflow, max_size_as_u64(), byte_count);
    }

    return static_cast<std::size_t>(byte_count);
}

}  // namespace local_gguf_engine::tensor
