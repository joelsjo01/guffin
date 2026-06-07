#include "local_gguf_engine/tensor/ops/elementwise.hpp"

#include "validation.hpp"

#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor::ops {

namespace {

[[nodiscard]] std::array<std::uint64_t, Shape::max_rank> indices_from_linear_index(
    const Shape& shape,
    std::uint64_t linear_index
) {
    std::array<std::uint64_t, Shape::max_rank> indices {};

    for (std::size_t reverse_index = shape.rank(); reverse_index > 0; --reverse_index) {
        const auto dimension_index = reverse_index - 1;
        const auto dimension = shape.dimension(dimension_index);
        indices[dimension_index] = linear_index % dimension;
        linear_index /= dimension;
    }

    return indices;
}

}  // namespace

TensorStorage add_f32(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f32_operand(right);
    require_same_shape(left.descriptor(), right.descriptor());

    auto output = TensorStorage::allocate(left.descriptor());
    for (std::uint64_t linear_index = 0; linear_index < left.descriptor().element_count; ++linear_index) {
        const auto indices = indices_from_linear_index(left.descriptor().shape, linear_index);
        const auto index_view = std::span<const std::uint64_t>(indices.data(), left.descriptor().shape.rank());
        write_f32(output, index_view, read_f32(left, index_view) + read_f32(right, index_view));
    }

    return output;
}

TensorStorage multiply_f32(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f32_operand(right);
    require_same_shape(left.descriptor(), right.descriptor());

    auto output = TensorStorage::allocate(left.descriptor());
    for (std::uint64_t linear_index = 0; linear_index < left.descriptor().element_count; ++linear_index) {
        const auto indices = indices_from_linear_index(left.descriptor().shape, linear_index);
        const auto index_view = std::span<const std::uint64_t>(indices.data(), left.descriptor().shape.rank());
        write_f32(output, index_view, read_f32(left, index_view) * read_f32(right, index_view));
    }

    return output;
}

TensorStorage silu_f32(const TensorStorage& input) {
    require_f32_operand(input);

    auto output = TensorStorage::allocate(input.descriptor());
    for (std::uint64_t linear_index = 0; linear_index < input.descriptor().element_count; ++linear_index) {
        const auto indices = indices_from_linear_index(input.descriptor().shape, linear_index);
        const auto index_view = std::span<const std::uint64_t>(indices.data(), input.descriptor().shape.rank());
        const auto value = read_f32(input, index_view);
        write_f32(output, index_view, value / (1.0F + std::exp(-value)));
    }

    return output;
}

TensorStorage copy_f32(const TensorStorage& input) {
    require_f32_operand(input);
    return TensorStorage::from_bytes(input.descriptor(), input.bytes());
}

TensorStorage fill_f32(const TensorDescriptor& descriptor, float value) {
    require_f32_descriptor(descriptor);

    auto output = TensorStorage::allocate(descriptor);
    for (std::uint64_t linear_index = 0; linear_index < descriptor.element_count; ++linear_index) {
        const auto indices = indices_from_linear_index(descriptor.shape, linear_index);
        const auto index_view = std::span<const std::uint64_t>(indices.data(), descriptor.shape.rank());
        write_f32(output, index_view, value);
    }

    return output;
}

}  // namespace local_gguf_engine::tensor::ops
