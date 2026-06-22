#include "local_gguf_engine/tensor/layout.hpp"

namespace local_gguf_engine::tensor {

std::size_t unquantized_element_byte_count(DType dtype) {
    switch (dtype) {
        case DType::f32:
            return 4;
        case DType::f16:
            return 2;
        case DType::q8_0:
        case DType::q4_k:
        case DType::q6_k:
            throw TensorError(
                TensorErrorCode::unsupported_indexed_dtype,
                0,
                static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype))
            );
    }

    throw TensorError(
        TensorErrorCode::unsupported_dtype,
        0,
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype))
    );
}

std::uint64_t linear_index(const Shape& shape, std::span<const std::uint64_t> indices) {
    if (indices.size() != shape.rank()) {
        throw TensorError(
            TensorErrorCode::invalid_tensor_index_rank,
            static_cast<std::uint64_t>(shape.rank()),
            static_cast<std::uint64_t>(indices.size())
        );
    }

    std::uint64_t index = 0;
    for (std::size_t dimension_index = 0; dimension_index < indices.size(); ++dimension_index) {
        const auto dimension = shape.dimension(dimension_index);
        const auto index_in_dimension = indices[dimension_index];
        if (index_in_dimension >= dimension) {
            throw TensorError(TensorErrorCode::tensor_index_out_of_range, dimension, index_in_dimension);
        }

        index = index * dimension + index_in_dimension;
    }

    return index;
}

std::size_t element_byte_offset(
    const TensorDescriptor& descriptor,
    std::span<const std::uint64_t> indices
) {
    const auto element_size = unquantized_element_byte_count(descriptor.dtype);
    const auto offset = linear_index(descriptor.shape, indices) * static_cast<std::uint64_t>(element_size);
    return static_cast<std::size_t>(offset);
}

std::span<const std::uint8_t> element_bytes(
    const TensorStorage& storage,
    std::span<const std::uint64_t> indices
) {
    const auto element_size = unquantized_element_byte_count(storage.descriptor().dtype);
    const auto offset = element_byte_offset(storage.descriptor(), indices);
    return storage.bytes().subspan(offset, element_size);
}

std::span<std::uint8_t> mutable_element_bytes(
    TensorStorage& storage,
    std::span<const std::uint64_t> indices
) {
    const auto element_size = unquantized_element_byte_count(storage.descriptor().dtype);
    const auto offset = element_byte_offset(storage.descriptor(), indices);
    return storage.mutable_bytes().subspan(offset, element_size);
}

}  // namespace local_gguf_engine::tensor
