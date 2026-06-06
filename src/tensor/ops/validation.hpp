#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

inline void require_f32_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.dtype != DType::f32) {
        throw TensorError(
            TensorErrorCode::unsupported_operator_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f32)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

inline void require_f32_operand(const TensorStorage& storage) {
    require_f32_descriptor(storage.descriptor());
}

inline void require_f16_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.dtype != DType::f16) {
        throw TensorError(
            TensorErrorCode::unsupported_operator_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f16)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

inline void require_f16_operand(const TensorStorage& storage) {
    require_f16_descriptor(storage.descriptor());
}

inline void require_same_shape(const TensorDescriptor& left, const TensorDescriptor& right) {
    if (left.shape.rank() != right.shape.rank()) {
        throw TensorError(
            TensorErrorCode::tensor_shape_mismatch,
            static_cast<std::uint64_t>(left.shape.rank()),
            static_cast<std::uint64_t>(right.shape.rank())
        );
    }

    for (std::size_t index = 0; index < left.shape.rank(); ++index) {
        const auto left_dimension = left.shape.dimension(index);
        const auto right_dimension = right.shape.dimension(index);
        if (left_dimension != right_dimension) {
            throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_dimension, right_dimension);
        }
    }
}

}  // namespace local_gguf_engine::tensor::ops
