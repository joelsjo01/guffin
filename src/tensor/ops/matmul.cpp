#include "local_gguf_engine/tensor/ops/matmul.hpp"

#include "validation.hpp"
#include "../backends/cpu_backend.hpp"
#include "../backends/metal_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/f16.hpp"

#include <cstring>

namespace local_gguf_engine::tensor::ops {

namespace {

constexpr std::size_t matrix_rank = 2;

void require_matrix_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.shape.rank() != matrix_rank) {
        throw TensorError(
            TensorErrorCode::invalid_shape_rank,
            static_cast<std::uint64_t>(matrix_rank),
            static_cast<std::uint64_t>(descriptor.shape.rank())
        );
    }
}

void require_matmul_inner_dimensions(const TensorDescriptor& left, const TensorDescriptor& right) {
    const auto left_columns = left.shape.dimension(1);
    const auto right_rows = right.shape.dimension(0);
    if (left_columns != right_rows) {
        throw TensorError(TensorErrorCode::tensor_shape_mismatch, left_columns, right_rows);
    }
}

void require_f32_f16_matmul_gguf_weight_operands(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f16_operand(right);
    require_matrix_descriptor(left.descriptor());
    require_matrix_descriptor(right.descriptor());
    require_matmul_inner_dimensions(left.descriptor(), right.descriptor());
}

void require_quantized_gguf_weight_descriptor(const TensorDescriptor& descriptor) {
    if (descriptor.dtype == DType::q8_0 || descriptor.dtype == DType::q4_k || descriptor.dtype == DType::q6_k) {
        return;
    }

    throw TensorError(
        TensorErrorCode::unsupported_operator_dtype,
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::q4_k)),
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
    );
}

void require_quantized_gguf_weight_inner_block_alignment(const TensorDescriptor& descriptor) {
    const auto block_elements = dtype_layout(descriptor.dtype).block_element_count;
    const auto inner_width = descriptor.shape.dimension(0);
    if (inner_width % block_elements != 0) {
        throw TensorError(
            TensorErrorCode::invalid_quantized_element_count,
            block_elements,
            inner_width
        );
    }
}

void require_f32_quantized_matmul_gguf_weight_operands(
    const TensorDescriptor& left,
    const TensorDescriptor& right
) {
    require_f32_descriptor(left);
    require_quantized_gguf_weight_descriptor(right);
    require_matrix_descriptor(left);
    require_matrix_descriptor(right);
    require_matmul_inner_dimensions(left, right);
    require_quantized_gguf_weight_inner_block_alignment(right);
}

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

[[nodiscard]] float read_f16_linear_as_f32(const TensorStorage& storage, std::uint64_t linear_index) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(std::uint16_t));
    const auto bytes = storage.bytes();
    const auto bits = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[byte_offset])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[byte_offset + 1]) << 8U)
    );
    return f16_bits_to_f32(bits);
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
}

template <typename RightReader>
[[nodiscard]] TensorStorage matmul_f32_with_right_reader(
    const TensorStorage& left,
    const TensorStorage& right,
    RightReader read_right
) {
    require_matrix_descriptor(left.descriptor());
    require_matrix_descriptor(right.descriptor());
    require_matmul_inner_dimensions(left.descriptor(), right.descriptor());

    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::uint64_t inner = 0; inner < shared; ++inner) {
                const auto left_index = row * shared + inner;
                const auto right_index = inner * columns + column;
                sum += read_f32_linear(left, left_index) * read_right(right, right_index);
            }

            write_f32_linear(output, row * columns + column, sum);
        }
    }

    return output;
}

}  // namespace

TensorStorage matmul_f32(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f32_operand(right);
    return matmul_f32_with_right_reader(left, right, read_f32_linear);
}

TensorStorage matmul_f32_f16(const TensorStorage& left, const TensorStorage& right) {
    require_f32_operand(left);
    require_f16_operand(right);
    return matmul_f32_with_right_reader(left, right, read_f16_linear_as_f32);
}

TensorStorage matmul_f32_f16_gguf_weight(const TensorStorage& left, const TensorStorage& right) {
    return matmul_f32_f16_gguf_weight(left, right, runtime::default_backend());
}

TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right,
    runtime::BackendKind backend
) {
    require_f32_f16_matmul_gguf_weight_operands(left, right);

    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_f16_gguf_weight(left, right);
        case runtime::BackendKind::metal:
            return backends::metal::matmul_f32_f16_gguf_weight(left, right);
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for F32xF16 GGUF weight matmul"
    );
}

TensorStorage matmul_f32_quantized_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
) {
    return matmul_f32_quantized_gguf_weight(left, right, runtime::default_backend());
}

TensorStorage matmul_f32_quantized_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right,
    runtime::BackendKind backend
) {
    require_f32_quantized_matmul_gguf_weight_operands(left.descriptor(), right.descriptor());

    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::matmul_f32_quantized_gguf_weight(left, right);
        case runtime::BackendKind::metal:
            return backends::metal::matmul_f32_quantized_gguf_weight(left, right);
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for F32xK-quant GGUF weight matmul"
    );
}

}  // namespace local_gguf_engine::tensor::ops
