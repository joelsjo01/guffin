#pragma once

#include <cstdint>
#include <stdexcept>

namespace local_gguf_engine::tensor {

enum class TensorErrorCode {
    invalid_shape_rank,
    invalid_shape_dimension,
    dimension_index_out_of_range,
    element_count_overflow,
    unsupported_dtype,
    invalid_quantized_element_count,
    byte_size_overflow,
    invalid_storage_byte_count,
    invalid_tensor_index_rank,
    tensor_index_out_of_range,
    unsupported_indexed_dtype,
    unsupported_typed_accessor_dtype,
    unsupported_operator_dtype,
    tensor_shape_mismatch,
    unsupported_quantized_dtype,
    quantized_block_index_out_of_range,
    quantized_block_element_index_out_of_range,
};

class TensorError final : public std::runtime_error {
public:
    TensorError(TensorErrorCode code, std::uint64_t expected_value, std::uint64_t actual_value);

    [[nodiscard]] TensorErrorCode code() const noexcept;
    [[nodiscard]] std::uint64_t expected_value() const noexcept;
    [[nodiscard]] std::uint64_t actual_value() const noexcept;

private:
    TensorErrorCode code_;
    std::uint64_t expected_value_;
    std::uint64_t actual_value_;
};

}  // namespace local_gguf_engine::tensor
