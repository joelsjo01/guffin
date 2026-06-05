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
