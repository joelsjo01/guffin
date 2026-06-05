#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <string>

namespace local_gguf_engine::tensor {

namespace {

[[nodiscard]] std::string error_message(TensorErrorCode code) {
    switch (code) {
        case TensorErrorCode::invalid_shape_rank:
            return "invalid tensor shape rank";
        case TensorErrorCode::invalid_shape_dimension:
            return "invalid tensor shape dimension";
        case TensorErrorCode::dimension_index_out_of_range:
            return "tensor shape dimension index out of range";
        case TensorErrorCode::element_count_overflow:
            return "tensor shape element count overflow";
        case TensorErrorCode::unsupported_dtype:
            return "unsupported tensor dtype";
        case TensorErrorCode::invalid_quantized_element_count:
            return "invalid quantized tensor element count";
        case TensorErrorCode::byte_size_overflow:
            return "tensor byte size overflow";
        case TensorErrorCode::invalid_storage_byte_count:
            return "invalid tensor storage byte count";
    }
}

}  // namespace

TensorError::TensorError(
    TensorErrorCode code,
    std::uint64_t expected_value,
    std::uint64_t actual_value
)
    : std::runtime_error(error_message(code)),
      code_(code),
      expected_value_(expected_value),
      actual_value_(actual_value) {}

TensorErrorCode TensorError::code() const noexcept {
    return code_;
}

std::uint64_t TensorError::expected_value() const noexcept {
    return expected_value_;
}

std::uint64_t TensorError::actual_value() const noexcept {
    return actual_value_;
}

}  // namespace local_gguf_engine::tensor
