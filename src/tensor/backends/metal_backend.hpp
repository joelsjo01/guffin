#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <string>

namespace local_gguf_engine::tensor::backends::metal {

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string unavailable_reason();

[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
);

}  // namespace local_gguf_engine::tensor::backends::metal
