#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::tensor::ops {

struct TopLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

[[nodiscard]] float sum_f32(const TensorStorage& input);
[[nodiscard]] float sum_squares_f32(const TensorStorage& input);
[[nodiscard]] float mean_square_f32(const TensorStorage& input);
[[nodiscard]] float rms_f32(const TensorStorage& input);
[[nodiscard]] TopLogit top_logit_f32(const TensorStorage& logits);

}  // namespace local_gguf_engine::tensor::ops
