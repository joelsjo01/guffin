#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] TensorStorage scaled_dot_product_attention_f32(
    const TensorStorage& query,
    const TensorStorage& key,
    const TensorStorage& value,
    float scale
);

}  // namespace local_gguf_engine::tensor::ops
