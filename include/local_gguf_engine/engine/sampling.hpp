#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>

namespace local_gguf_engine::engine {

[[nodiscard]] std::uint32_t sample_argmax_token_f32(const tensor::TensorStorage& logits);

}  // namespace local_gguf_engine::engine
