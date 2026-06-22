#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor::ops {

[[nodiscard]] BackendBuffer embed_f16_gguf_weight_token_ids_to_backend(
    std::span<const std::uint32_t> token_ids,
    const BackendBuffer& embedding_weight,
    float scale
);

}  // namespace local_gguf_engine::tensor::ops
