#pragma once

#include "local_gguf_engine/tensor/layout.hpp"

#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor {

[[nodiscard]] float read_f32(const TensorStorage& storage, std::span<const std::uint64_t> indices);
void write_f32(TensorStorage& storage, std::span<const std::uint64_t> indices, float value);
[[nodiscard]] float read_f16_as_f32(const TensorStorage& storage, std::span<const std::uint64_t> indices);
void write_f16_from_f32(TensorStorage& storage, std::span<const std::uint64_t> indices, float value);

}  // namespace local_gguf_engine::tensor
