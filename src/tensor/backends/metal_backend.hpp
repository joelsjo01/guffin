#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <string>
#include <cstdint>
#include <optional>

namespace local_gguf_engine::tensor::backends::metal {

[[nodiscard]] bool is_available() noexcept;
[[nodiscard]] std::string unavailable_reason();
struct MetalMemoryInfo {
    std::uint64_t physical_memory_bytes = 0;
    std::uint64_t recommended_working_set_bytes = 0;
    std::uint64_t max_buffer_bytes = 0;
    std::uint64_t current_allocated_bytes = 0;
};
// The recommended working set is advisory, not free memory or a reservation.
[[nodiscard]] std::optional<MetalMemoryInfo> memory_info() noexcept;
[[nodiscard]] std::optional<std::uint64_t> current_allocated_size_bytes() noexcept;

[[nodiscard]] TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
);
[[nodiscard]] TensorStorage matmul_f32_quantized_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
);

}  // namespace local_gguf_engine::tensor::backends::metal
