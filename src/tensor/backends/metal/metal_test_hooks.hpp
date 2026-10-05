#pragma once

#if defined(LOCAL_GGUF_ENGINE_TESTING)

#include "local_gguf_engine/tensor/backend_buffer.hpp"

namespace local_gguf_engine::tensor::backends::metal {

// Forces the device ownership boundary to behave as if no system Metal device
// exists. This symbol is compiled only into the repository test executable.
void set_metal_device_unavailable_for_testing(bool unavailable) noexcept;
// Uses an isolated fake device; returns false when the OS lacks the API.
// A supported device deliberately fails allocation to test error propagation.
[[nodiscard]] bool exercise_metal_buffer_residency_for_testing(bool supported);
[[nodiscard]] BackendBuffer shared_metal_buffer_view_for_testing(
    const BackendBuffer& parent, TensorDescriptor descriptor, std::size_t byte_offset
);

}  // namespace local_gguf_engine::tensor::backends::metal

#endif
