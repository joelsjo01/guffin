#pragma once

#if defined(LOCAL_GGUF_ENGINE_TESTING)

#include "local_gguf_engine/tensor/backend_buffer.hpp"

namespace local_gguf_engine::tensor::backends::metal {

// Forces the device ownership boundary to behave as if no system Metal device
// exists. This symbol is compiled only into the repository test executable.
void set_metal_device_unavailable_for_testing(bool unavailable) noexcept;
[[nodiscard]] BackendBuffer shared_metal_buffer_view_for_testing(
    const BackendBuffer& parent, TensorDescriptor descriptor, std::size_t byte_offset
);

}  // namespace local_gguf_engine::tensor::backends::metal

#endif
