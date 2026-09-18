#pragma once

#if defined(LOCAL_GGUF_ENGINE_TESTING)

namespace local_gguf_engine::tensor::backends::metal {

// Forces the device ownership boundary to behave as if no system Metal device
// exists. This symbol is compiled only into the repository test executable.
void set_metal_device_unavailable_for_testing(bool unavailable) noexcept;

}  // namespace local_gguf_engine::tensor::backends::metal

#endif
