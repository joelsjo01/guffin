#pragma once

namespace local_gguf_engine::tensor::backends::metal {

// Embedded source keeps installed inference independent of the checkout.
[[nodiscard]] const char* metal_kernel_source() noexcept;

}  // namespace local_gguf_engine::tensor::backends::metal
