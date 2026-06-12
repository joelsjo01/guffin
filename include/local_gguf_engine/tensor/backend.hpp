#pragma once

#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <string>

namespace local_gguf_engine::tensor {

[[nodiscard]] bool tensor_backend_is_available(runtime::BackendKind backend) noexcept;
[[nodiscard]] std::string tensor_backend_unavailable_reason(runtime::BackendKind backend);

}  // namespace local_gguf_engine::tensor
