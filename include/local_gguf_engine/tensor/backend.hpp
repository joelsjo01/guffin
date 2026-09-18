#pragma once

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"

#include <string>

namespace local_gguf_engine::tensor {

[[nodiscard]] bool tensor_backend_is_available(runtime::BackendKind backend) noexcept;
[[nodiscard]] std::string tensor_backend_unavailable_reason(runtime::BackendKind backend);
// Identifies the exact Metal source compiled into this binary. Trace/artifact
// receipts use this rather than assuming a dirty Git suffix identifies the
// kernel bytes.
[[nodiscard]] runtime::Sha256Digest metal_backend_library_source_digest();

}  // namespace local_gguf_engine::tensor
