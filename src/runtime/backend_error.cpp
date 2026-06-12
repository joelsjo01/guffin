#include "local_gguf_engine/runtime/backend_error.hpp"

#include <utility>

namespace local_gguf_engine::runtime {

BackendError::BackendError(BackendErrorCode code, BackendKind backend, std::string message)
    : std::runtime_error(std::move(message)),
      code_(code),
      backend_(backend) {}

BackendErrorCode BackendError::code() const noexcept {
    return code_;
}

BackendKind BackendError::backend() const noexcept {
    return backend_;
}

}  // namespace local_gguf_engine::runtime
