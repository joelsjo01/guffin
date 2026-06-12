#pragma once

#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <stdexcept>
#include <string>

namespace local_gguf_engine::runtime {

enum class BackendErrorCode {
    unsupported_backend,
    unavailable_metal_device,
    metal_execution_failed,
};

class BackendError final : public std::runtime_error {
public:
    BackendError(BackendErrorCode code, BackendKind backend, std::string message);

    [[nodiscard]] BackendErrorCode code() const noexcept;
    [[nodiscard]] BackendKind backend() const noexcept;

private:
    BackendErrorCode code_;
    BackendKind backend_;
};

}  // namespace local_gguf_engine::runtime
