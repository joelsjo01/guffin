#include "local_gguf_engine/tensor/backend.hpp"

#include "backends/cpu_backend.hpp"
#include "backends/metal_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

namespace local_gguf_engine::tensor {

bool tensor_backend_is_available(runtime::BackendKind backend) noexcept {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::is_available();
        case runtime::BackendKind::metal:
            return backends::metal::is_available();
    }

    return false;
}

std::string tensor_backend_unavailable_reason(runtime::BackendKind backend) {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::unavailable_reason();
        case runtime::BackendKind::metal:
            return backends::metal::unavailable_reason();
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported tensor backend"
    );
}

}  // namespace local_gguf_engine::tensor
