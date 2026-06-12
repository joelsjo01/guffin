#pragma once

namespace local_gguf_engine::runtime {

enum class BackendKind {
    cpu,
    metal,
};

[[nodiscard]] constexpr BackendKind default_backend() noexcept {
    return BackendKind::cpu;
}

}  // namespace local_gguf_engine::runtime
