#pragma once

#if defined(LOCAL_GGUF_ENGINE_TESTING)

#include <cstdint>

namespace local_gguf_engine::engine {

enum class TargetVerifierFailurePoint : std::uint8_t {
    none,
    allocation,
    encode,
    commit,
    wait,
    readback,
    restore,
    finalize,
};

void set_target_verifier_failure_for_testing(TargetVerifierFailurePoint point) noexcept;
void clear_target_verifier_failure_for_testing() noexcept;
void inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint point);

}  // namespace local_gguf_engine::engine

#endif
