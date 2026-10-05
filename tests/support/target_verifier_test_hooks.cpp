#include "target_verifier_test_hooks.hpp"

#if defined(LOCAL_GGUF_ENGINE_TESTING)

#include <atomic>
#include <stdexcept>

namespace local_gguf_engine::engine {

namespace {
std::atomic<TargetVerifierFailurePoint> target_verifier_failure {
    TargetVerifierFailurePoint::none
};
}

void set_target_verifier_failure_for_testing(TargetVerifierFailurePoint point) noexcept {
    target_verifier_failure.store(point, std::memory_order_release);
}

void clear_target_verifier_failure_for_testing() noexcept {
    set_target_verifier_failure_for_testing(TargetVerifierFailurePoint::none);
}

void inject_target_verifier_failure_for_testing(TargetVerifierFailurePoint point) {
    auto expected = point;
    if (target_verifier_failure.compare_exchange_strong(
            expected, TargetVerifierFailurePoint::none,
            std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        throw std::runtime_error("injected target verifier failure");
    }
}

}  // namespace local_gguf_engine::engine

#endif
