#include "prepared_plan_test_hooks.hpp"

#if defined(LOCAL_GGUF_ENGINE_TESTING)

#include <atomic>
#include <stdexcept>

namespace local_gguf_engine::engine {
namespace {

std::atomic<PreparedPlanFailurePoint> failure_point {PreparedPlanFailurePoint::none};
std::atomic<std::uint32_t> failure_occurrence {0U};

}  // namespace

void set_prepared_plan_failure_for_testing(
    PreparedPlanFailurePoint point,
    std::uint32_t occurrence
) noexcept {
    failure_occurrence.store(occurrence, std::memory_order_release);
    failure_point.store(point, std::memory_order_release);
}

void clear_prepared_plan_failure_for_testing() noexcept {
    failure_point.store(PreparedPlanFailurePoint::none, std::memory_order_release);
}

void inject_prepared_plan_failure_for_testing(
    PreparedPlanFailurePoint point,
    std::uint32_t occurrence
) {
    if (failure_occurrence.load(std::memory_order_acquire) != occurrence) return;
    auto expected = point;
    if (failure_point.compare_exchange_strong(
        expected, PreparedPlanFailurePoint::none,
        std::memory_order_acq_rel, std::memory_order_acquire
    )) {
        throw std::runtime_error("injected prepared-plan failure");
    }
}

}  // namespace local_gguf_engine::engine

#endif
