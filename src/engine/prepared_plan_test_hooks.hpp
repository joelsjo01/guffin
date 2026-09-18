#pragma once

#include <cstdint>

namespace local_gguf_engine::engine {

#if defined(LOCAL_GGUF_ENGINE_TESTING)

enum class PreparedPlanFailurePoint : std::uint8_t {
    none,
    prefill_encode,
    prefill_commit,
    prefill_wait,
    prefill_completion,
};

void set_prepared_plan_failure_for_testing(
    PreparedPlanFailurePoint point,
    std::uint32_t occurrence
) noexcept;
void clear_prepared_plan_failure_for_testing() noexcept;
void inject_prepared_plan_failure_for_testing(
    PreparedPlanFailurePoint point,
    std::uint32_t occurrence
);

#endif

}  // namespace local_gguf_engine::engine
