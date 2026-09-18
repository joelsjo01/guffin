#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

[[nodiscard]] std::span<const TestCase> speculative_controller_test_cases();

}  // namespace local_gguf_engine::tests
