#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

std::span<const TestCase> gemma2_metal_plan_test_cases();

}  // namespace local_gguf_engine::tests
