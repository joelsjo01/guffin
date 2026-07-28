#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {
std::span<const TestCase> conversation_benchmark_test_cases();
}
