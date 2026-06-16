#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

std::span<const TestCase> generate_cli_test_cases();

}  // namespace local_gguf_engine::tests
