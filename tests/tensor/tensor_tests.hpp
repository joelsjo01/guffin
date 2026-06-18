#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

std::span<const TestCase> tensor_core_test_cases();
std::span<const TestCase> tensor_post_backend_test_cases();

}  // namespace local_gguf_engine::tests
