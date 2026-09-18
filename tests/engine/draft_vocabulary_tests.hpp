#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

[[nodiscard]] std::span<const TestCase> draft_vocabulary_test_cases();

}  // namespace local_gguf_engine::tests
