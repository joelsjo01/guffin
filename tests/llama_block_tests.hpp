#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_block_primary_test_cases();
std::span<const TestCase> llama_block_validation_test_cases();

}  // namespace local_gguf_engine::tests
