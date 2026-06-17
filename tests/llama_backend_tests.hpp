#pragma once

#include "test_support.hpp"

#include <span>

namespace local_gguf_engine::tests {

std::span<const TestCase> llama_backend_attention_test_cases();
std::span<const TestCase> llama_backend_feed_forward_test_cases();
std::span<const TestCase> llama_backend_block_test_cases();
std::span<const TestCase> llama_backend_model_test_cases();

}  // namespace local_gguf_engine::tests
