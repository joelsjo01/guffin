#pragma once

#include "test_support.hpp"
#include <span>

namespace local_gguf_engine::tests {
std::span<const TestCase> standalone_chat_test_cases();
}
