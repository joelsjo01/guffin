#include "test_support.hpp"

#include <utility>

namespace local_gguf_engine::tests {

TestFailure::TestFailure(std::string message) : message_(std::move(message)) {}

const char* TestFailure::what() const noexcept {
    return message_.c_str();
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

}  // namespace local_gguf_engine::tests
