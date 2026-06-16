#pragma once

#include <exception>
#include <string>
#include <string_view>

namespace local_gguf_engine::tests {

class TestFailure final : public std::exception {
public:
    explicit TestFailure(std::string message);

    [[nodiscard]] const char* what() const noexcept override;

private:
    std::string message_;
};

struct TestCase {
    std::string_view name;
    void (*run)();
};

void expect(bool condition, std::string_view message);

}  // namespace local_gguf_engine::tests
