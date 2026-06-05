#include "local_gguf_engine/model_format/gguf_header.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using local_gguf_engine::model_format::GGUFHeader;
using local_gguf_engine::model_format::GGUFReadError;
using local_gguf_engine::model_format::GGUFReadErrorCode;
using local_gguf_engine::model_format::read_gguf_header;

class TestFailure final : public std::exception {
public:
    explicit TestFailure(std::string message) : message_(std::move(message)) {}

    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

std::vector<std::uint8_t> load_hex_fixture(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw TestFailure("fixture missing: " + path.string());
    }

    std::vector<std::uint8_t> bytes;
    std::string token;
    while (input >> token) {
        const auto value = std::stoul(token, nullptr, 16);
        if (value > 0xFFU) {
            throw TestFailure("fixture byte is out of range: " + token);
        }
        bytes.push_back(static_cast<std::uint8_t>(value));
    }

    return bytes;
}

std::vector<std::uint8_t> fixture(std::string_view name) {
    return load_hex_fixture(std::filesystem::path("tests") / "fixtures" / (std::string(name) + ".hex"));
}

void test_default_backend_starts_on_cpu() {
    expect(
        local_gguf_engine::runtime::default_backend()
            == local_gguf_engine::runtime::BackendKind::cpu,
        "default backend should start on CPU"
    );
}

void test_reads_valid_gguf_fixed_header_fixture() {
    const auto bytes = fixture("valid-header-v3");
    const auto header = read_gguf_header(std::span<const std::uint8_t>(bytes));

    expect(
        header == GGUFHeader {.version = 3, .tensor_count = 2, .metadata_kv_count = 5},
        "GGUF header should parse magic, version, tensor count, and metadata kv count"
    );
}

void test_reads_little_endian_header_counts() {
    const std::array<std::uint8_t, GGUFHeader::encoded_byte_count> bytes {
        0x47, 0x47, 0x55, 0x46,
        0x03, 0x00, 0x00, 0x00,
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
    };

    const auto header = read_gguf_header(std::span<const std::uint8_t>(bytes));

    expect(header.version == 3, "GGUF header version should be 3");
    expect(header.tensor_count == 0x0102030405060708ULL, "tensor count should parse as little-endian uint64");
    expect(
        header.metadata_kv_count == 0x1122334455667788ULL,
        "metadata kv count should parse as little-endian uint64"
    );
}

void test_rejects_invalid_gguf_magic() {
    const auto bytes = fixture("invalid-magic");

    try {
        (void)read_gguf_header(std::span<const std::uint8_t>(bytes));
        throw TestFailure("invalid magic should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::invalid_magic, "error code should be invalid_magic");
        expect(error.actual_magic() == std::array<std::uint8_t, 4> {0x42, 0x41, 0x44, 0x21}, "actual magic mismatch");
    }
}

void test_rejects_truncated_gguf_header() {
    const auto bytes = fixture("truncated-header");

    try {
        (void)read_gguf_header(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated header should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::truncated_header, "error code should be truncated_header");
        expect(error.expected_byte_count() == GGUFHeader::encoded_byte_count, "expected byte count mismatch");
        expect(error.actual_byte_count() == 3, "actual byte count mismatch");
    }
}

void test_rejects_truncated_count_data() {
    const auto bytes = fixture("truncated-count-data");

    try {
        (void)read_gguf_header(std::span<const std::uint8_t>(bytes));
        throw TestFailure("truncated count data should throw");
    } catch (const GGUFReadError& error) {
        expect(error.code() == GGUFReadErrorCode::truncated_header, "error code should be truncated_header");
        expect(error.expected_byte_count() == GGUFHeader::encoded_byte_count, "expected byte count mismatch");
        expect(error.actual_byte_count() == 16, "actual byte count mismatch");
    }
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"default backend starts on CPU", test_default_backend_starts_on_cpu},
        {"reads valid GGUF fixed header fixture", test_reads_valid_gguf_fixed_header_fixture},
        {"reads little-endian GGUF header counts", test_reads_little_endian_header_counts},
        {"rejects invalid GGUF magic", test_rejects_invalid_gguf_magic},
        {"rejects truncated GGUF header", test_rejects_truncated_gguf_header},
        {"rejects truncated GGUF count data", test_rejects_truncated_count_data},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << std::size(tests) << " test(s) passed\n";
    return EXIT_SUCCESS;
}
