#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace local_gguf_engine::benchmarks {

struct BenchmarkConfig {
    std::filesystem::path output_path;
    std::optional<std::filesystem::path> model_path;
    std::size_t iterations = 10;
};

struct BenchmarkArgParseResult {
    BenchmarkConfig config;
    bool should_exit = false;
    int exit_code = 0;
    std::string message;
};

struct BenchmarkMeasurement {
    std::string name;
    std::size_t iterations = 0;
    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    std::size_t bytes_processed = 0;
    std::uint64_t checksum = 0;
    std::string detail;

    [[nodiscard]] double average_ms() const noexcept;
};

struct BenchmarkReport {
    BenchmarkConfig config;
    std::vector<BenchmarkMeasurement> measurements;
};

[[nodiscard]] std::filesystem::path default_benchmark_output_path();
[[nodiscard]] bool benchmark_output_path_is_under_results(const std::filesystem::path& path);
[[nodiscard]] std::string benchmark_usage();
[[nodiscard]] BenchmarkArgParseResult parse_benchmark_args(int argc, const char* const* argv);
[[nodiscard]] BenchmarkReport run_benchmarks(const BenchmarkConfig& config);
[[nodiscard]] std::string format_benchmark_report(const BenchmarkReport& report);
int run_benchmark_cli(int argc, const char* const* argv, std::ostream& output, std::ostream& error);

}  // namespace local_gguf_engine::benchmarks
