#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace local_gguf_engine::benchmarks {

namespace {

[[nodiscard]] std::size_t parse_iteration_count(std::string_view text) {
    std::size_t parsed_characters = 0;
    const auto value = std::stoull(std::string(text), &parsed_characters, 10);
    if (parsed_characters != text.size() || value == 0) {
        throw std::invalid_argument("invalid iteration count");
    }

    if (value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::out_of_range("iteration count is too large");
    }

    return static_cast<std::size_t>(value);
}

[[nodiscard]] BenchmarkArgParseResult parse_error(std::string message) {
    BenchmarkArgParseResult result;
    result.config.output_path = default_benchmark_output_path();
    result.should_exit = true;
    result.exit_code = EXIT_FAILURE;
    result.message = "error: " + std::move(message) + "\n\n" + benchmark_usage();
    return result;
}

[[nodiscard]] bool path_has_prefix(
    const std::filesystem::path& path,
    const std::filesystem::path& prefix
) {
    auto path_iterator = path.begin();
    auto prefix_iterator = prefix.begin();

    for (; prefix_iterator != prefix.end(); ++prefix_iterator, ++path_iterator) {
        if (path_iterator == path.end() || *path_iterator != *prefix_iterator) {
            return false;
        }
    }

    return path_iterator != path.end();
}

}  // namespace

double BenchmarkMeasurement::average_ms() const noexcept {
    if (iterations == 0) {
        return 0.0;
    }

    return total_ms / static_cast<double>(iterations);
}

std::filesystem::path default_benchmark_output_path() {
    return std::filesystem::path("benchmarks") / "results" / "latest.txt";
}

bool benchmark_output_path_is_under_results(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) {
        return false;
    }

    return path_has_prefix(
        path.lexically_normal(),
        (std::filesystem::path("benchmarks") / "results").lexically_normal()
    );
}

std::string benchmark_usage() {
    return "usage: local-gguf-bench [--model <model.gguf>] "
           "[--output benchmarks/results/<name>.txt] [--iterations <count>]\n";
}

BenchmarkArgParseResult parse_benchmark_args(int argc, const char* const* argv) {
    BenchmarkArgParseResult result;
    result.config.output_path = default_benchmark_output_path();

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];

        if (argument == "--help" || argument == "-h") {
            result.should_exit = true;
            result.exit_code = EXIT_SUCCESS;
            result.message = benchmark_usage();
            return result;
        }

        if (argument == "--model") {
            if (index + 1 >= argc) {
                return parse_error("--model requires a path");
            }
            result.config.model_path = std::filesystem::path(argv[++index]);
            continue;
        }

        if (argument == "--output") {
            if (index + 1 >= argc) {
                return parse_error("--output requires a path under benchmarks/results");
            }
            result.config.output_path = std::filesystem::path(argv[++index]);
            continue;
        }

        if (argument == "--iterations") {
            if (index + 1 >= argc) {
                return parse_error("--iterations requires a positive count");
            }
            try {
                result.config.iterations = parse_iteration_count(argv[++index]);
            } catch (const std::exception&) {
                return parse_error("--iterations requires a positive count");
            }
            continue;
        }

        if (!argument.empty() && argument.front() == '-') {
            return parse_error("unknown option '" + std::string(argument) + "'");
        }

        if (result.config.model_path.has_value()) {
            return parse_error("expected at most one model path");
        }
        result.config.model_path = std::filesystem::path(argument);
    }

    if (!benchmark_output_path_is_under_results(result.config.output_path)) {
        return parse_error("benchmark output path must be under benchmarks/results");
    }

    return result;
}

std::string format_benchmark_report(const BenchmarkReport& report) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "local GGUF benchmark report\n";
    output << "iterations: " << report.config.iterations << '\n';
    output << "output: " << report.config.output_path.string() << '\n';
    output << "model: "
           << (report.config.model_path.has_value() ? report.config.model_path->string() : "<none>")
           << '\n';
    output << "benchmarks:\n";

    for (const auto& measurement : report.measurements) {
        output << "- " << measurement.name << '\n';
        output << "  iterations: " << measurement.iterations << '\n';
        output << "  total_ms: " << measurement.total_ms << '\n';
        output << "  avg_ms: " << measurement.average_ms() << '\n';
        output << "  min_ms: " << measurement.min_ms << '\n';
        output << "  max_ms: " << measurement.max_ms << '\n';
        output << "  bytes_processed: " << measurement.bytes_processed << '\n';
        output << "  checksum: " << measurement.checksum << '\n';
        output << "  detail: " << measurement.detail << '\n';
    }

    return output.str();
}

int run_benchmark_cli(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
    const auto parsed = parse_benchmark_args(argc, argv);
    if (parsed.should_exit) {
        auto& stream = parsed.exit_code == EXIT_SUCCESS ? output : error;
        stream << parsed.message;
        return parsed.exit_code;
    }

    try {
        const auto report = run_benchmarks(parsed.config);
        const auto report_text = format_benchmark_report(report);
        std::filesystem::create_directories(parsed.config.output_path.parent_path());

        std::ofstream file(parsed.config.output_path, std::ios::trunc);
        if (!file) {
            throw std::runtime_error("could not open benchmark output '" + parsed.config.output_path.string() + "'");
        }
        file << report_text;
        if (!file) {
            throw std::runtime_error("could not write benchmark output '" + parsed.config.output_path.string() + "'");
        }

        output << report_text;
        output << "wrote benchmark report: " << parsed.config.output_path.string() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        error << "error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}

}  // namespace local_gguf_engine::benchmarks
