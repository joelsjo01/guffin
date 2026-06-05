#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace local_gguf_engine::benchmarks {

namespace {

using Clock = std::chrono::steady_clock;
using model_format::load_gguf_tensor_payloads;
using model_format::read_gguf_model_view;
using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorStorage;
using tensor::write_f32;
using tensor::ops::add_f32;
using tensor::ops::sum_f32;

struct IterationMeasurement {
    std::size_t bytes_processed = 0;
    std::uint64_t checksum = 0;
};

[[nodiscard]] std::uint64_t checksum_byte_span(std::span<const std::uint8_t> bytes) noexcept {
    if (bytes.empty()) {
        return 0;
    }

    return static_cast<std::uint64_t>(bytes.size())
        ^ static_cast<std::uint64_t>(bytes.front())
        ^ (static_cast<std::uint64_t>(bytes.back()) << 8);
}

[[nodiscard]] std::uint64_t checksum_f32(float value) noexcept {
    return static_cast<std::uint64_t>(value >= 0.0F ? value * 1000.0F : -value * 1000.0F);
}

template <typename Callable>
[[nodiscard]] BenchmarkMeasurement measure(
    std::string name,
    std::size_t iterations,
    std::string detail,
    Callable callable
) {
    BenchmarkMeasurement measurement {
        .name = std::move(name),
        .iterations = iterations,
        .total_ms = 0.0,
        .min_ms = std::numeric_limits<double>::max(),
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = std::move(detail),
    };

    for (std::size_t index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        const auto iteration = callable();
        const auto end = Clock::now();
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        measurement.total_ms += elapsed_ms;
        measurement.min_ms = std::min(measurement.min_ms, elapsed_ms);
        measurement.max_ms = std::max(measurement.max_ms, elapsed_ms);
        measurement.bytes_processed += iteration.bytes_processed;
        measurement.checksum ^= iteration.checksum + index;
    }

    if (iterations == 0) {
        measurement.min_ms = 0.0;
    }

    return measurement;
}

[[nodiscard]] TensorStorage make_f32_storage(std::uint64_t element_count, float start_value) {
    auto storage = TensorStorage::allocate(TensorDescriptor::from(Shape::from({element_count}), DType::f32));
    for (std::uint64_t index = 0; index < element_count; ++index) {
        const std::array<std::uint64_t, 1> tensor_index {index};
        write_f32(storage, std::span<const std::uint64_t>(tensor_index), start_value + static_cast<float>(index));
    }

    return storage;
}

[[nodiscard]] std::vector<std::uint8_t> make_repeated_payload(std::size_t byte_count) {
    std::vector<std::uint8_t> bytes(byte_count);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(index % 251);
    }

    return bytes;
}

void add_synthetic_tensor_benchmarks(BenchmarkReport& report) {
    constexpr std::uint64_t add_element_count = 4096;
    constexpr std::uint64_t copy_element_count = 16 * 1024;

    const auto add_descriptor = TensorDescriptor::from(Shape::from({add_element_count}), DType::f32);
    const auto copy_descriptor = TensorDescriptor::from(Shape::from({copy_element_count}), DType::f32);
    const auto copy_payload = make_repeated_payload(copy_descriptor.byte_count);
    const auto add_left = make_f32_storage(add_element_count, 1.0F);
    const auto add_right = make_f32_storage(add_element_count, -0.5F);

    report.measurements.push_back(measure(
        "tensor.allocate_f32",
        report.config.iterations,
        "shape=[4096] dtype=F32",
        [add_descriptor]() {
            const auto storage = TensorStorage::allocate(add_descriptor);
            return IterationMeasurement {
                .bytes_processed = storage.bytes().size(),
                .checksum = checksum_byte_span(storage.bytes()),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.copy_f32",
        report.config.iterations,
        "shape=[16384] dtype=F32",
        [copy_descriptor, copy_payload]() {
            const auto storage = TensorStorage::from_bytes(copy_descriptor, std::span<const std::uint8_t>(copy_payload));
            return IterationMeasurement {
                .bytes_processed = storage.bytes().size(),
                .checksum = checksum_byte_span(storage.bytes()),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.add_f32",
        report.config.iterations,
        "shape=[4096] dtype=F32",
        [&add_left, &add_right]() {
            const auto output = add_f32(add_left, add_right);
            return IterationMeasurement {
                .bytes_processed = add_left.bytes().size() + add_right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));
}

[[nodiscard]] std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::error_code error_code;
    const auto file_size = std::filesystem::file_size(path, error_code);
    if (error_code) {
        throw std::runtime_error("could not read '" + path.string() + "': " + error_code.message());
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open '" + path.string() + "' for reading");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw std::runtime_error("could not read entire file '" + path.string() + "'");
        }
    }

    return bytes;
}

void add_model_load_benchmarks(BenchmarkReport& report, const std::filesystem::path& model_path) {
    report.measurements.push_back(measure(
        "gguf.read_file",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model_path]() {
            const auto bytes = read_binary_file(model_path);
            return IterationMeasurement {
                .bytes_processed = bytes.size(),
                .checksum = checksum_byte_span(bytes),
            };
        }
    ));

    const auto bytes = read_binary_file(model_path);
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));

    report.measurements.push_back(measure(
        "gguf.parse_model_view",
        report.config.iterations,
        "model=" + model_path.string(),
        [&bytes]() {
            const auto parsed = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
            return IterationMeasurement {
                .bytes_processed = bytes.size(),
                .checksum = parsed.directory.metadata.header.tensor_count
                    ^ parsed.directory.metadata.header.metadata_kv_count,
            };
        }
    ));

    report.measurements.push_back(measure(
        "gguf.load_tensor_payloads",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model]() {
            const auto loaded = load_gguf_tensor_payloads(model);
            std::size_t bytes_processed = 0;
            std::uint64_t checksum = 0;
            for (const auto& tensor : loaded) {
                bytes_processed += tensor.storage.bytes().size();
                checksum ^= checksum_byte_span(tensor.storage.bytes());
            }

            return IterationMeasurement {
                .bytes_processed = bytes_processed,
                .checksum = checksum,
            };
        }
    ));
}

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

BenchmarkReport run_benchmarks(const BenchmarkConfig& config) {
    BenchmarkReport report {
        .config = config,
        .measurements = {},
    };

    add_synthetic_tensor_benchmarks(report);

    if (config.model_path.has_value()) {
        add_model_load_benchmarks(report, *config.model_path);
    }

    return report;
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
