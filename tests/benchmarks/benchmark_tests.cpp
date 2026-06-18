#include "benchmark_tests.hpp"

#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using local_gguf_engine::benchmarks::benchmark_output_path_is_under_results;
using local_gguf_engine::benchmarks::BenchmarkBackend;
using local_gguf_engine::benchmarks::BenchmarkConfig;
using local_gguf_engine::benchmarks::BenchmarkMeasurement;
using local_gguf_engine::benchmarks::BenchmarkReport;
using local_gguf_engine::benchmarks::BenchmarkSuite;
using local_gguf_engine::benchmarks::default_benchmark_output_path;
using local_gguf_engine::benchmarks::format_benchmark_report;
using local_gguf_engine::benchmarks::parse_benchmark_args;
using local_gguf_engine::tests::expect;

void test_parses_default_benchmark_args() {
    const char *argv[] = {"local-gguf-bench"};
    const auto parsed = parse_benchmark_args(1, argv);

    expect(!parsed.should_exit, "benchmark default args should not exit");
    expect(parsed.config.output_path == default_benchmark_output_path(),
           "benchmark default output path mismatch");
    expect(!parsed.config.model_path.has_value(), "benchmark default model path should be absent");
    expect(parsed.config.iterations == 10, "benchmark default iteration count mismatch");
    expect(parsed.config.suite == BenchmarkSuite::all, "benchmark default suite mismatch");
    expect(parsed.config.backend == BenchmarkBackend::both, "benchmark default backend mismatch");
}

void test_parses_benchmark_model_output_iterations_suite_and_backend() {
    const char *argv[] = {
        "local-gguf-bench",
        "--model",
        "models/example.gguf",
        "--output",
        "benchmarks/results/example.txt",
        "--iterations",
        "3",
        "--suite",
        "resident-kv-decode",
        "--backend",
        "metal",
    };
    const auto parsed = parse_benchmark_args(11, argv);

    expect(!parsed.should_exit, "benchmark explicit args should not exit");
    expect(parsed.config.model_path.has_value(), "benchmark model path should be present");
    expect(*parsed.config.model_path == std::filesystem::path("models/example.gguf"),
           "benchmark model path mismatch");
    expect(parsed.config.output_path == std::filesystem::path("benchmarks/results/example.txt"),
           "benchmark output path mismatch");
    expect(parsed.config.iterations == 3, "benchmark explicit iteration count mismatch");
    expect(parsed.config.suite == BenchmarkSuite::resident_kv_decode,
           "benchmark explicit suite mismatch");
    expect(parsed.config.backend == BenchmarkBackend::metal, "benchmark explicit backend mismatch");
}

void test_parses_benchmark_suite_and_backend_aliases() {
    const char *argv[] = {"local-gguf-bench", "--suite", "decode", "--backend", "gpu"};
    const auto parsed = parse_benchmark_args(5, argv);

    expect(!parsed.should_exit, "benchmark suite/backend aliases should not exit");
    expect(parsed.config.suite == BenchmarkSuite::resident_kv_decode,
           "benchmark suite alias mismatch");
    expect(parsed.config.backend == BenchmarkBackend::metal, "benchmark backend alias mismatch");
}

void test_rejects_benchmark_output_outside_results() {
    const char *argv[] = {"local-gguf-bench", "--output", "benchmark-output.txt"};
    const auto parsed = parse_benchmark_args(3, argv);

    expect(parsed.should_exit, "benchmark output outside results should exit");
    expect(parsed.exit_code == EXIT_FAILURE, "benchmark output outside results exit code mismatch");
    expect(parsed.message.find("benchmarks/results") != std::string::npos,
           "benchmark output outside results error missing");
}

void test_rejects_invalid_benchmark_iterations() {
    const char *argv[] = {"local-gguf-bench", "--iterations", "0"};
    const auto parsed = parse_benchmark_args(3, argv);

    expect(parsed.should_exit, "benchmark zero iterations should exit");
    expect(parsed.exit_code == EXIT_FAILURE, "benchmark zero iterations exit code mismatch");
    expect(parsed.message.find("--iterations requires a positive count") != std::string::npos,
           "benchmark zero iterations error missing");
}

void test_rejects_invalid_benchmark_suite() {
    const char *argv[] = {"local-gguf-bench", "--suite", "everything"};
    const auto parsed = parse_benchmark_args(3, argv);

    expect(parsed.should_exit, "benchmark invalid suite should exit");
    expect(parsed.exit_code == EXIT_FAILURE, "benchmark invalid suite exit code mismatch");
    expect(parsed.message.find("--suite requires") != std::string::npos,
           "benchmark invalid suite error missing");
}

void test_rejects_invalid_benchmark_backend() {
    const char *argv[] = {"local-gguf-bench", "--backend", "cuda"};
    const auto parsed = parse_benchmark_args(3, argv);

    expect(parsed.should_exit, "benchmark invalid backend should exit");
    expect(parsed.exit_code == EXIT_FAILURE, "benchmark invalid backend exit code mismatch");
    expect(parsed.message.find("--backend requires") != std::string::npos,
           "benchmark invalid backend error missing");
}

void test_checks_benchmark_output_path_scope() {
    expect(benchmark_output_path_is_under_results("benchmarks/results/latest.txt"),
           "benchmark output path under results should be accepted");
    expect(benchmark_output_path_is_under_results("benchmarks/results/nested/run.txt"),
           "benchmark nested output path under results should be accepted");
    expect(!benchmark_output_path_is_under_results("benchmarks/results/../latest.txt"),
           "benchmark normalized path outside results should be rejected");
    expect(!benchmark_output_path_is_under_results("/tmp/local-gguf-bench.txt"),
           "benchmark absolute output path should be rejected");
}

void test_formats_benchmark_report() {
    BenchmarkReport report{
        .config =
            BenchmarkConfig{
                .output_path = std::filesystem::path("benchmarks/results/test.txt"),
                .model_path = std::filesystem::path("models/example.gguf"),
                .iterations = 2,
                .suite = BenchmarkSuite::tensor,
                .backend = BenchmarkBackend::cpu,
            },
        .measurements =
            {
                BenchmarkMeasurement{
                    .name = "tensor.add_f32",
                    .iterations = 2,
                    .total_ms = 3.0,
                    .min_ms = 1.0,
                    .max_ms = 2.0,
                    .bytes_processed = 96,
                    .checksum = 42,
                    .detail = "shape=[8] dtype=F32",
                },
            },
    };

    const auto text = format_benchmark_report(report);

    expect(text.find("local GGUF benchmark report\n") == 0, "benchmark report heading mismatch");
    expect(text.find("iterations: 2\n") != std::string::npos,
           "benchmark report iterations missing");
    expect(text.find("suite: tensor\n") != std::string::npos, "benchmark report suite missing");
    expect(text.find("backend: cpu\n") != std::string::npos, "benchmark report backend missing");
    expect(text.find("output: benchmarks/results/test.txt\n") != std::string::npos,
           "benchmark report output path missing");
    expect(text.find("model: models/example.gguf\n") != std::string::npos,
           "benchmark report model missing");
    expect(text.find("- tensor.add_f32\n") != std::string::npos,
           "benchmark report measurement missing");
    expect(text.find("  avg_ms: 1.500\n") != std::string::npos, "benchmark report average missing");
    expect(text.find("  bytes_processed: 96\n") != std::string::npos,
           "benchmark report bytes missing");
    expect(text.find("  checksum: 42\n") != std::string::npos, "benchmark report checksum missing");
}

} // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> benchmark_test_cases() {
    static constexpr TestCase tests[] = {
        {"parses default benchmark args", test_parses_default_benchmark_args},
        {
            "parses benchmark model output iterations suite and backend",
            test_parses_benchmark_model_output_iterations_suite_and_backend,
        },
        {"parses benchmark suite and backend aliases",
         test_parses_benchmark_suite_and_backend_aliases},
        {"rejects benchmark output outside results", test_rejects_benchmark_output_outside_results},
        {"rejects invalid benchmark iterations", test_rejects_invalid_benchmark_iterations},
        {"rejects invalid benchmark suite", test_rejects_invalid_benchmark_suite},
        {"rejects invalid benchmark backend", test_rejects_invalid_benchmark_backend},
        {"checks benchmark output path scope", test_checks_benchmark_output_path_scope},
        {"formats benchmark report", test_formats_benchmark_report},
    };

    return tests;
}

} // namespace local_gguf_engine::tests
