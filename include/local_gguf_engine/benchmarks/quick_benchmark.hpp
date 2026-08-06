#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace local_gguf_engine::benchmarks {

inline constexpr std::size_t quick_kernel_repetition_count = 3U;

struct BenchmarkStatistics {
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct QuickKernelResult {
    std::string name;
    std::string shape;
    BenchmarkStatistics gpu_duration_ns;
    double effective_weight_bandwidth_gb_per_second = 0.0;
    std::vector<std::uint64_t> raw_gpu_duration_ns;
};

struct QuickKernelBenchmarkResult {
    std::string engine_build;
    std::string model_path;
    std::vector<QuickKernelResult> kernels;
};

[[nodiscard]] BenchmarkStatistics summarize_values(std::span<const double> values);
[[nodiscard]] QuickKernelBenchmarkResult run_quick_kernel_benchmark(const std::string& model_path);

}  // namespace local_gguf_engine::benchmarks
