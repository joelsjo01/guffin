#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::benchmarks {

inline constexpr std::size_t quick_kernel_repetition_count = 3U;
inline constexpr std::size_t quick_kernel_warmup_count = 1U;
inline constexpr std::uint32_t quick_kernel_isolated_dispatch_count = 1U;
inline constexpr std::uint32_t quick_kernel_sustained_dispatch_count = 42U;
inline constexpr std::uint32_t quick_kernel_json_schema_version = 2U;

enum class QuickKernelMeasurementMode {
    isolated,
    sustained,
};

struct BenchmarkStatistics {
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct QuickKernelMeasurement {
    QuickKernelMeasurementMode mode = QuickKernelMeasurementMode::isolated;
    std::uint32_t dispatches_per_command = 0;
    BenchmarkStatistics gpu_duration_per_dispatch_ns;
    double effective_weight_bandwidth_gb_per_second = 0.0;
    std::vector<std::uint64_t> raw_command_gpu_duration_ns;
};

struct QuickKernelResult {
    std::string name;
    std::string shape;
    std::array<QuickKernelMeasurement, 2> measurements;
};

struct QuickKernelBenchmarkResult {
    std::string engine_build;
    std::string model_path;
    std::vector<QuickKernelResult> kernels;
};

[[nodiscard]] BenchmarkStatistics summarize_values(std::span<const double> values);
[[nodiscard]] std::string_view quick_kernel_measurement_mode_name(
    QuickKernelMeasurementMode mode
);
[[nodiscard]] QuickKernelMeasurement summarize_kernel_measurement(
    QuickKernelMeasurementMode mode,
    std::uint32_t dispatches_per_command,
    std::uint64_t weight_bytes,
    std::span<const std::uint64_t> raw_command_gpu_duration_ns
);
[[nodiscard]] QuickKernelBenchmarkResult run_quick_kernel_benchmark(const std::string& model_path);

}  // namespace local_gguf_engine::benchmarks
