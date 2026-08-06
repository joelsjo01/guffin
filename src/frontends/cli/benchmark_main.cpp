#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using local_gguf_engine::benchmarks::QuickKernelBenchmarkResult;

void write_json_string(std::ostream& output, std::string_view value) {
    output << '"';
    for (const auto byte : value) {
        if (byte == '"') output << "\\\"";
        else if (byte == '\\') output << "\\\\";
        else output << byte;
    }
    output << '"';
}

void write_human(std::ostream& output, const QuickKernelBenchmarkResult& result) {
    output << std::fixed << std::setprecision(1)
           << "Kernel          Shape                    Median (ms)  Range (ms)       Effective GB/s\n";
    for (const auto& kernel : result.kernels) {
        output << std::left << std::setw(16) << kernel.name
               << std::setw(25) << kernel.shape << std::right
               << std::setw(11) << kernel.gpu_duration_ns.median / 1.0e6 << "  "
               << std::setw(5) << kernel.gpu_duration_ns.minimum / 1.0e6 << '-'
               << std::setw(5) << kernel.gpu_duration_ns.maximum / 1.0e6
               << std::setw(21) << kernel.effective_weight_bandwidth_gb_per_second << '\n';
    }
}

void write_json(std::ostream& output, const QuickKernelBenchmarkResult& result) {
    output << std::setprecision(17) << "{\"engine_build\":";
    write_json_string(output, result.engine_build);
    output << ",\"model\":";
    write_json_string(output, result.model_path);
    output << ",\"repetitions\":"
           << local_gguf_engine::benchmarks::quick_kernel_repetition_count
           << ",\"kernels\":[";
    for (std::size_t index = 0; index < result.kernels.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& kernel = result.kernels[index];
        output << "{\"name\":";
        write_json_string(output, kernel.name);
        output << ",\"shape\":";
        write_json_string(output, kernel.shape);
        output << ",\"median_ns\":" << kernel.gpu_duration_ns.median
               << ",\"min_ns\":" << kernel.gpu_duration_ns.minimum
               << ",\"max_ns\":" << kernel.gpu_duration_ns.maximum
               << ",\"effective_gbps\":" << kernel.effective_weight_bandwidth_gb_per_second
               << ",\"raw_ns\":[";
        for (std::size_t raw_index = 0; raw_index < kernel.raw_gpu_duration_ns.size(); ++raw_index) {
            if (raw_index != 0U) output << ',';
            output << kernel.raw_gpu_duration_ns[raw_index];
        }
        output << "]}";
    }
    output << "]}\n";
}

}  // namespace

int main(int argc, const char* const* argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--json")) {
        std::cerr << "usage: local-gguf-bench <model.gguf> [--json]\n";
        return EXIT_FAILURE;
    }
    try {
        const auto result = local_gguf_engine::benchmarks::run_quick_kernel_benchmark(argv[1]);
        if (argc == 3) write_json(std::cout, result);
        else write_human(std::cout, result);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
