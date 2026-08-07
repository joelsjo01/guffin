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
    output << "Sustained mode uses "
           << local_gguf_engine::benchmarks::quick_kernel_sustained_dispatch_count
           << " serial, buffer-barrier-separated dispatches per command.\n"
           << std::fixed << std::setprecision(1)
           << "Kernel          Shape                    Mode       Dispatches  Median/dispatch ms"
              "  Range/dispatch ms  Effective GB/s\n";
    for (const auto& kernel : result.kernels) {
        for (const auto& measurement : kernel.measurements) {
            const auto& duration = measurement.gpu_duration_per_dispatch_ns;
            output << std::left << std::setw(16) << kernel.name
                   << std::setw(25) << kernel.shape
                   << std::setw(11) << local_gguf_engine::benchmarks::quick_kernel_measurement_mode_name(
                       measurement.mode
                   )
                   << std::right << std::setw(10) << measurement.dispatches_per_command
                   << std::setw(20) << duration.median / 1.0e6 << "  "
                   << std::setw(6) << duration.minimum / 1.0e6 << '-'
                   << std::setw(6) << duration.maximum / 1.0e6
                   << std::setw(16) << measurement.effective_weight_bandwidth_gb_per_second << '\n';
        }
    }
}

void write_json(std::ostream& output, const QuickKernelBenchmarkResult& result) {
    output << std::setprecision(17) << "{\"engine_build\":";
    write_json_string(output, result.engine_build);
    output << ",\"model\":";
    write_json_string(output, result.model_path);
    output << ",\"schema_version\":"
           << local_gguf_engine::benchmarks::quick_kernel_json_schema_version
           << ",\"measured_commands_per_mode\":"
           << local_gguf_engine::benchmarks::quick_kernel_repetition_count
           << ",\"warmup_commands_per_mode\":"
           << local_gguf_engine::benchmarks::quick_kernel_warmup_count
           << ",\"sustained_dispatches_per_command\":"
           << local_gguf_engine::benchmarks::quick_kernel_sustained_dispatch_count
           << ",\"kernels\":[";
    for (std::size_t index = 0; index < result.kernels.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& kernel = result.kernels[index];
        output << "{\"name\":";
        write_json_string(output, kernel.name);
        output << ",\"shape\":";
        write_json_string(output, kernel.shape);
        output << ",\"measurements\":[";
        for (std::size_t measurement_index = 0;
             measurement_index < kernel.measurements.size();
             ++measurement_index) {
            if (measurement_index != 0U) output << ',';
            const auto& measurement = kernel.measurements[measurement_index];
            output << "{\"mode\":";
            write_json_string(
                output,
                local_gguf_engine::benchmarks::quick_kernel_measurement_mode_name(measurement.mode)
            );
            output << ",\"dispatches_per_command\":" << measurement.dispatches_per_command
                   << ",\"median_per_dispatch_ns\":"
                   << measurement.gpu_duration_per_dispatch_ns.median
                   << ",\"min_per_dispatch_ns\":"
                   << measurement.gpu_duration_per_dispatch_ns.minimum
                   << ",\"max_per_dispatch_ns\":"
                   << measurement.gpu_duration_per_dispatch_ns.maximum
                   << ",\"effective_gbps\":"
                   << measurement.effective_weight_bandwidth_gb_per_second
                   << ",\"raw_command_ns\":[";
            for (std::size_t raw_index = 0;
                 raw_index < measurement.raw_command_gpu_duration_ns.size();
                 ++raw_index) {
                if (raw_index != 0U) output << ',';
                output << measurement.raw_command_gpu_duration_ns[raw_index];
            }
            output << "]}";
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
