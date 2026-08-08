#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"
#include "local_gguf_engine/benchmarks/immutable_weight_benchmark.hpp"
#include "local_gguf_engine/benchmarks/output_head_benchmark.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using local_gguf_engine::benchmarks::QuickKernelBenchmarkResult;
using local_gguf_engine::benchmarks::ImmutableWeightBenchmarkResult;

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

void write_resource_json(
    std::ostream& output,
    const local_gguf_engine::benchmarks::ImmutableWeightPolicyDescription& resource
) {
    output << "{\"label\":";
    write_json_string(output, resource.label);
    output << ",\"storage_mode\":";
    write_json_string(output, resource.storage_mode);
    output << ",\"cpu_cache_mode\":";
    write_json_string(output, resource.cpu_cache_mode);
    output << ",\"hazard_mode\":";
    write_json_string(output, resource.hazard_mode);
    output << ",\"parent_allocation_count\":" << resource.parent_allocation_count
           << ",\"logical_tensor_bytes\":" << resource.logical_tensor_bytes
           << ",\"staging_bytes\":" << resource.staging_bytes
           << ",\"copied_bytes\":" << resource.copied_bytes
           << ",\"file_backed_bytes\":" << resource.file_backed_bytes
           << ",\"upload_wall_ns\":" << resource.upload_wall_duration_ns
           << ",\"upload_gpu_ns\":" << resource.upload_gpu_duration_ns
           << ",\"cpu_contents_accessible\":"
           << (resource.cpu_contents_accessible ? "true" : "false") << '}';
}

void write_measurement_json(
    std::ostream& output,
    const local_gguf_engine::benchmarks::QuickKernelMeasurement& measurement
) {
    output << "{\"mode\":";
    write_json_string(
        output,
        local_gguf_engine::benchmarks::quick_kernel_measurement_mode_name(measurement.mode)
    );
    output << ",\"dispatches_per_command\":" << measurement.dispatches_per_command
           << ",\"median_per_dispatch_ns\":" << measurement.gpu_duration_per_dispatch_ns.median
           << ",\"min_per_dispatch_ns\":" << measurement.gpu_duration_per_dispatch_ns.minimum
           << ",\"max_per_dispatch_ns\":" << measurement.gpu_duration_per_dispatch_ns.maximum
           << ",\"effective_gbps\":" << measurement.effective_weight_bandwidth_gb_per_second
           << ",\"raw_command_ns\":[";
    for (std::size_t index = 0; index < measurement.raw_command_gpu_duration_ns.size(); ++index) {
        if (index != 0U) output << ',';
        output << measurement.raw_command_gpu_duration_ns[index];
    }
    output << "]}";
}

template <std::size_t Count>
void write_measurements_json(
    std::ostream& output,
    const std::array<local_gguf_engine::benchmarks::QuickKernelMeasurement, Count>& measurements
) {
    output << '[';
    for (std::size_t index = 0U; index < measurements.size(); ++index) {
        if (index != 0U) output << ',';
        write_measurement_json(output, measurements[index]);
    }
    output << ']';
}

void write_storage_json(std::ostream& output, const ImmutableWeightBenchmarkResult& result) {
    output << std::setprecision(17) << "{\"engine_build\":";
    write_json_string(output, result.engine_build);
    output << ",\"model\":";
    write_json_string(output, result.model_path);
    output << ",\"schema_version\":"
           << local_gguf_engine::benchmarks::immutable_weight_json_schema_version
           << ",\"policy\":";
    write_json_string(
        output, local_gguf_engine::benchmarks::immutable_weight_policy_name(result.policy)
    );
    output << ",\"device\":{\"name\":";
    write_json_string(output, result.device_name);
    output << ",\"unified_memory\":" << (result.unified_memory ? "true" : "false")
           << ",\"max_buffer_length\":" << result.maximum_buffer_length
           << ",\"recommended_max_working_set_size\":"
           << result.recommended_maximum_working_set_size
           << ",\"vm_page_size\":" << result.vm_page_size << '}'
           << ",\"model_file_bytes\":" << result.model_file_bytes
           << ",\"model_tensor_bytes\":" << result.model_tensor_bytes
           << ",\"model_tensor_count\":" << result.model_tensor_count
           << ",\"file_backed_no_copy\":" << (result.file_backed_no_copy ? "true" : "false")
           << ",\"mapped_extent_bytes\":" << result.mapped_extent_bytes
           << ",\"base_page_aligned\":" << (result.base_page_aligned ? "true" : "false")
           << ",\"single_vm_region\":" << (result.single_vm_region ? "true" : "false")
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
        output << ",\"tensor\":";
        write_json_string(output, kernel.tensor_name);
        output << ",\"shape\":";
        write_json_string(output, kernel.shape);
        output << ",\"weight_bytes\":" << kernel.weight_bytes
               << ",\"scheduled_class_weight_bytes\":" << kernel.scheduled_class_weight_bytes;
        output << ",\"resource\":";
        write_resource_json(output, kernel.resource);
        output << ",\"measurements\":[";
        for (std::size_t measurement = 0; measurement < kernel.measurements.size(); ++measurement) {
            if (measurement != 0U) output << ',';
            write_measurement_json(output, kernel.measurements[measurement]);
        }
        output << "]}";
    }
    output << "]}\n";
}

void write_storage_human(std::ostream& output, const ImmutableWeightBenchmarkResult& result) {
    output << "Immutable weight policy: "
           << local_gguf_engine::benchmarks::immutable_weight_policy_name(result.policy)
           << "\nDevice: " << result.device_name
           << ", unified=" << (result.unified_memory ? "yes" : "no")
           << ", max buffer=" << result.maximum_buffer_length
           << ", recommended working set=" << result.recommended_maximum_working_set_size
           << ", VM page=" << result.vm_page_size << " bytes\n"
           << "Model: file=" << result.model_file_bytes << ", tensors="
           << result.model_tensor_count << "/" << result.model_tensor_bytes << " bytes\n";
    for (const auto& kernel : result.kernels) {
        output << kernel.name << " (" << kernel.tensor_name << ", " << kernel.shape << ")\n"
               << "  resource=" << kernel.resource.storage_mode << '/'
               << kernel.resource.cpu_cache_mode << '/' << kernel.resource.hazard_mode
               << ", copied=" << kernel.resource.copied_bytes
               << ", staging=" << kernel.resource.staging_bytes
               << ", upload wall/GPU ns=" << kernel.resource.upload_wall_duration_ns
               << '/' << kernel.resource.upload_gpu_duration_ns << '\n';
        for (const auto& measurement : kernel.measurements) {
            output << "  "
                   << local_gguf_engine::benchmarks::quick_kernel_measurement_mode_name(measurement.mode)
                   << ": " << std::fixed << std::setprecision(1)
                   << measurement.gpu_duration_per_dispatch_ns.median / 1.0e6
                   << " ms, " << measurement.effective_weight_bandwidth_gb_per_second
                   << " GB/s\n";
        }
    }
}

}  // namespace

int main(int argc, const char* const* argv) {
    const bool output_quality_mode = argc >= 3 && std::string_view(argv[2]) == "--output-head-quality";
    const bool storage_mode = argc >= 4 && std::string_view(argv[2]) == "--storage-policy";
    const bool no_copy_mode = argc >= 3 && std::string_view(argv[2]) == "--no-copy-feasibility";
    const bool json = (argc == 3 && std::string_view(argv[2]) == "--json")
        || (argc == 4 && no_copy_mode && std::string_view(argv[3]) == "--json")
        || (argc == 5 && storage_mode && std::string_view(argv[4]) == "--json");
    if (argc < 2 || (!storage_mode && !no_copy_mode && !output_quality_mode && argc > 3)
        || (storage_mode && argc > 5) || (no_copy_mode && argc > 4)
        || (argc == 3 && !json && !no_copy_mode && !output_quality_mode)
        || (output_quality_mode && argc != 4)
        || (storage_mode && argc == 5 && !json) || (no_copy_mode && argc == 4 && !json)) {
        std::cerr << "usage: local-gguf-bench <model.gguf> [--json]\n"
                     "       local-gguf-bench <model.gguf> --output-head-quality --json\n"
                     "       local-gguf-bench <model.gguf> --storage-policy <policy> [--json]\n"
                     "       local-gguf-bench <model.gguf> --no-copy-feasibility [--json]\n";
        return EXIT_FAILURE;
    }
    try {
        if (output_quality_mode) {
            if (std::string_view(argv[3]) != "--json") {
                throw std::invalid_argument("output-head quality mode requires --json");
            }
            const auto result = local_gguf_engine::benchmarks::run_output_head_quality_benchmark(argv[1]);
            std::cout << std::setprecision(17)
                << "{\"schema_version\":"
                << local_gguf_engine::benchmarks::output_head_quality_schema_version
                << ",\"engine_build\":";
            write_json_string(std::cout, result.engine_build);
            std::cout << ",\"model\":";
            write_json_string(std::cout, result.model_path);
            std::cout << ",\"prediction_count\":" << result.prediction_count
                << ",\"vocabulary_size\":" << result.vocabulary_size
                << ",\"f16_resident_weight_bytes\":" << result.f16_resident_weight_bytes
                << ",\"q8_resident_weight_bytes\":" << result.q8_resident_weight_bytes
                << ",\"hidden_max_absolute_error\":" << result.hidden_max_absolute_error
                << ",\"logit_mean_absolute_error\":" << result.logit_mean_absolute_error
                << ",\"logit_mean_p95_absolute_error\":" << result.logit_mean_p95_absolute_error
                << ",\"logit_max_absolute_error\":" << result.logit_max_absolute_error
                << ",\"top1_agreement\":" << result.top1_agreement
                << ",\"top40_mean_overlap\":" << result.top40_mean_overlap
                << ",\"top40_minimum_overlap\":" << result.top40_minimum_overlap
                << ",\"mean_kl_divergence\":" << result.mean_kl_divergence
                << ",\"maximum_kl_divergence\":" << result.maximum_kl_divergence
                << ",\"f16_negative_log_likelihood\":" << result.f16_negative_log_likelihood
                << ",\"q8_negative_log_likelihood\":" << result.q8_negative_log_likelihood
                << ",\"f16_perplexity\":" << result.f16_perplexity
                << ",\"q8_perplexity\":" << result.q8_perplexity
                << ",\"kernel_measurements\":{\"f16\":";
            write_measurements_json(std::cout, result.f16_kernel_measurements);
            std::cout << ",\"q8_0\":";
            write_measurements_json(std::cout, result.q8_kernel_measurements);
            std::cout << "}"
                << ",\"finite\":" << (result.finite ? "true" : "false")
                << ",\"quality_gates_pass\":" << (result.quality_gates_pass ? "true" : "false")
                << "}\n";
            return result.quality_gates_pass ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (no_copy_mode) {
            const auto result = local_gguf_engine::benchmarks::run_no_copy_weight_feasibility_benchmark(
                argv[1]
            );
            if (json) write_storage_json(std::cout, result);
            else write_storage_human(std::cout, result);
            return EXIT_SUCCESS;
        }
        if (storage_mode) {
            const auto policy = local_gguf_engine::benchmarks::parse_immutable_weight_policy(argv[3]);
            const auto result = local_gguf_engine::benchmarks::run_immutable_weight_benchmark(
                argv[1], policy
            );
            if (json) write_storage_json(std::cout, result);
            else write_storage_human(std::cout, result);
            return EXIT_SUCCESS;
        }
        const auto result = local_gguf_engine::benchmarks::run_quick_kernel_benchmark(argv[1]);
        if (json) write_json(std::cout, result);
        else write_human(std::cout, result);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
