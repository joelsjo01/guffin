#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"
#include "local_gguf_engine/benchmarks/feature_capture_benchmark.hpp"
#include "local_gguf_engine/benchmarks/corrected_vocabulary_benchmark.hpp"
#include "local_gguf_engine/benchmarks/immutable_weight_benchmark.hpp"
#include "local_gguf_engine/benchmarks/output_head_benchmark.hpp"
#include "local_gguf_engine/benchmarks/target_verifier_benchmark.hpp"
#include "local_gguf_engine/benchmarks/width_two_verifier_benchmark.hpp"

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
    const bool width_two_mode = argc == 3 && std::string_view(argv[2]) == "--width-two";
    const bool target_verifier_mode = argc == 4
        && (std::string_view(argv[2]) == "--target-verifier"
            || std::string_view(argv[2]) == "--target-verifier-conditional"
            || std::string_view(argv[2]) == "--target-verifier-forest"
            || std::string_view(argv[2]) == "--target-verifier-repeated-root");
    const bool feature_capture_mode = argc == 4
        && std::string_view(argv[2]) == "--feature-capture";
    const bool corrected_vocabulary_mode = argc == 4
        && std::string_view(argv[2]) == "--corrected-vocabulary";
    const bool draft_vocabulary_shape_mode = argc == 3
        && std::string_view(argv[2]) == "--draft-vocabulary-shapes";
    const bool output_quality_mode = argc >= 3 && std::string_view(argv[2]) == "--output-head-quality";
    const bool storage_mode = argc >= 4 && std::string_view(argv[2]) == "--storage-policy";
    const bool no_copy_mode = argc >= 3 && std::string_view(argv[2]) == "--no-copy-feasibility";
    const bool json = (argc == 3 && std::string_view(argv[2]) == "--json")
        || (argc == 4 && no_copy_mode && std::string_view(argv[3]) == "--json")
        || (argc == 5 && storage_mode && std::string_view(argv[4]) == "--json");
    if (argc < 2 || (!storage_mode && !no_copy_mode && !output_quality_mode
            && !width_two_mode && !target_verifier_mode && !feature_capture_mode
            && !corrected_vocabulary_mode && !draft_vocabulary_shape_mode && argc > 3)
        || (storage_mode && argc > 5) || (no_copy_mode && argc > 4)
        || (target_verifier_mode && argc != 4)
        || (feature_capture_mode && argc != 4)
        || (corrected_vocabulary_mode && argc != 4)
        || (argc == 3 && !json && !no_copy_mode && !output_quality_mode && !width_two_mode
            && !draft_vocabulary_shape_mode)
        || (output_quality_mode && argc != 4)
        || (storage_mode && argc == 5 && !json) || (no_copy_mode && argc == 4 && !json)) {
        std::cerr << "usage: local-gguf-bench <model.gguf> [--json]\n"
                     "       local-gguf-bench <model.gguf> --width-two\n"
                     "       local-gguf-bench <model.gguf> --target-verifier <rows>\n"
                     "       local-gguf-bench <model.gguf> --target-verifier-conditional <horizon>\n"
                     "       local-gguf-bench <model.gguf> --target-verifier-forest <rows>\n"
                     "       local-gguf-bench <model.gguf> --target-verifier-repeated-root <beam-width>\n"
                     "       local-gguf-bench <model.gguf> --feature-capture <off|on>\n"
                     "       local-gguf-bench <model.gguf> --corrected-vocabulary <workload>\n"
                     "       local-gguf-bench <model.gguf> --draft-vocabulary-shapes\n"
                     "       local-gguf-bench <model.gguf> --output-head-quality --json\n"
                     "       local-gguf-bench <model.gguf> --storage-policy <policy> [--json]\n"
                     "       local-gguf-bench <model.gguf> --no-copy-feasibility [--json]\n";
        return EXIT_FAILURE;
    }
    try {
        if (draft_vocabulary_shape_mode) {
            const auto result = local_gguf_engine::benchmarks::run_draft_vocabulary_shape_benchmark(
                argv[1]
            );
            std::cout << result.interpretation_label
                << "\nstates/candidate k: " << result.state_count << "/"
                << result.candidate_k
                << "\nauthenticated fixture/owned weights/scratch bytes: "
                << result.authenticated_fixture_bytes << "/"
                << result.owned_weight_bytes << "/" << result.scratch_bytes
                << "\nverified load/upload/prefill/fixture setup: "
                << std::fixed << std::setprecision(3)
                << static_cast<double>(result.verified_load_wall_ns) / 1.0e6 << "/"
                << static_cast<double>(result.target_upload_wall_ns) / 1.0e6 << "/"
                << static_cast<double>(result.live_feature_prefill_wall_ns) / 1.0e6 << "/"
                << static_cast<double>(result.fixture_pack_plan_input_wall_ns) / 1.0e6
                << " ms\n";
            for (const auto& measurement : result.measurements) {
                std::cout << (measurement.invocation_count == 1U
                        ? "parallel-batched" : "recurrent-per-edge")
                    << " rows/invocations=" << measurement.row_count << "/"
                    << measurement.invocation_count
                    << " wall/GPU="
                    << static_cast<double>(measurement.wall_duration_ns) / 1.0e6 << "/"
                    << static_cast<double>(measurement.gpu_duration_ns) / 1.0e6 << " ms"
                    << " dispatches=" << measurement.dispatch_count
                    << " optimistic scheduled bytes="
                    << measurement.optimistic_scheduled_weight_bytes
                    << " effective lower-bound GB/s="
                    << measurement.optimistic_effective_bandwidth_gbps
                    << " conservative fallbacks="
                    << measurement.conservative_fallback_count
                    << " proposal sha256=" << measurement.proposal_digest.hex() << '\n';
            }
            const auto& pruner = result.arm_s_pruner;
            std::cout << "Arm-S rank-8 pruner: " << pruner.interpretation_label
                << "\nraw edges/survivors/fixture threshold="
                << pruner.raw_edge_count << "/" << pruner.survivor_count << "/"
                << std::scientific << std::setprecision(8)
                << pruner.frozen_fixture_threshold
                << std::fixed << std::setprecision(3)
                << " wall/GPU=" << static_cast<double>(pruner.wall_duration_ns) / 1.0e6
                << "/" << static_cast<double>(pruner.gpu_duration_ns) / 1.0e6 << " ms"
                << " dispatches=" << pruner.dispatch_count
                << " head MAC/FLOP=" << pruner.head_macs << "/" << pruner.head_flops
                << " owned/scratch bytes=" << pruner.owned_weight_bytes << "/"
                << pruner.scratch_bytes
                << " probability sha256=" << pruner.probability_digest.hex() << '\n';
            return EXIT_SUCCESS;
        }
        if (corrected_vocabulary_mode) {
            const auto workload = local_gguf_engine::benchmarks::parse_corrected_vocabulary_workload(
                argv[3]
            );
            const auto result = local_gguf_engine::benchmarks::run_corrected_vocabulary_benchmark(
                argv[1], workload
            );
            const auto& execution = result.execution;
            std::cout << std::fixed << std::setprecision(3)
                << "corrected vocabulary workload: "
                << local_gguf_engine::benchmarks::corrected_vocabulary_workload_name(workload)
                << "\nphysical head rows: " << execution.physical_head_rows
                << "\nraw lower/upper main/upper stream rows: "
                << result.workload.raw_lower_main_rows << "/"
                << result.workload.upper_main_rows << "/"
                << result.workload.upper_stream_rows
                << "\nhead MAC/FLOP: " << execution.head_macs << "/"
                << execution.head_flops
                << "\nwall/GPU: " << static_cast<double>(execution.wall_duration_ns) / 1.0e6
                << "/" << static_cast<double>(execution.gpu_duration_ns) / 1.0e6 << " ms"
                << "\ndispatches/command buffers: " << execution.dispatch_count << "/"
                << execution.command_buffer_count
                << "\nscratch bytes: " << execution.scratch_bytes
                << "\nlogical head-weight bytes: " << execution.logical_head_weight_bytes
                << "\nrescanned head rows: " << execution.rescanned_head_rows
                << "\nrank entries: " << result.workload.rank_entry_capacity
                << "\nrank sha256: " << result.rank_digest.hex()
                << "\ncompact next-tree nodes/sha256: "
                << result.compact_next_tree_nodes << "/"
                << result.compact_next_tree_digest.hex()
                << "\nverified load/upload/setup/prefill: "
                << static_cast<double>(result.verified_load_wall_ns) / 1.0e6 << "/"
                << static_cast<double>(result.target_upload_wall_ns) / 1.0e6 << "/"
                << static_cast<double>(result.scratch_allocation_wall_ns) / 1.0e6 << "/"
                << static_cast<double>(result.live_feature_prefill_wall_ns) / 1.0e6
                << " ms\n";
            return EXIT_SUCCESS;
        }
        if (feature_capture_mode) {
            const auto mode = std::string_view(argv[3]);
            if (mode != "off" && mode != "on") {
                throw std::invalid_argument("feature capture mode must be off or on");
            }
            const auto result = local_gguf_engine::benchmarks::run_feature_capture_benchmark(
                argv[1], mode == "on"
            );
            std::cout << std::fixed << std::setprecision(3)
                << "feature capture: " << (result.capture_enabled ? "on" : "off") << "\n"
                << "verified model load: "
                << static_cast<double>(result.verified_model_load_wall_ns) / 1.0e6 << " ms\n"
                << "target upload: "
                << static_cast<double>(result.target_upload_wall_ns) / 1.0e6 << " ms\n"
                << "feature setup: "
                << static_cast<double>(result.feature_setup_wall_ns) / 1.0e6 << " ms\n"
                << "prompt/TTFT wall/GPU: "
                << static_cast<double>(result.prompt_wall_ns) / 1.0e6 << " / "
                << static_cast<double>(result.prompt_gpu_ns) / 1.0e6 << " ms\n"
                << "prompt throughput: "
                << static_cast<double>(result.prompt_token_count) * 1.0e9
                    / static_cast<double>(result.prompt_wall_ns) << " tokens/s\n"
                << "commands/waits: " << result.prompt_command_buffer_count << "/"
                << result.prompt_synchronous_wait_count << "\n"
                << "resident target/feature bytes: " << result.target_resident_bytes << "/"
                << result.features.allocated_bytes << "\n"
                << "captured/projected rows: " << result.features.captured_row_count << "/"
                << result.features.eager_projection_row_count << "\n"
                << "logical capture/projection bytes: "
                << result.features.logical_capture_bytes << "/"
                << result.features.logical_projection_weight_bytes << "\n"
                << "logits sha256: " << result.logits_digest.hex() << "\n";
            return EXIT_SUCCESS;
        }
        if (target_verifier_mode) {
            char* end = nullptr;
            const auto rows = std::strtoul(argv[3], &end, 10);
            if (end == argv[3] || *end != '\0' || rows > 1'025UL) {
                throw std::invalid_argument("target verifier rows are invalid");
            }
            const auto result = local_gguf_engine::benchmarks::run_target_verifier_benchmark(
                argv[1], static_cast<std::uint32_t>(rows),
                std::string_view(argv[2]) == "--target-verifier-conditional"
                    ? local_gguf_engine::benchmarks::TargetVerifierBenchmarkTopology::registered_conditional
                    : std::string_view(argv[2]) == "--target-verifier-forest"
                    ? local_gguf_engine::benchmarks::TargetVerifierBenchmarkTopology::prefix_closed_forest
                    : std::string_view(argv[2]) == "--target-verifier-repeated-root"
                    ? local_gguf_engine::benchmarks::TargetVerifierBenchmarkTopology::repeated_root_beam_batch
                    : local_gguf_engine::benchmarks::TargetVerifierBenchmarkTopology::single_invocation
            );
            const auto repeated = std::string_view(argv[2])
                == "--target-verifier-repeated-root";
            std::cout << std::fixed << std::setprecision(3)
                << "target verifier rows: " << result.input_row_count
                << " (H=" << result.proposal_horizon << ")\n"
                << "topology: "
                << (repeated ? "repeated-root beam batch"
                    : result.forest_topology ? "prefix-closed forest"
                    : result.conditional_topology ? "registered conditional"
                    : "single invocation")
                << ", executed tiles: " << result.executed_tile_count << "\n"
                << "allocation wall: " << static_cast<double>(result.allocation_wall_ns) / 1.0e6
                << " ms, arena: " << result.verifier_allocated_bytes << " bytes\n"
                << "ordinary rows wall/GPU: "
                << static_cast<double>(result.ordinary_rows_wall_ns) / 1.0e6 << " / "
                << static_cast<double>(result.ordinary_rows_gpu_ns) / 1.0e6 << " ms\n"
                << "verifier wall/GPU: "
                << static_cast<double>(result.verifier_wall_ns) / 1.0e6 << " / "
                << static_cast<double>(result.verifier_gpu_ns) / 1.0e6 << " ms\n"
                << "controller wall: " << static_cast<double>(result.controller_wall_ns) / 1.0e6
                << " ms\n"
                << "commit wall/GPU: " << static_cast<double>(result.commit_wall_ns) / 1.0e6
                << " / " << static_cast<double>(result.commit_gpu_ns) / 1.0e6 << " ms\n"
                << "complete perfect episode wall: "
                << static_cast<double>(result.complete_episode_wall_ns) / 1.0e6 << " ms\n"
                << "perfect-progress ceiling: " << result.perfect_progress_ceiling << "x\n"
                << "compact readback: " << result.compact_readback_bytes << " bytes\n";
            if (result.candidate_identity_measured) {
                std::cout << "candidate identity: "
                    << (result.exact_candidate_match ? "exact" : "MISMATCH")
                    << " (token order "
                    << (result.candidate_token_order_match ? "exact" : "MISMATCH")
                    << ", max abs/rel " << std::scientific << std::setprecision(6)
                    << result.maximum_candidate_absolute_error << " / "
                    << result.maximum_candidate_relative_error << ")\n";
                if (!result.candidate_token_order_match) {
                    std::cout << std::fixed
                        << "candidate token mismatches: "
                        << result.candidate_token_mismatch_count
                        << ", first row/rank "
                        << result.first_candidate_token_mismatch_row << "/"
                        << result.first_candidate_token_mismatch_rank
                        << ", expected/actual "
                        << result.first_candidate_token_expected << "/"
                        << result.first_candidate_token_actual << "\n";
                }
                std::cout << std::scientific << std::setprecision(6)
                    << "final-hidden max abs: "
                    << result.maximum_final_hidden_absolute_error
                    << " at row " << result.maximum_final_hidden_error_row
                    << ", row 0/mismatch row "
                    << result.first_row_final_hidden_absolute_error << "/"
                    << result.mismatch_row_final_hidden_absolute_error << "\n";
            } else {
                std::cout << "candidate identity: not separately measured\n";
            }
            std::cout << std::fixed << std::setprecision(3)
                << "controller full match: "
                << (result.controller_full_match ? "yes" : "no")
                << "\nauthoritative token path: "
                << (result.authoritative_token_match ? "exact" : "MISMATCH") << "\n";
            return result.controller_full_match && result.authoritative_token_match
                && (!result.candidate_identity_measured || result.candidate_token_order_match)
                ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (width_two_mode) {
            const auto result = local_gguf_engine::benchmarks::run_width_two_verifier_benchmark(argv[1]);
            std::cout << std::fixed << std::setprecision(3)
                << "one-row GPU: " << static_cast<double>(result.first_decode_gpu_ns) / 1.0e6 << " / "
                << static_cast<double>(result.second_decode_gpu_ns) / 1.0e6 << " ms\n"
                << "width-two GPU: " << static_cast<double>(result.verifier_gpu_ns) / 1.0e6 << " ms\n"
                << "commit-two GPU: " << static_cast<double>(result.commit_two_gpu_ns) / 1.0e6 << " ms\n"
                << "verifier / mean one-row: " << result.verifier_to_mean_decode_ratio << "x\n"
                << "verifier / two rows: " << result.verifier_to_two_decode_ratio << "x\n"
                << "maximum logit error: " << result.maximum_logit_absolute_error << "\n"
                << "maximum post-softcap logit error: "
                << result.maximum_softcapped_logit_absolute_error << "\n"
                << "mean KL: " << result.mean_kl_divergence << "\n"
                << "mean top-40 overlap: " << result.mean_top40_overlap << "\n"
                << "top-1 agreement: " << (result.top1_agreement ? "yes" : "no") << "\n"
                << "maximum committed KV error: "
                << result.maximum_committed_kv_absolute_error << "\n"
                << "verifier scratch: " << result.verifier_scratch_bytes << " bytes\n"
                << "resident unchanged before commit: "
                << (result.resident_cache_unchanged_before_commit ? "yes" : "no") << "\n"
                << "committed cache matches sequential: "
                << (result.committed_cache_matches_sequential ? "yes" : "no") << "\n"
                << "1.35x gate: " << (result.gate_pass ? "PASS" : "FAIL") << "\n";
            return result.gate_pass ? EXIT_SUCCESS : EXIT_FAILURE;
        }
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
