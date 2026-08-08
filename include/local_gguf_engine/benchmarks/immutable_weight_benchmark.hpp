#pragma once

#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::benchmarks {

inline constexpr std::uint32_t immutable_weight_json_schema_version = 1U;

enum class ImmutableWeightPolicy {
    shared_default_tracked,
    shared_write_combined_tracked,
    shared_default_untracked,
    shared_write_combined_untracked,
    private_tracked,
    private_untracked,
};

struct ImmutableWeightPolicyDescription {
    std::string label;
    std::string storage_mode;
    std::string cpu_cache_mode;
    std::string hazard_mode;
    std::uint32_t parent_allocation_count = 0;
    std::uint64_t logical_tensor_bytes = 0;
    std::uint64_t staging_bytes = 0;
    std::uint64_t copied_bytes = 0;
    std::uint64_t file_backed_bytes = 0;
    std::uint64_t upload_wall_duration_ns = 0;
    std::uint64_t upload_gpu_duration_ns = 0;
    bool cpu_contents_accessible = false;
};

struct ImmutableWeightKernelResult {
    std::string name;
    std::string tensor_name;
    std::string shape;
    std::uint64_t weight_bytes = 0;
    std::uint64_t scheduled_class_weight_bytes = 0;
    ImmutableWeightPolicyDescription resource;
    std::vector<QuickKernelMeasurement> measurements;
};

struct ImmutableWeightBenchmarkResult {
    std::string engine_build;
    std::string model_path;
    std::string device_name;
    bool unified_memory = false;
    std::uint64_t maximum_buffer_length = 0;
    std::uint64_t recommended_maximum_working_set_size = 0;
    std::uint64_t vm_page_size = 0;
    std::uint64_t model_file_bytes = 0;
    std::uint64_t model_tensor_bytes = 0;
    std::uint64_t model_tensor_count = 0;
    ImmutableWeightPolicy policy = ImmutableWeightPolicy::shared_default_tracked;
    bool file_backed_no_copy = false;
    std::uint64_t mapped_extent_bytes = 0;
    bool base_page_aligned = false;
    bool single_vm_region = false;
    std::vector<ImmutableWeightKernelResult> kernels;
};

[[nodiscard]] std::string_view immutable_weight_policy_name(ImmutableWeightPolicy policy);
[[nodiscard]] ImmutableWeightPolicy parse_immutable_weight_policy(std::string_view value);
[[nodiscard]] ImmutableWeightPolicyDescription describe_immutable_weight_policy(
    ImmutableWeightPolicy policy,
    std::uint64_t logical_tensor_bytes,
    std::uint64_t upload_wall_duration_ns,
    std::uint64_t upload_gpu_duration_ns
);
[[nodiscard]] ImmutableWeightBenchmarkResult run_immutable_weight_benchmark(
    const std::string& model_path,
    ImmutableWeightPolicy policy
);
void validate_no_copy_mapping_constraints(
    std::uintptr_t base_address,
    std::uint64_t file_byte_count,
    std::uint64_t mapped_extent_byte_count,
    std::uint64_t page_size,
    std::uint64_t maximum_buffer_length,
    bool single_vm_region
);
[[nodiscard]] ImmutableWeightBenchmarkResult run_no_copy_weight_feasibility_benchmark(
    const std::string& model_path
);

}  // namespace local_gguf_engine::benchmarks
