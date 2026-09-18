#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace local_gguf_engine::benchmarks {

enum class TargetVerifierBenchmarkTopology : std::uint8_t {
    single_invocation,
    registered_conditional,
    prefix_closed_forest,
    repeated_root_beam_batch,
};

struct TargetVerifierBenchmarkResult {
    std::string model_path;
    std::uint32_t input_row_count = 0;
    std::uint32_t proposal_horizon = 0;
    std::uint64_t allocation_wall_ns = 0;
    std::uint64_t ordinary_rows_wall_ns = 0;
    std::uint64_t ordinary_rows_gpu_ns = 0;
    std::uint64_t verifier_wall_ns = 0;
    std::uint64_t verifier_gpu_ns = 0;
    std::uint64_t controller_wall_ns = 0;
    std::uint64_t commit_wall_ns = 0;
    std::uint64_t commit_gpu_ns = 0;
    std::uint64_t complete_episode_wall_ns = 0;
    std::size_t verifier_allocated_bytes = 0;
    std::uint32_t compact_readback_bytes = 0;
    std::uint32_t command_buffer_count = 0;
    std::uint32_t executed_tile_count = 0;
    bool conditional_topology = false;
    bool forest_topology = false;
    bool candidate_identity_measured = false;
    bool exact_candidate_match = false;
    bool candidate_token_order_match = false;
    float maximum_candidate_absolute_error = 0.0F;
    float maximum_candidate_relative_error = 0.0F;
    std::uint32_t candidate_token_mismatch_count = 0;
    std::uint32_t first_candidate_token_mismatch_row = 0;
    std::uint32_t first_candidate_token_mismatch_rank = 0;
    std::uint32_t first_candidate_token_expected = 0;
    std::uint32_t first_candidate_token_actual = 0;
    float maximum_final_hidden_absolute_error = 0.0F;
    std::uint32_t maximum_final_hidden_error_row = 0;
    float first_row_final_hidden_absolute_error = 0.0F;
    float mismatch_row_final_hidden_absolute_error = 0.0F;
    bool controller_full_match = false;
    bool authoritative_token_match = false;
    double perfect_progress_ceiling = 0.0;
};

[[nodiscard]] TargetVerifierBenchmarkResult run_target_verifier_benchmark(
    const std::string& model_path,
    std::uint32_t shape,
    TargetVerifierBenchmarkTopology topology =
        TargetVerifierBenchmarkTopology::single_invocation
);

}  // namespace local_gguf_engine::benchmarks
