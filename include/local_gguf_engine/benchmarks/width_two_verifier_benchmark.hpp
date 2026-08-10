#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace local_gguf_engine::benchmarks {

struct WidthTwoVerifierBenchmarkResult {
    std::string model_path;
    std::uint64_t first_decode_gpu_ns = 0;
    std::uint64_t second_decode_gpu_ns = 0;
    std::uint64_t verifier_gpu_ns = 0;
    std::uint64_t commit_two_gpu_ns = 0;
    double verifier_to_mean_decode_ratio = 0.0;
    double verifier_to_two_decode_ratio = 0.0;
    float maximum_logit_absolute_error = 0.0F;
    float maximum_softcapped_logit_absolute_error = 0.0F;
    double mean_kl_divergence = 0.0;
    double mean_top40_overlap = 0.0;
    bool top1_agreement = false;
    float maximum_committed_kv_absolute_error = 0.0F;
    std::size_t verifier_scratch_bytes = 0;
    bool resident_cache_unchanged_before_commit = false;
    bool committed_cache_matches_sequential = false;
    bool gate_pass = false;
};

[[nodiscard]] WidthTwoVerifierBenchmarkResult run_width_two_verifier_benchmark(
    const std::string& model_path
);

}  // namespace local_gguf_engine::benchmarks
