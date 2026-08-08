#pragma once

#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace local_gguf_engine::benchmarks {

inline constexpr std::uint32_t output_head_quality_schema_version = 2U;

struct OutputHeadQualityResult {
    std::string engine_build;
    std::string model_path;
    std::uint32_t prediction_count = 0;
    std::uint32_t vocabulary_size = 0;
    std::size_t f16_resident_weight_bytes = 0;
    std::size_t q8_resident_weight_bytes = 0;
    double hidden_max_absolute_error = 0.0;
    double logit_mean_absolute_error = 0.0;
    double logit_mean_p95_absolute_error = 0.0;
    double logit_max_absolute_error = 0.0;
    double top1_agreement = 0.0;
    double top40_mean_overlap = 0.0;
    std::uint32_t top40_minimum_overlap = 0;
    double mean_kl_divergence = 0.0;
    double maximum_kl_divergence = 0.0;
    double f16_negative_log_likelihood = 0.0;
    double q8_negative_log_likelihood = 0.0;
    double f16_perplexity = 0.0;
    double q8_perplexity = 0.0;
    std::array<QuickKernelMeasurement, 2> f16_kernel_measurements;
    std::array<QuickKernelMeasurement, 2> q8_kernel_measurements;
    bool finite = false;
    bool quality_gates_pass = false;
};

[[nodiscard]] OutputHeadQualityResult run_output_head_quality_benchmark(
    const std::string& model_path,
    std::uint32_t prediction_count = 512U
);

}  // namespace local_gguf_engine::benchmarks
