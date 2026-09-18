#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"

#include <cstdint>
#include <string>

namespace local_gguf_engine::benchmarks {

struct FeatureCaptureBenchmarkResult {
    std::string model_path;
    bool capture_enabled = false;
    std::uint32_t prompt_token_count = 0U;
    std::uint64_t verified_model_load_wall_ns = 0U;
    std::uint64_t target_upload_wall_ns = 0U;
    std::uint64_t feature_setup_wall_ns = 0U;
    std::uint64_t prompt_wall_ns = 0U;
    std::uint64_t prompt_gpu_ns = 0U;
    std::uint32_t prompt_command_buffer_count = 0U;
    std::uint32_t prompt_synchronous_wait_count = 0U;
    std::size_t target_resident_bytes = 0U;
    engine::FeatureSinkMetrics features;
    runtime::Sha256Digest logits_digest;
};

[[nodiscard]] FeatureCaptureBenchmarkResult run_feature_capture_benchmark(
    const std::string& model_path,
    bool capture_enabled
);

}  // namespace local_gguf_engine::benchmarks
