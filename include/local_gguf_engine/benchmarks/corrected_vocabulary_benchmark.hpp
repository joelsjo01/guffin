#pragma once

#include "local_gguf_engine/engine/draft_vocabulary.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"

#include <cstdint>
#include <array>
#include <string>
#include <string_view>

namespace local_gguf_engine::benchmarks {

struct CorrectedVocabularyBenchmarkResult {
    std::string model_path;
    engine::CorrectedVocabularyWorkloadSpec workload;
    engine::CorrectedVocabularyMetalResult execution;
    std::uint64_t verified_load_wall_ns = 0U;
    std::uint64_t target_upload_wall_ns = 0U;
    std::uint64_t scratch_allocation_wall_ns = 0U;
    std::uint64_t live_feature_prefill_wall_ns = 0U;
    runtime::Sha256Digest rank_digest;
    std::uint32_t compact_next_tree_nodes = 0U;
    runtime::Sha256Digest compact_next_tree_digest;
};

// These are fixture-weight component timings. They exercise the production
// artifact loader and Metal vocabulary path on a live target residual, but are
// deliberately not described as trained or end-to-end speculative results.
struct DraftVocabularyShapeMeasurement {
    std::uint32_t row_count = 0U;
    std::uint32_t invocation_count = 0U;
    std::uint64_t wall_duration_ns = 0U;
    std::uint64_t gpu_duration_ns = 0U;
    std::uint64_t dispatch_count = 0U;
    std::uint64_t optimistic_scheduled_weight_bytes = 0U;
    double optimistic_effective_bandwidth_gbps = 0.0;
    std::uint32_t conservative_fallback_count = 0U;
    runtime::Sha256Digest proposal_digest;
};

struct ArmSPrunerShapeMeasurement {
    std::string interpretation_label;
    std::uint32_t raw_edge_count = 0U;
    std::uint32_t survivor_count = 0U;
    float frozen_fixture_threshold = 0.0F;
    std::uint64_t wall_duration_ns = 0U;
    std::uint64_t gpu_duration_ns = 0U;
    std::uint32_t dispatch_count = 0U;
    std::uint64_t head_macs = 0U;
    std::uint64_t head_flops = 0U;
    std::uint64_t owned_weight_bytes = 0U;
    std::uint64_t scratch_bytes = 0U;
    runtime::Sha256Digest probability_digest;
};

struct DraftVocabularyShapeBenchmarkResult {
    std::string model_path;
    std::string interpretation_label;
    std::uint32_t state_count = 0U;
    std::uint32_t candidate_k = 0U;
    std::uint64_t verified_load_wall_ns = 0U;
    std::uint64_t target_upload_wall_ns = 0U;
    std::uint64_t live_feature_prefill_wall_ns = 0U;
    std::uint64_t fixture_pack_plan_input_wall_ns = 0U;
    std::uint64_t authenticated_fixture_bytes = 0U;
    std::uint64_t owned_weight_bytes = 0U;
    std::uint64_t scratch_bytes = 0U;
    std::array<DraftVocabularyShapeMeasurement, 2U> measurements;
    ArmSPrunerShapeMeasurement arm_s_pruner;
};

[[nodiscard]] engine::CorrectedVocabularyWorkload parse_corrected_vocabulary_workload(
    std::string_view name
);
[[nodiscard]] const char* corrected_vocabulary_workload_name(
    engine::CorrectedVocabularyWorkload workload
) noexcept;

[[nodiscard]] CorrectedVocabularyBenchmarkResult run_corrected_vocabulary_benchmark(
    const std::string& model_path,
    engine::CorrectedVocabularyWorkload workload
);

[[nodiscard]] DraftVocabularyShapeBenchmarkResult
run_draft_vocabulary_shape_benchmark(const std::string& model_path);

}  // namespace local_gguf_engine::benchmarks
