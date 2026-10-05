#pragma once

#include "local_gguf_engine/engine/target_verifier.hpp"

#include <cstdint>

namespace local_gguf_engine::benchmarks {

struct TargetVerifierEpisodeWorkspace {
    // This request is intentionally caller-owned and reusable. It is roughly
    // VerifierRowMax-shaped, so placing it inside a per-call stack frame would
    // both risk stack exhaustion and violate the steady-state allocation
    // contract.
    engine::VerifierEncodeRequest tile_request;
    engine::SpeculativeControllerScenario controller_scenario;
};

struct TargetVerifierEpisodeResult {
    engine::SpeculativeControllerOutcome controller;
    engine::TargetVerifierExecutionStats target_execution;
    engine::Gemma2MetalDecodeExecutionStats endpoint_execution;
    engine::Gemma2MetalDecodeExecutionStats ordinary_fallback_execution;
    engine::CompactTokenCandidates next_candidates;
    engine::CompactTokenCandidates ordinary_fallback_candidates;
    std::uint64_t controller_wall_ns = 0;
    std::uint64_t target_wall_ns = 0;
    std::uint64_t target_gpu_ns = 0;
    std::uint64_t commit_wall_ns = 0;
    std::uint64_t commit_gpu_ns = 0;
    std::uint64_t endpoint_wall_ns = 0;
    std::uint64_t ordinary_fallback_wall_ns = 0;
    std::uint32_t executed_tile_count = 0;
    std::uint32_t endpoint_count = 0;
    bool has_next_candidates = false;
    bool executed_ordinary_fallback = false;
};

// Drives a validated chain through the exact controller one available target
// sample at a time. A later tile is encoded only after the controller reaches
// it; all preceding gate, context, stop, budget, callback, and mismatch
// decisions therefore suppress that Metal submission rather than merely
// discounting it after execution. The authoritative sampler must already be
// at scenario.origin_checkpoint (the checkpoint after selecting x0).
[[nodiscard]] TargetVerifierEpisodeResult
run_target_verifier_chain_episode(
    engine::Gemma2MetalModelPlan& target,
    engine::Gemma2MetalTargetVerifierPlan& verifier,
    const engine::ValidatedVerifierEncodeRequest& full_request,
    const engine::SpeculativeControllerScenario& scenario,
    engine::TokenSampler& authoritative_sampler,
    TargetVerifierEpisodeWorkspace& workspace,
    float embedding_scale
);

// Single-submission prefix-closed forest/repeated-root counterpart. Sampling
// walks only the target-selected causal path; unused leaf rows never consume a
// sampler ticket, and only that path can reach resident K/V.
[[nodiscard]] TargetVerifierEpisodeResult
run_target_verifier_forest_episode(
    engine::Gemma2MetalModelPlan& target,
    engine::Gemma2MetalTargetVerifierPlan& verifier,
    const engine::ValidatedVerifierEncodeRequest& full_request,
    const engine::SpeculativeControllerScenario& scenario,
    engine::TokenSampler& authoritative_sampler,
    TargetVerifierEpisodeWorkspace& workspace,
    float embedding_scale
);

} // namespace local_gguf_engine::benchmarks
