#pragma once

#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"

#include <memory>
#include <span>

namespace local_gguf_engine::engine {

class StandaloneDraftArtifact;

struct VicunaDraftExecutionStats {
    std::uint64_t gpu_duration_ns = 0;
    std::uint64_t wall_duration_ns = 0;
    std::uint64_t readback_bytes = 0;
    std::uint64_t uploaded_bytes = 0;
    std::uint32_t command_buffer_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t token_count = 0;
};

struct VicunaDraftProposal {
    // Borrowed until the next mutation of this plan. Every returned draft
    // token already has K/V; position() is start_position + token count.
    std::span<const std::uint32_t> draft_tokens;
    std::span<const std::uint32_t> target_tokens;
    std::uint32_t start_position = 0;
    std::uint64_t epoch = 0;
    VicunaDraftExecutionStats execution;
};

// Complete native Llama/Vicuna-68M forward graph. The prepared artifact and
// injective TLI mapping are copied to resident storage once. This owns its own
// causal positions, F16 K/V, normalized-prefix checkpoints and proposal laws;
// it never borrows a Gemma feature or treats target positions as draft positions.
class VicunaDraftPlan final {
public:
    VicunaDraftPlan(const StandaloneDraftArtifact& artifact,
        std::span<const std::int32_t> draft_to_target,
        std::uint32_t target_vocabulary_size,
        std::uint32_t horizon_capacity = 3);
    ~VicunaDraftPlan();
    VicunaDraftPlan(VicunaDraftPlan&&) noexcept;
    VicunaDraftPlan& operator=(VicunaDraftPlan&&) noexcept;
    VicunaDraftPlan(const VicunaDraftPlan&) = delete;
    VicunaDraftPlan& operator=(const VicunaDraftPlan&) = delete;

    [[nodiscard]] std::uint32_t position() const noexcept;
    [[nodiscard]] std::uint32_t horizon_capacity() const noexcept;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    // Preparation only. Exercises prefill, decode, every dependent proposal
    // depth and probability readback without consuming any external RNG.
    void warm_up();
    void reset() noexcept;
    // Truncation is logical and invalidates proposal leases. The next head
    // evaluation uses the retained normalized prefix, never a full replay.
    void truncate(std::uint32_t position);
    [[nodiscard]] VicunaDraftExecutionStats append(std::span<const std::uint32_t> tokens);
    [[nodiscard]] VicunaDraftProposal propose(std::uint32_t target_root,
        std::span<const std::uint32_t> proposal_words, float temperature,
        const SamplerHistorySnapshot& history);
    // HF-4.50 TLI processor order: native-history repetition, intersection,
    // temperature, top-k (all cutoff ties), ascending-tail top-p, min-p.
    // Repetition sees the entire retained native context, as in the source;
    // options.repetition_window describes p and is not a draft-history limit.
    // Ties use score-descending/token-ID-ascending order, an explicit native
    // convention (the source's unstable torch.sort tie order is unspecified).
    [[nodiscard]] VicunaDraftProposal propose(std::uint32_t target_root,
        std::span<const std::uint32_t> proposal_words, const TokenSamplerOptions& options,
        const SamplerHistorySnapshot& history);
    [[nodiscard]] VerifierDeviceInputBinding verifier_inputs(std::uint64_t epoch) const;
    [[nodiscard]] GatheredDraftTokenDistribution gather(std::uint32_t depth,
        const PreparedTokenDistribution& target);
    [[nodiscard]] std::uint64_t probability_readback_bytes() const noexcept;

    // Diagnostics copy only into caller-owned storage and use the exact same
    // resident state as production. They are not a second inference path.
    [[nodiscard]] VicunaDraftExecutionStats read_logits(std::span<float> output);
    void read_cache(std::uint32_t layer, std::span<std::uint16_t> keys,
        std::span<std::uint16_t> values) const;

#if defined(LOCAL_GGUF_ENGINE_TESTING)
    // Test executable only: complete geometry, same execution graph, synthetic
    // weights or controlled logits. No artifact-authentication bypass exists
    // in a production executable.
    VicunaDraftPlan(std::span<const tensor::TensorView> tensors,
        std::span<const std::int32_t> draft_to_target,
        std::uint32_t target_vocabulary_size, std::uint32_t horizon_capacity);
    void sample_logits_for_testing(std::span<const float> logits,
        std::uint32_t word, float temperature);
    void sample_logits_for_testing(std::span<const float> logits,
        std::uint32_t word, const TokenSamplerOptions& options);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace local_gguf_engine::engine
