#pragma once

#include "local_gguf_engine/engine/target_verifier.hpp"

namespace local_gguf_engine::engine {

struct ChainCommittedToken {
    std::uint32_t token_id = 0;
    std::uint32_t depth = 0;
    std::uint32_t cached_token_count = 0;
    bool stop_token = false;
};

// Synchronous borrowed callbacks do not allocate or extend the producer's
// lifetime. The producer must retain each proposal law through verification.
class ChainCommitCallback final {
public:
    template<class Callable> ChainCommitCallback(Callable& callback) noexcept
        : context_(&callback), invoke_([](const void* context, const ChainCommittedToken& token) {
            return (*static_cast<const Callable*>(context))(token);
        }) {}
    [[nodiscard]] Gemma2MetalTokenStreamDecision operator()(const ChainCommittedToken& token) const {
        return invoke_(context_, token);
    }
private:
    const void* context_;
    Gemma2MetalTokenStreamDecision (*invoke_)(const void*, const ChainCommittedToken&);
};

class ChainSampleCallback final {
public:
    template<class Callable> ChainSampleCallback(Callable& callback) noexcept
        : context_(&callback), invoke_([](const void* context, TokenSamplerFork& fork,
            std::uint32_t depth, std::span<const TokenLogit> candidates) {
            return (*static_cast<const Callable*>(context))(fork, depth, candidates);
        }) {}
    [[nodiscard]] SpeculativeTokenSelection operator()(TokenSamplerFork& fork,
        std::uint32_t depth, std::span<const TokenLogit> candidates) const {
        return invoke_(context_, fork, depth, candidates);
    }
private:
    const void* context_;
    SpeculativeTokenSelection (*invoke_)(const void*, TokenSamplerFork&,
        std::uint32_t, std::span<const TokenLogit>);
};

struct ChainProposalBatch {
    std::span<const std::uint32_t> tokens;
    std::span<const float> confidence_logits;
    ProposalFeatureSource source = ProposalFeatureSource::draft_hidden;
    VerifierDeviceInputBinding device;
};

struct ChainCycleResult {
    TargetVerifierExecutionStats target;
    Gemma2MetalDecodeExecutionStats endpoint;
    std::uint32_t proposal_count = 0;
    std::uint32_t retained_rows = 0;
    std::uint32_t accepted_edges = 0;
    // Only an authoritative correction selected by the controller counts as
    // rejection. A preview discarded by a callback/stop/budget is censored.
    bool rejected = false;
    std::optional<std::uint32_t> carry;
    std::optional<std::uint32_t> stop_token;
    bool callback_stopped = false;
    bool has_next_candidates = false;
    CompactTokenCandidates next_candidates;
};

// Target transaction and callback-prefix ownership are independent of how a
// chain was drafted. R feature leases and autonomous draft K/V stay outside.
class ChainChatVerifier final {
public:
    ChainChatVerifier(const Gemma2MetalModelPlan& target, std::uint32_t horizon,
        SpeculativeSamplingMode mode, bool conditional);
    ~ChainChatVerifier();
    ChainChatVerifier(ChainChatVerifier&&) noexcept;
    ChainChatVerifier& operator=(ChainChatVerifier&&) noexcept;
    ChainChatVerifier(const ChainChatVerifier&) = delete;
    ChainChatVerifier& operator=(const ChainChatVerifier&) = delete;
    [[nodiscard]] static std::size_t estimate_owned_bytes(const Gemma2ModelConfig& config,
        std::uint32_t horizon, bool conditional);
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    [[nodiscard]] const tensor::BackendBuffer& last_hidden() const;
    void warm_up_tiles(Gemma2MetalModelPlan& target);
    [[nodiscard]] ChainCycleResult run(Gemma2MetalModelPlan& target, TokenSampler& sampler,
        std::uint32_t root, std::uint32_t position, std::size_t remaining,
        std::span<const std::uint32_t> stops, const ChainProposalBatch& batch,
        const ChainSampleCallback& sample, const ChainCommitCallback& callback);
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    [[nodiscard]] static ChainVerifierSchedule schedule_for_testing(std::uint32_t prepared_horizon,
        std::uint32_t active_horizon, bool conditional);
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace local_gguf_engine::engine
