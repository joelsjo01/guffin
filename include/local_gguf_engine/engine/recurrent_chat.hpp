#pragma once

#include "local_gguf_engine/engine/learned_recurrent.hpp"
#include "local_gguf_engine/engine/chain_chat.hpp"

namespace local_gguf_engine::engine {

struct RecurrentExecutionOptions {
    RecurrentDraftHead draft_head = RecurrentDraftHead::f16;
    bool conditional_verification = false;

    friend bool operator==(const RecurrentExecutionOptions&, const RecurrentExecutionOptions&) = default;
};

void validate_recurrent_execution_options(
    SpeculativeSamplingMode sampling_mode, const RecurrentExecutionOptions& execution
);

using RecurrentCommittedToken = ChainCommittedToken;
using RecurrentCommitCallback = ChainCommitCallback;
struct RecurrentCycleResult : ChainCycleResult {
    RecurrentExecutionResult draft;
};

// Explicit weights-only development execution. No fitted routing/admission
// policy is implied. Owns fixed proposal/verification storage and leases the
// final target feature until the next draft has consumed it, without replay.
class RecurrentChatPlan final {
public:
    RecurrentChatPlan(Gemma2SpeculativePlan draft, const Gemma2MetalModelPlan& target,
        SpeculativeSamplingMode sampling_mode = SpeculativeSamplingMode::exact_match,
        RecurrentExecutionOptions execution = {});
    ~RecurrentChatPlan();
    RecurrentChatPlan(RecurrentChatPlan&&) noexcept;
    RecurrentChatPlan& operator=(RecurrentChatPlan&&) noexcept;
    RecurrentChatPlan(const RecurrentChatPlan&) = delete;
    RecurrentChatPlan& operator=(const RecurrentChatPlan&) = delete;
    void validate_sampler(const TokenSamplerOptions&) const;
    [[nodiscard]] std::uint32_t horizon() const noexcept;
    [[nodiscard]] SpeculativeSamplingMode sampling_mode() const noexcept;
    [[nodiscard]] const RecurrentExecutionOptions& execution_options() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    void invalidate() noexcept;
    // Preparation only: after the first setup cycle, exercise every remaining
    // tile shape on setup K/V. Consumes no sampler draws or external callbacks;
    // the caller's logical context must still be empty.
    void warm_up_verifier_tiles(Gemma2MetalModelPlan& target);
    void publish_prefill_origin(std::uint32_t position);
    void publish_decode_origin(std::uint32_t position);
    void publish_retrieval_origin(std::uint32_t row, std::uint32_t position);
    [[nodiscard]] RecurrentCycleResult run(
        Gemma2MetalModelPlan& target, TokenSampler& sampler,
        std::uint32_t root, std::uint32_t cache_position,
        std::size_t remaining_tokens, std::span<const std::uint32_t> stops,
        const RecurrentCommitCallback& on_committed_token
    );
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace local_gguf_engine::engine
