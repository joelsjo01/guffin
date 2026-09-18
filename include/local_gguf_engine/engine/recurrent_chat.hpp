#pragma once

#include "local_gguf_engine/engine/learned_recurrent.hpp"

namespace local_gguf_engine::engine {

struct RecurrentCommittedToken {
    std::uint32_t token_id = 0;
    std::uint32_t depth = 0;
    std::uint32_t cached_token_count = 0;
    bool stop_token = false;
};
// Synchronous borrowed callback. Binding an lvalue makes its lifetime explicit
// and avoids constructing a heap-owning std::function in every draft cycle.
class RecurrentCommitCallback final {
public:
    template<class Callable> RecurrentCommitCallback(Callable& callback) noexcept
        : context_(&callback), invoke_([](const void* context, const RecurrentCommittedToken& token) {
            return (*static_cast<const Callable*>(context))(token);
        }) {}
    [[nodiscard]] Gemma2MetalTokenStreamDecision operator()(const RecurrentCommittedToken& token) const {
        return invoke_(context_, token);
    }
private:
    const void* context_;
    Gemma2MetalTokenStreamDecision (*invoke_)(const void*, const RecurrentCommittedToken&);
};

struct RecurrentCycleResult {
    RecurrentExecutionResult draft;
    TargetVerifierExecutionStats target;
    Gemma2MetalDecodeExecutionStats endpoint;
    std::uint32_t retained_rows = 0;
    std::uint32_t accepted_edges = 0;
    std::optional<std::uint32_t> carry;
    std::optional<std::uint32_t> stop_token;
    bool callback_stopped = false;
    bool has_next_candidates = false;
    CompactTokenCandidates next_candidates;
};

// Explicit weights-only development execution. No fitted routing/admission
// policy is implied. Owns fixed proposal/verification storage and leases the
// final target feature until the next draft has consumed it, without replay.
class RecurrentChatPlan final {
public:
    RecurrentChatPlan(Gemma2SpeculativePlan draft, const Gemma2MetalModelPlan& target);
    ~RecurrentChatPlan();
    RecurrentChatPlan(RecurrentChatPlan&&) noexcept;
    RecurrentChatPlan& operator=(RecurrentChatPlan&&) noexcept;
    RecurrentChatPlan(const RecurrentChatPlan&) = delete;
    RecurrentChatPlan& operator=(const RecurrentChatPlan&) = delete;
    void validate_sampler(const TokenSamplerOptions&) const;
    [[nodiscard]] std::uint32_t horizon() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    void invalidate() noexcept;
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
