#pragma once

#include "local_gguf_engine/engine/chain_chat.hpp"
#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"
#include "local_gguf_engine/engine/vicuna_draft.hpp"

#include <optional>

namespace local_gguf_engine::engine {

class Gemma3DraftPlan;

// Absent scalars inherit the authoritative target settings. Only q changes:
// proposal randomness remains checkpoint-owned and native draft repetition
// history remains the complete retained draft context, not a new window.
struct StandaloneDraftSamplingOptions {
    std::optional<float> temperature;
    std::optional<std::uint32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> repetition_penalty;
    friend bool operator==(const StandaloneDraftSamplingOptions&, const StandaloneDraftSamplingOptions&) = default;
};

// Full-vocabulary draft top_k may be zero (unrestricted) or exceed the target
// sampler's compact-support capacity. The returned seed/window are unchanged
// target metadata; this function neither creates nor advances a random stream.
[[nodiscard]] TokenSamplerOptions resolve_standalone_draft_sampler_options(
    const TokenSamplerOptions& target, const StandaloneDraftSamplingOptions& overrides,
    std::uint32_t draft_vocabulary_size);

struct StandaloneExecutionOptions {
    std::uint32_t horizon = 3;
    bool conditional_verification = false;
    StandaloneDraftSamplingOptions draft_sampling {};
    friend bool operator==(const StandaloneExecutionOptions&, const StandaloneExecutionOptions&) = default;
};

struct StandaloneCycleResult : ChainCycleResult {
    VicunaDraftExecutionStats draft;
};

struct StandaloneChatCheckpointState;
class StandaloneChatCheckpoint final {
public:
    StandaloneChatCheckpoint() = default;
    ~StandaloneChatCheckpoint();
    StandaloneChatCheckpoint(StandaloneChatCheckpoint&&) noexcept;
    StandaloneChatCheckpoint& operator=(StandaloneChatCheckpoint&&) noexcept;
    StandaloneChatCheckpoint(const StandaloneChatCheckpoint&) = delete;
    StandaloneChatCheckpoint& operator=(const StandaloneChatCheckpoint&) = delete;
    [[nodiscard]] bool valid() const noexcept;
private:
    friend class StandaloneChatPlan;
    std::shared_ptr<StandaloneChatCheckpointState> state_;
    std::uint64_t epoch_ = 0;
};

// The target ledger, draft emission ledger, and draft's local RoPE positions
// are distinct. Durable checkpoints record the actual append interpretation,
// including context-window rebuilds, rather than retokenizing accepted text.
class StandaloneChatPlan final {
public:
    StandaloneChatPlan(std::shared_ptr<StandaloneDraftArtifact> source,
        const tokenizer::Tokenizer& target_tokenizer, const Gemma2MetalModelPlan& target,
        StandaloneExecutionOptions execution);
    ~StandaloneChatPlan();
    StandaloneChatPlan(StandaloneChatPlan&&) noexcept;
    StandaloneChatPlan& operator=(StandaloneChatPlan&&) noexcept;
    StandaloneChatPlan(const StandaloneChatPlan&) = delete;
    StandaloneChatPlan& operator=(const StandaloneChatPlan&) = delete;
    [[nodiscard]] std::uint32_t horizon() const noexcept;
    [[nodiscard]] StandaloneDraftKind kind() const noexcept;
    [[nodiscard]] const StandaloneExecutionOptions& execution_options() const noexcept;
    [[nodiscard]] TokenSamplerOptions effective_draft_sampler_options(const TokenSamplerOptions& target) const;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::uint32_t target_position() const noexcept;
    [[nodiscard]] std::uint64_t rebuild_duration_ns() const noexcept;
    [[nodiscard]] VicunaDraftExecutionStats execution_stats() const noexcept;
    void reset();
    void warm_up(Gemma2MetalModelPlan& target, const TokenSampler& sampler);
    // Rebased prompts discard the old interpretation only after complete
    // conversion/capacity validation has succeeded.
    void synchronize_context(std::span<const std::uint32_t> target_tokens, bool replace_context = false);
    // Dry-run the complete native conversion before an ordinary/retrieval
    // target transaction publishes these tokens. No cache/ledger/RNG mutation.
    void validate_committed_prefix(std::span<const std::uint32_t> tokens);
    void observe_committed(std::uint32_t token, std::uint32_t target_position);
    // One prepare-allocated snapshot slot preserves the actual tokenizer
    // append interpretation across a failed regeneration. Explicit restore or
    // discard consumes its move-only ticket; nested snapshots fail closed.
    [[nodiscard]] StandaloneChatCheckpoint checkpoint();
    void restore(StandaloneChatCheckpoint& checkpoint);
    void discard(StandaloneChatCheckpoint& checkpoint);
    [[nodiscard]] StandaloneCycleResult run(Gemma2MetalModelPlan& target, TokenSampler& sampler,
        std::uint32_t root, std::uint32_t position, std::size_t remaining,
        std::span<const std::uint32_t> stops, const ChainCommitCallback& callback);
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    StandaloneChatPlan(std::span<const tensor::TensorView> full_draft_tensors,
        const tokenizer::Tokenizer& draft_tokenizer, const tokenizer::Tokenizer& target_tokenizer,
        const Gemma2MetalModelPlan& target, StandaloneExecutionOptions execution);
    [[nodiscard]] const VicunaDraftPlan& draft_plan_for_testing() const noexcept;
    [[nodiscard]] const Gemma3DraftPlan* gemma3_draft_plan_for_testing() const noexcept;
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace local_gguf_engine::engine
