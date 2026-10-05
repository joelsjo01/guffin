#pragma once

#include "local_gguf_engine/engine/standalone_draft_types.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"

namespace local_gguf_engine::engine {

class StandaloneDraftArtifact;

struct Gemma3DraftProposal {
    // The final token is deliberately pending, not yet forwarded through the
    // transformer. A subsequent append/head read materializes it; rejecting
    // that suffix discards it without executing work that cannot be reused.
    std::span<const std::uint32_t> draft_tokens;
    std::span<const std::uint32_t> target_tokens;
    std::uint32_t start_position = 0;
    std::uint32_t materialized_end = 0;
    std::uint64_t epoch = 0;
    StandaloneDraftExecutionStats execution;
};

// Complete Q8_0 Gemma3-270M transformer, with an independent native timeline.
// Resident local rings are never overwritten by uncommitted proposals. A
// transaction owns private append K/V and its normalized-prefix states until
// commit_prefix publishes the controller's durable prefix or abort discards it.
// K/V stays F32: intermediate F16 rounding amplifies otherwise benign Q8/F32
// reduction differences across layers, including under independent CPU replay.
class Gemma3DraftPlan final {
public:
    static constexpr std::uint32_t vocabulary_size = 262144;
    static constexpr std::uint32_t context_capacity = 32768;
    static constexpr std::uint32_t layer_count = 18;

    Gemma3DraftPlan(const StandaloneDraftArtifact& artifact,
        std::span<const std::int32_t> draft_to_target,
        std::uint32_t target_vocabulary_size, std::uint32_t horizon_capacity = 3,
        std::uint32_t transaction_capacity = 256,
        std::uint32_t context_rows = context_capacity);
    ~Gemma3DraftPlan();
    Gemma3DraftPlan(Gemma3DraftPlan&&) noexcept;
    Gemma3DraftPlan& operator=(Gemma3DraftPlan&&) noexcept;
    Gemma3DraftPlan(const Gemma3DraftPlan&) = delete;
    Gemma3DraftPlan& operator=(const Gemma3DraftPlan&) = delete;

    // Pure resident-layout estimate, including the full tied Q8 model, private
    // state, probability receipts and reverse mapping. No Metal allocation.
    [[nodiscard]] static std::size_t estimate_owned_bytes(std::uint32_t target_vocabulary_size,
        std::uint32_t horizon_capacity = 3, std::uint32_t transaction_capacity = 256,
        std::uint32_t context_rows = context_capacity);
    [[nodiscard]] std::uint32_t prepared_context_capacity() const noexcept;
    [[nodiscard]] std::uint32_t position() const noexcept;
    [[nodiscard]] std::uint32_t materialized_position() const noexcept;
    [[nodiscard]] std::uint32_t horizon_capacity() const noexcept;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] bool transaction_active() const noexcept;
    void warm_up();
    void reset() noexcept;
    [[nodiscard]] StandaloneDraftExecutionStats begin_transaction();
    // Finalizes and closes the transaction. Prefix includes a possible final
    // pending token: its K/V is materialized only when the next input needs it.
    [[nodiscard]] StandaloneDraftExecutionStats commit_prefix(std::uint32_t position);
    void abort_transaction() noexcept;
    [[nodiscard]] bool can_truncate(std::uint32_t position) const noexcept;
    void truncate(std::uint32_t position);
    [[nodiscard]] StandaloneDraftExecutionStats append(std::span<const std::uint32_t> tokens);
    // Restores a saved, already-selected native emission after replaying its
    // materialized prefix. This is not a proposal or a sampling operation.
    [[nodiscard]] StandaloneDraftExecutionStats restore_pending_token(std::uint32_t token);
    // Optional known root tokens and a prior pending token are ingested in the
    // same command submission as drafting. start_position is after that input.
    // Direct-ID pairing can explicitly use options.repetition_window on native
    // history; the default preserves the legacy full-prefix draft law.
    [[nodiscard]] Gemma3DraftProposal propose(std::uint32_t target_root,
        std::span<const std::uint32_t> proposal_words, const TokenSamplerOptions& options,
        const SamplerHistorySnapshot& history,
        std::span<const std::uint32_t> appended_tokens = {},
        bool bounded_repetition_history = false);
    [[nodiscard]] VerifierDeviceInputBinding verifier_inputs(std::uint64_t epoch) const;
    [[nodiscard]] GatheredDraftTokenDistribution gather(std::uint32_t depth,
        const PreparedTokenDistribution& target);
    [[nodiscard]] std::uint64_t probability_readback_bytes() const noexcept;
    [[nodiscard]] StandaloneDraftExecutionStats read_logits(std::span<float> output);
    // Logical-order diagnostic, limited to materialized rows. Local rows older
    // than their window are not present and are rejected, never fabricated.
    void read_cache(std::uint32_t layer, std::uint32_t first_position,
        std::span<float> keys, std::span<float> values) const;

#if defined(LOCAL_GGUF_ENGINE_TESTING)
    // Complete production geometry in native GGUF tensor order (embedding,
    // final norm; per layer q,k,v,o,gate,up,down,input,postattn,preff,postff,qnorm,knorm).
    Gemma3DraftPlan(std::span<const tensor::TensorView> tensors,
        std::span<const std::int32_t> map, std::uint32_t target_vocabulary,
        std::uint32_t horizon_capacity, std::uint32_t transaction_capacity = 256,
        std::uint32_t context_rows = context_capacity);
    void sample_logits_for_testing(std::span<const float> logits, std::uint32_t word,
        const TokenSamplerOptions& options, bool bounded_repetition_history = false);
    // Attention boundary fixture only: a declared prefix of exactly-zero K/V.
    // Does not claim those synthetic rows are a transformer-generated prompt.
    void seed_zero_cache_prefix_for_testing(std::uint32_t position);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace local_gguf_engine::engine
