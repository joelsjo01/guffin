#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <span>

namespace local_gguf_engine::engine {

inline constexpr std::size_t token_sampler_max_top_k = 256;
inline constexpr std::size_t token_sampler_max_repetition_window = 64;
inline constexpr std::size_t external_speculation_max_horizon = 23;
inline constexpr std::size_t stream_speculation_max_horizon = 23;
inline constexpr std::size_t sampler_max_preview_ticket_count =
    2U * stream_speculation_max_horizon + 1U;
inline constexpr std::size_t sampler_ticket_max_raw_engine_outputs = 4;
inline constexpr std::size_t token_sampler_max_base_candidates =
    token_sampler_max_top_k + token_sampler_max_repetition_window;
inline constexpr std::size_t token_sampler_max_candidates =
    token_sampler_max_base_candidates + token_sampler_max_repetition_window;

struct TokenLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

// Raw chronological repetition history is retained as a ring. Duplicates are
// intentional: hypothetical suffixes must evict occurrences before the
// candidate request deduplicates token IDs.
struct SamplerHistorySnapshot {
    std::array<std::uint32_t, token_sampler_max_repetition_window> token_ids {};
    std::uint32_t head = 0;
    std::uint32_t length = 0;
    std::uint32_t window_limit = 0;

    constexpr bool operator==(const SamplerHistorySnapshot&) const = default;
};

struct DeviceSelectionRequest {
    SamplerHistorySnapshot history;
    std::array<std::uint32_t, external_speculation_max_horizon> suffix_token_ids {};
    std::uint32_t suffix_count = 0;
};

struct TokenSelectionRequest {
    std::uint32_t top_k = 1;
    std::uint32_t base_candidate_count = 1;
    std::array<std::uint32_t, token_sampler_max_repetition_window> history_token_ids {};
    std::uint32_t history_token_count = 0;

    constexpr bool operator==(const TokenSelectionRequest&) const = default;
};

struct CompactTokenCandidates {
    std::array<TokenLogit, token_sampler_max_candidates> values {};
    std::uint32_t count = 0;
    std::uint32_t readback_bytes = 0;
};

struct PreparedTokenProbability {
    std::uint32_t token = 0;
    double probability = 0.0;
    double sampling_weight = 0.0;
};

struct PreparedTokenDistribution {
    std::array<PreparedTokenProbability, token_sampler_max_top_k> values {};
    std::uint32_t count = 0;
    double total_sampling_weight = 0.0;
};

enum class SpeculativeSamplingMode : std::uint8_t { exact_match, lossless };

// The proposal law is the actual inverse-CDF law on a uniform 32-bit word,
// not an ideal softmax reconstructed after device sampling. Positive masses
// must sum to 2^32, in precisely the order used by device selection.
inline constexpr std::uint64_t draft_probability_mass_total = std::uint64_t {1} << 32U;

struct DraftTokenProbability {
    std::uint32_t token = 0;
    std::uint64_t probability_mass = 0;
};

struct DraftTokenDistribution {
    std::array<DraftTokenProbability, token_sampler_max_top_k> values {};
    std::uint32_t count = 0;
};

struct DraftProbabilityLeaseState;
class DraftProbabilityLeaseOwner;

// A lease retains only its small lifetime record, not device probability
// storage. Destroying/invalidation of the producer therefore fails closed even
// if an old gathered readback outlives it. No allocation occurs when leasing.
class DraftProbabilityLease final {
public:
    DraftProbabilityLease() = default;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t row() const noexcept;

private:
    friend class DraftProbabilityLeaseOwner;
    std::shared_ptr<const DraftProbabilityLeaseState> state_;
    std::uint64_t epoch_ = 0;
    std::uint32_t row_ = 0;
};

class DraftProbabilityLeaseOwner final {
public:
    DraftProbabilityLeaseOwner();
    ~DraftProbabilityLeaseOwner();
    DraftProbabilityLeaseOwner(const DraftProbabilityLeaseOwner&) = delete;
    DraftProbabilityLeaseOwner& operator=(const DraftProbabilityLeaseOwner&) = delete;
    DraftProbabilityLeaseOwner(DraftProbabilityLeaseOwner&&) = delete;
    DraftProbabilityLeaseOwner& operator=(DraftProbabilityLeaseOwner&&) = delete;

    // One epoch covers all rows of one proposal chain; rewriting any of that
    // chain's distributions must start a new epoch or invalidate this owner.
    void begin_epoch();
    void invalidate() noexcept;
    [[nodiscard]] DraftProbabilityLease lease(std::uint32_t row) const;

private:
    std::shared_ptr<DraftProbabilityLeaseState> state_;
};

// Sparse lookup into a full, producer-validated 2^32-mass proposal law. Entries
// are exactly the prepared target support, in its order, including q=0 tokens.
// Their sum need NOT be 2^32 and must never be renormalized. The selected bin is
// contiguous because TLI's authenticated vocabulary mapping is injective.
struct GatheredDraftTokenDistribution {
    DraftProbabilityLease lease;
    DraftTokenProbability proposal;
    std::uint64_t proposal_cdf_begin = 0;
    std::array<DraftTokenProbability, token_sampler_max_top_k> target_values {};
    std::uint32_t count = 0;
};

struct ProposalRandomWords {
    std::array<std::uint32_t, external_speculation_max_horizon> words {};
    std::uint32_t count = 0;
};

struct SpeculativeTokenSelection {
    std::uint32_t token = 0;
    bool accepted = false;
};

struct PreparedSpeculativeDistribution {
    std::uint32_t proposal_token = 0;
    double acceptance_probability = 0.0;
    PreparedTokenDistribution residual;
};

[[nodiscard]] std::uint32_t select_draft_token(
    const DraftTokenDistribution& distribution,
    std::uint32_t random_word
);

[[nodiscard]] PreparedSpeculativeDistribution prepare_speculative_distribution(
    std::uint32_t proposal,
    const DraftTokenDistribution& draft,
    const PreparedTokenDistribution& target
);

[[nodiscard]] PreparedSpeculativeDistribution prepare_speculative_distribution(
    std::uint32_t proposal,
    const GatheredDraftTokenDistribution& draft,
    const PreparedTokenDistribution& target
);

// Injected independent unit-interval draws make the probability equations
// independently testable. The residual draw is used only on rejection.
[[nodiscard]] SpeculativeTokenSelection sample_speculative_token(
    std::uint32_t proposal,
    const PreparedSpeculativeDistribution& distribution,
    double acceptance_draw,
    double residual_draw
);

struct TokenSamplerOptions {
    static constexpr std::uint64_t random_seed = std::numeric_limits<std::uint64_t>::max();

    float temperature = 0.0F;
    std::size_t top_k = 40;
    float top_p = 0.95F;
    float min_p = 0.05F;
    float repetition_penalty = 1.0F;
    std::uint64_t seed = random_seed;
    std::size_t repetition_window = 64;

    friend bool operator==(const TokenSamplerOptions&, const TokenSamplerOptions&) = default;
};

[[nodiscard]] SamplerHistorySnapshot append_sampler_history(
    const SamplerHistorySnapshot& history,
    std::span<const std::uint32_t> suffix
);

[[nodiscard]] TokenSelectionRequest make_token_selection_request(
    const TokenSamplerOptions& options,
    const SamplerHistorySnapshot& history
);

[[nodiscard]] TokenSelectionRequest make_token_selection_request(
    const TokenSamplerOptions& options,
    const DeviceSelectionRequest& request
);

[[nodiscard]] PreparedTokenDistribution prepare_token_distribution(
    std::span<const TokenLogit> compact_candidates,
    const TokenSamplerOptions& options,
    const TokenSelectionRequest& request
);

class TokenSampler;
class TokenSamplerFork;

class SamplerCheckpoint final {
public:
    SamplerCheckpoint() = default;

    [[nodiscard]] const SamplerHistorySnapshot& history() const noexcept;
    [[nodiscard]] std::uint64_t draw_count() const noexcept;
    [[nodiscard]] std::uint64_t proposal_draw_count() const noexcept;
    [[nodiscard]] bool same_state(const SamplerCheckpoint& other) const noexcept;
    [[nodiscard]] bool follows_sample(
        const SamplerCheckpoint& previous,
        std::uint32_t observed_token
    ) const;
    [[nodiscard]] bool follows_speculative(
        const SamplerCheckpoint& previous,
        std::uint32_t observed_token,
        bool accepted
    ) const;

private:
    friend class TokenSampler;
    friend class TokenSamplerFork;

    SamplerCheckpoint(
        TokenSamplerOptions options,
        std::mt19937_64 rng,
        std::mt19937_64 proposal_rng,
        SamplerHistorySnapshot history,
        std::uint64_t lineage_id,
        std::uint64_t draw_count,
        std::uint64_t proposal_draw_count
    );

    TokenSamplerOptions options_;
    std::mt19937_64 rng_;
    std::mt19937_64 proposal_rng_;
    SamplerHistorySnapshot history_;
    std::uint64_t lineage_id_ = 0;
    std::uint64_t draw_count_ = 0;
    std::uint64_t proposal_draw_count_ = 0;
};

class SamplerTicket final {
public:
    SamplerTicket() = default;
    friend bool operator==(const SamplerTicket&, const SamplerTicket&) = default;

    [[nodiscard]] std::uint32_t offset() const noexcept;
    [[nodiscard]] std::uint64_t draw_index() const noexcept;
    [[nodiscard]] bool consumes_random_draw() const noexcept;
    [[nodiscard]] double unit_interval_draw() const noexcept;
    [[nodiscard]] std::uint32_t raw_engine_output_count() const noexcept;
    [[nodiscard]] std::uint64_t raw_engine_output(std::size_t index) const;
    [[nodiscard]] double mapped_draw(double total_sampling_weight) const;

private:
    friend class TokenSampler;
    friend class TokenSamplerFork;

    TokenSamplerOptions options_;
    std::mt19937_64 pre_rng_;
    std::mt19937_64 post_rng_;
    std::array<std::uint64_t, sampler_ticket_max_raw_engine_outputs> raw_engine_outputs_ {};
    std::uint64_t lineage_id_ = 0;
    std::uint64_t draw_index_ = 0;
    double unit_interval_draw_ = 0.0;
    std::uint32_t offset_ = 0;
    std::uint32_t raw_engine_output_count_ = 0;
    bool consumes_random_draw_ = false;
};

struct SamplerTicketPool {
    std::array<SamplerTicket, sampler_max_preview_ticket_count> tickets {};
    std::uint32_t count = 0;

    [[nodiscard]] const SamplerTicket& at(std::size_t offset) const;
};

class TokenSamplerFork final {
public:
    TokenSamplerFork(const TokenSamplerFork&) = default;
    TokenSamplerFork& operator=(const TokenSamplerFork&) = default;
    TokenSamplerFork(TokenSamplerFork&&) noexcept = default;
    TokenSamplerFork& operator=(TokenSamplerFork&&) noexcept = default;

    [[nodiscard]] bool is_greedy() const noexcept;
    [[nodiscard]] const SamplerHistorySnapshot& history_snapshot() const noexcept;
    [[nodiscard]] TokenSelectionRequest selection_request() const noexcept;
    [[nodiscard]] DeviceSelectionRequest device_selection_request(
        std::span<const std::uint32_t> hypothetical_suffix = {}
    ) const;
    [[nodiscard]] std::uint32_t sample_with_ticket(
        const SamplerTicket& ticket,
        std::span<const TokenLogit> compact_candidates
    );
    [[nodiscard]] std::uint32_t sample(std::span<const TokenLogit> compact_candidates);
    [[nodiscard]] PreparedTokenDistribution prepare_distribution(
        std::span<const TokenLogit> compact_candidates
    ) const;
    [[nodiscard]] SpeculativeTokenSelection sample_speculative(
        std::uint32_t proposal,
        const DraftTokenDistribution& draft,
        std::span<const TokenLogit> compact_target_candidates
    );
    [[nodiscard]] SpeculativeTokenSelection sample_speculative(
        std::uint32_t proposal,
        const GatheredDraftTokenDistribution& draft,
        std::span<const TokenLogit> compact_target_candidates
    );
    [[nodiscard]] ProposalRandomWords preview_proposal_words(std::size_t count) const;
    void observe_tokens(std::span<const std::uint32_t> token_ids);
    [[nodiscard]] SamplerCheckpoint checkpoint() const noexcept;

private:
    friend class TokenSampler;
    explicit TokenSamplerFork(const SamplerCheckpoint& checkpoint);

    TokenSamplerOptions options_;
    std::mt19937_64 rng_;
    std::mt19937_64 proposal_rng_;
    SamplerHistorySnapshot history_;
    std::uint64_t lineage_id_ = 0;
    std::uint64_t draw_count_ = 0;
    std::uint64_t proposal_draw_count_ = 0;
};

class TokenSampler final {
public:
    explicit TokenSampler(TokenSamplerOptions options = {});
    TokenSampler(const TokenSampler& other) = default;
    TokenSampler& operator=(const TokenSampler& other) = default;
    TokenSampler(TokenSampler&& other) noexcept = default;
    TokenSampler& operator=(TokenSampler&& other) noexcept = default;

    [[nodiscard]] bool is_greedy() const noexcept;
    [[nodiscard]] const TokenSamplerOptions& options() const noexcept;
    [[nodiscard]] const SamplerHistorySnapshot& history_snapshot() const noexcept;
    [[nodiscard]] TokenSelectionRequest selection_request() const noexcept;
    [[nodiscard]] TokenSelectionRequest selection_request_after_observing(
        std::uint32_t token_id
    ) const noexcept;
    [[nodiscard]] TokenSelectionRequest selection_request_after_observing(
        std::span<const std::uint32_t> token_ids
    ) const;
    [[nodiscard]] DeviceSelectionRequest device_selection_request(
        std::span<const std::uint32_t> hypothetical_suffix = {}
    ) const;
    [[nodiscard]] TokenSelectionRequest selection_request(
        const DeviceSelectionRequest& request
    ) const;
    [[nodiscard]] std::uint32_t sample(std::span<const TokenLogit> compact_candidates);
    void observe_tokens(std::span<const std::uint32_t> token_ids);
    void reset_history(std::span<const std::uint32_t> token_ids = {});

    [[nodiscard]] SamplerCheckpoint checkpoint() const noexcept;
    [[nodiscard]] TokenSamplerFork fork(const SamplerCheckpoint& checkpoint) const;
    [[nodiscard]] TokenSamplerFork fork() const;
    void adopt_checkpoint(const SamplerCheckpoint& checkpoint);
    [[nodiscard]] SamplerTicket preview_ticket(std::size_t offset) const;
    [[nodiscard]] SamplerTicketPool preview_tickets(std::size_t count) const;
    [[nodiscard]] SamplerTicketPool preview_external_tickets() const;
    [[nodiscard]] SamplerTicketPool preview_stream_tickets() const;
    [[nodiscard]] ProposalRandomWords preview_proposal_words(std::size_t count) const;

private:
    TokenSamplerOptions options_;
    std::mt19937_64 rng_;
    std::mt19937_64 proposal_rng_;
    SamplerHistorySnapshot history_;
    std::uint64_t lineage_id_ = 0;
    std::uint64_t draw_count_ = 0;
    std::uint64_t proposal_draw_count_ = 0;
};

}  // namespace local_gguf_engine::engine
