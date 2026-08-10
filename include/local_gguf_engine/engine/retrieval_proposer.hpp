#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::size_t retrieval_proposer_max_context = 8192U;
inline constexpr std::size_t retrieval_proposer_min_order = 2U;
inline constexpr std::size_t retrieval_proposer_max_order = 16U;

struct RetrievalProposal {
    std::uint32_t token_id = 0;
    std::uint32_t match_length = 0;
    std::uint32_t support_count = 0;
    std::uint32_t total_count = 0;
    double confidence = 0.0;
};

// A session-local, allocation-free-on-query suffix continuation proposer.
// Prefix hashes reject almost every non-match cheaply; exact token comparison
// makes hash collisions correctness-neutral.
class RetrievalProposer final {
public:
    RetrievalProposer();

    void rebuild(std::span<const std::uint32_t> live_token_ids);
    void observe(std::uint32_t token_id);
    [[nodiscard]] RetrievalProposal propose() const;
    [[nodiscard]] std::span<const std::uint32_t> history() const noexcept;

private:
    [[nodiscard]] std::uint64_t range_hash(std::size_t begin, std::size_t end) const noexcept;

    std::vector<std::uint32_t> history_;
    std::vector<std::uint64_t> prefix_hashes_;
    std::vector<std::uint64_t> hash_powers_;
    mutable std::vector<std::uint32_t> continuation_tokens_;
    mutable std::vector<std::uint32_t> continuation_counts_;
};

[[nodiscard]] bool retrieval_proposal_is_profitable(const RetrievalProposal& proposal) noexcept;

}  // namespace local_gguf_engine::engine
