#include "local_gguf_engine/engine/retrieval_proposer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace local_gguf_engine::engine {

namespace {

constexpr std::uint64_t hash_base = 0x9e3779b185ebca87ULL;

[[nodiscard]] std::uint64_t append_hash(std::uint64_t hash, std::uint32_t token) noexcept {
    return hash * hash_base + static_cast<std::uint64_t>(token) + 1U;
}

}  // namespace

RetrievalProposer::RetrievalProposer() {
    history_.reserve(retrieval_proposer_max_context);
    prefix_hashes_.reserve(retrieval_proposer_max_context + 1U);
    hash_powers_.resize(retrieval_proposer_max_context + 1U);
    continuation_tokens_.resize(retrieval_proposer_max_context);
    continuation_counts_.resize(retrieval_proposer_max_context);
    prefix_hashes_.push_back(0U);
    hash_powers_[0] = 1U;
    for (std::size_t index = 1; index < hash_powers_.size(); ++index) {
        hash_powers_[index] = hash_powers_[index - 1U] * hash_base;
    }
}

void RetrievalProposer::rebuild(std::span<const std::uint32_t> live_token_ids) {
    if (live_token_ids.size() > retrieval_proposer_max_context) {
        throw std::invalid_argument("retrieval proposer history exceeds the 8192-token context");
    }
    history_.assign(live_token_ids.begin(), live_token_ids.end());
    prefix_hashes_.clear();
    prefix_hashes_.push_back(0U);
    for (const auto token : history_) {
        prefix_hashes_.push_back(append_hash(prefix_hashes_.back(), token));
    }
}

void RetrievalProposer::observe(std::uint32_t token_id) {
    if (history_.size() == retrieval_proposer_max_context) {
        throw std::length_error("retrieval proposer cannot observe beyond the live context");
    }
    history_.push_back(token_id);
    prefix_hashes_.push_back(append_hash(prefix_hashes_.back(), token_id));
}

std::uint64_t RetrievalProposer::range_hash(std::size_t begin, std::size_t end) const noexcept {
    return prefix_hashes_[end] - prefix_hashes_[begin] * hash_powers_[end - begin];
}

RetrievalProposal RetrievalProposer::propose() const {
    if (history_.size() <= retrieval_proposer_min_order) {
        return {};
    }
    const auto max_order = std::min(retrieval_proposer_max_order, history_.size() - 1U);
    for (auto order = max_order; order >= retrieval_proposer_min_order; --order) {
        const auto current_begin = history_.size() - order;
        const auto wanted_hash = range_hash(current_begin, history_.size());
        auto distinct = std::size_t {0};
        auto total = std::uint32_t {0};
        for (std::size_t begin = 0; begin + order < history_.size(); ++begin) {
            if (range_hash(begin, begin + order) != wanted_hash
                || !std::equal(
                    history_.begin() + static_cast<std::ptrdiff_t>(begin),
                    history_.begin() + static_cast<std::ptrdiff_t>(begin + order),
                    history_.begin() + static_cast<std::ptrdiff_t>(current_begin)
                )) {
                continue;
            }
            const auto continuation = history_[begin + order];
            auto slot = std::size_t {0};
            while (slot < distinct && continuation_tokens_[slot] != continuation) {
                ++slot;
            }
            if (slot == distinct) {
                continuation_tokens_[distinct] = continuation;
                continuation_counts_[distinct] = 0U;
                ++distinct;
            }
            if (continuation_counts_[slot] != std::numeric_limits<std::uint32_t>::max()) {
                ++continuation_counts_[slot];
            }
            if (total != std::numeric_limits<std::uint32_t>::max()) {
                ++total;
            }
        }
        if (total == 0U) {
            if (order == retrieval_proposer_min_order) break;
            continue;
        }
        auto best = std::size_t {0};
        for (std::size_t slot = 1; slot < distinct; ++slot) {
            if (continuation_counts_[slot] > continuation_counts_[best]
                || (continuation_counts_[slot] == continuation_counts_[best]
                    && continuation_tokens_[slot] < continuation_tokens_[best])) {
                best = slot;
            }
        }
        if (continuation_counts_[best] < 2U) {
            if (order == retrieval_proposer_min_order) break;
            continue;
        }
        return RetrievalProposal {
            .token_id = continuation_tokens_[best],
            .match_length = static_cast<std::uint32_t>(order),
            .support_count = continuation_counts_[best],
            .total_count = total,
            .confidence = static_cast<double>(continuation_counts_[best]) / static_cast<double>(total),
        };
    }
    return {};
}

std::span<const std::uint32_t> RetrievalProposer::history() const noexcept {
    return history_;
}

bool retrieval_proposal_is_profitable(const RetrievalProposal& proposal) noexcept {
    // Cooled production telemetry found 0/6 matches below 2/3 confidence,
    // while the 2/3-to-1 bucket remained near the zero-copy verifier's
    // break-even rate and was strongly profitable in the first turn. Use exact
    // integer evidence for the boundary; confidence remains a diagnostic.
    return proposal.match_length >= retrieval_proposer_min_order
        && proposal.support_count >= 2U
        && proposal.total_count >= proposal.support_count
        && static_cast<std::uint64_t>(proposal.support_count) * 3U
            >= static_cast<std::uint64_t>(proposal.total_count) * 2U;
}

}  // namespace local_gguf_engine::engine
