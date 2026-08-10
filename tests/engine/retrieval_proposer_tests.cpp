#include "retrieval_proposer_tests.hpp"
#include "test_support.hpp"

#include "local_gguf_engine/engine/retrieval_proposer.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using local_gguf_engine::engine::RetrievalProposer;
using local_gguf_engine::engine::retrieval_proposal_is_profitable;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

void test_retrieval_proposer_finds_longest_frequent_continuation() {
    const std::array<std::uint32_t, 14> history {
        1, 2, 7, 1, 2, 7, 1, 2, 8, 9, 1, 2, 7, 1,
    };
    RetrievalProposer proposer;
    proposer.rebuild(history);
    proposer.observe(2U);
    const auto proposal = proposer.propose();
    expect(proposal.token_id == 7U, "retrieval proposer continuation mismatch");
    expect(proposal.match_length >= 2U, "retrieval proposer suffix order mismatch");
    expect(proposal.support_count == 3U && proposal.total_count == 4U,
        "retrieval proposer continuation counts mismatch");
    expect(proposal.confidence == 0.75, "retrieval proposer confidence mismatch");
    expect(retrieval_proposal_is_profitable(proposal), "supported proposal should pass gate");
}

void test_retrieval_proposer_rebuild_matches_incremental_and_stable_ties() {
    const std::array<std::uint32_t, 9> prefix {4, 5, 8, 4, 5, 7, 4, 5, 8};
    RetrievalProposer incremental;
    incremental.rebuild(std::span<const std::uint32_t>(prefix).first(7U));
    incremental.observe(5U);
    incremental.observe(8U);
    RetrievalProposer rebuilt;
    rebuilt.rebuild(prefix);
    incremental.observe(4U);
    incremental.observe(5U);
    rebuilt.observe(4U);
    rebuilt.observe(5U);
    const auto first = incremental.propose();
    const auto second = rebuilt.propose();
    expect(first.token_id == second.token_id && first.match_length == second.match_length
            && first.support_count == second.support_count && first.total_count == second.total_count,
        "incremental and rebuilt proposer state mismatch");
    expect(first.token_id == 8U, "equal continuation counts must prefer lower token id");
}

void test_retrieval_proposer_bounds_and_no_match() {
    RetrievalProposer proposer;
    const std::array<std::uint32_t, 2> short_history {1, 2};
    proposer.rebuild(short_history);
    expect(proposer.propose().match_length == 0U, "short history should not propose");
    proposer.observe(3U);
    expect(proposer.propose().match_length == 0U, "unique suffix should not propose");

    std::vector<std::uint32_t> too_long(8193U, 0U);
    try {
        proposer.rebuild(too_long);
        throw local_gguf_engine::tests::TestFailure("oversized proposer history should fail");
    } catch (const std::invalid_argument&) {
    }
}

void test_retrieval_proposer_profitability_gate_uses_two_thirds_evidence() {
    expect(!retrieval_proposal_is_profitable({
            .token_id = 1U, .match_length = 2U, .support_count = 3U,
            .total_count = 5U, .confidence = 0.6,
        }),
        "proposal below two-thirds confidence should fail gate");
    expect(retrieval_proposal_is_profitable({
            .token_id = 1U, .match_length = 2U, .support_count = 2U,
            .total_count = 3U, .confidence = 2.0 / 3.0,
        }),
        "proposal at two-thirds confidence should pass gate");
}

const TestCase kTests[] = {
    {"retrieval proposer finds longest frequent continuation",
        test_retrieval_proposer_finds_longest_frequent_continuation},
    {"retrieval proposer rebuild matches incremental and stable ties",
        test_retrieval_proposer_rebuild_matches_incremental_and_stable_ties},
    {"retrieval proposer bounds and no match", test_retrieval_proposer_bounds_and_no_match},
    {"retrieval proposer profitability gate uses two-thirds evidence",
        test_retrieval_proposer_profitability_gate_uses_two_thirds_evidence},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> retrieval_proposer_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
