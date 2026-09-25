#include "local_gguf_engine/engine/sampling.hpp"
#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>

namespace {

using namespace local_gguf_engine::engine;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

template <typename Function>
void expect_rejected(Function function, const char* message) {
    bool rejected = false;
    try { function(); }
    catch (const std::exception&) { rejected = true; }
    expect(rejected, message);
}

PreparedTokenDistribution target_distribution(std::initializer_list<TokenLogit> probabilities) {
    PreparedTokenDistribution result;
    result.total_sampling_weight = 1.0;
    for (const auto value : probabilities) {
        result.values[result.count++] = {value.token, value.logit, value.logit};
    }
    return result;
}

DraftTokenDistribution draft_distribution(
    std::initializer_list<DraftTokenProbability> probabilities
) {
    DraftTokenDistribution result;
    for (const auto value : probabilities) result.values[result.count++] = value;
    return result;
}

void expect_near(double actual, double expected, const char* message) {
    expect(std::abs(actual - expected) <= 16.0 * std::numeric_limits<double>::epsilon(), message);
}

void check_exact_marginal(const PreparedTokenDistribution& target, const DraftTokenDistribution& draft) {
    std::array<double, 512> marginal {};
    for (std::size_t index = 0; index < draft.count; ++index) {
        const auto& proposal = draft.values[index];
        const auto prepared = prepare_speculative_distribution(proposal.token, draft, target);
        const auto q = static_cast<double>(proposal.probability_mass)
            / static_cast<double>(draft_probability_mass_total);
        marginal[proposal.token] += q * prepared.acceptance_probability;
        for (std::size_t residual = 0; residual < prepared.residual.count; ++residual) {
            const auto& value = prepared.residual.values[residual];
            marginal[value.token] += q * (1.0 - prepared.acceptance_probability) * value.probability;
        }
    }
    std::array<double, 512> expected {};
    for (std::size_t index = 0; index < target.count; ++index) {
        expected[target.values[index].token] = target.values[index].probability;
    }
    for (std::size_t token = 0; token < marginal.size(); ++token) {
        expect_near(marginal[token], expected[token], "analytic acceptance plus correction changes target marginal");
    }
}

void test_actual_draft_integer_law_boundaries() {
    constexpr auto half = draft_probability_mass_total / 2U;
    const auto q = draft_distribution({{7U, 1U}, {9U, half - 1U}, {4U, half}});
    expect(select_draft_token(q, 0U) == 7U, "first one-word mass was lost");
    expect(select_draft_token(q, 1U) == 9U, "CDF comparison is not strict");
    expect(select_draft_token(q, static_cast<std::uint32_t>(half - 1U)) == 9U,
        "last word of second bin was lost");
    expect(select_draft_token(q, static_cast<std::uint32_t>(half)) == 4U,
        "CDF boundary selected its predecessor");
    expect(select_draft_token(q, std::numeric_limits<std::uint32_t>::max()) == 4U,
        "largest proposal word was not covered");
    const auto delta = draft_distribution({{91U, draft_probability_mass_total}});
    expect(select_draft_token(delta, std::numeric_limits<std::uint32_t>::max()) == 91U,
        "2^32 total was truncated to uint32");
}

void test_lossless_exact_analytic_marginals() {
    constexpr auto unit = draft_probability_mass_total / 8U;
    check_exact_marginal(
        target_distribution({{0U, 0.25F}, {1U, 0.5F}, {2U, 0.25F}}),
        draft_distribution({{0U, 4U * unit}, {1U, unit}, {2U, 3U * unit}})
    );
    check_exact_marginal(
        target_distribution({{0U, 0.25F}, {1U, 0.5F}, {2U, 0.25F}}),
        draft_distribution({{3U, 4U * unit}, {1U, 4U * unit}})
    );
    check_exact_marginal(
        target_distribution({{0U, 0.5F}, {1U, 0.5F}}),
        draft_distribution({{2U, 4U * unit}, {3U, 4U * unit}})
    );
    check_exact_marginal(
        target_distribution({{0U, 1.0F}, {1U, 0.0F}}),
        draft_distribution({{1U, draft_probability_mass_total}})
    );
    check_exact_marginal(
        target_distribution({{0U, 0.25F}, {1U, 0.75F}}),
        draft_distribution({{0U, 2U * unit}, {1U, 6U * unit}})
    );
}

void test_lossless_injected_draws_enumerate_exact_marginal() {
    const auto p = target_distribution({{0U, 0.25F}, {1U, 0.5F}, {2U, 0.25F}});
    constexpr auto quarter = draft_probability_mass_total / 4U;
    const auto q = draft_distribution({{0U, 2U * quarter}, {1U, quarter}, {2U, quarter}});
    std::array<std::uint32_t, 3> counts {};
    // An independent product of equal-sized draft, acceptance and residual
    // intervals exactly integrates this dyadic example, without Monte Carlo.
    for (std::uint32_t proposal_bin = 0; proposal_bin < 8U; ++proposal_bin) {
        const auto word = static_cast<std::uint32_t>(
            (2U * static_cast<std::uint64_t>(proposal_bin) + 1U)
                * draft_probability_mass_total / 16U
        );
        const auto proposal = select_draft_token(q, word);
        const auto prepared = prepare_speculative_distribution(proposal, q, p);
        for (std::uint32_t accept_bin = 0; accept_bin < 8U; ++accept_bin) {
            for (std::uint32_t residual_bin = 0; residual_bin < 8U; ++residual_bin) {
                const auto selected = sample_speculative_token(
                    proposal, prepared, (static_cast<double>(accept_bin) + 0.5) / 8.0,
                    (static_cast<double>(residual_bin) + 0.5) / 8.0
                );
                ++counts[selected.token];
            }
        }
    }
    expect(counts == std::array<std::uint32_t, 3> {128U, 256U, 128U},
        "independent joint draws do not recover the exact target marginal");
    const auto boundary = prepare_speculative_distribution(0U, q, p);
    expect(sample_speculative_token(0U, boundary, std::nextafter(0.5, 0.0), 0.0).accepted,
        "draw immediately below acceptance boundary rejected");
    const auto rejected = sample_speculative_token(0U, boundary, 0.5, 0.0);
    expect(!rejected.accepted && rejected.token == 1U, "acceptance boundary must reject to residual");
}

void test_lossless_near_equal_and_full_capacity() {
    constexpr auto half = draft_probability_mass_total / 2U;
    const auto q = draft_distribution({{0U, half}, {1U, half}});
    auto p = target_distribution({{0U, 0.5F}, {1U, 0.5F}});
    const auto epsilon = 1.0 / static_cast<double>(draft_probability_mass_total);
    p.values[0].probability = p.values[0].sampling_weight = 0.5 + epsilon;
    p.values[1].probability = p.values[1].sampling_weight = 0.5 - epsilon;
    const auto prepared = prepare_speculative_distribution(1U, q, p);
    expect(prepared.residual.count == 1U && prepared.residual.total_sampling_weight == epsilon,
        "near-equal probabilities discarded a real residual");
    const auto selection = sample_speculative_token(1U, prepared, std::nextafter(1.0, 0.0), 0.75);
    expect(!selection.accepted && selection.token == 0U, "near-equal rejection did not use residual law");
    check_exact_marginal(p, q);

    PreparedTokenDistribution full_target;
    DraftTokenDistribution full_draft;
    full_target.count = full_draft.count = static_cast<std::uint32_t>(token_sampler_max_top_k);
    full_target.total_sampling_weight = 1.0;
    for (std::uint32_t index = 0; index < full_target.count; ++index) {
        full_target.values[index] = {index, 1.0 / 256.0, 1.0 / 256.0};
        full_draft.values[index] = {index, draft_probability_mass_total / 256U};
    }
    check_exact_marginal(full_target, full_draft);
    const auto identical = prepare_speculative_distribution(255U, full_draft, full_target);
    expect(identical.acceptance_probability == 1.0 && identical.residual.count == 0U,
        "identical complete supports must accept every proposal without a residual");
}

void test_lossless_malformed_probabilities_fail_closed() {
    const auto p = target_distribution({{0U, 0.5F}, {1U, 0.5F}});
    const auto q = draft_distribution({{0U, draft_probability_mass_total / 2U},
        {1U, draft_probability_mass_total / 2U}});
    for (std::size_t fault = 0; fault < 7U; ++fault) {
        auto broken = q;
        if (fault == 0U) broken.count = 0U;
        if (fault == 1U) broken.count = 257U;
        if (fault == 2U) broken.values[0].probability_mass = 0U;
        if (fault == 3U) --broken.values[0].probability_mass;
        if (fault == 4U) ++broken.values[0].probability_mass;
        if (fault == 5U) broken.values[0].probability_mass = std::numeric_limits<std::uint64_t>::max();
        if (fault == 6U) broken.values[1].token = 0U;
        expect_rejected([&] { static_cast<void>(prepare_speculative_distribution(0U, broken, p)); },
            "malformed integer proposal law was accepted");
    }
    for (std::size_t fault = 0; fault < 10U; ++fault) {
        auto broken = p;
        if (fault == 0U) broken.count = 0U;
        if (fault == 1U) broken.count = 257U;
        if (fault == 2U) broken.values[0].probability = std::numeric_limits<double>::quiet_NaN();
        if (fault == 3U) broken.values[0].sampling_weight = std::numeric_limits<double>::infinity();
        if (fault == 4U) broken.values[0].probability = -0.5;
        if (fault == 5U) broken.values[1].token = 0U;
        if (fault == 6U) broken.total_sampling_weight = 0.0;
        if (fault == 7U) broken.total_sampling_weight = 2.0;
        if (fault == 8U) broken.values[0].probability = 0.25;
        if (fault == 9U) broken.values[0].sampling_weight = -0.5;
        expect_rejected([&] { static_cast<void>(prepare_speculative_distribution(0U, q, broken)); },
            "malformed target probability law was accepted");
    }
    expect_rejected([&] { static_cast<void>(prepare_speculative_distribution(9U, q, p)); },
        "zero-probability proposal was accepted");
    const auto prepared = prepare_speculative_distribution(0U, q, p);
    for (const auto invalid : {-1.0, 1.0, std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity()}) {
        expect_rejected([&] { static_cast<void>(sample_speculative_token(0U, prepared, invalid, 0.0)); },
            "invalid acceptance draw was accepted");
        expect_rejected([&] { static_cast<void>(sample_speculative_token(0U, prepared, 0.0, invalid)); },
            "invalid residual draw was accepted");
    }
    expect_rejected([&] { static_cast<void>(sample_speculative_token(1U, prepared, 0.0, 0.0)); },
        "prepared acceptance was reused for a different proposal");
}

void test_lossless_retains_target_sampler_transforms() {
    const std::array<TokenLogit, 4> logits {{{0U, 1.0F}, {1U, 0.7F}, {2U, 0.3F}, {3U, -2.0F}}};
    const auto q = draft_distribution({{0U, draft_probability_mass_total / 4U},
        {1U, draft_probability_mass_total / 4U}, {2U, draft_probability_mass_total / 4U},
        {3U, draft_probability_mass_total / 4U}});
    for (const auto options : {
        TokenSamplerOptions {.temperature = 0.6F, .top_k = 4U, .top_p = 0.8F, .min_p = 0.05F,
            .repetition_penalty = 1.2F, .seed = 42U, .repetition_window = 2U},
        TokenSamplerOptions {.temperature = 1.3F, .top_k = 3U, .top_p = 1.0F, .min_p = 0.4F,
            .repetition_penalty = 0.7F, .seed = 42U, .repetition_window = 2U},
    }) {
        TokenSampler sampler(options);
        const std::array<std::uint32_t, 3> history {2U, 0U, 0U};
        sampler.observe_tokens(history);
        const auto p = prepare_token_distribution(logits, options, sampler.selection_request());
        check_exact_marginal(p, q);
        auto fork = sampler.fork();
        const auto proposal = select_draft_token(q, sampler.preview_proposal_words(1U).words[0]);
        const auto expected_distribution = prepare_speculative_distribution(proposal, q, p);
        const auto tickets = sampler.preview_tickets(2U);
        const auto expected = sample_speculative_token(proposal, expected_distribution,
            tickets.at(0U).unit_interval_draw(), tickets.at(1U).unit_interval_draw());
        const auto actual = fork.sample_speculative(proposal, q, logits);
        expect(actual.token == expected.token && actual.accepted == expected.accepted,
            "speculative fork altered target transforms or draw order");
    }
}

void observe(TokenSamplerFork& fork, std::uint32_t token) {
    fork.observe_tokens(std::span<const std::uint32_t>(&token, 1U));
}

void test_lossless_rng_checkpoint_accept_reject_bonus() {
    TokenSampler sampler(TokenSamplerOptions {.temperature = 0.8F, .top_k = 1U, .seed = 17U});
    const auto anchor = sampler.checkpoint();
    const auto q = draft_distribution({{7U, draft_probability_mass_total}});
    const std::array<TokenLogit, 1> agreeing {{{7U, 0.0F}}};
    auto fork = sampler.fork();
    const auto accepted = fork.sample_speculative(7U, q, agreeing);
    observe(fork, accepted.token);
    const auto first = fork.checkpoint();
    expect(accepted.accepted && accepted.token == 7U, "identical delta proposal was rejected");
    expect(first.draw_count() == 1U && first.proposal_draw_count() == 1U,
        "acceptance must consume one target draw and one independent proposal word");
    expect(first.follows_speculative(anchor, 7U, true)
            && !first.follows_speculative(anchor, 7U, false)
            && !first.follows_sample(anchor, 7U),
        "accepted speculative checkpoint was confused with another transition law");
    const std::array<TokenLogit, 1> disagreeing {{{9U, 0.0F}}};
    const auto rejected = fork.sample_speculative(7U, q, disagreeing);
    observe(fork, rejected.token);
    const auto second = fork.checkpoint();
    expect(!rejected.accepted && rejected.token == 9U, "disjoint delta proposal did not correct to target");
    expect(second.draw_count() == 3U && second.proposal_draw_count() == 2U,
        "rejection must consume acceptance plus residual draws and one proposal word");
    expect(second.follows_speculative(first, 9U, false)
            && !second.follows_speculative(first, 9U, true),
        "rejection checkpoint accepted the wrong target draw count");
    const auto bonus = fork.sample(agreeing);
    observe(fork, bonus);
    const auto third = fork.checkpoint();
    expect(third.follows_sample(second, bonus) && third.draw_count() == 4U
            && third.proposal_draw_count() == 2U,
        "bonus token did not retain the ordinary target-only sampling law");
    expect(sampler.checkpoint().same_state(anchor), "fork evaluation mutated authoritative sampler");
    sampler.adopt_checkpoint(third);
    expect(sampler.checkpoint().same_state(third), "checkpoint adoption omitted proposal state");
    sampler.adopt_checkpoint(anchor);
    auto regenerated = sampler.fork();
    const auto replay = regenerated.sample_speculative(7U, q, agreeing);
    observe(regenerated, replay.token);
    expect(regenerated.checkpoint().same_state(first), "regeneration failed to restore both streams");
}

void test_lossless_proposal_preview_and_failure_are_transactional() {
    TokenSampler sampler(TokenSamplerOptions {.temperature = 0.8F, .top_k = 2U, .seed = 99U});
    const auto anchor = sampler.checkpoint();
    const auto words = sampler.preview_proposal_words(external_speculation_max_horizon);
    expect(words.count == external_speculation_max_horizon && sampler.checkpoint().same_state(anchor),
        "proposal preview advanced authoritative state");
    const auto target_ticket = sampler.preview_ticket(0U);
    expect(words.words[0] != static_cast<std::uint32_t>(target_ticket.raw_engine_output(0U) >> 32U),
        "proposal stream accidentally reuses the acceptance random word");
    const auto q = draft_distribution({{0U, draft_probability_mass_total / 2U},
        {1U, draft_probability_mass_total / 2U}});
    const auto proposal = select_draft_token(q, words.words[0]);
    const std::array<TokenLogit, 2> logits {{{0U, 0.0F}, {1U, 0.0F}}};
    auto fork = sampler.fork();
    expect_rejected([&] { static_cast<void>(fork.sample_speculative(1U - proposal, q, logits)); },
        "proposal not drawn from checkpoint-owned word was accepted");
    expect(fork.checkpoint().same_state(anchor), "proposal provenance failure mutated RNG");
    auto invalid_logits = logits;
    invalid_logits[0].logit = std::numeric_limits<float>::quiet_NaN();
    expect_rejected([&] { static_cast<void>(fork.sample_speculative(proposal, q, invalid_logits)); },
        "nonfinite target distribution was accepted");
    expect(fork.checkpoint().same_state(anchor), "target validation failure mutated RNG");
    const auto selected = fork.sample_speculative(proposal, q, logits);
    observe(fork, selected.token);
    const auto remaining = fork.preview_proposal_words(external_speculation_max_horizon - 1U);
    for (std::size_t index = 0; index < remaining.count; ++index) {
        expect(remaining.words[index] == words.words[index + 1U], "unused speculative suffix consumed proposal RNG");
    }
    expect_rejected([&] { static_cast<void>(sampler.preview_proposal_words(24U)); },
        "proposal preview exceeded fixed horizon capacity");
    expect_rejected([&] { static_cast<void>(fork.preview_proposal_words(24U)); },
        "fork proposal preview exceeded fixed horizon capacity");
    expect(sampler.preview_proposal_words(0U).count == 0U, "empty proposal preview is not empty");
    TokenSampler unrelated(TokenSamplerOptions {.temperature = 0.8F, .top_k = 2U, .seed = 99U});
    expect_rejected([&] { unrelated.adopt_checkpoint(fork.checkpoint()); },
        "independent proposal authority lineage was adopted");
    expect(sampler.checkpoint().same_state(anchor), "abandoned speculative fork consumed future words");
}

void test_lossless_greedy_target_never_draws_acceptance() {
    TokenSampler sampler(TokenSamplerOptions {.temperature = 0.0F, .top_k = 2U, .seed = 42U});
    const std::array<TokenLogit, 2> logits {{{3U, 1.0F}, {7U, 0.0F}}};
    for (const auto proposal : {3U, 7U}) {
        const auto before = sampler.checkpoint();
        const auto q = draft_distribution({{proposal, draft_probability_mass_total}});
        auto fork = sampler.fork();
        const auto selected = fork.sample_speculative(proposal, q, logits);
        observe(fork, selected.token);
        expect(selected.token == 3U && selected.accepted == (proposal == 3U),
            "greedy target was changed by speculative proposal");
        expect(fork.checkpoint().draw_count() == 0U
                && fork.checkpoint().proposal_draw_count() == 1U
                && fork.checkpoint().follows_speculative(before, selected.token, selected.accepted),
            "greedy speculation consumed target RNG or failed transition audit");
    }
}

GatheredDraftTokenDistribution gathered_distribution(
    const DraftProbabilityLeaseOwner& owner,
    std::uint32_t proposal,
    const DraftTokenDistribution& full,
    const PreparedTokenDistribution& target
) {
    GatheredDraftTokenDistribution result;
    result.lease = owner.lease(0U);
    result.count = target.count;
    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index < full.count; ++index) {
        if (full.values[index].token == proposal) {
            result.proposal = full.values[index];
            result.proposal_cdf_begin = cumulative;
        }
        cumulative += full.values[index].probability_mass;
    }
    for (std::size_t index = 0; index < target.count; ++index) {
        auto& value = result.target_values[index];
        value.token = target.values[index].token;
        for (std::size_t draft_index = 0; draft_index < full.count; ++draft_index) {
            if (full.values[draft_index].token == value.token) {
                value.probability_mass = full.values[draft_index].probability_mass;
            }
        }
    }
    return result;
}

void test_gathered_full_vocabulary_exact_marginal() {
    constexpr std::uint64_t vocabulary = 32000U;
    const auto endpoint = [](std::uint64_t index) {
        return index * draft_probability_mass_total / vocabulary;
    };
    const auto mass = [&](std::uint32_t token) {
        return token < vocabulary ? endpoint(token + 1U) - endpoint(token) : 0U;
    };
    const auto p = target_distribution({{0U, 0.125F}, {999U, 0.375F}, {40000U, 0.5F}});
    DraftProbabilityLeaseOwner owner;
    owner.begin_epoch();
    std::array<double, 3> marginal {};
    std::array<double, 3> corrections {};
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
        GatheredDraftTokenDistribution q;
        q.lease = owner.lease(0U);
        q.proposal = {token, mass(token)};
        q.proposal_cdf_begin = endpoint(token);
        q.count = p.count;
        for (std::size_t index = 0; index < p.count; ++index) {
            q.target_values[index] = {p.values[index].token, mass(p.values[index].token)};
        }
        const auto prepared = prepare_speculative_distribution(token, q, p);
        const auto proposal_probability = static_cast<double>(mass(token))
            / static_cast<double>(draft_probability_mass_total);
        for (std::size_t index = 0; index < p.count; ++index) {
            double contribution = p.values[index].token == token
                ? proposal_probability * prepared.acceptance_probability : 0.0;
            for (std::size_t residual = 0; residual < prepared.residual.count; ++residual) {
                const auto& value = prepared.residual.values[residual];
                if (value.token == p.values[index].token) {
                    contribution += proposal_probability * (1.0 - prepared.acceptance_probability)
                        * value.probability;
                }
            }
            const auto sum = marginal[index] + contribution;
            corrections[index] += marginal[index] >= contribution
                ? (marginal[index] - sum) + contribution
                : (contribution - sum) + marginal[index];
            marginal[index] = sum;
        }
        if (token == 17U) {
            expect(prepared.acceptance_probability == 0.0,
                "proposal outside target support has nonzero acceptance");
            expect_near(prepared.residual.total_sampling_weight,
                1.0 - static_cast<double>(mass(0U) + mass(999U))
                    / static_cast<double>(draft_probability_mass_total),
                "full-vocabulary gathered subset was renormalized");
            const auto selected = sample_speculative_token(token, prepared, 0.0, 0.99);
            expect(!selected.accepted && selected.token == 40000U,
                "correction cannot select a target token outside the vocabulary intersection");
        }
    }
    for (std::size_t index = 0; index < p.count; ++index) {
        expect_near(marginal[index] + corrections[index], p.values[index].probability,
            "32,000-token proposal law did not recover target marginal from sparse q lookups");
    }
}

void test_gathered_lease_lifetime_and_rows() {
    DraftProbabilityLease retained;
    {
        DraftProbabilityLeaseOwner owner;
        expect_rejected([&] { static_cast<void>(owner.lease(0U)); },
            "uninitialized producer issued a valid probability lease");
        owner.begin_epoch();
        retained = owner.lease(22U);
        expect(retained.valid() && retained.row() == 22U, "last supported proposal row cannot lease");
        expect_rejected([&] { static_cast<void>(owner.lease(23U)); },
            "proposal lease row overflow was accepted");
        owner.begin_epoch();
        expect(!retained.valid(), "overwritten chain retained a valid lease");
        retained = owner.lease(0U);
        owner.invalidate();
        expect(!retained.valid(), "explicit cache rollback did not invalidate probability lease");
        owner.begin_epoch();
        expect(!retained.valid(), "new epoch resurrected a previously invalidated lease");
        retained = owner.lease(0U);
    }
    expect(!retained.valid(), "destroyed producer retained a live probability lease");
    expect(!DraftProbabilityLease {}.valid(), "default probability lease was valid");
}

void test_gathered_probability_failures_are_transactional() {
    TokenSampler sampler(TokenSamplerOptions {.temperature = 0.8F, .top_k = 2U, .seed = 19U});
    const std::array<TokenLogit, 2> logits {{{0U, 0.0F}, {1U, 0.0F}}};
    const auto full = draft_distribution({{0U, draft_probability_mass_total / 4U},
        {1U, draft_probability_mass_total / 4U}, {2U, draft_probability_mass_total / 2U}});
    const auto proposal = select_draft_token(full, sampler.preview_proposal_words(1U).words[0]);
    auto fork = sampler.fork();
    const auto before = fork.checkpoint();
    const auto p = fork.prepare_distribution(logits);
    DraftProbabilityLeaseOwner owner;
    owner.begin_epoch();
    const auto good = gathered_distribution(owner, proposal, full, p);
    for (std::size_t fault = 0; fault < 12U; ++fault) {
        auto bad = good;
        if (fault == 0U) bad.lease = {};
        if (fault == 1U) bad.count = 0U;
        if (fault == 2U) bad.count = 257U;
        if (fault == 3U) bad.proposal.token = 17U;
        if (fault == 4U) bad.proposal.probability_mass = 0U;
        if (fault == 5U) bad.proposal.probability_mass = std::numeric_limits<std::uint64_t>::max();
        if (fault == 6U) bad.proposal_cdf_begin = draft_probability_mass_total;
        if (fault == 7U) bad.target_values[0].token = 17U;
        if (fault == 8U) bad.target_values[1].token = bad.target_values[0].token;
        if (fault == 9U) bad.target_values[0].probability_mass = draft_probability_mass_total;
        if (fault == 10U) bad.target_values[0].probability_mass = std::numeric_limits<std::uint64_t>::max();
        if (fault == 11U) {
            const auto word = sampler.preview_proposal_words(1U).words[0];
            bad.proposal.probability_mass = 1U;
            bad.proposal_cdf_begin = word == 0U ? 1U : 0U;
            for (std::size_t index = 0; index < bad.count; ++index) {
                if (bad.target_values[index].token == proposal) {
                    bad.target_values[index].probability_mass = 1U;
                }
            }
        }
        expect_rejected([&] { static_cast<void>(fork.sample_speculative(proposal, bad, logits)); },
            "malformed gathered probability or CDF provenance was accepted");
        expect(fork.checkpoint().same_state(before), "gathered probability failure mutated random streams");
    }
    auto nonfinite = logits;
    nonfinite[0].logit = std::numeric_limits<float>::quiet_NaN();
    expect_rejected([&] { static_cast<void>(fork.sample_speculative(proposal, good, nonfinite)); },
        "gathered speculation accepted a nonfinite target logit");
    expect(fork.checkpoint().same_state(before), "nonfinite target mutated gathered sampler streams");
    owner.begin_epoch();
    expect_rejected([&] { static_cast<void>(fork.sample_speculative(proposal, good, logits)); },
        "stale gathered probabilities were accepted");
    expect(fork.checkpoint().same_state(before), "stale lease mutated sampler streams");
}

void test_gathered_sampling_matches_complete_law_and_rollback() {
    for (const auto temperature : {0.0F, 0.7F}) {
        TokenSampler sampler(TokenSamplerOptions {.temperature = temperature, .top_k = 3U,
            .top_p = 0.9F, .min_p = 0.05F, .repetition_penalty = 1.2F, .seed = 27U});
        const std::array<std::uint32_t, 2> history {1U, 1U};
        sampler.observe_tokens(history);
        const auto anchor = sampler.checkpoint();
        auto complete = sampler.fork();
        auto gathered = sampler.fork();
        DraftProbabilityLeaseOwner owner;
        owner.begin_epoch();
        const auto full = draft_distribution({{0U, draft_probability_mass_total / 8U},
            {1U, draft_probability_mass_total / 8U}, {2U, draft_probability_mass_total / 4U},
            {7U, draft_probability_mass_total / 2U}});
        const std::array<TokenLogit, 3> logits {{{0U, 0.0F}, {1U, 1.0F}, {2U, 0.8F}}};
        SamplerCheckpoint first;
        for (std::size_t step = 0; step < 23U; ++step) {
            const auto word = complete.preview_proposal_words(1U).words[0];
            const auto proposal = select_draft_token(full, word);
            const auto p = gathered.prepare_distribution(logits);
            const auto sparse = gathered_distribution(owner, proposal, full, p);
            const auto expected = complete.sample_speculative(proposal, full, logits);
            const auto actual = gathered.sample_speculative(proposal, sparse, logits);
            expect(actual.token == expected.token && actual.accepted == expected.accepted,
                "gathered lookup changed the complete-law rejection sampler");
            observe(complete, expected.token);
            observe(gathered, actual.token);
            expect(complete.checkpoint().same_state(gathered.checkpoint()),
                "gathered lookup changed proposal, acceptance, or residual RNG consumption");
            if (step == 0U) first = gathered.checkpoint();
        }
        sampler.adopt_checkpoint(gathered.checkpoint());
        sampler.adopt_checkpoint(anchor);
        auto replay = sampler.fork();
        const auto proposal = select_draft_token(full, replay.preview_proposal_words(1U).words[0]);
        const auto sparse = gathered_distribution(owner, proposal, full, replay.prepare_distribution(logits));
        const auto selected = replay.sample_speculative(proposal, sparse, logits);
        observe(replay, selected.token);
        expect(replay.checkpoint().same_state(first), "gathered-law regeneration failed to restore both streams");
    }
}

void test_gathered_maximum_target_support_and_unit_bins() {
    PreparedTokenDistribution target;
    target.count = 256U;
    target.total_sampling_weight = 1.0;
    DraftProbabilityLeaseOwner owner;
    owner.begin_epoch();
    GatheredDraftTokenDistribution draft;
    draft.lease = owner.lease(0U);
    draft.proposal = {0U, 1U};
    draft.count = 256U;
    for (std::uint32_t index = 0; index < target.count; ++index) {
        target.values[index] = {index, 1.0 / 256.0, 1.0 / 256.0};
        draft.target_values[index] = {index, 1U};
    }
    const auto prepared = prepare_speculative_distribution(0U, draft, target);
    expect(prepared.acceptance_probability == 1.0 && prepared.residual.count == 256U,
        "maximum target support or one-word q bins were narrowed");
    expect_near(prepared.residual.total_sampling_weight,
        1.0 - 256.0 / static_cast<double>(draft_probability_mass_total),
        "small gathered mass was normalized to the whole vocabulary");

    const auto delta_target = target_distribution({{0U, 1.0F}});
    draft.count = 1U;
    draft.proposal.probability_mass = draft_probability_mass_total;
    draft.target_values[0].probability_mass = draft_probability_mass_total;
    const auto delta = prepare_speculative_distribution(0U, draft, delta_target);
    expect(delta.acceptance_probability == 1.0 && delta.residual.count == 0U,
        "gathered 2^32-mass delta was truncated or required a residual");
}

const TestCase kLosslessSamplingTests[] = {
    {"actual draft integer law covers exact CDF boundaries", test_actual_draft_integer_law_boundaries},
    {"lossless acceptance and residual preserve exact analytic marginals", test_lossless_exact_analytic_marginals},
    {"lossless independent injected draws enumerate exact target marginal", test_lossless_injected_draws_enumerate_exact_marginal},
    {"lossless nearly equal and maximum capacity distributions", test_lossless_near_equal_and_full_capacity},
    {"lossless malformed probabilities fail closed", test_lossless_malformed_probabilities_fail_closed},
    {"lossless speculation retains all target sampling transforms", test_lossless_retains_target_sampler_transforms},
    {"lossless RNG checkpoint accepts rejects bonus and regeneration", test_lossless_rng_checkpoint_accept_reject_bonus},
    {"lossless proposal preview and failures are transactional", test_lossless_proposal_preview_and_failure_are_transactional},
    {"lossless greedy target never draws acceptance randomness", test_lossless_greedy_target_never_draws_acceptance},
    {"gathered full 32000-token proposal law preserves exact target marginal", test_gathered_full_vocabulary_exact_marginal},
    {"gathered probability leases reject stale destroyed and overflowing rows", test_gathered_lease_lifetime_and_rows},
    {"gathered malformed probabilities and CDF provenance fail transactionally", test_gathered_probability_failures_are_transactional},
    {"gathered proposal lookup matches complete law and RNG rollback", test_gathered_sampling_matches_complete_law_and_rollback},
    {"gathered maximum target support retains one-word and full delta masses", test_gathered_maximum_target_support_and_unit_bins},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> lossless_sampling_test_cases() { return kLosslessSamplingTests; }

}  // namespace local_gguf_engine::tests
