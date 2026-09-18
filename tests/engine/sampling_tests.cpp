#include "sampling_tests.hpp"
#include "test_support.hpp"

#include "local_gguf_engine/engine/sampling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using local_gguf_engine::engine::TokenLogit;
using local_gguf_engine::engine::TokenSampler;
using local_gguf_engine::engine::TokenSamplerOptions;
using local_gguf_engine::engine::TokenSelectionRequest;
using local_gguf_engine::engine::SamplerHistorySnapshot;
using local_gguf_engine::engine::external_speculation_max_horizon;
using local_gguf_engine::engine::prepare_token_distribution;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

[[nodiscard]] bool better(const TokenLogit& left, const TokenLogit& right) {
    return left.logit > right.logit || (left.logit == right.logit && left.token < right.token);
}

void test_compact_sampler_stable_ties_and_repetition() {
    const std::array<TokenLogit, 4> candidates {{{2, 3.0F}, {1, 3.0F}, {0, 2.9F}, {3, -1.0F}}};
    TokenSampler tied(TokenSamplerOptions {.temperature = 0.0F, .top_k = 4});
    expect(tied.sample(candidates) == 1U, "equal logits must prefer lower token id");

    TokenSampler penalized(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 4, .repetition_penalty = 2.0F, .repetition_window = 2,
    });
    const std::array<std::uint32_t, 3> history {2U, 2U, 1U};
    penalized.observe_tokens(history);
    const auto request = penalized.selection_request();
    expect(request.history_token_count == 2U, "selection request must deduplicate history");
    expect(request.base_candidate_count == 3U, "greedy superset must be K plus unique history");
    expect(penalized.sample(candidates) == 0U, "positive repeated logits must be divided");

    const std::array<TokenLogit, 3> negative {{{0, -1.0F}, {1, -1.1F}, {2, -2.0F}}};
    TokenSampler negative_penalty(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 3, .repetition_penalty = 2.0F, .repetition_window = 1,
    });
    const std::array<std::uint32_t, 1> repeated {0U};
    negative_penalty.observe_tokens(repeated);
    expect(negative_penalty.sample(negative) == 1U, "negative repeated logits must be multiplied");
}

void test_compact_sampler_seed_snapshot_and_filter_order() {
    const std::array<TokenLogit, 4> candidates {{{0, 0.0F}, {1, -1.0F}, {2, -4.0F}, {3, -8.0F}}};
    const auto options = TokenSamplerOptions {
        .temperature = 2.0F, .top_k = 4, .top_p = 0.7F, .min_p = 0.05F,
        .repetition_penalty = 1.1F, .seed = 123,
    };
    TokenSampler first(options);
    const std::array<std::uint32_t, 2> history {3U, 3U};
    first.observe_tokens(history);
    auto second = first;
    for (std::size_t index = 0; index < 32U; ++index) {
        const auto a = first.sample(candidates);
        const auto b = second.sample(candidates);
        expect(a == b, "copied sampler must replay RNG and history");
        expect(a == 0U, "top-p must filter before temperature and min-p");
    }
    first.reset_history();
    expect(first.selection_request().history_token_count == 0U, "reset must clear sampler history");
}

void test_repetition_window_eviction_and_promotion() {
    const std::array<TokenLogit, 3> candidates {{{0, 1.0F}, {1, 0.75F}, {2, 0.5F}}};
    TokenSampler sampler(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 3, .repetition_penalty = 2.0F, .repetition_window = 2,
    });
    const std::array<std::uint32_t, 3> old {0U, 2U, 2U};
    sampler.observe_tokens(old);
    expect(sampler.sample(candidates) == 0U, "evicted token must not be penalized");
    const std::array<std::uint32_t, 1> recent {0U};
    sampler.observe_tokens(recent);
    expect(sampler.sample(candidates) == 1U, "recent token must be penalized");

    TokenSampler promoting(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 3, .repetition_penalty = 0.5F, .repetition_window = 1,
    });
    const std::array<std::uint32_t, 1> promote {1U};
    promoting.observe_tokens(promote);
    expect(promoting.sample(candidates) == 1U, "penalty below one must promote a positive history logit");
}

void test_selection_request_after_observing_matches_sampler_copy() {
    TokenSampler sampler(TokenSamplerOptions {
        .temperature = 0.8F, .top_k = 8U, .top_p = 1.0F, .min_p = 0.0F,
        .repetition_penalty = 1.1F, .seed = 42U, .repetition_window = 3U,
    });
    const std::array<std::uint32_t, 3> history {7U, 8U, 9U};
    sampler.observe_tokens(history);
    auto copied = sampler;
    const auto next = std::uint32_t {8U};
    copied.observe_tokens(std::span<const std::uint32_t>(&next, 1U));
    const auto direct = sampler.selection_request_after_observing(next);
    const auto reference = copied.selection_request();
    expect(direct.top_k == reference.top_k
            && direct.base_candidate_count == reference.base_candidate_count
            && direct.history_token_count == reference.history_token_count
            && direct.history_token_ids == reference.history_token_ids,
        "selection request after observing differs from sampler copy");
}

void test_sparse_change_candidate_superset_property() {
    std::mt19937 rng(77U);
    std::uniform_real_distribution<float> values(-5.0F, 5.0F);
    for (const std::size_t k : {1U, 40U, 256U}) {
        const auto vocab = std::max<std::size_t>(320U, k + 64U);
        for (const std::size_t history_count : {0U, 1U, 64U}) {
            for (const float penalty : {0.5F, 1.0F, 1.7F}) {
                std::vector<TokenLogit> base(vocab);
                for (std::size_t token = 0; token < vocab; ++token) {
                    base[token] = TokenLogit {
                        .token = static_cast<std::uint32_t>(token),
                        .logit = token % 17U == 0U ? 1.0F : values(rng),
                    };
                }
                std::vector<std::uint32_t> history;
                for (std::size_t index = 0; index < history_count; ++index) {
                    history.push_back(static_cast<std::uint32_t>((index * 37U + vocab / 2U) % vocab));
                }
                auto base_sorted = base;
                std::sort(base_sorted.begin(), base_sorted.end(), better);
                std::vector<std::uint32_t> union_tokens;
                for (std::size_t index = 0; index < std::min(vocab, k + history_count); ++index) {
                    union_tokens.push_back(base_sorted[index].token);
                }
                union_tokens.insert(union_tokens.end(), history.begin(), history.end());

                auto adjusted = base;
                for (auto& candidate : adjusted) {
                    if (std::find(history.begin(), history.end(), candidate.token) != history.end()) {
                        candidate.logit = candidate.logit < 0.0F
                            ? candidate.logit * penalty : candidate.logit / penalty;
                    }
                }
                std::sort(adjusted.begin(), adjusted.end(), better);
                for (std::size_t index = 0; index < k; ++index) {
                    expect(
                        std::find(union_tokens.begin(), union_tokens.end(), adjusted[index].token)
                            != union_tokens.end(),
                        "sparse-change top-K escaped the K+U union"
                    );
                }
            }
        }
    }
}

void test_rejects_invalid_compact_sampler_inputs() {
    for (const auto options : {
        TokenSamplerOptions {.top_k = 0},
        TokenSamplerOptions {.top_k = 257},
        TokenSamplerOptions {.repetition_window = 65},
        TokenSamplerOptions {.temperature = -1.0F},
    }) {
        bool rejected = false;
        try { static_cast<void>(TokenSampler(options)); } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "invalid sampler product bounds must fail");
    }
    TokenSampler sampler;
    const std::array<TokenLogit, 1> invalid {{{0, std::numeric_limits<float>::quiet_NaN()}}};
    bool rejected = false;
    try { static_cast<void>(sampler.sample(invalid)); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "non-finite compact candidate must fail");
}

void test_prepared_distribution_freezes_filter_order_and_probabilities() {
    const std::array<TokenLogit, 5> candidates {{
        {3U, 0.0F}, {1U, 2.0F}, {2U, 1.0F}, {1U, 2.0F}, {4U, -4.0F},
    }};
    const auto options = TokenSamplerOptions {
        .temperature = 1.0F,
        .top_k = 4U,
        .top_p = 1.0F,
        .min_p = 0.0F,
        .repetition_penalty = 2.0F,
        .seed = 9U,
        .repetition_window = 64U,
    };
    TokenSelectionRequest request {.top_k = 4U, .base_candidate_count = 5U};
    request.history_token_ids[0] = 1U;
    request.history_token_count = 1U;
    const auto prepared = prepare_token_distribution(candidates, options, request);
    expect(prepared.count == 4U, "prepared distribution candidate count mismatch");
    expect(prepared.values[0].token == 1U && prepared.values[1].token == 2U
            && prepared.values[2].token == 3U && prepared.values[3].token == 4U,
        "prepared distribution token order mismatch");
    const std::array<double, 4> raw {std::exp(0.0), std::exp(0.0), std::exp(-1.0), std::exp(-5.0)};
    const auto normalization = std::accumulate(raw.begin(), raw.end(), 0.0);
    double probability_sum = 0.0;
    for (std::size_t index = 0; index < prepared.count; ++index) {
        expect(
            std::abs(prepared.values[index].probability - raw[index] / normalization) < 1.0e-12,
            "prepared distribution probability mismatch"
        );
        probability_sum += prepared.values[index].probability;
    }
    expect(std::abs(probability_sum - 1.0) < 1.0e-12,
        "prepared distribution probabilities must normalize to one");

    const auto greedy = prepare_token_distribution(
        candidates,
        TokenSamplerOptions {.temperature = 0.0F, .top_k = 4U, .repetition_penalty = 2.0F},
        TokenSelectionRequest {
            .top_k = 1U,
            .base_candidate_count = 2U,
            .history_token_ids = {1U},
            .history_token_count = 1U,
        }
    );
    expect(greedy.count == 1U && greedy.values[0].token == 1U
            && greedy.values[0].probability == 1.0,
        "prepared greedy distribution mismatch");
}

void test_raw_history_suffixes_append_truncate_then_deduplicate() {
    const auto options = TokenSamplerOptions {
        .temperature = 0.8F,
        .top_k = 8U,
        .top_p = 1.0F,
        .min_p = 0.0F,
        .repetition_penalty = 1.1F,
        .seed = 17U,
        .repetition_window = 64U,
    };
    for (const std::size_t base_length : {0U, 1U, 63U, 64U, 65U}) {
        std::vector<std::uint32_t> base;
        for (std::size_t index = 0; index < base_length; ++index) {
            base.push_back(static_cast<std::uint32_t>((index * 7U) % 19U));
        }
        TokenSampler sampler(options);
        sampler.observe_tokens(base);

        for (std::size_t suffix_length = 0; suffix_length <= external_speculation_max_horizon;
             ++suffix_length) {
            std::array<std::uint32_t, external_speculation_max_horizon> suffix {};
            for (std::size_t index = 0; index < suffix_length; ++index) {
                suffix[index] = static_cast<std::uint32_t>((index * 5U + base_length) % 23U);
            }
            const auto suffix_span = std::span<const std::uint32_t>(suffix).first(suffix_length);
            const auto device = sampler.device_selection_request(suffix_span);
            const auto actual = sampler.selection_request(device);
            const auto direct = sampler.selection_request_after_observing(suffix_span);

            auto chronological = base;
            chronological.insert(chronological.end(), suffix_span.begin(), suffix_span.end());
            if (chronological.size() > options.repetition_window) {
                chronological.erase(
                    chronological.begin(),
                    chronological.end() - static_cast<std::ptrdiff_t>(options.repetition_window)
                );
            }
            std::vector<std::uint32_t> unique;
            for (const auto token : chronological) {
                if (std::find(unique.begin(), unique.end(), token) == unique.end()) {
                    unique.push_back(token);
                }
            }
            expect(actual.history_token_count == unique.size(),
                "hypothetical history unique count mismatch");
            expect(actual.base_candidate_count == actual.top_k + unique.size(),
                "hypothetical history base-candidate count mismatch");
            for (std::size_t index = 0; index < unique.size(); ++index) {
                expect(actual.history_token_ids[index] == unique[index],
                    "hypothetical history did not append/truncate before deduplication");
            }
            expect(actual.top_k == direct.top_k
                    && actual.base_candidate_count == direct.base_candidate_count
                    && actual.history_token_count == direct.history_token_count
                    && actual.history_token_ids == direct.history_token_ids,
                "device and bounded-span history requests diverged");
        }
    }

    TokenSampler zero_window(TokenSamplerOptions {
        .temperature = 0.8F, .top_k = 4U, .repetition_penalty = 1.2F,
        .seed = 1U, .repetition_window = 0U,
    });
    const std::array<std::uint32_t, 3> ignored {1U, 2U, 1U};
    zero_window.observe_tokens(ignored);
    expect(zero_window.history_snapshot().length == 0U
            && zero_window.selection_request_after_observing(ignored).history_token_count == 0U,
        "zero repetition window retained hypothetical history");
}

void test_sampler_tickets_reproduce_authoritative_stream_exactly() {
    const std::array<TokenLogit, 6> candidates {{
        {0U, 2.0F}, {1U, 1.25F}, {2U, 0.5F},
        {3U, -0.25F}, {4U, -1.0F}, {5U, -2.0F},
    }};
    for (const auto seed : {0U, 1U, 42U, 0xffff'ffffU}) {
        const auto options = TokenSamplerOptions {
            .temperature = 0.73F,
            .top_k = 6U,
            .top_p = 1.0F,
            .min_p = 0.0F,
            .repetition_penalty = 1.07F,
            .seed = seed,
            .repetition_window = 64U,
        };
        TokenSampler authority(options);
        const std::array<std::uint32_t, 4> initial_history {5U, 1U, 5U, 3U};
        authority.observe_tokens(initial_history);
        const auto anchor = authority.checkpoint();
        const auto tickets = authority.preview_stream_tickets();
        expect(authority.checkpoint().same_state(anchor),
            "ticket preview advanced sampler authority");

        auto baseline = authority;
        auto fork = authority.fork(anchor);
        for (std::size_t offset = 0; offset < tickets.count; ++offset) {
            auto varied = candidates;
            varied[(offset + 1U) % varied.size()].logit += static_cast<float>(offset % 5U) * 0.11F;
            const auto expected = baseline.sample(varied);
            const auto actual = fork.sample_with_ticket(tickets.at(offset), varied);
            expect(actual == expected, "ticketed sample changed the target-selected token");
            baseline.observe_tokens(std::span<const std::uint32_t>(&expected, 1U));
            fork.observe_tokens(std::span<const std::uint32_t>(&actual, 1U));
            expect(fork.checkpoint().same_state(baseline.checkpoint()),
                "ticketed fork checkpoint differs at an accepted depth");
        }
        authority.adopt_checkpoint(fork.checkpoint());
        expect(authority.checkpoint().same_state(baseline.checkpoint()),
            "adopted ticket checkpoint differs from direct target sampler");
    }

    constexpr std::uint64_t seed = 1234567U;
    TokenSampler sampler(TokenSamplerOptions {
        .temperature = 1.0F, .top_k = 4U, .top_p = 1.0F, .min_p = 0.0F,
        .seed = seed,
    });
    const auto ticket = sampler.preview_tickets(1U).at(0U);
    std::mt19937_64 raw_reference(seed);
    expect(ticket.raw_engine_output_count() == 1U
            && ticket.raw_engine_output(0U) == raw_reference(),
        "ticket did not preserve the exact mt19937_64 engine output");
    for (const double total : {0.125, 1.0, 3.141592653589793, 1.0e12}) {
        std::mt19937_64 reference(seed);
        std::uniform_real_distribution<double> distribution(0.0, total);
        const auto expected = distribution(reference);
        expect(std::bit_cast<std::uint64_t>(ticket.mapped_draw(total))
                == std::bit_cast<std::uint64_t>(expected),
            "ticket total-mass mapping changed an exact binary draw");
    }
}

void test_sampler_ticket_and_device_request_fail_closed() {
    const auto options = TokenSamplerOptions {
        .temperature = 0.9F, .top_k = 4U, .top_p = 1.0F, .min_p = 0.0F,
        .seed = 8U,
    };
    TokenSampler sampler(options);
    const std::array<TokenLogit, 4> candidates {{{0U, 1.0F}, {1U, 0.5F}, {2U, 0.0F}, {3U, -0.5F}}};
    const auto anchor = sampler.checkpoint();
    const auto tickets = sampler.preview_tickets(2U);

    auto fork = sampler.fork(anchor);
    bool rejected = false;
    try { static_cast<void>(fork.sample_with_ticket(tickets.at(1U), candidates)); }
    catch (const std::logic_error&) { rejected = true; }
    expect(rejected, "out-of-order sampler ticket was accepted");

    const auto selected = fork.sample_with_ticket(tickets.at(0U), candidates);
    fork.observe_tokens(std::span<const std::uint32_t>(&selected, 1U));
    rejected = false;
    try { static_cast<void>(fork.sample_with_ticket(tickets.at(0U), candidates)); }
    catch (const std::logic_error&) { rejected = true; }
    expect(rejected, "stale sampler ticket was accepted twice");

    TokenSampler other(options);
    rejected = false;
    try { static_cast<void>(other.fork(anchor)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "foreign sampler checkpoint was accepted");
    auto other_fork = other.fork();
    rejected = false;
    try { static_cast<void>(other_fork.sample_with_ticket(tickets.at(0U), candidates)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "foreign sampler ticket was accepted");

    rejected = false;
    try { static_cast<void>(sampler.preview_tickets(48U)); }
    catch (const std::length_error&) { rejected = true; }
    expect(rejected, "ticket pool overflow was accepted");
    rejected = false;
    try { static_cast<void>(tickets.at(2U)); }
    catch (const std::out_of_range&) { rejected = true; }
    expect(rejected, "ticket pool out-of-range access was accepted");

    std::array<std::uint32_t, external_speculation_max_horizon + 1U> oversized {};
    rejected = false;
    try { static_cast<void>(sampler.device_selection_request(oversized)); }
    catch (const std::length_error&) { rejected = true; }
    expect(rejected, "device history suffix overflow was accepted");

    auto stale_device = sampler.device_selection_request();
    const std::uint32_t next = 77U;
    sampler.observe_tokens(std::span<const std::uint32_t>(&next, 1U));
    rejected = false;
    try { static_cast<void>(sampler.selection_request(stale_device)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "stale device selection anchor was accepted");

    SamplerHistorySnapshot malformed {.head = 64U, .length = 1U, .window_limit = 64U};
    rejected = false;
    try { static_cast<void>(local_gguf_engine::engine::make_token_selection_request(options, malformed)); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "malformed raw sampler history was accepted");
}

void test_greedy_tickets_never_consume_rng() {
    TokenSampler sampler(TokenSamplerOptions {
        .temperature = 0.0F, .top_k = 4U, .repetition_penalty = 1.2F,
        .seed = 99U,
    });
    const std::array<TokenLogit, 4> candidates {{{0U, 1.0F}, {1U, 0.9F}, {2U, 0.8F}, {3U, 0.7F}}};
    const auto anchor = sampler.checkpoint();
    const auto tickets = sampler.preview_stream_tickets();
    auto fork = sampler.fork(anchor);
    for (std::size_t offset = 0; offset < tickets.count; ++offset) {
        expect(!tickets.at(offset).consumes_random_draw()
                && tickets.at(offset).raw_engine_output_count() == 0U,
            "greedy ticket consumed random state");
        const auto token = fork.sample_with_ticket(tickets.at(offset), candidates);
        fork.observe_tokens(std::span<const std::uint32_t>(&token, 1U));
    }
    expect(fork.checkpoint().draw_count() == anchor.draw_count()
            && sampler.checkpoint().same_state(anchor),
        "greedy preview changed authoritative RNG state");
}

const TestCase kSamplingTests[] = {
    {"compact sampler stable ties and repetition", test_compact_sampler_stable_ties_and_repetition},
    {"compact sampler seed snapshot and filter order", test_compact_sampler_seed_snapshot_and_filter_order},
    {"repetition window eviction and promotion", test_repetition_window_eviction_and_promotion},
    {"selection request after observing matches sampler copy",
        test_selection_request_after_observing_matches_sampler_copy},
    {"sparse-change candidate superset property", test_sparse_change_candidate_superset_property},
    {"rejects invalid compact sampler inputs", test_rejects_invalid_compact_sampler_inputs},
    {"prepared distribution freezes filter order and probabilities",
        test_prepared_distribution_freezes_filter_order_and_probabilities},
    {"raw history suffixes append truncate then deduplicate",
        test_raw_history_suffixes_append_truncate_then_deduplicate},
    {"sampler tickets reproduce authoritative stream exactly",
        test_sampler_tickets_reproduce_authoritative_stream_exactly},
    {"sampler ticket and device request fail closed",
        test_sampler_ticket_and_device_request_fail_closed},
    {"greedy tickets never consume RNG", test_greedy_tickets_never_consume_rng},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> sampling_test_cases() { return kSamplingTests; }

}  // namespace local_gguf_engine::tests
