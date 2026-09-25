#include "local_gguf_engine/engine/sampling.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>

namespace local_gguf_engine::engine {

struct DraftProbabilityLeaseState {
    std::uint64_t epoch = 0;
    bool active = false;
};

namespace {

std::atomic<std::uint64_t> next_sampler_lineage {1U};

[[nodiscard]] bool better_logit_candidate(const TokenLogit& left, const TokenLogit& right) noexcept {
    if (left.logit != right.logit) {
        return left.logit > right.logit;
    }
    return left.token < right.token;
}

[[nodiscard]] std::mt19937_64 make_sampler_rng(std::uint64_t seed) {
    if (seed != TokenSamplerOptions::random_seed) {
        return std::mt19937_64(seed);
    }

    std::random_device entropy;
    std::seed_seq random_seed {
        entropy(), entropy(), entropy(), entropy(),
        entropy(), entropy(), entropy(), entropy(),
    };
    return std::mt19937_64(random_seed);
}

[[nodiscard]] std::mt19937_64 make_proposal_rng(std::mt19937_64 target_rng) {
    // Domain separation does not consume the pre-existing target stream. The
    // second state is checkpoint-owned so rejected lookahead never advances it.
    const auto first = target_rng();
    const auto second = target_rng();
    std::seed_seq seed {
        0x4c475350U, 0x524f5031U,
        static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(first >> 32U),
        static_cast<std::uint32_t>(second), static_cast<std::uint32_t>(second >> 32U),
    };
    return std::mt19937_64(seed);
}

[[nodiscard]] std::uint32_t next_proposal_word(std::mt19937_64& rng) {
    return static_cast<std::uint32_t>(rng() >> 32U);
}

[[nodiscard]] ProposalRandomWords preview_proposal_random_words(
    std::mt19937_64 rng,
    std::uint64_t draw_count,
    std::size_t count
) {
    if (count > external_speculation_max_horizon) {
        throw std::length_error("proposal random words exceed ExternalHmax=23");
    }
    if (count > std::numeric_limits<std::uint64_t>::max() - draw_count) {
        throw std::overflow_error("proposal random draw count exhausted");
    }
    ProposalRandomWords result;
    result.count = static_cast<std::uint32_t>(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.words[index] = next_proposal_word(rng);
    }
    return result;
}

void validate_draft_distribution(const DraftTokenDistribution& draft) {
    if (draft.count == 0U || draft.count > draft.values.size()) {
        throw std::invalid_argument("draft probability support exceeds fixed capacity or is empty");
    }
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < draft.count; ++index) {
        const auto& value = draft.values[index];
        if (value.probability_mass == 0U
            || value.probability_mass > draft_probability_mass_total - total) {
            throw std::invalid_argument("draft probability masses are invalid");
        }
        total += value.probability_mass;
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (draft.values[earlier].token == value.token) {
                throw std::invalid_argument("draft probability support contains duplicate tokens");
            }
        }
    }
    if (total != draft_probability_mass_total) {
        throw std::invalid_argument("draft probability masses must sum to exactly 2^32");
    }
}

[[nodiscard]] double draft_probability(
    const DraftTokenDistribution& draft,
    std::uint32_t token
) noexcept {
    for (std::size_t index = 0; index < draft.count; ++index) {
        if (draft.values[index].token == token) {
            return static_cast<double>(draft.values[index].probability_mass)
                / static_cast<double>(draft_probability_mass_total);
        }
    }
    return 0.0;
}

[[nodiscard]] double validate_prepared_distribution(const PreparedTokenDistribution& target) {
    if (target.count == 0U || target.count > target.values.size()
        || !std::isfinite(target.total_sampling_weight) || target.total_sampling_weight <= 0.0) {
        throw std::invalid_argument("prepared probability distribution is invalid");
    }
    // Neumaier accumulation controls normalization error without relying on
    // long double, which is only binary64 on the Apple ARM product machine.
    double total = 0.0;
    double correction = 0.0;
    for (std::size_t index = 0; index < target.count; ++index) {
        const auto& value = target.values[index];
        if (!std::isfinite(value.probability) || value.probability < 0.0
            || value.probability > 1.0 || !std::isfinite(value.sampling_weight)
            || value.sampling_weight < 0.0) {
            throw std::invalid_argument("prepared probabilities and weights must be finite and nonnegative");
        }
        const auto expected = value.sampling_weight / target.total_sampling_weight;
        const auto tolerance = 4.0 * std::numeric_limits<double>::epsilon()
            * std::max(expected, value.probability);
        if (std::abs(value.probability - expected) > tolerance) {
            throw std::invalid_argument("prepared probabilities disagree with sampling weights");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (target.values[earlier].token == value.token) {
                throw std::invalid_argument("prepared probability support contains duplicate tokens");
            }
        }
        const auto sum = total + value.sampling_weight;
        correction += total >= value.sampling_weight
            ? (total - sum) + value.sampling_weight
            : (value.sampling_weight - sum) + total;
        total = sum;
    }
    total += correction;
    const auto tolerance = static_cast<double>(target.count)
        * std::numeric_limits<double>::epsilon() * total;
    if (!std::isfinite(total) || total <= 0.0
        || std::abs(total - target.total_sampling_weight) > tolerance) {
        throw std::invalid_argument("prepared probability normalization disagrees with its support");
    }
    return total;
}

void validate_unit_draw(double draw) {
    if (!std::isfinite(draw) || draw < 0.0 || draw >= 1.0) {
        throw std::invalid_argument("speculative random draw must lie in [0, 1)");
    }
}

[[nodiscard]] std::uint64_t allocate_sampler_lineage() {
    const auto lineage = next_sampler_lineage.fetch_add(1U, std::memory_order_relaxed);
    if (lineage == 0U) {
        throw std::overflow_error("sampler lineage identity exhausted");
    }
    return lineage;
}

void validate_sampler_options(const TokenSamplerOptions& options) {
    if (!std::isfinite(options.temperature) || options.temperature < 0.0F) {
        throw std::invalid_argument("sampling temperature must be finite and nonnegative");
    }
    if (options.top_k == 0U || options.top_k > token_sampler_max_top_k) {
        throw std::invalid_argument("sampling top_k must be in [1, 256]");
    }
    if (!std::isfinite(options.top_p) || options.top_p <= 0.0F || options.top_p > 1.0F) {
        throw std::invalid_argument("sampling top_p must be in (0, 1]");
    }
    if (!std::isfinite(options.min_p) || options.min_p < 0.0F || options.min_p > 1.0F) {
        throw std::invalid_argument("sampling min_p must be in [0, 1]");
    }
    if (!std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("sampling repetition_penalty must be finite and positive");
    }
    if (options.repetition_window > token_sampler_max_repetition_window) {
        throw std::invalid_argument("sampling repetition_window must be at most 64");
    }
}

void validate_history_snapshot(const SamplerHistorySnapshot& history) {
    if (history.window_limit > token_sampler_max_repetition_window) {
        throw std::invalid_argument("sampler history window exceeds fixed capacity");
    }
    if (history.length > history.window_limit) {
        throw std::invalid_argument("sampler history length exceeds its window");
    }
    if (history.head >= token_sampler_max_repetition_window) {
        throw std::invalid_argument("sampler history head is outside its ring");
    }
    if (history.length != 0U && history.head >= history.window_limit) {
        throw std::invalid_argument("sampler history head is outside its configured window");
    }
    if (history.length == 0U && history.head != 0U) {
        throw std::invalid_argument("empty sampler history must have a zero head");
    }
}

[[nodiscard]] std::uint32_t chronological_history_token(
    const SamplerHistorySnapshot& history,
    std::size_t index
) noexcept {
    return history.token_ids[(static_cast<std::size_t>(history.head) + index)
        % history.window_limit];
}

[[nodiscard]] bool same_history(
    const SamplerHistorySnapshot& left,
    const SamplerHistorySnapshot& right
) noexcept {
    if (left.length != right.length || left.window_limit != right.window_limit) return false;
    for (std::size_t index = 0; index < left.length; ++index) {
        if (chronological_history_token(left, index)
            != chronological_history_token(right, index)) return false;
    }
    return true;
}

[[nodiscard]] DeviceSelectionRequest make_device_request(
    const SamplerHistorySnapshot& history,
    std::span<const std::uint32_t> suffix
) {
    validate_history_snapshot(history);
    if (suffix.size() > external_speculation_max_horizon) {
        throw std::length_error("hypothetical sampler suffix exceeds ExternalHmax=23");
    }
    DeviceSelectionRequest request {.history = history};
    std::copy(suffix.begin(), suffix.end(), request.suffix_token_ids.begin());
    request.suffix_count = static_cast<std::uint32_t>(suffix.size());
    return request;
}

[[nodiscard]] std::uint32_t select_prepared_token(
    const PreparedTokenDistribution& prepared,
    double draw
) noexcept {
    double cumulative = 0.0;
    for (std::size_t index = 0; index < prepared.count; ++index) {
        cumulative += prepared.values[index].sampling_weight;
        if (draw < cumulative) return prepared.values[index].token;
    }
    return prepared.values[prepared.count - 1U].token;
}

class AuditedSamplerEngine final {
public:
    using result_type = std::mt19937_64::result_type;

    AuditedSamplerEngine(
        const std::mt19937_64& rng,
        std::array<std::uint64_t, sampler_ticket_max_raw_engine_outputs>& outputs,
        std::uint32_t& count
    ) : rng_(rng), outputs_(outputs), count_(count) {}

    [[nodiscard]] static constexpr result_type min() noexcept { return std::mt19937_64::min(); }
    [[nodiscard]] static constexpr result_type max() noexcept { return std::mt19937_64::max(); }

    result_type operator()() {
        if (count_ >= outputs_.size()) {
            throw std::runtime_error("uniform_real_distribution exceeded sampler ticket audit capacity");
        }
        const auto value = rng_();
        outputs_[count_++] = value;
        return value;
    }

    [[nodiscard]] const std::mt19937_64& state() const noexcept { return rng_; }

private:
    std::mt19937_64 rng_;
    std::array<std::uint64_t, sampler_ticket_max_raw_engine_outputs>& outputs_;
    std::uint32_t& count_;
};

}  // namespace

bool DraftProbabilityLease::valid() const noexcept {
    return state_ && epoch_ != 0U && state_->active && epoch_ == state_->epoch;
}

std::uint32_t DraftProbabilityLease::row() const noexcept { return row_; }

DraftProbabilityLeaseOwner::DraftProbabilityLeaseOwner()
    : state_(std::make_shared<DraftProbabilityLeaseState>()) {}

DraftProbabilityLeaseOwner::~DraftProbabilityLeaseOwner() { invalidate(); }

void DraftProbabilityLeaseOwner::begin_epoch() {
    invalidate();
    if (state_->epoch == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("draft probability lease epoch exhausted");
    }
    ++state_->epoch;
    state_->active = true;
}

void DraftProbabilityLeaseOwner::invalidate() noexcept { state_->active = false; }

DraftProbabilityLease DraftProbabilityLeaseOwner::lease(std::uint32_t row) const {
    if (!state_->active) {
        throw std::logic_error("draft probability lease owner has no active proposal chain");
    }
    if (row >= external_speculation_max_horizon) {
        throw std::out_of_range("draft probability lease row exceeds ExternalHmax=23");
    }
    DraftProbabilityLease result;
    result.state_ = state_;
    result.epoch_ = state_->epoch;
    result.row_ = row;
    return result;
}

SamplerHistorySnapshot append_sampler_history(
    const SamplerHistorySnapshot& history,
    std::span<const std::uint32_t> suffix
) {
    validate_history_snapshot(history);
    auto appended = history;
    if (appended.window_limit == 0U) return appended;

    for (const auto token : suffix) {
        if (appended.length < appended.window_limit) {
            const auto slot = (static_cast<std::size_t>(appended.head) + appended.length)
                % appended.window_limit;
            appended.token_ids[slot] = token;
            ++appended.length;
        } else {
            appended.token_ids[appended.head] = token;
            appended.head = static_cast<std::uint32_t>(
                (static_cast<std::size_t>(appended.head) + 1U)
                    % appended.window_limit
            );
        }
    }
    return appended;
}

TokenSelectionRequest make_token_selection_request(
    const TokenSamplerOptions& options,
    const SamplerHistorySnapshot& history
) {
    validate_sampler_options(options);
    validate_history_snapshot(history);
    if (history.window_limit != options.repetition_window) {
        throw std::invalid_argument("sampler history window does not match sampler options");
    }

    TokenSelectionRequest request;
    request.top_k = static_cast<std::uint32_t>(
        options.temperature == 0.0F ? 1U : options.top_k
    );
    if (options.repetition_penalty != 1.0F) {
        for (std::size_t index = 0; index < history.length; ++index) {
            const auto token = chronological_history_token(history, index);
            const auto begin = request.history_token_ids.begin();
            const auto end = begin + request.history_token_count;
            if (std::find(begin, end, token) == end) {
                request.history_token_ids[request.history_token_count++] = token;
            }
        }
    }
    request.base_candidate_count = request.top_k + request.history_token_count;
    return request;
}

TokenSelectionRequest make_token_selection_request(
    const TokenSamplerOptions& options,
    const DeviceSelectionRequest& request
) {
    if (request.suffix_count > external_speculation_max_horizon) {
        throw std::invalid_argument("device sampler suffix exceeds ExternalHmax=23");
    }
    const auto history = append_sampler_history(
        request.history,
        std::span<const std::uint32_t>(request.suffix_token_ids).first(request.suffix_count)
    );
    return make_token_selection_request(options, history);
}

PreparedTokenDistribution prepare_token_distribution(
    std::span<const TokenLogit> compact_candidates,
    const TokenSamplerOptions& options,
    const TokenSelectionRequest& request
) {
    validate_sampler_options(options);
    if (compact_candidates.empty() || compact_candidates.size() > token_sampler_max_candidates) {
        throw std::invalid_argument("compact sampling candidates must contain between 1 and 384 entries");
    }
    if (request.top_k == 0U || request.top_k > options.top_k
        || request.top_k > token_sampler_max_top_k
        || request.history_token_count > token_sampler_max_repetition_window
        || request.base_candidate_count != request.top_k + request.history_token_count
        || request.base_candidate_count > token_sampler_max_base_candidates) {
        throw std::invalid_argument("sampling distribution request is invalid");
    }

    std::array<TokenLogit, token_sampler_max_candidates> candidates {};
    std::size_t candidate_count = 0;
    for (const auto candidate : compact_candidates) {
        if (!std::isfinite(candidate.logit)) {
            throw std::invalid_argument("compact sampling candidate logits must be finite");
        }
        auto existing = std::find_if(
            candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count),
            [&](const TokenLogit& value) { return value.token == candidate.token; }
        );
        if (existing == candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count)) {
            candidates[candidate_count++] = candidate;
        } else if (candidate.logit != existing->logit) {
            throw std::invalid_argument("duplicate compact sampling candidates have different logits");
        }
    }
    if (candidate_count < request.top_k) {
        throw std::invalid_argument("compact sampling candidate set is smaller than top_k");
    }
    const auto history_end = request.history_token_ids.begin() + request.history_token_count;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        auto& candidate = candidates[index];
        if (std::find(request.history_token_ids.begin(), history_end, candidate.token) != history_end) {
            candidate.logit = candidate.logit < 0.0F
                ? candidate.logit * options.repetition_penalty
                : candidate.logit / options.repetition_penalty;
        }
    }
    std::sort(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count),
        better_logit_candidate
    );
    candidate_count = std::min(candidate_count, static_cast<std::size_t>(request.top_k));

    PreparedTokenDistribution prepared;
    if (options.temperature == 0.0F) {
        prepared.values[0] = {
            .token = candidates.front().token,
            .probability = 1.0,
            .sampling_weight = 1.0,
        };
        prepared.count = 1U;
        prepared.total_sampling_weight = 1.0;
        return prepared;
    }

    std::array<double, token_sampler_max_top_k> weights {};
    const auto max_logit = candidates.front().logit;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        weights[index] = std::exp(static_cast<double>(candidates[index].logit - max_logit));
    }
    const auto total_weight = std::accumulate(weights.begin(), weights.begin() + candidate_count, 0.0);
    double retained_weight = 0.0;
    std::size_t retained_count = 0;
    do {
        retained_weight += weights[retained_count++];
    } while (retained_count < candidate_count
        && retained_weight / total_weight < static_cast<double>(options.top_p));
    candidate_count = retained_count;
    if (options.min_p > 0.0F) {
        std::size_t min_p_count = 1;
        while (min_p_count < candidate_count
            && weights[min_p_count] >= static_cast<double>(options.min_p)) {
            ++min_p_count;
        }
        candidate_count = min_p_count;
    }
    double normalization = 0.0;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        weights[index] = std::exp(
            static_cast<double>(candidates[index].logit - max_logit)
                / static_cast<double>(options.temperature)
        );
        normalization += weights[index];
    }
    if (!std::isfinite(normalization) || normalization <= 0.0) {
        throw std::runtime_error("sampling distribution normalization is invalid");
    }
    prepared.count = static_cast<std::uint32_t>(candidate_count);
    prepared.total_sampling_weight = normalization;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        prepared.values[index] = {
            .token = candidates[index].token,
            .probability = weights[index] / normalization,
            .sampling_weight = weights[index],
        };
    }
    return prepared;
}

std::uint32_t select_draft_token(
    const DraftTokenDistribution& distribution,
    std::uint32_t random_word
) {
    validate_draft_distribution(distribution);
    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index < distribution.count; ++index) {
        cumulative += distribution.values[index].probability_mass;
        if (static_cast<std::uint64_t>(random_word) < cumulative) {
            return distribution.values[index].token;
        }
    }
    throw std::logic_error("validated draft distribution did not contain its random word");
}

PreparedSpeculativeDistribution prepare_speculative_distribution(
    std::uint32_t proposal,
    const DraftTokenDistribution& draft,
    const PreparedTokenDistribution& target
) {
    validate_draft_distribution(draft);
    const auto normalization = validate_prepared_distribution(target);
    const auto proposed_mass = draft_probability(draft, proposal);
    if (proposed_mass == 0.0) {
        throw std::invalid_argument("proposed token has zero probability under the actual draft law");
    }

    // Chen et al., arXiv:2302.01318, Algorithm 2: a=min(1,p(d)/q(d)),
    // and the rejection distribution is normalized max(p-q,0). Here p is
    // the target's fully transformed law, q the actual device proposal law.
    // This finite-precision implementation normalizes the existing weights;
    // it never changes top-k/top-p/min-p/temperature/repetition transforms.
    PreparedSpeculativeDistribution result {.proposal_token = proposal};
    for (std::size_t index = 0; index < target.count; ++index) {
        const auto& value = target.values[index];
        const auto probability = value.sampling_weight / normalization;
        if (value.token == proposal) {
            result.acceptance_probability = std::min(1.0, probability / proposed_mass);
        }
        const auto difference = probability - draft_probability(draft, value.token);
        if (difference <= 0.0) continue;
        auto& residual = result.residual;
        residual.values[residual.count++] = {
            .token = value.token,
            .probability = 0.0,
            .sampling_weight = difference,
        };
        residual.total_sampling_weight += difference;
    }
    if (result.residual.count != 0U) {
        for (std::size_t index = 0; index < result.residual.count; ++index) {
            auto& value = result.residual.values[index];
            value.probability = value.sampling_weight / result.residual.total_sampling_weight;
        }
    } else if (result.acceptance_probability < 1.0) {
        throw std::runtime_error("speculative rejection has no representable residual probability mass");
    }
    return result;
}

PreparedSpeculativeDistribution prepare_speculative_distribution(
    std::uint32_t proposal,
    const GatheredDraftTokenDistribution& draft,
    const PreparedTokenDistribution& target
) {
    if (!draft.lease.valid()) {
        throw std::invalid_argument("gathered draft probabilities have an expired producer lease");
    }
    const auto normalization = validate_prepared_distribution(target);
    if (draft.proposal.token != proposal || draft.proposal.probability_mass == 0U
        || draft.proposal.probability_mass > draft_probability_mass_total
        || draft.proposal_cdf_begin > draft_probability_mass_total - draft.proposal.probability_mass
        || draft.count != target.count) {
        throw std::invalid_argument("gathered draft proposal or support is invalid");
    }
    std::uint64_t gathered_mass = 0;
    bool proposal_in_target = false;
    for (std::size_t index = 0; index < target.count; ++index) {
        const auto& value = draft.target_values[index];
        if (value.token != target.values[index].token
            || value.probability_mass > draft_probability_mass_total - gathered_mass) {
            throw std::invalid_argument("gathered draft masses do not cover the exact target support");
        }
        gathered_mass += value.probability_mass;
        if (value.token == proposal) {
            proposal_in_target = true;
            if (value.probability_mass != draft.proposal.probability_mass) {
                throw std::invalid_argument("gathered proposal mass disagrees with its selected bin");
            }
        }
    }
    if (!proposal_in_target
        && draft.proposal.probability_mass > draft_probability_mass_total - gathered_mass) {
        throw std::invalid_argument("gathered draft and proposal masses exceed the full proposal law");
    }

    // TLI (arXiv:2502.05202v3, Algorithm 4) changes only q's support.
    // The producer has already normalized over the complete vocabulary
    // intersection. Here we look up q on p's support without renormalizing it.
    const auto proposed_probability = static_cast<double>(draft.proposal.probability_mass)
        / static_cast<double>(draft_probability_mass_total);
    PreparedSpeculativeDistribution result {.proposal_token = proposal};
    for (std::size_t index = 0; index < target.count; ++index) {
        const auto& value = target.values[index];
        const auto probability = value.sampling_weight / normalization;
        if (value.token == proposal) {
            result.acceptance_probability = std::min(1.0, probability / proposed_probability);
        }
        const auto q = static_cast<double>(draft.target_values[index].probability_mass)
            / static_cast<double>(draft_probability_mass_total);
        const auto difference = probability - q;
        if (difference <= 0.0) continue;
        auto& residual = result.residual;
        residual.values[residual.count++] = {
            .token = value.token,
            .probability = 0.0,
            .sampling_weight = difference,
        };
        residual.total_sampling_weight += difference;
    }
    if (result.residual.count != 0U) {
        for (std::size_t index = 0; index < result.residual.count; ++index) {
            auto& value = result.residual.values[index];
            value.probability = value.sampling_weight / result.residual.total_sampling_weight;
        }
    } else if (result.acceptance_probability < 1.0) {
        throw std::runtime_error("speculative rejection has no representable residual probability mass");
    }
    return result;
}

SpeculativeTokenSelection sample_speculative_token(
    std::uint32_t proposal,
    const PreparedSpeculativeDistribution& distribution,
    double acceptance_draw,
    double residual_draw
) {
    validate_unit_draw(acceptance_draw);
    validate_unit_draw(residual_draw);
    if (proposal != distribution.proposal_token
        || !std::isfinite(distribution.acceptance_probability)
        || distribution.acceptance_probability < 0.0
        || distribution.acceptance_probability > 1.0) {
        throw std::invalid_argument("speculative acceptance distribution is invalid");
    }
    if (acceptance_draw < distribution.acceptance_probability) {
        return {.token = proposal, .accepted = true};
    }
    static_cast<void>(validate_prepared_distribution(distribution.residual));
    return {
        .token = select_prepared_token(
            distribution.residual, residual_draw * distribution.residual.total_sampling_weight
        ),
        .accepted = false,
    };
}

const SamplerHistorySnapshot& SamplerCheckpoint::history() const noexcept { return history_; }

SamplerCheckpoint::SamplerCheckpoint(
    TokenSamplerOptions options,
    std::mt19937_64 rng,
    std::mt19937_64 proposal_rng,
    SamplerHistorySnapshot history,
    std::uint64_t lineage_id,
    std::uint64_t draw_count,
    std::uint64_t proposal_draw_count
) : options_(options),
    rng_(std::move(rng)),
    proposal_rng_(std::move(proposal_rng)),
    history_(history),
    lineage_id_(lineage_id),
    draw_count_(draw_count),
    proposal_draw_count_(proposal_draw_count) {}

std::uint64_t SamplerCheckpoint::draw_count() const noexcept { return draw_count_; }
std::uint64_t SamplerCheckpoint::proposal_draw_count() const noexcept { return proposal_draw_count_; }

bool SamplerCheckpoint::same_state(const SamplerCheckpoint& other) const noexcept {
    return options_ == other.options_ && rng_ == other.rng_
        && proposal_rng_ == other.proposal_rng_
        && same_history(history_, other.history_) && draw_count_ == other.draw_count_
        && proposal_draw_count_ == other.proposal_draw_count_;
}

bool SamplerCheckpoint::follows_sample(
    const SamplerCheckpoint& previous,
    std::uint32_t observed_token
) const {
    if (lineage_id_ == 0U || lineage_id_ != previous.lineage_id_
        || options_ != previous.options_ || proposal_rng_ != previous.proposal_rng_
        || proposal_draw_count_ != previous.proposal_draw_count_) return false;
    if (options_.temperature != 0.0F
        && previous.draw_count_ == std::numeric_limits<std::uint64_t>::max()) return false;
    const auto expected_draw_count = previous.draw_count_
        + (options_.temperature == 0.0F ? 0U : 1U);
    if (draw_count_ != expected_draw_count) return false;
    auto expected_rng = previous.rng_;
    if (options_.temperature != 0.0F) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        static_cast<void>(distribution(expected_rng));
    }
    if (rng_ != expected_rng) return false;
    const auto expected_history = append_sampler_history(
        previous.history_, std::span<const std::uint32_t>(&observed_token, 1U)
    );
    return same_history(history_, expected_history);
}

bool SamplerCheckpoint::follows_speculative(
    const SamplerCheckpoint& previous,
    std::uint32_t observed_token,
    bool accepted
) const {
    if (lineage_id_ == 0U || lineage_id_ != previous.lineage_id_
        || options_ != previous.options_
        || previous.proposal_draw_count_ == std::numeric_limits<std::uint64_t>::max()
        || proposal_draw_count_ != previous.proposal_draw_count_ + 1U) return false;
    const auto target_draws = options_.temperature == 0.0F ? 0U : (accepted ? 1U : 2U);
    if (target_draws > std::numeric_limits<std::uint64_t>::max() - previous.draw_count_
        || draw_count_ != previous.draw_count_ + target_draws) return false;
    auto expected_rng = previous.rng_;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (std::uint32_t index = 0; index < target_draws; ++index) {
        static_cast<void>(distribution(expected_rng));
    }
    auto expected_proposal_rng = previous.proposal_rng_;
    static_cast<void>(next_proposal_word(expected_proposal_rng));
    const auto expected_history = append_sampler_history(
        previous.history_, std::span<const std::uint32_t>(&observed_token, 1U)
    );
    return rng_ == expected_rng && proposal_rng_ == expected_proposal_rng
        && same_history(history_, expected_history);
}

std::uint32_t SamplerTicket::offset() const noexcept { return offset_; }

std::uint64_t SamplerTicket::draw_index() const noexcept { return draw_index_; }

bool SamplerTicket::consumes_random_draw() const noexcept { return consumes_random_draw_; }

double SamplerTicket::unit_interval_draw() const noexcept { return unit_interval_draw_; }

std::uint32_t SamplerTicket::raw_engine_output_count() const noexcept {
    return raw_engine_output_count_;
}

std::uint64_t SamplerTicket::raw_engine_output(std::size_t index) const {
    if (index >= raw_engine_output_count_) {
        throw std::out_of_range("sampler ticket raw engine output index is out of range");
    }
    return raw_engine_outputs_[index];
}

double SamplerTicket::mapped_draw(double total_sampling_weight) const {
    if (!consumes_random_draw_) return 0.0;
    if (!std::isfinite(total_sampling_weight) || total_sampling_weight <= 0.0) {
        throw std::invalid_argument("sampler ticket total sampling weight must be finite and positive");
    }
    auto rng = pre_rng_;
    std::uniform_real_distribution<double> distribution(0.0, total_sampling_weight);
    const auto draw = distribution(rng);
    if (rng != post_rng_) {
        throw std::runtime_error("sampler ticket distribution consumed an unexpected RNG state");
    }
    return draw;
}

const SamplerTicket& SamplerTicketPool::at(std::size_t offset) const {
    if (offset >= count) throw std::out_of_range("sampler ticket offset is out of range");
    return tickets[offset];
}

TokenSamplerFork::TokenSamplerFork(const SamplerCheckpoint& checkpoint)
    : options_(checkpoint.options_),
      rng_(checkpoint.rng_),
      proposal_rng_(checkpoint.proposal_rng_),
      history_(checkpoint.history_),
      lineage_id_(checkpoint.lineage_id_),
      draw_count_(checkpoint.draw_count_),
      proposal_draw_count_(checkpoint.proposal_draw_count_) {}

bool TokenSamplerFork::is_greedy() const noexcept { return options_.temperature == 0.0F; }

const SamplerHistorySnapshot& TokenSamplerFork::history_snapshot() const noexcept {
    return history_;
}

TokenSelectionRequest TokenSamplerFork::selection_request() const noexcept {
    return make_token_selection_request(options_, history_);
}

DeviceSelectionRequest TokenSamplerFork::device_selection_request(
    std::span<const std::uint32_t> hypothetical_suffix
) const {
    return make_device_request(history_, hypothetical_suffix);
}

std::uint32_t TokenSamplerFork::sample_with_ticket(
    const SamplerTicket& ticket,
    std::span<const TokenLogit> compact_candidates
) {
    if (lineage_id_ == 0U || ticket.lineage_id_ != lineage_id_
        || ticket.options_ != options_) {
        throw std::invalid_argument("sampler ticket is incompatible with this fork");
    }
    if (ticket.draw_index_ != draw_count_ || ticket.pre_rng_ != rng_) {
        throw std::logic_error("sampler ticket is stale or mapped to another accepted depth");
    }
    if (ticket.consumes_random_draw_ == is_greedy()) {
        throw std::logic_error("sampler ticket draw mode does not match the sampler");
    }

    const auto prepared = prepare_token_distribution(
        compact_candidates, options_, selection_request()
    );
    if (is_greedy()) {
        if (ticket.post_rng_ != rng_) {
            throw std::logic_error("greedy sampler ticket unexpectedly advances RNG state");
        }
        return prepared.values[0].token;
    }

    std::uniform_real_distribution<double> distribution(0.0, prepared.total_sampling_weight);
    const auto draw = distribution(rng_);
    if (rng_ != ticket.post_rng_
        || std::bit_cast<std::uint64_t>(draw)
            != std::bit_cast<std::uint64_t>(ticket.mapped_draw(prepared.total_sampling_weight))) {
        throw std::runtime_error("sampler ticket does not reproduce the authoritative draw");
    }
    ++draw_count_;
    return select_prepared_token(prepared, draw);
}

std::uint32_t TokenSamplerFork::sample(std::span<const TokenLogit> compact_candidates) {
    const auto prepared = prepare_token_distribution(compact_candidates, options_, selection_request());
    if (is_greedy()) return prepared.values[0].token;
    if (draw_count_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("target random draw count exhausted");
    }
    std::uniform_real_distribution<double> distribution(0.0, prepared.total_sampling_weight);
    const auto draw = distribution(rng_);
    ++draw_count_;
    return select_prepared_token(prepared, draw);
}

PreparedTokenDistribution TokenSamplerFork::prepare_distribution(
    std::span<const TokenLogit> compact_candidates
) const {
    return prepare_token_distribution(compact_candidates, options_, selection_request());
}

SpeculativeTokenSelection TokenSamplerFork::sample_speculative(
    std::uint32_t proposal,
    const DraftTokenDistribution& draft,
    std::span<const TokenLogit> compact_target_candidates
) {
    if (proposal_draw_count_ == std::numeric_limits<std::uint64_t>::max()
        || (!is_greedy() && draw_count_ > std::numeric_limits<std::uint64_t>::max() - 2U)) {
        throw std::overflow_error("speculative random draw count exhausted");
    }
    auto proposal_cursor = proposal_rng_;
    const auto word = next_proposal_word(proposal_cursor);
    if (select_draft_token(draft, word) != proposal) {
        throw std::invalid_argument("proposal does not match its checkpoint-owned random word and law");
    }
    const auto target = prepare_token_distribution(
        compact_target_candidates, options_, selection_request()
    );
    const auto distribution = prepare_speculative_distribution(proposal, draft, target);
    auto target_cursor = rng_;
    std::uint64_t target_draws = 0;
    double acceptance_draw = 0.0;
    double residual_draw = 0.0;
    if (!is_greedy()) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        acceptance_draw = uniform(target_cursor);
        ++target_draws;
        if (acceptance_draw >= distribution.acceptance_probability) {
            residual_draw = uniform(target_cursor);
            ++target_draws;
        }
    }
    const auto selected = sample_speculative_token(
        proposal, distribution, acceptance_draw, residual_draw
    );
    // Commit both streams together only after all probability and provenance
    // checks succeed. Observing the returned token remains the caller's job.
    rng_ = target_cursor;
    proposal_rng_ = proposal_cursor;
    draw_count_ += target_draws;
    ++proposal_draw_count_;
    return selected;
}

SpeculativeTokenSelection TokenSamplerFork::sample_speculative(
    std::uint32_t proposal,
    const GatheredDraftTokenDistribution& draft,
    std::span<const TokenLogit> compact_target_candidates
) {
    if (proposal_draw_count_ == std::numeric_limits<std::uint64_t>::max()
        || (!is_greedy() && draw_count_ > std::numeric_limits<std::uint64_t>::max() - 2U)) {
        throw std::overflow_error("speculative random draw count exhausted");
    }
    const auto target = prepare_distribution(compact_target_candidates);
    const auto distribution = prepare_speculative_distribution(proposal, draft, target);
    auto proposal_cursor = proposal_rng_;
    const auto word = static_cast<std::uint64_t>(next_proposal_word(proposal_cursor));
    if (word < draft.proposal_cdf_begin
        || word - draft.proposal_cdf_begin >= draft.proposal.probability_mass) {
        throw std::invalid_argument("gathered proposal does not match its checkpoint-owned random word");
    }
    auto target_cursor = rng_;
    std::uint64_t target_draws = 0;
    double acceptance_draw = 0.0;
    double residual_draw = 0.0;
    if (!is_greedy()) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        acceptance_draw = uniform(target_cursor);
        ++target_draws;
        if (acceptance_draw >= distribution.acceptance_probability) {
            residual_draw = uniform(target_cursor);
            ++target_draws;
        }
    }
    const auto selected = sample_speculative_token(
        proposal, distribution, acceptance_draw, residual_draw
    );
    rng_ = target_cursor;
    proposal_rng_ = proposal_cursor;
    draw_count_ += target_draws;
    ++proposal_draw_count_;
    return selected;
}

ProposalRandomWords TokenSamplerFork::preview_proposal_words(std::size_t count) const {
    return preview_proposal_random_words(proposal_rng_, proposal_draw_count_, count);
}

void TokenSamplerFork::observe_tokens(std::span<const std::uint32_t> token_ids) {
    history_ = append_sampler_history(history_, token_ids);
}

SamplerCheckpoint TokenSamplerFork::checkpoint() const noexcept {
    return SamplerCheckpoint(
        options_, rng_, proposal_rng_, history_, lineage_id_, draw_count_, proposal_draw_count_
    );
}

TokenSampler::TokenSampler(TokenSamplerOptions options)
    : options_(options), rng_(make_sampler_rng(options.seed)),
      proposal_rng_(make_proposal_rng(rng_)), lineage_id_(allocate_sampler_lineage()) {
    validate_sampler_options(options_);
    history_.window_limit = static_cast<std::uint32_t>(options_.repetition_window);
}

bool TokenSampler::is_greedy() const noexcept { return options_.temperature == 0.0F; }
const TokenSamplerOptions& TokenSampler::options() const noexcept { return options_; }

const SamplerHistorySnapshot& TokenSampler::history_snapshot() const noexcept { return history_; }

TokenSelectionRequest TokenSampler::selection_request() const noexcept {
    return make_token_selection_request(options_, history_);
}

TokenSelectionRequest TokenSampler::selection_request_after_observing(
    std::uint32_t token_id
) const noexcept {
    return make_token_selection_request(
        options_, append_sampler_history(history_, std::span<const std::uint32_t>(&token_id, 1U))
    );
}

TokenSelectionRequest TokenSampler::selection_request_after_observing(
    std::span<const std::uint32_t> token_ids
) const {
    if (token_ids.size() > external_speculation_max_horizon) {
        throw std::length_error("hypothetical sampler suffix exceeds ExternalHmax=23");
    }
    return make_token_selection_request(options_, append_sampler_history(history_, token_ids));
}

DeviceSelectionRequest TokenSampler::device_selection_request(
    std::span<const std::uint32_t> hypothetical_suffix
) const {
    return make_device_request(history_, hypothetical_suffix);
}

TokenSelectionRequest TokenSampler::selection_request(
    const DeviceSelectionRequest& request
) const {
    if (!same_history(request.history, history_)) {
        throw std::invalid_argument("device selection request is not anchored to current history");
    }
    return make_token_selection_request(options_, request);
}

std::uint32_t TokenSampler::sample(std::span<const TokenLogit> compact_candidates) {
    const auto prepared = prepare_token_distribution(compact_candidates, options_, selection_request());
    if (is_greedy()) return prepared.values[0].token;
    std::uniform_real_distribution<double> distribution(0.0, prepared.total_sampling_weight);
    const auto draw = distribution(rng_);
    ++draw_count_;
    return select_prepared_token(prepared, draw);
}

void TokenSampler::observe_tokens(std::span<const std::uint32_t> token_ids) {
    history_ = append_sampler_history(history_, token_ids);
}

void TokenSampler::reset_history(std::span<const std::uint32_t> token_ids) {
    history_ = SamplerHistorySnapshot {
        .window_limit = static_cast<std::uint32_t>(options_.repetition_window),
    };
    history_ = append_sampler_history(history_, token_ids);
}

SamplerCheckpoint TokenSampler::checkpoint() const noexcept {
    return SamplerCheckpoint(
        options_, rng_, proposal_rng_, history_, lineage_id_, draw_count_, proposal_draw_count_
    );
}

TokenSamplerFork TokenSampler::fork(const SamplerCheckpoint& checkpoint_value) const {
    if (checkpoint_value.lineage_id_ == 0U || checkpoint_value.lineage_id_ != lineage_id_) {
        throw std::invalid_argument("sampler checkpoint belongs to another authority lineage");
    }
    if (checkpoint_value.options_ != options_) {
        throw std::invalid_argument("sampler checkpoint options are incompatible");
    }
    validate_history_snapshot(checkpoint_value.history_);
    return TokenSamplerFork(checkpoint_value);
}

TokenSamplerFork TokenSampler::fork() const { return fork(checkpoint()); }

void TokenSampler::adopt_checkpoint(const SamplerCheckpoint& checkpoint_value) {
    if (checkpoint_value.lineage_id_ == 0U || checkpoint_value.lineage_id_ != lineage_id_) {
        throw std::invalid_argument("sampler checkpoint belongs to another authority lineage");
    }
    if (checkpoint_value.options_ != options_) {
        throw std::invalid_argument("sampler checkpoint options are incompatible");
    }
    validate_history_snapshot(checkpoint_value.history_);
    rng_ = checkpoint_value.rng_;
    proposal_rng_ = checkpoint_value.proposal_rng_;
    history_ = checkpoint_value.history_;
    draw_count_ = checkpoint_value.draw_count_;
    proposal_draw_count_ = checkpoint_value.proposal_draw_count_;
}

SamplerTicket TokenSampler::preview_ticket(std::size_t offset) const {
    if (offset >= sampler_max_preview_ticket_count) {
        throw std::out_of_range("sampler ticket offset exceeds Arm-S 47-ticket capacity");
    }
    return preview_tickets(offset + 1U).at(offset);
}

SamplerTicketPool TokenSampler::preview_tickets(std::size_t count) const {
    if (count > sampler_max_preview_ticket_count) {
        throw std::length_error("sampler ticket request exceeds Arm-S 47-ticket capacity");
    }

    SamplerTicketPool pool;
    pool.count = static_cast<std::uint32_t>(count);
    auto cursor = rng_;
    for (std::size_t offset = 0; offset < count; ++offset) {
        auto& ticket = pool.tickets[offset];
        ticket.options_ = options_;
        ticket.pre_rng_ = cursor;
        ticket.post_rng_ = cursor;
        ticket.lineage_id_ = lineage_id_;
        ticket.draw_index_ = draw_count_ + (is_greedy() ? 0U : offset);
        ticket.offset_ = static_cast<std::uint32_t>(offset);
        ticket.consumes_random_draw_ = !is_greedy();
        if (is_greedy()) continue;

        auto direct_rng = cursor;
        std::uniform_real_distribution<double> unit_distribution(0.0, 1.0);
        const auto direct_draw = unit_distribution(direct_rng);

        AuditedSamplerEngine audited(cursor, ticket.raw_engine_outputs_, ticket.raw_engine_output_count_);
        const auto audited_draw = unit_distribution(audited);
        if (std::bit_cast<std::uint64_t>(direct_draw)
                != std::bit_cast<std::uint64_t>(audited_draw)
            || direct_rng != audited.state()) {
            throw std::runtime_error("sampler ticket audit diverged from libc++ distribution");
        }
        ticket.unit_interval_draw_ = direct_draw;
        ticket.post_rng_ = direct_rng;
        cursor = direct_rng;
    }
    return pool;
}

SamplerTicketPool TokenSampler::preview_external_tickets() const {
    return preview_tickets(external_speculation_max_horizon);
}

SamplerTicketPool TokenSampler::preview_stream_tickets() const {
    return preview_tickets(sampler_max_preview_ticket_count);
}

ProposalRandomWords TokenSampler::preview_proposal_words(std::size_t count) const {
    return preview_proposal_random_words(proposal_rng_, proposal_draw_count_, count);
}

}  // namespace local_gguf_engine::engine
