#include "local_gguf_engine/engine/standalone_chat.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/engine/gemma3_draft.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {
TokenSamplerOptions resolve_standalone_draft_sampler_options(const TokenSamplerOptions& target,
    const StandaloneDraftSamplingOptions& overrides, std::uint32_t vocabulary) {
    // Reuse target validation without constructing a second sampler/RNG. Its
    // history is irrelevant here, but the public validator requires a matching
    // empty history shape. Invalid target fields cannot be hidden by overrides.
    SamplerHistorySnapshot history;
    history.window_limit = static_cast<std::uint32_t>(target.repetition_window);
    static_cast<void>(make_token_selection_request(target, history));
    auto result = target;
    if (overrides.temperature) result.temperature = *overrides.temperature;
    if (overrides.top_k) result.top_k = *overrides.top_k;
    if (overrides.top_p) result.top_p = *overrides.top_p;
    if (overrides.min_p) result.min_p = *overrides.min_p;
    if (overrides.repetition_penalty) result.repetition_penalty = *overrides.repetition_penalty;
    if (vocabulary == 0 || result.top_k > vocabulary
        || !std::isfinite(result.temperature) || result.temperature < 0.0F
        || !std::isfinite(result.top_p) || result.top_p <= 0.0F || result.top_p > 1.0F
        || !std::isfinite(result.min_p) || result.min_p < 0.0F || result.min_p > 1.0F
        || !std::isfinite(result.repetition_penalty) || result.repetition_penalty <= 0.0F)
        throw std::invalid_argument("standalone draft sampling scalars are invalid or exceed its native vocabulary");
    return result;
}

struct StandaloneChatCheckpointState {
    std::uint64_t epoch = 0;
    bool active = false;
};
bool StandaloneChatCheckpoint::valid() const noexcept {
    return state_ && state_->active && epoch_ != 0 && state_->epoch == epoch_;
}
StandaloneChatCheckpoint::~StandaloneChatCheckpoint() {
    // Abandoning a saved snapshot releases storage, never silently rolls back
    // an already delivered token or performs GPU work from a destructor.
    if (valid()) state_->active = false;
}
StandaloneChatCheckpoint::StandaloneChatCheckpoint(StandaloneChatCheckpoint&& other) noexcept
    : state_(std::move(other.state_)), epoch_(std::exchange(other.epoch_, 0)) {}
StandaloneChatCheckpoint& StandaloneChatCheckpoint::operator=(StandaloneChatCheckpoint&& other) noexcept {
    if (this != &other) {
        if (valid()) state_->active = false;
        state_ = std::move(other.state_); epoch_ = std::exchange(other.epoch_, 0);
    }
    return *this;
}
namespace {
constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();
std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::size_t text_capacity(const tokenizer::Tokenizer& tokenizer, std::uint32_t context) {
    std::size_t width = 1;
    for (const auto& token : tokenizer.vocabulary()) width = std::max(width, token.text.size() + 1);
    if (context == 0 || width > (std::numeric_limits<std::size_t>::max() / 4 - 4096) / context)
        throw std::length_error("standalone tokenizer capacity overflow");
    return width * context;
}
std::vector<std::int32_t> signed_mapping(std::span<const std::uint32_t> mapping) {
    std::vector<std::int32_t> result(mapping.size());
    for (std::size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] != StandaloneTliMapping::unmapped && mapping[i] > INT32_MAX)
            throw std::invalid_argument("standalone token mapping exceeds int32");
        result[i] = mapping[i] == StandaloneTliMapping::unmapped ? -1 : static_cast<std::int32_t>(mapping[i]);
    }
    return result;
}
void add_stats(StandaloneDraftExecutionStats& total, const StandaloneDraftExecutionStats& value) {
#define ADD(field) if (value.field > std::numeric_limits<decltype(total.field)>::max() - total.field) \
    throw std::overflow_error("standalone draft metric overflow"); total.field += value.field
    ADD(gpu_duration_ns); ADD(wall_duration_ns); ADD(readback_bytes); ADD(uploaded_bytes);
    ADD(command_buffer_count); ADD(compute_dispatch_count); ADD(token_count);
    ADD(memory_barrier_count);
#undef ADD
}
StandaloneDraftExecutionStats difference(const StandaloneDraftExecutionStats& end,
    const StandaloneDraftExecutionStats& start) {
    auto result = end;
#define SUB(field) result.field -= start.field
    SUB(gpu_duration_ns); SUB(wall_duration_ns); SUB(readback_bytes); SUB(uploaded_bytes);
    SUB(command_buffer_count); SUB(compute_dispatch_count); SUB(token_count);
    SUB(memory_barrier_count);
#undef SUB
    return result;
}

std::uint32_t transaction_capacity(const StandaloneTliMapping& mapping, std::uint32_t horizon) {
    // Reserve both root/correction conversions and an already committed
    // pending token using the authenticated tokenizer's expansion bound.
    const auto per_token = mapping.max_append_tokens();
    if (per_token > (std::numeric_limits<std::uint32_t>::max() - horizon - 1U) / 2U)
        throw std::length_error("standalone transaction capacity overflow");
    return static_cast<std::uint32_t>(2 * per_token + horizon + 1);
}
} // namespace

struct StandaloneChatPlan::Impl {
    std::shared_ptr<StandaloneDraftArtifact> source;
    // Immutable copies stay at stable addresses inside this non-moving owner.
    tokenizer::Tokenizer target_tokenizer;
    tokenizer::Tokenizer draft_tokenizer;
    StandaloneTliMapping mapping;
    Gemma3DraftPlan draft;
    ChainChatVerifier verifier;
    StandaloneExecutionOptions execution;
    std::vector<std::uint32_t> target_history, draft_history, encoded;
    std::vector<std::size_t> ends, bases;
    std::vector<char> text;
    tokenizer::TokenizerEncodeWorkspace workspace;
    struct SavedContext {
        std::vector<std::uint32_t> target_history, draft_history;
        std::vector<std::size_t> ends, bases;
        std::uint32_t target_count = 0;
        std::size_t draft_count = 0, base = 0;
        bool final_token_pending = false;
        SavedContext(std::size_t target_capacity, std::size_t draft_capacity)
            : target_history(target_capacity), draft_history(draft_capacity),
              ends(target_capacity + 1), bases(target_capacity + 1) {}
        std::size_t owned_bytes() const noexcept {
            return 4 * (target_history.capacity() + draft_history.capacity())
                + sizeof(std::size_t) * (ends.capacity() + bases.capacity());
        }
    } saved;
    std::shared_ptr<StandaloneChatCheckpointState> checkpoint_state =
        std::make_shared<StandaloneChatCheckpointState>();
    std::uint32_t target_count = 0;
    std::size_t draft_count = 0, base = 0;
    StandaloneDraftExecutionStats stats;
    std::uint64_t rebuild_ns = 0;
    bool failed = false;
    bool cycle_active = false;

    Impl(std::shared_ptr<StandaloneDraftArtifact> artifact, const tokenizer::Tokenizer& tokenizer,
        const Gemma2MetalModelPlan& target, StandaloneExecutionOptions options)
        : source(std::move(artifact)), target_tokenizer(tokenizer),
          draft_tokenizer(source->tokenizer()), mapping(target_tokenizer, draft_tokenizer),
          draft(*source, signed_mapping(mapping.draft_to_target()), target.config().vocab_size,
              options.horizon, transaction_capacity(mapping, options.horizon)),
          verifier(target, options.horizon, SpeculativeSamplingMode::lossless, options.conditional_verification),
          execution(options), target_history(target.config().context_length),
          draft_history(source->context_limit()),
          encoded(draft_history.size()), ends(target_history.size() + 1, absent),
          bases(target_history.size() + 1, absent), text(text_capacity(tokenizer, target.config().context_length)),
          workspace(text.size()), saved(target_history.size(), draft_history.size()) { ends[0] = bases[0] = 0; }

#if defined(LOCAL_GGUF_ENGINE_TESTING)
    Impl(std::span<const tensor::TensorView> tensors, const tokenizer::Tokenizer& draft_tokens,
        const tokenizer::Tokenizer& target_tokens, const Gemma2MetalModelPlan& target,
        StandaloneExecutionOptions options)
        : target_tokenizer(target_tokens), draft_tokenizer(draft_tokens), mapping(target_tokenizer, draft_tokenizer),
          draft(tensors, signed_mapping(mapping.draft_to_target()), target.config().vocab_size, options.horizon,
              transaction_capacity(mapping, options.horizon)),
          verifier(target, options.horizon, SpeculativeSamplingMode::lossless, options.conditional_verification),
          execution(options), target_history(target.config().context_length),
          draft_history(Gemma3DraftPlan::context_capacity),
          encoded(draft_history.size()), ends(target_history.size() + 1, absent),
          bases(target_history.size() + 1, absent), text(text_capacity(target_tokens, target.config().context_length)),
          workspace(text.size()), saved(target_history.size(), draft_history.size()) { ends[0] = bases[0] = 0; }
#endif
    ~Impl() { checkpoint_state->active = false; }

    void ready() const {
        if (failed || draft.poisoned()) throw std::logic_error("standalone draft state is poisoned");
    }
    void restore(std::size_t restored_base, std::size_t restored_end) {
        if (restored_base > restored_end || restored_end > draft_history.size()
            || restored_end - restored_base > Gemma3DraftPlan::context_capacity)
            throw std::logic_error("invalid standalone draft window receipt");
        const auto started = now_ns();
        draft.reset();
        if (restored_end != restored_base)
            add_stats(stats, draft.append(std::span<const std::uint32_t>(draft_history)
                .subspan(restored_base, restored_end - restored_base)));
        base = restored_base;
        rebuild_ns += now_ns() - started;
    }
    void append_emissions(std::span<const std::uint32_t> tokens) {
        if (tokens.size() > draft_history.size() - draft_count)
            throw std::length_error("standalone draft emission ledger overflow");
        if (tokens.size() > Gemma3DraftPlan::context_capacity - (draft_count - base))
            throw std::length_error("Gemma3 draft context capacity exhausted; target context was not shortened");
        const auto previous = draft_count;
        std::copy(tokens.begin(), tokens.end(), draft_history.begin() + static_cast<std::ptrdiff_t>(draft_count));
        draft_count += tokens.size();
        if (!tokens.empty()) {
            add_stats(stats, draft.append(std::span<const std::uint32_t>(draft_history)
                .subspan(previous, tokens.size())));
        }
    }
    void record(std::uint32_t token) {
        if (target_count == target_history.size()) throw std::length_error("standalone target ledger overflow");
        target_history[target_count++] = token;
        ends[target_count] = draft_count;
        bases[target_count] = base;
    }
    void observe(std::uint32_t token, std::uint32_t position) {
        if (position != target_count) throw std::logic_error("standalone commit target position mismatch");
        if (target_count == target_history.size()) throw std::length_error("standalone target ledger overflow");
        const auto count = mapping.append_target_token(token, encoded, text, workspace);
        if (count > Gemma3DraftPlan::context_capacity - (draft_count - base))
            throw std::length_error("standalone target token exceeds the draft's native context capacity");
        try {
            append_emissions(std::span<const std::uint32_t>(encoded).first(count));
            record(token);
        } catch (...) { failed = true; throw; }
    }
};

StandaloneChatPlan::StandaloneChatPlan(std::shared_ptr<StandaloneDraftArtifact> source,
    const tokenizer::Tokenizer& tokenizer, const Gemma2MetalModelPlan& target,
    StandaloneExecutionOptions options) {
    if (!source || options.horizon == 0 || options.horizon > external_speculation_max_horizon
        || tokenizer.vocabulary().size() != target.config().vocab_size)
        throw std::invalid_argument("standalone draft source, horizon or target tokenizer is incompatible");
    static_cast<void>(resolve_standalone_draft_sampler_options(TokenSamplerOptions{}, options.draft_sampling,
        static_cast<std::uint32_t>(source->tokenizer().vocabulary().size())));
    impl_ = std::make_unique<Impl>(std::move(source), tokenizer, target, options);
}
StandaloneChatPlan::~StandaloneChatPlan() = default;
#if defined(LOCAL_GGUF_ENGINE_TESTING)
StandaloneChatPlan::StandaloneChatPlan(std::span<const tensor::TensorView> tensors,
    const tokenizer::Tokenizer& draft_tokenizer, const tokenizer::Tokenizer& target_tokenizer,
    const Gemma2MetalModelPlan& target, StandaloneExecutionOptions options) {
    if (options.horizon == 0 || options.horizon > external_speculation_max_horizon
        || draft_tokenizer.vocabulary().size() != Gemma3DraftPlan::vocabulary_size
        || target_tokenizer.vocabulary().size() != target.config().vocab_size)
        throw std::invalid_argument("standalone test draft requires complete production geometry");
    static_cast<void>(resolve_standalone_draft_sampler_options(TokenSamplerOptions{}, options.draft_sampling,
        Gemma3DraftPlan::vocabulary_size));
    impl_ = std::make_unique<Impl>(tensors, draft_tokenizer, target_tokenizer, target, options);
}
const Gemma3DraftPlan& StandaloneChatPlan::draft_plan_for_testing() const noexcept { return impl_->draft; }
const Gemma3DraftPlan* StandaloneChatPlan::gemma3_draft_plan_for_testing() const noexcept {
    return &impl_->draft;
}
#endif
StandaloneChatPlan::StandaloneChatPlan(StandaloneChatPlan&&) noexcept = default;
StandaloneChatPlan& StandaloneChatPlan::operator=(StandaloneChatPlan&&) noexcept = default;
std::uint32_t StandaloneChatPlan::horizon() const noexcept { return impl_->execution.horizon; }
StandaloneDraftKind StandaloneChatPlan::kind() const noexcept {
    return StandaloneDraftKind::gemma3_270m_q8;
}
const StandaloneExecutionOptions& StandaloneChatPlan::execution_options() const noexcept { return impl_->execution; }
TokenSamplerOptions StandaloneChatPlan::effective_draft_sampler_options(const TokenSamplerOptions& target) const {
    return resolve_standalone_draft_sampler_options(target, impl_->execution.draft_sampling,
        static_cast<std::uint32_t>(impl_->mapping.draft_to_target().size()));
}
bool StandaloneChatPlan::poisoned() const noexcept { return impl_->failed || impl_->draft.poisoned(); }
std::uint32_t StandaloneChatPlan::target_position() const noexcept { return impl_->target_count; }
std::uint64_t StandaloneChatPlan::rebuild_duration_ns() const noexcept { return impl_->rebuild_ns; }
StandaloneDraftExecutionStats StandaloneChatPlan::execution_stats() const noexcept { return impl_->stats; }
std::size_t StandaloneChatPlan::owned_bytes() const noexcept {
    const auto& p = *impl_;
    return sizeof(Impl) + p.draft.owned_bytes() + p.verifier.owned_bytes() + p.saved.owned_bytes()
        + sizeof(StandaloneChatCheckpointState)
        + 4 * (p.target_history.capacity() + p.draft_history.capacity() + p.encoded.capacity())
        + sizeof(std::size_t) * (p.ends.capacity() + p.bases.capacity()) + p.text.capacity()
        + p.workspace.owned_bytes() + p.target_tokenizer.owned_bytes() + p.draft_tokenizer.owned_bytes()
        + p.mapping.owned_bytes();
}
StandaloneChatCheckpoint StandaloneChatPlan::checkpoint() {
    auto& p = *impl_; p.ready();
    if (p.cycle_active) throw std::logic_error("cannot checkpoint an active standalone cycle");
    if (p.checkpoint_state->active) throw std::logic_error("standalone checkpoint slot is already leased");
    if (p.checkpoint_state->epoch == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("standalone checkpoint epoch exhausted");
    auto& saved = p.saved;
    std::copy_n(p.target_history.begin(), p.target_count, saved.target_history.begin());
    std::copy_n(p.draft_history.begin(), p.draft_count, saved.draft_history.begin());
    std::copy_n(p.ends.begin(), p.target_count + 1, saved.ends.begin());
    std::copy_n(p.bases.begin(), p.target_count + 1, saved.bases.begin());
    saved.target_count = p.target_count; saved.draft_count = p.draft_count; saved.base = p.base;
    saved.final_token_pending = p.draft.materialized_position() < p.draft.position();
    ++p.checkpoint_state->epoch; p.checkpoint_state->active = true;
    StandaloneChatCheckpoint result;
    result.state_ = p.checkpoint_state; result.epoch_ = p.checkpoint_state->epoch;
    return result;
}
void StandaloneChatPlan::discard(StandaloneChatCheckpoint& checkpoint) {
    auto& p = *impl_;
    if (p.cycle_active) throw std::logic_error("cannot discard a checkpoint during an active standalone cycle");
    if (checkpoint.state_ != p.checkpoint_state || !checkpoint.valid())
        throw std::logic_error("standalone checkpoint is stale or belongs to another plan");
    p.checkpoint_state->active = false;
    checkpoint.state_.reset(); checkpoint.epoch_ = 0;
}
void StandaloneChatPlan::restore(StandaloneChatCheckpoint& checkpoint) {
    auto& p = *impl_;
    if (p.cycle_active) throw std::logic_error("cannot restore during an active standalone cycle");
    // A failed conversion/ledger operation can be reconstructed from this
    // independent receipt. A failed device cannot be made safe by replay.
    if (p.draft.poisoned()) throw std::logic_error("cannot restore a poisoned standalone device");
    if (checkpoint.state_ != p.checkpoint_state || !checkpoint.valid())
        throw std::logic_error("standalone checkpoint is stale or belongs to another plan");
    const auto started = now_ns();
    const auto& saved = p.saved;
    try {
        p.draft.reset();
        const auto cached_end = saved.draft_count - (saved.final_token_pending ? 1U : 0U);
        if (cached_end != saved.base)
            add_stats(p.stats, p.draft.append(std::span<const std::uint32_t>(saved.draft_history)
                .subspan(saved.base, cached_end - saved.base)));
        if (saved.final_token_pending)
            add_stats(p.stats, p.draft.restore_pending_token(saved.draft_history[cached_end]));
        std::copy_n(saved.target_history.begin(), saved.target_count, p.target_history.begin());
        std::copy_n(saved.draft_history.begin(), saved.draft_count, p.draft_history.begin());
        std::copy_n(saved.ends.begin(), saved.target_count + 1, p.ends.begin());
        std::copy_n(saved.bases.begin(), saved.target_count + 1, p.bases.begin());
        p.target_count = saved.target_count; p.draft_count = saved.draft_count; p.base = saved.base;
        p.failed = false;
        p.rebuild_ns += now_ns() - started;
        discard(checkpoint);
    } catch (...) {
        p.failed = true;
        p.checkpoint_state->active = false;
        throw;
    }
}
void StandaloneChatPlan::reset() {
    auto& p = *impl_; p.ready();
    if (p.cycle_active) throw std::logic_error("cannot reset during an active standalone cycle");
    p.draft.reset(); p.target_count = 0; p.draft_count = 0; p.base = 0;
    std::fill(p.ends.begin(), p.ends.end(), absent); std::fill(p.bases.begin(), p.bases.end(), absent);
    p.ends[0] = p.bases[0] = 0;
}
void StandaloneChatPlan::synchronize_context(std::span<const std::uint32_t> tokens, bool replace_context) {
    auto& p = *impl_; p.ready();
    if (p.cycle_active) throw std::logic_error("cannot synchronize during an active standalone cycle");
    if (tokens.size() > p.target_history.size()) throw std::length_error("standalone target context overflow");
    std::size_t common = 0;
    while (!replace_context && common < tokens.size() && common < p.target_count
        && tokens[common] == p.target_history[common]) ++common;
    if (!replace_context && common == tokens.size() && common == p.target_count) return;
    if (common != p.target_count && p.ends[common] == absent) common = 0;
    const auto restored_end = common == p.target_count ? p.draft_count : p.ends[common];
    const auto restored_base = common == p.target_count ? p.base : p.bases[common];
    // Complete potentially failing input conversion before rewinding either
    // the emission ledger or a sliding cache. The encoded scratch is private.
    const auto count = p.mapping.encode_target_context(tokens.subspan(common), p.encoded, p.text, p.workspace);
    if (count > p.draft_history.size() - restored_end
        || (count > Gemma3DraftPlan::context_capacity - (restored_end - restored_base)))
        throw std::length_error("standalone converted prompt exceeds its prepared native context capacity");
    try {
        if (replace_context) reset();
        else if (common != p.target_count) {
            if (p.ends[common] == absent) { reset(); common = 0; }
            else {
                const auto end = p.ends[common], base = p.bases[common];
                if (base == p.base && end >= base && end - base <= p.draft.position()
                    && p.draft.can_truncate(static_cast<std::uint32_t>(end - base)))
                    p.draft.truncate(static_cast<std::uint32_t>(end - base));
                else p.restore(base, end);
                p.target_count = static_cast<std::uint32_t>(common); p.draft_count = end; p.base = base;
            }
        }
        if (common == tokens.size()) return;
        p.append_emissions(std::span<const std::uint32_t>(p.encoded).first(count));
        for (std::size_t i = common; i < tokens.size(); ++i) {
            p.target_history[i] = tokens[i]; p.ends[i + 1] = p.bases[i + 1] = absent;
        }
        p.target_count = static_cast<std::uint32_t>(tokens.size());
        p.ends[p.target_count] = p.draft_count; p.bases[p.target_count] = p.base;
    } catch (...) { p.failed = true; throw; }
}
void StandaloneChatPlan::validate_committed_prefix(std::span<const std::uint32_t> tokens) {
    auto& p = *impl_; p.ready();
    if (tokens.size() > p.target_history.size() - p.target_count)
        throw std::length_error("standalone target commit ledger capacity exhausted");
    auto available = Gemma3DraftPlan::context_capacity - (p.draft_count - p.base);
    for (const auto token : tokens) {
        const auto count = p.mapping.append_target_token(token, p.encoded, p.text, p.workspace);
        if (count > available)
            throw std::length_error("standalone target commit exceeds the draft's native context capacity");
        available -= count;
    }
}
void StandaloneChatPlan::observe_committed(std::uint32_t token, std::uint32_t position) {
    auto& p = *impl_; p.ready();
    if (position < p.target_count && p.target_history[position] == token) return;
    if (p.cycle_active) throw std::logic_error("cannot publish a new observation during an active standalone cycle");
    p.observe(token, position);
}
StandaloneCycleResult StandaloneChatPlan::run(Gemma2MetalModelPlan& target, TokenSampler& sampler,
    std::uint32_t root, std::uint32_t position, std::size_t remaining,
    std::span<const std::uint32_t> stops, const ChainCommitCallback& callback) {
    auto& p = *impl_; p.ready();
    if (p.cycle_active) throw std::logic_error("standalone cycles are not reentrant");
    if (position != p.target_count) throw std::logic_error("standalone cycle target frontier mismatch");
    const auto draft_options = effective_draft_sampler_options(sampler.options());
    struct CycleGuard {
        bool& active;
        explicit CycleGuard(bool& value) : active(value) { active = true; }
        ~CycleGuard() { active = false; }
    } guard(p.cycle_active);
    const auto before = p.stats;
    const auto started = now_ns();
    const auto old_end = p.draft_count, old_base = p.base;
    bool delivered = false;
    bool transaction_started = false;
    try {
        const auto root_count = p.mapping.append_target_token(root, p.encoded, p.text, p.workspace);
        {
            const auto available = Gemma3DraftPlan::context_capacity - (p.draft_count - p.base);
            // The final target correction is not known until verification.
            // Prove room for its complete native encoding before a callback
            // can make any part of this cycle irreversible.
            const auto required = root_count + std::max<std::size_t>(horizon(),
                horizon() - 1U + p.mapping.max_append_tokens());
            if (required > available)
                throw std::length_error("Gemma3 draft lacks context capacity for the requested chain and target correction");
        }
        const auto transaction_stats = p.draft.begin_transaction();
        transaction_started = true;
        add_stats(p.stats, transaction_stats);
        p.append_emissions(std::span<const std::uint32_t>(p.encoded).first(root_count));
        if (p.draft.position() == 0) throw std::invalid_argument("standalone draft has no tokenized context");
        const auto after_root = p.draft_count;
        const auto words = sampler.preview_proposal_words(horizon());
        const auto proposal = p.draft.propose(root,
            std::span<const std::uint32_t>(words.words).first(horizon()),
            draft_options, sampler.history_snapshot());
        add_stats(p.stats, proposal.execution);
        auto draft_elapsed = now_ns() - started;
        if (proposal.target_tokens.size() != horizon() || proposal.draft_tokens.size() != horizon())
            throw std::logic_error("standalone draft omitted a requested proposal");
        const auto sample = [&](TokenSamplerFork& fork, std::uint32_t depth,
            std::span<const TokenLogit> candidates) {
            const auto gather_started = now_ns();
            const auto previous_readback = p.draft.probability_readback_bytes();
            const auto distribution = p.draft.gather(depth, fork.prepare_distribution(candidates));
            if (distribution.lease.row() != depth)
                throw std::logic_error("standalone probability lease has incorrect proposal depth");
            p.stats.readback_bytes += p.draft.probability_readback_bytes() - previous_readback;
            draft_elapsed += now_ns() - gather_started;
            return fork.sample_speculative(proposal.target_tokens[depth], distribution, candidates);
        };
        const auto committed = [&](const ChainCommittedToken& token) {
            if (token.depth == 0) p.record(root);
            else {
                // A stop or budget correction is cached by the target endpoint
                // but is not a draft proposal. Conversion follows finalization.
                if (token.depth > horizon() || token.token_id != proposal.target_tokens[token.depth - 1]) {
                    if (!token.stop_token && token.depth + 1 != remaining)
                        throw std::logic_error("non-endpoint correction committed inside standalone chain");
                    const auto correction_started = now_ns();
                    p.draft.truncate(static_cast<std::uint32_t>(p.draft_count - p.base));
                    p.observe(token.token_id, token.cached_token_count - 1);
                    draft_elapsed += now_ns() - correction_started;
                } else {
                    if (p.draft_count != after_root + token.depth - 1 || p.draft_count == p.draft_history.size())
                        throw std::logic_error("standalone committed draft prefix is discontinuous");
                    p.draft_history[p.draft_count++] = proposal.draft_tokens[token.depth - 1];
                    p.record(token.token_id);
                }
            }
            delivered = true;
            return callback(token);
        };
        StandaloneCycleResult result;
        static_cast<ChainCycleResult&>(result) = p.verifier.run(target, sampler, root, position,
            remaining, stops, {.tokens=proposal.target_tokens, .confidence_logits={},
                .source=ProposalFeatureSource::draft_hidden, .device=p.draft.verifier_inputs(proposal.epoch)},
            sample, committed);
        const auto finalize_started = now_ns();
        add_stats(p.stats, p.draft.commit_prefix(static_cast<std::uint32_t>(p.draft_count - p.base)));
        draft_elapsed += now_ns() - finalize_started;
        result.draft = difference(p.stats, before);
        result.draft.wall_duration_ns = draft_elapsed;
        return result;
    } catch (...) {
        try {
            if (!delivered) {
                if (transaction_started) p.draft.abort_transaction();
                p.draft_count = old_end;
                if (old_base == p.base) p.draft.truncate(static_cast<std::uint32_t>(old_end - old_base));
                else p.restore(old_base, old_end);
                p.base = old_base;
            } else add_stats(p.stats, p.draft.commit_prefix(static_cast<std::uint32_t>(p.draft_count - p.base)));
        } catch (...) { p.failed = true; }
        throw;
    }
}
void StandaloneChatPlan::warm_up(Gemma2MetalModelPlan& target, const TokenSampler& sampler) {
    auto& p = *impl_; p.ready();
    if (p.cycle_active) throw std::logic_error("cannot warm up during an active standalone cycle");
    static_cast<void>(effective_draft_sampler_options(sampler.options()));
    p.draft.warm_up();
    const auto mapped = p.mapping.draft_to_target();
    const auto found = std::find_if(mapped.begin(), mapped.end(), [](auto token) { return token != StandaloneTliMapping::unmapped; });
    if (found == mapped.end()) throw std::logic_error("standalone warm-up has no mapped token");
    const std::array<std::uint32_t, 1> prompt {*found};
    synchronize_context(prompt);
    auto warm_sampler = sampler;
    warm_sampler.observe_tokens(prompt);
    static_cast<void>(run_gemma2_metal_resident_prefill(target, prompt, 0,
        gemma2_input_embedding_scale(target.config().embedding_length), warm_sampler.selection_request()));
    warm_sampler.observe_tokens(prompt);
    const auto stop = [](const ChainCommittedToken&) { return Gemma2MetalTokenStreamDecision::stop_generation; };
    static_cast<void>(run(target, warm_sampler, prompt[0], 1, 2, {}, stop));
    p.verifier.warm_up_tiles(target);
    reset();
}
} // namespace local_gguf_engine::engine
