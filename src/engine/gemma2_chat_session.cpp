#include "local_gguf_engine/engine/gemma2_chat_session.hpp"

#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/engine/gemma2_model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

[[nodiscard]] SpeculativeSamplingMode resolve_speculative_sampling(const Gemma2ChatSessionOptions& options) {
    const auto sampling = options.speculative_sampling.value_or(options.standalone_draft_model
        ? SpeculativeSamplingMode::lossless : SpeculativeSamplingMode::exact_match);
    if (sampling != SpeculativeSamplingMode::exact_match && sampling != SpeculativeSamplingMode::lossless)
        throw std::invalid_argument("unsupported speculative sampling mode");
    if (options.standalone_draft_model) {
        if (!options.speculative_decode)
            throw std::invalid_argument("requested standalone draft conflicts with disabled speculation");
        if (sampling != SpeculativeSamplingMode::lossless)
            throw std::invalid_argument("standalone draft requires lossless sampling");
    } else if (sampling != SpeculativeSamplingMode::exact_match) {
        throw std::invalid_argument("lossless sampling requires an explicit standalone draft");
    }
    return sampling;
}

class ActiveTurn final {
public:
    explicit ActiveTurn(bool& active) : active_(active) {
        if (active_) throw std::logic_error("chat session mutation during an active turn");
        active_ = true;
    }
    ~ActiveTurn() { active_ = false; }
private:
    bool& active_;
};

[[nodiscard]] std::uint64_t monotonic_now_ns(const Gemma2MonotonicClock& clock) {
    if (clock) {
        return clock();
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count());
}

[[nodiscard]] std::uint64_t elapsed_ns(std::uint64_t start, std::uint64_t end) {
    if (end < start) {
        throw std::runtime_error("Gemma2 chat measurement clock moved backwards");
    }
    return end - start;
}

[[nodiscard]] std::optional<std::uint32_t> find_token_id_by_text(
    const tokenizer::Tokenizer& tokenizer,
    std::string_view text
) noexcept {
    const auto& vocabulary = tokenizer.vocabulary();
    for (std::size_t token_id = 0; token_id < vocabulary.size(); ++token_id) {
        if (vocabulary[token_id].text == text) {
            return static_cast<std::uint32_t>(token_id);
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::vector<std::uint32_t> chat_stop_token_ids(
    const tokenizer::Tokenizer& tokenizer,
    std::optional<std::uint32_t> end_of_turn_token_id
) {
    std::vector<std::uint32_t> token_ids;
    if (end_of_turn_token_id.has_value()) {
        token_ids.push_back(*end_of_turn_token_id);
    }
    if (
        tokenizer.special_tokens().eos_token_id.has_value()
        && std::find(
            token_ids.begin(),
            token_ids.end(),
            *tokenizer.special_tokens().eos_token_id
        ) == token_ids.end()
    ) {
        token_ids.push_back(*tokenizer.special_tokens().eos_token_id);
    }

    return token_ids;
}

[[nodiscard]] std::uint32_t checked_cached_token_count(std::size_t token_count) {
    if (token_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Gemma2 chat session cached token count exceeds uint32");
    }

    return static_cast<std::uint32_t>(token_count);
}

[[nodiscard]] std::size_t max_reverse_stop_string_length(std::span<const std::string> reverse_stop_strings) {
    std::size_t max_length = 0;
    for (const auto& stop_string : reverse_stop_strings) {
        max_length = std::max(max_length, stop_string.size());
    }

    return max_length;
}

[[nodiscard]] std::size_t safe_stream_text_size(
    std::string_view text,
    std::span<const std::string> reverse_stop_strings
) {
    const auto max_stop_length = max_reverse_stop_string_length(reverse_stop_strings);
    if (max_stop_length == 0U) {
        return text.size();
    }

    const auto holdback = max_stop_length - 1U;
    if (text.size() <= holdback) {
        return 0;
    }

    return text.size() - holdback;
}

[[nodiscard]] std::optional<std::pair<std::size_t, std::string>> first_reverse_stop_match(
    std::string_view text,
    std::span<const std::string> reverse_stop_strings,
    std::size_t search_offset
) {
    std::optional<std::pair<std::size_t, std::string>> first_match;
    for (const auto& stop_string : reverse_stop_strings) {
        if (stop_string.empty()) {
            continue;
        }

        const auto position = text.find(stop_string, search_offset);
        if (position == std::string_view::npos) {
            continue;
        }
        if (!first_match.has_value() || position < first_match->first) {
            first_match = std::pair<std::size_t, std::string>(position, stop_string);
        }
    }

    return first_match;
}

void emit_text_delta(
    const Gemma2ChatStreamCallback& on_text_delta,
    std::string_view text,
    std::size_t& emitted_size,
    std::size_t target_size
) {
    if (target_size <= emitted_size) {
        return;
    }
    if (target_size > text.size()) {
        throw std::invalid_argument("Gemma2 chat stream target exceeds text size");
    }

    if (on_text_delta) {
        on_text_delta(text.substr(emitted_size, target_size - emitted_size));
    }
    emitted_size = target_size;
}

}  // namespace

Gemma2ChatSessionState::Gemma2ChatSessionState(
    model_format::GGUFFileBackedModel file, EngineModel model,
    tokenizer::Tokenizer tokenizer, PreparedGemma2Plan prepared,
    std::shared_ptr<StandaloneDraftArtifact> standalone, std::optional<std::filesystem::path> standalone_path
) : file_model_(std::move(file)), model_(std::move(model)),
    tokenizer_(std::move(tokenizer)), prepared_(std::move(prepared)),
    standalone_source_(std::move(standalone)), requested_standalone_(std::move(standalone_path)) {}

Gemma2ChatSessionState Gemma2ChatSessionState::prepare(
    model_format::GGUFFileBackedModel file, const Gemma2ChatSessionOptions& options,
    bool authenticate_target
) {
    if (options.backend != runtime::BackendKind::metal) {
        throw std::invalid_argument("Gemma 2 chat session requires Metal backend");
    }
    static_cast<void>(resolve_speculative_sampling(options));
    if (!options.standalone_draft_model && options.standalone_execution != StandaloneExecutionOptions{})
        throw std::invalid_argument("standalone execution options require a draft artifact");
    // Validate sampler before any device allocation, including target-only mode.
    static_cast<void>(TokenSampler(options.sampler));
    auto standalone = options.standalone_draft_model
        ? std::make_shared<StandaloneDraftArtifact>(*options.standalone_draft_model) : nullptr;
    if (standalone) {
        static_cast<void>(resolve_standalone_draft_sampler_options(options.sampler,
            options.standalone_execution.draft_sampling,
            static_cast<std::uint32_t>(standalone->tokenizer().vocabulary().size())));
    }
    auto model = standalone || authenticate_target
        ? load_verified_engine_model_from_gguf(file.view(), file.bytes())
        : load_engine_model_from_gguf(file.view());
    auto tokenizer = tokenizer::load_tokenizer_from_gguf_metadata(file.view().directory.metadata);
    auto prepared = standalone
        ? prepare_gemma2_standalone_plan(model, tokenizer, standalone, options.standalone_execution,
            options.output_head_policy, options.context_length, options.kv_precision)
        : prepare_gemma2_target_only_plan(model, options.output_head_policy, options.context_length, options.kv_precision);
    return Gemma2ChatSessionState(std::move(file), std::move(model), std::move(tokenizer),
        std::move(prepared), std::move(standalone), options.standalone_draft_model);
}

Gemma2ChatSession::Gemma2ChatSession(
    model_format::GGUFFileBackedModel file_model,
    Gemma2ChatSessionOptions options
)
    : Gemma2ChatSession(Gemma2ChatSessionState::prepare(std::move(file_model), options), options) {}

Gemma2ChatSession::Gemma2ChatSession(
    Gemma2ChatSessionState state,
    Gemma2ChatSessionOptions options
)
    : file_model_(std::move(state.file_model_)),
      model_(std::move(state.model_)),
      tokenizer_(std::move(state.tokenizer_)),
      prepared_(std::move(state.prepared_)),
      standalone_source_(std::move(state.standalone_source_)),
      output_head_policy_(prepared_->target().output_head_policy()),
      kv_precision_(prepared_->target().config().kv_precision),
      reverse_stop_strings_(std::move(options.reverse_stop_strings)),
      end_of_turn_token_id_(find_token_id_by_text(tokenizer_, "<end_of_turn>")),
      max_new_tokens_(options.max_new_tokens),
      context_length_(prepared_->target().config().context_length),
      context_shift_(options.context_shift),
      speculative_decode_(options.speculative_decode && options.retrieval_decode.value_or(
          model_.architecture() == ModelArchitecture::gemma2)),
      speculative_sampling_(resolve_speculative_sampling(options)),
      standalone_execution_(options.standalone_execution),
      initial_sampler_options_(options.sampler),
      sampler_(initial_sampler_options_),
      clock_(std::move(options.clock)) {
    if (options.backend != runtime::BackendKind::metal || plan().backend() != runtime::BackendKind::metal) {
        throw std::runtime_error("Gemma 2 chat session requires Metal backend");
    }
    if (model_.architecture() != ModelArchitecture::gemma2
        && model_.architecture() != ModelArchitecture::gemma3) {
        throw std::runtime_error("Gemma chat session requires a supported Gemma model");
    }
    stop_token_ids_ = chat_stop_token_ids(tokenizer_, end_of_turn_token_id_);
    if (kv_precision_ != resolve_gemma_kv_precision(model_.architecture(), options.kv_precision)) {
        throw std::invalid_argument("prepared session and requested KV precision differ");
    }
    if ((options.context_length != 0U && options.context_length != context_length_)
        || bool(standalone_source_) != options.standalone_draft_model.has_value()
        || state.requested_standalone_ != options.standalone_draft_model
        || (standalone_source_ && !options.speculative_decode)
        || output_head_policy_ != resolve_gemma2_output_head_policy(model_.gemma2_config(),
            bind_gemma2_model_weights(model_), options.output_head_policy)) {
        throw std::invalid_argument("prepared session and requested speculation mode differ");
    }
    if (auto* standalone = prepared_->standalone()) {
        if (speculative_sampling_ != SpeculativeSamplingMode::lossless
            || standalone->execution_options() != options.standalone_execution)
            throw std::invalid_argument("prepared standalone execution differs from requested configuration");
    } else if (speculative_sampling_ != SpeculativeSamplingMode::exact_match
        || options.standalone_execution != StandaloneExecutionOptions{}) {
        throw std::invalid_argument("standalone execution requires an explicit draft artifact");
    }
    cache_segments_.reserve(plan().config().context_length);
    for (const auto& token : tokenizer_.vocabulary()) {
        maximum_decoded_token_bytes_ = std::max(maximum_decoded_token_bytes_, token.text.size() + 1U);
    }
    warm_up_plan();
}

Gemma2MetalModelPlan& Gemma2ChatSession::plan() {
    if (!prepared_) throw std::logic_error("chat session requires reset after failed rebuild");
    return prepared_->target();
}
const Gemma2MetalModelPlan& Gemma2ChatSession::plan() const {
    if (!prepared_) throw std::logic_error("chat session requires reset after failed rebuild");
    return prepared_->target();
}
bool Gemma2ChatSession::poisoned() const noexcept { return !prepared_ || !prepared_->ready(); }

void Gemma2ChatSession::rebuild_durable_cache() {
    const auto tokens = std::span<const std::uint32_t>(cached_token_ids_);
    const auto scale = gemma2_input_embedding_scale(model_.gemma2_config().embedding_length);
    std::uint32_t restored = 0;
    for (const auto& segment : cache_segments_) {
        if (segment.begin != restored || segment.prompt_end <= segment.begin
            || segment.end < segment.prompt_end || segment.end > tokens.size()) {
            throw std::logic_error("invalid durable chat cache execution receipt");
        }
        static_cast<void>(run_gemma2_metal_resident_prefill(plan(),
            tokens.subspan(segment.begin, segment.prompt_end - segment.begin), segment.begin, scale));
        for (auto position = segment.prompt_end; position < segment.end; ++position) {
            static_cast<void>(run_gemma2_metal_resident_decode_step(plan(),
                create_gemma2_metal_decode_schedule(plan(), position), tokens[position], scale,
                std::nullopt, Gemma2MetalDecodeOutputMode::kv_only));
        }
        restored = segment.end;
    }
    if (restored != tokens.size()) throw std::logic_error("incomplete durable chat cache receipt");
    if (prepared_->standalone()) prepared_->standalone()->synchronize_context(tokens);
}

void Gemma2ChatSession::warm_up_plan() {
    warmup_stats_.reset();
    const auto token_id = tokenizer_.special_tokens().bos_token_id.value_or(
        tokenizer_.special_tokens().eos_token_id.value_or(0U)
    );
    if (token_id >= plan().config().vocab_size) {
        throw std::runtime_error("Gemma2 chat warm-up token is outside vocabulary");
    }
    std::array<std::uint32_t, 16> token_ids {};
    token_ids.fill(token_id);
    warmup_stats_ = warm_up_gemma2_metal_plan(
        plan(), std::span<const std::uint32_t, 16>(token_ids), sampler_
    );
    if (prepared_->standalone()) prepared_->standalone()->warm_up(plan(), sampler_);
    if (!messages_.empty() || !cached_token_ids_.empty() || !turn_cache_positions_.empty()
        || !turn_sampler_states_.empty()) {
        throw std::runtime_error("Gemma2 chat warm-up mutated logical session state");
    }
}

std::vector<tokenizer::ChatMessage> Gemma2ChatSession::message_views_with_user(
    std::string_view content,
    std::size_t first_message_index
) const {
    if (first_message_index > messages_.size()) {
        throw std::invalid_argument("Gemma2 chat first message index is out of range");
    }

    std::vector<tokenizer::ChatMessage> messages;
    messages.reserve(messages_.size() - first_message_index + 1U);
    for (auto index = first_message_index; index < messages_.size(); ++index) {
        const auto& message = messages_[index];
        messages.push_back(tokenizer::ChatMessage {
            .role = message.role,
            .content = message.content,
        });
    }
    messages.push_back(tokenizer::ChatMessage {
        .role = "user",
        .content = content,
    });

    return messages;
}

std::vector<std::uint32_t> Gemma2ChatSession::encode_messages(
    std::span<const tokenizer::ChatMessage> messages
) const {
    const auto formatted_prompt = tokenizer::format_chat_prompt_from_gguf_metadata(
        file_model_.view().directory.metadata,
        messages,
        true
    );
    const auto default_encode_options = tokenizer_.default_encode_options();
    return tokenizer_.encode(
        formatted_prompt.text,
        tokenizer::TokenizerEncodeOptions {
            .add_bos = formatted_prompt.add_bos.value_or(default_encode_options.add_bos),
            .add_eos = formatted_prompt.add_eos.value_or(default_encode_options.add_eos),
            .encode_special_tokens = formatted_prompt.encode_special_tokens,
        }
    );
}

std::vector<std::uint32_t> Gemma2ChatSession::visible_assistant_token_ids(
    std::span<const std::uint32_t> generated_token_ids
) const {
    std::vector<std::uint32_t> visible_token_ids;
    visible_token_ids.reserve(generated_token_ids.size());
    for (const auto token_id : generated_token_ids) {
        if (tokenizer_.token_id_is_special(token_id)) {
            continue;
        }

        visible_token_ids.push_back(token_id);
    }

    return visible_token_ids;
}

Gemma2ChatTurnResult Gemma2ChatSession::generate_user_turn(std::string_view content) {
    return generate_user_turn(content, Gemma2ChatTurnOptions {});
}

Gemma2ChatTurnResult Gemma2ChatSession::generate_user_turn(
    std::string_view content,
    const Gemma2ChatStreamCallback& on_text_delta
) {
    return generate_user_turn(
        content,
        Gemma2ChatTurnOptions {
            .on_text_delta = on_text_delta,
        }
    );
}

std::vector<std::uint32_t> Gemma2ChatSession::visible_token_prefix_for_text(
    std::span<const std::uint32_t> visible_token_ids,
    std::string_view text
) const {
    std::vector<std::uint32_t> prefix_token_ids;
    prefix_token_ids.reserve(visible_token_ids.size());

    std::string decoded_prefix;
    for (const auto token_id : visible_token_ids) {
        prefix_token_ids.push_back(token_id);
        tokenizer_.append_decoded_token(
            decoded_prefix,
            token_id,
            tokenizer::TokenizerDecodeOptions {.skip_special_tokens = true}
        );
        if (decoded_prefix.size() > text.size() || text.substr(0, decoded_prefix.size()) != decoded_prefix) {
            prefix_token_ids.pop_back();
            break;
        }
    }

    return prefix_token_ids;
}

Gemma2ChatTurnResult Gemma2ChatSession::generate_user_turn(
    std::string_view content,
    const Gemma2ChatTurnOptions& turn_options
) {
    ActiveTurn active(turn_active_);
    static_cast<void>(plan());
    const auto full_turn_started_ns = monotonic_now_ns(clock_);
    if (publication_revision_ == UINT64_MAX) throw std::overflow_error("chat publication epoch overflow");
    std::size_t first_retained_message = 0;
    auto messages = message_views_with_user(content);
    auto rendered_token_ids = encode_messages(std::span<const tokenizer::ChatMessage>(messages));
    const auto context_length = static_cast<std::size_t>(plan().config().context_length);
    const auto exceeds_context = [&](std::size_t prompt_token_count) {
        if (max_new_tokens_ == 0U) {
            return prompt_token_count >= context_length;
        }
        return max_new_tokens_ > context_length
            || prompt_token_count > context_length - max_new_tokens_;
    };
    while (
        context_shift_ && exceeds_context(rendered_token_ids.size())
        && first_retained_message + 2U <= messages_.size()
    ) {
        first_retained_message += 2U;
        messages = message_views_with_user(content, first_retained_message);
        rendered_token_ids = encode_messages(std::span<const tokenizer::ChatMessage>(messages));
    }
    if (exceeds_context(rendered_token_ids.size())) {
        const auto response_budget = max_new_tokens_ == 0U
            ? std::string("unlimited")
            : std::to_string(max_new_tokens_);
        throw Gemma2ChatContextError(
            "turn requires prompt=" + std::to_string(rendered_token_ids.size())
                + " plus response-budget=" + response_budget
                + " tokens but the model limit is " + std::to_string(context_length)
                + "; use a shorter prompt"
                + (max_new_tokens_ == 0U ? "" : " or lower --max-new-tokens")
        );
    }

    const auto trimmed_turn_count = first_retained_message / 2U;
    // Context trimming is staged. Until a callback commits a new prefix, the
    // original conversation/checkpoints remain available for reconciliation.
    // llama.cpp begins prompt timing after tokenization, immediately before
    // cache reconciliation and prompt processing.
    const auto prompt_processing_started_ns = monotonic_now_ns(clock_);
    if (
        first_retained_message == 0U && (cached_token_ids_.size() > rendered_token_ids.size()
        || !std::equal(cached_token_ids_.begin(), cached_token_ids_.end(), rendered_token_ids.begin()))
    ) {
        if (first_retained_message > 0U || !end_of_turn_token_id_.has_value()) {
            throw std::runtime_error("Gemma2 chat session rendered prompt no longer matches cached token prefix");
        }

        // Decoded truncated output can re-tokenize differently from the exact
        // generated IDs already represented by resident KV. Append only the
        // next template turn to those exact IDs instead of replaying history.
        const auto next_turn_messages = message_views_with_user(content, messages_.size());
        auto next_turn_token_ids = encode_messages(
            std::span<const tokenizer::ChatMessage>(next_turn_messages)
        );
        const auto bos_token_id = tokenizer_.special_tokens().bos_token_id;
        if (
            bos_token_id.has_value() && !next_turn_token_ids.empty()
            && next_turn_token_ids.front() == *bos_token_id
        ) {
            next_turn_token_ids.erase(next_turn_token_ids.begin());
        }
        const auto boundary_token_ids = tokenizer_.encode(
            "\n",
            tokenizer::TokenizerEncodeOptions {
                .add_bos = false,
                .add_eos = false,
                .encode_special_tokens = false,
            }
        );
        if (boundary_token_ids.empty()) {
            throw std::runtime_error("Gemma2 chat template boundary did not encode any tokens");
        }
        rendered_token_ids = cached_token_ids_;
        if (rendered_token_ids.empty() || rendered_token_ids.back() != *end_of_turn_token_id_) {
            rendered_token_ids.push_back(*end_of_turn_token_id_);
        }
        rendered_token_ids.insert(
            rendered_token_ids.end(),
            boundary_token_ids.begin(),
            boundary_token_ids.end()
        );
        rendered_token_ids.insert(
            rendered_token_ids.end(),
            next_turn_token_ids.begin(),
            next_turn_token_ids.end()
        );
        if (exceeds_context(rendered_token_ids.size())) {
            throw Gemma2ChatContextError(
                "incremental turn plus response budget exceeds the model context; shorten the prompt"
            );
        }
    }

    const auto cache_position = first_retained_message > 0U ? 0U
        : checked_cached_token_count(cached_token_ids_.size());
    const auto generation_max_new_tokens = max_new_tokens_ == 0U
        ? context_length - rendered_token_ids.size()
        : max_new_tokens_;
    const auto suffix_begin = rendered_token_ids.begin()
        + static_cast<std::ptrdiff_t>(cache_position);
    std::vector<std::uint32_t> appended_token_ids(suffix_begin, rendered_token_ids.end());
    if (appended_token_ids.empty()) {
        throw std::runtime_error("Gemma2 chat session rendered prompt did not append any tokens");
    }
    auto sampler_before_turn = sampler_;
    if (first_retained_message > 0U) {
        sampler_before_turn.reset_history();
    }
    auto working_sampler = sampler_before_turn;
    if (first_retained_message > 0U) {
        working_sampler.reset_history(std::span<const std::uint32_t>(rendered_token_ids));
    } else {
        working_sampler.observe_tokens(std::span<const std::uint32_t>(appended_token_ids));
    }

    std::vector<std::uint32_t> visible_token_ids;
    std::string visible_text;
    std::string final_text;
    std::size_t emitted_text_size = 0;
    std::optional<std::string> reverse_stop_string;
    bool interrupted = false;
    bool callback_threw = false;
    bool published = false;
    std::vector<std::uint32_t> committed_storage(generation_max_new_tokens);
    Gemma2GenerationCommitJournal journal {.token_storage = committed_storage};
    auto next_cached_tokens = rendered_token_ids;
    next_cached_tokens.reserve(rendered_token_ids.size() + generation_max_new_tokens);
    OwnedChatMessage user_message {"user", std::string(content)};
    messages_.reserve(messages_.size() + 2U);
    turn_cache_positions_.reserve(turn_cache_positions_.size() + 1U);
    turn_sampler_states_.reserve(turn_sampler_states_.size() + 1U);
    if (cache_segments_.size() == cache_segments_.capacity()) {
        throw std::overflow_error("chat cache execution receipt capacity exhausted");
    }
    visible_token_ids.reserve(generation_max_new_tokens);
    if (generation_max_new_tokens > std::numeric_limits<std::size_t>::max() / maximum_decoded_token_bytes_) {
        throw std::overflow_error("chat decoded-text capacity overflow");
    }
    const auto text_capacity = generation_max_new_tokens * maximum_decoded_token_bytes_;
    visible_text.reserve(text_capacity);
    final_text.reserve(text_capacity);
    std::string committed_text;
    committed_text.reserve(text_capacity);
    const auto publish_turn = [&](std::string_view text, std::span<const std::uint32_t> tokens,
                                  const SamplerCheckpoint& checkpoint) {
        // Allocate text before mutating durable host state; the remaining
        // journal publication uses already-reserved storage and moved handles.
        committed_text.assign(text);
        OwnedChatMessage assistant {"model", std::move(committed_text)};
        next_cached_tokens.resize(rendered_token_ids.size());
        next_cached_tokens.insert(next_cached_tokens.end(), tokens.begin(), tokens.end());
        working_sampler.adopt_checkpoint(checkpoint);
        if (first_retained_message > 0U) {
            messages_.erase(messages_.begin(), messages_.begin()
                + static_cast<std::ptrdiff_t>(first_retained_message));
            turn_cache_positions_.clear();
            turn_sampler_states_.clear();
            cache_segments_.clear();
        }
        messages_.push_back(std::move(user_message));
        messages_.push_back(std::move(assistant));
        turn_cache_positions_.push_back(cache_position);
        turn_sampler_states_.push_back(sampler_before_turn);
        sampler_ = working_sampler;
        cached_token_ids_.swap(next_cached_tokens);
        cache_segments_.push_back({cache_position,
            checked_cached_token_count(rendered_token_ids.size()),
            checked_cached_token_count(cached_token_ids_.size())});
        ++publication_revision_;
        published = true;
    };
    const auto should_stop = [&]() {
        try { return turn_options.should_stop && turn_options.should_stop(); }
        catch (...) { callback_threw = true; throw; }
    };
    std::optional<std::uint64_t> first_text_ns;
    const Gemma2ChatStreamCallback stream_text_delta = [&](std::string_view text_delta) {
        if (text_delta.empty()) {
            return;
        }
        if (!first_text_ns) first_text_ns = monotonic_now_ns(clock_);
        if (turn_options.on_text_delta) {
            try { turn_options.on_text_delta(text_delta); }
            catch (...) { callback_threw = true; throw; }
        }
    };

    const auto on_generated_token = [&](const Gemma2MetalGeneratedTokenEvent& event) {
            if (should_stop()) {
                interrupted = true;
                return Gemma2MetalTokenStreamDecision::stop_generation;
            }

            if (tokenizer_.token_id_is_special(event.token_id)) {
                return Gemma2MetalTokenStreamDecision::continue_generation;
            }

            visible_token_ids.push_back(event.token_id);
            const auto previous_text_size = visible_text.size();
            tokenizer_.append_decoded_token(
                visible_text,
                event.token_id,
                tokenizer::TokenizerDecodeOptions {.skip_special_tokens = true}
            );

            const auto max_stop_length = max_reverse_stop_string_length(
                std::span<const std::string>(reverse_stop_strings_)
            );
            const auto boundary_overlap = max_stop_length == 0U ? 0U : max_stop_length - 1U;
            const auto reverse_stop_search_offset = previous_text_size > boundary_overlap
                ? previous_text_size - boundary_overlap : 0U;
            const auto stop_match = first_reverse_stop_match(
                visible_text,
                std::span<const std::string>(reverse_stop_strings_),
                reverse_stop_search_offset
            );
            if (stop_match.has_value()) {
                final_text = visible_text.substr(0, stop_match->first);
                reverse_stop_string = stop_match->second;
                emit_text_delta(stream_text_delta, final_text, emitted_text_size, final_text.size());
                return Gemma2MetalTokenStreamDecision::stop_generation;
            }

            emit_text_delta(
                stream_text_delta,
                visible_text,
                emitted_text_size,
                safe_stream_text_size(visible_text, std::span<const std::string>(reverse_stop_strings_))
            );
            if (should_stop()) {
                interrupted = true;
                return Gemma2MetalTokenStreamDecision::stop_generation;
            }
            return Gemma2MetalTokenStreamDecision::continue_generation;
    };
    const auto run_generation = [&]() {
        if (speculative_decode_)
            proposer_.rebuild(std::span<const std::uint32_t>(rendered_token_ids));
        return generate_gemma2_metal_sampled_tokens_with_plan(
            plan(),
            std::span<const std::uint32_t>(appended_token_ids),
            cache_position,
            generation_max_new_tokens,
            std::span<const std::uint32_t>(stop_token_ids_),
            working_sampler,
            on_generated_token,
            clock_,
            speculative_decode_ ? &proposer_ : nullptr,
            turn_options.trace,
            &journal,
            prepared_->standalone(),
            rendered_token_ids
        );
    };
    try {
    auto generation = run_generation();

    if (!reverse_stop_string.has_value()) {
        if (visible_token_ids.empty()) {
            visible_token_ids = visible_assistant_token_ids(
                std::span<const std::uint32_t>(generation.generated_token_ids)
            );
            visible_text = tokenizer_.decode(
                std::span<const std::uint32_t>(visible_token_ids),
                tokenizer::TokenizerDecodeOptions {.skip_special_tokens = true}
            );
        }
        final_text = visible_text;
        emit_text_delta(stream_text_delta, final_text, emitted_text_size, final_text.size());
    }
    auto cached_visible_token_ids = reverse_stop_string.has_value()
        ? visible_token_prefix_for_text(std::span<const std::uint32_t>(visible_token_ids), final_text)
        : visible_token_ids;

    // Every selected token is resident, including EOS and reverse-stop-hidden
    // tokens. Never infer cache identity by decoding/re-tokenizing display text.
    publish_turn(final_text, generation.selected_token_ids, working_sampler.checkpoint());

    auto generation_metrics = generation.metrics;
    if (interrupted) {
        generation_metrics.stop_reason = Gemma2GenerationStopReason::interrupted;
    } else if (reverse_stop_string.has_value()) {
        generation_metrics.stop_reason = Gemma2GenerationStopReason::reverse_stop;
    }
    const auto prompt_processing_duration_ns = elapsed_ns(
        prompt_processing_started_ns, generation_metrics.first_token_selected_ns
    );
    // llama.cpp counts every sampled token and measures from the first sample
    // through the final sample. It clamps the zero-length one-token case to 1 us.
    const auto generation_duration_ns = std::max<std::uint64_t>(
        1'000U,
        elapsed_ns(
            generation_metrics.first_token_selected_ns,
            generation_metrics.last_token_selected_ns
        )
    );
    auto result = Gemma2ChatTurnResult {
        .text = std::move(final_text),
        .cache_position = cache_position,
        .appended_token_count = checked_cached_token_count(appended_token_ids.size()),
        .cached_token_count = checked_cached_token_count(cached_token_ids_.size()),
        .generated_token_ids = cached_visible_token_ids,
        .rendered_prompt_token_ids = std::move(rendered_token_ids),
        .evaluated_suffix_token_ids = std::move(appended_token_ids),
        .selected_token_ids = std::move(generation.selected_token_ids),
        .visible_token_ids = std::move(cached_visible_token_ids),
        .stop_token_id = generation.stop_token_id,
        .reverse_stop_string = std::move(reverse_stop_string),
        .interrupted = interrupted,
        .trimmed_turn_count = checked_cached_token_count(trimmed_turn_count),
        .rebuilt_context = first_retained_message > 0U,
        .metrics = Gemma2ChatTurnMetrics {
            .generation = generation_metrics,
            .prompt_processing_duration_ns = prompt_processing_duration_ns,
            .generation_duration_ns = generation_duration_ns,
            .learned_owned_byte_count = prepared_->standalone() ? prepared_->standalone()->owned_bytes() : 0U,
            .learned_horizon = prepared_->standalone() ? prepared_->standalone()->horizon() : 0U,
            .standalone_draft = prepared_->standalone() != nullptr,
            .standalone_bounded_repetition_history = prepared_->standalone()
                && prepared_->standalone()->direct_ids(),
            .standalone_kind = prepared_->standalone() ? prepared_->standalone()->kind()
                : StandaloneDraftKind::gemma3_270m_q8,
            .standalone_execution = standalone_execution_,
            .standalone_prefill_ns = generation.metrics.standalone_prefill_ns,
            .standalone_rebuild_ns = generation.metrics.standalone_rebuild_ns,
            .standalone_effective_sampler = prepared_->standalone()
                ? std::optional(prepared_->standalone()->effective_draft_sampler_options(sampler_before_turn.options()))
                : std::nullopt,
            .speculative_sampling = speculative_sampling_,
            .initial_cache_position = cache_position,
            .reused_cached_token_count = cache_position,
            .selection_mode = sampler_before_turn.is_greedy()
                ? Gemma2TokenSelectionMode::greedy
                : Gemma2TokenSelectionMode::sampled,
            .trimmed_turn_count = checked_cached_token_count(trimmed_turn_count),
            .rebuilt_context = first_retained_message > 0U,
            .output_head_policy = plan().output_head_policy(),
            .model_architecture = plan().config().architecture,
            .context_capacity = plan().config().context_length,
            .kv_precision = plan().config().kv_precision,
            .retrieval_decode = speculative_decode_,
            .resident_weight_byte_count = plan().uploaded_weight_byte_count(),
            .resident_kv_cache_byte_count = plan().kv_cache_byte_count(),
            .resident_scratch_byte_count = plan().scratch_byte_count(),
            .warmup_kv_clear_pass_count = warmup_stats_->kv_clear_pass_count,
            .warmup_host_cleared_kv_bytes = warmup_stats_->host_cleared_kv_bytes,
            .decode_scheduled_weight_bytes = plan().decode_scheduled_weight_bytes(),
            .prefill_execution = generation.prefill_execution,
            .decode_execution = generation.decode_execution,
        },
    };
    const auto completed_ns = monotonic_now_ns(clock_);
    result.metrics.generation_completion_duration_ns = elapsed_ns(generation_metrics.prefill_completed_ns, completed_ns);
    result.metrics.full_turn_duration_ns = elapsed_ns(full_turn_started_ns, completed_ns);
    result.metrics.first_text_duration_ns = first_text_ns
        ? elapsed_ns(full_turn_started_ns, *first_text_ns) : 0U;
    return result;
    } catch (...) {
        const auto original = std::current_exception();
        try {
            if (!published && journal.count > 0U) {
                publish_turn(reverse_stop_string ? final_text : visible_text,
                    std::span<const std::uint32_t>(committed_storage).first(journal.count),
                    journal.sampler);
            }
            if (journal.completion.completion_unknown || poisoned()
                || plan().has_pending_verification()) {
                prepared_->poison();
            } else if (!callback_threw) {
                // Known completion: rebuild exactly the durable receipt (or
                // the untouched pre-turn prefix). This is error recovery only,
                // never a speculative performance path or token reselection.
                rebuild_durable_cache();
            }
        } catch (...) { if (prepared_) prepared_->poison(); }
        std::rethrow_exception(original);
    }
}

Gemma2ChatTurnResult Gemma2ChatSession::regenerate_last_turn() {
    return regenerate_last_turn(Gemma2ChatTurnOptions {});
}

Gemma2ChatTurnResult Gemma2ChatSession::regenerate_last_turn(
    const Gemma2ChatTurnOptions& turn_options
) {
    if (turn_active_ || poisoned()) throw std::logic_error("cannot regenerate in this chat state");
    const auto started = monotonic_now_ns(clock_);
    if (
        messages_.size() < 2U || turn_cache_positions_.empty()
        || messages_[messages_.size() - 2U].role != "user"
        || messages_.back().role != "model"
    ) {
        throw std::runtime_error("Gemma2 chat session has no assistant turn to regenerate");
    }

    const auto cache_position = turn_cache_positions_.back();
    std::vector<std::uint32_t> prefix(cached_token_ids_.begin(),
        cached_token_ids_.begin() + static_cast<std::ptrdiff_t>(cache_position));
    auto* standalone = prepared_->standalone();
    auto draft_checkpoint = standalone ? standalone->checkpoint() : StandaloneChatCheckpoint {};
    auto previous_cache = std::move(cached_token_ids_);
    auto previous_user = std::move(messages_[messages_.size() - 2U]);
    auto previous_assistant = std::move(messages_.back());
    const auto previous_sampler = sampler_;
    const auto turn_start_sampler = turn_sampler_states_.back();
    const auto revision = publication_revision_;
    const auto previous_segment = cache_segments_.back();
    cache_segments_.pop_back();
    messages_.resize(messages_.size() - 2U);
    turn_cache_positions_.pop_back();
    sampler_ = turn_sampler_states_.back();
    turn_sampler_states_.pop_back();
    cached_token_ids_ = std::move(prefix);
    try {
        // Rewinding a wrapped sliding ring cannot reuse rows overwritten by
        // the removed turn. Reconstruct the retained execution, without RNG.
        if (cache_position > 0U && previous_cache.size() > model_.gemma2_config().sliding_window) {
            try { rebuild_durable_cache(); }
            catch (...) { prepared_->poison(); throw; }
        }
        auto result = generate_user_turn(previous_user.content, turn_options);
        if (standalone) standalone->discard(draft_checkpoint);
        result.metrics.full_turn_duration_ns = elapsed_ns(started, monotonic_now_ns(clock_));
        return result;
    } catch (...) {
        if (publication_revision_ == revision) {
            messages_.push_back(std::move(previous_user));
            messages_.push_back(std::move(previous_assistant));
            turn_cache_positions_.push_back(cache_position);
            turn_sampler_states_.push_back(turn_start_sampler);
            sampler_ = previous_sampler;
            cached_token_ids_ = std::move(previous_cache);
            cache_segments_.push_back(previous_segment);
            if (!poisoned()) {
                try {
                    // Restore the original emission ledger before target
                    // reconstruction. Retokenizing a generated suffix is not
                    // equivalent to its original mapped/correction appends.
                    if (standalone) standalone->restore(draft_checkpoint);
                    rebuild_durable_cache();
                    if (!standalone) prepared_->reset_draft_state();
                } catch (...) { prepared_->poison(); }
            }
        }
        throw;
    }
}

void Gemma2ChatSession::clear() {
    if (turn_active_) throw std::logic_error("cannot clear an active chat turn");
    if (poisoned()) throw std::logic_error("poisoned chat requires reset");
    prepared_->reset_draft_state();
    messages_.clear();
    cached_token_ids_.clear();
    cache_segments_.clear();
    turn_cache_positions_.clear();
    turn_sampler_states_.clear();
    sampler_ = TokenSampler(initial_sampler_options_);
}

void Gemma2ChatSession::reset() {
    if (turn_active_) throw std::logic_error("cannot reset an active chat turn");
    // Release the old residency before allocating its replacement. The opened
    // weight source survives; a replaced pathname cannot silently change it.
    prepared_.reset();
    prepared_.emplace(standalone_source_
        ? prepare_gemma2_standalone_plan(model_, tokenizer_, standalone_source_, standalone_execution_,
            output_head_policy_, context_length_, kv_precision_)
        : prepare_gemma2_target_only_plan(model_, output_head_policy_, context_length_, kv_precision_));
    clear();
    try { warm_up_plan(); }
    catch (...) { prepared_->poison(); throw; }
    ++plan_revision_;
}

void Gemma2ChatSession::set_sampler_options(TokenSamplerOptions options) {
    if (turn_active_ || poisoned()) throw std::logic_error("chat sampler cannot change in this state");
    static_cast<void>(TokenSampler(options));
    initial_sampler_options_ = std::move(options);
    clear();
}

void Gemma2ChatSession::set_max_new_tokens(std::size_t max_new_tokens) {
    if (turn_active_) throw std::logic_error("cannot change token budget during a chat turn");
    max_new_tokens_ = max_new_tokens;
}

std::uint32_t Gemma2ChatSession::cached_token_count() const noexcept {
    return static_cast<std::uint32_t>(cached_token_ids_.size());
}

std::uint32_t Gemma2ChatSession::context_length() const noexcept {
    return context_length_;
}

std::uint32_t Gemma2ChatSession::remaining_context_tokens() const noexcept {
    return context_length() - cached_token_count();
}

std::size_t Gemma2ChatSession::max_new_tokens() const noexcept {
    return max_new_tokens_;
}

std::size_t Gemma2ChatSession::message_count() const noexcept {
    return messages_.size();
}

std::size_t Gemma2ChatSession::plan_revision() const noexcept {
    return plan_revision_;
}

std::span<const std::uint32_t> Gemma2ChatSession::stop_token_ids() const noexcept {
    return stop_token_ids_;
}

std::span<const std::string> Gemma2ChatSession::reverse_stop_strings() const noexcept {
    return reverse_stop_strings_;
}

std::array<std::uintptr_t, 4> Gemma2ChatSession::plan_allocation_identities() const noexcept {
    return poisoned() ? std::array<std::uintptr_t, 4>{} : plan().allocation_identities();
}

const std::optional<Gemma2MetalWarmupStats>& Gemma2ChatSession::warmup_stats() const noexcept {
    return warmup_stats_;
}

}  // namespace local_gguf_engine::engine
