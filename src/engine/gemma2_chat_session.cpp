#include "local_gguf_engine/engine/gemma2_chat_session.hpp"

#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include <algorithm>
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
        throw std::runtime_error("Gemma2 measurement clock moved backwards");
    }
    return end - start;
}

[[nodiscard]] Gemma2MetalModelPlan create_chat_plan(
    const EngineModel& model,
    const Gemma2ChatSessionOptions& options
) {
    if (options.backend != runtime::BackendKind::metal) {
        throw std::runtime_error("Gemma 2 chat session requires Metal backend");
    }
    if (model.architecture() != ModelArchitecture::gemma2) {
        throw std::runtime_error(
            "Gemma 2 chat session supports Gemma 2 only; architecture '"
                + std::string(model_architecture_name(model.architecture()))
                + "' is unsupported"
        );
    }

    return create_gemma2_metal_model_plan(model);
}

[[nodiscard]] Gemma2ChatSessionState create_chat_session_state(
    model_format::GGUFFileBackedModel file_model,
    const Gemma2ChatSessionOptions& options
) {
    auto model = load_engine_model_from_gguf(file_model.view());
    auto tokenizer = tokenizer::load_tokenizer_from_gguf_metadata(file_model.view().directory.metadata);
    auto plan = create_chat_plan(model, options);
    return Gemma2ChatSessionState {
        .file_model = std::move(file_model),
        .model = std::move(model),
        .tokenizer = std::move(tokenizer),
        .plan = std::move(plan),
    };
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
    std::span<const std::string> reverse_stop_strings
) {
    std::optional<std::pair<std::size_t, std::string>> first_match;
    for (const auto& stop_string : reverse_stop_strings) {
        if (stop_string.empty()) {
            continue;
        }

        const auto position = text.find(stop_string);
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

Gemma2ChatSession::Gemma2ChatSession(
    model_format::GGUFFileBackedModel file_model,
    Gemma2ChatSessionOptions options
)
    : Gemma2ChatSession(create_chat_session_state(std::move(file_model), options), options) {}

Gemma2ChatSession::Gemma2ChatSession(
    Gemma2ChatSessionState state,
    Gemma2ChatSessionOptions options
)
    : file_model_(std::move(state.file_model)),
      model_(std::move(state.model)),
      tokenizer_(std::move(state.tokenizer)),
      plan_(std::move(state.plan)),
      reverse_stop_strings_(std::move(options.reverse_stop_strings)),
      end_of_turn_token_id_(find_token_id_by_text(tokenizer_, "<end_of_turn>")),
      max_new_tokens_(options.max_new_tokens),
      context_shift_(options.context_shift),
      initial_sampler_options_(options.sampler),
      sampler_(initial_sampler_options_),
      clock_(std::move(options.clock)) {
    if (options.backend != runtime::BackendKind::metal || plan_.backend() != runtime::BackendKind::metal) {
        throw std::runtime_error("Gemma 2 chat session requires Metal backend");
    }
    if (model_.architecture() != ModelArchitecture::gemma2) {
        throw std::runtime_error("Gemma 2 chat session prepared state requires Gemma 2 model");
    }
    stop_token_ids_ = chat_stop_token_ids(tokenizer_, end_of_turn_token_id_);
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
        decoded_prefix = tokenizer_.decode(
            std::span<const std::uint32_t>(prefix_token_ids),
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
    const auto turn_accepted_ns = monotonic_now_ns(clock_);
    std::size_t first_retained_message = 0;
    auto messages = message_views_with_user(content);
    auto rendered_token_ids = encode_messages(std::span<const tokenizer::ChatMessage>(messages));
    const auto context_length = static_cast<std::size_t>(plan_.config().context_length);
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
    if (first_retained_message > 0U) {
        // Overflow rebuilds only the retained live history. The model plan and
        // uploaded weights stay resident; prefill overwrites KV from position 0.
        messages_.erase(
            messages_.begin(),
            messages_.begin() + static_cast<std::ptrdiff_t>(first_retained_message)
        );
        cached_token_ids_.clear();
        turn_cache_positions_.clear();
        turn_sampler_states_.clear();
    }
    if (
        cached_token_ids_.size() > rendered_token_ids.size()
        || !std::equal(cached_token_ids_.begin(), cached_token_ids_.end(), rendered_token_ids.begin())
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

    const auto cache_position = checked_cached_token_count(cached_token_ids_.size());
    const auto generation_max_new_tokens = max_new_tokens_ == 0U
        ? context_length - rendered_token_ids.size()
        : max_new_tokens_;
    const auto suffix_begin = rendered_token_ids.begin()
        + static_cast<std::ptrdiff_t>(cached_token_ids_.size());
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
    std::optional<std::uint64_t> first_visible_ready_ns;
    std::optional<std::uint64_t> first_visible_callback_completed_ns;
    std::optional<std::uint64_t> last_visible_callback_completed_ns;
    std::uint32_t visible_emission_count = 0;
    const Gemma2ChatStreamCallback measured_text_delta = [&](std::string_view text_delta) {
        if (text_delta.empty()) {
            return;
        }
        if (!first_visible_ready_ns.has_value()) {
            first_visible_ready_ns = monotonic_now_ns(clock_);
        }
        if (turn_options.on_text_delta) {
            turn_options.on_text_delta(text_delta);
        }
        const auto callback_completed_ns = monotonic_now_ns(clock_);
        if (!first_visible_callback_completed_ns.has_value()) {
            first_visible_callback_completed_ns = callback_completed_ns;
        }
        last_visible_callback_completed_ns = callback_completed_ns;
        ++visible_emission_count;
    };

    const auto on_generated_token = [&](const Gemma2MetalGeneratedTokenEvent& event) {
            if (turn_options.should_stop && turn_options.should_stop()) {
                interrupted = true;
                return Gemma2MetalTokenStreamDecision::stop_generation;
            }

            if (tokenizer_.token_id_is_special(event.token_id)) {
                return Gemma2MetalTokenStreamDecision::continue_generation;
            }

            visible_token_ids.push_back(event.token_id);
            visible_text = tokenizer_.decode(
                std::span<const std::uint32_t>(visible_token_ids),
                tokenizer::TokenizerDecodeOptions {.skip_special_tokens = true}
            );

            const auto stop_match = first_reverse_stop_match(
                visible_text,
                std::span<const std::string>(reverse_stop_strings_)
            );
            if (stop_match.has_value()) {
                final_text = visible_text.substr(0, stop_match->first);
                reverse_stop_string = stop_match->second;
                emit_text_delta(measured_text_delta, final_text, emitted_text_size, final_text.size());
                return Gemma2MetalTokenStreamDecision::stop_generation;
            }

            emit_text_delta(
                measured_text_delta,
                visible_text,
                emitted_text_size,
                safe_stream_text_size(visible_text, std::span<const std::string>(reverse_stop_strings_))
            );
            if (turn_options.should_stop && turn_options.should_stop()) {
                interrupted = true;
                return Gemma2MetalTokenStreamDecision::stop_generation;
            }
            return Gemma2MetalTokenStreamDecision::continue_generation;
    };
    const auto run_generation = [&]() {
        return generate_gemma2_metal_sampled_tokens_with_plan(
            plan_,
            std::span<const std::uint32_t>(appended_token_ids),
            cache_position,
            generation_max_new_tokens,
            std::span<const std::uint32_t>(stop_token_ids_),
            working_sampler,
            on_generated_token,
            clock_
        );
    };
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
        emit_text_delta(measured_text_delta, final_text, emitted_text_size, final_text.size());
    }
    auto cached_visible_token_ids = reverse_stop_string.has_value()
        ? visible_token_prefix_for_text(std::span<const std::uint32_t>(visible_token_ids), final_text)
        : visible_token_ids;

    messages_.push_back(OwnedChatMessage {
        .role = "user",
        .content = std::string(content),
    });
    messages_.push_back(OwnedChatMessage {
        .role = "model",
        .content = final_text,
    });
    turn_cache_positions_.push_back(cache_position);
    turn_sampler_states_.push_back(sampler_before_turn);
    sampler_ = std::move(working_sampler);

    // KV contains every selected token, including non-visible control tokens
    // and tokens hidden by reverse-stop handling. Cache identity must follow
    // those exact IDs rather than a decoded/re-tokenized display string.
    cached_token_ids_ = rendered_token_ids;
    cached_token_ids_.insert(
        cached_token_ids_.end(),
        generation.generated_token_ids.begin(),
        generation.generated_token_ids.end()
    );
    if (
        generation.stop_token_id.has_value()
        && end_of_turn_token_id_.has_value()
        && *generation.stop_token_id == *end_of_turn_token_id_
    ) {
        cached_token_ids_.push_back(*generation.stop_token_id);
    }

    auto generation_metrics = generation.metrics;
    if (interrupted) {
        generation_metrics.stop_reason = Gemma2GenerationStopReason::interrupted;
    } else if (reverse_stop_string.has_value()) {
        generation_metrics.stop_reason = Gemma2GenerationStopReason::reverse_stop;
    }
    const auto turn_completed_ns = monotonic_now_ns(clock_);
    const auto text_ready_ttft_ns = first_visible_ready_ns.has_value()
        ? std::optional<std::uint64_t>(elapsed_ns(turn_accepted_ns, *first_visible_ready_ns))
        : std::nullopt;
    const auto frontend_complete_ttft_ns = first_visible_callback_completed_ns.has_value()
        ? std::optional<std::uint64_t>(elapsed_ns(turn_accepted_ns, *first_visible_callback_completed_ns))
        : std::nullopt;
    const auto visible_stream_interval_ns =
        first_visible_callback_completed_ns.has_value()
        && last_visible_callback_completed_ns.has_value()
        && visible_emission_count >= 2U
        ? std::optional<std::uint64_t>(elapsed_ns(
            *first_visible_callback_completed_ns, *last_visible_callback_completed_ns
        ))
        : std::nullopt;

    return Gemma2ChatTurnResult {
        .text = final_text,
        .cache_position = cache_position,
        .appended_token_count = checked_cached_token_count(appended_token_ids.size()),
        .cached_token_count = checked_cached_token_count(cached_token_ids_.size()),
        .generated_token_ids = cached_visible_token_ids,
        .rendered_prompt_token_ids = std::move(rendered_token_ids),
        .evaluated_suffix_token_ids = appended_token_ids,
        .selected_token_ids = std::move(generation.selected_token_ids),
        .visible_token_ids = std::move(cached_visible_token_ids),
        .stop_token_id = generation.stop_token_id,
        .reverse_stop_string = std::move(reverse_stop_string),
        .interrupted = interrupted,
        .trimmed_turn_count = checked_cached_token_count(trimmed_turn_count),
        .rebuilt_context = first_retained_message > 0U,
        .metrics = Gemma2ChatTurnMetrics {
            .generation = generation_metrics,
            .turn_accepted_ns = turn_accepted_ns,
            .first_visible_ready_ns = first_visible_ready_ns,
            .first_visible_callback_completed_ns = first_visible_callback_completed_ns,
            .last_visible_callback_completed_ns = last_visible_callback_completed_ns,
            .turn_completed_ns = turn_completed_ns,
            .text_ready_ttft_ns = text_ready_ttft_ns,
            .frontend_complete_ttft_ns = frontend_complete_ttft_ns,
            .visible_stream_interval_ns = visible_stream_interval_ns,
            .visible_emission_count = visible_emission_count,
            .total_turn_duration_ns = elapsed_ns(turn_accepted_ns, turn_completed_ns),
            .initial_cache_position = cache_position,
            .reused_cached_token_count = cache_position,
            .selection_mode = sampler_before_turn.is_greedy()
                ? Gemma2TokenSelectionMode::greedy
                : Gemma2TokenSelectionMode::sampled,
            .trimmed_turn_count = checked_cached_token_count(trimmed_turn_count),
            .rebuilt_context = first_retained_message > 0U,
            .resident_weight_byte_count = plan_.uploaded_weight_byte_count(),
            .resident_kv_cache_byte_count = plan_.kv_cache_byte_count(),
            .resident_scratch_byte_count = plan_.scratch_byte_count(),
            .prefill_execution = generation.prefill_execution,
            .decode_execution = generation.decode_execution,
        },
    };
}

Gemma2ChatTurnResult Gemma2ChatSession::regenerate_last_turn() {
    return regenerate_last_turn(Gemma2ChatTurnOptions {});
}

Gemma2ChatTurnResult Gemma2ChatSession::regenerate_last_turn(
    const Gemma2ChatTurnOptions& turn_options
) {
    if (
        messages_.size() < 2U || turn_cache_positions_.empty()
        || messages_[messages_.size() - 2U].role != "user"
        || messages_.back().role != "model"
    ) {
        throw std::runtime_error("Gemma2 chat session has no assistant turn to regenerate");
    }

    auto user_content = std::move(messages_[messages_.size() - 2U].content);
    const auto cache_position = turn_cache_positions_.back();
    messages_.resize(messages_.size() - 2U);
    turn_cache_positions_.pop_back();
    sampler_ = turn_sampler_states_.back();
    turn_sampler_states_.pop_back();
    cached_token_ids_.resize(cache_position);
    return generate_user_turn(user_content, turn_options);
}

void Gemma2ChatSession::clear() {
    messages_.clear();
    cached_token_ids_.clear();
    turn_cache_positions_.clear();
    turn_sampler_states_.clear();
    sampler_ = TokenSampler(initial_sampler_options_);
}

void Gemma2ChatSession::reset() {
    plan_ = create_gemma2_metal_model_plan(model_);
    ++plan_revision_;
    clear();
}

void Gemma2ChatSession::set_sampler_options(TokenSamplerOptions options) {
    initial_sampler_options_ = std::move(options);
    clear();
}

void Gemma2ChatSession::set_max_new_tokens(std::size_t max_new_tokens) noexcept {
    max_new_tokens_ = max_new_tokens;
}

std::uint32_t Gemma2ChatSession::cached_token_count() const noexcept {
    return static_cast<std::uint32_t>(cached_token_ids_.size());
}

std::uint32_t Gemma2ChatSession::context_length() const noexcept {
    return plan_.config().context_length;
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
    return plan_.allocation_identities();
}

}  // namespace local_gguf_engine::engine
