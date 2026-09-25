#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/engine/retrieval_proposer.hpp"
#include "local_gguf_engine/engine/prepared_gemma2_plan.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::engine {

class Gemma2ChatContextError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Gemma2ChatSessionOptions {
    // Zero means generate until a stop token, cancellation, or context exhaustion.
    std::size_t max_new_tokens = 0;
    runtime::BackendKind backend = runtime::BackendKind::metal;
    Gemma2OutputHeadPolicy output_head_policy = Gemma2OutputHeadPolicy::automatic;
    bool context_shift = false;
    bool speculative_decode = true;
    bool retrieval_decode = true;
    // Explicit weights-only experiment, not a selected or admitted policy.
    std::optional<std::filesystem::path> development_recurrent_weights;
    std::optional<std::filesystem::path> standalone_draft_model;
    StandaloneExecutionOptions standalone_execution;
    SpeculativeSamplingMode speculative_sampling = SpeculativeSamplingMode::exact_match;
    RecurrentExecutionOptions recurrent_execution;
    std::vector<std::string> reverse_stop_strings;
    TokenSamplerOptions sampler;
    Gemma2MonotonicClock clock;
};

enum class Gemma2TokenSelectionMode {
    greedy,
    sampled,
};

struct Gemma2ChatTurnMetrics {
    Gemma2GenerationMetrics generation;
    // Matches llama.cpp result_timings: tokenized prompt processing through
    // first selection, then first selection through final selection.
    std::uint64_t prompt_processing_duration_ns = 0;
    std::uint64_t generation_duration_ns = 0;
    std::uint64_t generation_completion_duration_ns = 0;
    std::uint64_t full_turn_duration_ns = 0;
    // User-turn entry through the first nonempty decoded text delivery. Zero
    // means no visible text. Kept separate from legacy first-selection TTFT.
    std::uint64_t first_text_duration_ns = 0;
    std::size_t learned_owned_byte_count = 0;
    std::uint32_t learned_horizon = 0;
    bool standalone_draft = false;
    StandaloneDraftKind standalone_kind = StandaloneDraftKind::vicuna_68m_f16;
    StandaloneExecutionOptions standalone_execution;
    std::uint64_t standalone_prefill_ns = 0;
    std::uint64_t standalone_rebuild_ns = 0;
    // Resolved against this turn's target scalars and native draft vocabulary.
    // The target sampler and its history/RNG are not changed by draft overrides.
    std::optional<TokenSamplerOptions> standalone_effective_sampler;
    SpeculativeSamplingMode speculative_sampling = SpeculativeSamplingMode::exact_match;
    RecurrentExecutionOptions recurrent_execution;
    std::uint32_t initial_cache_position = 0;
    std::uint32_t reused_cached_token_count = 0;
    Gemma2TokenSelectionMode selection_mode = Gemma2TokenSelectionMode::greedy;
    std::uint32_t trimmed_turn_count = 0;
    bool rebuilt_context = false;
    Gemma2OutputHeadPolicy output_head_policy = Gemma2OutputHeadPolicy::f16;
    std::size_t resident_weight_byte_count = 0;
    std::size_t resident_kv_cache_byte_count = 0;
    std::size_t resident_scratch_byte_count = 0;
    std::uint32_t warmup_kv_clear_pass_count = 0;
    std::size_t warmup_host_cleared_kv_bytes = 0;
    Gemma2DecodeScheduledWeightBytes decode_scheduled_weight_bytes;
    Gemma2MetalPrefillExecutionStats prefill_execution;
    Gemma2MetalDecodeExecutionStats decode_execution;
};

struct Gemma2ChatTurnResult {
    std::string text;
    std::uint32_t cache_position = 0;
    std::uint32_t appended_token_count = 0;
    std::uint32_t cached_token_count = 0;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<std::uint32_t> rendered_prompt_token_ids;
    std::vector<std::uint32_t> evaluated_suffix_token_ids;
    std::vector<std::uint32_t> selected_token_ids;
    std::vector<std::uint32_t> visible_token_ids;
    std::optional<std::uint32_t> stop_token_id;
    std::optional<std::string> reverse_stop_string;
    bool interrupted = false;
    std::uint32_t trimmed_turn_count = 0;
    bool rebuilt_context = false;
    Gemma2ChatTurnMetrics metrics;
};

using Gemma2ChatStreamCallback = std::function<void(std::string_view text_delta)>;
using Gemma2ChatShouldStopCallback = std::function<bool()>;

struct Gemma2ChatTurnOptions {
    Gemma2ChatStreamCallback on_text_delta;
    Gemma2ChatShouldStopCallback should_stop;
    // Synchronous developer/offline capture binding. The caller retains the
    // binding and its sinks through generate_user_turn().
    const Gemma2GenerationTraceBinding* trace = nullptr;
};

class Gemma2ChatSessionState final {
public:
    [[nodiscard]] static Gemma2ChatSessionState prepare(
        model_format::GGUFFileBackedModel file_model,
        const Gemma2ChatSessionOptions& options, bool authenticate_target = false
    );
    Gemma2ChatSessionState(Gemma2ChatSessionState&&) noexcept = default;
    Gemma2ChatSessionState& operator=(Gemma2ChatSessionState&&) noexcept = default;
    [[nodiscard]] const EngineModel& model() const noexcept { return model_; }
    [[nodiscard]] const tokenizer::Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    [[nodiscard]] const PreparedGemma2Plan& prepared() const noexcept { return prepared_; }
private:
    friend class Gemma2ChatSession;
    Gemma2ChatSessionState(model_format::GGUFFileBackedModel, EngineModel,
        tokenizer::Tokenizer, PreparedGemma2Plan, std::shared_ptr<RecurrentWeightsFile>,
        std::optional<std::filesystem::path>, std::shared_ptr<StandaloneDraftArtifact>,
        std::optional<std::filesystem::path>);
    model_format::GGUFFileBackedModel file_model_;
    EngineModel model_;
    tokenizer::Tokenizer tokenizer_;
    PreparedGemma2Plan prepared_;
    std::shared_ptr<RecurrentWeightsFile> recurrent_source_;
    std::optional<std::filesystem::path> requested_weights_;
    std::shared_ptr<StandaloneDraftArtifact> standalone_source_;
    std::optional<std::filesystem::path> requested_standalone_;
};

class Gemma2ChatSession final {
public:
    Gemma2ChatSession(
        model_format::GGUFFileBackedModel file_model,
        Gemma2ChatSessionOptions options
    );
    Gemma2ChatSession(
        Gemma2ChatSessionState state,
        Gemma2ChatSessionOptions options
    );

    [[nodiscard]] Gemma2ChatTurnResult generate_user_turn(std::string_view content);
    [[nodiscard]] Gemma2ChatTurnResult generate_user_turn(
        std::string_view content,
        const Gemma2ChatStreamCallback& on_text_delta
    );
    [[nodiscard]] Gemma2ChatTurnResult generate_user_turn(
        std::string_view content,
        const Gemma2ChatTurnOptions& turn_options
    );
    [[nodiscard]] Gemma2ChatTurnResult regenerate_last_turn();
    [[nodiscard]] Gemma2ChatTurnResult regenerate_last_turn(
        const Gemma2ChatTurnOptions& turn_options
    );
    void clear();
    void reset();
    void set_sampler_options(TokenSamplerOptions options);
    void set_max_new_tokens(std::size_t max_new_tokens);
    [[nodiscard]] std::uint32_t cached_token_count() const noexcept;
    [[nodiscard]] std::uint32_t context_length() const noexcept;
    [[nodiscard]] std::uint32_t remaining_context_tokens() const noexcept;
    [[nodiscard]] std::size_t max_new_tokens() const noexcept;
    [[nodiscard]] std::size_t message_count() const noexcept;
    [[nodiscard]] std::size_t plan_revision() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> stop_token_ids() const noexcept;
    [[nodiscard]] std::span<const std::string> reverse_stop_strings() const noexcept;
    [[nodiscard]] std::array<std::uintptr_t, 4> plan_allocation_identities() const noexcept;
    [[nodiscard]] const std::optional<Gemma2MetalWarmupStats>& warmup_stats() const noexcept;
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    [[nodiscard]] const TokenSampler& sampler_for_testing() const noexcept { return sampler_; }
    [[nodiscard]] const Gemma2MetalModelPlan& plan_for_testing() const { return plan(); }
#endif

private:
    struct OwnedChatMessage {
        std::string role;
        std::string content;
    };

    [[nodiscard]] std::vector<tokenizer::ChatMessage> message_views_with_user(
        std::string_view content,
        std::size_t first_message_index = 0
    ) const;
    [[nodiscard]] std::vector<std::uint32_t> encode_messages(
        std::span<const tokenizer::ChatMessage> messages
    ) const;
    [[nodiscard]] std::vector<std::uint32_t> visible_assistant_token_ids(
        std::span<const std::uint32_t> generated_token_ids
    ) const;
    [[nodiscard]] std::vector<std::uint32_t> visible_token_prefix_for_text(
        std::span<const std::uint32_t> visible_token_ids,
        std::string_view text
    ) const;
    void warm_up_plan();
    void rebuild_durable_cache();
    [[nodiscard]] Gemma2MetalModelPlan& plan();
    [[nodiscard]] const Gemma2MetalModelPlan& plan() const;

    model_format::GGUFFileBackedModel file_model_;
    EngineModel model_;
    tokenizer::Tokenizer tokenizer_;
    std::optional<PreparedGemma2Plan> prepared_;
    std::shared_ptr<RecurrentWeightsFile> recurrent_source_;
    std::shared_ptr<StandaloneDraftArtifact> standalone_source_;
    Gemma2OutputHeadPolicy output_head_policy_;
    std::vector<OwnedChatMessage> messages_;
    std::vector<std::uint32_t> cached_token_ids_;
    // Preserve numerical execution boundaries for exceptional recovery:
    // replaying decoded tokens as a batched prefill changes rounded K/V.
    struct CacheSegment {
        std::uint32_t begin = 0;
        std::uint32_t prompt_end = 0;
        std::uint32_t end = 0;
    };
    std::vector<CacheSegment> cache_segments_;
    std::vector<std::size_t> turn_cache_positions_;
    std::vector<TokenSampler> turn_sampler_states_;
    std::vector<std::uint32_t> stop_token_ids_;
    std::vector<std::string> reverse_stop_strings_;
    std::optional<std::uint32_t> end_of_turn_token_id_;
    std::size_t max_new_tokens_ = 0;
    bool context_shift_ = false;
    bool speculative_decode_ = true;
    SpeculativeSamplingMode speculative_sampling_ = SpeculativeSamplingMode::exact_match;
    RecurrentExecutionOptions recurrent_execution_;
    StandaloneExecutionOptions standalone_execution_;
    bool turn_active_ = false;
    std::uint64_t publication_revision_ = 0;
    std::size_t maximum_decoded_token_bytes_ = 1;
    TokenSamplerOptions initial_sampler_options_;
    TokenSampler sampler_;
    RetrievalProposer proposer_;
    Gemma2MonotonicClock clock_;
    std::size_t plan_revision_ = 0;
    std::optional<Gemma2MetalWarmupStats> warmup_stats_;
};

}  // namespace local_gguf_engine::engine
