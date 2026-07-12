#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/sampling.hpp"
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
    std::size_t max_new_tokens = 128;
    runtime::BackendKind backend = runtime::BackendKind::metal;
    std::vector<std::string> reverse_stop_strings;
    TokenSamplerOptions sampler;
};

struct Gemma2ChatTurnResult {
    std::string text;
    std::uint32_t cache_position = 0;
    std::uint32_t appended_token_count = 0;
    std::uint32_t cached_token_count = 0;
    std::vector<std::uint32_t> generated_token_ids;
    std::optional<std::uint32_t> stop_token_id;
    std::optional<std::string> reverse_stop_string;
    bool interrupted = false;
    std::uint32_t trimmed_turn_count = 0;
    bool rebuilt_context = false;
};

using Gemma2ChatStreamCallback = std::function<void(std::string_view text_delta)>;
using Gemma2ChatShouldStopCallback = std::function<bool()>;

struct Gemma2ChatTurnOptions {
    Gemma2ChatStreamCallback on_text_delta;
    Gemma2ChatShouldStopCallback should_stop;
};

class Gemma2ChatSession final {
public:
    Gemma2ChatSession(
        model_format::GGUFFileBackedModel file_model,
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
    [[nodiscard]] std::uint32_t cached_token_count() const noexcept;
    [[nodiscard]] std::uint32_t context_length() const noexcept;
    [[nodiscard]] std::uint32_t remaining_context_tokens() const noexcept;
    [[nodiscard]] std::size_t max_new_tokens() const noexcept;
    [[nodiscard]] std::size_t message_count() const noexcept;

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

    model_format::GGUFFileBackedModel file_model_;
    EngineModel model_;
    tokenizer::Tokenizer tokenizer_;
    Gemma2MetalModelPlan plan_;
    std::vector<OwnedChatMessage> messages_;
    std::vector<std::uint32_t> cached_token_ids_;
    std::vector<std::size_t> turn_cache_positions_;
    std::vector<TokenSampler> turn_sampler_states_;
    std::vector<std::uint32_t> stop_token_ids_;
    std::vector<std::string> reverse_stop_strings_;
    std::optional<std::uint32_t> end_of_turn_token_id_;
    std::size_t max_new_tokens_ = 0;
    TokenSamplerOptions initial_sampler_options_;
    TokenSampler sampler_;
};

}  // namespace local_gguf_engine::engine
