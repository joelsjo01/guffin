#pragma once

#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::frontends::cli {

struct GenerateOptions {
    engine::SamplingConfig sampling;
    runtime::BackendKind backend = runtime::default_backend();
    bool debug_first_token = false;
};

struct GenerateFirstTokenLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
    std::string text;
};

struct GenerateFirstTokenDebugInfo {
    std::string formatted_prompt;
    bool encode_special_tokens = false;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<GenerateFirstTokenLogit> top_logits;
};

[[nodiscard]] std::string generate_text_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t max_new_tokens
);

[[nodiscard]] std::string generate_text_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t max_new_tokens,
    const GenerateOptions& options
);

[[nodiscard]] GenerateFirstTokenDebugInfo inspect_generate_first_token_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t top_token_count
);

[[nodiscard]] GenerateFirstTokenDebugInfo inspect_generate_first_token_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t top_token_count,
    const GenerateOptions& options
);

void write_generate_first_token_debug(
    const GenerateFirstTokenDebugInfo& info,
    std::ostream& output
);

int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error);

}  // namespace local_gguf_engine::frontends::cli
