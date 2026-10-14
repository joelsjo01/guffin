#pragma once

#include "local_gguf_engine/engine/gemma_chat_session.hpp"

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

namespace local_gguf_engine::frontends::cli {

struct CommandOptions {
    std::optional<std::filesystem::path> model_path;
    std::optional<std::string> prompt;
    bool interactive = false;
    bool stats = false;
    bool help = false;
    bool help_all = false;
    bool version = false;
    bool decode_diagnostics = false;
    std::optional<std::string> metal_capture_path;
    engine::GemmaChatSessionOptions session {
        .sampler = {
            .temperature = 0.8F,
            .top_k = 40U,
            .top_p = 0.95F,
            .min_p = 0.05F,
            .repetition_penalty = 1.0F,
            .seed = engine::TokenSamplerOptions::random_seed,
            .repetition_window = 64U,
        },
    };
};

void write_help(std::ostream& output, bool advanced);
[[nodiscard]] bool parse_command(int argc, const char* const* argv,
    CommandOptions& command, std::ostream& error);

} // namespace local_gguf_engine::frontends::cli
