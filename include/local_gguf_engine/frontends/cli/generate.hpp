#pragma once

#include "local_gguf_engine/runtime/backend_kind.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::engine {
struct Gemma2ChatTurnMetrics;
}

namespace local_gguf_engine::frontends::cli {

struct GenerateOptions {
    runtime::BackendKind backend = runtime::BackendKind::metal;
};

[[nodiscard]] std::string format_chat_timing_line(
    const engine::Gemma2ChatTurnMetrics& metrics
);

[[nodiscard]] std::string format_chat_timing_detail_line(
    const engine::Gemma2ChatTurnMetrics& metrics
);

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

int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error);
int run_generate(
    int argc,
    const char* const* argv,
    std::ostream& output,
    std::ostream& error,
    std::uint64_t process_started_ns
);
int run_generate(
    int argc,
    const char* const* argv,
    std::istream& input,
    std::ostream& output,
    std::ostream& error
);

}  // namespace local_gguf_engine::frontends::cli
