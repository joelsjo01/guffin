#pragma once

#include <string>

namespace local_gguf_engine::engine {
struct GemmaChatTurnMetrics;
}

namespace local_gguf_engine::frontends::cli {

[[nodiscard]] std::string format_chat_timing_line(const engine::GemmaChatTurnMetrics& metrics);
[[nodiscard]] std::string format_chat_timing_detail_line(const engine::GemmaChatTurnMetrics& metrics);
[[nodiscard]] std::string format_chat_decode_attribution_line(const engine::GemmaChatTurnMetrics& metrics);

} // namespace local_gguf_engine::frontends::cli
