#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>

namespace local_gguf_engine::frontends::cli {

[[nodiscard]] std::string generate_text_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t max_new_tokens
);
int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error);

}  // namespace local_gguf_engine::frontends::cli
