#pragma once

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
    runtime::BackendKind backend = runtime::BackendKind::metal;
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

int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error);
int run_generate(
    int argc,
    const char* const* argv,
    std::istream& input,
    std::ostream& output,
    std::ostream& error
);

}  // namespace local_gguf_engine::frontends::cli
