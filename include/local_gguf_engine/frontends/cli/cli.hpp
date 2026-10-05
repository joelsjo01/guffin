#pragma once

#include <iosfwd>

namespace local_gguf_engine::frontends::cli {

int run_cli(int argc, const char* const* argv, std::ostream& output, std::ostream& error);

// Injected streams select their UI explicitly; standard streams detect a TTY.
int run_cli(int argc, const char* const* argv, std::istream& input,
    std::ostream& output, std::ostream& error, bool interactive = true);

} // namespace local_gguf_engine::frontends::cli
