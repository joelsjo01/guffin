#include "local_gguf_engine/frontends/cli/generate.hpp"

#include <iostream>
#include <chrono>
#include <cstdint>

int main(int argc, char** argv) {
    const auto process_started_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
    return local_gguf_engine::frontends::cli::run_generate(
        argc,
        argv,
        std::cout,
        std::cerr,
        process_started_ns
    );
}
