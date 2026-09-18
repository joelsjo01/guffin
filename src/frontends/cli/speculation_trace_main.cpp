#include "local_gguf_engine/frontends/cli/speculation_trace.hpp"

#include <iostream>

int main(int argc, char** argv) {
    return local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        argc, argv, std::cout, std::cerr
    );
}
