#include "local_gguf_engine/frontends/cli/cli.hpp"

#include <iostream>

int main(int argc, char** argv) {
    return local_gguf_engine::frontends::cli::run_cli(
        argc, argv, std::cout, std::cerr
    );
}
