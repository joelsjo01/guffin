#include "local_gguf_engine/frontends/cli/generate.hpp"

#include <iostream>

int main(int argc, char** argv) {
    return local_gguf_engine::frontends::cli::run_generate(
        argc,
        argv,
        std::cout,
        std::cerr
    );
}
