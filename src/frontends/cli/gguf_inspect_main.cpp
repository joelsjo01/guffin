#include "local_gguf_engine/frontends/cli/gguf_inspect.hpp"

#include <iostream>

int main(int argc, char** argv) {
    return local_gguf_engine::frontends::cli::run_gguf_inspect(
        argc,
        argv,
        std::cout,
        std::cerr
    );
}
