#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include <iostream>

int main(int argc, char **argv) {
    return local_gguf_engine::benchmarks::run_benchmark_cli(argc, argv, std::cout, std::cerr);
}
