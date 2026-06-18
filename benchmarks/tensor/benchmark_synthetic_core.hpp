#pragma once

#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

namespace local_gguf_engine::benchmarks {

void add_synthetic_core_benchmarks(BenchmarkReport &report, BenchmarkBackendSelection selection);

} // namespace local_gguf_engine::benchmarks
