#pragma once

#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

namespace local_gguf_engine::benchmarks {

void add_focused_resident_kv_decode_stage_benchmarks(BenchmarkReport& report, bool metal_available);

}  // namespace local_gguf_engine::benchmarks
