#pragma once

#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include <filesystem>

namespace local_gguf_engine::benchmarks {

void add_model_load_benchmarks(BenchmarkReport& report, const std::filesystem::path& model_path);

}  // namespace local_gguf_engine::benchmarks
