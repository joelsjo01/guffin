#pragma once

#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <string_view>

namespace local_gguf_engine::benchmarks {

void warm_up_metal_projection_backend();

void add_projection_backend_benchmarks(
    BenchmarkReport& report,
    std::string_view label,
    const tensor::TensorStorage& left,
    const tensor::TensorStorage& right,
    bool metal_available
);

void add_unavailable_metal_projection_benchmark(BenchmarkReport& report);

}  // namespace local_gguf_engine::benchmarks
