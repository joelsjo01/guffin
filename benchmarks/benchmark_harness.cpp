#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include "benchmark_model_load.hpp"
#include "benchmark_synthetic_core.hpp"
#include "benchmark_synthetic_llama.hpp"

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"

namespace local_gguf_engine::benchmarks {

BenchmarkReport run_benchmarks(const BenchmarkConfig& config) {
    BenchmarkReport report {
        .config = config,
        .measurements = {},
    };

    const auto metal_available = tensor::tensor_backend_is_available(runtime::BackendKind::metal);
    add_synthetic_core_benchmarks(report, metal_available);
    add_synthetic_llama_benchmarks(report, metal_available);

    if (config.model_path.has_value()) {
        add_model_load_benchmarks(report, *config.model_path);
    }

    return report;
}

}  // namespace local_gguf_engine::benchmarks
