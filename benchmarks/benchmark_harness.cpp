#include "local_gguf_engine/benchmarks/benchmark_harness.hpp"

#include "engine/benchmark_synthetic_llama.hpp"
#include "engine/kv/benchmark_resident_kv_decode.hpp"
#include "model_format/benchmark_model_load.hpp"
#include "tensor/benchmark_synthetic_core.hpp"

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"

namespace local_gguf_engine::benchmarks {

namespace {

[[nodiscard]] BenchmarkBackendSelection make_backend_selection(BenchmarkBackend backend) {
    const auto include_cpu = backend == BenchmarkBackend::both || backend == BenchmarkBackend::cpu;
    const auto include_metal =
        backend == BenchmarkBackend::both || backend == BenchmarkBackend::metal;
    return BenchmarkBackendSelection{
        .include_cpu = include_cpu,
        .include_metal = include_metal,
        .metal_available =
            include_metal && tensor::tensor_backend_is_available(runtime::BackendKind::metal),
    };
}

[[nodiscard]] bool suite_includes(BenchmarkSuite selected, BenchmarkSuite suite) noexcept {
    return selected == BenchmarkSuite::all || selected == suite;
}

} // namespace

BenchmarkReport run_benchmarks(const BenchmarkConfig &config) {
    BenchmarkReport report{
        .config = config,
        .measurements = {},
    };

    const auto selection = make_backend_selection(config.backend);
    if (suite_includes(config.suite, BenchmarkSuite::tensor)) {
        add_synthetic_core_benchmarks(report, selection);
    }

    if (suite_includes(config.suite, BenchmarkSuite::engine)) {
        add_synthetic_llama_benchmarks(report, selection);
    }

    if (config.suite == BenchmarkSuite::resident_kv_decode) {
        add_focused_resident_kv_decode_stage_benchmarks(report, selection);
    }

    if (suite_includes(config.suite, BenchmarkSuite::model_format) && selection.include_cpu &&
        config.model_path.has_value()) {
        add_model_load_benchmarks(report, *config.model_path);
    }

    return report;
}

} // namespace local_gguf_engine::benchmarks
