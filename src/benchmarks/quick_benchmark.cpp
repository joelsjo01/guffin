#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"

#include "../tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace local_gguf_engine::benchmarks {
namespace {

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

using tensor::BackendBuffer;
using tensor::backends::metal::MetalCommandContext;

[[nodiscard]] QuickKernelResult run_kernel(
    std::string name,
    std::string shape,
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t rows
) {
    const auto dispatch = [&] {
        MetalCommandContext context("Quick kernel diagnostic");
        context.encode_matmul(input, weight, output, rows);
        const auto stats = context.commit_and_wait();
        if (stats.gpu_duration_ns == 0U) {
            throw std::runtime_error("Metal did not report a microbenchmark GPU duration");
        }
        return stats.gpu_duration_ns;
    };
    static_cast<void>(dispatch());
    std::vector<std::uint64_t> raw;
    std::vector<double> values;
    for (std::size_t repetition = 0; repetition < quick_kernel_repetition_count; ++repetition) {
        const auto duration = dispatch();
        raw.push_back(duration);
        values.push_back(static_cast<double>(duration));
    }
    const auto stats = summarize_values(values);
    return {
        .name = std::move(name),
        .shape = std::move(shape),
        .gpu_duration_ns = stats,
        .effective_weight_bandwidth_gb_per_second =
            static_cast<double>(weight.descriptor().byte_count) / stats.median,
        .raw_gpu_duration_ns = std::move(raw),
    };
}

}  // namespace

BenchmarkStatistics summarize_values(std::span<const double> values) {
    if (values.empty()) throw std::invalid_argument("benchmark summary requires at least one value");
    std::vector<double> sorted(values.begin(), values.end());
    for (const auto value : sorted) {
        if (!std::isfinite(value)) throw std::invalid_argument("benchmark summary value must be finite");
    }
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2U;
    const auto median = sorted.size() % 2U == 0U
        ? (sorted[middle - 1U] + sorted[middle]) / 2.0
        : sorted[middle];
    return {.median = median, .minimum = sorted.front(), .maximum = sorted.back()};
}

QuickKernelBenchmarkResult run_quick_kernel_benchmark(const std::string& model_path) {
    if (model_path.empty()) throw std::invalid_argument("benchmark model path must not be empty");
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    auto plan = engine::create_gemma2_metal_model_plan(model);

    QuickKernelBenchmarkResult result {
        .engine_build = LOCAL_GGUF_ENGINE_BUILD,
        .model_path = model_path,
    };
    const auto& weights = plan.weights();
    const auto& prefill = plan.prefill_scratch();
    const auto& block = weights.blocks.front();
    result.kernels.push_back(run_kernel(
        "Q4_K decode", "3584x14336, 1 row", prefill.normalized_feed_forward_input,
        block.feed_forward.gate_weight, prefill.feed_forward_gate, 1U
    ));
    result.kernels.push_back(run_kernel(
        "Q6_K decode", "14336x3584, 1 row", prefill.feed_forward_gated,
        block.feed_forward.down_weight, prefill.projected_feed_forward_output, 1U
    ));
    result.kernels.push_back(run_kernel(
        "F16 output", "3584x256000, 1 row", prefill.final_normalized_hidden,
        weights.output_weight, prefill.logits, 1U
    ));
    result.kernels.push_back(run_kernel(
        "Q4_K tail", "3584x14336, 15 rows", prefill.normalized_feed_forward_input,
        block.feed_forward.gate_weight, prefill.feed_forward_gate, 15U
    ));
    result.kernels.push_back(run_kernel(
        "Q4_K exact-16", "3584x14336, 16 rows", prefill.normalized_feed_forward_input,
        block.feed_forward.gate_weight, prefill.feed_forward_gate, 16U
    ));
    return result;
}

}  // namespace local_gguf_engine::benchmarks
