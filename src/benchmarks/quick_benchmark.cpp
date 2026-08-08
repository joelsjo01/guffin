#include "local_gguf_engine/benchmarks/quick_benchmark.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"

#include "../tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace local_gguf_engine::benchmarks {
namespace {

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

using tensor::BackendBuffer;
using tensor::backends::metal::MetalCommandContext;

[[nodiscard]] std::uint64_t run_kernel_command(
    std::string_view kernel_name,
    QuickKernelMeasurementMode mode,
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t rows,
    std::uint32_t dispatch_count
) {
    MetalCommandContext context("Quick kernel diagnostic");
    for (std::uint32_t dispatch = 0; dispatch < dispatch_count; ++dispatch) {
        context.encode_matmul(input, weight, output, rows);
        if (dispatch + 1U < dispatch_count) {
            context.encode_memory_barrier();
        }
    }
    const auto stats = context.commit_and_wait();
    const auto expected_barriers = dispatch_count - 1U;
    if (stats.command_buffer_count != 1U || stats.compute_encoder_count != 1U
        || stats.compute_dispatch_count != dispatch_count
        || stats.memory_barrier_count != expected_barriers
        || stats.commit_count != 1U || stats.synchronous_wait_count != 1U
        || stats.blit_encoder_count != 0U || stats.transient_buffer_allocation_count != 0U) {
        throw std::runtime_error(
            std::string("kernel diagnostic command accounting mismatch for ")
            + std::string(kernel_name) + " (" + std::string(quick_kernel_measurement_mode_name(mode)) + ")"
        );
    }
    if (stats.gpu_duration_ns == 0U) {
        throw std::runtime_error(
            std::string("Metal did not report a kernel diagnostic GPU duration for ")
            + std::string(kernel_name) + " (" + std::string(quick_kernel_measurement_mode_name(mode)) + ")"
        );
    }
    return stats.gpu_duration_ns;
}

[[nodiscard]] QuickKernelMeasurement measure_kernel(
    std::string_view kernel_name,
    QuickKernelMeasurementMode mode,
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t rows,
    std::uint32_t dispatch_count
) {
    for (std::size_t warmup = 0; warmup < quick_kernel_warmup_count; ++warmup) {
        static_cast<void>(run_kernel_command(
            kernel_name, mode, input, weight, output, rows, dispatch_count
        ));
    }
    std::vector<std::uint64_t> raw;
    raw.reserve(quick_kernel_repetition_count);
    for (std::size_t repetition = 0; repetition < quick_kernel_repetition_count; ++repetition) {
        raw.push_back(run_kernel_command(
            kernel_name, mode, input, weight, output, rows, dispatch_count
        ));
    }
    return summarize_kernel_measurement(
        mode,
        dispatch_count,
        static_cast<std::uint64_t>(weight.descriptor().byte_count),
        raw
    );
}

struct KernelInvocation {
    const BackendBuffer* input = nullptr;
    const BackendBuffer* weight = nullptr;
    const BackendBuffer* output = nullptr;
    std::uint32_t rows = 0;
};

[[nodiscard]] std::string shape_label(
    std::uint64_t shared,
    std::uint64_t columns,
    std::uint32_t rows
) {
    return std::to_string(shared) + "x" + std::to_string(columns) + ", "
        + std::to_string(rows) + (rows == 1U ? " row" : " rows");
}

void require_weight_shape(
    const BackendBuffer& weight,
    tensor::DType dtype,
    std::uint64_t shared,
    std::uint64_t columns,
    std::string_view name
) {
    const auto& descriptor = weight.descriptor();
    if (descriptor.dtype != dtype || descriptor.shape.rank() != 2U
        || descriptor.shape.dimension(0) != shared
        || descriptor.shape.dimension(1) != columns) {
        throw std::runtime_error("benchmark model is missing target tensor " + std::string(name));
    }
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

std::string_view quick_kernel_measurement_mode_name(QuickKernelMeasurementMode mode) {
    switch (mode) {
        case QuickKernelMeasurementMode::isolated:
            return "isolated";
        case QuickKernelMeasurementMode::sustained:
            return "sustained";
    }
    throw std::invalid_argument("kernel diagnostic measurement mode is invalid");
}

QuickKernelMeasurement summarize_kernel_measurement(
    QuickKernelMeasurementMode mode,
    std::uint32_t dispatches_per_command,
    std::uint64_t weight_bytes,
    std::span<const std::uint64_t> raw_command_gpu_duration_ns
) {
    if (dispatches_per_command == 0U) {
        throw std::invalid_argument("kernel measurement dispatch count must be positive");
    }
    if (weight_bytes == 0U) {
        throw std::invalid_argument("kernel measurement weight bytes must be positive");
    }
    if (weight_bytes > std::numeric_limits<std::uint64_t>::max() / dispatches_per_command) {
        throw std::overflow_error("kernel measurement streamed byte count overflows");
    }
    if (raw_command_gpu_duration_ns.empty()) {
        throw std::invalid_argument("kernel measurement requires at least one duration");
    }
    std::vector<double> command_durations;
    command_durations.reserve(raw_command_gpu_duration_ns.size());
    for (const auto duration : raw_command_gpu_duration_ns) {
        if (duration == 0U) {
            throw std::invalid_argument("kernel measurement duration must be positive");
        }
        command_durations.push_back(static_cast<double>(duration));
    }
    const auto command_stats = summarize_values(command_durations);
    const auto dispatch_denominator = static_cast<double>(dispatches_per_command);
    const BenchmarkStatistics per_dispatch {
        .median = command_stats.median / dispatch_denominator,
        .minimum = command_stats.minimum / dispatch_denominator,
        .maximum = command_stats.maximum / dispatch_denominator,
    };
    const auto streamed_bytes = weight_bytes * dispatches_per_command;
    return {
        .mode = mode,
        .dispatches_per_command = dispatches_per_command,
        .gpu_duration_per_dispatch_ns = per_dispatch,
        .effective_weight_bandwidth_gb_per_second =
            static_cast<double>(streamed_bytes) / command_stats.median,
        .raw_command_gpu_duration_ns = std::vector<std::uint64_t>(
            raw_command_gpu_duration_ns.begin(), raw_command_gpu_duration_ns.end()
        ),
    };
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
    const auto hidden = static_cast<std::uint64_t>(plan.config().embedding_length);
    const auto feed_forward = static_cast<std::uint64_t>(plan.config().feed_forward_length);
    const BackendBuffer* q4_gate = nullptr;
    const BackendBuffer* q4_down = nullptr;
    const BackendBuffer* converted_q4_down = nullptr;
    for (const auto& block : weights.blocks) {
        const auto& gate = block.feed_forward.gate_weight;
        const auto& down = block.feed_forward.down_weight;
        if (q4_gate == nullptr && gate.descriptor().dtype == tensor::DType::q4_k
            && gate.descriptor().shape.dimension(0) == hidden
            && gate.descriptor().shape.dimension(1) == feed_forward) {
            q4_gate = &gate;
        }
        if (down.descriptor().shape.dimension(0) == feed_forward
            && down.descriptor().shape.dimension(1) == hidden) {
            if (down.descriptor().dtype == tensor::DType::q4_k) {
                if (q4_down == nullptr) {
                    q4_down = &down;
                } else if (converted_q4_down == nullptr) {
                    converted_q4_down = &down;
                }
            }
        }
    }
    if (q4_gate == nullptr || q4_down == nullptr || converted_q4_down == nullptr) {
        throw std::runtime_error("benchmark model is missing target Q4_K FFN tensors");
    }
    require_weight_shape(*q4_gate, tensor::DType::q4_k, hidden, feed_forward, "Q4_K gate");
    require_weight_shape(*q4_down, tensor::DType::q4_k, feed_forward, hidden, "Q4_K down");
    require_weight_shape(
        *converted_q4_down, tensor::DType::q4_k, feed_forward, hidden, "converted Q4_K down"
    );
    require_weight_shape(
        weights.output_weight,
        tensor::DType::q8_0,
        hidden,
        plan.config().vocab_size,
        "Q8_0 output"
    );
    std::vector<KernelInvocation> invocations;
    const auto add_kernel = [&](
        std::string name,
        std::string shape,
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        std::uint32_t rows
    ) {
        result.kernels.push_back({
            .name = std::move(name),
            .shape = std::move(shape),
            .measurements = {},
        });
        invocations.push_back({
            .input = &input,
            .weight = &weight,
            .output = &output,
            .rows = rows,
        });
    };
    add_kernel(
        "Q4_K decode", shape_label(hidden, feed_forward, 1U),
        prefill.normalized_feed_forward_input, *q4_gate, prefill.feed_forward_gate, 1U
    );
    add_kernel(
        "Q4_K generic", shape_label(feed_forward, hidden, 1U),
        prefill.feed_forward_gated, *q4_down, prefill.projected_feed_forward_output, 1U
    );
    add_kernel(
        "Q4_K converted down", shape_label(feed_forward, hidden, 1U),
        prefill.feed_forward_gated, *converted_q4_down, prefill.projected_feed_forward_output, 1U
    );
    add_kernel(
        "Q8_0 output", shape_label(hidden, plan.config().vocab_size, 1U),
        prefill.normalized_hidden, weights.output_weight, prefill.logits, 1U
    );
    for (const std::uint32_t rows : {2U, 4U}) {
        add_kernel(
            "Q4_K verifier", shape_label(hidden, feed_forward, rows),
            prefill.normalized_feed_forward_input, *q4_gate, prefill.feed_forward_gate, rows
        );
        add_kernel(
            "Q4_K converted verifier", shape_label(feed_forward, hidden, rows),
            prefill.feed_forward_gated, *converted_q4_down, prefill.projected_feed_forward_output, rows
        );
    }
    for (const std::uint32_t rows : {15U, 16U}) {
        add_kernel(
            rows == 15U ? "Q4_K tail" : "Q4_K exact-16",
            shape_label(hidden, feed_forward, rows),
            prefill.normalized_feed_forward_input,
            *q4_gate,
            prefill.feed_forward_gate,
            rows
        );
    }
    for (std::size_t index = 0; index < invocations.size(); ++index) {
        const auto& invocation = invocations[index];
        result.kernels[index].measurements[0] = measure_kernel(
            result.kernels[index].name,
            QuickKernelMeasurementMode::isolated,
            *invocation.input,
            *invocation.weight,
            *invocation.output,
            invocation.rows,
            quick_kernel_isolated_dispatch_count
        );
    }
    for (std::size_t index = 0; index < invocations.size(); ++index) {
        const auto& invocation = invocations[index];
        result.kernels[index].measurements[1] = measure_kernel(
            result.kernels[index].name,
            QuickKernelMeasurementMode::sustained,
            *invocation.input,
            *invocation.weight,
            *invocation.output,
            invocation.rows,
            quick_kernel_sustained_dispatch_count
        );
    }
    return result;
}

}  // namespace local_gguf_engine::benchmarks
