#include "benchmark_projection.hpp"

#include "benchmark_support.hpp"

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"

#include <sstream>
#include <string>
#include <string_view>

namespace local_gguf_engine::benchmarks {

namespace {

using runtime::BackendKind;
using tensor::copy_tensor_to_backend;
using tensor::TensorStorage;
using tensor::tensor_backend_is_available;
using tensor::tensor_backend_unavailable_reason;
using tensor::ops::matmul_f32_f16_gguf_weight;
using tensor::ops::sum_f32;

[[nodiscard]] std::string projection_benchmark_detail(
    const TensorStorage& left,
    const TensorStorage& right,
    std::string_view backend,
    std::string_view cost_scope
) {
    std::ostringstream stream;
    stream << "left=[" << left.descriptor().shape.dimension(0)
           << "," << left.descriptor().shape.dimension(1)
           << "] right=[" << right.descriptor().shape.dimension(0)
           << "," << right.descriptor().shape.dimension(1)
           << "] dtype=F32/F16 layout=gguf-weight backend=" << backend
           << " cost=" << cost_scope;
    return stream.str();
}

}  // namespace

void warm_up_metal_projection_backend() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto left = make_f32_matrix(1, 4, 0.01F);
    const auto right = make_f16_gguf_weight_matrix(4, 4, 0.02F);
    (void)matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
}

void add_projection_backend_benchmarks(
    BenchmarkReport& report,
    std::string_view label,
    const TensorStorage& left,
    const TensorStorage& right,
    bool metal_available
) {
    const auto cpu_right_buffer = copy_tensor_to_backend(right, BackendKind::cpu);
    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.cpu." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(left, right, "cpu", "compute+allocation"),
        [&left, &right]() {
            const auto output = matmul_f32_f16_gguf_weight(left, right, BackendKind::cpu);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.cpu_buffered." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(left, right, "cpu", "compute+allocation preloaded-right"),
        [&left, cpu_right_buffer]() {
            const auto output = matmul_f32_f16_gguf_weight(left, cpu_right_buffer);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + cpu_right_buffer.descriptor().byte_count
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    if (!metal_available) {
        return;
    }

    const auto metal_right_buffer = copy_tensor_to_backend(right, BackendKind::metal);
    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.metal." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(
            left,
            right,
            "metal",
            "input-copy+dispatch+output-copy cached-pipeline"
        ),
        [&left, &right]() {
            const auto output = matmul_f32_f16_gguf_weight(left, right, BackendKind::metal);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + right.bytes().size() + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "tensor.matmul_f32_f16_gguf_weight.metal_buffered." + std::string(label),
        report.config.iterations,
        projection_benchmark_detail(
            left,
            right,
            "metal",
            "input-copy+dispatch+output-copy preloaded-right cached-pipeline"
        ),
        [&left, metal_right_buffer]() {
            const auto output = matmul_f32_f16_gguf_weight(left, metal_right_buffer);
            return IterationMeasurement {
                .bytes_processed = left.bytes().size() + metal_right_buffer.descriptor().byte_count
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));
}

void add_unavailable_metal_projection_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "tensor.matmul_f32_f16_gguf_weight.metal.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

}  // namespace local_gguf_engine::benchmarks
