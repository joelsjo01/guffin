#include "local_gguf_engine/benchmarks/output_head_benchmark.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include "../tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

namespace local_gguf_engine::benchmarks {
namespace {

using engine::GemmaOutputHeadPolicy;

struct BaselineRows {
    std::vector<float> logits;
    std::vector<float> hidden;
    std::size_t resident_weight_bytes = 0;
};

[[nodiscard]] std::uint64_t run_output_kernel_command(
    const tensor::BackendBuffer& input,
    const tensor::BackendBuffer& weight,
    const tensor::BackendBuffer& output,
    std::uint32_t dispatch_count
) {
    tensor::backends::metal::MetalCommandContext context("Output-head diagnostic");
    for (std::uint32_t dispatch = 0U; dispatch < dispatch_count; ++dispatch) {
        context.encode_matmul(input, weight, output, 1U);
        if (dispatch + 1U < dispatch_count) context.encode_memory_barrier();
    }
    const auto stats = context.commit_and_wait();
    const auto expected_f16 = weight.descriptor().dtype == tensor::DType::f16
        ? dispatch_count : 0U;
    const auto expected_q8 = weight.descriptor().dtype == tensor::DType::q8_0
        ? dispatch_count : 0U;
    if (stats.command_buffer_count != 1U || stats.compute_encoder_count != 1U
        || stats.compute_dispatch_count != dispatch_count
        || stats.memory_barrier_count != dispatch_count - 1U
        || stats.commit_count != 1U || stats.synchronous_wait_count != 1U
        || stats.transient_buffer_allocation_count != 0U
        || stats.f16_projection_count != expected_f16
        || stats.q8_decode_projection_count != expected_q8
        || stats.gpu_duration_ns == 0U) {
        throw std::runtime_error("output-head kernel command accounting mismatch");
    }
    return stats.gpu_duration_ns;
}

[[nodiscard]] QuickKernelMeasurement measure_output_kernel(
    QuickKernelMeasurementMode mode,
    const tensor::BackendBuffer& input,
    const tensor::BackendBuffer& weight,
    const tensor::BackendBuffer& output,
    std::uint32_t dispatch_count
) {
    static_cast<void>(run_output_kernel_command(input, weight, output, dispatch_count));
    std::vector<std::uint64_t> raw;
    raw.reserve(quick_kernel_repetition_count);
    for (std::size_t repetition = 0U; repetition < quick_kernel_repetition_count; ++repetition) {
        raw.push_back(run_output_kernel_command(input, weight, output, dispatch_count));
    }
    return summarize_kernel_measurement(
        mode, dispatch_count, weight.descriptor().byte_count, raw
    );
}

[[nodiscard]] std::array<QuickKernelMeasurement, 2> measure_output_kernel(
    const tensor::BackendBuffer& input,
    const tensor::BackendBuffer& weight,
    const tensor::BackendBuffer& output
) {
    return {
        measure_output_kernel(
            QuickKernelMeasurementMode::isolated, input, weight, output,
            quick_kernel_isolated_dispatch_count
        ),
        measure_output_kernel(
            QuickKernelMeasurementMode::sustained, input, weight, output,
            quick_kernel_sustained_dispatch_count
        ),
    };
}

[[nodiscard]] std::vector<float> f32_values(const tensor::BackendBuffer& buffer) {
    auto storage = tensor::copy_tensor_from_backend(buffer);
    if (storage.descriptor().dtype != tensor::DType::f32
        || storage.bytes().size() % sizeof(float) != 0U) {
        throw std::runtime_error("output-head evidence expected an F32 backend buffer");
    }
    std::vector<float> values(storage.bytes().size() / sizeof(float));
    std::memcpy(values.data(), storage.bytes().data(), storage.bytes().size());
    return values;
}

[[nodiscard]] std::vector<std::uint32_t> fixture_tokens(
    const tokenizer::Tokenizer& tokenizer,
    std::uint32_t prediction_count
) {
    constexpr std::string_view fixture =
        "A fast local inference engine should be measured as a complete system. "
        "Weights, caches, scheduling, numerical behavior, and token selection all contribute to the result.\n"
        "User: Explain why careful experiments need a control. Assistant: A control separates the proposed change "
        "from thermal drift, workload changes, and measurement noise.\n"
        "Reasoning example: If a matrix consumes eighteen milliseconds and compression saves eight milliseconds, "
        "the whole token improves only by those eight milliseconds, not by the matrix percentage alone.\n"
        "Code example: for (std::size_t i = 0; i < values.size(); ++i) { sum += values[i]; }\n";
    std::string text;
    std::vector<std::uint32_t> tokens;
    const auto required = static_cast<std::size_t>(prediction_count) + 1U;
    while (tokens.size() < required) {
        text.append(fixture);
        tokens = tokenizer.encode(text, tokenizer.default_encode_options());
    }
    tokens.resize(required);
    return tokens;
}

template <typename Consumer>
std::size_t run_teacher_forced(
    const engine::EngineModel& model,
    std::span<const std::uint32_t> tokens,
    GemmaOutputHeadPolicy policy,
    std::array<QuickKernelMeasurement, 2>& kernel_measurements,
    Consumer consume
) {
    auto plan = engine::create_gemma2_metal_model_plan(model, policy);
    const auto scale = engine::gemma_input_embedding_scale(plan.config().embedding_length);
    auto prefill = engine::run_gemma2_metal_resident_prefill(plan, tokens.first(1U), 0U, scale);
    consume(0U, prefill.logits, plan.prefill_scratch().final_normalized_hidden);
    kernel_measurements = measure_output_kernel(
        plan.prefill_scratch().final_normalized_hidden,
        plan.weights().output_weight,
        prefill.logits
    );
    for (std::size_t position = 1U; position + 1U < tokens.size(); ++position) {
        const auto schedule = engine::create_gemma2_metal_decode_schedule(
            plan, static_cast<std::uint32_t>(position)
        );
        auto result = engine::run_gemma2_metal_resident_decode_step(
            plan,
            schedule,
            tokens[position],
            scale,
            std::nullopt,
            engine::Gemma2MetalDecodeOutputMode::logits
        );
        consume(position, result.logits, plan.decode_scratch().normalized_hidden);
    }
    return plan.uploaded_weight_byte_count();
}

[[nodiscard]] std::vector<std::uint32_t> top_tokens(std::span<const float> logits, std::size_t count) {
    using Entry = std::pair<float, std::uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (std::uint32_t token = 0U; token < logits.size(); ++token) {
        const Entry entry {logits[token], token};
        if (heap.size() < count) heap.push(entry);
        else if (entry > heap.top()) {
            heap.pop();
            heap.push(entry);
        }
    }
    std::vector<std::uint32_t> tokens;
    tokens.reserve(heap.size());
    while (!heap.empty()) {
        tokens.push_back(heap.top().second);
        heap.pop();
    }
    std::sort(tokens.begin(), tokens.end());
    return tokens;
}

[[nodiscard]] float softcap(float value, float limit) {
    return limit * std::tanh(value / limit);
}

}  // namespace

OutputHeadQualityResult run_output_head_quality_benchmark(
    const std::string& model_path,
    std::uint32_t prediction_count
) {
    if (model_path.empty() || prediction_count == 0U) {
        throw std::invalid_argument("output-head quality benchmark requires a model and predictions");
    }
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    const auto model = engine::load_engine_model_from_gguf(file_model.view());
    const auto tokenizer = tokenizer::load_tokenizer_from_gguf_metadata(
        file_model.view().directory.metadata
    );
    const auto tokens = fixture_tokens(tokenizer, prediction_count);
    const auto vocab = model.gemma_config().vocab_size;
    const auto hidden = model.gemma_config().embedding_length;
    BaselineRows baseline;
    std::array<QuickKernelMeasurement, 2> f16_kernel_measurements;
    baseline.logits.reserve(static_cast<std::size_t>(prediction_count) * vocab);
    baseline.hidden.reserve(static_cast<std::size_t>(prediction_count) * hidden);
    baseline.resident_weight_bytes = run_teacher_forced(
        model,
        tokens,
        GemmaOutputHeadPolicy::f16,
        f16_kernel_measurements,
        [&](std::size_t, const tensor::BackendBuffer& logits, const tensor::BackendBuffer& normalized_hidden) {
            auto logit_values = f32_values(logits);
            auto hidden_values = f32_values(normalized_hidden);
            baseline.logits.insert(baseline.logits.end(), logit_values.begin(), logit_values.end());
            baseline.hidden.insert(baseline.hidden.end(), hidden_values.begin(), hidden_values.end());
        }
    );

    OutputHeadQualityResult result {
        .engine_build = LOCAL_GGUF_ENGINE_BUILD,
        .model_path = model_path,
        .prediction_count = prediction_count,
        .vocabulary_size = vocab,
        .f16_resident_weight_bytes = baseline.resident_weight_bytes,
        .top40_minimum_overlap = 40U,
        .finite = true,
    };
    result.f16_kernel_measurements = std::move(f16_kernel_measurements);
    long double absolute_error_sum = 0.0L;
    std::uint64_t absolute_error_count = 0U;
    long double p95_sum = 0.0L;
    std::uint32_t top1_matches = 0U;
    std::uint64_t top40_overlap_sum = 0U;
    long double kl_sum = 0.0L;
    long double f16_nll_sum = 0.0L;
    long double q8_nll_sum = 0.0L;
    constexpr float final_softcap = 30.0F;

    result.q8_resident_weight_bytes = run_teacher_forced(
        model,
        tokens,
        GemmaOutputHeadPolicy::q8_0,
        result.q8_kernel_measurements,
        [&](std::size_t position, const tensor::BackendBuffer& logits, const tensor::BackendBuffer& normalized_hidden) {
            const auto q8 = f32_values(logits);
            const auto q8_hidden = f32_values(normalized_hidden);
            const auto base_offset = position * static_cast<std::size_t>(vocab);
            const auto hidden_offset = position * static_cast<std::size_t>(hidden);
            const auto f16 = std::span<const float>(baseline.logits).subspan(base_offset, vocab);
            const auto f16_hidden = std::span<const float>(baseline.hidden).subspan(hidden_offset, hidden);

            std::vector<float> row_errors(vocab);
            float f16_max = -std::numeric_limits<float>::infinity();
            float q8_max = -std::numeric_limits<float>::infinity();
            std::uint32_t f16_top = 0U;
            std::uint32_t q8_top = 0U;
            for (std::uint32_t index = 0U; index < vocab; ++index) {
                if (!std::isfinite(f16[index]) || !std::isfinite(q8[index])) result.finite = false;
                const auto error = std::abs(f16[index] - q8[index]);
                row_errors[index] = error;
                absolute_error_sum += error;
                ++absolute_error_count;
                result.logit_max_absolute_error = std::max(result.logit_max_absolute_error, double(error));
                if (f16[index] > f16[f16_top]) f16_top = index;
                if (q8[index] > q8[q8_top]) q8_top = index;
                f16_max = std::max(f16_max, softcap(f16[index], final_softcap));
                q8_max = std::max(q8_max, softcap(q8[index], final_softcap));
            }
            if (f16_top == q8_top) ++top1_matches;
            const auto percentile_index = static_cast<std::size_t>(0.95 * double(vocab - 1U));
            std::nth_element(
                row_errors.begin(),
                row_errors.begin() + static_cast<std::ptrdiff_t>(percentile_index),
                row_errors.end()
            );
            p95_sum += row_errors[percentile_index];

            const auto f16_top40 = top_tokens(f16, 40U);
            const auto q8_top40 = top_tokens(q8, 40U);
            std::vector<std::uint32_t> overlap;
            std::set_intersection(
                f16_top40.begin(), f16_top40.end(), q8_top40.begin(), q8_top40.end(),
                std::back_inserter(overlap)
            );
            const auto overlap_count = static_cast<std::uint32_t>(overlap.size());
            top40_overlap_sum += overlap_count;
            result.top40_minimum_overlap = std::min(result.top40_minimum_overlap, overlap_count);

            long double f16_sum = 0.0L;
            long double q8_sum = 0.0L;
            for (std::uint32_t index = 0U; index < vocab; ++index) {
                f16_sum += std::exp(double(softcap(f16[index], final_softcap) - f16_max));
                q8_sum += std::exp(double(softcap(q8[index], final_softcap) - q8_max));
            }
            long double row_kl = 0.0L;
            for (std::uint32_t index = 0U; index < vocab; ++index) {
                const auto f16_log_probability = static_cast<long double>(softcap(f16[index], final_softcap) - f16_max)
                    - std::log(f16_sum);
                const auto q8_log_probability = static_cast<long double>(softcap(q8[index], final_softcap) - q8_max)
                    - std::log(q8_sum);
                const auto probability = std::exp(f16_log_probability);
                row_kl += probability * (f16_log_probability - q8_log_probability);
            }
            const auto target = tokens[position + 1U];
            f16_nll_sum -= static_cast<long double>(softcap(f16[target], final_softcap) - f16_max) - std::log(f16_sum);
            q8_nll_sum -= static_cast<long double>(softcap(q8[target], final_softcap) - q8_max) - std::log(q8_sum);
            kl_sum += row_kl;
            result.maximum_kl_divergence = std::max(result.maximum_kl_divergence, double(row_kl));

            for (std::uint32_t index = 0U; index < hidden; ++index) {
                result.hidden_max_absolute_error = std::max(
                    result.hidden_max_absolute_error,
                    double(std::abs(f16_hidden[index] - q8_hidden[index]))
                );
            }
        }
    );

    const auto predictions = static_cast<long double>(prediction_count);
    result.logit_mean_absolute_error = double(
        absolute_error_sum / static_cast<long double>(absolute_error_count)
    );
    result.logit_mean_p95_absolute_error = double(p95_sum / predictions);
    result.top1_agreement = double(static_cast<long double>(top1_matches) / predictions);
    result.top40_mean_overlap = double(static_cast<long double>(top40_overlap_sum) / predictions / 40.0L);
    result.mean_kl_divergence = double(kl_sum / predictions);
    result.f16_negative_log_likelihood = double(f16_nll_sum / predictions);
    result.q8_negative_log_likelihood = double(q8_nll_sum / predictions);
    result.f16_perplexity = std::exp(result.f16_negative_log_likelihood);
    result.q8_perplexity = std::exp(result.q8_negative_log_likelihood);
    const auto perplexity_regression = result.q8_perplexity / result.f16_perplexity - 1.0;
    result.quality_gates_pass = result.finite
        && result.hidden_max_absolute_error <= 1.0e-6
        && result.top1_agreement >= 0.99
        && result.top40_mean_overlap >= 0.975
        && result.mean_kl_divergence <= 1.0e-4
        && result.maximum_kl_divergence <= 1.0e-2
        && perplexity_regression <= 0.005;
    return result;
}

}  // namespace local_gguf_engine::benchmarks
