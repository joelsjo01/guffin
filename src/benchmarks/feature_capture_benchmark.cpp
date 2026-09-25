#include "local_gguf_engine/benchmarks/feature_capture_benchmark.hpp"

#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::benchmarks {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
    );
}

[[nodiscard]] std::vector<std::uint32_t> fixture_tokens(
    std::uint32_t count,
    std::uint32_t vocabulary_size
) {
    if (vocabulary_size <= 2U) {
        throw std::invalid_argument("feature capture benchmark vocabulary is too small");
    }
    std::vector<std::uint32_t> tokens(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        tokens[index] = 2U + index % (vocabulary_size - 2U);
    }
    return tokens;
}

}  // namespace

FeatureCaptureBenchmarkResult run_feature_capture_benchmark(
    const std::string& model_path,
    bool capture_enabled
) {
    if (model_path.empty()) {
        throw std::invalid_argument("feature capture benchmark model path is empty");
    }
    const auto load_begin = Clock::now();
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_verified_engine_model_from_gguf(
        file_model.view(), file_model.bytes()
    );
    const auto load_end = Clock::now();

    engine::ResidentTargetPayloadReceipt resident;
    const auto upload_begin = Clock::now();
    auto plan = engine::create_gemma2_metal_model_plan(
        model, engine::Gemma2OutputHeadPolicy::q8_0, resident
    );
    const auto upload_end = Clock::now();
    constexpr auto prompt_rows = std::uint32_t {128U};
    constexpr auto warmup_rows = std::uint32_t {16U};
    if (plan.config().context_length < prompt_rows) {
        throw std::invalid_argument("feature capture benchmark requires 128 context rows");
    }

    std::optional<engine::FeatureSinkSet> sinks;
    const auto feature_begin = Clock::now();
    if (capture_enabled) {
        constexpr auto projected_width = std::uint32_t {256U};
        const auto weight_descriptor = tensor::TensorDescriptor::from(
            tensor::Shape::from({plan.config().embedding_length, projected_width}),
            tensor::DType::f16
        );
        const auto weights = tensor::TensorStorage::allocate(weight_descriptor);
        engine::FeatureBankSpec spec;
        spec.bank_id = 1U;
        spec.tap_count = 1U;
        spec.taps[0] = {.stage = engine::FeatureTapStage::final_normalized};
        spec.child_count = 1U;
        spec.children[0] = {
            .child_id = 1U,
            .output_width = projected_width,
            .weight_dtype = tensor::DType::f16,
            .schedule = engine::FeatureProjectionSchedule::eager,
            .payload_offset = 0U,
            .payload_byte_count = weights.bytes().size(),
            .payload_digest = runtime::sha256(weights.bytes()),
        };
        const std::array specs {spec};
        sinks.emplace(engine::create_feature_sink_set(
            specs, weights.bytes(), plan.config().embedding_length,
            plan.config().block_count, plan.config().context_length,
            engine::gemma2_metal_verifier_row_max
        ));
    }
    const auto feature_end = Clock::now();

    const auto scale = engine::gemma2_input_embedding_scale(
        plan.config().embedding_length
    );
    const auto warmup = fixture_tokens(warmup_rows, plan.config().vocab_size);
    if (sinks.has_value()) {
        static_cast<void>(engine::run_gemma2_metal_resident_prefill(
            plan, warmup, 0U, scale, std::nullopt,
            engine::FeatureSinkBinding {
                .sinks = &*sinks,
                .destination = engine::FeatureCaptureDestination::canonical,
                .execution_epoch = 1U,
            }
        ));
        sinks->reset();
    } else {
        static_cast<void>(engine::run_gemma2_metal_resident_prefill(
            plan, warmup, 0U, scale
        ));
    }

    const auto prompt = fixture_tokens(prompt_rows, plan.config().vocab_size);
    const auto prompt_begin = Clock::now();
    const auto prefill = sinks.has_value()
        ? engine::run_gemma2_metal_resident_prefill(
            plan, prompt, 0U, scale, std::nullopt,
            engine::FeatureSinkBinding {
                .sinks = &*sinks,
                .destination = engine::FeatureCaptureDestination::canonical,
                .execution_epoch = 2U,
            }
        )
        : engine::run_gemma2_metal_resident_prefill(plan, prompt, 0U, scale);
    const auto prompt_end = Clock::now();
    const auto logits = tensor::copy_tensor_from_backend(prefill.logits);

    return FeatureCaptureBenchmarkResult {
        .model_path = model_path,
        .capture_enabled = capture_enabled,
        .prompt_token_count = prompt_rows,
        .verified_model_load_wall_ns = elapsed_ns(load_begin, load_end),
        .target_upload_wall_ns = elapsed_ns(upload_begin, upload_end),
        .feature_setup_wall_ns = elapsed_ns(feature_begin, feature_end),
        .prompt_wall_ns = elapsed_ns(prompt_begin, prompt_end),
        .prompt_gpu_ns = prefill.execution.gpu_duration_ns,
        .prompt_command_buffer_count = prefill.execution.command_buffer_count,
        .prompt_synchronous_wait_count = prefill.execution.synchronous_wait_count,
        .target_resident_bytes = plan.uploaded_weight_byte_count(),
        .features = sinks.has_value() ? sinks->metrics() : engine::FeatureSinkMetrics {},
        .logits_digest = runtime::sha256(logits.bytes()),
    };
}

}  // namespace local_gguf_engine::benchmarks
