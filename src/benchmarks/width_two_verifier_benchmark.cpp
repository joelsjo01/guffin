#include "local_gguf_engine/benchmarks/width_two_verifier_benchmark.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/f16.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace local_gguf_engine::benchmarks {
namespace {

[[nodiscard]] std::vector<float> f32_values(const tensor::BackendBuffer& buffer) {
    const auto storage = tensor::copy_tensor_from_backend(buffer);
    std::vector<float> values(storage.bytes().size() / sizeof(float));
    std::memcpy(values.data(), storage.bytes().data(), storage.bytes().size());
    return values;
}

[[nodiscard]] std::size_t buffer_bytes(const tensor::BackendBuffer& buffer) {
    return buffer.descriptor().byte_count;
}

[[nodiscard]] std::size_t verifier_scratch_bytes(const engine::Gemma2MetalVerifierScratch& scratch) {
    auto bytes = buffer_bytes(scratch.token_id_staging) + buffer_bytes(scratch.normalized_hidden)
        + buffer_bytes(scratch.raw_query) + buffer_bytes(scratch.raw_key)
        + buffer_bytes(scratch.raw_value) + buffer_bytes(scratch.query)
        + buffer_bytes(scratch.attention_partial_output) + buffer_bytes(scratch.attention_partial_maxima)
        + buffer_bytes(scratch.attention_partial_denominators) + buffer_bytes(scratch.attention_output)
        + buffer_bytes(scratch.projected_attention_output) + buffer_bytes(scratch.post_attention_residual)
        + buffer_bytes(scratch.normalized_feed_forward_input) + buffer_bytes(scratch.feed_forward_gate)
        + buffer_bytes(scratch.feed_forward_up) + buffer_bytes(scratch.feed_forward_gated)
        + buffer_bytes(scratch.projected_feed_forward_output) + buffer_bytes(scratch.feed_forward_residual)
        + buffer_bytes(scratch.logits) + buffer_bytes(scratch.backup_key) + buffer_bytes(scratch.backup_value);
    for (const auto& layer : scratch.tentative_kv) {
        bytes += buffer_bytes(layer.key) + buffer_bytes(layer.value);
    }
    return bytes;
}

[[nodiscard]] bool same_bytes(const tensor::TensorStorage& left, const tensor::TensorStorage& right) {
    return left.descriptor() == right.descriptor()
        && std::equal(left.bytes().begin(), left.bytes().end(), right.bytes().begin());
}

[[nodiscard]] float softcap(float value, float cap) {
    return std::tanh(value / cap) * cap;
}

[[nodiscard]] std::array<std::size_t, 40> top40(
    std::span<const float> values,
    float cap
) {
    std::vector<std::size_t> indices(values.size());
    for (std::size_t index = 0; index < indices.size(); ++index) indices[index] = index;
    std::partial_sort(indices.begin(), indices.begin() + 40, indices.end(), [&](auto left, auto right) {
        const auto left_value = softcap(values[left], cap);
        const auto right_value = softcap(values[right], cap);
        return left_value > right_value || (left_value == right_value && left < right);
    });
    std::array<std::size_t, 40> result {};
    std::copy_n(indices.begin(), 40, result.begin());
    return result;
}

[[nodiscard]] double kl_divergence(
    std::span<const float> reference,
    std::span<const float> candidate,
    float cap
) {
    const auto reference_max = softcap(*std::max_element(reference.begin(), reference.end()), cap);
    const auto candidate_max = softcap(*std::max_element(candidate.begin(), candidate.end()), cap);
    double reference_sum = 0.0;
    double candidate_sum = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        reference_sum += std::exp(static_cast<double>(softcap(reference[index], cap) - reference_max));
        candidate_sum += std::exp(static_cast<double>(softcap(candidate[index], cap) - candidate_max));
    }
    double kl = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const auto log_p = static_cast<double>(softcap(reference[index], cap) - reference_max)
            - std::log(reference_sum);
        const auto log_q = static_cast<double>(softcap(candidate[index], cap) - candidate_max)
            - std::log(candidate_sum);
        const auto p = std::exp(log_p);
        kl += p * (log_p - log_q);
    }
    return kl;
}

[[nodiscard]] float maximum_f16_error(
    const tensor::TensorStorage& left,
    const tensor::TensorStorage& right
) {
    float maximum = 0.0F;
    for (std::size_t offset = 0; offset < left.bytes().size(); offset += sizeof(std::uint16_t)) {
        std::uint16_t left_bits = 0;
        std::uint16_t right_bits = 0;
        std::memcpy(&left_bits, left.bytes().data() + offset, sizeof(left_bits));
        std::memcpy(&right_bits, right.bytes().data() + offset, sizeof(right_bits));
        maximum = std::max(maximum, std::abs(
            tensor::f16_bits_to_f32(left_bits) - tensor::f16_bits_to_f32(right_bits)
        ));
    }
    return maximum;
}

}  // namespace

WidthTwoVerifierBenchmarkResult run_width_two_verifier_benchmark(const std::string& model_path) {
    if (model_path.empty()) throw std::invalid_argument("width-two benchmark model path is empty");
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    auto plan = engine::create_gemma2_metal_model_plan(model);
    const auto& config = plan.config();
    if (config.context_length < 18U || config.vocab_size < 40U) {
        throw std::invalid_argument("width-two benchmark model is too small");
    }
    const auto embedding_scale = std::sqrt(static_cast<float>(config.embedding_length));
    std::array<std::uint32_t, 16> prefix {};
    for (std::uint32_t index = 0; index < prefix.size(); ++index) {
        prefix[index] = 2U + index % (config.vocab_size - 2U);
    }
    const std::array<std::uint32_t, 2> proposed {5U, 7U};
    (void)engine::run_gemma2_metal_resident_prefill(plan, prefix, 0U, embedding_scale);

    // One warm command of each shape establishes pipelines and resident pages;
    // these rows are outside the logical 16-token prefix used below.
    (void)engine::run_gemma2_metal_resident_decode_step(
        plan, engine::create_gemma2_metal_decode_schedule(plan, 16U),
        proposed[0], embedding_scale
    );
    (void)engine::run_gemma2_metal_resident_decode_step(
        plan, engine::create_gemma2_metal_decode_schedule(plan, 17U),
        proposed[1], embedding_scale
    );
    (void)engine::run_gemma2_metal_width_two_verifier(
        plan, engine::create_gemma2_metal_verifier_schedule(plan, 16U), proposed, embedding_scale
    );
    engine::discard_gemma2_metal_width_two_verification(plan);

    const auto first = engine::run_gemma2_metal_resident_decode_step(
        plan, engine::create_gemma2_metal_decode_schedule(plan, 16U),
        proposed[0], embedding_scale
    );
    const auto first_logits = f32_values(first.logits);
    const auto second = engine::run_gemma2_metal_resident_decode_step(
        plan, engine::create_gemma2_metal_decode_schedule(plan, 17U),
        proposed[1], embedding_scale
    );
    const auto second_logits = f32_values(second.logits);
    std::vector<tensor::TensorStorage> sequential_keys;
    std::vector<tensor::TensorStorage> sequential_values;
    for (const auto& layer : plan.kv_cache()) {
        sequential_keys.push_back(tensor::copy_tensor_from_backend(layer.key));
        sequential_values.push_back(tensor::copy_tensor_from_backend(layer.value));
    }

    const auto verified = engine::run_gemma2_metal_width_two_verifier(
        plan, engine::create_gemma2_metal_verifier_schedule(plan, 16U), proposed, embedding_scale
    );
    const auto verified_logits = f32_values(verified.logits);
    float maximum_error = 0.0F;
    float maximum_softcapped_error = 0.0F;
    for (std::size_t token = 0; token < first_logits.size(); ++token) {
        maximum_error = std::max(maximum_error, std::abs(verified_logits[token] - first_logits[token]));
        maximum_error = std::max(maximum_error,
            std::abs(verified_logits[first_logits.size() + token] - second_logits[token]));
        maximum_softcapped_error = std::max(maximum_softcapped_error, std::abs(
            softcap(verified_logits[token], config.final_logit_softcap)
                - softcap(first_logits[token], config.final_logit_softcap)
        ));
        maximum_softcapped_error = std::max(maximum_softcapped_error, std::abs(
            softcap(verified_logits[first_logits.size() + token], config.final_logit_softcap)
                - softcap(second_logits[token], config.final_logit_softcap)
        ));
    }
    bool unchanged = true;
    for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
        unchanged = unchanged
            && same_bytes(tensor::copy_tensor_from_backend(plan.kv_cache()[layer].key), sequential_keys[layer])
            && same_bytes(tensor::copy_tensor_from_backend(plan.kv_cache()[layer].value), sequential_values[layer]);
    }
    const auto commit = engine::commit_gemma2_metal_width_two_verification(plan, 2U);
    bool committed_matches = true;
    float maximum_kv_error = 0.0F;
    for (std::size_t layer = 0; layer < plan.kv_cache().size(); ++layer) {
        const auto actual_key = tensor::copy_tensor_from_backend(plan.kv_cache()[layer].key);
        const auto actual_value = tensor::copy_tensor_from_backend(plan.kv_cache()[layer].value);
        maximum_kv_error = std::max(maximum_kv_error, maximum_f16_error(actual_key, sequential_keys[layer]));
        maximum_kv_error = std::max(maximum_kv_error, maximum_f16_error(actual_value, sequential_values[layer]));
    }
    committed_matches = maximum_kv_error <= 0.02F;

    const std::array<std::span<const float>, 2> reference_rows {
        std::span<const float>(first_logits), std::span<const float>(second_logits),
    };
    const std::array<std::span<const float>, 2> candidate_rows {
        std::span<const float>(verified_logits).first(first_logits.size()),
        std::span<const float>(verified_logits).subspan(first_logits.size(), first_logits.size()),
    };
    double kl_sum = 0.0;
    double overlap_sum = 0.0;
    bool top1_matches = true;
    for (std::size_t row = 0; row < 2U; ++row) {
        kl_sum += kl_divergence(reference_rows[row], candidate_rows[row], config.final_logit_softcap);
        const auto reference_top = top40(reference_rows[row], config.final_logit_softcap);
        const auto candidate_top = top40(candidate_rows[row], config.final_logit_softcap);
        top1_matches = top1_matches && reference_top[0] == candidate_top[0];
        std::size_t overlap = 0U;
        for (const auto token : reference_top) {
            overlap += std::find(candidate_top.begin(), candidate_top.end(), token) != candidate_top.end();
        }
        overlap_sum += static_cast<double>(overlap) / 40.0;
    }
    const auto mean_kl = kl_sum / 2.0;
    const auto mean_overlap = overlap_sum / 2.0;

    const auto mean_decode = (static_cast<double>(first.execution.gpu_duration_ns)
        + static_cast<double>(second.execution.gpu_duration_ns)) / 2.0;
    const auto two_decode = static_cast<double>(first.execution.gpu_duration_ns)
        + static_cast<double>(second.execution.gpu_duration_ns);
    const auto ratio = static_cast<double>(verified.execution.gpu_duration_ns) / mean_decode;
    return WidthTwoVerifierBenchmarkResult {
        .model_path = model_path,
        .first_decode_gpu_ns = first.execution.gpu_duration_ns,
        .second_decode_gpu_ns = second.execution.gpu_duration_ns,
        .verifier_gpu_ns = verified.execution.gpu_duration_ns,
        .commit_two_gpu_ns = commit.execution.gpu_duration_ns,
        .verifier_to_mean_decode_ratio = ratio,
        .verifier_to_two_decode_ratio = static_cast<double>(verified.execution.gpu_duration_ns) / two_decode,
        .maximum_logit_absolute_error = maximum_error,
        .maximum_softcapped_logit_absolute_error = maximum_softcapped_error,
        .mean_kl_divergence = mean_kl,
        .mean_top40_overlap = mean_overlap,
        .top1_agreement = top1_matches,
        .maximum_committed_kv_absolute_error = maximum_kv_error,
        .verifier_scratch_bytes = verifier_scratch_bytes(plan.verifier_scratch()),
        .resident_cache_unchanged_before_commit = unchanged,
        .committed_cache_matches_sequential = committed_matches,
        .gate_pass = ratio <= 1.35 && unchanged && committed_matches
            && mean_kl <= 0.0001 && mean_overlap >= 0.975 && top1_matches,
    };
}

}  // namespace local_gguf_engine::benchmarks
