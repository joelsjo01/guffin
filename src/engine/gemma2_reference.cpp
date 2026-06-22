#include "local_gguf_engine/engine/gemma2_reference.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace local_gguf_engine::engine {

namespace {

void require_non_empty_span(std::span<const float> values, const char* name) {
    if (values.empty()) {
        throw std::invalid_argument(name);
    }
}

void require_same_size(std::span<const float> left, std::span<const float> right, const char* name) {
    if (left.size() != right.size()) {
        throw std::invalid_argument(name);
    }
}

void require_positive(float value, const char* name) {
    if (!(value > 0.0F)) {
        throw std::invalid_argument(name);
    }
}

}  // namespace

float gemma2_input_embedding_scale(std::uint32_t embedding_length) {
    if (embedding_length == 0) {
        throw std::invalid_argument("embedding_length");
    }

    return std::sqrt(static_cast<float>(embedding_length));
}

bool gemma2_layer_uses_sliding_window(std::uint32_t layer_index) noexcept {
    return layer_index % 2U == 0U;
}

float gemma2_softcap_value(float value, float softcap) {
    require_positive(softcap, "softcap");
    return std::tanh(value / softcap) * softcap;
}

std::vector<float> gemma2_softcap_f32(std::span<const float> values, float softcap) {
    require_positive(softcap, "softcap");

    std::vector<float> output;
    output.reserve(values.size());
    for (const auto value : values) {
        output.push_back(gemma2_softcap_value(value, softcap));
    }

    return output;
}

std::vector<float> gemma2_rms_norm_f32(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon
) {
    require_non_empty_span(input, "input");
    require_same_size(input, weight, "weight");
    require_positive(epsilon, "epsilon");

    const auto sum_squares = std::inner_product(input.begin(), input.end(), input.begin(), 0.0F);
    const auto mean_square = sum_squares / static_cast<float>(input.size());
    const auto inverse_rms = 1.0F / std::sqrt(mean_square + epsilon);

    std::vector<float> output;
    output.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output.push_back(input[index] * inverse_rms * weight[index]);
    }

    return output;
}

std::vector<float> gemma2_post_norm_residual_f32(
    std::span<const float> residual,
    std::span<const float> branch_output,
    std::span<const float> post_norm_weight,
    float epsilon
) {
    require_same_size(residual, branch_output, "branch_output");
    auto normalized = gemma2_rms_norm_f32(branch_output, post_norm_weight, epsilon);
    for (std::size_t index = 0; index < normalized.size(); ++index) {
        normalized[index] += residual[index];
    }

    return normalized;
}

}  // namespace local_gguf_engine::engine
