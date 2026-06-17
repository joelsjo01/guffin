#include "sampling_tests.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

using local_gguf_engine::engine::sample_argmax_token_f32;
using local_gguf_engine::engine::sample_token_f32;
using local_gguf_engine::engine::SamplingConfig;
using local_gguf_engine::engine::SamplingState;
using local_gguf_engine::engine::top_logit_tokens_f32;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::TensorError;
using local_gguf_engine::tensor::TensorErrorCode;
using local_gguf_engine::tensor::TensorStorage;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;
using local_gguf_engine::tests::tensor_f32_matrix;
using local_gguf_engine::tests::tensor_f32_vector;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;

void test_samples_argmax_token_from_f32_logits_vector() {
    const std::array<float, 4> logits_values {0.5F, 3.0F, 3.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));

    const auto token = sample_argmax_token_f32(logits);

    expect(token == 1, "argmax sampler should pick the first max token");
}

void test_samples_argmax_token_from_last_logits_row() {
    const std::array<float, 6> logits_values {
        0.0F, 1.0F, 9.0F,
        4.0F, 3.0F, 4.0F,
    };
    const auto logits = tensor_f32_matrix(2, 3, std::span<const float>(logits_values));

    const auto token = sample_argmax_token_f32(logits);

    expect(token == 0, "argmax sampler should use the last logits row and first-max tie break");
}

void test_selects_top_logit_tokens_from_last_row() {
    const std::array<float, 8> logits_values {
        9.0F, 0.0F, 1.0F, 2.0F,
        4.0F, 5.0F, 5.0F, 1.0F,
    };
    const auto logits = tensor_f32_matrix(2, 4, std::span<const float>(logits_values));

    const auto top_logits = top_logit_tokens_f32(logits, 3);

    expect(top_logits.size() == 3, "top logits count mismatch");
    expect(top_logits[0].token == 1, "top logits first token mismatch");
    expect(top_logits[1].token == 2, "top logits tie should prefer lower token id");
    expect(top_logits[2].token == 0, "top logits third token mismatch");
    expect_close(top_logits[0].logit, 5.0F, 0.000001F, "top logits first value mismatch");
}

void test_rejects_argmax_sampler_dtype_mismatch() {
    const auto logits = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 3}), DType::f16));

    try {
        (void)sample_argmax_token_f32(logits);
        throw TestFailure("argmax sampler should reject F16 logits");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "argmax sampler dtype error mismatch");
        expect(error.expected_value() == 0, "argmax sampler expected dtype mismatch");
        expect(error.actual_value() == 1, "argmax sampler actual dtype mismatch");
    }
}

void test_rejects_argmax_sampler_rank_above_two() {
    const auto logits = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 1, 2}), DType::f32));

    try {
        (void)sample_argmax_token_f32(logits);
        throw TestFailure("argmax sampler should reject logits rank above two");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "argmax sampler rank error mismatch");
        expect(error.expected_value() == 2, "argmax sampler expected rank mismatch");
        expect(error.actual_value() == 3, "argmax sampler actual rank mismatch");
    }
}

void test_samples_temperature_top_k_token_from_f32_logits_vector() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState first_state(sampling.seed);
    SamplingState second_state(sampling.seed);

    const auto first_token = sample_token_f32(logits, sampling, first_state);
    const auto second_token = sample_token_f32(logits, sampling, second_state);

    expect(first_token == 1, "temperature top-k sampler should draw the second top-k candidate for seed 0");
    expect(second_token == first_token, "temperature top-k sampler should be deterministic for the same seed");
}

void test_samples_temperature_top_k_token_from_last_logits_row() {
    const std::array<float, 6> logits_values {
        100.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 2.0F,
    };
    const auto logits = tensor_f32_matrix(2, 3, std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state);

    expect(token == 1, "temperature top-k sampler should use the final logits row");
}

void test_samples_temperature_top_p_token_from_f32_logits_vector() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.5F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state);

    expect(token == 2, "temperature top-p sampler should keep at least the highest-probability token");
}

void test_samples_temperature_repetition_penalty_token_from_f32_logits_vector() {
    const std::array<float, 3> logits_values {0.0F, 4.0F, 5.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const std::array<std::uint32_t, 1> repetition_tokens {2};
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.01F,
        .repetition_penalty = 2.0F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state, std::span<const std::uint32_t>(repetition_tokens));

    expect(token == 1, "repetition penalty should lower a repeated positive-logit token before top-p filtering");
}

void test_sampling_temperature_zero_keeps_argmax_path() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 0.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    const auto token = sample_token_f32(logits, sampling, state);

    expect(token == 2, "zero-temperature sampler should preserve argmax behavior");
}

void test_rejects_sampling_negative_temperature() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = -1.0F,
        .top_k = 2,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    try {
        (void)sample_token_f32(logits, sampling, state);
        throw TestFailure("temperature sampler should reject negative temperature");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling temperature must be zero or a positive finite value",
            "temperature sampler error text mismatch"
        );
    }
}

void test_rejects_sampling_invalid_top_p() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .top_p = 0.0F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    try {
        (void)sample_token_f32(logits, sampling, state);
        throw TestFailure("temperature sampler should reject invalid top-p");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling top-p must be in (0, 1]",
            "top-p sampler error text mismatch"
        );
    }
}

void test_rejects_sampling_invalid_repetition_penalty() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const SamplingConfig sampling {
        .temperature = 1.0F,
        .repetition_penalty = 0.5F,
        .seed = 0,
    };
    SamplingState state(sampling.seed);

    try {
        (void)sample_token_f32(logits, sampling, state);
        throw TestFailure("temperature sampler should reject invalid repetition penalty");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling repetition penalty must be at least 1",
            "repetition-penalty sampler error text mismatch"
        );
    }
}

const TestCase kSamplingTests[] = {
    {"samples argmax token from F32 logits vector", test_samples_argmax_token_from_f32_logits_vector},
    {"samples argmax token from last logits row", test_samples_argmax_token_from_last_logits_row},
    {"selects top logit tokens from last row", test_selects_top_logit_tokens_from_last_row},
    {"rejects argmax sampler dtype mismatch", test_rejects_argmax_sampler_dtype_mismatch},
    {"rejects argmax sampler rank above two", test_rejects_argmax_sampler_rank_above_two},
    {
        "samples temperature top-k token from F32 logits vector",
        test_samples_temperature_top_k_token_from_f32_logits_vector,
    },
    {
        "samples temperature top-k token from last logits row",
        test_samples_temperature_top_k_token_from_last_logits_row,
    },
    {
        "samples temperature top-p token from F32 logits vector",
        test_samples_temperature_top_p_token_from_f32_logits_vector,
    },
    {
        "samples temperature repetition-penalty token from F32 logits vector",
        test_samples_temperature_repetition_penalty_token_from_f32_logits_vector,
    },
    {"sampling temperature zero keeps argmax path", test_sampling_temperature_zero_keeps_argmax_path},
    {"rejects sampling negative temperature", test_rejects_sampling_negative_temperature},
    {"rejects sampling invalid top-p", test_rejects_sampling_invalid_top_p},
    {"rejects sampling invalid repetition penalty", test_rejects_sampling_invalid_repetition_penalty},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> sampling_test_cases() {
    return kSamplingTests;
}

}  // namespace local_gguf_engine::tests
