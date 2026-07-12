#include "sampling_tests.hpp"
#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/sampling.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {

using local_gguf_engine::engine::top_logit_tokens_f32;
using local_gguf_engine::engine::TokenSampler;
using local_gguf_engine::engine::TokenSamplerOptions;
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

void test_selects_top_logit_tokens_from_vector() {
    const std::array<float, 4> logits_values {0.5F, 3.0F, 3.0F, 2.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));

    const auto top_logits = top_logit_tokens_f32(logits, 2);

    expect(top_logits.size() == 2, "top logits vector count mismatch");
    expect(top_logits[0].token == 1, "top logits vector first token mismatch");
    expect(top_logits[1].token == 2, "top logits vector tie token mismatch");
    expect_close(top_logits[0].logit, 3.0F, 0.000001F, "top logits vector first value mismatch");
}

void test_selects_top_logit_tokens_from_last_row() {
    const std::array<float, 8> logits_values {
        9.0F, 0.0F, 1.0F, 2.0F,
        4.0F, 5.0F, 5.0F, 1.0F,
    };
    const auto logits = tensor_f32_matrix(2, 4, std::span<const float>(logits_values));

    const auto top_logits = top_logit_tokens_f32(logits, 3);

    expect(top_logits.size() == 3, "top logits matrix count mismatch");
    expect(top_logits[0].token == 1, "top logits matrix first token mismatch");
    expect(top_logits[1].token == 2, "top logits matrix tie should prefer lower token id");
    expect(top_logits[2].token == 0, "top logits matrix third token mismatch");
    expect_close(top_logits[0].logit, 5.0F, 0.000001F, "top logits matrix first value mismatch");
}

void test_top_logit_tokens_allows_full_vocab_request() {
    const std::array<float, 3> logits_values {0.0F, 2.0F, 1.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));

    const auto top_logits = top_logit_tokens_f32(logits, 8);

    expect(top_logits.size() == 3, "top logits full-vocab count mismatch");
    expect(top_logits[0].token == 1, "top logits full-vocab first token mismatch");
    expect(top_logits[1].token == 2, "top logits full-vocab second token mismatch");
    expect(top_logits[2].token == 0, "top logits full-vocab third token mismatch");
}

void test_rejects_top_logit_tokens_dtype_mismatch() {
    const auto logits = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 3}), DType::f16));

    try {
        (void)top_logit_tokens_f32(logits, 1);
        throw TestFailure("top logits should reject F16 logits");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::unsupported_operator_dtype, "top logits dtype error mismatch");
        expect(error.expected_value() == 0, "top logits expected dtype mismatch");
        expect(error.actual_value() == 1, "top logits actual dtype mismatch");
    }
}

void test_rejects_top_logit_tokens_rank_above_two() {
    const auto logits = TensorStorage::allocate(TensorDescriptor::from(Shape::from({1, 1, 2}), DType::f32));

    try {
        (void)top_logit_tokens_f32(logits, 1);
        throw TestFailure("top logits should reject logits rank above two");
    } catch (const TensorError& error) {
        expect(error.code() == TensorErrorCode::invalid_shape_rank, "top logits rank error mismatch");
        expect(error.expected_value() == 2, "top logits expected rank mismatch");
        expect(error.actual_value() == 3, "top logits actual rank mismatch");
    }
}

void test_rejects_top_logit_tokens_nonfinite_logit() {
    const std::array<float, 3> logits_values {0.0F, 1.0F, std::numeric_limits<float>::infinity()};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));

    try {
        (void)top_logit_tokens_f32(logits, 1);
        throw TestFailure("top logits should reject non-finite logits");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string_view(error.what()) == "sampling logits must be finite",
            "top logits non-finite error text mismatch"
        );
    }
}

void test_seeded_token_sampler_is_reproducible_and_respects_top_k() {
    const std::array<float, 4> logits_values {1.0F, 1.0F, 1.0F, 1.0F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    const auto options = TokenSamplerOptions {
        .temperature = 1.0F,
        .top_k = 2,
        .top_p = 1.0F,
        .seed = 123,
    };
    auto first = TokenSampler(options);
    auto second = TokenSampler(options);

    for (std::size_t index = 0; index < 16U; ++index) {
        const auto first_token = first.sample(logits);
        const auto second_token = second.sample(logits);
        expect(first_token == second_token, "seeded token sampler sequence mismatch");
        expect(first_token < 2U, "top-k sampler selected filtered token");
    }
}

void test_token_sampler_snapshot_replays_rng_and_history() {
    const std::array<float, 3> logits_values {1.0F, 0.9F, 0.8F};
    const auto logits = tensor_f32_vector(std::span<const float>(logits_values));
    auto sampler = TokenSampler(TokenSamplerOptions {
        .temperature = 0.8F,
        .top_k = 3,
        .top_p = 1.0F,
        .repetition_penalty = 1.2F,
        .seed = 9,
    });
    const std::array<std::uint32_t, 2> history {0U, 1U};
    sampler.observe_tokens(std::span<const std::uint32_t>(history));
    const auto snapshot = sampler;

    const auto selected = sampler.sample(logits);
    auto replay = snapshot;
    expect(replay.sample(logits) == selected, "token sampler snapshot should replay selection");
}

void test_rejects_invalid_token_sampler_options() {
    const std::array<TokenSamplerOptions, 3> invalid_options {{
        TokenSamplerOptions {.temperature = -1.0F},
        TokenSamplerOptions {.temperature = 1.0F, .top_p = 0.0F},
        TokenSamplerOptions {.temperature = 1.0F, .repetition_penalty = 0.0F},
    }};
    for (const auto& options : invalid_options) {
        bool rejected = false;
        try {
            static_cast<void>(TokenSampler(options));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "invalid token sampler options should fail");
    }
}

const TestCase kSamplingTests[] = {
    {"selects top logit tokens from vector", test_selects_top_logit_tokens_from_vector},
    {"selects top logit tokens from last row", test_selects_top_logit_tokens_from_last_row},
    {"top logit tokens allows full vocab request", test_top_logit_tokens_allows_full_vocab_request},
    {"rejects top logit tokens dtype mismatch", test_rejects_top_logit_tokens_dtype_mismatch},
    {"rejects top logit tokens rank above two", test_rejects_top_logit_tokens_rank_above_two},
    {"rejects top logit tokens nonfinite logit", test_rejects_top_logit_tokens_nonfinite_logit},
    {"seeded token sampler is reproducible and respects top-k", test_seeded_token_sampler_is_reproducible_and_respects_top_k},
    {"token sampler snapshot replays RNG and history", test_token_sampler_snapshot_replays_rng_and_history},
    {"rejects invalid token sampler options", test_rejects_invalid_token_sampler_options},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> sampling_test_cases() {
    return kSamplingTests;
}

}  // namespace local_gguf_engine::tests
