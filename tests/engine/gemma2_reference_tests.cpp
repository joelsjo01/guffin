#include "gemma2_reference_tests.hpp"

#include "tensor_test_support.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"

#include <array>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {

using local_gguf_engine::engine::gemma2_input_embedding_scale;
using local_gguf_engine::engine::gemma2_layer_uses_sliding_window;
using local_gguf_engine::engine::gemma2_post_norm_residual_f32;
using local_gguf_engine::engine::gemma2_rms_norm_f32;
using local_gguf_engine::engine::gemma2_softcap_f32;
using local_gguf_engine::engine::gemma2_softcap_value;
using local_gguf_engine::tests::TestCase;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::expect_close;

void test_computes_gemma2_embedding_scale() {
    expect_close(gemma2_input_embedding_scale(3584), 59.86652F, 0.0001F, "Gemma2 embedding scale mismatch");
    expect_close(gemma2_input_embedding_scale(9), 3.0F, 0.0F, "Gemma2 tiny embedding scale mismatch");
}

void test_marks_even_layers_as_sliding_window() {
    expect(gemma2_layer_uses_sliding_window(0), "Gemma2 layer 0 should use sliding window");
    expect(!gemma2_layer_uses_sliding_window(1), "Gemma2 layer 1 should use full attention");
    expect(gemma2_layer_uses_sliding_window(2), "Gemma2 layer 2 should use sliding window");
}

void test_applies_gemma2_softcap() {
    const std::array<float, 3> values {-4.0F, 0.0F, 4.0F};
    const auto capped = gemma2_softcap_f32(std::span<const float>(values), 2.0F);

    expect_close(gemma2_softcap_value(3.0F, 2.0F), 1.81030F, 0.00001F, "Gemma2 scalar softcap mismatch");
    expect_close(capped[0], -1.92806F, 0.00001F, "Gemma2 softcap negative mismatch");
    expect_close(capped[1], 0.0F, 0.0F, "Gemma2 softcap zero mismatch");
    expect_close(capped[2], 1.92806F, 0.00001F, "Gemma2 softcap positive mismatch");
}

void test_applies_gemma2_multiplicative_rms_norm() {
    const std::array<float, 2> input {3.0F, 4.0F};
    const std::array<float, 2> weight {0.5F, 2.0F};
    const auto output = gemma2_rms_norm_f32(std::span<const float>(input), std::span<const float>(weight), 0.000001F);

    expect_close(output[0], 0.424264F, 0.00001F, "Gemma2 RMSNorm value 0 mismatch");
    expect_close(output[1], 2.26274F, 0.00001F, "Gemma2 RMSNorm value 1 mismatch");
}

void test_applies_gemma2_post_norm_before_residual_add() {
    const std::array<float, 2> residual {10.0F, -10.0F};
    const std::array<float, 2> branch {3.0F, 4.0F};
    const std::array<float, 2> post_norm_weight {1.0F, 1.0F};
    const auto output = gemma2_post_norm_residual_f32(
        std::span<const float>(residual),
        std::span<const float>(branch),
        std::span<const float>(post_norm_weight),
        0.000001F
    );

    expect_close(output[0], 10.84853F, 0.00001F, "Gemma2 post-norm residual value 0 mismatch");
    expect_close(output[1], -8.86863F, 0.00001F, "Gemma2 post-norm residual value 1 mismatch");
}

void test_rejects_invalid_gemma2_reference_shapes() {
    const std::array<float, 2> input {1.0F, 2.0F};
    const std::array<float, 1> weight {1.0F};

    try {
        (void)gemma2_rms_norm_f32(std::span<const float>(input), std::span<const float>(weight), 0.000001F);
        throw TestFailure("Gemma2 RMSNorm should reject weight size mismatch");
    } catch (const std::invalid_argument& error) {
        expect(std::string_view(error.what()) == "weight", "Gemma2 RMSNorm mismatch detail mismatch");
    }
}

void test_rejects_nonpositive_gemma2_reference_parameters() {
    const std::array<float, 1> input {1.0F};
    const std::array<float, 1> weight {1.0F};

    try {
        (void)gemma2_rms_norm_f32(std::span<const float>(input), std::span<const float>(weight), 0.0F);
        throw TestFailure("Gemma2 RMSNorm should reject nonpositive epsilon");
    } catch (const std::invalid_argument& error) {
        expect(std::string_view(error.what()) == "epsilon", "Gemma2 epsilon detail mismatch");
    }

    try {
        (void)gemma2_softcap_value(1.0F, 0.0F);
        throw TestFailure("Gemma2 softcap should reject nonpositive cap");
    } catch (const std::invalid_argument& error) {
        expect(std::string_view(error.what()) == "softcap", "Gemma2 softcap detail mismatch");
    }
}

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gemma2_reference_test_cases() {
    static constexpr TestCase tests[] = {
        {"computes Gemma2 embedding scale", test_computes_gemma2_embedding_scale},
        {"marks even Gemma2 layers as sliding window", test_marks_even_layers_as_sliding_window},
        {"applies Gemma2 softcap", test_applies_gemma2_softcap},
        {"applies Gemma2 multiplicative RMSNorm", test_applies_gemma2_multiplicative_rms_norm},
        {
            "applies Gemma2 post-norm before residual add",
            test_applies_gemma2_post_norm_before_residual_add,
        },
        {"rejects invalid Gemma2 reference shapes", test_rejects_invalid_gemma2_reference_shapes},
        {"rejects nonpositive Gemma2 reference parameters", test_rejects_nonpositive_gemma2_reference_parameters},
    };

    return tests;
}

}  // namespace local_gguf_engine::tests
