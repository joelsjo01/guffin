#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

[[nodiscard]] float gemma2_input_embedding_scale(std::uint32_t embedding_length);
[[nodiscard]] bool gemma2_layer_uses_sliding_window(std::uint32_t layer_index) noexcept;
[[nodiscard]] float gemma2_softcap_value(float value, float softcap);
[[nodiscard]] std::vector<float> gemma2_softcap_f32(std::span<const float> values, float softcap);
[[nodiscard]] std::vector<float> gemma2_rms_norm_f32(
    std::span<const float> input,
    std::span<const float> weight,
    float epsilon
);
[[nodiscard]] std::vector<float> gemma2_post_norm_residual_f32(
    std::span<const float> residual,
    std::span<const float> branch_output,
    std::span<const float> post_norm_weight,
    float epsilon
);

}  // namespace local_gguf_engine::engine
