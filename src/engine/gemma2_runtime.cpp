#include "local_gguf_engine/engine/gemma2_runtime.hpp"

#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/tensor/ops/embedding.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

namespace {

[[nodiscard]] std::uint32_t checked_prompt_token_count(std::span<const std::uint32_t> prompt_token_ids) {
    if (prompt_token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Gemma2 Metal runtime shell prompt token count exceeds uint32");
    }

    return static_cast<std::uint32_t>(prompt_token_ids.size());
}

}  // namespace

Gemma2MetalGenerationRuntimeShell create_gemma2_metal_generation_runtime_shell(
    const EngineModel& model,
    std::span<const std::uint32_t> prompt_token_ids
) {
    auto plan = create_gemma2_metal_model_plan(model);
    const auto prompt_token_count = checked_prompt_token_count(prompt_token_ids);
    auto prefill = create_gemma2_metal_prefill_schedule(plan, prompt_token_count);
    auto first_decode = create_gemma2_metal_decode_schedule(plan, prompt_token_count);
    const auto embedding_scale = gemma2_input_embedding_scale(plan.config().embedding_length);
    auto prefill_input = Gemma2MetalInputActivations {
        .token_ids = std::vector<std::uint32_t>(prompt_token_ids.begin(), prompt_token_ids.end()),
        .hidden_states = tensor::ops::embed_f16_gguf_weight_token_ids_to_backend(
            prompt_token_ids,
            plan.weights().token_embedding_weight,
            embedding_scale
        ),
        .embedding_scale = embedding_scale,
    };

    return Gemma2MetalGenerationRuntimeShell {
        .plan = std::move(plan),
        .prefill = std::move(prefill),
        .first_decode = std::move(first_decode),
        .prefill_input = std::move(prefill_input),
    };
}

}  // namespace local_gguf_engine::engine
