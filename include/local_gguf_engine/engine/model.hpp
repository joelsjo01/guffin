#pragma once

#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::engine {

enum class EngineModelErrorCode {
    missing_metadata,
    invalid_metadata_type,
    unsupported_architecture,
    duplicate_tensor_name,
    missing_tensor,
    invalid_tensor,
};

class EngineModelError final : public std::runtime_error {
public:
    EngineModelError(EngineModelErrorCode code, std::string detail);

    [[nodiscard]] EngineModelErrorCode code() const noexcept;
    [[nodiscard]] const std::string& detail() const noexcept;

private:
    EngineModelErrorCode code_;
    std::string detail_;
};

enum class ModelArchitecture {
    llama,
    gemma2,
};

struct LlamaModelConfig {
    std::uint32_t context_length = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t feed_forward_length = 0;
    std::uint32_t attention_head_count = 0;
    std::uint32_t attention_head_count_kv = 0;
    std::uint32_t attention_key_length = 0;
    std::uint32_t attention_value_length = 0;
    std::uint32_t rope_dimension_count = 0;
    std::uint32_t vocab_size = 0;
    float rope_freq_base = 0.0F;
    float rms_norm_epsilon = 0.0F;

    constexpr bool operator==(const LlamaModelConfig&) const = default;
};

struct Gemma2ModelConfig {
    std::uint32_t context_length = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t feed_forward_length = 0;
    std::uint32_t attention_head_count = 0;
    std::uint32_t attention_head_count_kv = 0;
    std::uint32_t attention_key_length = 0;
    std::uint32_t attention_value_length = 0;
    std::uint32_t sliding_window = 0;
    std::uint32_t vocab_size = 0;
    float rms_norm_epsilon = 0.0F;
    float attention_logit_softcap = 0.0F;
    float final_logit_softcap = 0.0F;
    bool uses_tied_output_weight = false;

    constexpr bool operator==(const Gemma2ModelConfig&) const = default;
};

struct EngineTensor {
    std::string name;
    tensor::TensorStorage storage;
};

struct EngineTensorView {
    std::string name;
    tensor::TensorView view;
};

class EngineModel final {
public:
    EngineModel(
        ModelArchitecture architecture,
        std::optional<std::string> name,
        LlamaModelConfig llama_config,
        std::vector<EngineTensor> tensors
    );
    EngineModel(
        ModelArchitecture architecture,
        std::optional<std::string> name,
        LlamaModelConfig llama_config,
        Gemma2ModelConfig gemma2_config,
        std::vector<EngineTensor> tensors
    );
    EngineModel(
        ModelArchitecture architecture,
        std::optional<std::string> name,
        LlamaModelConfig llama_config,
        Gemma2ModelConfig gemma2_config,
        std::vector<EngineTensor> tensors,
        std::vector<EngineTensorView> tensor_views
    );

    [[nodiscard]] ModelArchitecture architecture() const noexcept;
    [[nodiscard]] const std::optional<std::string>& name() const noexcept;
    [[nodiscard]] const LlamaModelConfig& llama_config() const noexcept;
    [[nodiscard]] const Gemma2ModelConfig& gemma2_config() const noexcept;
    [[nodiscard]] const std::vector<EngineTensor>& tensors() const noexcept;
    [[nodiscard]] const std::vector<EngineTensorView>& tensor_views() const noexcept;
    [[nodiscard]] const EngineTensor* find_tensor(std::string_view name) const noexcept;
    [[nodiscard]] const EngineTensorView* find_tensor_view(std::string_view name) const noexcept;

private:
    ModelArchitecture architecture_;
    std::optional<std::string> name_;
    LlamaModelConfig llama_config_;
    Gemma2ModelConfig gemma2_config_;
    std::vector<EngineTensor> tensors_;
    std::vector<EngineTensorView> tensor_views_;
};

[[nodiscard]] std::string_view model_architecture_name(ModelArchitecture architecture) noexcept;
[[nodiscard]] EngineModel load_engine_model_from_gguf(const model_format::GGUFModelView& model);

}  // namespace local_gguf_engine::engine
