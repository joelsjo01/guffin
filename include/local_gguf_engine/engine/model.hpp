#pragma once

#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"
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
    gemma2,
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

struct EngineTensorView {
    std::string name;
    tensor::TensorView view;
};

inline constexpr std::uint32_t engine_model_payload_receipt_version = 1U;

// Computed only for prepared learned-mode plans. Ordinary target-only loading
// deliberately avoids hashing a multi-gigabyte GGUF on every session start.
struct EngineModelPayloadReceipt {
    std::uint32_t version = engine_model_payload_receipt_version;
    std::uint64_t whole_gguf_byte_count = 0U;
    std::uint32_t source_tensor_count = 0U;
    runtime::Sha256Digest whole_gguf;
    runtime::Sha256Digest canonical_consumed_metadata;
    runtime::Sha256Digest source_tensors;

    constexpr bool operator==(const EngineModelPayloadReceipt&) const = default;
};

class EngineModel final {
public:
    EngineModel(
        ModelArchitecture architecture,
        std::optional<std::string> name,
        Gemma2ModelConfig gemma2_config,
        std::vector<EngineTensorView> tensor_views,
        std::optional<EngineModelPayloadReceipt> payload_receipt = std::nullopt
    );

    [[nodiscard]] ModelArchitecture architecture() const noexcept;
    [[nodiscard]] const std::optional<std::string>& name() const noexcept;
    [[nodiscard]] const Gemma2ModelConfig& gemma2_config() const noexcept;
    [[nodiscard]] const std::vector<EngineTensorView>& tensor_views() const noexcept;
    [[nodiscard]] const std::optional<EngineModelPayloadReceipt>& payload_receipt() const noexcept;
    [[nodiscard]] const EngineTensorView* find_tensor_view(std::string_view name) const noexcept;

private:
    ModelArchitecture architecture_;
    std::optional<std::string> name_;
    Gemma2ModelConfig gemma2_config_;
    std::vector<EngineTensorView> tensor_views_;
    std::optional<EngineModelPayloadReceipt> payload_receipt_;
};

[[nodiscard]] std::string_view model_architecture_name(ModelArchitecture architecture) noexcept;
[[nodiscard]] EngineModel load_engine_model_from_gguf(const model_format::GGUFModelView& model);
[[nodiscard]] EngineModel load_verified_engine_model_from_gguf(
    const model_format::GGUFModelView& model,
    std::span<const std::uint8_t> whole_gguf_bytes
);

}  // namespace local_gguf_engine::engine
