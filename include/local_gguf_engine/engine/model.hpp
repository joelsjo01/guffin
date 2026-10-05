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
    invalid_metadata_value,
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
    gemma3,
};

enum class GemmaKVPrecision { automatic, f32, f16 };

[[nodiscard]] GemmaKVPrecision resolve_gemma_kv_precision(
    ModelArchitecture architecture, GemmaKVPrecision requested
);
[[nodiscard]] std::string_view gemma_kv_precision_name(GemmaKVPrecision precision) noexcept;

struct GemmaModelConfig {
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

    // Shared resident execution policy. The shared configuration is retained
    // so Gemma 2 cache, chat and transaction ownership remain one foundation.
    ModelArchitecture architecture = ModelArchitecture::gemma2;
    // Zero on legacy hand-built Gemma 2 fixtures means context_length.
    std::uint32_t model_context_length = 0;
    std::uint32_t attention_global_layer_period = 2;
    float rope_freq_base = 10000.0F;
    float local_rope_freq_base = 10000.0F;
    float rope_freq_scale = 1.0F;
    bool uses_qk_norm = false;
    // Source metadata leaves this automatic, resolving to F16 for both Gemma
    // families. Gemma 3 also supports explicit F32. A resident plan owns one
    // resolved precision for durable, current-chunk, private and rollback K/V.
    GemmaKVPrecision kv_precision = GemmaKVPrecision::automatic;

    constexpr bool operator==(const GemmaModelConfig&) const = default;
};

[[nodiscard]] float gemma_input_embedding_scale(std::uint32_t embedding_length);

[[nodiscard]] bool gemma_layer_uses_sliding_window(
    const GemmaModelConfig& config, std::uint32_t layer_index
) noexcept;
[[nodiscard]] float gemma_layer_rope_freq_base(
    const GemmaModelConfig& config, std::uint32_t layer_index
) noexcept;
[[nodiscard]] float gemma_layer_rope_freq_scale(
    const GemmaModelConfig& config, std::uint32_t layer_index
) noexcept;
[[nodiscard]] tensor::DType gemma_kv_dtype(const GemmaModelConfig& config);

struct EngineTensorView {
    std::string name;
    tensor::TensorView view;
};

inline constexpr std::uint32_t engine_model_payload_receipt_version = 1U;

// Computed for authenticated speculative plans. Ordinary target-only loading
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

// Identity of the actual uploaded target weights, including any prepared
// output-head conversion. This is separate from the source GGUF identity.
struct ResidentTargetPayloadReceipt {
    std::uint32_t version = 1U;
    std::uint32_t allocation_count = 0U;
    std::uint64_t byte_count = 0U;
    runtime::Sha256Digest bytes;

    constexpr bool operator==(const ResidentTargetPayloadReceipt&) const = default;
};

class EngineModel final {
public:
    EngineModel(
        ModelArchitecture architecture,
        std::optional<std::string> name,
        GemmaModelConfig gemma_config,
        std::vector<EngineTensorView> tensor_views,
        std::optional<EngineModelPayloadReceipt> payload_receipt = std::nullopt,
        std::optional<runtime::Sha256Digest> tokenizer_identity = std::nullopt
    );

    [[nodiscard]] ModelArchitecture architecture() const noexcept;
    [[nodiscard]] const std::optional<std::string>& name() const noexcept;
    [[nodiscard]] const GemmaModelConfig& gemma_config() const noexcept;
    [[nodiscard]] const std::vector<EngineTensorView>& tensor_views() const noexcept;
    [[nodiscard]] const std::optional<EngineModelPayloadReceipt>& payload_receipt() const noexcept;
    [[nodiscard]] const std::optional<runtime::Sha256Digest>& tokenizer_identity() const noexcept;
    [[nodiscard]] const EngineTensorView* find_tensor_view(std::string_view name) const noexcept;

private:
    ModelArchitecture architecture_;
    std::optional<std::string> name_;
    GemmaModelConfig gemma_config_;
    std::vector<EngineTensorView> tensor_views_;
    std::optional<EngineModelPayloadReceipt> payload_receipt_;
    // Standalone plans bind the actual source tokenizer independently of the
    // target tensor and metadata receipt.
    std::optional<runtime::Sha256Digest> tokenizer_identity_;
};

[[nodiscard]] std::string_view model_architecture_name(ModelArchitecture architecture) noexcept;
[[nodiscard]] EngineModel load_engine_model_from_gguf(const model_format::GGUFModelView& model);
[[nodiscard]] EngineModel load_verified_engine_model_from_gguf(
    const model_format::GGUFModelView& model,
    std::span<const std::uint8_t> whole_gguf_bytes
);

}  // namespace local_gguf_engine::engine
