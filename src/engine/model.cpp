#include "local_gguf_engine/engine/model.hpp"

#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>

namespace local_gguf_engine::engine {

namespace {

using model_format::GGUFMetadataEntry;
using model_format::GGUFMetadataSection;
using model_format::GGUFMetadataValueType;
using model_format::load_gguf_tensor_payload;

[[nodiscard]] std::string error_message(EngineModelErrorCode code, std::string_view detail) {
    std::string message;
    switch (code) {
        case EngineModelErrorCode::missing_metadata:
            message = "missing model metadata";
            break;
        case EngineModelErrorCode::invalid_metadata_type:
            message = "invalid model metadata type";
            break;
        case EngineModelErrorCode::unsupported_architecture:
            message = "unsupported model architecture";
            break;
        case EngineModelErrorCode::duplicate_tensor_name:
            message = "duplicate model tensor name";
            break;
        case EngineModelErrorCode::missing_tensor:
            message = "missing model tensor";
            break;
        case EngineModelErrorCode::invalid_tensor:
            message = "invalid model tensor";
            break;
    }

    if (!detail.empty()) {
        message += ": ";
        message += detail;
    }

    return message;
}

[[nodiscard]] const GGUFMetadataEntry* find_metadata_entry(
    const GGUFMetadataSection& metadata,
    std::string_view key
) noexcept {
    const auto found = std::find_if(
        metadata.entries.begin(),
        metadata.entries.end(),
        [key](const GGUFMetadataEntry& entry) {
            return entry.key == key;
        }
    );

    if (found == metadata.entries.end()) {
        return nullptr;
    }

    return &*found;
}

[[nodiscard]] const GGUFMetadataEntry& require_metadata_entry(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        throw EngineModelError(EngineModelErrorCode::missing_metadata, std::string(key));
    }

    return *entry;
}

[[nodiscard]] std::string require_metadata_string(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto& entry = require_metadata_entry(metadata, key);
    if (entry.value.type != GGUFMetadataValueType::string
        || !std::holds_alternative<std::string>(entry.value.data)) {
        throw EngineModelError(EngineModelErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<std::string>(entry.value.data);
}

[[nodiscard]] std::optional<std::string> optional_metadata_string(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto* entry = find_metadata_entry(metadata, key);
    if (entry == nullptr) {
        return std::nullopt;
    }

    if (entry->value.type != GGUFMetadataValueType::string
        || !std::holds_alternative<std::string>(entry->value.data)) {
        throw EngineModelError(EngineModelErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<std::string>(entry->value.data);
}

[[nodiscard]] std::uint32_t require_metadata_u32(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto& entry = require_metadata_entry(metadata, key);
    if (entry.value.type != GGUFMetadataValueType::uint32
        || !std::holds_alternative<std::uint32_t>(entry.value.data)) {
        throw EngineModelError(EngineModelErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<std::uint32_t>(entry.value.data);
}

[[nodiscard]] float require_metadata_f32(
    const GGUFMetadataSection& metadata,
    std::string_view key
) {
    const auto& entry = require_metadata_entry(metadata, key);
    if (entry.value.type != GGUFMetadataValueType::float32
        || !std::holds_alternative<float>(entry.value.data)) {
        throw EngineModelError(EngineModelErrorCode::invalid_metadata_type, std::string(key));
    }

    return std::get<float>(entry.value.data);
}

[[nodiscard]] ModelArchitecture parse_architecture(const GGUFMetadataSection& metadata) {
    const auto architecture = require_metadata_string(metadata, "general.architecture");
    if (architecture == "llama") {
        return ModelArchitecture::llama;
    }

    if (architecture == "gemma2") {
        return ModelArchitecture::gemma2;
    }

    throw EngineModelError(EngineModelErrorCode::unsupported_architecture, architecture);
}

[[nodiscard]] LlamaModelConfig load_llama_config(const GGUFMetadataSection& metadata) {
    return LlamaModelConfig {
        .context_length = require_metadata_u32(metadata, "llama.context_length"),
        .embedding_length = require_metadata_u32(metadata, "llama.embedding_length"),
        .block_count = require_metadata_u32(metadata, "llama.block_count"),
        .feed_forward_length = require_metadata_u32(metadata, "llama.feed_forward_length"),
        .attention_head_count = require_metadata_u32(metadata, "llama.attention.head_count"),
        .attention_head_count_kv = require_metadata_u32(metadata, "llama.attention.head_count_kv"),
        .attention_key_length = require_metadata_u32(metadata, "llama.attention.key_length"),
        .attention_value_length = require_metadata_u32(metadata, "llama.attention.value_length"),
        .rope_dimension_count = require_metadata_u32(metadata, "llama.rope.dimension_count"),
        .vocab_size = require_metadata_u32(metadata, "llama.vocab_size"),
        .rope_freq_base = require_metadata_f32(metadata, "llama.rope.freq_base"),
        .rms_norm_epsilon = require_metadata_f32(metadata, "llama.attention.layer_norm_rms_epsilon"),
    };
}

[[nodiscard]] const model_format::GGUFTensorPayloadView* find_tensor_payload(
    const model_format::GGUFModelView& model,
    std::string_view name
) noexcept {
    const auto found = std::find_if(
        model.tensor_payloads.begin(),
        model.tensor_payloads.end(),
        [name](const model_format::GGUFTensorPayloadView& payload) {
            return payload.tensor.name == name;
        }
    );

    if (found == model.tensor_payloads.end()) {
        return nullptr;
    }

    return &*found;
}

[[nodiscard]] const model_format::GGUFTensorPayloadView& require_tensor_payload(
    const model_format::GGUFModelView& model,
    std::string_view name
) {
    const auto* payload = find_tensor_payload(model, name);
    if (payload == nullptr) {
        throw EngineModelError(EngineModelErrorCode::missing_tensor, std::string(name));
    }

    return *payload;
}

[[nodiscard]] std::uint32_t checked_tensor_dimension_u32(
    const model_format::GGUFTensorInfo& tensor,
    std::size_t dimension_index
) {
    if (dimension_index >= tensor.dimensions.size()) {
        throw EngineModelError(EngineModelErrorCode::invalid_tensor, tensor.name);
    }

    const auto dimension = tensor.dimensions[dimension_index];
    if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw EngineModelError(EngineModelErrorCode::invalid_tensor, tensor.name);
    }

    return static_cast<std::uint32_t>(dimension);
}

[[nodiscard]] std::uint32_t infer_vocab_size_from_token_embedding(
    const model_format::GGUFModelView& model,
    std::uint32_t embedding_length
) {
    const auto& token_embedding = require_tensor_payload(model, "token_embd.weight").tensor;
    if (token_embedding.dimensions.size() != 2 || checked_tensor_dimension_u32(token_embedding, 0) != embedding_length) {
        throw EngineModelError(EngineModelErrorCode::invalid_tensor, token_embedding.name);
    }

    return checked_tensor_dimension_u32(token_embedding, 1);
}

[[nodiscard]] Gemma2ModelConfig load_gemma2_config(const model_format::GGUFModelView& model) {
    const auto& metadata = model.directory.metadata;
    const auto embedding_length = require_metadata_u32(metadata, "gemma2.embedding_length");

    return Gemma2ModelConfig {
        .context_length = require_metadata_u32(metadata, "gemma2.context_length"),
        .embedding_length = embedding_length,
        .block_count = require_metadata_u32(metadata, "gemma2.block_count"),
        .feed_forward_length = require_metadata_u32(metadata, "gemma2.feed_forward_length"),
        .attention_head_count = require_metadata_u32(metadata, "gemma2.attention.head_count"),
        .attention_head_count_kv = require_metadata_u32(metadata, "gemma2.attention.head_count_kv"),
        .attention_key_length = require_metadata_u32(metadata, "gemma2.attention.key_length"),
        .attention_value_length = require_metadata_u32(metadata, "gemma2.attention.value_length"),
        .sliding_window = require_metadata_u32(metadata, "gemma2.attention.sliding_window"),
        .vocab_size = infer_vocab_size_from_token_embedding(model, embedding_length),
        .rms_norm_epsilon = require_metadata_f32(metadata, "gemma2.attention.layer_norm_rms_epsilon"),
        .attention_logit_softcap = require_metadata_f32(metadata, "gemma2.attn_logit_softcapping"),
        .final_logit_softcap = require_metadata_f32(metadata, "gemma2.final_logit_softcapping"),
        .uses_tied_output_weight = find_tensor_payload(model, "output.weight") == nullptr,
    };
}

[[nodiscard]] std::vector<EngineTensor> load_unique_tensors(const model_format::GGUFModelView& model) {
    std::vector<EngineTensor> tensors;
    tensors.reserve(model.tensor_payloads.size());

    std::unordered_set<std::string> names;
    names.reserve(model.tensor_payloads.size());

    for (const auto& payload : model.tensor_payloads) {
        if (!names.insert(payload.tensor.name).second) {
            throw EngineModelError(EngineModelErrorCode::duplicate_tensor_name, payload.tensor.name);
        }

        auto loaded = load_gguf_tensor_payload(payload);
        tensors.push_back(EngineTensor {
            .name = std::move(loaded.name),
            .storage = std::move(loaded.storage),
        });
    }

    return tensors;
}

}  // namespace

EngineModelError::EngineModelError(EngineModelErrorCode code, std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

EngineModelErrorCode EngineModelError::code() const noexcept {
    return code_;
}

const std::string& EngineModelError::detail() const noexcept {
    return detail_;
}

EngineModel::EngineModel(
    ModelArchitecture architecture,
    std::optional<std::string> name,
    LlamaModelConfig llama_config,
    std::vector<EngineTensor> tensors
)
    : EngineModel(
        architecture,
        std::move(name),
        llama_config,
        Gemma2ModelConfig {},
        std::move(tensors)
    ) {}

EngineModel::EngineModel(
    ModelArchitecture architecture,
    std::optional<std::string> name,
    LlamaModelConfig llama_config,
    Gemma2ModelConfig gemma2_config,
    std::vector<EngineTensor> tensors
)
    : architecture_(architecture),
      name_(std::move(name)),
      llama_config_(llama_config),
      gemma2_config_(gemma2_config),
      tensors_(std::move(tensors)) {}

ModelArchitecture EngineModel::architecture() const noexcept {
    return architecture_;
}

const std::optional<std::string>& EngineModel::name() const noexcept {
    return name_;
}

const LlamaModelConfig& EngineModel::llama_config() const noexcept {
    return llama_config_;
}

const Gemma2ModelConfig& EngineModel::gemma2_config() const noexcept {
    return gemma2_config_;
}

const std::vector<EngineTensor>& EngineModel::tensors() const noexcept {
    return tensors_;
}

const EngineTensor* EngineModel::find_tensor(std::string_view name) const noexcept {
    const auto found = std::find_if(
        tensors_.begin(),
        tensors_.end(),
        [name](const EngineTensor& tensor) {
            return tensor.name == name;
        }
    );

    if (found == tensors_.end()) {
        return nullptr;
    }

    return &*found;
}

std::string_view model_architecture_name(ModelArchitecture architecture) noexcept {
    switch (architecture) {
        case ModelArchitecture::llama:
            return "llama";
        case ModelArchitecture::gemma2:
            return "gemma2";
    }

    return "unknown";
}

EngineModel load_engine_model_from_gguf(const model_format::GGUFModelView& model) {
    const auto architecture = parse_architecture(model.directory.metadata);
    const auto name = optional_metadata_string(model.directory.metadata, "general.name");

    if (architecture == ModelArchitecture::gemma2) {
        const auto gemma2_config = load_gemma2_config(model);
        auto tensors = load_unique_tensors(model);
        return EngineModel(
            architecture,
            name,
            LlamaModelConfig {},
            gemma2_config,
            std::move(tensors)
        );
    }

    const auto llama_config = load_llama_config(model.directory.metadata);
    auto tensors = load_unique_tensors(model);
    return EngineModel(
        architecture,
        name,
        llama_config,
        std::move(tensors)
    );
}

}  // namespace local_gguf_engine::engine
