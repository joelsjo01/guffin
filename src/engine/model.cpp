#include "local_gguf_engine/engine/model.hpp"

#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <algorithm>
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
    : architecture_(architecture),
      name_(std::move(name)),
      llama_config_(llama_config),
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
    }

    return "unknown";
}

EngineModel load_engine_model_from_gguf(const model_format::GGUFModelView& model) {
    const auto architecture = parse_architecture(model.directory.metadata);
    return EngineModel(
        architecture,
        optional_metadata_string(model.directory.metadata, "general.name"),
        load_llama_config(model.directory.metadata),
        load_unique_tensors(model)
    );
}

}  // namespace local_gguf_engine::engine
