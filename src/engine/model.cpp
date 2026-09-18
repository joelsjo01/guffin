#include "local_gguf_engine/engine/model.hpp"

#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
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
using model_format::view_gguf_tensor_payload;

void hash_u32(runtime::Sha256& hash, std::uint32_t value) {
    const std::array<std::uint8_t, 4> bytes {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8U),
        static_cast<std::uint8_t>(value >> 16U),
        static_cast<std::uint8_t>(value >> 24U),
    };
    hash.update(bytes);
}

void hash_u64(runtime::Sha256& hash, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes {};
    for (std::size_t index = 0U; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
    hash.update(bytes);
}

void hash_string(runtime::Sha256& hash, std::string_view value) {
    hash_u64(hash, value.size());
    hash.update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size()
    ));
}

[[nodiscard]] runtime::Sha256Digest hash_consumed_metadata(
    ModelArchitecture architecture,
    const std::optional<std::string>& name,
    const Gemma2ModelConfig& config
) {
    runtime::Sha256 hash;
    hash_string(hash, "local-gguf-engine/consumed-model-metadata/v1");
    hash_u32(hash, static_cast<std::uint32_t>(architecture));
    hash_u32(hash, name.has_value() ? 1U : 0U);
    if (name.has_value()) hash_string(hash, *name);
    for (const auto value : {
             config.context_length, config.embedding_length, config.block_count,
             config.feed_forward_length, config.attention_head_count,
             config.attention_head_count_kv, config.attention_key_length,
             config.attention_value_length, config.sliding_window, config.vocab_size,
         }) {
        hash_u32(hash, value);
    }
    hash_u32(hash, std::bit_cast<std::uint32_t>(config.rms_norm_epsilon));
    hash_u32(hash, std::bit_cast<std::uint32_t>(config.attention_logit_softcap));
    hash_u32(hash, std::bit_cast<std::uint32_t>(config.final_logit_softcap));
    hash_u32(hash, config.uses_tied_output_weight ? 1U : 0U);
    return hash.finalize();
}

[[nodiscard]] runtime::Sha256Digest hash_source_tensors(
    const std::vector<EngineTensorView>& tensors
) {
    std::vector<const EngineTensorView*> ordered;
    ordered.reserve(tensors.size());
    for (const auto& tensor : tensors) ordered.push_back(&tensor);
    std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
        return left->name < right->name;
    });
    runtime::Sha256 hash;
    hash_string(hash, "local-gguf-engine/source-tensors/v1");
    hash_u32(hash, static_cast<std::uint32_t>(ordered.size()));
    for (const auto* tensor : ordered) {
        hash_string(hash, tensor->name);
        const auto& descriptor = tensor->view.descriptor();
        hash_u32(hash, static_cast<std::uint32_t>(descriptor.dtype));
        hash_u32(hash, static_cast<std::uint32_t>(descriptor.shape.rank()));
        for (std::size_t dimension = 0U; dimension < descriptor.shape.rank(); ++dimension) {
            hash_u64(hash, descriptor.shape.dimension(dimension));
        }
        hash_u64(hash, descriptor.byte_count);
        hash.update(tensor->view.bytes());
    }
    return hash.finalize();
}

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
    if (architecture == "gemma2") {
        return ModelArchitecture::gemma2;
    }

    throw EngineModelError(EngineModelErrorCode::unsupported_architecture, architecture);
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

[[nodiscard]] std::vector<EngineTensorView> load_unique_tensor_views(const model_format::GGUFModelView& model) {
    std::vector<EngineTensorView> tensors;
    tensors.reserve(model.tensor_payloads.size());

    std::unordered_set<std::string> names;
    names.reserve(model.tensor_payloads.size());

    for (const auto& payload : model.tensor_payloads) {
        if (!names.insert(payload.tensor.name).second) {
            throw EngineModelError(EngineModelErrorCode::duplicate_tensor_name, payload.tensor.name);
        }

        auto borrowed = view_gguf_tensor_payload(payload);
        tensors.push_back(EngineTensorView {
            .name = std::move(borrowed.name),
            .view = borrowed.view,
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
    Gemma2ModelConfig gemma2_config,
    std::vector<EngineTensorView> tensor_views,
    std::optional<EngineModelPayloadReceipt> payload_receipt
)
    : architecture_(architecture),
      name_(std::move(name)),
      gemma2_config_(gemma2_config),
      tensor_views_(std::move(tensor_views)),
      payload_receipt_(std::move(payload_receipt)) {}

ModelArchitecture EngineModel::architecture() const noexcept {
    return architecture_;
}

const std::optional<std::string>& EngineModel::name() const noexcept {
    return name_;
}

const Gemma2ModelConfig& EngineModel::gemma2_config() const noexcept {
    return gemma2_config_;
}

const std::vector<EngineTensorView>& EngineModel::tensor_views() const noexcept {
    return tensor_views_;
}

const std::optional<EngineModelPayloadReceipt>& EngineModel::payload_receipt() const noexcept {
    return payload_receipt_;
}

const EngineTensorView* EngineModel::find_tensor_view(std::string_view name) const noexcept {
    const auto found = std::find_if(
        tensor_views_.begin(),
        tensor_views_.end(),
        [name](const EngineTensorView& tensor) {
            return tensor.name == name;
        }
    );

    if (found == tensor_views_.end()) {
        return nullptr;
    }

    return &*found;
}

std::string_view model_architecture_name(ModelArchitecture architecture) noexcept {
    switch (architecture) {
        case ModelArchitecture::gemma2:
            return "gemma2";
    }

    return "unknown";
}

EngineModel load_engine_model_from_gguf(const model_format::GGUFModelView& model) {
    const auto architecture = parse_architecture(model.directory.metadata);
    const auto name = optional_metadata_string(model.directory.metadata, "general.name");
    const auto gemma2_config = load_gemma2_config(model);
    auto tensor_views = load_unique_tensor_views(model);
    return EngineModel(
        architecture,
        name,
        gemma2_config,
        std::move(tensor_views)
    );
}

EngineModel load_verified_engine_model_from_gguf(
    const model_format::GGUFModelView& model,
    std::span<const std::uint8_t> whole_gguf_bytes
) {
    if (whole_gguf_bytes.empty()) {
        throw EngineModelError(EngineModelErrorCode::invalid_tensor, "empty whole-GGUF receipt input");
    }
    const auto architecture = parse_architecture(model.directory.metadata);
    const auto name = optional_metadata_string(model.directory.metadata, "general.name");
    const auto gemma2_config = load_gemma2_config(model);
    auto tensor_views = load_unique_tensor_views(model);
    if (tensor_views.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw EngineModelError(EngineModelErrorCode::invalid_tensor, "source tensor receipt count");
    }
    EngineModelPayloadReceipt receipt {
        .whole_gguf_byte_count = whole_gguf_bytes.size(),
        .source_tensor_count = static_cast<std::uint32_t>(tensor_views.size()),
        .whole_gguf = runtime::sha256(whole_gguf_bytes),
        .canonical_consumed_metadata = hash_consumed_metadata(
            architecture, name, gemma2_config
        ),
        .source_tensors = hash_source_tensors(tensor_views),
    };
    return EngineModel(
        architecture, name, gemma2_config, std::move(tensor_views), receipt
    );
}

}  // namespace local_gguf_engine::engine
