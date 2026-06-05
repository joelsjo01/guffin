#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include "local_gguf_engine/model_format/gguf_reader.hpp"

#include <span>
#include <utility>

namespace local_gguf_engine::model_format {

namespace {

[[nodiscard]] tensor::Shape shape_from_gguf_tensor_info(const GGUFTensorInfo& tensor) {
    return tensor::Shape::from(
        std::span<const std::uint64_t>(tensor.dimensions.data(), tensor.dimensions.size())
    );
}

}  // namespace

tensor::DType dtype_from_gguf_tensor_type(std::uint32_t type_id) {
    switch (type_id) {
        case 0:
            return tensor::DType::f32;
        case 1:
            return tensor::DType::f16;
        case 8:
            return tensor::DType::q8_0;
        default:
            throw GGUFReadError(GGUFReadErrorCode::unsupported_tensor_type, 0, type_id);
    }
}

tensor::TensorDescriptor tensor_descriptor_from_gguf_tensor_info(const GGUFTensorInfo& tensor) {
    return tensor::TensorDescriptor::from(
        shape_from_gguf_tensor_info(tensor),
        dtype_from_gguf_tensor_type(tensor.type_id)
    );
}

LoadedGGUFTensor load_gguf_tensor_payload(const GGUFTensorPayloadView& payload) {
    const auto descriptor = tensor_descriptor_from_gguf_tensor_info(payload.tensor);

    return LoadedGGUFTensor {
        .name = payload.tensor.name,
        .storage = tensor::TensorStorage::from_bytes(descriptor, payload.bytes),
    };
}

std::vector<LoadedGGUFTensor> load_gguf_tensor_payloads(const GGUFModelView& model) {
    std::vector<LoadedGGUFTensor> loaded_tensors;
    loaded_tensors.reserve(model.tensor_payloads.size());

    for (const auto& payload : model.tensor_payloads) {
        loaded_tensors.push_back(load_gguf_tensor_payload(payload));
    }

    return loaded_tensors;
}

}  // namespace local_gguf_engine::model_format
