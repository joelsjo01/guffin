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
        case 12:
            return tensor::DType::q4_k;
        case 14:
            return tensor::DType::q6_k;
        default:
            throw GGUFReadError(GGUFReadErrorCode::unsupported_tensor_type, 0, type_id);
    }
}

tensor::TensorDescriptor tensor_descriptor_from_gguf_tensor_info(const GGUFTensorInfo& tensor) {
    const auto shape = shape_from_gguf_tensor_info(tensor);
    const auto dtype = dtype_from_gguf_tensor_type(tensor.type_id);
    if (shape.rank() == 2U && tensor::dtype_layout(dtype).block_element_count > 1U) {
        return tensor::TensorDescriptor::from_quantized_gguf_weight_matrix(shape, dtype);
    }
    return tensor::TensorDescriptor::from(
        shape, dtype
    );
}

LoadedGGUFTensor load_gguf_tensor_payload(const GGUFTensorPayloadView& payload) {
    auto borrowed = view_gguf_tensor_payload(payload);

    return LoadedGGUFTensor {
        .name = std::move(borrowed.name),
        .storage = tensor::TensorStorage::from_bytes(borrowed.view.descriptor(), borrowed.view.bytes()),
    };
}

BorrowedGGUFTensor view_gguf_tensor_payload(const GGUFTensorPayloadView& payload) {
    const auto descriptor = tensor_descriptor_from_gguf_tensor_info(payload.tensor);

    return BorrowedGGUFTensor {
        .name = payload.tensor.name,
        .view = tensor::TensorView::from_bytes(descriptor, payload.bytes),
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
