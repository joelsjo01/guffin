#pragma once

#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <string>
#include <vector>

namespace local_gguf_engine::model_format {

struct LoadedGGUFTensor {
    std::string name;
    tensor::TensorStorage storage;
};

struct BorrowedGGUFTensor {
    std::string name;
    tensor::TensorView view;
};

[[nodiscard]] tensor::DType dtype_from_gguf_tensor_type(std::uint32_t type_id);
[[nodiscard]] tensor::TensorDescriptor tensor_descriptor_from_gguf_tensor_info(const GGUFTensorInfo& tensor);
[[nodiscard]] BorrowedGGUFTensor view_gguf_tensor_payload(const GGUFTensorPayloadView& payload);
[[nodiscard]] LoadedGGUFTensor load_gguf_tensor_payload(const GGUFTensorPayloadView& payload);
[[nodiscard]] std::vector<LoadedGGUFTensor> load_gguf_tensor_payloads(const GGUFModelView& model);

}  // namespace local_gguf_engine::model_format
