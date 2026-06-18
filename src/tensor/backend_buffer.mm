#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include "backends/cpu_backend.hpp"
#include "backends/metal/metal_buffer.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#include <memory>
#include <utility>

namespace local_gguf_engine::tensor::detail {

BackendBufferStorage::~BackendBufferStorage() = default;

BackendBuffer make_backend_buffer(std::shared_ptr<BackendBufferStorage> storage) {
    return BackendBuffer(std::move(storage));
}

}  // namespace local_gguf_engine::tensor::detail

namespace local_gguf_engine::tensor {

BackendBuffer::BackendBuffer(std::shared_ptr<detail::BackendBufferStorage> storage)
    : storage_(std::move(storage)) {}

runtime::BackendKind BackendBuffer::backend() const noexcept {
    return storage_->backend();
}

const TensorDescriptor& BackendBuffer::descriptor() const noexcept {
    return storage_->descriptor();
}

BackendBuffer copy_tensor_to_backend(const TensorStorage& storage, runtime::BackendKind backend) {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::copy_tensor_to_cpu_backend(storage);
        case runtime::BackendKind::metal:
            return backends::metal::copy_tensor_to_metal_backend(storage);
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for tensor buffer copy"
    );
}

TensorStorage copy_tensor_from_backend(const BackendBuffer& buffer) {
    const auto& storage = backend_buffer_storage(buffer);
    switch (storage.backend()) {
        case runtime::BackendKind::cpu:
            return backends::cpu::copy_tensor_from_cpu_backend(storage);
        case runtime::BackendKind::metal:
            return backends::metal::copy_tensor_from_metal_backend(storage);
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        storage.backend(),
        "unsupported backend for tensor buffer readback"
    );
}

const detail::BackendBufferStorage& backend_buffer_storage(const BackendBuffer& buffer) noexcept {
    return *buffer.storage_;
}

}  // namespace local_gguf_engine::tensor
