#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include "local_gguf_engine/tensor/backend.hpp"

#include "backends/cpu_backend.hpp"
#include "backends/metal/metal_buffer.hpp"
#include "backends/metal/metal_kernels.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#include <memory>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::tensor::detail {

BackendBufferStorage::~BackendBufferStorage() = default;

BackendBuffer make_backend_buffer(std::shared_ptr<BackendBufferStorage> storage) {
    return BackendBuffer(std::move(storage));
}

}  // namespace local_gguf_engine::tensor::detail

namespace local_gguf_engine::tensor {

runtime::Sha256Digest metal_backend_library_source_digest() {
    const auto* source = backends::metal::metal_kernel_source();
    return runtime::sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(source), std::strlen(source)
    ));
}

BackendBuffer::BackendBuffer(std::shared_ptr<detail::BackendBufferStorage> storage)
    : storage_(std::move(storage)) {}

runtime::BackendKind BackendBuffer::backend() const noexcept {
    return storage_->backend();
}

const TensorDescriptor& BackendBuffer::descriptor() const noexcept {
    return storage_->descriptor();
}

BackendBuffer copy_tensor_to_backend(const TensorStorage& storage, runtime::BackendKind backend) {
    return copy_tensor_view_to_backend(storage.view(), backend);
}

BackendBuffer copy_tensor_view_to_backend(const TensorView& view, runtime::BackendKind backend) {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::copy_tensor_view_to_cpu_backend(view);
        case runtime::BackendKind::metal:
            return backends::metal::copy_tensor_view_to_metal_backend(view);
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for tensor buffer copy"
    );
}

BackendBuffer allocate_backend_buffer(const TensorDescriptor& descriptor, runtime::BackendKind backend) {
    switch (backend) {
        case runtime::BackendKind::cpu:
            return backends::cpu::allocate_cpu_backend_buffer(descriptor);
        case runtime::BackendKind::metal:
            return backends::metal::allocate_metal_backend_buffer(descriptor);
    }

    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        backend,
        "unsupported backend for tensor buffer allocation"
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

std::vector<std::uint8_t> copy_backend_buffer_bytes(
    const BackendBuffer& buffer,
    std::size_t byte_offset,
    std::size_t byte_count
) {
    const auto& storage = backend_buffer_storage(buffer);
    if (byte_offset > storage.descriptor().byte_count
        || byte_count > storage.descriptor().byte_count - byte_offset) {
        throw std::out_of_range("backend byte readback exceeds buffer");
    }
    switch (storage.backend()) {
        case runtime::BackendKind::metal:
            return backends::metal::copy_metal_backend_buffer_bytes(
                storage, byte_offset, byte_count
            );
        case runtime::BackendKind::cpu: {
            const auto full = backends::cpu::copy_tensor_from_cpu_backend(storage);
            std::vector<std::uint8_t> output(byte_count);
            if (byte_count != 0U) {
                std::memcpy(output.data(), full.bytes().data() + byte_offset, byte_count);
            }
            return output;
        }
    }
    throw runtime::BackendError(
        runtime::BackendErrorCode::unsupported_backend,
        storage.backend(),
        "unsupported backend for bounded tensor buffer readback"
    );
}

const detail::BackendBufferStorage& backend_buffer_storage(const BackendBuffer& buffer) noexcept {
    return *buffer.storage_;
}

}  // namespace local_gguf_engine::tensor
