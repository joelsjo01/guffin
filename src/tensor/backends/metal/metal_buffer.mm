#include "metal_buffer.hpp"

#include "metal_context.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#include <cstring>
#include <memory>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

CpuBackendBufferStorage::CpuBackendBufferStorage(const TensorStorage& storage)
    : storage_(TensorStorage::from_bytes(storage.descriptor(), storage.bytes())) {}

runtime::BackendKind CpuBackendBufferStorage::backend() const noexcept {
    return runtime::BackendKind::cpu;
}

const TensorDescriptor& CpuBackendBufferStorage::descriptor() const noexcept {
    return storage_.descriptor();
}

const TensorStorage& CpuBackendBufferStorage::storage() const noexcept {
    return storage_;
}

MetalBackendBufferStorage::MetalBackendBufferStorage(
    const TensorStorage& storage,
    id<MTLBuffer> buffer
)
    : MetalBackendBufferStorage(storage.descriptor(), buffer) {}

MetalBackendBufferStorage::MetalBackendBufferStorage(
    TensorDescriptor descriptor,
    id<MTLBuffer> buffer
)
    : descriptor_(std::move(descriptor)),
      buffer_(buffer) {}

runtime::BackendKind MetalBackendBufferStorage::backend() const noexcept {
    return runtime::BackendKind::metal;
}

const TensorDescriptor& MetalBackendBufferStorage::descriptor() const noexcept {
    return descriptor_;
}

id<MTLBuffer> MetalBackendBufferStorage::buffer() const noexcept {
    return buffer_;
}

const CpuBackendBufferStorage& require_cpu_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const CpuBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain CPU storage"
        );
    }

    return *typed_storage;
}

const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const MetalBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain Metal storage"
        );
    }

    return *typed_storage;
}

BackendBuffer make_metal_backend_buffer(TensorDescriptor descriptor, id<MTLBuffer> buffer) {
    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(std::move(descriptor), buffer)
    );
}

}  // namespace local_gguf_engine::tensor::backends::metal

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
            return BackendBuffer(std::make_shared<backends::metal::CpuBackendBufferStorage>(storage));
        case runtime::BackendKind::metal: {
            @autoreleasepool {
                auto& state = backends::metal::shared_matmul_pipeline_state();
                id<MTLBuffer> buffer = [state.device newBufferWithBytes:storage.bytes().data()
                                                                  length:static_cast<NSUInteger>(storage.bytes().size())
                                                                 options:MTLResourceStorageModeShared];
                if (buffer == nil) {
                    throw runtime::BackendError(
                        runtime::BackendErrorCode::metal_execution_failed,
                        runtime::BackendKind::metal,
                        "Metal backend buffer allocation failed"
                    );
                }

                return BackendBuffer(
                    std::make_shared<backends::metal::MetalBackendBufferStorage>(storage, buffer)
                );
            }
        }
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
        case runtime::BackendKind::cpu: {
            const auto& cpu_storage = backends::metal::require_cpu_backend_buffer(storage).storage();
            return TensorStorage::from_bytes(cpu_storage.descriptor(), cpu_storage.bytes());
        }
        case runtime::BackendKind::metal: {
            @autoreleasepool {
                const auto& metal_storage = backends::metal::require_metal_backend_buffer(storage);
                auto output = TensorStorage::allocate(metal_storage.descriptor());
                std::memcpy(
                    output.mutable_bytes().data(),
                    [metal_storage.buffer() contents],
                    output.bytes().size()
                );
                return output;
            }
        }
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
