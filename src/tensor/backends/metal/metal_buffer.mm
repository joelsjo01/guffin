#include "metal_buffer.hpp"

#include "metal_context.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#include <cstring>
#include <memory>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

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

BackendBuffer copy_tensor_to_metal_backend(const TensorStorage& storage) {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();
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

        return detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(storage, buffer)
        );
    }
}

TensorStorage copy_tensor_from_metal_backend(const detail::BackendBufferStorage& storage) {
    @autoreleasepool {
        const auto& metal_storage = require_metal_backend_buffer(storage);
        auto output = TensorStorage::allocate(metal_storage.descriptor());
        std::memcpy(
            output.mutable_bytes().data(),
            [metal_storage.buffer() contents],
            output.bytes().size()
        );
        return output;
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal
