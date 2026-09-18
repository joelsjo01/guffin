#include "metal_buffer.hpp"

#include "metal_context.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::tensor::backends::metal {

MetalBackendBufferStorage::MetalBackendBufferStorage(
    const TensorStorage& storage,
    id<MTLBuffer> buffer
)
    : MetalBackendBufferStorage(storage.descriptor(), buffer) {}

MetalBackendBufferStorage::MetalBackendBufferStorage(
    TensorDescriptor descriptor,
    id<MTLBuffer> buffer,
    std::size_t base_byte_offset
)
    : descriptor_(std::move(descriptor)),
      buffer_(buffer),
      base_byte_offset_(base_byte_offset) {
    if (buffer_ == nil) throw std::invalid_argument("Metal backend buffer must not be nil");
    if (base_byte_offset_ % alignof(std::uint32_t) != 0U) {
        throw std::invalid_argument("Metal backend buffer base offset must be four-byte aligned");
    }
    if (base_byte_offset_ > buffer_.length
        || descriptor_.byte_count > buffer_.length - base_byte_offset_) {
        throw std::out_of_range("Metal backend buffer view exceeds its parent allocation");
    }
}

runtime::BackendKind MetalBackendBufferStorage::backend() const noexcept {
    return runtime::BackendKind::metal;
}

const TensorDescriptor& MetalBackendBufferStorage::descriptor() const noexcept {
    return descriptor_;
}

id<MTLBuffer> MetalBackendBufferStorage::buffer() const noexcept {
    return buffer_;
}

std::size_t MetalBackendBufferStorage::base_byte_offset() const noexcept {
    return base_byte_offset_;
}

std::size_t MetalBackendBufferStorage::parent_byte_count() const noexcept {
    return buffer_.length;
}

std::uintptr_t MetalBackendBufferStorage::parent_allocation_identity() const noexcept {
    return reinterpret_cast<std::uintptr_t>((__bridge const void*)buffer_);
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

BackendBuffer make_metal_backend_buffer_view(
    TensorDescriptor descriptor,
    id<MTLBuffer> parent_buffer,
    std::size_t base_byte_offset
) {
    return detail::make_backend_buffer(std::make_shared<MetalBackendBufferStorage>(
        std::move(descriptor), parent_buffer, base_byte_offset
    ));
}

NSUInteger metal_buffer_binding_offset(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset
) {
    if (operation_byte_offset > storage.descriptor().byte_count) {
        throw std::out_of_range("Metal backend operation offset exceeds child buffer");
    }
    const auto base = static_cast<std::uint64_t>(storage.base_byte_offset());
    if (operation_byte_offset > std::numeric_limits<std::uint64_t>::max() - base) {
        throw std::overflow_error("Metal backend binding offset overflows");
    }
    const auto combined = base + operation_byte_offset;
    if (combined > storage.parent_byte_count()) {
        throw std::out_of_range("Metal backend binding offset exceeds parent allocation");
    }
    return static_cast<NSUInteger>(combined);
}

void* metal_buffer_contents(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset
) {
    void* contents = storage.buffer().contents;
    if (contents == nullptr) {
        throw std::logic_error("Metal private buffer contents are not CPU-accessible");
    }
    return static_cast<std::byte*>(contents) + metal_buffer_binding_offset(storage, operation_byte_offset);
}

BackendBuffer copy_tensor_to_metal_backend(const TensorStorage& storage) {
    return copy_tensor_view_to_metal_backend(storage.view());
}

BackendBuffer copy_tensor_view_to_metal_backend(const TensorView& view) {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();
        id<MTLBuffer> buffer = [state.device newBufferWithBytes:view.bytes().data()
                                                         length:static_cast<NSUInteger>(view.bytes().size())
                                                        options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw runtime::BackendError(
                runtime::BackendErrorCode::metal_execution_failed,
                runtime::BackendKind::metal,
                "Metal backend buffer allocation failed"
            );
        }

        return detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(view.descriptor(), buffer)
        );
    }
}

BackendBuffer allocate_metal_backend_buffer(const TensorDescriptor& descriptor) {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();
        id<MTLBuffer> buffer = [state.device newBufferWithLength:static_cast<NSUInteger>(descriptor.byte_count)
                                                         options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw runtime::BackendError(
                runtime::BackendErrorCode::metal_execution_failed,
                runtime::BackendKind::metal,
                "Metal backend buffer allocation failed"
            );
        }

        return make_metal_backend_buffer(descriptor, buffer);
    }
}

TensorStorage copy_tensor_from_metal_backend(const detail::BackendBufferStorage& storage) {
    @autoreleasepool {
        const auto& metal_storage = require_metal_backend_buffer(storage);
        auto output = TensorStorage::allocate(metal_storage.descriptor());
        std::memcpy(
            output.mutable_bytes().data(),
            metal_buffer_contents(metal_storage),
            output.bytes().size()
        );
        return output;
    }
}

std::vector<std::uint8_t> copy_metal_backend_buffer_bytes(
    const detail::BackendBufferStorage& storage,
    std::size_t byte_offset,
    std::size_t byte_count
) {
    @autoreleasepool {
        const auto& metal_storage = require_metal_backend_buffer(storage);
        if (byte_offset > metal_storage.descriptor().byte_count
            || byte_count > metal_storage.descriptor().byte_count - byte_offset) {
            throw std::out_of_range("Metal backend byte readback exceeds buffer");
        }
        std::vector<std::uint8_t> output(byte_count);
        if (byte_count != 0U) {
            std::memcpy(
                output.data(), metal_buffer_contents(metal_storage, byte_offset), byte_count
            );
        }
        return output;
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal
