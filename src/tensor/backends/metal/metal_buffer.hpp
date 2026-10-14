#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace local_gguf_engine::tensor::backends::metal {

class MetalBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    MetalBackendBufferStorage(const TensorStorage& storage, id<MTLBuffer> buffer);
    MetalBackendBufferStorage(
        TensorDescriptor descriptor,
        id<MTLBuffer> buffer,
        std::size_t base_byte_offset = 0,
        bool read_only = false
    );
    ~MetalBackendBufferStorage() override;
    MetalBackendBufferStorage(const MetalBackendBufferStorage&) = delete;
    MetalBackendBufferStorage& operator=(const MetalBackendBufferStorage&) = delete;

    [[nodiscard]] runtime::BackendKind backend() const noexcept override;
    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override;
    // Borrow the owned ARC reference; returning id by value would add an
    // autorelease whose lifetime can outlive a C++ caller's entire model.
    [[nodiscard]] id<MTLBuffer> const& buffer() const noexcept;
    [[nodiscard]] std::size_t base_byte_offset() const noexcept;
    [[nodiscard]] bool read_only() const noexcept;
    [[nodiscard]] std::size_t parent_byte_count() const noexcept;
    [[nodiscard]] std::uintptr_t parent_allocation_identity() const noexcept;

private:
    TensorDescriptor descriptor_;
    id<MTLBuffer> buffer_ = nil;
    std::size_t base_byte_offset_ = 0;
    bool read_only_ = false;
};

// Renew the idle residency window for owned target and draft buffers.
void keep_metal_buffers_resident();

[[nodiscard]] const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
);
[[nodiscard]] BackendBuffer make_metal_backend_buffer(
    TensorDescriptor descriptor,
    id<MTLBuffer> buffer
);
[[nodiscard]] BackendBuffer make_metal_backend_buffer_view(
    TensorDescriptor descriptor,
    id<MTLBuffer> parent_buffer,
    std::size_t base_byte_offset,
    bool read_only = false
);
[[nodiscard]] NSUInteger metal_buffer_binding_offset(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset = 0
);
void require_writable_metal_buffer(const MetalBackendBufferStorage& storage);
[[nodiscard]] void* mutable_metal_buffer_contents(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset = 0
);
[[nodiscard]] const void* metal_buffer_contents(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset = 0
);
[[nodiscard]] BackendBuffer copy_tensor_to_metal_backend(const TensorStorage& storage);
[[nodiscard]] BackendBuffer copy_tensor_view_to_metal_backend(const TensorView& view);
[[nodiscard]] BackendBuffer allocate_metal_backend_buffer(const TensorDescriptor& descriptor);
[[nodiscard]] TensorStorage copy_tensor_from_metal_backend(
    const detail::BackendBufferStorage& storage
);
[[nodiscard]] std::vector<std::uint8_t> copy_metal_backend_buffer_bytes(
    const detail::BackendBufferStorage& storage,
    std::size_t byte_offset,
    std::size_t byte_count
);

}  // namespace local_gguf_engine::tensor::backends::metal
