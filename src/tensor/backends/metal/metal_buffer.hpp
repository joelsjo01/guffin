#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"

#import <Metal/Metal.h>

namespace local_gguf_engine::tensor::backends::metal {

class MetalBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    MetalBackendBufferStorage(const TensorStorage& storage, id<MTLBuffer> buffer);
    MetalBackendBufferStorage(TensorDescriptor descriptor, id<MTLBuffer> buffer);

    [[nodiscard]] runtime::BackendKind backend() const noexcept override;
    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override;
    [[nodiscard]] id<MTLBuffer> buffer() const noexcept;

private:
    TensorDescriptor descriptor_;
    id<MTLBuffer> buffer_ = nil;
};

[[nodiscard]] const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
);
[[nodiscard]] BackendBuffer make_metal_backend_buffer(
    TensorDescriptor descriptor,
    id<MTLBuffer> buffer
);
[[nodiscard]] BackendBuffer copy_tensor_to_metal_backend(const TensorStorage& storage);
[[nodiscard]] BackendBuffer copy_tensor_view_to_metal_backend(const TensorView& view);
[[nodiscard]] BackendBuffer allocate_metal_backend_buffer(const TensorDescriptor& descriptor);
[[nodiscard]] TensorStorage copy_tensor_from_metal_backend(
    const detail::BackendBufferStorage& storage
);

}  // namespace local_gguf_engine::tensor::backends::metal
