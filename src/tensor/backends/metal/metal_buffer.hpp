#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"

#import <Metal/Metal.h>

namespace local_gguf_engine::tensor::backends::metal {

class CpuBackendBufferStorage final : public detail::BackendBufferStorage {
public:
    explicit CpuBackendBufferStorage(const TensorStorage& storage);

    [[nodiscard]] runtime::BackendKind backend() const noexcept override;
    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept override;
    [[nodiscard]] const TensorStorage& storage() const noexcept;

private:
    TensorStorage storage_;
};

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

[[nodiscard]] const CpuBackendBufferStorage& require_cpu_backend_buffer(
    const detail::BackendBufferStorage& storage
);
[[nodiscard]] const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
);
[[nodiscard]] BackendBuffer make_metal_backend_buffer(
    TensorDescriptor descriptor,
    id<MTLBuffer> buffer
);

}  // namespace local_gguf_engine::tensor::backends::metal
