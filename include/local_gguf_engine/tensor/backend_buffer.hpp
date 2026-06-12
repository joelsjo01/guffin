#pragma once

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <memory>

namespace local_gguf_engine::tensor {

namespace detail {

class BackendBufferStorage {
public:
    virtual ~BackendBufferStorage();

    [[nodiscard]] virtual runtime::BackendKind backend() const noexcept = 0;
    [[nodiscard]] virtual const TensorDescriptor& descriptor() const noexcept = 0;
};

}  // namespace detail

class BackendBuffer final {
public:
    [[nodiscard]] runtime::BackendKind backend() const noexcept;
    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept;

private:
    explicit BackendBuffer(std::shared_ptr<detail::BackendBufferStorage> storage);

    friend BackendBuffer copy_tensor_to_backend(const TensorStorage&, runtime::BackendKind);
    friend TensorStorage copy_tensor_from_backend(const BackendBuffer&);
    friend const detail::BackendBufferStorage& backend_buffer_storage(const BackendBuffer&) noexcept;

    std::shared_ptr<detail::BackendBufferStorage> storage_;
};

[[nodiscard]] BackendBuffer copy_tensor_to_backend(
    const TensorStorage& storage,
    runtime::BackendKind backend
);
[[nodiscard]] TensorStorage copy_tensor_from_backend(const BackendBuffer& buffer);
[[nodiscard]] const detail::BackendBufferStorage& backend_buffer_storage(
    const BackendBuffer& buffer
) noexcept;

}  // namespace local_gguf_engine::tensor
