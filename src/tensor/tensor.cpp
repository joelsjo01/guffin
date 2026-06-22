#include "local_gguf_engine/tensor/tensor.hpp"

#include <algorithm>
#include <utility>

namespace local_gguf_engine::tensor {

namespace {

void validate_storage_byte_count(const TensorDescriptor& descriptor, std::size_t actual_byte_count) {
    if (actual_byte_count != descriptor.byte_count) {
        throw TensorError(
            TensorErrorCode::invalid_storage_byte_count,
            static_cast<std::uint64_t>(descriptor.byte_count),
            static_cast<std::uint64_t>(actual_byte_count)
        );
    }
}

}  // namespace

TensorDescriptor TensorDescriptor::from(const Shape& shape, DType dtype) {
    return TensorDescriptor(
        shape,
        dtype,
        local_gguf_engine::tensor::element_count(shape),
        byte_size(shape, dtype)
    );
}

TensorDescriptor::TensorDescriptor(
    Shape shape,
    DType dtype,
    std::uint64_t element_count,
    std::size_t byte_count
)
    : shape(shape),
      dtype(dtype),
      element_count(element_count),
      byte_count(byte_count) {
}

TensorStorage TensorStorage::allocate(const TensorDescriptor& descriptor) {
    return TensorStorage(descriptor, std::vector<std::uint8_t>(descriptor.byte_count));
}

TensorStorage TensorStorage::from_bytes(
    const TensorDescriptor& descriptor,
    std::span<const std::uint8_t> bytes
) {
    validate_storage_byte_count(descriptor, bytes.size());
    std::vector<std::uint8_t> storage(bytes.size());
    std::copy(bytes.begin(), bytes.end(), storage.begin());
    return TensorStorage(descriptor, std::move(storage));
}

const TensorDescriptor& TensorStorage::descriptor() const noexcept {
    return descriptor_;
}

std::span<const std::uint8_t> TensorStorage::bytes() const noexcept {
    return std::span<const std::uint8_t>(bytes_.data(), bytes_.size());
}

std::span<std::uint8_t> TensorStorage::mutable_bytes() noexcept {
    return std::span<std::uint8_t>(bytes_.data(), bytes_.size());
}

TensorView TensorStorage::view() const noexcept {
    return TensorView::from_bytes(descriptor_, bytes());
}

TensorStorage::TensorStorage(TensorDescriptor descriptor, std::vector<std::uint8_t> bytes)
    : descriptor_(descriptor),
      bytes_(std::move(bytes)) {}

TensorView TensorView::from_bytes(
    const TensorDescriptor& descriptor,
    std::span<const std::uint8_t> bytes
) {
    validate_storage_byte_count(descriptor, bytes.size());
    return TensorView(descriptor, bytes);
}

const TensorDescriptor& TensorView::descriptor() const noexcept {
    return descriptor_;
}

std::span<const std::uint8_t> TensorView::bytes() const noexcept {
    return bytes_;
}

TensorView::TensorView(TensorDescriptor descriptor, std::span<const std::uint8_t> bytes)
    : descriptor_(descriptor),
      bytes_(bytes) {}

}  // namespace local_gguf_engine::tensor
