#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/kquant_packing.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::tensor {

namespace {

void validate_tensor_packing(const TensorDescriptor& descriptor) {
    if (descriptor.packing == TensorPacking::canonical) {
        return;
    }
    if (descriptor.packing != TensorPacking::kquant_columns8) {
        throw std::invalid_argument("unknown tensor packing");
    }
    const auto expected = TensorDescriptor::from_interleaved_kquant_weight_matrix(
        descriptor.shape, descriptor.dtype);
    if (descriptor != expected) {
        throw std::invalid_argument("interleaved K-quant tensor descriptor is inconsistent");
    }
}

void validate_storage_byte_count(const TensorDescriptor& descriptor, std::size_t actual_byte_count) {
    validate_tensor_packing(descriptor);
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

TensorDescriptor TensorDescriptor::from_quantized_gguf_weight_matrix(
    const Shape& shape,
    DType dtype
) {
    if (shape.rank() != 2) {
        throw TensorError(TensorErrorCode::invalid_shape_rank, 2, static_cast<std::uint64_t>(shape.rank()));
    }
    if (dtype != DType::q8_0 && dtype != DType::q4_k && dtype != DType::q6_k) {
        throw TensorError(
            TensorErrorCode::unsupported_quantized_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::q4_k)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype))
        );
    }

    const auto layout = dtype_layout(dtype);
    const auto shared = shape.dimension(0);
    const auto columns = shape.dimension(1);
    if (shared % layout.block_element_count != 0) {
        throw TensorError(TensorErrorCode::invalid_quantized_element_count, layout.block_element_count, shared);
    }
    if (shared > std::numeric_limits<std::uint64_t>::max() / columns) {
        throw TensorError(TensorErrorCode::element_count_overflow, shared, columns);
    }
    const auto elements = shared * columns;
    const auto blocks = (shared / layout.block_element_count) * columns;
    if (blocks > std::numeric_limits<std::uint64_t>::max() / layout.block_byte_count) {
        throw TensorError(TensorErrorCode::byte_size_overflow, blocks, layout.block_byte_count);
    }
    const auto bytes = blocks * layout.block_byte_count;
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw TensorError(
            TensorErrorCode::byte_size_overflow,
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
            bytes
        );
    }

    return TensorDescriptor(shape, dtype, elements, static_cast<std::size_t>(bytes));
}

TensorDescriptor TensorDescriptor::from_interleaved_kquant_weight_matrix(
    const Shape& shape,
    DType dtype
) {
    if (dtype != DType::q4_k && dtype != DType::q6_k) {
        throw std::invalid_argument("interleaved K-quant weights require Q4_K or Q6_K");
    }
    const auto native = from_quantized_gguf_weight_matrix(shape, dtype);
    const auto columns = shape.dimension(1);
    const auto tiles = columns / kquant_packing_columns
        + (columns % kquant_packing_columns != 0U);
    const auto blocks = shape.dimension(0) / 256U;
    const auto tile_bytes = dtype == DType::q4_k
        ? kquant_q4_columns8_tile_bytes : kquant_q6_columns8_tile_bytes;
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (blocks > limit / tile_bytes || tiles > limit / (blocks * tile_bytes)) {
        throw TensorError(TensorErrorCode::byte_size_overflow, limit, tiles);
    }
    return TensorDescriptor(shape, dtype, native.element_count,
        static_cast<std::size_t>(tiles * blocks * tile_bytes), TensorPacking::kquant_columns8);
}

TensorDescriptor::TensorDescriptor(
    Shape shape,
    DType dtype,
    std::uint64_t element_count,
    std::size_t byte_count,
    TensorPacking packing
)
    : shape(shape),
      dtype(dtype),
      packing(packing),
      element_count(element_count),
      byte_count(byte_count) {
}

TensorStorage TensorStorage::allocate(const TensorDescriptor& descriptor) {
    validate_tensor_packing(descriptor);
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
