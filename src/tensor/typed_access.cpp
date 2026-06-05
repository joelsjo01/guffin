#include "local_gguf_engine/tensor/typed_access.hpp"

#include "local_gguf_engine/tensor/f16.hpp"

#include <cstring>

namespace local_gguf_engine::tensor {

namespace {

void require_f32_storage(const TensorDescriptor& descriptor) {
    if (descriptor.dtype != DType::f32) {
        throw TensorError(
            TensorErrorCode::unsupported_typed_accessor_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f32)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

void require_f16_storage(const TensorDescriptor& descriptor) {
    if (descriptor.dtype != DType::f16) {
        throw TensorError(
            TensorErrorCode::unsupported_typed_accessor_dtype,
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(DType::f16)),
            static_cast<std::uint64_t>(static_cast<std::uint8_t>(descriptor.dtype))
        );
    }
}

}  // namespace

float read_f32(const TensorStorage& storage, std::span<const std::uint64_t> indices) {
    require_f32_storage(storage.descriptor());

    const auto bytes = element_bytes(storage, indices);
    float value = 0.0F;
    std::memcpy(&value, bytes.data(), sizeof(float));
    return value;
}

void write_f32(TensorStorage& storage, std::span<const std::uint64_t> indices, float value) {
    require_f32_storage(storage.descriptor());

    auto bytes = mutable_element_bytes(storage, indices);
    std::memcpy(bytes.data(), &value, sizeof(float));
}

float read_f16_as_f32(const TensorStorage& storage, std::span<const std::uint64_t> indices) {
    require_f16_storage(storage.descriptor());

    const auto bytes = element_bytes(storage, indices);
    const auto bits = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U)
    );
    return f16_bits_to_f32(bits);
}

void write_f16_from_f32(TensorStorage& storage, std::span<const std::uint64_t> indices, float value) {
    require_f16_storage(storage.descriptor());

    const auto bits = f32_to_f16_bits(value);
    auto bytes = mutable_element_bytes(storage, indices);
    bytes[0] = static_cast<std::uint8_t>(bits & 0x00FFU);
    bytes[1] = static_cast<std::uint8_t>((bits >> 8U) & 0x00FFU);
}

}  // namespace local_gguf_engine::tensor
