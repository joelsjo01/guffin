#include "cpu_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"

#include <cstdint>
#include <cstring>
#include <memory>

namespace local_gguf_engine::tensor::backends::cpu {

namespace {

[[nodiscard]] float read_f32_linear(const TensorStorage& storage, std::uint64_t linear_index) {
    float value = 0.0F;
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(&value, storage.bytes().data() + byte_offset, sizeof(float));
    return value;
}

[[nodiscard]] float read_f16_linear_as_f32(const TensorStorage& storage, std::uint64_t linear_index) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(std::uint16_t));
    const auto bytes = storage.bytes();
    const auto bits = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[byte_offset])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[byte_offset + 1]) << 8U)
    );
    return f16_bits_to_f32(bits);
}

void write_f32_linear(TensorStorage& storage, std::uint64_t linear_index, float value) {
    const auto byte_offset = static_cast<std::size_t>(linear_index * sizeof(float));
    std::memcpy(storage.mutable_bytes().data() + byte_offset, &value, sizeof(float));
}

}  // namespace

bool is_available() noexcept {
    return true;
}

std::string unavailable_reason() {
    return {};
}

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

const CpuBackendBufferStorage& require_cpu_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const CpuBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw runtime::BackendError(
            runtime::BackendErrorCode::unsupported_backend,
            runtime::BackendKind::cpu,
            "backend buffer does not contain CPU storage"
        );
    }

    return *typed_storage;
}

BackendBuffer copy_tensor_to_cpu_backend(const TensorStorage& storage) {
    return detail::make_backend_buffer(std::make_shared<CpuBackendBufferStorage>(storage));
}

TensorStorage copy_tensor_from_cpu_backend(const detail::BackendBufferStorage& storage) {
    const auto& cpu_storage = require_cpu_backend_buffer(storage).storage();
    return TensorStorage::from_bytes(cpu_storage.descriptor(), cpu_storage.bytes());
}

TensorStorage matmul_f32_f16_gguf_weight(
    const TensorStorage& left,
    const TensorStorage& right
) {
    const auto rows = left.descriptor().shape.dimension(0);
    const auto shared = left.descriptor().shape.dimension(1);
    const auto columns = right.descriptor().shape.dimension(1);
    auto output = TensorStorage::allocate(TensorDescriptor::from(Shape::from({rows, columns}), DType::f32));

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::uint64_t inner = 0; inner < shared; ++inner) {
                const auto left_index = row * shared + inner;
                // GGUF/GGML stores dimension 0 contiguously, so each output column owns a packed input row.
                const auto right_index = column * shared + inner;
                sum += read_f32_linear(left, left_index) * read_f16_linear_as_f32(right, right_index);
            }

            write_f32_linear(output, row * columns + column, sum);
        }
    }

    return output;
}

TensorStorage matmul_f32_f16_gguf_weight_with_cpu_right_buffer(
    const TensorStorage& left,
    const CpuBackendBufferStorage& right
) {
    return matmul_f32_f16_gguf_weight(left, right.storage());
}

BackendBuffer matmul_f32_f16_gguf_weight_with_cpu_buffers(
    const CpuBackendBufferStorage& left,
    const CpuBackendBufferStorage& right
) {
    return copy_tensor_to_cpu_backend(matmul_f32_f16_gguf_weight(left.storage(), right.storage()));
}

ops::F32F16GGUFWeightTripleMatmulBackendBuffers
matmul_f32_f16_gguf_weight_triple_with_cpu_buffers(
    const CpuBackendBufferStorage& left,
    const CpuBackendBufferStorage& first_right,
    const CpuBackendBufferStorage& second_right,
    const CpuBackendBufferStorage& third_right
) {
    return ops::F32F16GGUFWeightTripleMatmulBackendBuffers {
        .first = copy_tensor_to_cpu_backend(matmul_f32_f16_gguf_weight(left.storage(), first_right.storage())),
        .second = copy_tensor_to_cpu_backend(matmul_f32_f16_gguf_weight(left.storage(), second_right.storage())),
        .third = copy_tensor_to_cpu_backend(matmul_f32_f16_gguf_weight(left.storage(), third_right.storage())),
    };
}

BackendBuffer silu_multiply_f32_with_cpu_buffers(
    const CpuBackendBufferStorage& gate,
    const CpuBackendBufferStorage& up
) {
    const auto output = ops::multiply_f32(ops::silu_f32(gate.storage()), up.storage());
    return copy_tensor_to_cpu_backend(output);
}

BackendBuffer gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
    const CpuBackendBufferStorage& left,
    const CpuBackendBufferStorage& gate_weight,
    const CpuBackendBufferStorage& up_weight,
    const CpuBackendBufferStorage& down_weight
) {
    const auto gate = matmul_f32_f16_gguf_weight(left.storage(), gate_weight.storage());
    const auto up = matmul_f32_f16_gguf_weight(left.storage(), up_weight.storage());
    const auto gated = ops::multiply_f32(ops::silu_f32(gate), up);
    return copy_tensor_to_cpu_backend(matmul_f32_f16_gguf_weight(gated, down_weight.storage()));
}

BackendBuffer residual_gated_silu_projection_f32_f16_gguf_weight_with_cpu_buffers(
    const CpuBackendBufferStorage& input,
    const CpuBackendBufferStorage& attention_output,
    const CpuBackendBufferStorage& attention_output_weight,
    const CpuBackendBufferStorage& feed_forward_norm_weight,
    const CpuBackendBufferStorage& gate_weight,
    const CpuBackendBufferStorage& up_weight,
    const CpuBackendBufferStorage& down_weight,
    float epsilon
) {
    const auto attention_projected = matmul_f32_f16_gguf_weight(
        attention_output.storage(),
        attention_output_weight.storage()
    );
    const auto attention_residual = ops::add_f32(input.storage(), attention_projected);
    const auto normalized = ops::rms_norm_f32(
        attention_residual,
        feed_forward_norm_weight.storage(),
        epsilon
    );
    const auto gate = matmul_f32_f16_gguf_weight(normalized, gate_weight.storage());
    const auto up = matmul_f32_f16_gguf_weight(normalized, up_weight.storage());
    const auto gated = ops::multiply_f32(ops::silu_f32(gate), up);
    const auto feed_forward_projected = matmul_f32_f16_gguf_weight(gated, down_weight.storage());
    return copy_tensor_to_cpu_backend(ops::add_f32(attention_residual, feed_forward_projected));
}

}  // namespace local_gguf_engine::tensor::backends::cpu
