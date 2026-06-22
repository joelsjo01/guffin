#include "local_gguf_engine/tensor/dtype.hpp"

namespace local_gguf_engine::tensor {

DTypeLayout dtype_layout(DType dtype) {
    switch (dtype) {
        case DType::f32:
            return DTypeLayout {.block_element_count = 1, .block_byte_count = 4};
        case DType::f16:
            return DTypeLayout {.block_element_count = 1, .block_byte_count = 2};
        case DType::q8_0:
            return DTypeLayout {.block_element_count = 32, .block_byte_count = 34};
        case DType::q4_k:
            return DTypeLayout {.block_element_count = 256, .block_byte_count = 144};
        case DType::q6_k:
            return DTypeLayout {.block_element_count = 256, .block_byte_count = 210};
    }

    throw TensorError(
        TensorErrorCode::unsupported_dtype,
        0,
        static_cast<std::uint64_t>(static_cast<std::uint8_t>(dtype))
    );
}

}  // namespace local_gguf_engine::tensor
