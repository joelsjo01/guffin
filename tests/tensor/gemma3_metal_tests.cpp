#include "tensor_test_support.hpp"
#include "test_support.hpp"

#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "local_gguf_engine/tensor/kquant_packing.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "../../src/tensor/backends/metal/metal_command_context.hpp"
#include "../../src/tensor/backends/metal/metal_test_hooks.hpp"
#include "../../src/tensor/backends/metal_backend.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace local_gguf_engine::tests {
namespace {
using namespace tensor;
using tensor::backends::metal::MetalCommandContext;
using runtime::BackendKind;

BackendBuffer buffer(std::initializer_list<std::uint64_t> shape, DType dtype = DType::f32) {
    return allocate_backend_buffer(TensorDescriptor::from(Shape::from(shape), dtype), BackendKind::metal);
}
BackendBuffer upload(std::uint32_t rows, std::uint32_t columns, const std::vector<float>& values) {
    return copy_tensor_to_backend(tensor_f32_matrix(rows, columns, values), BackendKind::metal);
}
std::vector<float> values(const BackendBuffer& input) {
    const auto copied = copy_tensor_from_backend(input);
    std::vector<float> result(copied.bytes().size() / sizeof(float));
    std::memcpy(result.data(), copied.bytes().data(), copied.bytes().size());
    return result;
}
std::vector<std::uint16_t> half_bits(const BackendBuffer& input) {
    const auto copied = copy_tensor_from_backend(input);
    expect(copied.descriptor().dtype == DType::f16, "half readback requires F16 storage");
    std::vector<std::uint16_t> result(copied.descriptor().element_count);
    std::memcpy(result.data(), copied.bytes().data(), copied.bytes().size());
    return result;
}
BackendBuffer cache_upload(std::uint32_t rows, std::uint32_t columns,
    const std::vector<float>& data, DType dtype) {
    return copy_tensor_to_backend(dtype == DType::f16
        ? tensor_f16_matrix(rows, columns, data) : tensor_f32_matrix(rows, columns, data), BackendKind::metal);
}
std::vector<float> cache_values(const BackendBuffer& input) {
    if (input.descriptor().dtype == DType::f32) return values(input);
    const auto bits = half_bits(input);
    std::vector<float> result(bits.size());
    std::transform(bits.begin(), bits.end(), result.begin(), f16_bits_to_f32);
    return result;
}
BackendBuffer words(std::span<const std::uint32_t> source) {
    auto result = buffer({source.size()});
    MetalCommandContext command("Gemma3 test staging", false, true);
    command.write_u32s(result, 0U, source);
    return result;
}
std::vector<float> pattern(std::size_t count, float scale = 0.15F) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = std::sin(static_cast<float>(i % 8191U) * 0.071F) * scale
            + std::cos(static_cast<float>(i % 997U) * 0.027F) * scale * 0.37F;
    }
    return result;
}
bool metal_available() { return tensor_backend_is_available(BackendKind::metal); }

// These modest magnitudes lie exactly on the binary16 grid. Attention tests
// therefore have an independent cache oracle without using the engine's F32
// narrowing helper to define rounding; the native cast test pins that separately.
std::vector<float> cache_pattern(std::size_t count, float scale, DType dtype) {
    auto result = pattern(count, scale);
    if (dtype == DType::f16)
        for (auto& value : result) value = std::round(value * 1024.0F) / 1024.0F;
    return result;
}

void native_half_cache_cast_rounding() {
    if (!metal_available()) return;
    struct Case { float input; std::uint16_t bits; };
    // Independently specified IEEE binary16 nearest-even results, including
    // negative zero, both tie parities, normal/subnormal transition and overflow.
    // No engine narrowing helper generates either inputs or expected bits.
    const std::array cases {
        Case {0.0F, 0x0000U}, Case {-0.0F, 0x8000U},
        Case {1.0F, 0x3c00U}, Case {-1.0F, 0xbc00U},
        Case {1.0F + 0x1p-11F, 0x3c00U}, Case {-1.0F - 0x1p-11F, 0xbc00U},
        Case {1.0F + 0x1.8p-10F, 0x3c02U}, Case {-1.0F - 0x1.8p-10F, 0xbc02U},
        Case {std::nextafter(1.0F + 0x1p-11F, 0.0F), 0x3c00U},
        Case {std::nextafter(1.0F + 0x1p-11F, 2.0F), 0x3c01U},
        Case {0x1p-24F, 0x0001U}, Case {-0x1p-24F, 0x8001U},
        Case {0x1p-25F, 0x0000U}, Case {-0x1p-25F, 0x8000U},
        Case {0x1.8p-24F, 0x0002U}, Case {-0x1.8p-24F, 0x8002U},
        Case {0x1p-14F - 0x1p-24F, 0x03ffU}, Case {0x1p-14F - 0x1p-25F, 0x0400U},
        Case {0x1p-14F, 0x0400U}, Case {-0x1p-14F, 0x8400U},
        Case {65504.0F, 0x7bffU}, Case {-65504.0F, 0xfbffU},
        Case {65519.0F, 0x7bffU}, Case {-65519.0F, 0xfbffU},
        Case {65520.0F, 0x7c00U}, Case {-65520.0F, 0xfc00U},
        Case {65536.0F, 0x7c00U}, Case {-65536.0F, 0xfc00U},
    };
    constexpr std::uint32_t width = 256U;
    std::vector<float> input(width);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = cases[i % cases.size()].input;
    const auto raw = upload(1U, width, input);
    const auto key = buffer({1U, width}, DType::f16), value = buffer({1U, width}, DType::f16);
    const std::array<std::uint32_t, 1> position {0U};
    const auto positions = words(position);
    MetalCommandContext command("precise binary16 cache narrowing", false, true);
    // Disable rotation to isolate the narrowing operation, including -0.
    command.encode_verifier_kv_store(raw, raw, positions, key, value,
        1U, 1U, width, 0U, 10000.0F);
    static_cast<void>(command.commit_and_wait());
    const auto keys = half_bits(key), vals = half_bits(value);
    for (std::size_t i = 0; i < input.size(); ++i) {
        expect(keys[i] == cases[i % cases.size()].bits && vals[i] == cases[i % cases.size()].bits,
            "native precise K/V narrowing must implement the pinned binary16 rounding contract at case "
                + std::to_string(i % cases.size()));
    }
}

void half_cache_numerical_status() {
    if (!metal_available()) return;
    constexpr std::uint32_t width = 256U;
    const auto key = buffer({1U, width}, DType::f16), value = buffer({1U, width}, DType::f16);
    const auto status = buffer({1U});
    const std::array<std::uint32_t, 1> position {0U};
    const auto positions = words(position);
    std::vector<float> finite(width, 0.25F);
    finite[3U] = 65519.0F; // Rounds to the maximum finite half; it is not overflow.
    finite[4U] = -65519.0F;
    finite[5U] = 0x1p-24F;
    const auto finite_raw = upload(1U, width, finite);
    for (const auto fault : {0U, 1U, 2U, 3U, 4U}) {
        auto raw_values = finite;
        if (fault == 1U || fault == 3U) raw_values[19U] = 65520.0F;
        if (fault == 2U) raw_values[19U] = std::numeric_limits<float>::quiet_NaN();
        if (fault == 4U) raw_values[19U] = -std::numeric_limits<float>::infinity();
        const auto changed = upload(1U, width, raw_values);
        const auto& raw_key = fault == 1U || fault == 4U ? changed : finite_raw;
        const auto& raw_value = fault == 2U || fault == 3U ? changed : finite_raw;
        MetalCommandContext command("checked half-cache numerical status", false, true);
        command.write_u32(status, 5U);
        command.encode_verifier_kv_store(raw_key, raw_value, positions, key, value,
            1U, 1U, width, 0U, 10000.0F, 1.0F, &status);
        // A later finite layer must not clear the first layer's failure bit.
        command.encode_memory_barrier();
        command.encode_verifier_kv_store(finite_raw, finite_raw, positions, key, value,
            1U, 1U, width, 0U, 10000.0F, 1.0F, &status);
        const auto stats = command.commit_and_wait();
        const auto expected = 5U | (fault == 0U ? 0U : tensor::backends::metal::verifier_kv_nonfinite_status);
        expect(command.read_verifier_status(status) == expected,
            "checked half stores report nonfinite K/V and preserve earlier status bits");
        expect(stats.compute_dispatch_count == 2U,
            "checking half-cache numerical status adds no dispatch");
    }
    const auto half_status = buffer({2U}, DType::f16);
    for (const bool precise : {false, true}) {
        const auto reject = [&](const BackendBuffer& destination, const BackendBuffer& flag) {
            MetalCommandContext command("checked half-cache invalid status", false, precise);
            bool rejected = false;
            try {
                command.encode_verifier_kv_store(finite_raw, finite_raw, positions, destination, destination,
                    1U, 1U, width, 0U, 10000.0F, 1.0F, &flag);
            } catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected && command.commit_and_wait().compute_dispatch_count == 0U,
                "invalid checked cache policy/status fails before dispatch");
        };
        reject(key, half_status);
        reject(key, finite_raw);
        reject(finite_raw, status);
        if (!precise) reject(key, status);
    }
}

// This follows the original Q4 consumer's worker/component offsets, rather
// than the inverse permutation used by the fused producers.
void check_q4_sum_receipt(const std::vector<float>& input, const BackendBuffer& receipt,
    std::uint32_t rows, std::uint32_t width) {
    const auto sums = values(receipt);
    constexpr std::array<std::uint32_t, 4> offsets {0U, 32U, 128U, 160U};
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t block = 0; block < width / 256U; ++block)
            for (std::uint32_t worker = 0; worker < 8U; ++worker)
                for (std::uint32_t component = 0; component < 4U; ++component) {
                    const auto source = row * width + block * 256U + 64U * (worker / 4U)
                        + 8U * (worker % 4U) + offsets[component];
                    float sum = 0.0F;
                    for (std::uint32_t i = 0; i < 8U; ++i) sum += input[source + i];
                    const auto index = row * width / 8U + block * 32U + worker * 4U + component;
                    expect(std::bit_cast<std::uint32_t>(sums[index]) == std::bit_cast<std::uint32_t>(sum),
                        "fused Q4 sums preserve the consumer's eight ordered F32 additions");
                }
}

void compact_q4_worker_partitions_and_column_tails() {
    if (!metal_available()) return;
    // Fewer than four blocks leaves logical K workers inactive; five blocks
    // gives only one partition a second iteration. Every padded column must
    // still reach the transpose barrier and publish zero partials.
    for (const auto width : {256U, 768U, 1024U, 1280U})
        for (const auto columns : {1U, 7U, 8U, 9U}) {
            auto host = tensor_q4_k_stress_gguf_weight_matrix(width, columns, 419U);
            // The shared fixture repeats its nibble pattern every 16 values.
            // Give all eight groups different codes so swapped low/high
            // groups or incorrect SIMD sharing cannot hide behind that repeat.
            auto bytes = host.mutable_bytes();
            for (std::uint32_t column = 0U; column < columns; ++column)
                for (std::uint32_t block = 0U; block < width / 256U; ++block)
                    for (std::uint32_t pair = 0U; pair < 4U; ++pair)
                        for (std::uint32_t i = 0U; i < 32U; ++i) {
                            const auto low = (7U * i + 6U * pair + 5U * block + 11U * column) % 16U;
                            const auto high = (low + 3U) % 16U;
                            bytes[(column * (width / 256U) + block) * 144U + 16U + pair * 32U + i]
                                = static_cast<std::uint8_t>(low | (high << 4U));
                        }
            const auto native = copy_tensor_to_backend(host, BackendKind::metal);
            const auto packed = copy_tensor_to_backend(repack_kquant_weight_columns8(host.view()), BackendKind::metal);
            const auto input = upload(2U, width, pattern(2U * width, 0.73F));
            const auto gate = upload(2U, width, pattern(2U * width, 1.3F));
            for (const auto rows : {1U, 2U}) {
                const auto activation = buffer({rows, width});
                const auto sums = buffer({rows, width / 8U});
                const auto expected = buffer({rows, columns});
                const auto ordinary = buffer({rows, columns});
                const auto prepared = buffer({rows, columns});
                MetalCommandContext command("compact Q4 worker and column boundaries", false, true);
                command.encode_gelu_tanh_multiply(gate, input, activation, rows * width, &sums);
                command.encode_memory_barrier();
                command.encode_verifier_matmul(activation, native, expected, rows);
                command.encode_verifier_matmul(activation, packed, ordinary, rows);
                command.encode_verifier_matmul(activation, packed, prepared, rows, false, &sums);
                static_cast<void>(command.commit_and_wait());
                const auto reference = values(expected);
                expect(values(ordinary) == reference && values(prepared) == reference,
                    "compact Q4 retains canonical R1/R2 arithmetic with inactive workers and padded columns");
                check_q4_sum_receipt(values(activation), sums, rows, width);
            }
        }
}

void fused_q4_rms_and_gelu_receipts(bool precise = true) {
    if (!metal_available()) return;
    for (const std::uint32_t rows : {1U, 2U}) for (const std::uint32_t width : {3584U, 3840U, 4096U, 14336U, 15360U}) {
        auto data = pattern((rows + 1U) * width, 2.1F);
        data[width + 3U] = 1.0F + std::ldexp(1.0F, -20);
        data[width + 7U] = -data[width + 3U];
        const auto input = upload(rows + 1U, width, data);
        auto norm_values = pattern(width, 0.2F);
        for (auto& value : norm_values) value += 0.9F;
        const auto norm = upload(1U, width, norm_values);
        const auto reference = buffer({rows, width});
        constexpr float guard = -123.25F;
        const auto owner = upload(1U, rows * width + 34U, std::vector<float>(rows * width + 34U, guard));
        const auto sums_owner = upload(1U, rows * width / 8U + 34U,
            std::vector<float>(rows * width / 8U + 34U, guard));
        const auto output = tensor::backends::metal::shared_metal_buffer_view_for_testing(owner,
            TensorDescriptor::from(Shape::from({rows, width}), DType::f32), 17U * sizeof(float));
        const auto sums = tensor::backends::metal::shared_metal_buffer_view_for_testing(sums_owner,
            TensorDescriptor::from(Shape::from({rows, width / 8U}), DType::f32), 17U * sizeof(float));
        const auto packed_host = repack_kquant_weight_columns8(tensor_q4_k_stress_gguf_weight_matrix(width, 37U, 713U).view());
        const auto weight = copy_tensor_to_backend(packed_host, BackendKind::metal);
        const auto ordinary_projection = buffer({rows, 37U}), prepared_projection = buffer({rows, 37U});
        const auto up = upload(rows, width, pattern(rows * width, 0.7F));
        const auto gate = upload(rows, width, pattern(rows * width, 1.9F));
        for (const bool gelu : {false, true}) {
            MetalCommandContext command("fused Q4 producer parity", false, precise);
            if (gelu) {
                command.encode_gelu_tanh_multiply(gate, up, reference, rows * width);
                command.encode_gelu_tanh_multiply(gate, up, output, rows * width, &sums);
            } else {
                command.encode_rms_norm(input, norm, reference, 1.0e-6F, rows, 1U);
                command.encode_rms_norm(input, norm, output, 1.0e-6F, rows, 1U, &sums);
            }
            command.encode_memory_barrier();
            command.encode_verifier_matmul(reference, weight, ordinary_projection, rows);
            command.encode_verifier_matmul(output, weight, prepared_projection, rows, false, &sums);
            const auto stats = command.commit_and_wait();
            expect(stats.compute_dispatch_count == 4U && stats.memory_barrier_count == 1U,
                "fused Q4 producers and consumers add no preparation dispatch or barrier");
            const auto expected = values(reference);
            expect(values(output) == expected, "fused RMS/GELU output retains precise F32 arithmetic");
            check_q4_sum_receipt(expected, sums, rows, width);
            expect(values(prepared_projection) == values(ordinary_projection),
                "prepared R1/R2 Q4 projection preserves native output including column tails");
            for (const auto* parent : {&owner, &sums_owner}) {
                const auto guarded = values(*parent);
                expect(std::all_of(guarded.begin(), guarded.begin() + 17U, [](float v) { return v == guard; })
                    && std::all_of(guarded.end() - 17U, guarded.end(), [](float v) { return v == guard; }),
                    "fused producers respect nonzero views and output/sum guard regions");
            }
        }
    }
}

void fused_q4_attention_receipts(bool precise = true) {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 2U, heads = 16U, kv_heads = 8U, dimension = 256U, capacity = 4096U;
    constexpr auto width = heads * dimension;
    const auto query = upload(rows, width, pattern(rows * width, 0.3F));
    const std::array<std::uint32_t, 3> ancestor_ids {0U, 0U, 1U};
    const auto ancestors = words(ancestor_ids);
    const auto reference = buffer({rows, width}), scalar = buffer({1U, width});
    const auto scalar_sums = buffer({1U, width / 8U});
    constexpr float guard = -123.25F;
    const auto output_owner = upload(1U, rows * width + 34U, std::vector<float>(rows * width + 34U, guard));
    const auto sums_owner = upload(1U, rows * width / 8U + 34U,
        std::vector<float>(rows * width / 8U + 34U, guard));
    const auto output = tensor::backends::metal::shared_metal_buffer_view_for_testing(output_owner,
        TensorDescriptor::from(Shape::from({rows, width}), DType::f32), 17U * sizeof(float));
    const auto sums = tensor::backends::metal::shared_metal_buffer_view_for_testing(sums_owner,
        TensorDescriptor::from(Shape::from({rows, width / 8U}), DType::f32), 17U * sizeof(float));
    const auto partial = buffer({rows, 32U, heads, dimension});
    const auto maxima = buffer({rows, 32U, heads}), denominators = buffer({rows, 32U, heads});
    for (const auto dtype : {DType::f32, DType::f16}) {
        const auto key = cache_upload(capacity, kv_heads * dimension,
            cache_pattern(capacity * kv_heads * dimension, 0.19F, dtype), dtype);
        const auto value = cache_upload(capacity, kv_heads * dimension,
            cache_pattern(capacity * kv_heads * dimension, 0.73F, dtype), dtype);
        const auto private_key = cache_upload(rows, kv_heads * dimension,
            cache_pattern(rows * kv_heads * dimension, 0.27F, dtype), dtype);
        const auto private_value = cache_upload(rows, kv_heads * dimension,
            cache_pattern(rows * kv_heads * dimension, 0.91F, dtype), dtype);
        for (const auto visible : std::array<std::array<std::uint32_t, 2>, 6>{{
                 {1U, 2U}, {255U, 257U}, {511U, 513U}, {1023U, 1025U}, {2047U, 2049U}, {4095U, 4097U}}}) {
            const std::array<std::uint32_t, 8> schedule {0U, visible[0] - 1U, 0U, 1U, 0U, visible[1] - 2U, 1U, 2U};
            const auto schedules = words(schedule);
            std::uint32_t ordinary_dispatches = 0U;
            for (const bool fused : {false, true}) {
                MetalCommandContext command("fused Q4 attention parity", false, precise);
                command.encode_verifier_attention(query, key, value, private_key, private_value, schedules,
                    0U, ancestors, fused ? output : reference, partial, maxima, denominators, rows,
                    capacity - 1U, heads, kv_heads, dimension, dimension, 1.0F / 16.0F, 0.0F,
                    fused ? &sums : nullptr);
                const auto stats = command.commit_and_wait();
                if (!fused) ordinary_dispatches = stats.compute_dispatch_count;
                else expect(stats.compute_dispatch_count == ordinary_dispatches
                    && stats.compute_dispatch_count == (visible[1] <= 256U ? 1U : 2U),
                    "attention receipts add no dispatch for single, mixed or multiple partitions");
            }
            const auto expected = values(reference);
            expect(values(output) == expected, "fused half/float-cache attention retains final precise F32 values");
            check_q4_sum_receipt(expected, sums, rows, width);
            MetalCommandContext command("scalar versus mixed-row fused attention", false, precise);
            command.encode_verifier_attention(query, key, value, private_key, private_value, schedules,
                0U, ancestors, scalar, partial, maxima, denominators, 1U,
                capacity - 1U, heads, kv_heads, dimension, dimension, 1.0F / 16.0F, 0.0F,
                &scalar_sums);
            static_cast<void>(command.commit_and_wait());
            const auto scalar_output = values(scalar);
            expect(std::equal(scalar_output.begin(), scalar_output.end(), expected.begin()),
                "each ordinary/verifier row retains its own attention partition policy");
            check_q4_sum_receipt(scalar_output, scalar_sums, 1U, width);
            for (const auto* owner : {&output_owner, &sums_owner}) {
                const auto guarded = values(*owner);
                expect(std::all_of(guarded.begin(), guarded.begin() + 17U, [](float v) { return v == guard; })
                    && std::all_of(guarded.end() - 17U, guarded.end(), [](float v) { return v == guard; }),
                    "attention receipts preserve offset views and output guards");
            }
        }
    }
}

void fused_q4_receipt_validation() {
    if (!metal_available()) return;
    const auto input = buffer({3U, 256U}), weight = buffer({256U}), output = buffer({3U, 256U});
    const auto sums = buffer({3U, 32U}), small = buffer({31U}), half_sums = buffer({3U, 32U}, DType::f16);
    const auto bad_width = buffer({1U, 260U});
    const auto native_host = tensor_q4_k_stress_gguf_weight_matrix(256U, 37U, 721U);
    const auto compact = copy_tensor_to_backend(repack_kquant_weight_columns8(native_host.view()), BackendKind::metal);
    const auto native = copy_tensor_to_backend(native_host, BackendKind::metal);
    const auto q6 = copy_tensor_to_backend(repack_kquant_weight_columns8(
        tensor_q6_k_stress_gguf_weight_matrix(256U, 37U, 723U).view()), BackendKind::metal);
    const auto reject = [&](const auto& encode, bool precise = true) {
        MetalCommandContext command("invalid fused Q4 receipt", false, precise);
        bool rejected = false;
        try { encode(command); } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "invalid fused Q4 storage or execution contract must be rejected");
        expect(command.commit_and_wait().compute_dispatch_count == 0U,
            "invalid fused Q4 contract must fail before dispatch");
    };
    for (const auto* invalid : {&small, &half_sums, &input, &output, &weight}) {
        reject([&](auto& c) { c.encode_rms_norm(input, weight, output, 1.0e-6F, 1U, 0U, invalid); });
        reject([&](auto& c) { c.encode_gelu_tanh_multiply(input, weight, output, 256U, invalid); });
    }
    reject([&](auto& c) { c.encode_rms_norm(input, weight, input, 1.0e-6F, 1U, 0U, &sums); });
    reject([&](auto& c) { c.encode_rms_norm(bad_width, output, output, 1.0e-6F, 1U, 0U, &sums); });
    reject([&](auto& c) { c.encode_gelu_tanh_multiply(input, input, input, 256U, &sums); });
    reject([&](auto& c) { c.encode_gelu_tanh_multiply(input, input, output, 255U, &sums); });
    for (const auto* invalid : {&small, &half_sums, &input, &output})
        reject([&](auto& c) { c.encode_verifier_matmul(input, compact, output, 1U, false, invalid); });
    reject([&](auto& c) { c.encode_verifier_matmul(input, compact, input, 1U, false, &sums); });
    reject([&](auto& c) { c.encode_verifier_matmul(input, compact, output, 1U, true, &sums); });
    reject([&](auto& c) { c.encode_verifier_matmul(input, compact, output, 3U, false, &sums); });
    reject([&](auto& c) { c.encode_verifier_matmul(input, native, output, 1U, false, &sums); });
    reject([&](auto& c) { c.encode_verifier_matmul(input, q6, output, 1U, false, &sums); });
    const auto query = buffer({1U, 512U}), cache = buffer({1U, 256U});
    const auto half_cache = buffer({1U, 256U}, DType::f16), attention_output = buffer({1U, 512U});
    const auto partial = buffer({1U, 32U, 2U, 256U}), maxima = buffer({1U, 32U, 2U}), denominators = buffer({1U, 32U, 2U});
    const std::array<std::uint32_t, 4> schedule {0U, 0U, 0U, 1U};
    const std::array<std::uint32_t, 1> ancestor {0U};
    const auto schedules = words(schedule), ancestors = words(ancestor);
    const auto attention = [&](auto& c, const BackendBuffer& source, const BackendBuffer& receipt) {
        c.encode_verifier_attention(query, source, source, source, source, schedules, 0U, ancestors,
            attention_output, partial, maxima, denominators, 1U, 0U, 2U, 1U, 256U, 256U,
            1.0F / 16.0F, 0.0F, &receipt);
    };
    for (const auto* invalid : {&small, &half_sums, &query, &cache, &attention_output, &partial, &maxima, &denominators})
        reject([&](auto& c) { attention(c, cache, *invalid); });
    reject([&](auto& c) { attention(c, half_cache, half_sums); });
    reject([&](auto& c) {
        c.encode_verifier_attention(query, half_cache, half_cache, half_cache, cache, schedules, 0U, ancestors,
            attention_output, partial, maxima, denominators, 1U, 0U, 2U, 1U, 256U, 256U,
            1.0F / 16.0F, 0.0F, &sums);
    });
}

void gemma2_compact_projection_parity() {
    if (!metal_available()) return;
    constexpr std::uint32_t columns = 37U;
    for (const auto width : {3584U, 4096U, 14336U}) {
        const auto input_values = pattern(3U * width, 0.027F);
        const auto input = upload(3U, width, input_values);
        for (const auto dtype : {DType::q4_k, DType::q6_k}) {
            const auto host = dtype == DType::q4_k
                ? tensor_q4_k_stress_gguf_weight_matrix(width, columns, 51U)
                : tensor_q6_k_stress_gguf_weight_matrix(width, columns, 71U);
            const auto weight = copy_tensor_to_backend(host, BackendKind::metal);
            const auto packed = dtype == DType::q4_k
                ? copy_tensor_to_backend(repack_kquant_weight_columns8(host.view()), BackendKind::metal)
                : weight;
            const auto output = buffer({2U, columns}), scalar = buffer({1U, columns});
            const auto baseline = buffer({2U, columns});
            MetalCommandContext command("Gemma2 compact actual inner dimensions");
            command.encode_matmul(input, weight, baseline, 2U);
            command.encode_verifier_matmul(input, packed, output, 2U);
            static_cast<void>(command.commit_and_wait());
            const auto batch = values(output), before = values(baseline);
            for (std::uint32_t row = 0U; row < 2U; ++row) {
                command.restart();
                command.encode_target_head_rows(input, row, packed, scalar, 1U);
                static_cast<void>(command.commit_and_wait());
                const auto single = values(scalar);
                expect(std::equal(single.begin(), single.end(), batch.begin() + row * columns),
                    "Gemma2 compact R1/R2 must compute the same conditional projection");
                for (std::uint32_t column = 0U; column < columns; ++column) {
                    double reference = 0.0, absolute_sum = 0.0;
                    for (std::uint32_t inner = 0U; inner < width; ++inner) {
                        const auto block = column * (width / 256U) + inner / 256U;
                        const auto decoded = dtype == DType::q4_k
                            ? q4_k_block_view(host, block).dequantized_value(inner % 256U)
                            : q6_k_block_view(host, block).dequantized_value(inner % 256U);
                        const auto product = double(input_values[row * width + inner]) * decoded;
                        reference += product; absolute_sum += std::abs(product);
                    }
                    const auto rounding = double(width / 32U + 32U) * std::numeric_limits<float>::epsilon();
                    const auto bound = float(rounding / (1.0 - rounding) * absolute_sum + 1.0e-6);
                    expect_close(single[column], float(reference), bound,
                        "Gemma2 compact projection retains the existing independent dot bound");
                    expect_close(before[row * columns + column], float(reference), bound,
                        "Gemma2 original projection satisfies the same independent dot bound");
                }
            }
        }
    }
}

void gemma2_packed_prompt_retains_half_operands() {
    if (!metal_available()) return;
    // All distinct retained Gemma 2 body shapes, plus output tails, exercise
    // the old M32/M16 K32 half-operand pipeline against its packed specialization.
    for (const auto shape : std::array<std::array<std::uint32_t, 2>, 6>{{
             {3584U, 4096U}, {3584U, 2048U}, {4096U, 3584U},
             {3584U, 14336U}, {14336U, 3584U}, {3584U, 37U}}}) {
        const auto width = shape[0], columns = shape[1];
        const auto input = upload(33U, width, pattern(33U * width, 0.027F));
        for (const auto dtype : {DType::q4_k}) {
            const auto host = dtype == DType::q4_k
                ? tensor_q4_k_stress_gguf_weight_matrix(width, columns, 117U)
                : tensor_q6_k_stress_gguf_weight_matrix(width, columns, 137U);
            const auto native = copy_tensor_to_backend(host, BackendKind::metal);
            const auto packed = copy_tensor_to_backend(repack_kquant_weight_columns8(host.view()), BackendKind::metal);
            const auto expected = buffer({33U, columns}), actual = buffer({33U, columns});
            const auto generic = buffer({33U, columns});
            for (const auto rows : {1U, 2U, 4U, 8U, 9U, 16U, 17U, 33U}) {
                MetalCommandContext command("Gemma2 packed prompt numerical contract");
                command.encode_matmul(input, native, expected, rows);
                command.encode_verifier_matmul(input, packed, actual, rows, true);
                command.encode_matmul(input, packed, generic, rows);
                static_cast<void>(command.commit_and_wait());
                const auto a = values(actual), b = values(expected), g = values(generic);
                expect(std::equal(g.begin(), g.begin() + rows * columns, b.begin()),
                    "legacy Gemma2 prefill encoder reads its sole packed body exactly: dtype="
                        + std::to_string(static_cast<int>(dtype)) + " rows=" + std::to_string(rows)
                        + " K=" + std::to_string(width));
                expect(std::equal(a.begin(), a.begin() + rows * columns, b.begin()),
                    "Gemma2 packed prompt must preserve every legacy half-operand result exactly");
            }
        }
    }
}

void native_q6_embedding_and_head() {
    if (!metal_available()) return;
    constexpr std::uint32_t hidden = 3840U, vocabulary = 7U, rows = 3U;
    const auto weight = tensor_q6_k_stress_gguf_weight_matrix(hidden, vocabulary, 137U);
    const auto resident = copy_tensor_to_backend(weight, BackendKind::metal);
    const std::array<std::uint32_t, 4> ids {6U, 2U, 0U, 5U};
    const auto tokens = words(ids);
    const auto embeddings = buffer({rows, hidden});
    const float scale = std::sqrt(static_cast<float>(hidden));
    MetalCommandContext command("Gemma3 native tied Q6 embedding/head", false, true);
    command.encode_embedding(tokens, resident, embeddings, scale, 1U, rows);
    static_cast<void>(command.commit_and_wait());
    const auto actual = values(embeddings);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < hidden; ++column) {
            const auto block = q6_k_block_view(weight,
                ids[row + 1U] * (hidden / 256U) + column / 256U);
            expect_close(actual[row * hidden + column], block.dequantized_value(column % 256U) * scale, 2.0e-6F,
                "native Q6 embedding uses every signed scale and high-bit group");
        }
    }
    const auto input_values = pattern(4U * hidden, 0.03F);
    const auto input_host = tensor_f32_matrix(4U, hidden, input_values);
    const auto input = copy_tensor_to_backend(input_host, BackendKind::metal);
    const auto logits = buffer({rows, vocabulary});
    command.restart();
    command.encode_target_head_rows(input, 1U, resident, logits, rows);
    static_cast<void>(command.commit_and_wait());
    const auto head = values(logits);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t token = 0; token < vocabulary; ++token) {
            double sum = 0.0;
            for (std::uint32_t column = 0; column < hidden; ++column) {
                const auto block = q6_k_block_view(weight, token * (hidden / 256U) + column / 256U);
                sum += static_cast<double>(input_values[(row + 1U) * hidden + column])
                    * block.dequantized_value(column % 256U);
            }
            expect_close(head[row * vocabulary + token], static_cast<float>(sum), 3.0e-5F,
                "native Q6 output head agrees with independent decoded dot product");
        }
    }
}

void head_norm_and_scaled_rope() {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 4U, heads = 3U, dimension = 256U;
    auto input_values = pattern(rows * heads * dimension, 1.3F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t head = 0; head < heads; ++head) {
            for (std::uint32_t column = 0; column < dimension; ++column) {
                input_values[(row * heads + head) * dimension + column] *= static_cast<float>(head + 1U);
            }
        }
    }
    auto weights = pattern(dimension, 0.2F);
    for (auto& value : weights) value += 0.7F;
    weights[19] = -0.25F;
    const std::array<std::uint32_t, rows> position_values {1U, 1024U, 32769U, 131071U};
    const auto positions = words(position_values);
    const auto input = upload(rows, heads * dimension, input_values);
    const auto weight = upload(1U, dimension, weights);
    const auto normalized = buffer({rows, heads * dimension});
    const auto output = buffer({rows, heads * dimension});
    const auto value = upload(rows, heads * dimension, pattern(rows * heads * dimension, 0.07F));
    const auto stored_key = buffer({rows, heads, dimension});
    const auto stored_value = buffer({rows, heads, dimension});
    const auto half_key = buffer({rows, heads, dimension}, DType::f16);
    const auto half_value = buffer({rows, heads, dimension}, DType::f16);
    constexpr float epsilon = 1.0e-6F;
    for (const auto frequency : std::array<std::array<float, 2>, 2>{{{10000.0F, 1.0F}, {1000000.0F, 0.125F}}}) {
        MetalCommandContext command("Gemma3 per-head normalization and position policy", false, true);
        command.encode_head_rms_norm(input, weight, normalized, epsilon, rows, heads, dimension);
        command.encode_memory_barrier();
        command.encode_verifier_rope(normalized, positions, output, rows, heads, dimension,
            dimension, frequency[0], frequency[1]);
        command.encode_verifier_kv_store(normalized, value, positions, stored_key, stored_value,
            rows, heads, dimension, dimension, frequency[0], frequency[1]);
        command.encode_memory_barrier();
        // The V input here is the independently checked F32 query rotation.
        // Equal half outputs prove that key narrowing happens after rotation,
        // including at high positions and with global frequency scaling.
        command.encode_verifier_kv_store(normalized, output, positions, half_key, half_value,
            rows, heads, dimension, dimension, frequency[0], frequency[1]);
        static_cast<void>(command.commit_and_wait());
        const auto norm = values(normalized);
        const auto rotated = values(output);
        const auto cache = values(stored_key);
        expect(cache == rotated, "query and key use identical half-split RoPE math");
        expect(values(stored_value) == values(value), "F32 value store does not narrow precision");
        expect(half_bits(half_key) == half_bits(half_value), "F16 key store rounds only after F32 RoPE");
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t head = 0; head < heads; ++head) {
                const auto offset = (row * heads + head) * dimension;
                double squares = 0.0;
                for (std::uint32_t column = 0; column < dimension; ++column) {
                    squares += static_cast<double>(input_values[offset + column]) * input_values[offset + column];
                }
                const double factor = 1.0 / std::sqrt(squares / dimension + epsilon);
                for (std::uint32_t column = 0; column < dimension; ++column) {
                    expect_close(norm[offset + column], static_cast<float>(input_values[offset + column]
                        * factor * weights[column]), 6.0e-7F, "independent per-head RMS denominator");
                }
                for (std::uint32_t pair = 0; pair < dimension / 2U; ++pair) {
                    const double angle = static_cast<double>(position_values[row])
                        * std::pow(static_cast<double>(frequency[0]), -static_cast<double>(2U * pair) / dimension)
                        * frequency[1];
                    const float a = norm[offset + pair], b = norm[offset + pair + dimension / 2U];
                    // GPU pow and division form an F32 phase. At long context,
                    // their rounding error grows with the position. Bound four
                    // relative F32 rounding units, then use |sin'|,|cos'| <= 1
                    // to propagate phase error through the rotation. At pos1
                    // this remains a few micro-units rather than a broad floor.
                    const auto tolerance = static_cast<float>(4.0 * std::numeric_limits<float>::epsilon()
                        * std::abs(angle) * (std::abs(a) + std::abs(b)) + 2.0e-6);
                    expect_close(rotated[offset + pair], static_cast<float>(a * std::cos(angle) - b * std::sin(angle)),
                        tolerance, "scaled half-split first RoPE component at high logical position");
                    expect_close(rotated[offset + pair + dimension / 2U], static_cast<float>(a * std::sin(angle) + b * std::cos(angle)),
                        tolerance, "scaled half-split second RoPE component at high logical position");
                }
            }
        }
    }
}

void attention_dispatch_uses_host_staged_visible_rows() {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 3U, heads = 2U, dimension = 256U, capacity = 8192U;
    const auto query = upload(rows, heads * dimension, std::vector<float>(rows * heads * dimension, 0.0F));
    std::vector<float> keys(capacity * dimension, 0.0F), resident_values(keys.size());
    std::vector<float> private_keys(rows * dimension, 0.0F), private_values(private_keys.size());
    for (std::uint32_t token = 0; token < capacity; ++token) {
        for (std::uint32_t column = 0; column < dimension; ++column) {
            resident_values[token * dimension + column] = static_cast<float>(int(token % 17U) - 8) / 32.0F
                + static_cast<float>(column % 13U) / 128.0F;
        }
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < dimension; ++column) {
            private_values[row * dimension + column] = static_cast<float>(row + 1U) / 8.0F
                + static_cast<float>(column % 13U) / 128.0F;
        }
    }
    const std::array<std::uint32_t, rows> ancestors_values {0U, 1U, 2U};
    const auto ancestors = words(ancestors_values);
    const auto output = buffer({rows, heads, dimension});
    const auto partial = buffer({rows, 32U, heads, dimension});
    const auto maxima = buffer({rows, 32U, heads});
    const auto denominators = buffer({rows, 32U, heads});
    constexpr std::array<std::array<std::uint32_t, rows>, 8> counts {{
        {1U, 1U, 1U}, {1U, 256U, 255U}, {1U, 256U, 257U}, {512U, 1U, 513U},
        {1U, 257U, 1024U}, {1024U, 256U, 1025U}, {2048U, 1U, 2049U}, {4096U, 1U, 4097U}
    }};
    for (const auto dtype : {DType::f16, DType::f32}) {
        const auto cache = [&](std::uint32_t count, const std::vector<float>& data) {
            return copy_tensor_to_backend(dtype == DType::f16
                ? tensor_f16_matrix(count, dimension, data)
                : tensor_f32_matrix(count, dimension, data), BackendKind::metal);
        };
        const auto key = cache(capacity, keys), value = cache(capacity, resident_values);
        const auto private_key = cache(rows, private_keys), private_value = cache(rows, private_values);
        for (const auto& visible : counts) {
            // Poison padding exposes accidental omission of either the parent
            // view offset or the per-layer schedule offset. All values are exact
            // in F16 and F32; zero queries make the independent result a mean.
            std::array<std::uint32_t, 24> packed;
            packed.fill(std::numeric_limits<std::uint32_t>::max());
            for (std::uint32_t row = 0; row < rows; ++row) {
                const std::array<std::uint32_t, 4> schedule {0U, visible[row] - 1U, row, 1U};
                std::copy(schedule.begin(), schedule.end(), packed.begin() + 8U + row * 4U);
            }
            const auto parent = words(packed);
#if defined(LOCAL_GGUF_ENGINE_TESTING)
            const auto schedules = tensor::backends::metal::shared_metal_buffer_view_for_testing(parent,
                TensorDescriptor::from(Shape::from({16U}), DType::f32), 4U * sizeof(std::uint32_t));
            constexpr auto offset = 4U * sizeof(std::uint32_t);
#else
            const auto& schedules = parent;
            constexpr auto offset = 8U * sizeof(std::uint32_t);
#endif
            MetalCommandContext command("attention visible-row dispatch", false, dtype == DType::f32);
            command.encode_verifier_attention(query, key, value, private_key, private_value, schedules,
                offset, ancestors, output, partial, maxima, denominators, rows, capacity - 1U,
                heads, 1U, dimension, dimension, 1.0F / 16.0F, dtype == DType::f16 ? 50.0F : 0.0F);
            const auto stats = command.commit_and_wait();
            const bool single = *std::max_element(visible.begin(), visible.end()) <= 256U;
            expect(stats.compute_dispatch_count == (single ? 1U : 2U)
                && stats.memory_barrier_count == (single ? 0U : 1U),
                "attention omits reduction only when every visible row has one partition");
            const auto actual = values(output);
            for (std::uint32_t row = 0; row < rows; ++row) {
                for (std::uint32_t column = 0; column < dimension; ++column) {
                    double sum = private_values[row * dimension + column];
                    for (std::uint32_t token = 0; token + 1U < visible[row]; ++token)
                        sum += resident_values[token * dimension + column];
                    const auto expected = static_cast<float>(sum / visible[row]);
                    for (std::uint32_t head = 0; head < heads; ++head)
                        expect_close(actual[(row * heads + head) * dimension + column], expected, 2.0e-6F,
                            "pruned F16/F32 attention covers every resident and private token");
                }
            }
        }
    }
}

void attention_dispatch_rejects_invalid_host_schedules() {
    if (!metal_available()) return;
    const auto query = buffer({1U, 2U, 256U}), cache = buffer({1U, 256U});
    const auto output = buffer({1U, 2U, 256U}), partial = buffer({1U, 32U, 2U, 256U});
    const auto scalars = buffer({1U, 32U, 2U});
    const std::array<std::uint32_t, 1> ancestor {0U};
    const auto ancestors = words(ancestor);
    const auto reject = [&](std::span<const std::uint32_t> packed, std::uint64_t offset) {
        const auto schedules = words(packed);
        MetalCommandContext command("invalid attention schedule", false, true);
        bool rejected = false;
        try {
            command.encode_verifier_attention(query, cache, cache, cache, cache, schedules,
                offset, ancestors, output, partial, scalars, scalars, 1U, 0U,
                2U, 1U, 256U, 256U, 1.0F / 16.0F, 0.0F);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "invalid host attention schedule must fail before dispatch");
        expect(command.commit_and_wait().compute_dispatch_count == 0U,
            "invalid attention schedule encoded GPU work");
    };
    const std::array<std::uint32_t, 4> valid {0U, 0U, 0U, 1U};
    reject(std::span(valid).first(3U), 0U);
    reject(valid, 4U);
    reject(valid, 1U);
    reject(valid, std::numeric_limits<std::uint64_t>::max() - 3U);
    const std::array<std::uint32_t, 4> empty {0U, 0U, 0U, 0U};
    reject(empty, 0U);
    const std::array<std::uint32_t, 4> overflow {0U, std::numeric_limits<std::uint32_t>::max(), 0U, 1U};
    reject(overflow, 0U);
}

void private_attention_ring_and_publication(DType dtype) {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 3U, dimension = 256U, capacity = 1024U;
    const std::uint32_t query_heads = dtype == DType::f16 ? 16U : 4U, kv_heads = query_heads / 2U;
    const auto q = pattern(rows * query_heads * dimension, 0.3F);
    const auto k = cache_pattern(capacity * kv_heads * dimension, 0.19F, dtype);
    const auto v = cache_pattern(capacity * kv_heads * dimension, 0.73F, dtype);
    const auto pk = cache_pattern(rows * kv_heads * dimension, 0.27F, dtype);
    auto pv = cache_pattern(rows * kv_heads * dimension, 0.91F, dtype);
    // Distinguish sibling 1 strongly without changing the branch 0 -> 2.
    if (dtype == DType::f16)
        for (std::uint32_t column = 0; column < kv_heads * dimension; ++column)
            pv[kv_heads * dimension + column] = 4.0F
                + std::round(pv[kv_heads * dimension + column] * 128.0F) / 128.0F;
    const auto query = upload(rows, query_heads * dimension, q);
    constexpr float guard = -13.25F;
    std::vector<BackendBuffer> cache_owners;
    const auto cache = [&](std::uint32_t count, const std::vector<float>& data) {
        if (dtype == DType::f32) return upload(count, kv_heads * dimension, data);
        // Twelve bytes is four-byte aligned but not half4 aligned; this also
        // exercises application of the backing-view offset on each cache read.
        std::vector<float> guarded(data.size() + 12U, guard);
        std::copy(data.begin(), data.end(), guarded.begin() + 6U);
        auto owner = cache_upload(1U, static_cast<std::uint32_t>(guarded.size()), guarded, dtype);
        auto view = tensor::backends::metal::shared_metal_buffer_view_for_testing(owner,
            TensorDescriptor::from(Shape::from({count, kv_heads * dimension}), dtype), 12U);
        cache_owners.push_back(std::move(owner));
        return view;
    };
    const auto key = cache(capacity, k), value = cache(capacity, v);
    const auto private_key = cache(rows, pk), private_value = cache(rows, pv);
    // Last row branches from private row 0 through row 2, excluding sibling 1.
    const std::array<std::uint32_t, 5> ancestor_values {0U, 0U, 1U, 0U, 2U};
    const auto ancestors = words(ancestor_values);
    const std::array<std::array<std::uint32_t, 4>, rows> schedules {{
        {capacity - 3U, 1023U, 0U, 1U}, {capacity - 2U, 1022U, 1U, 2U},
        {capacity - 1U, 1021U, 3U, 2U}
    }};
    std::vector<std::uint32_t> schedule_words(8U, std::numeric_limits<std::uint32_t>::max());
    for (const auto& schedule : schedules) schedule_words.insert(schedule_words.end(), schedule.begin(), schedule.end());
    const auto schedule_owner = words(schedule_words);
    const auto schedule_buffer = tensor::backends::metal::shared_metal_buffer_view_for_testing(schedule_owner,
        TensorDescriptor::from(Shape::from({4U + rows * 4U}), DType::f32), 4U * sizeof(std::uint32_t));
    const auto output = buffer({rows, query_heads * dimension});
    const auto partial = buffer({rows, 32U, query_heads, dimension});
    const auto maxima = buffer({rows, 32U, query_heads});
    const auto denominators = buffer({rows, 32U, query_heads});
    MetalCommandContext command("Gemma3 precise private attention", false, true);
    command.encode_verifier_attention(query, key, value, private_key, private_value, schedule_buffer,
        4U * sizeof(std::uint32_t), ancestors, output, partial, maxima, denominators, rows, capacity - 1U,
        query_heads, kv_heads, dimension, dimension, 1.0F / 16.0F, 0.0F);
    static_cast<void>(command.commit_and_wait());
    const auto actual = values(output);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const auto& schedule = schedules[row];
        const auto count = schedule[1] + schedule[3];
        for (std::uint32_t head = 0; head < query_heads; ++head) {
            std::vector<double> scores(count);
            for (std::uint32_t token = 0; token < count; ++token) {
                const bool is_private = token >= schedule[1];
                const auto source = is_private ? ancestor_values[schedule[2] + token - schedule[1]]
                    : (schedule[0] + token) & (capacity - 1U);
                const auto& keys = is_private ? pk : k;
                double dot = 0.0;
                for (std::uint32_t column = 0; column < dimension; ++column) {
                    dot += static_cast<double>(q[(row * query_heads + head) * dimension + column])
                        * keys[(source * kv_heads + head / 2U) * dimension + column];
                }
                scores[token] = dot / 16.0;
            }
            const auto maximum = *std::max_element(scores.begin(), scores.end());
            double denominator = 0.0;
            for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (std::uint32_t column = 0; column < dimension; ++column) {
                double reference = 0.0;
                for (std::uint32_t token = 0; token < count; ++token) {
                    const bool is_private = token >= schedule[1];
                    const auto source = is_private ? ancestor_values[schedule[2] + token - schedule[1]]
                        : (schedule[0] + token) & (capacity - 1U);
                    const auto& source_values = is_private ? pv : v;
                    reference += scores[token] / denominator
                        * source_values[(source * kv_heads + head / 2U) * dimension + column];
                }
                expect_close(actual[(row * query_heads + head) * dimension + column],
                    static_cast<float>(reference), 2.0e-7F, "independent precise tree and wrapped ring attention");
            }
        }
    }
    // Commit an accepted branch across the local ring edge, then copy it to a
    // larger cache. These values deliberately distinguish F32 from F16.
    const auto resident_key = buffer({capacity, kv_heads, dimension}, dtype);
    const auto resident_value = buffer({capacity, kv_heads, dimension}, dtype);
    const auto copied_key = buffer({2U * capacity, kv_heads, dimension}, dtype);
    const auto copied_value = buffer({2U * capacity, kv_heads, dimension}, dtype);
    const std::array<std::uint32_t, 3> selected_rows {2U, 0U, 1U};
    const auto selected = words(selected_rows);
    command.restart();
    command.encode_verifier_kv_scatter(private_key, private_value, selected, resident_key, resident_value,
        capacity - 1U, rows, kv_heads * dimension, capacity - 1U);
    command.encode_memory_barrier();
    command.encode_gemma2_kv_cache_copy(resident_key, resident_value, copied_key, copied_value,
        capacity - 1U, rows, capacity - 1U, 2U * capacity - 1U);
    static_cast<void>(command.commit_and_wait());
    const auto published_key = cache_values(copied_key), published_value = cache_values(copied_value);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < kv_heads * dimension; ++column) {
            const auto destination = (capacity - 1U + row) * kv_heads * dimension + column;
            const auto source = selected_rows[row] * kv_heads * dimension + column;
            expect(published_key[destination] == pk[source] && published_value[destination] == pv[source],
                "accepted-path publication and wrapped cache copy are lossless");
        }
    }
    if (dtype == DType::f16) {
        const auto source_k = half_bits(private_key), source_v = half_bits(private_value);
        const auto destination_k = half_bits(copied_key), destination_v = half_bits(copied_value);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const auto destination = (capacity - 1U + row) * kv_heads * dimension;
            const auto source = selected_rows[row] * kv_heads * dimension;
            expect(std::equal(source_k.begin() + source, source_k.begin() + source + kv_heads * dimension,
                       destination_k.begin() + destination)
                && std::equal(source_v.begin() + source, source_v.begin() + source + kv_heads * dimension,
                       destination_v.begin() + destination),
                "half publication and differing cache masks copy already rounded bits");
        }
        for (const auto& owner : cache_owners) {
            const auto guarded = cache_values(owner);
            expect(std::all_of(guarded.begin(), guarded.begin() + 6U, [](float v) { return v == guard; })
                && std::all_of(guarded.end() - 6U, guarded.end(), [](float v) { return v == guard; }),
                "half attention/publication preserve cache-view guard regions");
        }
    }
}

void f32_private_attention_ring_and_publication() { private_attention_ring_and_publication(DType::f32); }
void f16_private_attention_ring_and_publication() { private_attention_ring_and_publication(DType::f16); }

void f16_attention_extreme_finite_scores() {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 2U, heads = 16U, kv_heads = 8U, width = 256U, capacity = 512U;
    std::vector<float> q(rows * heads * width), k(capacity * kv_heads * width), v(k.size());
    std::vector<float> pk(rows * kv_heads * width), pv(pk.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t head = 0; head < heads; ++head)
            q[(row * heads + head) * width] = head % 2U == 0U ? 1024.0F : -1024.0F;
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            pk[(row * kv_heads + head) * width] = row == 0U ? 32.0F : -32.0F;
            for (std::uint32_t column = 0; column < width; ++column)
                pv[(row * kv_heads + head) * width + column] = row == 0U ? 65504.0F : -65504.0F;
        }
    }
    for (std::uint32_t token = 0; token < capacity; ++token)
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            k[(token * kv_heads + head) * width] = token % 2U == 0U ? 16.0F : -16.0F;
            for (std::uint32_t column = 0; column < width; ++column)
                v[(token * kv_heads + head) * width + column] = static_cast<float>(head)
                    + static_cast<float>(column) * 0.25F;
        }
    const auto query = upload(rows, heads * width, q);
    const auto key = cache_upload(capacity, kv_heads * width, k, DType::f16);
    const auto value = cache_upload(capacity, kv_heads * width, v, DType::f16);
    const auto private_key = cache_upload(rows, kv_heads * width, pk, DType::f16);
    const auto private_value = cache_upload(rows, kv_heads * width, pv, DType::f16);
    const std::array<std::uint32_t, 3> ancestor_ids {0U, 0U, 1U};
    const std::array<std::uint32_t, 8> packed {0U, capacity, 0U, 1U, 0U, capacity, 1U, 2U};
    const auto ancestors = words(ancestor_ids), schedules = words(packed);
    const auto output = buffer({rows, heads, width}), sums = buffer({rows, heads * width / 8U});
    const auto partial = buffer({rows, 32U, heads, width});
    const auto maxima = buffer({rows, 32U, heads}), denominators = buffer({rows, 32U, heads});
    for (const bool fused : {false, true}) {
        MetalCommandContext command("precise half attention extreme finite scores", false, true);
        command.encode_verifier_attention(query, key, value, private_key, private_value,
            schedules, 0U, ancestors, output, partial, maxima, denominators, rows,
            capacity - 1U, heads, kv_heads, width, width, 1.0F / 16.0F, 0.0F, fused ? &sums : nullptr);
        static_cast<void>(command.commit_and_wait());
        const auto actual = values(output);
        // Scores are exactly +/-1024 or +/-2048. Stable F32 softmax discards
        // the losing exponentials; maxima from different partitions remain
        // finite even with the largest finite half values on the winning row.
        for (std::uint32_t row = 0; row < rows; ++row)
            for (std::uint32_t head = 0; head < heads; ++head)
                for (std::uint32_t column = 0; column < width; ++column) {
                    const auto expected = head % 2U == 0U ? 65504.0F : row == 1U ? -65504.0F
                        : static_cast<float>(head / 2U) + static_cast<float>(column) * 0.25F;
                    expect(actual[(row * heads + head) * width + column] == expected,
                        "half storage retains stable uncapped F32 softmax and accumulation at extreme scores");
                }
        if (fused) check_q4_sum_receipt(actual, sums, rows, heads * width);
    }
}

void oversized_chunk_publication_keeps_newest_rows() {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 12U, capacity = 8U, width = 256U, position = 7U;
    auto input = pattern(rows * width, 0.3F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) input[row * width + column] += static_cast<float>(row);
    }
    std::array<std::uint32_t, rows> selected {};
    for (std::uint32_t row = 0; row < rows; ++row) selected[row] = row * 5U % rows;
    const auto indices = words(selected);
    for (const auto dtype : {DType::f16, DType::f32}) {
        const auto host = dtype == DType::f32 ? tensor_f32_matrix(rows, width, input)
            : tensor_f16_matrix(rows, width, input);
        const auto source = copy_tensor_to_backend(host, BackendKind::metal);
        const auto key = buffer({capacity, 1U, width}, dtype), value = buffer({capacity, 1U, width}, dtype);
        MetalCommandContext command("Gemma3 oversized private chunk publication", false, true);
        command.encode_verifier_kv_scatter(source, source, indices, key, value,
            position, rows, width, capacity - 1U);
        static_cast<void>(command.commit_and_wait());
        const auto actual_key = copy_tensor_from_backend(key), actual_value = copy_tensor_from_backend(value);
        const auto bytes_per_row = width * (dtype == DType::f32 ? sizeof(float) : sizeof(std::uint16_t));
        for (std::uint32_t row = rows - capacity; row < rows; ++row) {
            const auto destination = (position + row) & (capacity - 1U);
            const auto expected = host.bytes().subspan(selected[row] * bytes_per_row, bytes_per_row);
            const auto actual = actual_key.bytes().subspan(destination * bytes_per_row, bytes_per_row);
            const auto value_bytes = actual_value.bytes().subspan(destination * bytes_per_row, bytes_per_row);
            expect(std::equal(expected.begin(), expected.end(), actual.begin())
                && std::equal(expected.begin(), expected.end(), value_bytes.begin()),
                "only the newest logical row owns each wrapped publication slot");
        }
    }
}

void uncapped_candidate_selection() {
    if (!metal_available()) return;
    constexpr std::uint32_t vocabulary = 1025U;
    std::vector<float> logits(vocabulary, -2.0F);
    logits[1000] = 101.0F; logits[1001] = 101.0F; logits[1010] = 100.0F; logits[7] = -99.75F;
    const auto input = upload(1U, vocabulary, logits);
    const engine::Gemma2MetalSelectionWorkspace workspace {
        buffer({2U * 3U * engine::token_sampler_max_candidates}),
        buffer({2U * 3U * engine::token_sampler_max_candidates}),
        buffer({2U * (2U + 2U * engine::token_sampler_max_candidates)}),
        buffer({2U * engine::token_sampler_max_repetition_window}),
    };
    MetalCommandContext command("Gemma3 uncapped selection", false, true);
    engine::TokenSelectionRequest request;
    request.top_k = 2U; request.base_candidate_count = 3U;
    request.history_token_count = 1U; request.history_token_ids[0] = 7U;
    static_cast<void>(command.encode_token_candidates(input, workspace, request, 0.0F));
    static_cast<void>(command.commit_and_wait());
    auto result = command.read_token_candidates(workspace);
    expect(result.count == 4U && result.values[0].token == 1000U && result.values[1].token == 1001U,
        "uncapped compact selector preserves ordering and deterministic ties");
    for (std::uint32_t i = 0; i < result.count; ++i) {
        expect(result.values[i].logit == logits[result.values[i].token], "uncapped logits retain native values exactly");
    }
    request = {}; command.restart();
    static_cast<void>(command.encode_token_candidates(input, workspace, request, 0.0F));
    static_cast<void>(command.commit_and_wait());
    result = command.read_token_candidates(workspace);
    expect(result.count == 1U && result.values[0].token == 1000U && result.values[0].logit == 101.0F,
        "uncapped greedy selector uses the same tie and logit policy");
}

void hybrid_prefill_preserves_native_precision() {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 128U, columns = 37U;
    for (const auto width : {3840U, 4096U, 15360U}) {
        const auto input_values = pattern(rows * width, 0.027F);
        const auto input = upload(rows, width, input_values);
        for (const auto dtype : {DType::q4_k, DType::q6_k}) {
            const auto host_weight = dtype == DType::q4_k
                ? tensor_q4_k_stress_gguf_weight_matrix(width, columns, 51U)
                : tensor_q6_k_stress_gguf_weight_matrix(width, columns, 71U);
            const auto weight = copy_tensor_to_backend(host_weight, BackendKind::metal);
            const auto repacked_host = repack_kquant_weight_columns8(host_weight.view());
            const auto repacked_weight = copy_tensor_to_backend(repacked_host, BackendKind::metal);
            const auto output = buffer({rows, columns});
            const auto packed_output = buffer({rows, columns});
            MetalCommandContext command("Gemma3 native F32 hybrid projection parity", false, true);
            command.encode_verifier_matmul(input, weight, output, rows, true);
            command.encode_verifier_matmul(input, weight, packed_output, rows);
            static_cast<void>(command.commit_and_wait());
            const auto batched = values(output);
            const auto packed = values(packed_output);
#if defined(LOCAL_GGUF_ENGINE_TESTING)
            // Eight useful rows force the retained M8 specialization. Check
            // every row of a larger tile, including nonzero input/output views.
            const auto m8_output = buffer({rows, columns});
            command.restart();
            for (std::uint32_t first = 0U; first < rows; first += 8U) {
                const auto input_view = tensor::backends::metal::shared_metal_buffer_view_for_testing(
                    input, TensorDescriptor::from(Shape::from({8U, width}), DType::f32),
                    first * width * sizeof(float));
                const auto output_view = tensor::backends::metal::shared_metal_buffer_view_for_testing(
                    m8_output, TensorDescriptor::from(Shape::from({8U, columns}), DType::f32),
                    first * columns * sizeof(float));
                command.encode_verifier_matmul(input_view, weight, output_view, 8U, true);
            }
            static_cast<void>(command.commit_and_wait());
            expect(values(m8_output) == batched,
                "larger prompt tiles preserve every M8 row's exact F32 arithmetic");
#endif
            command.restart();
            command.encode_verifier_matmul(input, repacked_weight, output, rows, true);
            command.encode_verifier_matmul(input, repacked_weight, packed_output, rows);
            static_cast<void>(command.commit_and_wait());
            expect(values(output) == batched,
                "columns8 prefill preserves native-layout F32 matrix arithmetic");
            expect(values(packed_output) == packed,
                "columns8 verifier preserves native-layout arithmetic for all rows");
            for (const auto tail_rows : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U, 33U, 63U, 64U, 65U, 127U}) {
                const auto tail_output = buffer({tail_rows, columns});
                command.restart();
                command.encode_verifier_matmul(input, weight, tail_output, tail_rows, true);
                static_cast<void>(command.commit_and_wait());
                const auto tail = values(tail_output);
                if (tail_rows >= 4U) {
                    expect(std::equal(tail.begin(), tail.end(), batched.begin()),
                        "MMA row tiles preserve arithmetic across even and odd batch tails");
                } else {
                    // Packed rows deliberately have a different reduction and
                    // scale association. Compare to packed scalar here; both
                    // regimes are checked against decoded dots below.
                    for (std::uint32_t row = 0U; row < tail_rows; ++row) {
                        const auto single = buffer({1U, columns});
                        command.restart();
                        command.encode_target_head_rows(input, row, weight, single, 1U);
                        static_cast<void>(command.commit_and_wait());
                        const auto scalar = values(single);
                        expect(std::equal(scalar.begin(), scalar.end(), tail.begin() + row * columns),
                            "packed one/two/three-row arithmetic remains identical");
                    }
                }
                command.restart();
                command.encode_verifier_matmul(input, weight, tail_output, tail_rows);
                static_cast<void>(command.commit_and_wait());
                const auto packed_tail = values(tail_output);
                expect(std::equal(packed_tail.begin(), packed_tail.end(), packed.begin()),
                    "verifier batches retain exact packed arithmetic for every row count");
                command.restart();
                command.encode_verifier_matmul(input, repacked_weight, tail_output, tail_rows);
                static_cast<void>(command.commit_and_wait());
                expect(values(tail_output) == packed_tail,
                    "columns8 scalar and short verifier tails match original native weights");
                command.restart();
                command.encode_verifier_matmul(input, repacked_weight, tail_output, tail_rows, true);
                static_cast<void>(command.commit_and_wait());
                expect(values(tail_output) == tail,
                    "columns8 prefill tails retain the native matrix/packed dispatch contract");
            }
            for (const auto row : {0U, 1U, 2U, 63U, 127U}) {
                const auto single = buffer({1U, columns});
                command.restart();
                command.encode_target_head_rows(input, row, weight, single, 1U);
                static_cast<void>(command.commit_and_wait());
                const auto scalar = values(single);
                command.restart();
                command.encode_target_head_rows(input, row, repacked_weight, single, 1U);
                static_cast<void>(command.commit_and_wait());
                expect(values(single) == scalar, "columns8 row offsets preserve native scalar results");
                for (std::uint32_t column = 0; column < columns; ++column) {
                    expect(scalar[column] == packed[row * columns + column],
                        "parallel verifier rows retain canonical scalar arithmetic");
                    double reference = 0.0;
                    double absolute_sum = 0.0;
                    for (std::uint32_t inner = 0; inner < width; ++inner) {
                        const auto block = column * (width / 256U) + inner / 256U;
                        const float decoded = dtype == DType::q4_k
                            ? q4_k_block_view(host_weight, block).dequantized_value(inner % 256U)
                            : q6_k_block_view(host_weight, block).dequantized_value(inner % 256U);
                        const double product = static_cast<double>(input_values[row * width + inner]) * decoded;
                        reference += product;
                        absolute_sum += std::abs(product);
                    }
                    // Preserve the strict regression envelope inherited from
                    // the original 32-worker packed reduction. This is not a
                    // general error bound for factored scale/min cancellation.
                    const double operations = static_cast<double>(width / 32U + 32U);
                    const double rounding = operations * std::numeric_limits<float>::epsilon();
                    const float bound = static_cast<float>(rounding / (1.0 - rounding) * absolute_sum + 1.0e-6);
                    expect_close(scalar[column], static_cast<float>(reference), bound,
                        "native packed F32 projection agrees with independent decoded dot product");
                    // MMA accumulates the dot in a different order. Use the
                    // standard gamma_n absolute dot-product bound, with n
                    // covering all F32 products/accumulations plus decoding;
                    // this is fixed from the operator, before candidate runs.
                    const double matrix_rounding = static_cast<double>(width + 8U)
                        * std::numeric_limits<float>::epsilon();
                    const float matrix_bound = static_cast<float>(matrix_rounding
                        / (1.0 - matrix_rounding) * absolute_sum + 1.0e-6);
                    expect_close(batched[row * columns + column], static_cast<float>(reference), matrix_bound,
                        "native MMA F32 projection agrees with independent decoded dot product");
                }
            }
        }
    }

    // Each native Q4 weight is the small difference of two terms roughly
    // 1365 times larger. Pack and retain its exact dyadic value independently
    // of production dequantization; signed F32 activations exercise both
    // within-fragment and across-block cancellation.
    constexpr std::uint32_t cancellation_rows = 5U, cancellation_columns = 3U;
    for (const auto width : {3840U, 15360U}) {
        auto input_values = pattern(cancellation_rows * width, 0.027F);
        for (std::size_t i = 0U; i < input_values.size(); ++i) {
            input_values[i] += float(i % 11U + 1U) * 0x1p-20F;
            if ((i / 16U) % 2U != 0U) input_values[i] = -input_values[i];
        }
        const auto input = upload(cancellation_rows, width, input_values);
        auto host_weight = tensor_q4_k_stress_gguf_weight_matrix(width, cancellation_columns, 0U);
        auto bytes = host_weight.mutable_bytes();
        const auto blocks = width / 256U;
        for (std::uint32_t column = 0U; column < cancellation_columns; ++column) {
            for (std::uint32_t block = 0U; block < blocks; ++block) {
                auto* packed = bytes.data() + (column * blocks + block) * 144U;
                // d=1365/4096, dmin=(1364 or 1366)/4096,
                // group scale=1, group minimum=13, every nibble=13.
                packed[0] = 0x55U; packed[1] = 0x35U;
                packed[2] = (column + block) % 2U == 0U ? 0x54U : 0x56U;
                packed[3] = 0x35U;
                for (std::uint32_t group = 0U; group < 4U; ++group) {
                    packed[4U + group] = 1U;
                    packed[8U + group] = 13U;
                    packed[12U + group] = 0xd1U;
                }
                std::fill(packed + 16U, packed + 144U, std::uint8_t {0xddU});
            }
        }
        const auto weight = copy_tensor_to_backend(host_weight, BackendKind::metal);
        std::vector<BackendBuffer> outputs, scalar_outputs;
        outputs.reserve(cancellation_rows);
        scalar_outputs.reserve(cancellation_rows);
        MetalCommandContext command("Gemma3 Q4 scale/min cancellation", false, true);
        const auto repacked_host = repack_kquant_weight_columns8(host_weight.view());
        const auto repacked_weight = copy_tensor_to_backend(repacked_host, BackendKind::metal);
        for (std::uint32_t count = 1U; count <= cancellation_rows; ++count) {
            outputs.push_back(buffer({count, cancellation_columns}));
            scalar_outputs.push_back(buffer({1U, cancellation_columns}));
            command.encode_verifier_matmul(input, weight, outputs.back(), count);
            command.encode_target_head_rows(input, count - 1U, weight, scalar_outputs.back(), 1U);
        }
        static_cast<void>(command.commit_and_wait());
        for (std::uint32_t count = 1U; count <= cancellation_rows; ++count) {
            const auto actual = values(outputs[count - 1U]);
            command.restart();
            command.encode_verifier_matmul(input, repacked_weight, outputs[count - 1U], count);
            static_cast<void>(command.commit_and_wait());
            expect(values(outputs[count - 1U]) == actual,
                "columns8 preserves the adversarial native Q4 cancellation result exactly");
            for (std::uint32_t row = 0U; row < count; ++row) {
                const auto scalar = values(scalar_outputs[row]);
                for (std::uint32_t column = 0U; column < cancellation_columns; ++column) {
                    double reference = 0.0, absolute_sum = 0.0;
                    for (std::uint32_t inner = 0U; inner < width; ++inner) {
                        const double decoded = (column + inner / 256U) % 2U == 0U
                            ? 13.0 / 4096.0 : -13.0 / 4096.0;
                        const double product = double(input_values[row * width + inner]) * decoded;
                        reference += product;
                        absolute_sum += std::abs(product);
                    }
                    const double operations = static_cast<double>(width / 32U + 32U);
                    const double rounding = operations * std::numeric_limits<float>::epsilon();
                    const float bound = static_cast<float>(rounding / (1.0 - rounding) * absolute_sum + 1.0e-6);
                    const auto value = actual[row * cancellation_columns + column];
                    expect(std::isfinite(value), "Q4 cancellation must produce finite results");
                    expect(value == scalar[column], "Q4 cancellation changed ordinary/verifier arithmetic");
                    expect_close(value, static_cast<float>(reference), bound,
                        "Q4 scale/min cancellation agrees with independently retained dense weights");
                }
            }
        }
    }
}

void matrix_operands_retain_f32_information() {
    if (!metal_available()) return;
    constexpr std::uint32_t rows = 33U, width = 3840U, columns = 37U;
    std::vector<float> input_values(rows * width, 0.0F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        input_values[row * width + (row % 8U) * 33U] = 1.0F + float(row + 1U) * 0x1p-20F;
    }
    const auto input = upload(rows, width, input_values);
    for (const auto dtype : {DType::q4_k, DType::q6_k}) {
        auto host_weight = dtype == DType::q4_k
            ? tensor_q4_k_stress_gguf_weight_matrix(width, columns, 51U)
            : tensor_q6_k_stress_gguf_weight_matrix(width, columns, 71U);
        const auto block_bytes = dtype == DType::q4_k ? 144U : 210U;
        const auto scale_offset = dtype == DType::q4_k ? 0U : 208U;
        auto bytes = host_weight.mutable_bytes();
        for (std::uint32_t column = 0; column < columns; ++column) {
            // A non-power-of-two half scale produces decoded F32 weights
            // that cannot be represented exactly by a staged half operand.
            bytes[column * (width / 256U) * block_bytes + scale_offset] = 0x55U;
            bytes[column * (width / 256U) * block_bytes + scale_offset + 1U] = 0x35U;
        }
        const auto weight = copy_tensor_to_backend(host_weight, BackendKind::metal);
        const auto output = buffer({rows, columns});
        std::vector<BackendBuffer> precise_outputs;
        precise_outputs.reserve(5U);
        MetalCommandContext command("Gemma3 FP32 operand precision", false, true);
        command.encode_verifier_matmul(input, weight, output, rows, true);
        for (std::uint32_t count = 1U; count <= 5U; ++count) {
            precise_outputs.push_back(buffer({count, columns}));
            command.encode_verifier_matmul(input, weight, precise_outputs.back(), count);
        }
        static_cast<void>(command.commit_and_wait());
        const auto actual = values(output);
        std::vector<std::vector<float>> precise_values;
        for (const auto& precise : precise_outputs) precise_values.push_back(values(precise));
        const auto repacked_host = repack_kquant_weight_columns8(host_weight.view());
        const auto repacked_weight = copy_tensor_to_backend(repacked_host, BackendKind::metal);
        command.restart();
        command.encode_verifier_matmul(input, repacked_weight, output, rows, true);
        for (std::uint32_t count = 1U; count <= 5U; ++count) {
            command.encode_verifier_matmul(input, repacked_weight, precise_outputs[count - 1U], count);
        }
        static_cast<void>(command.commit_and_wait());
        expect(values(output) == actual, "columns8 prefill retains all native F32 operand information");
        for (std::size_t index = 0; index < precise_outputs.size(); ++index) {
            expect(values(precise_outputs[index]) == precise_values[index],
                "columns8 ordinary and verifier retain all native F32 operand information");
        }
        bool discriminates_half = false;
        bool precise_discriminates_half = false;
        for (std::uint32_t row = 0; row < rows; ++row) {
            const auto inner = (row % 8U) * 33U;
            const auto activation = input_values[row * width + inner];
            for (std::uint32_t column = 0; column < columns; ++column) {
                const auto decoded = dtype == DType::q4_k
                    ? q4_k_block_view(host_weight, column * (width / 256U)).dequantized_value(inner)
                    : q6_k_block_view(host_weight, column * (width / 256U)).dequantized_value(inner);
                const auto expected = activation * decoded;
                const auto narrowed = f16_bits_to_f32(f32_to_f16_bits(activation))
                    * f16_bits_to_f32(f32_to_f16_bits(decoded));
                const auto bound = 4.0F * std::numeric_limits<float>::epsilon()
                    * std::abs(expected) + 1.0e-6F;
                const bool detects_half = std::abs(expected - narrowed) > 16.0F * bound;
                discriminates_half = discriminates_half || detects_half;
                precise_discriminates_half = precise_discriminates_half
                    || (row < precise_values.size() && detects_half);
                expect_close(actual[row * columns + column], expected, bound,
                    "matrix operands preserve F32 activation and native decoded weight information");
                for (std::uint32_t count = row + 1U; count <= precise_values.size(); ++count) {
                    const auto value = precise_values[count - 1U][row * columns + column];
                    expect(std::isfinite(value), "precise packed projection must remain finite");
                    expect_close(value, expected, bound,
                        "precise packed rows preserve F32 activation and native decoded weight information");
                }
            }
        }
        expect(discriminates_half, "precision fixture must reject half operand staging");
        expect(precise_discriminates_half,
            "precise packed-row fixture must reject half operand staging");
    }
}

void metal_resource_and_input_validation() {
    if (!metal_available()) return;
    const auto limits = tensor::backends::metal::memory_info();
    expect(limits && limits->physical_memory_bytes > 0U && limits->max_buffer_bytes > 0U
        && limits->recommended_working_set_bytes > 0U, "Metal reports physical and device memory limits");
    bool rejected = false;
    try { static_cast<void>(buffer({limits->max_buffer_bytes / sizeof(float) + 1U})); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "oversized Metal allocation is rejected before creating a buffer");
    const auto input = buffer({1U, 256U}), weight = buffer({256U}), output = buffer({1U, 256U});
    const std::array<std::uint32_t, 1> zero {0U};
    const auto positions = words(zero);
    MetalCommandContext command("Gemma3 primitive validation", false, true);
    rejected = false;
    try { command.encode_head_rms_norm(input, weight, output, 1.0e-6F, 1U, 1U, 128U); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "per-head normalization rejects unsupported width");
    rejected = false;
    try { command.encode_verifier_rope(input, positions, output, 1U, 1U, 256U, 256U, 1.0e6F, 0.0F); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "RoPE rejects invalid scaling");
    const auto half_output = buffer({1U, 256U}, DType::f16);
    rejected = false;
    try { command.encode_verifier_kv_store(input, input, positions, output, half_output,
        1U, 1U, 256U, 256U, 1.0e6F, 0.125F); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "private cache rejects mismatched K/V precision");
}

void columns8_storage_boundaries() {
    if (!metal_available()) return;
    constexpr std::uint32_t width = 256U, columns = 9U, rows = 65U;
    const auto input = upload(rows, width, pattern(rows * width));
    const auto output = buffer({rows, columns});
    const std::array<std::uint32_t, 1> token {0U};
    const auto tokens = words(token);
    const auto embedding = buffer({1U, width});
    for (const auto dtype : {DType::q4_k, DType::q6_k}) {
        const auto native = dtype == DType::q4_k
            ? tensor_q4_k_stress_gguf_weight_matrix(width, columns, 67U)
            : tensor_q6_k_stress_gguf_weight_matrix(width, columns, 83U);
        const auto packed = repack_kquant_weight_columns8(native.view());
        const auto resident = copy_tensor_to_backend(packed, BackendKind::metal);
        const auto copied = copy_tensor_from_backend(resident);
        expect(copied.descriptor() == packed.descriptor()
            && std::equal(copied.bytes().begin(), copied.bytes().end(), packed.bytes().begin()),
            "packed readback preserves the physical descriptor and bytes");
        MetalCommandContext precise("columns8 boundaries", false, true);
        bool rejected = false;
        try { precise.encode_matmul(input, resident, output, rows); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "generic native matmul rejects packed weights");
        rejected = false;
        try { precise.encode_embedding(tokens, resident, embedding, 1.0F); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "native embedding rejects packed weights");
        MetalCommandContext ordinary("columns8 established Gemma2 compiler policy");
        if (dtype == DType::q4_k) {
            ordinary.encode_verifier_matmul(input, resident, output, rows);
            expect(ordinary.commit_and_wait().compute_dispatch_count == 1U,
                "packed Q4 supports the retained Gemma2 compiler policy");
        } else {
            rejected = false;
            try { ordinary.encode_verifier_matmul(input, resident, output, rows); }
            catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected, "Gemma2 keeps the established canonical Q6 arithmetic");
        }
        const auto native_resident = copy_tensor_to_backend(native, BackendKind::metal);
        const auto native_output = buffer({rows, columns});
        for (const auto count : {4U, 5U}) {
            precise.restart();
            precise.encode_target_head_rows(input, 1U, native_resident, native_output, count);
            precise.encode_target_head_rows(input, 1U, resident, output, count);
            static_cast<void>(precise.commit_and_wait());
            const auto expected = values(native_output);
            const auto actual = values(output);
            expect(std::equal(actual.begin(), actual.begin() + count * columns, expected.begin()),
                "multirow packed projections preserve native arithmetic at a nonzero input row offset");
        }
#if defined(LOCAL_GGUF_ENGINE_TESTING)
        auto padded = TensorStorage::allocate(TensorDescriptor::from(
            Shape::from({(packed.bytes().size() + 8U) / sizeof(float)}), DType::f32));
        std::fill(padded.mutable_bytes().begin(), padded.mutable_bytes().end(), 0xA5U);
        std::copy(packed.bytes().begin(), packed.bytes().end(), padded.mutable_bytes().begin() + 4U);
        const auto owner = copy_tensor_to_backend(padded, BackendKind::metal);
        const auto view = tensor::backends::metal::shared_metal_buffer_view_for_testing(
            owner, packed.descriptor(), 4U);
        const auto guarded_output = upload(1U, rows * columns + 2U,
            std::vector<float>(rows * columns + 2U, -123.25F));
        const auto output_view = tensor::backends::metal::shared_metal_buffer_view_for_testing(
            guarded_output, output.descriptor(), sizeof(float));
        for (const auto count : {1U, 2U, 3U, 4U, 5U, 7U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, rows}) {
            precise.restart();
            precise.encode_verifier_matmul(input, resident, output, count, true);
            static_cast<void>(precise.commit_and_wait());
            const auto expected = values(output);
            precise.restart();
            precise.encode_verifier_matmul(input, view, output, count, true);
            static_cast<void>(precise.commit_and_wait());
            expect(values(output) == expected, "packed kernels honor nonzero allocation offsets");
            precise.restart();
            precise.encode_verifier_matmul(input, view, output_view, count, true);
            static_cast<void>(precise.commit_and_wait());
            const auto guarded = values(guarded_output);
            expect(guarded.front() == -123.25F && guarded.back() == -123.25F
                && std::equal(expected.begin(), expected.begin() + count * columns, guarded.begin() + 1U),
                "packed projection preserves output allocation guards and nonzero offsets");
            expect(std::all_of(guarded.begin() + 1U + count * columns, guarded.end(),
                [](float value) { return value == -123.25F; }),
                "packed partial row tiles do not write inactive output rows");
        }
        const auto owner_after = copy_tensor_from_backend(owner);
        expect(std::equal(owner_after.bytes().begin(), owner_after.bytes().end(), padded.bytes().begin()),
            "packed kernels preserve source bytes and surrounding canaries");
#endif
    }
}

void discarded_commands_release_queue_slots() {
    if (!metal_available()) return;
    const std::vector<float> input_values {0.25F, -0.75F, 1.0F, 4.5F};
    const std::vector<float> initial_output(4U, -13.0F);
    const auto input = upload(1U, 4U, input_values);
    for (const bool precise : {false, true}) {
        const auto output = upload(1U, 4U, initial_output);
        // Exceed the command queue's ordinary capacity without any external
        // autorelease pool. Both empty and encoded commands must relinquish
        // their leases when discarded or replaced before submission.
        for (std::uint32_t iteration = 0; iteration < 160U; ++iteration) {
            MetalCommandContext discarded("discarded Metal command", false, precise);
            if ((iteration & 1U) != 0U) discarded.encode_add(input, input, output);
            discarded.restart();
            discarded.encode_add(input, input, output);
        }
        expect(values(output) == initial_output, "discarded commands must not execute work");
        MetalCommandContext submitted("command after discarded contexts", false, precise);
        submitted.encode_add(input, input, output);
        static_cast<void>(submitted.commit_and_wait());
        const auto actual = values(output);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            expect(actual[i] == input_values[i] * 2.0F,
                "queue remains usable after hundreds of discarded and restarted contexts");
        }
    }
}

void shared_buffer_access_preserves_owner_lifetime() {
    if (!metal_available()) return;
    // Establish both queues and complete one dispatch before taking the
    // allocation baseline, excluding persistent pipeline/driver initialization.
    for (const bool precise : {false, true}) {
        const auto warm_buffer = buffer({4U});
        MetalCommandContext warm("shared buffer lifetime warm-up", false, precise);
        warm.encode_add(warm_buffer, warm_buffer, warm_buffer);
        static_cast<void>(warm.commit_and_wait());
    }
    const auto before = tensor::backends::metal::memory_info()->current_allocated_bytes;
    for (std::uint32_t iteration = 0; iteration < 4U; ++iteration) {
        const auto staging = buffer({4U * 1024U * 1024U});
        tensor::backends::metal::fill_shared_metal_buffer_u32(staging, 0x12345678U);
        std::array<std::uint8_t, sizeof(std::uint32_t)> bytes {};
        tensor::backends::metal::read_shared_metal_buffer_bytes(staging, 0U, bytes);
        expect(bytes[0] == 0x78U && bytes[3] == 0x12U, "shared staging access preserves contents");
        MetalCommandContext command("completed shared buffer lifetime", false, (iteration & 1U) != 0U);
        command.encode_add(staging, staging, staging, 4U);
        static_cast<void>(command.commit_and_wait());
    }
    const auto after = tensor::backends::metal::memory_info()->current_allocated_bytes;
    expect(after == before,
        "CPU contents access must not retain destroyed Metal buffers in a caller autorelease pool");
}

void residency_skips_unsupported_devices() {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    // This exercises construction, buffer registration and destruction even
    // when no system Metal device exists. The fake throws if creation is tried.
    static_cast<void>(tensor::backends::metal::exercise_metal_buffer_residency_for_testing(false));
#endif
}

void residency_preserves_supported_device_errors() {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    bool failed = false;
    try {
        if (!tensor::backends::metal::exercise_metal_buffer_residency_for_testing(true)) return;
    } catch (const runtime::BackendError& error) {
        failed = error.code() == runtime::BackendErrorCode::metal_execution_failed
            && error.backend() == BackendKind::metal;
    }
    expect(failed, "supported devices must report residency allocation failures");
#endif
}

void precise_context_unavailable_device() {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    using tensor::backends::metal::set_metal_device_unavailable_for_testing;
    set_metal_device_unavailable_for_testing(true);
    bool unavailable = false;
    try { MetalCommandContext command("Gemma3 unavailable device", false, true); }
    catch (const runtime::BackendError& error) {
        unavailable = error.code() == runtime::BackendErrorCode::unavailable_metal_device;
    } catch (...) { set_metal_device_unavailable_for_testing(false); throw; }
    set_metal_device_unavailable_for_testing(false);
    expect(unavailable, "precise context must not bypass the unavailable-device boundary");
#endif
}
}  // namespace
std::span<const TestCase> gemma3_metal_test_cases() {
    static const std::array cases {
        TestCase{"Gemma3 native half-cache rounding boundaries", native_half_cache_cast_rounding},
        TestCase{"Gemma3 checked half-cache numerical status", half_cache_numerical_status},
        TestCase{"Gemma3 fused Q4 RMS/GELU sums preserve precise projections", [] { fused_q4_rms_and_gelu_receipts(); }},
        TestCase{"Gemma2 fused Q4 RMS/GELU sums preserve default compiler projections", [] { fused_q4_rms_and_gelu_receipts(false); }},
        TestCase{"Gemma3 compact Q4 worker partitions and column tails", compact_q4_worker_partitions_and_column_tails},
        TestCase{"Gemma3 fused Q4 attention sums cover partition boundaries", [] { fused_q4_attention_receipts(); }},
        TestCase{"Gemma2 fused Q4 attention sums cover partition boundaries", [] { fused_q4_attention_receipts(false); }},
        TestCase{"Gemma3 fused Q4 receipt validation", fused_q4_receipt_validation},
        TestCase{"Gemma2 compact projection actual inner widths", gemma2_compact_projection_parity},
        TestCase{"Gemma2 packed prompt retains half operands at actual dimensions", gemma2_packed_prompt_retains_half_operands},
        TestCase{"Gemma3 native Q6 embedding and tied output head", native_q6_embedding_and_head},
        TestCase{"Gemma3 per-head RMSNorm and scaled RoPE positions", head_norm_and_scaled_rope},
        TestCase{"Gemma3 F32 private attention and ring publication", f32_private_attention_ring_and_publication},
        TestCase{"Gemma3 F16 precise private attention and ring publication", f16_private_attention_ring_and_publication},
        TestCase{"Gemma3 F16 attention extreme finite scores", f16_attention_extreme_finite_scores},
        TestCase{"Metal attention dispatch follows host-staged visible rows", attention_dispatch_uses_host_staged_visible_rows},
        TestCase{"Metal attention dispatch rejects invalid host schedules", attention_dispatch_rejects_invalid_host_schedules},
        TestCase{"Gemma3 oversized private chunk publication", oversized_chunk_publication_keeps_newest_rows},
        TestCase{"Gemma3 uncapped compact token selection", uncapped_candidate_selection},
        TestCase{"Gemma3 hybrid F32 prefill and scalar independent projection parity", hybrid_prefill_preserves_native_precision},
        TestCase{"Gemma3 FP32 matrix operands retain native information", matrix_operands_retain_f32_information},
        TestCase{"Gemma3 Metal resource and primitive validation", metal_resource_and_input_validation},
        TestCase{"Gemma3 columns8 storage boundaries", columns8_storage_boundaries},
        TestCase{"Metal discarded commands release queue slots", discarded_commands_release_queue_slots},
        TestCase{"Metal shared buffer access preserves owner lifetime", shared_buffer_access_preserves_owner_lifetime},
        TestCase{"Metal residency skips unsupported devices", residency_skips_unsupported_devices},
        TestCase{"Metal residency preserves supported device errors", residency_preserves_supported_device_errors},
        TestCase{"Gemma3 precise Metal context unavailable device", precise_context_unavailable_device},
    };
    return cases;
}
}  // namespace local_gguf_engine::tests
