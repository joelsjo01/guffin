#include "benchmark_synthetic_core.hpp"

#include "../support/benchmark_support.hpp"
#include "benchmark_projection.hpp"

#include "local_gguf_engine/tensor/ops/attention.hpp"
#include "local_gguf_engine/tensor/ops/elementwise.hpp"
#include "local_gguf_engine/tensor/ops/matmul.hpp"
#include "local_gguf_engine/tensor/ops/normalization.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/ops/rope.hpp"
#include "local_gguf_engine/tensor/ops/softmax.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace local_gguf_engine::benchmarks {

namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::TensorStorage;
using tensor::ops::add_f32;
using tensor::ops::causal_scaled_dot_product_attention_f32;
using tensor::ops::matmul_f32;
using tensor::ops::matmul_f32_f16;
using tensor::ops::multiply_f32;
using tensor::ops::rms_norm_f32;
using tensor::ops::rope_f32;
using tensor::ops::rope_f32_with_position_start;
using tensor::ops::scaled_dot_product_attention_f32;
using tensor::ops::silu_f32;
using tensor::ops::softmax_f32;
using tensor::ops::sum_f32;
using tokenizer::load_tokenizer_from_gguf_metadata;
using tokenizer::TokenizerDecodeOptions;
using tokenizer::TokenizerEncodeOptions;

} // namespace

void add_synthetic_core_benchmarks(BenchmarkReport &report, BenchmarkBackendSelection selection) {
    constexpr std::uint64_t add_element_count = 4096;
    constexpr std::uint64_t copy_element_count = 16 * 1024;
    constexpr std::uint64_t matmul_size = 64;
    constexpr std::uint64_t focused_hidden = 576;
    constexpr std::uint64_t focused_ffn = 1536;

    const auto add_descriptor =
        TensorDescriptor::from(Shape::from({add_element_count}), DType::f32);
    const auto copy_descriptor =
        TensorDescriptor::from(Shape::from({copy_element_count}), DType::f32);
    const auto copy_payload = make_repeated_payload(copy_descriptor.byte_count);
    const auto add_left = make_f32_storage(add_element_count, 1.0F);
    const auto add_right = make_f32_storage(add_element_count, -0.5F);
    const auto matmul_left = make_f32_matrix(matmul_size, matmul_size, 0.01F);
    const auto matmul_right = make_f32_matrix(matmul_size, matmul_size, 0.02F);
    const auto matmul_right_f16 = make_f16_matrix(matmul_size, matmul_size, 0.02F);
    const auto projection_small_left = make_f32_matrix(8, matmul_size, 0.01F);
    const auto projection_small_weight =
        make_f16_gguf_weight_matrix(matmul_size, matmul_size, 0.02F);
    const auto projection_next_token_left = make_f32_matrix(1, focused_hidden, 0.001F);
    const auto projection_hidden_weight =
        make_f16_gguf_weight_matrix(focused_hidden, focused_hidden, 0.0005F);
    const auto projection_prefill_left = make_f32_matrix(8, focused_hidden, 0.001F);
    const auto projection_ffn_weight =
        make_f16_gguf_weight_matrix(focused_hidden, focused_ffn, 0.00025F);
    const auto rms_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto attention_query = make_f32_matrix(matmul_size, matmul_size, 0.01F);
    const auto attention_key = make_f32_matrix(matmul_size, matmul_size, 0.02F);
    const auto attention_value = make_f32_matrix(matmul_size, matmul_size, 0.03F);
    const auto tokenizer_metadata = make_synthetic_tokenizer_metadata();
    const auto synthetic_tokenizer = load_tokenizer_from_gguf_metadata(tokenizer_metadata);
    const std::string tokenizer_text = "Hello world!Hello world!";
    const auto tokenizer_ids = synthetic_tokenizer.encode(
        tokenizer_text, TokenizerEncodeOptions{.add_bos = true, .add_eos = true});

    if (selection.metal_available) {
        warm_up_metal_projection_backend();
    }

    if (selection.include_cpu) {
        report.measurements.push_back(
            measure("tensor.allocate_f32", report.config.iterations, "shape=[4096] dtype=F32",
                    [add_descriptor]() {
                        const auto storage = TensorStorage::allocate(add_descriptor);
                        return IterationMeasurement{
                            .bytes_processed = storage.bytes().size(),
                            .checksum = checksum_byte_span(storage.bytes()),
                        };
                    }));

        report.measurements.push_back(
            measure("tensor.copy_f32", report.config.iterations, "shape=[16384] dtype=F32",
                    [copy_descriptor, copy_payload]() {
                        const auto storage = TensorStorage::from_bytes(
                            copy_descriptor, std::span<const std::uint8_t>(copy_payload));
                        return IterationMeasurement{
                            .bytes_processed = storage.bytes().size(),
                            .checksum = checksum_byte_span(storage.bytes()),
                        };
                    }));

        report.measurements.push_back(measure("tensor.add_f32", report.config.iterations,
                                              "shape=[4096] dtype=F32", [&add_left, &add_right]() {
                                                  const auto output = add_f32(add_left, add_right);
                                                  return IterationMeasurement{
                                                      .bytes_processed = add_left.bytes().size() +
                                                                         add_right.bytes().size() +
                                                                         output.bytes().size(),
                                                      .checksum = checksum_f32(sum_f32(output)),
                                                  };
                                              }));

        report.measurements.push_back(
            measure("tensor.multiply_f32", report.config.iterations, "shape=[4096] dtype=F32",
                    [&add_left, &add_right]() {
                        const auto output = multiply_f32(add_left, add_right);
                        return IterationMeasurement{
                            .bytes_processed = add_left.bytes().size() + add_right.bytes().size() +
                                               output.bytes().size(),
                            .checksum = checksum_f32(sum_f32(output)),
                        };
                    }));

        report.measurements.push_back(measure(
            "tensor.silu_f32", report.config.iterations, "shape=[4096] dtype=F32", [&add_left]() {
                const auto output = silu_f32(add_left);
                return IterationMeasurement{
                    .bytes_processed = add_left.bytes().size() + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }));

        report.measurements.push_back(
            measure("tensor.matmul_f32", report.config.iterations,
                    "left=[64,64] right=[64,64] dtype=F32", [&matmul_left, &matmul_right]() {
                        const auto output = matmul_f32(matmul_left, matmul_right);
                        return IterationMeasurement{
                            .bytes_processed = matmul_left.bytes().size() +
                                               matmul_right.bytes().size() + output.bytes().size(),
                            .checksum = checksum_f32(sum_f32(output)),
                        };
                    }));

        report.measurements.push_back(measure(
            "tensor.matmul_f32_f16", report.config.iterations,
            "left=[64,64] dtype=F32 right=[64,64] dtype=F16", [&matmul_left, &matmul_right_f16]() {
                const auto output = matmul_f32_f16(matmul_left, matmul_right_f16);
                return IterationMeasurement{
                    .bytes_processed = matmul_left.bytes().size() +
                                       matmul_right_f16.bytes().size() + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }));
    }

    add_projection_backend_benchmarks(report, "small_prefill", projection_small_left,
                                      projection_small_weight, selection);
    add_projection_backend_benchmarks(report, "next_token_hidden", projection_next_token_left,
                                      projection_hidden_weight, selection);
    add_projection_backend_benchmarks(report, "prefill_hidden", projection_prefill_left,
                                      projection_hidden_weight, selection);
    add_projection_backend_benchmarks(report, "next_token_ffn", projection_next_token_left,
                                      projection_ffn_weight, selection);
    if (selection.include_metal && !selection.metal_available) {
        add_unavailable_metal_projection_benchmark(report);
    }

    if (selection.include_cpu) {
        report.measurements.push_back(measure(
            "tensor.rms_norm_f32", report.config.iterations,
            "input=[64,64] weight=[64] dtype=F32 epsilon=1e-5", [&matmul_left, &rms_norm_weight]() {
                const auto output = rms_norm_f32(matmul_left, rms_norm_weight, 0.00001F);
                return IterationMeasurement{
                    .bytes_processed = matmul_left.bytes().size() + rms_norm_weight.bytes().size() +
                                       output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }));

        report.measurements.push_back(measure(
            "tensor.rope_f32", report.config.iterations,
            "input=[64,64] position=1 rope_dim=64 freq_base=10000 dtype=F32", [&matmul_left]() {
                const auto output = rope_f32(matmul_left, 1, 64, 10000.0F);
                return IterationMeasurement{
                    .bytes_processed = matmul_left.bytes().size() + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }));

        report.measurements.push_back(measure(
            "tensor.rope_f32_with_position_start", report.config.iterations,
            "input=[64,64] position_start=0 rope_dim=64 freq_base=10000 dtype=F32",
            [&matmul_left]() {
                const auto output = rope_f32_with_position_start(matmul_left, 0, 64, 10000.0F);
                return IterationMeasurement{
                    .bytes_processed = matmul_left.bytes().size() + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }));

        report.measurements.push_back(
            measure("tensor.softmax_f32", report.config.iterations,
                    "input=[64,64] dtype=F32 rowwise", [&matmul_left]() {
                        const auto output = softmax_f32(matmul_left);
                        return IterationMeasurement{
                            .bytes_processed = matmul_left.bytes().size() + output.bytes().size(),
                            .checksum = checksum_f32(sum_f32(output)),
                        };
                    }));

        report.measurements.push_back(measure(
            "tokenizer.encode_greedy", report.config.iterations,
            "model=gpt2 vocab=7 text_bytes=24 bos/eos", [&synthetic_tokenizer, &tokenizer_text]() {
                const auto token_ids = synthetic_tokenizer.encode(
                    tokenizer_text, TokenizerEncodeOptions{.add_bos = true, .add_eos = true});
                return IterationMeasurement{
                    .bytes_processed =
                        tokenizer_text.size() + token_ids.size() * sizeof(std::uint32_t),
                    .checksum = token_ids.empty() ? 0
                                                  : static_cast<std::uint64_t>(token_ids.back() +
                                                                               token_ids.size()),
                };
            }));

        report.measurements.push_back(measure(
            "tokenizer.decode", report.config.iterations, "model=gpt2 tokens=6 skip_special=true",
            [&synthetic_tokenizer, &tokenizer_ids]() {
                const auto text =
                    synthetic_tokenizer.decode(std::span<const std::uint32_t>(tokenizer_ids),
                                               TokenizerDecodeOptions{.skip_special_tokens = true});
                return IterationMeasurement{
                    .bytes_processed = tokenizer_ids.size() * sizeof(std::uint32_t) + text.size(),
                    .checksum = checksum_byte_span(std::span<const std::uint8_t>(
                        reinterpret_cast<const std::uint8_t *>(text.data()), text.size())),
                };
            }));

        report.measurements.push_back(measure(
            "tensor.scaled_dot_product_attention_f32", report.config.iterations,
            "query=[64,64] key=[64,64] value=[64,64] dtype=F32 scale=0.125",
            [&attention_query, &attention_key, &attention_value]() {
                const auto output = scaled_dot_product_attention_f32(attention_query, attention_key,
                                                                     attention_value, 0.125F);
                return IterationMeasurement{
                    .bytes_processed = attention_query.bytes().size() +
                                       attention_key.bytes().size() +
                                       attention_value.bytes().size() + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }));

        report.measurements.push_back(
            measure("tensor.causal_scaled_dot_product_attention_f32", report.config.iterations,
                    "query=[64,64] key=[64,64] value=[64,64] dtype=F32 scale=0.125 start=0",
                    [&attention_query, &attention_key, &attention_value]() {
                        const auto output = causal_scaled_dot_product_attention_f32(
                            attention_query, attention_key, attention_value, 0.125F, 0);
                        return IterationMeasurement{
                            .bytes_processed =
                                attention_query.bytes().size() + attention_key.bytes().size() +
                                attention_value.bytes().size() + output.bytes().size(),
                            .checksum = checksum_f32(sum_f32(output)),
                        };
                    }));
    }
}

} // namespace local_gguf_engine::benchmarks
