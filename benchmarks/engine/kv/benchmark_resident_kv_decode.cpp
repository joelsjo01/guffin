#include "benchmark_resident_kv_decode.hpp"

#include "../../support/benchmark_support.hpp"

#include "../../../src/tensor/backends/metal/metal_benchmark.hpp"
#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::benchmarks {

namespace {

using engine::copy_llama_model_block_weights_to_backend;
using engine::create_llama_backend_kv_cache;
using engine::LlamaAttentionLayerWeights;
using engine::LlamaBackendKVCache;
using engine::LlamaBlockWeights;
using engine::LlamaFeedForwardLayerWeights;
using engine::LlamaModelConfig;
using engine::LlamaModelWeights;
using engine::run_llama_model_blocks_with_kv_cache_f32;
using runtime::BackendKind;
using tensor::copy_tensor_to_backend;
using tensor::tensor_backend_unavailable_reason;
using tensor::TensorStorage;
using tensor::ops::sum_f32;

[[nodiscard]] std::string focused_resident_kv_decode_detail(std::string_view stage,
                                                            std::string_view backend,
                                                            std::uint64_t cache_length,
                                                            std::uint64_t blocks,
                                                            std::string_view cost_scope) {
    std::ostringstream stream;
    stream << "input=[1,576] blocks=" << blocks
           << " heads=9 kv_heads=3 key=64 value=64 ffn=1536 backend=" << backend
           << " resident-kv cache_prefilled=" << cache_length << " stage=" << stage
           << " cost=" << cost_scope;
    return stream.str();
}

[[nodiscard]] LlamaBackendKVCache make_prefilled_focused_backend_kv_cache(
    const LlamaModelConfig &config, BackendKind backend, std::uint64_t filled_token_count,
    const TensorStorage &key_storage, const TensorStorage &value_storage) {
    auto cache = create_llama_backend_kv_cache(config, backend);
    for (auto &layer : cache.layers) {
        layer.key = copy_tensor_to_backend(key_storage, backend);
        layer.value = copy_tensor_to_backend(value_storage, backend);
        layer.filled_token_count = filled_token_count;
    }
    return cache;
}

void add_unavailable_metal_focused_resident_kv_decode_stage_benchmark(BenchmarkReport &report) {
    report.measurements.push_back(BenchmarkMeasurement{
        .name = "engine.llama_cached_decode_stage_sweep.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

} // namespace

void add_focused_resident_kv_decode_stage_benchmarks(BenchmarkReport &report,
                                                     BenchmarkBackendSelection selection) {
    constexpr std::uint64_t hidden = 576;
    constexpr std::uint64_t feed_forward = 1536;
    constexpr std::uint64_t head_count = 9;
    constexpr std::uint64_t kv_head_count = 3;
    constexpr std::uint64_t key_length = 64;
    constexpr std::uint64_t value_length = 64;
    constexpr std::uint64_t context_length = 4096;
    constexpr std::uint64_t vocab_size = 16;
    constexpr float rms_norm_epsilon = 0.00001F;
    constexpr float rope_freq_base = 10000.0F;
    constexpr float attention_scale = 0.125F;
    const std::array<std::uint64_t, 5> cache_lengths{7, 64, 256, 1024, 2048};

    const auto decode_input = make_f32_matrix(1, hidden, 0.0005F);
    const auto attention_norm_weight = make_f32_storage(hidden, 1.0F);
    const auto query_weight =
        make_f16_gguf_weight_matrix(hidden, head_count * key_length, 0.00012F);
    const auto key_weight =
        make_f16_gguf_weight_matrix(hidden, kv_head_count * key_length, 0.00014F);
    const auto value_weight =
        make_f16_gguf_weight_matrix(hidden, kv_head_count * value_length, 0.00016F);
    const auto output_weight =
        make_f16_gguf_weight_matrix(head_count * value_length, hidden, 0.00011F);
    const auto feed_forward_norm_weight = make_f32_storage(hidden, 1.0F);
    const auto gate_weight = make_f16_gguf_weight_matrix(hidden, feed_forward, 0.00007F);
    const auto up_weight = make_f16_gguf_weight_matrix(hidden, feed_forward, 0.00008F);
    const auto down_weight = make_f16_gguf_weight_matrix(feed_forward, hidden, 0.00006F);
    const auto token_embedding_weight = make_f16_gguf_weight_matrix(hidden, vocab_size, 0.00003F);
    const auto final_norm_weight = make_f32_storage(hidden, 1.0F);
    const auto logits_weight = make_f16_gguf_weight_matrix(hidden, vocab_size, 0.00004F);
    const auto query_fixture = make_f32_matrix(1, head_count * key_length, 0.00023F);
    const auto key_fixture = make_f32_matrix(1, kv_head_count * key_length, 0.00017F);
    const auto value_fixture = make_f32_matrix(1, kv_head_count * value_length, 0.00018F);
    const auto attention_output_fixture = make_f32_matrix(1, head_count * value_length, 0.00019F);
    const auto attention_residual_fixture = make_f32_matrix(1, hidden, 0.00021F);
    const auto gated_fixture = make_f32_matrix(1, feed_forward, 0.00010F);
    const auto cache_key_storage =
        make_f32_matrix(context_length, kv_head_count * key_length, 0.00009F);
    const auto cache_value_storage =
        make_f32_matrix(context_length, kv_head_count * value_length, 0.00007F);

    LlamaModelConfig focused_config{
        .context_length = static_cast<std::uint32_t>(context_length),
        .embedding_length = static_cast<std::uint32_t>(hidden),
        .block_count = 2,
        .feed_forward_length = static_cast<std::uint32_t>(feed_forward),
        .attention_head_count = static_cast<std::uint32_t>(head_count),
        .attention_head_count_kv = static_cast<std::uint32_t>(kv_head_count),
        .attention_key_length = static_cast<std::uint32_t>(key_length),
        .attention_value_length = static_cast<std::uint32_t>(value_length),
        .rope_dimension_count = static_cast<std::uint32_t>(key_length),
        .vocab_size = static_cast<std::uint32_t>(vocab_size),
        .rope_freq_base = rope_freq_base,
        .rms_norm_epsilon = rms_norm_epsilon,
    };
    const LlamaAttentionLayerWeights attention_weights{
        .attention_norm_weight = attention_norm_weight,
        .query_weight = query_weight,
        .key_weight = key_weight,
        .value_weight = value_weight,
        .output_weight = output_weight,
    };
    const LlamaFeedForwardLayerWeights feed_forward_weights{
        .feed_forward_norm_weight = feed_forward_norm_weight,
        .gate_weight = gate_weight,
        .up_weight = up_weight,
        .down_weight = down_weight,
    };
    const LlamaBlockWeights block_weights{
        .attention = attention_weights,
        .feed_forward = feed_forward_weights,
    };
    const LlamaModelWeights model_weights{
        .token_embedding_weight = token_embedding_weight,
        .output_norm_weight = final_norm_weight,
        .output_weight = logits_weight,
        .blocks = {block_weights, block_weights},
    };
    if (selection.include_cpu) {
        const auto cpu_backend_weights =
            copy_llama_model_block_weights_to_backend(model_weights, BackendKind::cpu);

        for (const auto cache_length : cache_lengths) {
            std::vector<LlamaBackendKVCache> cpu_decode_caches;
            cpu_decode_caches.reserve(report.config.iterations);
            for (std::size_t index = 0; index < report.config.iterations; ++index) {
                cpu_decode_caches.push_back(make_prefilled_focused_backend_kv_cache(
                    focused_config, BackendKind::cpu, cache_length, cache_key_storage,
                    cache_value_storage));
            }

            report.measurements.push_back(measure(
                "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.cpu_"
                "buffered.focused_cache_" +
                    std::to_string(cache_length),
                report.config.iterations,
                focused_resident_kv_decode_detail("full_model_decode", "cpu", cache_length,
                                                  focused_config.block_count,
                                                  "model-resident fused-cached-block one-readback"),
                [&decode_input, &cpu_backend_weights, &focused_config, &cpu_decode_caches,
                 cache_index = std::size_t{0}, cache_length]() mutable {
                    auto &cache = cpu_decode_caches[cache_index];
                    ++cache_index;
                    const auto output = run_llama_model_blocks_with_kv_cache_f32(
                        decode_input, cpu_backend_weights, focused_config, cache_length, cache);
                    return IterationMeasurement{
                        .bytes_processed = decode_input.bytes().size() +
                                           llama_backend_kv_cache_bytes(cache) +
                                           llama_model_block_weight_bytes(cpu_backend_weights) +
                                           output.bytes().size(),
                        .checksum = checksum_f32(sum_f32(output)) ^
                                    llama_backend_kv_cache_fill_checksum(cache),
                    };
                }));
        }
    }

    if (!selection.include_metal) {
        return;
    }

    if (!selection.metal_available) {
        add_unavailable_metal_focused_resident_kv_decode_stage_benchmark(report);
        return;
    }

    const auto metal_backend_weights =
        copy_llama_model_block_weights_to_backend(model_weights, BackendKind::metal);
    const auto input_buffer = copy_tensor_to_backend(decode_input, BackendKind::metal);
    const auto query_buffer = copy_tensor_to_backend(query_fixture, BackendKind::metal);
    const auto key_buffer = copy_tensor_to_backend(key_fixture, BackendKind::metal);
    const auto value_buffer = copy_tensor_to_backend(value_fixture, BackendKind::metal);
    const auto attention_output_buffer =
        copy_tensor_to_backend(attention_output_fixture, BackendKind::metal);
    const auto attention_residual_buffer =
        copy_tensor_to_backend(attention_residual_fixture, BackendKind::metal);
    const auto gated_buffer = copy_tensor_to_backend(gated_fixture, BackendKind::metal);
    const auto short_decode_cache_length = cache_lengths.front();

    report.measurements.push_back(
        measure("tensor.metal_noop_dispatch", report.config.iterations,
                "backend=metal stage=noop cost=command-buffer+one-empty-dispatch", []() {
                    tensor::backends::metal::benchmark_noop_dispatch();
                    return IterationMeasurement{
                        .bytes_processed = 0,
                        .checksum = 1,
                    };
                }));

    auto append_cache = make_prefilled_focused_backend_kv_cache(
        focused_config, BackendKind::metal, short_decode_cache_length, cache_key_storage,
        cache_value_storage);
    {
        const auto &block = metal_backend_weights.blocks[0];
        (void)tensor::backends::metal::benchmark_llama_cached_decode_attention_norm_to_backend(
            input_buffer, block.attention.attention_norm_weight_buffer, rms_norm_epsilon);
        (void)tensor::backends::metal::benchmark_llama_cached_decode_qkv_projection_to_backend(
            input_buffer, block.attention.query_weight, block.attention.key_weight,
            block.attention.value_weight);
        auto warmup_append_cache = make_prefilled_focused_backend_kv_cache(
            focused_config, BackendKind::metal, short_decode_cache_length, cache_key_storage,
            cache_value_storage);
        tensor::backends::metal::benchmark_llama_cached_decode_kv_append_to_backend(
            key_buffer, value_buffer, warmup_append_cache.layers[0].key,
            warmup_append_cache.layers[0].value, short_decode_cache_length, kv_head_count,
            key_length, value_length, key_length, rope_freq_base);
        (void)tensor::backends::metal::benchmark_llama_cached_decode_attention_residual_to_backend(
            input_buffer, attention_output_buffer, block.attention.output_weight);
        (void)tensor::backends::metal::benchmark_llama_cached_decode_ffn_gate_up_to_backend(
            attention_residual_buffer, block.feed_forward.feed_forward_norm_weight_buffer,
            block.feed_forward.gate_weight, block.feed_forward.up_weight, rms_norm_epsilon);
        (void)tensor::backends::metal::benchmark_llama_cached_decode_ffn_down_to_backend(
            gated_buffer, attention_residual_buffer, block.feed_forward.down_weight);
    }

    report.measurements.push_back(measure(
        "engine.llama_cached_decode_attention_norm_f32.metal_buffered.focused",
        report.config.iterations,
        focused_resident_kv_decode_detail("attention_norm", "metal", short_decode_cache_length, 1,
                                          "RMSNorm one-dispatch no-readback"),
        [&input_buffer, &metal_backend_weights]() {
            const auto &block = metal_backend_weights.blocks[0];
            const auto output =
                tensor::backends::metal::benchmark_llama_cached_decode_attention_norm_to_backend(
                    input_buffer, block.attention.attention_norm_weight_buffer, rms_norm_epsilon);
            return IterationMeasurement{
                .bytes_processed =
                    backend_buffer_byte_count(input_buffer) +
                    backend_buffer_byte_count(block.attention.attention_norm_weight_buffer) +
                    backend_buffer_byte_count(output),
                .checksum = output.descriptor().byte_count,
            };
        }));

    report.measurements.push_back(measure(
        "engine.llama_cached_decode_qkv_projection_f32.metal_buffered.focused",
        report.config.iterations,
        focused_resident_kv_decode_detail("qkv_projection", "metal", short_decode_cache_length, 1,
                                          "Q/K/V matmuls one-command-buffer no-readback"),
        [&input_buffer, &metal_backend_weights]() {
            const auto &block = metal_backend_weights.blocks[0];
            const auto outputs =
                tensor::backends::metal::benchmark_llama_cached_decode_qkv_projection_to_backend(
                    input_buffer, block.attention.query_weight, block.attention.key_weight,
                    block.attention.value_weight);
            return IterationMeasurement{
                .bytes_processed = backend_buffer_byte_count(input_buffer) +
                                   backend_buffer_byte_count(block.attention.query_weight) +
                                   backend_buffer_byte_count(block.attention.key_weight) +
                                   backend_buffer_byte_count(block.attention.value_weight) +
                                   outputs[0].descriptor().byte_count +
                                   outputs[1].descriptor().byte_count +
                                   outputs[2].descriptor().byte_count,
                .checksum = outputs[0].descriptor().byte_count ^
                            (outputs[1].descriptor().byte_count << 1U) ^
                            (outputs[2].descriptor().byte_count << 2U),
            };
        }));

    report.measurements.push_back(
        measure("engine.llama_cached_decode_kv_append_f32.metal_buffered.focused_cache_" +
                    std::to_string(short_decode_cache_length),
                report.config.iterations,
                focused_resident_kv_decode_detail("kv_append", "metal", short_decode_cache_length,
                                                  1, "RoPE-key+KV-append one-dispatch no-readback"),
                [&key_buffer, &value_buffer, &append_cache, short_decode_cache_length]() {
                    tensor::backends::metal::benchmark_llama_cached_decode_kv_append_to_backend(
                        key_buffer, value_buffer, append_cache.layers[0].key,
                        append_cache.layers[0].value, short_decode_cache_length, kv_head_count,
                        key_length, value_length, key_length, rope_freq_base);
                    append_cache.layers[0].filled_token_count = short_decode_cache_length + 1;
                    return IterationMeasurement{
                        .bytes_processed = backend_buffer_byte_count(key_buffer) +
                                           backend_buffer_byte_count(value_buffer) +
                                           llama_backend_kv_cache_bytes(append_cache),
                        .checksum = llama_backend_kv_cache_fill_checksum(append_cache) ^
                                    backend_buffer_byte_count(append_cache.layers[0].key),
                    };
                }));

    report.measurements.push_back(measure(
        "engine.llama_cached_decode_attention_residual_f32.metal_buffered."
        "focused",
        report.config.iterations,
        focused_resident_kv_decode_detail(
            "attention_residual", "metal", short_decode_cache_length, 1,
            "attention-output-projection+residual one-dispatch no-readback"),
        [&input_buffer, &attention_output_buffer, &metal_backend_weights]() {
            const auto &block = metal_backend_weights.blocks[0];
            const auto output = tensor::backends::metal::
                benchmark_llama_cached_decode_attention_residual_to_backend(
                    input_buffer, attention_output_buffer, block.attention.output_weight);
            return IterationMeasurement{
                .bytes_processed = backend_buffer_byte_count(input_buffer) +
                                   backend_buffer_byte_count(attention_output_buffer) +
                                   backend_buffer_byte_count(block.attention.output_weight) +
                                   backend_buffer_byte_count(output),
                .checksum = output.descriptor().byte_count,
            };
        }));

    report.measurements.push_back(measure(
        "engine.llama_cached_decode_ffn_gate_up_f32.metal_buffered.focused",
        report.config.iterations,
        focused_resident_kv_decode_detail(
            "ffn_gate_up_silu", "metal", short_decode_cache_length, 1,
            "FFN RMSNorm+gate/up+SiLU one-command-buffer no-readback"),
        [&attention_residual_buffer, &metal_backend_weights]() {
            const auto &block = metal_backend_weights.blocks[0];
            const auto output =
                tensor::backends::metal::benchmark_llama_cached_decode_ffn_gate_up_to_backend(
                    attention_residual_buffer, block.feed_forward.feed_forward_norm_weight_buffer,
                    block.feed_forward.gate_weight, block.feed_forward.up_weight, rms_norm_epsilon);
            return IterationMeasurement{
                .bytes_processed =
                    backend_buffer_byte_count(attention_residual_buffer) +
                    backend_buffer_byte_count(block.feed_forward.feed_forward_norm_weight_buffer) +
                    backend_buffer_byte_count(block.feed_forward.gate_weight) +
                    backend_buffer_byte_count(block.feed_forward.up_weight) +
                    backend_buffer_byte_count(output),
                .checksum = output.descriptor().byte_count,
            };
        }));

    report.measurements.push_back(measure(
        "engine.llama_cached_decode_ffn_down_f32.metal_buffered.focused", report.config.iterations,
        focused_resident_kv_decode_detail("ffn_down_residual", "metal", short_decode_cache_length,
                                          1,
                                          "FFN down-projection+residual one-dispatch no-readback"),
        [&gated_buffer, &attention_residual_buffer, &metal_backend_weights]() {
            const auto &block = metal_backend_weights.blocks[0];
            const auto output =
                tensor::backends::metal::benchmark_llama_cached_decode_ffn_down_to_backend(
                    gated_buffer, attention_residual_buffer, block.feed_forward.down_weight);
            return IterationMeasurement{
                .bytes_processed = backend_buffer_byte_count(gated_buffer) +
                                   backend_buffer_byte_count(attention_residual_buffer) +
                                   backend_buffer_byte_count(block.feed_forward.down_weight) +
                                   backend_buffer_byte_count(output),
                .checksum = output.descriptor().byte_count,
            };
        }));

    for (const auto cache_length : cache_lengths) {
        std::vector<LlamaBackendKVCache> metal_decode_caches;
        metal_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            metal_decode_caches.push_back(make_prefilled_focused_backend_kv_cache(
                focused_config, BackendKind::metal, cache_length, cache_key_storage,
                cache_value_storage));
        }
        auto front_end_cache = make_prefilled_focused_backend_kv_cache(
            focused_config, BackendKind::metal, cache_length, cache_key_storage,
            cache_value_storage);
        auto attention_cache = make_prefilled_focused_backend_kv_cache(
            focused_config, BackendKind::metal, cache_length + 1, cache_key_storage,
            cache_value_storage);

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.metal_"
            "buffered.focused_cache_" +
                std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail("full_model_decode", "metal", cache_length,
                                              focused_config.block_count,
                                              "model-resident fused-cached-block one-readback"),
            [&decode_input, &metal_backend_weights, &focused_config, &metal_decode_caches,
             cache_index = std::size_t{0}, cache_length]() mutable {
                auto &cache = metal_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    decode_input, metal_backend_weights, focused_config, cache_length, cache);
                return IterationMeasurement{
                    .bytes_processed = decode_input.bytes().size() +
                                       llama_backend_kv_cache_bytes(cache) +
                                       llama_model_block_weight_bytes(metal_backend_weights) +
                                       output.bytes().size(),
                    .checksum =
                        checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }));

        report.measurements.push_back(measure(
            "engine.llama_cached_decode_attention_frontend_f32.metal_buffered."
            "focused_cache_" +
                std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "attention_frontend", "metal", cache_length, 1,
                "RMSNorm+QKV+KV-append one-command-buffer no-readback"),
            [&input_buffer, &metal_backend_weights, &front_end_cache, cache_length]() {
                const auto &block = metal_backend_weights.blocks[0];
                auto outputs = tensor::backends::metal::
                    benchmark_llama_cached_decode_attention_frontend_to_backend(
                        input_buffer, block.attention.attention_norm_weight_buffer,
                        block.attention.query_weight, block.attention.key_weight,
                        block.attention.value_weight, front_end_cache.layers[0].key,
                        front_end_cache.layers[0].value, cache_length, head_count, kv_head_count,
                        key_length, value_length, key_length, rope_freq_base, rms_norm_epsilon);
                front_end_cache.layers[0].filled_token_count = cache_length + 1;
                return IterationMeasurement{
                    .bytes_processed = backend_buffer_byte_count(input_buffer) +
                                       llama_attention_weight_bytes(block.attention) +
                                       llama_backend_kv_cache_bytes(front_end_cache) +
                                       outputs[0].descriptor().byte_count +
                                       outputs[1].descriptor().byte_count +
                                       outputs[2].descriptor().byte_count,
                    .checksum = outputs[0].descriptor().byte_count ^
                                (outputs[1].descriptor().byte_count << 1U) ^
                                (outputs[2].descriptor().byte_count << 2U) ^
                                llama_backend_kv_cache_fill_checksum(front_end_cache),
                };
            }));

        report.measurements.push_back(measure(
            "engine.llama_cached_decode_attention_output_f32.metal_buffered."
            "focused_cache_" +
                std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail("cached_attention_output", "metal", cache_length, 1,
                                              "attention-output-only no-append no-readback"),
            [&query_buffer, &attention_cache, cache_length]() {
                const auto output = tensor::backends::metal::
                    benchmark_llama_cached_decode_attention_output_to_backend(
                        query_buffer, attention_cache.layers[0].key,
                        attention_cache.layers[0].value, cache_length, head_count, kv_head_count,
                        key_length, value_length, key_length, rope_freq_base, attention_scale);
                return IterationMeasurement{
                    .bytes_processed = backend_buffer_byte_count(query_buffer) +
                                       llama_backend_kv_cache_bytes(attention_cache) +
                                       backend_buffer_byte_count(output),
                    .checksum = output.descriptor().byte_count ^
                                llama_backend_kv_cache_fill_checksum(attention_cache),
                };
            }));

        report.measurements.push_back(measure(
            "engine.llama_cached_decode_block_tail_f32.metal_buffered.focused_"
            "cache_" +
                std::to_string(cache_length),
            report.config.iterations,
            focused_resident_kv_decode_detail(
                "block_tail", "metal", cache_length, 1,
                "attention-output-projection+residual+FFN no-readback"),
            [&input_buffer, &attention_output_buffer, &metal_backend_weights]() {
                const auto &block = metal_backend_weights.blocks[0];
                const auto output =
                    tensor::backends::metal::benchmark_llama_cached_decode_block_tail_to_backend(
                        input_buffer, attention_output_buffer, block.attention.output_weight,
                        block.feed_forward.feed_forward_norm_weight_buffer,
                        block.feed_forward.gate_weight, block.feed_forward.up_weight,
                        block.feed_forward.down_weight, rms_norm_epsilon);
                return IterationMeasurement{
                    .bytes_processed = backend_buffer_byte_count(input_buffer) +
                                       backend_buffer_byte_count(attention_output_buffer) +
                                       llama_feed_forward_weight_bytes(block.feed_forward) +
                                       backend_buffer_byte_count(block.attention.output_weight) +
                                       backend_buffer_byte_count(output),
                    .checksum = output.descriptor().byte_count,
                };
            }));
    }
}

} // namespace local_gguf_engine::benchmarks
