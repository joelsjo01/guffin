#include "benchmark_synthetic_llama.hpp"

#include "benchmark_resident_kv_decode.hpp"
#include "benchmark_support.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_layer.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace local_gguf_engine::benchmarks {

namespace {

using engine::copy_llama_model_block_weights_to_backend;
using engine::copy_llama_model_logits_weights_to_backend;
using engine::create_llama_backend_kv_cache;
using engine::create_llama_kv_cache;
using engine::generate_llama_argmax_tokens_f32;
using engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using engine::LlamaAttentionLayerWeights;
using engine::LlamaBlockWeights;
using engine::LlamaFeedForwardLayerWeights;
using engine::LlamaBackendKVCache;
using engine::LlamaKVCache;
using engine::LlamaModelConfig;
using engine::LlamaModelWeights;
using engine::run_llama_attention_layer_f32;
using engine::run_llama_block_f32;
using engine::run_llama_feed_forward_layer_f32;
using engine::run_llama_model_blocks_f32;
using engine::run_llama_model_blocks_with_kv_cache_f32;
using engine::run_llama_model_logits_f32;
using engine::run_llama_model_logits_with_kv_cache_f32;
using engine::run_llama_model_next_token_argmax_f32;
using engine::run_llama_model_next_token_argmax_with_kv_cache_f32;
using engine::run_llama_model_next_token_sampled_f32;
using engine::SamplingConfig;
using engine::SamplingState;
using runtime::BackendKind;
using tensor::tensor_backend_unavailable_reason;
using tensor::ops::sum_f32;

void add_unavailable_metal_model_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_model_logits_next_token_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_cached_model_block_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_model_blocks_with_kv_cache_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_backend_kv_cached_model_block_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_model_blocks_with_backend_kv_cache_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_attention_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_attention_layer_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_unavailable_metal_feed_forward_backend_benchmark(BenchmarkReport& report) {
    report.measurements.push_back(BenchmarkMeasurement {
        .name = "engine.llama_feed_forward_layer_f32.metal_buffered.unavailable",
        .iterations = 0,
        .total_ms = 0.0,
        .min_ms = 0.0,
        .max_ms = 0.0,
        .bytes_processed = 0,
        .checksum = 0,
        .detail = tensor_backend_unavailable_reason(BackendKind::metal),
    });
}

void add_synthetic_llama_benchmarks_impl(BenchmarkReport& report, bool metal_available) {
    constexpr std::uint64_t matmul_size = 64;

    const auto llama_layer_input = make_f32_matrix(8, matmul_size, 0.01F);
    const auto llama_cached_model_block_prefill_input = make_f32_matrix(7, matmul_size, 0.011F);
    const auto llama_cached_model_block_decode_input = make_f32_matrix(1, matmul_size, 0.017F);
    const auto llama_attention_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto llama_query_weight = make_f16_matrix(matmul_size, matmul_size, 0.01F);
    const auto llama_key_weight = make_f16_matrix(matmul_size, 32, 0.02F);
    const auto llama_value_weight = make_f16_matrix(matmul_size, 32, 0.03F);
    const auto llama_output_weight = make_f16_matrix(matmul_size, matmul_size, 0.04F);
    const auto llama_feed_forward_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto llama_gate_weight = make_f16_matrix(matmul_size, 128, 0.001F);
    const auto llama_up_weight = make_f16_matrix(matmul_size, 128, 0.0015F);
    const auto llama_down_weight = make_f16_matrix(128, matmul_size, 0.002F);
    const auto llama_token_embedding_weight = make_f16_matrix(matmul_size, 1024, 0.0005F);
    const auto llama_final_norm_weight = make_f32_storage(matmul_size, 1.0F);
    const auto llama_logits_weight = make_f16_matrix(matmul_size, 1024, 0.00075F);
    const std::array<std::uint32_t, 8> llama_token_ids {1, 5, 9, 13, 17, 21, 25, 29};
    const LlamaModelConfig llama_layer_config {
        .context_length = 128,
        .embedding_length = 64,
        .block_count = 1,
        .feed_forward_length = 128,
        .attention_head_count = 4,
        .attention_head_count_kv = 2,
        .attention_key_length = 16,
        .attention_value_length = 16,
        .rope_dimension_count = 16,
        .vocab_size = 1024,
        .rope_freq_base = 10000.0F,
        .rms_norm_epsilon = 0.00001F,
    };
    const LlamaAttentionLayerWeights llama_layer_weights {
        .attention_norm_weight = llama_attention_norm_weight,
        .query_weight = llama_query_weight,
        .key_weight = llama_key_weight,
        .value_weight = llama_value_weight,
        .output_weight = llama_output_weight,
    };
    const LlamaFeedForwardLayerWeights llama_feed_forward_weights {
        .feed_forward_norm_weight = llama_feed_forward_norm_weight,
        .gate_weight = llama_gate_weight,
        .up_weight = llama_up_weight,
        .down_weight = llama_down_weight,
    };
    const LlamaBlockWeights llama_block_weights {
        .attention = llama_layer_weights,
        .feed_forward = llama_feed_forward_weights,
    };
    auto llama_model_blocks_config = llama_layer_config;
    llama_model_blocks_config.block_count = 2;
    const LlamaModelWeights llama_model_weights {
        .token_embedding_weight = llama_token_embedding_weight,
        .output_norm_weight = llama_final_norm_weight,
        .output_weight = llama_logits_weight,
        .blocks = {llama_block_weights, llama_block_weights},
    };
    const auto llama_model_cpu_block_backend_weights = copy_llama_model_block_weights_to_backend(
        llama_model_weights,
        BackendKind::cpu
    );
    const auto llama_attention_cpu_backend_weights = copy_llama_attention_layer_weights_to_backend(
        llama_layer_weights,
        BackendKind::cpu
    );
    const auto llama_feed_forward_cpu_backend_weights = copy_llama_feed_forward_layer_weights_to_backend(
        llama_feed_forward_weights,
        BackendKind::cpu
    );
    const auto llama_model_cpu_backend_weights = copy_llama_model_logits_weights_to_backend(
        llama_model_weights,
        BackendKind::cpu
    );
    const SamplingConfig llama_sampling {
        .temperature = 0.7F,
        .top_k = 40,
        .top_p = 0.9F,
        .repetition_penalty = 1.1F,
        .seed = 1,
    };
    std::vector<LlamaKVCache> cached_model_block_decode_caches;
    cached_model_block_decode_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_kv_cache(llama_model_blocks_config);
        (void)run_llama_model_blocks_with_kv_cache_f32(
            llama_cached_model_block_prefill_input,
            llama_model_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_model_block_decode_caches.push_back(std::move(cache));
    }
    std::vector<LlamaKVCache> cached_model_block_cpu_backend_decode_caches;
    cached_model_block_cpu_backend_decode_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_kv_cache(llama_model_blocks_config);
        (void)run_llama_model_blocks_with_kv_cache_f32(
            llama_cached_model_block_prefill_input,
            llama_model_cpu_block_backend_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_model_block_cpu_backend_decode_caches.push_back(std::move(cache));
    }
    std::vector<LlamaBackendKVCache> cached_model_block_cpu_backend_resident_decode_caches;
    cached_model_block_cpu_backend_resident_decode_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::cpu);
        (void)run_llama_model_blocks_with_kv_cache_f32(
            llama_cached_model_block_prefill_input,
            llama_model_cpu_block_backend_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_model_block_cpu_backend_resident_decode_caches.push_back(std::move(cache));
    }
    std::vector<LlamaKVCache> cached_next_token_caches;
    cached_next_token_caches.reserve(report.config.iterations);
    for (std::size_t index = 0; index < report.config.iterations; ++index) {
        auto cache = create_llama_kv_cache(llama_model_blocks_config);
        (void)run_llama_model_logits_with_kv_cache_f32(
            std::span<const std::uint32_t>(llama_token_ids.data(), llama_token_ids.size() - 1),
            llama_model_weights,
            llama_model_blocks_config,
            6,
            cache
        );
        cached_next_token_caches.push_back(std::move(cache));
    }
    report.measurements.push_back(measure(
        "engine.llama_attention_layer_f32",
        report.config.iterations,
        "input=[8,64] heads=4 kv_heads=2 key=16 value=16 dtype=F32/F16",
        [&llama_layer_input, &llama_layer_weights, &llama_layer_config]() {
            const auto output = run_llama_attention_layer_f32(
                llama_layer_input,
                llama_layer_weights,
                llama_layer_config,
                7
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_layer_weights.attention_norm_weight.bytes().size()
                    + llama_layer_weights.query_weight.bytes().size()
                    + llama_layer_weights.key_weight.bytes().size()
                    + llama_layer_weights.value_weight.bytes().size()
                    + llama_layer_weights.output_weight.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_attention_layer_f32.cpu_buffered",
        report.config.iterations,
        "input=[8,64] heads=4 kv_heads=2 key=16 value=16 backend=cpu preloaded-projections fused-qkv",
        [&llama_layer_input, &llama_attention_cpu_backend_weights, &llama_layer_config]() {
            const auto output = run_llama_attention_layer_f32(
                llama_layer_input,
                llama_attention_cpu_backend_weights,
                llama_layer_config,
                7
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_attention_weight_bytes(llama_attention_cpu_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_feed_forward_layer_f32",
        report.config.iterations,
        "input=[8,64] hidden=64 ffn=128 dtype=F32/F16",
        [&llama_layer_input, &llama_feed_forward_weights, &llama_layer_config]() {
            const auto output = run_llama_feed_forward_layer_f32(
                llama_layer_input,
                llama_feed_forward_weights,
                llama_layer_config
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_feed_forward_weights.feed_forward_norm_weight.bytes().size()
                    + llama_feed_forward_weights.gate_weight.bytes().size()
                    + llama_feed_forward_weights.up_weight.bytes().size()
                    + llama_feed_forward_weights.down_weight.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_feed_forward_layer_f32.cpu_buffered",
        report.config.iterations,
        "input=[8,64] hidden=64 ffn=128 backend=cpu preloaded-projections fused-ffn",
        [&llama_layer_input, &llama_feed_forward_cpu_backend_weights, &llama_layer_config]() {
            const auto output = run_llama_feed_forward_layer_f32(
                llama_layer_input,
                llama_feed_forward_cpu_backend_weights,
                llama_layer_config
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_feed_forward_weight_bytes(llama_feed_forward_cpu_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_block_f32",
        report.config.iterations,
        "input=[8,64] heads=4 kv_heads=2 ffn=128 dtype=F32/F16",
        [&llama_layer_input, &llama_block_weights, &llama_layer_config]() {
            const auto output = run_llama_block_f32(
                llama_layer_input,
                llama_block_weights,
                llama_layer_config,
                7
            );
            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_block_weights.attention.attention_norm_weight.bytes().size()
                    + llama_block_weights.attention.query_weight.bytes().size()
                    + llama_block_weights.attention.key_weight.bytes().size()
                    + llama_block_weights.attention.value_weight.bytes().size()
                    + llama_block_weights.attention.output_weight.bytes().size()
                    + llama_block_weights.feed_forward.feed_forward_norm_weight.bytes().size()
                    + llama_block_weights.feed_forward.gate_weight.bytes().size()
                    + llama_block_weights.feed_forward.up_weight.bytes().size()
                    + llama_block_weights.feed_forward.down_weight.bytes().size()
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_f32",
        report.config.iterations,
        "input=[8,64] blocks=2 heads=4 kv_heads=2 ffn=128 dtype=F32/F16",
        [&llama_layer_input, &llama_model_weights, &llama_model_blocks_config]() {
            const auto output = run_llama_model_blocks_f32(
                llama_layer_input,
                llama_model_weights,
                llama_model_blocks_config,
                7
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = llama_layer_input.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_f32",
        report.config.iterations,
        "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu cache_prefill",
        [&llama_cached_model_block_prefill_input, &llama_model_weights, &llama_model_blocks_config]() {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_weights,
                llama_model_blocks_config,
                6,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_f32.cpu_buffered",
        report.config.iterations,
        "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections fused-qkv fused-block-tail cache_prefill",
        [
            &llama_cached_model_block_prefill_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config
        ]() {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_decode_f32",
        report.config.iterations,
        "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu cache_prefilled=7",
        [
            &llama_cached_model_block_decode_input,
            &llama_model_weights,
            &llama_model_blocks_config,
            &cached_model_block_decode_caches,
            cache_index = std::size_t {0}
        ]() mutable {
            auto& cache = cached_model_block_decode_caches[cache_index];
            ++cache_index;
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_decode_input,
                llama_model_weights,
                llama_model_blocks_config,
                7,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_kv_cache_decode_f32.cpu_buffered",
        report.config.iterations,
        "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections fused-qkv fused-block-tail cache_prefilled=7",
        [
            &llama_cached_model_block_decode_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config,
            &cached_model_block_cpu_backend_decode_caches,
            cache_index = std::size_t {0}
        ]() mutable {
            auto& cache = cached_model_block_cpu_backend_decode_caches[cache_index];
            ++cache_index;
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_decode_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                7,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                    + llama_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_backend_kv_cache_f32.cpu_buffered",
        report.config.iterations,
        "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections resident-kv fused-cached-block model-resident cache_prefill",
        [
            &llama_cached_model_block_prefill_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config
        ]() {
            auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::cpu);
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                    + llama_backend_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.cpu_buffered",
        report.config.iterations,
        "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=cpu preloaded-projections resident-kv fused-cached-block model-resident cache_prefilled=7",
        [
            &llama_cached_model_block_decode_input,
            &llama_model_cpu_block_backend_weights,
            &llama_model_blocks_config,
            &cached_model_block_cpu_backend_resident_decode_caches,
            cache_index = std::size_t {0}
        ]() mutable {
            auto& cache = cached_model_block_cpu_backend_resident_decode_caches[cache_index];
            ++cache_index;
            const auto output = run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_decode_input,
                llama_model_cpu_block_backend_weights,
                llama_model_blocks_config,
                7,
                cache
            );

            return IterationMeasurement {
                .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                    + llama_backend_kv_cache_bytes(cache)
                    + llama_model_block_weight_bytes(llama_model_cpu_block_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_logits_f32",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 dtype=F32/F16",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            const auto output = run_llama_model_logits_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_weights,
                llama_model_blocks_config,
                7
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_model_logits_f32.cpu_buffered",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 dtype=F32/F16 backend=cpu preloaded-projections fused-qkv fused-block-tail",
        [&llama_token_ids, &llama_model_cpu_backend_weights, &llama_model_blocks_config]() {
            const auto output = run_llama_model_logits_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_cpu_backend_weights,
                llama_model_blocks_config,
                7
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_cpu_backend_weights)
                    + output.bytes().size(),
                .checksum = checksum_f32(sum_f32(output)),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_argmax_f32",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 sampler=argmax",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            const auto token = run_llama_model_next_token_argmax_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_weights,
                llama_model_blocks_config,
                7
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = token,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_argmax_f32.cpu_buffered",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 backend=cpu preloaded-projections fused-qkv fused-block-tail sampler=argmax",
        [&llama_token_ids, &llama_model_cpu_backend_weights, &llama_model_blocks_config]() {
            const auto token = run_llama_model_next_token_argmax_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_cpu_backend_weights,
                llama_model_blocks_config,
                7
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_cpu_backend_weights),
                .checksum = token,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_sampled_f32",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 sampler=temp0.7/top-k40/top-p0.9/repeat1.1",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config, &llama_sampling]() {
            SamplingState sampling_state(llama_sampling.seed);
            const auto token = run_llama_model_next_token_sampled_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_weights,
                llama_model_blocks_config,
                7,
                llama_sampling,
                sampling_state,
                std::span<const std::uint32_t>(llama_token_ids)
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_weights),
                .checksum = token,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_next_token_sampled_f32.cpu_buffered",
        report.config.iterations,
        "tokens=8 hidden=64 vocab=1024 blocks=2 backend=cpu preloaded-projections fused-qkv fused-block-tail sampler=temp0.7/top-k40/top-p0.9/repeat1.1",
        [&llama_token_ids, &llama_model_cpu_backend_weights, &llama_model_blocks_config, &llama_sampling]() {
            SamplingState sampling_state(llama_sampling.seed);
            const auto token = run_llama_model_next_token_sampled_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                llama_model_cpu_backend_weights,
                llama_model_blocks_config,
                7,
                llama_sampling,
                sampling_state,
                std::span<const std::uint32_t>(llama_token_ids)
            );

            return IterationMeasurement {
                .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                    + llama_model_weight_bytes(llama_model_cpu_backend_weights),
                .checksum = token,
            };
        }
    ));

    if (metal_available) {
        const auto llama_attention_metal_backend_weights = copy_llama_attention_layer_weights_to_backend(
            llama_layer_weights,
            BackendKind::metal
        );
        const auto llama_feed_forward_metal_backend_weights = copy_llama_feed_forward_layer_weights_to_backend(
            llama_feed_forward_weights,
            BackendKind::metal
        );
        const auto llama_model_metal_block_backend_weights = copy_llama_model_block_weights_to_backend(
            llama_model_weights,
            BackendKind::metal
        );
        const auto llama_model_metal_backend_weights = copy_llama_model_logits_weights_to_backend(
            llama_model_weights,
            BackendKind::metal
        );
        std::vector<LlamaKVCache> cached_model_block_metal_backend_decode_caches;
        cached_model_block_metal_backend_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            (void)run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_metal_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );
            cached_model_block_metal_backend_decode_caches.push_back(std::move(cache));
        }
        std::vector<LlamaBackendKVCache> cached_model_block_metal_backend_resident_decode_caches;
        cached_model_block_metal_backend_resident_decode_caches.reserve(report.config.iterations);
        for (std::size_t index = 0; index < report.config.iterations; ++index) {
            auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::metal);
            (void)run_llama_model_blocks_with_kv_cache_f32(
                llama_cached_model_block_prefill_input,
                llama_model_metal_block_backend_weights,
                llama_model_blocks_config,
                6,
                cache
            );
            cached_model_block_metal_backend_resident_decode_caches.push_back(std::move(cache));
        }

        report.measurements.push_back(measure(
            "engine.llama_attention_layer_f32.metal_buffered",
            report.config.iterations,
            "input=[8,64] heads=4 kv_heads=2 key=16 value=16 backend=metal preloaded-projections fused-qkv",
            [&llama_layer_input, &llama_attention_metal_backend_weights, &llama_layer_config]() {
                const auto output = run_llama_attention_layer_f32(
                    llama_layer_input,
                    llama_attention_metal_backend_weights,
                    llama_layer_config,
                    7
                );
                return IterationMeasurement {
                    .bytes_processed = llama_layer_input.bytes().size()
                        + llama_attention_weight_bytes(llama_attention_metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_feed_forward_layer_f32.metal_buffered",
            report.config.iterations,
            "input=[8,64] hidden=64 ffn=128 backend=metal preloaded-projections fused-ffn one-command-buffer",
            [&llama_layer_input, &llama_feed_forward_metal_backend_weights, &llama_layer_config]() {
                const auto output = run_llama_feed_forward_layer_f32(
                    llama_layer_input,
                    llama_feed_forward_metal_backend_weights,
                    llama_layer_config
                );
                return IterationMeasurement {
                    .bytes_processed = llama_layer_input.bytes().size()
                        + llama_feed_forward_weight_bytes(llama_feed_forward_metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_kv_cache_f32.metal_buffered",
            report.config.iterations,
            "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections fused-qkv fused-block-tail cache_prefill",
            [
                &llama_cached_model_block_prefill_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config
            ]() {
                auto cache = create_llama_kv_cache(llama_model_blocks_config);
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_prefill_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    6,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                        + llama_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_kv_cache_decode_f32.metal_buffered",
            report.config.iterations,
            "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections fused-qkv fused-block-tail cache_prefilled=7",
            [
                &llama_cached_model_block_decode_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config,
                &cached_model_block_metal_backend_decode_caches,
                cache_index = std::size_t {0}
            ]() mutable {
                auto& cache = cached_model_block_metal_backend_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_decode_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    7,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                        + llama_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_f32.metal_buffered",
            report.config.iterations,
            "input=[7,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections resident-kv fused-cached-block model-resident one-readback cache_prefill",
            [
                &llama_cached_model_block_prefill_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config
            ]() {
                auto cache = create_llama_backend_kv_cache(llama_model_blocks_config, BackendKind::metal);
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_prefill_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    6,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_prefill_input.bytes().size()
                        + llama_backend_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_blocks_with_backend_kv_cache_decode_f32.metal_buffered",
            report.config.iterations,
            "input=[1,64] blocks=2 heads=4 kv_heads=2 ffn=128 backend=metal preloaded-projections resident-kv fused-cached-block model-resident one-readback cache_prefilled=7",
            [
                &llama_cached_model_block_decode_input,
                &llama_model_metal_block_backend_weights,
                &llama_model_blocks_config,
                &cached_model_block_metal_backend_resident_decode_caches,
                cache_index = std::size_t {0}
            ]() mutable {
                auto& cache = cached_model_block_metal_backend_resident_decode_caches[cache_index];
                ++cache_index;
                const auto output = run_llama_model_blocks_with_kv_cache_f32(
                    llama_cached_model_block_decode_input,
                    llama_model_metal_block_backend_weights,
                    llama_model_blocks_config,
                    7,
                    cache
                );

                return IterationMeasurement {
                    .bytes_processed = llama_cached_model_block_decode_input.bytes().size()
                        + llama_backend_kv_cache_bytes(cache)
                        + llama_model_block_weight_bytes(llama_model_metal_block_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)) ^ llama_backend_kv_cache_fill_checksum(cache),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_model_logits_f32.metal_buffered",
            report.config.iterations,
            "tokens=8 hidden=64 vocab=1024 blocks=2 dtype=F32/F16 backend=metal preloaded-projections fused-qkv fused-block-tail",
            [&llama_token_ids, &llama_model_metal_backend_weights, &llama_model_blocks_config]() {
                const auto output = run_llama_model_logits_f32(
                    std::span<const std::uint32_t>(llama_token_ids),
                    llama_model_metal_backend_weights,
                    llama_model_blocks_config,
                    7
                );

                return IterationMeasurement {
                    .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                        + llama_model_weight_bytes(llama_model_metal_backend_weights)
                        + output.bytes().size(),
                    .checksum = checksum_f32(sum_f32(output)),
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_next_token_argmax_f32.metal_buffered",
            report.config.iterations,
            "tokens=8 hidden=64 vocab=1024 blocks=2 backend=metal preloaded-projections fused-qkv fused-block-tail sampler=argmax",
            [&llama_token_ids, &llama_model_metal_backend_weights, &llama_model_blocks_config]() {
                const auto token = run_llama_model_next_token_argmax_f32(
                    std::span<const std::uint32_t>(llama_token_ids),
                    llama_model_metal_backend_weights,
                    llama_model_blocks_config,
                    7
                );

                return IterationMeasurement {
                    .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                        + llama_model_weight_bytes(llama_model_metal_backend_weights),
                    .checksum = token,
                };
            }
        ));

        report.measurements.push_back(measure(
            "engine.llama_next_token_sampled_f32.metal_buffered",
            report.config.iterations,
            "tokens=8 hidden=64 vocab=1024 blocks=2 backend=metal preloaded-projections fused-qkv fused-block-tail sampler=temp0.7/top-k40/top-p0.9/repeat1.1",
            [&llama_token_ids, &llama_model_metal_backend_weights, &llama_model_blocks_config, &llama_sampling]() {
                SamplingState sampling_state(llama_sampling.seed);
                const auto token = run_llama_model_next_token_sampled_f32(
                    std::span<const std::uint32_t>(llama_token_ids),
                    llama_model_metal_backend_weights,
                    llama_model_blocks_config,
                    7,
                    llama_sampling,
                    sampling_state,
                    std::span<const std::uint32_t>(llama_token_ids)
                );

                return IterationMeasurement {
                    .bytes_processed = llama_token_ids.size() * sizeof(std::uint32_t)
                        + llama_model_weight_bytes(llama_model_metal_backend_weights),
                    .checksum = token,
                };
            }
        ));
    } else {
        add_unavailable_metal_attention_backend_benchmark(report);
        add_unavailable_metal_feed_forward_backend_benchmark(report);
        add_unavailable_metal_model_backend_benchmark(report);
        add_unavailable_metal_cached_model_block_backend_benchmark(report);
        add_unavailable_metal_backend_kv_cached_model_block_backend_benchmark(report);
    }

    add_focused_resident_kv_decode_stage_benchmarks(report, metal_available);

    report.measurements.push_back(measure(
        "engine.llama_kv_cache_prefill_f32",
        report.config.iterations,
        "prompt_tokens=7 hidden=64 vocab=1024 blocks=2",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            auto cache = create_llama_kv_cache(llama_model_blocks_config);
            const auto logits = run_llama_model_logits_with_kv_cache_f32(
                std::span<const std::uint32_t>(llama_token_ids.data(), llama_token_ids.size() - 1),
                llama_model_weights,
                llama_model_blocks_config,
                6,
                cache
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = (llama_token_ids.size() - 1) * sizeof(std::uint32_t)
                    + cache.layers.size() * (
                        cache.layers[0].key.bytes().size()
                        + cache.layers[0].value.bytes().size()
                    )
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = checksum_f32(sum_f32(logits))
                    ^ static_cast<std::uint64_t>(cache.layers[0].filled_token_count),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_cached_next_token_argmax_f32",
        report.config.iterations,
        "next_tokens=1 hidden=64 vocab=1024 blocks=2 cache_prefilled=7",
        [&llama_token_ids,
         &llama_model_weights,
         &llama_model_blocks_config,
         &cached_next_token_caches,
         cache_index = std::size_t {0}]() mutable {
            auto& cache = cached_next_token_caches[cache_index];
            ++cache_index;
            const auto token = run_llama_model_next_token_argmax_with_kv_cache_f32(
                std::span<const std::uint32_t>(llama_token_ids.data() + (llama_token_ids.size() - 1), 1),
                llama_model_weights,
                llama_model_blocks_config,
                7,
                cache
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = sizeof(std::uint32_t)
                    + cache.layers.size() * (
                        cache.layers[0].key.bytes().size()
                        + cache.layers[0].value.bytes().size()
                    )
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = token ^ static_cast<std::uint64_t>(cache.layers[0].filled_token_count),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_generate_argmax_f32",
        report.config.iterations,
        "prompt_tokens=8 max_new_tokens=2 hidden=64 vocab=1024 blocks=2",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            constexpr std::size_t max_new_tokens = 2;
            const auto tokens = generate_llama_argmax_tokens_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                max_new_tokens,
                llama_model_weights,
                llama_model_blocks_config
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = tokens.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + max_new_tokens * llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = tokens.empty() ? 0 : tokens.back(),
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.llama_generate_argmax_with_kv_cache_f32",
        report.config.iterations,
        "prompt_tokens=8 max_new_tokens=2 hidden=64 vocab=1024 blocks=2",
        [&llama_token_ids, &llama_model_weights, &llama_model_blocks_config]() {
            constexpr std::size_t max_new_tokens = 2;
            const auto tokens = generate_llama_argmax_tokens_with_kv_cache_f32(
                std::span<const std::uint32_t>(llama_token_ids),
                max_new_tokens,
                llama_model_weights,
                llama_model_blocks_config
            );
            const auto block_weight_bytes =
                llama_model_weights.blocks[0].attention.attention_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.query_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.key_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.value_weight.bytes().size()
                + llama_model_weights.blocks[0].attention.output_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.feed_forward_norm_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.gate_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.up_weight.bytes().size()
                + llama_model_weights.blocks[0].feed_forward.down_weight.bytes().size();

            return IterationMeasurement {
                .bytes_processed = tokens.size() * sizeof(std::uint32_t)
                    + llama_model_weights.token_embedding_weight.bytes().size()
                    + llama_model_weights.output_norm_weight.bytes().size()
                    + llama_model_weights.output_weight.bytes().size()
                    + max_new_tokens * llama_model_weights.blocks.size() * block_weight_bytes,
                .checksum = tokens.empty() ? 0 : tokens.back(),
            };
        }
    ));
}

}  // namespace

void add_synthetic_llama_benchmarks(BenchmarkReport& report, bool metal_available) {
    add_synthetic_llama_benchmarks_impl(report, metal_available);
}

}  // namespace local_gguf_engine::benchmarks
