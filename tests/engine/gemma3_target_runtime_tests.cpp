#include "gemma3_target_runtime_tests.hpp"
#include "gemma3_target_test_support.hpp"
#include "gemma3_target_golden_hashes.hpp"

#include "local_gguf_engine/engine/chain_chat.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/typed_access.hpp"
#include "../../src/tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace local_gguf_engine;
using namespace local_gguf_engine::engine;
using tests::expect;
using tests::TestFailure;
using tensor::DType;
using Floats = std::vector<float>;

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16U; value *= 0x7feb352dU;
    value ^= value >> 15U; value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

// Explicit native block packing is paired with independently retained dense
// values. The oracle never calls production quantization or tensor operations.
struct NumericalFixture : tests::Gemma3TargetFixture {
    std::map<std::string, Floats> dense;

    NumericalFixture() {
        for (std::size_t t = 0U; t < tensors.size(); ++t) {
            auto& tensor = tensors[t];
            const auto& name = view.tensor_payloads[t].tensor.name;
            auto bytes = tensor.mutable_bytes();
            auto& values = dense[name];
            values.resize(static_cast<std::size_t>(tensor.descriptor().element_count));
            std::uint32_t seed = 2166136261U;
            for (const auto c : name) seed = (seed ^ static_cast<std::uint8_t>(c)) * 16777619U;
            if (tensor.descriptor().dtype == DType::f32) {
                for (std::size_t i = 0U; i < values.size(); ++i) {
                    values[i] = 0.75F + static_cast<float>(mix(seed + static_cast<std::uint32_t>(i)) % 65U) / 128.0F;
                }
                std::memcpy(bytes.data(), values.data(), bytes.size());
                continue;
            }
            const bool q4 = tensor.descriptor().dtype == DType::q4_k;
            const auto block_bytes = q4 ? 144U : 210U;
            for (std::size_t b = 0U; b < values.size() / 256U; ++b) {
                auto* packed = bytes.data() + b * block_bytes;
                if (q4) {
                    // d=dmin=2^-9; all eight group scales=1, offsets=8.
                    packed[1] = 0x18U; packed[3] = 0x18U;
                    for (std::size_t i = 0U; i < 4U; ++i) {
                        packed[4U + i] = 1U; packed[8U + i] = 8U; packed[12U + i] = 0x81U;
                    }
                } else {
                    packed[209] = 0x10U;  // d=2^-11
                    for (std::size_t group = 0U; group < 16U; ++group) {
                        const auto scale = static_cast<std::int8_t>((group % 3U) + 1U)
                            * (group % 2U == 0U ? 1 : -1);
                        packed[192U + group] = static_cast<std::uint8_t>(scale);
                    }
                }
                for (std::size_t i = 0U; i < 256U; ++i) {
                    const auto random = mix(seed + static_cast<std::uint32_t>(b * 256U + i));
                    if (q4) {
                        const auto quant = random & 15U;
                        packed[16U + (i / 64U) * 32U + i % 32U] |=
                            static_cast<std::uint8_t>(quant << (i % 64U < 32U ? 0U : 4U));
                        values[b * 256U + i] = static_cast<float>(static_cast<int>(quant) - 8) / 512.0F;
                    } else {
                        const auto quant = random & 63U;
                        const auto within = i % 128U;
                        const auto low_index = (i / 128U) * 64U + (within / 32U % 2U) * 32U + within % 32U;
                        packed[low_index] |= static_cast<std::uint8_t>((quant & 15U) << (within < 64U ? 0U : 4U));
                        packed[128U + (i / 128U) * 32U + within % 32U] |=
                            static_cast<std::uint8_t>((quant >> 4U) << (2U * (within / 32U)));
                        const auto group = i / 16U;
                        const auto scale = static_cast<int>(group % 3U + 1U) * (group % 2U == 0U ? 1 : -1);
                        values[b * 256U + i] = static_cast<float>((static_cast<int>(quant) - 32) * scale) / 2048.0F;
                    }
                }
            }
        }
    }

    Floats matrix(const std::string& name, std::span<const float> input) const {
        const auto& weight = dense.at(name);
        Floats output(weight.size() / input.size());
        for (std::size_t row = 0U; row < output.size(); ++row) {
            double sum = 0.0;
            for (std::size_t i = 0U; i < input.size(); ++i) sum += double(input[i]) * weight[row * input.size() + i];
            output[row] = static_cast<float>(sum);
        }
        return output;
    }

    Floats norm(const std::string& name, std::span<const float> input) const {
        const auto& gamma = dense.at(name);
        double squares = 0.0;
        for (const auto x : input) squares += double(x) * x;
        const auto scale = static_cast<float>(1.0 / std::sqrt(squares / double(input.size()) + 1.0e-6));
        Floats output(input.size());
        for (std::size_t i = 0U; i < input.size(); ++i) output[i] = input[i] * scale * gamma[i];
        return output;
    }
};

struct Reference {
    const NumericalFixture& fixture;
    const GemmaKVPrecision precision;
    std::array<Floats, 6> key, value;
    std::vector<Floats> logits;

    explicit Reference(const NumericalFixture& input, GemmaKVPrecision cache_precision = GemmaKVPrecision::f32)
        : fixture(input), precision(cache_precision) {}

    // Independent binary16 RNE oracle. Work in exact powers of two and round
    // the integer significand; do not share the production conversion helper.
    static float rounded_half(float input) {
        const auto magnitude = std::abs(double(input));
        if (magnitude == 0.0 || !std::isfinite(magnitude)) return input;
        int exponent = 0;
        static_cast<void>(std::frexp(magnitude, &exponent));
        const auto quantum = std::ldexp(1.0, std::max(exponent - 11, -24));
        const auto scaled = magnitude / quantum;
        auto integer = std::floor(scaled);
        const auto fraction = scaled - integer;
        if (fraction > 0.5 || (fraction == 0.5 && std::fmod(integer, 2.0) != 0.0)) ++integer;
        const auto rounded = integer * quantum;
        return std::copysign(rounded > 65504.0 ? std::numeric_limits<float>::infinity()
            : static_cast<float>(rounded), input);
    }

    Floats append(std::uint32_t token) {
        const auto position = logits.size();
        const auto& embedding = fixture.dense.at("token_embd.weight");
        Floats hidden(256U);
        for (std::size_t i = 0U; i < 256U; ++i) hidden[i] = 16.0F * embedding[token * 256U + i];
        for (std::size_t layer = 0U; layer < 6U; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto normalized = fixture.norm(prefix + "attn_norm.weight", hidden);
            auto q = fixture.matrix(prefix + "attn_q.weight", normalized);
            auto k = fixture.matrix(prefix + "attn_k.weight", normalized);
            auto v = fixture.matrix(prefix + "attn_v.weight", normalized);
            for (std::size_t head = 0U; head < 2U; ++head) {
                const auto normalized_head = fixture.norm(prefix + "attn_q_norm.weight",
                    std::span<const float>(q).subspan(head * 256U, 256U));
                std::copy(normalized_head.begin(), normalized_head.end(), q.begin() + static_cast<std::ptrdiff_t>(head * 256U));
            }
            k = fixture.norm(prefix + "attn_k_norm.weight", k);
            const bool local = layer != 5U;
            const auto rotate = [&](Floats& input) {
                for (std::size_t head = 0U; head < input.size() / 256U; ++head) {
                    for (std::size_t i = 0U; i < 128U; ++i) {
                        const auto angle = double(position) * (local ? 1.0 : 0.125)
                            * std::pow(local ? 10000.0 : 1000000.0, -2.0 * double(i) / 256.0);
                        const auto a = input[head * 256U + i], b = input[head * 256U + i + 128U];
                        input[head * 256U + i] = static_cast<float>(a * std::cos(angle) - b * std::sin(angle));
                        input[head * 256U + i + 128U] = static_cast<float>(a * std::sin(angle) + b * std::cos(angle));
                    }
                }
            };
            rotate(q); rotate(k);
            if (precision == GemmaKVPrecision::f16) {
                // The current token is rounded before its own attention, as
                // are earlier private rows in a verifier or prompt chunk.
                for (auto& x : k) x = rounded_half(x);
                for (auto& x : v) x = rounded_half(x);
            }
            key[layer].insert(key[layer].end(), k.begin(), k.end());
            value[layer].insert(value[layer].end(), v.begin(), v.end());
            Floats attended(512U);
            const auto first = local && position >= 8U ? position - 7U : 0U;
            for (std::size_t head = 0U; head < 2U; ++head) {
                std::vector<double> scores(position - first + 1U);
                for (std::size_t p = first; p <= position; ++p) {
                    double sum = 0.0;
                    for (std::size_t i = 0U; i < 256U; ++i) sum += double(q[head * 256U + i]) * key[layer][p * 256U + i];
                    scores[p - first] = sum / 16.0;
                }
                const auto maximum = *std::max_element(scores.begin(), scores.end());
                double denominator = 0.0;
                for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
                for (std::size_t i = 0U; i < 256U; ++i) {
                    double sum = 0.0;
                    for (std::size_t p = first; p <= position; ++p) sum += scores[p - first] * value[layer][p * 256U + i];
                    attended[head * 256U + i] = static_cast<float>(sum / denominator);
                }
            }
            auto projected = fixture.matrix(prefix + "attn_output.weight", attended);
            projected = fixture.norm(prefix + "post_attention_norm.weight", projected);
            for (std::size_t i = 0U; i < hidden.size(); ++i) hidden[i] += projected[i];
            const auto ffn_input = fixture.norm(prefix + "ffn_norm.weight", hidden);
            auto gate = fixture.matrix(prefix + "ffn_gate.weight", ffn_input);
            const auto up = fixture.matrix(prefix + "ffn_up.weight", ffn_input);
            for (std::size_t i = 0U; i < gate.size(); ++i) {
                const auto x = gate[i];
                gate[i] = 0.5F * x * (1.0F + std::tanh(0.7978845608028654F * (x + 0.044715F * x * x * x))) * up[i];
            }
            projected = fixture.matrix(prefix + "ffn_down.weight", gate);
            projected = fixture.norm(prefix + "post_ffw_norm.weight", projected);
            for (std::size_t i = 0U; i < hidden.size(); ++i) hidden[i] += projected[i];
        }
        logits.push_back(fixture.matrix("token_embd.weight", fixture.norm("output_norm.weight", hidden)));
        return logits.back();
    }
};

Floats read(const tensor::BackendBuffer& buffer) {
    const auto storage = tensor::copy_tensor_from_backend(buffer);
    Floats values(static_cast<std::size_t>(storage.descriptor().element_count));
    if (storage.descriptor().dtype == DType::f32) {
        std::memcpy(values.data(), storage.bytes().data(), storage.bytes().size());
    } else {
        expect(storage.descriptor().dtype == DType::f16, "target readback requires F32 or F16");
        for (std::size_t i = 0U; i < values.size(); ++i) {
            std::uint16_t bits;
            std::memcpy(&bits, storage.bytes().data() + 2U * i, sizeof(bits));
            values[i] = tensor::f16_bits_to_f32(bits);
        }
    }
    return values;
}

void close(std::span<const float> actual, std::span<const float> expected, std::string_view label,
    float absolute = 3.0e-4F, float relative = 3.0e-4F) {
    expect(actual.size() == expected.size(), std::string(label) + " size mismatch");
    for (std::size_t i = 0U; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > absolute + relative * std::abs(expected[i])) {
            throw TestFailure(std::string(label) + " differs at " + std::to_string(i)
                + ": actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
        }
    }
}

struct F16CacheErrorMetrics {
    double component_budget_ratio = 0.0;
    double rms_budget_ratio = 0.0;
};

F16CacheErrorMetrics close_f16_checkpoint_cache(std::span<const float> actual, std::span<const float> expected,
    std::string_view label) {
    constexpr std::size_t head_width = 256U;
    constexpr double absolute = 5.0e-4;
    constexpr double relative = 3.0e-4 + 1.0 / 1024.0;
    expect(actual.size() == expected.size() && actual.size() % head_width == 0U,
        std::string(label) + " head extent mismatch");
    // Independent F32 projection orders can straddle an F16 midpoint in the
    // first layer. Both casts are correct, but the resulting half difference
    // propagates through later projections, including nearly cancelling
    // components. Retain the F32 budget plus one binary16 ULP; scale those
    // components by their own head's RMS, and also bound the whole head's RMS
    // error. These are precision-specific regression acceptance budgets.
    // Native cast bit fixtures and exact ordinary/verifier comparisons
    // independently constrain conversion and the production target semantics.
    F16CacheErrorMetrics metrics;
    for (std::size_t first = 0U; first < actual.size(); first += head_width) {
        double expected_square = 0.0, error_square = 0.0;
        for (std::size_t offset = 0U; offset < head_width; ++offset) {
            const auto index = first + offset;
            expect(std::isfinite(actual[index]) && std::isfinite(expected[index]),
                std::string(label) + " contains nonfinite cache values");
            const auto value = double(expected[index]);
            const auto error = double(actual[index]) - value;
            expected_square += value * value;
            error_square += error * error;
        }
        const auto expected_rms = std::sqrt(expected_square / head_width);
        const auto error_rms = std::sqrt(error_square / head_width);
        metrics.rms_budget_ratio = std::max(metrics.rms_budget_ratio,
            error_rms / (absolute + relative * expected_rms));
        expect(error_rms <= absolute + relative * expected_rms,
            std::string(label) + " RMS error exceeds the F16 budget at head "
                + std::to_string(first / head_width));
        for (std::size_t offset = 0U; offset < head_width; ++offset) {
            const auto index = first + offset;
            const auto scale = std::max(std::abs(double(expected[index])), expected_rms);
            const auto error = std::abs(double(actual[index]) - expected[index]);
            metrics.component_budget_ratio = std::max(metrics.component_budget_ratio,
                error / (absolute + relative * scale));
            if (error > absolute + relative * scale) {
                throw TestFailure(std::string(label) + " differs at " + std::to_string(index)
                    + ": actual=" + std::to_string(actual[index])
                    + " expected=" + std::to_string(expected[index]));
            }
        }
    }
    return metrics;
}

void check_cache(const Gemma2MetalModelPlan& plan, const Reference& reference, std::size_t count) {
    // A tiny difference in otherwise valid F32 arithmetic can cross a half
    // midpoint. Add one binary16 ULP to the existing arithmetic budget, only
    // for the rounded oracle; native ordinary/verifier bits are checked below.
    const auto rounding = reference.precision == GemmaKVPrecision::f16 ? std::ldexp(1.0F, -10) : 0.0F;
    for (std::size_t layer = 0U; layer < 6U; ++layer) {
        const auto& cache = plan.kv_cache()[layer];
        const auto k = read(cache.key), v = read(cache.value);
        const auto first = layer != 5U && count > 8U ? count - 8U : 0U;
        for (std::size_t position = first; position < count; ++position) {
            const auto slot = position & cache.row_mask;
            close(std::span(k).subspan(slot * 256U, 256U), std::span(reference.key[layer]).subspan(position * 256U, 256U),
                "layer " + std::to_string(layer) + " key position " + std::to_string(position), 3.0e-4F, 3.0e-4F + rounding);
            close(std::span(v).subspan(slot * 256U, 256U), std::span(reference.value[layer]).subspan(position * 256U, 256U),
                "layer " + std::to_string(layer) + " value position " + std::to_string(position), 3.0e-4F, 3.0e-4F + rounding);
        }
    }
}

bool have_metal() { return tensor::tensor_backend_is_available(runtime::BackendKind::metal); }

void exact_cache(const Gemma2MetalModelPlan& actual, const Gemma2MetalModelPlan& expected, std::size_t count) {
    expect(actual.kv_cache().size() == expected.kv_cache().size(), "target cache layer count changed");
    for (std::size_t layer = 0U; layer < actual.kv_cache().size(); ++layer) {
        const auto& a = actual.kv_cache()[layer];
        const auto& e = expected.kv_cache()[layer];
        const auto first = count > a.row_capacity ? count - a.row_capacity : 0U;
        for (const auto values : {false, true}) {
            const auto x = tensor::copy_tensor_from_backend(values ? a.value : a.key);
            const auto y = tensor::copy_tensor_from_backend(values ? e.value : e.key);
            expect(x.descriptor().dtype == y.descriptor().dtype, "target cache precision differs");
            const auto row_bytes = x.bytes().size() / a.row_capacity;
            for (std::size_t position = first; position < count; ++position) {
                const auto row = x.bytes().subspan((position & a.row_mask) * row_bytes, row_bytes);
                const auto other = y.bytes().subspan((position & e.row_mask) * row_bytes, row_bytes);
                expect(std::equal(row.begin(), row.end(), other.begin()),
                    "native ordinary/verifier or replay cache bits differ in layer " + std::to_string(layer)
                        + " at position " + std::to_string(position));
            }
        }
    }
}

void test_complete_synthetic_reference_and_wrapped_batch(GemmaKVPrecision precision) {
    if (!have_metal()) return;
    NumericalFixture fixture;
    Reference reference(fixture, precision);
    auto batch = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
    auto scalar = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
    constexpr std::array<std::uint32_t, 13> tokens {2U, 7U, 1U, 19U, 3U, 11U, 5U, 13U, 17U, 29U, 23U, 0U, 31U};
    // 512-term dense sums have gamma_n below 3.1e-5. Mixed 3e-4 bounds
    // accommodate six residual/norm blocks and F32 nonlinear reductions while
    // independently detecting Q/K norm, scaling, mask and quantization errors.
    for (std::size_t i = 0U; i < 12U; ++i) {
        const auto expected = reference.append(tokens[i]);
        const auto actual = i == 0U
            ? run_gemma2_metal_resident_prefill(scalar, std::span(tokens).first(1U), 0U, 16.0F).logits
            : run_gemma2_metal_resident_decode_step(scalar,
                create_gemma2_metal_decode_schedule(scalar, static_cast<std::uint32_t>(i)), tokens[i], 16.0F).logits;
        close(read(actual), expected, "complete scalar Gemma 3 logits");
        if (i == 6U || i == 7U || i == 8U || i == 11U) check_cache(scalar, reference, i + 1U);
    }
    auto result = run_gemma2_metal_resident_prefill(batch, std::span(tokens).first(12U), 0U, 16.0F);
    close(read(result.logits), reference.logits.back(), "twelve-row batched Gemma 3 logits");
    check_cache(batch, reference, 12U);
    const auto expected = reference.append(tokens.back());
    result = run_gemma2_metal_resident_prefill(batch, std::span(tokens).last(1U), 12U, 16.0F);
    close(read(result.logits), expected, "continuation after wrapped batch");
    check_cache(batch, reference, 13U);
    // Replaying the same prefill/decode segmentation must preserve the stored
    // representation exactly, even after local slots have wrapped.
    auto replay = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
    (void)run_gemma2_metal_resident_prefill(replay, std::span(tokens).first(12U), 0U, 16.0F);
    const auto replay_last = run_gemma2_metal_resident_prefill(replay, std::span(tokens).last(1U), 12U, 16.0F);
    expect(read(replay_last.logits) == read(result.logits), "segmented target replay changed logits");
    exact_cache(replay, batch, 13U);
}

void test_width_two_prefix_publication_and_rollback(GemmaKVPrecision precision) {
    if (!have_metal()) return;
    NumericalFixture fixture;
    constexpr std::array<std::uint32_t, 9> tokens {2U, 7U, 1U, 19U, 3U, 11U, 5U, 13U, 17U};
    Reference reference(fixture, precision);
    for (const auto token : tokens) (void)reference.append(token);
    auto ordinary = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
    (void)run_gemma2_metal_resident_prefill(ordinary, std::span(tokens).first(7U), 0U, 16.0F);
    const auto ordinary_rows = run_gemma2_metal_resident_prefill(ordinary, std::span(tokens).last(2U), 7U, 16.0F);
    (void)run_gemma2_metal_resident_prefill(ordinary, std::span(tokens).first(7U), 0U, 16.0F);
    std::array<Floats, 2> ordinary_logits;
    for (std::uint32_t i = 0U; i < 2U; ++i) ordinary_logits[i] = read(run_gemma2_metal_resident_decode_step(ordinary,
        create_gemma2_metal_decode_schedule(ordinary, 7U + i), tokens[7U + i], 16.0F).logits);
    for (std::uint32_t retained = 0U; retained <= 2U; ++retained) {
        auto plan = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
        (void)run_gemma2_metal_resident_prefill(plan, std::span(tokens).first(7U), 0U, 16.0F);
        std::vector<std::vector<std::uint8_t>> before;
        for (const auto& layer : plan.kv_cache()) {
            before.push_back(tensor::copy_backend_buffer_bytes(layer.key, 0U, layer.key.descriptor().byte_count));
            before.push_back(tensor::copy_backend_buffer_bytes(layer.value, 0U, layer.value.descriptor().byte_count));
        }
        const auto verified = run_gemma2_metal_width_two_verifier(plan,
            create_gemma2_metal_verifier_schedule(plan, 7U), std::span<const std::uint32_t, 2>(tokens.data() + 7U, 2U), 16.0F);
        expect(verified.execution.memory_barrier_count == ordinary_rows.execution.memory_barrier_count,
            "width-two backups must join the existing initial barrier without per-layer waits");
        expect(verified.execution.compute_dispatch_count == ordinary_rows.execution.compute_dispatch_count
            + plan.config().block_count, "width-two rollback should add only one backup copy per layer");
        const auto logits = read(verified.logits);
        expect(std::equal(ordinary_logits[0].begin(), ordinary_logits[0].end(), logits.begin())
            && std::equal(ordinary_logits[1].begin(), ordinary_logits[1].end(), logits.begin() + 32U),
            "width-two target conditionals differ from ordinary logits");
        close(std::span(logits).first(32U), reference.logits[7U], "verifier root logits");
        close(std::span(logits).subspan(32U, 32U), reference.logits[8U], "verifier successor logits");
        (void)provisionally_commit_gemma2_metal_width_two_verification(plan, 2U);
        (void)finalize_gemma2_metal_width_two_verification(plan, retained);
        expect(!plan.has_pending_verification(), "Gemma 3 verifier leaked transaction state");
        if (retained == 0U) {
            std::size_t index = 0U;
            for (const auto& layer : plan.kv_cache()) {
                expect(tensor::copy_backend_buffer_bytes(layer.key, 0U, layer.key.descriptor().byte_count) == before[index++],
                    "full rejection did not exactly restore every physical K slot across wrap");
                expect(tensor::copy_backend_buffer_bytes(layer.value, 0U, layer.value.descriptor().byte_count) == before[index++],
                    "full rejection did not exactly restore every physical V slot across wrap");
            }
        }
        check_cache(plan, reference, 7U + retained);
        (void)run_gemma2_metal_resident_prefill(ordinary, std::span(tokens).first(7U), 0U, 16.0F);
        for (std::uint32_t i = 0U; i < retained; ++i) (void)run_gemma2_metal_resident_decode_step(ordinary,
            create_gemma2_metal_decode_schedule(ordinary, 7U + i), tokens[7U + i], 16.0F);
        exact_cache(plan, ordinary, 7U + retained);
        if (retained < 2U) {
            const auto result = run_gemma2_metal_resident_decode_step(plan,
                create_gemma2_metal_decode_schedule(plan, 7U + retained), tokens[7U + retained], 16.0F);
            close(read(result.logits), reference.logits[7U + retained], "decode after speculative suffix rollback");
        }
    }
}

void test_actual_plan_bounds_and_invalid_scales_preserve_state(GemmaKVPrecision precision) {
    if (!have_metal()) return;
    NumericalFixture fixture;
    auto target = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 16U, precision);
    const auto larger = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
    constexpr std::array<std::uint32_t, 2> tokens {2U, 7U};
    (void)run_gemma2_metal_resident_prefill(target, tokens, 0U, 16.0F);
    Reference reference(fixture, precision);
    for (const auto token : tokens) (void)reference.append(token);
    const auto reject = [&](auto operation) {
        try {
            operation();
            throw TestFailure("Gemma 3 accepted invalid actual-plan bounds or embedding scale");
        } catch (const std::invalid_argument&) {}
        expect(!target.has_pending_verification(), "invalid Gemma 3 request changed target lifecycle");
        check_cache(target, reference, 2U);
    };
    reject([&] { (void)run_gemma2_metal_resident_decode_step(target,
        create_gemma2_metal_decode_schedule(larger, 16U), 3U, 16.0F); });
    reject([&] { (void)run_gemma2_metal_width_two_verifier(target,
        create_gemma2_metal_verifier_schedule(larger, 15U), tokens, 16.0F); });
    for (const auto scale : {0.0F, -1.0F, std::numeric_limits<float>::quiet_NaN()}) {
        reject([&] { (void)run_gemma2_metal_width_two_verifier(target,
            create_gemma2_metal_verifier_schedule(target, 2U), tokens, scale); });
    }
    const auto expected = reference.append(3U);
    const auto result = run_gemma2_metal_resident_decode_step(target,
        create_gemma2_metal_decode_schedule(target, 2U), 3U, 16.0F);
    close(read(result.logits), expected, "valid decode after rejected request");
}

tensor::BackendBuffer stage_bytes(std::span<const std::uint8_t> bytes) {
    const auto descriptor = tensor::TensorDescriptor::from(tensor::Shape::from({bytes.size() / 4U}), DType::f32);
    return tensor::copy_tensor_to_backend(tensor::TensorStorage::from_bytes(descriptor, bytes), runtime::BackendKind::metal);
}

template<class T>
std::span<const std::uint8_t> bytes_of(std::span<const T> values) {
    return {reinterpret_cast<const std::uint8_t*>(values.data()), values.size_bytes()};
}

void test_scripted_chain_callbacks_stops_full_and_conditional(GemmaKVPrecision precision) {
    if (!have_metal()) return;
    NumericalFixture fixture;
    constexpr std::array<std::uint32_t, 7> prefix {2U, 7U, 1U, 19U, 3U, 11U, 5U};
    Reference reference(fixture, precision);
    Floats last;
    for (const auto token : prefix) last = reference.append(token);
    const TokenSamplerOptions options {.temperature = 0.0F, .top_k = 1U, .top_p = 1.0F, .min_p = 0.0F,
        .repetition_penalty = 4.0F, .seed = 42U, .repetition_window = 16U};
    TokenSampler reference_sampler(options);
    reference_sampler.observe_tokens(prefix);
    std::array<std::uint32_t, 4> chain {};
    for (auto& token : chain) {
        std::vector<TokenLogit> candidates;
        for (std::uint32_t id = 0U; id < last.size(); ++id) candidates.push_back({id, last[id]});
        token = reference_sampler.sample(candidates);
        reference_sampler.observe_tokens(std::span<const std::uint32_t>(&token, 1U));
        last = reference.append(token);
    }
    auto ordinary = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
    (void)run_gemma2_metal_resident_prefill(ordinary, prefix, 0U, 16.0F);
    std::array<Floats, 4> ordinary_logits;
    for (std::uint32_t row = 0U; row < chain.size(); ++row) ordinary_logits[row] = read(
        run_gemma2_metal_resident_decode_step(ordinary,
            create_gemma2_metal_decode_schedule(ordinary, 7U + row), chain[row], 16.0F).logits);
    for (const bool conditional : {false, true}) {
        for (const auto outcome : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}) {
            auto plan = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, precision);
            (void)run_gemma2_metal_resident_prefill(plan, prefix, 0U, 16.0F);
            TokenSampler sampler(options);
            sampler.observe_tokens(prefix); sampler.observe_tokens(std::span(chain).first(1U));
            auto proposed = chain;
            const auto wrong_depth = outcome == 4U || outcome == 7U ? 1U : outcome == 5U ? 2U : 0U;
            if (wrong_depth != 0U) proposed[wrong_depth] = (proposed[wrong_depth] + 1U) % 32U;
            const auto& history = sampler.history_snapshot();
            std::array<VerifierDeviceSelectionRow, 4> requests {};
            for (std::size_t row = 0U; row < requests.size(); ++row) {
                auto& request = requests[row];
                request.history_token_ids = history.token_ids;
                request.history_head = history.head; request.history_length = history.length;
                request.history_window_limit = history.window_limit;
                request.suffix_count = static_cast<std::uint32_t>(row);
                for (std::size_t i = 0U; i < row; ++i) request.suffix_token_ids[i] = proposed[i + 1U];
            }
            auto device_tokens = stage_bytes(bytes_of<std::uint32_t>(proposed));
            auto device_requests = stage_bytes(bytes_of<VerifierDeviceSelectionRow>(requests));
            ChainProposalBatch proposals {.tokens = std::span(proposed).subspan(1U), .confidence_logits = {},
                .device = {.token_ids = device_tokens, .raw_selection_requests = device_requests, .arena_epoch = 1U}};
            ChainChatVerifier verifier(plan, 3U, SpeculativeSamplingMode::exact_match, conditional);
            std::vector<ChainCommittedToken> observed;
            const auto sample = [&](TokenSamplerFork& fork, std::uint32_t depth, std::span<const TokenLogit> candidates) {
                for (const auto candidate : candidates) expect(candidate.logit == ordinary_logits[depth][candidate.token],
                    "full/conditional graph target differs from ordinary conditional");
                const auto token = fork.sample(candidates);
                return SpeculativeTokenSelection {.token = token, .accepted = token == proposed[depth + 1U]};
            };
            const auto callback = [&](const ChainCommittedToken& token) {
                observed.push_back(token);
                if (outcome == 3U && observed.size() == 2U) throw std::runtime_error("scripted consumer failure");
                return outcome == 1U && observed.size() == 2U
                    ? Gemma2MetalTokenStreamDecision::stop_generation : Gemma2MetalTokenStreamDecision::continue_generation;
            };
            const auto distinct = std::find_if(chain.begin() + 1, chain.end(), [&](auto token) { return token != chain[0]; });
            expect(outcome != 2U || distinct != chain.end(), "scripted stop fixture requires a non-root terminal token");
            const std::array<std::uint32_t, 1> stop {distinct == chain.end() ? 0U : *distinct};
            const bool can_stop = outcome == 2U;
            std::size_t retained_count = 7U;
            try {
                const auto result = verifier.run(plan, sampler, chain[0], 7U, outcome >= 6U ? 2U : 8U,
                    can_stop ? std::span<const std::uint32_t>(stop) : std::span<const std::uint32_t>(),
                    proposals, ChainSampleCallback(sample), ChainCommitCallback(callback));
                expect(outcome != 3U, "throwing speculative callback did not propagate");
                expect(result.rejected == (wrong_depth != 0U), "scripted target proposal rejection mismatch");
                expect(result.callback_stopped == (outcome == 1U), "chain callback stop was lost");
                if (can_stop) expect(result.stop_token == stop[0] && observed.back().stop_token,
                    "chain terminal token must have an internal durable receipt");
                if (outcome == 0U) expect(result.accepted_edges == 3U && result.retained_rows == 4U && result.has_next_candidates,
                    "complete chain must publish root/proposals and expose post-full distribution");
                if (outcome == 4U || outcome == 5U) expect(result.carry == chain[wrong_depth]
                    && result.retained_rows == wrong_depth && result.accepted_edges == wrong_depth - 1U,
                    "rejected proposal must retain only accepted rows and expose authoritative correction carry");
                if (outcome == 6U) expect(result.retained_rows == 2U && result.accepted_edges == 1U && !result.carry,
                    "selection budget must retain precisely the delivered accepted prefix");
                if (outcome == 7U) expect(result.retained_rows == 2U && result.endpoint.command_buffer_count != 0U && !result.carry,
                    "terminal correction must materialize its cache before ending the budget");
                check_cache(plan, reference, 7U + result.retained_rows);
                retained_count += result.retained_rows;
            } catch (const std::runtime_error& error) {
                expect(outcome == 3U && std::string_view(error.what()) == "scripted consumer failure",
                    "unexpected chain runtime failure");
                check_cache(plan, reference, observed.back().cached_token_count);
                retained_count = observed.back().cached_token_count;
            }
            expect(!plan.has_pending_verification(), "chain callback/terminal path leaked a target transaction");
            for (std::size_t i = 0U; i < observed.size(); ++i) expect(observed[i].token_id == chain[i],
                "chain callback tokens differ from independent greedy target history");
            (void)run_gemma2_metal_resident_prefill(ordinary, prefix, 0U, 16.0F);
            for (std::uint32_t row = 0U; row + 7U < retained_count; ++row)
                (void)run_gemma2_metal_resident_decode_step(ordinary,
                    create_gemma2_metal_decode_schedule(ordinary, 7U + row), chain[row], 16.0F);
            exact_cache(plan, ordinary, retained_count);
        }
    }
}

void test_f16_cache_overflow_preserves_transaction_boundaries() {
    if (!have_metal()) return;
    NumericalFixture fixture;
    // Change a real Q6 V projection's finite scale from 2^-11 to 2^11.
    // F32 still executes this transformer; narrowing its first V row must fail.
    for (std::size_t i = 0U; i < fixture.tensors.size(); ++i) {
        auto bytes = fixture.tensors[i].mutable_bytes();
        const auto& name = fixture.view.tensor_payloads[i].tensor.name;
        if (name == "token_embd.weight") {
            // The public decode schedules require an existing prefix. With
            // no model biases this zero embedding produces finite zero KV,
            // even under the large V projection used for the next token.
            std::fill_n(bytes.begin(), 210U, std::uint8_t {0U});
        }
        if (name != "blk.0.attn_v.weight") continue;
        for (std::size_t block = 0U; block < bytes.size() / 210U; ++block) {
            bytes[block * 210U + 208U] = 0U;
            bytes[block * 210U + 209U] = 0x68U;
        }
    }
    const auto snapshot = [](const Gemma2MetalModelPlan& plan) {
        std::vector<std::vector<std::uint8_t>> result;
        for (const auto& layer : plan.kv_cache()) {
            result.push_back(tensor::copy_backend_buffer_bytes(layer.key, 0U, layer.key.descriptor().byte_count));
            result.push_back(tensor::copy_backend_buffer_bytes(layer.value, 0U, layer.value.descriptor().byte_count));
        }
        return result;
    };
    const auto overflow = [](auto operation) {
        bool rejected = false;
        try { operation(); } catch (const std::overflow_error&) { rejected = true; }
        expect(rejected, "F16 target accepted a nonfinite narrowed K/V row");
    };
    constexpr std::array<std::uint32_t, 1> prefix {0U};
    const auto prefill_prefix = [&](Gemma2MetalModelPlan& plan) {
        (void)run_gemma2_metal_resident_prefill(plan, prefix, 0U, 16.0F);
        for (const auto& layer : plan.kv_cache()) {
            for (const auto* buffer : {&layer.key, &layer.value}) {
                const auto values = read(*buffer);
                expect(std::all_of(values.begin(), values.begin() + 256U,
                    [](float x) { return x == 0.0F; }),
                    "overflow fixture prefix must contain finite zero KV");
            }
        }
    };
    constexpr std::array<std::uint32_t, 2> tokens {2U, 7U};
    {
        auto plan = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, GemmaKVPrecision::f32);
        prefill_prefix(plan);
        (void)run_gemma2_metal_resident_decode_step(plan, create_gemma2_metal_decode_schedule(plan, 1U),
            tokens[0], 16.0F, std::nullopt, Gemma2MetalDecodeOutputMode::kv_only);
        const auto value = read(plan.kv_cache()[0].value);
        expect(std::all_of(value.begin() + 256U, value.begin() + 512U, [](float x) { return std::isfinite(x); })
            && std::any_of(value.begin() + 256U, value.begin() + 512U, [](float x) { return std::abs(x) > 65504.0F; }),
            "overflow fixture must distinguish finite F32 V from unrepresentable F16 V");
    }
    {
        auto plan = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, GemmaKVPrecision::f16);
        prefill_prefix(plan);
        overflow([&] { (void)run_gemma2_metal_resident_decode_step(plan,
            create_gemma2_metal_decode_schedule(plan, 1U), tokens[0], 16.0F,
            std::nullopt, Gemma2MetalDecodeOutputMode::kv_only); });
        expect(plan.verification_poisoned(), "ordinary F16 overflow must poison already published cache state");
    }
    {
        auto plan = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, GemmaKVPrecision::f16);
        prefill_prefix(plan);
        const auto before = snapshot(plan);
        overflow([&] { (void)run_gemma2_metal_width_two_verifier(plan,
            create_gemma2_metal_verifier_schedule(plan, 1U), tokens, 16.0F); });
        expect(!plan.has_pending_verification() && !plan.verification_poisoned(),
            "width-two F16 overflow must end its rollback transaction");
        expect(snapshot(plan) == before, "width-two F16 overflow did not restore every physical half cache bit");
    }
    for (const bool conditional : {false, true}) {
        auto plan = create_gemma2_metal_model_plan(fixture.model(), Gemma2OutputHeadPolicy::automatic, 32U, GemmaKVPrecision::f16);
        prefill_prefix(plan);
        const auto before = snapshot(plan);
        TokenSampler sampler({.temperature = 0.0F, .top_k = 1U});
        sampler.observe_tokens(prefix);
        sampler.observe_tokens(std::span(tokens).first(1U));
        const auto& history = sampler.history_snapshot();
        std::array<VerifierDeviceSelectionRow, 2> requests {};
        for (std::size_t row = 0U; row < requests.size(); ++row) {
            requests[row].history_token_ids = history.token_ids;
            requests[row].history_head = history.head;
            requests[row].history_length = history.length;
            requests[row].history_window_limit = history.window_limit;
            requests[row].suffix_count = static_cast<std::uint32_t>(row);
            if (row) requests[row].suffix_token_ids[0] = tokens[1];
        }
        const auto device_tokens = stage_bytes(bytes_of<std::uint32_t>(tokens));
        const auto device_requests = stage_bytes(bytes_of<VerifierDeviceSelectionRow>(requests));
        ChainProposalBatch proposals {.tokens = std::span(tokens).last(1U), .confidence_logits = {},
            .device = {.token_ids = device_tokens, .raw_selection_requests = device_requests, .arena_epoch = 1U}};
        const auto sample = [](TokenSamplerFork&, std::uint32_t, std::span<const TokenLogit>) -> SpeculativeTokenSelection {
            throw TestFailure("F16 overflow must fail before target sampling");
        };
        const auto callback = [](const ChainCommittedToken&) -> Gemma2MetalTokenStreamDecision {
            throw TestFailure("F16 overflow must fail before cache publication callback");
        };
        ChainChatVerifier verifier(plan, 1U, SpeculativeSamplingMode::exact_match, conditional);
        // A second attempt checks that the first error released its lease.
        // Every submission consumes its producer epoch, including failed work.
        for (unsigned retry = 0U; retry < 2U; ++retry) {
            proposals.device.arena_epoch = retry + 1U;
            overflow([&] { (void)verifier.run(plan, sampler, tokens[0], 1U, 4U, {}, proposals,
                ChainSampleCallback(sample), ChainCommitCallback(callback)); });
            expect(!plan.has_pending_verification() && !plan.verification_poisoned(),
                "private F16 overflow retained or poisoned a target lease");
            expect(snapshot(plan) == before, "private F16 overflow changed durable cache bytes");
        }
    }
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw TestFailure("missing target golden " + path.string());
    const auto size = file.tellg();
    expect(size >= 0, "cannot measure target golden");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    expect(bool(file), "cannot read target golden");
    return bytes;
}

Floats checkpoint_golden(const std::filesystem::path& directory, const std::string& filename,
    GemmaKVPrecision precision = GemmaKVPrecision::f32) {
    const auto& digests = precision == GemmaKVPrecision::f16 ? tests::gemma3_target_f16_golden_digests
        : tests::gemma3_target_golden_digests;
    const auto found = std::find_if(digests.begin(), digests.end(),
        [&](const auto& golden) { return golden.filename == filename; });
    expect(found != digests.end(), "target golden is not in pinned manifest");
    const auto bytes = read_file(directory / filename);
    const bool half = filename.ends_with(".f16");
    expect(bytes.size() == found->elements * (half ? 2U : sizeof(float)), "target golden shape differs from pinned manifest");
    expect(runtime::sha256(bytes) == runtime::parse_sha256_hex(found->sha256), "target golden hash differs from independent artifact");
    Floats values(found->elements);
    if (half) {
        for (std::size_t i = 0U; i < values.size(); ++i) {
            std::uint16_t bits;
            std::memcpy(&bits, bytes.data() + 2U * i, sizeof(bits));
            values[i] = tensor::f16_bits_to_f32(bits);
        }
    } else std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

// This reports numerical sensitivity through the existing production sampler
// transforms. It does not infer exact finite-grid sampling-law equality from
// close logits. Independent golden assertions below remain the acceptance gate.
void report_checkpoint_probabilities(std::string_view label, std::span<const float> actual,
    std::span<const float> expected, std::span<const std::uint32_t> prefix) {
    for (const auto penalty : {1.0F, 1.1F}) {
        const TokenSamplerOptions options {.temperature = 0.8F, .top_k = 40U,
            .top_p = 0.95F, .min_p = 0.05F, .repetition_penalty = penalty,
            .seed = 42U, .repetition_window = 64U};
        TokenSampler sampler(options);
        sampler.observe_tokens(prefix);
        const auto request = sampler.selection_request();
        const auto prepare = [&](std::span<const float> logits) {
            std::vector<TokenLogit> candidates;
            candidates.reserve(logits.size());
            for (std::uint32_t token = 0U; token < logits.size(); ++token) {
                candidates.push_back({token, logits[token]});
            }
            const auto better = [](const auto& a, const auto& b) {
                return a.logit == b.logit ? a.token < b.token : a.logit > b.logit;
            };
            std::partial_sort(candidates.begin(), candidates.begin() + request.base_candidate_count,
                candidates.end(), better);
            candidates.resize(request.base_candidate_count);
            for (std::uint32_t i = 0U; i < request.history_token_count; ++i) {
                const auto token = request.history_token_ids[i];
                candidates.push_back({token, logits[token]});
            }
            return prepare_token_distribution(candidates, options, request);
        };
        const auto p = prepare(actual), q = prepare(expected);
        std::map<std::uint32_t, std::array<double, 2>> probabilities;
        for (std::uint32_t i = 0U; i < p.count; ++i) probabilities[p.values[i].token][0] = p.values[i].probability;
        for (std::uint32_t i = 0U; i < q.count; ++i) probabilities[q.values[i].token][1] = q.values[i].probability;
        double maximum = 0.0, variation = 0.0;
        std::size_t support_difference = 0U;
        for (const auto& [token, values] : probabilities) {
            (void)token;
            const auto error = std::abs(values[0] - values[1]);
            maximum = std::max(maximum, error);
            variation += error * 0.5;
            if ((values[0] == 0.0) != (values[1] == 0.0)) ++support_difference;
        }
        float logit_error = 0.0F;
        for (std::size_t i = 0U; i < actual.size(); ++i) {
            logit_error = std::max(logit_error, std::abs(actual[i] - expected[i]));
        }
        std::cout << "[GEMMA3 PREPARED PROBABILITIES] " << label
            << " temperature=0.8 top_k=40 top_p=0.95 min_p=0.05 repetition=" << penalty
            << " max_logit_abs=" << std::scientific << std::setprecision(8) << logit_error
            << " max_probability_abs=" << maximum << " total_variation=" << variation
            << " support_difference=" << support_difference
            << " support_counts=" << p.count << '/' << q.count << std::defaultfloat << '\n';
    }
}

void test_actual_checkpoint_independent_goldens_when_requested(GemmaKVPrecision precision) {
    const bool half = precision == GemmaKVPrecision::f16;
    const auto* variable = half ? "LOCAL_GGUF_GEMMA3_TARGET_F16_GOLDENS" : "LOCAL_GGUF_GEMMA3_TARGET_GOLDENS";
    const auto* path = std::getenv(variable);
    if (path == nullptr || *path == '\0') {
        std::cout << "[SKIP GEMMA3 TARGET CHECKPOINT] set " << variable << " to the authenticated golden directory\n";
        return;
    }
    expect(have_metal(), "requested Gemma 3 target golden run requires Metal");
    const std::filesystem::path directory(path);
    expect(runtime::sha256(read_file(directory / "identity.json"))
        == runtime::parse_sha256_hex(half ? tests::gemma3_target_f16_golden_identity_sha256
            : tests::gemma3_target_golden_identity_sha256), "target golden identity changed");
    const auto mapped = model_format::read_file_backed_gguf_model("models/gemma-3-12b-it-Q4_K_M.gguf");
    expect(runtime::sha256(mapped.bytes()) == runtime::parse_sha256_hex("7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5"),
        "Gemma 3 target checkpoint does not match golden source");
    const auto model = load_engine_model_from_gguf(mapped.view());
    // Exercise the production default against the independent F16 checkpoint;
    // retain an explicit F32 run against its original numerical target.
    auto plan = create_gemma2_metal_model_plan(model, Gemma2OutputHeadPolicy::automatic, 1024U,
        half ? GemmaKVPrecision::automatic : precision);
    expect(plan.config().kv_precision == precision,
        "12B checkpoint must use the requested numerical target");
    const auto golden = [&](const std::string& filename) { return checkpoint_golden(directory, filename, precision); };
    constexpr std::array<std::uint32_t, 4> tokens {2U, 9259U, 12074U, 108U};
    const auto embedding_scale = std::sqrt(3840.0F);
    Floats scalar_last;
    std::array<Floats, 4> scalar_logits;
    for (std::size_t position = 0U; position < tokens.size(); ++position) {
        const auto result = position == 0U
            ? run_gemma2_metal_resident_prefill(plan, std::span(tokens).first(1U), 0U, embedding_scale).logits
            : run_gemma2_metal_resident_decode_step(plan,
                create_gemma2_metal_decode_schedule(plan, static_cast<std::uint32_t>(position)), tokens[position], embedding_scale).logits;
        const auto expected = golden("position-00000" + std::to_string(position) + "-logits.f32");
        const auto actual = read(result);
        scalar_logits[position] = actual;
        if (position + 1U == tokens.size()) scalar_last = actual;
        // The 15360-term down reduction has gamma_n below 9.2e-4. This
        // budget covers F32 reduction order through 48 norm/residual blocks;
        // all vocabulary entries and all layer caches are checked separately.
        close(actual, expected, "12B independent logits position " + std::to_string(position), 2.0e-3F, 3.0e-4F);
        expect(std::max_element(actual.begin(), actual.end()) - actual.begin()
            == std::max_element(expected.begin(), expected.end()) - expected.begin(), "12B independent top token mismatch");
    }
    const auto check_checkpoint_cache = [&](std::size_t count, std::string_view phase) {
        F16CacheErrorMetrics metrics;
        for (std::size_t layer = 0U; layer < 48U; ++layer) {
            const auto prefix = "layer-" + std::string(layer < 10U ? "0" : "") + std::to_string(layer);
            const auto k = read(plan.kv_cache()[layer].key), v = read(plan.kv_cache()[layer].value);
            const auto expected_k = golden(prefix + (half ? "-key.f16" : "-key.f32"));
            const auto expected_v = golden(prefix + (half ? "-value.f16" : "-value.f32"));
            const auto elements = count * 8U * 256U;
            if (half) {
                const auto key_error = close_f16_checkpoint_cache(std::span(k).first(elements), std::span(expected_k).first(elements),
                    std::string(phase) + prefix + " independent key");
                const auto value_error = close_f16_checkpoint_cache(std::span(v).first(elements), std::span(expected_v).first(elements),
                    std::string(phase) + prefix + " independent value");
                metrics.component_budget_ratio = std::max({metrics.component_budget_ratio,
                    key_error.component_budget_ratio, value_error.component_budget_ratio});
                metrics.rms_budget_ratio = std::max({metrics.rms_budget_ratio,
                    key_error.rms_budget_ratio, value_error.rms_budget_ratio});
            } else {
                close(std::span(k).first(elements), std::span(expected_k).first(elements),
                    std::string(phase) + prefix + " independent key", 5.0e-4F, 3.0e-4F);
                close(std::span(v).first(elements), std::span(expected_v).first(elements),
                    std::string(phase) + prefix + " independent value", 5.0e-4F, 3.0e-4F);
            }
        }
        if (half) {
            std::cout << "[GEMMA3 F16 CACHE PARITY] " << phase
                << "component_budget_ratio=" << std::scientific << std::setprecision(8)
                << metrics.component_budget_ratio << " rms_budget_ratio=" << metrics.rms_budget_ratio
                << " absolute_budget=5e-4 relative_budget=3e-4+2^-10" << std::defaultfloat << '\n';
        }
    };
    check_checkpoint_cache(4U, "scalar ");
    // Exercise the production H3 four-row body/head against the same scalar
    // path already checked against independent CPU goldens. Script acceptance
    // to cover all four known input tokens, independently of proposal quality.
    std::array<std::array<Floats, 2>, 48> scalar_four_kv;
    for (std::size_t layer = 0U; layer < scalar_four_kv.size(); ++layer) {
        const auto key = read(plan.kv_cache()[layer].key), value = read(plan.kv_cache()[layer].value);
        scalar_four_kv[layer][0].assign(key.begin(), key.begin() + 4U * 8U * 256U);
        scalar_four_kv[layer][1].assign(value.begin(), value.begin() + 4U * 8U * 256U);
    }
    {
        const TokenSamplerOptions options {.temperature = 0.8F, .top_k = 40U,
            .top_p = 0.95F, .min_p = 0.05F, .repetition_penalty = 1.0F,
            .seed = 42U, .repetition_window = 64U};
        TokenSampler sampler(options);
        sampler.observe_tokens(std::span(tokens).first(1U));
        std::array<VerifierDeviceSelectionRow, 4> requests {};
        const auto& history = sampler.history_snapshot();
        for (std::size_t row = 0U; row < requests.size(); ++row) {
            auto& request = requests[row];
            request.history_token_ids = history.token_ids;
            request.history_head = history.head; request.history_length = history.length;
            request.history_window_limit = history.window_limit;
            request.suffix_count = static_cast<std::uint32_t>(row);
            for (std::size_t i = 0U; i < row; ++i) request.suffix_token_ids[i] = tokens[i + 1U];
        }
        auto device_tokens = stage_bytes(bytes_of<std::uint32_t>(tokens));
        auto device_requests = stage_bytes(bytes_of<VerifierDeviceSelectionRow>(requests));
        const ChainProposalBatch proposals {.tokens = std::span(tokens).subspan(1U), .confidence_logits = {},
            .device = {.token_ids = device_tokens, .raw_selection_requests = device_requests, .arena_epoch = 1U}};
        const auto check_candidates = [&](std::uint32_t row, std::span<const TokenLogit> candidates) {
            expect(!candidates.empty(), "12B H3 must expose target candidates for every row");
            for (const auto candidate : candidates)
                expect(candidate.logit == scalar_logits[row][candidate.token],
                    "12B four-row target candidate differs from ordinary target evaluation");
        };
        const auto sample = [&](TokenSamplerFork& fork, std::uint32_t depth, std::span<const TokenLogit> candidates) {
            check_candidates(depth, candidates);
            // Advance the normal sampler checkpoint, but force the known
            // reference input sequence: this checks evaluation/publication,
            // while the lossless probability law has its own tests.
            static_cast<void>(fork.sample(candidates));
            return SpeculativeTokenSelection {.token = tokens[depth + 1U], .accepted = true};
        };
        std::size_t published = 0U;
        const auto callback = [&](const ChainCommittedToken& token) {
            expect(published < tokens.size() && token.token_id == tokens[published++],
                "12B H3 publication must preserve the scripted prefix");
            return Gemma2MetalTokenStreamDecision::continue_generation;
        };
        ChainChatVerifier verifier(plan, 3U, SpeculativeSamplingMode::exact_match, false);
        const auto result = verifier.run(plan, sampler, tokens[0], 0U, 8U, {}, proposals,
            ChainSampleCallback(sample), ChainCommitCallback(callback));
        expect(published == 4U && result.retained_rows == 4U && result.accepted_edges == 3U
            && result.has_next_candidates && !plan.has_pending_verification(),
            "12B H3 must publish all rows and finish its target transaction");
        check_candidates(3U, std::span(result.next_candidates.values).first(result.next_candidates.count));
    }
    check_checkpoint_cache(4U, "four-row verifier ");
    for (std::size_t layer = 0U; layer < scalar_four_kv.size(); ++layer) {
        const auto key = read(plan.kv_cache()[layer].key), value = read(plan.kv_cache()[layer].value);
        expect(std::equal(scalar_four_kv[layer][0].begin(), scalar_four_kv[layer][0].end(), key.begin())
            && std::equal(scalar_four_kv[layer][1].begin(), scalar_four_kv[layer][1].end(), value.begin()),
            "12B four-row verifier must preserve ordinary all-layer K/V exactly");
    }
    const auto batched = run_gemma2_metal_resident_prefill(plan, tokens, 0U, embedding_scale);
    const auto batch_last = read(batched.logits);
    const auto expected_last = golden("position-000003-logits.f32");
    close(batch_last, expected_last, "12B four-row prefill", 2.0e-3F, 3.0e-4F);
    check_checkpoint_cache(4U, "batch ");
    report_checkpoint_probabilities("packed_vs_cpu_position3", scalar_last, expected_last, tokens);
    report_checkpoint_probabilities("matrix_prefill_vs_cpu_position3", batch_last, expected_last, tokens);
    report_checkpoint_probabilities("matrix_prefill_vs_packed_position3", batch_last, scalar_last, tokens);
    // The first four rows of a 32-row prompt have the same causal input as
    // the short golden. Check their KV and row-3 head to cover M32 projections
    // with an authenticated full-transformer oracle.
    std::array<std::uint32_t, 32> long_prompt {};
    for (std::size_t i = 0U; i < long_prompt.size(); ++i) long_prompt[i] = tokens[i % tokens.size()];
    (void)run_gemma2_metal_resident_prefill(plan, long_prompt, 0U, embedding_scale);
    const auto& prompt_scratch = plan.target_execution_workspace()->scratch();
    tensor::backends::metal::MetalCommandContext prompt_head("authenticated causal prompt prefix", false, true);
    prompt_head.encode_target_head_rows(prompt_scratch.normalized_hidden, 3U,
        plan.weights().output_weight, batched.logits, 1U);
    (void)prompt_head.commit_and_wait();
    const auto causal_prefix_logits = read(batched.logits);
    close(causal_prefix_logits, expected_last, "12B first four rows of 32-row prompt", 2.0e-3F, 3.0e-4F);
    check_checkpoint_cache(4U, "32-row causal prompt prefix ");
    report_checkpoint_probabilities("causal_prompt_prefix_vs_cpu_position3", causal_prefix_logits, expected_last, tokens);
    // Truncate the logical prompt to three tokens, retaining its MMA-produced
    // KV. The fourth cached row is causally invisible and is overwritten by
    // packed scalar continuation, so the same authenticated fourth-token
    // golden checks the mixed transition without generating another oracle.
    const auto mixed = run_gemma2_metal_resident_decode_step(plan,
        create_gemma2_metal_decode_schedule(plan, 3U), tokens[3U], embedding_scale);
    const auto mixed_last = read(mixed.logits);
    close(mixed_last, expected_last, "12B packed continuation from matrix prompt", 2.0e-3F, 3.0e-4F);
    check_checkpoint_cache(4U, "matrix prompt then packed ");
    report_checkpoint_probabilities("matrix_prompt_packed_continuation_vs_cpu_position3", mixed_last, expected_last, tokens);
    // Both ordinary and speculative continuations start from the identical
    // MMA-produced prompt cache. Reconstruct it before verification so this
    // exact comparison isolates the canonical packed continuation arithmetic.
    (void)run_gemma2_metal_resident_prefill(plan, tokens, 0U, embedding_scale);
    const auto ordinary_one = read(run_gemma2_metal_resident_decode_step(plan,
        create_gemma2_metal_decode_schedule(plan, 1U), tokens[1U], embedding_scale).logits);
    const auto ordinary_two = read(run_gemma2_metal_resident_decode_step(plan,
        create_gemma2_metal_decode_schedule(plan, 2U), tokens[2U], embedding_scale).logits);
    std::array<std::array<Floats, 2>, 48> ordinary_kv;
    for (std::size_t layer = 0U; layer < ordinary_kv.size(); ++layer) {
        const auto key = read(plan.kv_cache()[layer].key), value = read(plan.kv_cache()[layer].value);
        ordinary_kv[layer][0].assign(key.begin(), key.begin() + 3U * 8U * 256U);
        ordinary_kv[layer][1].assign(value.begin(), value.begin() + 3U * 8U * 256U);
    }
    (void)run_gemma2_metal_resident_prefill(plan, tokens, 0U, embedding_scale);
    const std::array<std::uint32_t, 2> proposed {tokens[1], tokens[2]};
    const auto verified = run_gemma2_metal_width_two_verifier(plan,
        create_gemma2_metal_verifier_schedule(plan, 1U), proposed, embedding_scale);
    const auto verified_logits = read(verified.logits);
    expect(std::equal(ordinary_one.begin(), ordinary_one.end(), verified_logits.begin())
        && std::equal(ordinary_two.begin(), ordinary_two.end(), verified_logits.begin() + 262144U),
        "12B packed ordinary/verifier logits must be exact from the same matrix prompt cache");
    close(std::span(verified_logits).first(262144U), golden("position-000001-logits.f32"), "12B verification root", 2.0e-3F, 3.0e-4F);
    close(std::span(verified_logits).subspan(262144U, 262144U), golden("position-000002-logits.f32"), "12B verification successor", 2.0e-3F, 3.0e-4F);
    (void)provisionally_commit_gemma2_metal_width_two_verification(plan, 2U);
    (void)finalize_gemma2_metal_width_two_verification(plan, 1U);
    check_checkpoint_cache(2U, "accepted ");
    const auto continued = run_gemma2_metal_resident_decode_step(plan,
        create_gemma2_metal_decode_schedule(plan, 2U), tokens[2], embedding_scale);
    const auto continued_logits = read(continued.logits);
    expect(continued_logits == ordinary_two, "12B packed continuation after rollback changed canonical arithmetic");
    close(continued_logits, golden("position-000002-logits.f32"), "12B decode after verification rollback", 2.0e-3F, 3.0e-4F);
    check_checkpoint_cache(3U, "continued ");
    for (std::size_t layer = 0U; layer < ordinary_kv.size(); ++layer) {
        const auto key = read(plan.kv_cache()[layer].key), value = read(plan.kv_cache()[layer].value);
        expect(std::equal(ordinary_kv[layer][0].begin(), ordinary_kv[layer][0].end(), key.begin())
            && std::equal(ordinary_kv[layer][1].begin(), ordinary_kv[layer][1].end(), value.begin()),
            "12B packed verifier publication/rollback changed canonical all-layer KV");
    }
    std::cout << "[GEMMA3 TARGET PARITY] " << (half ? "F16 KV " : "F32 KV ") << "scalar, matrix prefill, packed mixed continuation and two-row verification logits/all48-layer K/V match authenticated CPU goldens; four-row H3 candidates/all-layer K/V and packed ordinary/verifier continuation are exact\n";
}

const tests::TestCase cases[] {
    {"Gemma3 F16 overflow poisons ordinary state and restores verifier transactions", test_f16_cache_overflow_preserves_transaction_boundaries},
    {"Gemma3 F16 complete transformer matches rounded independent equations across local wrap", [] { test_complete_synthetic_reference_and_wrapped_batch(GemmaKVPrecision::f16); }},
    {"Gemma3 F16 width-two verification preserves accepted cache prefixes and rollback bits", [] { test_width_two_prefix_publication_and_rollback(GemmaKVPrecision::f16); }},
    {"Gemma3 F16 invalid requests preserve rounded target state", [] { test_actual_plan_bounds_and_invalid_scales_preserve_state(GemmaKVPrecision::f16); }},
    {"Gemma3 F16 full and conditional verification preserves callback and stop receipts", [] { test_scripted_chain_callbacks_stops_full_and_conditional(GemmaKVPrecision::f16); }},
    {"Gemma3 complete native transformer matches independent equations across local wrap", [] { test_complete_synthetic_reference_and_wrapped_batch(GemmaKVPrecision::f32); }},
    {"Gemma3 width-two verification publishes and restores accepted cache prefixes", [] { test_width_two_prefix_publication_and_rollback(GemmaKVPrecision::f32); }},
    {"Gemma3 actual-plan bounds and invalid scales preserve target state", [] { test_actual_plan_bounds_and_invalid_scales_preserve_state(GemmaKVPrecision::f32); }},
    {"Gemma3 scripted full and conditional target verification preserves callback and stop receipts", [] { test_scripted_chain_callbacks_stops_full_and_conditional(GemmaKVPrecision::f32); }},
    {"Gemma3 12B target matches independent checkpoint goldens when requested", [] { test_actual_checkpoint_independent_goldens_when_requested(GemmaKVPrecision::f32); }},
    {"Gemma3 F16 12B target matches independent checkpoint goldens when requested", [] { test_actual_checkpoint_independent_goldens_when_requested(GemmaKVPrecision::f16); }},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gemma3_target_runtime_test_cases() { return cases; }

}  // namespace local_gguf_engine::tests
