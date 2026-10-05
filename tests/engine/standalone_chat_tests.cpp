#include "standalone_chat_tests.hpp"
#include "gemma3_target_test_support.hpp"

#include "local_gguf_engine/engine/standalone_chat.hpp"
#include "local_gguf_engine/engine/gemma3_draft.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "gemma2_reference.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "metal_test_hooks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace local_gguf_engine::tests {
std::vector<std::uint8_t> standalone_chat_target_fixture_bytes(std::uint32_t context);
namespace {
using namespace engine;
constexpr std::uint32_t hello = 5;
constexpr std::uint32_t spaced_hello = 6;
constexpr std::uint32_t draft_width = 640;
constexpr std::uint32_t draft_vocabulary = Gemma3DraftPlan::vocabulary_size;
constexpr std::uint32_t draft_layers = Gemma3DraftPlan::layer_count;

template<class F> void refuses(F&& function, const char* message) {
    bool failed = false;
    try { function(); } catch (const std::exception&) { failed = true; }
    expect(failed, message);
}

tokenizer::Tokenizer draft_tokenizer() {
    std::vector<std::string> tokens(draft_vocabulary);
    std::vector<std::int32_t> types(draft_vocabulary, 1);
    std::vector<float> scores(draft_vocabulary, -10.0F);
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = "test-draft-piece-" + std::to_string(i);
    tokens[0] = "<unk>"; types[0] = 2;
    tokens[1] = "<bos>"; types[1] = 3;
    tokens[2] = "<eos>"; types[2] = 3;
    tokens[3] = "▁"; tokens[4] = "H";
    tokens[5] = "Hello"; tokens[6] = "▁Hello";
    tokens[7] = "e"; tokens[8] = "l"; tokens[9] = "o";
    tokens[10] = "He"; tokens[11] = "Hel"; tokens[12] = "Hell";
    tokens[13] = "<start_of_turn>"; types[13] = 3;
    tokens[14] = "<end_of_turn>"; types[14] = 3;
    for (unsigned i = 3; i < 13; ++i) scores[i] = static_cast<float>(i);
    for (unsigned i = 0; i < 256; ++i) {
        std::array<char, 7> text {};
        std::snprintf(text.data(), text.size(), "<0x%02X>", i);
        tokens[1000 + i] = text.data(); types[1000 + i] = 6;
    }
    return tokenizer::load_tokenizer_from_gguf_metadata(metadata_section({
        metadata_string("general.architecture", "gemma3"), metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string("tokenizer.ggml.normalizer", "identity-no-collapse-byte-fallback"),
        metadata_bool("tokenizer.ggml.add_space_prefix", false),
        metadata_string_array("tokenizer.ggml.tokens", std::move(tokens)),
        metadata_f32_array("tokenizer.ggml.scores", std::move(scores)),
        metadata_i32_array("tokenizer.ggml.token_type", std::move(types)),
        metadata_u32("tokenizer.ggml.unknown_token_id", 0), metadata_u32("tokenizer.ggml.bos_token_id", 1),
        metadata_u32("tokenizer.ggml.eos_token_id", 2),
    }));
}

// Complete Gemma3-270M geometry, all 18 layers and native Q8 kernels. Two equal
// tied embedding/head rows yield exact half-mass bins and greedy Hello.
// Nonzero K/V exposes causal-position/transaction bugs; zero attention output
// and FFN keep this diagnostic proposal law independently predictable.
struct DraftFixture {
    std::vector<tensor::TensorStorage> storage;
    std::vector<tensor::TensorView> views;
    tokenizer::Tokenizer tokenizer = draft_tokenizer();
    DraftFixture() {
        const auto add = [&](std::uint32_t input, std::uint32_t output, unsigned kind) {
            auto value = tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
                output ? tensor::Shape::from({input, output}) : tensor::Shape::from({input}),
                output ? tensor::DType::q8_0 : tensor::DType::f32));
            std::memset(value.mutable_bytes().data(), 0, value.mutable_bytes().size());
            if (!output) {
                auto* weights = reinterpret_cast<float*>(value.mutable_bytes().data());
                std::fill_n(weights, input, 1.0F);
            } else if (kind != 0) {
                auto* bytes = reinterpret_cast<std::uint8_t*>(value.mutable_bytes().data());
                const auto scale = tensor::f32_to_f16_bits(1.0F / 64.0F);
                for (std::uint32_t row = 0; row < output; ++row) {
                    const auto block = std::size_t(row) * (input / 32U) * 34U;
                    std::memcpy(bytes + block, &scale, sizeof(scale));
                    const auto coefficient = static_cast<std::int8_t>(kind == 2
                        ? (row == hello || row == spaced_hello ? 64 : -64) : 16);
                    std::memcpy(bytes + block + 2U, &coefficient, sizeof(coefficient));
                }
            }
            storage.push_back(std::move(value));
        };
        add(draft_width, draft_vocabulary, 2); add(draft_width, 0, 0);
        for (unsigned layer = 0; layer < draft_layers; ++layer) {
            add(draft_width, 1024, 1); add(draft_width, 256, 1);
            add(draft_width, 256, 1); add(1024, draft_width, 0);
            add(draft_width, 2048, 0); add(draft_width, 2048, 0); add(2048, draft_width, 0);
            for (unsigned norm = 0; norm < 4; ++norm) add(draft_width, 0, 0);
            add(256, 0, 0); add(256, 0, 0);
        }
        for (const auto& tensor : storage) views.push_back(tensor.view());
    }
};
const DraftFixture& fixture() { static const DraftFixture value; return value; }

struct TargetFixture {
    std::vector<std::uint8_t> bytes;
    model_format::GGUFModelView view;
    EngineModel model;
    tokenizer::Tokenizer tokenizer;
    Gemma2MetalModelPlan plan;
    explicit TargetFixture(std::uint32_t context = 128)
        : bytes(standalone_chat_target_fixture_bytes(context)), view(model_format::read_gguf_model_view(bytes)),
          model(load_verified_engine_model_from_gguf(view, bytes)),
          tokenizer(tokenizer::load_tokenizer_from_gguf_metadata(view.directory.metadata)),
          plan(create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::q8_0)) {}
};

struct DirectTargetFixture : Gemma3TargetFixture {
    Gemma2MetalModelPlan plan;
    explicit DirectTargetFixture(std::uint32_t context = 16,
        GemmaKVPrecision precision = GemmaKVPrecision::automatic) : plan(prepare(context, precision)) {}
private:
    Gemma2MetalModelPlan prepare(std::uint32_t context, GemmaKVPrecision precision) {
        // Full token-ID space and native six-layer Gemma3 target semantics;
        // the independent scalar target law is a delta at Hello. The draft
        // remains the complete 18-layer 270M fixture with nonzero private K/V.
        tensors[0] = tensor::TensorStorage::allocate(tensor::TensorDescriptor::from_quantized_gguf_weight_matrix(
            tensor::Shape::from({256U, draft_vocabulary}), tensor::DType::q6_k));
        view.tensor_payloads[0].tensor.dimensions = {256U, draft_vocabulary};
        view.tensor_payloads[0].bytes = tensors[0].bytes();
        for (auto& value : tensors) {
            auto bytes = value.mutable_bytes();
            std::fill(bytes.begin(), bytes.end(), 0);
            if (value.descriptor().dtype == tensor::DType::f32)
                std::fill_n(reinterpret_cast<float*>(bytes.data()), value.descriptor().element_count, 1.0F);
        }
        auto embedding = tensors[0].mutable_bytes();
        for (std::uint32_t token = 0; token < draft_vocabulary; ++token) {
            const auto offset = std::size_t(token) * 210;
            embedding[offset] = 1;
            std::fill_n(embedding.begin() + static_cast<std::ptrdiff_t>(offset + 128), 64, 0xaaU);
            std::fill_n(embedding.begin() + static_cast<std::ptrdiff_t>(offset + 192), 16, 1U);
            embedding[offset + 209] = token == hello ? 0x40U : 0x3cU;
        }
        return create_gemma2_metal_model_plan(model(), GemmaOutputHeadPolicy::automatic, context, precision);
    }
};

void prefill(Gemma2MetalModelPlan& plan, TokenSampler& sampler, std::span<const std::uint32_t> prompt) {
    sampler.observe_tokens(prompt);
    static_cast<void>(run_gemma2_metal_resident_prefill(plan, prompt, 0,
        gemma_input_embedding_scale(plan.config().embedding_length), sampler.selection_request()));
}

void compare_target_cache(const Gemma2MetalModelPlan& actual, const Gemma2MetalModelPlan& expected,
    std::uint32_t position) {
    for (std::size_t layer = 0; layer < actual.kv_cache().size(); ++layer) {
        for (bool keys : {true, false}) {
            const auto& one = keys ? actual.kv_cache()[layer].key : actual.kv_cache()[layer].value;
            const auto& two = keys ? expected.kv_cache()[layer].key : expected.kv_cache()[layer].value;
            const auto rows = actual.kv_cache()[layer].row_capacity;
            const auto count = std::min(position, rows) * (one.descriptor().byte_count / rows);
            expect(tensor::copy_backend_buffer_bytes(one, 0, count) == tensor::copy_backend_buffer_bytes(two, 0, count),
                "standalone target cache differs from committed-prefix replay");
        }
    }
}

void compare_draft_cache(const StandaloneChatPlan& actual, std::span<const std::uint32_t> expected_tokens) {
    const auto& f = fixture();
    std::vector<std::int32_t> map(draft_vocabulary, -1); map[hello] = hello; map[spaced_hello] = spaced_hello;
    Gemma3DraftPlan expected(f.views, map, 110, 3);
    const auto& plan = actual.draft_plan_for_testing();
    expect(plan.position() == expected_tokens.size(), "standalone draft position differs from its committed ledger");
    const auto materialized = plan.materialized_position();
    expect(materialized <= expected_tokens.size() && expected_tokens.size() - materialized <= 1U,
        "standalone draft omitted more than the single legal pending emission");
    static_cast<void>(expected.append(expected_tokens.first(materialized)));
    if (materialized != expected_tokens.size())
        static_cast<void>(expected.restore_pending_token(expected_tokens.back()));
    expect(expected.position() == plan.position() && expected.materialized_position() == materialized,
        "standalone draft pending emission receipt differs from its independent replay");
    for (std::uint32_t layer = 0; layer < draft_layers; ++layer) {
        const auto first = layer % 6U != 5U && materialized > 512U ? materialized - 512U : 0U;
        const auto count = std::size_t(materialized - first) * 256U;
        std::vector<float> ak(count), av(count), ek(count), ev(count);
        plan.read_cache(layer, first, ak, av); expected.read_cache(layer, first, ek, ev);
        expect(std::memcmp(ak.data(), ek.data(), count * sizeof(float)) == 0
                && std::memcmp(av.data(), ev.data(), count * sizeof(float)) == 0,
            "standalone draft K/V differs from independently replayed materialized emissions");
        expect(std::any_of(ak.begin(), ak.end(), [](float value) { return value != 0.0F; })
                && std::any_of(av.begin(), av.end(), [](float value) { return value != 0.0F; }),
            "standalone transaction fixture lost its nonzero K/V witness");
    }
}

std::uint64_t seed_for_rejection(std::uint32_t depth) {
    for (std::uint64_t seed = 0; seed < 10000; ++seed) {
        TokenSampler sampler({.temperature=0.8F, .top_k=1, .seed=seed});
        const auto words = sampler.preview_proposal_words(3);
        bool matches = true;
        for (std::uint32_t i = 0; i <= depth; ++i)
            matches = matches && ((words.words[i] < 0x80000000U) == (i < depth));
        if (matches) return seed;
    }
    throw TestFailure("could not construct deterministic rejection seed");
}

void test_standalone_draft_sampling_resolution() {
    const TokenSamplerOptions target {.temperature=0.8F, .top_k=3, .top_p=0.9F,
        .min_p=0.01F, .repetition_penalty=1.2F, .seed=71, .repetition_window=7};
    TokenSampler sampler(target);
    const auto checkpoint = sampler.checkpoint();
    expect(resolve_standalone_draft_sampler_options(target, {}, draft_vocabulary) == target,
        "absent draft overrides changed inherited sampler options");
    StandaloneDraftSamplingOptions all {.temperature=0.0F, .top_k=262144,
        .top_p=1.0F, .min_p=0.0F, .repetition_penalty=1.0F};
    auto effective = resolve_standalone_draft_sampler_options(target, all, 262144);
    expect(effective.temperature == 0.0F && effective.top_k == 262144 && effective.top_p == 1.0F
            && effective.min_p == 0.0F && effective.repetition_penalty == 1.0F
            && effective.seed == target.seed && effective.repetition_window == target.repetition_window,
        "draft scalar resolution lost an override or changed RNG/history ownership");
    all.top_k = 0;
    expect(resolve_standalone_draft_sampler_options(target, all, draft_vocabulary).top_k == 0,
        "draft unrestricted full-vocabulary top-k was rejected");
    expect(sampler.options() == target && sampler.checkpoint().same_state(checkpoint),
        "draft resolution mutated the authoritative sampler");
    auto greedy_target = target; greedy_target.temperature = 0.0F;
    all.temperature = 1.0F;
    expect(resolve_standalone_draft_sampler_options(greedy_target, all, draft_vocabulary).temperature == 1.0F,
        "sampled draft was incorrectly coupled to greedy target transforms");

    const auto invalid = [&](StandaloneDraftSamplingOptions bad, std::uint32_t vocabulary = draft_vocabulary) {
        refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(target, bad, vocabulary)); },
            "malformed or incompatible draft sampling override was accepted");
    };
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    for (auto value : {-1.0F, nan, inf, -inf}) invalid({.temperature=value});
    for (auto value : {0.0F, -1.0F, 1.01F, nan, inf}) invalid({.top_p=value});
    for (auto value : {-0.01F, 1.01F, nan, inf}) invalid({.min_p=value});
    for (auto value : {0.0F, -1.0F, nan, inf}) invalid({.repetition_penalty=value});
    invalid({.top_k=(draft_vocabulary + 1U)}); invalid({.top_k=std::numeric_limits<std::uint32_t>::max()});
    invalid({}, 0); invalid({}, 2);
    all.top_p = std::numeric_limits<float>::denorm_min(); all.min_p = 1.0F;
    expect(resolve_standalone_draft_sampler_options(target, all, draft_vocabulary).top_p == *all.top_p,
        "valid draft probability endpoints were narrowed");
    auto malformed_target = target; malformed_target.temperature = nan;
    refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(malformed_target, all, draft_vocabulary)); },
        "draft override concealed a malformed target temperature");
    malformed_target = target; malformed_target.top_k = 0;
    refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(malformed_target, all, draft_vocabulary)); },
        "draft override concealed invalid authoritative top-k");
    malformed_target = target; malformed_target.repetition_window = 65;
    refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(malformed_target, all, draft_vocabulary)); },
        "draft override concealed invalid authoritative history capacity");
}

void test_shortened_chain_schedules_fit_every_prepared_arena() {
    for (bool conditional : {false, true}) {
        for (std::uint32_t maximum = 1; maximum <= external_speculation_max_horizon; ++maximum) {
            const auto prepared = ChainChatVerifier::schedule_for_testing(maximum, maximum, conditional);
            std::uint32_t row_capacity = 0;
            for (std::uint32_t tile = 0; tile < prepared.tile_count; ++tile)
                row_capacity = std::max(row_capacity, std::uint32_t(prepared.tiles[tile].input_row_count));
            for (std::uint32_t active = 1; active <= maximum; ++active) {
                const auto schedule = ChainChatVerifier::schedule_for_testing(maximum, active, conditional);
                validate_chain_verifier_schedule(schedule, active);
                std::uint32_t rows = 0;
                for (std::uint32_t tile = 0; tile < schedule.tile_count; ++tile) {
                    expect(schedule.tiles[tile].input_row_count <= row_capacity,
                        "shortened chain requires storage beyond its prepared arena");
                    rows += schedule.tiles[tile].input_row_count;
                }
                expect(rows == active + 1, "shortened schedule replays accepted rows or omits its bonus row");
            }
        }
    }
    refuses([] { static_cast<void>(ChainChatVerifier::schedule_for_testing(3, 0, true)); },
        "zero active horizon was accepted as a verifier cycle");
    refuses([] { static_cast<void>(ChainChatVerifier::schedule_for_testing(3, 4, true)); },
        "active horizon exceeded its reusable preparation");
}

void test_standalone_draft_sampling_changes_only_proposal_law() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    std::vector<std::int32_t> map(draft_vocabulary, -1);
    map[hello] = hello; map[spaced_hello] = spaced_hello; map[7] = 7;
    Gemma3DraftPlan draft(f.views, map, 110, 3);
    static_cast<void>(draft.append(std::span(&hello, 1)));
    const TokenSamplerOptions target_options {.temperature=1.0F, .top_k=3, .top_p=1.0F,
        .min_p=0.0F, .repetition_penalty=1.0F, .seed=71, .repetition_window=0};
    TokenSampler sampler(target_options);
    const auto checkpoint = sampler.checkpoint();
    const std::array<TokenLogit, 3> candidates {{{hello, 2.0F}, {spaced_hello, 1.0F}, {7, -4.0F}}};
    const auto target_law = sampler.fork().prepare_distribution(candidates);
    std::vector<float> logits(draft_vocabulary, -20.0F);
    for (const auto& entry : candidates) logits[entry.token] = entry.logit;
    const auto check = [&](StandaloneDraftSamplingOptions overrides, std::array<double, 3> expected) {
        const auto effective = resolve_standalone_draft_sampler_options(target_options, overrides, draft_vocabulary);
        draft.sample_logits_for_testing(logits, 0, effective);
        const auto q = draft.gather(0, target_law);
        std::uint64_t mass = 0;
        for (std::uint32_t i = 0; i < q.count; ++i) {
            mass += q.target_values[i].probability_mass;
            const auto value = double(q.target_values[i].probability_mass) / double(draft_probability_mass_total);
            expect(std::abs(value - expected[i]) < 2.e-6, "resolved draft transform differs from independent q equation");
        }
        expect(mass == draft_probability_mass_total, "draft override changed full-q normalization");
        expect(sampler.options() == target_options && sampler.checkpoint().same_state(checkpoint),
            "draft probability transform changed target sampler or RNG state");
        const auto after = sampler.fork().prepare_distribution(candidates);
        expect(after.count == target_law.count && after.total_sampling_weight == target_law.total_sampling_weight,
            "draft override changed target support or normalization");
        for (std::uint32_t i = 0; i < after.count; ++i)
            expect(after.values[i].token == target_law.values[i].token
                && after.values[i].probability == target_law.values[i].probability
                && after.values[i].sampling_weight == target_law.values[i].sampling_weight,
                "draft override changed an authoritative target probability");
        return std::array<std::uint64_t, 5>{q.target_values[0].probability_mass,
            q.target_values[1].probability_mass, q.target_values[2].probability_mass,
            q.proposal.token, q.proposal_cdf_begin};
    };
    const auto softmax = [](double a, double b, double c) {
        const auto total = std::exp(a) + std::exp(b) + std::exp(c);
        return std::array<double, 3>{std::exp(a) / total, std::exp(b) / total, std::exp(c) / total};
    };
    const auto inherited = check({}, softmax(2, 1, -4));
    expect(check({.temperature=1.0F, .top_k=3, .top_p=1.0F, .min_p=0.0F,
            .repetition_penalty=1.0F}, softmax(2, 1, -4)) == inherited,
        "explicit inherited controls changed native proposal IDs or CDF masses");
    expect(check({.top_k=0}, softmax(2, 1, -4)) == inherited
            && check({.top_k=draft_vocabulary}, softmax(2, 1, -4)) == inherited,
        "draft unrestricted or larger-than-target-cap top-k changed its complete mapped law");
    check({.temperature=2.0F}, softmax(1, 0.5, -2));
    check({.top_k=1}, {1, 0, 0});
    check({.top_p=0.6F}, {1, 0, 0});
    check({.min_p=0.5F}, {1, 0, 0});
    check({.repetition_penalty=2.0F}, softmax(1, 1, -4));
    check({.temperature=0.0F, .top_k=0}, {1, 0, 0});
}

void test_standalone_draft_sampling_integration() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    for (bool conditional : {false, true}) {
        TokenSampler initial({.temperature=0.8F, .top_k=1, .seed=seed_for_rejection(0)});
        SamplerCheckpoint inherited_end;
        for (unsigned mode = 0; mode < 3; ++mode) {
            TargetFixture target;
            StandaloneExecutionOptions options {3, conditional};
            if (mode == 1) options.draft_sampling = {.temperature=initial.options().temperature,
                .top_k=static_cast<std::uint32_t>(initial.options().top_k), .top_p=initial.options().top_p,
                .min_p=initial.options().min_p, .repetition_penalty=initial.options().repetition_penalty};
            if (mode == 2) options.draft_sampling = {.temperature=0.0F, .top_k=0, .top_p=1.0F, .min_p=0.0F};
            StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, options);
            auto sampler = initial;
            const std::array<std::uint32_t, 1> prompt {hello};
            prefill(target.plan, sampler, prompt); draft.synchronize_context(prompt); sampler.observe_tokens(prompt);
            const auto callback = [](const ChainCommittedToken& token) {
                expect(token.token_id == hello, "draft-only override altered the target's delta law");
                return Gemma2MetalTokenStreamDecision::continue_generation;
            };
            const auto result = draft.run(target.plan, sampler, hello, 1, 8, {}, callback);
            expect(result.accepted_edges == (mode == 2 ? 3U : 0U) && result.rejected == (mode != 2),
                "engine did not apply draft-only transforms before target verification");
            expect(sampler.options() == initial.options() && sampler.selection_request().top_k == 1,
                "draft full-vocabulary settings leaked into the target verifier");
            if (result.carry) expect(*result.carry == hello, "draft override changed the target residual law");
            if (mode == 0) inherited_end = sampler.checkpoint();
            if (mode == 1) expect(sampler.checkpoint().same_state(inherited_end),
                "explicit inherited draft settings changed default RNG consumption");
            TargetFixture ordinary;
            TokenSampler ordinary_sampler(initial.options());
            std::vector<std::uint32_t> committed(draft.target_position(), hello);
            prefill(ordinary.plan, ordinary_sampler, committed);
            compare_target_cache(target.plan, ordinary.plan, draft.target_position());
        }
    }
}

void test_standalone_joint_callbacks_and_cache_prefixes() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    for (bool conditional : {false, true}) {
        TargetFixture target;
        StandaloneExecutionOptions options {3, conditional};
        // Keep the known all-accepted draft path while exercising both greedy
        // and stochastic target RNG receipts at every callback boundary.
        options.draft_sampling = {.temperature=0.0F, .top_k=0, .top_p=1.0F, .min_p=0.0F};
        StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, options);
        for (float temperature : {0.0F, 0.8F}) for (bool throwing : {false, true}) for (std::uint32_t depth = 0; depth <= 3; ++depth) {
            draft.reset();
            TokenSampler sampler({.temperature=temperature, .top_k=1, .seed=42, .repetition_window=2});
            const std::array<std::uint32_t, 2> prompt {hello, spaced_hello};
            prefill(target.plan, sampler, prompt); draft.synchronize_context(prompt);
            sampler.observe_tokens(std::span<const std::uint32_t>(&hello, 1));
            expect(sampler.history_snapshot().head == 1 && sampler.history_snapshot().window_limit == 2,
                "joint callback fixture did not wrap its two-token repetition history");
            std::uint32_t calls = 0;
            const auto callback = [&](const ChainCommittedToken& token) {
                expect(token.depth == calls++, "standalone callback order changed");
                if (token.depth == 0) {
                    refuses([&] { draft.reset(); }, "callback reset invalidated an active draft cycle");
                    refuses([&] { draft.synchronize_context(prompt); }, "callback rewound an active draft cycle");
                    refuses([&] { static_cast<void>(draft.checkpoint()); }, "callback snapshotted uncommitted draft state");
                    refuses([&] { draft.observe_committed(hello, draft.target_position()); },
                        "callback published an unverified token into the draft transaction");
                }
                if (token.depth == depth) {
                    if (throwing) throw std::runtime_error("injected standalone callback failure");
                    return Gemma2MetalTokenStreamDecision::stop_generation;
                }
                return Gemma2MetalTokenStreamDecision::continue_generation;
            };
            if (throwing) refuses([&] { static_cast<void>(draft.run(target.plan, sampler, hello, 2, 10, {}, callback)); },
                "standalone throwing callback was swallowed");
            else {
                const auto result = draft.run(target.plan, sampler, hello, 2, 10, {}, callback);
                expect(result.callback_stopped && result.retained_rows == depth + 1,
                    "standalone callback stop committed an unseen suffix");
                expect(!result.rejected, "callback-censored prefix was counted as a rejection");
            }
            expect(calls == depth + 1 && draft.target_position() == depth + 3 && !draft.poisoned(),
                "standalone callback prefix did not remain usable");
            std::vector<std::uint32_t> expected_draft(depth + 3, hello);
            expected_draft[1] = spaced_hello;
            compare_draft_cache(draft, expected_draft);
            TargetFixture ordinary;
            TokenSampler ordinary_sampler({.temperature=0.0F, .top_k=1, .seed=42});
            std::vector<std::uint32_t> target_prefix(depth + 3, hello); target_prefix[1] = spaced_hello;
            prefill(ordinary.plan, ordinary_sampler, target_prefix);
            compare_target_cache(target.plan, ordinary.plan, depth + 3);
        }
    }
}

void test_standalone_rejection_budget_correction_and_regeneration() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    for (bool conditional : {false, true}) {
        TargetFixture target;
        StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, {3, conditional});
        for (std::uint32_t rejection = 0; rejection < 3; ++rejection) {
            for (const auto remaining : {rejection + 2U, 8U}) {
                const auto seed = seed_for_rejection(rejection);
                SamplerCheckpoint first;
                for (unsigned replay = 0; replay < 2; ++replay) {
                    TokenSampler sampler({.temperature=0.8F, .top_k=1, .seed=seed});
                    const std::array<std::uint32_t, 1> prompt {hello};
                    prefill(target.plan, sampler, prompt);
                    draft.synchronize_context(prompt); sampler.observe_tokens(prompt);
                    const auto callback = [](const ChainCommittedToken&) { return Gemma2MetalTokenStreamDecision::continue_generation; };
                    const auto result = draft.run(target.plan, sampler, hello, 1, remaining, {}, callback);
                    expect(result.accepted_edges == rejection,
                        "native half-mass fixture did not reject at its independently selected depth");
                    expect(result.rejected, "actual residual selection was omitted from rejection accounting");
                    const bool correction_cached = remaining == rejection + 2;
                    expect(result.retained_rows == 1 + rejection + (correction_cached ? 1U : 0U),
                        "budget correction cache receipt is incorrect");
                    expect(result.carry.has_value() != correction_cached,
                        "budget endpoint confused a cached correction with pending carry");
                    if (result.carry) {
                        expect(*result.carry == hello, "rejection correction changed the target delta law");
                        draft.observe_committed(*result.carry, draft.target_position());
                    }
                    std::vector<std::uint32_t> expected_draft(rejection + 3, hello);
                    compare_draft_cache(draft, expected_draft);
                    if (replay == 0) first = sampler.checkpoint();
                    else expect(sampler.checkpoint().same_state(first),
                        "restored context/regeneration changed standalone sampling RNG");
                }
            }
        }
    }
}

void test_standalone_ordinary_sync_rebuild_and_fail_closed() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    TargetFixture target(4096);
    StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, {3, false});
    const std::array<std::uint32_t, 1> prompt {hello};
    draft.synchronize_context(prompt);
    draft.observe_committed(hello, 1);
    draft.observe_committed(hello, 1); // Shared runtime journal repeats accepted chain receipts.
    draft.observe_committed(spaced_hello, 2);
    expect(draft.target_position() == 3, "ordinary/retrieval catch-up duplicated a committed token");
    const std::array<std::uint32_t, 3> expected {hello, hello, spaced_hello};
    compare_draft_cache(draft, expected);
    refuses([&] { draft.observe_committed(hello, 9); }, "noncausal target frontier was accepted");
    draft.synchronize_context(prompt);
    compare_draft_cache(draft, std::span(expected).first(1));
    // Wrap the native 512-row local rings, then replace the context explicitly.
    // Gemma3 never silently rebases a suffix at Vicuna's old 2048-token limit.
    std::vector<std::uint32_t> long_prefix(600, hello);
    draft.synchronize_context(long_prefix);
    draft.observe_committed(hello, 600);
    expect(draft.draft_plan_for_testing().position() == 601,
        "Gemma3 changed its native positions after local-ring wrap");
    long_prefix.push_back(hello);
    auto wrapped = draft.checkpoint();
    draft.synchronize_context(prompt, true);
    compare_draft_cache(draft, std::span(expected).first(1));
    draft.restore(wrapped);
    expect(!wrapped.valid() && draft.target_position() == 601,
        "context replacement consumed or corrupted the saved native timeline");
    compare_draft_cache(draft, long_prefix);
    const std::vector<std::uint32_t> overflow(4097, hello);
    refuses([&] { draft.synchronize_context(overflow, true); },
        "context replacement accepted an overflowing target ledger");
    expect(!draft.poisoned() && draft.target_position() == 601
            && draft.draft_plan_for_testing().position() == 601,
        "failed context replacement damaged the prior native timeline");
    draft.synchronize_context(prompt, true);
    compare_draft_cache(draft, std::span(expected).first(1));
    draft.reset(); expect(draft.target_position() == 0 && draft.draft_plan_for_testing().position() == 0,
        "standalone reset retained a logical cache frontier");
    refuses([&] { StandaloneChatPlan invalid(f.views, f.tokenizer, target.tokenizer, target.plan, {0, false}); },
        "standalone accepted a zero horizon");
    refuses([&] { StandaloneChatPlan invalid(f.views, f.tokenizer, target.tokenizer, target.plan, {24, false}); },
        "standalone accepted horizon capacity overflow");
    StandaloneExecutionOptions invalid_sampling;
    invalid_sampling.draft_sampling.temperature = std::numeric_limits<float>::quiet_NaN();
    refuses([&] { StandaloneChatPlan invalid(f.views, f.tokenizer, target.tokenizer, target.plan, invalid_sampling); },
        "standalone plan accepted a nonfinite draft override");
    invalid_sampling.draft_sampling = {.top_k=(draft_vocabulary + 1U)};
    refuses([&] { StandaloneChatPlan invalid(f.views, f.tokenizer, target.tokenizer, target.plan, invalid_sampling); },
        "standalone plan accepted a draft top-k incompatible with its artifact");
    tensor::backends::metal::set_metal_device_unavailable_for_testing(true);
    try {
        refuses([&] { StandaloneChatPlan unavailable(f.views, f.tokenizer, target.tokenizer, target.plan, {3, false}); },
            "standalone silently executed without Metal");
    } catch (...) { tensor::backends::metal::set_metal_device_unavailable_for_testing(false); throw; }
    tensor::backends::metal::set_metal_device_unavailable_for_testing(false);
}

void test_standalone_complete_horizon_and_stop_endpoints() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    for (bool conditional : {false, true}) {
        for (std::uint32_t horizon : {1U, 2U, 5U, 23U}) {
            TargetFixture target;
            StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, {horizon, conditional});
            TokenSampler sampler({.temperature=0.0F, .top_k=1, .seed=17});
            const std::array<std::uint32_t, 1> prompt {hello};
            prefill(target.plan, sampler, prompt); draft.synchronize_context(prompt); sampler.observe_tokens(prompt);
            const auto callback = [](const ChainCommittedToken&) { return Gemma2MetalTokenStreamDecision::continue_generation; };
            const auto result = draft.run(target.plan, sampler, hello, 1, horizon + 2, {}, callback);
            expect(result.accepted_edges == horizon && result.retained_rows == horizon + 1
                    && result.has_next_candidates && draft.target_position() == horizon + 2,
                "standalone full or conditional verification narrowed the requested horizon");
            expect(!result.rejected, "fully accepted draft chain was counted as a rejection");
            const auto& native = draft.draft_plan_for_testing();
            expect(native.position() == horizon + 2U && native.materialized_position() == horizon + 1U,
                "fully accepted chain did not preserve exactly one pending native emission");
            auto pending = draft.checkpoint();
            draft.reset();
            draft.synchronize_context(prompt);
            draft.restore(pending);
            expect(!pending.valid() && native.position() == horizon + 2U
                    && native.materialized_position() == horizon + 1U,
                "checkpoint restoration materialized or discarded the accepted pending emission");
            const std::vector<std::uint32_t> emissions(horizon + 2U, hello);
            compare_draft_cache(draft, emissions);
        }
        for (bool residual : {false, true}) {
            TargetFixture target;
            StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, {3, conditional});
            TokenSampler sampler({.temperature=residual ? 0.8F : 0.0F, .top_k=1,
                .seed=seed_for_rejection(0)});
            const std::array<std::uint32_t, 1> prompt {spaced_hello};
            const std::array<std::uint32_t, 1> stops {hello};
            prefill(target.plan, sampler, prompt); draft.synchronize_context(prompt); sampler.observe_tokens(prompt);
            std::uint32_t visible = 0, stop_receipts = 0;
            const auto callback = [&](const ChainCommittedToken& token) {
                if (token.stop_token) ++stop_receipts;
                else ++visible;
                return Gemma2MetalTokenStreamDecision::continue_generation;
            };
            const auto result = draft.run(target.plan, sampler, spaced_hello, 1, 8, stops, callback);
            expect(result.stop_token == hello && result.retained_rows == 2 && !result.carry
                    && visible == 1 && stop_receipts == 1 && draft.target_position() == 3,
                "standalone accepted/residual stop token lost its joint K/V receipt");
            expect(result.rejected == residual, "accepted and residual stops were conflated in rejection accounting");
        }
    }
}

void test_gemma3_direct_context_and_shortened_chain_tails() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    for (bool conditional : {false, true}) {
        DirectTargetFixture target;
        StandaloneChatPlan draft(f.views, f.tokenizer, f.tokenizer, target.plan, {3, conditional});
        expect(draft.direct_ids() && draft.tokenizer_workspace_bytes_for_testing() == 0
                && draft.draft_plan_for_testing().prepared_context_capacity() == 16,
            "Gemma3 direct conditioning retained generic conversion or oversized native context storage");
        const StandaloneTliMapping mapping(f.tokenizer, f.tokenizer);
        expect(StandaloneChatPlan::estimate_owned_bytes(target.plan.config(), f.tokenizer, f.tokenizer,
                mapping, {3, conditional}) >= draft.owned_bytes(),
            "Gemma3 combined preparation underestimated standalone owned payload");
        // Preserve exact byte fragments, unknown/control IDs and native
        // boundaries, including tokens deliberately excluded from proposals.
        const std::array<std::uint32_t, 6> prompt {hello, 1032, 1072, 0, 13, hello};
        draft.synchronize_context(prompt);
        compare_draft_cache(draft, prompt);
        draft.synchronize_context(std::span(prompt).first(3));
        compare_draft_cache(draft, std::span(prompt).first(3));
        draft.synchronize_context(prompt);
        for (const auto remaining : {2U, 3U, 4U}) {
            draft.synchronize_context(prompt, true);
            TokenSampler sampler({.temperature=0.0F, .top_k=1, .seed=17});
            prefill(target.plan, sampler, prompt);
            sampler.observe_tokens(std::span(&hello, 1));
            const auto initial = sampler.checkpoint();
            expect(draft.cycle_horizon(prompt.size(), remaining) == remaining - 1
                    && sampler.checkpoint().same_state(initial),
                "actual standalone horizon ignored output budget or advanced randomness");
            std::uint32_t callbacks = 0;
            const auto callback = [&](const ChainCommittedToken& token) {
                expect(token.depth == callbacks++, "shortened chain callback order changed");
                return Gemma2MetalTokenStreamDecision::continue_generation;
            };
            const auto result = draft.run(target.plan, sampler, hello, prompt.size(), remaining, {}, callback);
            expect(result.proposal_count == remaining - 1 && result.accepted_edges == remaining - 1
                    && result.retained_rows == remaining && callbacks == remaining
                    && !result.rejected && !result.carry,
                "shortened chain omitted, overshot or misclassified its durable budget prefix");
            std::vector<std::uint32_t> committed(prompt.begin(), prompt.end());
            committed.insert(committed.end(), remaining, hello);
            compare_draft_cache(draft, committed);
            DirectTargetFixture ordinary;
            TokenSampler replay({.temperature=0.0F, .top_k=1, .seed=17});
            prefill(ordinary.plan, replay, committed);
            compare_target_cache(target.plan, ordinary.plan, static_cast<std::uint32_t>(committed.size()));
        }
        std::vector<std::uint32_t> context_tail(13, hello);
        draft.synchronize_context(context_tail, true);
        TokenSampler sampler({.temperature=0.0F, .top_k=1, .seed=31});
        prefill(target.plan, sampler, context_tail);
        sampler.observe_tokens(std::span(&hello, 1));
        expect(draft.cycle_horizon(13, 20) == 2 && draft.cycle_horizon(13, 1) == 0,
            "standalone horizon did not account for root and target context tail");
        const auto callback = [](const ChainCommittedToken&) {
            return Gemma2MetalTokenStreamDecision::continue_generation;
        };
        const auto result = draft.run(target.plan, sampler, hello, 13, 20, {}, callback);
        expect(result.proposal_count == 2 && result.accepted_edges == 2 && result.retained_rows == 3
                && draft.target_position() == 16 && draft.cycle_horizon(16, 20) == 0,
            "short context tail did not use the reusable arena through its final legal row");
        context_tail.resize(16, hello);
        compare_draft_cache(draft, context_tail);
        refuses([&] { draft.validate_committed_prefix(std::span(&hello, 1)); },
            "full paired context accepted an additional authoritative token");
        expect(!draft.poisoned(), "full-context preflight damaged the completed paired session");
    }
}

void test_gemma3_half_target_h1_preserves_draft_precision_and_callback_prefixes() {
    expect(StandaloneExecutionOptions{}.horizon == 1U && !StandaloneExecutionOptions{}.conditional_verification,
        "standalone defaults must select H1 full verification");
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    DirectTargetFixture target(16U, GemmaKVPrecision::f16), ordinary(16U, GemmaKVPrecision::f16);
    StandaloneChatPlan draft(f.views, f.tokenizer, f.tokenizer, target.plan, {});
    expect(draft.horizon() == 1U && !draft.execution_options().conditional_verification,
        "default standalone plan did not retain H1 full verification");
    const StandaloneTliMapping mapping(f.tokenizer, f.tokenizer);
    const auto half_estimate = StandaloneChatPlan::estimate_owned_bytes(target.plan.config(),
        f.tokenizer, f.tokenizer, mapping, {1U, false});
    auto full_config = target.plan.config();
    full_config.kv_precision = GemmaKVPrecision::f32;
    const auto full_estimate = StandaloneChatPlan::estimate_owned_bytes(full_config,
        f.tokenizer, f.tokenizer, mapping, {1U, false});
    // Only target verifier private/undo rows change. The 270M drafter keeps
    // its F32 cache, full vocabulary law, sampler storage and host ledgers.
    expect(half_estimate >= draft.owned_bytes()
            && full_estimate - half_estimate == 6U * (2U + 2U) * 2U * 256U * 2U,
        "paired F16 accounting changed draft storage or missed target private/undo precision");
    const std::vector<std::uint32_t> prompt(10U, hello); // Wrap the target's eight-row local rings.
    for (const bool stop_callback : {false, true}) {
        draft.reset();
        TokenSampler sampler({.temperature=0.0F, .top_k=1U, .seed=17U});
        sampler.observe_tokens(prompt);
        std::uint32_t callbacks = 0U;
        const auto callback = [&](const Gemma2MetalGeneratedTokenEvent& event) {
            expect(event.token_id == hello, "H1 half-cache target changed the independent delta law");
            ++callbacks;
            return stop_callback && callbacks == 3U ? Gemma2MetalTokenStreamDecision::stop_generation
                : Gemma2MetalTokenStreamDecision::continue_generation;
        };
        const auto generated = generate_gemma2_metal_sampled_tokens_with_plan(target.plan,
            prompt, 0U, 4U, {}, sampler, callback, {}, nullptr, nullptr, &draft, prompt);
        const auto count = stop_callback ? 3U : 4U;
        expect(generated.selected_token_ids == std::vector<std::uint32_t>(count, hello)
                && callbacks == count && generated.metrics.learned_cycle_count > 0U
                && !target.plan.has_pending_verification() && !draft.poisoned(),
            "H1 half-cache generation lost its verified callback prefix");
        TokenSampler ordinary_sampler({.temperature=0.0F, .top_k=1U, .seed=17U});
        ordinary_sampler.observe_tokens(prompt);
        const auto scalar = generate_gemma2_metal_sampled_tokens_with_plan(ordinary.plan,
            prompt, 0U, count, {}, ordinary_sampler);
        expect(scalar.selected_token_ids == generated.selected_token_ids,
            "H1 and ordinary half-cache target disagree on the independent delta law");
        compare_target_cache(target.plan, ordinary.plan, 10U + count);
        auto expected = prompt;
        expected.insert(expected.end(), count, hello);
        compare_draft_cache(draft, expected);
        auto checkpoint = draft.checkpoint();
        draft.reset();
        draft.restore(checkpoint);
        expect(!checkpoint.valid() && draft.target_position() == 10U + count,
            "paired half-cache checkpoint did not preserve target/draft positions");
        compare_draft_cache(draft, expected);
    }
}

void test_gemma3_first_token_precedes_draft_work_and_recovers_every_exit() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    DirectTargetFixture target, ordinary;
    StandaloneChatPlan draft(f.views, f.tokenizer, f.tokenizer, target.plan, {3, false});
    const std::array<std::uint32_t, 2> prompt {hello, hello};
    TokenSampler ordinary_sampler({.temperature=0.0F, .top_k=1, .seed=17});
    ordinary_sampler.observe_tokens(prompt);
    const auto ordinary_result = generate_gemma2_metal_sampled_tokens_with_plan(
        ordinary.plan, prompt, 0, 1, {}, ordinary_sampler);
    expect(ordinary_result.selected_token_ids == std::vector<std::uint32_t>{hello},
        "deferred publication fixture lost its independent first-token target law");
    enum class Exit {complete, stop_callback, throw_callback, eos, budget_one};
    for (auto exit : {Exit::complete, Exit::stop_callback, Exit::throw_callback, Exit::eos, Exit::budget_one}) {
        draft.reset();
        const auto before = draft.execution_stats();
        const std::array<std::uint32_t, 1> invalid {draft_vocabulary};
        refuses([&] { draft.validate_direct_context(prompt, invalid); },
            "deferred direct-ID validation admitted an invalid suffix");
        expect(draft.target_position() == 0 && draft.draft_plan_for_testing().position() == 0
                && draft.execution_stats().command_buffer_count == before.command_buffer_count,
            "deferred direct-ID validation mutated draft state");
        TokenSampler sampler({.temperature=0.0F, .top_k=1, .seed=17});
        sampler.observe_tokens(prompt);
        std::array<std::uint32_t, 5> committed {};
        Gemma2GenerationCommitJournal journal {.token_storage=committed};
        std::uint32_t calls = 0;
        const auto callback = [&](const Gemma2MetalGeneratedTokenEvent& event) {
            if (++calls == 1) {
                expect(journal.count == 1 && journal.cached_token_count == prompt.size() + 1
                        && journal.sampler.same_state(sampler.checkpoint())
                        && !journal.completion.completion_unknown && !target.plan.has_pending_verification(),
                    "first-token callback preceded its authoritative target cache/journal receipt");
                expect(event.token_id == hello && draft.target_position() == 0
                        && draft.draft_plan_for_testing().position() == 0
                        && draft.execution_stats().command_buffer_count == before.command_buffer_count,
                    "first-token publication waited for draft prompt synchronization or speculation");
                compare_target_cache(target.plan, ordinary.plan, 3);
                if (exit == Exit::throw_callback) throw std::runtime_error("first-token callback sentinel");
                if (exit == Exit::stop_callback) return Gemma2MetalTokenStreamDecision::stop_generation;
            }
            return Gemma2MetalTokenStreamDecision::continue_generation;
        };
        std::uint64_t tick = 0;
        const Gemma2MonotonicClock clock = [&] { return tick += 100; };
        const auto maximum = exit == Exit::budget_one ? 1U : 5U;
        const auto stops = exit == Exit::eos ? std::span<const std::uint32_t>(&hello, 1)
            : std::span<const std::uint32_t>{};
        std::optional<Gemma2MetalPlanGeneratedTokens> result;
        bool threw = false;
        try {
            result = generate_gemma2_metal_sampled_tokens_with_plan(target.plan, prompt, 0, maximum,
                stops, sampler, callback, clock, nullptr, &journal, &draft, prompt);
        } catch (const std::runtime_error& error) {
            threw = std::string_view(error.what()) == "first-token callback sentinel";
            if (!threw) throw;
        }
        const auto selected = exit == Exit::complete ? 5U : 1U;
        expect(threw == (exit == Exit::throw_callback) && journal.count == selected
                && draft.target_position() == prompt.size() + selected && !draft.poisoned(),
            "deferred first-token exit lost its delivered prefix or swallowed the callback exception");
        expect(calls == (exit == Exit::eos ? 0U : selected), "deferred first-token callback count changed");
        std::vector<std::uint32_t> expected(prompt.begin(), prompt.end());
        expected.insert(expected.end(), selected, hello);
        compare_draft_cache(draft, expected);
        if (result) {
            expect(result->metrics.standalone_prefill_ns > 0
                    && result->metrics.standalone_draft_wall_ns >= result->metrics.standalone_prefill_ns
                    && result->metrics.decode_duration_ns >= result->metrics.standalone_prefill_ns
                    && result->metrics.prompt_evaluation_duration_ns == 100,
                "deferred draft synchronization was hidden outside generation or charged as target prefill");
            const auto commands = draft.execution_stats().command_buffer_count - before.command_buffer_count;
            expect(result->metrics.standalone_draft_command_buffer_count == commands
                    && result->decode_execution.command_buffer_count >= commands + 1,
                "deferred draft GPU work was omitted from the generation execution ledger");
            expect(result->metrics.learned_cycle_count == (exit == Exit::complete ? 1U : 0U),
                "first token unexpectedly ran a speculative cycle");
        }
        if (exit == Exit::budget_one) {
            expect(draft.draft_plan_for_testing().materialized_position() == prompt.size(),
                "first ordinary token was unnecessarily forwarded through the drafter");
            auto checkpoint = draft.checkpoint();
            draft.reset(); draft.restore(checkpoint);
            expect(draft.draft_plan_for_testing().position() == 3
                    && draft.draft_plan_for_testing().materialized_position() == 2,
                "deferred native pending token was lost by checkpoint restoration");
            const std::array<std::uint32_t, 4> context {hello, hello, hello, hello};
            const auto suffix = std::span(context).last(1);
            sampler.observe_tokens(suffix);
            const auto follow_before = draft.execution_stats();
            std::uint32_t follow_calls = 0;
            const auto follow_callback = [&](const Gemma2MetalGeneratedTokenEvent&) {
                ++follow_calls;
                expect(draft.target_position() == 3 && draft.draft_plan_for_testing().materialized_position() == 2
                        && draft.execution_stats().command_buffer_count == follow_before.command_buffer_count,
                    "follow-up synchronized its draft before the first target publication");
                return Gemma2MetalTokenStreamDecision::continue_generation;
            };
            const auto follow = generate_gemma2_metal_sampled_tokens_with_plan(target.plan, suffix, 3, 1,
                {}, sampler, follow_callback, clock, nullptr, nullptr, &draft, context);
            expect(follow_calls == 1 && follow.cached_token_count == 5 && draft.target_position() == 5
                    && draft.draft_plan_for_testing().materialized_position() == 4,
                "follow-up failed to reconcile the previous pending token and new prompt");
            compare_draft_cache(draft, std::array<std::uint32_t, 5>{hello, hello, hello, hello, hello});
        }
    }
}

void test_gemma3_deferred_draft_cached_continuation_preserves_pending_prefix() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    DirectTargetFixture target, ordinary;
    StandaloneChatPlan draft(f.views, f.tokenizer, f.tokenizer, target.plan, {3, false});
    const std::array<std::uint32_t, 2> prompt {hello, spaced_hello};
    const std::array<std::uint32_t, 3> context {hello, spaced_hello, hello};
    const std::array<std::uint32_t, 4> committed_context {hello, spaced_hello, hello, hello};
    TokenSampler ordinary_sampler({.temperature=0.0F, .top_k=1, .seed=17});
    ordinary_sampler.observe_tokens(prompt);
    static_cast<void>(generate_gemma2_metal_sampled_tokens_with_plan(
        ordinary.plan, prompt, 0, 2, {}, ordinary_sampler));
    for (bool throwing_callback : {false, true}) {
        draft.reset();
        TokenSampler sampler({.temperature=0.0F, .top_k=1, .seed=17});
        sampler.observe_tokens(prompt);
        const auto first = generate_gemma2_metal_sampled_tokens_with_plan(target.plan, prompt, 0, 1,
            {}, sampler, {}, {}, nullptr, nullptr, &draft, prompt);
        expect(first.selected_token_ids == std::vector<std::uint32_t>{hello}
                && draft.target_position() == 3 && draft.draft_plan_for_testing().position() == 3
                && draft.draft_plan_for_testing().materialized_position() == 2,
            "cached continuation fixture did not retain the first selected token as pending");
        const auto before = draft.execution_stats();
        std::array<std::uint32_t, 1> committed {};
        Gemma2GenerationCommitJournal journal {.token_storage=committed};
        std::uint32_t calls = 0;
        const auto callback = [&](const Gemma2MetalGeneratedTokenEvent& event) {
            ++calls;
            expect(event.token_id == hello && journal.count == 1 && journal.cached_token_count == 4
                    && journal.sampler.same_state(sampler.checkpoint())
                    && !journal.completion.completion_unknown && !target.plan.has_pending_verification(),
                "cached continuation callback preceded authoritative target publication");
            expect(draft.target_position() == 3 && draft.draft_plan_for_testing().materialized_position() == 2
                    && draft.execution_stats().command_buffer_count == before.command_buffer_count,
                "cached continuation materialized its pending draft prefix before publication");
            compare_target_cache(target.plan, ordinary.plan, 4);
            if (throwing_callback) throw std::runtime_error("cached continuation callback sentinel");
            return Gemma2MetalTokenStreamDecision::continue_generation;
        };
        bool threw = false;
        try {
            // The caller already owns all three history tokens. Re-evaluating
            // the final cached target row creates candidates without appending
            // prompt history or changing the matching native draft ledger.
            const auto result = generate_gemma2_metal_sampled_tokens_with_plan(target.plan,
                std::span(context).last(1), 2, 1, {}, sampler, callback, {},
                nullptr, &journal, &draft, context);
            expect(result.cached_token_count == 4 && result.metrics.standalone_draft_command_buffer_count > 0,
                "cached continuation omitted its pending-prefix reconciliation work");
        } catch (const std::runtime_error& error) {
            threw = std::string_view(error.what()) == "cached continuation callback sentinel";
            if (!threw) throw;
        }
        expect(threw == throwing_callback && calls == 1 && journal.count == 1
                && draft.target_position() == 4 && draft.draft_plan_for_testing().position() == 4
                && draft.draft_plan_for_testing().materialized_position() == 4 && !draft.poisoned(),
            "cached continuation lost its committed prefix or the original callback exception");
        compare_draft_cache(draft, committed_context);
        // Both callback exits leave a usable authoritative prefix, including
        // another exact-context continuation after the exceptional exit.
        const auto next = generate_gemma2_metal_sampled_tokens_with_plan(target.plan,
            std::span(committed_context).last(1), 3, 1, {}, sampler, {}, {},
            nullptr, nullptr, &draft, committed_context);
        expect(next.cached_token_count == 5 && draft.target_position() == 5
                && draft.draft_plan_for_testing().materialized_position() == 4 && !draft.poisoned(),
            "reconciled cached continuation could not be reused after callback completion");
        compare_draft_cache(draft, std::array<std::uint32_t, 5>{hello, spaced_hello, hello, hello, hello});
    }
}

void test_gemma3_direct_draft_inherits_bounded_repetition_history() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    for (bool conditional : {false, true}) {
        DirectTargetFixture target;
        StandaloneChatPlan draft(f.views, f.tokenizer, f.tokenizer, target.plan, {3, conditional});
        const std::array<std::uint32_t, 2> prompt {hello, spaced_hello};
        for (const std::size_t window : {0U, 1U, 2U}) {
            draft.synchronize_context(prompt, true);
            TokenSampler sampler({.temperature=0.0F, .top_k=1, .repetition_penalty=1.1F,
                .seed=23, .repetition_window=window});
            prefill(target.plan, sampler, prompt);
            sampler.observe_tokens(std::span(&hello, 1));
            const auto callback = [](const ChainCommittedToken& token) {
                expect(token.token_id == hello, "bounded draft repetition changed the target's delta law");
                return Gemma2MetalTokenStreamDecision::continue_generation;
            };
            // The draft's two positive head rows are identical. Window one
            // penalizes only the root Hello, so it proposes spaced-Hello and
            // rejects against p. Window zero applies no penalty; window two
            // penalizes both equal rows, preserving the Hello tie winner.
            const auto result = draft.run(target.plan, sampler, hello, 2, 2, {}, callback);
            expect(result.proposal_count == 1 && result.rejected == (window == 1)
                    && result.accepted_edges == (window == 1 ? 0U : 1U)
                    && result.retained_rows == 2 && !result.carry,
                "direct-ID draft repetition ignored its inherited bounded window");
            const std::array<std::uint32_t, 4> committed {hello, spaced_hello, hello, hello};
            compare_draft_cache(draft, committed);
        }
    }
    TargetFixture legacy_target;
    StandaloneChatPlan legacy(f.views, f.tokenizer, legacy_target.tokenizer, legacy_target.plan, {3, false});
    expect(!legacy.direct_ids(), "cross-generation TLI incorrectly opted into native window alignment");
}

void test_standalone_checkpoint_preserves_append_interpretation_and_lifetime() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    TargetFixture target;
    StandaloneChatCheckpoint outlives_owner;
    {
        StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, {3, false});
        const std::array<std::uint32_t, 1> prompt {hello};
        draft.synchronize_context(prompt);
        draft.observe_committed(hello, 1);
        draft.observe_committed(spaced_hello, 2);
        const std::array<std::uint32_t, 3> expected {hello, hello, spaced_hello};
        auto saved = draft.checkpoint();
        expect(saved.valid(), "standalone snapshot did not issue a live ticket");
        refuses([&] { static_cast<void>(draft.checkpoint()); }, "single-slot standalone snapshot was overwritten");
        auto moved = std::move(saved);
        expect(!saved.valid() && moved.valid(), "standalone snapshot move duplicated ownership");
        draft.reset(); draft.synchronize_context(prompt);
        draft.observe_committed(spaced_hello, 1);
        draft.restore(moved);
        expect(!moved.valid() && draft.target_position() == 3,
            "standalone snapshot restore did not consume its ticket or restore target position");
        compare_draft_cache(draft, expected);
        refuses([&] { draft.restore(moved); }, "consumed standalone snapshot restored again");
        StandaloneChatCheckpoint empty;
        refuses([&] { draft.restore(empty); }, "default standalone snapshot was accepted");
        saved = draft.checkpoint();
        {
            StandaloneChatPlan other(f.views, f.tokenizer, target.tokenizer, target.plan, {3, false});
            refuses([&] { other.restore(saved); }, "standalone checkpoint crossed model-plan ownership");
        }
        expect(saved.valid(), "wrong-owner restore consumed the real checkpoint");
        draft.discard(saved);
        expect(!saved.valid(), "standalone discard retained a live snapshot");
        {
            auto abandoned = draft.checkpoint();
            expect(abandoned.valid(), "abandoned snapshot was not created");
        }
        outlives_owner = draft.checkpoint();
        expect(outlives_owner.valid(), "abandoning a snapshot leaked its fixed-capacity slot");
    }
    expect(!outlives_owner.valid(), "destroyed draft plan left a valid snapshot ticket");
}

void test_gemma3_checkpoint_restores_wrapped_context_after_conversion_failure() {
    const std::filesystem::path path("models/gemma-3-270m-it-Q8_0.gguf");
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal) || !std::filesystem::exists(path)) return;
    TargetFixture target(1024);
    auto source = std::make_shared<StandaloneDraftArtifact>(path);
    StandaloneChatPlan draft(source, target.tokenizer, target.plan, {3, false});
    std::vector<std::uint32_t> prompt(600, spaced_hello);
    draft.synchronize_context(prompt);
    for (std::uint32_t i = 600; i < 604; ++i) draft.observe_committed(hello, i);
    const auto* native = draft.gemma3_draft_plan_for_testing();
    expect(native && native->position() > 512 && native->position() == native->materialized_position(),
        "Gemma3 checkpoint test did not wrap its real sliding cache");
    const auto position = native->position();
    std::array<std::vector<float>, 18> keys, values;
    for (std::uint32_t layer = 0; layer < 18; ++layer) {
        const auto first = layer % 6 == 5 ? 0U : position - 512U;
        keys[layer].resize(std::size_t(position - first) * 256);
        values[layer].resize(keys[layer].size());
        native->read_cache(layer, first, keys[layer], values[layer]);
    }
    auto checkpoint = draft.checkpoint();
    const std::array<std::uint32_t, 1> invalid {UINT32_MAX};
    refuses([&] { draft.validate_committed_prefix(invalid); }, "malformed Gemma3 commit passed preflight");
    refuses([&] { draft.validate_committed_prefix(prompt); }, "Gemma3 commit ledger overflow passed preflight");
    const std::array<std::uint32_t, 2> pair {hello, spaced_hello};
    draft.validate_committed_prefix(pair);
    expect(!draft.poisoned() && draft.target_position() == 604 && native->position() == position,
        "Gemma3 commit preflight mutated or poisoned a published frontier");
    refuses([&] { draft.synchronize_context(invalid); }, "malformed Gemma3 context did not fail closed");
    refuses([&] { draft.synchronize_context(invalid, true); }, "malformed rebased Gemma3 context did not fail closed");
    expect(!draft.poisoned() && draft.target_position() == 604 && native->position() == position,
        "failed Gemma3 context replacement destroyed the old emission ledger");
    draft.restore(checkpoint);
    expect(!checkpoint.valid() && !draft.poisoned() && draft.target_position() == 604
            && native->position() == position,
        "recoverable Gemma3 conversion failure lost the saved native emission ledger");
    for (std::uint32_t layer = 0; layer < 18; ++layer) {
        const auto first = layer % 6 == 5 ? 0U : position - 512U;
        std::vector<float> k(keys[layer].size()), v(k.size());
        native->read_cache(layer, first, k, v);
        expect(std::memcmp(k.data(), keys[layer].data(), k.size() * sizeof(float)) == 0
                && std::memcmp(v.data(), values[layer].data(), v.size() * sizeof(float)) == 0,
            "Gemma3 wrapped checkpoint changed real K/V bits");
    }
    // A historical published prefix is no longer rewindable by counter alone.
    // Reusing its saved emission receipt must reconstruct the overwritten rows.
    draft.synchronize_context(prompt);
    expect(draft.target_position() == prompt.size() && native->position() < position
            && !draft.poisoned(), "Gemma3 historical-prefix replay failed after ring overwrite");
}
} // namespace

std::span<const TestCase> standalone_chat_test_cases() {
    static constexpr TestCase cases[] {
        {"shortened chain schedules fit every prepared full and conditional arena", test_shortened_chain_schedules_fit_every_prepared_arena},
        {"standalone draft sampling overrides resolve without changing target authority", test_standalone_draft_sampling_resolution},
        {"standalone draft scalar overrides preserve independent full-q equations and target law", test_standalone_draft_sampling_changes_only_proposal_law},
        {"standalone draft-only sampling changes proposals with default-equivalent target replay", test_standalone_draft_sampling_integration},
        {"Gemma3 wrapped real-checkpoint context restores after conversion failure", test_gemma3_checkpoint_restores_wrapped_context_after_conversion_failure},
        {"standalone joint target and draft caches retain every callback prefix", test_standalone_joint_callbacks_and_cache_prefixes},
        {"standalone native rejection honors budget correction and regeneration", test_standalone_rejection_budget_correction_and_regeneration},
        {"standalone ordinary synchronization context rebuild and unavailable device fail closed", test_standalone_ordinary_sync_rebuild_and_fail_closed},
        {"standalone complete horizons and accepted residual stop endpoints", test_standalone_complete_horizon_and_stop_endpoints},
        {"Gemma3 direct context and shortened chain tails preserve native cache receipts", test_gemma3_direct_context_and_shortened_chain_tails},
        {"Gemma3 F16 target H1 keeps F32 draft receipts and callback prefixes", test_gemma3_half_target_h1_preserves_draft_precision_and_callback_prefixes},
        {"Gemma3 first target publication precedes draft work and preserves every exit", test_gemma3_first_token_precedes_draft_work_and_recovers_every_exit},
        {"Gemma3 deferred draft cached continuation preserves pending prefix across callbacks", test_gemma3_deferred_draft_cached_continuation_preserves_pending_prefix},
        {"Gemma3 direct draft inherits bounded repetition history without changing target authority", test_gemma3_direct_draft_inherits_bounded_repetition_history},
        {"standalone checkpoint preserves append interpretation and single-slot lifetime", test_standalone_checkpoint_preserves_append_interpretation_and_lifetime},
    };
    return cases;
}
} // namespace local_gguf_engine::tests
