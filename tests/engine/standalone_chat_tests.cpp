#include "standalone_chat_tests.hpp"

#include "local_gguf_engine/engine/standalone_chat.hpp"
#include "local_gguf_engine/engine/gemma3_draft.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "../../src/tensor/backends/metal/metal_test_hooks.hpp"

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

template<class F> void refuses(F&& function, const char* message) {
    bool failed = false;
    try { function(); } catch (const std::exception&) { failed = true; }
    expect(failed, message);
}

tokenizer::Tokenizer draft_tokenizer() {
    std::vector<std::string> tokens(32000);
    std::vector<std::int32_t> types(32000, 1);
    std::vector<float> scores(32000, -10.0F);
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = "test-draft-piece-" + std::to_string(i);
    tokens[0] = "<unk>"; types[0] = 2;
    tokens[1] = "<s>"; types[1] = 3;
    tokens[2] = "</s>"; types[2] = 3;
    tokens[3] = "▁"; tokens[4] = "H";
    tokens[5] = "Hello"; tokens[6] = "▁Hello";
    tokens[7] = "e"; tokens[8] = "l"; tokens[9] = "o";
    tokens[10] = "He"; tokens[11] = "Hel"; tokens[12] = "Hell";
    for (unsigned i = 3; i < 13; ++i) scores[i] = static_cast<float>(i);
    for (unsigned i = 0; i < 256; ++i) {
        std::array<char, 7> text {};
        std::snprintf(text.data(), text.size(), "<0x%02X>", i);
        tokens[1000 + i] = text.data(); types[1000 + i] = 6;
    }
    return tokenizer::load_tokenizer_from_gguf_metadata(metadata_section({
        metadata_string("general.architecture", "llama"), metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string("tokenizer.ggml.normalizer", "identity-no-collapse-byte-fallback"),
        metadata_bool("tokenizer.ggml.add_space_prefix", true),
        metadata_string_array("tokenizer.ggml.tokens", std::move(tokens)),
        metadata_f32_array("tokenizer.ggml.scores", std::move(scores)),
        metadata_i32_array("tokenizer.ggml.token_type", std::move(types)),
        metadata_u32("tokenizer.ggml.unknown_token_id", 0), metadata_u32("tokenizer.ggml.bos_token_id", 1),
        metadata_u32("tokenizer.ggml.eos_token_id", 2),
    }));
}

// Full 68M geometry and native kernels, not a reduced production architecture.
// Two equal head rows yield exact half-mass bins when sampled and Hello when
// greedy. Nonzero K/V exposes causal-position/truncation bugs; zero attention
// output keeps the proposal law independently predictable at every depth.
struct DraftFixture {
    std::vector<tensor::TensorStorage> storage;
    std::vector<tensor::TensorView> views;
    tokenizer::Tokenizer tokenizer = draft_tokenizer();
    DraftFixture() {
        const auto add = [&](std::uint32_t input, std::uint32_t output, unsigned kind) {
            auto value = tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
                output == 1 ? tensor::Shape::from({input}) : tensor::Shape::from({input, output}), tensor::DType::f16));
            auto* bits = reinterpret_cast<std::uint16_t*>(value.mutable_bytes().data());
            std::fill_n(bits, value.descriptor().element_count, std::uint16_t {0});
            if (kind == 1) for (std::uint32_t i = 0; i < output; ++i) bits[std::size_t(i) * input] = tensor::f32_to_f16_bits(1.0F);
            if (kind == 2) for (std::uint32_t i = 0; i < output; ++i)
                bits[std::size_t(i) * input] = tensor::f32_to_f16_bits(i == hello || i == spaced_hello ? 1.0F : -1.0F);
            if (kind == 3) std::fill_n(bits, input, tensor::f32_to_f16_bits(1.0F));
            if (kind == 4) for (std::uint32_t i = 0; i < output; ++i)
                bits[std::size_t(i) * input + (i % input)] = tensor::f32_to_f16_bits(0.25F);
            storage.push_back(std::move(value));
        };
        add(768, 32000, 1); add(768, 32000, 2); add(768, 1, 3);
        for (unsigned layer = 0; layer < 2; ++layer) {
            add(768, 768, 4); add(768, 768, 4); add(768, 768, 4); add(768, 768, 0);
            add(768, 3072, 0); add(768, 3072, 0); add(3072, 768, 0);
            add(768, 1, 3); add(768, 1, 3);
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
          plan(create_gemma2_metal_model_plan(model, Gemma2OutputHeadPolicy::q8_0)) {}
};

void prefill(Gemma2MetalModelPlan& plan, TokenSampler& sampler, std::span<const std::uint32_t> prompt) {
    sampler.observe_tokens(prompt);
    static_cast<void>(run_gemma2_metal_resident_prefill(plan, prompt, 0,
        gemma2_input_embedding_scale(plan.config().embedding_length), sampler.selection_request()));
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
    std::vector<std::int32_t> map(32000, -1); map[hello] = hello; map[spaced_hello] = spaced_hello;
    VicunaDraftPlan expected(f.views, map, 110, 3);
    static_cast<void>(expected.append(expected_tokens));
    const auto& plan = actual.draft_plan_for_testing();
    expect(plan.position() == expected_tokens.size(), "standalone draft position differs from its committed ledger");
    const auto count = expected_tokens.size() * 768;
    std::vector<std::uint16_t> ak(count), av(count), ek(count), ev(count);
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        plan.read_cache(layer, ak, av); expected.read_cache(layer, ek, ev);
        expect(ak == ek && av == ev, "standalone draft K/V differs from independently replayed append policy");
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
    expect(resolve_standalone_draft_sampler_options(target, {}, 32000) == target,
        "absent draft overrides changed inherited sampler options");
    StandaloneDraftSamplingOptions all {.temperature=0.0F, .top_k=262144,
        .top_p=1.0F, .min_p=0.0F, .repetition_penalty=1.0F};
    auto effective = resolve_standalone_draft_sampler_options(target, all, 262144);
    expect(effective.temperature == 0.0F && effective.top_k == 262144 && effective.top_p == 1.0F
            && effective.min_p == 0.0F && effective.repetition_penalty == 1.0F
            && effective.seed == target.seed && effective.repetition_window == target.repetition_window,
        "draft scalar resolution lost an override or changed RNG/history ownership");
    all.top_k = 0;
    expect(resolve_standalone_draft_sampler_options(target, all, 32000).top_k == 0,
        "draft unrestricted full-vocabulary top-k was rejected");
    expect(sampler.options() == target && sampler.checkpoint().same_state(checkpoint),
        "draft resolution mutated the authoritative sampler");
    auto greedy_target = target; greedy_target.temperature = 0.0F;
    all.temperature = 1.0F;
    expect(resolve_standalone_draft_sampler_options(greedy_target, all, 32000).temperature == 1.0F,
        "sampled draft was incorrectly coupled to greedy target transforms");

    const auto invalid = [&](StandaloneDraftSamplingOptions bad, std::uint32_t vocabulary = 32000) {
        refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(target, bad, vocabulary)); },
            "malformed or incompatible draft sampling override was accepted");
    };
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    for (auto value : {-1.0F, nan, inf, -inf}) invalid({.temperature=value});
    for (auto value : {0.0F, -1.0F, 1.01F, nan, inf}) invalid({.top_p=value});
    for (auto value : {-0.01F, 1.01F, nan, inf}) invalid({.min_p=value});
    for (auto value : {0.0F, -1.0F, nan, inf}) invalid({.repetition_penalty=value});
    invalid({.top_k=32001}); invalid({.top_k=std::numeric_limits<std::uint32_t>::max()});
    invalid({}, 0); invalid({}, 2);
    all.top_p = std::numeric_limits<float>::denorm_min(); all.min_p = 1.0F;
    expect(resolve_standalone_draft_sampler_options(target, all, 32000).top_p == *all.top_p,
        "valid draft probability endpoints were narrowed");
    auto malformed_target = target; malformed_target.temperature = nan;
    refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(malformed_target, all, 32000)); },
        "draft override concealed a malformed target temperature");
    malformed_target = target; malformed_target.top_k = 0;
    refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(malformed_target, all, 32000)); },
        "draft override concealed invalid authoritative top-k");
    malformed_target = target; malformed_target.repetition_window = 65;
    refuses([&] { static_cast<void>(resolve_standalone_draft_sampler_options(malformed_target, all, 32000)); },
        "draft override concealed invalid authoritative history capacity");
}

void test_standalone_draft_sampling_changes_only_proposal_law() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    const auto& f = fixture();
    std::vector<std::int32_t> map(32000, -1);
    map[hello] = hello; map[spaced_hello] = spaced_hello; map[7] = 7;
    VicunaDraftPlan draft(f.views, map, 110, 3);
    static_cast<void>(draft.append(std::span(&hello, 1)));
    const TokenSamplerOptions target_options {.temperature=1.0F, .top_k=3, .top_p=1.0F,
        .min_p=0.0F, .repetition_penalty=1.0F, .seed=71, .repetition_window=0};
    TokenSampler sampler(target_options);
    const auto checkpoint = sampler.checkpoint();
    const std::array<TokenLogit, 3> candidates {{{hello, 2.0F}, {spaced_hello, 1.0F}, {7, -4.0F}}};
    const auto target_law = sampler.fork().prepare_distribution(candidates);
    std::vector<float> logits(32000, -20.0F);
    for (const auto& entry : candidates) logits[entry.token] = entry.logit;
    const auto check = [&](StandaloneDraftSamplingOptions overrides, std::array<double, 3> expected) {
        const auto effective = resolve_standalone_draft_sampler_options(target_options, overrides, 32000);
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
            && check({.top_k=32000}, softmax(2, 1, -4)) == inherited,
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
        StandaloneChatPlan draft(f.views, f.tokenizer, target.tokenizer, target.plan, {3, conditional});
        for (bool throwing : {false, true}) for (std::uint32_t depth = 0; depth <= 3; ++depth) {
            draft.reset();
            TokenSampler sampler({.temperature=0.0F, .top_k=1, .seed=42, .repetition_window=2});
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
            expected_draft[0] = expected_draft[1] = spaced_hello;
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
                    std::vector<std::uint32_t> expected_draft(rejection + 3, hello); expected_draft[0] = spaced_hello;
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
    const std::array<std::uint32_t, 3> expected {spaced_hello, hello, spaced_hello};
    compare_draft_cache(draft, expected);
    refuses([&] { draft.observe_committed(hello, 9); }, "noncausal target frontier was accepted");
    draft.synchronize_context(prompt);
    compare_draft_cache(draft, std::span(expected).first(1));
    // Fill the legal 2045-token draft prefix with batched prefill, then cross
    // its reserved-H3 boundary through the ordinary single-token commit path.
    std::vector<std::uint32_t> long_prefix(2045, hello);
    draft.synchronize_context(long_prefix);
    expect(draft.draft_plan_for_testing().position() == 2045,
        "draft reserved-horizon boundary was not reached exactly");
    draft.observe_committed(hello, 2045);
    expect(draft.rebuild_duration_ns() > 0 && draft.draft_plan_for_testing().position() < 2048,
        "draft context limit did not rebuild an explicitly rebased suffix");
    std::vector<std::uint32_t> suffix(draft.draft_plan_for_testing().position(), hello);
    compare_draft_cache(draft, suffix);
    draft.synchronize_context(prompt);
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
    invalid_sampling.draft_sampling = {.top_k=32001};
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
        const std::array<std::uint32_t, 3> expected {spaced_hello, hello, spaced_hello};
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
        {"standalone draft sampling overrides resolve without changing target authority", test_standalone_draft_sampling_resolution},
        {"standalone draft scalar overrides preserve independent full-q equations and target law", test_standalone_draft_sampling_changes_only_proposal_law},
        {"standalone draft-only sampling changes proposals with default-equivalent target replay", test_standalone_draft_sampling_integration},
        {"Gemma3 wrapped real-checkpoint context restores after conversion failure", test_gemma3_checkpoint_restores_wrapped_context_after_conversion_failure},
        {"standalone joint target and draft caches retain every callback prefix", test_standalone_joint_callbacks_and_cache_prefixes},
        {"standalone native rejection honors budget correction and regeneration", test_standalone_rejection_budget_correction_and_regeneration},
        {"standalone ordinary synchronization context rebuild and unavailable device fail closed", test_standalone_ordinary_sync_rebuild_and_fail_closed},
        {"standalone complete horizons and accepted residual stop endpoints", test_standalone_complete_horizon_and_stop_endpoints},
        {"standalone checkpoint preserves append interpretation and single-slot lifetime", test_standalone_checkpoint_preserves_append_interpretation_and_lifetime},
    };
    return cases;
}
} // namespace local_gguf_engine::tests
