#include "speculation_artifact_tests.hpp"

#include "local_gguf_engine/engine/speculation_trace.hpp"
#include "local_gguf_engine/engine/speculative_artifact.hpp"
#include "local_gguf_engine/frontends/cli/speculation_trace.hpp"
#include "local_gguf_engine/runtime/sha256.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using namespace local_gguf_engine::engine;
using local_gguf_engine::runtime::Sha256Digest;
using local_gguf_engine::runtime::sha256;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::TestCase;

[[nodiscard]] Sha256Digest digest(std::uint8_t seed) {
    Sha256Digest result;
    for (std::size_t index = 0U; index < result.bytes.size(); ++index) {
        result.bytes[index] = static_cast<std::uint8_t>(seed + index);
    }
    return result;
}

[[nodiscard]] EngineModelPayloadReceipt model_receipt() {
    return EngineModelPayloadReceipt {
        .version = engine_model_payload_receipt_version,
        .whole_gguf_byte_count = 4096U,
        .source_tensor_count = 3U,
        .whole_gguf = digest(1U),
        .canonical_consumed_metadata = digest(2U),
        .source_tensors = digest(3U),
    };
}

[[nodiscard]] ResidentTargetPayloadReceipt resident_receipt() {
    return ResidentTargetPayloadReceipt {
        .version = 1U,
        .allocation_count = 3U,
        .byte_count = 2048U,
        .bytes = digest(4U),
    };
}

[[nodiscard]] SpeculativeArtifactDescriptor artifact_descriptor() {
    SpeculativeArtifactDescriptor descriptor;
    descriptor.artifact_id = 17U;
    descriptor.target.model = model_receipt();
    descriptor.target.resident = resident_receipt();
    descriptor.target.config.context_length = 32U;
    descriptor.target.config.embedding_length = 2U;
    descriptor.target.config.block_count = 2U;
    descriptor.target.config.vocab_size = 16U;
    descriptor.target.output_head_policy = 2U;
    descriptor.feature_bank_count = 1U;
    auto& bank = descriptor.feature_banks[0];
    bank.bank_id = 1U;
    bank.tap_count = 1U;
    bank.taps[0] = {
        .stage = FeatureTapStage::layer_residual,
        .layer_index = 1U,
    };
    bank.child_count = 1U;
    bank.children[0] = {
        .child_id = 7U,
        .output_width = 2U,
        .weight_dtype = DType::f16,
        .schedule = FeatureProjectionSchedule::eager,
        .payload_offset = 0U,
        .payload_byte_count = 8U,
    };
    descriptor.draft_cache = {
        .layer_count = 2U,
        .stream_count = 1U,
        .key_width = 4U,
        .value_width = 4U,
        .maximum_depth = 23U,
        .maximum_active_width = 64U,
        .linear_row_capacity = 23U,
        .tree_row_capacity = 1'025U,
    };
    return descriptor;
}

void test_sha256_matches_published_abc_vector() {
    constexpr std::string_view input = "abc";
    const auto bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(input.data()), input.size()
    );
    expect(
        sha256(bytes).hex()
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc vector differs"
    );
}

void test_fixture_artifact_roundtrip_authenticates_every_payload_slice() {
    const std::array<std::uint8_t, 8U> payload {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
    const auto encoded_a = encode_fixture_speculative_artifact(
        artifact_descriptor(), payload
    );
    const auto encoded_b = encode_fixture_speculative_artifact(
        artifact_descriptor(), payload
    );
    expect(encoded_a == encoded_b, "fixture artifact writer is not deterministic");
    const auto loaded = load_speculative_artifact(encoded_a);
    expect(loaded.descriptor().artifact_id == 17U, "artifact id changed");
    expect(std::equal(
        loaded.payload().begin(), loaded.payload().end(), payload.begin()
    ), "artifact payload changed");
    validate_speculative_artifact_target(
        loaded.descriptor(), model_receipt(), resident_receipt(),
        loaded.descriptor().target.config, 2U
    );

    auto corrupt = encoded_a;
    corrupt.back() ^= 1U;
    auto rejected = false;
    try {
        static_cast<void>(load_speculative_artifact(corrupt));
    } catch (const SpeculativeArtifactError& error) {
        rejected = error.code() == SpeculativeArtifactErrorCode::digest_mismatch;
    }
    expect(rejected, "artifact payload corruption was not rejected");
}

void test_fixture_artifact_authenticates_phase4_scoring_tensors() {
    auto descriptor = artifact_descriptor();
    auto payload = std::vector<std::uint8_t> {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
    auto& draft = descriptor.draft_vocabulary;
    draft.version = draft_vocabulary_artifact_version;
    draft.semantic_width = 2U;
    draft.selector_width = 112U;
    draft.selector_storage_width = 112U;
    draft.target_width = 2U;
    draft.vocabulary_size = 16U;
    draft.state_capacity = draft_vocabulary_external_state_max;
    draft.candidate_capacity = 16U;
    const auto append_tensor = [&](DraftArtifactTensorRole role,
                                   DraftArtifactElementType type,
                                   std::uint32_t d0,
                                   std::uint32_t d1,
                                   std::size_t bytes) {
        auto& tensor = draft.tensors[draft_artifact_tensor_index(role)];
        tensor.storage = DraftArtifactTensorStorage::payload;
        tensor.element_type = type;
        tensor.dimension_0 = d0;
        tensor.dimension_1 = d1;
        tensor.payload_offset = payload.size();
        tensor.payload_byte_count = bytes;
        payload.resize(payload.size() + bytes, 0U);
    };
    append_tensor(DraftArtifactTensorRole::selector_projection,
        DraftArtifactElementType::f16, 2U, 112U, 2U * 112U * 2U);
    append_tensor(DraftArtifactTensorRole::selector_head,
        DraftArtifactElementType::f16, 112U, 16U, 112U * 16U * 2U);
    append_tensor(DraftArtifactTensorRole::candidate_head,
        DraftArtifactElementType::f16, 2U, 16U, 2U * 16U * 2U);
    append_tensor(DraftArtifactTensorRole::proposal_exp_lut,
        DraftArtifactElementType::u32, 4'097U, 1U, 4'097U * 4U);
    auto& lut = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::proposal_exp_lut
    )];
    for (std::uint32_t index = 0U; index < 4'097U; ++index) {
        const auto value = 4'097U - index;
        const auto offset = static_cast<std::size_t>(lut.payload_offset) + index * 4U;
        for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
            payload[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8U));
        }
    }
    const auto encoded = encode_fixture_speculative_artifact(descriptor, payload);
    const auto loaded = load_speculative_artifact(encoded);
    expect(loaded.descriptor().draft_vocabulary.present()
            && loaded.descriptor().draft_vocabulary.selector_width == 112U
            && loaded.descriptor().draft_vocabulary.state_capacity == 23U,
        "Phase-4 artifact scoring descriptor changed during roundtrip");
    const auto& loaded_head = loaded.descriptor().draft_vocabulary.tensors[
        draft_artifact_tensor_index(DraftArtifactTensorRole::selector_head)
    ];
    expect(!loaded_head.payload_digest.is_zero(),
        "Phase-4 artifact tensor was not independently authenticated");

    auto malformed = descriptor;
    malformed.draft_vocabulary.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::candidate_head
    )].element_type = DraftArtifactElementType::u32;
    auto rejected = false;
    try {
        static_cast<void>(encode_fixture_speculative_artifact(malformed, payload));
    } catch (const SpeculativeArtifactError& error) {
        rejected = error.code() == SpeculativeArtifactErrorCode::invalid_descriptor;
    }
    expect(rejected, "artifact accepted a non-scoring candidate-head element type");

    malformed = descriptor;
    malformed.draft_vocabulary.candidate_capacity = 3U;
    rejected = false;
    try {
        static_cast<void>(encode_fixture_speculative_artifact(malformed, payload));
    } catch (const SpeculativeArtifactError& error) {
        rejected = error.code() == SpeculativeArtifactErrorCode::invalid_descriptor;
    }
    expect(rejected, "artifact accepted an unexecutable candidate capacity");
}

[[nodiscard]] SpeculationTraceIdentity trace_identity() {
    return SpeculationTraceIdentity {
        .model = model_receipt(),
        .resident = resident_receipt(),
        .artifact = digest(5U),
        .tokenizer = digest(6U),
        .chat_template = digest(7U),
        .source_manifest = digest(8U),
        .registry = digest(9U),
        .sampler = {
            .temperature = 0.8F,
            .top_k = 40U,
            .top_p = 0.95F,
            .min_p = 0.05F,
            .repetition_penalty = 1.1F,
            .seed = 42U,
            .repetition_window = 32U,
        },
    };
}

[[nodiscard]] SpeculationTraceRecord trace_record(
    std::uint32_t position,
    std::optional<std::uint64_t> parent
) {
    SpeculationTraceFeature feature;
    feature.state = {
        .execution_epoch = 3U,
        .arena_epoch = 0U,
        .bank_id = 1U,
        .child_id = 7U,
        .source_stage = FeatureTapStage::layer_residual,
        .source_layer = 1U,
        .coordinate = {
            .target_position = position,
            .draft_position = position + 1U,
        },
        .row_count = 1U,
        .feature_width = 2U,
    };
    feature.dtype = DType::f16;
    feature.bytes = {1U, 2U, 3U, 4U};
    SpeculationTraceRecord record;
    record.sequence_id = 11U;
    record.turn_index = 0U;
    record.kind = position == 0U
        ? SpeculationTraceRowKind::prompt : SpeculationTraceRowKind::generated;
    record.parent_record_index = parent;
    record.target_position = position;
    record.draft_position = position + 1U;
    record.token_id = 3U + position;
    record.target_label_token_id = 4U + position;
    record.sampler_draw_count = position;
    record.source = {
        .source_group = digest(10U),
        .source_cluster = digest(11U),
        .conversation = digest(12U),
        .prompt_derivation = digest(13U),
        .split = SpeculationTraceSplit::training_development,
        .workload = SpeculationTraceWorkload::instruction_chat,
        .generation_seed = 42U,
    };
    record.features.push_back(std::move(feature));
    return record;
}

void test_speculation_trace_roundtrip_and_causality_audit() {
    std::vector<SpeculationTraceRecord> records;
    records.push_back(trace_record(0U, std::nullopt));
    records.push_back(trace_record(1U, 0U));
    records[1].kind = SpeculationTraceRowKind::prompt;
    records[1].sampler_draw_count = 0U;
    records[1].terminal = true;
    const auto encoded = encode_speculation_trace(trace_identity(), records);
    const auto loaded = load_speculation_trace(encoded);
    const auto audit = audit_speculation_trace(loaded);
    expect(audit.record_count == 2U, "trace record count changed");
    expect(audit.sequence_count == 1U, "trace sequence count changed");
    expect(audit.feature_count == 2U && audit.feature_byte_count == 8U,
        "trace feature accounting changed");
    expect(loaded.records()[1].parent_record_index == 0U,
        "trace parent edge changed");

    auto future = records;
    future[1].features[0].state.coordinate.target_position = 2U;
    future[1].features[0].state.coordinate.draft_position = 3U;
    auto rejected = false;
    try {
        static_cast<void>(encode_speculation_trace(trace_identity(), future));
    } catch (const SpeculationTraceError& error) {
        rejected = error.code() == SpeculationTraceErrorCode::invalid_causality;
    }
    expect(rejected, "future-leaking trace feature was not rejected");

    auto corrupt = encoded;
    corrupt.back() ^= 1U;
    rejected = false;
    try {
        static_cast<void>(load_speculation_trace(corrupt));
    } catch (const SpeculationTraceError& error) {
        rejected = error.code() == SpeculationTraceErrorCode::digest_mismatch;
    }
    expect(rejected, "trace envelope corruption was not rejected");

    auto invalid_identity = trace_identity();
    invalid_identity.sampler.top_p = std::nanf("");
    rejected = false;
    try {
        static_cast<void>(encode_speculation_trace(invalid_identity, records));
    } catch (const SpeculationTraceError& error) {
        rejected = error.code() == SpeculationTraceErrorCode::invalid_identity;
    }
    expect(rejected, "trace accepted a non-finite authoritative sampler option");

    auto invalid_stage = records;
    invalid_stage[0].features[0].state.source_stage =
        static_cast<FeatureTapStage>(255U);
    rejected = false;
    try {
        static_cast<void>(encode_speculation_trace(trace_identity(), invalid_stage));
    } catch (const SpeculationTraceError& error) {
        rejected = error.code() == SpeculationTraceErrorCode::invalid_record;
    }
    expect(rejected, "trace accepted an unknown feature tap stage");
}

void test_speculation_trace_preserves_and_recomputes_exact_sampler_evidence() {
    auto identity = trace_identity();
    identity.sampler.top_k = 4U;
    TokenSampler sampler(identity.sampler);
    const std::array<std::uint32_t, 2U> prefix {3U, 4U};
    sampler.observe_tokens(prefix);
    const auto pre = sampler.checkpoint();
    const auto request = sampler.selection_request();
    const auto ticket = sampler.preview_ticket(0U);
    const std::array<TokenLogit, 4U> logits {{
        {.token = 5U, .logit = 2.0F},
        {.token = 6U, .logit = 1.0F},
        {.token = 7U, .logit = 0.0F},
        {.token = 8U, .logit = -1.0F},
    }};
    const auto selected = sampler.sample(logits);
    sampler.observe_tokens(std::span<const std::uint32_t>(&selected, 1U));
    const auto post = sampler.checkpoint();

    auto record = trace_record(1U, std::nullopt);
    record.kind = SpeculationTraceRowKind::generated;
    record.token_id = 4U;
    record.target_label_token_id = selected;
    record.sampler_draw_count = pre.draw_count();
    record.terminal = true;
    record.selection = make_speculation_trace_selection(
        logits, identity.sampler, request, pre, ticket, post, selected
    );
    const std::array records {record};
    const auto loaded = load_speculation_trace(encode_speculation_trace(identity, records));
    const auto audit = audit_speculation_trace(loaded);
    expect(audit.selection_count == 1U
            && audit.prepared_candidate_count
                == loaded.records()[0].selection->prepared_candidates.size(),
        "trace did not preserve exact prepared sampler evidence");

    auto corrupted = record;
    corrupted.selection->prepared_candidates[0].probability = 0.5;
    auto rejected = false;
    try {
        const std::array invalid {corrupted};
        static_cast<void>(encode_speculation_trace(identity, invalid));
    } catch (const SpeculationTraceError& error) {
        rejected = error.code() == SpeculationTraceErrorCode::invalid_record;
    }
    expect(rejected, "trace accepted a prepared distribution that did not recompute");
}

void test_speculation_trace_cli_writes_audits_and_fails_closed() {
    const auto path = std::filesystem::temp_directory_path()
        / "local-gguf-engine-speculation-trace-cli-test.lgtrace";
    auto partial = path;
    partial += ".partial";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(partial, ignored);
    struct Cleanup final {
        std::filesystem::path path;
        std::filesystem::path partial;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
            std::filesystem::remove(partial, error);
        }
    } cleanup {path, partial};

    const auto path_text = path.string();
    const std::array<const char*, 3U> write_args {
        "local-gguf-speculation-trace", "write-fixture", path_text.c_str(),
    };
    std::ostringstream output;
    std::ostringstream error;
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(write_args.size()), write_args.data(), output, error
    ) == 0, "speculation trace CLI fixture write failed");
    expect(std::filesystem::exists(path) && !std::filesystem::exists(partial),
        "speculation trace writer did not atomically publish its file");

    const std::array<const char*, 3U> audit_args {
        "local-gguf-speculation-trace", "audit", path_text.c_str(),
    };
    output.str({});
    error.str({});
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(audit_args.size()), audit_args.data(), output, error
    ) == 0 && output.str().find("\"records\":1") != std::string::npos,
        "speculation trace CLI audit failed");

    output.str({});
    error.str({});
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(write_args.size()), write_args.data(), output, error
    ) == 1 && error.str().find("refusing to overwrite") != std::string::npos,
        "speculation trace CLI overwrote an immutable trace");
}

void test_speculation_trace_resume_plans_only_a_contiguous_group_prefix() {
    using local_gguf_engine::frontends::cli::plan_speculation_trace_resume_group;
    const std::array<std::uint8_t, 3U> complete {1U, 1U, 1U};
    const auto complete_plan = plan_speculation_trace_resume_group(complete);
    expect(complete_plan.fully_published
            && complete_plan.published_prefix_turn_count == complete.size(),
        "complete trace group was not planned as an authenticated skip");

    const std::array<std::uint8_t, 4U> partial {1U, 1U, 0U, 0U};
    const auto partial_plan = plan_speculation_trace_resume_group(partial);
    expect(!partial_plan.fully_published
            && partial_plan.published_prefix_turn_count == 2U,
        "partial trace group did not preserve its exact replay prefix");

    const std::array<std::uint8_t, 1U> absent {0U};
    const auto absent_plan = plan_speculation_trace_resume_group(absent);
    expect(!absent_plan.fully_published
            && absent_plan.published_prefix_turn_count == 0U,
        "absent single-turn group unexpectedly requested replay");

    auto rejected = false;
    try {
        const std::array<std::uint8_t, 3U> hole {1U, 0U, 1U};
        static_cast<void>(plan_speculation_trace_resume_group(hole));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    expect(rejected, "trace resume accepted a publication hole");

    rejected = false;
    try {
        const std::array<std::uint8_t, 1U> invalid {2U};
        static_cast<void>(plan_speculation_trace_resume_group(invalid));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "trace resume accepted a non-boolean publication state");

    rejected = false;
    try {
        static_cast<void>(plan_speculation_trace_resume_group(
            std::span<const std::uint8_t> {}
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "trace resume accepted an empty conversation group");
}

void test_speculation_trace_collection_profiles_are_exact_and_bounded() {
    using local_gguf_engine::engine::FeatureTapStage;
    using local_gguf_engine::engine::target_feature_trace_tap_max_count;
    using local_gguf_engine::frontends::cli::SpeculationTraceFeatureProfile;
    using local_gguf_engine::frontends::cli::plan_speculation_trace_features;

    const auto final = plan_speculation_trace_features(
        SpeculationTraceFeatureProfile::final_normalized, 42U
    );
    expect(final.size() == 1U
            && final.front().stage == FeatureTapStage::final_normalized
            && final.front().layer_index == 0U,
        "final-only collection profile retained an unrelated target tap");

    const auto all = plan_speculation_trace_features(
        SpeculationTraceFeatureProfile::all_target_taps, 42U
    );
    expect(all.size() == target_feature_trace_tap_max_count
            && all.front().stage == FeatureTapStage::embedding_residual
            && all.back().stage == FeatureTapStage::final_normalized,
        "all-target collection profile lost its exact boundary taps");
    for (std::size_t layer = 0U; layer < 42U; ++layer) {
        expect(all[layer + 1U].stage == FeatureTapStage::layer_residual
                && all[layer + 1U].layer_index == layer,
            "all-target collection profile changed layer order");
    }

    auto rejected = false;
    try {
        static_cast<void>(plan_speculation_trace_features(
            SpeculationTraceFeatureProfile::all_target_taps, 43U
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "all-target collection profile exceeded fixed tap capacity");

    rejected = false;
    try {
        static_cast<void>(plan_speculation_trace_features(
            static_cast<SpeculationTraceFeatureProfile>(255U), 42U
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "trace collection accepted an unknown feature profile");
}

void test_speculation_trace_cli_discards_only_an_explicit_unpublished_turn() {
    const auto directory = std::filesystem::temp_directory_path()
        / "local-gguf-engine-speculation-trace-discard-test";
    const auto published = directory / "00000000000000000001.lgtrace";
    auto partial = published;
    partial += ".partial";
    auto body = published;
    body += ".body.partial";
    std::error_code ignored;
    std::filesystem::remove(partial, ignored);
    std::filesystem::remove(body, ignored);
    std::filesystem::remove(published, ignored);
    std::filesystem::remove(directory, ignored);
    expect(std::filesystem::create_directory(directory),
        "could not create partial-discard test directory");
    struct Cleanup final {
        std::filesystem::path directory;
        std::filesystem::path published;
        std::filesystem::path partial;
        std::filesystem::path body;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(partial, error);
            std::filesystem::remove(body, error);
            std::filesystem::remove(published, error);
            std::filesystem::remove(directory, error);
        }
    } cleanup {directory, published, partial, body};

    {
        std::ofstream(partial, std::ios::binary).put('p');
        std::ofstream(body, std::ios::binary).put('b');
    }
    const auto directory_text = directory.string();
    const std::array<const char*, 4U> args {
        "local-gguf-speculation-trace", "discard-partial",
        directory_text.c_str(), "1",
    };
    std::ostringstream output;
    std::ostringstream error;
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(args.size()), args.data(), output, error
    ) == 0 && !std::filesystem::exists(partial) && !std::filesystem::exists(body)
        && output.str().find("\"sequence_id\":1") != std::string::npos,
        "partial recovery did not discard both exact first-sequence fragments");

    output.str({});
    error.str({});
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(args.size()), args.data(), output, error
    ) == 1 && error.str().find("no unpublished fragments") != std::string::npos,
        "partial recovery silently succeeded without a fragment");

    std::ofstream(body, std::ios::binary).put('b');
    std::ofstream(published, std::ios::binary).put('x');
    output.str({});
    error.str({});
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(args.size()), args.data(), output, error
    ) == 1 && std::filesystem::exists(body)
        && error.str().find("already published") != std::string::npos,
        "partial recovery modified an immutable published trace");

    const std::array<const char*, 4U> zero_args {
        "local-gguf-speculation-trace", "discard-partial",
        directory_text.c_str(), "0",
    };
    output.str({});
    error.str({});
    expect(local_gguf_engine::frontends::cli::run_speculation_trace_cli(
        static_cast<int>(zero_args.size()), zero_args.data(), output, error
    ) == 1 && error.str().find("positive") != std::string::npos,
        "partial recovery accepted sequence zero despite one-based manifests");
}

const TestCase kTests[] = {
    {"SHA-256 matches published abc vector", test_sha256_matches_published_abc_vector},
    {"fixture speculative artifact authenticates descriptor and payload",
        test_fixture_artifact_roundtrip_authenticates_every_payload_slice},
    {"fixture speculative artifact authenticates Phase-4 scoring tensors",
        test_fixture_artifact_authenticates_phase4_scoring_tensors},
    {"speculation trace roundtrips and rejects future leakage",
        test_speculation_trace_roundtrip_and_causality_audit},
    {"speculation trace preserves exact sampler evidence",
        test_speculation_trace_preserves_and_recomputes_exact_sampler_evidence},
    {"speculation trace CLI writes audits and refuses overwrite",
        test_speculation_trace_cli_writes_audits_and_fails_closed},
    {"speculation trace resume requires a contiguous published prefix",
        test_speculation_trace_resume_plans_only_a_contiguous_group_prefix},
    {"speculation trace collection profiles are exact and bounded",
        test_speculation_trace_collection_profiles_are_exact_and_bounded},
    {"speculation trace partial recovery is explicit and publication-safe",
        test_speculation_trace_cli_discards_only_an_explicit_unpublished_turn},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> speculation_artifact_test_cases() { return kTests; }

}  // namespace local_gguf_engine::tests
