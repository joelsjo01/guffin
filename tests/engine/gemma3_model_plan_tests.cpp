#include "gemma3_model_plan_tests.hpp"
#include "gemma3_target_test_support.hpp"

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/prepared_gemma_plan.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/kquant_packing.hpp"
#include "metal_test_hooks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace local_gguf_engine;
using namespace local_gguf_engine::engine;
using tests::expect;
using tests::TestFailure;
using tensor::DType;

using Gemma3Fixture = tests::Gemma3TargetFixture;

template<class Function>
void rejects_model(Function function, EngineModelErrorCode code, std::string_view detail) {
    try {
        function();
        throw TestFailure("invalid Gemma 3 input was accepted");
    } catch (const EngineModelError& error) {
        expect(error.code() == code, "Gemma 3 validation returned wrong error code");
        expect(error.detail().find(detail) != std::string::npos, "Gemma 3 validation lost offending metadata/tensor");
    }
}

void test_gemma3_metadata_policy_and_native_binding() {
    Gemma3Fixture fixture;
    const auto model = fixture.model();
    const auto& config = model.gemma_config();
    expect(model.architecture() == ModelArchitecture::gemma3 && model_architecture_name(model.architecture()) == "gemma3",
        "Gemma 3 architecture was not preserved");
    expect(config.model_context_length == 131072U && config.context_length == 16384U,
        "Gemma 3 must separate advertised and allocated context");
    expect(config.uses_qk_norm && gemma_kv_dtype(config) == DType::f16, "Gemma 3 normalization/cache policy mismatch");
    expect(config.attention_logit_softcap == 0.0F && config.final_logit_softcap == 0.0F,
        "Gemma 3 must not inherit Gemma 2 softcaps");
    for (std::uint32_t layer = 0U; layer < 48U; ++layer) {
        const bool local = layer % 6U != 5U;
        expect(gemma_layer_uses_sliding_window(config, layer) == local, "Gemma 3 five-local/one-global cadence mismatch");
        expect(gemma_layer_rope_freq_base(config, layer) == (local ? 10000.0F : 1000000.0F),
            "Gemma 3 RoPE base mismatch");
        expect(gemma_layer_rope_freq_scale(config, layer) == (local ? 1.0F : 0.125F),
            "Gemma 3 linear scaling must apply only on global layers");
    }
    const auto weights = bind_gemma_model_weights(model);
    expect(weights.output_weight.bytes().data() == weights.token_embedding_weight.bytes().data(),
        "Gemma 3 tied head lost source alias");
    expect(weights.blocks[0].attention.query_norm_weight && weights.blocks[0].attention.key_norm_weight,
        "Gemma 3 Q/K norms were not bound");
    expect(resolve_gemma_output_head_policy(config, weights, GemmaOutputHeadPolicy::automatic) == GemmaOutputHeadPolicy::q6_k,
        "Gemma 3 automatic head must preserve Q6_K");
    expect(std::string(gemma_output_head_policy_name(GemmaOutputHeadPolicy::q6_k)) == "q6_k",
        "Gemma 3 head policy name mismatch");
    for (const auto policy : {GemmaOutputHeadPolicy::f16, GemmaOutputHeadPolicy::q8_0}) {
        try {
            (void)resolve_gemma_output_head_policy(config, weights, policy);
            throw TestFailure("Gemma 3 accepted implicit head conversion");
        } catch (const std::invalid_argument&) {}
    }
}

void test_gemma3_rejects_bad_metadata_and_shapes() {
    Gemma3Fixture fixture;
    fixture.metadata(tests::metadata_f32("gemma3.rope.scaling.factor", std::numeric_limits<float>::infinity()));
    rejects_model([&] { (void)fixture.model(); }, EngineModelErrorCode::invalid_metadata_value, "scaling.factor");
    fixture.metadata(tests::metadata_f32("gemma3.rope.scaling.factor", 8.0F));
    fixture.metadata(tests::metadata_string("gemma3.rope.scaling.type", "yarn"));
    rejects_model([&] { (void)fixture.model(); }, EngineModelErrorCode::invalid_metadata_value, "scaling.type");
    fixture.metadata(tests::metadata_string("gemma3.rope.scaling.type", "linear"));
    fixture.metadata(tests::metadata_f32("gemma3.attn_logit_softcapping", 50.0F));
    rejects_model([&] { (void)fixture.model(); }, EngineModelErrorCode::invalid_metadata_value, "softcapping");
    fixture.metadata(tests::metadata_f32("gemma3.attn_logit_softcapping", 0.0F));
    fixture.metadata(tests::metadata_u32("gemma3.attention.sliding_window_pattern", 3U));
    rejects_model([&] { (void)fixture.model(); }, EngineModelErrorCode::invalid_metadata_value, "sliding_window_pattern");
    fixture.metadata(tests::metadata_u32("gemma3.attention.sliding_window_pattern", 6U));
    fixture.metadata(tests::metadata_f32("gemma3.rope.freq_base_swa", -1.0F));
    rejects_model([&] { (void)fixture.model(); }, EngineModelErrorCode::invalid_metadata_value, "freq_base_swa");
    fixture.metadata(tests::metadata_f32("gemma3.rope.freq_base_swa", 20000.0F));
    expect(fixture.model().gemma_config().local_rope_freq_base == 20000.0F,
        "explicit local RoPE base must be consumed");
    auto unsupported = fixture.model().gemma_config();
    unsupported.attention_head_count = 4U;
    auto valid_weights = bind_gemma_model_weights(fixture.model());
    rejects_model([&] { validate_gemma3_model_weights(unsupported, valid_weights); },
        EngineModelErrorCode::invalid_tensor, "paired GQA");
    unsupported = fixture.model().gemma_config();
    unsupported.context_length = 4U;
    rejects_model([&] { validate_gemma3_model_weights(unsupported, valid_weights); },
        EngineModelErrorCode::invalid_tensor, "local attention window");
    auto malformed = bind_gemma_model_weights(fixture.model());
    malformed.blocks[0].attention.query_norm_weight.reset();
    rejects_model([&] { validate_gemma3_model_weights(fixture.model().gemma_config(), malformed); },
        EngineModelErrorCode::invalid_tensor, "Q/K norm");
    malformed = bind_gemma_model_weights(fixture.model());
    malformed.blocks[0].attention.query_weight = malformed.blocks[0].attention.key_weight;
    rejects_model([&] { validate_gemma3_model_weights(fixture.model().gemma_config(), malformed); },
        EngineModelErrorCode::invalid_tensor, "attn_q.weight");
    std::erase_if(fixture.view.tensor_payloads, [](const auto& payload) { return payload.tensor.name == "blk.0.attn_k_norm.weight"; });
    rejects_model([&] { (void)bind_gemma_model_weights(fixture.model()); }, EngineModelErrorCode::missing_tensor, "attn_k_norm.weight");
}

void test_gemma3_policy_receipt_binds_global_rope_scaling() {
    Gemma3Fixture fixture;
    const std::array<std::uint8_t, 3> provenance {1U, 2U, 3U};
    const auto first = load_verified_engine_model_from_gguf(fixture.view, provenance);
    fixture.metadata(tests::metadata_f32("gemma3.rope.scaling.factor", 4.0F));
    const auto second = load_verified_engine_model_from_gguf(fixture.view, provenance);
    expect(first.payload_receipt()->source_tensors == second.payload_receipt()->source_tensors,
        "metadata-only change must leave source weight identity unchanged");
    expect(first.payload_receipt()->canonical_consumed_metadata != second.payload_receipt()->canonical_consumed_metadata,
        "Gemma 3 source receipt must bind resolved RoPE scaling");
}

void test_gemma3_rejects_execution_packing_at_source_boundary() {
    Gemma3Fixture fixture;
    const auto model = fixture.model();
    const auto& config = model.gemma_config();
    const auto native = bind_gemma_model_weights(model);
    for (const auto kind : {0U, 1U, 2U}) {
        const auto source = kind == 0U ? native.token_embedding_weight
            : kind == 1U ? native.blocks[0].attention.query_weight
            : native.blocks[0].feed_forward.down_weight;
        const auto packed = tensor::repack_kquant_weight_columns8(source);
        auto malformed = native;
        const auto name = kind == 0U ? "token_embd.weight"
            : kind == 1U ? "blk.0.attn_q.weight" : "blk.0.ffn_down.weight";
        if (kind == 0U) {
            // Preserve the alias so rejection specifically tests the source
            // representation, not an unrelated tied-weight contract.
            malformed.token_embedding_weight = packed.view();
            malformed.output_weight = packed.view();
        } else if (kind == 1U) {
            malformed.blocks[0].attention.query_weight = packed.view();
        } else {
            malformed.blocks[0].feed_forward.down_weight = packed.view();
        }
        rejects_model([&] { validate_gemma3_model_weights(config, malformed); },
            EngineModelErrorCode::invalid_tensor, name);
        rejects_model([&] { (void)estimate_gemma3_metal_memory(config, malformed); },
            EngineModelErrorCode::invalid_tensor, name);
    }
}

std::size_t expected_resident_weight_bytes(const EngineModel& model) {
    std::size_t bytes = 0U;
    for (const auto& source : model.tensor_views()) {
        const auto& descriptor = source.view.descriptor();
        if (source.name.starts_with("blk.")
            && (descriptor.dtype == DType::q4_k || descriptor.dtype == DType::q6_k)) {
            // Independent physical ledger: eight columns per group, retaining
            // every native code, with four extra metadata bytes per Q4 block.
            const auto columns = descriptor.shape.dimension(1);
            const auto blocks = descriptor.shape.dimension(0) / 256U;
            bytes += ((columns + 7U) / 8U) * blocks
                * (descriptor.dtype == DType::q4_k ? 1184U : 1680U);
        } else {
            bytes += source.view.bytes().size();
        }
    }
    return bytes;
}

void test_gemma3_memory_estimate_counts_execution_packing_without_a_device() {
    Gemma3Fixture fixture;
    const auto model = fixture.model();
    auto config = model.gemma_config();
    config.context_length = 16U;
    const auto weights = bind_gemma_model_weights(model);
    const auto estimate = estimate_gemma3_metal_memory(config, weights);
    expect(estimate.weights == expected_resident_weight_bytes(model),
        "Gemma 3 admission must count packed projection bytes before device upload");
    expect(estimate.weights == 2281024U,
        "Gemma 3 fixture must retain the independently calculated physical weight ledger");
    config.context_length = 32U;
    expect(estimate_gemma3_metal_memory(config, weights).weights == estimate.weights,
        "Gemma 3 execution packing must not depend on context capacity");
}

void test_gemma_kv_precision_policy_and_estimates() {
    expect(resolve_gemma_kv_precision(ModelArchitecture::gemma2, GemmaKVPrecision::automatic)
            == GemmaKVPrecision::f16
        && resolve_gemma_kv_precision(ModelArchitecture::gemma3, GemmaKVPrecision::automatic)
            == GemmaKVPrecision::f16,
        "automatic KV precision must select F16 for both Gemma families");
    expect(gemma_kv_precision_name(GemmaKVPrecision::automatic) == "automatic"
        && gemma_kv_precision_name(GemmaKVPrecision::f32) == "f32"
        && gemma_kv_precision_name(GemmaKVPrecision::f16) == "f16",
        "KV precision reporting must distinguish requests and resolved policies");
    for (const auto [architecture, precision] : {
             std::pair {ModelArchitecture::gemma2, GemmaKVPrecision::f32},
             std::pair {ModelArchitecture::gemma3, static_cast<GemmaKVPrecision>(99)},
             std::pair {static_cast<ModelArchitecture>(99), GemmaKVPrecision::automatic}}) {
        try {
            (void)resolve_gemma_kv_precision(architecture, precision);
            throw TestFailure("unsupported Gemma KV policy was accepted");
        } catch (const std::invalid_argument&) {}
    }

    Gemma3Fixture fixture;
    const auto model = fixture.model();
    auto config = model.gemma_config();
    config.context_length = 16U;
    const auto weights = bind_gemma_model_weights(model);
    const auto original = estimate_gemma3_metal_memory(config, weights);
    config.kv_precision = GemmaKVPrecision::f32;
    const auto f32 = estimate_gemma3_metal_memory(config, weights);
    config.kv_precision = GemmaKVPrecision::f16;
    const auto f16 = estimate_gemma3_metal_memory(config, weights);
    expect(f16.weights == original.weights && f16.kv_cache == original.kv_cache
            && f16.scratch == original.scratch && f16.weights == f32.weights,
        "automatic KV admission must match explicit F16 without changing immutable weight accounting");
    expect(f32.kv_cache == 2U * f16.kv_cache && f16.kv_cache == 57344U,
        "half-cache admission did not count both local and global durable storage");
    // Six layers, K/V width 256: ordinary 128-row staging, two-row
    // legacy backups, and the generic 128-private/32-backup-row arena.
    constexpr auto saved_scratch = (128U + 6U * 2U + 6U * (128U + 32U)) * 2U * 256U * 2U;
    expect(f32.scratch - f16.scratch == saved_scratch,
        "half-cache admission must halve every KV-shaped temporary without changing F32 activations");
    for (const auto rows : {1U, 2U, 3U, 24U, gemma2_metal_prefill_chunk_size}) {
        const auto half_bytes = estimate_target_verifier_bytes(config, rows);
        config.kv_precision = GemmaKVPrecision::automatic;
        expect(estimate_target_verifier_bytes(config, rows) == half_bytes,
            "automatic verifier admission must match explicit F16 private and rollback storage");
        config.kv_precision = GemmaKVPrecision::f32;
        const auto full_bytes = estimate_target_verifier_bytes(config, rows);
        config.kv_precision = GemmaKVPrecision::f16;
        const auto backups = std::bit_ceil(std::min(rows, 24U));
        expect(full_bytes - half_bytes == 6U * (rows + backups) * 2U * 256U * 2U,
            "verifier half-cache estimate changed storage outside private K/V and accepted backups");
    }
    expect(model.gemma_config().kv_precision == GemmaKVPrecision::automatic,
        "memory estimation mutated loaded model metadata");
    config.kv_precision = static_cast<GemmaKVPrecision>(99);
    try {
        (void)estimate_gemma3_metal_memory(config, weights);
        throw TestFailure("memory admission accepted an invalid KV policy");
    } catch (const std::invalid_argument&) {}
}

void test_gemma3_precision_owns_all_cache_storage_and_preserves_receipts() {
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    Gemma3Fixture fixture;
    const std::array<std::uint8_t, 3> provenance {1U, 2U, 3U};
    const auto model = load_verified_engine_model_from_gguf(fixture.view, provenance);
    const auto source_receipt = model.payload_receipt();
    ResidentTargetPayloadReceipt f32_receipt, f16_receipt, automatic_receipt;
    auto full = create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::automatic,
        f32_receipt, 16U, GemmaKVPrecision::f32);
    auto half = create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::automatic,
        f16_receipt, 16U, GemmaKVPrecision::f16);
    auto automatic = create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::automatic,
        automatic_receipt, 16U);
    expect(full.config().kv_precision == GemmaKVPrecision::f32
            && half.config().kv_precision == GemmaKVPrecision::f16
            && gemma_kv_dtype(half.config()) == DType::f16,
        "resident configuration did not retain its resolved KV policy");
    expect(f16_receipt == f32_receipt && model.payload_receipt() == source_receipt
            && model.gemma_config().kv_precision == GemmaKVPrecision::automatic,
        "cache precision changed authenticated source or resident weight identity");
    expect(automatic.config() == half.config() && automatic_receipt == f16_receipt
            && automatic.kv_cache_byte_count() == half.kv_cache_byte_count()
            && automatic.scratch_byte_count() == half.scratch_byte_count(),
        "automatic resident allocation and weight identity must match explicit F16");
    expect(full.kv_cache_byte_count() == 2U * half.kv_cache_byte_count(),
        "resident half-cache ledger did not halve durable bytes");
    for (const auto& layer : half.kv_cache())
        expect(layer.key.descriptor().dtype == DType::f16 && layer.value.descriptor().dtype == DType::f16,
            "resident target contains mixed KV precision");
    expect(half.prefill_scratch().current_chunk_key.descriptor().dtype == DType::f16
            && half.prefill_scratch().current_chunk_value.descriptor().dtype == DType::f16,
        "current-chunk staging ignored resolved KV precision");
    for (const auto& layer : half.verifier_scratch().kv_backups)
        expect(layer.backup_key.descriptor().dtype == DType::f16
                && layer.backup_value.descriptor().dtype == DType::f16,
            "width-two rollback storage ignored resolved KV precision");
    const auto estimate = estimate_gemma3_metal_memory(half.config(), bind_gemma_model_weights(model));
    for (const auto rows : {1U, 2U, 3U, 4U, 24U, gemma2_metal_prefill_chunk_size}) {
        const auto arena = create_gemma2_metal_target_verifier_plan(half, rows);
        const auto automatic_arena = create_gemma2_metal_target_verifier_plan(automatic, rows);
        expect(arena.allocated_byte_count() == estimate_target_verifier_bytes(half.config(), rows),
            "half-cache verifier allocation disagrees with admission");
        expect(automatic_arena.allocated_byte_count() == arena.allocated_byte_count(),
            "automatic verifier allocation must match explicit F16");
        for (const auto& layer : arena.scratch().kv_layers)
            expect(layer.private_key.descriptor().dtype == DType::f16
                    && layer.private_value.descriptor().dtype == DType::f16
                    && layer.accepted_backup_key.descriptor().dtype == DType::f16
                    && layer.accepted_backup_value.descriptor().dtype == DType::f16,
                "private or accepted-backup precision differs from durable cache");
        expect(arena.scratch().query.descriptor().dtype == DType::f32
                && arena.scratch().attention_partial_output.descriptor().dtype == DType::f32,
            "half-cache policy changed attention activation/accumulator storage");
        if (rows == gemma2_metal_prefill_chunk_size)
            expect(estimate.scratch == half.scratch_byte_count() + arena.allocated_byte_count(),
                "half-cache admission omitted ordinary lazy workspace");
    }
    auto configured = create_gemma2_metal_model_plan(half.config(), bind_gemma_model_weights(model));
    expect(configured.config().kv_precision == GemmaKVPrecision::f16
            && configured.kv_cache_byte_count() == half.kv_cache_byte_count(),
        "low-level plan factory discarded configuration precision");
    auto automatic_config = half.config();
    automatic_config.kv_precision = GemmaKVPrecision::automatic;
    auto configured_automatic = create_gemma2_metal_model_plan(automatic_config, bind_gemma_model_weights(model));
    expect(configured_automatic.config() == configured.config()
            && configured_automatic.kv_cache_byte_count() == configured.kv_cache_byte_count()
            && configured_automatic.scratch_byte_count() == configured.scratch_byte_count(),
        "low-level automatic plan factory must resolve to explicit F16 allocation");
    auto prepared = prepare_gemma_target_only_plan(model, GemmaOutputHeadPolicy::automatic,
        16U, GemmaKVPrecision::f16);
    expect(prepared.target().config().kv_precision == GemmaKVPrecision::f16
            && prepared.resident_receipt() == f16_receipt
            && prepared.model_receipt() == source_receipt,
        "prepared target factory lost precision or changed weight identity");
    auto prepared_automatic = prepare_gemma_target_only_plan(model, GemmaOutputHeadPolicy::automatic, 16U);
    expect(prepared_automatic.target().config() == prepared.target().config()
            && prepared_automatic.resident_receipt() == prepared.resident_receipt()
            && prepared_automatic.model_receipt() == prepared.model_receipt(),
        "automatic prepared target must resolve to explicit F16 without changing authenticated weights");
}

void test_gemma3_resident_plan_native_weights_context_and_bounded_scratch() {
    Gemma3Fixture fixture;
    const std::array<std::uint8_t, 3> provenance {1U, 2U, 3U};
    const auto model = load_verified_engine_model_from_gguf(fixture.view, provenance);
    const auto source_receipt = model.payload_receipt();
    if (!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    ResidentTargetPayloadReceipt resident, larger_resident;
    auto plan = create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::automatic, resident, 16U);
    auto larger = create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::automatic, larger_resident, 32U);
    expect(plan.config().context_length == 16U && plan.config().model_context_length == 131072U,
        "runtime context override must preserve model maximum");
    expect(&tensor::backend_buffer_storage(plan.weights().token_embedding_weight)
        == &tensor::backend_buffer_storage(plan.weights().output_weight), "resident tied Q6 head allocated twice");
    expect(plan.uploaded_weight_byte_count() == expected_resident_weight_bytes(model),
        "Gemma 3 resident ledger must count execution packing without another native copy");
    expect(resident.byte_count == plan.uploaded_weight_byte_count()
        && resident.allocation_count == model.tensor_views().size(),
        "Gemma 3 resident receipt must cover each unique uploaded allocation once");
    expect(resident == larger_resident, "Gemma 3 resident weight identity must be independent of context capacity");
    // This zero-filled six-layer fixture has an independently serialized
    // receipt: 80 allocations and 2,281,024 physical bytes. Pin the layout tag
    // as well as the upload payload; hashing only GGUF bytes must not pass.
    expect(resident.bytes == runtime::parse_sha256_hex(
        "2eba2cb7d43b1b613ec6ff3c84df6f989538e1893fde1f281a9c5c650df69145"),
        "Gemma 3 resident receipt must bind the versioned execution layout and physical bytes");
    expect(load_verified_engine_model_from_gguf(fixture.view, provenance).payload_receipt() == source_receipt,
        "Gemma 3 execution repacking must not mutate source weights or their authenticated receipt");
    expect(plan.weights().token_embedding_weight.descriptor().packing == tensor::TensorPacking::canonical
        && plan.weights().output_weight.descriptor().packing == tensor::TensorPacking::canonical
        && plan.weights().output_norm_weight.descriptor().packing == tensor::TensorPacking::canonical,
        "Gemma 3 tied embedding/head and output norm must retain native storage");
    const auto source_weights = bind_gemma_model_weights(model);
    for (std::size_t layer = 0U; layer < plan.weights().blocks.size(); ++layer) {
        const auto& block = plan.weights().blocks[layer];
        const auto& source = source_weights.blocks[layer];
        const std::array projections {
            std::pair {&block.attention.query_weight, &source.attention.query_weight},
            std::pair {&block.attention.key_weight, &source.attention.key_weight},
            std::pair {&block.attention.value_weight, &source.attention.value_weight},
            std::pair {&block.attention.output_weight, &source.attention.output_weight},
            std::pair {&block.feed_forward.gate_weight, &source.feed_forward.gate_weight},
            std::pair {&block.feed_forward.up_weight, &source.feed_forward.up_weight},
            std::pair {&block.feed_forward.down_weight, &source.feed_forward.down_weight},
        };
        for (const auto& [actual, native] : projections) {
            expect(actual->descriptor().packing == tensor::TensorPacking::kquant_columns8
                && actual->descriptor().dtype == native->descriptor().dtype,
                "Gemma 3 projections must select execution packing without requantizing native values");
            const auto packed = tensor::repack_kquant_weight_columns8(*native);
            const auto copied = tensor::copy_tensor_from_backend(*actual);
            expect(copied.descriptor() == packed.descriptor()
                && std::equal(copied.bytes().begin(), copied.bytes().end(), packed.bytes().begin(), packed.bytes().end()),
                "Gemma 3 resident projection must own the complete repacked payload");
        }
        for (const auto* norm : {&block.attention.attention_norm_weight,
                 &block.attention.post_attention_norm_weight, &block.attention.query_norm_weight.value(),
                 &block.attention.key_norm_weight.value(), &block.feed_forward.feed_forward_norm_weight,
                 &block.feed_forward.post_feed_forward_norm_weight}) {
            expect(norm->descriptor().packing == tensor::TensorPacking::canonical,
                "Gemma 3 normalization weights must retain native storage");
        }
    }
    expect(plan.weights().blocks[0].feed_forward.down_weight.descriptor().dtype == DType::q6_k,
        "Gemma 3 must not inherit Gemma 2 down requantization");
    for (std::uint32_t layer = 0U; layer < 6U; ++layer) {
        expect(plan.kv_cache()[layer].key.descriptor().dtype == DType::f16, "Gemma 3 automatic KV must use F16");
        expect(plan.kv_cache()[layer].row_capacity == (layer == 5U ? 16U : 8U), "Gemma 3 local/global cache capacity mismatch");
    }
    expect(plan.kv_cache_byte_count() == 2U * 256U * sizeof(std::uint16_t) * (5U * 8U + 16U), "Gemma 3 KV allocation ledger mismatch");
    const auto estimate = estimate_gemma3_metal_memory(plan.config(), bind_gemma_model_weights(model));
    expect(estimate.weights == plan.uploaded_weight_byte_count()
        && estimate.kv_cache == plan.kv_cache_byte_count(), "Gemma 3 admission estimate must match actual resident ownership");
    for (const auto rows : {1U, 2U, 3U, 4U, 24U, gemma2_metal_prefill_chunk_size}) {
        auto arena = create_gemma2_metal_target_verifier_plan(plan, rows);
        expect(estimate_target_verifier_bytes(plan.config(), rows) == arena.allocated_byte_count(),
            "Gemma verifier admission estimate must count actual private/undo/selection allocations");
        if (rows == gemma2_metal_prefill_chunk_size)
            expect(estimate.scratch == plan.scratch_byte_count() + arena.allocated_byte_count(),
                "Gemma 3 admission must include lazily prepared ordinary workspace and count aliases once");
    }
    expect(larger.scratch_byte_count() - plan.scratch_byte_count() == (32U - 16U) * sizeof(float),
        "Gemma 3 scratch must grow only for token staging, not context-sized attention scores");
    const auto schedule = create_gemma2_metal_decode_schedule(plan, 12U);
    expect(schedule.layers()[0].attention_token_count == 8U && schedule.layers()[5].attention_token_count == 13U,
        "Gemma 3 schedule must attend populated local/global prefixes");
    for (const auto context : {0x40000U, 24U}) {
        try {
            (void)create_gemma2_metal_model_plan(model, GemmaOutputHeadPolicy::automatic, context);
            throw TestFailure("Gemma 3 accepted invalid runtime context");
        } catch (const std::invalid_argument&) {}
    }
}

void test_gemma3_plan_unavailable_device() {
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    Gemma3Fixture fixture;
    struct RestoreDevice {
        ~RestoreDevice() { tensor::backends::metal::set_metal_device_unavailable_for_testing(false); }
    } restore;
    tensor::backends::metal::set_metal_device_unavailable_for_testing(true);
    for (const auto precision : {GemmaKVPrecision::automatic, GemmaKVPrecision::f32, GemmaKVPrecision::f16}) {
        try {
            (void)create_gemma2_metal_model_plan(fixture.model(), GemmaOutputHeadPolicy::automatic, 16U, precision);
            throw TestFailure("Gemma 3 plan should reject unavailable Metal");
        } catch (const runtime::BackendError& error) {
            expect(error.code() == runtime::BackendErrorCode::unavailable_metal_device,
                "Gemma 3 unavailable device must report backend availability");
        }
    }
    try {
        (void)create_gemma2_metal_model_plan(fixture.model(), GemmaOutputHeadPolicy::automatic,
            16U, static_cast<GemmaKVPrecision>(99));
        throw TestFailure("invalid precision was accepted before device preparation");
    } catch (const std::invalid_argument&) {
    }
#endif
}

const tests::TestCase cases[] {
    {"Gemma3 metadata, local/global RoPE and native binding", test_gemma3_metadata_policy_and_native_binding},
    {"Gemma3 rejects malformed metadata, Q/K norms and projection shapes", test_gemma3_rejects_bad_metadata_and_shapes},
    {"Gemma3 authenticated policy includes global RoPE scaling", test_gemma3_policy_receipt_binds_global_rope_scaling},
    {"Gemma3 source validation and admission reject execution packing", test_gemma3_rejects_execution_packing_at_source_boundary},
    {"Gemma3 memory admission counts execution packing without Metal", test_gemma3_memory_estimate_counts_execution_packing_without_a_device},
    {"Gemma KV precision validates policy and counts every cache allocation", test_gemma_kv_precision_policy_and_estimates},
    {"Gemma3 half-cache ownership preserves weights and matches memory admission", test_gemma3_precision_owns_all_cache_storage_and_preserves_receipts},
    {"Gemma3 native resident plan has configurable context and bounded scratch", test_gemma3_resident_plan_native_weights_context_and_bounded_scratch},
    {"Gemma3 resident plan reports unavailable Metal", test_gemma3_plan_unavailable_device},
};

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gemma3_model_plan_test_cases() { return cases; }

}  // namespace local_gguf_engine::tests
