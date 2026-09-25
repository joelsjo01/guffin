#include "local_gguf_engine/engine/prepared_gemma2_plan.hpp"

#include "local_gguf_engine/engine/gemma2_model.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {
namespace {

[[noreturn]] void incompatible(const char* detail) {
    throw SpeculativeArtifactError(
        SpeculativeArtifactErrorCode::incompatible_target, detail
    );
}

void validate_preallocation_identity(
    const SpeculativeArtifactDescriptor& artifact,
    const EngineModelPayloadReceipt& model,
    const Gemma2ModelConfig& config,
    Gemma2OutputHeadPolicy output_policy
) {
    if (artifact.kind != SpeculativeArtifactKind::fixture_feature_cache) {
        incompatible("Phase-3 prepared plan accepts only fixture feature/cache artifacts");
    }
    if (artifact.target.model != model) {
        incompatible("artifact verified GGUF identity does not match");
    }
    if (artifact.target.config != config) {
        incompatible("artifact target configuration does not match");
    }
    if (artifact.target.output_head_policy != static_cast<std::uint32_t>(output_policy)) {
        incompatible("artifact target output-head policy does not match");
    }
}

}  // namespace

PreparedGemma2Plan::PreparedGemma2Plan(
    PreparedGemma2Mode mode,
    Gemma2MetalModelPlan target,
    std::optional<FeatureSinkSet> feature_sinks,
    std::optional<DraftCacheArena> draft_cache,
    std::optional<EngineModelPayloadReceipt> model_receipt,
    std::optional<ResidentTargetPayloadReceipt> resident_receipt,
    std::optional<runtime::Sha256Digest> artifact_digest
) : mode_(mode), target_(std::move(target)),
    feature_sinks_(std::move(feature_sinks)), draft_cache_(std::move(draft_cache)),
    model_receipt_(std::move(model_receipt)), resident_receipt_(std::move(resident_receipt)),
    artifact_digest_(std::move(artifact_digest)) {}

PreparedGemma2Mode PreparedGemma2Plan::mode() const noexcept { return mode_; }
PreparedGemma2Lifecycle PreparedGemma2Plan::lifecycle() const noexcept {
    if (lifecycle_ == PreparedGemma2Lifecycle::poisoned
        || target_.verification_poisoned()
        || (recurrent_.has_value() && recurrent_->poisoned())
        || (standalone_.has_value() && standalone_->poisoned())
        || (feature_sinks_.has_value() && feature_sinks_->poisoned())
        || (draft_cache_.has_value() && draft_cache_->poisoned())) {
        return PreparedGemma2Lifecycle::poisoned;
    }
    return PreparedGemma2Lifecycle::ready;
}
bool PreparedGemma2Plan::ready() const noexcept {
    return lifecycle() == PreparedGemma2Lifecycle::ready;
}

Gemma2MetalModelPlan& PreparedGemma2Plan::target() {
    if (!ready()) throw std::logic_error("prepared Gemma2 plan is poisoned");
    return target_;
}
const Gemma2MetalModelPlan& PreparedGemma2Plan::target() const {
    if (!ready()) throw std::logic_error("prepared Gemma2 plan is poisoned");
    return target_;
}
FeatureSinkSet* PreparedGemma2Plan::feature_sinks() noexcept {
    return feature_sinks_ ? &*feature_sinks_ : nullptr;
}
const FeatureSinkSet* PreparedGemma2Plan::feature_sinks() const noexcept {
    return feature_sinks_ ? &*feature_sinks_ : nullptr;
}
DraftCacheArena* PreparedGemma2Plan::draft_cache() noexcept {
    return draft_cache_ ? &*draft_cache_ : nullptr;
}
const DraftCacheArena* PreparedGemma2Plan::draft_cache() const noexcept {
    return draft_cache_ ? &*draft_cache_ : nullptr;
}
const std::optional<EngineModelPayloadReceipt>& PreparedGemma2Plan::model_receipt() const noexcept {
    return model_receipt_;
}
const std::optional<ResidentTargetPayloadReceipt>&
PreparedGemma2Plan::resident_receipt() const noexcept { return resident_receipt_; }
const std::optional<runtime::Sha256Digest>& PreparedGemma2Plan::artifact_digest() const noexcept {
    return artifact_digest_;
}

FeatureSinkBinding PreparedGemma2Plan::canonical_feature_binding() {
    if (!ready() || !feature_sinks_) {
        throw std::logic_error("prepared plan has no ready feature sinks");
    }
    if (execution_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
        poison();
        throw std::overflow_error("prepared plan execution epoch overflow");
    }
    ++execution_epoch_;
    return FeatureSinkBinding {
        .sinks = &*feature_sinks_,
        .destination = FeatureCaptureDestination::canonical,
        .execution_epoch = execution_epoch_,
        .arena_epoch = 0U,
    };
}

FeatureSinkBinding PreparedGemma2Plan::speculative_feature_binding(
    std::uint64_t arena_epoch
) {
    if (!ready() || !feature_sinks_ || arena_epoch == 0U) {
        throw std::logic_error("prepared plan has no ready speculative feature sinks");
    }
    if (execution_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
        poison();
        throw std::overflow_error("prepared plan execution epoch overflow");
    }
    ++execution_epoch_;
    return FeatureSinkBinding {
        .sinks = &*feature_sinks_,
        .destination = FeatureCaptureDestination::speculative,
        .execution_epoch = execution_epoch_,
        .arena_epoch = arena_epoch,
    };
}

DraftCacheBinding PreparedGemma2Plan::draft_cache_binding(
    DraftCacheScratchKind scratch,
    std::uint64_t arena_epoch
) {
    if (!ready() || !draft_cache_ || arena_epoch == 0U) {
        throw std::logic_error("prepared plan has no ready draft cache");
    }
    return DraftCacheBinding {
        .arena = &*draft_cache_,
        .scratch = scratch,
        .arena_epoch = arena_epoch,
    };
}

void PreparedGemma2Plan::reset_learned_state() {
    if (lifecycle() == PreparedGemma2Lifecycle::poisoned
        || target_.has_pending_verification()) {
        throw std::logic_error(
            "cannot reset a poisoned prepared plan or one with a pending target ticket"
        );
    }
    if (feature_sinks_) feature_sinks_->reset();
    if (draft_cache_) draft_cache_->reset();
    if (recurrent_) recurrent_->invalidate();
    if (standalone_) standalone_->reset();
    lifecycle_ = PreparedGemma2Lifecycle::ready;
    execution_epoch_ = 0U;
}

void PreparedGemma2Plan::poison() noexcept {
    lifecycle_ = PreparedGemma2Lifecycle::poisoned;
}
RecurrentChatPlan* PreparedGemma2Plan::recurrent() noexcept {
    return recurrent_ ? &*recurrent_ : nullptr;
}
const RecurrentChatPlan* PreparedGemma2Plan::recurrent() const noexcept {
    return recurrent_ ? &*recurrent_ : nullptr;
}
StandaloneChatPlan* PreparedGemma2Plan::standalone() noexcept {
    return standalone_ ? &*standalone_ : nullptr;
}
const StandaloneChatPlan* PreparedGemma2Plan::standalone() const noexcept {
    return standalone_ ? &*standalone_ : nullptr;
}

PreparedGemma2Plan prepare_gemma2_standalone_plan(const EngineModel& model,
    const tokenizer::Tokenizer& tokenizer, std::shared_ptr<StandaloneDraftArtifact> source,
    StandaloneExecutionOptions execution, Gemma2OutputHeadPolicy policy) {
    if (!model.payload_receipt())
        throw std::invalid_argument("standalone draft requires authenticated model sources");
    if (!model.tokenizer_identity() || *model.tokenizer_identity() != tokenizer.identity())
        throw std::invalid_argument("standalone target tokenizer does not match authenticated GGUF metadata");
    if (!source)
        throw std::invalid_argument("standalone draft requires authenticated model sources");
    if (execution.horizon == 0 || execution.horizon > external_speculation_max_horizon)
        throw std::invalid_argument("standalone draft horizon is outside prepared capacity");
    ResidentTargetPayloadReceipt resident;
    auto target = create_gemma2_metal_model_plan(model, policy, resident);
    const auto mode = source->kind() == StandaloneDraftKind::gemma3_270m_q8
        ? PreparedGemma2Mode::standalone_gemma3_tli : PreparedGemma2Mode::standalone_vicuna_tli;
    PreparedGemma2Plan result(mode, std::move(target),
        std::nullopt, std::nullopt, *model.payload_receipt(), resident, source->digest());
    result.standalone_.emplace(std::move(source), tokenizer, result.target_, execution);
    return result;
}

PreparedGemma2Plan prepare_gemma2_target_only_plan(
    const EngineModel& model,
    Gemma2OutputHeadPolicy policy
) {
    if (model.payload_receipt()) {
        ResidentTargetPayloadReceipt resident;
        auto target = create_gemma2_metal_model_plan(model, policy, resident);
        return PreparedGemma2Plan(PreparedGemma2Mode::target_only, std::move(target),
            std::nullopt, std::nullopt, *model.payload_receipt(), resident, std::nullopt);
    }
    return PreparedGemma2Plan(
        PreparedGemma2Mode::target_only,
        create_gemma2_metal_model_plan(model, policy),
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt
    );
}

PreparedGemma2Plan prepare_gemma2_recurrent_plan(
    const EngineModel& model, std::shared_ptr<RecurrentWeightsFile> source,
    const TokenSamplerOptions& sampler, Gemma2OutputHeadPolicy policy,
    SpeculativeSamplingMode sampling_mode, RecurrentExecutionOptions execution
) {
    validate_recurrent_execution_options(sampling_mode, execution);
    if (!source || !model.payload_receipt()) {
        throw std::invalid_argument("recurrent prepared plan requires authenticated sources");
    }
    const auto artifact = source->read();
    const auto& descriptor = artifact.descriptor();
    const auto resolved = resolve_gemma2_output_head_policy(
        model.gemma2_config(), bind_gemma2_model_weights(model), policy);
    if (execution.draft_head == RecurrentDraftHead::q8_0
        && resolved != Gemma2OutputHeadPolicy::q8_0) {
        incompatible("Q8 recurrent drafting requires a resident Q8 target head");
    }
    const auto& s = descriptor.recurrent;
    if (descriptor.kind != SpeculativeArtifactKind::trained_candidate
        || descriptor.target.model != *model.payload_receipt()
        || descriptor.target.config != model.gemma2_config()
        || descriptor.target.output_head_policy != static_cast<std::uint32_t>(resolved)
        || sampler.temperature != s.temperature || sampler.top_k != s.top_k
        || sampler.top_p != s.top_p || sampler.min_p != s.min_p
        || sampler.repetition_penalty != 1.0F || sampler.repetition_window != 64U) {
        incompatible("recurrent weights, target, output head, or sampler are incompatible");
    }
    ResidentTargetPayloadReceipt resident;
    auto target = create_gemma2_metal_model_plan(model, resolved, resident);
    auto draft = Gemma2SpeculativePlan::create(source, *model.payload_receipt(), resident, target,
        execution.draft_head);
    PreparedGemma2Plan result(PreparedGemma2Mode::development_recurrent, std::move(target),
        std::nullopt, std::nullopt, *model.payload_receipt(), resident, artifact.envelope_digest());
    result.recurrent_.emplace(std::move(draft), result.target_, sampling_mode, execution);
    return result;
}

PreparedGemma2Plan prepare_gemma2_fixture_learned_plan(
    const EngineModel& verified_model,
    AuthenticatedSpeculativeArtifact&& artifact,
    Gemma2OutputHeadPolicy policy
) {
    if (!verified_model.payload_receipt().has_value()) {
        throw std::invalid_argument("learned prepared plan requires a verified EngineModel receipt");
    }
    const auto weights = bind_gemma2_model_weights(verified_model);
    const auto resolved_policy = resolve_gemma2_output_head_policy(
        verified_model.gemma2_config(), weights, policy
    );
    const auto descriptor = artifact.descriptor();
    const auto artifact_digest = artifact.envelope_digest();
    validate_preallocation_identity(
        descriptor, *verified_model.payload_receipt(),
        verified_model.gemma2_config(), resolved_policy
    );

    ResidentTargetPayloadReceipt resident_receipt;
    auto target = create_gemma2_metal_model_plan(
        verified_model, resolved_policy, resident_receipt
    );
    validate_speculative_artifact_target(
        descriptor, *verified_model.payload_receipt(), resident_receipt,
        verified_model.gemma2_config(), static_cast<std::uint32_t>(resolved_policy)
    );

    auto feature_sinks = create_feature_sink_set(
        std::span<const FeatureBankSpec>(descriptor.feature_banks)
            .first(descriptor.feature_bank_count),
        artifact.payload(), verified_model.gemma2_config().embedding_length,
        verified_model.gemma2_config().block_count,
        verified_model.gemma2_config().context_length,
        descriptor.draft_cache.tree_row_capacity
    );
    auto draft_cache = create_draft_cache_arena(
        descriptor.draft_cache, verified_model.gemma2_config().context_length
    );

    return PreparedGemma2Plan(
        PreparedGemma2Mode::fixture_learned, std::move(target),
        std::move(feature_sinks), std::move(draft_cache),
        *verified_model.payload_receipt(), resident_receipt, artifact_digest
    );
}

}  // namespace local_gguf_engine::engine
