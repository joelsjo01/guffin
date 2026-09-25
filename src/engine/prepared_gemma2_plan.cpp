#include "local_gguf_engine/engine/prepared_gemma2_plan.hpp"

#include <stdexcept>
#include <utility>

namespace local_gguf_engine::engine {

PreparedGemma2Plan::PreparedGemma2Plan(
    PreparedGemma2Mode mode, Gemma2MetalModelPlan target,
    std::optional<EngineModelPayloadReceipt> model_receipt,
    std::optional<ResidentTargetPayloadReceipt> resident_receipt,
    std::optional<runtime::Sha256Digest> artifact_digest
) : mode_(mode), target_(std::move(target)), model_receipt_(std::move(model_receipt)),
    resident_receipt_(std::move(resident_receipt)), artifact_digest_(std::move(artifact_digest)) {}

PreparedGemma2Mode PreparedGemma2Plan::mode() const noexcept { return mode_; }
PreparedGemma2Lifecycle PreparedGemma2Plan::lifecycle() const noexcept {
    if (lifecycle_ == PreparedGemma2Lifecycle::poisoned || target_.verification_poisoned()
        || (standalone_ && standalone_->poisoned())) return PreparedGemma2Lifecycle::poisoned;
    return PreparedGemma2Lifecycle::ready;
}
bool PreparedGemma2Plan::ready() const noexcept { return lifecycle() == PreparedGemma2Lifecycle::ready; }
Gemma2MetalModelPlan& PreparedGemma2Plan::target() {
    if (!ready()) throw std::logic_error("prepared Gemma2 plan is poisoned");
    return target_;
}
const Gemma2MetalModelPlan& PreparedGemma2Plan::target() const {
    if (!ready()) throw std::logic_error("prepared Gemma2 plan is poisoned");
    return target_;
}
const std::optional<EngineModelPayloadReceipt>& PreparedGemma2Plan::model_receipt() const noexcept {
    return model_receipt_;
}
const std::optional<ResidentTargetPayloadReceipt>& PreparedGemma2Plan::resident_receipt() const noexcept {
    return resident_receipt_;
}
const std::optional<runtime::Sha256Digest>& PreparedGemma2Plan::artifact_digest() const noexcept {
    return artifact_digest_;
}
void PreparedGemma2Plan::reset_draft_state() {
    if (lifecycle() == PreparedGemma2Lifecycle::poisoned || target_.has_pending_verification()) {
        throw std::logic_error("cannot reset a poisoned prepared plan or one with a pending target ticket");
    }
    if (standalone_) standalone_->reset();
}
void PreparedGemma2Plan::poison() noexcept { lifecycle_ = PreparedGemma2Lifecycle::poisoned; }
StandaloneChatPlan* PreparedGemma2Plan::standalone() noexcept { return standalone_ ? &*standalone_ : nullptr; }
const StandaloneChatPlan* PreparedGemma2Plan::standalone() const noexcept { return standalone_ ? &*standalone_ : nullptr; }

PreparedGemma2Plan prepare_gemma2_standalone_plan(const EngineModel& model,
    const tokenizer::Tokenizer& tokenizer, std::shared_ptr<StandaloneDraftArtifact> source,
    StandaloneExecutionOptions execution, Gemma2OutputHeadPolicy policy) {
    if (!model.payload_receipt())
        throw std::invalid_argument("standalone draft requires authenticated model sources");
    if (!model.tokenizer_identity() || *model.tokenizer_identity() != tokenizer.identity())
        throw std::invalid_argument("standalone target tokenizer does not match authenticated GGUF metadata");
    if (!source)
        throw std::invalid_argument("standalone draft requires authenticated model sources");
    if (source->kind() != StandaloneDraftKind::gemma3_270m_q8)
        throw std::invalid_argument("unsupported authenticated standalone draft kind");
    if (execution.horizon == 0 || execution.horizon > external_speculation_max_horizon)
        throw std::invalid_argument("standalone draft horizon is outside prepared capacity");
    ResidentTargetPayloadReceipt resident;
    auto target = create_gemma2_metal_model_plan(model, policy, resident);
    PreparedGemma2Plan result(PreparedGemma2Mode::standalone_gemma3_tli, std::move(target),
        *model.payload_receipt(), resident, source->digest());
    result.standalone_.emplace(std::move(source), tokenizer, result.target_, execution);
    return result;
}

PreparedGemma2Plan prepare_gemma2_target_only_plan(const EngineModel& model, Gemma2OutputHeadPolicy policy) {
    if (model.payload_receipt()) {
        ResidentTargetPayloadReceipt resident;
        auto target = create_gemma2_metal_model_plan(model, policy, resident);
        return PreparedGemma2Plan(PreparedGemma2Mode::target_only, std::move(target),
            *model.payload_receipt(), resident, std::nullopt);
    }
    return PreparedGemma2Plan(PreparedGemma2Mode::target_only,
        create_gemma2_metal_model_plan(model, policy), std::nullopt, std::nullopt, std::nullopt);
}

} // namespace local_gguf_engine::engine
