#include "local_gguf_engine/engine/prepared_gemma_plan.hpp"
#include "../tensor/backends/metal_backend.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace local_gguf_engine::engine {

PreparedGemmaPlan::PreparedGemmaPlan(
    PreparedGemmaMode mode, Gemma2MetalModelPlan target,
    std::optional<EngineModelPayloadReceipt> model_receipt,
    std::optional<ResidentTargetPayloadReceipt> resident_receipt,
    std::optional<runtime::Sha256Digest> artifact_digest
) : mode_(mode), target_(std::move(target)), model_receipt_(std::move(model_receipt)),
    resident_receipt_(std::move(resident_receipt)), artifact_digest_(std::move(artifact_digest)) {}

PreparedGemmaMode PreparedGemmaPlan::mode() const noexcept { return mode_; }
PreparedGemmaLifecycle PreparedGemmaPlan::lifecycle() const noexcept {
    if (lifecycle_ == PreparedGemmaLifecycle::poisoned || target_.verification_poisoned()
        || (standalone_ && standalone_->poisoned())) return PreparedGemmaLifecycle::poisoned;
    return PreparedGemmaLifecycle::ready;
}
bool PreparedGemmaPlan::ready() const noexcept { return lifecycle() == PreparedGemmaLifecycle::ready; }
Gemma2MetalModelPlan& PreparedGemmaPlan::target() {
    if (!ready()) throw std::logic_error("prepared Gemma2 plan is poisoned");
    return target_;
}
const Gemma2MetalModelPlan& PreparedGemmaPlan::target() const {
    if (!ready()) throw std::logic_error("prepared Gemma2 plan is poisoned");
    return target_;
}
const std::optional<EngineModelPayloadReceipt>& PreparedGemmaPlan::model_receipt() const noexcept {
    return model_receipt_;
}
const std::optional<ResidentTargetPayloadReceipt>& PreparedGemmaPlan::resident_receipt() const noexcept {
    return resident_receipt_;
}
const std::optional<runtime::Sha256Digest>& PreparedGemmaPlan::artifact_digest() const noexcept {
    return artifact_digest_;
}
void PreparedGemmaPlan::reset_draft_state() {
    if (lifecycle() == PreparedGemmaLifecycle::poisoned || target_.has_pending_verification()) {
        throw std::logic_error("cannot reset a poisoned prepared plan or one with a pending target ticket");
    }
    if (standalone_) standalone_->reset();
}
void PreparedGemmaPlan::poison() noexcept { lifecycle_ = PreparedGemmaLifecycle::poisoned; }
StandaloneChatPlan* PreparedGemmaPlan::standalone() noexcept { return standalone_ ? &*standalone_ : nullptr; }
const StandaloneChatPlan* PreparedGemmaPlan::standalone() const noexcept { return standalone_ ? &*standalone_ : nullptr; }

PreparedGemmaPlan prepare_gemma_standalone_plan(const EngineModel& model,
    const tokenizer::Tokenizer& tokenizer, std::shared_ptr<StandaloneDraftArtifact> source,
    StandaloneExecutionOptions execution, GemmaOutputHeadPolicy policy, std::uint32_t context_length,
    GemmaKVPrecision kv_precision) {
    const auto resolved_precision = resolve_gemma_kv_precision(model.architecture(), kv_precision);
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
    std::optional<std::size_t> admitted_standalone_bytes;
    // Validate the tokenizer relationship before the multi-gigabyte target
    // upload. Family names alone do not establish a conditioning/proposal map.
    {
        const StandaloneTliMapping mapping(tokenizer, source->tokenizer());
        if (model.architecture() == ModelArchitecture::gemma3) {
            auto config = model.gemma_config();
            config.kv_precision = resolved_precision;
            if (context_length) config.context_length = context_length;
            const auto weights = bind_gemma_model_weights(model);
            static_cast<void>(resolve_gemma_output_head_policy(config, weights, policy));
            const auto target_bytes = estimate_gemma3_metal_memory(config, weights);
            admitted_standalone_bytes = StandaloneChatPlan::estimate_owned_bytes(
                config, tokenizer, source->tokenizer(), mapping, execution);
            std::size_t required = sizeof(PreparedGemmaPlan);
            const auto add = [&](std::size_t bytes) {
                if (bytes > std::numeric_limits<std::size_t>::max() - required)
                    throw std::overflow_error("paired Gemma 3 resource estimate overflow");
                required += bytes;
            };
            add(Gemma2MetalModelPlan::host_storage_byte_count());
            add(target_bytes.weights); add(target_bytes.kv_cache); add(target_bytes.scratch);
            add(*admitted_standalone_bytes);
            // These tokenizers also remain live outside StandaloneChatPlan's
            // copies. GGUF mappings are reclaimable file-backed sources, not
            // second immutable device-weight allocations.
            add(tokenizer.owned_bytes()); add(source->tokenizer().owned_bytes());
            if (const auto memory = tensor::backends::metal::memory_info()) {
                const auto budget = std::min(memory->recommended_working_set_bytes, memory->physical_memory_bytes);
                const auto available = memory->current_allocated_bytes < budget
                    ? budget - memory->current_allocated_bytes : 0U;
                if (required > available)
                    throw std::invalid_argument("Gemma 3 target/draft context capacity "
                        + std::to_string(config.context_length) + " requires " + std::to_string(required)
                        + " accounted bytes including target/draft caches, scratch and host ledgers; available engine Metal budget is "
                        + std::to_string(available) + "; select a smaller --context-size or target-only execution");
            }
        }
    }
    ResidentTargetPayloadReceipt resident;
    auto target = create_gemma2_metal_model_plan(model, policy, resident, context_length, resolved_precision);
    PreparedGemmaPlan result(PreparedGemmaMode::standalone_gemma3_tli, std::move(target),
        *model.payload_receipt(), resident, source->digest());
    result.standalone_.emplace(std::move(source), tokenizer, result.target_, execution);
    if (admitted_standalone_bytes && result.standalone_->owned_bytes() > *admitted_standalone_bytes)
        throw std::logic_error("paired Gemma 3 owned storage exceeds its pre-upload admission estimate");
    return result;
}

PreparedGemmaPlan prepare_gemma_target_only_plan(const EngineModel& model, GemmaOutputHeadPolicy policy,
    std::uint32_t context_length, GemmaKVPrecision kv_precision) {
    if (model.payload_receipt()) {
        ResidentTargetPayloadReceipt resident;
        auto target = create_gemma2_metal_model_plan(model, policy, resident, context_length, kv_precision);
        return PreparedGemmaPlan(PreparedGemmaMode::target_only, std::move(target),
            *model.payload_receipt(), resident, std::nullopt);
    }
    return PreparedGemmaPlan(PreparedGemmaMode::target_only,
        create_gemma2_metal_model_plan(model, policy, context_length, kv_precision),
        std::nullopt, std::nullopt, std::nullopt);
}

} // namespace local_gguf_engine::engine
