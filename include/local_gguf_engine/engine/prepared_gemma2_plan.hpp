#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"
#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/speculative_artifact.hpp"
#include "local_gguf_engine/engine/recurrent_chat.hpp"

#include <cstdint>
#include <optional>

namespace local_gguf_engine::engine {

enum class PreparedGemma2Mode : std::uint8_t {
    target_only,
    fixture_learned,
    development_recurrent,
};

enum class PreparedGemma2Lifecycle : std::uint8_t {
    ready,
    poisoned,
};

// Session-owned aggregate. Its factory functions publish the object only
// after every requested component has been validated and allocated.
class PreparedGemma2Plan final {
public:
    PreparedGemma2Plan(const PreparedGemma2Plan&) = delete;
    PreparedGemma2Plan& operator=(const PreparedGemma2Plan&) = delete;
    PreparedGemma2Plan(PreparedGemma2Plan&&) noexcept = default;
    PreparedGemma2Plan& operator=(PreparedGemma2Plan&&) noexcept = default;

    [[nodiscard]] PreparedGemma2Mode mode() const noexcept;
    [[nodiscard]] PreparedGemma2Lifecycle lifecycle() const noexcept;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] Gemma2MetalModelPlan& target();
    [[nodiscard]] const Gemma2MetalModelPlan& target() const;
    [[nodiscard]] FeatureSinkSet* feature_sinks() noexcept;
    [[nodiscard]] const FeatureSinkSet* feature_sinks() const noexcept;
    [[nodiscard]] DraftCacheArena* draft_cache() noexcept;
    [[nodiscard]] const DraftCacheArena* draft_cache() const noexcept;
    [[nodiscard]] const std::optional<EngineModelPayloadReceipt>& model_receipt() const noexcept;
    [[nodiscard]] const std::optional<ResidentTargetPayloadReceipt>& resident_receipt() const noexcept;
    [[nodiscard]] const std::optional<runtime::Sha256Digest>& artifact_digest() const noexcept;

    [[nodiscard]] FeatureSinkBinding canonical_feature_binding();
    [[nodiscard]] FeatureSinkBinding speculative_feature_binding(std::uint64_t arena_epoch);
    [[nodiscard]] DraftCacheBinding draft_cache_binding(
        DraftCacheScratchKind scratch,
        std::uint64_t arena_epoch
    );
    void reset_learned_state();
    void poison() noexcept;
    [[nodiscard]] RecurrentChatPlan* recurrent() noexcept;
    [[nodiscard]] const RecurrentChatPlan* recurrent() const noexcept;

private:
    friend PreparedGemma2Plan prepare_gemma2_target_only_plan(
        const EngineModel&, Gemma2OutputHeadPolicy
    );
    friend PreparedGemma2Plan prepare_gemma2_fixture_learned_plan(
        const EngineModel&, AuthenticatedSpeculativeArtifact&&,
        Gemma2OutputHeadPolicy
    );
    friend PreparedGemma2Plan prepare_gemma2_recurrent_plan(
        const EngineModel&, std::shared_ptr<RecurrentWeightsFile>,
        const TokenSamplerOptions&, Gemma2OutputHeadPolicy
    );

    PreparedGemma2Plan(
        PreparedGemma2Mode mode,
        Gemma2MetalModelPlan target,
        std::optional<FeatureSinkSet> feature_sinks,
        std::optional<DraftCacheArena> draft_cache,
        std::optional<EngineModelPayloadReceipt> model_receipt,
        std::optional<ResidentTargetPayloadReceipt> resident_receipt,
        std::optional<runtime::Sha256Digest> artifact_digest
    );

    PreparedGemma2Mode mode_ = PreparedGemma2Mode::target_only;
    PreparedGemma2Lifecycle lifecycle_ = PreparedGemma2Lifecycle::ready;
    Gemma2MetalModelPlan target_;
    std::optional<FeatureSinkSet> feature_sinks_;
    std::optional<DraftCacheArena> draft_cache_;
    std::optional<EngineModelPayloadReceipt> model_receipt_;
    std::optional<ResidentTargetPayloadReceipt> resident_receipt_;
    std::optional<runtime::Sha256Digest> artifact_digest_;
    std::uint64_t execution_epoch_ = 0U;
    std::optional<RecurrentChatPlan> recurrent_;
};

[[nodiscard]] PreparedGemma2Plan prepare_gemma2_target_only_plan(
    const EngineModel& model,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic
);

// Consumes the authenticated artifact so its bulk CPU payload is released as
// soon as the persistent Metal feature/cache state has been uploaded.
[[nodiscard]] PreparedGemma2Plan prepare_gemma2_fixture_learned_plan(
    const EngineModel& verified_model,
    AuthenticatedSpeculativeArtifact&& artifact,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic
);

[[nodiscard]] PreparedGemma2Plan prepare_gemma2_recurrent_plan(
    const EngineModel& verified_model, std::shared_ptr<RecurrentWeightsFile> source,
    const TokenSamplerOptions& sampler,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic
);

}  // namespace local_gguf_engine::engine
