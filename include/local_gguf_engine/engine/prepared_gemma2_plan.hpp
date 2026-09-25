#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/standalone_chat.hpp"

#include <cstdint>
#include <optional>

namespace local_gguf_engine::engine {

enum class PreparedGemma2Mode : std::uint8_t {
    target_only,
    standalone_gemma3_tli,
};

enum class PreparedGemma2Lifecycle : std::uint8_t { ready, poisoned };

// Session-owned aggregate. Factories publish only after every requested
// component has been authenticated, validated and allocated.
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
    [[nodiscard]] const std::optional<EngineModelPayloadReceipt>& model_receipt() const noexcept;
    [[nodiscard]] const std::optional<ResidentTargetPayloadReceipt>& resident_receipt() const noexcept;
    [[nodiscard]] const std::optional<runtime::Sha256Digest>& artifact_digest() const noexcept;
    void reset_draft_state();
    void poison() noexcept;
    [[nodiscard]] StandaloneChatPlan* standalone() noexcept;
    [[nodiscard]] const StandaloneChatPlan* standalone() const noexcept;

private:
    friend PreparedGemma2Plan prepare_gemma2_standalone_plan(
        const EngineModel&, const tokenizer::Tokenizer&, std::shared_ptr<StandaloneDraftArtifact>,
        StandaloneExecutionOptions, Gemma2OutputHeadPolicy
    );
    friend PreparedGemma2Plan prepare_gemma2_target_only_plan(
        const EngineModel&, Gemma2OutputHeadPolicy
    );
    PreparedGemma2Plan(PreparedGemma2Mode, Gemma2MetalModelPlan,
        std::optional<EngineModelPayloadReceipt>, std::optional<ResidentTargetPayloadReceipt>,
        std::optional<runtime::Sha256Digest>);

    PreparedGemma2Mode mode_ = PreparedGemma2Mode::target_only;
    PreparedGemma2Lifecycle lifecycle_ = PreparedGemma2Lifecycle::ready;
    Gemma2MetalModelPlan target_;
    std::optional<EngineModelPayloadReceipt> model_receipt_;
    std::optional<ResidentTargetPayloadReceipt> resident_receipt_;
    std::optional<runtime::Sha256Digest> artifact_digest_;
    std::optional<StandaloneChatPlan> standalone_;
};

[[nodiscard]] PreparedGemma2Plan prepare_gemma2_standalone_plan(
    const EngineModel& verified_model, const tokenizer::Tokenizer& target_tokenizer,
    std::shared_ptr<StandaloneDraftArtifact> source, StandaloneExecutionOptions execution,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic
);
[[nodiscard]] PreparedGemma2Plan prepare_gemma2_target_only_plan(
    const EngineModel& model,
    Gemma2OutputHeadPolicy policy = Gemma2OutputHeadPolicy::automatic
);

} // namespace local_gguf_engine::engine
