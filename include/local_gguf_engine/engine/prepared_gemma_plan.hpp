#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/engine/standalone_chat.hpp"

#include <cstdint>
#include <optional>

namespace local_gguf_engine::engine {

enum class PreparedGemmaMode : std::uint8_t {
    target_only,
    standalone_gemma3_tli,
};

enum class PreparedGemmaLifecycle : std::uint8_t { ready, poisoned };

// Session-owned aggregate. Factories publish only after every requested
// component has been authenticated, validated and allocated.
class PreparedGemmaPlan final {
public:
    PreparedGemmaPlan(const PreparedGemmaPlan&) = delete;
    PreparedGemmaPlan& operator=(const PreparedGemmaPlan&) = delete;
    PreparedGemmaPlan(PreparedGemmaPlan&&) noexcept = default;
    PreparedGemmaPlan& operator=(PreparedGemmaPlan&&) noexcept = default;

    [[nodiscard]] PreparedGemmaMode mode() const noexcept;
    [[nodiscard]] PreparedGemmaLifecycle lifecycle() const noexcept;
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
    friend PreparedGemmaPlan prepare_gemma_standalone_plan(
        const EngineModel&, const tokenizer::Tokenizer&, std::shared_ptr<StandaloneDraftArtifact>,
        StandaloneExecutionOptions, GemmaOutputHeadPolicy, std::uint32_t, GemmaKVPrecision
    );
    friend PreparedGemmaPlan prepare_gemma_target_only_plan(
        const EngineModel&, GemmaOutputHeadPolicy, std::uint32_t, GemmaKVPrecision
    );
    PreparedGemmaPlan(PreparedGemmaMode, Gemma2MetalModelPlan,
        std::optional<EngineModelPayloadReceipt>, std::optional<ResidentTargetPayloadReceipt>,
        std::optional<runtime::Sha256Digest>);

    PreparedGemmaMode mode_ = PreparedGemmaMode::target_only;
    PreparedGemmaLifecycle lifecycle_ = PreparedGemmaLifecycle::ready;
    Gemma2MetalModelPlan target_;
    std::optional<EngineModelPayloadReceipt> model_receipt_;
    std::optional<ResidentTargetPayloadReceipt> resident_receipt_;
    std::optional<runtime::Sha256Digest> artifact_digest_;
    std::optional<StandaloneChatPlan> standalone_;
};

[[nodiscard]] PreparedGemmaPlan prepare_gemma_standalone_plan(
    const EngineModel& verified_model, const tokenizer::Tokenizer& target_tokenizer,
    std::shared_ptr<StandaloneDraftArtifact> source, StandaloneExecutionOptions execution,
    GemmaOutputHeadPolicy policy = GemmaOutputHeadPolicy::automatic,
    std::uint32_t context_length = 0U,
    GemmaKVPrecision kv_precision = GemmaKVPrecision::automatic
);
[[nodiscard]] PreparedGemmaPlan prepare_gemma_target_only_plan(
    const EngineModel& model,
    GemmaOutputHeadPolicy policy = GemmaOutputHeadPolicy::automatic,
    std::uint32_t context_length = 0U,
    GemmaKVPrecision kv_precision = GemmaKVPrecision::automatic
);

} // namespace local_gguf_engine::engine
