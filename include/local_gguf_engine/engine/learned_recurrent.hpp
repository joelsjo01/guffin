#pragma once

#include "local_gguf_engine/engine/speculative_artifact.hpp"
#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"

#include <memory>

namespace local_gguf_engine::engine {

// Retains the opened byte source across upload/reset/poison. Rebuild checks
// the same descriptor and digest, not a potentially replaced pathname.
class RecurrentWeightsFile final {
public:
    explicit RecurrentWeightsFile(const std::filesystem::path& path);
    ~RecurrentWeightsFile();
    RecurrentWeightsFile(const RecurrentWeightsFile&) = delete;
    RecurrentWeightsFile& operator=(const RecurrentWeightsFile&) = delete;
    [[nodiscard]] AuthenticatedSpeculativeArtifact read() const;
private:
    int fd_ = -1;
    runtime::Sha256Digest digest_;
    std::array<std::int64_t,5> fingerprint_ {};
};

struct RecurrentGolden {
    std::vector<std::uint32_t> tokens;
    std::vector<float> target_states;
    std::vector<float> confidence_logits;
};

// Offline equations, with an exhaustive vocabulary and the unchanged sampler
// operation order. Never used to execute a production draft or recover a
// numerically uncertain device decision.
[[nodiscard]] RecurrentGolden recurrent_cpu_golden(
    const AuthenticatedSpeculativeArtifact& artifact,
    const tensor::TensorView& token_table, std::span<const float> output_norm,
    std::span<const float> final_feature, std::uint32_t root_token,
    std::span<const double> unit_draws
);
[[nodiscard]] std::uint32_t recurrent_sample_cpu(
    std::span<const float> logits, double draw, const LearnedRecurrentSpec& spec
);

struct RecurrentExecutionResult {
    // 0 = selected, 1 = conservative numeric refusal, 2 = malformed input.
    std::uint32_t status = 2U;
    std::array<std::uint32_t,23> tokens {};
    std::array<float,23> confidence_logits {};
    std::uint32_t count = 0U;
    std::uint64_t execution_epoch = 0U;
    std::uint64_t gpu_duration_ns = 0U;
    std::uint64_t wall_duration_ns = 0U;
    std::uint32_t dispatch_count = 0U;
    std::uint32_t transient_buffer_allocations = 0U;
    Gemma2MetalDecodeExecutionStats execution;
};

class Gemma2SpeculativePlan final {
public:
    ~Gemma2SpeculativePlan();
    Gemma2SpeculativePlan(Gemma2SpeculativePlan&&) noexcept;
    Gemma2SpeculativePlan& operator=(Gemma2SpeculativePlan&&) noexcept;
    Gemma2SpeculativePlan(const Gemma2SpeculativePlan&) = delete;
    Gemma2SpeculativePlan& operator=(const Gemma2SpeculativePlan&) = delete;
    [[nodiscard]] static Gemma2SpeculativePlan create(
        std::shared_ptr<RecurrentWeightsFile> source,
        const EngineModelPayloadReceipt& model, const ResidentTargetPayloadReceipt& resident,
        const Gemma2MetalModelPlan& target
    );
    // Weights-only execution is an explicit development operation. Tokens
    // remain on device throughout the dependent recurrence; one compact
    // completion is read after the whole chain. There is no routing policy.
    // The live F32 final tap is rounded through the trained F16 capture
    // boundary on device before BF16 fusion; trace inputs use the same path.
    [[nodiscard]] RecurrentExecutionResult execute_development(
        const tensor::BackendBuffer& final_feature, const TargetStateView& provenance,
        std::uint32_t root_token, std::span<const double> unit_draws
    );
    // Live common-draw entry point. The caller retains the opaque ticket pool
    // for target verification; drafting never advances the authoritative RNG.
    [[nodiscard]] RecurrentExecutionResult execute_live(
        const tensor::BackendBuffer& final_feature, const TargetStateView& provenance,
        std::uint32_t root_token, const TokenSampler& sampler,
        const SamplerTicketPool& tickets
    );
    void validate_sampler(const TokenSamplerOptions& options) const;
    [[nodiscard]] VerifierDeviceInputBinding verifier_inputs(std::uint64_t epoch) const;
    [[nodiscard]] const SpeculativeArtifactDescriptor& descriptor() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::size_t owned_bytes() const noexcept;
    [[nodiscard]] std::vector<float> diagnostic_target_states() const;
#if defined(LOCAL_GGUF_ENGINE_TESTING)
    [[nodiscard]] static Gemma2SpeculativePlan fixture_for_testing(
        const AuthenticatedSpeculativeArtifact& artifact,
        const tensor::BackendBuffer& table, const tensor::BackendBuffer& norm
    );
#endif
private:
    struct Impl;
    explicit Gemma2SpeculativePlan(std::unique_ptr<Impl> impl);
    [[nodiscard]] RecurrentExecutionResult execute(
        const tensor::BackendBuffer&, const TargetStateView&, std::uint32_t,
        std::span<const double>, const SamplerHistorySnapshot*
    );
    std::unique_ptr<Impl> impl_;
};

}  // namespace local_gguf_engine::engine
