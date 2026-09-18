#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"

namespace local_gguf_engine::engine {

struct Gemma2MetalVerificationStateAccess final {
    static void set_pending(
        Gemma2MetalModelPlan& plan,
        std::uint32_t cache_position,
        std::uint32_t row_count = 2U,
        VerifierTransactionState state = VerifierTransactionState::submitted
    ) {
        plan.set_pending_verification(cache_position, row_count, state);
    }

    static void mark_submitted(Gemma2MetalModelPlan& plan) {
        plan.mark_pending_verification_submitted();
    }

    static void mark_completed(Gemma2MetalModelPlan& plan) {
        plan.mark_pending_verification_completed();
    }

    static void set_provisional(Gemma2MetalModelPlan& plan, std::uint32_t token_count) {
        plan.set_provisionally_committed_token_count(token_count);
    }

    static void clear(Gemma2MetalModelPlan& plan) noexcept {
        plan.clear_pending_verification();
    }

    static void poison(Gemma2MetalModelPlan& plan) noexcept {
        plan.poison_pending_verification();
    }
};

}  // namespace local_gguf_engine::engine
