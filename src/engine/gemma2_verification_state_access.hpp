#pragma once

#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"

namespace local_gguf_engine::engine {

struct Gemma2MetalVerificationStateAccess final {
    using Storage = Gemma2MetalModelPlan::Storage;

    static std::shared_ptr<Storage> owner(Gemma2MetalModelPlan& plan) noexcept {
        return plan.storage_;
    }

    static void set_pending(
        Gemma2MetalModelPlan& plan,
        std::uint32_t cache_position,
        std::uint32_t row_count = 2U,
        VerifierTransactionState state = VerifierTransactionState::submitted
    ) {
        plan.storage_->set_pending_verification(cache_position, row_count, state);
    }

    static void mark_submitted(Gemma2MetalModelPlan& plan) {
        plan.storage_->mark_pending_verification_submitted();
    }

    static void mark_submitted(Storage& storage) {
        storage.mark_pending_verification_submitted();
    }

    static void mark_completed(Gemma2MetalModelPlan& plan) {
        plan.storage_->mark_pending_verification_completed();
    }

    static void mark_completed(Storage& storage) {
        storage.mark_pending_verification_completed();
    }

    static void set_provisional(Gemma2MetalModelPlan& plan, std::uint32_t token_count) {
        plan.storage_->set_provisionally_committed_token_count(token_count);
    }

    static void set_provisional(Storage& storage, std::uint32_t token_count) {
        storage.set_provisionally_committed_token_count(token_count);
    }

    static void clear(Gemma2MetalModelPlan& plan) noexcept {
        plan.storage_->clear_pending_verification();
    }

    static void clear(Storage& storage) noexcept {
        storage.clear_pending_verification();
    }

    static void poison(Gemma2MetalModelPlan& plan) noexcept {
        plan.storage_->poison_pending_verification();
    }

    static void poison(Storage& storage) noexcept {
        storage.poison_pending_verification();
    }
};

}  // namespace local_gguf_engine::engine
