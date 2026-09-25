#include "local_gguf_engine/engine/recurrent_chat.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "../tensor/backends/metal/metal_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace local_gguf_engine::engine {
namespace {
using tensor::BackendBuffer;
BackendBuffer row_view(const BackendBuffer& buffer,std::uint32_t row,std::uint32_t width) {
    using namespace tensor::backends::metal;
    const auto& storage=require_metal_backend_buffer(tensor::backend_buffer_storage(buffer));
    if((std::size_t(row)+1)*width*4>buffer.descriptor().byte_count) {
        throw std::invalid_argument("recurrent feature row exceeds its owner");
    }
    return make_metal_backend_buffer_view(tensor::TensorDescriptor::from(
        tensor::Shape::from({1,width}),tensor::DType::f32),storage.buffer(),
        storage.base_byte_offset()+std::size_t(row)*width*4);
}
} // namespace

void validate_recurrent_execution_options(SpeculativeSamplingMode mode,
    const RecurrentExecutionOptions& execution) {
    if(mode!=SpeculativeSamplingMode::exact_match && mode!=SpeculativeSamplingMode::lossless)
        throw std::invalid_argument("invalid recurrent sampling mode");
    if(execution.draft_head!=RecurrentDraftHead::f16 && execution.draft_head!=RecurrentDraftHead::q8_0)
        throw std::invalid_argument("invalid recurrent draft head");
    if(mode!=SpeculativeSamplingMode::lossless && execution!=RecurrentExecutionOptions{})
        throw std::invalid_argument("recurrent execution alternatives require lossless sampling");
}

struct RecurrentChatPlan::Impl {
    Gemma2SpeculativePlan draft;
    ChainChatVerifier verifier;
    SamplerTicketPool tickets;
    BackendBuffer prefill_feature,decode_feature;
    std::vector<BackendBuffer> retrieval_features;
    const BackendBuffer* origin=nullptr;
    std::uint32_t origin_position=0;
    std::uint64_t epoch=0;
    SpeculativeSamplingMode sampling_mode;
    RecurrentExecutionOptions execution_options;

    Impl(Gemma2SpeculativePlan value,const Gemma2MetalModelPlan& target,SpeculativeSamplingMode mode,
        RecurrentExecutionOptions execution)
        : draft(std::move(value)),
          verifier(target,draft.descriptor().recurrent.horizon,mode,execution.conditional_verification),
          prefill_feature(target.prefill_scratch().final_normalized_hidden),
          decode_feature(target.decode_scratch().normalized_hidden),sampling_mode(mode),
          execution_options(execution) {
        const auto width=target.config().embedding_length;
        retrieval_features.reserve(2);
        for(std::uint32_t i=0;i<2;++i) retrieval_features.push_back(
            row_view(target.verifier_scratch().normalized_hidden,i,width));

    }
};
RecurrentChatPlan::RecurrentChatPlan(Gemma2SpeculativePlan draft,const Gemma2MetalModelPlan& target,
    SpeculativeSamplingMode mode,RecurrentExecutionOptions execution)
    : impl_(nullptr) {
    validate_recurrent_execution_options(mode,execution);
    if(draft.draft_head()!=execution.draft_head)
        throw std::invalid_argument("recurrent execution head disagrees with prepared draft");
    impl_=std::make_unique<Impl>(std::move(draft),target,mode,execution);
}
RecurrentChatPlan::~RecurrentChatPlan()=default;
RecurrentChatPlan::RecurrentChatPlan(RecurrentChatPlan&&) noexcept=default;
RecurrentChatPlan& RecurrentChatPlan::operator=(RecurrentChatPlan&&) noexcept=default;
void RecurrentChatPlan::validate_sampler(const TokenSamplerOptions& options) const {impl_->draft.validate_sampler(options);}
std::uint32_t RecurrentChatPlan::horizon() const noexcept {return impl_->draft.descriptor().recurrent.horizon;}
SpeculativeSamplingMode RecurrentChatPlan::sampling_mode() const noexcept {return impl_->sampling_mode;}
const RecurrentExecutionOptions& RecurrentChatPlan::execution_options() const noexcept {
    return impl_->execution_options;
}
bool RecurrentChatPlan::poisoned() const noexcept {return impl_->draft.poisoned();}
std::size_t RecurrentChatPlan::owned_bytes() const noexcept {
    return sizeof(Impl)+impl_->draft.owned_bytes()+impl_->verifier.owned_bytes();
}
void RecurrentChatPlan::invalidate() noexcept {impl_->origin=nullptr;}
void RecurrentChatPlan::warm_up_verifier_tiles(Gemma2MetalModelPlan& target) {
    if(impl_->draft.poisoned()) throw std::logic_error("cannot warm a poisoned recurrent draft");
    invalidate();
    impl_->verifier.warm_up_tiles(target);
}
void RecurrentChatPlan::publish_prefill_origin(std::uint32_t position) {
    impl_->origin=&impl_->prefill_feature;impl_->origin_position=position;
}
void RecurrentChatPlan::publish_decode_origin(std::uint32_t position) {
    impl_->origin=&impl_->decode_feature;impl_->origin_position=position;
}
void RecurrentChatPlan::publish_retrieval_origin(std::uint32_t row,std::uint32_t position) {
    impl_->origin=&impl_->retrieval_features.at(row);impl_->origin_position=position;
}

RecurrentCycleResult RecurrentChatPlan::run(
    Gemma2MetalModelPlan& target,TokenSampler& sampler,std::uint32_t root,
    std::uint32_t position,std::size_t remaining,std::span<const std::uint32_t> stops,
    const RecurrentCommitCallback& callback) {
    auto& p=*impl_;const auto h=horizon();const auto& config=target.config();
    if(!p.origin || position==0 || p.origin_position!=position-1 || remaining<2
        || position>config.context_length || h+1>config.context_length-position
        || std::find(stops.begin(),stops.end(),root)!=stops.end() || p.epoch==UINT64_MAX) {
        throw std::invalid_argument("recurrent cycle has no compatible live target origin or capacity");
    }
    RecurrentCycleResult result;
    const TargetStateView provenance {
        .execution_epoch=++p.epoch,.source_stage=FeatureTapStage::final_normalized,
        .coordinate={position-1,position},.row_count=1,.feature_width=config.embedding_length
    };
    const bool lossless=p.sampling_mode==SpeculativeSamplingMode::lossless;
    if(lossless) {
        const auto proposal_words=sampler.preview_proposal_words(h);
        result.draft=p.draft.execute_lossless(*p.origin,provenance,root,sampler,
            std::span<const std::uint32_t>(proposal_words.words).first(h));
        if(result.draft.status!=0 || !result.draft.has_distributions || result.draft.count!=h) {
            throw std::runtime_error("lossless drafter did not publish its actual proposal laws");
        }
    } else {
        p.tickets=sampler.preview_tickets(h);
        result.draft=p.draft.execute_live(*p.origin,provenance,root,sampler,p.tickets);
    }
    if(result.draft.status!=0) return result; // certified numeric refusal; target untouched

    const auto sample=[&](TokenSamplerFork& fork,std::uint32_t depth,
        std::span<const TokenLogit> candidates) {
        if(lossless) return fork.sample_speculative(result.draft.tokens[depth],
            result.draft.proposal_distributions[depth],candidates);
        return SpeculativeTokenSelection{fork.sample_with_ticket(p.tickets.at(depth),candidates),false};
    };
    try {
        static_cast<ChainCycleResult&>(result)=p.verifier.run(target,sampler,root,position,
            remaining,stops,ChainProposalBatch{
                .tokens=std::span<const std::uint32_t>(result.draft.tokens).first(h),
                .confidence_logits=std::span<const float>(result.draft.confidence_logits).first(h),
                .source=ProposalFeatureSource::recurrent_state,.device=p.draft.verifier_inputs(p.epoch)},
            sample,callback);
        if(result.has_next_candidates) {
            p.origin=&p.verifier.last_hidden();p.origin_position=position+h;
        }
    } catch(...) {invalidate();throw;}
    return result;
}
}  // namespace local_gguf_engine::engine
