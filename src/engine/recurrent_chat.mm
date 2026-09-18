#include "local_gguf_engine/engine/recurrent_chat.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "../tensor/backends/metal/metal_buffer.hpp"
#include "gemma2_verification_state_access.hpp"

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
bool is_stop(std::uint32_t token,std::span<const std::uint32_t> stops) {
    return std::find(stops.begin(),stops.end(),token)!=stops.end();
}
void add_commit_stats(TargetVerifierExecutionStats& total,const TargetVerifierExecutionStats& value) {
    total.command_buffer_count+=value.command_buffer_count;
    total.commit_count+=value.commit_count;
    total.synchronous_wait_count+=value.synchronous_wait_count;
    total.compute_encoder_count+=value.compute_encoder_count;
    total.compute_dispatch_count+=value.compute_dispatch_count;
    total.blit_encoder_count+=value.blit_encoder_count;
    total.memory_barrier_count+=value.memory_barrier_count;
    total.transient_buffer_allocation_count+=value.transient_buffer_allocation_count;
    total.gpu_duration_ns+=value.gpu_duration_ns;
}
std::uint64_t append_digest(std::uint64_t value,std::uint32_t token) {
    for(unsigned byte=0;byte<4;++byte) {value^=(token>>(8*byte))&255U;value*=1099511628211ULL;}
    return value;
}
}

struct RecurrentChatPlan::Impl {
    Gemma2SpeculativePlan draft;
    Gemma2MetalTargetVerifierPlan verifier;
    ProposalGraphStorage graph;
    VerifierEncodeRequest request;
    std::optional<ValidatedProposalGraph> validated_graph;
    SamplerTicketPool tickets;
    SpeculativeControllerScenario scenario;
    BackendBuffer prefill_feature,decode_feature;
    std::vector<BackendBuffer> retrieval_features,verified_features;
    const BackendBuffer* origin=nullptr;
    std::uint32_t origin_position=0;
    std::uint64_t epoch=0;

    Impl(Gemma2SpeculativePlan value,const Gemma2MetalModelPlan& target)
        : draft(std::move(value)),
          verifier(create_gemma2_metal_target_verifier_plan(target,draft.descriptor().recurrent.horizon+1)),
          prefill_feature(target.prefill_scratch().final_normalized_hidden),
          decode_feature(target.decode_scratch().normalized_hidden) {
        const auto width=target.config().embedding_length;
        retrieval_features.reserve(2);
        verified_features.reserve(verifier.row_capacity());
        for(std::uint32_t i=0;i<2;++i) retrieval_features.push_back(
            row_view(target.verifier_scratch().normalized_hidden,i,width));
        for(std::uint32_t i=0;i<verifier.row_capacity();++i) verified_features.push_back(
            row_view(verifier.scratch().normalized_hidden,i,width));
    }
};
RecurrentChatPlan::RecurrentChatPlan(Gemma2SpeculativePlan draft,const Gemma2MetalModelPlan& target)
    : impl_(std::make_unique<Impl>(std::move(draft),target)) {}
RecurrentChatPlan::~RecurrentChatPlan()=default;
RecurrentChatPlan::RecurrentChatPlan(RecurrentChatPlan&&) noexcept=default;
RecurrentChatPlan& RecurrentChatPlan::operator=(RecurrentChatPlan&&) noexcept=default;
void RecurrentChatPlan::validate_sampler(const TokenSamplerOptions& options) const {impl_->draft.validate_sampler(options);}
std::uint32_t RecurrentChatPlan::horizon() const noexcept {return impl_->draft.descriptor().recurrent.horizon;}
bool RecurrentChatPlan::poisoned() const noexcept {return impl_->draft.poisoned();}
std::size_t RecurrentChatPlan::owned_bytes() const noexcept {
    return sizeof(Impl)+impl_->draft.owned_bytes()+impl_->verifier.allocated_byte_count();
}
void RecurrentChatPlan::invalidate() noexcept {impl_->origin=nullptr;}
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
        || is_stop(root,stops) || p.epoch==UINT64_MAX) {
        throw std::invalid_argument("recurrent cycle has no compatible live target origin or capacity");
    }
    p.tickets=sampler.preview_tickets(h);
    RecurrentCycleResult result;
    result.draft=p.draft.execute_live(*p.origin,TargetStateView {
        .execution_epoch=++p.epoch,.source_stage=FeatureTapStage::final_normalized,
        .coordinate={position-1,position},.row_count=1,.feature_width=config.embedding_length
    },root,sampler,p.tickets);
    if(result.draft.status!=0) return result; // certified numeric refusal; target untouched

    auto& graph=p.graph;
    graph.header={.kind=ProposalGraphKind::chain,.horizon=h,.materialized_node_count=h+1,
        .materialized_edge_count=h,.root_count=1,.root_indices={0},
        .graph_identity=p.epoch,.device_arena_epoch=p.epoch};
    auto digest=std::uint64_t{14695981039346656037ULL};
    const auto& history=sampler.history_snapshot();
    for(std::uint32_t i=0;i<history.length;++i) digest=append_digest(digest,
        history.token_ids[(history.head+i)%token_sampler_max_repetition_window]);
    const ProposalPathHistoryDescriptor root_history {digest,position+1};
    for(std::uint32_t row=0;row<=h;++row) {
        const auto token=row==0?root:result.draft.tokens[row-1];
        if(row) digest=append_digest(digest,token);
        graph.nodes.token_ids[row]=token;
        graph.nodes.parent_indices[row]=row==0?proposal_node_none:row-1;
        graph.nodes.depths[row]=static_cast<std::uint16_t>(row);
        graph.nodes.absolute_positions[row]=position+row;
        graph.nodes.mask_component_roots[row]=0;
        graph.nodes.mask_ancestor_counts[row]=static_cast<std::uint16_t>(row+1);
        graph.nodes.conditional_confidences[row]=row==0?1.0F:
            1.0F/(1.0F+std::exp(-result.draft.confidence_logits[row-1]));
        graph.nodes.path_histories[row]={digest,position+row+1};
        graph.nodes.proposal_ticket_offsets[row]=row==0?proposal_ticket_none:static_cast<std::uint16_t>(row-1);
        graph.nodes.feature_provenance[row]={.source=ProposalFeatureSource::recurrent_state,.arena_epoch=p.epoch};
        graph.nodes.output_roles[row]=row==h?ProposalRowOutputRole::post_full_candidate:
            ProposalRowOutputRole::verifies_proposals;
        if(row) {
            const auto edge=row-1;
            graph.materialized_edges.parent_node_indices[edge]=edge;
            graph.materialized_edges.child_node_indices[edge]=row;
            graph.materialized_edges.token_ids[edge]=token;
            graph.materialized_edges.conditional_confidences[edge]=graph.nodes.conditional_confidences[row];
            graph.materialized_edges.sibling_ranks[edge]=0;
            graph.materialized_edges.proposal_ticket_offsets[edge]=static_cast<std::uint16_t>(edge);
        }
        p.request.selection_requests[row]=sampler.device_selection_request(
            std::span<const std::uint32_t>(result.draft.tokens).first(row));
    }
    p.validated_graph.emplace(validate_proposal_graph(graph,{
        .vocabulary_size=config.vocab_size,.context_length=config.context_length,
        .expected_root_token=root,.expected_root_position=position,
        .expected_root_history=root_history,.expected_device_arena_epoch=p.epoch}));
    ChainVerifierSchedule schedule;
    schedule.tiles[0]={static_cast<std::uint16_t>(h),static_cast<std::uint16_t>(h+1)};
    schedule.tile_count=1;
    p.request.topology=make_verifier_topology_descriptor(*p.validated_graph,schedule,
        config.block_count,config.context_length,config.sliding_window);
    p.request.selection_request_count=h+1;
    p.request.sampler_options=sampler.options();
    p.request.input_residency=VerifierInputResidency::device_resident;
    p.request.device_inputs=p.draft.verifier_inputs(p.epoch);
    auto request=validate_verifier_encode_request(p.request,config.vocab_size,
        config.block_count,config.context_length,config.sliding_window);
    auto ticket=encode_gemma2_metal_target_verifier(target,p.verifier,request,
        gemma2_input_embedding_scale(config.embedding_length));
    submit_gemma2_metal_target_verifier(ticket);
    const auto completion=complete_gemma2_metal_target_verifier(ticket);
    result.target=completion.execution;

    // Previews stay on a fork. The pure controller, resumed after each live
    // callback, owns selection/adoption/stop/budget/carry decisions.
    auto& scenario=p.scenario;
    scenario={};
    scenario.graph=&*p.validated_graph;
    scenario.chain_schedule=schedule;
    scenario.proposal_source=SpeculativeProposalSource::learned_external;
    scenario.origin_token=root;scenario.origin_checkpoint=sampler.checkpoint();
    scenario.remaining_selected_token_budget=static_cast<std::uint32_t>(remaining);
    scenario.initial_logical_cached_count=position;
    scenario.available_context_rows=config.context_length-position;
    scenario.callback_decision_count=0;
    auto fork=sampler.fork();
    std::uint32_t matched=0;
    for(std::uint32_t i=0;i<h && i+1<remaining;++i) {
        const auto& candidates=completion.candidates[i];
        const auto token=fork.sample_with_ticket(p.tickets.at(i),
            std::span<const TokenLogit>(candidates.values).first(candidates.count));
        fork.observe_tokens(std::span<const std::uint32_t>(&token,1));
        scenario.target_samples[i]={token,fork.checkpoint()};
        scenario.token_is_stop[i+1]=is_stop(token,stops);
        ++scenario.target_sample_count;
        if(token!=result.draft.tokens[i]) break;
        ++matched;
        if(is_stop(token,stops)) break;
    }
    std::array<std::uint32_t,24> path {};
    for(std::uint32_t i=0;i<=matched;++i) path[i]=i;
    add_commit_stats(result.target,provisionally_commit_gemma2_metal_target_verifier(
        ticket,std::span<const std::uint32_t>(path).first(matched+1)).execution);
    BoundedSpeculativeController controller;
    bool finalized=false,endpoint_done=false;
    std::uint32_t delivered_rows=0,reported_tokens=0;
    SpeculativeControllerOutcome progress;
    const auto finalize=[&](std::uint32_t retained) {
        if(finalized) return;
        add_commit_stats(result.target,finalize_gemma2_metal_target_verifier(ticket,retained).execution);
        finalized=true;
    };
    try {
        while(true) {
            progress=controller.run_available(scenario);
            if(progress.requires_target_sample) {
                throw std::logic_error("live chain controller requested an unavailable causal preview");
            }
            sampler.adopt_checkpoint(progress.adopted_checkpoint);
            if(progress.endpoint_kv_only_count && !endpoint_done) {
                if(progress.endpoint_kv_only_count!=1) throw std::logic_error("chain endpoint count");
                finalize(progress.retained_row_count);
                try {
                    result.endpoint=run_gemma2_metal_target_verifier_endpoint(target,
                        progress.endpoint_kv_only_tokens[0],position+progress.retained_row_count,
                        gemma2_input_embedding_scale(config.embedding_length));
                } catch(...) {
                    Gemma2MetalVerificationStateAccess::poison(target);
                    throw;
                }
                endpoint_done=true;
            }
            if(!progress.requires_callback) break;
            const auto depth=progress.required_callback_depth;
            if(depth!=reported_tokens) throw std::logic_error("non-causal live callback depth");
            delivered_rows=progress.retained_row_count;
            ++reported_tokens; // the throwing callback is an irreversible receipt
            const auto decision=callback({progress.selected_tokens[depth],depth,
                progress.logical_cached_count,false});
            scenario.callbacks[depth]=decision==Gemma2MetalTokenStreamDecision::continue_generation
                ?ControllerCallbackDecision::continue_generation:ControllerCallbackDecision::stop_generation;
            scenario.callback_decision_count=depth+1;
        }
        if(progress.next_candidate_lease.source==TargetStateLeaseSource::selected_verifier_row) {
            const auto row=progress.next_candidate_lease.node_index;
            if(row!=h || progress.retained_row_count!=h+1) throw std::logic_error("invalid post-full lease");
            result.next_candidates=completion.candidates[row];
            result.has_next_candidates=true;
            p.origin=&p.verified_features[row];p.origin_position=position+row;
        }
        finalize(progress.retained_row_count);
        result.retained_rows=progress.retained_row_count+progress.endpoint_kv_only_count;
        result.accepted_edges=progress.accepted_edge_count;
        result.callback_stopped=progress.stop_reason==ControllerStopReason::callback_stop;
        if(progress.stop_reason==ControllerStopReason::stop_token) {
            const auto depth=progress.selected_count-1;
            result.stop_token=progress.selected_tokens[depth];
            // Internal commit notification only: stop tokens never invoke the
            // external text callback, but their sampler/K/V receipt is durable.
            static_cast<void>(callback({*result.stop_token,depth,
                progress.logical_cached_count,true}));
        }
        if(progress.has_carry && !progress.carry_is_cached) result.carry=progress.carry_token;
    } catch(...) {
        if(!finalized) finalize(delivered_rows);
        invalidate();
        throw;
    }
    return result;
}
}  // namespace local_gguf_engine::engine
