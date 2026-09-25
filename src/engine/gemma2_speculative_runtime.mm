#include "local_gguf_engine/engine/learned_recurrent.hpp"
#include "../tensor/backends/metal/metal_command_context.hpp"
#include "../tensor/backends/metal/metal_context.hpp"
#include "../tensor/backends/metal/metal_buffer.hpp"

#include <bit>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace local_gguf_engine::engine {
namespace {
using tensor::BackendBuffer;
using namespace tensor::backends::metal;
BackendBuffer scratch(std::uint32_t rows,std::uint32_t columns) {
    return tensor::allocate_backend_buffer(tensor::TensorDescriptor::from(
        tensor::Shape::from({rows,columns}),tensor::DType::f32),runtime::BackendKind::metal);
}
void validate_head(const SpeculativeArtifactDescriptor& artifact,const BackendBuffer& table,
                   const BackendBuffer& norm,RecurrentDraftHead head,const BackendBuffer* packed) {
    const auto& config=artifact.target.config;
    const auto shape=tensor::Shape::from({config.embedding_length,config.vocab_size});
    if(head!=RecurrentDraftHead::f16 && head!=RecurrentDraftHead::q8_0) {
        throw std::invalid_argument("invalid recurrent draft head");
    }
    if(!config.uses_tied_output_weight || table.backend()!=runtime::BackendKind::metal
        || table.descriptor().dtype!=tensor::DType::f16 || table.descriptor().shape!=shape
        || norm.backend()!=runtime::BackendKind::metal || norm.descriptor().dtype!=tensor::DType::f32
        || norm.descriptor().element_count!=config.embedding_length) {
        throw std::invalid_argument("recurrent control requires resident tied F16 embeddings and F32 norm");
    }
    if(head==RecurrentDraftHead::q8_0 && (!packed || packed->backend()!=runtime::BackendKind::metal
        || packed->descriptor().dtype!=tensor::DType::q8_0 || packed->descriptor().shape!=shape
        || config.embedding_length%32!=0)) {
        throw std::invalid_argument("Q8_0 recurrent draft requires a compatible resident Q8_0 target head");
    }
    if(head==RecurrentDraftHead::f16 && packed) {
        throw std::invalid_argument("F16 recurrent draft cannot receive a packed head");
    }
}
}

struct Gemma2SpeculativePlan::Impl {
    MetalCommandContext command {"Trained recurrent full-vocabulary chain"};
    std::shared_ptr<RecurrentWeightsFile> source;
    SpeculativeArtifactDescriptor descriptor;
    std::vector<BackendBuffer> weights;
    BackendBuffer table, norm, output_head;
    RecurrentDraftHead head;
    BackendBuffer fusion, state, embedding, projected, joined, gate, candidate;
    BackendBuffer normalized, logits, runs_a, runs_b, status, tokens, draws, confidences, states;
    BackendBuffer base_request, verifier_requests, proposal_support;
    std::vector<BackendBuffer> state_rows;
    std::uint64_t last_epoch=0;
    bool poison=false;
    std::uint64_t published_epoch=0;
    std::size_t bytes=0;

    Impl(std::shared_ptr<RecurrentWeightsFile> file,const AuthenticatedSpeculativeArtifact& artifact,
         const BackendBuffer& target_table,const BackendBuffer& target_norm,
         RecurrentDraftHead selected_head,const BackendBuffer* packed_output_head)
        : source(std::move(file)),descriptor(artifact.descriptor()),
          table(target_table),norm(target_norm),
          output_head(packed_output_head?*packed_output_head:target_table),head(selected_head),
          fusion(scratch(1,descriptor.recurrent.width)),state(scratch(1,descriptor.recurrent.width)),
          embedding(scratch(1,descriptor.target.config.embedding_length)),
          projected(scratch(1,descriptor.recurrent.width)),joined(scratch(1,2*descriptor.recurrent.width)),
          gate(scratch(1,descriptor.recurrent.width)),candidate(scratch(1,descriptor.recurrent.width)),
          normalized(scratch(1,descriptor.target.config.embedding_length)),
          logits(scratch(1,descriptor.target.config.vocab_size)),
          runs_a(scratch(1,2*descriptor.target.config.vocab_size)),
          runs_b(scratch(1,2*descriptor.target.config.vocab_size)),
          status(scratch(1,1)),tokens(scratch(1,24)),draws(scratch(1,46)),confidences(scratch(1,23)),
          states(scratch(descriptor.recurrent.horizon,descriptor.target.config.embedding_length)),
          base_request(scratch(1,91)), verifier_requests(scratch(descriptor.recurrent.horizon+1,91)),
          proposal_support(scratch(descriptor.recurrent.horizon,769)) {
        static_assert(sizeof(VerifierDeviceSelectionRow)==91*4);
        weights.reserve(recurrent_tensor_count);
        state_rows.reserve(descriptor.recurrent.horizon);
        const auto& storage=require_metal_backend_buffer(tensor::backend_buffer_storage(states));
        for(std::uint32_t i=0;i<descriptor.recurrent.horizon;++i) {
            state_rows.push_back(make_metal_backend_buffer_view(tensor::TensorDescriptor::from(
                tensor::Shape::from({1,descriptor.target.config.embedding_length}),tensor::DType::f32),
                storage.buffer(),storage.base_byte_offset()+std::size_t(i)*descriptor.target.config.embedding_length*4));
        }
        for(const auto& t:descriptor.recurrent.tensors) {
            auto view=tensor::TensorView::from_bytes(tensor::TensorDescriptor::from(
                tensor::Shape::from({t.rows,t.columns}),tensor::DType::f32),
                artifact.payload().subspan(t.offset,t.bytes));
            weights.push_back(tensor::copy_tensor_view_to_backend(view,runtime::BackendKind::metal));
            bytes+=view.bytes().size();
        }
        for(const auto* b:{&fusion,&state,&embedding,&projected,&joined,&gate,&candidate,
            &normalized,&logits,&runs_a,&runs_b,&status,&tokens,&draws,&confidences,&states,
            &base_request,&verifier_requests,&proposal_support}) {
            bytes+=b->descriptor().byte_count;
        }
    }
};

Gemma2SpeculativePlan::Gemma2SpeculativePlan(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
Gemma2SpeculativePlan::~Gemma2SpeculativePlan()=default;
Gemma2SpeculativePlan::Gemma2SpeculativePlan(Gemma2SpeculativePlan&&) noexcept=default;
Gemma2SpeculativePlan& Gemma2SpeculativePlan::operator=(Gemma2SpeculativePlan&&) noexcept=default;
const SpeculativeArtifactDescriptor& Gemma2SpeculativePlan::descriptor() const noexcept {return impl_->descriptor;}
RecurrentDraftHead Gemma2SpeculativePlan::draft_head() const noexcept {return impl_->head;}
bool Gemma2SpeculativePlan::poisoned() const noexcept {return impl_->poison;}
std::size_t Gemma2SpeculativePlan::owned_bytes() const noexcept {return impl_->bytes;}

Gemma2SpeculativePlan Gemma2SpeculativePlan::create(std::shared_ptr<RecurrentWeightsFile> source,
    const EngineModelPayloadReceipt& model,const ResidentTargetPayloadReceipt& resident,
    const Gemma2MetalModelPlan& target,RecurrentDraftHead head) {
    if(!source || target.verification_poisoned()) throw std::invalid_argument("missing source or poisoned target");
    auto artifact=source->read();
    validate_speculative_artifact_target(artifact.descriptor(),model,resident,target.config(),
                                         static_cast<std::uint32_t>(target.output_head_policy()));
    const auto* packed=head==RecurrentDraftHead::q8_0?&target.weights().output_weight:nullptr;
    validate_head(artifact.descriptor(),target.weights().token_embedding_weight,
                  target.weights().output_norm_weight,head,packed);
    prepare_recurrent_pipelines();
    auto result=Gemma2SpeculativePlan(std::make_unique<Impl>(std::move(source),artifact,
        target.weights().token_embedding_weight,target.weights().output_norm_weight,head,packed));
    // State-neutral warmup uses the real recurrent/head/selection pipelines.
    // No warmup proposal is published or consumes authoritative RNG.
    auto feature=scratch(1,target.config().embedding_length);
    fill_shared_metal_buffer_u32(feature,0U);
    std::array<double,23> tickets {};tickets.fill(0.5);
    TargetStateView provenance {.execution_epoch=1,.source_stage=FeatureTapStage::final_normalized,
        .coordinate={0,1},.row_count=1,.feature_width=target.config().embedding_length};
    if(head==RecurrentDraftHead::f16) {
        static_cast<void>(result.execute_development(feature,provenance,0,
            std::span<const double>(tickets).first(artifact.descriptor().recurrent.horizon)));
    }
    std::array<std::uint32_t,23> words {};words.fill(0x80000000U);
    provenance.execution_epoch=2;
    static_cast<void>(result.execute(feature,provenance,0,{},nullptr,
        std::span<const std::uint32_t>(words).first(artifact.descriptor().recurrent.horizon)));
    result.impl_->last_epoch=0;
    return result;
}

#if defined(LOCAL_GGUF_ENGINE_TESTING)
Gemma2SpeculativePlan Gemma2SpeculativePlan::fixture_for_testing(
    const AuthenticatedSpeculativeArtifact& artifact,const BackendBuffer& table,const BackendBuffer& norm,
    RecurrentDraftHead head,const BackendBuffer* packed_output_head) {
    validate_head(artifact.descriptor(),table,norm,head,packed_output_head);
    prepare_recurrent_pipelines();
    return Gemma2SpeculativePlan(std::make_unique<Impl>(nullptr,artifact,table,norm,head,packed_output_head));
}
#endif

RecurrentExecutionResult Gemma2SpeculativePlan::execute_development(
    const BackendBuffer& feature,const TargetStateView& provenance,std::uint32_t root,
    std::span<const double> tickets) {
    return execute(feature,provenance,root,tickets,nullptr);
}

void Gemma2SpeculativePlan::validate_sampler(const TokenSamplerOptions& options) const {
    const auto& s=impl_->descriptor.recurrent;
    if(options.temperature!=s.temperature || options.top_k!=s.top_k
        || options.top_p!=s.top_p || options.min_p!=s.min_p
        || options.repetition_penalty!=1.0F || options.repetition_window!=64) {
        throw std::invalid_argument("recurrent weights are incompatible with the requested sampler");
    }
}

RecurrentExecutionResult Gemma2SpeculativePlan::execute_live(
    const BackendBuffer& feature,const TargetStateView& provenance,std::uint32_t root,
    const TokenSampler& sampler,const SamplerTicketPool& tickets) {
    validate_sampler(sampler.options());
    const auto horizon=impl_->descriptor.recurrent.horizon;
    if(tickets.count!=horizon) throw std::invalid_argument("live recurrent ticket count");
    std::array<double,23> draws {};
    // Compare against freshly previewed opaque tickets, including the exact
    // raw engine outputs. Reject another origin/seed without consuming RNG.
    const auto expected=sampler.preview_tickets(horizon);
    for(std::uint32_t i=0;i<horizon;++i) {
        const auto& a=tickets.at(i);const auto& b=expected.at(i);
        if(a!=b) {
            throw std::invalid_argument("stale live recurrent sampler ticket");
        }
        draws[i]=a.unit_interval_draw();
    }
    return execute(feature,provenance,root,std::span<const double>(draws).first(horizon),
                   &sampler.history_snapshot());
}

RecurrentExecutionResult Gemma2SpeculativePlan::execute_lossless(
    const BackendBuffer& feature,const TargetStateView& provenance,std::uint32_t root,
    const TokenSampler& sampler,std::span<const std::uint32_t> words) {
    validate_sampler(sampler.options());
    const auto horizon=impl_->descriptor.recurrent.horizon;
    if(words.size()!=horizon) throw std::invalid_argument("lossless recurrent proposal word count");
    const auto expected=sampler.preview_proposal_words(horizon);
    if(!std::equal(words.begin(),words.end(),expected.words.begin())) {
        throw std::invalid_argument("stale lossless recurrent proposal words");
    }
    return execute(feature,provenance,root,{},&sampler.history_snapshot(),words);
}

VerifierDeviceInputBinding Gemma2SpeculativePlan::verifier_inputs(std::uint64_t epoch) const {
    if(impl_->poison || !epoch || epoch!=impl_->published_epoch) {
        throw std::logic_error("no current live recurrent device proposal lease");
    }
    return {.token_ids=impl_->tokens,.raw_selection_requests=impl_->verifier_requests,
            .arena_epoch=epoch};
}

RecurrentExecutionResult Gemma2SpeculativePlan::execute(
    const BackendBuffer& feature,const TargetStateView& provenance,std::uint32_t root,
    std::span<const double> tickets,const SamplerHistorySnapshot* history,
    std::span<const std::uint32_t> proposal_words) {
    auto& p=*impl_;const auto& s=p.descriptor.recurrent;const auto& c=p.descriptor.target.config;
    const bool lossless=!proposal_words.empty();
    if(p.head==RecurrentDraftHead::q8_0 && !lossless) {
        throw std::invalid_argument("Q8_0 recurrent draft requires lossless rejection sampling");
    }
    if(p.poison) throw std::logic_error("recurrent plan is poisoned; rebuild from authenticated source");
    if(provenance.execution_epoch==0 || provenance.execution_epoch<=p.last_epoch
        || provenance.source_stage!=FeatureTapStage::final_normalized || provenance.source_layer!=0
        || provenance.arena_epoch!=0 || provenance.row_count!=1 || provenance.feature_width!=c.embedding_length
        || provenance.coordinate.target_position==UINT32_MAX
        || provenance.coordinate.draft_position!=provenance.coordinate.target_position+1
        || feature.backend()!=runtime::BackendKind::metal || feature.descriptor().dtype!=tensor::DType::f32
        || feature.descriptor().byte_count!=std::size_t(c.embedding_length)*4
        || root>=c.vocab_size || (lossless ? proposal_words.size()!=s.horizon || !tickets.empty()
                                         : tickets.size()!=s.horizon)) {
        throw std::invalid_argument("recurrent input shape, feature provenance, ticket capacity, or stale epoch");
    }
    std::array<std::uint32_t,46> ticket_bits {};
    for(std::size_t i=0;i<tickets.size();++i) {
        if(!std::isfinite(tickets[i]) || tickets[i]<0 || tickets[i]>=1 || std::signbit(tickets[i])) {
            throw std::invalid_argument("invalid recurrent unit draw");
        }
        const auto bits=std::bit_cast<std::uint64_t>(tickets[i]);
        ticket_bits[2*i]=static_cast<std::uint32_t>(bits);
        ticket_bits[2*i+1]=static_cast<std::uint32_t>(bits>>32);
    }
    p.last_epoch=provenance.execution_epoch;
    p.published_epoch=0;
    const auto start=std::chrono::steady_clock::now();
    try {
        auto& command=p.command;
        command.restart();
        command.write_u32(p.status,0);command.write_u32(p.tokens,root);
        command.write_u32s(p.draws,0,lossless ? proposal_words
            : std::span<const std::uint32_t>(ticket_bits).first(2*s.horizon));
        if(history) {
            std::array<std::uint32_t,91> base {};
            std::copy(history->token_ids.begin(),history->token_ids.end(),base.begin());
            base[87]=history->head;base[88]=history->length;base[89]=history->window_limit;
            command.write_u32s(p.base_request,0,base);
        }
        auto stage=[&](RecurrentStage which,std::initializer_list<const BackendBuffer*> buffers,
                       std::array<std::uint32_t,4> dims,std::array<float,4> scalars=std::array<float,4>{}) {
            command.encode_recurrent_stage(which,std::span<const BackendBuffer* const>(buffers.begin(),buffers.size()),dims,scalars);
            command.encode_memory_barrier();
        };
        auto linear=[&](const BackendBuffer& x,unsigned weight,unsigned bias,BackendBuffer& y,
                        std::uint32_t in,std::uint32_t out,std::uint32_t offset=0) {
            stage(RecurrentStage::linear,{&x,&p.weights[weight],&p.weights[bias<10?bias:weight],&y,&p.status},
                  {in,out,offset,(bias<10?1U:0U)|(weight==0?2U:0U)});
        };
        linear(feature,0,10,p.fusion,c.embedding_length,s.width);
        stage(RecurrentStage::norm,{&p.fusion,&p.weights[1],&p.state,&p.status},{s.width,0,0,0});
        std::uint32_t selector_dispatches=0;
        for(std::uint32_t depth=0;depth<s.horizon;++depth) {
            stage(RecurrentStage::embedding,{&p.table,&p.tokens,&p.embedding,&p.status},
                  {c.embedding_length,c.vocab_size,depth,s.width});
            linear(p.embedding,2,10,p.projected,c.embedding_length,s.width);
            stage(RecurrentStage::join,{&p.state,&p.projected,&p.joined},{s.width,0,0,0});
            linear(p.joined,3,4,p.gate,2*s.width,s.width);
            linear(p.joined,5,6,p.candidate,2*s.width,s.width);
            stage(RecurrentStage::update,{&p.gate,&p.candidate,&p.state},{s.width,0,0,0});
            linear(p.state,7,10,p.state_rows[depth],s.width,c.embedding_length);
            linear(p.state,8,9,p.confidences,s.width,1,depth);
            command.encode_rms_norm(p.state_rows[depth],p.norm,p.normalized,c.rms_norm_epsilon,1);
            command.encode_memory_barrier();
            command.encode_matmul(p.normalized,p.output_head,p.logits,1);
            command.encode_memory_barrier();
            command.encode_softcap(p.logits,p.logits,c.final_logit_softcap,c.vocab_size);
            command.encode_memory_barrier();
            const auto sorted=command.encode_draft_stable_top_k(p.logits,p.runs_a,p.runs_b,p.status,1,c.vocab_size,s.top_k);
            selector_dispatches+=sorted.dispatch_count+1U; // stable top-k and certified sampling
            command.encode_memory_barrier();
            if(lossless) {
                stage(RecurrentStage::sample_lossless,
                    {sorted.final_buffer_is_a?&p.runs_a:&p.runs_b,&p.draws,&p.tokens,&p.status,&p.proposal_support},
                    {depth,s.top_k,0,0},{s.temperature,s.top_p,s.min_p,0});
            } else {
                stage(RecurrentStage::sample,{sorted.final_buffer_is_a?&p.runs_a:&p.runs_b,&p.draws,&p.tokens,&p.status},
                      {depth,s.top_k,0,0},{s.temperature,s.top_p,s.min_p,0});
            }
        }
        if(history) stage(RecurrentStage::verifier_requests,
            {&p.base_request,&p.tokens,&p.verifier_requests},{s.horizon,0,0,0});
        const auto stats=command.commit_and_wait();
        RecurrentExecutionResult out;
        out.status=command.read_verifier_status(p.status);
        out.execution_epoch=p.last_epoch;out.gpu_duration_ns=stats.gpu_duration_ns;
        out.execution.command_buffer_count=stats.command_buffer_count;
        out.execution.commit_count=stats.commit_count;
        out.execution.synchronous_wait_count=stats.synchronous_wait_count;
        out.execution.compute_encoder_count=stats.compute_encoder_count;
        out.execution.compute_dispatch_count=stats.compute_dispatch_count;
        out.execution.blit_encoder_count=stats.blit_encoder_count;
        out.execution.memory_barrier_count=stats.memory_barrier_count;
        out.execution.transient_buffer_allocation_count=stats.transient_buffer_allocation_count;
        out.execution.gpu_duration_ns=stats.gpu_duration_ns;
        out.execution.f16_projection_count=stats.f16_projection_count;
        out.execution.q8_decode_projection_count=stats.q8_decode_projection_count;
        out.execution.selector_dispatch_count=selector_dispatches;
        out.execution.compact_readback_bytes=4U+(out.status==0?s.horizon*8U:0U);
        out.dispatch_count=stats.compute_dispatch_count;out.transient_buffer_allocations=stats.transient_buffer_allocation_count;
        if(out.status==0) {
            out.count=s.horizon;
            read_shared_metal_buffer_bytes(p.tokens,4,std::span<std::uint8_t>(
                reinterpret_cast<std::uint8_t*>(out.tokens.data()),s.horizon*4));
            read_shared_metal_buffer_bytes(p.confidences,0,std::span<std::uint8_t>(
                reinterpret_cast<std::uint8_t*>(out.confidence_logits.data()),s.horizon*4));
            for(std::uint32_t i=0;i<s.horizon;++i) {
                if(out.tokens[i]>=c.vocab_size || !std::isfinite(out.confidence_logits[i])) {
                    throw std::runtime_error("malformed recurrent completion");
                }
                if(lossless) {
                    auto& distribution=out.proposal_distributions[i];
                    const auto byte_offset=std::size_t(i)*769*4;
                    read_shared_metal_buffer_bytes(p.proposal_support,byte_offset,std::span<std::uint8_t>(
                        reinterpret_cast<std::uint8_t*>(&distribution.count),4));
                    if(distribution.count==0 || distribution.count>s.top_k) {
                        throw std::runtime_error("malformed lossless recurrent support count");
                    }
                    std::array<std::uint32_t,768> support {};
                    read_shared_metal_buffer_bytes(p.proposal_support,byte_offset+4,std::span<std::uint8_t>(
                        reinterpret_cast<std::uint8_t*>(support.data()),distribution.count*12));
                    out.execution.compact_readback_bytes+=4+distribution.count*12;
                    std::uint64_t previous=0;std::uint32_t selected=c.vocab_size;
                    for(std::uint32_t j=0;j<distribution.count;++j) {
                        const auto token=support[3*j];
                        const auto boundary=std::uint64_t(support[3*j+1])|(std::uint64_t(support[3*j+2])<<32);
                        if(token>=c.vocab_size || boundary<=previous || boundary>(1ULL<<32)) {
                            throw std::runtime_error("malformed lossless recurrent CDF");
                        }
                        for(std::uint32_t k=0;k<j;++k) if(distribution.values[k].token==token) {
                            throw std::runtime_error("duplicate lossless recurrent support token");
                        }
                        distribution.values[j]={token,boundary-previous};
                        if(selected==c.vocab_size && proposal_words[i]<boundary) selected=token;
                        previous=boundary;
                    }
                    if(previous!=(1ULL<<32) || selected!=out.tokens[i]) {
                        throw std::runtime_error("lossless recurrent law differs from proposal");
                    }
                }
            }
            out.has_distributions=lossless;
            if(history) p.published_epoch=p.last_epoch;
        } else if(lossless || out.status!=1) throw std::runtime_error("invalid recurrent device input or arithmetic");
        out.wall_duration_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count());
        return out;
    } catch(...) {p.poison=true;throw;}
}

std::vector<float> Gemma2SpeculativePlan::diagnostic_target_states() const {
    if(impl_->poison || impl_->last_epoch==0) throw std::logic_error("no completed recurrent diagnostics");
    std::vector<float> result(impl_->states.descriptor().element_count);
    read_shared_metal_buffer_bytes(impl_->states,0,std::span<std::uint8_t>(
        reinterpret_cast<std::uint8_t*>(result.data()),result.size()*sizeof(float)));
    return result;
}
}  // namespace local_gguf_engine::engine
