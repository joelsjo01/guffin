#include "learned_recurrent_tests.hpp"
#include "local_gguf_engine/engine/learned_recurrent.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "tensor_test_support.hpp"
#include "../../src/tensor/backends/metal/metal_command_context.hpp"
#include "../../src/tensor/backends/metal/metal_test_hooks.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <unistd.h>

namespace {
using namespace local_gguf_engine;
using namespace engine;
using tests::expect;
using namespace tensor::backends::metal;
template<class F> void refuses(F&& f,const char* message) {
    bool failed=false;try {f();} catch(const std::exception&) {failed=true;}
    expect(failed,message);
}

struct Fixture {
    SpeculativeArtifactDescriptor a;
    std::vector<std::uint8_t> payload;
    std::vector<float> feature,norm;
    tensor::TensorStorage table;
    explicit Fixture(std::uint32_t h=3):table(tensor::TensorStorage::allocate(
        tensor::TensorDescriptor::from(tensor::Shape::from({32,65}),tensor::DType::f16))) {
        a.schema_version=3;a.kind=SpeculativeArtifactKind::trained_candidate;a.artifact_id=1;
        auto hash=runtime::sha256(std::array<std::uint8_t,1>{1});
        a.target.model={1,4096,2,hash,hash,hash};a.target.resident={1,2,4096,hash};
        a.target.config.embedding_length=32;a.target.config.vocab_size=65;
        a.target.config.uses_tied_output_weight=true;a.target.config.rms_norm_epsilon=1e-6F;
        a.target.config.final_logit_softcap=30;a.target.output_head_policy=2;
        auto& r=a.recurrent;r.version=1;r.width=16;r.horizon=h;r.top_k=8;
        r.temperature=0.8F;r.top_p=0.95F;r.min_p=0.05F;r.provenance.fill(hash);
        const std::array<std::array<unsigned,2>,10> shapes {{{16,32},{1,16},{16,32},{16,32},{1,16},
            {16,32},{1,16},{32,16},{1,16},{1,1}}};
        for(std::size_t t=0;t<10;++t) {
            auto& w=r.tensors[t];w.rows=shapes[t][0];w.columns=shapes[t][1];w.offset=payload.size();
            w.bytes=std::uint64_t(w.rows)*w.columns*4;
            for(unsigned i=0;i<w.rows*w.columns;++i) {
                float value=t==1?1.0F:float(int((i*13+t*7)%17)-8)/64.0F;
                auto bits=std::bit_cast<std::uint32_t>(value);
                for(unsigned b=0;b<4;++b) payload.push_back(static_cast<std::uint8_t>(bits>>(8*b)));
            }
            w.digest=runtime::sha256(std::span<const std::uint8_t>(payload).subspan(w.offset,w.bytes));
        }
        for(std::size_t i=0;i<table.descriptor().element_count;++i) {
            auto bits=tensor::f32_to_f16_bits(float(int((i*19)%37)-18)/32.0F);
            table.mutable_bytes()[2*i]=static_cast<std::uint8_t>(bits);
            table.mutable_bytes()[2*i+1]=static_cast<std::uint8_t>(bits>>8);
        }
        feature.resize(32);norm.assign(32,1.0F);
        for(unsigned i=0;i<32;++i) feature[i]=float(int(i%11)-5)/8.0F;
        feature[0]=1.00395F; // F16 -> BF16 differs from a direct BF16 cast.
    }
    auto encode() const {return encode_recurrent_weights_artifact(a,payload);}
};

void artifact_contract() {
    Fixture f;auto bytes=f.encode();auto loaded=load_speculative_artifact(bytes);
    expect(f.encode()==bytes,"trained artifact packer is not deterministic");
    expect(loaded.descriptor()==f.a,"trained artifact descriptor changed");
    auto corrupted=bytes;corrupted.back()^=1;
    refuses([&]{static_cast<void>(load_speculative_artifact(corrupted));},"corrupt weights accepted");
    for(std::size_t n:{0U,8U,64U,128U}) {
        refuses([&]{static_cast<void>(load_speculative_artifact(std::span(bytes).first(n)));},"truncated weights accepted");
    }
    for(unsigned change=0;change<7;++change) {
        auto invalid=f.a;
        switch(change) {
            case 0: invalid.recurrent.role=2;break;
            case 1: invalid.recurrent.width=3585;break;
            case 2: invalid.recurrent.horizon=2;break;
            case 3: invalid.recurrent.tensors[1].offset=0;break;
            case 4: invalid.recurrent.tensors[0].rows=1;break;
            case 5: invalid.recurrent.provenance[0]={};break;
            case 6: invalid.recurrent.temperature=std::numeric_limits<float>::quiet_NaN();break;
        }
        refuses([&]{static_cast<void>(encode_recurrent_weights_artifact(invalid,f.payload));},"invalid recurrent descriptor accepted");
    }
    f.payload[0]=0;f.payload[1]=0;f.payload[2]=0x80;f.payload[3]=0x7f;
    f.a.recurrent.tensors[0].digest=runtime::sha256(std::span(f.payload).first(f.a.recurrent.tensors[0].bytes));
    refuses([&]{static_cast<void>(f.encode());},"authenticated nonfinite weight accepted");
}

void file_lifetime() {
    std::array<char,64> name {};std::strcpy(name.data(),"/tmp/lg-recurrent-test-XXXXXX");
    int fd=mkstemp(name.data());expect(fd>=0,"temporary weights file");close(fd);
    const auto path=std::filesystem::path(name.data());
    Fixture f;auto bytes=f.encode();
    {std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));}
    RecurrentWeightsFile source(path);
    expect(source.read().payload().size()==f.payload.size(),"file-backed payload changed");
    {std::fstream out(path,std::ios::in|std::ios::out|std::ios::binary);out.seekp(-1,std::ios::end);out.put('\0');}
    refuses([&]{static_cast<void>(source.read());},"mutated opened weights accepted");
    std::filesystem::remove(path);
}

void full_rollout() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    for(unsigned h:{3U,4U,7U,8U,15U,16U,23U}) {
        Fixture f(h);auto a=load_speculative_artifact(f.encode());
        auto table=tensor::copy_tensor_to_backend(f.table,runtime::BackendKind::metal);
        auto norm=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(f.norm),runtime::BackendKind::metal);
        auto feature=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,f.feature),runtime::BackendKind::metal);
        auto plan=Gemma2SpeculativePlan::fixture_for_testing(a,table,norm);
        std::array<double,23> draws {};for(unsigned i=0;i<h;++i) draws[i]=0.13+double(i%5)*0.151;
        auto golden=recurrent_cpu_golden(a,f.table.view(),f.norm,f.feature,7,std::span(draws).first(h));
        TargetStateView provenance {.execution_epoch=1,.source_stage=FeatureTapStage::final_normalized,
            .coordinate={4,5},.row_count=1,.feature_width=32};
        auto actual=plan.execute_development(feature,provenance,7,std::span(draws).first(h));
        expect(actual.status==0 && actual.count==h,"ordinary recurrent rollout refused");
        expect(std::equal(golden.tokens.begin(),golden.tokens.end(),actual.tokens.begin()),"dependent proposal tokens differ from CPU equations");
        auto states=plan.diagnostic_target_states();
        for(std::size_t i=0;i<states.size();++i) {
            tests::expect_close(states[i],golden.target_states[i],0.00001F,"recurrent state CPU/Metal parity");
        }
        expect(actual.transient_buffer_allocations==0,"hot-loop recurrent allocation");
        refuses([&]{static_cast<void>(plan.execute_development(feature,provenance,7,std::span(draws).first(h)));},"stale feature epoch accepted");
        ++provenance.execution_epoch;
        refuses([&]{static_cast<void>(plan.execute_development(feature,provenance,65,std::span(draws).first(h)));},"out-of-vocabulary root accepted");
        refuses([&]{static_cast<void>(plan.execute_development(feature,provenance,7,std::span(draws).first(h-1)));},"short ticket pool accepted");
        auto repeated=plan.execute_development(feature,provenance,7,std::span(draws).first(h));
        expect(repeated.tokens==actual.tokens,"failed request changed future recurrent output");
        ++provenance.execution_epoch;
        auto bad=provenance;bad.coordinate.target_position=UINT32_MAX;bad.coordinate.draft_position=0;
        refuses([&]{static_cast<void>(plan.execute_development(feature,bad,7,std::span(draws).first(h)));},"wrapped position accepted");
        draws[0]=std::numeric_limits<double>::quiet_NaN();
        refuses([&]{static_cast<void>(plan.execute_development(feature,provenance,7,std::span(draws).first(h)));},"nonfinite ticket accepted");
        draws[0]=0.13;
        fill_shared_metal_buffer_u32(feature,0x7fc00000U);
        refuses([&]{static_cast<void>(plan.execute_development(feature,provenance,7,std::span(draws).first(h)));},"nonfinite feature accepted");
        expect(plan.poisoned(),"invalid device arithmetic did not poison unpublished state");
    }
}

void unavailable_device() {
    set_metal_device_unavailable_for_testing(true);
    try {
        refuses([]{prepare_recurrent_pipelines();},"prepared recurrent pipelines bypassed unavailable device");
    } catch(...) {set_metal_device_unavailable_for_testing(false);throw;}
    set_metal_device_unavailable_for_testing(false);
}

void live_tickets_and_histories() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    Fixture f(23);auto a=load_speculative_artifact(f.encode());
    auto table=tensor::copy_tensor_to_backend(f.table,runtime::BackendKind::metal);
    auto norm=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(f.norm),runtime::BackendKind::metal);
    auto feature=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,f.feature),runtime::BackendKind::metal);
    auto plan=Gemma2SpeculativePlan::fixture_for_testing(a,table,norm);
    TokenSampler sampler({.temperature=0.8F,.top_k=8,.seed=71});
    std::array<std::uint32_t,90> history {};for(unsigned i=0;i<90;++i) history[i]=i%65;
    sampler.observe_tokens(history);
    const auto checkpoint=sampler.checkpoint();
    TargetStateView origin {.execution_epoch=1,.source_stage=FeatureTapStage::final_normalized,
        .coordinate={89,90},.row_count=1,.feature_width=32};
    const auto tickets=sampler.preview_tickets(23);
    auto foreign=sampler.options();foreign.seed=72;
    refuses([&]{static_cast<void>(plan.execute_live(feature,origin,7,sampler,
        TokenSampler(foreign).preview_tickets(23)));},"foreign ticket lineage accepted");
    auto result=plan.execute_live(feature,origin,7,sampler,tickets);
    expect(result.status==0,"live-ticket fixture numeric refusal");
    expect(sampler.checkpoint().same_state(checkpoint),"draft advanced authoritative sampler");
    const auto inputs=plan.verifier_inputs(1);
    const auto bytes=tensor::copy_backend_buffer_bytes(inputs.raw_selection_requests,0,24*sizeof(VerifierDeviceSelectionRow));
    for(unsigned row=0;row<24;++row) {
        VerifierDeviceSelectionRow actual;
        std::memcpy(&actual,bytes.data()+row*sizeof(actual),sizeof(actual));
        const auto expected=sampler.device_selection_request(std::span(result.tokens).first(row));
        expect(actual.history_token_ids==expected.history.token_ids
            && actual.history_head==expected.history.head && actual.history_length==expected.history.length
            && actual.history_window_limit==expected.history.window_limit
            && actual.suffix_count==row && actual.suffix_token_ids==expected.suffix_token_ids,
            "device hypothetical histories differ from independent CPU ring/suffix equations");
    }
    refuses([&]{static_cast<void>(plan.verifier_inputs(2));},"stale device proposal lease accepted");
    auto incompatible=sampler.options();incompatible.repetition_penalty=1.1F;
    refuses([&]{plan.validate_sampler(incompatible);},"incompatible sampler silently routed");
}

void cancellation_boundary() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    prepare_recurrent_pipelines();
    std::vector<float> x(32,1), w(32,0), zeros(1,0);
    w[0]=0x1p24F;w[1]=1;w[2]=-0x1p24F;w[3]=0x1p-7F;
    auto input=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,x),runtime::BackendKind::metal);
    auto weight=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,w),runtime::BackendKind::metal);
    auto bias=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(zeros),runtime::BackendKind::metal);
    auto output=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(zeros),runtime::BackendKind::metal);
    auto status=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(zeros),runtime::BackendKind::metal);
    MetalCommandContext cmd("BF16 cancellation regression");
    const std::array<const tensor::BackendBuffer*,5> buffers {&input,&weight,&bias,&output,&status};
    refuses([&]{cmd.encode_recurrent_stage(RecurrentStage::linear,buffers,{33,1,0,0},{});},"undersized linear buffer accepted");
    cmd.encode_recurrent_stage(RecurrentStage::linear,buffers,{32,1,0,0},{});
    static_cast<void>(cmd.commit_and_wait());
    float actual=0;
    read_shared_metal_buffer_bytes(output,0,std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(&actual),4));
    expect(actual==1.0078125F,"BF16 cancellation crossed an exact rounding boundary");
}

void sampler_boundaries() {
    LearnedRecurrentSpec s;s.top_k=4;s.temperature=0.8F;s.top_p=1;s.min_p=0;
    std::array<float,4> equal {0,0,0,0};
    expect(recurrent_sample_cpu(equal,0,s)==0 && recurrent_sample_cpu(equal,0.25,s)==1,
        "CPU strict boundary/lower-token tie contract");
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    prepare_recurrent_pipelines();
    auto scores=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,4,equal),runtime::BackendKind::metal);
    auto alloc=[](unsigned n){return tensor::allocate_backend_buffer(tensor::TensorDescriptor::from(
        tensor::Shape::from({1,n}),tensor::DType::f32),runtime::BackendKind::metal);};
    auto a=alloc(8),b=alloc(8),status=alloc(1),tickets=alloc(2),tokens=alloc(2);
    for(double draw:{0.0,0.125,0.25,0.9}) {
        MetalCommandContext cmd("recurrent sample boundaries");cmd.write_u32(status,0);
        auto bits=std::bit_cast<std::uint64_t>(draw);
        const std::array words {static_cast<std::uint32_t>(bits),static_cast<std::uint32_t>(bits>>32)};
        cmd.write_u32s(tickets,0,words);
        auto sorted=cmd.encode_draft_stable_top_k(scores,a,b,status,1,4,4);cmd.encode_memory_barrier();
        const std::array<const tensor::BackendBuffer*,4> buffers {sorted.final_buffer_is_a?&a:&b,&tickets,&tokens,&status};
        cmd.encode_recurrent_stage(RecurrentStage::sample,buffers,{0,4,0,0},{0.8F,1,0,0});
        static_cast<void>(cmd.commit_and_wait());auto code=cmd.read_verifier_status(status);
        if(draw==0.25) expect(code==1,"ambiguous exact CDF boundary did not fail closed");
        else {
            expect(code==0,"unambiguous CDF decision refused");
            std::uint32_t token=UINT32_MAX;
            read_shared_metal_buffer_bytes(tokens,4,std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(&token),4));
            expect(token==recurrent_sample_cpu(equal,draw,s),"GPU sampler differs from CPU");
        }
    }
    // Exercise top-p/min-p before temperature, full retained support, stable
    // ties and wide score ranges. Refusal is allowed; a wrong token is not.
    unsigned decisions=0;
    for(const auto& values:std::array<std::array<float,4>,3>{{
            {0,0,0,0},{-30,30,29.9F,2},{1,0,-1,-2}}}) {
        auto input=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,4,values),runtime::BackendKind::metal);
        for(float temperature:{0.01F,0.8F,2.0F}) {
            for(float top_p:{0.5F,0.95F,1.0F}) {
                s.temperature=temperature;s.top_p=top_p;s.min_p=0.05F;
                for(unsigned i=0;i<16;++i) {
                    const double draw=(double(i)+0.173)/16.0;
                    MetalCommandContext cmd("recurrent sampler equation sweep");cmd.write_u32(status,0);
                    const auto bits=std::bit_cast<std::uint64_t>(draw);
                    const std::array words {static_cast<std::uint32_t>(bits),static_cast<std::uint32_t>(bits>>32)};
                    cmd.write_u32s(tickets,0,words);
                    auto sorted=cmd.encode_draft_stable_top_k(input,a,b,status,1,4,4);cmd.encode_memory_barrier();
                    const std::array<const tensor::BackendBuffer*,4> buffers {sorted.final_buffer_is_a?&a:&b,&tickets,&tokens,&status};
                    cmd.encode_recurrent_stage(RecurrentStage::sample,buffers,{0,4,0,0},{temperature,top_p,s.min_p,0});
                    static_cast<void>(cmd.commit_and_wait());const auto code=cmd.read_verifier_status(status);
                    expect(code<=1,"valid sampler sweep produced invalid-input status");
                    if(code==0) {
                        ++decisions;std::uint32_t token=UINT32_MAX;
                        read_shared_metal_buffer_bytes(tokens,4,std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(&token),4));
                        expect(token==recurrent_sample_cpu(values,draw,s),"interval sampler chose a wrong CPU bin");
                    }
                }
            }
        }
    }
    expect(decisions>=300,"interval sampler refused ordinary unambiguous decisions");
}

void lossless_finite_grid_equations() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    prepare_recurrent_pipelines();
    auto alloc=[](unsigned n){return tensor::allocate_backend_buffer(tensor::TensorDescriptor::from(
        tensor::Shape::from({1,n}),tensor::DType::f32),runtime::BackendKind::metal);};
    const auto check=[&](const std::vector<float>& values,std::uint32_t top_k,float top_p,
                         float min_p,float temperature,std::uint32_t expected_count) {
        auto scores=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(values),runtime::BackendKind::metal);
        auto a=alloc(static_cast<unsigned>(2*values.size())),b=alloc(static_cast<unsigned>(2*values.size()));
        auto status=alloc(1),words=alloc(1),tokens=alloc(2),support=alloc(769);
        std::array<std::uint32_t,769> previous {};bool first=true;
        // Include both sides of exact inverse-CDF boundaries. These must not
        // use the old interval sampler's draw-dependent conservative refusal.
        for(std::uint32_t word:{0U,1U,0x3fffffffU,0x40000000U,0x7fffffffU,0x80000000U,UINT32_MAX}) {
            MetalCommandContext cmd("lossless actual finite-grid draft law");
            cmd.write_u32(status,0);cmd.write_u32(words,word);
            const auto sorted=cmd.encode_draft_stable_top_k(scores,a,b,status,1,
                static_cast<std::uint32_t>(values.size()),top_k);
            cmd.encode_memory_barrier();
            const std::array<const tensor::BackendBuffer*,5> buffers {
                sorted.final_buffer_is_a?&a:&b,&words,&tokens,&status,&support};
            cmd.encode_recurrent_stage(RecurrentStage::sample_lossless,buffers,
                {0,top_k,0,0},{temperature,top_p,min_p,0});
            const auto stats=cmd.commit_and_wait();
            expect(cmd.read_verifier_status(status)==0,"valid lossless law refused a random word");
            expect(stats.transient_buffer_allocation_count==0,"lossless selector allocated transient storage");
            std::uint32_t selected=UINT32_MAX,count=0;
            read_shared_metal_buffer_bytes(tokens,4,std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(&selected),4));
            read_shared_metal_buffer_bytes(support,0,std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(&count),4));
            expect(count==expected_count,"top-p/min-p/underflow changed finite-grid support");
            std::array<std::uint32_t,769> raw {};
            read_shared_metal_buffer_bytes(support,0,std::span<std::uint8_t>(
                reinterpret_cast<std::uint8_t*>(raw.data()),(1+3*count)*4));
            if(!first) expect(raw==previous,"proposal law depends on proposal draw");
            previous=raw;first=false;
            std::uint64_t cumulative=0;DraftTokenDistribution distribution;distribution.count=count;
            double cpu_total=0,cpu_partial=0;
            const bool uniform=std::all_of(values.begin(),values.begin()+count,
                [&](float value){return value==values[0];});
            for(std::uint32_t i=0;i<count;++i) cpu_total+=std::exp(
                double(values[i]-values[0])/double(temperature));
            for(std::uint32_t i=0;i<count;++i) {
                const auto boundary=std::uint64_t(raw[2+3*i])|(std::uint64_t(raw[3+3*i])<<32);
                expect(raw[1+3*i]==i,"stable top-k proposal IDs differ from CPU score order");
                if(uniform) {
                    // Equal retained logits have exactly equal integer bins.
                    const auto expected_boundary=(draft_probability_mass_total/count)*(i+1);
                    expect(boundary==expected_boundary,"CPU analytic uniform-bin equation differs from Metal");
                } else {
                    cpu_partial+=std::exp(double(values[i]-values[0])/double(temperature));
                    // Precise Metal exp permits four ulps. Bound both sum
                    // roundoffs, division and CDF quantization independently;
                    // this is parity evidence, not the q used for correction.
                    const auto bound=double(8*count)*std::numeric_limits<float>::epsilon()
                        +1.0/double(draft_probability_mass_total);
                    expect(std::abs(double(boundary)/double(draft_probability_mass_total)-cpu_partial/cpu_total)<=bound,
                        "CPU exponential/CDF equation differs from Metal beyond operation error bound");
                }
                distribution.values[i]={raw[1+3*i],boundary-cumulative};cumulative=boundary;
            }
            expect(cumulative==draft_probability_mass_total,"draft mass not exactly normalized");
            if(uniform) expect(selected==word/(draft_probability_mass_total/count),"GPU boundary chose wrong uniform bin");
            expect(selected==select_draft_token(distribution,word),"CPU integer-CDF inversion differs from Metal");
        }
    };
    check({0,0,0,0},4,1,0,0.8F,4);
    check({0,0,0,0},4,0.5F,0,0.8F,2);
    check({0,0,0,0},4,0.01F,0,0.8F,1);
    check({0,0,-1000,-1000},4,1,0,0.0001F,2);
    check({0,0,-1000,-1000},4,1,0.5F,0.8F,2);
    check({0,-1000,-1000,-1000},4,1,0,0.8F,1);
    check(std::vector<float>(257,0.0F),256,1,0,0.8F,256);
    check({3,2,0,-7},4,1,0,0.8F,4);
    check({3,2,0,-7},4,0.95F,0,0.8F,2);
    check({3,2,0,-7},4,1,0.05F,2.0F,2);

    auto a=alloc(8),words=alloc(1),tokens=alloc(2),status=alloc(1),support=alloc(768);
    const std::array<const tensor::BackendBuffer*,5> short_buffers {&a,&words,&tokens,&status,&support};
    MetalCommandContext cmd("lossless proposal capacity validation");
    refuses([&]{cmd.encode_recurrent_stage(RecurrentStage::sample_lossless,short_buffers,
        {0,4,0,0},{0.8F,1,0,0});},"undersized probability arena accepted");
    refuses([&]{cmd.encode_recurrent_stage(RecurrentStage::sample_lossless,short_buffers,
        {23,4,0,0},{0.8F,1,0,0});},"lossless proposal horizon overflow accepted");
}

void lossless_live_chain() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    for(unsigned horizon:{3U,4U,7U,8U,15U,16U,23U}) {
        Fixture f(horizon);auto artifact=load_speculative_artifact(f.encode());
        auto table=tensor::copy_tensor_to_backend(f.table,runtime::BackendKind::metal);
        auto norm=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(f.norm),runtime::BackendKind::metal);
        auto feature=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,f.feature),runtime::BackendKind::metal);
        auto plan=Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm);
        TokenSampler sampler({.temperature=0.8F,.top_k=8,.seed=71});
        const auto checkpoint=sampler.checkpoint();const auto preview=sampler.preview_proposal_words(horizon);
        const auto words=std::span(preview.words).first(horizon);
        TargetStateView origin {.execution_epoch=1,.source_stage=FeatureTapStage::final_normalized,
            .coordinate={4,5},.row_count=1,.feature_width=32};
        auto wrong=preview;wrong.words[0]^=1U;
        refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,
            std::span(wrong.words).first(horizon)));},"foreign lossless proposal words accepted");
        refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,
            words.first(horizon-1)));},"short lossless proposal word pool accepted");
        const auto actual=plan.execute_lossless(feature,origin,7,sampler,words);
        expect(actual.status==0 && actual.count==horizon && actual.has_distributions,
            "lossless recurrent did not return complete actual proposal laws");
        expect(actual.transient_buffer_allocations==0 && actual.execution.command_buffer_count==1,
            "lossless chain did not retain one-command fixed-storage execution");
        expect(sampler.checkpoint().same_state(checkpoint),"draft advanced sampler random stream");
        std::array<double,23> ideal_draws {};
        for(unsigned i=0;i<horizon;++i) {
            const auto& law=actual.proposal_distributions[i];
            expect(law.count>0 && law.count<=8 && select_draft_token(law,words[i])==actual.tokens[i],
                "published proposal differs from actual integer law");
            ideal_draws[i]=double(words[i])/double(draft_probability_mass_total);
        }
        // The native proposal law is explicitly quantized, not the old CPU
        // ideal law; these ordinary non-boundary inputs independently check
        // that the unchanged recurrent equations still produce the same path.
        const auto golden=recurrent_cpu_golden(artifact,f.table.view(),f.norm,f.feature,7,
            std::span(ideal_draws).first(horizon));
        expect(std::equal(golden.tokens.begin(),golden.tokens.end(),actual.tokens.begin()),
            "lossless ordinary recurrent chain differs from independent CPU equations");
        const auto states=plan.diagnostic_target_states();
        for(std::size_t i=0;i<states.size();++i) tests::expect_close(states[i],golden.target_states[i],
            0.00001F,"lossless recurrent state CPU/Metal parity");
        const auto binding=plan.verifier_inputs(1);
        expect(binding.arena_epoch==1,"lossless proposals lack verifier lease");
        refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,words));},
            "stale lossless feature epoch accepted");
        ++origin.execution_epoch;
        const auto repeated=plan.execute_lossless(feature,origin,7,sampler,words);
        expect(repeated.tokens==actual.tokens,"uncommitted lossless preview is not reproducible");
        fill_shared_metal_buffer_u32(feature,0x7fc00000U);++origin.execution_epoch;
        refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,words));},
            "lossless nonfinite arithmetic silently fell back");
        expect(plan.poisoned(),"lossless malformed arithmetic did not poison plan");
        refuses([&]{static_cast<void>(plan.verifier_inputs(2));},"poisoned lossless proposal lease survived");
    }
}

void lossless_quantized_head_chain() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    for(unsigned horizon:{3U,4U,7U,8U,15U,16U,23U}) {
        Fixture f(horizon);
        // Retain all top-k entries so this equation test isolates projection,
        // normalization and the actual finite-grid law. Truncation boundaries
        // have their independent tests above, shared by both head variants.
        f.a.recurrent.top_p=1;f.a.recurrent.min_p=0;
        auto artifact=load_speculative_artifact(f.encode());
        auto packed=tensor::quantize_f16_gguf_weight_q8_0(f.table.view());
        const auto packed_view=packed.view();
        auto table=tensor::copy_tensor_to_backend(f.table,runtime::BackendKind::metal);
        auto head=tensor::copy_tensor_to_backend(packed,runtime::BackendKind::metal);
        auto norm=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(f.norm),runtime::BackendKind::metal);
        auto feature=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,f.feature),runtime::BackendKind::metal);
        auto plan=Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,RecurrentDraftHead::q8_0,&head);
        auto control=Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm);
        expect(plan.draft_head()==RecurrentDraftHead::q8_0 && control.draft_head()==RecurrentDraftHead::f16,
            "recurrent projection variant lost its explicit identity");
        expect(plan.owned_bytes()==control.owned_bytes(),"aliased Q8 head was counted as new owned storage");
        TokenSampler sampler({.temperature=0.8F,.top_k=8,.top_p=1,.min_p=0,.seed=71});
        const auto checkpoint=sampler.checkpoint();const auto preview=sampler.preview_proposal_words(horizon);
        const auto words=std::span(preview.words).first(horizon);
        TargetStateView origin {.execution_epoch=1,.source_stage=FeatureTapStage::final_normalized,
            .coordinate={4,5},.row_count=1,.feature_width=32};
        std::array<double,23> draws {};
        for(unsigned i=0;i<horizon;++i) draws[i]=double(words[i])/double(draft_probability_mass_total);
        refuses([&]{static_cast<void>(plan.execute_development(feature,origin,7,std::span(draws).first(horizon)));},
            "quantized head silently changed the old development control");
        refuses([&]{static_cast<void>(plan.execute_live(feature,origin,7,sampler,sampler.preview_tickets(horizon)));},
            "quantized head entered the old common-draw live route");
        expect(!plan.poisoned(),"invalid Q8 mode poisoned an unused plan");
        const auto actual=plan.execute_lossless(feature,origin,7,sampler,words);
        expect(actual.status==0 && actual.count==horizon && actual.has_distributions,
            "quantized lossless chain did not publish complete proposal laws");
        expect(actual.execution.q8_decode_projection_count==horizon && actual.execution.f16_projection_count==0,
            "quantized draft projection accounting reports the wrong head");
        expect(actual.transient_buffer_allocations==0 && actual.execution.command_buffer_count==1,
            "quantized full-vocabulary chain allocates or stages dependent tokens on host");
        expect(sampler.checkpoint().same_state(checkpoint),"quantized draft consumed authoritative randomness");
        const auto golden=recurrent_cpu_golden(artifact,f.table.view(),f.norm,f.feature,7,
            std::span(draws).first(horizon),&packed_view);
        const auto f16_golden=recurrent_cpu_golden(artifact,f.table.view(),f.norm,f.feature,7,
            std::span(draws).first(horizon));
        expect(!std::equal(golden.logits.begin(),golden.logits.begin()+65,f16_golden.logits.begin()),
            "Q8 head fixture did not exercise quantization differences from the F16 control");
        expect(std::equal(golden.tokens.begin(),golden.tokens.end(),actual.tokens.begin()),
            "Q8 dependent path differs from independent CPU dequantization equations");
        const auto states=plan.diagnostic_target_states();
        for(std::size_t i=0;i<states.size();++i) tests::expect_close(states[i],golden.target_states[i],
            0.00001F,"Q8 recurrent state CPU/Metal parity");
        for(unsigned depth=0;depth<horizon;++depth) {
            const auto logits=std::span(golden.logits).subspan(depth*65,65);
            std::array<unsigned,65> order {};std::iota(order.begin(),order.end(),0U);
            std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b) {
                return logits[a]>logits[b] || (logits[a]==logits[b] && a<b);
            });
            const auto& law=actual.proposal_distributions[depth];
            expect(law.count==8 && select_draft_token(law,words[depth])==actual.tokens[depth],
                "Q8 proposal law is not its actual inverse-CDF law");
            double squared=0;
            for(unsigned i=0;i<32;++i) squared+=double(golden.target_states[depth*32+i])*golden.target_states[depth*32+i];
            const double normalization=1/std::sqrt(squared/32+f.a.target.config.rms_norm_epsilon);
            double largest_l1_score=0;
            for(unsigned token=0;token<65;++token) {
                const auto at=token*34;
                const auto scale=tensor::f16_bits_to_f32(static_cast<std::uint16_t>(
                    std::uint16_t(packed.bytes()[at])|(std::uint16_t(packed.bytes()[at+1])<<8)));
                double l1=0;
                for(unsigned i=0;i<32;++i) l1+=std::abs(double(golden.target_states[depth*32+i])*normalization
                    *scale*std::bit_cast<std::int8_t>(packed.bytes()[at+2+i]));
                largest_l1_score=std::max(largest_l1_score,l1);
            }
            // gamma(4*d+16) bounds RMS reduction, two scaled products,
            // accumulation and softcap evaluation for this 32-wide fixture.
            // Softmax perturbs a subset probability by at most exp(4*eta/T)-1;
            // add the selector's exp/sum/divide roundoff and integer CDF grid.
            const double n_epsilon=(4*32+16)*double(std::numeric_limits<float>::epsilon());
            const double eta=n_epsilon/(1-n_epsilon)*std::max(1.0,largest_l1_score);
            const double cdf_bound=std::expm1(4*eta/f.a.recurrent.temperature)
                +8*law.count*double(std::numeric_limits<float>::epsilon())+1.0/draft_probability_mass_total;
            double total=0,partial=0;std::uint64_t cumulative=0;
            for(unsigned i=0;i<8;++i) total+=std::exp(double(logits[order[i]]-logits[order[0]])/f.a.recurrent.temperature);
            for(unsigned i=0;i<8;++i) {
                expect(law.values[i].token==order[i],"Q8 full-vocabulary support differs from independent CPU projection");
                partial+=std::exp(double(logits[order[i]]-logits[order[0]])/f.a.recurrent.temperature);
                cumulative+=law.values[i].probability_mass;
                expect(std::abs(double(cumulative)/draft_probability_mass_total-partial/total)<=cdf_bound,
                    "Q8 actual proposal CDF exceeds independent numerical error budget");
            }
            expect(cumulative==draft_probability_mass_total,"Q8 proposal mass is not exactly normalized");
        }
        expect(plan.verifier_inputs(1).arena_epoch==1,"Q8 proposal did not publish a verifier lease");
        refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,words));},
            "Q8 draft accepted stale provenance");
        ++origin.execution_epoch;
        expect(plan.execute_lossless(feature,origin,7,sampler,words).tokens==actual.tokens,
            "uncommitted Q8 proposal preview is not reproducible");
        fill_shared_metal_buffer_u32(feature,0x7fc00000U);++origin.execution_epoch;
        refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,words));},
            "Q8 draft invalid arithmetic silently fell back");
        expect(plan.poisoned(),"Q8 arithmetic failure did not poison draft state");
        refuses([&]{static_cast<void>(plan.verifier_inputs(2));},"Q8 poisoned lease survived");
    }
}

void quantized_head_configuration() {
    if(!tensor::tensor_backend_is_available(runtime::BackendKind::metal)) return;
    Fixture f;auto artifact=load_speculative_artifact(f.encode());
    auto table=tensor::copy_tensor_to_backend(f.table,runtime::BackendKind::metal);
    auto norm=tensor::copy_tensor_to_backend(tests::tensor_f32_vector(f.norm),runtime::BackendKind::metal);
    refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
        RecurrentDraftHead::q8_0));},"missing Q8 draft head silently fell back to F16");
    refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
        RecurrentDraftHead::q8_0,&table));},"F16 head accepted as packed Q8");
    refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
        static_cast<RecurrentDraftHead>(2)));},"unknown draft head accepted");
    auto packed=tensor::quantize_f16_gguf_weight_q8_0(f.table.view());
    auto head=tensor::copy_tensor_to_backend(packed,runtime::BackendKind::metal);
    refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
        RecurrentDraftHead::f16,&head));},"F16 route accepted a silently substituted Q8 head");
    auto wrong_shape=tensor::allocate_backend_buffer(tensor::TensorDescriptor::from(
        tensor::Shape::from({32,64}),tensor::DType::q8_0),runtime::BackendKind::metal);
    refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
        RecurrentDraftHead::q8_0,&wrong_shape));},"reduced-vocabulary Q8 head accepted");
    auto cpu_head=tensor::copy_tensor_to_backend(packed,runtime::BackendKind::cpu);
    refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
        RecurrentDraftHead::q8_0,&cpu_head));},"CPU draft head silently substituted for resident Metal");
    set_metal_device_unavailable_for_testing(true);
    try {
        refuses([&]{static_cast<void>(Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,
            RecurrentDraftHead::q8_0,&head));},"Q8 plan bypassed unavailable Metal");
    } catch(...) {set_metal_device_unavailable_for_testing(false);throw;}
    set_metal_device_unavailable_for_testing(false);
    // A malformed packed block must fail through device arithmetic validation,
    // never publish a proposal or recover by switching to the F16 table.
    packed.mutable_bytes()[0]=0;packed.mutable_bytes()[1]=0x7e;
    auto malformed=tensor::copy_tensor_to_backend(packed,runtime::BackendKind::metal);
    auto plan=Gemma2SpeculativePlan::fixture_for_testing(artifact,table,norm,RecurrentDraftHead::q8_0,&malformed);
    auto feature=tensor::copy_tensor_to_backend(tests::tensor_f32_matrix(1,32,f.feature),runtime::BackendKind::metal);
    TokenSampler sampler({.temperature=0.8F,.top_k=8,.seed=71});
    const auto words=sampler.preview_proposal_words(3);
    TargetStateView origin {.execution_epoch=1,.source_stage=FeatureTapStage::final_normalized,
        .coordinate={4,5},.row_count=1,.feature_width=32};
    refuses([&]{static_cast<void>(plan.execute_lossless(feature,origin,7,sampler,
        std::span(words.words).first(3)));},"malformed Q8 scale silently selected a draft token");
    expect(plan.poisoned(),"malformed Q8 scale did not poison draft plan");
    refuses([&]{static_cast<void>(plan.verifier_inputs(1));},"malformed Q8 head published a lease");
}
}

namespace local_gguf_engine::tests {
std::span<const TestCase> learned_recurrent_test_cases() {
    static const std::array<TestCase,11> cases {{
        {"trained recurrent artifact validates complete numeric and weight contract",artifact_contract},
        {"trained recurrent source detects mutation after opening",file_lifetime},
        {"trained recurrent dependent CPU/Metal rollout and lifetime parity",full_rollout},
        {"trained recurrent common-draw boundaries fail closed",sampler_boundaries},
        {"trained recurrent fails closed without a Metal device",unavailable_device},
        {"trained recurrent preserves BF16 cancellation and rejects capacity overflow",cancellation_boundary},
        {"live recurrent opaque tickets and device hypothetical histories match CPU",live_tickets_and_histories},
        {"lossless recurrent actual grid law matches CPU boundaries and underflow",lossless_finite_grid_equations},
        {"lossless recurrent full chains preserve probabilities and transactional lifetime",lossless_live_chain},
        {"lossless Q8 recurrent chains match independent dequantized equations at every horizon",lossless_quantized_head_chain},
        {"lossless Q8 recurrent head selection rejects incompatible storage and unavailable Metal",quantized_head_configuration},
    }};return cases;
}
}
