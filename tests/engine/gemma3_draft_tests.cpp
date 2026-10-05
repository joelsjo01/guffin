#include "local_gguf_engine/engine/gemma3_draft.hpp"
#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "metal_test_hooks.hpp"
#include "test_support.hpp"
#include "gemma3_golden_hashes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace local_gguf_engine::tests {
namespace {
using namespace engine;
constexpr std::uint32_t width=640,ff=2048,vocab=262144,layers=18;
bool bit_equal(std::span<const float> a,std::span<const float> b){
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size_bytes())==0);
}
template<class F>void refuses(F&& f,const char* label){bool failed=false;
    try{f();}catch(const std::exception&){failed=true;}expect(failed,label);}
bool have_metal(){const auto available=tensor::tensor_backend_is_available(runtime::BackendKind::metal);
    if(!available)std::cout<<"[GEMMA3 METAL] unavailable; no production CPU fallback\n";return available;}
float norm_weight(std::size_t col){return 1.0F+float(int(col%7)-3)/64;}
float embedding_weight(std::size_t row,std::size_t col){return float(int((row*17+col*13)%53)-26)/64;}

// Exact complete geometry, Q8 layout, and all18 nonzero transformer layers.
// Sparse known matrices make an independent scalar oracle economical without
// replacing any production operation with a reduced model.
struct Fixture {
    std::vector<tensor::TensorStorage> storage;
    std::vector<tensor::TensorView> views;
    std::vector<std::int32_t> map;
    Fixture():map(vocab){
        std::iota(map.begin(),map.end(),0);
        const auto add=[&](std::uint32_t in,std::uint32_t out,int matrix){
            auto t=tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
                out?tensor::Shape::from({in,out}):tensor::Shape::from({in}),out?tensor::DType::q8_0:tensor::DType::f32));
            if(!out){auto* w=reinterpret_cast<float*>(t.mutable_bytes().data());
                for(std::uint32_t j=0;j<in;++j)w[j]=norm_weight(j);
            }else{
                auto* bytes=reinterpret_cast<std::uint8_t*>(t.mutable_bytes().data());std::memset(bytes,0,t.mutable_bytes().size());
                const auto scale=tensor::f32_to_f16_bits(1.0F/64);
                for(std::uint32_t row=0;row<out;++row){
                    for(std::uint32_t block=0;block<in/32;++block)
                        std::memcpy(bytes+(std::size_t(row)*(in/32)+block)*34,&scale,2);
                    if(matrix<0){for(std::uint32_t col=0;col<in;++col){
                        const auto value=static_cast<std::int8_t>(int((std::size_t(row)*17+col*13)%53)-26);
                        std::memcpy(bytes+(std::size_t(row)*(in/32)+col/32)*34+2+col%32,&value,1);
                    }}else{
                        const auto col=(row*7+11+std::uint32_t(matrix)*3)%in;
                        bytes[(std::size_t(row)*(in/32)+col/32)*34+2+col%32]=8;
                    }
                }
            }
            storage.push_back(std::move(t));
        };
        add(width,vocab,-1);add(width,0,0);
        constexpr std::array<std::uint32_t,7> in={width,width,width,1024,width,width,ff};
        constexpr std::array<std::uint32_t,7> out={1024,256,256,width,ff,ff,width};
        for(unsigned layer=0;layer<layers;++layer){
            for(unsigned m=0;m<7;++m)add(in[m],out[m],int(m));
            for(unsigned n=0;n<4;++n)add(width,0,0);add(256,0,0);add(256,0,0);
        }
        for(const auto& t:storage)views.push_back(t.view());
    }
};
const Fixture& fixture(){static const Fixture value;return value;}

struct Reference {
    std::array<std::vector<float>,layers> keys,values;
    std::vector<float> hidden;
    std::uint32_t position=0;
    static std::vector<float> norm(const std::vector<float>& input,std::size_t length){
        std::vector<float> output(input.size());
        for(std::size_t row=0;row<input.size()/length;++row){double sum=0;
            for(std::size_t j=0;j<length;++j)sum+=double(input[row*length+j])*input[row*length+j];
            const auto inv=1.0/std::sqrt(sum/double(length)+1.e-6);
            for(std::size_t j=0;j<length;++j)output[row*length+j]=float(double(input[row*length+j])*inv*norm_weight(j));
        }return output;
    }
    static std::vector<float> linear(const std::vector<float>& input,std::size_t output,unsigned matrix){
        std::vector<float> result(output);
        for(std::size_t row=0;row<output;++row)result[row]=0.125F*input[(row*7+11+matrix*3)%input.size()];
        return result;
    }
    static void add(std::vector<float>& input,const std::vector<float>& branch){
        for(std::size_t j=0;j<input.size();++j)input[j]+=branch[j];
    }
    void step(std::uint32_t token,bool zero_prefix=false){
        std::vector<float> x(width);for(std::size_t j=0;j<width;++j)x[j]=embedding_weight(token,j)*std::sqrt(float(width));
        for(unsigned layer=0;layer<layers;++layer){
            const auto n=norm(x,width);auto q=norm(linear(n,1024,0),256),k=norm(linear(n,256,1),256);
            const auto v=linear(n,256,2);const bool local=layer%6!=5;
            for(auto* value:{&q,&k})for(std::size_t head=0;head<value->size()/256;++head)
                for(std::size_t j=0;j<128;++j){
                    const double angle=double(position)*std::pow(local?10000.0:1000000.0,-double(j)/128);
                    const auto one=(*value)[head*256+j],two=(*value)[head*256+j+128];
                    (*value)[head*256+j]=float(double(one)*std::cos(angle)-double(two)*std::sin(angle));
                    (*value)[head*256+j+128]=float(double(one)*std::sin(angle)+double(two)*std::cos(angle));
                }
            for(std::size_t j=0;j<256;++j){keys[layer].push_back(k[j]);values[layer].push_back(v[j]);}
            std::vector<float> attended(1024);const auto first=local&&position>=511?position-511:0;
            for(std::size_t head=0;head<4;++head){
                if(zero_prefix){
                    double score=0;for(std::size_t j=0;j<256;++j)
                        score+=double(q[head*256+j])*keys[layer][j];
                    const double current=1.0/(1.0+double(position-first)*std::exp(-score/16));
                    for(std::size_t j=0;j<256;++j)attended[head*256+j]=float(current*values[layer][j]);
                    continue;
                }
                std::vector<double> scores(position+1-first);double maximum=-std::numeric_limits<double>::infinity();
                for(std::uint32_t t=first;t<=position;++t){double sum=0;
                    for(std::size_t j=0;j<256;++j)sum+=double(q[head*256+j])*keys[layer][t*256+j];
                    scores[t-first]=sum/16;maximum=std::max(maximum,scores[t-first]);
                }
                double total=0;for(auto& value:scores){value=std::exp(value-maximum);total+=value;}
                for(std::size_t j=0;j<256;++j){double sum=0;
                    for(std::uint32_t t=first;t<=position;++t)
                        sum+=scores[t-first]/total*values[layer][t*256+j];
                    attended[head*256+j]=float(sum);
                }
            }
            add(x,norm(linear(attended,width,3),width));const auto nf=norm(x,width);
            auto gate=linear(nf,ff,4);const auto up=linear(nf,ff,5);
            for(std::size_t j=0;j<ff;++j){double z=gate[j];
                gate[j]=float(0.5*z*(1+std::tanh(std::sqrt(2.0/std::acos(-1.0))*(z+0.044715*z*z*z)))*up[j]);}
            add(x,norm(linear(gate,width,6),width));
        }
        hidden=norm(x,width);++position;
    }
    std::vector<float> logits()const{
        std::array<float,53> distinct{};for(std::size_t row=0;row<53;++row){double sum=0;
            for(std::size_t j=0;j<width;++j)sum+=double(hidden[j])*embedding_weight(row,j);distinct[row]=float(sum);}
        std::vector<float> result(vocab);for(std::size_t i=0;i<result.size();++i)result[i]=distinct[i%53];return result;
    }
};
void close_logits(std::span<const float> actual,std::span<const float> expected,const std::string& label){
    expect(actual.size()==expected.size(),"Gemma3 reference logit shape differs");double maximum=0;
    for(std::size_t i=0;i<actual.size();++i){const double error=std::abs(double(actual[i])-expected[i]);
        maximum=std::max(maximum,error);
        if(!std::isfinite(actual[i])||error>0.03+0.001*std::abs(double(expected[i])))
            throw TestFailure(label+" token "+std::to_string(i)+" error "+std::to_string(error));
    }
    std::cout<<"[GEMMA3 PARITY] "<<label<<" max_abs="<<maximum<<'\n';
}
void cache_close(std::span<const float> actual,std::span<const float> expected,const char* label){
    expect(actual.size()==expected.size(),"Gemma3 cache reference shape differs");
    for(std::size_t i=0;i<actual.size();++i){
        const auto a=actual[i],b=expected[i];
        if(!std::isfinite(a)||std::abs(a-b)>0.004F+0.002F*std::abs(b))
            throw TestFailure(std::string(label)+" element "+std::to_string(i)+" actual "+std::to_string(a)
                +" expected "+std::to_string(b)+" error "+std::to_string(std::abs(a-b)));
    }
}
void synthetic_forward(){
    if(!have_metal())return;const auto& f=fixture();Gemma3DraftPlan plan(f.views,f.map,vocab,23);
    constexpr std::array<std::uint32_t,4> tokens={2,9259,12074,108};Reference reference;
    (void)plan.append(std::span(tokens).first(3));for(auto token:std::span(tokens).first(3))reference.step(token);
    std::vector<float> actual(vocab),replayed(vocab);(void)plan.read_logits(actual);
    close_logits(actual,reference.logits(),"synthetic prefill independent equations");
    (void)plan.append(std::span(tokens).last(1));reference.step(tokens.back());(void)plan.read_logits(actual);
    close_logits(actual,reference.logits(),"synthetic decode independent equations");
    for(unsigned layer=0;layer<layers;++layer){std::vector<float> k(4*256),v(k.size());
        plan.read_cache(layer,0,k,v);cache_close(k,reference.keys[layer],"synthetic K");cache_close(v,reference.values[layer],"synthetic V");}
    plan.reset();for(auto token:tokens)(void)plan.append(std::span(&token,1));(void)plan.read_logits(replayed);
    expect(bit_equal(actual,replayed),"Gemma3 scalar and batched replay differ");
}

std::vector<std::uint32_t> stream(std::size_t size){
    std::vector<std::uint32_t> result(size);for(std::size_t i=0;i<size;++i)result[i]=i?3+std::uint32_t(i*499%255997):2;return result;
}
struct CacheSnapshot {
    std::array<std::vector<float>,layers> keys,values;
    std::array<std::uint32_t,layers> first{};
    explicit CacheSnapshot(const Gemma3DraftPlan& plan){
        for(unsigned layer=0;layer<layers;++layer){
            const auto count=layer%6!=5?std::min(plan.materialized_position(),512U):plan.materialized_position();
            first[layer]=plan.materialized_position()-count;keys[layer].resize(count*256);values[layer].resize(count*256);
            plan.read_cache(layer,first[layer],keys[layer],values[layer]);
        }
    }
    void matches(const Gemma3DraftPlan& plan)const{
        for(unsigned layer=0;layer<layers;++layer){
            std::vector<float> k(keys[layer].size()),v(values[layer].size());
            plan.read_cache(layer,first[layer],k,v);
            expect(bit_equal(k,keys[layer])&&bit_equal(v,values[layer]),"Gemma3 transaction changed committed K/V");
        }
    }
};
void transaction_lifecycle(){
    if(!have_metal())return;const auto& f=fixture();Gemma3DraftPlan plan(f.views,f.map,vocab,23,96);
    auto prompt=stream(510);(void)plan.append(prompt);CacheSnapshot before(plan);
    TokenSamplerOptions options;options.temperature=0;SamplerHistorySnapshot history;history.window_limit=64;
    std::array<std::uint32_t,23> words{};const std::uint32_t root=55,next_root=719;
    (void)plan.begin_transaction();(void)plan.append(std::span(&root,1));
    (void)plan.propose(root,std::span(words).first(3),options,history);
    plan.abort_transaction();expect(plan.position()==510,"Gemma3 abort moved prefix");before.matches(plan);
    for(std::uint32_t accepted=0;accepted<=3;++accepted){
        (void)plan.begin_transaction();(void)plan.append(std::span(&root,1));
        const auto proposal=plan.propose(root,std::span(words).first(3),options,history);
        expect(proposal.execution.command_buffer_count==1&&proposal.execution.token_count==2,
            "Gemma3 H3 performed per-token synchronization or forwarded unused final token");
        expect(plan.position()==514&&plan.materialized_position()==513,"Gemma3 pending frontier is wrong");
        std::array<std::uint32_t,3> proposed;std::copy_n(proposal.draft_tokens.begin(),3,proposed.begin());
        {
            (void)plan.commit_prefix(511+accepted);
            if(accepted==3){
                CacheSnapshot pending_cache(plan);const auto old_materialized=plan.materialized_position();
                (void)plan.begin_transaction();(void)plan.append(std::span(&next_root,1));
                (void)plan.propose(next_root,std::span(words).first(1),options,history);plan.abort_transaction();
                expect(plan.position()==514&&plan.materialized_position()==old_materialized,
                    "Gemma3 abort failed to restore the previous committed pending token");
                pending_cache.matches(plan);
            }
            (void)plan.append(std::span(&next_root,1));
            std::vector<float> actual(vocab),expected(vocab);(void)plan.read_logits(actual);CacheSnapshot committed(plan);
            auto replay=prompt;replay.push_back(root);replay.insert(replay.end(),proposed.begin(),proposed.begin()+accepted);replay.push_back(next_root);
            plan.reset();(void)plan.append(replay);(void)plan.read_logits(expected);
            expect(bit_equal(actual,expected),"Gemma3 accepted/pending prefix and different next root changed replay logits");committed.matches(plan);
            plan.reset();(void)plan.append(prompt);
        }
    }
    // Truncate an encoded speculative suffix before appending a correction.
    (void)plan.begin_transaction();(void)plan.append(std::span(&root,1));
    const auto correction_proposals=plan.propose(root,std::span(words).first(3),options,history);
    const auto accepted_token=correction_proposals.draft_tokens[0];plan.truncate(512);
    const std::uint32_t correction=111;(void)plan.append(std::span(&correction,1));(void)plan.commit_prefix(513);
    expect(!plan.can_truncate(512),"Gemma3 allowed published ring truncation");
    refuses([&]{plan.truncate(512);},"Gemma3 erased durable ring logically");
    std::vector<float> correction_logits(vocab),correction_replay(vocab);(void)plan.read_logits(correction_logits);
    CacheSnapshot corrected(plan);auto corrected_tokens=prompt;
    corrected_tokens.insert(corrected_tokens.end(),{root,accepted_token,correction});
    plan.reset();(void)plan.append(corrected_tokens);(void)plan.read_logits(correction_replay);
    expect(bit_equal(correction_logits,correction_replay),"Gemma3 truncated suffix/correction changed replay");corrected.matches(plan);
    plan.reset();(void)plan.append(std::span(&root,1));
    for(auto h:{1U,2U,3U,5U,23U}){
        (void)plan.begin_transaction();const auto proposal=plan.propose(root,std::span(words).first(h),options,history);
        expect(proposal.draft_tokens.size()==h&&plan.position()==h+1,"Gemma3 narrowed requested horizon");
        plan.abort_transaction();expect(plan.position()==1,"Gemma3 horizon abort lost prefix");
    }
    (void)plan.begin_transaction();
    refuses([&]{(void)plan.begin_transaction();},"Gemma3 nested transaction admitted");
    const auto oversized=stream(97);refuses([&]{(void)plan.append(oversized);},"Gemma3 private cache overflow admitted");
    expect(plan.position()==1,"Gemma3 overflow mutated state");plan.abort_transaction();
    const auto oversized_context=stream(Gemma3DraftPlan::context_capacity);
    refuses([&]{(void)plan.append(oversized_context);},"Gemma3 context overflow admitted");
    const std::uint32_t invalid=vocab;refuses([&]{(void)plan.append(std::span(&invalid,1));},"Gemma3 invalid token admitted");
    plan.reset();(void)plan.append(std::span(&root,1));
    const auto restored=plan.restore_pending_token(correction);
    expect(restored.token_count==0&&plan.position()==2&&plan.materialized_position()==1,
        "Gemma3 saved pending token was eagerly forwarded");
    refuses([&]{(void)plan.restore_pending_token(correction);},"Gemma3 accepted a second pending token");
    (void)plan.begin_transaction();
    refuses([&]{(void)plan.restore_pending_token(root);},"Gemma3 restored pending state inside a transaction");
    (void)plan.append(std::span(&next_root,1));plan.abort_transaction();
    expect(plan.position()==2&&plan.materialized_position()==1,"Gemma3 pending checkpoint abort changed frontier");
    (void)plan.append(std::span(&next_root,1));(void)plan.read_logits(correction_logits);
    plan.reset();const std::array<std::uint32_t,3> saved_tokens={root,correction,next_root};
    (void)plan.append(saved_tokens);(void)plan.read_logits(correction_replay);
    expect(bit_equal(correction_logits,correction_replay),"Gemma3 restored pending token changed future logits");
}

PreparedTokenDistribution target_tokens(std::uint32_t first,std::uint32_t count){
    PreparedTokenDistribution target;target.count=count;
    for(std::uint32_t i=0;i<count;++i)target.values[i]={first+i,1,1};return target;
}
void probability_law(){
    if(!have_metal())return;const auto& f=fixture();Gemma3DraftPlan plan(f.views,f.map,vocab,3);
    std::vector<float> logits(vocab,0);TokenSamplerOptions o;o.temperature=1;o.top_k=1;o.top_p=1;o.min_p=0;o.repetition_penalty=1;
    // All262144 cutoff ties survive top-k1; each exact interval is16384 words.
    plan.sample_logits_for_testing(logits,std::numeric_limits<std::uint32_t>::max(),o);
    std::uint64_t total=0;
    for(std::uint32_t first=0;first<vocab;first+=256){const auto q=plan.gather(0,target_tokens(first,256));
        for(unsigned i=0;i<256;++i){expect(q.target_values[i].probability_mass==16384,"Gemma3 truncated full-vocabulary ties");
            total+=q.target_values[i].probability_mass;}
        expect(q.proposal.token==vocab-1&&q.proposal_cdf_begin==draft_probability_mass_total-16384,
            "Gemma3 final CDF interval is wrong");
    }
    expect(total==draft_probability_mass_total,"Gemma3 full law is not normalized");
    auto stale=plan.gather(0,target_tokens(0,2));o.top_p=0.5F;
    plan.sample_logits_for_testing(logits,0,o);
    expect(!stale.lease.valid(),"Gemma3 overwritten probability law retained its lease");
    auto first=plan.gather(0,target_tokens(0,2)),last=plan.gather(0,target_tokens(vocab-2,2));
    expect(first.target_values[0].probability_mass==32768&&last.target_values[0].probability_mass==0,
        "Gemma3 ascending-tail nucleus filtering changed tied support");
    o.top_p=std::numeric_limits<float>::denorm_min();plan.sample_logits_for_testing(logits,0,o);
    auto tiny=plan.gather(0,target_tokens(0,2));expect(tiny.proposal.token==0&&tiny.proposal.probability_mass==draft_probability_mass_total,
        "Gemma3 tiny nucleus resurrected filtered logits");
    // Repetition must use retained native history beyond the target64 window.
    auto native=stream(70);native[0]=37;for(std::size_t i=1;i<native.size();++i)native[i]=2;
    (void)plan.append(native);std::fill(logits.begin(),logits.end(),-100);
    logits[37]=2;logits[38]=1.5F;o.temperature=0;o.top_p=1;o.repetition_penalty=2;
    plan.sample_logits_for_testing(logits,0,o);
    expect(plan.gather(0,target_tokens(37,2)).proposal.token==38,"Gemma3 draft repetition discarded older native history");
    // The explicit direct-ID law penalizes only the selected native window.
    // Keep an old token, a recent token and an unseen token distinguishable,
    // then compare the actual q masses against independent softmax equations.
    plan.reset();native.back()=38;(void)plan.append(native);
    std::fill(logits.begin(),logits.end(),-100);logits[37]=3;logits[38]=2.5F;logits[39]=2;
    o.temperature=1;o.top_k=3;o.top_p=1;o.min_p=0;o.repetition_window=64;
    const auto check_repetition=[&](bool bounded,const std::array<double,3>& scores){
        plan.sample_logits_for_testing(logits,0,o,bounded);const auto q=plan.gather(0,target_tokens(37,3));
        const auto maximum=*std::max_element(scores.begin(),scores.end());double total=0;
        for(auto score:scores)total+=std::exp(score-maximum);
        std::uint64_t mass=0;
        for(std::uint32_t i=0;i<3;++i){const auto expected=std::exp(scores[i]-maximum)/total;
            const auto actual=double(q.target_values[i].probability_mass)/double(draft_probability_mass_total);
            expect(std::abs(actual-expected)<0.000001,"Gemma3 repetition policy changed its declared q law");
            mass+=q.target_values[i].probability_mass;
        }
        expect(mass==draft_probability_mass_total,"Gemma3 repetition policy q does not sum to its integer grid");
    };
    check_repetition(false,{1.5,1.25,2});check_repetition(true,{3,1.25,2});
    o.repetition_window=0;check_repetition(true,{3,2.5,2});
    // A prior zero-window call must not suppress legacy full-history marking.
    check_repetition(false,{1.5,1.25,2});
    plan.reset();logits[0]=std::numeric_limits<float>::infinity();
    refuses([&]{plan.sample_logits_for_testing(logits,0,o);},"Gemma3 accepted nonfinite raw logits");
    expect(plan.poisoned(),"Gemma3 numerical failure did not poison the plan");

    auto masked=f.map;masked.back()=-1;Gemma3DraftPlan masked_plan(f.views,masked,vocab,3);
    std::fill(logits.begin(),logits.end(),0);o.temperature=1;o.top_k=0;o.repetition_penalty=1;
    masked_plan.sample_logits_for_testing(logits,std::numeric_limits<std::uint32_t>::max(),o);
    auto endpoint=masked_plan.gather(0,target_tokens(vocab-2,2));
    expect(endpoint.proposal.token==vocab-2&&endpoint.target_values[1].probability_mass==0,
        "Gemma3 forced terminal CDF assigned probability to unmapped final token");
    logits[vocab-2]=-1000;masked_plan.sample_logits_for_testing(logits,std::numeric_limits<std::uint32_t>::max(),o);
    endpoint=masked_plan.gather(0,target_tokens(vocab-2,2));
    expect(endpoint.proposal.token<vocab-2&&endpoint.target_values[0].probability_mass==0
        &&endpoint.target_values[1].probability_mass==0,"Gemma3 CDF resurrected a zero-mass tail");
    o.temperature=std::numeric_limits<float>::min();logits[0]=std::numeric_limits<float>::max();
    refuses([&]{masked_plan.sample_logits_for_testing(logits,0,o);},"Gemma3 ignored finite temperature overflow");
}
void compact_probability_law(){
    if(!have_metal())return;const auto& f=fixture();Gemma3DraftPlan plan(f.views,f.map,vocab,3,6,769);
    std::vector<float> logits(vocab,-100);TokenSamplerOptions o;o.temperature=1;o.top_k=40;
    o.top_p=0.87F;o.min_p=0.05F;o.repetition_penalty=1;
    struct Candidate {std::uint32_t token;float score;};std::vector<Candidate> sorted;
    // Each candidate is in a different original sort tile. Cutoff ties extend
    // the retained support beyond k and compaction arrival order is irrelevant.
    for(std::uint32_t i=0;i<48;++i){const auto token=37+i*4099;
        logits[token]=i<38?4.0F-float(i)*0.07F:1.34F;sorted.push_back({token,logits[token]});}
    std::sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){
        return a.score>b.score||(a.score==b.score&&a.token<b.token);});
    const auto cutoff=sorted[o.top_k-1].score;
    sorted.erase(std::remove_if(sorted.begin(),sorted.end(),[&](const auto& v){return v.score<cutoff;}),sorted.end());
    std::vector<double> weights(sorted.size());double total=0;
    for(std::size_t i=0;i<sorted.size();++i){weights[i]=std::exp(double(sorted[i].score)-sorted[0].score);total+=weights[i];}
    double tail=0;std::vector<std::pair<std::uint32_t,double>> expected;
    for(std::size_t j=sorted.size();j>0;--j){const auto i=j-1;tail+=weights[i];
        if(i==0||(tail/total>1.0-double(o.top_p)&&weights[i]>=o.min_p))expected.push_back({sorted[i].token,weights[i]});}
    std::sort(expected.begin(),expected.end());total=0;for(const auto& value:expected)total+=value.second;
    plan.sample_logits_for_testing(logits,0,o);PreparedTokenDistribution support;support.count=48;
    for(std::uint32_t i=0;i<48;++i)support.values[i]={37+i*4099,1,1};
    const auto q=plan.gather(0,support);std::uint64_t mass=0;
    for(std::uint32_t i=0;i<support.count;++i){const auto token=support.values[i].token;
        const auto found=std::find_if(expected.begin(),expected.end(),[&](const auto& value){return value.first==token;});
        const double probability=found==expected.end()?0:found->second/total;
        const double actual=double(q.target_values[i].probability_mass)/double(draft_probability_mass_total);
        expect(std::abs(probability-actual)<0.000001,"Gemma3 compact receipt differs from independent filtered softmax");
        if(found==expected.end())expect(q.target_values[i].probability_mass==0,"Gemma3 compact filter resurrected a token");
        mass+=q.target_values[i].probability_mass;
    }
    expect(mass==draft_probability_mass_total,"Gemma3 compact integer law is not normalized");
    // Selected interval boundaries must be the very same rounded masses used
    // for correction lookup, including both endpoints of nontrivial bins.
    std::uint64_t begin=0;std::uint32_t checked=0;
    for(std::uint32_t i=0;i<support.count;++i){const auto interval_mass=q.target_values[i].probability_mass;
        if(interval_mass&&checked++<4){for(auto word:{begin,begin+interval_mass-1}){
            plan.sample_logits_for_testing(logits,static_cast<std::uint32_t>(word),o);
            const auto selected=plan.gather(0,support);
            expect(selected.proposal.token==support.values[i].token&&selected.proposal_cdf_begin==begin
                &&selected.proposal.probability_mass==interval_mass,"Gemma3 compact sampler and q receipt disagree");
        }}begin+=interval_mass;
    }
    // Exercise both sides of the bounded-support fallback with cutoff ties;
    // the general path must replace any compact receipt from the previous call.
    for(auto count:{256U,257U,64U}){
        std::fill(logits.begin(),logits.end(),-100);std::fill_n(logits.begin(),count,0);
        o.top_k=1;o.top_p=1;o.min_p=0;
        plan.sample_logits_for_testing(logits,std::numeric_limits<std::uint32_t>::max(),o);
        const auto selected=plan.gather(0,target_tokens(count-1,2));
        expect(selected.proposal.token==count-1&&selected.target_values[1].probability_mass==0,
            "Gemma3 cutoff tie fallback lost support or reused stale compact data");
        expect(std::abs(double(selected.proposal.probability_mass)/double(draft_probability_mass_total)-1.0/count)<0.000001,
            "Gemma3 cutoff tie fallback changed the uniform law");
    }
    // A compact receipt lives in native draft-ID order. Probability lookup
    // must map target support back by ID, without assuming numeric identity or
    // renormalizing away the excluded target control token.
    auto mapped=f.map;std::swap(mapped[12],mapped[50000]);mapped[0]=-1;
    {
        Gemma3DraftPlan mapped_plan(f.views,mapped,vocab,3,6,769);
        std::fill(logits.begin(),logits.end(),-100);logits[0]=100;logits[12]=0;logits[511]=0;
        o.temperature=1;o.top_k=1;o.top_p=1;o.min_p=0;
        PreparedTokenDistribution target;target.count=4;
        target.values[0]={0,1,1};target.values[1]={50000,1,1};target.values[2]={511,1,1};target.values[3]={12,1,1};
        mapped_plan.sample_logits_for_testing(logits,0,o);const auto left=mapped_plan.gather(0,target);
        expect(left.proposal.token==50000&&left.proposal_cdf_begin==0
            &&left.proposal.probability_mass==draft_probability_mass_total/2
            &&left.target_values[0].probability_mass==0&&left.target_values[3].probability_mass==0
            &&left.target_values[1].probability_mass==draft_probability_mass_total/2
            &&left.target_values[2].probability_mass==draft_probability_mass_total/2,
            "Gemma3 compact receipt assumed identity mapping or included excluded controls");
        mapped_plan.sample_logits_for_testing(logits,std::numeric_limits<std::uint32_t>::max(),o);
        const auto right=mapped_plan.gather(0,target);
        expect(right.proposal.token==511&&right.proposal_cdf_begin==draft_probability_mass_total/2,
            "Gemma3 compact mapped selection changed its native-ID CDF interval");
        logits[0]=std::numeric_limits<float>::quiet_NaN();
        refuses([&]{mapped_plan.sample_logits_for_testing(logits,0,o);},
            "Gemma3 compact path ignored a NaN in an excluded raw logit");
        expect(mapped_plan.poisoned(),"Gemma3 compact NaN refusal did not poison the plan");
    }
    {
        Gemma3DraftPlan overflow_plan(f.views,f.map,vocab,3,6,769);
        std::fill(logits.begin(),logits.end(),0);logits[12]=std::numeric_limits<float>::max();
        o.temperature=std::numeric_limits<float>::min();
        refuses([&]{overflow_plan.sample_logits_for_testing(logits,0,o);},
            "Gemma3 compact path accepted finite temperature-transform overflow");
        expect(overflow_plan.poisoned(),"Gemma3 compact transform refusal did not poison the plan");
    }
}
void bounded_resources_and_batched_ingestion(){
    const auto full=Gemma3DraftPlan::estimate_owned_bytes(vocab,3,6,32768);
    const auto half=Gemma3DraftPlan::estimate_owned_bytes(vocab,3,6,16384);
    expect(full-half==std::size_t(32768-16384)*(3*256*2+1)*sizeof(float),
        "Gemma3 context estimate does not account for exactly global KV and native history");
    refuses([&]{(void)Gemma3DraftPlan::estimate_owned_bytes(vocab,3,6,0);},"Gemma3 estimated zero context");
    refuses([&]{(void)Gemma3DraftPlan::estimate_owned_bytes(vocab,3,6,32769);},"Gemma3 estimated beyond model context");
    if(!have_metal())return;const auto& f=fixture();Gemma3DraftPlan plan(f.views,f.map,vocab,3,6,769);
    expect(plan.prepared_context_capacity()==769&&plan.owned_bytes()==Gemma3DraftPlan::estimate_owned_bytes(vocab,3,6,769),
        "Gemma3 aggregate admission estimate differs from allocated ownership");
    const std::array<std::uint32_t,2> prefix={2,71};const std::uint32_t root=111,next_root=739;
    TokenSamplerOptions options;options.temperature=0;SamplerHistorySnapshot history;history.window_limit=64;
    std::array<std::uint32_t,3> words{};(void)plan.append(prefix);
    (void)plan.begin_transaction();(void)plan.append(std::span(&root,1));
    const auto original=plan.propose(root,words,options,history);
    const std::vector<std::uint32_t> expected(original.draft_tokens.begin(),original.draft_tokens.end());
    plan.abort_transaction();(void)plan.begin_transaction();
    const auto combined=plan.propose(root,words,options,history,std::span(&root,1));
    expect(combined.execution.command_buffer_count==1&&combined.execution.token_count==3
        &&combined.start_position==3&&std::equal(expected.begin(),expected.end(),combined.draft_tokens.begin()),
        "Gemma3 batched root ingestion changed proposals or added a submission");
    (void)plan.commit_prefix(plan.position());
    (void)plan.begin_transaction();const auto pending=plan.propose(next_root,std::span(words).first(1),options,history,std::span(&next_root,1));
    expect(pending.execution.command_buffer_count==1&&pending.execution.token_count==2,
        "Gemma3 pending/root ingestion submitted separately or forwarded final proposal");
    plan.abort_transaction();expect(plan.position()==6&&plan.materialized_position()==5,
        "Gemma3 combined transaction abort lost the durable pending token");
    (void)plan.append(std::span(&next_root,1));std::vector<float> actual(vocab),replayed(vocab);(void)plan.read_logits(actual);
    std::vector<std::uint32_t> replay(prefix.begin(),prefix.end());replay.push_back(root);
    replay.insert(replay.end(),expected.begin(),expected.end());replay.push_back(next_root);
    plan.reset();(void)plan.append(replay);(void)plan.read_logits(replayed);
    expect(bit_equal(actual,replayed),"Gemma3 combined root/pending path changed exact replay");
    // Non-power-of-two global capacity must use logical positions, never a
    // capacity-minus-one bit mask when publishing the final global cache row.
    plan.reset();plan.seed_zero_cache_prefix_for_testing(768);Reference reference;reference.position=768;
    (void)plan.append(std::span(&root,1));reference.step(root,true);(void)plan.read_logits(actual);
    close_logits(actual,reference.logits(),"session-sized non-power-of-two context boundary");
    for(unsigned layer=0;layer<layers;++layer){std::vector<float> k(256),v(256);plan.read_cache(layer,768,k,v);
        cache_close(k,reference.keys[layer],"session-sized endpoint K");cache_close(v,reference.values[layer],"session-sized endpoint V");}
    const auto frontier=plan.position();(void)plan.begin_transaction();
    refuses([&]{(void)plan.propose(root,std::span(words).first(1),options,history,std::span(&root,1));},
        "Gemma3 combined root/proposal accepted a context overflow");
    expect(plan.position()==frontier,"Gemma3 combined capacity refusal mutated the timeline");plan.abort_transaction();
}
void context_endpoint(){
    if(!have_metal())return;const auto& f=fixture();Gemma3DraftPlan plan(f.views,f.map,vocab,3);
    constexpr auto last=Gemma3DraftPlan::context_capacity-1;
    plan.seed_zero_cache_prefix_for_testing(last);Reference reference;reference.position=last;
    const std::uint32_t token=119;(void)plan.append(std::span(&token,1));reference.step(token,true);
    std::vector<float> logits(vocab);(void)plan.read_logits(logits);
    close_logits(logits,reference.logits(),"full32768 attention boundary with independent zero-prefix equations");
    for(unsigned layer=0;layer<layers;++layer){std::vector<float> k(256),v(256);
        plan.read_cache(layer,last,k,v);cache_close(k,reference.keys[layer],"last context K");cache_close(v,reference.values[layer],"last context V");}
    CacheSnapshot cache(plan);refuses([&]{(void)plan.append(std::span(&token,1));},"Gemma3 wrote beyond final context row");cache.matches(plan);
    (void)plan.begin_transaction();std::array<std::uint32_t,1> words{};
    TokenSamplerOptions options;SamplerHistorySnapshot history;history.window_limit=64;
    refuses([&]{(void)plan.propose(token,words,options,history);},"Gemma3 proposed beyond context endpoint");
    plan.abort_transaction();cache.matches(plan);
}
void unavailable_and_malformed(){
    const auto& f=fixture();using namespace tensor::backends::metal;
    set_metal_device_unavailable_for_testing(true);
    refuses([&]{Gemma3DraftPlan p(f.views,f.map,vocab,3);},"Gemma3 fell back without Metal");
    set_metal_device_unavailable_for_testing(false);
    auto map=f.map;map[1]=map[0];refuses([&]{Gemma3DraftPlan p(f.views,map,vocab,3);},"Gemma3 accepted duplicate mapping");
    std::fill(map.begin(),map.end(),-1);refuses([&]{Gemma3DraftPlan p(f.views,map,vocab,3);},"Gemma3 accepted empty mapping");
    refuses([&]{Gemma3DraftPlan p(f.views,f.map,vocab,24);},"Gemma3 accepted horizon overflow");
    refuses([&]{Gemma3DraftPlan p(f.views,f.map,vocab,3,4);},"Gemma3 accepted undersized transaction arena");
    refuses([&]{Gemma3DraftPlan p(std::span(f.views).first(3),f.map,vocab,3);},"Gemma3 accepted incomplete model");
}

std::vector<float> golden_logits(const char* filename){
    const auto entry=std::find_if(gemma3_golden_digests.begin(),gemma3_golden_digests.end(),
        [&](const auto& value){return value.filename==filename;});
    expect(entry!=gemma3_golden_digests.end(),"Gemma3 golden logit hash is absent");
    const auto path=std::filesystem::path("build/gemma3-reference-f32-golden")/filename;
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    expect(file&&file.tellg()==std::streamoff(vocab*sizeof(float)),"Gemma3 pinned golden is missing or has wrong size");
    std::vector<float> result(vocab);file.seekg(0);file.read(reinterpret_cast<char*>(result.data()),std::streamsize(result.size()*sizeof(float)));
    expect(bool(file),"Gemma3 golden read failed");
    const auto bytes=std::span(reinterpret_cast<const std::uint8_t*>(result.data()),result.size()*sizeof(float));
    expect(runtime::sha256(bytes)==runtime::parse_sha256_hex(entry->sha256),"Gemma3 golden digest differs");return result;
}
void golden_cache(const Gemma3DraftPlan& plan,std::string_view prefix){
    unsigned checked=0;
    for(const auto& entry:gemma3_golden_digests){
        const auto stem=std::string(prefix)+"-layer-";
        if(!entry.filename.starts_with(stem))continue;
        const auto layer=static_cast<std::uint32_t>(std::stoul(std::string(entry.filename.substr(stem.size(),2))));
        const auto path=std::filesystem::path("build/gemma3-reference-f32-golden")/entry.filename;
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        const auto count=entry.rows*256;
        expect(file&&file.tellg()==std::streamoff(count*sizeof(float)),"Gemma3 golden cache size mismatch");
        std::vector<float> expected(count),k(count),v(count);
        file.seekg(0);file.read(reinterpret_cast<char*>(expected.data()),std::streamsize(count*sizeof(float)));
        expect(bool(file),"Gemma3 golden cache read failed");
        const auto bytes=std::span(reinterpret_cast<const std::uint8_t*>(expected.data()),count*sizeof(float));
        expect(runtime::sha256(bytes)==runtime::parse_sha256_hex(entry.sha256),"Gemma3 golden cache digest mismatch");
        plan.read_cache(layer,plan.materialized_position()-static_cast<std::uint32_t>(entry.rows),k,v);
        cache_close(entry.filename.ends_with("-key.f32")?std::span(k):std::span(v),expected,entry.filename.data());
        ++checked;
    }
    expect(checked==2*layers,"Gemma3 golden omitted a layer's K/V");
    std::cout<<"[GEMMA3 PARITY] "<<prefix<<" all18-layer K/V matches pinned CPU goldens\n";
}
void checkpoint_goldens(){
    if(!have_metal())return;const std::filesystem::path path="models/gemma-3-270m-it-Q8_0.gguf";
    if(!std::filesystem::exists(path)){
        std::cout<<"[GEMMA3 CHECKPOINT] absent; complete synthetic independent equations still execute\n";return;}
    StandaloneDraftArtifact artifact(path);std::vector<std::int32_t> map(vocab);std::iota(map.begin(),map.end(),0);
    Gemma3DraftPlan plan(artifact,map,vocab,23);std::vector<float> actual(vocab),replayed(vocab);
    const bool goldens=std::filesystem::exists("build/gemma3-reference-f32-golden/identity.json");
    if(!goldens)std::cout<<"[GEMMA3 CHECKPOINT] pinned offline goldens absent; real-checkpoint replay still executes\n";
    constexpr std::array<std::uint32_t,4> short_tokens={2,9259,12074,108};
    (void)plan.append(std::span(short_tokens).first(3));(void)plan.read_logits(actual);
    if(goldens)close_logits(actual,golden_logits("prefill-logits.f32"),"checkpoint prefill");
    (void)plan.append(std::span(short_tokens).last(1));(void)plan.read_logits(actual);
    if(goldens)close_logits(actual,golden_logits("decode-logits.f32"),"checkpoint decode");
    if(goldens)golden_cache(plan,"decode");
    CacheSnapshot short_cache(plan);plan.reset();for(auto token:short_tokens)(void)plan.append(std::span(&token,1));
    (void)plan.read_logits(replayed);expect(bit_equal(actual,replayed),"Gemma3 checkpoint scalar/batched logits differ");short_cache.matches(plan);
    auto tokens=stream(536);plan.reset();(void)plan.append(std::span(tokens).first(512));(void)plan.read_logits(actual);
    if(goldens)close_logits(actual,golden_logits("wrap-before-logits.f32"),"checkpoint before local wrap");
    (void)plan.append(std::span(tokens).subspan(512,1));(void)plan.read_logits(actual);
    if(goldens)close_logits(actual,golden_logits("wrap-crossing-logits.f32"),"checkpoint first local wrap");
    (void)plan.append(std::span(tokens).subspan(513));(void)plan.read_logits(actual);
    if(goldens)close_logits(actual,golden_logits("wrap-after-logits.f32"),"checkpoint wrapped prefill");
    if(goldens)golden_cache(plan,"wrap-after");
    CacheSnapshot wrapped(plan);plan.reset();(void)plan.append(tokens);(void)plan.read_logits(replayed);
    expect(bit_equal(actual,replayed),"Gemma3 checkpoint chunk boundaries changed logits");wrapped.matches(plan);
}
} // namespace
std::span<const TestCase> gemma3_draft_test_cases(){
    static const std::array<TestCase,8> cases={{
        {"Gemma3 complete Q8 transformer matches independent CPU equations and exact replay",synthetic_forward},
        {"Gemma3 private K/V transactions preserve wrap, rejection, correction and pending-token replay",transaction_lifecycle},
        {"Gemma3 full262k probability law preserves ties, nucleus and native-history semantics",probability_law},
        {"Gemma3 compact top-k receipts match independent sampling and preserve cutoff tie fallback",compact_probability_law},
        {"Gemma3 session resources and batched pending/root ingestion preserve exact replay",bounded_resources_and_batched_ingestion},
        {"Gemma3 full32768 context endpoint matches independent attention equations and refuses overflow",context_endpoint},
        {"Gemma3 requires Metal, complete geometry and bounded injective mappings",unavailable_and_malformed},
        {"Gemma3 actual checkpoint matches pinned CPU goldens and exact wrapped replay when available",checkpoint_goldens},
    }};return cases;
}
} // namespace local_gguf_engine::tests
