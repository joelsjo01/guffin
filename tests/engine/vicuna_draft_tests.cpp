#include "local_gguf_engine/engine/vicuna_draft.hpp"
#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"
#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "../../src/tensor/backends/metal/metal_test_hooks.hpp"
#include "test_support.hpp"

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
constexpr std::size_t width=768,ff=3072,vocab=32000;
template<class F>void refuses(F&& f,const char* what){bool caught=false;
    try{f();}catch(const std::exception&){caught=true;}expect(caught,what);}

std::vector<tensor::TensorView> artifact_views(const StandaloneDraftArtifact& artifact){
    std::vector<tensor::TensorView> result;
    for(const auto* name:{"model.embed_tokens.weight","lm_head.weight","model.norm.weight"})result.push_back(artifact.tensor(name));
    constexpr std::array<const char*,9> suffixes={"self_attn.q_proj.weight","self_attn.k_proj.weight",
        "self_attn.v_proj.weight","self_attn.o_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight",
        "mlp.down_proj.weight","input_layernorm.weight","post_attention_layernorm.weight"};
    for(unsigned layer=0;layer<2;++layer)for(const auto* suffix:suffixes)
        result.push_back(artifact.tensor("model.layers."+std::to_string(layer)+"."+suffix));
    return result;
}
struct Fixture {
    std::vector<tensor::TensorStorage> storage;
    std::vector<tensor::TensorView> views;
    std::vector<std::int32_t> map;
    Fixture():map(vocab){
        std::iota(map.begin(),map.end(),0);
        const auto add=[&](std::size_t input,std::size_t output,unsigned kind){
            auto value=tensor::TensorStorage::allocate(tensor::TensorDescriptor::from(
                output==1?tensor::Shape::from({input}):tensor::Shape::from({input,output}),tensor::DType::f16));
            auto* dst=reinterpret_cast<std::uint16_t*>(value.mutable_bytes().data());
            for(std::size_t row=0;row<output;++row)for(std::size_t col=0;col<input;++col){
                float a=0;
                if(kind==0)a=float(int((row*17+col*13)%53)-26)/64.0F;
                else if(kind==1)a=float(int((row*19+col*7)%31)-15)/128.0F;
                else if(kind==2)a=1.0F+float(int(col%7)-3)/64.0F;
                else if(col==row%input)a=0.25F;
                else if(col==(row*7+11)%input)a=-0.0625F;
                dst[row*input+col]=tensor::f32_to_f16_bits(a);
            }
            storage.push_back(std::move(value));
        };
        add(width,vocab,0);add(width,vocab,1);add(width,1,2);
        for(unsigned layer=0;layer<2;++layer){for(unsigned j=0;j<4;++j)add(width,width,3);
            add(width,ff,3);add(width,ff,3);add(ff,width,3);add(width,1,2);add(width,1,2);}
        for(const auto& item:storage)views.push_back(item.view());
    }
};
const Fixture& fixture(){static const Fixture value;return value;}

// Independent scalar equations: double accumulators, ordinary RMS weights,
// explicit split-half RoPE, uncapped softmax and separately rounded F16 K/V.
// This lives only in tests and cannot execute a production inference request.
struct Reference {
    std::vector<std::vector<float>> weights;
    std::array<std::vector<std::uint16_t>,2> keys,values;
    std::vector<float> logits;
    explicit Reference(std::span<const tensor::TensorView> tensors){
        for(const auto& tensor:tensors){std::vector<float> w(tensor.descriptor().element_count);
            const auto* bits=reinterpret_cast<const std::uint16_t*>(tensor.bytes().data());
            for(std::size_t i=0;i<w.size();++i)w[i]=tensor::f16_bits_to_f32(bits[i]);weights.push_back(std::move(w));}
    }
    std::vector<float> linear(const std::vector<float>& x,std::size_t index,std::size_t output){
        std::vector<float> y(output);const auto& w=weights[index];
        for(std::size_t o=0;o<output;++o){double s=0;for(std::size_t i=0;i<x.size();++i)s+=double(x[i])*w[o*x.size()+i];y[o]=float(s);}return y;
    }
    std::vector<float> norm(const std::vector<float>& x,std::size_t index){
        double s=0;for(float v:x)s+=double(v)*v;const auto inv=1.0/std::sqrt(s/double(x.size())+1.e-6);
        std::vector<float> y(x.size());for(std::size_t i=0;i<x.size();++i)y[i]=float(double(x[i])*inv*weights[index][i]);return y;
    }
    void step(std::uint32_t token){
        const std::size_t position=keys[0].size()/width;
        std::vector<float> x(weights[0].data()+std::size_t(token)*width,weights[0].data()+(std::size_t(token)+1)*width);
        for(std::size_t layer=0;layer<2;++layer){const auto b=3+9*layer;const auto n=norm(x,b+7);
            auto q=linear(n,b,width),k=linear(n,b+1,width);const auto v=linear(n,b+2,width);
            for(std::size_t h=0;h<12;++h)for(std::size_t j=0;j<32;++j){
                const double angle=double(position)*std::pow(10000.0,-double(j)/32.0),c=std::cos(angle),s=std::sin(angle);
                for(auto* a:{&q,&k}){const auto one=(*a)[h*64+j],two=(*a)[h*64+j+32];
                    (*a)[h*64+j]=float(double(one)*c-double(two)*s);(*a)[h*64+j+32]=float(double(one)*s+double(two)*c);}
            }
            for(std::size_t j=0;j<width;++j){keys[layer].push_back(tensor::f32_to_f16_bits(k[j]));values[layer].push_back(tensor::f32_to_f16_bits(v[j]));}
            std::vector<float> attended(width);
            for(std::size_t h=0;h<12;++h){std::vector<double> scores(position+1);double mx=-std::numeric_limits<double>::infinity();
                for(std::size_t t=0;t<=position;++t){double s=0;for(std::size_t j=0;j<64;++j){
                    s+=double(q[h*64+j])*tensor::f16_bits_to_f32(keys[layer][t*width+h*64+j]);}
                    scores[t]=s/8;mx=std::max(mx,scores[t]);}
                double total=0;for(auto& s:scores){s=std::exp(s-mx);total+=s;}
                for(std::size_t j=0;j<64;++j){double s=0;for(std::size_t t=0;t<=position;++t){
                    s+=(scores[t]/total)*tensor::f16_bits_to_f32(values[layer][t*width+h*64+j]);}
                    attended[h*64+j]=float(s);}
            }
            const auto out=linear(attended,b+3,width);for(std::size_t j=0;j<width;++j)x[j]+=out[j];
            const auto n2=norm(x,b+8);auto gate=linear(n2,b+4,ff);const auto up=linear(n2,b+5,ff);
            for(std::size_t j=0;j<ff;++j)gate[j]=float(double(gate[j])/(1.0+std::exp(-double(gate[j])))*up[j]);
            const auto down=linear(gate,b+6,width);for(std::size_t j=0;j<width;++j)x[j]+=down[j];
        }
        logits=linear(norm(x,2),1,vocab);
    }
};
void close_logits(std::span<const float> actual,std::span<const float> expected,const char* label){
    double worst=0;for(std::size_t i=0;i<actual.size();++i){const double error=std::abs(double(actual[i])-expected[i]);
        worst=std::max(worst,error);if(!std::isfinite(actual[i])||error>0.006+0.001*std::abs(double(expected[i])))
            throw TestFailure(std::string(label)+": token="+std::to_string(i)+" error="+std::to_string(error));}
    std::cout<<"[VICUNA PARITY] "<<label<<" max_abs="<<worst<<'\n';
}
void check_upstream_golden(std::span<const float> actual,const char* name,const char* digest){
    const auto path=std::filesystem::path("build/vicuna-reference-golden")/name;
    if(!std::filesystem::exists(path))return;
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    expect(file&&file.tellg()==std::streamoff(vocab*sizeof(float)),"Vicuna upstream golden size mismatch");
    std::vector<float> expected(vocab);file.seekg(0);file.read(reinterpret_cast<char*>(expected.data()),std::streamsize(expected.size()*sizeof(float)));
    expect(bool(file),"Vicuna upstream golden read failed");
    const auto bytes=std::span(reinterpret_cast<const std::uint8_t*>(expected.data()),expected.size()*sizeof(float));
    expect(runtime::sha256(bytes)==runtime::parse_sha256_hex(digest),"Vicuna upstream golden identity mismatch");
    close_logits(actual,expected,name);
}
void verify_forward(std::span<const tensor::TensorView> views,VicunaDraftPlan& plan,bool upstream=false){
    constexpr std::array<std::uint32_t,4> tokens={1,15043,29892,3186};Reference cpu(views);
    (void)plan.append(std::span(tokens).first(3));for(auto t:std::span(tokens).first(3))cpu.step(t);
    std::vector<float> batched(vocab),incremental(vocab);(void)plan.read_logits(batched);
    close_logits(batched,cpu.logits,"prefill full-vocabulary CPU/Metal");
    if(upstream)check_upstream_golden(batched,"prefill-logits.f32","4c043145e6e29b82f4541287e4963796dc11de9be67d55570a76523abbb195f1");
    (void)plan.append(std::span(tokens).last(1));cpu.step(tokens.back());(void)plan.read_logits(batched);
    close_logits(batched,cpu.logits,"cached decode full-vocabulary CPU/Metal");
    if(upstream)check_upstream_golden(batched,"decode-logits.f32","c70f20e57bb31acb94bd3fac90f739d25e19b0d1f696507d66d46274a7e80301");
    for(std::uint32_t layer=0;layer<2;++layer){std::vector<std::uint16_t> k(tokens.size()*width),v(k.size());plan.read_cache(layer,k,v);
        for(std::size_t i=0;i<k.size();++i)for(auto pair:{std::pair{k[i],cpu.keys[layer][i]},std::pair{v[i],cpu.values[layer][i]}}){
            const float a=tensor::f16_bits_to_f32(pair.first),b=tensor::f16_bits_to_f32(pair.second);
            expect(std::abs(a-b)<=0.002F+0.001F*std::abs(b),"Vicuna K/V differs beyond F16 rounding and F32 accumulation error");}}
    plan.reset();for(auto t:tokens)(void)plan.append(std::span(&t,1));(void)plan.read_logits(incremental);
    expect(batched==incremental,"Vicuna batched/incremental replay changed logits");
    plan.truncate(3);(void)plan.append(std::span(tokens).last(1));(void)plan.read_logits(incremental);
    expect(batched==incremental,"Vicuna truncated cache regeneration changed logits");
}
bool have_metal(){const bool available=tensor::tensor_backend_is_available(runtime::BackendKind::metal);
    if(!available)std::cout<<"[VICUNA METAL] unavailable; GPU parity not executed\n";return available;}
void synthetic_forward(){if(!have_metal())return;const auto& f=fixture();VicunaDraftPlan plan(f.views,f.map,vocab,23);verify_forward(f.views,plan);}
void checkpoint_forward(){const std::filesystem::path path="models/vicuna_68m/vicuna-68m-f16.gguf";
    if(!have_metal())return;
    if(!std::filesystem::exists(path)){std::cout<<"[VICUNA CHECKPOINT] unavailable; synthetic complete-geometry parity remains mandatory\n";return;}
    const StandaloneDraftArtifact artifact(path);auto views=artifact_views(artifact);std::vector<std::int32_t> map(vocab);std::iota(map.begin(),map.end(),0);
    VicunaDraftPlan plan(artifact,map,vocab,23);verify_forward(views,plan,true);
    // Replay is a state restoration mechanism, so mere numerical closeness
    // is insufficient: full real-checkpoint K/V and logits must be identical
    // across the production 64-row prefill and single-token decode schedules.
    std::array<std::uint32_t,64> tokens;for(std::size_t i=0;i<tokens.size();++i)tokens[i]=3+std::uint32_t(i*499%31997);
    tokens[0]=1;plan.reset();(void)plan.append(tokens);
    std::vector<float> batched(vocab),decoded(vocab);(void)plan.read_logits(batched);
    std::array<std::vector<std::uint16_t>,2> keys,values;
    for(std::uint32_t layer=0;layer<2;++layer){keys[layer].resize(tokens.size()*width);values[layer].resize(keys[layer].size());plan.read_cache(layer,keys[layer],values[layer]);}
    plan.reset();for(auto token:tokens)(void)plan.append(std::span(&token,1));(void)plan.read_logits(decoded);
    expect(batched==decoded,"Vicuna real checkpoint 64-row replay changed logits");
    for(std::uint32_t layer=0;layer<2;++layer){std::vector<std::uint16_t> k(keys[layer].size()),v(k.size());plan.read_cache(layer,k,v);
        expect(k==keys[layer]&&v==values[layer],"Vicuna real checkpoint 64-row replay changed K/V bits");}
}

std::vector<double> processor_reference(std::span<const float> raw,std::span<const std::int32_t> map,
    std::span<const std::uint32_t> history,const TokenSamplerOptions& options){
    struct Entry {std::uint32_t token;float score;bool keep;};std::vector<Entry> sorted(vocab);
    for(std::uint32_t i=0;i<vocab;++i){float score=raw[i];
        if(std::find(history.begin(),history.end(),i)!=history.end())
            score=score<0?score*options.repetition_penalty:score/options.repetition_penalty;
        if(options.temperature>0)score/=options.temperature;
        sorted[i]={i,map[i]>=0?score:-std::numeric_limits<float>::infinity(),map[i]>=0};
    }
    std::sort(sorted.begin(),sorted.end(),[](const Entry& a,const Entry& b){
        if(a.keep!=b.keep)return a.keep>b.keep;
        return a.score>b.score||(a.score==b.score&&a.token<b.token);
    });
    std::vector<double> probabilities(vocab);
    if(options.temperature==0){probabilities[sorted[0].token]=1;return probabilities;}
    const float peak=sorted[0].score;
    if(options.top_k){const float cutoff=sorted[options.top_k-1].score;
        for(auto& entry:sorted)if(entry.score<cutoff)entry.keep=false;}
    double total=0;for(const auto& entry:sorted)if(entry.keep)total+=std::exp(double(entry.score)-peak);
    double tail=0;const float tail_limit=1.0F-options.top_p;
    for(std::size_t i=sorted.size();i-->0;){auto& entry=sorted[i];if(!entry.keep)continue;
        tail+=std::exp(double(entry.score)-peak);
        if(i!=0&&options.top_p<1&&tail/total<=tail_limit)entry.keep=false;
        if(i!=0&&options.min_p>0&&std::exp(double(entry.score)-peak)<options.min_p)entry.keep=false;
    }
    total=0;for(const auto& entry:sorted)if(entry.keep){const double w=std::exp(double(entry.score)-peak);probabilities[entry.token]=w;total+=w;}
    for(auto& p:probabilities)p/=total;return probabilities;
}
void verify_processor_law(VicunaDraftPlan& plan,std::span<const float> logits,
    std::span<const std::int32_t> map,std::span<const std::uint32_t> history,const TokenSamplerOptions& options,
    std::uint32_t word=0x715acd31U){
    const auto expected=processor_reference(logits,map,history,options);
    plan.sample_logits_for_testing(logits,word,options);std::uint64_t total=0;
    for(std::uint32_t first=0;first<vocab;first+=256){PreparedTokenDistribution target;target.count=std::min(256U,32000U-first);
        for(std::uint32_t i=0;i<target.count;++i)target.values[i]={first+i,1.0/double(target.count),1};
        const auto q=plan.gather(0,target);
        expect(std::uint64_t(word)>=q.proposal_cdf_begin&&std::uint64_t(word)<q.proposal_cdf_begin+q.proposal.probability_mass,"Vicuna filtered q interval does not contain its proposal word");
        for(std::uint32_t i=0;i<target.count;++i){const auto mass=q.target_values[i].probability_mass;total+=mass;
            expect(std::abs(double(mass)/4294967296.0-expected[first+i])<=1.e-6,"Vicuna source-order draft processors disagree with independent CPU equations at token "+std::to_string(first+i));
            if(expected[first+i]==0)expect(mass==0,"Vicuna filtered token retained positive mass");
        }
    }expect(total==draft_probability_mass_total,"Vicuna filtered full-vocabulary q does not sum to 2^32");
}
void source_processors(){
    if(!have_metal())return;const auto& f=fixture();auto map=f.map;
    for(std::size_t i=0;i<vocab;i+=7)map[i]=-1;map[255]=map[256]=map[31999]=-1;
    VicunaDraftPlan plan(f.views,map,vocab,23);std::vector<float> logits(vocab,-10.0F);
    std::vector<std::uint32_t> history={1,5,5,123,31998};(void)plan.append(history);
    TokenSamplerOptions options;options.temperature=0.8F;options.top_k=40;options.top_p=0.95F;
    options.min_p=0.05F;options.repetition_penalty=1.1F;
    for(std::size_t i=0;i<vocab;++i)logits[i]=float(std::sin(double(i)*0.173)*7.0+double(i%31)*0.03125);
    verify_processor_law(plan,logits,map,history,options);
    const auto golden_dir=std::filesystem::path("build/vicuna-sampler-golden");
    if(std::filesystem::exists(golden_dir/"raw-input.f32")){
        const auto read=[&](const char* name,const char* digest){
            std::ifstream file(golden_dir/name,std::ios::binary|std::ios::ate);
            expect(file&&file.tellg()==std::streamoff(vocab*sizeof(float)),"Vicuna sampler golden size mismatch");
            std::vector<float> values(vocab);file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(float)));
            expect(bool(file),"Vicuna sampler golden read failed");
            const auto bytes=std::span(reinterpret_cast<const std::uint8_t*>(values.data()),values.size()*sizeof(float));
            expect(runtime::sha256(bytes)==runtime::parse_sha256_hex(digest),"Vicuna sampler golden identity mismatch");return values;
        };
        logits=read("raw-input.f32","c9f7f0351dec5ad9362473a1e3051d6dc3473aba3739f90b9c5272237ee4a43b");
        const auto filtered=read("filtered-logits.f32","4d9c690339c1bd260c83f79d3cb70654edf935bb19a4c9e00c6a85f89edd418f");
        const float maximum=*std::max_element(filtered.begin(),filtered.end());double normalizer=0;
        for(float value:filtered)normalizer+=std::exp(double(value)-maximum);
        plan.sample_logits_for_testing(logits,0xa593147eU,options);std::size_t positive=0;
        for(std::uint32_t first=0;first<vocab;first+=256){PreparedTokenDistribution target;target.count=std::min(256U,32000U-first);
            for(std::uint32_t i=0;i<target.count;++i)target.values[i]={first+i,1.0/double(target.count),1};
            const auto q=plan.gather(0,target);
            for(std::uint32_t i=0;i<target.count;++i){const auto mass=q.target_values[i].probability_mass;
                const double expected=std::exp(double(filtered[first+i])-maximum)/normalizer;
                expect(std::abs(double(mass)/4294967296.0-expected)<=1.e-6,"Vicuna q differs from pinned Transformers processor golden");
                if(!std::isfinite(filtered[first+i]))expect(mass==0,"Vicuna retained a token suppressed by pinned HF processors");
                if(mass)++positive;
            }
        }expect(positive==38,"Vicuna pinned HF processor support differs");
        std::cout<<"[VICUNA PARITY] pinned HF sampler support=38 full-vocabulary mass parity\n";
    }
    // Full-vocabulary cutoff ties must not be narrowed to either top_k or the
    // target's 256-entry compact probability representation.
    std::fill(logits.begin(),logits.end(),-10);for(std::size_t i=1;i<=601;++i)logits[i]=2;
    options.temperature=1;options.top_k=1;options.top_p=1;options.min_p=0;options.repetition_penalty=1;
    verify_processor_law(plan,logits,map,history,options,0xffffffffU);
    std::uint32_t positive=0;for(std::uint32_t first=0;first<768;first+=256){PreparedTokenDistribution target;target.count=256;
        for(std::uint32_t i=0;i<256;++i)target.values[i]={first+i,1.0/256,1};const auto q=plan.gather(0,target);
        for(std::uint32_t i=0;i<256;++i)if(q.target_values[i].probability_mass)++positive;}
    expect(positive>256,"Vicuna top-k cutoff ties were silently narrowed to 256 tokens");
    options.top_p=0.5F;options.min_p=1;verify_processor_law(plan,logits,map,history,options,0);
    // Repetition is per unique native ID over the complete retained prefix,
    // not frequency, target IDs, or a silently substituted 64-token window.
    history.assign(70,2);history[0]=1;history[1]=5;history[2]=5;
    plan.reset();(void)plan.append(history);std::fill(logits.begin(),logits.end(),-100);
    logits[1]=10;logits[3]=9;logits[5]=-1;options.temperature=1;options.top_k=0;options.top_p=1;
    options.min_p=0;options.repetition_penalty=2;verify_processor_law(plan,logits,map,history,options);
    options.temperature=0;options.top_k=1;options.top_p=0.01F;options.min_p=1;
    verify_processor_law(plan,logits,map,history,options);
    // A truncated native suffix must no longer affect repetition filtering.
    plan.truncate(1);history.resize(1);options.temperature=1;options.top_k=32000;options.top_p=1;options.min_p=0;
    logits[2]=6;verify_processor_law(plan,logits,map,history,options);
    // Intentional suppression is never encoded as a finite sentinel. Even
    // actual -FLT_MAX logits remain legal and cannot resurrect filtered IDs.
    std::fill(logits.begin(),logits.end(),-std::numeric_limits<float>::max());
    options.repetition_penalty=1;options.top_k=1;options.top_p=std::numeric_limits<float>::min();
    verify_processor_law(plan,logits,map,history,options,0xffffffffU);
    options.top_p=1;verify_processor_law(plan,logits,map,history,options);
    auto invalid=options;invalid.top_k=32001;refuses([&]{plan.sample_logits_for_testing(logits,0,invalid);},"Vicuna draft top-k overflow accepted");
    invalid=options;invalid.repetition_penalty=0;refuses([&]{plan.sample_logits_for_testing(logits,0,invalid);},"Vicuna invalid repetition accepted");
    invalid=options;invalid.top_p=0;refuses([&]{plan.sample_logits_for_testing(logits,0,invalid);},"Vicuna empty nucleus accepted");
}
void full_probability(){
    if(!have_metal())return;
    const auto& f=fixture();auto map=f.map;for(std::size_t i=0;i<vocab;i+=7)map[i]=-1;
    // A masked block tail and the final vocabulary token must receive zero
    // mass, even at tile-offset and forced-final-CDF boundaries.
    map[255]=-1;map[256]=-1;map[31999]=-1;
    VicunaDraftPlan plan(f.views,map,vocab,23);std::vector<float> logits(vocab);
    // Exact equal weights make all expected CDF boundaries independently
    // calculable in integer arithmetic, including >256 nonzero support.
    std::fill(logits.begin(),logits.end(),0.0F);
    const auto eligible=static_cast<std::uint32_t>(std::count_if(map.begin(),map.end(),[](auto id){return id>=0;}));
    const auto boundary=[&](std::uint32_t n){return static_cast<std::uint64_t>((float(n)/float(eligible))*4294967296.0F);};
    const std::array<std::uint32_t,7> words={0,1,std::uint32_t(boundary(100)-1),
        std::uint32_t(boundary(100)),0x7fffffffU,0xfffffffeU,0xffffffffU};
    for(auto word:words){plan.sample_logits_for_testing(logits,word,1.0F);std::uint32_t prefix=0;std::uint64_t total=0;
        for(std::uint32_t first=0;first<vocab;first+=256){PreparedTokenDistribution target;target.count=std::min(256U,32000U-first);
            for(std::uint32_t i=0;i<target.count;++i)target.values[i]={first+i,1.0/double(target.count),1.0};
            auto q=plan.gather(0,target);
            expect(q.lease.valid(),"Vicuna probability lease is not current");
            expect(std::uint64_t(word)>=q.proposal_cdf_begin&&std::uint64_t(word)<q.proposal_cdf_begin+q.proposal.probability_mass,"Vicuna sample interval misses its random word");
            for(std::uint32_t i=0;i<target.count;++i){auto before=boundary(prefix);if(map[first+i]>=0)++prefix;
                auto mass=boundary(prefix)-before;expect(q.target_values[i].probability_mass==mass,"Vicuna full-q integer CDF differs from independent uniform equations");total+=mass;}
        }expect(total==draft_probability_mass_total,"Vicuna complete q is not normalized on all 2^32 words");
    }
    // Nonuniform full support cross-checks the actual quantized law against
    // independent double softmax. The 1e-6 absolute budget covers F32 positive
    // prefix accumulation and CDF rounding; it is not a p/target tolerance.
    double normalizer=0;
    for(std::size_t i=0;i<vocab;++i){logits[i]=float(int((i*37)%251)-125)/16.0F;
        if(map[i]>=0)normalizer+=std::exp(double(logits[i]));}
    plan.sample_logits_for_testing(logits,0x715acd31U,1.0F);std::uint64_t complete_mass=0;
    for(std::uint32_t first=0;first<vocab;first+=256){PreparedTokenDistribution target;
        target.count=std::min(256U,32000U-first);
        for(std::uint32_t i=0;i<target.count;++i)target.values[i]={first+i,1.0/double(target.count),1};
        auto q=plan.gather(0,target);
        for(std::uint32_t i=0;i<target.count;++i){const auto token=first+i;
            const double ideal=map[token]<0?0:std::exp(double(logits[token]))/normalizer;
            const auto mass=q.target_values[i].probability_mass;complete_mass+=mass;
            expect(std::abs(double(mass)/4294967296.0-ideal)<=1.e-6,"Vicuna quantized q disagrees with independent full softmax");
            if(map[token]<0)expect(mass==0,"Vicuna masked token acquired positive probability");
        }
    }expect(complete_mass==draft_probability_mass_total,"Vicuna nonuniform q mass does not sum to 2^32");
    logits[31998]=100;plan.sample_logits_for_testing(logits,0xffffffffU,0.0F);
    PreparedTokenDistribution target;target.count=2;target.values[0]={31998,0.5,1};target.values[1]={1,0.5,1};
    auto q=plan.gather(0,target);expect(q.proposal.token==31998&&q.proposal.probability_mass==draft_probability_mass_total,"Vicuna greedy draft is not a unit-mass law");
    plan.reset();expect(!q.lease.valid(),"Vicuna reset kept probability lease live");
    refuses([&]{(void)plan.gather(0,target);},"Vicuna stale q gather accepted");
    std::fill(logits.begin(),logits.end(),-1000);logits[1]=1000;plan.sample_logits_for_testing(logits,0,1);
    q=plan.gather(0,target);expect(q.proposal.token==1&&q.proposal.probability_mass==draft_probability_mass_total,"Vicuna underflow produced an incorrect law");
    logits[1]=std::numeric_limits<float>::quiet_NaN();
    refuses([&]{plan.sample_logits_for_testing(logits,0,1);},"Vicuna nonfinite logits selected a token");
    expect(plan.poisoned(),"Vicuna invalid numeric execution did not poison the plan");
}
void chain_lifecycle(){if(!have_metal())return;const auto& f=fixture();VicunaDraftPlan plan(f.views,f.map,vocab,23);
    plan.warm_up();expect(plan.position()==0,"Vicuna warmup leaked setup tokens into generation");
    const std::array<std::uint32_t,2> prompt={1,15043};(void)plan.append(prompt);
    SamplerHistorySnapshot history;history.window_limit=64;history.length=2;history.token_ids[0]=1;history.token_ids[1]=15043;
    std::array<std::uint32_t,23> words{};for(std::size_t i=0;i<words.size();++i)words[i]=std::uint32_t(i)*0x9e3779b9U;
    const auto generated=plan.propose(15043,words,0.8F,history);
    std::array<std::uint32_t,23> saved;std::copy(generated.draft_tokens.begin(),generated.draft_tokens.end(),saved.begin());
    expect(plan.position()==25&&generated.execution.command_buffer_count==1,"Vicuna full horizon was host-staged or incompletely cached");
    (void)plan.verifier_inputs(generated.epoch);PreparedTokenDistribution target;target.count=1;target.values[0]={saved[0],1,1};
    auto q=plan.gather(0,target);plan.truncate(2);expect(!q.lease.valid(),"Vicuna truncation kept an old probability lease live");
    refuses([&]{(void)plan.verifier_inputs(generated.epoch);},"Vicuna truncation kept a verifier input lease live");
    const auto repeated=plan.propose(15043,words,0.8F,history);
    expect(std::equal(saved.begin(),saved.end(),repeated.draft_tokens.begin()),"Vicuna proposal rollback changed its own deterministic sequence");
    plan.reset();(void)plan.append(prompt);std::vector<float> host_logits(vocab);
    for(std::uint32_t depth=0;depth<23;++depth){
        (void)plan.read_logits(host_logits);plan.sample_logits_for_testing(host_logits,words[depth],0.8F);
        auto one=plan.gather(0,target);expect(one.proposal.token==saved[depth],"Vicuna device chain differs from separately synchronized causal token steps");
        (void)plan.append(std::span(&saved[depth],1));
    }
    refuses([&]{plan.truncate(26);},"Vicuna truncation extended its cache");
    std::vector<std::uint32_t> too_many(2048,1);refuses([&]{(void)plan.append(too_many);},"Vicuna append overflow accepted");
    expect(plan.position()==25,"Vicuna overflow mutated the cache frontier");
    const std::array<std::uint32_t,1> bad={32000};refuses([&]{(void)plan.append(bad);},"Vicuna out-of-vocabulary append accepted");
    plan.reset();refuses([&]{(void)plan.propose(1,std::span(words).first(3),1,history);},"Vicuna empty prefix sampled uninitialized logits");
    // Full declared capacity is real, not a smaller diagnostic window. The
    // last legal row must survive rollback; overflow cannot alter its K/V.
    (void)plan.append(too_many);expect(plan.position()==2048,"Vicuna did not retain all configured cache rows");
    std::vector<float> endpoint(vocab),replayed(vocab);(void)plan.read_logits(endpoint);
    refuses([&]{(void)plan.propose(1,std::span(words).first(1),1,history);},"Vicuna proposal crossed context boundary");
    plan.truncate(2047);(void)plan.append(std::span(too_many).first(1));(void)plan.read_logits(replayed);
    expect(endpoint==replayed,"Vicuna final context row rollback changed logits");
}
void unavailable_and_mapping(){const auto& f=fixture();using namespace tensor::backends::metal;
    set_metal_device_unavailable_for_testing(true);
    refuses([&]{VicunaDraftPlan plan(f.views,f.map,vocab,3);},"Vicuna silently used CPU without Metal");
    set_metal_device_unavailable_for_testing(false);
    auto map=f.map;map[1]=map[0];refuses([&]{VicunaDraftPlan plan(f.views,map,vocab,3);},"Vicuna accepted non-injective TLI map");
    std::fill(map.begin(),map.end(),-1);refuses([&]{VicunaDraftPlan plan(f.views,map,vocab,3);},"Vicuna accepted empty intersection");
    refuses([&]{VicunaDraftPlan plan(f.views,f.map,vocab,24);},"Vicuna accepted horizon overflow");
}
} // namespace
std::span<const TestCase> vicuna_draft_test_cases(){static const std::array<TestCase,6> cases={{
    {"Vicuna complete native transformer matches independent CPU equations and exact replay",synthetic_forward},
    {"Vicuna released checkpoint has full-vocabulary CPU/Metal parity when present",checkpoint_forward},
    {"Vicuna full-vocabulary TLI finite-grid probabilities cover boundaries, underflow and stale leases",full_probability},
    {"Vicuna source-order draft filters preserve full support ties and native history",source_processors},
    {"Vicuna 23-token device chain preserves rollback and fails closed on overflow",chain_lifecycle},
    {"Vicuna requires Metal and an injective nonempty mapping",unavailable_and_mapping},
}};return cases;}
} // namespace local_gguf_engine::tests
