#include "local_gguf_engine/engine/vicuna_draft.hpp"
#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"
#include "../tensor/backends/metal/metal_buffer.hpp"
#include "../tensor/backends/metal/metal_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace local_gguf_engine::engine {
namespace {
using tensor::BackendBuffer;
using tensor::DType;
using namespace tensor::backends::metal;
constexpr std::uint32_t hidden=768, intermediate=3072, vocabulary=32000;
constexpr std::uint32_t capacity=2048, chunk=64, tiles=vocabulary/256;

// Independent precise library: no changes to target arithmetic. Equations
// follow transformers v4.50.0 modeling_llama.py (RMSNorm, rotate_half,
// eager_attention_forward and LlamaDecoderLayer); F16 weights/KV with F32
// activations is an explicitly identified native precision transfer.
constexpr const char* source=R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void vd_embed(device const uint* ids [[buffer(0)]], device const half* w [[buffer(1)]],
 device float* x [[buffer(2)]], device atomic_uint* status [[buffer(3)]],
 constant uint4& p [[buffer(4)]],device uint* history [[buffer(5)]],uint2 i [[thread_position_in_grid]]) {
 if(i.x>=768 || i.y>=p.x) return;
 uint token=ids[p.y+i.y];
 if(token>=32000) {atomic_store_explicit(status,1u,memory_order_relaxed);token=0;}
 if(i.x==0)history[p.z+i.y]=token;
 x[i.y*768+i.x]=float(w[token*768+i.x]);
}
kernel void vd_norm(device const float* x [[buffer(0)]],device const half* w [[buffer(1)]],
 device float* y [[buffer(2)]],device atomic_uint* status [[buffer(3)]],
 constant uint4& p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]]) {
 float sum=0;
 for(uint j=lane;j<768;j+=32) {float a=x[(p.y+row)*768+j];sum+=a*a;}
 sum=simd_sum(sum);float inv=rsqrt(sum/768.0f+1.e-6f);
 for(uint j=lane;j<768;j+=32) {
   float a=x[(p.y+row)*768+j]*inv*float(w[j]);
   y[(p.z+row)*768+j]=a;
   if(!isfinite(a)) atomic_store_explicit(status,2u,memory_order_relaxed);
 }
}
// Q remains F32. K is rotated before its sole F16 store; V is unrotated.
kernel void vd_rope_store(device const float* raw_q [[buffer(0)]],
 device const float* raw_k [[buffer(1)]],device const float* raw_v [[buffer(2)]],
 device float* q [[buffer(3)]],device half* kc [[buffer(4)]],device half* vc [[buffer(5)]],
 device atomic_uint* status [[buffer(6)]],constant uint2& p [[buffer(7)]],
 uint2 i [[thread_position_in_grid]]) {
 if(i.x>=768 || i.y>=p.x) return;
 uint j=i.x%64,b=i.y*768+(i.x/64)*64+(j%32);
 float a=float(p.y+i.y)*pow(10000.0f,-float(2*(j%32))/64.0f);
 float c=cos(a),s=sin(a);
 float qr=j<32?raw_q[b]*c-raw_q[b+32]*s:raw_q[b]*s+raw_q[b+32]*c;
 float kr=j<32?raw_k[b]*c-raw_k[b+32]*s:raw_k[b]*s+raw_k[b+32]*c;
 half k=half(kr),v=half(raw_v[i.y*768+i.x]);
 q[i.y*768+i.x]=qr;kc[(p.y+i.y)*768+i.x]=k;vc[(p.y+i.y)*768+i.x]=v;
 if(!isfinite(qr)||!isfinite(float(k))||!isfinite(float(v)))
   atomic_store_explicit(status,3u,memory_order_relaxed);
}
// One cooperative group per (row,head), true uncapped causal MHA. All cache
// reads stop at the row's own position, including batched prefill chunks.
kernel void vd_attention(device const float* q [[buffer(0)]],
 device const half* kc [[buffer(1)]],device const half* vc [[buffer(2)]],
 device float* out [[buffer(3)]],device atomic_uint* status [[buffer(4)]],
 constant uint2& p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],
 uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float scores[2048];threadgroup float red[8];
 uint row=group.y,head=group.x,count=p.y+row+1;
 float mx=-INFINITY;
 for(uint token=t;token<count;token+=256) {
   float a=0;for(uint j=0;j<64;++j) a+=q[row*768+head*64+j]*float(kc[token*768+head*64+j]);
   a*=0.125f;scores[token]=a;mx=max(mx,a);
 }
 mx=simd_max(mx);if(lane==0)red[sg]=mx;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0){float m=simd_max(lane<8?red[lane]:-INFINITY);if(lane==0)red[0]=m;}
 threadgroup_barrier(mem_flags::mem_threadgroup);mx=red[0];
 float sum=0;
 for(uint token=t;token<count;token+=256){float a=exp(scores[token]-mx);scores[token]=a;sum+=a;}
 sum=simd_sum(sum);threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lane==0)red[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0){float s=simd_sum(lane<8?red[lane]:0.0f);if(lane==0)red[0]=s;}
 threadgroup_barrier(mem_flags::mem_threadgroup);sum=red[0];
 if(t<64){float a=0;for(uint token=0;token<count;++token)
     a+=(scores[token]/sum)*float(vc[token*768+head*64+t]);
   out[row*768+head*64+t]=a;
   if(!isfinite(a))atomic_store_explicit(status,4u,memory_order_relaxed);
 }
}
kernel void vd_add(device float* x [[buffer(0)]],device const float* y [[buffer(1)]],
 constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<count)x[i]+=y[i];}
kernel void vd_silu(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
 constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<count){float a=gate[i];gate[i]=(a/(1.0f+exp(-a)))*up[i];}
}
kernel void vd_history_clear(device uint* bits [[buffer(0)]],uint i [[thread_position_in_grid]]) {
 if(i<1000)bits[i]=0;
}
kernel void vd_history_mark(device const uint* history [[buffer(0)]],device atomic_uint* bits [[buffer(1)]],
 constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<count){uint token=history[i];atomic_fetch_or_explicit(bits+token/32,1u<<(token%32),memory_order_relaxed);}
}
kernel void vd_process(device const float* logits [[buffer(0)]],device const int* map [[buffer(1)]],
 device const uint* history_bits [[buffer(2)]],device float* scores [[buffer(3)]],
 device atomic_uint* status [[buffer(4)]],constant float2& options [[buffer(5)]],
 device uint* eligible [[buffer(6)]],uint i [[thread_position_in_grid]]) {
 if(i>=32000)return;float score=logits[i];
 if(!isfinite(score)){atomic_store_explicit(status,10u,memory_order_relaxed);score=0;}
 if(options.y!=1.0f && (history_bits[i/32]&(1u<<(i%32))))score=score<0?score*options.y:score/options.y;
 eligible[i]=map[i]>=0?1u:0u;
 if(map[i]<0){scores[i]=-INFINITY;return;}
 if(options.x>0)score/=options.x;
 if(!isfinite(score)){atomic_store_explicit(status,11u,memory_order_relaxed);score=0;}
 scores[i]=score;
}
struct VDPair {uint token;float score;};
bool vd_better(VDPair a,VDPair b){return a.score>b.score||(a.score==b.score&&a.token<b.token);}
// Unlike the generic draft selector, this boundary has already validated the
// raw logits. Negative infinity now intentionally means suppressed, never a
// finite sentinel which could collide with a valid extreme logit.
kernel void vd_sort_tile(device const float* scores [[buffer(0)]],device VDPair* output [[buffer(1)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 threadgroup VDPair tile[512];uint token=group*512+t;
 tile[t]=VDPair{token<32000?token:uint(-1),token<32000?scores[token]:-INFINITY};
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint width=2;width<=512;width<<=1)for(uint stride=width>>1;stride;stride>>=1){
   uint other=t^stride;
   if(other>t){bool ascending=(t&width)!=0;
     if((!ascending&&vd_better(tile[other],tile[t]))||(ascending&&vd_better(tile[t],tile[other])))
       {VDPair swap=tile[t];tile[t]=tile[other];tile[other]=swap;}}
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(token<32000)output[token]=tile[t];
}
// Ascending tail prefix for HF top-p; all cutoff ties survive top-k.
kernel void vd_filter_prefix(device const VDPair* sorted [[buffer(0)]],device const int* map [[buffer(1)]],
 device float* prefix [[buffer(2)]],device float* totals [[buffer(3)]],constant uint& top_k [[buffer(4)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 threadgroup float weights[256];uint i=group*256+t;VDPair item=sorted[31999-i];
 float cutoff=top_k?sorted[top_k-1].score:-INFINITY;
 weights[t]=(map[item.token]>=0 && item.score>=cutoff)?exp(item.score-sorted[0].score):0.0f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(t==0){float total=0;for(uint j=0;j<256;++j){total+=weights[j];prefix[group*256+j]=total;}totals[group]=total;}
}
kernel void vd_filter_apply(device const VDPair* sorted [[buffer(0)]],device const int* map [[buffer(1)]],
 device const float* prefix [[buffer(2)]],device const float* offsets [[buffer(3)]],
 device float* scores [[buffer(4)]],constant uint& top_k [[buffer(5)]],constant float2& filters [[buffer(6)]],
 device uint* eligible [[buffer(7)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=32000)return;VDPair item=sorted[31999-i];
 float cutoff=top_k?sorted[top_k-1].score:-INFINITY;
 bool keep=map[item.token]>=0&&item.score>=cutoff;
 float tail=(offsets[i/256]+prefix[i])/offsets[125];
 if(filters.x<1.0f&&tail<=1.0f-filters.x&&i!=31999)keep=false;
 if(filters.y>0.0f&&exp(item.score-sorted[0].score)<filters.y&&i!=31999)keep=false;
 eligible[item.token]=keep?1u:0u;scores[item.token]=keep?item.score:-INFINITY;
}
kernel void vd_max(device const float* logits [[buffer(0)]],device const uint* eligible [[buffer(1)]],
 device float* maxima [[buffer(2)]],device uint* argmax [[buffer(3)]],
 device atomic_uint* status [[buffer(4)]],uint group [[threadgroup_position_in_grid]],
 uint t [[thread_index_in_threadgroup]]) {
 threadgroup float m[256];threadgroup uint ids[256];uint i=group*256+t;
 float v=logits[i];if(eligible[i]&&!isfinite(v))atomic_store_explicit(status,5u,memory_order_relaxed);
 m[t]=eligible[i]?v:-INFINITY;ids[t]=i;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s>>=1){if(t<s && (m[t+s]>m[t]||(m[t+s]==m[t]&&ids[t+s]<ids[t])))
   {m[t]=m[t+s];ids[t]=ids[t+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(t==0){maxima[group]=m[0];argmax[group]=ids[0];}
}
kernel void vd_max_finish(device float* maxima [[buffer(0)]],device uint* argmax [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]]) {
 float m=-INFINITY;uint id=0;
 for(uint i=0;i<125;++i) if(maxima[i]>m||(maxima[i]==m&&argmax[i]<id)){m=maxima[i];id=argmax[i];}
 maxima[125]=m;argmax[125]=id;
 if(!isfinite(m))atomic_store_explicit(status,6u,memory_order_relaxed);
}
// Parallel exp, fixed-order positive prefix within each 256-token block.
// Subsequent block offsets use that same order. This guarantees monotone
// boundaries without a draw-dependent refusal or a top-k approximation.
kernel void vd_prefix(device const float* logits [[buffer(0)]],device const uint* eligible [[buffer(1)]],
 device const float* maxima [[buffer(2)]],device float* prefix [[buffer(3)]],
 device float* totals [[buffer(4)]],constant float& temperature [[buffer(5)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 threadgroup float w[256];uint i=group*256+t;
 w[t]=(eligible[i] && temperature>0)?exp((logits[i]-maxima[125])/temperature):0.0f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(t==0){float s=0;for(uint j=0;j<256;++j){s+=w[j];prefix[group*256+j]=s;}totals[group]=s;}
}
kernel void vd_offsets(device const float* totals [[buffer(0)]],device float* offsets [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]],constant float& temperature [[buffer(3)]]) {
 float s=0;offsets[0]=0;for(uint i=0;i<125;++i){s+=totals[i];offsets[i+1]=s;}
 if(temperature>0 && (!(s>0)||!isfinite(s)))atomic_store_explicit(status,7u,memory_order_relaxed);
}
kernel void vd_cdf(device const float* prefix [[buffer(0)]],device const float* offsets [[buffer(1)]],
 device const uint* argmax [[buffer(2)]],device ulong* cdf [[buffer(3)]],
 constant float& temperature [[buffer(4)]],device atomic_uint* status [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>32000)return;
 if(atomic_load_explicit(status,memory_order_relaxed)!=0){cdf[i]=0;return;}
 if(temperature==0){cdf[i]=i<=argmax[125]?0ul:4294967296ul;return;}
 if(i==0){cdf[i]=0;return;}if(i==32000){cdf[i]=4294967296ul;return;}
 uint b=i/256;float sum=(i%256==0)?offsets[b]:offsets[b]+prefix[i-1];
 cdf[i]=ulong(clamp((sum/offsets[125])*4294967296.0f,0.0f,4294967296.0f));
}
kernel void vd_select(device const ulong* cdf [[buffer(0)]],device const int* map [[buffer(1)]],
 device const uint* words [[buffer(2)]],device uint* draft [[buffer(3)]],
 device uint* target [[buffer(4)]],device atomic_uint* status [[buffer(5)]],
 constant uint& depth [[buffer(6)]],device const uint* eligible [[buffer(7)]],uint i [[thread_position_in_grid]]) {
 if(i>=32000)return;
 if(atomic_load_explicit(status,memory_order_relaxed)!=0)return;
 if(cdf[i]>cdf[i+1]||cdf[i+1]>4294967296ul)
   atomic_store_explicit(status,8u,memory_order_relaxed);
 if(cdf[i]<cdf[i+1]&&(map[i]<0||!eligible[i])){atomic_store_explicit(status,9u,memory_order_relaxed);return;}
 if(ulong(words[depth])>=cdf[i] && ulong(words[depth])<cdf[i+1]) {
   draft[depth]=i;target[depth+1]=uint(map[i]);
 }
}
kernel void vd_requests(device const uint* base [[buffer(0)]],device const uint* tokens [[buffer(1)]],
 device uint* rows [[buffer(2)]],constant uint& horizon [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>horizon)return;for(uint j=0;j<91;++j)rows[i*91+j]=base[j];
 for(uint j=0;j<i;++j)rows[i*91+64+j]=tokens[j+1];rows[i*91+90]=i;
}
)metal";

enum Pipeline : std::size_t { embed,norm,rope,attention,add,silu,maxima,max_finish,
    prefix,offsets,cdf,select,requests,history_clear,history_mark,process,sort_tile,filter_prefix,filter_apply,pipeline_count };
constexpr std::array<const char*,pipeline_count> pipeline_names={"vd_embed","vd_norm",
    "vd_rope_store","vd_attention","vd_add","vd_silu","vd_max","vd_max_finish",
    "vd_prefix","vd_offsets","vd_cdf","vd_select","vd_requests","vd_history_clear","vd_history_mark",
    "vd_process","vd_sort_tile","vd_filter_prefix","vd_filter_apply"};
std::uint64_t now_ns(){return static_cast<std::uint64_t>(std::chrono::duration_cast<
    std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
const MetalBackendBufferStorage& storage(const BackendBuffer& b){
    return require_metal_backend_buffer(tensor::backend_buffer_storage(b));
}
void* contents(const BackendBuffer& b){return metal_buffer_contents(storage(b));}
BackendBuffer allocate(std::initializer_list<std::uint64_t> dimensions,DType dtype=DType::f32){
    return allocate_metal_backend_buffer(tensor::TensorDescriptor::from(tensor::Shape::from(dimensions),dtype));
}
void bind(id<MTLComputeCommandEncoder> e,const BackendBuffer& b,NSUInteger index,std::size_t offset=0){
    const auto& s=storage(b);[e setBuffer:s.buffer() offset:s.base_byte_offset()+offset atIndex:index];
}
void combine(VicunaDraftExecutionStats& a,const VicunaDraftExecutionStats& b){
    a.gpu_duration_ns+=b.gpu_duration_ns;a.wall_duration_ns+=b.wall_duration_ns;
    a.readback_bytes+=b.readback_bytes;a.uploaded_bytes+=b.uploaded_bytes;
    a.command_buffer_count+=b.command_buffer_count;a.compute_dispatch_count+=b.compute_dispatch_count;
    a.memory_barrier_count+=b.memory_barrier_count;
    a.token_count+=b.token_count;
}
// Construction is staged so malformed mappings fail before any Metal upload.
// Engaged once during preparation; no optional state exists in a live plan.
struct ResidentBuffer {
    std::optional<BackendBuffer> value;
    ResidentBuffer& operator=(BackendBuffer buffer){value=std::move(buffer);return *this;}
    operator const BackendBuffer&()const{return value.value();}
};
struct TensorSource {
    const void* context;
    tensor::TensorView (*get)(const void*,std::string_view);
};
TokenSamplerOptions unrestricted_options(float temperature){
    TokenSamplerOptions options;options.temperature=temperature;options.top_k=0;options.top_p=1;
    options.min_p=0;options.repetition_penalty=1;return options;
}
void validate_draft_options(const TokenSamplerOptions& options){
    if(!std::isfinite(options.temperature)||options.temperature<0||options.top_k>vocabulary
        ||!std::isfinite(options.top_p)||options.top_p<=0||options.top_p>1
        ||!std::isfinite(options.min_p)||options.min_p<0||options.min_p>1
        ||!std::isfinite(options.repetition_penalty)||options.repetition_penalty<=0)
        throw std::invalid_argument("Vicuna draft sampler options are invalid");
}
} // namespace

struct VicunaDraftPlan::Impl {
    struct Layer {ResidentBuffer q,k,v,o,gate,up,down,input_norm,post_norm,key_cache,value_cache;};
    std::array<Layer,2> layers;
    ResidentBuffer embedding,head,final_norm;
    ResidentBuffer ids,x,normalized,q,k,v,rotated_q,attended,projected,gate,up;
    ResidentBuffer normalized_prefix,logits,status,mapping,words,draft_tokens,target_tokens;
    ResidentBuffer tile_max,tile_argmax,local_prefix,tile_totals,tile_offsets,cdfs,base_history,device_requests;
    ResidentBuffer native_history,history_bits,processed_logits,active,sort_a,sort_b;
    std::array<id<MTLComputePipelineState>,pipeline_count> pipelines {};
    MatmulPipelineState* shared=nullptr;
    id<MTLCommandBuffer> command=nil;
    id<MTLComputeCommandEncoder> encoder=nil;
    std::vector<std::int32_t> reverse_map;
    DraftProbabilityLeaseOwner probability_owner;
    std::uint32_t frontier=0,horizon=0,target_vocab=0,active_horizon=0;
    std::uint64_t epoch=0,published_epoch=0,probability_bytes=0;
    std::size_t bytes=0;
    bool poison=false,logits_current=false;
    VicunaDraftExecutionStats running;
    std::uint64_t wall_start=0;

    Impl(TensorSource source_tensors,std::span<const std::int32_t> map,
        std::uint32_t target_vocabulary,std::uint32_t max_horizon):horizon(max_horizon),target_vocab(target_vocabulary){
        if(map.size()!=vocabulary || target_vocab==0 || horizon==0 || horizon>external_speculation_max_horizon)
            throw std::invalid_argument("Vicuna draft geometry or horizon is invalid");
        reverse_map.assign(target_vocab,-1);std::size_t mapped=0;
        for(std::uint32_t i=0;i<vocabulary;++i){
            if(map[i]<-1 || (map[i]>=0&&static_cast<std::uint32_t>(map[i])>=target_vocab))
                throw std::invalid_argument("Vicuna TLI map token is out of bounds");
            if(map[i]<0)continue;
            auto& reverse=reverse_map[static_cast<std::size_t>(map[i])];
            if(reverse!=-1)throw std::invalid_argument("Vicuna TLI map is not injective");
            reverse=static_cast<std::int32_t>(i);++mapped;
        }
        if(!mapped)throw std::invalid_argument("Vicuna TLI intersection is empty");
        (void)metal_device_or_throw();shared=&shared_matmul_pipeline_state();
        MTLCompileOptions* options=[[MTLCompileOptions alloc]init];options.mathMode=MTLMathModeSafe;
        options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError* error=nil;
        id<MTLLibrary> library=[shared->device newLibraryWithSource:[NSString stringWithUTF8String:source]
            options:options error:&error];
        if(library==nil)throw std::runtime_error("Vicuna Metal library: "+ns_error_message(error,"compilation failed"));
        for(std::size_t i=0;i<pipeline_count;++i){
            id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:pipeline_names[i]]];
            pipelines[i]=[shared->device newComputePipelineStateWithFunction:fn error:&error];
            if(pipelines[i]==nil||pipelines[i].threadExecutionWidth!=32
                ||pipelines[i].maxTotalThreadsPerThreadgroup<(i==sort_tile?512U:256U))
                throw std::runtime_error("Vicuna Metal pipeline: "+ns_error_message(error,"unsupported device or creation failed"));
        }
        const auto load=[&](std::string_view name){auto view=source_tensors.get(source_tensors.context,name);
            if(view.descriptor().dtype!=DType::f16)throw std::invalid_argument("Vicuna weight must be F16");
            const auto& shape=view.descriptor().shape;
            const bool is_norm=name.find("norm.weight")!=std::string_view::npos;
            const auto input=name.find("down_proj")!=std::string_view::npos?intermediate:hidden;
            const auto output=(name=="model.embed_tokens.weight"||name=="lm_head.weight")?vocabulary:
                (name.find("gate_proj")!=std::string_view::npos||name.find("up_proj")!=std::string_view::npos)?intermediate:hidden;
            if(is_norm?(shape.rank()!=1||shape.dimension(0)!=hidden):
                (shape.rank()!=2||shape.dimension(0)!=input||shape.dimension(1)!=output))
                throw std::invalid_argument("Vicuna weight geometry is invalid: "+std::string(name));
            auto b=copy_tensor_view_to_metal_backend(view);
            bytes+=b.descriptor().byte_count;return b;};
        const auto alloc=[&](std::initializer_list<std::uint64_t> shape,DType dt=DType::f32){
            auto b=allocate(shape,dt);bytes+=b.descriptor().byte_count;
            // Fault pages into residency during preparation, not the first
            // visit to a long-context cache row in steady generation.
            std::memset(contents(b),0,b.descriptor().byte_count);return b;};
        embedding=load("model.embed_tokens.weight");head=load("lm_head.weight");final_norm=load("model.norm.weight");
        for(std::size_t i=0;i<layers.size();++i){auto& l=layers[i];const auto p="model.layers."+std::to_string(i)+".";
            l.q=load(p+"self_attn.q_proj.weight");l.k=load(p+"self_attn.k_proj.weight");
            l.v=load(p+"self_attn.v_proj.weight");l.o=load(p+"self_attn.o_proj.weight");
            l.gate=load(p+"mlp.gate_proj.weight");l.up=load(p+"mlp.up_proj.weight");l.down=load(p+"mlp.down_proj.weight");
            l.input_norm=load(p+"input_layernorm.weight");l.post_norm=load(p+"post_attention_layernorm.weight");
            l.key_cache=alloc({capacity,hidden},DType::f16);l.value_cache=alloc({capacity,hidden},DType::f16);
        }
        // The shared tensor descriptor has no integer dtype. These explicit
        // four-byte opaque slots use the same ABI as target verifier storage.
        ids=alloc({chunk});x=alloc({chunk,hidden});normalized=alloc({chunk,hidden});
        q=alloc({chunk,hidden});k=alloc({chunk,hidden});v=alloc({chunk,hidden});rotated_q=alloc({chunk,hidden});
        attended=alloc({chunk,hidden});projected=alloc({chunk,hidden});gate=alloc({chunk,intermediate});up=alloc({chunk,intermediate});
        normalized_prefix=alloc({capacity,hidden});logits=alloc({1,vocabulary});status=alloc({1});
        mapping=alloc({vocabulary});words=alloc({horizon});
        draft_tokens=alloc({horizon});target_tokens=alloc({horizon+1});
        tile_max=alloc({tiles+1});tile_argmax=alloc({tiles+1});local_prefix=alloc({vocabulary});
        tile_totals=alloc({tiles});tile_offsets=alloc({tiles+1});
        cdfs=alloc({horizon,vocabulary+1,2});
        base_history=alloc({91});device_requests=alloc({horizon+1,91});
        native_history=alloc({capacity});history_bits=alloc({vocabulary/32});processed_logits=alloc({vocabulary});active=alloc({vocabulary});
        sort_a=alloc({vocabulary,2});sort_b=alloc({vocabulary,2});
        std::memcpy(contents(mapping),map.data(),map.size_bytes());
        bytes+=reverse_map.capacity()*sizeof(std::int32_t);
    }
    void require_live()const{if(poison)throw std::logic_error("Vicuna draft plan is poisoned");}
    void invalidate()noexcept{probability_owner.invalidate();published_epoch=0;active_horizon=0;}
    void start(){
        require_live();running={};wall_start=now_ns();*static_cast<std::uint32_t*>(contents(status))=0;
        running.uploaded_bytes=sizeof(std::uint32_t);
        command=[shared->queue commandBuffer];if(command==nil)throw std::runtime_error("Vicuna command allocation failed");
        command.label=@"Vicuna resident draft";encoder=[command computeCommandEncoder];
        if(encoder==nil)throw std::runtime_error("Vicuna compute encoder allocation failed");
    }
    VicunaDraftExecutionStats finish(){
        [encoder endEncoding];encoder=nil;[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted){poison=true;invalidate();
            throw std::runtime_error("Vicuna Metal execution: "+ns_error_message(command.error,"command failed"));}
        if(*static_cast<const std::uint32_t*>(contents(status))!=0){poison=true;invalidate();
            throw std::runtime_error("Vicuna Metal execution produced an invalid numeric state (status "+
                std::to_string(*static_cast<const std::uint32_t*>(contents(status)))+")");}
        running.command_buffer_count=1;running.readback_bytes+=sizeof(std::uint32_t);
        if(command.GPUEndTime>=command.GPUStartTime)running.gpu_duration_ns=
            static_cast<std::uint64_t>((command.GPUEndTime-command.GPUStartTime)*1.e9);
        running.wall_duration_ns=now_ns()-wall_start;command=nil;return running;
    }
    void pipeline(Pipeline p){[encoder setComputePipelineState:pipelines[p]];}
    template<class T>void constant(const T& value,NSUInteger index){[encoder setBytes:&value length:sizeof(value) atIndex:index];}
    void dispatch(std::uint32_t groups,std::uint32_t threads=256,std::uint32_t rows=1){
        [encoder dispatchThreadgroups:MTLSizeMake(groups,rows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        ++running.compute_dispatch_count;
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];++running.memory_barrier_count;
    }
    void linear(const BackendBuffer& input,const BackendBuffer& weight,const BackendBuffer& output,
        std::uint32_t rows,std::uint32_t in,std::uint32_t out,std::size_t input_offset=0){
        const std::array<std::uint32_t,3> dims{rows,in,out};
        auto d=select_f16_matmul_dispatch(*shared,rows,out);
        [encoder setComputePipelineState:d.pipeline];bind(encoder,input,0,input_offset);
        bind(encoder,weight,1);bind(encoder,output,2);constant(dims,3);
        [encoder dispatchThreadgroups:d.threadgroups threadsPerThreadgroup:d.threads_per_threadgroup];
        ++running.compute_dispatch_count;
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];++running.memory_barrier_count;
    }
    void normalize(const BackendBuffer& input,const BackendBuffer& weight,const BackendBuffer& output,
        std::uint32_t rows,std::uint32_t input_row=0,std::uint32_t output_row=0){
        pipeline(norm);bind(encoder,input,0);bind(encoder,weight,1);bind(encoder,output,2);bind(encoder,status,3);
        constant(std::array<std::uint32_t,4>{rows,input_row,output_row,0},4);dispatch(rows,32);
    }
    void add_residual(std::uint32_t rows){pipeline(add);bind(encoder,x,0);bind(encoder,projected,1);
        constant(rows*hidden,2);dispatch((rows*hidden+255)/256);}
    void forward(const BackendBuffer& tokens,std::uint32_t token_offset,std::uint32_t rows,std::uint32_t position){
        pipeline(embed);bind(encoder,tokens,0);bind(encoder,embedding,1);bind(encoder,x,2);bind(encoder,status,3);
        constant(std::array<std::uint32_t,4>{rows,token_offset,position,0},4);bind(encoder,native_history,5);
        dispatch(hidden/256,256,rows);
        for(auto& l:layers){
            normalize(x,l.input_norm,normalized,rows);
            linear(normalized,l.q,q,rows,hidden,hidden);linear(normalized,l.k,k,rows,hidden,hidden);
            linear(normalized,l.v,v,rows,hidden,hidden);
            pipeline(rope);bind(encoder,q,0);bind(encoder,k,1);bind(encoder,v,2);bind(encoder,rotated_q,3);
            bind(encoder,l.key_cache,4);bind(encoder,l.value_cache,5);bind(encoder,status,6);
            constant(std::array<std::uint32_t,2>{rows,position},7);dispatch(hidden/256,256,rows);
            pipeline(attention);bind(encoder,rotated_q,0);bind(encoder,l.key_cache,1);bind(encoder,l.value_cache,2);
            bind(encoder,attended,3);bind(encoder,status,4);constant(std::array<std::uint32_t,2>{rows,position},5);
            dispatch(12,256,rows);linear(attended,l.o,projected,rows,hidden,hidden);add_residual(rows);
            normalize(x,l.post_norm,normalized,rows);
            linear(normalized,l.gate,gate,rows,hidden,intermediate);linear(normalized,l.up,up,rows,hidden,intermediate);
            pipeline(silu);bind(encoder,gate,0);bind(encoder,up,1);constant(rows*intermediate,2);
            dispatch((rows*intermediate+255)/256);linear(gate,l.down,projected,rows,intermediate,hidden);add_residual(rows);
        }
        normalize(x,final_norm,normalized_prefix,rows,0,position);running.token_count+=rows;
    }
    void project(std::uint32_t position){linear(normalized_prefix,head,logits,1,hidden,vocabulary,
        std::size_t(position-1)*hidden*sizeof(float));}
    void sample(std::uint32_t depth,const TokenSamplerOptions& options,std::uint32_t history_count){
        if(options.repetition_penalty!=1.0F){
            pipeline(history_clear);bind(encoder,history_bits,0);dispatch((vocabulary/32+255)/256);
            pipeline(history_mark);bind(encoder,native_history,0);bind(encoder,history_bits,1);
            constant(history_count,2);if(history_count)dispatch((history_count+255)/256);
        }
        pipeline(process);bind(encoder,logits,0);bind(encoder,mapping,1);bind(encoder,history_bits,2);
        bind(encoder,processed_logits,3);bind(encoder,status,4);
        constant(std::array<float,2>{options.temperature,options.repetition_penalty},5);bind(encoder,active,6);dispatch(tiles);
        const float temperature=options.temperature==0?0.0F:1.0F;
        if(temperature>0&&(options.top_k!=0||options.top_p<1||options.min_p>0)){
            pipeline(sort_tile);bind(encoder,processed_logits,0);bind(encoder,sort_a,1);
            dispatch((vocabulary+511)/512,512);
            bool in_a=true;
            for(std::uint32_t run_width=512;run_width<vocabulary;run_width*=2){
                [encoder setComputePipelineState:shared->draft_topk_merge_pipeline];
                bind(encoder,in_a?sort_a:sort_b,0);bind(encoder,in_a?sort_b:sort_a,1);
                constant(std::array<std::uint32_t,4>{1,vocabulary,vocabulary,run_width},2);
                dispatch(tiles);in_a=!in_a;
            }
            const auto& sorted=in_a?sort_a:sort_b;
            const auto top_k=static_cast<std::uint32_t>(options.top_k);
            pipeline(filter_prefix);bind(encoder,sorted,0);bind(encoder,mapping,1);bind(encoder,local_prefix,2);
            bind(encoder,tile_totals,3);constant(top_k,4);dispatch(tiles);
            pipeline(offsets);bind(encoder,tile_totals,0);bind(encoder,tile_offsets,1);bind(encoder,status,2);
            constant(temperature,3);dispatch(1,1);
            pipeline(filter_apply);bind(encoder,sorted,0);bind(encoder,mapping,1);bind(encoder,local_prefix,2);
            bind(encoder,tile_offsets,3);bind(encoder,processed_logits,4);constant(top_k,5);
            constant(std::array<float,2>{options.top_p,options.min_p},6);bind(encoder,active,7);dispatch(tiles);
        }
        pipeline(maxima);bind(encoder,processed_logits,0);bind(encoder,active,1);bind(encoder,tile_max,2);
        bind(encoder,tile_argmax,3);bind(encoder,status,4);dispatch(tiles);
        pipeline(max_finish);bind(encoder,tile_max,0);bind(encoder,tile_argmax,1);bind(encoder,status,2);dispatch(1,1);
        pipeline(prefix);bind(encoder,processed_logits,0);bind(encoder,active,1);bind(encoder,tile_max,2);
        bind(encoder,local_prefix,3);bind(encoder,tile_totals,4);constant(temperature,5);dispatch(tiles);
        pipeline(offsets);bind(encoder,tile_totals,0);bind(encoder,tile_offsets,1);bind(encoder,status,2);
        constant(temperature,3);dispatch(1,1);
        const auto offset=std::size_t(depth)*(vocabulary+1)*sizeof(std::uint64_t);
        pipeline(cdf);bind(encoder,local_prefix,0);bind(encoder,tile_offsets,1);bind(encoder,tile_argmax,2);
        bind(encoder,cdfs,3,offset);constant(temperature,4);bind(encoder,status,5);dispatch((vocabulary+1+255)/256);
        pipeline(select);bind(encoder,cdfs,0,offset);bind(encoder,mapping,1);bind(encoder,words,2);
        bind(encoder,draft_tokens,3);bind(encoder,target_tokens,4);bind(encoder,status,5);constant(depth,6);bind(encoder,active,7);dispatch(tiles);
    }
};

VicunaDraftPlan::VicunaDraftPlan(const StandaloneDraftArtifact& artifact,std::span<const std::int32_t> map,
    std::uint32_t target_vocab,std::uint32_t horizon):impl_(std::make_unique<Impl>(TensorSource{&artifact,
        [](const void* context,std::string_view name){return static_cast<const StandaloneDraftArtifact*>(context)->tensor(name);}},
        map,target_vocab,horizon)){}
VicunaDraftPlan::~VicunaDraftPlan()=default;
VicunaDraftPlan::VicunaDraftPlan(VicunaDraftPlan&&)noexcept=default;
VicunaDraftPlan& VicunaDraftPlan::operator=(VicunaDraftPlan&&)noexcept=default;
std::uint32_t VicunaDraftPlan::position()const noexcept{return impl_->frontier;}
std::uint32_t VicunaDraftPlan::horizon_capacity()const noexcept{return impl_->horizon;}
std::size_t VicunaDraftPlan::owned_bytes()const noexcept{return sizeof(Impl)+impl_->bytes;}
bool VicunaDraftPlan::poisoned()const noexcept{return impl_->poison;}
void VicunaDraftPlan::warm_up(){
    auto& p=*impl_;p.require_live();if(p.frontier!=0||p.published_epoch)
        throw std::logic_error("Vicuna warmup requires an empty prepared plan");
    std::array<std::uint32_t,chunk> input{};std::array<std::uint32_t,external_speculation_max_horizon> random{};
    (void)append(input);(void)append(std::span(input).first(1));
    SamplerHistorySnapshot history;history.window_limit=64;
    TokenSamplerOptions options;options.temperature=0.8F;options.repetition_penalty=1.1F;
    const auto proposal=propose(0,std::span(random).first(p.horizon),options,history);
    for(std::uint32_t depth=0;depth<p.horizon;++depth){PreparedTokenDistribution target;
        target.count=1;target.values[0]={proposal.target_tokens[depth],1,1};(void)gather(depth,target);}
    reset();
}
void VicunaDraftPlan::reset()noexcept{impl_->invalidate();impl_->frontier=0;impl_->logits_current=false;}
void VicunaDraftPlan::truncate(std::uint32_t position){auto& p=*impl_;p.require_live();
    if(position>p.frontier)throw std::invalid_argument("Vicuna draft truncation would grow its cache");
    p.invalidate();if(position!=p.frontier)p.logits_current=false;p.frontier=position;
}
VicunaDraftExecutionStats VicunaDraftPlan::append(std::span<const std::uint32_t> tokens){
    auto& p=*impl_;p.require_live();
    if(tokens.size()>capacity-p.frontier)throw std::invalid_argument("Vicuna draft context capacity exceeded");
    for(auto token:tokens)if(token>=vocabulary)throw std::invalid_argument("Vicuna draft input token out of vocabulary");
    if(tokens.empty())return {};
    p.invalidate();VicunaDraftExecutionStats result;const auto start=now_ns();
    @autoreleasepool {for(std::size_t offset=0;offset<tokens.size();offset+=chunk){
        const auto rows=static_cast<std::uint32_t>(std::min<std::size_t>(chunk,tokens.size()-offset));
        p.start();std::memcpy(contents(p.ids),tokens.data()+offset,rows*sizeof(std::uint32_t));
        p.running.uploaded_bytes+=rows*sizeof(std::uint32_t);p.forward(p.ids,0,rows,p.frontier);
        combine(result,p.finish());p.frontier+=rows;
    }}
    p.logits_current=false;result.wall_duration_ns=now_ns()-start;return result;
}
VicunaDraftProposal VicunaDraftPlan::propose(std::uint32_t root,std::span<const std::uint32_t> proposal_words,
    float temperature,const SamplerHistorySnapshot& history){
    return propose(root,proposal_words,unrestricted_options(temperature),history);
}
VicunaDraftProposal VicunaDraftPlan::propose(std::uint32_t root,std::span<const std::uint32_t> proposal_words,
    const TokenSamplerOptions& options,const SamplerHistorySnapshot& history){
    auto& p=*impl_;p.require_live();const auto count=static_cast<std::uint32_t>(proposal_words.size());
    validate_draft_options(options);
    if(proposal_words.empty()||proposal_words.size()>p.horizon||count>capacity-p.frontier||p.frontier==0
        ||root>=p.target_vocab)
        throw std::invalid_argument("Vicuna proposal request or remaining draft capacity is invalid");
    if(history.head>=token_sampler_max_repetition_window||history.length>history.window_limit
        ||history.window_limit>token_sampler_max_repetition_window
        ||(history.length!=0&&history.head>=history.window_limit)||(history.length==0&&history.head!=0))
        throw std::invalid_argument("Vicuna proposal sampler history is invalid");
    if(p.epoch==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Vicuna proposal epoch overflow");
    p.invalidate();p.probability_owner.begin_epoch();const auto start_position=p.frontier;
    @autoreleasepool {
        p.start();std::memcpy(contents(p.words),proposal_words.data(),proposal_words.size_bytes());
        auto* target=static_cast<std::uint32_t*>(contents(p.target_tokens));target[0]=root;
        std::fill_n(static_cast<std::uint32_t*>(contents(p.draft_tokens)),count,0U);
        std::fill_n(target+1,count,std::numeric_limits<std::uint32_t>::max());
        VerifierDeviceSelectionRow row;row.history_token_ids=history.token_ids;row.history_head=history.head;
        row.history_length=history.length;row.history_window_limit=history.window_limit;
        static_assert(sizeof(row)==91*sizeof(std::uint32_t));std::memcpy(contents(p.base_history),&row,sizeof(row));
        p.running.uploaded_bytes+=proposal_words.size_bytes()+sizeof(root)+sizeof(row)+2*count*sizeof(std::uint32_t);
        for(std::uint32_t i=0;i<count;++i){p.project(start_position+i);p.sample(i,options,start_position+i);
            p.forward(p.draft_tokens,i,1,start_position+i);}
        p.pipeline(requests);bind(p.encoder,p.base_history,0);bind(p.encoder,p.target_tokens,1);
        bind(p.encoder,p.device_requests,2);p.constant(count,3);p.dispatch(1,32);
        auto stats=p.finish();p.frontier+=count;p.logits_current=false;p.published_epoch=++p.epoch;
        p.active_horizon=count;p.probability_bytes=0;
        const auto* draft=static_cast<const std::uint32_t*>(contents(p.draft_tokens));
        for(std::uint32_t i=0;i<count;++i)if(target[i+1]>=p.target_vocab||draft[i]>=vocabulary){
            p.poison=true;p.invalidate();throw std::runtime_error("Vicuna draft selection did not publish a valid token");}
        stats.readback_bytes+=count*2*sizeof(std::uint32_t);
        stats.wall_duration_ns=now_ns()-p.wall_start;
        return {.draft_tokens={draft,count},.target_tokens={target+1,count},.start_position=start_position,
            .epoch=p.published_epoch,.execution=stats};
    }
}
VerifierDeviceInputBinding VicunaDraftPlan::verifier_inputs(std::uint64_t epoch)const{
    const auto& p=*impl_;p.require_live();if(!epoch||epoch!=p.published_epoch)
        throw std::logic_error("Vicuna verifier proposal lease is stale");
    return {.token_ids=p.target_tokens,.raw_selection_requests=p.device_requests,.arena_epoch=epoch};
}
GatheredDraftTokenDistribution VicunaDraftPlan::gather(std::uint32_t depth,const PreparedTokenDistribution& target){
    auto& p=*impl_;p.require_live();if(!p.published_epoch||depth>=p.active_horizon||target.count==0
        ||target.count>token_sampler_max_top_k)throw std::invalid_argument("Vicuna probability gather is stale or out of bounds");
    GatheredDraftTokenDistribution result;result.lease=p.probability_owner.lease(depth);result.count=target.count;
    const auto* c=static_cast<const std::uint64_t*>(contents(p.cdfs))+std::size_t(depth)*(vocabulary+1);
    const auto selected=static_cast<const std::uint32_t*>(contents(p.draft_tokens))[depth];
    result.proposal.token=static_cast<const std::uint32_t*>(contents(p.target_tokens))[depth+1];
    result.proposal.probability_mass=c[selected+1]-c[selected];result.proposal_cdf_begin=c[selected];
    p.probability_bytes+=2*sizeof(std::uint32_t)+2*sizeof(std::uint64_t);
    for(std::uint32_t i=0;i<target.count;++i){const auto token=target.values[i].token;
        if(token>=p.target_vocab)throw std::invalid_argument("Vicuna probability gather target token out of bounds");
        const auto id=p.reverse_map[token];result.target_values[i]={token,0};
        if(id>=0){result.target_values[i].probability_mass=c[std::uint32_t(id)+1]-c[std::uint32_t(id)];
            p.probability_bytes+=2*sizeof(std::uint64_t);}
    }
    return result;
}
std::uint64_t VicunaDraftPlan::probability_readback_bytes()const noexcept{return impl_->probability_bytes;}
VicunaDraftExecutionStats VicunaDraftPlan::read_logits(std::span<float> output){
    const auto started=now_ns();
    auto& p=*impl_;p.require_live();if(output.size()!=vocabulary||!p.frontier)
        throw std::invalid_argument("Vicuna logit readback requires a populated prefix and full vocabulary destination");
    VicunaDraftExecutionStats stats;@autoreleasepool {if(!p.logits_current){p.start();p.project(p.frontier);stats=p.finish();p.logits_current=true;}}
    std::memcpy(output.data(),contents(p.logits),output.size_bytes());stats.readback_bytes+=output.size_bytes();
    stats.wall_duration_ns=now_ns()-started;return stats;
}
void VicunaDraftPlan::read_cache(std::uint32_t layer,std::span<std::uint16_t> keys,std::span<std::uint16_t> values)const{
    const auto& p=*impl_;p.require_live();if(layer>=p.layers.size()||keys.size()!=std::size_t(p.frontier)*hidden||values.size()!=keys.size())
        throw std::invalid_argument("Vicuna cache readback shape mismatch");
    std::memcpy(keys.data(),contents(p.layers[layer].key_cache),keys.size_bytes());
    std::memcpy(values.data(),contents(p.layers[layer].value_cache),values.size_bytes());
}
#if defined(LOCAL_GGUF_ENGINE_TESTING)
VicunaDraftPlan::VicunaDraftPlan(std::span<const tensor::TensorView> tensors,std::span<const std::int32_t> map,
    std::uint32_t target_vocab,std::uint32_t horizon):impl_(std::make_unique<Impl>(TensorSource{&tensors,
        [](const void* context,std::string_view name){
            const auto views=*static_cast<const std::span<const tensor::TensorView>*>(context);
            if(views.size()!=21)throw std::invalid_argument("Vicuna test source requires all 21 tensors");
            if(name=="model.embed_tokens.weight")return views[0];
            if(name=="lm_head.weight")return views[1];
            if(name=="model.norm.weight")return views[2];
            constexpr std::array<std::string_view,9> suffixes={"self_attn.q_proj.weight","self_attn.k_proj.weight",
                "self_attn.v_proj.weight","self_attn.o_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight",
                "mlp.down_proj.weight","input_layernorm.weight","post_attention_layernorm.weight"};
            for(std::size_t layer=0;layer<2;++layer)for(std::size_t i=0;i<suffixes.size();++i)
                if(name=="model.layers."+std::to_string(layer)+"."+std::string(suffixes[i]))return views[3+layer*9+i];
            throw std::invalid_argument("Vicuna test tensor name");
        }},map,target_vocab,horizon)){}
void VicunaDraftPlan::sample_logits_for_testing(std::span<const float> input,std::uint32_t word,float temperature){
    sample_logits_for_testing(input,word,unrestricted_options(temperature));
}
void VicunaDraftPlan::sample_logits_for_testing(std::span<const float> input,std::uint32_t word,const TokenSamplerOptions& options){
    validate_draft_options(options);
    auto& p=*impl_;p.require_live();if(input.size()!=vocabulary)
        throw std::invalid_argument("Vicuna test logits shape or temperature");
    p.invalidate();p.probability_owner.begin_epoch();p.start();
    std::memcpy(contents(p.logits),input.data(),input.size_bytes());
    *static_cast<std::uint32_t*>(contents(p.words))=word;p.sample(0,options,p.frontier);(void)p.finish();
    p.active_horizon=1;p.published_epoch=++p.epoch;p.probability_bytes=0;p.logits_current=false;
}
#endif
} // namespace local_gguf_engine::engine
