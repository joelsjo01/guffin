#include "local_gguf_engine/engine/gemma3_draft.hpp"
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

class StandaloneDraftArtifact;
namespace {
using tensor::BackendBuffer;
using tensor::DType;
using namespace tensor::backends::metal;
constexpr std::uint32_t hidden=640, intermediate=2048, vocabulary=262144;
constexpr std::uint32_t capacity=32768, chunk=64, tiles=vocabulary/256;
constexpr bool local_layer(std::uint32_t layer) { return layer%6!=5; }

// The native GGUF has already folded Gemma's (1+w) into its F32 norm vectors.
// Q8 is never expanded into a second resident full-vocabulary matrix.
constexpr const char* source=R"metal(
#include <metal_stdlib>
using namespace metal;
// A quantized weight is scale*int8, rounded to F32 before the activation
// product. This keeps the quantizer equation explicit instead of moving the
// activation through dequantization as (activation*scale)*int8. The target's
// established Q8 kernels are intentionally not changed by this draft path.
kernel void gd_q8(device const float* input [[buffer(0)]],device const uchar* weights [[buffer(1)]],
 device float* output [[buffer(2)]],constant packed_uint3& p [[buffer(3)]],
 uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]]) {
 uint row=group.y,first=group.x*4+sg*2,second=first+1,blocks=p.y/32,stride=blocks*34;
 float a=0,b=0;
 for(uint block=0;block<blocks;++block){
   float x=input[row*p.y+block*32+lane];
   if(first<p.z){device const uchar* w=weights+first*stride+block*34;
     float value=float(*reinterpret_cast<device const half*>(w))*float(as_type<char>(w[2+lane]));
     a=fma(x,value,a);}
   if(second<p.z){device const uchar* w=weights+second*stride+block*34;
     float value=float(*reinterpret_cast<device const half*>(w))*float(as_type<char>(w[2+lane]));
     b=fma(x,value,b);}
 }
 a=simd_sum(a);b=simd_sum(b);
 if(lane==0){if(first<p.z)output[row*p.z+first]=a;if(second<p.z)output[row*p.z+second]=b;}
}
kernel void gd_embed(device const uint* ids [[buffer(0)]],device const uchar* weights [[buffer(1)]],
 device float* output [[buffer(2)]],device atomic_uint* status [[buffer(3)]],
 constant uint4& p [[buffer(4)]],device uint* history [[buffer(5)]],uint2 i [[thread_position_in_grid]]) {
 if(i.x>=640||i.y>=p.x)return;uint token=ids[p.y+i.y];
 if(token>=262144){atomic_store_explicit(status,1u,memory_order_relaxed);token=0;}
 uint block=token*20+i.x/32;device const uchar* w=weights+block*34;
 float scale=float(*reinterpret_cast<device const half*>(w));
 float value=scale*float(*reinterpret_cast<device const char*>(w+2+i.x%32))*sqrt(640.0f);
 output[i.y*640+i.x]=value;if(i.x==0)history[p.z+i.y]=token;
 if(!isfinite(value))atomic_store_explicit(status,2u,memory_order_relaxed);
}
kernel void gd_pending(device const uint* ids [[buffer(0)]],device uint* history [[buffer(1)]],
 constant uint2& p [[buffer(2)]]) {history[p.y]=ids[p.x];}
kernel void gd_norm(device const float* input [[buffer(0)]],device const float* weight [[buffer(1)]],
 device float* output [[buffer(2)]],device atomic_uint* status [[buffer(3)]],
 constant uint2& p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]]) {
 float sum=0;for(uint j=lane;j<p.y;j+=32){float a=input[row*p.y+j];sum+=a*a;}
 sum=simd_sum(sum);float inv=rsqrt(sum/float(p.y)+1.e-6f);
 if(!isfinite(sum)||!isfinite(inv)||!(inv>0))atomic_store_explicit(status,3u,memory_order_relaxed);
 for(uint j=lane;j<p.y;j+=32){float a=input[row*p.y+j]*inv*weight[j];output[row*p.y+j]=a;
   if(!isfinite(a))atomic_store_explicit(status,3u,memory_order_relaxed);}
}
kernel void gd_rope_store(device const float* raw_q [[buffer(0)]],device const float* raw_k [[buffer(1)]],
 device const float* raw_v [[buffer(2)]],device float* q [[buffer(3)]],
 device float* k [[buffer(4)]],device float* v [[buffer(5)]],device atomic_uint* status [[buffer(6)]],
 constant uint4& p [[buffer(7)]],constant float& theta [[buffer(8)]],uint2 i [[thread_position_in_grid]]) {
 if(i.x>=1024||i.y>=p.x)return;uint j=i.x%256,b=i.y*1024+(i.x/256)*256+j%128;
 float angle=float(p.y+i.y)*pow(theta,-float(j%128)/128.0f),c=cos(angle),s=sin(angle);
 float qr=j<128?raw_q[b]*c-raw_q[b+128]*s:raw_q[b]*s+raw_q[b+128]*c;
 q[i.y*1024+i.x]=qr;if(!isfinite(qr))atomic_store_explicit(status,4u,memory_order_relaxed);
 if(i.x<256){b=i.y*256+j%128;
   float kr=j<128?raw_k[b]*c-raw_k[b+128]*s:raw_k[b]*s+raw_k[b+128]*c;
   float key=kr,value=raw_v[i.y*256+j];uint dst=(p.y+i.y-p.z)*256+j;
   k[dst]=key;v[dst]=value;
   if(!isfinite(float(key))||!isfinite(float(value)))atomic_store_explicit(status,4u,memory_order_relaxed);
 }
}
struct GDAttention {uint rows;uint position;uint private_begin;uint local;uint max_parts;};
// Four independent SIMD online-softmax streams per query head/partition.
// Partition and summation order depend only on the logical query position,
// never prefill batch size, preserving exact replay across batching changes.
kernel void gd_attention(device const float4* q [[buffer(0)]],
 device const float4* resident_k [[buffer(1)]],device const float4* resident_v [[buffer(2)]],
 device const float4* private_k [[buffer(3)]],device const float4* private_v [[buffer(4)]],
 device float4* partial [[buffer(5)]],device float2* scales [[buffer(6)]],
 constant GDAttention& p [[buffer(7)]],uint3 group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group.y,head=group.x,position=p.position+row;
 uint first=p.local&&position>=511?position-511:0,count=position-first+1;
 uint parts=min(32u,(count+511)/512),partition=group.z;
 if(row>=p.rows||partition>=parts)return;
 float4 q0=q[(row*4+head)*64+lane*2],q1=q[(row*4+head)*64+lane*2+1];
 float maximum=-INFINITY,denom=0;float4 acc0=0,acc1=0;
 uint blocks=(count+31)/32;
 for(uint block=partition*4+sg;block<blocks;block+=parts*4){
   for(uint token=block*32;token<min(block*32+32,count);++token){
     uint pos=first+token,slot=pos>=p.private_begin?pos-p.private_begin:(p.local?pos&511:pos);
     device const float4* ks=pos>=p.private_begin?private_k:resident_k;
     device const float4* vs=pos>=p.private_begin?private_v:resident_v;
     float score=simd_sum(dot(q0,float4(ks[slot*64+lane*2]))+dot(q1,float4(ks[slot*64+lane*2+1])))*0.0625f;
     float next=max(maximum,score),old=denom==0?0.0f:exp(maximum-next),add=exp(score-next);
     acc0=acc0*old+float4(vs[slot*64+lane*2])*add;
     acc1=acc1*old+float4(vs[slot*64+lane*2+1])*add;
     denom=denom*old+add;maximum=next;
   }
 }
 threadgroup float4 accum[4*64];threadgroup float2 reductions[4];
 accum[sg*64+lane*2]=acc0;accum[sg*64+lane*2+1]=acc1;
 if(lane==0)reductions[sg]=float2(maximum,denom);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg!=0)return;
 maximum=-INFINITY;denom=0;acc0=0;acc1=0;
 for(uint i=0;i<4;++i){float2 a=reductions[i];if(a.y==0)continue;
   float next=max(maximum,a.x),old=denom==0?0.0f:exp(maximum-next),add=exp(a.x-next);
   acc0=acc0*old+accum[i*64+lane*2]*add;acc1=acc1*old+accum[i*64+lane*2+1]*add;
   denom=denom*old+a.y*add;maximum=next;
 }
 uint index=(row*4+head)*32+partition;
 partial[index*64+lane*2]=acc0;partial[index*64+lane*2+1]=acc1;
 if(lane==0)scales[index]=float2(maximum,denom);
}
kernel void gd_attention_reduce(device const float4* partial [[buffer(0)]],device const float2* scales [[buffer(1)]],
 device float4* output [[buffer(2)]],device atomic_uint* status [[buffer(3)]],
 constant GDAttention& p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint row=group.y,head=group.x,position=p.position+row,count=p.local?min(position+1,512u):position+1;
 uint parts=min(32u,(count+511)/512);float maximum=-INFINITY,denom=0;float4 acc0=0,acc1=0;
 for(uint part=0;part<parts;++part){uint index=(row*4+head)*32+part;float2 a=scales[index];
   float next=max(maximum,a.x),old=denom==0?0.0f:exp(maximum-next),add=exp(a.x-next);
   acc0=acc0*old+partial[index*64+lane*2]*add;acc1=acc1*old+partial[index*64+lane*2+1]*add;
   denom=denom*old+a.y*add;maximum=next;
 }
 acc0/=denom;acc1/=denom;
 output[(row*4+head)*64+lane*2]=acc0;output[(row*4+head)*64+lane*2+1]=acc1;
 if(!all(isfinite(acc0))||!all(isfinite(acc1)))atomic_store_explicit(status,5u,memory_order_relaxed);
}
kernel void gd_gelu(device float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
 constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<count){float x=gate[i];gate[i]=0.5f*x*(1.0f+tanh(0.7978845608028654f*(x+0.044715f*x*x*x)))*up[i];}
}
kernel void gd_add(device float* residual [[buffer(0)]],device const float* branch [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]],constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<count){float x=residual[i]+branch[i];residual[i]=x;
 if(!isfinite(x))atomic_store_explicit(status,6u,memory_order_relaxed);}
}
kernel void gd_publish(device const float* sk [[buffer(0)]],device const float* sv [[buffer(1)]],
 device float* dk [[buffer(2)]],device float* dv [[buffer(3)]],constant uint4& p [[buffer(4)]],
 uint2 i [[thread_position_in_grid]]) {
 if(i.x>=256||i.y>=p.y)return;uint position=p.x+i.y;
 uint source=(position-p.z)*256+i.x,dest=(position&p.w)*256+i.x;dk[dest]=sk[source];dv[dest]=sv[source];
}
kernel void gd_copy(device const float* src [[buffer(0)]],device float* dst [[buffer(1)]],
 uint i [[thread_position_in_grid]]){if(i<640)dst[i]=src[i];}
kernel void gd_history_clear(device uint* bits [[buffer(0)]],uint i [[thread_position_in_grid]]) {
 if(i<8192)bits[i]=0;
}
kernel void gd_history_mark(device const uint* history [[buffer(0)]],device atomic_uint* bits [[buffer(1)]],
 constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<count){uint token=history[i];atomic_fetch_or_explicit(bits+token/32,1u<<(token%32),memory_order_relaxed);}
}
kernel void gd_process(device const float* logits [[buffer(0)]],device const int* map [[buffer(1)]],
 device const uint* history_bits [[buffer(2)]],device float* scores [[buffer(3)]],
 device atomic_uint* status [[buffer(4)]],constant float2& options [[buffer(5)]],
 device uint* eligible [[buffer(6)]],uint i [[thread_position_in_grid]]) {
 if(i>=262144)return;float score=logits[i];
 if(!isfinite(score)){atomic_store_explicit(status,10u,memory_order_relaxed);score=0;}
 if(options.y!=1.0f && (history_bits[i/32]&(1u<<(i%32))))score=score<0?score*options.y:score/options.y;
 eligible[i]=map[i]>=0?1u:0u;
 if(map[i]<0){scores[i]=-INFINITY;return;}
 if(options.x>0)score/=options.x;
 if(!isfinite(score)){atomic_store_explicit(status,11u,memory_order_relaxed);score=0;}
 scores[i]=score;
}
struct GDPair {uint token;float score;};
bool gd_better(GDPair a,GDPair b){return a.score>b.score||(a.score==b.score&&a.token<b.token);}
// Unlike the generic draft selector, this boundary has already validated the
// raw logits. Negative infinity now intentionally means suppressed, never a
// finite sentinel which could collide with a valid extreme logit.
kernel void gd_sort_tile(device const float* scores [[buffer(0)]],device GDPair* output [[buffer(1)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 threadgroup GDPair tile[512];uint token=group*512+t;
 tile[t]=GDPair{token<262144?token:uint(-1),token<262144?scores[token]:-INFINITY};
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint width=2;width<=512;width<<=1)for(uint stride=width>>1;stride;stride>>=1){
   uint other=t^stride;
   if(other>t){bool ascending=(t&width)!=0;
     if((!ascending&&gd_better(tile[other],tile[t]))||(ascending&&gd_better(tile[t],tile[other])))
       {GDPair swap=tile[t];tile[t]=tile[other];tile[other]=swap;}}
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(token<262144)output[token]=tile[t];
}
// Ascending tail prefix for HF top-p; all cutoff ties survive top-k.
kernel void gd_filter_prefix(device const GDPair* sorted [[buffer(0)]],device const int* map [[buffer(1)]],
 device float* prefix [[buffer(2)]],device float* totals [[buffer(3)]],constant uint& top_k [[buffer(4)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 threadgroup float weights[256];uint i=group*256+t;GDPair item=sorted[262143-i];
 float cutoff=top_k?sorted[top_k-1].score:-INFINITY;
 weights[t]=(map[item.token]>=0 && item.score>=cutoff)?exp(item.score-sorted[0].score):0.0f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(t==0){float total=0;for(uint j=0;j<256;++j){total+=weights[j];prefix[group*256+j]=total;}totals[group]=total;}
}
kernel void gd_filter_apply(device const GDPair* sorted [[buffer(0)]],device const int* map [[buffer(1)]],
 device const float* prefix [[buffer(2)]],device const float* offsets [[buffer(3)]],
 device float* scores [[buffer(4)]],constant uint& top_k [[buffer(5)]],constant float2& filters [[buffer(6)]],
 device uint* eligible [[buffer(7)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=262144)return;GDPair item=sorted[262143-i];
 float cutoff=top_k?sorted[top_k-1].score:-INFINITY;
 bool keep=map[item.token]>=0&&item.score>=cutoff;
 float tail=(offsets[i/256]+prefix[i])/offsets[1024];
 if(filters.x<1.0f&&tail<=1.0f-filters.x&&i!=262143)keep=false;
 if(filters.y>0.0f&&exp(item.score-sorted[0].score)<filters.y&&i!=262143)keep=false;
 eligible[item.token]=keep?1u:0u;scores[item.token]=keep?item.score:-INFINITY;
}
kernel void gd_max(device const float* logits [[buffer(0)]],device const uint* eligible [[buffer(1)]],
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
kernel void gd_max_finish(device float* maxima [[buffer(0)]],device uint* argmax [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]]) {
 float m=-INFINITY;uint id=0;
 for(uint i=0;i<1024;++i) if(maxima[i]>m||(maxima[i]==m&&argmax[i]<id)){m=maxima[i];id=argmax[i];}
 maxima[1024]=m;argmax[1024]=id;
 if(!isfinite(m))atomic_store_explicit(status,6u,memory_order_relaxed);
}
// Parallel exp, fixed-order positive prefix within each 256-token block.
// Subsequent block offsets use that same order. This guarantees monotone
// boundaries without a draw-dependent refusal or a top-k approximation.
kernel void gd_prefix(device const float* logits [[buffer(0)]],device const uint* eligible [[buffer(1)]],
 device const float* maxima [[buffer(2)]],device float* prefix [[buffer(3)]],
 device float* totals [[buffer(4)]],constant float& temperature [[buffer(5)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 threadgroup float w[256];uint i=group*256+t;
 w[t]=(eligible[i] && temperature>0)?exp((logits[i]-maxima[1024])/temperature):0.0f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(t==0){float s=0;for(uint j=0;j<256;++j){s+=w[j];prefix[group*256+j]=s;}totals[group]=s;}
}
kernel void gd_offsets(device const float* totals [[buffer(0)]],device float* offsets [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]],constant float& temperature [[buffer(3)]]) {
 float s=0;offsets[0]=0;for(uint i=0;i<1024;++i){s+=totals[i];offsets[i+1]=s;}
 if(temperature>0 && (!(s>0)||!isfinite(s)))atomic_store_explicit(status,7u,memory_order_relaxed);
}
kernel void gd_cdf(device const float* prefix [[buffer(0)]],device const float* offsets [[buffer(1)]],
 device const uint* argmax [[buffer(2)]],device ulong* cdf [[buffer(3)]],
 constant float& temperature [[buffer(4)]],device atomic_uint* status [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>262144)return;
 if(atomic_load_explicit(status,memory_order_relaxed)!=0){cdf[i]=0;return;}
 if(temperature==0){cdf[i]=i<=argmax[1024]?0ul:4294967296ul;return;}
 if(i==0){cdf[i]=0;return;}if(i==262144){cdf[i]=4294967296ul;return;}
 uint b=i/256;float sum=(i%256==0)?offsets[b]:offsets[b]+prefix[i-1];
 cdf[i]=ulong(clamp((sum/offsets[1024])*4294967296.0f,0.0f,4294967296.0f));
}
kernel void gd_select(device const ulong* cdf [[buffer(0)]],device const int* map [[buffer(1)]],
 device const uint* words [[buffer(2)]],device uint* draft [[buffer(3)]],
 device uint* target [[buffer(4)]],device atomic_uint* status [[buffer(5)]],
 constant uint& depth [[buffer(6)]],device const uint* eligible [[buffer(7)]],uint i [[thread_position_in_grid]]) {
 if(i>=262144)return;
 if(atomic_load_explicit(status,memory_order_relaxed)!=0)return;
 if(cdf[i]>cdf[i+1]||cdf[i+1]>4294967296ul)
   atomic_store_explicit(status,8u,memory_order_relaxed);
 if(cdf[i]<cdf[i+1]&&(map[i]<0||!eligible[i])){atomic_store_explicit(status,9u,memory_order_relaxed);return;}
 if(ulong(words[depth])>=cdf[i] && ulong(words[depth])<cdf[i+1]) {
   draft[depth]=i;target[depth+1]=uint(map[i]);
 }
}
kernel void gd_requests(device const uint* base [[buffer(0)]],device const uint* tokens [[buffer(1)]],
 device uint* rows [[buffer(2)]],constant uint& horizon [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>horizon)return;for(uint j=0;j<91;++j)rows[i*91+j]=base[j];
 for(uint j=0;j<i;++j)rows[i*91+64+j]=tokens[j+1];rows[i*91+90]=i;
}
)metal";
enum Pipeline : std::size_t {q8,embed,pending,norm,rope,attention,attention_reduce,gelu,add,publish,copy,
    maxima,max_finish,prefix,offsets,cdf,select,requests,history_clear,history_mark,process,
    sort_tile,filter_prefix,filter_apply,pipeline_count};
constexpr std::array<const char*,pipeline_count> names={"gd_q8","gd_embed","gd_pending","gd_norm","gd_rope_store",
    "gd_attention","gd_attention_reduce","gd_gelu","gd_add","gd_publish","gd_copy",
    "gd_max","gd_max_finish","gd_prefix","gd_offsets","gd_cdf","gd_select","gd_requests",
    "gd_history_clear","gd_history_mark","gd_process","gd_sort_tile","gd_filter_prefix","gd_filter_apply"};
std::uint64_t now_ns(){return static_cast<std::uint64_t>(std::chrono::duration_cast<
    std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
const MetalBackendBufferStorage& storage(const BackendBuffer& b){
    return require_metal_backend_buffer(tensor::backend_buffer_storage(b));}
void* contents(const BackendBuffer& b){return metal_buffer_contents(storage(b));}
void bind(id<MTLComputeCommandEncoder> encoder,const BackendBuffer& b,NSUInteger index,std::size_t offset=0){
    const auto& s=storage(b);[encoder setBuffer:s.buffer() offset:s.base_byte_offset()+offset atIndex:index];}
void combine(StandaloneDraftExecutionStats& a,const StandaloneDraftExecutionStats& b){
#define ADD(field) if(b.field>std::numeric_limits<decltype(a.field)>::max()-a.field) \
    throw std::overflow_error("Gemma3 draft statistics overflow"); a.field+=b.field
    ADD(gpu_duration_ns);ADD(wall_duration_ns);ADD(readback_bytes);ADD(uploaded_bytes);
    ADD(command_buffer_count);ADD(compute_dispatch_count);ADD(memory_barrier_count);ADD(token_count);
#undef ADD
}
struct Buffer {
    std::optional<BackendBuffer> value;
    Buffer& operator=(BackendBuffer b){value=std::move(b);return *this;}
    operator const BackendBuffer&() const{return value.value();}
};
struct TensorSource {const void* context;tensor::TensorView (*get)(const void*,std::string_view);};
constexpr std::array<const char*,13> suffixes={"attn_q.weight","attn_k.weight","attn_v.weight",
    "attn_output.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight","attn_norm.weight",
    "post_attention_norm.weight","ffn_norm.weight","post_ffw_norm.weight","attn_q_norm.weight","attn_k_norm.weight"};
void validate_options(const TokenSamplerOptions& o){
    if(!std::isfinite(o.temperature)||o.temperature<0||o.top_k>vocabulary
        ||!std::isfinite(o.top_p)||o.top_p<=0||o.top_p>1||!std::isfinite(o.min_p)||o.min_p<0||o.min_p>1
        ||!std::isfinite(o.repetition_penalty)||o.repetition_penalty<=0)
        throw std::invalid_argument("Gemma3 draft sampler options are invalid");
}
} // namespace

struct Gemma3DraftPlan::Impl {
    struct Layer {
        std::array<Buffer,13> weights;
        Buffer key,value,private_key,private_value;
    };
    std::array<Layer,layer_count> layers;
    Buffer embedding,final_norm,ids,pending_input,x,normed,q,k,v,norm_q,norm_k,rotated_q,attended,projected,branch,gate,up;
    Buffer partial,scales,durable_norm,private_norm,logits,status,mapping,words,draft_tokens,target_tokens;
    Buffer tile_max,tile_argmax,local_prefix,tile_totals,tile_offsets,cdfs,base_history,device_requests;
    Buffer native_history,history_bits,processed_logits,active,sort_a,sort_b;
    std::array<id<MTLComputePipelineState>,pipeline_count> pipelines {};
    MatmulPipelineState* shared=nullptr;
    id<MTLCommandBuffer> command=nil;
    id<MTLComputeCommandEncoder> encoder=nil;
    std::vector<std::int32_t> reverse_map;
    DraftProbabilityLeaseOwner probability_owner;
    std::uint32_t frontier=0,materialized=0,horizon=0,target_vocab=0,transaction_capacity=0,active_horizon=0;
    std::uint32_t transaction_begin=0,saved_frontier=0,saved_materialized=0,saved_pending_token=0,pending_token=0;
    std::uint64_t epoch=0,published_epoch=0,probability_bytes=0,wall_start=0;
    std::size_t bytes=0;
    bool poison=false,in_transaction=false,has_pending=false,saved_pending=false;
    StandaloneDraftExecutionStats running;

    Impl(TensorSource source_tensors,std::span<const std::int32_t> map,std::uint32_t target_vocabulary,
        std::uint32_t max_horizon,std::uint32_t transaction_rows)
        :horizon(max_horizon),target_vocab(target_vocabulary),transaction_capacity(transaction_rows){
        if(map.size()!=vocabulary||target_vocab==0||horizon==0||horizon>external_speculation_max_horizon
            ||transaction_rows<horizon+2||transaction_rows>capacity)
            throw std::invalid_argument("Gemma3 draft geometry, transaction capacity or horizon is invalid");
        reverse_map.assign(target_vocab,-1);std::size_t mapped=0;
        for(std::uint32_t i=0;i<vocabulary;++i){
            if(map[i]<-1||(map[i]>=0&&static_cast<std::uint32_t>(map[i])>=target_vocab))
                throw std::invalid_argument("Gemma3 TLI map token is out of bounds");
            if(map[i]<0)continue;auto& inverse=reverse_map[std::size_t(map[i])];
            if(inverse!=-1)throw std::invalid_argument("Gemma3 TLI map is not injective");
            inverse=static_cast<std::int32_t>(i);++mapped;
        }
        if(!mapped)throw std::invalid_argument("Gemma3 TLI intersection is empty");
        (void)metal_device_or_throw();shared=&shared_matmul_pipeline_state();
        MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        NSError* error=nil;
        auto library=[shared->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
        if(!library)throw std::runtime_error("Gemma3 Metal library: "+ns_error_message(error,"compilation failed"));
        for(std::size_t i=0;i<pipeline_count;++i){
            auto fn=[library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
            pipelines[i]=[shared->device newComputePipelineStateWithFunction:fn error:&error];
            if(!pipelines[i]||pipelines[i].threadExecutionWidth!=32
                ||pipelines[i].maxTotalThreadsPerThreadgroup<(i==sort_tile?512U:256U))
                throw std::runtime_error("Gemma3 Metal pipeline: "+ns_error_message(error,"unsupported device"));
        }
        const auto load=[&](std::string_view name,std::uint32_t input,std::uint32_t output,DType dtype){
            auto view=source_tensors.get(source_tensors.context,name);const auto& d=view.descriptor();
            if(d.dtype!=dtype||(output==0?(d.shape.rank()!=1||d.shape.dimension(0)!=input):
                (d.shape.rank()!=2||d.shape.dimension(0)!=input||d.shape.dimension(1)!=output)))
                throw std::invalid_argument("Gemma3 weight has incompatible shape or dtype: "+std::string(name));
            auto b=copy_tensor_view_to_metal_backend(view);bytes+=b.descriptor().byte_count;return b;
        };
        const auto alloc=[&](std::initializer_list<std::uint64_t> shape,DType dtype=DType::f32){
            auto b=allocate_metal_backend_buffer(tensor::TensorDescriptor::from(tensor::Shape::from(shape),dtype));
            bytes+=b.descriptor().byte_count;std::memset(contents(b),0,b.descriptor().byte_count);return b;
        };
        embedding=load("token_embd.weight",hidden,vocabulary,DType::q8_0);
        final_norm=load("output_norm.weight",hidden,0,DType::f32);
        constexpr std::array<std::uint32_t,7> inputs={hidden,hidden,hidden,1024,hidden,hidden,intermediate};
        constexpr std::array<std::uint32_t,7> outputs={1024,256,256,hidden,intermediate,intermediate,hidden};
        for(std::uint32_t i=0;i<layer_count;++i){auto& layer=layers[i];const auto prefix="blk."+std::to_string(i)+".";
            for(std::uint32_t w=0;w<13;++w)layer.weights[w]=load(prefix+suffixes[w],
                w<7?inputs[w]:(w>=11?256:hidden),w<7?outputs[w]:0,w<7?DType::q8_0:DType::f32);
            const auto rows=local_layer(i)?512:capacity;
            layer.key=alloc({rows,256});layer.value=alloc({rows,256});
            layer.private_key=alloc({std::max(chunk,transaction_rows),256});
            layer.private_value=alloc({std::max(chunk,transaction_rows),256});
        }
        ids=alloc({chunk});pending_input=alloc({1});x=alloc({chunk,hidden});normed=alloc({chunk,hidden});q=alloc({chunk,1024});
        k=alloc({chunk,256});v=alloc({chunk,256});norm_q=alloc({chunk,1024});norm_k=alloc({chunk,256});
        rotated_q=alloc({chunk,1024});attended=alloc({chunk,1024});projected=alloc({chunk,hidden});
        branch=alloc({chunk,hidden});gate=alloc({chunk,intermediate});up=alloc({chunk,intermediate});
        partial=alloc({chunk,4,32,256});scales=alloc({chunk,4,32,2});
        durable_norm=alloc({1,hidden});private_norm=alloc({std::max(chunk,transaction_rows),hidden});
        logits=alloc({vocabulary});status=alloc({1});mapping=alloc({vocabulary});words=alloc({horizon});
        draft_tokens=alloc({horizon});target_tokens=alloc({horizon+1});tile_max=alloc({tiles+1});
        tile_argmax=alloc({tiles+1});local_prefix=alloc({vocabulary});tile_totals=alloc({tiles});
        tile_offsets=alloc({tiles+1});cdfs=alloc({horizon,vocabulary+1,2});base_history=alloc({91});
        device_requests=alloc({horizon+1,91});native_history=alloc({capacity});
        history_bits=alloc({vocabulary/32});processed_logits=alloc({vocabulary});active=alloc({vocabulary});
        sort_a=alloc({vocabulary,2});sort_b=alloc({vocabulary,2});
        std::memcpy(contents(mapping),map.data(),map.size_bytes());bytes+=reverse_map.capacity()*sizeof(std::int32_t);
    }
    void require_live()const{if(poison)throw std::logic_error("Gemma3 draft plan is poisoned");}
    void invalidate()noexcept{probability_owner.invalidate();published_epoch=0;active_horizon=0;}
    void start(){
        require_live();running={};wall_start=now_ns();*static_cast<std::uint32_t*>(contents(status))=0;
        running.uploaded_bytes=sizeof(std::uint32_t);command=[shared->queue commandBuffer];
        if(!command)throw std::runtime_error("Gemma3 command allocation failed");
        command.label=@"Gemma3 resident draft";encoder=[command computeCommandEncoder];
        if(!encoder)throw std::runtime_error("Gemma3 encoder allocation failed");
    }
    StandaloneDraftExecutionStats finish(){
        [encoder endEncoding];encoder=nil;[command commit];[command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted){
            poison=true;invalidate();throw std::runtime_error("Gemma3 Metal execution: "+ns_error_message(command.error,"command failed"));
        }
        const auto code=*static_cast<const std::uint32_t*>(contents(status));
        if(code){poison=true;invalidate();throw std::runtime_error("Gemma3 draft numerical refusal: "+std::to_string(code));}
        running.command_buffer_count=1;running.readback_bytes+=sizeof(std::uint32_t);
        if(command.GPUEndTime>=command.GPUStartTime)
            running.gpu_duration_ns=static_cast<std::uint64_t>((command.GPUEndTime-command.GPUStartTime)*1.e9);
        running.wall_duration_ns=now_ns()-wall_start;command=nil;return running;
    }
    void pipeline(Pipeline p){[encoder setComputePipelineState:pipelines[p]];}
    template<class T>void constant(const T& value,NSUInteger index){[encoder setBytes:&value length:sizeof(value) atIndex:index];}
    void barrier(){[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];++running.memory_barrier_count;}
    void dispatch(std::uint32_t groups,std::uint32_t threads=256,std::uint32_t rows=1,std::uint32_t partitions=1){
        [encoder dispatchThreadgroups:MTLSizeMake(groups,rows,partitions)
                  threadsPerThreadgroup:MTLSizeMake(threads,1,1)];++running.compute_dispatch_count;barrier();
    }
    void linear(const BackendBuffer& input,const BackendBuffer& weight,const BackendBuffer& output,
        std::uint32_t rows,std::uint32_t in,std::uint32_t out,std::size_t input_offset=0){
        // Same row-parallel reduction for scalar decode and every prefill row.
        // A Q8 row is 20 blocks at width640; no Q4/width-two surrogate is used.
        pipeline(q8);
        bind(encoder,input,0,input_offset);bind(encoder,weight,1);bind(encoder,output,2);
        constant(std::array<std::uint32_t,3>{rows,in,out},3);dispatch((out+3)/4,64,rows);
    }
    void normalize(const BackendBuffer& input,const BackendBuffer& weight,const BackendBuffer& output,
        std::uint32_t rows,std::uint32_t width,std::size_t input_offset=0,std::size_t output_offset=0){
        pipeline(norm);bind(encoder,input,0,input_offset);bind(encoder,weight,1);bind(encoder,output,2,output_offset);
        bind(encoder,status,3);constant(std::array<std::uint32_t,2>{rows,width},4);dispatch(rows,32);
    }
    void residual(std::uint32_t rows){
        pipeline(add);bind(encoder,x,0);bind(encoder,branch,1);bind(encoder,status,2);
        constant(rows*hidden,3);dispatch((rows*hidden+255)/256);
    }
    void publish_rows(std::uint32_t begin,std::uint32_t end,std::uint32_t private_begin){
        if(begin==end)return;
        for(std::uint32_t i=0;i<layer_count;++i){auto& l=layers[i];
            const auto first=local_layer(i)&&end-begin>512?end-512:begin;
            pipeline(publish);bind(encoder,l.private_key,0);bind(encoder,l.private_value,1);
            bind(encoder,l.key,2);bind(encoder,l.value,3);
            constant(std::array<std::uint32_t,4>{first,end-first,private_begin,local_layer(i)?511U:capacity-1},4);
            dispatch(1,256,end-first);
        }
        pipeline(copy);bind(encoder,private_norm,0,std::size_t(end-1-private_begin)*hidden*sizeof(float));
        bind(encoder,durable_norm,1);dispatch(3);
    }
    void forward(const BackendBuffer& tokens,std::uint32_t token_offset,std::uint32_t rows,std::uint32_t position){
        const auto private_begin=in_transaction?transaction_begin:position;
        pipeline(embed);bind(encoder,tokens,0);bind(encoder,embedding,1);bind(encoder,x,2);bind(encoder,status,3);
        constant(std::array<std::uint32_t,4>{rows,token_offset,position,0},4);bind(encoder,native_history,5);
        dispatch(3,256,rows);
        for(std::uint32_t i=0;i<layer_count;++i){auto& l=layers[i];auto& w=l.weights;
            normalize(x,w[7],normed,rows,hidden);
            linear(normed,w[0],q,rows,hidden,1024);linear(normed,w[1],k,rows,hidden,256);
            linear(normed,w[2],v,rows,hidden,256);
            normalize(q,w[11],norm_q,rows*4,256);normalize(k,w[12],norm_k,rows,256);
            pipeline(rope);bind(encoder,norm_q,0);bind(encoder,norm_k,1);bind(encoder,v,2);
            bind(encoder,rotated_q,3);bind(encoder,l.private_key,4);bind(encoder,l.private_value,5);bind(encoder,status,6);
            constant(std::array<std::uint32_t,4>{rows,position,private_begin,0},7);
            constant(local_layer(i)?10000.0F:1000000.0F,8);dispatch(4,256,rows);
            const auto max_parts=local_layer(i)?1U:std::min(32U,(position+rows+511)/512);
            const std::array<std::uint32_t,5> params{rows,position,private_begin,local_layer(i)?1U:0U,max_parts};
            pipeline(attention);bind(encoder,rotated_q,0);bind(encoder,l.key,1);bind(encoder,l.value,2);
            bind(encoder,l.private_key,3);bind(encoder,l.private_value,4);bind(encoder,partial,5);bind(encoder,scales,6);
            constant(params,7);dispatch(4,128,rows,max_parts);
            pipeline(attention_reduce);bind(encoder,partial,0);bind(encoder,scales,1);
            bind(encoder,attended,2);bind(encoder,status,3);constant(params,4);dispatch(4,32,rows);
            linear(attended,w[3],projected,rows,1024,hidden);
            normalize(projected,w[8],branch,rows,hidden);residual(rows);
            normalize(x,w[9],normed,rows,hidden);
            linear(normed,w[4],gate,rows,hidden,intermediate);linear(normed,w[5],up,rows,hidden,intermediate);
            pipeline(gelu);bind(encoder,gate,0);bind(encoder,up,1);constant(rows*intermediate,2);
            dispatch((rows*intermediate+255)/256);
            linear(gate,w[6],projected,rows,intermediate,hidden);
            normalize(projected,w[10],branch,rows,hidden);residual(rows);
        }
        normalize(x,final_norm,private_norm,rows,hidden,0,std::size_t(position-private_begin)*hidden*sizeof(float));
        if(!in_transaction)publish_rows(position,position+rows,private_begin);
        running.token_count+=rows;
    }
    void materialize_pending(){
        if(!has_pending)return;
        *static_cast<std::uint32_t*>(contents(pending_input))=pending_token;running.uploaded_bytes+=sizeof(pending_token);
        forward(pending_input,0,1,materialized);++materialized;has_pending=false;
    }
    void project(){
        if(!materialized||has_pending)throw std::logic_error("Gemma3 head has no materialized frontier");
        if(in_transaction&&materialized>transaction_begin)
            linear(private_norm,embedding,logits,1,hidden,vocabulary,
                std::size_t(materialized-1-transaction_begin)*hidden*sizeof(float));
        else linear(durable_norm,embedding,logits,1,hidden,vocabulary);
    }
    void sample(std::uint32_t depth,const TokenSamplerOptions& options,std::uint32_t history_count){
        if(options.repetition_penalty!=1.0F){
            pipeline(history_clear);bind(encoder,history_bits,0);dispatch((vocabulary/32+255)/256);
            pipeline(history_mark);bind(encoder,native_history,0);bind(encoder,history_bits,1);
            constant(history_count,2);if(history_count)dispatch((history_count+255)/256);
        }
        pipeline(process);bind(encoder,logits,0);bind(encoder,mapping,1);bind(encoder,history_bits,2);
        bind(encoder,processed_logits,3);bind(encoder,status,4);
        constant(std::array<float,2>{options.temperature,options.repetition_penalty},5);
        bind(encoder,active,6);dispatch(tiles);
        const float temperature=options.temperature==0?0.0F:1.0F;
        if(temperature>0&&(options.top_k!=0||options.top_p<1||options.min_p>0)){
            pipeline(sort_tile);bind(encoder,processed_logits,0);bind(encoder,sort_a,1);dispatch(vocabulary/512,512);
            bool in_a=true;
            for(std::uint32_t width=512;width<vocabulary;width*=2){
                [encoder setComputePipelineState:shared->draft_topk_merge_pipeline];
                bind(encoder,in_a?sort_a:sort_b,0);bind(encoder,in_a?sort_b:sort_a,1);
                constant(std::array<std::uint32_t,4>{1,vocabulary,vocabulary,width},2);
                dispatch(tiles);in_a=!in_a;
            }
            const auto& sorted=in_a?sort_a:sort_b;const auto top_k=static_cast<std::uint32_t>(options.top_k);
            pipeline(filter_prefix);bind(encoder,sorted,0);bind(encoder,mapping,1);bind(encoder,local_prefix,2);
            bind(encoder,tile_totals,3);constant(top_k,4);dispatch(tiles);
            pipeline(offsets);bind(encoder,tile_totals,0);bind(encoder,tile_offsets,1);bind(encoder,status,2);
            constant(temperature,3);dispatch(1,1);
            pipeline(filter_apply);bind(encoder,sorted,0);bind(encoder,mapping,1);bind(encoder,local_prefix,2);
            bind(encoder,tile_offsets,3);bind(encoder,processed_logits,4);constant(top_k,5);
            constant(std::array<float,2>{options.top_p,options.min_p},6);bind(encoder,active,7);dispatch(tiles);
        }
        pipeline(maxima);bind(encoder,processed_logits,0);bind(encoder,active,1);
        bind(encoder,tile_max,2);bind(encoder,tile_argmax,3);bind(encoder,status,4);dispatch(tiles);
        pipeline(max_finish);bind(encoder,tile_max,0);bind(encoder,tile_argmax,1);bind(encoder,status,2);dispatch(1,1);
        pipeline(prefix);bind(encoder,processed_logits,0);bind(encoder,active,1);bind(encoder,tile_max,2);
        bind(encoder,local_prefix,3);bind(encoder,tile_totals,4);constant(temperature,5);dispatch(tiles);
        pipeline(offsets);bind(encoder,tile_totals,0);bind(encoder,tile_offsets,1);bind(encoder,status,2);
        constant(temperature,3);dispatch(1,1);
        const auto offset=std::size_t(depth)*(vocabulary+1)*sizeof(std::uint64_t);
        pipeline(cdf);bind(encoder,local_prefix,0);bind(encoder,tile_offsets,1);bind(encoder,tile_argmax,2);
        bind(encoder,cdfs,3,offset);constant(temperature,4);bind(encoder,status,5);dispatch((vocabulary+256)/256);
        pipeline(select);bind(encoder,cdfs,0,offset);bind(encoder,mapping,1);bind(encoder,words,2);
        bind(encoder,draft_tokens,3);bind(encoder,target_tokens,4);bind(encoder,status,5);
        constant(depth,6);bind(encoder,active,7);dispatch(tiles);
    }
};

Gemma3DraftPlan::Gemma3DraftPlan(const StandaloneDraftArtifact& artifact,
    std::span<const std::int32_t> map,std::uint32_t target_vocabulary,
    std::uint32_t max_horizon,std::uint32_t transaction_rows):impl_(nullptr){
    if(artifact.kind()!=StandaloneDraftKind::gemma3_270m_q8)
        throw std::invalid_argument("Gemma3 draft requires its authenticated Q8 artifact");
    impl_=std::make_unique<Impl>(TensorSource{&artifact,
        [](const void* context,std::string_view name){return static_cast<const StandaloneDraftArtifact*>(context)->tensor(name);}},
        map,target_vocabulary,max_horizon,transaction_rows);
}
Gemma3DraftPlan::~Gemma3DraftPlan()=default;
Gemma3DraftPlan::Gemma3DraftPlan(Gemma3DraftPlan&&)noexcept=default;
Gemma3DraftPlan& Gemma3DraftPlan::operator=(Gemma3DraftPlan&&)noexcept=default;
std::uint32_t Gemma3DraftPlan::position()const noexcept{return impl_->frontier;}
std::uint32_t Gemma3DraftPlan::materialized_position()const noexcept{return impl_->materialized;}
std::uint32_t Gemma3DraftPlan::horizon_capacity()const noexcept{return impl_->horizon;}
std::size_t Gemma3DraftPlan::owned_bytes()const noexcept{return sizeof(Impl)+impl_->bytes;}
bool Gemma3DraftPlan::poisoned()const noexcept{return impl_->poison;}
bool Gemma3DraftPlan::transaction_active()const noexcept{return impl_->in_transaction;}
void Gemma3DraftPlan::reset()noexcept{
    auto& p=*impl_;p.invalidate();p.frontier=p.materialized=0;p.has_pending=false;p.in_transaction=false;
}
StandaloneDraftExecutionStats Gemma3DraftPlan::begin_transaction(){
    auto& p=*impl_;p.require_live();if(p.in_transaction)throw std::logic_error("Gemma3 transaction is already active");
    p.invalidate();p.in_transaction=true;p.transaction_begin=p.materialized;
    p.saved_frontier=p.frontier;p.saved_materialized=p.materialized;
    p.saved_pending=p.has_pending;p.saved_pending_token=p.pending_token;return {};
}
bool Gemma3DraftPlan::can_truncate(std::uint32_t position)const noexcept{
    const auto& p=*impl_;return !p.poison&&position<=p.frontier
        &&(position==p.frontier||(p.in_transaction&&position>=p.saved_frontier));
}
void Gemma3DraftPlan::truncate(std::uint32_t position){
    auto& p=*impl_;p.require_live();if(!can_truncate(position))
        throw std::invalid_argument("Gemma3 cannot logically truncate published sliding K/V");
    if(position!=p.frontier){p.invalidate();p.frontier=position;p.materialized=std::min(p.materialized,position);p.has_pending=false;}
}
void Gemma3DraftPlan::abort_transaction()noexcept{
    auto& p=*impl_;p.invalidate();if(!p.in_transaction){p.poison=true;return;}
    p.frontier=p.saved_frontier;p.materialized=p.saved_materialized;
    p.has_pending=p.saved_pending;p.pending_token=p.saved_pending_token;p.in_transaction=false;
}
StandaloneDraftExecutionStats Gemma3DraftPlan::commit_prefix(std::uint32_t position){
    auto& p=*impl_;p.require_live();
    if(!p.in_transaction||position<p.saved_frontier||position>p.frontier)
        throw std::invalid_argument("Gemma3 commit is outside its transaction");
    const auto materialized=std::min(position,p.materialized);
    const bool pending=p.has_pending&&position==p.frontier;
    p.invalidate();StandaloneDraftExecutionStats stats;
    if(materialized>p.transaction_begin){
        @autoreleasepool {p.start();p.publish_rows(p.transaction_begin,materialized,p.transaction_begin);stats=p.finish();}
    }
    p.frontier=position;p.materialized=materialized;p.has_pending=pending;p.in_transaction=false;return stats;
}
StandaloneDraftExecutionStats Gemma3DraftPlan::append(std::span<const std::uint32_t> tokens){
    auto& p=*impl_;p.require_live();
    if(tokens.size()>capacity-p.frontier
        ||(p.in_transaction&&tokens.size()>p.transaction_capacity-(p.frontier-p.transaction_begin)))
        throw std::length_error("Gemma3 draft append exceeds its prepared capacity");
    for(auto token:tokens)if(token>=vocabulary)throw std::invalid_argument("Gemma3 draft token is outside vocabulary");
    if(tokens.empty())return {};
    p.invalidate();StandaloneDraftExecutionStats result;const auto started=now_ns();
    @autoreleasepool {
        for(std::size_t offset=0;offset<tokens.size();offset+=chunk){
            const auto rows=static_cast<std::uint32_t>(std::min<std::size_t>(chunk,tokens.size()-offset));
            p.start();std::memcpy(contents(p.ids),tokens.data()+offset,rows*sizeof(std::uint32_t));
            p.running.uploaded_bytes+=rows*sizeof(std::uint32_t);p.materialize_pending();
            p.forward(p.ids,0,rows,p.materialized);p.materialized+=rows;p.frontier+=rows;
            combine(result,p.finish());
        }
    }
    result.wall_duration_ns=now_ns()-started;return result;
}
Gemma3DraftProposal Gemma3DraftPlan::propose(std::uint32_t root,std::span<const std::uint32_t> words,
    const TokenSamplerOptions& options,const SamplerHistorySnapshot& history){
    auto& p=*impl_;p.require_live();validate_options(options);
    if(!p.in_transaction||words.empty()||words.size()>p.horizon||p.frontier==0
        ||words.size()>capacity-p.frontier
        ||words.size()>p.transaction_capacity-(p.frontier-p.transaction_begin)||root>=p.target_vocab)
        throw std::invalid_argument("Gemma3 proposal has incompatible transaction, horizon or capacity");
    if(history.head>=token_sampler_max_repetition_window||history.length>history.window_limit
        ||history.window_limit>token_sampler_max_repetition_window
        ||(history.length&&history.head>=history.window_limit)||(!history.length&&history.head))
        throw std::invalid_argument("Gemma3 proposal sampler history is malformed");
    if(p.epoch==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Gemma3 proposal epoch overflow");
    p.invalidate();p.probability_owner.begin_epoch();
    const auto count=static_cast<std::uint32_t>(words.size()),start_position=p.frontier;
    @autoreleasepool {
        p.start();std::memcpy(contents(p.words),words.data(),words.size_bytes());
        auto* target=static_cast<std::uint32_t*>(contents(p.target_tokens));target[0]=root;
        std::fill_n(static_cast<std::uint32_t*>(contents(p.draft_tokens)),count,vocabulary);
        std::fill_n(target+1,count,p.target_vocab);
        VerifierDeviceSelectionRow row;row.history_token_ids=history.token_ids;row.history_head=history.head;
        row.history_length=history.length;row.history_window_limit=history.window_limit;
        static_assert(sizeof(row)==91*sizeof(std::uint32_t));std::memcpy(contents(p.base_history),&row,sizeof(row));
        p.running.uploaded_bytes+=words.size_bytes()+sizeof(root)+sizeof(row)+2*count*sizeof(std::uint32_t);
        p.materialize_pending();
        for(std::uint32_t i=0;i<count;++i){
            p.project();p.sample(i,options,p.frontier);
            if(i+1<count){p.forward(p.draft_tokens,i,1,p.frontier);++p.materialized;}
            else {
                p.pipeline(pending);bind(p.encoder,p.draft_tokens,0);bind(p.encoder,p.native_history,1);
                p.constant(std::array<std::uint32_t,2>{i,p.frontier},2);p.dispatch(1,1);
            }
            ++p.frontier;
        }
        p.pipeline(requests);bind(p.encoder,p.base_history,0);bind(p.encoder,p.target_tokens,1);
        bind(p.encoder,p.device_requests,2);p.constant(count,3);p.dispatch(1,32);
        auto stats=p.finish();
        const auto* draft=static_cast<const std::uint32_t*>(contents(p.draft_tokens));
        for(std::uint32_t i=0;i<count;++i)if(target[i+1]>=p.target_vocab||draft[i]>=vocabulary){
            p.poison=true;p.invalidate();throw std::runtime_error("Gemma3 draft failed to publish a selected token");
        }
        p.has_pending=true;p.pending_token=draft[count-1];p.published_epoch=++p.epoch;
        p.active_horizon=count;p.probability_bytes=0;stats.readback_bytes+=count*2*sizeof(std::uint32_t);
        stats.wall_duration_ns=now_ns()-p.wall_start;
        return {.draft_tokens={draft,count},.target_tokens={target+1,count},.start_position=start_position,.materialized_end=p.materialized,
            .epoch=p.published_epoch,.execution=stats};
    }
}
StandaloneDraftExecutionStats Gemma3DraftPlan::restore_pending_token(std::uint32_t token){
    auto& p=*impl_;p.require_live();const auto started=now_ns();
    if(p.in_transaction||p.has_pending||p.frontier!=p.materialized||p.frontier==capacity||token>=vocabulary)
        throw std::invalid_argument("Gemma3 saved pending token is incompatible with its restored prefix");
    p.invalidate();static_cast<std::uint32_t*>(contents(p.native_history))[p.frontier]=token;
    p.pending_token=token;p.has_pending=true;++p.frontier;
    return {.wall_duration_ns=now_ns()-started,.uploaded_bytes=sizeof(token)};
}
VerifierDeviceInputBinding Gemma3DraftPlan::verifier_inputs(std::uint64_t epoch)const{
    const auto& p=*impl_;p.require_live();
    if(!epoch||epoch!=p.published_epoch)throw std::logic_error("Gemma3 proposal device lease is stale");
    return {.token_ids=p.target_tokens,.raw_selection_requests=p.device_requests,.arena_epoch=epoch};
}
GatheredDraftTokenDistribution Gemma3DraftPlan::gather(std::uint32_t depth,const PreparedTokenDistribution& target){
    auto& p=*impl_;p.require_live();
    if(!p.published_epoch||depth>=p.active_horizon||!target.count||target.count>token_sampler_max_top_k)
        throw std::invalid_argument("Gemma3 proposal probability lease is stale or malformed");
    GatheredDraftTokenDistribution result;result.lease=p.probability_owner.lease(depth);result.count=target.count;
    const auto* c=static_cast<const std::uint64_t*>(contents(p.cdfs))+std::size_t(depth)*(vocabulary+1);
    const auto selected=static_cast<const std::uint32_t*>(contents(p.draft_tokens))[depth];
    result.proposal.token=static_cast<const std::uint32_t*>(contents(p.target_tokens))[depth+1];
    result.proposal.probability_mass=c[selected+1]-c[selected];result.proposal_cdf_begin=c[selected];
    p.probability_bytes+=2*sizeof(std::uint32_t)+2*sizeof(std::uint64_t);
    for(std::uint32_t i=0;i<target.count;++i){
        const auto token=target.values[i].token;if(token>=p.target_vocab)
            throw std::invalid_argument("Gemma3 probability gather target token is out of bounds");
        result.target_values[i]={token,0};const auto id=p.reverse_map[token];
        if(id>=0){result.target_values[i].probability_mass=c[std::uint32_t(id)+1]-c[std::uint32_t(id)];
            p.probability_bytes+=2*sizeof(std::uint64_t);}
    }
    return result;
}
std::uint64_t Gemma3DraftPlan::probability_readback_bytes()const noexcept{return impl_->probability_bytes;}
StandaloneDraftExecutionStats Gemma3DraftPlan::read_logits(std::span<float> output){
    auto& p=*impl_;p.require_live();
    if(output.size()!=vocabulary||p.frontier==0)throw std::invalid_argument("Gemma3 logit diagnostic shape or frontier is invalid");
    p.invalidate();
    @autoreleasepool {p.start();p.materialize_pending();p.project();auto stats=p.finish();
        std::memcpy(output.data(),contents(p.logits),output.size_bytes());stats.readback_bytes+=output.size_bytes();return stats;}
}
void Gemma3DraftPlan::read_cache(std::uint32_t layer,std::uint32_t first,
    std::span<float> keys,std::span<float> values)const{
    const auto& p=*impl_;p.require_live();
    if(layer>=layer_count||keys.size()!=values.size()||keys.size()%256||first>p.materialized
        ||keys.size()/256>p.materialized-first
        ||(local_layer(layer)&&p.materialized>512&&first<p.materialized-512))
        throw std::invalid_argument("Gemma3 cache diagnostic is outside the retained logical range");
    const auto& l=p.layers[layer];
    for(std::size_t row=0;row<keys.size()/256;++row){const auto position=first+static_cast<std::uint32_t>(row);
        const bool priv=p.in_transaction&&position>=p.transaction_begin;
        const auto slot=priv?position-p.transaction_begin:(local_layer(layer)?position&511:position);
        const auto* k=static_cast<const float*>(contents(priv?l.private_key:l.key));
        const auto* v=static_cast<const float*>(contents(priv?l.private_value:l.value));
        std::copy_n(k+std::size_t(slot)*256,256,keys.data()+row*256);
        std::copy_n(v+std::size_t(slot)*256,256,values.data()+row*256);
    }
}
void Gemma3DraftPlan::warm_up(){
    auto& p=*impl_;p.require_live();
    if(p.frontier||p.in_transaction)throw std::logic_error("Gemma3 warmup requires an empty prepared plan");
    std::array<std::uint32_t,chunk> tokens{};tokens.fill(2);
    (void)append(tokens);(void)begin_transaction();(void)append(std::span(tokens).first(1));
    std::array<std::uint32_t,external_speculation_max_horizon> words{};
    TokenSamplerOptions options;options.temperature=0.8F;options.repetition_penalty=1.1F;
    SamplerHistorySnapshot history;history.window_limit=64;
    const auto batch=propose(0,std::span(words).first(p.horizon),options,history);
    for(std::uint32_t row=0;row<p.horizon;++row){
        PreparedTokenDistribution target;target.count=1;target.values[0]={batch.target_tokens[row],1,1};
        (void)gather(row,target);
    }
    (void)commit_prefix(position());(void)append(std::span(tokens).first(1));reset();
}

#if defined(LOCAL_GGUF_ENGINE_TESTING)
Gemma3DraftPlan::Gemma3DraftPlan(std::span<const tensor::TensorView> tensors,std::span<const std::int32_t> map,
    std::uint32_t target_vocabulary,std::uint32_t max_horizon,std::uint32_t transaction_rows):impl_(nullptr){
    if(tensors.size()!=2+13*layer_count)throw std::invalid_argument("Gemma3 fixture must contain the entire production model");
    impl_=std::make_unique<Impl>(TensorSource{&tensors,[](const void* context,std::string_view name){
        const auto views=*static_cast<const std::span<const tensor::TensorView>*>(context);
        if(name=="token_embd.weight")return views[0];if(name=="output_norm.weight")return views[1];
        for(std::uint32_t layer=0;layer<layer_count;++layer)for(std::uint32_t w=0;w<13;++w)
            if(name=="blk."+std::to_string(layer)+"."+suffixes[w])return views[2+13*layer+w];
        throw std::invalid_argument("Gemma3 fixture tensor name is invalid");
    }},map,target_vocabulary,max_horizon,transaction_rows);
}
void Gemma3DraftPlan::sample_logits_for_testing(std::span<const float> logits,std::uint32_t word,
    const TokenSamplerOptions& options){
    auto& p=*impl_;p.require_live();validate_options(options);
    if(logits.size()!=vocabulary)throw std::invalid_argument("Gemma3 diagnostic logits are not full vocabulary");
    p.invalidate();p.probability_owner.begin_epoch();
    @autoreleasepool {p.start();std::memcpy(contents(p.logits),logits.data(),logits.size_bytes());
        *static_cast<std::uint32_t*>(contents(p.words))=word;
        p.sample(0,options,p.frontier);(void)p.finish();}
    p.published_epoch=++p.epoch;p.active_horizon=1;p.probability_bytes=0;
}
void Gemma3DraftPlan::seed_zero_cache_prefix_for_testing(std::uint32_t position){
    auto& p=*impl_;p.require_live();
    if(position>capacity||p.frontier||p.in_transaction)
        throw std::invalid_argument("Gemma3 zero-cache fixture requires empty bounded state");
    for(auto& layer:p.layers)for(const Buffer* buffer:{&layer.key,&layer.value})
        std::memset(contents(*buffer),0,buffer->value->descriptor().byte_count);
    std::memset(contents(p.durable_norm),0,hidden*sizeof(float));
    std::memset(contents(p.native_history),0,capacity*sizeof(std::uint32_t));
    p.invalidate();p.materialized=p.frontier=position;
}
#endif
} // namespace local_gguf_engine::engine
