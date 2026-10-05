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
constexpr std::uint32_t hidden_width = 640;
constexpr std::uint32_t intermediate_width = 2048;
constexpr std::uint32_t draft_vocabulary_size = 262144;
constexpr std::uint32_t maximum_context_rows = 32768;
constexpr std::uint32_t prefill_chunk_rows = 64;
constexpr std::uint32_t vocabulary_tile_count = draft_vocabulary_size / 256;
constexpr bool is_local_layer(std::uint32_t layer) {
    return layer % 6 != 5;
}

// The native GGUF has already folded Gemma's (1+w) into its F32 norm vectors.
// Q8 is never expanded into a second resident full-vocabulary matrix.
constexpr const char* draft_kernel_source = R"metal(
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
 constant GDAttention& p [[buffer(7)]],device float4* output [[buffer(8)]],
 device atomic_uint* status [[buffer(9)]],uint3 group [[threadgroup_position_in_grid]],
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
 if(p.max_parts==1){
   // Preserve the one-partition reducer's F32 arithmetic without round-tripping
   // its entire partial output through another kernel and command dependency.
   float add=exp(maximum-maximum);acc0=(float4(0)*0.0f+acc0*add)/(0.0f+denom*add);
   acc1=(float4(0)*0.0f+acc1*add)/(0.0f+denom*add);
   output[(row*4+head)*64+lane*2]=acc0;output[(row*4+head)*64+lane*2+1]=acc1;
   if(!all(isfinite(acc0))||!all(isfinite(acc1)))atomic_store_explicit(status,5u,memory_order_relaxed);
   return;
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
 constant uint2& range [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<range.y){uint token=history[range.x+i];atomic_fetch_or_explicit(bits+token/32,1u<<(token%32),memory_order_relaxed);}
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
// Merge only the best 64 entries from each run. The initial 512-token
// tiles remain intact for the rare large-support fallback.
kernel void gd_bounded_merge(device const GDPair* input [[buffer(0)]],device GDPair* output [[buffer(1)]],
 constant uint2& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.x*64)return;uint group=i/64,rank=i%64;
 device const GDPair* a=input+group*2*p.y;device const GDPair* b=a+p.y;
 int low=0,high=int(rank);
 while(low<=high){int probe=(low+high)/2;uint ai=uint(probe),bi=rank-ai;
   if(ai>0&&bi<64&&gd_better(b[bi],a[ai-1]))high=probe-1;
   else if(bi>0&&ai<64&&gd_better(a[ai],b[bi-1]))low=probe+1;
   else{low=probe;break;}}
 uint ai=uint(low),bi=rank-ai;
 output[i]=ai>=64?b[bi]:bi>=64?a[ai]:gd_better(b[bi],a[ai])?b[bi]:a[ai];
}
kernel void gd_compact_collect(device const float* scores [[buffer(0)]],device const int* map [[buffer(1)]],
 device const GDPair* best [[buffer(2)]],device GDPair* support [[buffer(3)]],
 device atomic_uint* count [[buffer(4)]],constant uint& top_k [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>=262144||map[i]<0||scores[i]<best[top_k-1].score)return;
 uint index=atomic_fetch_add_explicit(count,1u,memory_order_relaxed);
 if(index<256)support[index]=GDPair{i,scores[i]};
}
// The compact law uses fixed token-ID order and F32 positive accumulation.
// Its uint64 bin widths are the probability law, not an idealized softmax.
// Support overflow is decided from logits before inspecting any random word.
kernel void gd_compact_sample(device const GDPair* support [[buffer(0)]],device const uint* count [[buffer(1)]],
 device uint* ids [[buffer(2)]],device ulong* cdf [[buffer(3)]],device const int* map [[buffer(4)]],
 device const uint* words [[buffer(5)]],device uint* draft [[buffer(6)]],device uint* target [[buffer(7)]],
 device atomic_uint* status [[buffer(8)]],constant uint& depth [[buffer(9)]],
 constant float2& filters [[buffer(10)]],uint t [[thread_index_in_threadgroup]]) {
 uint n=*count;if(n>256)return;
 if(n==0){if(t==0)atomic_store_explicit(status,12u,memory_order_relaxed);return;}
 threadgroup GDPair values[256];threadgroup float weights[256];
 values[t]=t<n?support[t]:GDPair{uint(-1),-INFINITY};threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint width=2;width<=256;width<<=1)for(uint stride=width>>1;stride;stride>>=1){uint other=t^stride;
   if(other>t){bool ascending=(t&width)!=0;
     if((!ascending&&gd_better(values[other],values[t]))||(ascending&&gd_better(values[t],values[other])))
       {GDPair swap=values[t];values[t]=values[other];values[other]=swap;}}
   threadgroup_barrier(mem_flags::mem_threadgroup);}
 weights[t]=t<n?exp(values[t].score-values[0].score):0.0f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(t==0){float total=0,tail=0;for(uint j=n;j>0;--j)total+=weights[j-1];
   for(uint j=n;j>0;--j){uint i=j-1;tail+=weights[i];
     bool keep=i==0||((filters.x>=1.0f||tail/total>1.0f-filters.x)
       &&(filters.y<=0.0f||weights[i]>=filters.y));
     if(!keep)values[i].score=-INFINITY;}}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 // Sorting IDs after filtering makes receipt lookup deterministic even though
 // support compaction itself is parallel and may arrive in any order.
 for(uint width=2;width<=256;width<<=1)for(uint stride=width>>1;stride;stride>>=1){uint other=t^stride;
   if(other>t){bool ascending=(t&width)==0;
     if((ascending&&values[other].token<values[t].token)||(!ascending&&values[t].token<values[other].token))
       {GDPair swap=values[t];values[t]=values[other];values[other]=swap;}}
   threadgroup_barrier(mem_flags::mem_threadgroup);}
 // Recompute using the original maximum to avoid a reduction dependent on
 // support compaction or filtering. The best candidate always survives.
 if(t==0){float maximum=-INFINITY;for(uint i=0;i<n;++i)maximum=max(maximum,values[i].score);
   float total=0;for(uint i=0;i<n;++i){weights[i]=exp(values[i].score-maximum);total+=weights[i];}
   if(!(total>0)||!isfinite(total)){atomic_store_explicit(status,13u,memory_order_relaxed);return;}
   float prefix=0;ulong last=0;cdf[0]=0;
   for(uint i=0;i<n;++i){ids[i]=values[i].token;prefix+=weights[i];
     ulong next=ulong(clamp((prefix/total)*4294967296.0f,0.0f,4294967296.0f));
     // Last positive prefix equals total; zero-weight tails inherit its exact
     // endpoint without acquiring a forced probability mass.
     cdf[i+1]=next;
     if(next<last||next>4294967296ul||(next>last&&!isfinite(values[i].score)))
       atomic_store_explicit(status,14u,memory_order_relaxed);
     if(ulong(words[depth])>=last&&ulong(words[depth])<next){draft[depth]=ids[i];target[depth+1]=uint(map[ids[i]]);}
     last=next;
   }
   if(last!=4294967296ul)atomic_store_explicit(status,15u,memory_order_relaxed);
 }
}
kernel void gd_full_merge(device const GDPair* input [[buffer(0)]],device GDPair* output [[buffer(1)]],
 constant uint& width [[buffer(2)]],device const uint* compact_count [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(*compact_count<=256||i>=262144)return;
 uint group=i/(width*2),rank=i%(width*2);device const GDPair* a=input+group*width*2;
 device const GDPair* b=a+width;int low=int(rank>width?rank-width:0),high=int(min(rank,width));
 while(low<=high){int probe=(low+high)/2;uint ai=uint(probe),bi=rank-ai;
   if(ai>0&&bi<width&&gd_better(b[bi],a[ai-1]))high=probe-1;
   else if(bi>0&&ai<width&&gd_better(a[ai],b[bi-1]))low=probe+1;
   else{low=probe;break;}}
 uint ai=uint(low),bi=rank-ai;
 output[i]=ai>=width?b[bi]:bi>=width?a[ai]:gd_better(b[bi],a[ai])?b[bi]:a[ai];
}
// Ascending tail prefix for HF top-p; all cutoff ties survive top-k.
kernel void gd_filter_prefix(device const GDPair* sorted [[buffer(0)]],device const int* map [[buffer(1)]],
 device float* prefix [[buffer(2)]],device float* totals [[buffer(3)]],constant uint& top_k [[buffer(4)]],
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],device const uint* compact_count [[buffer(5)]]) {
 if(*compact_count<=256)return;
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
 uint i [[thread_position_in_grid]],device const uint* compact_count [[buffer(8)]]) {
 if(*compact_count<=256)return;
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
 uint t [[thread_index_in_threadgroup]],device const uint* compact_count [[buffer(5)]]) {
 if(*compact_count<=256)return;
 threadgroup float m[256];threadgroup uint ids[256];uint i=group*256+t;
 float v=logits[i];if(eligible[i]&&!isfinite(v))atomic_store_explicit(status,5u,memory_order_relaxed);
 m[t]=eligible[i]?v:-INFINITY;ids[t]=i;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s;s>>=1){if(t<s && (m[t+s]>m[t]||(m[t+s]==m[t]&&ids[t+s]<ids[t])))
   {m[t]=m[t+s];ids[t]=ids[t+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(t==0){maxima[group]=m[0];argmax[group]=ids[0];}
}
kernel void gd_max_finish(device float* maxima [[buffer(0)]],device uint* argmax [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]],device const uint* compact_count [[buffer(3)]]) {
 if(*compact_count<=256)return;
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
 uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],device const uint* compact_count [[buffer(6)]]) {
 if(*compact_count<=256)return;
 threadgroup float w[256];uint i=group*256+t;
 w[t]=(eligible[i] && temperature>0)?exp((logits[i]-maxima[1024])/temperature):0.0f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(t==0){float s=0;for(uint j=0;j<256;++j){s+=w[j];prefix[group*256+j]=s;}totals[group]=s;}
}
kernel void gd_offsets(device const float* totals [[buffer(0)]],device float* offsets [[buffer(1)]],
 device atomic_uint* status [[buffer(2)]],constant float& temperature [[buffer(3)]],device const uint* compact_count [[buffer(4)]]) {
 if(*compact_count<=256)return;
 float s=0;offsets[0]=0;for(uint i=0;i<1024;++i){s+=totals[i];offsets[i+1]=s;}
 if(temperature>0 && (!(s>0)||!isfinite(s)))atomic_store_explicit(status,7u,memory_order_relaxed);
}
kernel void gd_cdf(device const float* prefix [[buffer(0)]],device const float* offsets [[buffer(1)]],
 device const uint* argmax [[buffer(2)]],device ulong* cdf [[buffer(3)]],
 constant float& temperature [[buffer(4)]],device atomic_uint* status [[buffer(5)]],uint i [[thread_position_in_grid]],device const uint* compact_count [[buffer(6)]]) {
 if(*compact_count<=256)return;
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
 constant uint& depth [[buffer(6)]],device const uint* eligible [[buffer(7)]],uint i [[thread_position_in_grid]],device const uint* compact_count [[buffer(8)]]) {
 if(*compact_count<=256)return;
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
enum Pipeline : std::size_t {
    q8,
    embed,
    pending,
    norm,
    rope,
    attention,
    attention_reduce,
    gelu,
    add,
    publish,
    copy,
    maxima,
    max_finish,
    prefix,
    offsets,
    cdf,
    select,
    requests,
    history_clear,
    history_mark,
    process,
    sort_tile,
    filter_prefix,
    filter_apply,
    bounded_merge,
    compact_collect,
    compact_sample,
    full_merge,
    pipeline_count
};
constexpr std::array<const char*, pipeline_count> pipeline_names = {"gd_q8", "gd_embed",
    "gd_pending", "gd_norm", "gd_rope_store", "gd_attention", "gd_attention_reduce", "gd_gelu",
    "gd_add", "gd_publish", "gd_copy", "gd_max", "gd_max_finish", "gd_prefix", "gd_offsets",
    "gd_cdf", "gd_select", "gd_requests", "gd_history_clear", "gd_history_mark", "gd_process",
    "gd_sort_tile", "gd_filter_prefix", "gd_filter_apply", "gd_bounded_merge", "gd_compact_collect",
    "gd_compact_sample", "gd_full_merge"};
std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

const MetalBackendBufferStorage& storage(const BackendBuffer& buffer) {
    return require_metal_backend_buffer(tensor::backend_buffer_storage(buffer));
}

void* contents(const BackendBuffer& buffer) {
    return metal_buffer_contents(storage(buffer));
}

void bind(id<MTLComputeCommandEncoder> encoder, const BackendBuffer& buffer, NSUInteger index,
    std::size_t offset = 0) {
    const auto& metal_storage = storage(buffer);
    [encoder setBuffer:metal_storage.buffer()
                offset:metal_storage.base_byte_offset() + offset
               atIndex:index];
}

void combine(StandaloneDraftExecutionStats& total, const StandaloneDraftExecutionStats& increment) {
#define ADD(field)                                                                                 \
    if (increment.field > std::numeric_limits<decltype(total.field)>::max() - total.field)         \
        throw std::overflow_error("Gemma3 draft statistics overflow");                             \
    total.field += increment.field
    ADD(gpu_duration_ns);
    ADD(wall_duration_ns);
    ADD(readback_bytes);
    ADD(uploaded_bytes);
    ADD(command_buffer_count);
    ADD(compute_dispatch_count);
    ADD(memory_barrier_count);
    ADD(token_count);
#undef ADD
}

struct Buffer {
    std::optional<BackendBuffer> value;
    Buffer& operator=(BackendBuffer buffer) {
        value = std::move(buffer);
        return *this;
    }
    operator const BackendBuffer&() const {
        return value.value();
    }
};
struct TensorSource {
    const void* source_context;
    tensor::TensorView (*get)(const void*, std::string_view);
};
constexpr std::array<const char*, 13> weight_suffixes = {"attn_q.weight", "attn_k.weight",
    "attn_v.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight",
    "attn_q_norm.weight", "attn_k_norm.weight"};
void validate_options(const TokenSamplerOptions& options) {
    if (!std::isfinite(options.temperature) || options.temperature < 0 ||
        options.top_k > draft_vocabulary_size || !std::isfinite(options.top_p) ||
        options.top_p <= 0 || options.top_p > 1 || !std::isfinite(options.min_p) ||
        options.min_p < 0 || options.min_p > 1 || !std::isfinite(options.repetition_penalty) ||
        options.repetition_penalty <= 0)
        throw std::invalid_argument("Gemma3 draft sampler options are invalid");
}
} // namespace

struct Gemma3DraftPlan::Impl {
    struct Layer {
        std::array<Buffer, 13> weights;
        Buffer key;
        Buffer value;
        Buffer private_key;
        Buffer private_value;
    };
    std::array<Layer, layer_count> layers;
    Buffer embedding;
    Buffer final_norm;
    Buffer ids;
    Buffer proposal_input;
    Buffer pending_input;
    Buffer hidden_states;
    Buffer normalized_hidden;
    Buffer raw_queries;
    Buffer raw_keys;
    Buffer raw_values;
    Buffer normalized_queries;
    Buffer normalized_keys;
    Buffer rotated_queries;
    Buffer attention_output;
    Buffer projected_output;
    Buffer residual_branch;
    Buffer gate_states;
    Buffer up_states;
    Buffer partial;
    Buffer scales;
    Buffer durable_norm;
    Buffer private_norm;
    Buffer logits;
    Buffer status;
    Buffer mapping;
    Buffer words;
    Buffer draft_tokens;
    Buffer target_tokens;
    Buffer tile_max;
    Buffer tile_argmax;
    Buffer local_prefix;
    Buffer tile_totals;
    Buffer tile_offsets;
    Buffer cdfs;
    Buffer base_history;
    Buffer device_requests;
    Buffer native_history;
    Buffer history_bits;
    Buffer processed_logits;
    Buffer active;
    Buffer sort_a;
    Buffer sort_b;
    Buffer bounded_a;
    Buffer bounded_b;
    Buffer compact_support;
    Buffer compact_counts;
    Buffer compact_ids;
    Buffer compact_cdfs;
    std::array<id<MTLComputePipelineState>, pipeline_count> pipelines{};
    MatmulPipelineState* shared = nullptr;
    id<MTLCommandBuffer> command = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    std::vector<std::int32_t> reverse_map;
    DraftProbabilityLeaseOwner probability_owner;
    // Logical history includes a final selected token that may still be pending.
    // Materialized history has transformer K/V and normalized head state.
    std::uint32_t logical_position = 0;
    std::uint32_t materialized_position = 0;
    std::uint32_t horizon_capacity = 0;
    std::uint32_t target_vocabulary_size = 0;
    std::uint32_t transaction_capacity = 0;
    std::uint32_t context_rows = 0;
    std::uint32_t proposal_length = 0;
    std::uint32_t transaction_begin = 0;
    std::uint32_t saved_logical_position = 0;
    std::uint32_t saved_materialized_position = 0;
    std::uint32_t saved_pending_token = 0;
    std::uint32_t pending_token = 0;
    std::uint64_t epoch = 0;
    std::uint64_t published_epoch = 0;
    std::uint64_t probability_readback_bytes = 0;
    std::uint64_t command_started_ns = 0;
    std::size_t resident_bytes = 0;
    bool is_poisoned = false;
    bool in_transaction = false;
    bool has_pending = false;
    bool saved_has_pending = false;
    StandaloneDraftExecutionStats running;

    template <class Visit>
    static void scratch_layout(std::uint32_t horizon_capacity, std::uint32_t transaction_rows,
        std::uint32_t context_rows, Visit&& visit) {
        const auto add = [&](Buffer Impl::* member, std::initializer_list<std::uint64_t> shape) {
            visit(member, shape);
        };
        add(&Impl::ids, {prefill_chunk_rows});
        add(&Impl::proposal_input, {transaction_rows});
        add(&Impl::pending_input, {1});
        add(&Impl::hidden_states, {prefill_chunk_rows, hidden_width});
        add(&Impl::normalized_hidden, {prefill_chunk_rows, hidden_width});
        add(&Impl::raw_queries, {prefill_chunk_rows, 1024});
        add(&Impl::raw_keys, {prefill_chunk_rows, 256});
        add(&Impl::raw_values, {prefill_chunk_rows, 256});
        add(&Impl::normalized_queries, {prefill_chunk_rows, 1024});
        add(&Impl::normalized_keys, {prefill_chunk_rows, 256});
        add(&Impl::rotated_queries, {prefill_chunk_rows, 1024});
        add(&Impl::attention_output, {prefill_chunk_rows, 1024});
        add(&Impl::projected_output, {prefill_chunk_rows, hidden_width});
        add(&Impl::residual_branch, {prefill_chunk_rows, hidden_width});
        add(&Impl::gate_states, {prefill_chunk_rows, intermediate_width});
        add(&Impl::up_states, {prefill_chunk_rows, intermediate_width});
        add(&Impl::partial, {prefill_chunk_rows, 4, 32, 256});
        add(&Impl::scales, {prefill_chunk_rows, 4, 32, 2});
        add(&Impl::durable_norm, {1, hidden_width});
        add(&Impl::private_norm, {std::max(prefill_chunk_rows, transaction_rows), hidden_width});
        add(&Impl::logits, {draft_vocabulary_size});
        add(&Impl::status, {1});
        add(&Impl::mapping, {draft_vocabulary_size});
        add(&Impl::words, {horizon_capacity});
        add(&Impl::draft_tokens, {horizon_capacity});
        add(&Impl::target_tokens, {horizon_capacity + 1});
        add(&Impl::tile_max, {vocabulary_tile_count + 1});
        add(&Impl::tile_argmax, {vocabulary_tile_count + 1});
        add(&Impl::local_prefix, {draft_vocabulary_size});
        add(&Impl::tile_totals, {vocabulary_tile_count});
        add(&Impl::tile_offsets, {vocabulary_tile_count + 1});
        add(&Impl::cdfs, {horizon_capacity, draft_vocabulary_size + 1, 2});
        add(&Impl::base_history, {91});
        add(&Impl::device_requests, {horizon_capacity + 1, 91});
        add(&Impl::native_history, {context_rows});
        add(&Impl::history_bits, {draft_vocabulary_size / 32});
        add(&Impl::processed_logits, {draft_vocabulary_size});
        add(&Impl::active, {draft_vocabulary_size});
        add(&Impl::sort_a, {draft_vocabulary_size, 2});
        add(&Impl::sort_b, {draft_vocabulary_size, 2});
        add(&Impl::bounded_a, {draft_vocabulary_size / 512 / 2, 64, 2});
        add(&Impl::bounded_b, {draft_vocabulary_size / 512 / 2, 64, 2});
        add(&Impl::compact_support, {256, 2});
        add(&Impl::compact_counts, {horizon_capacity});
        add(&Impl::compact_ids, {horizon_capacity, 256});
        add(&Impl::compact_cdfs, {horizon_capacity, 257, 2});
    }

    Impl(TensorSource source_tensors, std::span<const std::int32_t> draft_to_target,
        std::uint32_t target_vocabulary, std::uint32_t max_horizon, std::uint32_t transaction_rows,
        std::uint32_t context_rows)
        : horizon_capacity(max_horizon), target_vocabulary_size(target_vocabulary),
          transaction_capacity(transaction_rows), context_rows(context_rows) {
        if (draft_to_target.size() != draft_vocabulary_size || target_vocabulary_size == 0 ||
            horizon_capacity == 0 || horizon_capacity > external_speculation_max_horizon ||
            transaction_rows < horizon_capacity + 2 || transaction_rows > maximum_context_rows ||
            context_rows == 0 || context_rows > maximum_context_rows)
            throw std::invalid_argument(
                "Gemma3 draft geometry, transaction capacity or horizon is invalid");
        reverse_map.assign(target_vocabulary_size, -1);
        std::size_t mapped_count = 0;
        for (std::uint32_t index = 0; index < draft_vocabulary_size; ++index) {
            if (draft_to_target[index] < -1 ||
                (draft_to_target[index] >= 0 &&
                    static_cast<std::uint32_t>(draft_to_target[index]) >= target_vocabulary_size))
                throw std::invalid_argument("Gemma3 TLI map token is out of bounds");
            if (draft_to_target[index] < 0)
                continue;
            auto& inverse = reverse_map[std::size_t(draft_to_target[index])];
            if (inverse != -1)
                throw std::invalid_argument("Gemma3 TLI map is not injective");
            inverse = static_cast<std::int32_t>(index);
            ++mapped_count;
        }
        if (!mapped_count)
            throw std::invalid_argument("Gemma3 TLI intersection is empty");
        (void)metal_device_or_throw();
        shared = &shared_matmul_pipeline_state();
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        auto library =
            [shared->device newLibraryWithSource:[NSString stringWithUTF8String:draft_kernel_source]
                                         options:options
                                           error:&error];
        if (!library)
            throw std::runtime_error(
                "Gemma3 Metal library: " + ns_error_message(error, "compilation failed"));
        for (std::size_t index = 0; index < pipeline_count; ++index) {
            auto function =
                [library newFunctionWithName:[NSString stringWithUTF8String:pipeline_names[index]]];
            pipelines[index] = [shared->device newComputePipelineStateWithFunction:function
                                                                             error:&error];
            if (!pipelines[index] || pipelines[index].threadExecutionWidth != 32 ||
                pipelines[index].maxTotalThreadsPerThreadgroup < (index == sort_tile ? 512U : 256U))
                throw std::runtime_error(
                    "Gemma3 Metal pipeline: " + ns_error_message(error, "unsupported device"));
        }
        const auto load = [&](std::string_view name, std::uint32_t input, std::uint32_t output,
                              DType dtype) {
            auto view = source_tensors.get(source_tensors.source_context, name);
            const auto& descriptor = view.descriptor();
            if (descriptor.dtype != dtype ||
                (output == 0
                        ? (descriptor.shape.rank() != 1 || descriptor.shape.dimension(0) != input)
                        : (descriptor.shape.rank() != 2 || descriptor.shape.dimension(0) != input ||
                              descriptor.shape.dimension(1) != output)))
                throw std::invalid_argument(
                    "Gemma3 weight has incompatible shape or dtype: " + std::string(name));
            auto buffer = copy_tensor_view_to_metal_backend(view);
            resident_bytes += buffer.descriptor().byte_count;
            return buffer;
        };
        const auto alloc = [&](std::initializer_list<std::uint64_t> shape,
                               DType dtype = DType::f32) {
            auto buffer = allocate_metal_backend_buffer(
                tensor::TensorDescriptor::from(tensor::Shape::from(shape), dtype));
            resident_bytes += buffer.descriptor().byte_count;
            std::memset(contents(buffer), 0, buffer.descriptor().byte_count);
            return buffer;
        };
        embedding = load("token_embd.weight", hidden_width, draft_vocabulary_size, DType::q8_0);
        final_norm = load("output_norm.weight", hidden_width, 0, DType::f32);
        constexpr std::array<std::uint32_t, 7> inputs = {hidden_width, hidden_width, hidden_width,
            1024, hidden_width, hidden_width, intermediate_width};
        constexpr std::array<std::uint32_t, 7> outputs = {
            1024, 256, 256, hidden_width, intermediate_width, intermediate_width, hidden_width};
        for (std::uint32_t index = 0; index < layer_count; ++index) {
            auto& layer = layers[index];
            const auto prefix = "blk." + std::to_string(index) + ".";
            for (std::uint32_t weight_index = 0; weight_index < 13; ++weight_index)
                layer.weights[weight_index] = load(prefix + weight_suffixes[weight_index],
                    weight_index < 7 ? inputs[weight_index]
                                     : (weight_index >= 11 ? 256 : hidden_width),
                    weight_index < 7 ? outputs[weight_index] : 0,
                    weight_index < 7 ? DType::q8_0 : DType::f32);
            const auto rows = is_local_layer(index) ? 512 : context_rows;
            layer.key = alloc({rows, 256});
            layer.value = alloc({rows, 256});
            layer.private_key = alloc({std::max(prefill_chunk_rows, transaction_rows), 256});
            layer.private_value = alloc({std::max(prefill_chunk_rows, transaction_rows), 256});
        }
        scratch_layout(horizon_capacity, transaction_rows, context_rows,
            [&](Buffer Impl::* member, std::initializer_list<std::uint64_t> shape) {
                this->*member = alloc(shape);
            });
        std::memcpy(contents(mapping), draft_to_target.data(), draft_to_target.size_bytes());
        resident_bytes += reverse_map.capacity() * sizeof(std::int32_t);
    }

    void require_live() const {
        if (is_poisoned)
            throw std::logic_error("Gemma3 draft plan is poisoned");
    }

    void invalidate() noexcept {
        probability_owner.invalidate();
        published_epoch = 0;
        proposal_length = 0;
    }

    void start() {
        require_live();
        running = {};
        command_started_ns = now_ns();
        *static_cast<std::uint32_t*>(contents(status)) = 0;
        running.uploaded_bytes = sizeof(std::uint32_t);
        command = [shared->queue commandBuffer];
        if (!command)
            throw std::runtime_error("Gemma3 command allocation failed");
        command.label = @"Gemma3 resident draft";
        encoder = [command computeCommandEncoder];
        if (!encoder)
            throw std::runtime_error("Gemma3 encoder allocation failed");
    }

    StandaloneDraftExecutionStats finish() {
        keep_metal_buffers_resident();
        [encoder endEncoding];
        encoder = nil;
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            is_poisoned = true;
            invalidate();
            throw std::runtime_error(
                "Gemma3 Metal execution: " + ns_error_message(command.error, "command failed"));
        }
        const auto status_code = *static_cast<const std::uint32_t*>(contents(status));
        if (status_code) {
            is_poisoned = true;
            invalidate();
            throw std::runtime_error(
                "Gemma3 draft numerical refusal: " + std::to_string(status_code));
        }
        running.command_buffer_count = 1;
        running.readback_bytes += sizeof(std::uint32_t);
        if (command.GPUEndTime >= command.GPUStartTime)
            running.gpu_duration_ns =
                static_cast<std::uint64_t>((command.GPUEndTime - command.GPUStartTime) * 1.e9);
        running.wall_duration_ns = now_ns() - command_started_ns;
        command = nil;
        return running;
    }

    void pipeline(Pipeline pipeline_id) {
        [encoder setComputePipelineState:pipelines[pipeline_id]];
    }

    template <class T> void constant(const T& value, NSUInteger index) {
        [encoder setBytes:&value length:sizeof(value) atIndex:index];
    }

    void barrier() {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        ++running.memory_barrier_count;
    }

    void dispatch(std::uint32_t groups, std::uint32_t threads = 256, std::uint32_t rows = 1,
        std::uint32_t partitions = 1, bool synchronize = true) {
        [encoder dispatchThreadgroups:MTLSizeMake(groups, rows, partitions)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        ++running.compute_dispatch_count;
        if (synchronize)
            barrier();
    }

    void linear(const BackendBuffer& input, const BackendBuffer& weight,
        const BackendBuffer& output, std::uint32_t rows, std::uint32_t input_width,
        std::uint32_t output_width, std::size_t input_offset = 0, bool synchronize = true) {
        // Same row-parallel reduction for scalar decode and every prefill row.
        // A Q8 row is 20 blocks at width640; no Q4/width-two surrogate is used.
        pipeline(q8);
        bind(encoder, input, 0, input_offset);
        bind(encoder, weight, 1);
        bind(encoder, output, 2);
        constant(std::array<std::uint32_t, 3>{rows, input_width, output_width}, 3);
        dispatch((output_width + 3) / 4, 64, rows, 1, synchronize);
    }

    void normalize(const BackendBuffer& input, const BackendBuffer& weight,
        const BackendBuffer& output, std::uint32_t rows, std::uint32_t width,
        std::size_t input_offset = 0, std::size_t output_offset = 0, bool synchronize = true) {
        pipeline(norm);
        bind(encoder, input, 0, input_offset);
        bind(encoder, weight, 1);
        bind(encoder, output, 2, output_offset);
        bind(encoder, status, 3);
        constant(std::array<std::uint32_t, 2>{rows, width}, 4);
        dispatch(rows, 32, 1, 1, synchronize);
    }

    void residual(std::uint32_t rows) {
        pipeline(add);
        bind(encoder, hidden_states, 0);
        bind(encoder, residual_branch, 1);
        bind(encoder, status, 2);
        constant(rows * hidden_width, 3);
        dispatch((rows * hidden_width + 255) / 256);
    }
    // Only publication writes private append state into durable cache storage.
    // Local layers retain the newest 512 rows in their resident ring.
    void publish_rows(std::uint32_t begin, std::uint32_t end, std::uint32_t private_begin) {
        if (begin == end)
            return;
        for (std::uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            auto& layer = layers[layer_index];
            const auto first = is_local_layer(layer_index) && end - begin > 512 ? end - 512 : begin;
            pipeline(publish);
            bind(encoder, layer.private_key, 0);
            bind(encoder, layer.private_value, 1);
            bind(encoder, layer.key, 2);
            bind(encoder, layer.value, 3);
            constant(
                std::array<std::uint32_t, 4>{first, end - first, private_begin,
                    is_local_layer(layer_index) ? 511U : std::numeric_limits<std::uint32_t>::max()},
                4);
            dispatch(1, 256, end - first, 1, false);
        }
        pipeline(copy);
        bind(encoder, private_norm, 0,
            std::size_t(end - 1 - private_begin) * hidden_width * sizeof(float));
        bind(encoder, durable_norm, 1);
        dispatch(3);
    }
    // Inference writes private K/V and normalized prefix states. Outside a
    // transaction, the same submission also publishes these known input rows.
    void forward(const BackendBuffer& tokens, std::uint32_t token_offset, std::uint32_t rows,
        std::uint32_t position) {
        const auto private_begin = in_transaction ? transaction_begin : position;
        pipeline(embed);
        bind(encoder, tokens, 0);
        bind(encoder, embedding, 1);
        bind(encoder, hidden_states, 2);
        bind(encoder, status, 3);
        constant(std::array<std::uint32_t, 4>{rows, token_offset, position, 0}, 4);
        bind(encoder, native_history, 5);
        dispatch(3, 256, rows);
        for (std::uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            auto& layer = layers[layer_index];
            auto& weights = layer.weights;
            // Attention branch: normalize, project Q/K/V, rotate, attend.
            normalize(hidden_states, weights[7], normalized_hidden, rows, hidden_width);
            // Q/K/V share a read-only input and have disjoint outputs. Q/K
            // norms likewise join before RoPE; status writes are atomic.
            linear(normalized_hidden, weights[0], raw_queries, rows, hidden_width, 1024, 0, false);
            linear(normalized_hidden, weights[1], raw_keys, rows, hidden_width, 256, 0, false);
            linear(normalized_hidden, weights[2], raw_values, rows, hidden_width, 256);
            normalize(raw_queries, weights[11], normalized_queries, rows * 4, 256, 0, 0, false);
            normalize(raw_keys, weights[12], normalized_keys, rows, 256);
            pipeline(rope);
            bind(encoder, normalized_queries, 0);
            bind(encoder, normalized_keys, 1);
            bind(encoder, raw_values, 2);
            bind(encoder, rotated_queries, 3);
            bind(encoder, layer.private_key, 4);
            bind(encoder, layer.private_value, 5);
            bind(encoder, status, 6);
            constant(std::array<std::uint32_t, 4>{rows, position, private_begin, 0}, 7);
            constant(is_local_layer(layer_index) ? 10000.0F : 1000000.0F, 8);
            dispatch(4, 256, rows);
            const auto max_parts =
                is_local_layer(layer_index) ? 1U : std::min(32U, (position + rows + 511) / 512);
            const std::array<std::uint32_t, 5> params{
                rows, position, private_begin, is_local_layer(layer_index) ? 1U : 0U, max_parts};
            pipeline(attention);
            bind(encoder, rotated_queries, 0);
            bind(encoder, layer.key, 1);
            bind(encoder, layer.value, 2);
            bind(encoder, layer.private_key, 3);
            bind(encoder, layer.private_value, 4);
            bind(encoder, partial, 5);
            bind(encoder, scales, 6);
            constant(params, 7);
            bind(encoder, attention_output, 8);
            bind(encoder, status, 9);
            dispatch(4, 128, rows, max_parts);
            if (max_parts > 1) {
                pipeline(attention_reduce);
                bind(encoder, partial, 0);
                bind(encoder, scales, 1);
                bind(encoder, attention_output, 2);
                bind(encoder, status, 3);
                constant(params, 4);
                dispatch(4, 32, rows);
            }
            linear(attention_output, weights[3], projected_output, rows, 1024, hidden_width);
            normalize(projected_output, weights[8], residual_branch, rows, hidden_width);
            residual(rows);
            // Feed-forward branch: normalize, gate/up, GELU, down, residual.
            normalize(hidden_states, weights[9], normalized_hidden, rows, hidden_width);
            linear(normalized_hidden, weights[4], gate_states, rows, hidden_width,
                intermediate_width, 0, false);
            linear(
                normalized_hidden, weights[5], up_states, rows, hidden_width, intermediate_width);
            pipeline(gelu);
            bind(encoder, gate_states, 0);
            bind(encoder, up_states, 1);
            constant(rows * intermediate_width, 2);
            dispatch((rows * intermediate_width + 255) / 256);
            linear(
                gate_states, weights[6], projected_output, rows, intermediate_width, hidden_width);
            normalize(projected_output, weights[10], residual_branch, rows, hidden_width);
            residual(rows);
        }
        // Retain each final normalized state until its prefix is committed.
        normalize(hidden_states, final_norm, private_norm, rows, hidden_width, 0,
            std::size_t(position - private_begin) * hidden_width * sizeof(float));
        if (!in_transaction)
            publish_rows(position, position + rows, private_begin);
        running.token_count += rows;
    }

    void materialize_pending() {
        if (!has_pending)
            return;
        *static_cast<std::uint32_t*>(contents(pending_input)) = pending_token;
        running.uploaded_bytes += sizeof(pending_token);
        forward(pending_input, 0, 1, materialized_position);
        ++materialized_position;
        has_pending = false;
    }

    void project() {
        if (!materialized_position || has_pending)
            throw std::logic_error("Gemma3 head has no materialized frontier");
        if (in_transaction && materialized_position > transaction_begin)
            linear(private_norm, embedding, logits, 1, hidden_width, draft_vocabulary_size,
                std::size_t(materialized_position - 1 - transaction_begin) * hidden_width *
                    sizeof(float));
        else
            linear(durable_norm, embedding, logits, 1, hidden_width, draft_vocabulary_size);
    }

    void sample(std::uint32_t depth, const TokenSamplerOptions& options,
        std::uint32_t history_count, bool bounded_repetition_history) {
        if (options.repetition_penalty != 1.0F) {
            pipeline(history_clear);
            bind(encoder, history_bits, 0);
            dispatch((draft_vocabulary_size / 32 + 255) / 256);
            pipeline(history_mark);
            bind(encoder, native_history, 0);
            bind(encoder, history_bits, 1);
            const auto count = bounded_repetition_history
                                   ? static_cast<std::uint32_t>(std::min<std::size_t>(
                                         history_count, options.repetition_window))
                                   : history_count;
            constant(std::array<std::uint32_t, 2>{history_count - count, count}, 2);
            if (count)
                dispatch((count + 255) / 256);
        }
        // Apply the validated sampler policy before constructing its GPU CDF.
        pipeline(process);
        bind(encoder, logits, 0);
        bind(encoder, mapping, 1);
        bind(encoder, history_bits, 2);
        bind(encoder, processed_logits, 3);
        bind(encoder, status, 4);
        constant(std::array<float, 2>{options.temperature, options.repetition_penalty}, 5);
        bind(encoder, active, 6);
        dispatch(vocabulary_tile_count);
        const float temperature = options.temperature == 0 ? 0.0F : 1.0F;
        const bool compact = temperature > 0 && options.top_k > 0 && options.top_k <= 64;
        static_cast<std::uint32_t*>(contents(compact_counts))[depth] =
            compact ? 0U : std::numeric_limits<std::uint32_t>::max();
        running.uploaded_bytes += sizeof(std::uint32_t);
        const auto bind_count = [&](NSUInteger index) {
            bind(encoder, compact_counts, index, std::size_t(depth) * sizeof(std::uint32_t));
        };
        if (temperature > 0 && (options.top_k != 0 || options.top_p < 1 || options.min_p > 0)) {
            pipeline(sort_tile);
            bind(encoder, processed_logits, 0);
            bind(encoder, sort_a, 1);
            dispatch(draft_vocabulary_size / 512, 512);
            // Bounded support keeps the common top-k path compact. The full
            // path below remains encoded for support-overflow fallback.
            if (compact) {
                const BackendBuffer* input = &static_cast<const BackendBuffer&>(sort_a);
                bool write_first_buffer = true;
                std::uint32_t stride = 512;
                for (std::uint32_t groups = draft_vocabulary_size / 512 / 2; groups; groups /= 2) {
                    auto& output = write_first_buffer ? bounded_a : bounded_b;
                    pipeline(bounded_merge);
                    bind(encoder, *input, 0);
                    bind(encoder, output, 1);
                    constant(std::array<std::uint32_t, 2>{groups, stride}, 2);
                    dispatch((groups * 64 + 255) / 256);
                    input = &static_cast<const BackendBuffer&>(output);
                    stride = 64;
                    write_first_buffer = !write_first_buffer;
                }
                pipeline(compact_collect);
                bind(encoder, processed_logits, 0);
                bind(encoder, mapping, 1);
                bind(encoder, *input, 2);
                bind(encoder, compact_support, 3);
                bind_count(4);
                constant(static_cast<std::uint32_t>(options.top_k), 5);
                dispatch(vocabulary_tile_count);
                pipeline(compact_sample);
                bind(encoder, compact_support, 0);
                bind_count(1);
                bind(encoder, compact_ids, 2, std::size_t(depth) * 256 * sizeof(std::uint32_t));
                bind(encoder, compact_cdfs, 3, std::size_t(depth) * 257 * sizeof(std::uint64_t));
                bind(encoder, mapping, 4);
                bind(encoder, words, 5);
                bind(encoder, draft_tokens, 6);
                bind(encoder, target_tokens, 7);
                bind(encoder, status, 8);
                constant(depth, 9);
                constant(std::array<float, 2>{options.top_p, options.min_p}, 10);
                dispatch(1);
            }
            bool sorted_in_first_buffer = true;
            for (std::uint32_t width = 512; width < draft_vocabulary_size; width *= 2) {
                pipeline(full_merge);
                bind(encoder, sorted_in_first_buffer ? sort_a : sort_b, 0);
                bind(encoder, sorted_in_first_buffer ? sort_b : sort_a, 1);
                constant(width, 2);
                bind_count(3);
                dispatch(vocabulary_tile_count);
                sorted_in_first_buffer = !sorted_in_first_buffer;
            }
            const auto& sorted = sorted_in_first_buffer ? sort_a : sort_b;
            const auto top_k = static_cast<std::uint32_t>(options.top_k);
            pipeline(filter_prefix);
            bind(encoder, sorted, 0);
            bind(encoder, mapping, 1);
            bind(encoder, local_prefix, 2);
            bind(encoder, tile_totals, 3);
            constant(top_k, 4);
            bind_count(5);
            dispatch(vocabulary_tile_count);
            pipeline(offsets);
            bind(encoder, tile_totals, 0);
            bind(encoder, tile_offsets, 1);
            bind(encoder, status, 2);
            constant(temperature, 3);
            bind_count(4);
            dispatch(1, 1);
            pipeline(filter_apply);
            bind(encoder, sorted, 0);
            bind(encoder, mapping, 1);
            bind(encoder, local_prefix, 2);
            bind(encoder, tile_offsets, 3);
            bind(encoder, processed_logits, 4);
            constant(top_k, 5);
            constant(std::array<float, 2>{options.top_p, options.min_p}, 6);
            bind(encoder, active, 7);
            bind_count(8);
            dispatch(vocabulary_tile_count);
        }
        // The full-vocabulary receipt and selection kernels retain their
        // fixed dispatch order; compact-count guards choose the active law.
        pipeline(maxima);
        bind(encoder, processed_logits, 0);
        bind(encoder, active, 1);
        bind(encoder, tile_max, 2);
        bind(encoder, tile_argmax, 3);
        bind(encoder, status, 4);
        bind_count(5);
        dispatch(vocabulary_tile_count);
        pipeline(max_finish);
        bind(encoder, tile_max, 0);
        bind(encoder, tile_argmax, 1);
        bind(encoder, status, 2);
        bind_count(3);
        dispatch(1, 1);
        pipeline(prefix);
        bind(encoder, processed_logits, 0);
        bind(encoder, active, 1);
        bind(encoder, tile_max, 2);
        bind(encoder, local_prefix, 3);
        bind(encoder, tile_totals, 4);
        constant(temperature, 5);
        bind_count(6);
        dispatch(vocabulary_tile_count);
        pipeline(offsets);
        bind(encoder, tile_totals, 0);
        bind(encoder, tile_offsets, 1);
        bind(encoder, status, 2);
        constant(temperature, 3);
        bind_count(4);
        dispatch(1, 1);
        const auto offset =
            std::size_t(depth) * (draft_vocabulary_size + 1) * sizeof(std::uint64_t);
        pipeline(cdf);
        bind(encoder, local_prefix, 0);
        bind(encoder, tile_offsets, 1);
        bind(encoder, tile_argmax, 2);
        bind(encoder, cdfs, 3, offset);
        constant(temperature, 4);
        bind(encoder, status, 5);
        bind_count(6);
        dispatch((draft_vocabulary_size + 256) / 256);
        pipeline(select);
        bind(encoder, cdfs, 0, offset);
        bind(encoder, mapping, 1);
        bind(encoder, words, 2);
        bind(encoder, draft_tokens, 3);
        bind(encoder, target_tokens, 4);
        bind(encoder, status, 5);
        constant(depth, 6);
        bind(encoder, active, 7);
        bind_count(8);
        dispatch(vocabulary_tile_count);
    }
};

Gemma3DraftPlan::Gemma3DraftPlan(const StandaloneDraftArtifact& artifact,
    std::span<const std::int32_t> draft_to_target, std::uint32_t target_vocabulary,
    std::uint32_t max_horizon, std::uint32_t transaction_rows, std::uint32_t context_rows)
    : impl_(nullptr) {
    if (artifact.kind() != StandaloneDraftKind::gemma3_270m_q8)
        throw std::invalid_argument("Gemma3 draft requires its authenticated Q8 artifact");
    impl_ = std::make_unique<Impl>(
        TensorSource{&artifact,
            [](const void* source_context, std::string_view name) {
                return static_cast<const StandaloneDraftArtifact*>(source_context)->tensor(name);
            }},
        draft_to_target, target_vocabulary, max_horizon, transaction_rows, context_rows);
}

std::size_t Gemma3DraftPlan::estimate_owned_bytes(std::uint32_t target_vocabulary,
    std::uint32_t horizon_capacity, std::uint32_t transaction_rows, std::uint32_t context_rows) {
    if (target_vocabulary == 0 || horizon_capacity == 0 ||
        horizon_capacity > external_speculation_max_horizon ||
        transaction_rows < horizon_capacity + 2 || transaction_rows > maximum_context_rows ||
        context_rows == 0 || context_rows > maximum_context_rows)
        throw std::invalid_argument("Gemma3 draft estimate has incompatible geometry");
    std::size_t resident_bytes = sizeof(Impl);
    const auto add = [&](std::size_t amount) {
        if (amount > std::numeric_limits<std::size_t>::max() - resident_bytes)
            throw std::overflow_error("Gemma3 draft resident estimate overflow");
        resident_bytes += amount;
    };
    const auto shape_bytes = [](std::initializer_list<std::uint64_t> shape) {
        std::size_t result = sizeof(float);
        for (auto count : shape) {
            if (count > std::numeric_limits<std::size_t>::max() / result)
                throw std::overflow_error("Gemma3 draft buffer estimate overflow");
            result *= count;
        }
        return result;
    };
    // GGUF Q8 rows contain a half scale plus 32 int8 values per block. Tied
    // output embeddings are counted once, exactly as resident preparation.
    add(std::size_t(hidden_width / 32) * 34 * draft_vocabulary_size);
    add(hidden_width * sizeof(float));
    constexpr std::array<std::uint32_t, 7> inputs = {hidden_width, hidden_width, hidden_width, 1024,
        hidden_width, hidden_width, intermediate_width};
    constexpr std::array<std::uint32_t, 7> outputs = {
        1024, 256, 256, hidden_width, intermediate_width, intermediate_width, hidden_width};
    for (std::uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        for (std::uint32_t weight_index = 0; weight_index < 7; ++weight_index)
            add(std::size_t(inputs[weight_index] / 32) * 34 * outputs[weight_index]);
        add((4 * hidden_width + 2 * 256) * sizeof(float));
        add(shape_bytes({2, is_local_layer(layer_index) ? 512U : context_rows, 256}));
        add(shape_bytes({2, std::max(prefill_chunk_rows, transaction_rows), 256}));
    }
    Impl::scratch_layout(horizon_capacity, transaction_rows, context_rows,
        [&](Buffer Impl::*, std::initializer_list<std::uint64_t> shape) {
            add(shape_bytes(shape));
        });
    add(shape_bytes({target_vocabulary}));
    return resident_bytes;
}

Gemma3DraftPlan::~Gemma3DraftPlan() = default;
Gemma3DraftPlan::Gemma3DraftPlan(Gemma3DraftPlan&&) noexcept = default;
Gemma3DraftPlan& Gemma3DraftPlan::operator=(Gemma3DraftPlan&&) noexcept = default;
std::uint32_t Gemma3DraftPlan::prepared_context_capacity() const noexcept {
    return impl_->context_rows;
}

std::uint32_t Gemma3DraftPlan::position() const noexcept {
    return impl_->logical_position;
}

std::uint32_t Gemma3DraftPlan::materialized_position() const noexcept {
    return impl_->materialized_position;
}

std::uint32_t Gemma3DraftPlan::horizon_capacity() const noexcept {
    return impl_->horizon_capacity;
}

std::size_t Gemma3DraftPlan::owned_bytes() const noexcept {
    return sizeof(Impl) + impl_->resident_bytes;
}

bool Gemma3DraftPlan::poisoned() const noexcept {
    return impl_->is_poisoned;
}

bool Gemma3DraftPlan::transaction_active() const noexcept {
    return impl_->in_transaction;
}

void Gemma3DraftPlan::reset() noexcept {
    auto& plan = *impl_;
    plan.invalidate();
    plan.logical_position = plan.materialized_position = 0;
    plan.has_pending = false;
    plan.in_transaction = false;
}

StandaloneDraftExecutionStats Gemma3DraftPlan::begin_transaction() {
    auto& plan = *impl_;
    plan.require_live();
    if (plan.in_transaction)
        throw std::logic_error("Gemma3 transaction is already active");
    plan.invalidate();
    plan.in_transaction = true;
    plan.transaction_begin = plan.materialized_position;
    plan.saved_logical_position = plan.logical_position;
    plan.saved_materialized_position = plan.materialized_position;
    plan.saved_has_pending = plan.has_pending;
    plan.saved_pending_token = plan.pending_token;
    return {};
}

bool Gemma3DraftPlan::can_truncate(std::uint32_t position) const noexcept {
    const auto& plan = *impl_;
    return !plan.is_poisoned && position <= plan.logical_position &&
           (position == plan.logical_position ||
               (plan.in_transaction && position >= plan.saved_logical_position));
}

void Gemma3DraftPlan::truncate(std::uint32_t position) {
    auto& plan = *impl_;
    plan.require_live();
    if (!can_truncate(position))
        throw std::invalid_argument("Gemma3 cannot logically truncate published sliding K/V");
    if (position != plan.logical_position) {
        plan.invalidate();
        plan.logical_position = position;
        plan.materialized_position = std::min(plan.materialized_position, position);
        plan.has_pending = false;
    }
}

void Gemma3DraftPlan::abort_transaction() noexcept {
    auto& plan = *impl_;
    plan.invalidate();
    if (!plan.in_transaction) {
        plan.is_poisoned = true;
        return;
    }
    plan.logical_position = plan.saved_logical_position;
    plan.materialized_position = plan.saved_materialized_position;
    plan.has_pending = plan.saved_has_pending;
    plan.pending_token = plan.saved_pending_token;
    plan.in_transaction = false;
}
// Publish the durable materialized prefix before advancing the logical ledger.
// A retained pending token stays selected without being forwarded here.
StandaloneDraftExecutionStats Gemma3DraftPlan::commit_prefix(std::uint32_t position) {
    auto& plan = *impl_;
    plan.require_live();
    if (!plan.in_transaction || position < plan.saved_logical_position ||
        position > plan.logical_position)
        throw std::invalid_argument("Gemma3 commit is outside its transaction");
    const auto materialized_position = std::min(position, plan.materialized_position);
    const bool pending = plan.has_pending && position == plan.logical_position;
    plan.invalidate();
    StandaloneDraftExecutionStats stats;
    if (materialized_position > plan.transaction_begin) {
        @autoreleasepool {
            plan.start();
            plan.publish_rows(
                plan.transaction_begin, materialized_position, plan.transaction_begin);
            stats = plan.finish();
        }
    }
    plan.logical_position = position;
    plan.materialized_position = materialized_position;
    plan.has_pending = pending;
    plan.in_transaction = false;
    return stats;
}

StandaloneDraftExecutionStats Gemma3DraftPlan::append(std::span<const std::uint32_t> tokens) {
    auto& plan = *impl_;
    plan.require_live();
    if (tokens.size() > plan.context_rows - plan.logical_position ||
        (plan.in_transaction &&
            tokens.size() >
                plan.transaction_capacity - (plan.logical_position - plan.transaction_begin)))
        throw std::length_error("Gemma3 draft append exceeds its prepared capacity");
    for (auto token : tokens)
        if (token >= draft_vocabulary_size)
            throw std::invalid_argument("Gemma3 draft token is outside vocabulary");
    if (tokens.empty())
        return {};
    plan.invalidate();
    StandaloneDraftExecutionStats result;
    const auto started = now_ns();
    @autoreleasepool {
        for (std::size_t offset = 0; offset < tokens.size(); offset += prefill_chunk_rows) {
            const auto rows = static_cast<std::uint32_t>(
                std::min<std::size_t>(prefill_chunk_rows, tokens.size() - offset));
            plan.start();
            std::memcpy(contents(plan.ids), tokens.data() + offset, rows * sizeof(std::uint32_t));
            plan.running.uploaded_bytes += rows * sizeof(std::uint32_t);
            plan.materialize_pending();
            plan.forward(plan.ids, 0, rows, plan.materialized_position);
            plan.materialized_position += rows;
            plan.logical_position += rows;
            combine(result, plan.finish());
        }
    }
    result.wall_duration_ns = now_ns() - started;
    return result;
}

Gemma3DraftProposal Gemma3DraftPlan::propose(std::uint32_t target_root,
    std::span<const std::uint32_t> words, const TokenSamplerOptions& options,
    const SamplerHistorySnapshot& history, std::span<const std::uint32_t> appended_tokens,
    bool bounded_repetition_history) {
    auto& plan = *impl_;
    plan.require_live();
    validate_options(options);
    if (!plan.in_transaction || words.empty() || words.size() > plan.horizon_capacity ||
        (plan.logical_position == 0 && appended_tokens.empty()) ||
        appended_tokens.size() > plan.context_rows - plan.logical_position ||
        words.size() > plan.context_rows - plan.logical_position - appended_tokens.size() ||
        appended_tokens.size() >
            plan.transaction_capacity - (plan.logical_position - plan.transaction_begin) ||
        words.size() > plan.transaction_capacity -
                           (plan.logical_position - plan.transaction_begin) -
                           appended_tokens.size() ||
        target_root >= plan.target_vocabulary_size)
        throw std::invalid_argument(
            "Gemma3 proposal has incompatible transaction, horizon or capacity");
    for (auto token : appended_tokens)
        if (token >= draft_vocabulary_size)
            throw std::invalid_argument("Gemma3 proposal root token is outside vocabulary");
    if (history.head >= token_sampler_max_repetition_window ||
        history.length > history.window_limit ||
        history.window_limit > token_sampler_max_repetition_window ||
        (history.length && history.head >= history.window_limit) ||
        (!history.length && history.head))
        throw std::invalid_argument("Gemma3 proposal sampler history is malformed");
    if (plan.epoch == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Gemma3 proposal epoch overflow");
    plan.invalidate();
    plan.probability_owner.begin_epoch();
    const auto count = static_cast<std::uint32_t>(words.size()),
               start_position =
                   plan.logical_position + static_cast<std::uint32_t>(appended_tokens.size());
    @autoreleasepool {
        plan.start();
        std::memcpy(contents(plan.words), words.data(), words.size_bytes());
        auto* target_tokens = static_cast<std::uint32_t*>(contents(plan.target_tokens));
        target_tokens[0] = target_root;
        std::fill_n(
            static_cast<std::uint32_t*>(contents(plan.draft_tokens)), count, draft_vocabulary_size);
        std::fill_n(target_tokens + 1, count, plan.target_vocabulary_size);
        VerifierDeviceSelectionRow row;
        row.history_token_ids = history.token_ids;
        row.history_head = history.head;
        row.history_length = history.length;
        row.history_window_limit = history.window_limit;
        static_assert(sizeof(row) == 91 * sizeof(std::uint32_t));
        std::memcpy(contents(plan.base_history), &row, sizeof(row));
        plan.running.uploaded_bytes += words.size_bytes() + sizeof(target_root) + sizeof(row) +
                                       2 * count * sizeof(std::uint32_t);
        if (appended_tokens.empty())
            plan.materialize_pending();
        else {
            // Both a saved pending emission and the root are already known.
            // Their causal rows can share ingestion kernels; future sampled
            // draft tokens still execute sequentially below.
            auto* input = static_cast<std::uint32_t*>(contents(plan.proposal_input));
            const std::uint32_t pending_rows = plan.has_pending ? 1U : 0U;
            if (pending_rows)
                input[0] = plan.pending_token;
            std::memcpy(input + pending_rows, appended_tokens.data(), appended_tokens.size_bytes());
            const auto input_rows =
                static_cast<std::uint32_t>(appended_tokens.size()) + pending_rows;
            plan.running.uploaded_bytes += std::size_t(input_rows) * sizeof(std::uint32_t);
            for (std::uint32_t offset = 0; offset < input_rows; offset += prefill_chunk_rows) {
                const auto rows = std::min(prefill_chunk_rows, input_rows - offset);
                plan.forward(plan.proposal_input, offset, rows, plan.materialized_position);
                plan.materialized_position += rows;
            }
            plan.logical_position += static_cast<std::uint32_t>(appended_tokens.size());
            plan.has_pending = false;
        }
        // Sample each causal head in order. The final emission remains pending
        // so a rejected suffix never executes an otherwise unused forward pass.
        for (std::uint32_t depth = 0; depth < count; ++depth) {
            plan.project();
            plan.sample(depth, options, plan.logical_position, bounded_repetition_history);
            if (depth + 1 < count) {
                plan.forward(plan.draft_tokens, depth, 1, plan.logical_position);
                ++plan.materialized_position;
            } else {
                plan.pipeline(pending);
                bind(plan.encoder, plan.draft_tokens, 0);
                bind(plan.encoder, plan.native_history, 1);
                plan.constant(std::array<std::uint32_t, 2>{depth, plan.logical_position}, 2);
                plan.dispatch(1, 1);
            }
            ++plan.logical_position;
        }
        plan.pipeline(requests);
        bind(plan.encoder, plan.base_history, 0);
        bind(plan.encoder, plan.target_tokens, 1);
        bind(plan.encoder, plan.device_requests, 2);
        plan.constant(count, 3);
        plan.dispatch(1, 32);
        auto stats = plan.finish();
        const auto* draft_tokens = static_cast<const std::uint32_t*>(contents(plan.draft_tokens));
        for (std::uint32_t depth = 0; depth < count; ++depth)
            if (target_tokens[depth + 1] >= plan.target_vocabulary_size ||
                draft_tokens[depth] >= draft_vocabulary_size) {
                plan.is_poisoned = true;
                plan.invalidate();
                throw std::runtime_error("Gemma3 draft failed to publish a selected token");
            }
        // Expose tokens and their probability lease only after GPU completion
        // and selected-token validation. Cache publication is a separate commit.
        plan.has_pending = true;
        plan.pending_token = draft_tokens[count - 1];
        plan.published_epoch = ++plan.epoch;
        plan.proposal_length = count;
        plan.probability_readback_bytes = 0;
        stats.readback_bytes += count * 2 * sizeof(std::uint32_t);
        stats.wall_duration_ns = now_ns() - plan.command_started_ns;
        return {.draft_tokens = {draft_tokens, count},
            .target_tokens = {target_tokens + 1, count},
            .start_position = start_position,
            .materialized_end = plan.materialized_position,
            .epoch = plan.published_epoch,
            .execution = stats};
    }
}

StandaloneDraftExecutionStats Gemma3DraftPlan::restore_pending_token(std::uint32_t token) {
    auto& plan = *impl_;
    plan.require_live();
    const auto started = now_ns();
    if (plan.in_transaction || plan.has_pending ||
        plan.logical_position != plan.materialized_position ||
        plan.logical_position == plan.context_rows || token >= draft_vocabulary_size)
        throw std::invalid_argument(
            "Gemma3 saved pending token is incompatible with its restored prefix");
    plan.invalidate();
    static_cast<std::uint32_t*>(contents(plan.native_history))[plan.logical_position] = token;
    plan.pending_token = token;
    plan.has_pending = true;
    ++plan.logical_position;
    return {.wall_duration_ns = now_ns() - started, .uploaded_bytes = sizeof(token)};
}

VerifierDeviceInputBinding Gemma3DraftPlan::verifier_inputs(std::uint64_t epoch) const {
    const auto& plan = *impl_;
    plan.require_live();
    if (!epoch || epoch != plan.published_epoch)
        throw std::logic_error("Gemma3 proposal device lease is stale");
    return {.token_ids = plan.target_tokens,
        .raw_selection_requests = plan.device_requests,
        .arena_epoch = epoch};
}

GatheredDraftTokenDistribution Gemma3DraftPlan::gather(
    std::uint32_t depth, const PreparedTokenDistribution& target) {
    auto& plan = *impl_;
    plan.require_live();
    if (!plan.published_epoch || depth >= plan.proposal_length || !target.count ||
        target.count > token_sampler_max_top_k)
        throw std::invalid_argument("Gemma3 proposal probability lease is stale or malformed");
    GatheredDraftTokenDistribution result;
    result.lease = plan.probability_owner.lease(depth);
    result.count = target.count;
    const auto compact_count =
        static_cast<const std::uint32_t*>(contents(plan.compact_counts))[depth];
    const bool compact = compact_count <= 256;
    const auto* cdf_entries =
        static_cast<const std::uint64_t*>(contents(compact ? plan.compact_cdfs : plan.cdfs)) +
        std::size_t(depth) * (compact ? 257 : draft_vocabulary_size + 1);
    const auto* support_tokens =
        compact ? static_cast<const std::uint32_t*>(contents(plan.compact_ids)) +
                      std::size_t(depth) * 256
                : nullptr;
    const auto interval = [&](std::uint32_t token) {
        if (!compact)
            return std::pair{cdf_entries[token], cdf_entries[token + 1]};
        const auto* found = std::lower_bound(support_tokens, support_tokens + compact_count, token);
        if (found == support_tokens + compact_count || *found != token)
            return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        const auto index = static_cast<std::size_t>(found - support_tokens);
        return std::pair{cdf_entries[index], cdf_entries[index + 1]};
    };
    const auto selected_draft_token =
        static_cast<const std::uint32_t*>(contents(plan.draft_tokens))[depth];
    result.proposal.token =
        static_cast<const std::uint32_t*>(contents(plan.target_tokens))[depth + 1];
    const auto selected_interval = interval(selected_draft_token);
    result.proposal.probability_mass = selected_interval.second - selected_interval.first;
    result.proposal_cdf_begin = selected_interval.first;
    plan.probability_readback_bytes +=
        2 * sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t) + sizeof(std::uint32_t);
    if (compact)
        plan.probability_readback_bytes += compact_count * sizeof(std::uint32_t);
    for (std::uint32_t index = 0; index < target.count; ++index) {
        const auto token = target.values[index].token;
        if (token >= plan.target_vocabulary_size)
            throw std::invalid_argument("Gemma3 probability gather target token is out of bounds");
        result.target_values[index] = {token, 0};
        const auto draft_token = plan.reverse_map[token];
        if (draft_token >= 0) {
            const auto bounds = interval(static_cast<std::uint32_t>(draft_token));
            result.target_values[index].probability_mass = bounds.second - bounds.first;
            plan.probability_readback_bytes += 2 * sizeof(std::uint64_t);
        }
    }
    return result;
}

std::uint64_t Gemma3DraftPlan::probability_readback_bytes() const noexcept {
    return impl_->probability_readback_bytes;
}

StandaloneDraftExecutionStats Gemma3DraftPlan::read_logits(std::span<float> output) {
    auto& plan = *impl_;
    plan.require_live();
    if (output.size() != draft_vocabulary_size || plan.logical_position == 0)
        throw std::invalid_argument("Gemma3 logit diagnostic shape or frontier is invalid");
    plan.invalidate();
    @autoreleasepool {
        plan.start();
        plan.materialize_pending();
        plan.project();
        auto stats = plan.finish();
        std::memcpy(output.data(), contents(plan.logits), output.size_bytes());
        stats.readback_bytes += output.size_bytes();
        return stats;
    }
}

void Gemma3DraftPlan::read_cache(std::uint32_t layer_index, std::uint32_t first_position,
    std::span<float> keys, std::span<float> values) const {
    const auto& plan = *impl_;
    plan.require_live();
    if (layer_index >= layer_count || keys.size() != values.size() || keys.size() % 256 ||
        first_position > plan.materialized_position ||
        keys.size() / 256 > plan.materialized_position - first_position ||
        (is_local_layer(layer_index) && plan.materialized_position > 512 &&
            first_position < plan.materialized_position - 512))
        throw std::invalid_argument(
            "Gemma3 cache diagnostic is outside the retained logical range");
    const auto& layer = plan.layers[layer_index];
    for (std::size_t row = 0; row < keys.size() / 256; ++row) {
        const auto position = first_position + static_cast<std::uint32_t>(row);
        const bool is_private = plan.in_transaction && position >= plan.transaction_begin;
        const auto slot = is_private ? position - plan.transaction_begin
                                     : (is_local_layer(layer_index) ? position & 511 : position);
        const auto* key_data =
            static_cast<const float*>(contents(is_private ? layer.private_key : layer.key));
        const auto* value_data =
            static_cast<const float*>(contents(is_private ? layer.private_value : layer.value));
        std::copy_n(key_data + std::size_t(slot) * 256, 256, keys.data() + row * 256);
        std::copy_n(value_data + std::size_t(slot) * 256, 256, values.data() + row * 256);
    }
}

void Gemma3DraftPlan::warm_up() {
    auto& plan = *impl_;
    plan.require_live();
    if (plan.logical_position || plan.in_transaction)
        throw std::logic_error("Gemma3 warmup requires an empty prepared plan");
    std::array<std::uint32_t, prefill_chunk_rows> tokens{};
    tokens.fill(2);
    if (plan.context_rows < 3) {
        (void)append(std::span(tokens).first(1));
        @autoreleasepool {
            plan.start();
            plan.project();
            (void)plan.finish();
        }
        reset();
        return;
    }
    const auto proposal_length = std::min(plan.horizon_capacity, plan.context_rows - 2);
    (void)append(std::span(tokens).first(
        std::min(prefill_chunk_rows, plan.context_rows - proposal_length - 2)));
    (void)begin_transaction();
    std::array<std::uint32_t, external_speculation_max_horizon> words{};
    TokenSamplerOptions options;
    options.temperature = 0.8F;
    options.repetition_penalty = 1.1F;
    SamplerHistorySnapshot history;
    history.window_limit = 64;
    const auto batch = propose(
        0, std::span(words).first(proposal_length), options, history, std::span(tokens).first(1));
    for (std::uint32_t row = 0; row < proposal_length; ++row) {
        PreparedTokenDistribution target;
        target.count = 1;
        target.values[0] = {batch.target_tokens[row], 1, 1};
        (void)gather(row, target);
    }
    (void)commit_prefix(position());
    (void)append(std::span(tokens).first(1));
    reset();
}

#if defined(LOCAL_GGUF_ENGINE_TESTING)
Gemma3DraftPlan::Gemma3DraftPlan(std::span<const tensor::TensorView> tensors,
    std::span<const std::int32_t> draft_to_target, std::uint32_t target_vocabulary,
    std::uint32_t max_horizon, std::uint32_t transaction_rows, std::uint32_t context_rows)
    : impl_(nullptr) {
    if (tensors.size() != 2 + 13 * layer_count)
        throw std::invalid_argument("Gemma3 fixture must contain the entire production model");
    impl_ = std::make_unique<Impl>(
        TensorSource{&tensors,
            [](const void* source_context, std::string_view name) {
                const auto views =
                    *static_cast<const std::span<const tensor::TensorView>*>(source_context);
                if (name == "token_embd.weight")
                    return views[0];
                if (name == "output_norm.weight")
                    return views[1];
                for (std::uint32_t layer = 0; layer < layer_count; ++layer)
                    for (std::uint32_t weight_index = 0; weight_index < 13; ++weight_index)
                        if (name ==
                            "blk." + std::to_string(layer) + "." + weight_suffixes[weight_index])
                            return views[2 + 13 * layer + weight_index];
                throw std::invalid_argument("Gemma3 fixture tensor name is invalid");
            }},
        draft_to_target, target_vocabulary, max_horizon, transaction_rows, context_rows);
}

void Gemma3DraftPlan::sample_logits_for_testing(std::span<const float> logits, std::uint32_t word,
    const TokenSamplerOptions& options, bool bounded_repetition_history) {
    auto& plan = *impl_;
    plan.require_live();
    validate_options(options);
    if (logits.size() != draft_vocabulary_size)
        throw std::invalid_argument("Gemma3 diagnostic logits are not full vocabulary");
    plan.invalidate();
    plan.probability_owner.begin_epoch();
    @autoreleasepool {
        plan.start();
        std::memcpy(contents(plan.logits), logits.data(), logits.size_bytes());
        *static_cast<std::uint32_t*>(contents(plan.words)) = word;
        plan.sample(0, options, plan.logical_position, bounded_repetition_history);
        (void)plan.finish();
    }
    plan.published_epoch = ++plan.epoch;
    plan.proposal_length = 1;
    plan.probability_readback_bytes = 0;
}

void Gemma3DraftPlan::seed_zero_cache_prefix_for_testing(std::uint32_t position) {
    auto& plan = *impl_;
    plan.require_live();
    if (position > plan.context_rows || plan.logical_position || plan.in_transaction)
        throw std::invalid_argument("Gemma3 zero-cache fixture requires empty bounded state");
    for (auto& layer : plan.layers)
        for (const Buffer* buffer : {&layer.key, &layer.value})
            std::memset(contents(*buffer), 0, buffer->value->descriptor().byte_count);
    std::memset(contents(plan.durable_norm), 0, hidden_width * sizeof(float));
    std::memset(contents(plan.native_history), 0, plan.context_rows * sizeof(std::uint32_t));
    plan.invalidate();
    plan.materialized_position = plan.logical_position = position;
}
#endif
} // namespace local_gguf_engine::engine
