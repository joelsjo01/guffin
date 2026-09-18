#pragma once

namespace local_gguf_engine::tensor::backends::metal {
// A separate precise library preserves the existing target library's numeric
// behavior. It is compiled during plan construction, never during drafting.
inline constexpr const char* learned_recurrent_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

float rb(float x) {
    uint b=as_type<uint>(x);
    if ((b & 0x7f800000u)==0x7f800000u) return x;
    return as_type<float>((b+0x7fffu+((b>>16)&1u)) & 0xffff0000u);
}

// BF16 products are exact in F32, but an ordinary F32 reduction can cross a
// BF16 midpoint after cancellation. Preserve the low sum through the SIMD
// tree; round once at the F32 linear boundary before the trained BF16 cast.
float2 accum(float2 a,float x) {
    float s=a.x+x, v=s-a.x;
    return float2(s,a.y+((a.x-(s-v))+(x-v)));
}

// Constants: [input width, output width, output offset, flags].
// Flag 1: bias; flag 2: F16 feature-capture boundary before BF16 autocast.
kernel void recurrent_linear(device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]], device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant uint4& p [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float2 partial=float2(0);
    for (uint i=lane;i<p.x;i+=32) {
        float input=(p.w&2u) ? float(half(x[i])) : x[i];
        partial=accum(partial,rb(input)*rb(w[group*p.x+i]));
    }
    for(uint shift=16;shift;shift>>=1) {
        float2 other=simd_shuffle_down(partial,shift);
        if(lane+shift<32) {partial=accum(partial,other.x);partial=accum(partial,other.y);}
    }
    float sum=partial.x+partial.y;
    if (lane==0) {
        float result=rb(sum+((p.w&1u) ? rb(bias[group]) : 0.0f));
        if(!isfinite(result)) atomic_store_explicit(status,2u,memory_order_relaxed);
        y[p.z+group]=result;
    }
}

kernel void recurrent_norm(device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant uint& width [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
    float sum=0;
    for(uint i=lane;i<width;i+=32) sum+=x[i]*x[i];
    sum=simd_sum(sum);
    if(!isfinite(sum)) atomic_store_explicit(status,2u,memory_order_relaxed);
    float inv=rsqrt(sum/float(width)+1.e-6f);
    for(uint i=lane;i<width;i+=32) {
        y[i]=rb((x[i]*inv)*w[i]);
        if(!isfinite(y[i])) atomic_store_explicit(status,2u,memory_order_relaxed);
    }
}

// [target width, vocabulary, proposal depth, state width].
kernel void recurrent_embedding(device const half* table [[buffer(0)]],
    device const uint* tokens [[buffer(1)]], device float* x [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant uint4& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.x) return;
    uint token=tokens[p.z];
    if(token>=p.y) {atomic_store_explicit(status,2u,memory_order_relaxed); token=0;}
    x[i]=rb(rb(float(table[ulong(token)*p.x+i]))*sqrt(float(p.x)));
}

kernel void recurrent_join(device const float* state [[buffer(0)]],
    device const float* token [[buffer(1)]], device float* joined [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i<width) {joined[i]=state[i]; joined[width+i]=token[i];}
}

kernel void recurrent_update(device const float* gate [[buffer(0)]],
    device const float* candidate [[buffer(1)]], device float* state [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i>=width) return;
    float g=rb(1.0f/(1.0f+exp(-gate[i])));
    float c=rb(candidate[i]/(1.0f+exp(-candidate[i])));
    state[i]=rb(rb(g*state[i])+rb(rb(1.0f-g)*c));
}

// CPU equations use double exp/sums, unavailable as a native Metal scalar.
// Outward F32 intervals enclose those decisions. MSL 4 specification
// (2026-06-04), Table 8.1, specifies precise exp <=4 ulp;
// allowance includes rounded division, input subtraction (already F32),
// double-libm error, and directed endpoint arithmetic. Ambiguous decisions
// produce status 1 and no usable chain; they never choose an arbitrary bin.
float down(float x) {return nextafter(x,-INFINITY);}
float up(float x) {return nextafter(x,INFINITY);}
float2 addi(float2 a,float2 b) {return float2(max(0.0f,down(a.x+b.x)),up(a.y+b.y));}
float2 muli(float2 a,float2 b) {return float2(max(0.0f,down(a.x*b.x)),up(a.y*b.y));}
float2 weight_interval(float delta,float temperature) {
    if(delta==0) return float2(1);
    float x=delta/temperature;
    if(x < -80.0f) return float2(0,2.e-35f);
    float value=exp(x);
    float error=fabs(x)*0x1p-22f+0x1p-19f;
    return float2(max(0.0f,down(value*(1-error))),up(value*(1+error)));
}
struct RPair {uint token; float score;};
// [depth, top-k, unused, unused], [temperature, top-p, min-p, unused].
kernel void recurrent_sample(device const RPair* sorted [[buffer(0)]],
    device const ulong* tickets [[buffer(1)]], device uint* tokens [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant uint4& p [[buffer(4)]],
    constant float4& f [[buffer(5)]]) {
    tokens[p.x+1]=0;
    if(atomic_load_explicit(status,memory_order_relaxed)!=0) return;
    float2 pre[256], total=float2(0);
    for(uint i=0;i<p.y;++i) {
        pre[i]=weight_interval(sorted[i].score-sorted[0].score,1.0f);
        total=addi(total,pre[i]);
    }
    uint count=p.y;
    if(f.y<1.0f) {
        float2 partial=float2(0), threshold=muli(total,float2(f.y));
        for(uint i=0;i<p.y;++i) {
            partial=addi(partial,pre[i]);
            if(partial.x>=threshold.y) {count=i+1;break;}
            if(partial.y>=threshold.x) {atomic_store_explicit(status,1u,memory_order_relaxed);return;}
        }
    }
    if(f.z>0.0f) {
        for(uint i=1;i<count;++i) {
            if(pre[i].y<f.z) {count=i;break;}
            if(pre[i].x<f.z) {atomic_store_explicit(status,1u,memory_order_relaxed);return;}
        }
    }
    ulong bits=tickets[p.x];
    uint exponent=uint((bits>>52)&2047ul);
    if((bits>>63)!=0 || exponent>=1023) {atomic_store_explicit(status,2u,memory_order_relaxed);return;}
    if(bits==0 || count==1) {tokens[p.x+1]=sorted[0].token;return;}
    float draw=ldexp(float((bits&0xffffffffffffful)|(exponent ? 0x10000000000000ul:0ul)),
                     int(exponent ? exponent:1)-1075);
    float2 unit=float2(max(0.0f,down(draw)),up(draw));
    total=float2(0);
    for(uint i=0;i<count;++i) {
        pre[i]=weight_interval(sorted[i].score-sorted[0].score,f.x);
        total=addi(total,pre[i]);
    }
    float2 threshold=muli(unit,total), partial=float2(0);
    for(uint i=0;i<count;++i) {
        partial=addi(partial,pre[i]);
        if(threshold.y<partial.x) {tokens[p.x+1]=sorted[i].token;return;}
        if(threshold.x<partial.y) {atomic_store_explicit(status,1u,memory_order_relaxed);return;}
    }
    atomic_store_explicit(status,1u,memory_order_relaxed);
}
// ABI: 64 ring tokens, 23 suffix tokens, head/length/window/suffix_count.
// Only the authoritative base history crosses from the host. Hypothetical
// histories are constructed directly from the device-resident proposal chain.
kernel void recurrent_verifier_requests(device const uint* base [[buffer(0)]],
    device const uint* tokens [[buffer(1)]], device uint* rows [[buffer(2)]],
    constant uint4& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i>p.x) return;
    for (uint j=0;j<91;++j) rows[i*91+j]=base[j];
    for (uint j=0;j<i;++j) rows[i*91+64+j]=tokens[j+1];
    rows[i*91+90]=i;
}
)metal";
}  // namespace local_gguf_engine::tensor::backends::metal
