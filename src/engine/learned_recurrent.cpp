#include "local_gguf_engine/engine/learned_recurrent.hpp"
#include "local_gguf_engine/tensor/f16.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>

namespace local_gguf_engine::engine {
namespace {
std::array<std::int64_t,5> fingerprint(int fd) {
    struct stat s {};
    if (fstat(fd,&s)!=0 || !S_ISREG(s.st_mode) || s.st_size<72 || s.st_size>(1LL<<30)) {
        throw std::runtime_error("recurrent weights must be a bounded regular file");
    }
    return {s.st_size,s.st_mtimespec.tv_sec,s.st_mtimespec.tv_nsec,
            s.st_ctimespec.tv_sec,s.st_ctimespec.tv_nsec};
}
float bf16(float x) {
    auto b=std::bit_cast<std::uint32_t>(x);
    if ((b&0x7f800000U)==0x7f800000U) return x;
    return std::bit_cast<float>((b+0x7fffU+((b>>16U)&1U))&0xffff0000U);
}
float scalar(std::span<const std::uint8_t> bytes,std::size_t i) {
    std::uint32_t bits=0;
    for(unsigned b=0;b<4;++b) bits|=std::uint32_t(bytes[4*i+b])<<(8*b);
    return std::bit_cast<float>(bits);
}
std::vector<float> linear(std::span<const float> x, std::span<const float> weight,
                          std::span<const float> bias, std::size_t rows) {
    std::vector<float> out(rows);
    for(std::size_t row=0;row<rows;++row) {
        // Accumulate the independent CPU equation in double, then round at
        // the trained linear boundary. BF16 operands have exact products.
        double sum=0;
        for(std::size_t i=0;i<x.size();++i) sum+=double(bf16(x[i]))*bf16(weight[row*x.size()+i]);
        out[row]=bf16(static_cast<float>(sum)+(bias.empty()?0.0F:bf16(bias[row])));
    }
    return out;
}
}

RecurrentWeightsFile::RecurrentWeightsFile(const std::filesystem::path& path) {
    fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(fd_<0) throw std::runtime_error("cannot open recurrent weights: "+path.string());
    try {fingerprint_=fingerprint(fd_); digest_=read().envelope_digest();}
    catch(...) {close(fd_);fd_=-1;throw;}
}
RecurrentWeightsFile::~RecurrentWeightsFile() {if(fd_>=0) close(fd_);}
AuthenticatedSpeculativeArtifact RecurrentWeightsFile::read() const {
    auto before=fingerprint(fd_);
    if(before!=fingerprint_) throw std::runtime_error("opened recurrent weights changed");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(before[0]));
    std::size_t done=0;
    while(done<bytes.size()) {
        auto n=pread(fd_,bytes.data()+done,bytes.size()-done,static_cast<off_t>(done));
        if(n<=0) throw std::runtime_error("recurrent weights read failed");
        done+=static_cast<std::size_t>(n);
    }
    if(fingerprint(fd_)!=before) throw std::runtime_error("recurrent weights mutated during read");
    auto artifact=load_speculative_artifact(bytes);
    if(artifact.descriptor().schema_version!=3U || artifact.descriptor().recurrent.version!=1U
        || (!digest_.is_zero() && artifact.envelope_digest()!=digest_)) {
        throw std::runtime_error("recurrent weights source identity mismatch");
    }
    return artifact;
}

std::uint32_t recurrent_sample_cpu(std::span<const float> logits,double draw,const LearnedRecurrentSpec& s) {
    if(logits.empty() || s.top_k==0 || s.top_k>std::min<std::size_t>(logits.size(),256)
        || !std::isfinite(draw) || draw<0 || draw>=1 || std::signbit(draw) || !std::isfinite(s.temperature)
        || s.temperature<=0 || !(s.top_p>0 && s.top_p<=1) || !(s.min_p>=0 && s.min_p<=1)) {
        throw std::invalid_argument("recurrent sampler input");
    }
    std::vector<std::uint32_t> order(logits.size());
    std::iota(order.begin(),order.end(),0U);
    for(float v:logits) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite recurrent logit");
    std::partial_sort(order.begin(),order.begin()+s.top_k,order.end(),[&](auto a,auto b){
        return logits[a]>logits[b] || (logits[a]==logits[b] && a<b);
    });
    std::array<double,256> weights {};
    double total=0;
    for(std::uint32_t i=0;i<s.top_k;++i) total+=(weights[i]=std::exp(double(logits[order[i]]-logits[order[0]])));
    std::uint32_t count=0;double sum=0;
    do {sum+=weights[count++];} while(count<s.top_k && sum/total<double(s.top_p));
    if(s.min_p>0) {std::uint32_t n=1;while(n<count && weights[n]>=double(s.min_p)) ++n;count=n;}
    total=0;
    for(std::uint32_t i=0;i<count;++i) total+=(weights[i]=std::exp(double(logits[order[i]]-logits[order[0]])/double(s.temperature)));
    const double threshold=draw*total;sum=0;
    for(std::uint32_t i=0;i<count;++i) {sum+=weights[i];if(threshold<sum) return order[i];}
    return order[count-1];
}

RecurrentGolden recurrent_cpu_golden(const AuthenticatedSpeculativeArtifact& artifact,
    const tensor::TensorView& table,std::span<const float> norm,std::span<const float> feature,
    std::uint32_t root,std::span<const double> draws,const tensor::TensorView* packed_output_head) {
    const auto& a=artifact.descriptor();const auto& s=a.recurrent;const auto& c=a.target.config;
    const auto d=c.embedding_length,w=s.width,v=c.vocab_size;
    if(s.version!=1 || draws.size()!=s.horizon || feature.size()!=d || norm.size()!=d
        || table.descriptor().dtype!=tensor::DType::f16 || table.bytes().size()!=std::uint64_t(d)*v*2 || root>=v) {
        throw std::invalid_argument("recurrent golden shape");
    }
    if(packed_output_head && (packed_output_head->descriptor().dtype!=tensor::DType::q8_0
        || packed_output_head->descriptor().shape!=table.descriptor().shape || d%32!=0)) {
        throw std::invalid_argument("recurrent golden packed head shape");
    }
    std::array<std::vector<float>,10> weights;
    for(std::size_t i=0;i<10;++i) {
        const auto& t=s.tensors[i];auto bytes=artifact.payload().subspan(t.offset,t.bytes);
        for(std::size_t at=0;at<t.bytes/4;++at) weights[i].push_back(scalar(bytes,at));
    }
    // Training consumes the F16 trace tap, not the unrounded live F32 tap.
    // Preserve that boundary before BF16 autocast, including double-rounding
    // cases near a BF16 midpoint.
    std::vector<float> captured(feature.begin(),feature.end());
    for(auto& x:captured) x=tensor::f16_bits_to_f32(tensor::f32_to_f16_bits(x));
    auto state=linear(captured,weights[0],{},w);
    float squares=0;for(auto x:state) squares+=x*x;
    const float inv=1.0F/std::sqrt(squares/float(w)+1.e-6F);
    for(std::uint32_t i=0;i<w;++i) state[i]=bf16((state[i]*inv)*weights[1][i]);
    auto table_value=[&](std::uint32_t token,std::uint32_t i){
        const auto at=(std::uint64_t(token)*d+i)*2;
        return tensor::f16_bits_to_f32(static_cast<std::uint16_t>(
            std::uint16_t(table.bytes()[at])|(std::uint16_t(table.bytes()[at+1])<<8)));
    };
    auto head_value=[&](std::uint32_t token,std::uint32_t i) {
        if(!packed_output_head) return table_value(token,i);
        // Independent Q8_0 equation: each dimension-0 block is a little-endian
        // half scale followed by 32 signed bytes. No Metal dequantizer or
        // quantizer is used as this projection's numerical oracle.
        const auto bytes=packed_output_head->bytes();
        const auto at=(std::uint64_t(token)*(d/32)+i/32)*34;
        const auto scale=tensor::f16_bits_to_f32(static_cast<std::uint16_t>(
            std::uint16_t(bytes[at])|(std::uint16_t(bytes[at+1])<<8)));
        return scale*float(std::bit_cast<std::int8_t>(bytes[at+2+i%32]));
    };
    RecurrentGolden result;std::uint32_t token=root;
    for(std::uint32_t depth=0;depth<s.horizon;++depth) {
        std::vector<float> embedding(d);
        for(std::uint32_t i=0;i<d;++i) embedding[i]=bf16(bf16(table_value(token,i))*std::sqrt(float(d)));
        auto projected=linear(embedding,weights[2],{},w);
        auto joined=state;joined.insert(joined.end(),projected.begin(),projected.end());
        auto gate=linear(joined,weights[3],weights[4],w);
        auto candidate=linear(joined,weights[5],weights[6],w);
        for(std::uint32_t i=0;i<w;++i) {
            float g=bf16(1.0F/(1.0F+std::exp(-gate[i])));
            float x=bf16(candidate[i]/(1.0F+std::exp(-candidate[i])));
            state[i]=bf16(bf16(g*state[i])+bf16(bf16(1.0F-g)*x));
        }
        auto target=linear(state,weights[7],{},d);
        result.target_states.insert(result.target_states.end(),target.begin(),target.end());
        result.confidence_logits.push_back(linear(state,weights[8],weights[9],1)[0]);
        float sum=0;for(auto x:target) sum+=x*x;
        float scale=1.0F/std::sqrt(sum/float(d)+c.rms_norm_epsilon);
        for(std::uint32_t i=0;i<d;++i) target[i]=(target[i]*scale)*norm[i];
        std::vector<float> logits(v);
        for(std::uint32_t t=0;t<v;++t) {
            double score=0;for(std::uint32_t i=0;i<d;++i) score+=double(target[i])*head_value(t,i);
            logits[t]=c.final_logit_softcap*std::tanh(static_cast<float>(score)/c.final_logit_softcap);
        }
        result.logits.insert(result.logits.end(),logits.begin(),logits.end());
        token=recurrent_sample_cpu(logits,draws[depth],s);result.tokens.push_back(token);
    }
    return result;
}
}  // namespace local_gguf_engine::engine
