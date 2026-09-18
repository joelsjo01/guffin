#include "local_gguf_engine/engine/learned_recurrent.hpp"
#include "local_gguf_engine/engine/speculation_trace.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/f16.hpp"

#include <charconv>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace local_gguf_engine;

// Development-only weights inspector/parity harness. It cannot create an
// admitted chat policy and does not report these trace inputs as live chat.
int main(int argc,char** argv) {
    try {
        if(argc==3 && std::string_view(argv[1])=="inspect") {
            engine::RecurrentWeightsFile file(argv[2]);auto artifact=file.read();
            const auto& a=artifact.descriptor();
            std::cout << "{\"role\":\"weights_only\",\"width\":" << a.recurrent.width
                << ",\"horizon\":" << a.recurrent.horizon << ",\"vocabulary\":" << a.target.config.vocab_size
                << ",\"weight_bytes\":" << artifact.payload().size()
                << ",\"checkpoint_sha256\":\"" << a.recurrent.provenance[0].hex() << "\""
                << ",\"envelope_sha256\":\"" << artifact.envelope_digest().hex() << "\"}\n";
            return 0;
        }
        if(argc!=7 || (std::string_view(argv[1])!="trace-metal" && std::string_view(argv[1])!="trace-cpu")) {
            std::cerr << "usage: local-gguf-speculation-draft inspect <weights.lgweights>\n"
                << "       local-gguf-speculation-draft <trace-metal|trace-cpu> <model.gguf> <weights.lgweights> <trace.lgtrace> <origins> <compact|states>\n";
            return 2;
        }
        unsigned limit=0;std::string_view number(argv[5]);
        auto parsed=std::from_chars(number.begin(),number.end(),limit);
        if(parsed.ec!=std::errc{} || parsed.ptr!=number.end() || limit==0 || limit>1024) {
            throw std::invalid_argument("origin limit must be in [1,1024]");
        }
        const bool dump=std::string_view(argv[6])=="states";
        if(!dump && std::string_view(argv[6])!="compact") throw std::invalid_argument("invalid output mode");
        auto source=std::make_shared<engine::RecurrentWeightsFile>(argv[3]);
        auto artifact=source->read();const auto& a=artifact.descriptor();
        auto trace=engine::load_speculation_trace_file(argv[4]);
        if(trace.identity().model!=a.target.model || trace.identity().resident!=a.target.resident
            || trace.identity().artifact!=a.recurrent.provenance[8]
            || trace.identity().registry!=a.recurrent.provenance[2]) {
            throw std::invalid_argument("weights/trace identity mismatch");
        }
        for(const auto& row:trace.records()) {
            if(row.source.split!=engine::SpeculationTraceSplit::training_development) {
                throw std::invalid_argument("development probe refuses non-development traces");
            }
        }
        auto file=model_format::read_file_backed_gguf_model(argv[2]);
        auto model=engine::load_verified_engine_model_from_gguf(file.view(),file.bytes());
        if(*model.payload_receipt()!=a.target.model || model.gemma2_config()!=a.target.config) {
            throw std::invalid_argument("weights/model identity mismatch");
        }
        auto weights=engine::bind_gemma2_model_weights(model);
        std::vector<float> norm(a.target.config.embedding_length);
        if(weights.output_norm_weight.bytes().size()!=norm.size()*4) throw std::invalid_argument("F32 norm required");
        std::memcpy(norm.data(),weights.output_norm_weight.bytes().data(),norm.size()*4);
        std::optional<engine::Gemma2MetalModelPlan> target;
        std::optional<engine::Gemma2SpeculativePlan> draft;
        if(std::string_view(argv[1])=="trace-metal") {
            engine::ResidentTargetPayloadReceipt resident;
            target.emplace(engine::create_gemma2_metal_model_plan(model,engine::Gemma2OutputHeadPolicy::q8_0,resident));
            draft.emplace(engine::Gemma2SpeculativePlan::create(source,*model.payload_receipt(),resident,*target));
        }
        unsigned count=0;auto records=trace.records();const auto h=a.recurrent.horizon;
        for(std::size_t origin=0;origin+h<records.size() && count<limit;++origin) {
            const auto& root=records[origin];bool valid=true;
            std::array<double,23> draws {};
            for(std::uint32_t i=0;i<=h;++i) {
                const auto& row=records[origin+i];
                valid=valid && row.selection.has_value() && row.target_label_token_id.has_value()
                    && row.target_position==root.target_position+i
                    && row.sequence_id==root.sequence_id && row.turn_index==root.turn_index
                    && row.source.context_generation==root.source.context_generation;
                if(i && valid) {
                    valid=records[origin+i-1].target_label_token_id==row.token_id;
                    draws[i-1]=row.selection->draw.unit_interval_draw;
                }
            }
            if(!valid) continue;
            auto feature=std::find_if(root.features.begin(),root.features.end(),[](const auto& f){
                return f.state.source_stage==engine::FeatureTapStage::final_normalized;
            });
            if(feature==root.features.end() || feature->dtype!=tensor::DType::f16
                || feature->bytes.size()!=std::size_t(a.target.config.embedding_length)*2) {
                throw std::invalid_argument("trace final feature missing or malformed");
            }
            std::vector<float> values(a.target.config.embedding_length);
            for(std::size_t i=0;i<values.size();++i) values[i]=tensor::f16_bits_to_f32(
                static_cast<std::uint16_t>(std::uint16_t(feature->bytes[2*i])|(std::uint16_t(feature->bytes[2*i+1])<<8)));
            std::vector<std::uint32_t> tokens;std::vector<float> states,confidence;
            std::uint32_t status=0,dispatches=0,allocations=0;std::uint64_t ns=0,gpu=0;
            if(draft) {
                auto storage=tensor::TensorStorage::from_bytes(tensor::TensorDescriptor::from(
                    tensor::Shape::from({1,values.size()}),tensor::DType::f32),std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(values.data()),values.size()*4));
                auto input=tensor::copy_tensor_to_backend(storage,runtime::BackendKind::metal);
                engine::TargetStateView provenance {.execution_epoch=count+1U,
                    .source_stage=engine::FeatureTapStage::final_normalized,
                    .coordinate={root.target_position,root.target_position+1},.row_count=1,
                    .feature_width=a.target.config.embedding_length};
                auto result=draft->execute_development(input,provenance,*root.target_label_token_id,
                    std::span<const double>(draws).first(h));
                status=result.status;ns=result.wall_duration_ns;gpu=result.gpu_duration_ns;
                dispatches=result.dispatch_count;allocations=result.transient_buffer_allocations;
                tokens.assign(result.tokens.begin(),result.tokens.begin()+result.count);
                confidence.assign(result.confidence_logits.begin(),result.confidence_logits.begin()+result.count);
                if(dump) states=draft->diagnostic_target_states();
            } else {
                auto result=engine::recurrent_cpu_golden(artifact,weights.token_embedding_weight,norm,values,
                    *root.target_label_token_id,std::span<const double>(draws).first(h));
                tokens=std::move(result.tokens);confidence=std::move(result.confidence_logits);
                if(dump) states=std::move(result.target_states);
            }
            std::cout << std::setprecision(9) << "{\"origin\":" << origin << ",\"status\":" << status
                << ",\"wall_ns\":" << ns << ",\"gpu_ns\":" << gpu << ",\"dispatches\":" << dispatches
                << ",\"transient_buffer_allocations\":" << allocations
                << ",\"owned_bytes\":" << (draft?draft->owned_bytes():0) << ",\"tokens\":[";
            for(std::size_t i=0;i<tokens.size();++i) std::cout << (i?",":"") << tokens[i];
            std::cout << "],\"confidence_logits\":[";
            for(std::size_t i=0;i<confidence.size();++i) std::cout << (i?",":"") << confidence[i];
            std::cout << "]";
            if(dump) {
                std::cout << ",\"states\":[";
                for(std::size_t i=0;i<states.size();++i) std::cout << (i?",":"") << states[i];
                std::cout << "]";
            }
            std::cout << "}\n" << std::flush;++count;
        }
        if(count<limit) throw std::invalid_argument("trace has fewer eligible origins than requested");
        return 0;
    } catch(const std::exception& e) {std::cerr << "speculation draft: " << e.what() << '\n';return 1;}
}
