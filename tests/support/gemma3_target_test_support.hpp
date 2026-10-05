#pragma once

#include "test_support.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <string>
#include <vector>

namespace local_gguf_engine::tests {

using tensor::DType;
using engine::EngineModel;
using engine::load_engine_model_from_gguf;

// Small widths retain complete six-layer architecture and native K-quant
// layouts. No checkpoint is required to validate ownership and allocation.
struct Gemma3TargetFixture {
    std::vector<tensor::TensorStorage> tensors;
    model_format::GGUFModelView view;

    Gemma3TargetFixture() {
        view.directory.metadata = tests::metadata_section({
            tests::metadata_string("general.architecture", "gemma3"),
            tests::metadata_string("general.name", "Gemma 3 plan fixture"),
            tests::metadata_u32("gemma3.context_length", 131072U),
            tests::metadata_u32("gemma3.embedding_length", 256U),
            tests::metadata_u32("gemma3.block_count", 6U),
            tests::metadata_u32("gemma3.feed_forward_length", 256U),
            tests::metadata_u32("gemma3.attention.head_count", 2U),
            tests::metadata_u32("gemma3.attention.head_count_kv", 1U),
            tests::metadata_u32("gemma3.attention.key_length", 256U),
            tests::metadata_u32("gemma3.attention.value_length", 256U),
            tests::metadata_u32("gemma3.attention.sliding_window", 8U),
            tests::metadata_f32("gemma3.attention.layer_norm_rms_epsilon", 1.0e-6F),
            tests::metadata_f32("gemma3.rope.freq_base", 1000000.0F),
            tests::metadata_string("gemma3.rope.scaling.type", "linear"),
            tests::metadata_f32("gemma3.rope.scaling.factor", 8.0F),
        });
        tensors.reserve(80U);
        add("token_embd.weight", {256U, 32U}, DType::q6_k);
        add("output_norm.weight", {256U}, DType::f32);
        for (std::uint32_t i = 0U; i < 6U; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            for (const auto* suffix : {"attn_norm.weight", "post_attention_norm.weight",
                 "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight"}) {
                add(prefix + suffix, {256U}, DType::f32);
            }
            add(prefix + "attn_q.weight", {256U, 512U}, DType::q4_k);
            add(prefix + "attn_k.weight", {256U, 256U}, DType::q4_k);
            add(prefix + "attn_v.weight", {256U, 256U}, DType::q6_k);
            add(prefix + "attn_output.weight", {512U, 256U}, DType::q4_k);
            add(prefix + "ffn_gate.weight", {256U, 256U}, DType::q4_k);
            add(prefix + "ffn_up.weight", {256U, 256U}, DType::q4_k);
            add(prefix + "ffn_down.weight", {256U, 256U}, DType::q6_k);
        }
    }

    void add(std::string name, std::initializer_list<std::uint64_t> dimensions, DType dtype) {
        const auto shape = tensor::Shape::from(dimensions);
        const auto descriptor = dtype == DType::f32 ? tensor::TensorDescriptor::from(shape, dtype)
            : tensor::TensorDescriptor::from_quantized_gguf_weight_matrix(shape, dtype);
        tensors.push_back(tensor::TensorStorage::allocate(descriptor));
        const auto type_id = dtype == DType::q6_k ? 14U : dtype == DType::q4_k ? 12U : 0U;
        view.tensor_payloads.push_back({
            .tensor = {.name = std::move(name), .dimensions = dimensions, .type_id = type_id},
            .bytes = tensors.back().bytes(),
        });
    }

    void metadata(model_format::GGUFMetadataEntry replacement) {
        for (auto& entry : view.directory.metadata.entries) {
            if (entry.key == replacement.key) {
                entry = std::move(replacement);
                return;
            }
        }
        view.directory.metadata.entries.push_back(std::move(replacement));
    }

    EngineModel model() const { return load_engine_model_from_gguf(view); }
};

}  // namespace local_gguf_engine::tests
