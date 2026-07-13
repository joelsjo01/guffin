#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace local_gguf_engine::tensor::backends::metal {

struct MetalCommandStats {
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint64_t gpu_duration_ns = 0;
};

// Owns one command buffer for a complete resident decode. Encoding operations
// take explicit destinations so this context never allocates activation or
// constant buffers in the token hot path.
class MetalCommandContext final {
public:
    explicit MetalCommandContext(std::string_view label);
    ~MetalCommandContext();

    MetalCommandContext(const MetalCommandContext&) = delete;
    MetalCommandContext& operator=(const MetalCommandContext&) = delete;
    MetalCommandContext(MetalCommandContext&&) noexcept;
    MetalCommandContext& operator=(MetalCommandContext&&) noexcept;

    void write_u32(const BackendBuffer& destination, std::uint32_t value);
    void encode_embedding(
        const BackendBuffer& token_ids,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        float scale
    );
    void encode_matmul(
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& output
    );
    void encode_rms_norm(
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        float epsilon
    );
    void encode_rope(
        const BackendBuffer& input,
        const BackendBuffer& output,
        std::uint32_t position_start,
        std::uint32_t head_count,
        std::uint32_t head_dimension,
        std::uint32_t rope_dimension_count,
        float frequency_base
    );
    void encode_kv_cache_write(
        const BackendBuffer& key,
        const BackendBuffer& value,
        const BackendBuffer& cache_key,
        const BackendBuffer& cache_value,
        std::uint32_t cache_position
    );
    void encode_attention_scores(
        const BackendBuffer& query,
        const BackendBuffer& key_cache,
        const BackendBuffer& output,
        std::uint32_t query_position_start,
        std::uint32_t head_count,
        std::uint32_t kv_head_count,
        std::uint32_t key_length,
        std::uint32_t attention_token_count,
        float scale,
        float logit_softcap
    );
    void encode_softmax(
        const BackendBuffer& input,
        const BackendBuffer& output,
        std::uint32_t rows,
        std::uint32_t columns
    );
    void encode_attention_values(
        const BackendBuffer& probabilities,
        const BackendBuffer& value_cache,
        const BackendBuffer& output,
        std::uint32_t query_position_start,
        std::uint32_t head_count,
        std::uint32_t kv_head_count,
        std::uint32_t value_length,
        std::uint32_t attention_token_count
    );
    void encode_add(
        const BackendBuffer& left,
        const BackendBuffer& right,
        const BackendBuffer& output
    );
    void encode_silu_multiply(
        const BackendBuffer& gate,
        const BackendBuffer& up,
        const BackendBuffer& output
    );
    void encode_softcap(
        const BackendBuffer& input,
        const BackendBuffer& output,
        float softcap
    );
    void encode_top_logit(
        const BackendBuffer& logits,
        const BackendBuffer& result
    );

    [[nodiscard]] MetalCommandStats commit_and_wait();
    [[nodiscard]] ops::TopLogit read_top_logit(const BackendBuffer& result) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace local_gguf_engine::tensor::backends::metal
