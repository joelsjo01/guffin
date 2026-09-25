#pragma once

#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/ops/reduction.hpp"
#include "local_gguf_engine/engine/gemma2_metal_plan.hpp"
#include "local_gguf_engine/engine/target_verifier.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <string_view>

namespace local_gguf_engine::tensor::backends::metal {

enum class RecurrentStage : std::uint32_t { linear, norm, embedding, join, update, sample, verifier_requests, sample_lossless };
void prepare_recurrent_pipelines();

namespace draft_detail {
struct TopKResult {
    bool final_buffer_is_a = true;
    std::uint32_t dispatch_count = 0U;
};
}

struct MetalCommandStats {
    std::uint32_t command_buffer_count = 0;
    std::uint32_t commit_count = 0;
    std::uint32_t synchronous_wait_count = 0;
    std::uint32_t compute_encoder_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t blit_encoder_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t transient_buffer_allocation_count = 0;
    std::uint64_t gpu_duration_ns = 0;
    std::uint32_t q4_decode_projection_count = 0;
    std::uint32_t q8_decode_projection_count = 0;
    std::uint32_t q8_width_two_projection_count = 0;
    std::uint32_t q4_small_batch_projection_count = 0;
    std::uint32_t q4_width_two_projection_count = 0;
    std::uint32_t q4_exact_16_projection_count = 0;
    std::uint32_t q4_generic_tiled_projection_count = 0;
    std::uint32_t q6_decode_projection_count = 0;
    std::uint32_t q6_small_batch_projection_count = 0;
    std::uint32_t q6_width_two_projection_count = 0;
    std::uint32_t q6_exact_16_projection_count = 0;
    std::uint32_t q6_generic_tiled_projection_count = 0;
    std::uint32_t f16_projection_count = 0;
    std::uint32_t decode_attention_dispatch_count = 0;
    std::uint32_t decode_attention_partition_count = 0;
    std::uint64_t decode_attention_token_count = 0;
};

[[nodiscard]] std::uint32_t gemma2_decode_attention_partition_count(
    std::uint32_t attention_token_count
) noexcept;

// Owns one command buffer for a complete resident Gemma execution. Encoding operations
// take explicit destinations so this context never allocates activation or
// constant buffers in the token hot path.
class MetalCommandContext final {
public:
    explicit MetalCommandContext(std::string_view label, bool concurrent_compute = false);
    // Reuses engine-owned host storage. Metal command buffers/encoders remain
    // single-use driver objects; this is not a claim of zero driver allocation.
    void restart();
    ~MetalCommandContext();

    MetalCommandContext(const MetalCommandContext&) = delete;
    MetalCommandContext& operator=(const MetalCommandContext&) = delete;
    MetalCommandContext(MetalCommandContext&&) noexcept;
    MetalCommandContext& operator=(MetalCommandContext&&) noexcept;

    void write_u32(const BackendBuffer& destination, std::uint32_t value);
    void write_u32s(
        const BackendBuffer& destination,
        std::uint32_t element_offset,
        std::span<const std::uint32_t> values
    );
    void encode_embedding(
        const BackendBuffer& token_ids,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        float scale,
        std::uint32_t token_offset = 0,
        std::uint32_t token_count = 0
    );
    void encode_matmul(
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        std::uint32_t row_count = 0,
        std::uint64_t output_byte_offset = 0
    );
    void encode_verifier_matmul(
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        std::uint32_t row_count
    );
    void encode_target_head_q8_rows(
        const BackendBuffer& input,
        std::uint32_t input_row_start,
        const BackendBuffer& weight,
        const BackendBuffer& logits_tile,
        std::uint32_t row_count
    );
    void encode_feature_tap_copy(
        const BackendBuffer& input,
        std::uint32_t input_row_start,
        const BackendBuffer& fused_output,
        std::uint32_t output_column_start,
        std::uint32_t row_count
    );
    void encode_feature_tap_store_f16(
        const BackendBuffer& input,
        std::uint32_t input_row_start,
        const BackendBuffer& output,
        std::uint32_t output_column_start,
        std::uint32_t row_count
    );
    void encode_feature_scatter(
        const BackendBuffer& source_f32,
        const BackendBuffer& source_rows_u32,
        const BackendBuffer& destination_rows_u32,
        const BackendBuffer& destination,
        const BackendBuffer& status_u32,
        std::uint32_t row_count,
        std::uint32_t width
    );
    void encode_f16_row_scatter(
        const BackendBuffer& source,
        const BackendBuffer& source_rows_u32,
        const BackendBuffer& destination_rows_u32,
        const BackendBuffer& destination,
        const BackendBuffer& status_u32,
        std::uint32_t row_count
    );
    void encode_rms_norm(
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& output,
        float epsilon,
        std::uint32_t row_count = 0,
        std::uint32_t input_row_start = 0
    );
    void encode_rms_norm_add(
        const BackendBuffer& input,
        const BackendBuffer& weight,
        const BackendBuffer& residual,
        const BackendBuffer& output,
        float epsilon,
        std::uint32_t row_count = 0
    );
    void encode_rope(
        const BackendBuffer& input,
        const BackendBuffer& output,
        std::uint32_t position_start,
        std::uint32_t head_count,
        std::uint32_t head_dimension,
        std::uint32_t rope_dimension_count,
        float frequency_base,
        std::uint32_t row_count = 0,
        std::uint64_t output_byte_offset = 0
    );
    void encode_gemma2_kv_cache_store(
        const BackendBuffer& raw_key,
        const BackendBuffer& raw_value,
        const BackendBuffer& cache_key,
        const BackendBuffer& cache_value,
        std::uint32_t position_start,
        std::uint32_t head_count,
        std::uint32_t head_dimension,
        std::uint32_t rope_dimension_count,
        float frequency_base,
        std::uint32_t cache_row_mask,
        std::uint32_t token_count = 0,
        std::uint32_t source_row_start = 0
    );
    void encode_gemma2_kv_cache_copy(
        const BackendBuffer& source_key,
        const BackendBuffer& source_value,
        const BackendBuffer& destination_key,
        const BackendBuffer& destination_value,
        std::uint32_t position_start,
        std::uint32_t token_count,
        std::uint32_t source_row_mask,
        std::uint32_t destination_row_mask
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
        float logit_softcap,
        std::uint32_t cache_row_mask,
        std::uint32_t query_row_count = 0
    );
    void encode_wrapped_prefill_attention_scores(
        const BackendBuffer& query,
        const BackendBuffer& key_cache,
        const BackendBuffer& current_chunk_key,
        const BackendBuffer& output,
        std::uint32_t query_position_start,
        std::uint32_t head_count,
        std::uint32_t kv_head_count,
        std::uint32_t key_length,
        std::uint32_t attention_token_count,
        float scale,
        float logit_softcap,
        std::uint32_t cache_row_mask,
        std::uint32_t query_row_count = 0
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
        std::uint32_t attention_token_count,
        std::uint32_t cache_row_mask,
        std::uint32_t query_row_count = 0
    );
    void encode_wrapped_prefill_attention_values(
        const BackendBuffer& probabilities,
        const BackendBuffer& value_cache,
        const BackendBuffer& current_chunk_value,
        const BackendBuffer& output,
        std::uint32_t query_position_start,
        std::uint32_t head_count,
        std::uint32_t kv_head_count,
        std::uint32_t value_length,
        std::uint32_t attention_token_count,
        std::uint32_t cache_row_mask,
        std::uint32_t query_row_count = 0
    );
    void encode_gemma2_decode_attention(
        const BackendBuffer& query,
        const BackendBuffer& key_cache,
        const BackendBuffer& value_cache,
        const BackendBuffer& output,
        const BackendBuffer& partial_output,
        const BackendBuffer& partial_maxima,
        const BackendBuffer& partial_denominators,
        std::uint32_t cache_position,
        std::uint32_t attention_token_count,
        float scale,
        float logit_softcap,
        std::uint32_t cache_row_mask,
        std::uint32_t query_row = 0,
        std::uint32_t output_row = 0
    );
    void encode_verifier_input_validation(
        const BackendBuffer& actual_token_ids,
        std::uint64_t actual_byte_offset,
        const BackendBuffer& expected_token_ids,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size
    );
    void encode_verifier_selection_requests(
        const BackendBuffer& raw_requests,
        std::uint64_t raw_byte_offset,
        const BackendBuffer& resolved_requests,
        const BackendBuffer& histories,
        const BackendBuffer& status,
        std::uint32_t row_count,
        const engine::TokenSamplerOptions& options,
        std::uint32_t vocabulary_size
    );
    void encode_verifier_rope(
        const BackendBuffer& input,
        const BackendBuffer& positions,
        const BackendBuffer& output,
        std::uint32_t row_count,
        std::uint32_t head_count,
        std::uint32_t head_dimension,
        std::uint32_t rope_dimension_count,
        float frequency_base
    );
    void encode_verifier_kv_store(
        const BackendBuffer& raw_key,
        const BackendBuffer& raw_value,
        const BackendBuffer& positions,
        const BackendBuffer& private_key,
        const BackendBuffer& private_value,
        std::uint32_t row_count,
        std::uint32_t head_count,
        std::uint32_t head_dimension,
        std::uint32_t rope_dimension_count,
        float frequency_base
    );
    void encode_verifier_attention(
        const BackendBuffer& query,
        const BackendBuffer& resident_key,
        const BackendBuffer& resident_value,
        const BackendBuffer& private_key,
        const BackendBuffer& private_value,
        const BackendBuffer& schedules,
        std::uint64_t schedule_byte_offset,
        const BackendBuffer& ancestors,
        const BackendBuffer& output,
        const BackendBuffer& partial_output,
        const BackendBuffer& partial_maxima,
        const BackendBuffer& partial_denominators,
        std::uint32_t row_count,
        std::uint32_t cache_row_mask,
        std::uint32_t query_head_count,
        std::uint32_t kv_head_count,
        std::uint32_t key_dimension,
        std::uint32_t value_dimension,
        float scale,
        float logit_softcap
    );
    void encode_verifier_kv_scatter(
        const BackendBuffer& private_key,
        const BackendBuffer& private_value,
        const BackendBuffer& selected_nodes,
        const BackendBuffer& resident_key,
        const BackendBuffer& resident_value,
        std::uint32_t destination_position,
        std::uint32_t row_count,
        std::uint32_t row_width,
        std::uint32_t destination_row_mask
    );
    void encode_add(
        const BackendBuffer& left,
        const BackendBuffer& right,
        const BackendBuffer& output,
        std::uint64_t element_count = 0
    );
    void encode_silu_multiply(
        const BackendBuffer& gate,
        const BackendBuffer& up,
        const BackendBuffer& output,
        std::uint64_t element_count = 0
    );
    void encode_gelu_tanh_multiply(
        const BackendBuffer& gate,
        const BackendBuffer& up,
        const BackendBuffer& output,
        std::uint64_t element_count = 0
    );
    void encode_softcap(
        const BackendBuffer& input,
        const BackendBuffer& output,
        float softcap,
        std::uint64_t element_count = 0
    );
    [[nodiscard]] std::uint32_t encode_token_candidates(
        const BackendBuffer& logits,
        const engine::Gemma2MetalSelectionWorkspace& workspace,
        const engine::TokenSelectionRequest& request,
        float softcap
    );
    [[nodiscard]] std::uint32_t encode_token_candidates_width_two(
        const BackendBuffer& logits,
        const engine::Gemma2MetalSelectionWorkspace& workspace,
        const std::array<engine::TokenSelectionRequest, 2>& requests,
        float softcap
    );
    [[nodiscard]] std::uint32_t encode_target_token_candidates(
        const BackendBuffer& logits_tile,
        const engine::TargetSelectionWorkspace& workspace,
        std::uint32_t first_request_row,
        std::uint32_t row_count,
        float softcap,
        bool greedy_without_history
    );
    [[nodiscard]] draft_detail::TopKResult encode_draft_stable_top_k(
        const BackendBuffer& scores,
        const BackendBuffer& runs_a,
        const BackendBuffer& runs_b,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size,
        std::uint32_t retain_count
    );
    void encode_draft_candidate_union(
        const BackendBuffer& selected,
        const BackendBuffer& transition_counts,
        const BackendBuffer& transition_tokens,
        const BackendBuffer& history_ids,
        const BackendBuffer& history_counts,
        const BackendBuffer& required_ids,
        const BackendBuffer& required_counts,
        const BackendBuffer& retrieval_ids,
        const BackendBuffer& retrieval_counts,
        const BackendBuffer& candidate_bits,
        const BackendBuffer& tile_counts,
        const BackendBuffer& tile_offsets,
        const BackendBuffer& candidate_ids,
        const BackendBuffer& candidate_counts,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size,
        std::uint32_t selector_count,
        std::uint32_t transition_width,
        bool full_vocabulary
    );
    void encode_draft_indexed_dot(
        const BackendBuffer& states,
        const BackendBuffer& weights,
        const BackendBuffer& candidate_ids,
        const BackendBuffer& candidate_counts,
        const BackendBuffer& output,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size,
        std::uint32_t candidate_dispatch_width
    );
    void encode_draft_score_correction(
        const BackendBuffer& scores,
        const BackendBuffer& corrections,
        const BackendBuffer& candidate_counts,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t candidate_stride,
        std::uint32_t candidate_dispatch_width,
        float softcap
    );
    void encode_draft_proposals(
        const BackendBuffer& candidate_ids,
        const BackendBuffer& candidate_counts,
        const BackendBuffer& scores,
        const BackendBuffer& exp_lut,
        const BackendBuffer& draws,
        const BackendBuffer& weights,
        const BackendBuffer& results,
        std::uint32_t row_count,
        std::uint32_t candidate_stride,
        float temperature
    );
    void encode_corrected_logsumexp(
        const BackendBuffer& logits,
        const BackendBuffer& logsumexp,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size,
        float temperature
    );
    void encode_corrected_rank_finalize(
        const BackendBuffer& sorted,
        const BackendBuffer& logsumexp,
        const BackendBuffer& rank_entries,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size,
        std::uint32_t rank_width,
        std::uint32_t output_row_start,
        float temperature
    );
    void encode_arm_s_transition_scores(
        const BackendBuffer& logits,
        const BackendBuffer& logsumexp,
        const BackendBuffer& child_token_ids,
        std::uint32_t child_token_offset,
        const BackendBuffer& probabilities,
        const BackendBuffer& status,
        std::uint32_t row_count,
        std::uint32_t vocabulary_size,
        std::uint32_t output_row_start
    );
    void encode_memory_barrier();
    void encode_recurrent_stage(
        RecurrentStage stage, std::span<const BackendBuffer* const> buffers,
        const std::array<std::uint32_t,4>& dimensions,
        const std::array<float,4>& scalars = {}
    );
    void push_debug_group(std::string_view label);
    void pop_debug_group();

    [[nodiscard]] MetalCommandStats commit();
    [[nodiscard]] MetalCommandStats wait_until_completed();
    [[nodiscard]] MetalCommandStats completion_stats() const;
    [[nodiscard]] MetalCommandStats commit_and_wait();
    [[nodiscard]] engine::CompactTokenCandidates read_token_candidates(
        const engine::Gemma2MetalSelectionWorkspace& workspace
    ) const;
    [[nodiscard]] std::array<engine::CompactTokenCandidates, 2>
    read_token_candidates_width_two(const engine::Gemma2MetalSelectionWorkspace& workspace) const;
    [[nodiscard]] std::uint32_t read_verifier_status(const BackendBuffer& status) const;
    void read_target_token_candidates(
        const engine::TargetSelectionWorkspace& workspace,
        std::uint32_t first_row,
        std::span<engine::CompactTokenCandidates> output
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Deterministically initializes shared Metal storage before a diagnostic
// command. It is intentionally outside command-buffer accounting.
void fill_shared_metal_buffer_u32(
    const BackendBuffer& buffer,
    std::uint32_t pattern
);
void read_shared_metal_buffer_bytes(
    const BackendBuffer& buffer,
    std::uint64_t byte_offset,
    std::span<std::uint8_t> destination
);

}  // namespace local_gguf_engine::tensor::backends::metal
