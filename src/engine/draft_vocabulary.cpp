#include "local_gguf_engine/engine/draft_vocabulary.hpp"

#include "local_gguf_engine/tensor/f16.hpp"
#include "local_gguf_engine/tensor/quantization.hpp"
#include "../tensor/backends/metal/metal_command_context.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>

namespace local_gguf_engine::engine {
namespace {

using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;

[[nodiscard]] bool candidate_better(
    const DraftVocabularyCandidate& left,
    const DraftVocabularyCandidate& right
) noexcept {
    return left.score > right.score
        || (left.score == right.score && left.token_id < right.token_id);
}

void require_extent(std::size_t actual, std::uint64_t required, const char* name) {
    if (required > std::numeric_limits<std::size_t>::max()
        || actual < static_cast<std::size_t>(required)) {
        throw std::invalid_argument(std::string(name) + " extent is too small");
    }
}

[[nodiscard]] std::uint16_t read_u16(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

[[nodiscard]] float f16_column_value(
    std::span<const std::uint8_t> bytes,
    std::uint32_t shared,
    std::uint32_t token,
    std::uint32_t index
) {
    const auto offset = (static_cast<std::uint64_t>(token) * shared + index) * 2U;
    return tensor::f16_bits_to_f32(read_u16(bytes.data() + offset));
}

[[nodiscard]] float q8_column_value(
    std::span<const std::uint8_t> bytes,
    std::uint32_t shared,
    std::uint32_t token,
    std::uint32_t index
) {
    const auto blocks = shared / 32U;
    const auto block_offset = (static_cast<std::uint64_t>(token) * blocks
        + index / 32U) * 34U;
    const auto scale = tensor::f16_bits_to_f32(read_u16(bytes.data() + block_offset));
    const auto quantized = std::bit_cast<std::int8_t>(bytes[block_offset + 2U + index % 32U]);
    return scale * static_cast<float>(quantized);
}

[[nodiscard]] float q4_column_value(
    std::span<const std::uint8_t> bytes,
    std::uint32_t shared,
    std::uint32_t token,
    std::uint32_t index
) {
    const auto blocks = shared / 256U;
    const auto block_offset = (static_cast<std::uint64_t>(token) * blocks
        + index / 256U) * 144U;
    const auto block = tensor::Q4_KBlockView(
        read_u16(bytes.data() + block_offset),
        read_u16(bytes.data() + block_offset + 2U),
        bytes.subspan(static_cast<std::size_t>(block_offset + 4U), 12U),
        bytes.subspan(static_cast<std::size_t>(block_offset + 16U), 128U)
    );
    return block.dequantized_value(index % 256U);
}

[[nodiscard]] float q6_column_value(
    std::span<const std::uint8_t> bytes,
    std::uint32_t shared,
    std::uint32_t token,
    std::uint32_t index
) {
    const auto blocks = shared / 256U;
    const auto block_offset = (static_cast<std::uint64_t>(token) * blocks
        + index / 256U) * 210U;
    const auto block = tensor::Q6_KBlockView(
        bytes.subspan(static_cast<std::size_t>(block_offset), 128U),
        bytes.subspan(static_cast<std::size_t>(block_offset + 128U), 64U),
        bytes.subspan(static_cast<std::size_t>(block_offset + 192U), 16U),
        read_u16(bytes.data() + block_offset + 208U)
    );
    return block.dequantized_value(index % 256U);
}

[[nodiscard]] float packed_column_value(
    const tensor::TensorView& weight,
    std::uint32_t shared,
    std::uint32_t token,
    std::uint32_t index
) {
    switch (weight.descriptor().dtype) {
        case tensor::DType::f16:
            return f16_column_value(weight.bytes(), shared, token, index);
        case tensor::DType::q8_0:
            return q8_column_value(weight.bytes(), shared, token, index);
        case tensor::DType::q4_k:
            return q4_column_value(weight.bytes(), shared, token, index);
        case tensor::DType::q6_k:
            return q6_column_value(weight.bytes(), shared, token, index);
        case tensor::DType::f32:
            break;
    }
    throw std::invalid_argument("indexed vocabulary table dtype is unsupported");
}

[[nodiscard]] std::uint64_t multiply_checked(
    std::uint64_t left,
    std::uint64_t right,
    const char* name
) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(name) + " overflows");
    }
    return left * right;
}

[[nodiscard]] DType artifact_dtype(DraftArtifactElementType type) {
    switch (type) {
        case DraftArtifactElementType::f16: return DType::f16;
        case DraftArtifactElementType::q8_0: return DType::q8_0;
        case DraftArtifactElementType::q6_k: return DType::q6_k;
        case DraftArtifactElementType::q4_k: return DType::q4_k;
        case DraftArtifactElementType::u32: return DType::f32;
    }
    throw std::invalid_argument("draft artifact tensor element type is invalid");
}

[[nodiscard]] TensorDescriptor artifact_tensor_descriptor(
    const DraftArtifactTensorSpec& spec
) {
    const auto shape = Shape::from({spec.dimension_0, spec.dimension_1});
    const auto dtype = artifact_dtype(spec.element_type);
    return spec.element_type == DraftArtifactElementType::q8_0
            || spec.element_type == DraftArtifactElementType::q6_k
            || spec.element_type == DraftArtifactElementType::q4_k
        ? TensorDescriptor::from_quantized_gguf_weight_matrix(shape, dtype)
        : TensorDescriptor::from(shape, dtype);
}

[[nodiscard]] tensor::BackendBuffer allocate_f32_buffer(
    std::uint64_t rows,
    std::uint64_t columns
) {
    return tensor::allocate_backend_buffer(
        TensorDescriptor::from(Shape::from({rows, columns}), DType::f32),
        runtime::BackendKind::metal
    );
}

[[nodiscard]] tensor::BackendBuffer upload_artifact_tensor(
    const AuthenticatedSpeculativeArtifact& artifact,
    const DraftArtifactTensorSpec& spec
) {
    if (spec.storage != DraftArtifactTensorStorage::payload) {
        throw std::invalid_argument("draft artifact tensor is not payload-owned");
    }
    const auto bytes = artifact.payload().subspan(
        static_cast<std::size_t>(spec.payload_offset),
        static_cast<std::size_t>(spec.payload_byte_count)
    );
    const auto storage = tensor::TensorStorage::from_bytes(
        artifact_tensor_descriptor(spec), bytes
    );
    return tensor::copy_tensor_to_backend(storage, runtime::BackendKind::metal);
}

void validate_selector_padding(
    const AuthenticatedSpeculativeArtifact& artifact,
    const DraftVocabularyArtifactSpec& draft
) {
    if (draft.selector_storage_width == draft.selector_width) return;
    const auto validate_matrix = [&](const DraftArtifactTensorSpec& tensor_spec,
                                     std::uint32_t logical_rows,
                                     const char* name) {
        if (tensor_spec.storage != DraftArtifactTensorStorage::payload) return;
        const auto bytes = artifact.payload().subspan(
            static_cast<std::size_t>(tensor_spec.payload_offset),
            static_cast<std::size_t>(tensor_spec.payload_byte_count)
        );
        const auto storage = tensor::TensorStorage::from_bytes(
            artifact_tensor_descriptor(tensor_spec), bytes
        );
        const auto view = storage.view();
        const auto columns = static_cast<std::uint32_t>(view.descriptor().shape.dimension(1));
        for (std::uint32_t column = 0U; column < columns; ++column) {
            for (std::uint32_t row = logical_rows;
                 row < draft.selector_storage_width; ++row) {
                const auto value = packed_column_value(
                    view, draft.selector_storage_width, column, row
                );
                if (!std::isfinite(value) || value != 0.0F) {
                    throw SpeculativeArtifactError(
                        SpeculativeArtifactErrorCode::invalid_descriptor,
                        std::string(name) + " has nonzero padded selector weights"
                    );
                }
            }
        }
    };
    const auto& projection = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::selector_projection
    )];
    if (projection.storage == DraftArtifactTensorStorage::payload) {
        // The projection's columns are selector coordinates, so its padded
        // output columns are contiguous F16 columns of semantic_width values.
        const auto bytes = artifact.payload().subspan(
            static_cast<std::size_t>(projection.payload_offset),
            static_cast<std::size_t>(projection.payload_byte_count)
        );
        for (std::uint32_t column = draft.selector_width;
             column < draft.selector_storage_width; ++column) {
            for (std::uint32_t row = 0U; row < draft.semantic_width; ++row) {
                const auto offset = (static_cast<std::uint64_t>(column)
                    * draft.semantic_width + row) * 2U;
                if (tensor::f16_bits_to_f32(read_u16(bytes.data() + offset)) != 0.0F) {
                    throw SpeculativeArtifactError(
                        SpeculativeArtifactErrorCode::invalid_descriptor,
                        "selector projection has nonzero padded output weights"
                    );
                }
            }
        }
    }
    validate_matrix(
        draft.tensors[draft_artifact_tensor_index(DraftArtifactTensorRole::selector_head)],
        draft.selector_width, "selector head"
    );
}

}  // namespace

bool is_supported_draft_vocabulary_state_batch(std::uint32_t row_count) noexcept {
    return std::find(
        draft_vocabulary_state_batches.begin(),
        draft_vocabulary_state_batches.end(), row_count
    ) != draft_vocabulary_state_batches.end();
}

bool is_supported_draft_vocabulary_candidate_k(
    std::uint32_t candidate_k,
    std::uint32_t vocabulary_size
) noexcept {
    if (candidate_k == vocabulary_size && vocabulary_size != 0U) return true;
    return candidate_k >= 1'024U && candidate_k <= 131'072U
        && std::has_single_bit(candidate_k) && candidate_k <= vocabulary_size;
}

std::vector<DraftVocabularyCandidate> stable_top_k_cpu(
    std::span<const float> scores,
    std::uint32_t row_count,
    std::uint32_t vocabulary_size,
    std::uint32_t candidate_k
) {
    if (row_count == 0U || vocabulary_size == 0U || candidate_k == 0U
        || candidate_k > vocabulary_size) {
        throw std::invalid_argument("stable top-k dimensions are invalid");
    }
    require_extent(scores.size(), static_cast<std::uint64_t>(row_count) * vocabulary_size,
        "stable top-k scores");
    std::vector<DraftVocabularyCandidate> output(
        static_cast<std::size_t>(row_count) * candidate_k
    );
    std::vector<DraftVocabularyCandidate> row(vocabulary_size);
    for (std::uint32_t row_index = 0U; row_index < row_count; ++row_index) {
        for (std::uint32_t token = 0U; token < vocabulary_size; ++token) {
            const auto score = scores[static_cast<std::size_t>(row_index) * vocabulary_size + token];
            if (!std::isfinite(score)) {
                throw std::invalid_argument("stable top-k score must be finite");
            }
            row[token] = {token, score};
        }
        std::partial_sort(row.begin(), row.begin() + candidate_k, row.end(), candidate_better);
        std::copy_n(row.begin(), candidate_k,
            output.begin() + static_cast<std::ptrdiff_t>(row_index * candidate_k));
    }
    return output;
}

std::vector<std::uint32_t> candidate_union_cpu(
    std::span<const DraftVocabularyCandidate> selector_candidates,
    std::uint32_t row_count,
    std::uint32_t selector_stride,
    std::span<const std::uint32_t> selector_counts,
    std::span<const std::uint32_t> history_ids,
    std::span<const std::uint32_t> history_counts,
    std::span<const std::uint32_t> required_ids,
    std::span<const std::uint32_t> required_counts,
    std::span<const std::uint32_t> retrieval_ids,
    std::span<const std::uint32_t> retrieval_counts,
    std::span<const std::uint32_t> transition_counts,
    std::span<const std::uint32_t> transition_tokens,
    std::uint32_t transition_width,
    std::uint32_t vocabulary_size,
    std::span<std::uint32_t> output_counts
) {
    if (row_count == 0U || selector_stride == 0U || vocabulary_size == 0U) {
        throw std::invalid_argument("candidate union dimensions are invalid");
    }
    require_extent(selector_candidates.size(),
        static_cast<std::uint64_t>(row_count) * selector_stride, "selector candidates");
    require_extent(selector_counts.size(), row_count, "selector counts");
    require_extent(history_counts.size(), row_count, "history counts");
    require_extent(required_counts.size(), row_count, "required counts");
    require_extent(retrieval_counts.size(), row_count, "retrieval counts");
    require_extent(history_ids.size(),
        static_cast<std::uint64_t>(row_count) * draft_vocabulary_auxiliary_width,
        "history ids");
    require_extent(required_ids.size(),
        static_cast<std::uint64_t>(row_count) * draft_vocabulary_auxiliary_width,
        "required ids");
    require_extent(retrieval_ids.size(),
        static_cast<std::uint64_t>(row_count) * draft_vocabulary_auxiliary_width,
        "retrieval ids");
    require_extent(output_counts.size(), row_count, "candidate output counts");
    if (transition_width != 0U) {
        require_extent(transition_counts.size(), vocabulary_size, "transition counts");
        require_extent(transition_tokens.size(),
            static_cast<std::uint64_t>(vocabulary_size) * transition_width,
            "transition tokens");
    }
    std::vector<std::uint32_t> output(
        static_cast<std::size_t>(row_count) * vocabulary_size
    );
    std::vector<bool> present(vocabulary_size);
    const auto mark = [&](std::vector<bool>& bits, std::uint32_t token) {
        if (token >= vocabulary_size) {
            throw std::invalid_argument("candidate union token is outside vocabulary");
        }
        bits[token] = true;
    };
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        std::fill(present.begin(), present.end(), false);
        if (selector_counts[row] > selector_stride
            || history_counts[row] > draft_vocabulary_auxiliary_width
            || required_counts[row] > draft_vocabulary_auxiliary_width
            || retrieval_counts[row] > draft_vocabulary_auxiliary_width) {
            throw std::invalid_argument("candidate union source count exceeds capacity");
        }
        for (std::uint32_t rank = 0U; rank < selector_counts[row]; ++rank) {
            const auto token = selector_candidates[
                static_cast<std::size_t>(row) * selector_stride + rank
            ].token_id;
            mark(present, token);
            if (transition_width != 0U) {
                if (transition_counts[token] > transition_width) {
                    throw std::invalid_argument("transition count exceeds artifact width");
                }
                for (std::uint32_t edge = 0U; edge < transition_counts[token]; ++edge) {
                    mark(present, transition_tokens[
                        static_cast<std::size_t>(token) * transition_width + edge
                    ]);
                }
            }
        }
        const auto mark_aux = [&](std::span<const std::uint32_t> ids,
                                  std::uint32_t count) {
            const auto base = static_cast<std::size_t>(row)
                * draft_vocabulary_auxiliary_width;
            for (std::uint32_t index = 0U; index < count; ++index) {
                mark(present, ids[base + index]);
            }
        };
        mark_aux(history_ids, history_counts[row]);
        mark_aux(required_ids, required_counts[row]);
        mark_aux(retrieval_ids, retrieval_counts[row]);
        auto count = std::uint32_t {0U};
        for (std::uint32_t token = 0U; token < vocabulary_size; ++token) {
            if (present[token]) {
                output[static_cast<std::size_t>(row) * vocabulary_size + count++] = token;
            }
        }
        output_counts[row] = count;
    }
    return output;
}

std::vector<float> indexed_vocabulary_dot_cpu(
    std::span<const float> states,
    std::uint32_t row_count,
    std::uint32_t state_width,
    const tensor::TensorView& packed_columns,
    std::span<const std::uint32_t> candidate_ids,
    std::span<const std::uint32_t> candidate_counts,
    std::uint32_t candidate_stride
) {
    const auto& descriptor = packed_columns.descriptor();
    if (descriptor.shape.rank() != 2U
        || descriptor.shape.dimension(0) != state_width
        || row_count == 0U || candidate_stride == 0U) {
        throw std::invalid_argument("indexed vocabulary dot dimensions are invalid");
    }
    const auto vocabulary_size = static_cast<std::uint32_t>(descriptor.shape.dimension(1));
    require_extent(states.size(), static_cast<std::uint64_t>(row_count) * state_width,
        "indexed states");
    require_extent(candidate_ids.size(),
        static_cast<std::uint64_t>(row_count) * candidate_stride, "indexed candidates");
    require_extent(candidate_counts.size(), row_count, "indexed candidate counts");
    std::vector<float> output(static_cast<std::size_t>(row_count) * candidate_stride);
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        if (candidate_counts[row] > candidate_stride) {
            throw std::invalid_argument("indexed candidate count exceeds stride");
        }
        for (std::uint32_t candidate = 0U; candidate < candidate_counts[row]; ++candidate) {
            const auto token = candidate_ids[
                static_cast<std::size_t>(row) * candidate_stride + candidate
            ];
            if (token >= vocabulary_size) {
                throw std::invalid_argument("indexed candidate token is outside vocabulary");
            }
            auto sum = 0.0F;
            for (std::uint32_t index = 0U; index < state_width; ++index) {
                sum += states[static_cast<std::size_t>(row) * state_width + index]
                    * packed_column_value(packed_columns, state_width, token, index);
            }
            output[static_cast<std::size_t>(row) * candidate_stride + candidate] = sum;
        }
    }
    return output;
}

DraftProposalResult select_draft_proposal_cpu(
    std::span<const std::uint32_t> candidate_ids,
    std::span<const float> corrected_scores,
    std::span<const std::uint32_t> exp_lut,
    float temperature,
    std::uint32_t common_draw
) {
    if (candidate_ids.empty() || candidate_ids.size() != corrected_scores.size()
        || exp_lut.size() != draft_vocabulary_proposal_lut_size
        || !std::isfinite(temperature) || temperature <= 0.0F) {
        throw std::invalid_argument("draft proposal transform input is invalid");
    }
    for (std::size_t index = 0U; index < exp_lut.size(); ++index) {
        if ((index != 0U && exp_lut[index] > exp_lut[index - 1U])
            || (index == 0U && exp_lut[index] == 0U)) {
            throw std::invalid_argument("draft proposal LUT is not a positive monotone table");
        }
    }
    auto maximum = -std::numeric_limits<float>::infinity();
    for (const auto score : corrected_scores) {
        if (!std::isfinite(score)) {
            return {.status = DraftProposalStatus::conservative_numeric_fallback};
        }
        maximum = std::max(maximum, score);
    }
    std::vector<std::uint32_t> weights(candidate_ids.size());
    auto total = std::uint64_t {0U};
    for (std::size_t index = 0U; index < corrected_scores.size(); ++index) {
        const float scaled = ((maximum - corrected_scores[index]) / temperature) * 256.0F;
        if (!std::isfinite(scaled)) {
            return {.status = DraftProposalStatus::conservative_numeric_fallback};
        }
        const auto floor_value = std::floor(scaled);
        const auto fraction = scaled - floor_value;
        if (scaled < 4'096.0F
            && std::abs(fraction - 0.5F) <= (1.0F / 4'096.0F)) {
            return {.status = DraftProposalStatus::conservative_numeric_fallback};
        }
        const auto bin = scaled >= 4'096.0F
            ? 4'096U : static_cast<std::uint32_t>(floor_value + 0.5F);
        weights[index] = exp_lut[bin];
        total += weights[index];
    }
    if (total == 0U) {
        return {.status = DraftProposalStatus::conservative_numeric_fallback};
    }
    const auto threshold = static_cast<std::uint64_t>(common_draw) * (total >> 32U)
        + ((static_cast<std::uint64_t>(common_draw)
            * (total & 0xFFFF'FFFFULL)) >> 32U);
    auto cumulative = std::uint64_t {0U};
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        cumulative += weights[index];
        if (cumulative > threshold) {
            return {
                .status = DraftProposalStatus::selected,
                .token_id = candidate_ids[index],
                .total_weight = total,
                .selected_rank = static_cast<std::uint32_t>(index),
            };
        }
    }
    return {.status = DraftProposalStatus::conservative_numeric_fallback};
}

DraftVocabularyMetalPlan::DraftVocabularyMetalPlan(
    DraftVocabularyArtifactSpec spec,
    DraftVocabularyMetalScratch scratch,
    std::optional<tensor::BackendBuffer> selector_projection,
    tensor::BackendBuffer selector_head,
    tensor::BackendBuffer candidate_head,
    std::optional<tensor::BackendBuffer> target_up_projection,
    std::optional<tensor::BackendBuffer> transition_counts,
    std::optional<tensor::BackendBuffer> transition_tokens,
    tensor::BackendBuffer proposal_exp_lut,
    bool candidate_head_aliases_target,
    std::size_t owned_weight_bytes,
    std::size_t scratch_bytes
)
    : spec_(std::move(spec)),
      scratch_(std::move(scratch)),
      selector_projection_(std::move(selector_projection)),
      selector_head_(std::move(selector_head)),
      candidate_head_(std::move(candidate_head)),
      target_up_projection_(std::move(target_up_projection)),
      transition_counts_(std::move(transition_counts)),
      transition_tokens_(std::move(transition_tokens)),
      proposal_exp_lut_(std::move(proposal_exp_lut)),
      candidate_head_aliases_target_(candidate_head_aliases_target),
      owned_weight_bytes_(owned_weight_bytes),
      scratch_bytes_(scratch_bytes) {}

const DraftVocabularyArtifactSpec& DraftVocabularyMetalPlan::spec() const noexcept {
    return spec_;
}

const DraftVocabularyMetalScratch& DraftVocabularyMetalPlan::scratch() const noexcept {
    return scratch_;
}

const tensor::BackendBuffer* DraftVocabularyMetalPlan::selector_projection() const noexcept {
    return selector_projection_ ? &*selector_projection_ : nullptr;
}

const tensor::BackendBuffer& DraftVocabularyMetalPlan::selector_head() const noexcept {
    return selector_head_;
}

const tensor::BackendBuffer& DraftVocabularyMetalPlan::candidate_head() const noexcept {
    return candidate_head_;
}

const tensor::BackendBuffer* DraftVocabularyMetalPlan::target_up_projection() const noexcept {
    return target_up_projection_ ? &*target_up_projection_ : nullptr;
}

const tensor::BackendBuffer* DraftVocabularyMetalPlan::transition_counts() const noexcept {
    return transition_counts_ ? &*transition_counts_ : nullptr;
}

const tensor::BackendBuffer* DraftVocabularyMetalPlan::transition_tokens() const noexcept {
    return transition_tokens_ ? &*transition_tokens_ : nullptr;
}

const tensor::BackendBuffer& DraftVocabularyMetalPlan::proposal_exp_lut() const noexcept {
    return proposal_exp_lut_;
}

bool DraftVocabularyMetalPlan::candidate_head_aliases_target() const noexcept {
    return candidate_head_aliases_target_;
}

std::size_t DraftVocabularyMetalPlan::owned_weight_bytes() const noexcept {
    return owned_weight_bytes_;
}

std::size_t DraftVocabularyMetalPlan::scratch_bytes() const noexcept {
    return scratch_bytes_;
}

DraftVocabularyMetalPlan create_draft_vocabulary_metal_plan(
    const AuthenticatedSpeculativeArtifact& artifact,
    const EngineModelPayloadReceipt& model_receipt,
    const ResidentTargetPayloadReceipt& resident_receipt,
    const Gemma2MetalModelPlan& target
) {
    const auto& descriptor = artifact.descriptor();
    validate_speculative_artifact_target(
        descriptor, model_receipt, resident_receipt, target.config(),
        static_cast<std::uint32_t>(target.output_head_policy())
    );
    const auto& draft = descriptor.draft_vocabulary;
    if (!draft.present()) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "artifact has no draft vocabulary plan"
        );
    }
    if (target.backend() != runtime::BackendKind::metal) {
        throw std::invalid_argument("draft vocabulary target must be Metal resident");
    }
    validate_selector_padding(artifact, draft);
    const auto& tensors = draft.tensors;
    const auto load_optional = [&](DraftArtifactTensorRole role)
        -> std::optional<tensor::BackendBuffer> {
        const auto& spec = tensors[draft_artifact_tensor_index(role)];
        if (spec.storage == DraftArtifactTensorStorage::absent) return std::nullopt;
        return upload_artifact_tensor(artifact, spec);
    };
    auto selector_projection = load_optional(DraftArtifactTensorRole::selector_projection);
    auto selector_head = upload_artifact_tensor(
        artifact, tensors[draft_artifact_tensor_index(DraftArtifactTensorRole::selector_head)]
    );
    const auto& candidate_spec = tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::candidate_head
    )];
    const auto candidate_alias = candidate_spec.storage
        == DraftArtifactTensorStorage::resident_target_output_q8_0;
    auto candidate_head = candidate_alias
        ? target.weights().output_weight
        : upload_artifact_tensor(artifact, candidate_spec);
    if (candidate_head.descriptor() != artifact_tensor_descriptor(candidate_spec)) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::incompatible_target,
            "resident candidate-head descriptor"
        );
    }
    auto target_up = load_optional(DraftArtifactTensorRole::target_up_projection);
    auto transitions_counts = load_optional(DraftArtifactTensorRole::transition_counts);
    auto transitions_tokens = load_optional(DraftArtifactTensorRole::transition_tokens);
    const auto& lut_spec = tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::proposal_exp_lut
    )];
    const auto lut_bytes = artifact.payload().subspan(
        static_cast<std::size_t>(lut_spec.payload_offset),
        static_cast<std::size_t>(lut_spec.payload_byte_count)
    );
    auto previous = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t index = 0U; index < draft_vocabulary_proposal_lut_size; ++index) {
        std::uint32_t value = 0U;
        std::memcpy(&value, lut_bytes.data() + static_cast<std::size_t>(index) * 4U, 4U);
        if ((index == 0U && value == 0U) || value > previous) {
            throw SpeculativeArtifactError(
                SpeculativeArtifactErrorCode::invalid_descriptor,
                "proposal exp LUT is not positive monotone"
            );
        }
        previous = value;
    }
    auto proposal_lut = upload_artifact_tensor(artifact, lut_spec);

    const auto rows = static_cast<std::uint64_t>(draft.state_capacity);
    const auto vocabulary = static_cast<std::uint64_t>(draft.vocabulary_size);
    const auto tile_count = (vocabulary + draft_vocabulary_selection_tile_size - 1U)
        / draft_vocabulary_selection_tile_size;
    const auto bit_words = (vocabulary + 31U) / 32U;
    auto score_storage = allocate_f32_buffer(rows, vocabulary);
    auto runs_a = allocate_f32_buffer(rows, vocabulary * 2U);
    auto runs_b = allocate_f32_buffer(rows, vocabulary * 2U);
    auto selector_states = allocate_f32_buffer(rows, draft.selector_storage_width);
    auto candidate_bits = allocate_f32_buffer(rows, bit_words);
    auto candidate_tile_counts = allocate_f32_buffer(rows, tile_count);
    auto candidate_tile_offsets = allocate_f32_buffer(rows, tile_count);
    auto candidate_ids = allocate_f32_buffer(rows, vocabulary);
    auto candidate_counts = allocate_f32_buffer(rows, 1U);
    auto target_states = allocate_f32_buffer(rows, draft.target_width);
    auto normalized_states = allocate_f32_buffer(rows, draft.target_width);
    auto proposal_weights = allocate_f32_buffer(rows, vocabulary);
    auto proposal_results = allocate_f32_buffer(rows, 6U);
    auto status = allocate_f32_buffer(1U, 1U);
    DraftVocabularyMetalScratch scratch {
        .selector_states = std::move(selector_states),
        .selector_logits = score_storage,
        .candidate_runs_a = std::move(runs_a),
        .candidate_runs_b = std::move(runs_b),
        .candidate_bits = std::move(candidate_bits),
        .candidate_tile_counts = std::move(candidate_tile_counts),
        .candidate_tile_offsets = std::move(candidate_tile_offsets),
        .candidate_ids = std::move(candidate_ids),
        .candidate_counts = std::move(candidate_counts),
        .target_states = std::move(target_states),
        .normalized_target_states = std::move(normalized_states),
        .indexed_scores = score_storage,
        .proposal_weights = std::move(proposal_weights),
        .proposal_results = std::move(proposal_results),
        .status = std::move(status),
    };
    auto scratch_bytes = std::size_t {0U};
    const auto add_scratch = [&](const tensor::BackendBuffer& buffer) {
        if (buffer.descriptor().byte_count
            > std::numeric_limits<std::size_t>::max() - scratch_bytes) {
            throw std::overflow_error("draft vocabulary scratch byte count overflows");
        }
        scratch_bytes += buffer.descriptor().byte_count;
    };
    for (const auto* buffer : {
             &scratch.selector_states, &scratch.selector_logits,
             &scratch.candidate_runs_a, &scratch.candidate_runs_b,
             &scratch.candidate_bits, &scratch.candidate_tile_counts,
             &scratch.candidate_tile_offsets, &scratch.candidate_ids,
             &scratch.candidate_counts, &scratch.target_states,
             &scratch.normalized_target_states, &scratch.proposal_weights,
             &scratch.proposal_results, &scratch.status,
         }) add_scratch(*buffer);
    auto owned_bytes = std::size_t {0U};
    for (const auto& tensor_spec : tensors) {
        if (tensor_spec.storage != DraftArtifactTensorStorage::payload) continue;
        if (tensor_spec.payload_byte_count
            > std::numeric_limits<std::size_t>::max() - owned_bytes) {
            throw std::overflow_error("draft vocabulary owned byte count overflows");
        }
        owned_bytes += static_cast<std::size_t>(tensor_spec.payload_byte_count);
    }
    return DraftVocabularyMetalPlan(
        draft, std::move(scratch), std::move(selector_projection),
        std::move(selector_head), std::move(candidate_head), std::move(target_up),
        std::move(transitions_counts), std::move(transitions_tokens),
        std::move(proposal_lut), candidate_alias, owned_bytes, scratch_bytes
    );
}

DraftVocabularyMetalResult execute_draft_vocabulary_metal(
    DraftVocabularyMetalPlan& plan,
    const Gemma2MetalModelPlan& target,
    const DraftVocabularyDeviceInputs& inputs,
    const DraftVocabularyExecution& execution
) {
    const auto& spec = plan.spec_;
    if (!is_supported_draft_vocabulary_state_batch(execution.row_count)
        || !is_supported_draft_vocabulary_candidate_k(
            execution.candidate_k, spec.vocabulary_size)
        || execution.candidate_k > spec.candidate_capacity
        || !std::isfinite(execution.temperature) || execution.temperature <= 0.0F
        || target.config().embedding_length != spec.target_width
        || target.config().vocab_size != spec.vocabulary_size) {
        throw std::invalid_argument("draft vocabulary execution contract is invalid");
    }
    const auto require_f32_matrix = [&](const tensor::BackendBuffer& buffer,
                                        std::uint64_t rows,
                                        std::uint64_t columns,
                                        const char* name) {
        const auto& descriptor = buffer.descriptor();
        if (buffer.backend() != runtime::BackendKind::metal
            || descriptor.dtype != DType::f32 || descriptor.shape.rank() != 2U
            || descriptor.shape.dimension(0) < rows
            || descriptor.shape.dimension(1) != columns) {
            throw std::invalid_argument(std::string(name) + " device layout is invalid");
        }
    };
    require_f32_matrix(inputs.semantic_states, execution.row_count, spec.semantic_width,
        "draft semantic states");
    for (const auto* ids : {&inputs.history_ids, &inputs.required_ids,
                            &inputs.retrieval_ids}) {
        require_f32_matrix(*ids, execution.row_count,
            draft_vocabulary_auxiliary_width, "draft auxiliary IDs");
    }
    for (const auto* counts : {&inputs.history_counts, &inputs.required_counts,
                               &inputs.retrieval_counts, &inputs.common_draws}) {
        require_f32_matrix(*counts, execution.row_count, 1U, "draft scalar inputs");
    }
    require_f32_matrix(inputs.correction_scores, execution.row_count,
        spec.vocabulary_size, "draft correction scores");

    auto& scratch = plan.scratch_;
    tensor::backends::metal::MetalCommandContext command("Draft vocabulary scoring");
    command.write_u32(scratch.status, 0U);
    const auto start = std::chrono::steady_clock::now();
    const tensor::BackendBuffer* selector_states = &inputs.semantic_states;
    if (plan.selector_projection_) {
        command.encode_matmul(
            inputs.semantic_states, *plan.selector_projection_, scratch.selector_states,
            execution.row_count
        );
        command.encode_memory_barrier();
        selector_states = &scratch.selector_states;
    }
    if (plan.selector_head_.descriptor().dtype == DType::q8_0) {
        command.encode_target_head_q8_rows(
            *selector_states, 0U, plan.selector_head_, scratch.selector_logits,
            execution.row_count
        );
    } else {
        command.encode_matmul(
            *selector_states, plan.selector_head_, scratch.selector_logits,
            execution.row_count
        );
    }
    command.encode_memory_barrier();
    const auto full_vocabulary = execution.candidate_k == spec.vocabulary_size;
    auto selected = &scratch.candidate_runs_a;
    if (!full_vocabulary) {
        const auto topk = command.encode_draft_stable_top_k(
            scratch.selector_logits, scratch.candidate_runs_a,
            scratch.candidate_runs_b, scratch.status, execution.row_count,
            spec.vocabulary_size, execution.candidate_k
        );
        selected = topk.final_buffer_is_a
            ? &scratch.candidate_runs_a : &scratch.candidate_runs_b;
        command.encode_memory_barrier();
    }
    const auto& transition_counts = plan.transition_counts_
        ? *plan.transition_counts_ : scratch.status;
    const auto& transition_tokens = plan.transition_tokens_
        ? *plan.transition_tokens_ : scratch.status;
    command.encode_draft_candidate_union(
        *selected, transition_counts, transition_tokens,
        inputs.history_ids, inputs.history_counts,
        inputs.required_ids, inputs.required_counts,
        inputs.retrieval_ids, inputs.retrieval_counts,
        scratch.candidate_bits, scratch.candidate_tile_counts,
        scratch.candidate_tile_offsets, scratch.candidate_ids,
        scratch.candidate_counts, scratch.status, execution.row_count,
        spec.vocabulary_size, execution.candidate_k, spec.transition_width,
        full_vocabulary
    );
    command.encode_memory_barrier();

    const tensor::BackendBuffer* target_states = &inputs.semantic_states;
    if (plan.target_up_projection_) {
        command.encode_matmul(
            inputs.semantic_states, *plan.target_up_projection_, scratch.target_states,
            execution.row_count
        );
        command.encode_memory_barrier();
        target_states = &scratch.target_states;
    }
    command.encode_rms_norm(
        *target_states, target.weights().output_norm_weight,
        scratch.normalized_target_states, target.config().rms_norm_epsilon,
        execution.row_count
    );
    command.encode_memory_barrier();
    const auto expanded = static_cast<std::uint64_t>(execution.candidate_k)
        * (1U + spec.transition_width) + 3U * draft_vocabulary_auxiliary_width;
    const auto dispatch_width = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        spec.vocabulary_size, expanded
    ));
    command.encode_draft_indexed_dot(
        scratch.normalized_target_states, plan.candidate_head_,
        scratch.candidate_ids, scratch.candidate_counts, scratch.indexed_scores,
        scratch.status, execution.row_count, spec.vocabulary_size, dispatch_width
    );
    command.encode_memory_barrier();
    command.encode_draft_score_correction(
        scratch.indexed_scores, inputs.correction_scores, scratch.candidate_counts,
        scratch.status, execution.row_count, spec.vocabulary_size, dispatch_width,
        target.config().final_logit_softcap
    );
    command.encode_memory_barrier();
    command.encode_draft_proposals(
        scratch.candidate_ids, scratch.candidate_counts, scratch.indexed_scores,
        plan.proposal_exp_lut_, inputs.common_draws, scratch.proposal_weights,
        scratch.proposal_results, execution.row_count, spec.vocabulary_size,
        execution.temperature
    );
    const auto stats = command.commit_and_wait();
    const auto status = command.read_verifier_status(scratch.status);
    if (status != 0U) {
        throw std::runtime_error(
            "Metal draft vocabulary rejected malformed or non-finite device input"
        );
    }
    tensor::backends::metal::read_shared_metal_buffer_bytes(
        scratch.proposal_results, 0U,
        std::span<std::uint8_t>(
            reinterpret_cast<std::uint8_t*>(plan.host_proposals_.data()),
            static_cast<std::size_t>(execution.row_count) * sizeof(DraftProposalResult)
        )
    );
    const auto end = std::chrono::steady_clock::now();
    tensor::backends::metal::read_shared_metal_buffer_bytes(
        scratch.candidate_counts, 0U,
        std::span<std::uint8_t>(
            reinterpret_cast<std::uint8_t*>(plan.host_candidate_counts_.data()),
            static_cast<std::size_t>(execution.row_count) * sizeof(std::uint32_t)
        )
    );
    return {
        .proposals = std::span<const DraftProposalResult>(
            plan.host_proposals_.data(), execution.row_count
        ),
        .candidate_counts = std::span<const std::uint32_t>(
            plan.host_candidate_counts_.data(), execution.row_count
        ),
        .gpu_duration_ns = stats.gpu_duration_ns,
        .wall_duration_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        ),
        .dispatch_count = stats.compute_dispatch_count,
    };
}

CorrectedVocabularyWorkloadSpec corrected_vocabulary_workload_spec(
    CorrectedVocabularyWorkload workload,
    std::uint32_t target_width,
    std::uint32_t vocabulary_size
) {
    if (target_width == 0U || vocabulary_size == 0U) {
        throw std::invalid_argument("corrected-vocabulary target dimensions are empty");
    }
    CorrectedVocabularyWorkloadSpec spec {.workload = workload};
    switch (workload) {
        case CorrectedVocabularyWorkload::treeflash_released_h15_m16_top16:
            spec.fidelity = CorrectedVocabularySourceFidelity::released_source_exact;
            spec.semantic_rows = spec.physical_head_rows = 240U;
            spec.conditioning_width = 16U;
            spec.rank_width = 16U;
            spec.rank_entry_capacity = 3'840U;
            break;
        case CorrectedVocabularyWorkload::treeflash_h15_m64_top16:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 960U;
            spec.conditioning_width = 64U;
            spec.rank_width = 16U;
            spec.rank_entry_capacity = 15'360U;
            break;
        case CorrectedVocabularyWorkload::treeflash_h15_m64_top64:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 960U;
            spec.conditioning_width = 64U;
            spec.rank_width = 64U;
            spec.rank_entry_capacity = 61'440U;
            break;
        case CorrectedVocabularyWorkload::treeflash_h15_m64_full_v_lazy:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 960U;
            spec.conditioning_width = 64U;
            spec.rank_width = 512U;
            spec.rank_entry_capacity = 491'520U;
            break;
        case CorrectedVocabularyWorkload::treeflash_paper_h16_m64_full_v_lazy:
            spec.fidelity = CorrectedVocabularySourceFidelity::paper_semantic_transfer;
            spec.semantic_rows = spec.physical_head_rows = 961U;
            spec.conditioning_width = 64U;
            spec.rank_width = 512U;
            spec.rank_entry_capacity = 492'032U;
            break;
        case CorrectedVocabularyWorkload::treeflash_code_h16_m64_full_v_lazy:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 1'024U;
            spec.conditioning_width = 64U;
            spec.rank_width = 512U;
            spec.rank_entry_capacity = 524'288U;
            break;
        case CorrectedVocabularyWorkload::arm_s_cross_root_excluded:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 160U;
            spec.conditioning_width = 4U;
            spec.rank_width = 3U;
            spec.rank_entry_capacity = 384U;
            spec.raw_lower_main_rows = 121U;
            spec.upper_main_rows = 32U;
            spec.upper_stream_rows = 128U;
            break;
        case CorrectedVocabularyWorkload::arm_s_cross_root_included:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 165U;
            spec.conditioning_width = 4U;
            spec.rank_width = 3U;
            spec.rank_entry_capacity = 396U;
            spec.raw_lower_main_rows = 121U;
            spec.upper_main_rows = 33U;
            spec.upper_stream_rows = 132U;
            break;
        case CorrectedVocabularyWorkload::arm_s_deep_gamma23_k1:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 552U;
            spec.conditioning_width = 23U;
            spec.rank_width = 1U;
            spec.rank_entry_capacity = 552U;
            spec.raw_lower_main_rows = 24U;
            spec.upper_stream_rows = 552U;
            break;
        case CorrectedVocabularyWorkload::arm_s_threshold_gamma4_k3:
            spec.fidelity = CorrectedVocabularySourceFidelity::paper_semantic_transfer;
            spec.semantic_rows = spec.physical_head_rows = 605U;
            spec.conditioning_width = 4U;
            spec.rank_width = 3U;
            spec.rank_entry_capacity = 1'452U;
            spec.raw_lower_main_rows = 121U;
            spec.upper_main_rows = 121U;
            spec.upper_stream_rows = 484U;
            break;
        case CorrectedVocabularyWorkload::arm_s_post_walk_selected_anchor:
            spec.fidelity = CorrectedVocabularySourceFidelity::local_complete_interpretation;
            spec.semantic_rows = spec.physical_head_rows = 4U;
            spec.conditioning_width = 4U;
            spec.rank_width = 3U;
            spec.rank_entry_capacity = 12U;
            spec.upper_stream_rows = 4U;
            break;
    }
    spec.head_macs = multiply_checked(
        multiply_checked(spec.physical_head_rows, target_width, "head MAC count"),
        vocabulary_size, "head MAC count"
    );
    spec.forbidden_dense_f32_bytes = multiply_checked(
        multiply_checked(spec.physical_head_rows, vocabulary_size, "dense score bytes"),
        sizeof(float), "dense score bytes"
    );
    return spec;
}

std::vector<DraftVocabularyCandidate> corrected_full_vocabulary_cpu(
    std::span<const float> states,
    std::uint32_t row_count,
    std::span<const float> final_norm_weight,
    float epsilon,
    float softcap,
    float temperature,
    const tensor::TensorView& target_q8_head,
    std::uint32_t rank_width
) {
    const auto& descriptor = target_q8_head.descriptor();
    if (descriptor.dtype != DType::q8_0 || descriptor.shape.rank() != 2U
        || row_count == 0U || rank_width == 0U
        || rank_width > descriptor.shape.dimension(1)
        || !std::isfinite(epsilon) || epsilon <= 0.0F
        || !std::isfinite(softcap) || softcap <= 0.0F
        || !std::isfinite(temperature) || temperature <= 0.0F) {
        throw std::invalid_argument("corrected full-vocabulary CPU dimensions are invalid");
    }
    const auto hidden = static_cast<std::uint32_t>(descriptor.shape.dimension(0));
    const auto vocabulary = static_cast<std::uint32_t>(descriptor.shape.dimension(1));
    require_extent(states.size(), static_cast<std::uint64_t>(row_count) * hidden,
        "corrected CPU states");
    require_extent(final_norm_weight.size(), hidden, "corrected CPU norm weight");
    std::vector<float> normalized(static_cast<std::size_t>(row_count) * hidden);
    std::vector<float> logits(static_cast<std::size_t>(row_count) * vocabulary);
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        auto sum_squares = 0.0F;
        for (std::uint32_t index = 0U; index < hidden; ++index) {
            const auto value = states[static_cast<std::size_t>(row) * hidden + index];
            if (!std::isfinite(value) || !std::isfinite(final_norm_weight[index])) {
                throw std::invalid_argument("corrected CPU input must be finite");
            }
            sum_squares += value * value;
        }
        const auto inverse_rms = 1.0F / std::sqrt(
            sum_squares / static_cast<float>(hidden) + epsilon
        );
        for (std::uint32_t index = 0U; index < hidden; ++index) {
            normalized[static_cast<std::size_t>(row) * hidden + index]
                = states[static_cast<std::size_t>(row) * hidden + index]
                * inverse_rms * final_norm_weight[index];
        }
        for (std::uint32_t token = 0U; token < vocabulary; ++token) {
            auto score = 0.0F;
            for (std::uint32_t index = 0U; index < hidden; ++index) {
                score += normalized[static_cast<std::size_t>(row) * hidden + index]
                    * q8_column_value(target_q8_head.bytes(), hidden, token, index);
            }
            logits[static_cast<std::size_t>(row) * vocabulary + token]
                = softcap * std::tanh(score / softcap);
        }
    }
    auto ranks = stable_top_k_cpu(logits, row_count, vocabulary, rank_width);
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        auto maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0U; token < vocabulary; ++token) {
            maximum = std::max(maximum,
                logits[static_cast<std::size_t>(row) * vocabulary + token] / temperature);
        }
        auto sum = 0.0F;
        for (std::uint32_t token = 0U; token < vocabulary; ++token) {
            sum += std::exp(logits[static_cast<std::size_t>(row) * vocabulary + token]
                / temperature - maximum);
        }
        const auto logsumexp = maximum + std::log(sum);
        for (std::uint32_t rank = 0U; rank < rank_width; ++rank) {
            auto& entry = ranks[static_cast<std::size_t>(row) * rank_width + rank];
            entry.score = entry.score / temperature - logsumexp;
        }
    }
    return ranks;
}

ArmSSurvivingPriorTree prune_arm_s_raw_tree_cpu(
    std::span<const std::uint32_t> parent_indices,
    std::span<const float> transition_probabilities,
    float threshold
) {
    if (parent_indices.empty() || parent_indices.size() > arm_s_raw_prior_node_max
        || transition_probabilities.size() + 1U != parent_indices.size()
        || parent_indices[0] != std::numeric_limits<std::uint32_t>::max()
        || !std::isfinite(threshold) || threshold < 0.0F || threshold > 1.0F) {
        throw std::invalid_argument("Arm-S raw-tree pruning contract is invalid");
    }
    ArmSSurvivingPriorTree output;
    std::array<bool, arm_s_raw_prior_node_max> kept {};
    kept[0] = true;
    output.raw_node_indices[output.survivor_count++] = 0U;
    for (std::uint32_t node = 1U; node < parent_indices.size(); ++node) {
        const auto parent = parent_indices[node];
        const auto probability = transition_probabilities[node - 1U];
        if (parent >= node || !std::isfinite(probability)
            || probability < 0.0F || probability > 1.0F) {
            throw std::invalid_argument("Arm-S raw-tree edge is malformed");
        }
        kept[node] = kept[parent] && probability >= threshold;
        if (kept[node]) {
            output.raw_node_indices[output.survivor_count++] = node;
        }
    }
    return output;
}

ArmSCompactNextTree build_arm_s_compact_next_tree_cpu(
    std::span<const std::uint32_t> anchor_root_token_ids,
    std::span<const DraftVocabularyCandidate> stream_rank_entries,
    std::uint32_t gamma,
    std::uint32_t branch_width,
    std::uint32_t output_capacity
) {
    if (anchor_root_token_ids.empty()
        || anchor_root_token_ids.size() > arm_s_raw_prior_node_max
        || gamma == 0U || gamma > draft_vocabulary_external_state_max
        || branch_width == 0U || branch_width > 8U
        || output_capacity == 0U
        || output_capacity > arm_s_gamma4_k3_compact_node_max) {
        throw std::invalid_argument("Arm-S compact-tree dimensions are invalid");
    }
    const auto required_ranks = static_cast<std::uint64_t>(anchor_root_token_ids.size())
        * gamma * branch_width;
    if (required_ranks != stream_rank_entries.size()) {
        throw std::invalid_argument("Arm-S compact-tree rank extent is invalid");
    }
    auto level_nodes = std::uint64_t {1U};
    auto nodes_per_anchor = std::uint64_t {1U};
    for (std::uint32_t depth = 0U; depth < gamma; ++depth) {
        if (level_nodes > output_capacity / branch_width) {
            throw std::length_error("Arm-S compact tree exceeds fixed capacity");
        }
        level_nodes *= branch_width;
        if (nodes_per_anchor > output_capacity - level_nodes) {
            throw std::length_error("Arm-S compact tree exceeds fixed capacity");
        }
        nodes_per_anchor += level_nodes;
    }
    if (nodes_per_anchor > output_capacity / anchor_root_token_ids.size()) {
        throw std::length_error("Arm-S compact tree exceeds fixed capacity");
    }
    ArmSCompactNextTree output;
    for (std::uint32_t anchor = 0U; anchor < anchor_root_token_ids.size(); ++anchor) {
        const auto root = output.node_count++;
        output.token_ids[root] = anchor_root_token_ids[anchor];
        output.parent_indices[root] = std::numeric_limits<std::uint32_t>::max();
        output.anchor_indices[root] = anchor;
        output.depths[root] = 0U;
        auto level_start = root;
        auto level_count = std::uint32_t {1U};
        for (std::uint32_t depth = 0U; depth < gamma; ++depth) {
            const auto next_start = output.node_count;
            const auto rank_base = (static_cast<std::size_t>(anchor) * gamma + depth)
                * branch_width;
            for (std::uint32_t rank = 0U; rank < branch_width; ++rank) {
                const auto& candidate = stream_rank_entries[rank_base + rank];
                if (!std::isfinite(candidate.score)) {
                    throw std::invalid_argument("Arm-S compact-tree rank is non-finite");
                }
                for (std::uint32_t prior = 0U; prior < rank; ++prior) {
                    if (stream_rank_entries[rank_base + prior].token_id
                        == candidate.token_id) {
                        throw std::invalid_argument(
                            "Arm-S compact-tree stream menu contains duplicate token "
                            + std::to_string(candidate.token_id) + " at anchor "
                            + std::to_string(anchor) + ", depth "
                            + std::to_string(depth)
                        );
                    }
                }
            }
            for (std::uint32_t parent_offset = 0U;
                 parent_offset < level_count; ++parent_offset) {
                const auto parent = level_start + parent_offset;
                for (std::uint32_t rank = 0U; rank < branch_width; ++rank) {
                    const auto node = output.node_count++;
                    output.token_ids[node] = stream_rank_entries[rank_base + rank].token_id;
                    output.parent_indices[node] = parent;
                    output.anchor_indices[node] = anchor;
                    output.depths[node] = static_cast<std::uint16_t>(depth + 1U);
                }
            }
            level_start = next_start;
            level_count *= branch_width;
        }
    }
    return output;
}

ArmSEarlyExitMetalPlan::ArmSEarlyExitMetalPlan(
    tensor::BackendBuffer down,
    tensor::BackendBuffer up,
    ArmSEarlyExitMetalScratch scratch,
    std::size_t owned_weight_bytes,
    std::size_t scratch_bytes
)
    : down_(std::move(down)), up_(std::move(up)), scratch_(std::move(scratch)),
      owned_weight_bytes_(owned_weight_bytes), scratch_bytes_(scratch_bytes) {}

ArmSEarlyExitInterpretation ArmSEarlyExitMetalPlan::interpretation() const noexcept {
    return ArmSEarlyExitInterpretation::local_bias_free_rank8_projection;
}

std::uint32_t ArmSEarlyExitMetalPlan::row_capacity() const noexcept {
    return arm_s_gamma4_k3_raw_edge_count;
}

std::size_t ArmSEarlyExitMetalPlan::owned_weight_bytes() const noexcept {
    return owned_weight_bytes_;
}

std::size_t ArmSEarlyExitMetalPlan::scratch_bytes() const noexcept {
    return scratch_bytes_;
}

ArmSEarlyExitMetalPlan create_arm_s_early_exit_metal_plan(
    const AuthenticatedSpeculativeArtifact& artifact,
    const EngineModelPayloadReceipt& model_receipt,
    const ResidentTargetPayloadReceipt& resident_receipt,
    const Gemma2MetalModelPlan& target
) {
    validate_speculative_artifact_target(
        artifact.descriptor(), model_receipt, resident_receipt, target.config(),
        static_cast<std::uint32_t>(target.output_head_policy())
    );
    if (target.output_head_policy() != Gemma2OutputHeadPolicy::q8_0) {
        throw std::invalid_argument("Arm-S early exit requires the resident target-Q8 head");
    }
    const auto& draft = artifact.descriptor().draft_vocabulary;
    if (!draft.present()) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "Arm-S early exit artifact has no draft vocabulary specification"
        );
    }
    const auto& down_spec = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::arm_s_rank8_down
    )];
    const auto& up_spec = draft.tensors[draft_artifact_tensor_index(
        DraftArtifactTensorRole::arm_s_rank8_up
    )];
    if (down_spec.storage != DraftArtifactTensorStorage::payload
        || up_spec.storage != DraftArtifactTensorStorage::payload) {
        throw SpeculativeArtifactError(
            SpeculativeArtifactErrorCode::invalid_descriptor,
            "Arm-S rank-8 local interpretation tensors are absent"
        );
    }
    auto down = upload_artifact_tensor(artifact, down_spec);
    auto up = upload_artifact_tensor(artifact, up_spec);
    constexpr auto rows = std::uint64_t {arm_s_gamma4_k3_raw_edge_count};
    constexpr auto tile_rows = std::uint64_t {16U};
    const auto vocabulary = static_cast<std::uint64_t>(target.config().vocab_size);
    ArmSEarlyExitMetalScratch scratch {
        .rank8_states = allocate_f32_buffer(rows, 8U),
        .projected_target_states = allocate_f32_buffer(rows, target.config().embedding_length),
        .normalized_rows = allocate_f32_buffer(tile_rows, target.config().embedding_length),
        .logits_tile = allocate_f32_buffer(tile_rows, vocabulary),
        .logsumexp = allocate_f32_buffer(tile_rows, 1U),
        .transition_scores = allocate_f32_buffer(rows, 1U),
        .status = allocate_f32_buffer(1U, 1U),
    };
    auto scratch_bytes = std::size_t {0U};
    for (const auto* buffer : {
             &scratch.rank8_states, &scratch.projected_target_states,
             &scratch.normalized_rows, &scratch.logits_tile, &scratch.logsumexp,
             &scratch.transition_scores, &scratch.status,
         }) {
        if (buffer->descriptor().byte_count
            > std::numeric_limits<std::size_t>::max() - scratch_bytes) {
            throw std::overflow_error("Arm-S early-exit scratch bytes overflow");
        }
        scratch_bytes += buffer->descriptor().byte_count;
    }
    const auto owned = static_cast<std::size_t>(
        down_spec.payload_byte_count + up_spec.payload_byte_count
    );
    return ArmSEarlyExitMetalPlan(
        std::move(down), std::move(up), std::move(scratch), owned, scratch_bytes
    );
}

ArmSEarlyExitMetalResult execute_arm_s_early_exit_metal(
    ArmSEarlyExitMetalPlan& plan,
    const Gemma2MetalModelPlan& target,
    const tensor::BackendBuffer& parent_states,
    const tensor::BackendBuffer& child_token_ids,
    std::uint32_t edge_count
) {
    if (edge_count == 0U || edge_count > plan.row_capacity()
        || parent_states.backend() != runtime::BackendKind::metal
        || parent_states.descriptor().dtype != DType::f32
        || parent_states.descriptor().shape.rank() != 2U
        || parent_states.descriptor().shape.dimension(0) < edge_count
        || parent_states.descriptor().shape.dimension(1)
            != plan.down_.descriptor().shape.dimension(0)
        || child_token_ids.backend() != runtime::BackendKind::metal
        || child_token_ids.descriptor().byte_count
            < static_cast<std::uint64_t>(edge_count) * sizeof(std::uint32_t)) {
        throw std::invalid_argument("Arm-S early-exit device input is incompatible");
    }
    auto& scratch = plan.scratch_;
    tensor::backends::metal::MetalCommandContext command("Arm-S rank-8 early exit");
    command.write_u32(scratch.status, 0U);
    const auto start = std::chrono::steady_clock::now();
    command.encode_matmul(parent_states, plan.down_, scratch.rank8_states, edge_count);
    command.encode_memory_barrier();
    command.encode_matmul(
        scratch.rank8_states, plan.up_, scratch.projected_target_states, edge_count
    );
    command.encode_memory_barrier();
    constexpr auto tile_rows = std::uint32_t {16U};
    for (std::uint32_t first = 0U; first < edge_count; first += tile_rows) {
        const auto count = std::min(tile_rows, edge_count - first);
        command.encode_rms_norm(
            scratch.projected_target_states, target.weights().output_norm_weight,
            scratch.normalized_rows, target.config().rms_norm_epsilon, count, first
        );
        command.encode_memory_barrier();
        command.encode_target_head_q8_rows(
            scratch.normalized_rows, 0U, target.weights().output_weight,
            scratch.logits_tile, count
        );
        command.encode_memory_barrier();
        command.encode_softcap(
            scratch.logits_tile, scratch.logits_tile,
            target.config().final_logit_softcap,
            static_cast<std::uint64_t>(count) * target.config().vocab_size
        );
        command.encode_memory_barrier();
        command.encode_corrected_logsumexp(
            scratch.logits_tile, scratch.logsumexp, scratch.status,
            count, target.config().vocab_size, 1.0F
        );
        command.encode_memory_barrier();
        command.encode_arm_s_transition_scores(
            scratch.logits_tile, scratch.logsumexp, child_token_ids, first,
            scratch.transition_scores, scratch.status, count,
            target.config().vocab_size, first
        );
        if (first + count < edge_count) command.encode_memory_barrier();
    }
    const auto stats = command.commit_and_wait();
    if (command.read_verifier_status(scratch.status) != 0U) {
        throw std::runtime_error("Arm-S early-exit scoring rejected its device input");
    }
    tensor::backends::metal::read_shared_metal_buffer_bytes(
        scratch.transition_scores, 0U,
        std::span<std::uint8_t>(
            reinterpret_cast<std::uint8_t*>(plan.host_scores_.data()),
            static_cast<std::size_t>(edge_count) * sizeof(float)
        )
    );
    const auto end = std::chrono::steady_clock::now();
    return {
        .transition_probabilities = std::span<const float>(
            plan.host_scores_.data(), edge_count
        ),
        .gpu_duration_ns = stats.gpu_duration_ns,
        .wall_duration_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        ),
        .dispatch_count = stats.compute_dispatch_count,
        .full_vocabulary_rows = edge_count,
        .scratch_bytes = plan.scratch_bytes_,
    };
}

CorrectedVocabularyMetalPlan::CorrectedVocabularyMetalPlan(
    CorrectedVocabularyWorkloadSpec spec,
    CorrectedVocabularyMetalScratch scratch,
    std::size_t scratch_bytes
)
    : spec_(spec), scratch_(std::move(scratch)), scratch_bytes_(scratch_bytes),
      host_rank_entries_(spec.rank_entry_capacity) {}

const CorrectedVocabularyWorkloadSpec& CorrectedVocabularyMetalPlan::spec() const noexcept {
    return spec_;
}

std::size_t CorrectedVocabularyMetalPlan::scratch_bytes() const noexcept {
    return scratch_bytes_;
}

CorrectedVocabularyMetalPlan create_corrected_vocabulary_metal_plan(
    const Gemma2MetalModelPlan& target,
    CorrectedVocabularyWorkload workload
) {
    if (target.backend() != runtime::BackendKind::metal
        || target.output_head_policy() != Gemma2OutputHeadPolicy::q8_0
        || target.weights().output_weight.descriptor().dtype != DType::q8_0) {
        throw std::invalid_argument(
            "corrected vocabulary scoring requires the resident target-Q8 head"
        );
    }
    auto spec = corrected_vocabulary_workload_spec(
        workload, target.config().embedding_length, target.config().vocab_size
    );
    constexpr auto tile_rows = std::uint64_t {16U};
    const auto vocabulary = static_cast<std::uint64_t>(target.config().vocab_size);
    CorrectedVocabularyMetalScratch scratch {
        .normalized_rows = allocate_f32_buffer(tile_rows, target.config().embedding_length),
        .logits_tile = allocate_f32_buffer(tile_rows, vocabulary),
        .candidate_runs_a = allocate_f32_buffer(tile_rows, vocabulary * 2U),
        .candidate_runs_b = allocate_f32_buffer(tile_rows, vocabulary * 2U),
        .logsumexp = allocate_f32_buffer(tile_rows, 1U),
        .rank_entries = allocate_f32_buffer(spec.rank_entry_capacity, 2U),
        .status = allocate_f32_buffer(1U, 1U),
    };
    auto bytes = std::size_t {0U};
    for (const auto* buffer : {
             &scratch.normalized_rows, &scratch.logits_tile,
             &scratch.candidate_runs_a, &scratch.candidate_runs_b,
             &scratch.logsumexp, &scratch.rank_entries, &scratch.status,
         }) {
        if (buffer->descriptor().byte_count > std::numeric_limits<std::size_t>::max() - bytes) {
            throw std::overflow_error("corrected vocabulary scratch byte count overflows");
        }
        bytes += buffer->descriptor().byte_count;
    }
    return CorrectedVocabularyMetalPlan(spec, std::move(scratch), bytes);
}

CorrectedVocabularyMetalResult execute_corrected_vocabulary_metal(
    CorrectedVocabularyMetalPlan& plan,
    const Gemma2MetalModelPlan& target,
    const tensor::BackendBuffer& corrected_states,
    float temperature
) {
    const auto& spec = plan.spec_;
    const auto& descriptor = corrected_states.descriptor();
    if (corrected_states.backend() != runtime::BackendKind::metal
        || descriptor.dtype != DType::f32 || descriptor.shape.rank() != 2U
        || descriptor.shape.dimension(0) < spec.physical_head_rows
        || descriptor.shape.dimension(1) != target.config().embedding_length
        || target.output_head_policy() != Gemma2OutputHeadPolicy::q8_0
        || !std::isfinite(temperature) || temperature <= 0.0F) {
        throw std::invalid_argument("corrected vocabulary Metal input is incompatible");
    }
    const auto cached_rows = spec.rank_entry_capacity / spec.rank_width;
    if (cached_rows * spec.rank_width != spec.rank_entry_capacity
        || cached_rows > spec.physical_head_rows) {
        throw std::logic_error("corrected vocabulary rank ledger is inconsistent");
    }
    const auto cache_first_row = spec.physical_head_rows - cached_rows;
    auto& scratch = plan.scratch_;
    tensor::backends::metal::MetalCommandContext command("Corrected full-vocabulary head");
    command.write_u32(scratch.status, 0U);
    const auto start = std::chrono::steady_clock::now();
    const auto encode_range = [&](std::uint32_t first_row,
                                  std::uint32_t row_count,
                                  bool cache_ranks,
                                  std::uint32_t output_row_start) {
        command.encode_rms_norm(
            corrected_states, target.weights().output_norm_weight,
            scratch.normalized_rows, target.config().rms_norm_epsilon,
            row_count, first_row
        );
        command.encode_memory_barrier();
        command.encode_target_head_q8_rows(
            scratch.normalized_rows, 0U, target.weights().output_weight,
            scratch.logits_tile, row_count
        );
        command.encode_memory_barrier();
        command.encode_softcap(
            scratch.logits_tile, scratch.logits_tile,
            target.config().final_logit_softcap,
            static_cast<std::uint64_t>(row_count) * target.config().vocab_size
        );
        if (!cache_ranks) return;
        command.encode_memory_barrier();
        command.encode_corrected_logsumexp(
            scratch.logits_tile, scratch.logsumexp, scratch.status,
            row_count, target.config().vocab_size, temperature
        );
        command.encode_memory_barrier();
        const auto topk = command.encode_draft_stable_top_k(
            scratch.logits_tile, scratch.candidate_runs_a,
            scratch.candidate_runs_b, scratch.status, row_count,
            target.config().vocab_size, spec.rank_width
        );
        command.encode_memory_barrier();
        command.encode_corrected_rank_finalize(
            topk.final_buffer_is_a ? scratch.candidate_runs_a : scratch.candidate_runs_b,
            scratch.logsumexp, scratch.rank_entries, scratch.status,
            row_count, target.config().vocab_size, spec.rank_width,
            output_row_start, temperature
        );
    };
    constexpr auto tile_rows = std::uint32_t {16U};
    for (std::uint32_t first = 0U; first < cache_first_row; first += tile_rows) {
        encode_range(first, std::min(tile_rows, cache_first_row - first), false, 0U);
        command.encode_memory_barrier();
    }
    for (std::uint32_t first = cache_first_row; first < spec.physical_head_rows;
         first += tile_rows) {
        const auto count = std::min(tile_rows, spec.physical_head_rows - first);
        encode_range(first, count, true, first - cache_first_row);
        if (first + count < spec.physical_head_rows) command.encode_memory_barrier();
    }
    const auto stats = command.commit_and_wait();
    if (command.read_verifier_status(scratch.status) != 0U) {
        throw std::runtime_error("corrected vocabulary Metal scoring produced non-finite output");
    }
    tensor::backends::metal::read_shared_metal_buffer_bytes(
        scratch.rank_entries, 0U,
        std::span<std::uint8_t>(
            reinterpret_cast<std::uint8_t*>(plan.host_rank_entries_.data()),
            plan.host_rank_entries_.size() * sizeof(DraftVocabularyCandidate)
        )
    );
    const auto end = std::chrono::steady_clock::now();
    const auto head_bytes = target.weights().output_weight.descriptor().byte_count;
    const auto head_tiles = (spec.physical_head_rows + tile_rows - 1U) / tile_rows;
    return {
        .rank_entries = plan.host_rank_entries_,
        .gpu_duration_ns = stats.gpu_duration_ns,
        .wall_duration_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        ),
        .dispatch_count = stats.compute_dispatch_count,
        .command_buffer_count = stats.command_buffer_count,
        .physical_head_rows = spec.physical_head_rows,
        .rescanned_head_rows = 0U,
        .head_macs = spec.head_macs,
        .head_flops = multiply_checked(spec.head_macs, 2U, "corrected head FLOPs"),
        .logical_head_weight_bytes = multiply_checked(
            head_tiles, head_bytes, "corrected head logical weight bytes"
        ),
        .scratch_bytes = plan.scratch_bytes_,
    };
}

TreeFlashTree build_treeflash_tree_cpu(
    const TreeFlashRankTableView& table,
    std::uint32_t tree_size
) {
    if (tree_size == 0U || table.depth_count == 0U
        || table.depth_count > draft_vocabulary_external_state_max
        || table.conditioning_width == 0U || table.rank_width == 0U) {
        throw std::invalid_argument("TreeFlash rank table dimensions are invalid");
    }
    const auto entries = static_cast<std::uint64_t>(table.depth_count)
        * table.conditioning_width * table.rank_width;
    require_extent(table.ranks.size(), entries, "TreeFlash ranks");
    require_extent(table.transition_next_indices.size(), entries,
        "TreeFlash transition indices");
    struct HeapEntry {
        float cumulative = 0.0F;
        std::array<std::uint16_t, draft_vocabulary_external_state_max> path {};
        std::uint16_t path_length = 0U;
        std::uint32_t parent = 0U;
        std::uint32_t depth = 1U;
        std::uint32_t rank = 0U;
        std::uint32_t source_depth = 0U;
        std::uint32_t source_condition = 0U;
    };
    const auto worse = [](const HeapEntry& left, const HeapEntry& right) {
        if (left.cumulative != right.cumulative) return left.cumulative < right.cumulative;
        const auto common = std::min(left.path_length, right.path_length);
        for (std::uint16_t index = 0U; index < common; ++index) {
            if (left.path[index] != right.path[index]) return left.path[index] > right.path[index];
        }
        if (left.path_length != right.path_length) return left.path_length > right.path_length;
        if (left.parent != right.parent) return left.parent > right.parent;
        if (left.depth != right.depth) return left.depth > right.depth;
        if (left.rank != right.rank) return left.rank > right.rank;
        if (left.source_depth != right.source_depth) return left.source_depth > right.source_depth;
        return left.source_condition > right.source_condition;
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(worse)> heap(worse);
    const auto at = [&](std::uint32_t depth, std::uint32_t condition, std::uint32_t rank) {
        return (static_cast<std::size_t>(depth) * table.conditioning_width + condition)
            * table.rank_width + rank;
    };
    const auto first = table.ranks[at(0U, 0U, 0U)].score;
    if (std::isnan(first)) throw std::invalid_argument("TreeFlash rank score is NaN");
    if (std::isfinite(first) && tree_size > 1U) {
        HeapEntry entry;
        entry.cumulative = first;
        entry.path[0] = 0U;
        entry.path_length = 1U;
        heap.push(entry);
    }
    TreeFlashTree output;
    output.token_ids.assign(tree_size, 0U);
    output.parent_indices.assign(tree_size, 0U);
    output.depths.assign(tree_size, 0U);
    auto produced = std::uint32_t {1U};
    while (produced < tree_size && !heap.empty()) {
        const auto current = heap.top();
        heap.pop();
        const auto index = at(current.source_depth, current.source_condition, current.rank);
        output.token_ids[produced] = table.ranks[index].token_id;
        output.parent_indices[produced] = current.parent;
        output.depths[produced] = static_cast<std::uint16_t>(current.depth);
        const auto next_condition = table.transition_next_indices[index];
        const auto node_index = produced++;
        if (current.rank + 1U < table.rank_width) {
            const auto sibling_index = at(
                current.source_depth, current.source_condition, current.rank + 1U
            );
            const auto sibling_score = table.ranks[sibling_index].score;
            if (std::isnan(sibling_score)) {
                throw std::invalid_argument("TreeFlash sibling score is NaN");
            }
            if (std::isfinite(sibling_score)) {
                auto sibling = current;
                sibling.cumulative = current.cumulative
                    - table.ranks[index].score + sibling_score;
                sibling.rank += 1U;
                sibling.path[sibling.path_length - 1U]
                    = static_cast<std::uint16_t>(sibling.rank);
                if (std::isfinite(sibling.cumulative)) heap.push(sibling);
            }
        }
        if (current.depth < table.depth_count && next_condition >= 0
            && static_cast<std::uint32_t>(next_condition) < table.conditioning_width) {
            const auto child_index = at(
                current.source_depth + 1U,
                static_cast<std::uint32_t>(next_condition), 0U
            );
            const auto child_score = table.ranks[child_index].score;
            if (std::isnan(child_score)) {
                throw std::invalid_argument("TreeFlash child score is NaN");
            }
            if (std::isfinite(child_score)) {
                auto child = current;
                child.cumulative += child_score;
                child.parent = node_index;
                child.depth += 1U;
                child.rank = 0U;
                child.source_depth += 1U;
                child.source_condition = static_cast<std::uint32_t>(next_condition);
                child.path[child.path_length++] = 0U;
                if (std::isfinite(child.cumulative)) heap.push(child);
            }
        }
    }
    output.token_ids.resize(produced);
    output.parent_indices.resize(produced);
    output.depths.resize(produced);
    return output;
}

}  // namespace local_gguf_engine::engine
