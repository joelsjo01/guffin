#pragma once

#include "local_gguf_engine/runtime/sha256.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"
#include "local_gguf_engine/tensor/dtype.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace local_gguf_engine::engine {

inline constexpr std::size_t feature_bank_max_count = 8U;
// The shared DFlash/DSpark feature bank uses five distinct target-layer taps.
// Artifact banks serialize only tap_count entries, so older banks keep their
// existing wire representation when this in-memory capacity grows.
inline constexpr std::size_t feature_bank_tap_max_count = 5U;
inline constexpr std::size_t feature_bank_child_max_count = 8U;
inline constexpr std::size_t feature_bank_total_child_max_count =
    feature_bank_max_count * feature_bank_child_max_count;
inline constexpr std::uint32_t feature_projection_width_max = 4'096U;

enum class FeatureTapStage : std::uint8_t {
    embedding_residual,
    layer_residual,
    final_normalized,
};

enum class FeatureFusionKind : std::uint8_t {
    concatenate,
};

enum class FeatureProjectionSchedule : std::uint8_t {
    eager,
    lazy,
};

struct FeatureTapSpec {
    FeatureTapStage stage = FeatureTapStage::layer_residual;
    std::uint16_t layer_index = 0U;

    constexpr bool operator==(const FeatureTapSpec&) const = default;
};

struct FeatureProjectionSpec {
    std::uint32_t child_id = 0U;
    std::uint32_t output_width = 0U;
    tensor::DType weight_dtype = tensor::DType::f16;
    FeatureProjectionSchedule schedule = FeatureProjectionSchedule::eager;
    std::uint64_t payload_offset = 0U;
    std::uint64_t payload_byte_count = 0U;
    runtime::Sha256Digest payload_digest;

    constexpr bool operator==(const FeatureProjectionSpec&) const = default;
};

struct FeatureBankSpec {
    std::uint32_t bank_id = 0U;
    FeatureFusionKind fusion = FeatureFusionKind::concatenate;
    std::array<FeatureTapSpec, feature_bank_tap_max_count> taps {};
    std::uint16_t tap_count = 0U;
    std::array<FeatureProjectionSpec, feature_bank_child_max_count> children {};
    std::uint16_t child_count = 0U;

    constexpr bool operator==(const FeatureBankSpec&) const = default;
};

struct DraftCacheSpec {
    std::uint16_t layer_count = 0U;
    std::uint16_t stream_count = 0U;
    std::uint16_t key_width = 0U;
    std::uint16_t value_width = 0U;
    std::uint16_t maximum_depth = 0U;
    std::uint16_t maximum_active_width = 0U;
    std::uint32_t linear_row_capacity = 0U;
    std::uint32_t tree_row_capacity = 0U;

    constexpr bool operator==(const DraftCacheSpec&) const = default;
};

struct TargetStateCoordinate {
    std::uint32_t target_position = 0U;
    // A draft state generated from target p predicts p+1.
    std::uint32_t draft_position = 0U;

    constexpr bool operator==(const TargetStateCoordinate&) const = default;
};

struct TargetStateView {
    std::uint64_t execution_epoch = 0U;
    std::uint64_t arena_epoch = 0U;
    std::uint32_t bank_id = 0U;
    std::uint32_t child_id = 0U;
    FeatureTapStage source_stage = FeatureTapStage::layer_residual;
    std::uint16_t source_layer = 0U;
    TargetStateCoordinate coordinate;
    std::uint32_t row_count = 0U;
    std::uint32_t feature_width = 0U;

    constexpr bool operator==(const TargetStateView&) const = default;
};

enum class FeatureCaptureDestination : std::uint8_t {
    canonical,
    speculative,
};

struct FeatureSinkBinding {
    class FeatureSinkSet* sinks = nullptr;
    class TargetFeatureTraceSink* trace_sink = nullptr;
    FeatureCaptureDestination destination = FeatureCaptureDestination::canonical;
    std::uint64_t execution_epoch = 0U;
    std::uint64_t arena_epoch = 0U;
};

inline constexpr std::size_t target_feature_trace_tap_max_count = 44U;
inline constexpr std::uint32_t target_feature_trace_row_max = 1'025U;

struct TargetFeatureTraceBatch {
    std::uint64_t execution_epoch = 0U;
    std::vector<std::uint32_t> absolute_positions;
    // Exact token consumed to produce each target row. Keeping this beside
    // the row prevents an offline collector from reconstructing prompt or
    // incremental-chat tokenization after execution.
    std::vector<std::uint32_t> input_token_ids;
    std::vector<FeatureTapSpec> taps;
    std::uint32_t hidden_width = 0U;
    tensor::DType dtype = tensor::DType::f16;
    // Row-major [row, tap, hidden]. The callback owns no view into device
    // memory, so it may serialize this batch immediately and release it.
    std::vector<std::uint8_t> bytes;
};

using TargetFeatureTraceCallback =
    std::function<void(TargetFeatureTraceBatch&&)>;

class TargetFeatureTraceSink final {
public:
    TargetFeatureTraceSink(const TargetFeatureTraceSink&) = delete;
    TargetFeatureTraceSink& operator=(const TargetFeatureTraceSink&) = delete;
    TargetFeatureTraceSink(TargetFeatureTraceSink&&) noexcept;
    TargetFeatureTraceSink& operator=(TargetFeatureTraceSink&&) noexcept;
    ~TargetFeatureTraceSink();

    [[nodiscard]] std::span<const FeatureTapSpec> taps() const noexcept;
    [[nodiscard]] std::uint32_t hidden_width() const noexcept;
    [[nodiscard]] std::uint32_t row_capacity() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;

private:
    struct Impl;
    friend struct FeatureSinkRuntimeAccess;
    friend TargetFeatureTraceSink create_target_feature_trace_sink(
        std::span<const FeatureTapSpec>,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t,
        TargetFeatureTraceCallback
    );
    explicit TargetFeatureTraceSink(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] TargetFeatureTraceSink create_target_feature_trace_sink(
    std::span<const FeatureTapSpec> taps,
    std::uint32_t target_hidden_width,
    std::uint32_t target_layer_count,
    std::uint32_t row_capacity,
    TargetFeatureTraceCallback callback
);

[[nodiscard]] constexpr bool feature_sink_bound(const FeatureSinkBinding& binding) noexcept {
    return binding.sinks != nullptr || binding.trace_sink != nullptr;
}

struct FeatureSinkMetrics {
    std::uint64_t captured_row_count = 0U;
    std::uint64_t eager_projection_row_count = 0U;
    std::uint64_t lazy_stale_row_count = 0U;
    std::uint64_t lazy_rebuild_row_count = 0U;
    std::uint64_t logical_capture_bytes = 0U;
    std::uint64_t logical_projection_weight_bytes = 0U;
    std::uint64_t trace_readback_row_count = 0U;
    std::uint64_t trace_readback_bytes = 0U;
    std::uint64_t allocated_bytes = 0U;
};

class FeatureSinkSet final {
public:
    FeatureSinkSet(const FeatureSinkSet&) = delete;
    FeatureSinkSet& operator=(const FeatureSinkSet&) = delete;
    FeatureSinkSet(FeatureSinkSet&&) noexcept;
    FeatureSinkSet& operator=(FeatureSinkSet&&) noexcept;
    ~FeatureSinkSet();

    [[nodiscard]] std::span<const FeatureBankSpec> specs() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::uint32_t canonical_frontier() const noexcept;
    [[nodiscard]] const FeatureSinkMetrics& metrics() const noexcept;
    [[nodiscard]] std::optional<TargetStateView> target_state_view(
        std::uint32_t child_id,
        std::uint32_t target_position
    ) const;
    // Offline trace collection only. The requested canonical rows are gathered
    // into the fixed speculative arena before CPU readback; production decode
    // never exposes or copies the full context-sized feature allocation.
    [[nodiscard]] tensor::TensorStorage copy_canonical_child_feature_rows(
        std::uint32_t child_id,
        std::uint32_t first_target_position,
        std::uint32_t row_count
    ) const;
    void rebuild_lazy_child(
        std::uint32_t child_id,
        std::uint32_t first_target_position,
        std::uint32_t row_count
    );
    void reset() noexcept;

private:
    struct Impl;
    friend struct FeatureSinkRuntimeAccess;
    friend FeatureSinkSet create_feature_sink_set(
        std::span<const FeatureBankSpec>,
        std::span<const std::uint8_t>,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t
    );
    explicit FeatureSinkSet(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] FeatureSinkSet create_feature_sink_set(
    std::span<const FeatureBankSpec> specs,
    std::span<const std::uint8_t> artifact_payload,
    std::uint32_t target_hidden_width,
    std::uint32_t target_layer_count,
    std::uint32_t target_context_length,
    std::uint32_t speculative_row_capacity
);

struct DraftCacheLayerView {
    std::uint16_t layer_index = 0U;
    std::uint16_t stream_index = 0U;
    tensor::BackendBuffer canonical_key;
    tensor::BackendBuffer canonical_value;
    tensor::BackendBuffer linear_key;
    tensor::BackendBuffer linear_value;
    tensor::BackendBuffer tree_key;
    tensor::BackendBuffer tree_value;
};

enum class DraftCacheScratchKind : std::uint8_t {
    linear,
    tree,
};

struct DraftCacheBinding {
    class DraftCacheArena* arena = nullptr;
    DraftCacheScratchKind scratch = DraftCacheScratchKind::linear;
    std::uint64_t arena_epoch = 0U;
};

struct DraftCacheStateView {
    std::uint64_t arena_epoch = 0U;
    std::uint32_t target_position = 0U;
    std::uint32_t draft_position = 1U;
    std::uint32_t target_frontier = 0U;

    constexpr bool operator==(const DraftCacheStateView&) const = default;
};

class DraftCacheArena final {
public:
    DraftCacheArena(const DraftCacheArena&) = delete;
    DraftCacheArena& operator=(const DraftCacheArena&) = delete;
    DraftCacheArena(DraftCacheArena&&) noexcept = default;
    DraftCacheArena& operator=(DraftCacheArena&&) noexcept = default;

    [[nodiscard]] const DraftCacheSpec& spec() const noexcept;
    [[nodiscard]] std::span<const DraftCacheLayerView> layers() const noexcept;
    [[nodiscard]] std::uint32_t target_frontier() const noexcept;
    [[nodiscard]] std::uint32_t draft_frontier() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] std::optional<DraftCacheStateView> state_view(
        std::uint32_t target_position
    ) const noexcept;
    [[nodiscard]] tensor::TensorStorage copy_canonical_key(
        std::uint16_t layer_index,
        std::uint16_t stream_index
    ) const;
    [[nodiscard]] tensor::TensorStorage copy_canonical_value(
        std::uint16_t layer_index,
        std::uint16_t stream_index
    ) const;
    void reset() noexcept;

private:
    friend DraftCacheArena create_draft_cache_arena(
        const DraftCacheSpec&,
        std::uint32_t
    );
    friend struct DraftCacheRuntimeAccess;
    DraftCacheArena() = default;
    DraftCacheSpec spec_;
    std::vector<DraftCacheLayerView> layers_;
    std::uint32_t target_frontier_ = 0U;
    std::uint32_t draft_frontier_ = 1U;
    std::size_t allocated_bytes_ = 0U;
    std::optional<tensor::BackendBuffer> source_rows_;
    std::optional<tensor::BackendBuffer> destination_rows_;
    std::optional<tensor::BackendBuffer> status_;
    std::array<std::uint32_t, 1'025U> source_row_values_ {};
    std::array<std::uint32_t, 1'025U> destination_row_values_ {};
    std::uint64_t arena_epoch_ = 0U;
    std::uint32_t promotion_row_count_ = 0U;
    bool promotion_active_ = false;
    bool poisoned_ = false;
};

[[nodiscard]] DraftCacheArena create_draft_cache_arena(
    const DraftCacheSpec& spec,
    std::uint32_t target_context_length
);

}  // namespace local_gguf_engine::engine
