#include "local_gguf_engine/engine/feature_bank.hpp"

#include "feature_bank_runtime.hpp"
#include "draft_cache_runtime.hpp"
#include "../tensor/backends/metal/metal_command_context.hpp"

#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace local_gguf_engine::engine {
namespace {

using runtime::BackendKind;
using tensor::DType;
using tensor::Shape;
using tensor::TensorDescriptor;
using tensor::backends::metal::MetalCommandContext;

static_assert(feature_bank_tap_max_count <= std::numeric_limits<std::uint8_t>::digits,
    "feature capture mask must cover every supported tap");

[[nodiscard]] TensorDescriptor descriptor(
    std::uint64_t rows,
    std::uint64_t columns,
    DType dtype
) {
    return TensorDescriptor::from(Shape::from({rows, columns}), dtype);
}

void checked_add(std::size_t& total, std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - total) {
        throw std::overflow_error("feature/draft-cache allocation byte count overflow");
    }
    total += value;
}

[[nodiscard]] tensor::BackendBuffer allocate(
    const TensorDescriptor& value,
    std::size_t& bytes
) {
    checked_add(bytes, value.byte_count);
    return tensor::allocate_backend_buffer(value, BackendKind::metal);
}

[[nodiscard]] bool tap_matches(
    const FeatureTapSpec& tap,
    FeatureTapStage stage,
    std::uint16_t layer
) noexcept {
    return tap.stage == stage
        && (stage != FeatureTapStage::layer_residual || tap.layer_index == layer);
}

[[nodiscard]] std::uint32_t tap_order(const FeatureTapSpec& tap) {
    switch (tap.stage) {
        case FeatureTapStage::embedding_residual:
            return 0U;
        case FeatureTapStage::layer_residual:
            return 1U + tap.layer_index;
        case FeatureTapStage::final_normalized:
            return std::numeric_limits<std::uint32_t>::max();
    }
    throw std::invalid_argument("feature tap stage is invalid");
}

}  // namespace

struct FeatureSinkSet::Impl {
    struct Child {
        FeatureProjectionSpec spec;
        tensor::BackendBuffer weight;
        tensor::BackendBuffer projected;
        tensor::BackendBuffer canonical;
        tensor::BackendBuffer speculative;
        std::vector<std::optional<TargetStateView>> provenance;
        std::vector<std::optional<TargetStateView>> speculative_provenance;
        std::optional<std::uint32_t> stale_frontier;
    };

    struct Bank {
        FeatureBankSpec spec;
        std::uint32_t fused_width = 0U;
        tensor::BackendBuffer fused;
        std::optional<tensor::BackendBuffer> lazy_canonical;
        std::optional<tensor::BackendBuffer> lazy_speculative;
        std::vector<std::optional<TargetStateView>> lazy_canonical_provenance;
        std::vector<std::optional<TargetStateView>> lazy_speculative_provenance;
        std::vector<Child> children;

        Bank(
            FeatureBankSpec value_spec,
            std::uint32_t value_fused_width,
            tensor::BackendBuffer value_fused
        ) : spec(value_spec), fused_width(value_fused_width),
            fused(std::move(value_fused)) {}
    };

    std::vector<FeatureBankSpec> specs;
    std::vector<Bank> banks;
    std::array<std::uint32_t, 1'025U> source_row_values {};
    std::array<std::uint32_t, 1'025U> destination_row_values {};
    std::array<std::uint32_t, 1'025U> active_absolute_positions {};
    std::array<std::uint8_t, feature_bank_max_count> active_tap_masks {};
    tensor::BackendBuffer source_rows;
    tensor::BackendBuffer destination_rows;
    tensor::BackendBuffer status;
    std::uint32_t hidden_width = 0U;
    std::uint32_t context_length = 0U;
    std::uint32_t speculative_capacity = 0U;
    std::uint32_t canonical_frontier = 0U;
    FeatureSinkMetrics metrics;
    bool poisoned = false;
    bool capture_active = false;
    bool promotion_active = false;
    std::uint32_t promotion_row_count = 0U;
    FeatureCaptureDestination active_destination = FeatureCaptureDestination::canonical;
    std::uint64_t active_execution_epoch = 0U;
    std::uint64_t active_arena_epoch = 0U;
    std::uint32_t active_row_count = 0U;

    Impl(
        tensor::BackendBuffer value_source_rows,
        tensor::BackendBuffer value_destination_rows,
        tensor::BackendBuffer value_status
    ) : source_rows(std::move(value_source_rows)),
        destination_rows(std::move(value_destination_rows)),
        status(std::move(value_status)) {}
};

struct TargetFeatureTraceSink::Impl {
    std::vector<FeatureTapSpec> taps;
    tensor::BackendBuffer rows;
    TargetFeatureTraceCallback callback;
    std::vector<std::uint32_t> active_positions;
    std::vector<std::uint32_t> active_token_ids;
    std::array<bool, target_feature_trace_tap_max_count> captured {};
    std::uint32_t hidden_width = 0U;
    std::uint32_t row_capacity = 0U;
    std::uint32_t active_row_count = 0U;
    std::uint64_t active_execution_epoch = 0U;
    bool active = false;
    bool poisoned = false;

    Impl(
        std::vector<FeatureTapSpec> value_taps,
        tensor::BackendBuffer value_rows,
        TargetFeatureTraceCallback value_callback,
        std::uint32_t value_hidden_width,
        std::uint32_t value_row_capacity
    ) : taps(std::move(value_taps)), rows(std::move(value_rows)),
        callback(std::move(value_callback)), hidden_width(value_hidden_width),
        row_capacity(value_row_capacity) {}
};

TargetFeatureTraceSink::TargetFeatureTraceSink(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
TargetFeatureTraceSink::TargetFeatureTraceSink(TargetFeatureTraceSink&&) noexcept = default;
TargetFeatureTraceSink& TargetFeatureTraceSink::operator=(TargetFeatureTraceSink&&) noexcept = default;
TargetFeatureTraceSink::~TargetFeatureTraceSink() = default;

std::span<const FeatureTapSpec> TargetFeatureTraceSink::taps() const noexcept {
    return impl_ ? std::span<const FeatureTapSpec>(impl_->taps)
                 : std::span<const FeatureTapSpec> {};
}
std::uint32_t TargetFeatureTraceSink::hidden_width() const noexcept {
    return impl_ ? impl_->hidden_width : 0U;
}
std::uint32_t TargetFeatureTraceSink::row_capacity() const noexcept {
    return impl_ ? impl_->row_capacity : 0U;
}
bool TargetFeatureTraceSink::poisoned() const noexcept {
    return !impl_ || impl_->poisoned;
}

TargetFeatureTraceSink create_target_feature_trace_sink(
    std::span<const FeatureTapSpec> taps,
    std::uint32_t target_hidden_width,
    std::uint32_t target_layer_count,
    std::uint32_t row_capacity,
    TargetFeatureTraceCallback callback
) {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) {
        throw std::runtime_error("target feature trace sink requires Metal");
    }
    if (taps.empty() || taps.size() > target_feature_trace_tap_max_count
        || target_hidden_width == 0U || target_layer_count == 0U
        || row_capacity == 0U || row_capacity > target_feature_trace_row_max
        || !callback) {
        throw std::invalid_argument("target feature trace sink contract is invalid");
    }
    auto previous = std::uint32_t {0U};
    for (std::size_t index = 0U; index < taps.size(); ++index) {
        if (taps[index].stage == FeatureTapStage::layer_residual
            && taps[index].layer_index >= target_layer_count) {
            throw std::invalid_argument("target feature trace tap is outside target layers");
        }
        const auto order = tap_order(taps[index]);
        if (index != 0U && order <= previous) {
            throw std::invalid_argument("target feature trace taps are not unique and ordered");
        }
        previous = order;
    }
    const auto width = static_cast<std::uint64_t>(target_hidden_width) * taps.size();
    if (width > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("target feature trace row width overflow");
    }
    auto rows = tensor::allocate_backend_buffer(
        descriptor(row_capacity, width, DType::f16), BackendKind::metal
    );
    return TargetFeatureTraceSink(std::make_unique<TargetFeatureTraceSink::Impl>(
        std::vector<FeatureTapSpec>(taps.begin(), taps.end()), std::move(rows),
        std::move(callback), target_hidden_width, row_capacity
    ));
}

FeatureSinkSet::FeatureSinkSet(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FeatureSinkSet::FeatureSinkSet(FeatureSinkSet&&) noexcept = default;
FeatureSinkSet& FeatureSinkSet::operator=(FeatureSinkSet&&) noexcept = default;
FeatureSinkSet::~FeatureSinkSet() = default;

std::span<const FeatureBankSpec> FeatureSinkSet::specs() const noexcept {
    return impl_ ? std::span<const FeatureBankSpec>(impl_->specs) : std::span<const FeatureBankSpec> {};
}

bool FeatureSinkSet::poisoned() const noexcept { return !impl_ || impl_->poisoned; }
std::uint32_t FeatureSinkSet::canonical_frontier() const noexcept {
    return impl_ ? impl_->canonical_frontier : 0U;
}
const FeatureSinkMetrics& FeatureSinkSet::metrics() const noexcept {
    static const FeatureSinkMetrics empty;
    return impl_ ? impl_->metrics : empty;
}

std::optional<TargetStateView> FeatureSinkSet::target_state_view(
    std::uint32_t child_id,
    std::uint32_t target_position
) const {
    if (!impl_ || impl_->poisoned || target_position >= impl_->context_length) {
        return std::nullopt;
    }
    for (const auto& bank : impl_->banks) {
        for (const auto& child : bank.children) {
            if (child.spec.child_id == child_id) return child.provenance[target_position];
        }
    }
    return std::nullopt;
}

tensor::TensorStorage FeatureSinkSet::copy_canonical_child_feature_rows(
    std::uint32_t child_id,
    std::uint32_t first_target_position,
    std::uint32_t row_count
) const {
    if (!impl_ || impl_->poisoned || impl_->capture_active || impl_->promotion_active
        || row_count == 0U || row_count > impl_->speculative_capacity
        || first_target_position > impl_->canonical_frontier
        || row_count > impl_->canonical_frontier - first_target_position) {
        throw std::logic_error("feature sink is absent or poisoned");
    }
    for (const auto& bank : impl_->banks) {
        if (std::any_of(
            bank.lazy_speculative_provenance.begin(),
            bank.lazy_speculative_provenance.end(),
            [](const auto& value) { return value.has_value(); }
        )) {
            throw std::logic_error("feature trace readback conflicts with speculative rows");
        }
        for (const auto& child : bank.children) {
            if (std::any_of(
                child.speculative_provenance.begin(),
                child.speculative_provenance.end(),
                [](const auto& value) { return value.has_value(); }
            )) {
                throw std::logic_error("feature trace readback conflicts with speculative rows");
            }
        }
    }
    for (auto& bank : impl_->banks) {
        for (auto& child : bank.children) {
            if (child.spec.child_id == child_id) {
                for (std::uint32_t row = 0U; row < row_count; ++row) {
                    const auto position = first_target_position + row;
                    const auto& state = child.provenance[position];
                    if (!state.has_value()
                        || state->child_id != child_id
                        || state->coordinate.target_position != position
                        || state->coordinate.draft_position != position + 1U) {
                        throw std::logic_error(
                            "feature trace readback lacks exact canonical provenance"
                        );
                    }
                    impl_->source_row_values[row] = position;
                    impl_->destination_row_values[row] = row;
                }
                MetalCommandContext command("Bounded feature trace readback", true);
                auto submission_attempted = false;
                try {
                    command.write_u32(impl_->status, 0U);
                    command.write_u32s(
                        impl_->source_rows, 0U,
                        std::span<const std::uint32_t>(impl_->source_row_values)
                            .first(row_count)
                    );
                    command.write_u32s(
                        impl_->destination_rows, 0U,
                        std::span<const std::uint32_t>(impl_->destination_row_values)
                            .first(row_count)
                    );
                    command.encode_f16_row_scatter(
                        child.canonical, impl_->source_rows, impl_->destination_rows,
                        child.speculative, impl_->status, row_count
                    );
                    submission_attempted = true;
                    static_cast<void>(command.commit_and_wait());
                    if (command.read_verifier_status(impl_->status) != 0U) {
                        throw std::runtime_error("feature trace row gather failed");
                    }
                    const auto staging = tensor::copy_tensor_from_backend(child.speculative);
                    const auto result_descriptor = TensorDescriptor::from(
                        Shape::from({row_count, child.spec.output_width}), DType::f16
                    );
                    impl_->metrics.trace_readback_row_count += row_count;
                    impl_->metrics.trace_readback_bytes += staging.bytes().size();
                    return tensor::TensorStorage::from_bytes(
                        result_descriptor,
                        staging.bytes().first(result_descriptor.byte_count)
                    );
                } catch (...) {
                    if (submission_attempted) impl_->poisoned = true;
                    throw;
                }
            }
        }
    }
    throw std::invalid_argument("unknown feature child id");
}

void FeatureSinkSet::rebuild_lazy_child(
    std::uint32_t child_id,
    std::uint32_t first_target_position,
    std::uint32_t row_count
) {
    if (!impl_ || impl_->poisoned || impl_->capture_active || impl_->promotion_active
        || row_count == 0U || row_count > impl_->speculative_capacity
        || first_target_position > impl_->canonical_frontier
        || row_count > impl_->canonical_frontier - first_target_position) {
        throw std::logic_error("lazy feature rebuild range or lifecycle is invalid");
    }
    for (auto& bank : impl_->banks) {
        for (auto& child : bank.children) {
            if (child.spec.child_id != child_id) continue;
            if (child.spec.schedule != FeatureProjectionSchedule::lazy
                || !bank.lazy_canonical.has_value()
                || !child.stale_frontier.has_value()
                || *child.stale_frontier != first_target_position) {
                throw std::logic_error("lazy feature child is not stale at the requested frontier");
            }
            for (std::uint32_t row = 0U; row < row_count; ++row) {
                if (!bank.lazy_canonical_provenance[first_target_position + row].has_value()) {
                    throw std::logic_error("lazy feature rebuild lacks canonical raw provenance");
                }
                impl_->source_row_values[row] = row;
                impl_->destination_row_values[row] = first_target_position + row;
            }
            tensor::backends::metal::MetalCommandContext command(
                "Gemma2 lazy feature projection rebuild", true
            );
            auto submission_attempted = false;
            try {
                command.write_u32(impl_->status, 0U);
                command.write_u32s(
                    impl_->source_rows, 0U,
                    std::span<const std::uint32_t>(impl_->source_row_values).first(row_count)
                );
                command.write_u32s(
                    impl_->destination_rows, 0U,
                    std::span<const std::uint32_t>(impl_->destination_row_values).first(row_count)
                );
                command.encode_feature_tap_copy(
                    *bank.lazy_canonical, first_target_position,
                    bank.fused, 0U, row_count
                );
                command.encode_memory_barrier();
                command.encode_verifier_matmul(
                    bank.fused, child.weight, child.projected, row_count
                );
                command.encode_memory_barrier();
                command.encode_feature_scatter(
                    child.projected, impl_->source_rows, impl_->destination_rows,
                    child.canonical, impl_->status, row_count, child.spec.output_width
                );
                submission_attempted = true;
                static_cast<void>(command.commit_and_wait());
                if (command.read_verifier_status(impl_->status) != 0U) {
                    throw std::runtime_error("lazy feature projection rebuild scatter failed");
                }
                for (std::uint32_t row = 0U; row < row_count; ++row) {
                    auto view = *bank.lazy_canonical_provenance[first_target_position + row];
                    view.child_id = child.spec.child_id;
                    view.feature_width = child.spec.output_width;
                    child.provenance[first_target_position + row] = view;
                }
                impl_->metrics.lazy_rebuild_row_count += row_count;
                const auto next = first_target_position + row_count;
                child.stale_frontier = next < impl_->canonical_frontier
                    ? std::optional<std::uint32_t>(next) : std::nullopt;
                return;
            } catch (...) {
                if (submission_attempted) impl_->poisoned = true;
                throw;
            }
        }
    }
    throw std::invalid_argument("unknown feature child id");
}

void FeatureSinkSet::reset() noexcept {
    if (!impl_ || impl_->poisoned) return;
    impl_->canonical_frontier = 0U;
    impl_->capture_active = false;
    impl_->promotion_active = false;
    impl_->active_row_count = 0U;
    impl_->metrics.captured_row_count = 0U;
    impl_->metrics.eager_projection_row_count = 0U;
    impl_->metrics.lazy_stale_row_count = 0U;
    impl_->metrics.lazy_rebuild_row_count = 0U;
    impl_->metrics.logical_capture_bytes = 0U;
    impl_->metrics.logical_projection_weight_bytes = 0U;
    impl_->metrics.trace_readback_row_count = 0U;
    impl_->metrics.trace_readback_bytes = 0U;
    for (auto& bank : impl_->banks) {
        for (auto& child : bank.children) {
            std::fill(child.provenance.begin(), child.provenance.end(), std::nullopt);
            std::fill(
                child.speculative_provenance.begin(),
                child.speculative_provenance.end(), std::nullopt
            );
            child.stale_frontier.reset();
        }
        std::fill(
            bank.lazy_canonical_provenance.begin(),
            bank.lazy_canonical_provenance.end(), std::nullopt
        );
        std::fill(
            bank.lazy_speculative_provenance.begin(),
            bank.lazy_speculative_provenance.end(), std::nullopt
        );
    }
}

FeatureSinkSet create_feature_sink_set(
    std::span<const FeatureBankSpec> specs,
    std::span<const std::uint8_t> artifact_payload,
    std::uint32_t target_hidden_width,
    std::uint32_t target_layer_count,
    std::uint32_t target_context_length,
    std::uint32_t speculative_row_capacity
) {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) {
        throw std::runtime_error("feature sink requires Metal");
    }
    if (specs.empty() || specs.size() > feature_bank_max_count
        || target_hidden_width == 0U || target_layer_count == 0U
        || target_context_length == 0U || speculative_row_capacity == 0U
        || speculative_row_capacity > 1'025U) {
        throw std::invalid_argument("feature sink dimensions are invalid");
    }
    std::size_t allocated_bytes = 0U;
    auto source_rows = allocate(
        descriptor(speculative_row_capacity, 1U, DType::f32), allocated_bytes
    );
    auto destination_rows = allocate(
        descriptor(speculative_row_capacity, 1U, DType::f32), allocated_bytes
    );
    auto status = allocate(descriptor(1U, 1U, DType::f32), allocated_bytes);
    auto impl = std::make_unique<FeatureSinkSet::Impl>(
        std::move(source_rows), std::move(destination_rows), std::move(status)
    );
    impl->hidden_width = target_hidden_width;
    impl->context_length = target_context_length;
    impl->speculative_capacity = speculative_row_capacity;
    impl->specs.assign(specs.begin(), specs.end());
    impl->banks.reserve(specs.size());
    std::unordered_set<std::uint32_t> child_ids;
    auto previous_bank_id = std::uint32_t {0U};

    for (std::size_t bank_index = 0U; bank_index < specs.size(); ++bank_index) {
        const auto& spec = specs[bank_index];
        if (spec.bank_id == 0U
            || (bank_index != 0U && spec.bank_id <= previous_bank_id)
            || spec.fusion != FeatureFusionKind::concatenate
            || spec.tap_count == 0U || spec.tap_count > spec.taps.size()
            || spec.child_count == 0U || spec.child_count > spec.children.size()) {
            throw std::invalid_argument("feature bank entry count is invalid");
        }
        previous_bank_id = spec.bank_id;
        auto previous_tap_order = std::uint32_t {0U};
        for (std::uint16_t tap = 0U; tap < spec.tap_count; ++tap) {
            const auto order = tap_order(spec.taps[tap]);
            if (spec.taps[tap].stage == FeatureTapStage::layer_residual
                && spec.taps[tap].layer_index >= target_layer_count) {
                throw std::invalid_argument("feature tap layer is outside target");
            }
            if (tap != 0U && order <= previous_tap_order) {
                throw std::invalid_argument("feature taps are not in causal order");
            }
            previous_tap_order = order;
        }
        const auto fused_width_64 = static_cast<std::uint64_t>(target_hidden_width)
            * spec.tap_count;
        if (fused_width_64 > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("feature fusion width overflow");
        }
        FeatureSinkSet::Impl::Bank bank(
            spec,
            static_cast<std::uint32_t>(fused_width_64),
            allocate(
                descriptor(speculative_row_capacity, fused_width_64, DType::f32),
                allocated_bytes
            )
        );
        auto has_lazy = false;
        bank.children.reserve(spec.child_count);
        for (std::uint16_t child_index = 0U;
             child_index < spec.child_count; ++child_index) {
            const auto& child_spec = spec.children[child_index];
            if (!child_ids.insert(child_spec.child_id).second) {
                throw std::invalid_argument("feature child id is not globally unique");
            }
            if (child_spec.child_id == 0U
                || child_spec.weight_dtype != DType::f16
                || child_spec.output_width == 0U
                || child_spec.output_width > feature_projection_width_max
                || (child_spec.schedule != FeatureProjectionSchedule::eager
                    && child_spec.schedule != FeatureProjectionSchedule::lazy)) {
                throw std::invalid_argument(
                    "Phase-3 fixture feature projection must be bounded F16"
                );
            }
            const auto weight_descriptor = TensorDescriptor::from(
                Shape::from({fused_width_64, child_spec.output_width}), DType::f16
            );
            if (child_spec.payload_offset > artifact_payload.size()
                || child_spec.payload_byte_count != weight_descriptor.byte_count
                || child_spec.payload_byte_count
                    > artifact_payload.size() - child_spec.payload_offset) {
                throw std::invalid_argument("feature projection payload shape is invalid");
            }
            const auto weight_bytes = artifact_payload.subspan(
                static_cast<std::size_t>(child_spec.payload_offset),
                static_cast<std::size_t>(child_spec.payload_byte_count)
            );
            if (child_spec.payload_digest.is_zero()
                || runtime::sha256(weight_bytes) != child_spec.payload_digest) {
                throw std::invalid_argument("feature projection payload digest is invalid");
            }
            auto weight_view = tensor::TensorView::from_bytes(weight_descriptor, weight_bytes);
            FeatureSinkSet::Impl::Child child {
                .spec = child_spec,
                .weight = tensor::copy_tensor_view_to_backend(weight_view, BackendKind::metal),
                .projected = allocate(
                    descriptor(speculative_row_capacity, child_spec.output_width, DType::f32),
                    allocated_bytes
                ),
                .canonical = allocate(
                    descriptor(target_context_length, child_spec.output_width, DType::f16),
                    allocated_bytes
                ),
                .speculative = allocate(
                    descriptor(speculative_row_capacity, child_spec.output_width, DType::f16),
                    allocated_bytes
                ),
                .provenance = std::vector<std::optional<TargetStateView>>(
                    target_context_length
                ),
                .speculative_provenance =
                    std::vector<std::optional<TargetStateView>>(speculative_row_capacity),
            };
            checked_add(allocated_bytes, weight_descriptor.byte_count);
            has_lazy = has_lazy
                || child_spec.schedule == FeatureProjectionSchedule::lazy;
            bank.children.push_back(std::move(child));
        }
        if (has_lazy) {
            bank.lazy_canonical = allocate(
                descriptor(target_context_length, fused_width_64, DType::f32),
                allocated_bytes
            );
            bank.lazy_speculative = allocate(
                descriptor(speculative_row_capacity, fused_width_64, DType::f32),
                allocated_bytes
            );
            bank.lazy_canonical_provenance.resize(target_context_length);
            bank.lazy_speculative_provenance.resize(speculative_row_capacity);
        }
        impl->banks.push_back(std::move(bank));
    }
    impl->metrics.allocated_bytes = allocated_bytes;
    return FeatureSinkSet(std::move(impl));
}

bool FeatureSinkRuntimeAccess::needs_tap(
    const FeatureSinkBinding& binding,
    FeatureTapStage stage,
    std::uint16_t layer_index
) {
    if (binding.sinks != nullptr && binding.sinks->impl_) {
        for (const auto& bank : binding.sinks->impl_->banks) {
            for (std::uint16_t tap = 0U; tap < bank.spec.tap_count; ++tap) {
                if (tap_matches(bank.spec.taps[tap], stage, layer_index)) return true;
            }
        }
    }
    if (binding.trace_sink != nullptr && binding.trace_sink->impl_) {
        return std::any_of(
            binding.trace_sink->impl_->taps.begin(), binding.trace_sink->impl_->taps.end(),
            [&](const FeatureTapSpec& tap) { return tap_matches(tap, stage, layer_index); }
        );
    }
    return false;
}

void FeatureSinkRuntimeAccess::begin_capture(
    const FeatureSinkBinding& binding,
    MetalCommandContext& command,
    std::span<const std::uint32_t> absolute_positions,
    std::span<const std::uint32_t> input_token_ids,
    std::uint32_t row_count
) {
    if (!feature_sink_bound(binding)) return;
    if (binding.sinks != nullptr && binding.sinks->impl_) {
        auto& impl = *binding.sinks->impl_;
        if (impl.poisoned || impl.capture_active || impl.promotion_active
            || binding.execution_epoch == 0U
            || row_count == 0U || row_count > impl.speculative_capacity
            || absolute_positions.size() != row_count) {
            throw std::logic_error("feature capture binding or lifecycle is invalid");
        }
        for (std::uint32_t row = 0U; row < row_count; ++row) {
            impl.source_row_values[row] = row;
            if (absolute_positions[row] >= impl.context_length) {
                throw std::invalid_argument("feature capture position exceeds target context");
            }
            impl.destination_row_values[row] = binding.destination == FeatureCaptureDestination::canonical
                ? absolute_positions[row] : row;
            impl.active_absolute_positions[row] = absolute_positions[row];
        }
        if (binding.destination == FeatureCaptureDestination::canonical) {
            if (binding.arena_epoch != 0U) {
                throw std::invalid_argument("canonical feature capture cannot bind an arena epoch");
            }
            for (std::uint32_t row = 1U; row < row_count; ++row) {
                if (absolute_positions[row] != absolute_positions[0] + row) {
                    throw std::invalid_argument("canonical feature capture positions are not contiguous");
                }
            }
            if (absolute_positions[0] != impl.canonical_frontier) {
                throw std::logic_error("canonical feature capture does not begin at its frontier");
            }
        } else if (binding.arena_epoch == 0U) {
            throw std::invalid_argument("speculative feature capture requires an arena epoch");
        }
        command.write_u32(impl.status, 0U);
        command.write_u32s(
            impl.source_rows, 0U,
            std::span<const std::uint32_t>(impl.source_row_values).first(row_count)
        );
        command.write_u32s(
            impl.destination_rows, 0U,
            std::span<const std::uint32_t>(impl.destination_row_values).first(row_count)
        );
        impl.active_tap_masks.fill(0U);
        impl.capture_active = true;
        impl.active_destination = binding.destination;
        impl.active_execution_epoch = binding.execution_epoch;
        impl.active_arena_epoch = binding.arena_epoch;
        impl.active_row_count = row_count;
    }
    if (binding.trace_sink != nullptr && binding.trace_sink->impl_) {
        auto& trace = *binding.trace_sink->impl_;
        if (trace.poisoned || trace.active || binding.execution_epoch == 0U
            || row_count == 0U || row_count > trace.row_capacity
            || absolute_positions.size() != row_count
            || input_token_ids.size() != row_count) {
            throw std::logic_error("trace feature capture binding or lifecycle is invalid");
        }
        trace.active_positions.assign(absolute_positions.begin(), absolute_positions.end());
        trace.active_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
        trace.captured.fill(false);
        trace.active_row_count = row_count;
        trace.active_execution_epoch = binding.execution_epoch;
        trace.active = true;
    }
}

void FeatureSinkRuntimeAccess::encode_tap(
    const FeatureSinkBinding& binding,
    MetalCommandContext& command,
    FeatureTapStage stage,
    std::uint16_t layer_index,
    const tensor::BackendBuffer& input,
    std::uint32_t input_row_start,
    std::uint32_t row_count
) {
    if (!feature_sink_bound(binding)) return;
    if (binding.sinks != nullptr && binding.sinks->impl_) {
        auto& impl = *binding.sinks->impl_;
        if (!impl.capture_active || impl.active_execution_epoch != binding.execution_epoch
            || impl.active_arena_epoch != binding.arena_epoch
            || impl.active_destination != binding.destination) {
            throw std::logic_error("feature tap does not belong to the active capture");
        }
        if (row_count != impl.active_row_count) {
            throw std::invalid_argument("feature tap row count differs from active capture");
        }
        for (std::size_t bank_index = 0U; bank_index < impl.banks.size(); ++bank_index) {
            auto& bank = impl.banks[bank_index];
            for (std::uint16_t tap = 0U; tap < bank.spec.tap_count; ++tap) {
                if (!tap_matches(bank.spec.taps[tap], stage, layer_index)) continue;
                const auto bit = static_cast<std::uint8_t>(1U << tap);
                if ((impl.active_tap_masks[bank_index] & bit) != 0U) {
                    throw std::logic_error("feature tap was encoded twice for one target row set");
                }
                // Projection is emitted with the final tap. Merely checking
                // completeness after submission could accept a final tap
                // encoded before earlier columns had been filled.
                if (impl.active_tap_masks[bank_index] != bit - 1U) {
                    throw std::logic_error("feature taps were not encoded in causal order");
                }
                impl.active_tap_masks[bank_index] |= bit;
                command.encode_feature_tap_copy(
                    input, input_row_start, bank.fused,
                    static_cast<std::uint32_t>(tap) * impl.hidden_width, row_count
                );
                command.encode_memory_barrier();
                impl.metrics.logical_capture_bytes += static_cast<std::uint64_t>(row_count)
                    * impl.hidden_width * sizeof(float);
                if (tap + 1U != bank.spec.tap_count) continue;

                if (bank.lazy_canonical.has_value()) {
                    command.encode_feature_scatter(
                        bank.fused, impl.source_rows, impl.destination_rows,
                        binding.destination == FeatureCaptureDestination::canonical
                            ? *bank.lazy_canonical : *bank.lazy_speculative,
                        impl.status, row_count, bank.fused_width
                    );
                }
                for (auto& child : bank.children) {
                    if (child.spec.schedule == FeatureProjectionSchedule::lazy) continue;
                    command.encode_verifier_matmul(
                        bank.fused, child.weight, child.projected, row_count
                    );
                    command.encode_memory_barrier();
                    command.encode_feature_scatter(
                        child.projected, impl.source_rows, impl.destination_rows,
                        binding.destination == FeatureCaptureDestination::canonical
                            ? child.canonical : child.speculative,
                        impl.status, row_count, child.spec.output_width
                    );
                    impl.metrics.logical_projection_weight_bytes +=
                        static_cast<std::uint64_t>(row_count)
                        * bank.fused_width * child.spec.output_width * sizeof(std::uint16_t);
                }
            }
        }
    }
    if (binding.trace_sink != nullptr && binding.trace_sink->impl_) {
        auto& trace = *binding.trace_sink->impl_;
        if (!trace.active || trace.active_execution_epoch != binding.execution_epoch
            || row_count != trace.active_row_count) {
            throw std::logic_error("trace feature tap does not belong to active capture");
        }
        for (std::size_t tap = 0U; tap < trace.taps.size(); ++tap) {
            if (!tap_matches(trace.taps[tap], stage, layer_index)) continue;
            if (trace.captured[tap]) {
                throw std::logic_error("trace feature tap was encoded twice");
            }
            trace.captured[tap] = true;
            command.encode_feature_tap_store_f16(
                input, input_row_start, trace.rows,
                static_cast<std::uint32_t>(tap) * trace.hidden_width, row_count
            );
            command.encode_memory_barrier();
        }
    }
}

void FeatureSinkRuntimeAccess::complete_capture(
    const FeatureSinkBinding& binding,
    MetalCommandContext& command,
    std::span<const std::uint32_t> absolute_positions,
    std::uint32_t row_count
) {
    if (!feature_sink_bound(binding)) return;
    if (binding.sinks != nullptr && binding.sinks->impl_) {
    auto& impl = *binding.sinks->impl_;
    if (!impl.capture_active || row_count == 0U
        || row_count != impl.active_row_count
        || absolute_positions.size() != row_count
        || impl.active_execution_epoch != binding.execution_epoch
        || impl.active_arena_epoch != binding.arena_epoch
        || impl.active_destination != binding.destination
        || command.read_verifier_status(impl.status) != 0U) {
        impl.poisoned = true;
        impl.capture_active = false;
        impl.active_row_count = 0U;
        throw std::runtime_error("feature capture completion is invalid");
    }
    for (std::uint32_t row = 0U; row < row_count; ++row) {
        if (absolute_positions[row] != impl.active_absolute_positions[row]) {
            impl.poisoned = true;
            impl.capture_active = false;
            impl.active_row_count = 0U;
            throw std::runtime_error("feature capture completion positions changed");
        }
    }
    for (std::size_t bank = 0U; bank < impl.banks.size(); ++bank) {
        const auto expected = static_cast<std::uint8_t>(
            (1U << impl.banks[bank].spec.tap_count) - 1U
        );
        if (impl.active_tap_masks[bank] != expected) {
            impl.poisoned = true;
            impl.capture_active = false;
            throw std::runtime_error("feature capture omitted a declared target tap");
        }
    }
    for (auto& bank : impl.banks) {
        const auto& final_tap = bank.spec.taps[bank.spec.tap_count - 1U];
        if (bank.lazy_canonical.has_value()) {
            for (std::uint32_t row = 0U; row < row_count; ++row) {
                const auto view = TargetStateView {
                    .execution_epoch = binding.execution_epoch,
                    .arena_epoch = binding.arena_epoch,
                    .bank_id = bank.spec.bank_id,
                    .child_id = 0U,
                    .source_stage = final_tap.stage,
                    .source_layer = final_tap.layer_index,
                    .coordinate = {
                        .target_position = absolute_positions[row],
                        .draft_position = absolute_positions[row] + 1U,
                    },
                    .row_count = 1U,
                    .feature_width = bank.fused_width,
                };
                if (binding.destination == FeatureCaptureDestination::canonical) {
                    bank.lazy_canonical_provenance[absolute_positions[row]] = view;
                } else {
                    bank.lazy_speculative_provenance[row] = view;
                }
            }
        }
        for (auto& child : bank.children) {
            for (std::uint32_t row = 0U; row < row_count; ++row) {
                const auto view = TargetStateView {
                    .execution_epoch = binding.execution_epoch,
                    .arena_epoch = binding.arena_epoch,
                    .bank_id = bank.spec.bank_id,
                    .child_id = child.spec.child_id,
                    .source_stage = final_tap.stage,
                    .source_layer = final_tap.layer_index,
                    .coordinate = {
                        .target_position = absolute_positions[row],
                        .draft_position = absolute_positions[row] + 1U,
                    },
                    .row_count = 1U,
                    .feature_width = child.spec.output_width,
                };
                if (child.spec.schedule == FeatureProjectionSchedule::eager) {
                    if (binding.destination == FeatureCaptureDestination::canonical) {
                        child.provenance[absolute_positions[row]] = view;
                    } else {
                        child.speculative_provenance[row] = view;
                    }
                }
            }
            if (child.spec.schedule == FeatureProjectionSchedule::eager) {
                impl.metrics.eager_projection_row_count += row_count;
            } else if (binding.destination == FeatureCaptureDestination::canonical) {
                impl.metrics.lazy_stale_row_count += row_count;
                const auto first = absolute_positions[0];
                child.stale_frontier = child.stale_frontier.has_value()
                    ? std::min(*child.stale_frontier, first) : first;
            }
        }
    }
    impl.metrics.captured_row_count += row_count;
    if (binding.destination == FeatureCaptureDestination::canonical) {
        impl.canonical_frontier = absolute_positions.back() + 1U;
    }
    impl.capture_active = false;
    impl.active_row_count = 0U;
    }
    if (binding.trace_sink != nullptr && binding.trace_sink->impl_) {
        auto& trace = *binding.trace_sink->impl_;
        const auto valid = trace.active && !trace.poisoned && row_count != 0U
            && row_count == trace.active_row_count
            && absolute_positions.size() == row_count
            && trace.active_execution_epoch == binding.execution_epoch
            && std::equal(
                absolute_positions.begin(), absolute_positions.end(),
                trace.active_positions.begin(), trace.active_positions.end()
            )
            && std::all_of(
                trace.captured.begin(),
                trace.captured.begin() + static_cast<std::ptrdiff_t>(trace.taps.size()),
                [](bool captured) { return captured; }
            );
        if (!valid) {
            trace.poisoned = true;
            trace.active = false;
            throw std::runtime_error("trace feature capture completion is invalid");
        }
        const auto row_bytes = static_cast<std::uint64_t>(trace.taps.size())
            * trace.hidden_width * sizeof(std::uint16_t);
        const auto byte_count = row_bytes * row_count;
        if (byte_count > std::numeric_limits<std::size_t>::max()) {
            trace.poisoned = true;
            trace.active = false;
            throw std::overflow_error("trace feature batch byte count overflow");
        }
        TargetFeatureTraceBatch batch {
            .execution_epoch = binding.execution_epoch,
            .absolute_positions = trace.active_positions,
            .input_token_ids = trace.active_token_ids,
            .taps = trace.taps,
            .hidden_width = trace.hidden_width,
            .dtype = DType::f16,
            .bytes = tensor::copy_backend_buffer_bytes(
                trace.rows, 0U,
                static_cast<std::size_t>(byte_count)
            ),
        };
        trace.active = false;
        trace.active_row_count = 0U;
        try {
            trace.callback(std::move(batch));
        } catch (...) {
            trace.poisoned = true;
            throw;
        }
    }
}

void FeatureSinkRuntimeAccess::encode_speculative_promotion(
    const FeatureSinkBinding& binding,
    MetalCommandContext& command,
    std::span<const std::uint32_t> selected_source_rows,
    std::span<const std::uint32_t> absolute_positions
) {
    if (binding.sinks == nullptr || !binding.sinks->impl_) return;
    auto& impl = *binding.sinks->impl_;
    const auto row_count = selected_source_rows.size();
    if (impl.poisoned || impl.capture_active || impl.promotion_active
        || binding.destination != FeatureCaptureDestination::speculative
        || binding.execution_epoch == 0U || binding.arena_epoch == 0U
        || row_count == 0U || row_count != absolute_positions.size()
        || row_count > impl.speculative_capacity
        || absolute_positions[0] != impl.canonical_frontier) {
        throw std::logic_error("speculative feature promotion lifecycle is invalid");
    }
    for (std::size_t row = 0U; row < row_count; ++row) {
        if (selected_source_rows[row] >= impl.speculative_capacity
            || absolute_positions[row] != absolute_positions[0] + row) {
            throw std::invalid_argument("speculative feature promotion path is invalid");
        }
        impl.source_row_values[row] = selected_source_rows[row];
        impl.destination_row_values[row] = absolute_positions[row];
        for (const auto& bank : impl.banks) {
            if (bank.lazy_speculative.has_value()) {
                const auto& raw = bank.lazy_speculative_provenance[selected_source_rows[row]];
                if (!raw.has_value()
                    || raw->execution_epoch != binding.execution_epoch
                    || raw->arena_epoch != binding.arena_epoch
                    || raw->coordinate.target_position != absolute_positions[row]) {
                    throw std::logic_error("speculative raw feature provenance is stale");
                }
            }
            for (const auto& child : bank.children) {
                if (child.spec.schedule == FeatureProjectionSchedule::lazy) continue;
                const auto& view = child.speculative_provenance[selected_source_rows[row]];
                if (!view.has_value()
                    || view->execution_epoch != binding.execution_epoch
                    || view->arena_epoch != binding.arena_epoch
                    || view->coordinate.target_position != absolute_positions[row]) {
                    throw std::logic_error("speculative child feature provenance is stale");
                }
            }
        }
    }
    command.write_u32(impl.status, 0U);
    command.write_u32s(
        impl.source_rows, 0U,
        std::span<const std::uint32_t>(impl.source_row_values).first(row_count)
    );
    command.write_u32s(
        impl.destination_rows, 0U,
        std::span<const std::uint32_t>(impl.destination_row_values).first(row_count)
    );
    for (auto& bank : impl.banks) {
        if (bank.lazy_canonical.has_value()) {
            command.encode_feature_scatter(
                *bank.lazy_speculative, impl.source_rows, impl.destination_rows,
                *bank.lazy_canonical, impl.status,
                static_cast<std::uint32_t>(row_count), bank.fused_width
            );
        }
        for (auto& child : bank.children) {
            if (child.spec.schedule == FeatureProjectionSchedule::lazy) continue;
            command.encode_f16_row_scatter(
                child.speculative, impl.source_rows, impl.destination_rows,
                child.canonical, impl.status, static_cast<std::uint32_t>(row_count)
            );
        }
    }
    impl.promotion_active = true;
    impl.promotion_row_count = static_cast<std::uint32_t>(row_count);
}

void FeatureSinkRuntimeAccess::complete_speculative_promotion(
    const FeatureSinkBinding& binding,
    MetalCommandContext& command
) {
    if (binding.sinks == nullptr || !binding.sinks->impl_) return;
    auto& impl = *binding.sinks->impl_;
    if (!impl.promotion_active || command.read_verifier_status(impl.status) != 0U) {
        impl.poisoned = true;
        impl.promotion_active = false;
        throw std::runtime_error("speculative feature promotion failed");
    }
    for (auto& bank : impl.banks) {
        for (std::uint32_t row = 0U; row < impl.promotion_row_count; ++row) {
            const auto source = impl.source_row_values[row];
            const auto destination = impl.destination_row_values[row];
            if (bank.lazy_canonical.has_value()) {
                bank.lazy_canonical_provenance[destination] =
                    bank.lazy_speculative_provenance[source];
            }
            for (auto& child : bank.children) {
                if (child.spec.schedule == FeatureProjectionSchedule::eager) {
                    child.provenance[destination] = child.speculative_provenance[source];
                } else {
                    child.provenance[destination].reset();
                    ++impl.metrics.lazy_stale_row_count;
                    child.stale_frontier = child.stale_frontier.has_value()
                        ? std::min(*child.stale_frontier, destination) : destination;
                }
            }
        }
    }
    impl.canonical_frontier += impl.promotion_row_count;
    impl.promotion_active = false;
    discard_speculative(binding);
}

void FeatureSinkRuntimeAccess::abort_promotion(const FeatureSinkBinding& binding) noexcept {
    if (binding.sinks == nullptr || !binding.sinks->impl_) return;
    binding.sinks->impl_->promotion_active = false;
    binding.sinks->impl_->promotion_row_count = 0U;
}

void FeatureSinkRuntimeAccess::discard_speculative(
    const FeatureSinkBinding& binding
) noexcept {
    if (binding.sinks == nullptr || !binding.sinks->impl_) return;
    auto& impl = *binding.sinks->impl_;
    for (auto& bank : impl.banks) {
        std::fill(
            bank.lazy_speculative_provenance.begin(),
            bank.lazy_speculative_provenance.end(), std::nullopt
        );
        for (auto& child : bank.children) {
            std::fill(
                child.speculative_provenance.begin(),
                child.speculative_provenance.end(), std::nullopt
            );
        }
    }
    impl.promotion_active = false;
    impl.promotion_row_count = 0U;
}

void FeatureSinkRuntimeAccess::abort_unsubmitted(const FeatureSinkBinding& binding) noexcept {
    if (binding.sinks != nullptr && binding.sinks->impl_) {
        binding.sinks->impl_->capture_active = false;
        binding.sinks->impl_->active_row_count = 0U;
    }
    if (binding.trace_sink != nullptr && binding.trace_sink->impl_) {
        binding.trace_sink->impl_->active = false;
        binding.trace_sink->impl_->active_row_count = 0U;
    }
}

void FeatureSinkRuntimeAccess::poison(const FeatureSinkBinding& binding) noexcept {
    if (binding.sinks != nullptr && binding.sinks->impl_) {
        binding.sinks->impl_->capture_active = false;
        binding.sinks->impl_->active_row_count = 0U;
        binding.sinks->impl_->promotion_active = false;
        binding.sinks->impl_->poisoned = true;
    }
    if (binding.trace_sink != nullptr && binding.trace_sink->impl_) {
        binding.trace_sink->impl_->active = false;
        binding.trace_sink->impl_->active_row_count = 0U;
        binding.trace_sink->impl_->poisoned = true;
    }
}

void FeatureSinkRuntimeAccess::truncate(FeatureSinkSet& sinks, std::uint32_t frontier) {
    if (!sinks.impl_ || sinks.impl_->poisoned || frontier > sinks.impl_->canonical_frontier) {
        throw std::logic_error("feature sink truncate frontier is invalid");
    }
    for (auto& bank : sinks.impl_->banks) {
        if (!bank.lazy_canonical_provenance.empty()) {
            std::fill(
                bank.lazy_canonical_provenance.begin() + frontier,
                bank.lazy_canonical_provenance.end(), std::nullopt
            );
        }
        for (auto& child : bank.children) {
            std::fill(child.provenance.begin() + frontier, child.provenance.end(), std::nullopt);
            if (child.stale_frontier.has_value() && *child.stale_frontier >= frontier) {
                child.stale_frontier.reset();
            }
        }
    }
    sinks.impl_->canonical_frontier = frontier;
}

DraftCacheArena create_draft_cache_arena(
    const DraftCacheSpec& spec,
    std::uint32_t target_context_length
) {
    if (!tensor::tensor_backend_is_available(BackendKind::metal)) {
        throw std::runtime_error("draft cache requires Metal");
    }
    if (spec.layer_count == 0U || spec.layer_count > 64U
        || spec.stream_count == 0U || spec.stream_count > 256U
        || spec.key_width == 0U || spec.key_width > 4'096U
        || spec.value_width == 0U || spec.value_width > 4'096U
        || spec.maximum_depth == 0U || spec.maximum_depth > 23U
        || spec.maximum_active_width == 0U || spec.maximum_active_width > 256U
        || spec.linear_row_capacity == 0U || spec.linear_row_capacity > 23U
        || spec.tree_row_capacity == 0U || spec.tree_row_capacity > 1'025U
        || target_context_length == 0U) {
        throw std::invalid_argument("draft cache spec is invalid");
    }
    DraftCacheArena arena;
    arena.spec_ = spec;
    const auto staging_row_capacity = std::max(
        spec.tree_row_capacity, spec.linear_row_capacity + 1U
    );
    arena.source_rows_ = allocate(
        descriptor(staging_row_capacity, 1U, DType::f32), arena.allocated_bytes_
    );
    arena.destination_rows_ = allocate(
        descriptor(staging_row_capacity, 1U, DType::f32), arena.allocated_bytes_
    );
    arena.status_ = allocate(
        descriptor(1U, 1U, DType::f32), arena.allocated_bytes_
    );
    const auto layer_count = static_cast<std::size_t>(spec.layer_count)
        * spec.stream_count;
    arena.layers_.reserve(layer_count);
    for (std::uint16_t layer = 0U; layer < spec.layer_count; ++layer) {
        for (std::uint16_t stream = 0U; stream < spec.stream_count; ++stream) {
            arena.layers_.push_back(DraftCacheLayerView {
                .layer_index = layer,
                .stream_index = stream,
                .canonical_key = allocate(
                    descriptor(target_context_length, spec.key_width, DType::f16),
                    arena.allocated_bytes_
                ),
                .canonical_value = allocate(
                    descriptor(target_context_length, spec.value_width, DType::f16),
                    arena.allocated_bytes_
                ),
                .linear_key = allocate(
                    descriptor(
                        static_cast<std::uint64_t>(spec.linear_row_capacity) + 1U,
                        spec.key_width, DType::f16
                    ),
                    arena.allocated_bytes_
                ),
                .linear_value = allocate(
                    descriptor(
                        static_cast<std::uint64_t>(spec.linear_row_capacity) + 1U,
                        spec.value_width, DType::f16
                    ),
                    arena.allocated_bytes_
                ),
                .tree_key = allocate(
                    descriptor(spec.tree_row_capacity, spec.key_width, DType::f16),
                    arena.allocated_bytes_
                ),
                .tree_value = allocate(
                    descriptor(spec.tree_row_capacity, spec.value_width, DType::f16),
                    arena.allocated_bytes_
                ),
            });
        }
    }
    return arena;
}

const DraftCacheSpec& DraftCacheArena::spec() const noexcept { return spec_; }
std::span<const DraftCacheLayerView> DraftCacheArena::layers() const noexcept { return layers_; }
std::uint32_t DraftCacheArena::target_frontier() const noexcept { return target_frontier_; }
std::uint32_t DraftCacheArena::draft_frontier() const noexcept { return draft_frontier_; }
std::size_t DraftCacheArena::allocated_bytes() const noexcept { return allocated_bytes_; }
bool DraftCacheArena::poisoned() const noexcept { return poisoned_; }
std::optional<DraftCacheStateView> DraftCacheArena::state_view(
    std::uint32_t target_position
) const noexcept {
    if (poisoned_ || target_position >= target_frontier_ || arena_epoch_ == 0U) {
        return std::nullopt;
    }
    return DraftCacheStateView {
        .arena_epoch = arena_epoch_,
        .target_position = target_position,
        .draft_position = target_position + 1U,
        .target_frontier = target_frontier_,
    };
}

tensor::TensorStorage DraftCacheArena::copy_canonical_key(
    std::uint16_t layer_index,
    std::uint16_t stream_index
) const {
    if (poisoned_ || layer_index >= spec_.layer_count
        || stream_index >= spec_.stream_count) {
        throw std::logic_error("draft cache key view is invalid");
    }
    const auto index = static_cast<std::size_t>(layer_index) * spec_.stream_count
        + stream_index;
    return tensor::copy_tensor_from_backend(layers_[index].canonical_key);
}

tensor::TensorStorage DraftCacheArena::copy_canonical_value(
    std::uint16_t layer_index,
    std::uint16_t stream_index
) const {
    if (poisoned_ || layer_index >= spec_.layer_count
        || stream_index >= spec_.stream_count) {
        throw std::logic_error("draft cache value view is invalid");
    }
    const auto index = static_cast<std::size_t>(layer_index) * spec_.stream_count
        + stream_index;
    return tensor::copy_tensor_from_backend(layers_[index].canonical_value);
}

void DraftCacheArena::reset() noexcept {
    if (poisoned_) return;
    target_frontier_ = 0U;
    draft_frontier_ = 1U;
    arena_epoch_ = 0U;
    promotion_row_count_ = 0U;
    promotion_active_ = false;
}

void DraftCacheRuntimeAccess::validate_binding(
    const DraftCacheBinding& binding,
    std::uint64_t expected_arena_epoch,
    std::uint32_t required_rows
) {
    if (binding.arena == nullptr) return;
    const auto& arena = *binding.arena;
    const auto capacity = binding.scratch == DraftCacheScratchKind::linear
        ? arena.spec_.linear_row_capacity + 1U : arena.spec_.tree_row_capacity;
    if (arena.poisoned_ || arena.promotion_active_
        || binding.arena_epoch == 0U
        || binding.arena_epoch != expected_arena_epoch
        || required_rows == 0U || required_rows > capacity
        || (arena.arena_epoch_ != 0U && arena.arena_epoch_ != binding.arena_epoch)) {
        throw std::logic_error("draft cache binding is stale or exceeds its shape");
    }
}

void DraftCacheRuntimeAccess::encode_promotion(
    const DraftCacheBinding& binding,
    MetalCommandContext& command,
    std::span<const std::uint32_t> selected_source_rows,
    std::span<const std::uint32_t> absolute_positions
) {
    if (binding.arena == nullptr) return;
    auto& arena = *binding.arena;
    const auto row_count = selected_source_rows.size();
    const auto capacity = binding.scratch == DraftCacheScratchKind::linear
        ? arena.spec_.linear_row_capacity + 1U : arena.spec_.tree_row_capacity;
    if (arena.poisoned_ || arena.promotion_active_
        || binding.arena_epoch == 0U || row_count == 0U
        || row_count != absolute_positions.size() || row_count > capacity
        || absolute_positions[0] != arena.target_frontier_) {
        throw std::logic_error("draft cache promotion lifecycle is invalid");
    }
    for (std::size_t row = 0U; row < row_count; ++row) {
        if (selected_source_rows[row] >= capacity
            || absolute_positions[row] != absolute_positions[0] + row) {
            throw std::invalid_argument("draft cache selected path is invalid");
        }
        arena.source_row_values_[row] = selected_source_rows[row];
        arena.destination_row_values_[row] = absolute_positions[row];
    }
    command.write_u32(*arena.status_, 0U);
    command.write_u32s(
        *arena.source_rows_, 0U,
        std::span<const std::uint32_t>(arena.source_row_values_).first(row_count)
    );
    command.write_u32s(
        *arena.destination_rows_, 0U,
        std::span<const std::uint32_t>(arena.destination_row_values_).first(row_count)
    );
    for (auto& layer : arena.layers_) {
        const auto& source_key = binding.scratch == DraftCacheScratchKind::linear
            ? layer.linear_key : layer.tree_key;
        const auto& source_value = binding.scratch == DraftCacheScratchKind::linear
            ? layer.linear_value : layer.tree_value;
        command.encode_f16_row_scatter(
            source_key, *arena.source_rows_, *arena.destination_rows_,
            layer.canonical_key, *arena.status_, static_cast<std::uint32_t>(row_count)
        );
        command.encode_f16_row_scatter(
            source_value, *arena.source_rows_, *arena.destination_rows_,
            layer.canonical_value, *arena.status_, static_cast<std::uint32_t>(row_count)
        );
    }
    arena.promotion_row_count_ = static_cast<std::uint32_t>(row_count);
    arena.promotion_active_ = true;
}

void DraftCacheRuntimeAccess::complete_promotion(
    const DraftCacheBinding& binding,
    MetalCommandContext& command
) {
    if (binding.arena == nullptr) return;
    auto& arena = *binding.arena;
    if (!arena.promotion_active_ || command.read_verifier_status(*arena.status_) != 0U) {
        arena.poisoned_ = true;
        arena.promotion_active_ = false;
        throw std::runtime_error("draft cache promotion failed");
    }
    arena.target_frontier_ += arena.promotion_row_count_;
    arena.draft_frontier_ = arena.target_frontier_ + 1U;
    arena.arena_epoch_ = binding.arena_epoch;
    arena.promotion_row_count_ = 0U;
    arena.promotion_active_ = false;
}

void DraftCacheRuntimeAccess::abort_promotion(const DraftCacheBinding& binding) noexcept {
    if (binding.arena == nullptr) return;
    binding.arena->promotion_row_count_ = 0U;
    binding.arena->promotion_active_ = false;
}

void DraftCacheRuntimeAccess::truncate(DraftCacheArena& arena, std::uint32_t frontier) {
    if (arena.poisoned_ || arena.promotion_active_ || frontier > arena.target_frontier_) {
        throw std::logic_error("draft cache truncate frontier is invalid");
    }
    arena.target_frontier_ = frontier;
    arena.draft_frontier_ = frontier + 1U;
}

void DraftCacheRuntimeAccess::poison(const DraftCacheBinding& binding) noexcept {
    if (binding.arena == nullptr) return;
    binding.arena->promotion_active_ = false;
    binding.arena->poisoned_ = true;
}

}  // namespace local_gguf_engine::engine
