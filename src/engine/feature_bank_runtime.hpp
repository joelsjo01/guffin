#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"

#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor::backends::metal {
class MetalCommandContext;
}

namespace local_gguf_engine::engine {

struct FeatureSinkRuntimeAccess final {
    [[nodiscard]] static bool needs_tap(
        const FeatureSinkBinding& binding,
        FeatureTapStage stage,
        std::uint16_t layer_index
    );
    static void begin_capture(
        const FeatureSinkBinding& binding,
        tensor::backends::metal::MetalCommandContext& command,
        std::span<const std::uint32_t> absolute_positions,
        std::span<const std::uint32_t> input_token_ids,
        std::uint32_t row_count
    );
    static void encode_tap(
        const FeatureSinkBinding& binding,
        tensor::backends::metal::MetalCommandContext& command,
        FeatureTapStage stage,
        std::uint16_t layer_index,
        const tensor::BackendBuffer& input,
        std::uint32_t input_row_start,
        std::uint32_t row_count
    );
    static void complete_capture(
        const FeatureSinkBinding& binding,
        tensor::backends::metal::MetalCommandContext& command,
        std::span<const std::uint32_t> absolute_positions,
        std::uint32_t row_count
    );
    static void encode_speculative_promotion(
        const FeatureSinkBinding& binding,
        tensor::backends::metal::MetalCommandContext& command,
        std::span<const std::uint32_t> selected_source_rows,
        std::span<const std::uint32_t> absolute_positions
    );
    static void complete_speculative_promotion(
        const FeatureSinkBinding& binding,
        tensor::backends::metal::MetalCommandContext& command
    );
    static void abort_promotion(const FeatureSinkBinding& binding) noexcept;
    static void discard_speculative(const FeatureSinkBinding& binding) noexcept;
    static void abort_unsubmitted(const FeatureSinkBinding& binding) noexcept;
    static void poison(const FeatureSinkBinding& binding) noexcept;
    static void truncate(FeatureSinkSet& sinks, std::uint32_t frontier);
};

}  // namespace local_gguf_engine::engine
