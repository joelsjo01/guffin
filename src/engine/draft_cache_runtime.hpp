#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"

#include <cstdint>
#include <span>

namespace local_gguf_engine::tensor::backends::metal {
class MetalCommandContext;
}

namespace local_gguf_engine::engine {

struct DraftCacheRuntimeAccess final {
    static void validate_binding(
        const DraftCacheBinding& binding,
        std::uint64_t expected_arena_epoch,
        std::uint32_t required_rows
    );
    static void encode_promotion(
        const DraftCacheBinding& binding,
        tensor::backends::metal::MetalCommandContext& command,
        std::span<const std::uint32_t> selected_source_rows,
        std::span<const std::uint32_t> absolute_positions
    );
    static void complete_promotion(
        const DraftCacheBinding& binding,
        tensor::backends::metal::MetalCommandContext& command
    );
    static void abort_promotion(const DraftCacheBinding& binding) noexcept;
    static void truncate(DraftCacheArena& arena, std::uint32_t frontier);
    static void poison(const DraftCacheBinding& binding) noexcept;
};

}  // namespace local_gguf_engine::engine
