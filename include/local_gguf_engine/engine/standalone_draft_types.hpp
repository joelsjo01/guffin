#pragma once

#include <cstdint>

namespace local_gguf_engine::engine {

// Work performed by the native draft executor, including synchronization and
// transfers. Target verification and cache publication are counted separately.
struct StandaloneDraftExecutionStats {
    std::uint64_t gpu_duration_ns = 0;
    std::uint64_t wall_duration_ns = 0;
    std::uint64_t readback_bytes = 0;
    std::uint64_t uploaded_bytes = 0;
    std::uint32_t command_buffer_count = 0;
    std::uint32_t compute_dispatch_count = 0;
    std::uint32_t memory_barrier_count = 0;
    std::uint32_t token_count = 0;
};

} // namespace local_gguf_engine::engine
