#include "metal_backend.hpp"

#include "metal/metal_context.hpp"
#include "metal/metal_buffer.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

namespace local_gguf_engine::tensor::backends::metal {


bool is_available() noexcept {
    return system_metal_device() != nil;
}

std::string unavailable_reason() {
    if (system_metal_device() != nil) {
        return {};
    }

    return "Metal backend unavailable: no system default Metal device";
}

std::optional<MetalMemoryInfo> memory_info() noexcept {
    @autoreleasepool {
        id<MTLDevice> device = system_metal_device();
        if (device == nil) return std::nullopt;
        return MetalMemoryInfo {
            .physical_memory_bytes = NSProcessInfo.processInfo.physicalMemory,
            .recommended_working_set_bytes = device.recommendedMaxWorkingSetSize,
            .max_buffer_bytes = device.maxBufferLength,
            .current_allocated_bytes = device.currentAllocatedSize,
        };
    }
}

std::optional<std::uint64_t> current_allocated_size_bytes() noexcept {
    @autoreleasepool {
        id<MTLDevice> device = system_metal_device();
        if (device == nil) return std::nullopt;
        return static_cast<std::uint64_t>(device.currentAllocatedSize);
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal
