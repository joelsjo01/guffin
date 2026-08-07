#pragma once

#include <cstdint>
#include <string>

namespace local_gguf_engine::tensor::backends::metal {

enum class MetalTraceCaptureStatus : std::uint8_t {
    idle,
    armed,
    active,
    completed,
};

// Arms a one-shot capture of the next production command buffer whose label is
// "Gemma2 resident decode". Capture begins before that command buffer is
// created and ends after it completes.
void arm_next_gemma2_decode_trace(std::string output_path);
[[nodiscard]] MetalTraceCaptureStatus metal_trace_capture_status() noexcept;
void reset_completed_metal_trace_capture();
void cancel_armed_metal_trace_capture();

}  // namespace local_gguf_engine::tensor::backends::metal
