#pragma once

#include "local_gguf_engine/tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <thread>

namespace local_gguf_engine::tensor::detail {

inline void validate_preparation_destination(TensorView source,
    const TensorDescriptor& destination, const TensorDescriptor& expected,
    std::span<std::uint8_t> bytes) {
    if (destination != expected || bytes.size() != expected.byte_count)
        throw std::invalid_argument("weight preparation destination descriptor or extent is inconsistent");
    const auto input = reinterpret_cast<std::uintptr_t>(source.bytes().data());
    const auto output = reinterpret_cast<std::uintptr_t>(bytes.data());
    if (input <= output ? output - input < source.bytes().size() : input - output < bytes.size())
        throw std::invalid_argument("weight preparation source and destination overlap");
}

// Preparation is synchronous and block-local. Static contiguous ranges avoid
// an atomic queue per block and keep each worker's destination writes together.
// Four workers match this engine's M4 performance-core preparation policy.
// jthread also joins already-started workers if thread creation itself fails.
template <typename Function>
void prepare_weight_ranges(std::size_t count, std::size_t parallel_threshold, Function&& function) {
    if (count < parallel_threshold) {
        function(0U, count);
        return;
    }
    constexpr std::size_t worker_count = 4U;
    std::array<std::exception_ptr, worker_count> errors {};
    std::array<std::jthread, worker_count> workers;
    const auto width = count / worker_count;
    const auto remainder = count % worker_count;
    for (std::size_t worker = 0U; worker < worker_count; ++worker) {
        const auto begin = worker * width + std::min(worker, remainder);
        const auto end = begin + width + (worker < remainder ? 1U : 0U);
        workers[worker] = std::jthread([&, worker, begin, end] {
            try {
                function(begin, end);
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

}  // namespace local_gguf_engine::tensor::detail
