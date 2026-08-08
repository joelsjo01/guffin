#include "local_gguf_engine/benchmarks/immutable_weight_benchmark.hpp"

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/backend_buffer.hpp"

#include "../tensor/backends/metal/metal_buffer.hpp"
#include "../tensor/backends/metal/metal_command_context.hpp"
#include "../tensor/backends/metal/metal_context.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

namespace local_gguf_engine::benchmarks {
namespace {

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

using tensor::BackendBuffer;
using tensor::TensorDescriptor;
using tensor::backends::metal::MetalCommandContext;
using tensor::backends::metal::make_metal_backend_buffer;
using tensor::backends::metal::make_metal_backend_buffer_view;
using tensor::backends::metal::shared_matmul_pipeline_state;

struct ResourceOptions {
    MTLResourceOptions value;
    MTLStorageMode storage_mode;
    MTLCPUCacheMode cache_mode;
    MTLHazardTrackingMode hazard_mode;
};

struct UploadedWeight {
    BackendBuffer buffer;
    std::uint64_t wall_duration_ns = 0;
    std::uint64_t gpu_duration_ns = 0;
};

[[nodiscard]] std::uint64_t checked_round_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::invalid_argument("no-copy VM page size must be a power of two");
    }
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U)) {
        throw std::overflow_error("no-copy mapped extent rounding overflows");
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

[[nodiscard]] bool mapping_is_one_vm_region(const void* base, std::uint64_t extent) {
    mach_vm_address_t region_address = reinterpret_cast<mach_vm_address_t>(base);
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info {};
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    const auto result = mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &info_count, &object_name
    );
    if (object_name != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object_name);
    if (result != KERN_SUCCESS || (info.protection & VM_PROT_READ) == 0) return false;
    const auto target = reinterpret_cast<mach_vm_address_t>(base);
    return region_address <= target && target - region_address <= region_size
        && extent <= region_size - (target - region_address);
}

[[nodiscard]] ResourceOptions resource_options(ImmutableWeightPolicy policy) {
    MTLStorageMode storage = MTLStorageModeShared;
    MTLCPUCacheMode cache = MTLCPUCacheModeDefaultCache;
    MTLHazardTrackingMode hazard = MTLHazardTrackingModeTracked;
    switch (policy) {
        case ImmutableWeightPolicy::shared_default_tracked:
            break;
        case ImmutableWeightPolicy::shared_write_combined_tracked:
            cache = MTLCPUCacheModeWriteCombined;
            break;
        case ImmutableWeightPolicy::shared_default_untracked:
            hazard = MTLHazardTrackingModeUntracked;
            break;
        case ImmutableWeightPolicy::shared_write_combined_untracked:
            cache = MTLCPUCacheModeWriteCombined;
            hazard = MTLHazardTrackingModeUntracked;
            break;
        case ImmutableWeightPolicy::private_tracked:
            storage = MTLStorageModePrivate;
            break;
        case ImmutableWeightPolicy::private_untracked:
            storage = MTLStorageModePrivate;
            hazard = MTLHazardTrackingModeUntracked;
            break;
    }
    return {
        .value = static_cast<MTLResourceOptions>(
            storage << MTLResourceStorageModeShift
            | cache << MTLResourceCPUCacheModeShift
            | hazard << MTLResourceHazardTrackingModeShift
        ),
        .storage_mode = storage,
        .cache_mode = cache,
        .hazard_mode = hazard,
    };
}

void require_actual_options(id<MTLBuffer> buffer, const ResourceOptions& expected) {
    if (buffer.storageMode != expected.storage_mode
        || buffer.cpuCacheMode != expected.cache_mode
        || buffer.hazardTrackingMode != expected.hazard_mode) {
        throw std::runtime_error("Metal immutable-weight resource properties differ from request");
    }
}

[[nodiscard]] std::uint64_t command_gpu_duration(id<MTLCommandBuffer> command) {
    const auto start = command.GPUStartTime;
    const auto end = command.GPUEndTime;
    if (start <= 0.0 || end < start) {
        throw std::runtime_error("Metal immutable-weight upload reported zero GPU timing");
    }
    return static_cast<std::uint64_t>((end - start) * 1.0e9);
}

[[nodiscard]] UploadedWeight upload_weight(
    const tensor::TensorView& view,
    ImmutableWeightPolicy policy
) {
    @autoreleasepool {
        auto& state = shared_matmul_pipeline_state();
        const auto options = resource_options(policy);
        const auto begin = std::chrono::steady_clock::now();
        id<MTLBuffer> destination = nil;
        std::uint64_t gpu_duration_ns = 0;
        if (options.storage_mode == MTLStorageModePrivate) {
            id<MTLBuffer> staging = [state.device newBufferWithBytes:view.bytes().data()
                                                               length:view.bytes().size()
                                                              options:MTLResourceStorageModeShared];
            destination = [state.device newBufferWithLength:view.bytes().size() options:options.value];
            if (staging == nil || destination == nil) {
                throw std::runtime_error("Metal immutable-weight private allocation failed");
            }
            id<MTLCommandBuffer> command = [state.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (command == nil || blit == nil) {
                throw std::runtime_error("Metal immutable-weight upload command creation failed");
            }
            [blit copyFromBuffer:staging
                    sourceOffset:0
                        toBuffer:destination
               destinationOffset:0
                            size:view.bytes().size()];
            [blit endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                throw std::runtime_error("Metal immutable-weight private upload failed");
            }
            gpu_duration_ns = command_gpu_duration(command);
            staging = nil;
        } else {
            destination = [state.device newBufferWithBytes:view.bytes().data()
                                                     length:view.bytes().size()
                                                    options:options.value];
            if (destination == nil) {
                throw std::runtime_error("Metal immutable-weight shared allocation failed");
            }
        }
        require_actual_options(destination, options);
        const auto end = std::chrono::steady_clock::now();
        return {
            .buffer = make_metal_backend_buffer(view.descriptor(), destination),
            .wall_duration_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
            ),
            .gpu_duration_ns = gpu_duration_ns,
        };
    }
}

void require_byte_parity(const tensor::TensorView& source, const BackendBuffer& candidate) {
    @autoreleasepool {
        const auto& storage = tensor::backends::metal::require_metal_backend_buffer(
            tensor::backend_buffer_storage(candidate)
        );
        id<MTLBuffer> readback = nil;
        if (storage.buffer().storageMode == MTLStorageModePrivate) {
            auto& state = shared_matmul_pipeline_state();
            readback = [state.device newBufferWithLength:source.bytes().size()
                                             options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = [state.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (readback == nil || command == nil || blit == nil) {
                throw std::runtime_error("Metal immutable-weight parity readback allocation failed");
            }
            [blit copyFromBuffer:storage.buffer()
                    sourceOffset:0
                        toBuffer:readback
               destinationOffset:0
                            size:source.bytes().size()];
            [blit endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                throw std::runtime_error("Metal immutable-weight parity readback failed");
            }
        } else {
            readback = storage.buffer();
        }
        const void* bytes = storage.buffer().storageMode == MTLStorageModePrivate
            ? readback.contents : tensor::backends::metal::metal_buffer_contents(storage);
        if (std::memcmp(source.bytes().data(), bytes, source.bytes().size()) != 0) {
            throw std::runtime_error("Metal immutable-weight copied-byte parity mismatch");
        }
    }
}

[[nodiscard]] std::uint64_t run_kernel_command(
    std::string_view kernel_name,
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t dispatch_count
) {
    MetalCommandContext context("Immutable weight policy diagnostic");
    for (std::uint32_t dispatch = 0; dispatch < dispatch_count; ++dispatch) {
        context.encode_matmul(input, weight, output, 1U);
        if (dispatch + 1U < dispatch_count) context.encode_memory_barrier();
    }
    const auto stats = context.commit_and_wait();
    if (stats.command_buffer_count != 1U || stats.compute_encoder_count != 1U
        || stats.compute_dispatch_count != dispatch_count
        || stats.memory_barrier_count != dispatch_count - 1U
        || stats.blit_encoder_count != 0U || stats.commit_count != 1U
        || stats.synchronous_wait_count != 1U
        || stats.transient_buffer_allocation_count != 0U) {
        throw std::runtime_error(
            "immutable-weight command accounting mismatch for " + std::string(kernel_name)
        );
    }
    if (stats.gpu_duration_ns == 0U) {
        throw std::runtime_error(
            "immutable-weight kernel reported zero GPU timing for " + std::string(kernel_name)
        );
    }
    return stats.gpu_duration_ns;
}

[[nodiscard]] QuickKernelMeasurement measure(
    std::string_view name,
    QuickKernelMeasurementMode mode,
    const BackendBuffer& input,
    const BackendBuffer& weight,
    const BackendBuffer& output,
    std::uint32_t dispatch_count
) {
    static_cast<void>(run_kernel_command(name, input, weight, output, dispatch_count));
    std::vector<std::uint64_t> raw;
    raw.reserve(quick_kernel_repetition_count);
    for (std::size_t repetition = 0; repetition < quick_kernel_repetition_count; ++repetition) {
        raw.push_back(run_kernel_command(name, input, weight, output, dispatch_count));
    }
    return summarize_kernel_measurement(
        mode, dispatch_count, weight.descriptor().byte_count, raw
    );
}

void require_matmul_output_parity(
    const tensor::TensorView& source,
    const BackendBuffer& candidate,
    std::string_view name
) {
    const auto shared = source.descriptor().shape.dimension(0);
    const auto columns = source.descriptor().shape.dimension(1);
    const auto input_descriptor = TensorDescriptor::from(
        tensor::Shape::from({1U, shared}), tensor::DType::f32
    );
    auto input_host = tensor::TensorStorage::allocate(input_descriptor);
    for (std::uint64_t index = 0; index < shared; ++index) {
        const float value = static_cast<float>(static_cast<int>(index % 17U) - 8) * 0.03125F;
        std::memcpy(
            input_host.mutable_bytes().data() + index * sizeof(float), &value, sizeof(value)
        );
    }
    auto input = tensor::copy_tensor_to_backend(input_host, runtime::BackendKind::metal);
    const auto output_descriptor = TensorDescriptor::from(
        tensor::Shape::from({1U, columns}), tensor::DType::f32
    );
    auto candidate_output = tensor::allocate_backend_buffer(
        output_descriptor, runtime::BackendKind::metal
    );
    auto control_output = tensor::allocate_backend_buffer(
        output_descriptor, runtime::BackendKind::metal
    );
    auto control = upload_weight(source, ImmutableWeightPolicy::shared_default_tracked);
    static_cast<void>(run_kernel_command(name, input, candidate, candidate_output, 1U));
    static_cast<void>(run_kernel_command(name, input, control.buffer, control_output, 1U));
    const auto candidate_host = tensor::copy_tensor_from_backend(candidate_output);
    const auto control_host = tensor::copy_tensor_from_backend(control_output);
    if (candidate_host.bytes().size() != control_host.bytes().size()
        || std::memcmp(
            candidate_host.bytes().data(), control_host.bytes().data(), candidate_host.bytes().size()
        ) != 0) {
        throw std::runtime_error(
            "Metal no-copy real-offset matmul parity mismatch for " + std::string(name)
        );
    }
}

[[nodiscard]] std::string shape_label(const TensorDescriptor& descriptor) {
    return std::to_string(descriptor.shape.dimension(0)) + "x"
        + std::to_string(descriptor.shape.dimension(1));
}

[[nodiscard]] const engine::EngineTensorView& require_representative(
    const engine::EngineModel& model,
    tensor::DType dtype,
    std::uint64_t first,
    std::uint64_t second,
    std::string_view label
) {
    const auto found = std::find_if(
        model.tensor_views().begin(), model.tensor_views().end(),
        [&](const engine::EngineTensorView& tensor_view) {
            const auto& descriptor = tensor_view.view.descriptor();
            return descriptor.dtype == dtype && descriptor.shape.rank() == 2U
                && descriptor.shape.dimension(0) == first
                && descriptor.shape.dimension(1) == second;
        }
    );
    if (found == model.tensor_views().end()) {
        throw std::runtime_error("benchmark model is missing representative " + std::string(label));
    }
    return *found;
}

[[nodiscard]] std::uint64_t scheduled_class_bytes(
    const engine::EngineModel& model,
    std::string_view class_name
) {
    std::uint64_t total = 0;
    for (const auto& tensor_view : model.tensor_views()) {
        const auto& name = tensor_view.name;
        const auto dtype = tensor_view.view.descriptor().dtype;
        bool matches = false;
        if (class_name == "Q4_K gate/up") {
            matches = dtype == tensor::DType::q4_k
                && (name.find(".ffn_gate.weight") != std::string::npos
                    || name.find(".ffn_up.weight") != std::string::npos);
        } else if (class_name == "Q4_K down") {
            matches = dtype == tensor::DType::q4_k
                && name.find(".ffn_down.weight") != std::string::npos;
        } else if (class_name == "Q6_K down") {
            matches = dtype == tensor::DType::q6_k
                && name.find(".ffn_down.weight") != std::string::npos;
        } else if (class_name == "F16 output") {
            matches = name == "token_embd.weight";
        }
        if (!matches) continue;
        const auto bytes = static_cast<std::uint64_t>(tensor_view.view.bytes().size());
        if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error("scheduled class byte accounting overflows");
        }
        total += bytes;
    }
    if (total == 0U) throw std::runtime_error("scheduled class has no target weight bytes");
    return total;
}

}  // namespace

std::string_view immutable_weight_policy_name(ImmutableWeightPolicy policy) {
    switch (policy) {
        case ImmutableWeightPolicy::shared_default_tracked: return "shared-default-tracked";
        case ImmutableWeightPolicy::shared_write_combined_tracked: return "shared-write-combined-tracked";
        case ImmutableWeightPolicy::shared_default_untracked: return "shared-default-untracked";
        case ImmutableWeightPolicy::shared_write_combined_untracked: return "shared-write-combined-untracked";
        case ImmutableWeightPolicy::private_tracked: return "private-tracked";
        case ImmutableWeightPolicy::private_untracked: return "private-untracked";
    }
    throw std::invalid_argument("immutable-weight policy is invalid");
}

ImmutableWeightPolicy parse_immutable_weight_policy(std::string_view value) {
    for (const auto policy : {
             ImmutableWeightPolicy::shared_default_tracked,
             ImmutableWeightPolicy::shared_write_combined_tracked,
             ImmutableWeightPolicy::shared_default_untracked,
             ImmutableWeightPolicy::shared_write_combined_untracked,
             ImmutableWeightPolicy::private_tracked,
             ImmutableWeightPolicy::private_untracked,
         }) {
        if (value == immutable_weight_policy_name(policy)) return policy;
    }
    throw std::invalid_argument("unknown immutable-weight policy: " + std::string(value));
}

ImmutableWeightPolicyDescription describe_immutable_weight_policy(
    ImmutableWeightPolicy policy,
    std::uint64_t logical_tensor_bytes,
    std::uint64_t upload_wall_duration_ns,
    std::uint64_t upload_gpu_duration_ns
) {
    const auto options = resource_options(policy);
    return {
        .label = std::string(immutable_weight_policy_name(policy)),
        .storage_mode = options.storage_mode == MTLStorageModePrivate ? "private" : "shared",
        .cpu_cache_mode = options.cache_mode == MTLCPUCacheModeWriteCombined
            ? "write-combined" : "default",
        .hazard_mode = options.hazard_mode == MTLHazardTrackingModeUntracked
            ? "untracked" : "tracked",
        .parent_allocation_count = 1U,
        .logical_tensor_bytes = logical_tensor_bytes,
        .staging_bytes = options.storage_mode == MTLStorageModePrivate ? logical_tensor_bytes : 0U,
        .copied_bytes = logical_tensor_bytes,
        .file_backed_bytes = 0U,
        .upload_wall_duration_ns = upload_wall_duration_ns,
        .upload_gpu_duration_ns = upload_gpu_duration_ns,
        .cpu_contents_accessible = options.storage_mode != MTLStorageModePrivate,
    };
}

void validate_no_copy_mapping_constraints(
    std::uintptr_t base_address,
    std::uint64_t file_byte_count,
    std::uint64_t mapped_extent_byte_count,
    std::uint64_t page_size,
    std::uint64_t maximum_buffer_length,
    bool single_vm_region
) {
    if (file_byte_count == 0U) throw std::invalid_argument("no-copy file mapping is empty");
    if (page_size == 0U || (page_size & (page_size - 1U)) != 0U) {
        throw std::invalid_argument("no-copy VM page size must be a power of two");
    }
    if (base_address % page_size != 0U) {
        throw std::invalid_argument("no-copy mapped base is not page aligned");
    }
    if (mapped_extent_byte_count % page_size != 0U) {
        throw std::invalid_argument("no-copy mapped extent is not page aligned");
    }
    if (mapped_extent_byte_count < file_byte_count) {
        throw std::out_of_range("no-copy mapped extent is shorter than the file");
    }
    if (mapped_extent_byte_count > maximum_buffer_length) {
        throw std::out_of_range("no-copy mapped extent exceeds Metal maxBufferLength");
    }
    if (!single_vm_region) {
        throw std::invalid_argument("no-copy mapped extent is not one readable VM region");
    }
}

ImmutableWeightBenchmarkResult run_no_copy_weight_feasibility_benchmark(
    const std::string& model_path
) {
    if (model_path.empty()) throw std::invalid_argument("benchmark model path must not be empty");
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    auto& state = shared_matmul_pipeline_state();
    const long raw_page_size = sysconf(_SC_PAGESIZE);
    if (raw_page_size <= 0) throw std::runtime_error("could not query VM page size");
    const auto page_size = static_cast<std::uint64_t>(raw_page_size);
    const auto extent = checked_round_up(file_model.bytes().size(), page_size);
    const bool one_region = mapping_is_one_vm_region(file_model.bytes().data(), extent);
    validate_no_copy_mapping_constraints(
        reinterpret_cast<std::uintptr_t>(file_model.bytes().data()),
        file_model.bytes().size(), extent, page_size, state.device.maxBufferLength, one_region
    );
    const auto begin = std::chrono::steady_clock::now();
    id<MTLBuffer> parent = [state.device
        newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(file_model.bytes().data())
                         length:static_cast<NSUInteger>(extent)
                        options:(MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked)
                    deallocator:nil];
    const auto end = std::chrono::steady_clock::now();
    if (parent == nil) throw std::runtime_error("Metal rejected the read-only file-backed no-copy mapping");
    if (parent.length != extent || parent.storageMode != MTLStorageModeShared
        || parent.cpuCacheMode != MTLCPUCacheModeDefaultCache
        || parent.hazardTrackingMode != MTLHazardTrackingModeTracked) {
        throw std::runtime_error("Metal no-copy parent properties differ from request");
    }

    const auto& config = model.gemma2_config();
    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto feed_forward = static_cast<std::uint64_t>(config.feed_forward_length);
    const std::vector<std::pair<std::string, const engine::EngineTensorView*>> representatives {
        {"Q4_K gate/up", &require_representative(model, tensor::DType::q4_k, hidden, feed_forward, "Q4_K gate/up")},
        {"Q4_K down", &require_representative(model, tensor::DType::q4_k, feed_forward, hidden, "Q4_K down")},
        {"Q6_K down", &require_representative(model, tensor::DType::q6_k, feed_forward, hidden, "Q6_K down")},
        {"F16 output", &require_representative(model, tensor::DType::f16, hidden, config.vocab_size, "F16 output")},
    };
    std::uint64_t tensor_bytes = 0;
    for (const auto& tensor_view : model.tensor_views()) tensor_bytes += tensor_view.view.bytes().size();
    ImmutableWeightBenchmarkResult result {
        .engine_build = LOCAL_GGUF_ENGINE_BUILD,
        .model_path = model_path,
        .device_name = state.device.name.UTF8String,
        .unified_memory = state.device.hasUnifiedMemory,
        .maximum_buffer_length = state.device.maxBufferLength,
        .recommended_maximum_working_set_size = state.device.recommendedMaxWorkingSetSize,
        .vm_page_size = page_size,
        .model_file_bytes = file_model.bytes().size(),
        .model_tensor_bytes = tensor_bytes,
        .model_tensor_count = model.tensor_views().size(),
        .policy = ImmutableWeightPolicy::shared_default_tracked,
        .file_backed_no_copy = true,
        .mapped_extent_bytes = extent,
        .base_page_aligned = true,
        .single_vm_region = true,
    };
    for (const auto& [name, representative] : representatives) {
        const auto payload = std::find_if(
            file_model.view().tensor_payloads.begin(), file_model.view().tensor_payloads.end(),
            [&](const model_format::GGUFTensorPayloadView& value) {
                return value.tensor.name == representative->name;
            }
        );
        if (payload == file_model.view().tensor_payloads.end()) {
            throw std::runtime_error("no-copy representative payload metadata is missing");
        }
        const auto offset = payload->tensor.file_offset;
        if (offset > extent || representative->view.bytes().size() > extent - offset) {
            throw std::out_of_range("no-copy representative tensor exceeds mapped parent");
        }
        auto weight = make_metal_backend_buffer_view(
            representative->view.descriptor(), parent, static_cast<std::size_t>(offset)
        );
        require_byte_parity(representative->view, weight);
        require_matmul_output_parity(representative->view, weight, name);
        const auto columns = representative->view.descriptor().shape.dimension(1);
        const auto shared = representative->view.descriptor().shape.dimension(0);
        auto input = tensor::allocate_backend_buffer(
            TensorDescriptor::from(tensor::Shape::from({1U, shared}), tensor::DType::f32),
            runtime::BackendKind::metal
        );
        auto output = tensor::allocate_backend_buffer(
            TensorDescriptor::from(tensor::Shape::from({1U, columns}), tensor::DType::f32),
            runtime::BackendKind::metal
        );
        auto resource = describe_immutable_weight_policy(
            ImmutableWeightPolicy::shared_default_tracked,
            representative->view.bytes().size(),
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
            ),
            0U
        );
        resource.label = "file-backed-no-copy";
        resource.copied_bytes = 0U;
        resource.file_backed_bytes = representative->view.bytes().size();
        ImmutableWeightKernelResult kernel {
            .name = name,
            .tensor_name = representative->name,
            .shape = shape_label(representative->view.descriptor()),
            .weight_bytes = representative->view.bytes().size(),
            .scheduled_class_weight_bytes = scheduled_class_bytes(model, name),
            .resource = std::move(resource),
        };
        kernel.measurements.push_back(measure(
            name, QuickKernelMeasurementMode::isolated, input, weight, output,
            quick_kernel_isolated_dispatch_count
        ));
        kernel.measurements.push_back(measure(
            name, QuickKernelMeasurementMode::sustained, input, weight, output,
            quick_kernel_sustained_dispatch_count
        ));
        result.kernels.push_back(std::move(kernel));
    }
    const auto& last = file_model.view().tensor_payloads.back();
    const auto* last_tensor = model.find_tensor_view(last.tensor.name);
    if (last_tensor == nullptr) throw std::runtime_error("no-copy final tensor metadata is missing");
    static_cast<void>(make_metal_backend_buffer_view(
        last_tensor->view.descriptor(),
        parent,
        static_cast<std::size_t>(last.tensor.file_offset)
    ));
    return result;
}

ImmutableWeightBenchmarkResult run_immutable_weight_benchmark(
    const std::string& model_path,
    ImmutableWeightPolicy policy
) {
    if (model_path.empty()) throw std::invalid_argument("benchmark model path must not be empty");
    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    auto model = engine::load_engine_model_from_gguf(file_model.view());
    auto& state = shared_matmul_pipeline_state();
    const auto& config = model.gemma2_config();
    const auto hidden = static_cast<std::uint64_t>(config.embedding_length);
    const auto feed_forward = static_cast<std::uint64_t>(config.feed_forward_length);
    const std::vector<std::pair<std::string, const engine::EngineTensorView*>> representatives {
        {"Q4_K gate/up", &require_representative(model, tensor::DType::q4_k, hidden, feed_forward, "Q4_K gate/up")},
        {"Q4_K down", &require_representative(model, tensor::DType::q4_k, feed_forward, hidden, "Q4_K down")},
        {"Q6_K down", &require_representative(model, tensor::DType::q6_k, feed_forward, hidden, "Q6_K down")},
        {"F16 output", &require_representative(model, tensor::DType::f16, hidden, config.vocab_size, "F16 output")},
    };
    std::uint64_t tensor_bytes = 0;
    for (const auto& tensor_view : model.tensor_views()) {
        if (tensor_bytes > std::numeric_limits<std::uint64_t>::max() - tensor_view.view.bytes().size()) {
            throw std::overflow_error("model tensor byte accounting overflows");
        }
        tensor_bytes += tensor_view.view.bytes().size();
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) throw std::runtime_error("could not query VM page size");
    ImmutableWeightBenchmarkResult result {
        .engine_build = LOCAL_GGUF_ENGINE_BUILD,
        .model_path = model_path,
        .device_name = state.device.name.UTF8String,
        .unified_memory = state.device.hasUnifiedMemory,
        .maximum_buffer_length = state.device.maxBufferLength,
        .recommended_maximum_working_set_size = state.device.recommendedMaxWorkingSetSize,
        .vm_page_size = static_cast<std::uint64_t>(page_size),
        .model_file_bytes = file_model.bytes().size(),
        .model_tensor_bytes = tensor_bytes,
        .model_tensor_count = model.tensor_views().size(),
        .policy = policy,
    };
    for (const auto& [name, representative] : representatives) {
        const auto& view = representative->view;
        auto uploaded = upload_weight(view, policy);
        require_byte_parity(view, uploaded.buffer);
        const auto shared = view.descriptor().shape.dimension(0);
        const auto columns = view.descriptor().shape.dimension(1);
        const auto input_descriptor = TensorDescriptor::from(
            tensor::Shape::from({1U, shared}), tensor::DType::f32
        );
        const auto output_descriptor = TensorDescriptor::from(
            tensor::Shape::from({1U, columns}), tensor::DType::f32
        );
        auto input = tensor::allocate_backend_buffer(input_descriptor, runtime::BackendKind::metal);
        auto output = tensor::allocate_backend_buffer(output_descriptor, runtime::BackendKind::metal);
        ImmutableWeightKernelResult kernel {
            .name = name,
            .tensor_name = representative->name,
            .shape = shape_label(view.descriptor()),
            .weight_bytes = view.bytes().size(),
            .scheduled_class_weight_bytes = scheduled_class_bytes(model, name),
            .resource = describe_immutable_weight_policy(
                policy, view.bytes().size(), uploaded.wall_duration_ns, uploaded.gpu_duration_ns
            ),
        };
        kernel.measurements.push_back(measure(
            name, QuickKernelMeasurementMode::isolated, input, uploaded.buffer, output,
            quick_kernel_isolated_dispatch_count
        ));
        kernel.measurements.push_back(measure(
            name, QuickKernelMeasurementMode::sustained, input, uploaded.buffer, output,
            quick_kernel_sustained_dispatch_count
        ));
        result.kernels.push_back(std::move(kernel));
    }
    return result;
}

}  // namespace local_gguf_engine::benchmarks
