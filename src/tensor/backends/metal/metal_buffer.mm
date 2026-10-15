#include "metal_buffer.hpp"

#include "metal_context.hpp"
#include "../metal_backend.hpp"

#include "local_gguf_engine/runtime/backend_error.hpp"

#include <cstring>
#include <bit>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(LOCAL_GGUF_ENGINE_TESTING)
#include "metal_test_hooks.hpp"

// Exercise the real residency lifecycle without requiring a particular GPU.
@interface LocalGGUFResidencyTestDevice : NSObject
@property(nonatomic) BOOL residencySupported;
- (BOOL)supportsFamily:(MTLGPUFamily)family;
- (id<MTLResidencySet>)newResidencySetWithDescriptor:(MTLResidencySetDescriptor*)descriptor
    error:(NSError**)error API_AVAILABLE(macos(15.0));
@end

@implementation LocalGGUFResidencyTestDevice
- (BOOL)supportsFamily:(MTLGPUFamily)family {
    return self.residencySupported && family == MTLGPUFamilyApple6;
}
- (id<MTLResidencySet>)newResidencySetWithDescriptor:(MTLResidencySetDescriptor*)descriptor
    error:(NSError**)error {
    if (!self.residencySupported) {
        throw std::logic_error("unsupported device must not attempt residency set creation");
    }
    if (error != nullptr) *error = nil;
    return nil;
}
@end
#endif

namespace local_gguf_engine::tensor::backends::metal {

namespace {

#if defined(LOCAL_GGUF_ENGINE_TESTING)
__weak id<MTLBuffer> last_initializer_buffer;
#endif

// Retaining MTLBuffer objects alone does not keep their backing memory resident
// between turns. Match ggml 0.13.1's 500 ms requests and three-minute idle window.
// One set covers both numerical targets and the drafter without extra GPU work.
class MetalBufferResidency final {
public:
    explicit MetalBufferResidency(id<MTLDevice> device) {
        if (@available(macOS 15.0, *)) {
            @autoreleasepool {
                // OS availability alone is insufficient: residency sets require
                // Apple6 or later. Other Metal GPUs still use ordinary buffers.
                if (![device supportsFamily:MTLGPUFamilyApple6]) return;
                auto* descriptor = [MTLResidencySetDescriptor new];
                descriptor.label = @"local-gguf-engine model buffers";
                descriptor.initialCapacity = 256;
                NSError* error = nil;
                set_ = [device newResidencySetWithDescriptor:descriptor error:&error];
                if (set_ == nil) {
                    throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                        "Metal residency set creation failed: " + ns_error_message(error, "unknown error"));
                }
            }
            worker_ = std::thread([this] {
                std::unique_lock lock(mutex_);
                while (!stopping_) {
                    if (!owners_.empty() && now_ns() < active_until_ns_.load(std::memory_order_relaxed)) {
                        @autoreleasepool { [set_ requestResidency]; }
                    }
                    wake_.wait_for(lock, std::chrono::milliseconds(500), [this] { return stopping_; });
                }
            });
        }
    }

    ~MetalBufferResidency() {
        if (set_ == nil) return;
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_one();
        if (worker_.joinable()) worker_.join();
        if (@available(macOS 15.0, *)) {
            @autoreleasepool {
                [set_ endResidency];
                [set_ removeAllAllocations];
                [set_ commit];
                set_ = nil;
            }
        }
    }

    void add(id<MTLBuffer> buffer) {
        if (set_ == nil) return;
        if (@available(macOS 15.0, *)) {
            std::lock_guard lock(mutex_);
            // Packed projections and cache views share a parent allocation.
            // Only its first owner adds it; only its last owner removes it.
            auto& count = owners_[identity(buffer)];
            if (count++ == 0) {
                @autoreleasepool {
                    [set_ addAllocation:buffer];
                    [set_ commit];
                    [set_ requestResidency];
                }
            }
            keep_alive();
        }
    }

    void remove(id<MTLBuffer> buffer) noexcept {
        if (set_ == nil) return;
        if (@available(macOS 15.0, *)) {
            std::lock_guard lock(mutex_);
            const auto entry = owners_.find(identity(buffer));
            if (entry == owners_.end() || --entry->second != 0) return;
            @autoreleasepool {
                [set_ removeAllocation:buffer];
                [set_ commit];
                owners_.erase(entry);
                if (owners_.empty()) [set_ endResidency];
            }
        }
    }

    void keep_alive() noexcept {
        if (set_ == nil) return;
        active_until_ns_.store(now_ns() + 180'000'000'000LL, std::memory_order_relaxed);
    }

private:
    static std::int64_t now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static std::uintptr_t identity(id<MTLBuffer> buffer) noexcept {
        return reinterpret_cast<std::uintptr_t>((__bridge const void*)buffer);
    }

    id<MTLResidencySet> set_ = nil;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::unordered_map<std::uintptr_t, std::size_t> owners_;
    std::atomic<std::int64_t> active_until_ns_ {0};
    bool stopping_ = false;
    std::thread worker_;
};

MetalBufferResidency& buffer_residency() {
    static MetalBufferResidency residency(metal_device_or_throw());
    return residency;
}

} // namespace

#if defined(LOCAL_GGUF_ENGINE_TESTING)
bool exercise_metal_buffer_residency_for_testing(bool supported) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            auto* device = [LocalGGUFResidencyTestDevice new];
            device.residencySupported = supported;
            MetalBufferResidency residency((id<MTLDevice>)device);
            residency.add(nil);
            residency.keep_alive();
            residency.remove(nil);
        }
        return true;
    }
    return false;
}
#endif

void keep_metal_buffers_resident() {
    buffer_residency().keep_alive();
}

MetalBackendBufferStorage::MetalBackendBufferStorage(
    const TensorStorage& storage,
    id<MTLBuffer> buffer
)
    : MetalBackendBufferStorage(storage.descriptor(), buffer) {}

MetalBackendBufferStorage::MetalBackendBufferStorage(
    TensorDescriptor descriptor,
    id<MTLBuffer> buffer,
    std::size_t base_byte_offset,
    bool read_only
)
    : descriptor_(std::move(descriptor)),
      buffer_(buffer),
      base_byte_offset_(base_byte_offset),
      read_only_(read_only) {
    validate_tensor_descriptor(descriptor_);
    if (buffer_ == nil) throw std::invalid_argument("Metal backend buffer must not be nil");
    if (base_byte_offset_ % alignof(std::uint32_t) != 0U) {
        throw std::invalid_argument("Metal backend buffer base offset must be four-byte aligned");
    }
    if (base_byte_offset_ > buffer_.length
        || descriptor_.byte_count > buffer_.length - base_byte_offset_) {
        throw std::out_of_range("Metal backend buffer view exceeds its parent allocation");
    }
    buffer_residency().add(buffer_);
}

MetalBackendBufferStorage::~MetalBackendBufferStorage() {
    buffer_residency().remove(buffer_);
}

runtime::BackendKind MetalBackendBufferStorage::backend() const noexcept {
    return runtime::BackendKind::metal;
}

const TensorDescriptor& MetalBackendBufferStorage::descriptor() const noexcept {
    return descriptor_;
}

id<MTLBuffer> const& MetalBackendBufferStorage::buffer() const noexcept {
    return buffer_;
}

std::size_t MetalBackendBufferStorage::base_byte_offset() const noexcept {
    return base_byte_offset_;
}

bool MetalBackendBufferStorage::read_only() const noexcept { return read_only_; }

std::size_t MetalBackendBufferStorage::parent_byte_count() const noexcept {
    return buffer_.length;
}

std::uintptr_t MetalBackendBufferStorage::parent_allocation_identity() const noexcept {
    return reinterpret_cast<std::uintptr_t>((__bridge const void*)buffer_);
}

const MetalBackendBufferStorage& require_metal_backend_buffer(
    const detail::BackendBufferStorage& storage
) {
    const auto* typed_storage = dynamic_cast<const MetalBackendBufferStorage*>(&storage);
    if (typed_storage == nullptr) {
        throw metal_error(
            runtime::BackendErrorCode::unsupported_backend,
            "backend buffer does not contain Metal storage"
        );
    }

    return *typed_storage;
}

BackendBuffer make_metal_backend_buffer(TensorDescriptor descriptor, id<MTLBuffer> buffer) {
    return detail::make_backend_buffer(
        std::make_shared<MetalBackendBufferStorage>(std::move(descriptor), buffer)
    );
}

BackendBuffer make_metal_backend_buffer_view(
    TensorDescriptor descriptor,
    id<MTLBuffer> parent_buffer,
    std::size_t base_byte_offset,
    bool read_only
) {
    return detail::make_backend_buffer(std::make_shared<MetalBackendBufferStorage>(
        std::move(descriptor), parent_buffer, base_byte_offset, read_only
    ));
}

NSUInteger metal_buffer_binding_offset(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset
) {
    if (operation_byte_offset > storage.descriptor().byte_count) {
        throw std::out_of_range("Metal backend operation offset exceeds child buffer");
    }
    const auto base = static_cast<std::uint64_t>(storage.base_byte_offset());
    if (operation_byte_offset > std::numeric_limits<std::uint64_t>::max() - base) {
        throw std::overflow_error("Metal backend binding offset overflows");
    }
    const auto combined = base + operation_byte_offset;
    if (combined > storage.parent_byte_count()) {
        throw std::out_of_range("Metal backend binding offset exceeds parent allocation");
    }
    return static_cast<NSUInteger>(combined);
}

const void* metal_buffer_contents(
    const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset
) {
    // Metal marks contents as an interior pointer. An explicitly scoped
    // strong receiver prevents ARC from autoreleasing the owner to extend
    // that pointer's lifetime; the caller already owns storage for every use.
    id<MTLBuffer> __attribute__((objc_precise_lifetime)) buffer = storage.buffer();
    void* contents = buffer.contents;
    if (contents == nullptr) {
        throw std::logic_error("Metal private buffer contents are not CPU-accessible");
    }
    return static_cast<const std::byte*>(contents) + metal_buffer_binding_offset(storage, operation_byte_offset);
}

void require_writable_metal_buffer(const MetalBackendBufferStorage& storage) {
    if (storage.read_only()) throw std::invalid_argument("immutable Metal weights cannot be written");
}

void* mutable_metal_buffer_contents(const MetalBackendBufferStorage& storage,
    std::uint64_t operation_byte_offset) {
    require_writable_metal_buffer(storage);
    return const_cast<void*>(metal_buffer_contents(storage, operation_byte_offset));
}

namespace {

struct MappedTensorWindow {
    const std::uint8_t* begin;
    std::size_t byte_count;
    std::size_t tensor_offset;
};

MappedTensorWindow mapped_tensor_window(const TensorView& view,
    const runtime::OwnedReadOnlyBytes& backing) {
    const auto source = backing.bytes();
    if (!backing.owner() || source.empty() || view.bytes().empty()
        || view.descriptor().shape.rank() != 2U
        || view.descriptor().packing != TensorPacking::canonical) {
        throw std::invalid_argument("immutable Metal mapping requires an owned canonical matrix");
    }
    const auto page_result = ::sysconf(_SC_PAGESIZE);
    if (page_result <= 0) throw std::runtime_error("could not determine immutable Metal mapping page size");
    const auto page = static_cast<std::size_t>(page_result);
    const auto address = reinterpret_cast<std::uintptr_t>(source.data());
    const auto tensor_address = reinterpret_cast<std::uintptr_t>(view.bytes().data());
    if (!std::has_single_bit(page) || address % page != 0U
        || backing.accessible_byte_count() % page != 0U
        || backing.accessible_byte_count() > std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::invalid_argument("immutable Metal backing must own an aligned page extent");
    }
    if (tensor_address < address || tensor_address - address > source.size()
        || view.bytes().size() > source.size() - (tensor_address - address)) {
        throw std::invalid_argument("immutable Metal tensor is outside its logical source allocation");
    }
    const auto offset = static_cast<std::size_t>(tensor_address - address);
    const auto end = offset + view.bytes().size();
    if (end > std::numeric_limits<std::size_t>::max() - (page - 1U))
        throw std::overflow_error("immutable Metal tensor page extent overflows");
    const auto first_page = offset / page * page;
    const auto end_page = (end + page - 1U) / page * page;
    if (end_page > backing.accessible_byte_count())
        throw std::out_of_range("immutable Metal tensor page extent exceeds its owned allocation");
    return {source.data() + first_page, end_page - first_page, offset - first_page};
}

bool is_readable_vm_region(const MappedTensorWindow& window) {
    mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(window.begin);
    mach_vm_size_t byte_count = 0;
    vm_region_basic_info_data_64_t info {};
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = mach_vm_region(mach_task_self(), &address, &byte_count,
        VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &info_count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (result != KERN_SUCCESS || (info.protection & VM_PROT_READ) == 0) return false;
    const auto target = reinterpret_cast<mach_vm_address_t>(window.begin);
    return address <= target && target - address <= byte_count
        && window.byte_count <= byte_count - (target - address);
}

}  // namespace

std::size_t mapped_immutable_tensor_byte_count(const TensorView& view,
    const runtime::OwnedReadOnlyBytes& backing) {
    return mapped_tensor_window(view, backing).byte_count;
}

std::size_t metal_backend_buffer_allocation_byte_count(const BackendBuffer& buffer) {
    return require_metal_backend_buffer(backend_buffer_storage(buffer)).parent_byte_count();
}

BackendBuffer map_immutable_tensor_view_to_metal_backend(const TensorView& view,
    const runtime::OwnedReadOnlyBytes& backing) {
    const auto window = mapped_tensor_window(view, backing);
    if (window.tensor_offset % alignof(std::uint32_t) != 0U)
        throw std::invalid_argument("immutable Metal tensor offset must be four-byte aligned");
    if (!is_readable_vm_region(window))
        throw std::invalid_argument("immutable Metal tensor must occupy one readable VM region");
    @autoreleasepool {
        id<MTLDevice> device = metal_device_or_throw();
        if (!device.hasUnifiedMemory)
            throw metal_error(runtime::BackendErrorCode::unsupported_backend,
                "immutable Metal mappings require unified memory");
        if (window.byte_count > device.maxBufferLength)
            throw std::invalid_argument("immutable Metal tensor page extent exceeds maxBufferLength");
        // Retain the mmap in Metal's deallocator, so submitted command buffers
        // and child views remain valid after every C++ model facade is gone.
        const auto owner = backing.owner();
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(window.begin)
            length:static_cast<NSUInteger>(window.byte_count) options:MTLResourceStorageModeShared
            deallocator:^(void*, NSUInteger) { static_cast<void>(owner); }];
        if (buffer == nil)
            throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                "immutable Metal tensor registration failed");
        return make_metal_backend_buffer_view(view.descriptor(), buffer, window.tensor_offset, true);
    }
}

BackendBuffer initialize_metal_backend_buffer(const TensorDescriptor& descriptor,
    const std::function<void(std::span<std::uint8_t>)>& initializer) {
    validate_tensor_descriptor(descriptor);
    if (!initializer) throw std::invalid_argument("Metal buffer initializer must not be empty");
    std::exception_ptr failure;
    @autoreleasepool {
        try {
            id<MTLDevice> device = metal_device_or_throw();
            if (descriptor.byte_count > device.maxBufferLength)
                throw std::invalid_argument("Metal tensor initialization exceeds maxBufferLength");
            // contents returns an interior pointer; the local owner already
            // spans every callback use, so ARC need not autorelease the buffer.
            id<MTLBuffer> __attribute__((objc_precise_lifetime)) buffer =
                [device newBufferWithLength:static_cast<NSUInteger>(descriptor.byte_count)
                    options:MTLResourceStorageModeShared];
            if (buffer == nil)
                throw metal_error(runtime::BackendErrorCode::metal_execution_failed,
                    "Metal initialized buffer allocation failed");
#if defined(LOCAL_GGUF_ENGINE_TESTING)
            last_initializer_buffer = buffer;
#endif
            initializer({static_cast<std::uint8_t*>(buffer.contents), static_cast<std::size_t>(descriptor.byte_count)});
            return make_metal_backend_buffer(descriptor, buffer);
        } catch (...) {
            // C++ unwinding skips autorelease-pool drainage. Leave the pool
            // normally before propagating failure, including callback failures.
            failure = std::current_exception();
        }
    }
    std::rethrow_exception(failure);
}

#if defined(LOCAL_GGUF_ENGINE_TESTING)
bool initialized_metal_buffer_is_alive_for_testing() noexcept {
    @autoreleasepool {
        return last_initializer_buffer != nil;
    }
}
#endif

BackendBuffer copy_tensor_to_metal_backend(const TensorStorage& storage) {
    return copy_tensor_view_to_metal_backend(storage.view());
}

BackendBuffer copy_tensor_view_to_metal_backend(const TensorView& view) {
    @autoreleasepool {
        id<MTLDevice> device = metal_device_or_throw();
        if (view.bytes().size() > device.maxBufferLength) {
            throw std::invalid_argument("Metal tensor upload exceeds the device maxBufferLength");
        }
        id<MTLBuffer> buffer = [device newBufferWithBytes:view.bytes().data()
                                                         length:static_cast<NSUInteger>(view.bytes().size())
                                                        options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw runtime::BackendError(
                runtime::BackendErrorCode::metal_execution_failed,
                runtime::BackendKind::metal,
                "Metal backend buffer allocation failed"
            );
        }

        return detail::make_backend_buffer(
            std::make_shared<MetalBackendBufferStorage>(view.descriptor(), buffer)
        );
    }
}

BackendBuffer allocate_metal_backend_buffer(const TensorDescriptor& descriptor) {
    validate_tensor_descriptor(descriptor);
    @autoreleasepool {
        id<MTLDevice> device = metal_device_or_throw();
        if (descriptor.byte_count > device.maxBufferLength) {
            throw std::invalid_argument("Metal tensor allocation exceeds the device maxBufferLength");
        }
        id<MTLBuffer> buffer = [device newBufferWithLength:static_cast<NSUInteger>(descriptor.byte_count)
                                                         options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw runtime::BackendError(
                runtime::BackendErrorCode::metal_execution_failed,
                runtime::BackendKind::metal,
                "Metal backend buffer allocation failed"
            );
        }

        return make_metal_backend_buffer(descriptor, buffer);
    }
}

TensorStorage copy_tensor_from_metal_backend(const detail::BackendBufferStorage& storage) {
    @autoreleasepool {
        const auto& metal_storage = require_metal_backend_buffer(storage);
        auto output = TensorStorage::allocate(metal_storage.descriptor());
        std::memcpy(
            output.mutable_bytes().data(),
            metal_buffer_contents(metal_storage),
            output.bytes().size()
        );
        return output;
    }
}

std::vector<std::uint8_t> copy_metal_backend_buffer_bytes(
    const detail::BackendBufferStorage& storage,
    std::size_t byte_offset,
    std::size_t byte_count
) {
    @autoreleasepool {
        const auto& metal_storage = require_metal_backend_buffer(storage);
        if (byte_offset > metal_storage.descriptor().byte_count
            || byte_count > metal_storage.descriptor().byte_count - byte_offset) {
            throw std::out_of_range("Metal backend byte readback exceeds buffer");
        }
        std::vector<std::uint8_t> output(byte_count);
        if (byte_count != 0U) {
            std::memcpy(
                output.data(), metal_buffer_contents(metal_storage, byte_offset), byte_count
            );
        }
        return output;
    }
}

}  // namespace local_gguf_engine::tensor::backends::metal
