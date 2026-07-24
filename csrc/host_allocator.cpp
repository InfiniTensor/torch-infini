#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include "torch_infini.h"

namespace torch_infini {

namespace {

struct HostAllocationContext {
  void* ptr;
  std::size_t nbytes;
  std::mutex mutex;
  std::vector<c10::Stream> streams;
};

class LiveAllocationRegistry final {
 public:
  void add(HostAllocationContext* allocation) {
    const auto address = reinterpret_cast<std::uintptr_t>(allocation->ptr);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto [it, inserted] = allocations_.emplace(address, allocation);
    TORCH_CHECK(inserted, "InfiniRT MallocHost returned a live address twice");
    try {
      const auto [context_it, context_inserted] = contexts_.insert(allocation);
      TORCH_INTERNAL_ASSERT(context_inserted);
      (void)context_it;
    } catch (...) {
      allocations_.erase(it);
      throw;
    }
  }

  void remove(HostAllocationContext* allocation) {
    const auto address = reinterpret_cast<std::uintptr_t>(allocation->ptr);
    const std::lock_guard<std::mutex> lock(mutex_);
    contexts_.erase(allocation);
    allocations_.erase(address);
  }

  bool contains(const void* data) const {
    if (data == nullptr) {
      return false;
    }

    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const std::lock_guard<std::mutex> lock(mutex_);
    auto allocation = allocations_.upper_bound(address);
    if (allocation == allocations_.begin()) {
      return false;
    }
    --allocation;
    return address - allocation->first < allocation->second->nbytes;
  }

  bool record_stream(
      const void* data,
      const void* allocation_context,
      const c10::Stream& stream) {
    const std::lock_guard<std::mutex> registry_lock(mutex_);
    HostAllocationContext* context = nullptr;
    const auto context_it = contexts_.find(
        static_cast<const HostAllocationContext*>(allocation_context));
    if (context_it != contexts_.end()) {
      context = const_cast<HostAllocationContext*>(*context_it);
    } else if (data != nullptr) {
      const auto address = reinterpret_cast<std::uintptr_t>(data);
      auto allocation = allocations_.upper_bound(address);
      if (allocation != allocations_.begin()) {
        --allocation;
        if (address - allocation->first < allocation->second->nbytes) {
          context = allocation->second;
        }
      }
    }

    if (context == nullptr) {
      return false;
    }

    const std::lock_guard<std::mutex> allocation_lock{context->mutex};
    if (std::find(context->streams.begin(), context->streams.end(), stream) ==
        context->streams.end()) {
      context->streams.push_back(stream);
    }
    return true;
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::uintptr_t, HostAllocationContext*> allocations_;
  std::unordered_set<const HostAllocationContext*> contexts_;
};

LiveAllocationRegistry& live_allocations() {
  // Pinned DataPtrs may be released during process teardown, after ordinary
  // static objects in this extension have already been destroyed.
  static auto* registry = new LiveAllocationRegistry();
  return *registry;
}

void delete_host_allocation(void* context) {
  auto* allocation = static_cast<HostAllocationContext*>(context);
  if (allocation == nullptr) {
    return;
  }
  live_allocations().remove(allocation);
  if (try_ensure_runtime_backend_for_current_thread()) {
    try {
      std::vector<c10::Stream> streams;
      {
        const std::lock_guard<std::mutex> lock{allocation->mutex};
        streams = std::move(allocation->streams);
      }
      for (const auto& stream : streams) {
        synchronize_stream(stream);
      }
      (void)rt::FreeHost(allocation->ptr);
    } catch (...) {
    }
  }
  delete allocation;
}

class HostAllocator final : public at::HostAllocator {
 public:
  c10::DataPtr allocate(std::size_t nbytes) override {
    ensure_runtime_backend_for_current_thread();
    TORCH_CHECK(
        runtime_capabilities(infini::rt::runtime_device_type())
            .pinned_host_allocation,
        "the selected InfiniRT backend does not support pinned host memory");
    void* data = nullptr;
    HostAllocationContext* context = nullptr;
    if (nbytes != 0) {
      check(rt::MallocHost(&data, nbytes), "MallocHost");
      try {
        context = new HostAllocationContext{data, nbytes, {}, {}};
        live_allocations().add(context);
      } catch (...) {
        delete context;
        (void)rt::FreeHost(data);
        throw;
      }
    }
    return {
        data,
        context,
        &delete_host_allocation,
        c10::Device{c10::DeviceType::CPU}};
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    std::memcpy(dest, src, count);
  }

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
    return live_allocations().record_stream(ptr, ctx, stream);
  }

  void empty_cache() override {}

  at::HostStats get_stats() override {
    return {};
  }

  void reset_accumulated_stats() override {}

  void reset_peak_stats() override {}
};

HostAllocator g_host_allocator;

} // namespace

c10::Allocator* get_host_allocator() {
  return &g_host_allocator;
}

bool is_pinned_ptr(const void* data) {
  return live_allocations().contains(data);
}

bool record_host_allocation_stream(
    const at::Tensor& tensor,
    const c10::Stream& stream) {
  const auto& data_ptr = tensor.storage().data_ptr();
  return g_host_allocator.record_event(
      tensor.data_ptr(), data_ptr.get_context(), stream);
}

REGISTER_HOST_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_host_allocator);

} // namespace torch_infini
