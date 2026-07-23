#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>

#include "torch_infini.h"

namespace torch_infini {

namespace {

class LiveAllocationRegistry final {
 public:
  void add(void* data, std::size_t nbytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto [it, inserted] = allocations_.emplace(address, nbytes);
    TORCH_CHECK(inserted, "InfiniRT MallocHost returned a live address twice");
    (void)it;
  }

  void remove(void* data) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const std::lock_guard<std::mutex> lock(mutex_);
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
    return address - allocation->first < allocation->second;
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::uintptr_t, std::size_t> allocations_;
};

LiveAllocationRegistry& live_allocations() {
  // Pinned DataPtrs may be released during process teardown, after ordinary
  // static objects in this extension have already been destroyed.
  static auto* registry = new LiveAllocationRegistry();
  return *registry;
}

void delete_host_allocation(void* data) {
  if (data == nullptr) {
    return;
  }
  live_allocations().remove(data);
  if (try_ensure_runtime_backend_for_current_thread()) {
    (void)rt::FreeHost(data);
  }
}

class HostAllocator final : public at::HostAllocator {
 public:
  c10::DataPtr allocate(std::size_t nbytes) override {
    ensure_runtime_backend_for_current_thread();
    void* data = nullptr;
    if (nbytes != 0) {
      check(rt::MallocHost(&data, nbytes), "MallocHost");
      try {
        live_allocations().add(data, nbytes);
      } catch (...) {
        (void)rt::FreeHost(data);
        throw;
      }
    }
    return {
        data, data, &delete_host_allocation, c10::Device{c10::DeviceType::CPU}};
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    std::memcpy(dest, src, count);
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &delete_host_allocation;
  }

  // Copies are synchronized before returning, so allocations are never cached
  // or held alive by stream events.
  bool record_event(void*, void*, c10::Stream) override {
    return false;
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

REGISTER_HOST_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_host_allocator);

} // namespace torch_infini
