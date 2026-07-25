#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "torch_infini.h"

namespace torch_infini {

namespace {

struct AllocationContext {
  void* ptr;
  int device;
  std::mutex mutex;
  std::vector<c10::Stream> streams;
};

void synchronize_recorded_streams(AllocationContext* allocation) {
  std::vector<c10::Stream> streams;
  {
    const std::lock_guard<std::mutex> lock{allocation->mutex};
    streams = std::move(allocation->streams);
  }
  for (const auto& stream : streams) {
    synchronize_stream(stream);
  }
}

void delete_allocation(void* context) {
  auto* allocation = static_cast<AllocationContext*>(context);
  if (allocation == nullptr) {
    return;
  }
  // A DataPtr may be destroyed on a different thread from its allocation.
  if (!try_ensure_runtime_backend_for_current_thread()) {
    delete allocation;
    return;
  }
  try {
    const c10::DeviceGuard guard{c10::Device{
        kDeviceType, static_cast<c10::DeviceIndex>(allocation->device)}};
    synchronize_recorded_streams(allocation);
    (void)rt::Free(allocation->ptr);
  } catch (...) {
  }
  delete allocation;
}

class Allocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    const int device_index = current_device();
    if (nbytes != 0) {
      check(rt::Malloc(&data, nbytes), "Malloc");
    }
    const auto device =
        c10::Device{kDeviceType, static_cast<c10::DeviceIndex>(device_index)};
    return {
        data,
        new AllocationContext{data, device_index, {}, {}},
        &delete_allocation,
        device};
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    ensure_runtime_backend_for_current_thread();
    check(
        rt::Memcpy(dest, src, count, rt::kMemcpyDeviceToDevice),
        "Memcpy(DeviceToDevice)");
  }
};

Allocator g_allocator;

} // namespace

c10::Allocator* get_allocator() {
  return &g_allocator;
}

bool can_record_allocation_stream(
    const at::Tensor& tensor,
    const c10::Stream& stream) {
  const auto& data_ptr = tensor.storage().data_ptr();
  if (data_ptr.get_deleter() != &delete_allocation) {
    return false;
  }
  const auto* allocation =
      static_cast<const AllocationContext*>(data_ptr.get_context());
  return allocation != nullptr && allocation->device == stream.device_index();
}

void record_allocation_stream(
    const at::Tensor& tensor,
    const c10::Stream& stream) {
  const auto& data_ptr = tensor.storage().data_ptr();
  TORCH_INTERNAL_ASSERT(can_record_allocation_stream(tensor, stream));
  auto* allocation = static_cast<AllocationContext*>(data_ptr.get_context());

  const std::lock_guard<std::mutex> lock{allocation->mutex};
  if (std::find(
          allocation->streams.begin(), allocation->streams.end(), stream) ==
      allocation->streams.end()) {
    allocation->streams.push_back(stream);
  }
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

} // namespace torch_infini
