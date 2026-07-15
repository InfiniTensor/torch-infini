#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "torch_infini.h"

namespace torch_infini {

namespace {

struct AllocationContext {
  void* ptr;
  int device;
};

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
  (void)rt::SetDevice(allocation->device);
  (void)rt::Free(allocation->ptr);
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
        new AllocationContext{data, device_index},
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

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

} // namespace torch_infini
