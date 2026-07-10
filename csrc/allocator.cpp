#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "infini_torch.h"

namespace torch_infini {

namespace {

struct AllocationContext {
  void* ptr;
  int device;
};

void InfiniDelete(void* context) {
  auto* allocation = static_cast<AllocationContext*>(context);
  if (allocation == nullptr) {
    return;
  }
  (void)rt::SetDevice(allocation->device);
  (void)rt::Free(allocation->ptr);
  delete allocation;
}

class InfiniAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    const int device_index = CurrentDevice();
    if (nbytes != 0) {
      Check(rt::Malloc(&data, nbytes), "Malloc");
    }
    const auto device =
        c10::Device{kDeviceType, static_cast<c10::DeviceIndex>(device_index)};
    return {
        data, new AllocationContext{data, device_index}, &InfiniDelete, device};
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    Check(
        rt::Memcpy(dest, src, count, rt::kMemcpyDeviceToDevice),
        "Memcpy(DeviceToDevice)");
  }
};

InfiniAllocator g_allocator;

} // namespace

c10::Allocator* GetInfiniAllocator() {
  return &g_allocator;
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

} // namespace torch_infini
