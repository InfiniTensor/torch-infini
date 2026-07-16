#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

#include "torch_infini.h"

namespace torch_infini {

namespace {

c10::Device checked_device(c10::Device device) {
  TORCH_CHECK(
      device.type() == kDeviceType, "expected an infini device, got ", device);
  const auto index = device.has_index() ? device.index() : current_device();
  return c10::Device{kDeviceType, static_cast<c10::DeviceIndex>(index)};
}

class DeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override {
    return kDeviceType;
  }

  c10::Device exchangeDevice(c10::Device device) const override {
    auto previous = getDevice();
    setDevice(device);
    return previous;
  }

  c10::Device getDevice() const override {
    return c10::Device{
        kDeviceType, static_cast<c10::DeviceIndex>(current_device())};
  }

  void setDevice(c10::Device device) const override {
    set_device(checked_device(device).index());
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    if (device.type() != kDeviceType) {
      return;
    }
    if (!try_ensure_runtime_backend_for_current_thread()) {
      return;
    }
    const auto index = device.has_index() ? device.index() : 0;
    (void)rt::SetDevice(index);
  }

  c10::Stream getStream(c10::Device device) const override {
    return get_current_stream(device);
  }

  c10::Stream getDefaultStream(c10::Device device) const override {
    return get_default_stream(device);
  }

  c10::Stream getStreamFromGlobalPool(
      c10::Device device,
      bool is_high_priority = false) const override {
    return get_stream_from_global_pool(device, is_high_priority);
  }

  c10::Stream getNewStream(c10::Device device, int priority = 0)
      const override {
    return create_stream(device, priority);
  }

  c10::Stream exchangeStream(c10::Stream stream) const override {
    return exchange_current_stream(stream);
  }

  void* getStreamNativeHandle(c10::Stream stream) const {
    return get_native_stream_handle(stream);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(device_count());
  }

  bool queryStream(const c10::Stream& stream) const override {
    return query_stream(stream);
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    synchronize_stream(stream);
  }

  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    synchronize(device_index);
  }
};

} // namespace

C10_REGISTER_GUARD_IMPL(PrivateUse1, DeviceGuardImpl);

} // namespace torch_infini
