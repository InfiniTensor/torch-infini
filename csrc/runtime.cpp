#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <mutex>

#include "infini_torch.h"

namespace torch_infini {

namespace {

std::once_flag g_backend_init_once;

} // namespace

std::string RuntimeErrorMessage(rt::Error status, const char* call) {
  return std::string{"InfiniRT "} + call + " failed with status " +
      std::to_string(static_cast<int>(status));
}

void Check(rt::Error status, const char* call) {
  TORCH_CHECK(status == rt::kSuccess, RuntimeErrorMessage(status, call));
}

int DeviceCount() noexcept {
  int count = 0;
  if (rt::GetDeviceCount(&count) != rt::kSuccess) {
    return 0;
  }
  return count;
}

int CurrentDevice() {
  int device = 0;
  Check(rt::GetDevice(&device), "GetDevice");
  return device;
}

void SetDevice(int device) {
  TORCH_CHECK(device >= 0, "infini device index must be non-negative");
  Check(rt::SetDevice(device), "SetDevice");
}

void Synchronize(int device) {
  SetDevice(device);
  Check(rt::DeviceSynchronize(), "DeviceSynchronize");
}

bool IsAvailable() {
  return DeviceCount() > 0;
}

std::string DeviceName(int device) {
  TORCH_CHECK(device >= 0, "infini device index must be non-negative");
  const int count = DeviceCount();
  TORCH_CHECK(
      device < count,
      "infini device index ",
      device,
      " is out of range for ",
      count,
      " devices");
  return std::string{kBackendName} + ":" + std::to_string(device);
}

void InitializeBackend() {
  std::call_once(g_backend_init_once, [] {
    if (c10::is_privateuse1_backend_registered()) {
      TORCH_CHECK(
          c10::get_privateuse1_backend(true) == kBackendName,
          "PrivateUse1 backend is already registered as ",
          c10::get_privateuse1_backend(true),
          ", not ",
          kBackendName);
      return;
    }
    c10::register_privateuse1_backend(kBackendName);
  });
}

} // namespace torch_infini
