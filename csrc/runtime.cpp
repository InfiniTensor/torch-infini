#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <mutex>

#include "torch_infini.h"

namespace torch_infini {

namespace {

std::once_flag g_backend_init_once;

} // namespace

std::string runtime_error_message(rt::Error status, const char* call) {
  return std::string{"InfiniRT "} + call + " failed with status " +
      std::to_string(static_cast<int>(status));
}

void check(rt::Error status, const char* call) {
  TORCH_CHECK(status == rt::kSuccess, runtime_error_message(status, call));
}

int device_count() noexcept {
  int count = 0;
  if (rt::GetDeviceCount(&count) != rt::kSuccess) {
    return 0;
  }
  return count;
}

int current_device() {
  int device = 0;
  check(rt::GetDevice(&device), "GetDevice");
  return device;
}

void set_device(int device) {
  TORCH_CHECK(device >= 0, "infini device index must be non-negative");
  check(rt::SetDevice(device), "SetDevice");
}

void synchronize(int device) {
  set_device(device);
  check(rt::DeviceSynchronize(), "DeviceSynchronize");
}

bool is_available() {
  return device_count() > 0;
}

std::string device_name(int device) {
  TORCH_CHECK(device >= 0, "infini device index must be non-negative");
  const int count = device_count();
  TORCH_CHECK(
      device < count,
      "infini device index ",
      device,
      " is out of range for ",
      count,
      " devices");
  return std::string{kBackendName} + ":" + std::to_string(device);
}

void initialize_backend() {
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
