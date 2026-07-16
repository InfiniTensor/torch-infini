#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <mutex>

#include "torch_infini.h"

namespace torch_infini {

namespace {

std::once_flag g_backend_registration_once;
std::once_flag g_runtime_selection_once;
std::atomic<int> g_runtime_device_type{-1};

template <auto... device_types>
infini::rt::Device::Type preferred_runtime_backend(
    infini::rt::List<device_types...>) {
  auto selected = infini::rt::Device::Type::kCpu;
  bool found_accelerator = false;

  const auto consider = [&](infini::rt::Device::Type candidate) {
    if (candidate == infini::rt::Device::Type::kCpu) {
      return;
    }
    if (!found_accelerator) {
      infini::rt::set_runtime_device_type(candidate);
      int count = 0;
      if (rt::GetDeviceCount(&count) == rt::kSuccess && count > 0) {
        found_accelerator = true;
        selected = candidate;
      }
    }
  };

  (consider(device_types), ...);
  return selected;
}

void select_runtime_backend() {
  std::call_once(g_runtime_selection_once, [] {
    using ActiveDeviceTypes = infini::rt::ActiveDevices<void>;
    static_assert(
        infini::rt::
            ContainsValue<ActiveDeviceTypes, infini::rt::Device::Type::kCpu>,
        "torch-infini requires InfiniRT to be built with CPU support");
    const auto selected = preferred_runtime_backend(ActiveDeviceTypes{});
    infini::rt::set_runtime_device_type(selected);
    TORCH_CHECK(
        infini::rt::runtime_device_type() == selected,
        "InfiniRT did not select the preferred runtime backend ",
        infini::rt::Device::StringFromType(selected));
    g_runtime_device_type.store(
        static_cast<int>(selected), std::memory_order_release);
  });
}

} // namespace

bool try_ensure_runtime_backend_for_current_thread() noexcept {
  const int selected = g_runtime_device_type.load(std::memory_order_acquire);
  if (selected < 0) {
    return false;
  }
  try {
    const auto selected_type = static_cast<infini::rt::Device::Type>(selected);
    if (infini::rt::runtime_device_type() != selected_type) {
      infini::rt::set_runtime_device_type(selected_type);
    }
    return infini::rt::runtime_device_type() == selected_type;
  } catch (...) {
    return false;
  }
}

void ensure_runtime_backend_for_current_thread() {
  const int selected = g_runtime_device_type.load(std::memory_order_acquire);
  TORCH_CHECK(
      try_ensure_runtime_backend_for_current_thread(),
      "failed to bind the preferred InfiniRT runtime backend ",
      selected >= 0 ? infini::rt::Device::StringFromType(
                          static_cast<infini::rt::Device::Type>(selected))
                    : "before backend registration");
}

std::string runtime_backend_name() {
  ensure_runtime_backend_for_current_thread();
  const auto selected = static_cast<infini::rt::Device::Type>(
      g_runtime_device_type.load(std::memory_order_acquire));
  return std::string{infini::rt::Device::StringFromType(selected)};
}

std::string runtime_error_message(rt::Error status, const char* call) {
  return std::string{"InfiniRT "} + call + " failed with status " +
      std::to_string(static_cast<int>(status));
}

void check(rt::Error status, const char* call) {
  TORCH_CHECK(status == rt::kSuccess, runtime_error_message(status, call));
}

int device_count() noexcept {
  if (!try_ensure_runtime_backend_for_current_thread()) {
    return 0;
  }
  int count = 0;
  if (rt::GetDeviceCount(&count) != rt::kSuccess) {
    return 0;
  }
  return count;
}

int current_device() {
  ensure_runtime_backend_for_current_thread();
  int device = 0;
  check(rt::GetDevice(&device), "GetDevice");
  return device;
}

void set_device(int device) {
  ensure_runtime_backend_for_current_thread();
  TORCH_CHECK(device >= 0, "infini device index must be non-negative");
  check(rt::SetDevice(device), "SetDevice");
}

void synchronize(int device) {
  TORCH_CHECK(device >= 0, "infini device index must be non-negative");
  const c10::DeviceGuard guard{
      c10::Device{kDeviceType, static_cast<c10::DeviceIndex>(device)}};
  check(rt::DeviceSynchronize(), "DeviceSynchronize");
}

bool is_available() {
  return device_count() > 0;
}

std::string device_name(int device) {
  ensure_runtime_backend_for_current_thread();
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

void register_backend() {
  std::call_once(g_backend_registration_once, [] {
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
  select_runtime_backend();
}

} // namespace torch_infini
