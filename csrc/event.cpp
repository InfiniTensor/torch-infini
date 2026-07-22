#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include "torch_infini.h"

namespace torch_infini {

namespace {

c10::Device event_device(c10::DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0,
      "expected a recorded infini event, got device index ",
      static_cast<int>(device_index));
  return c10::Device{kDeviceType, device_index};
}

rt::Event native_event(void* event) {
  TORCH_CHECK(event != nullptr, "expected a recorded infini event");
  return event;
}

rt::Stream native_stream(const c10::Stream& stream) {
  return reinterpret_cast<rt::Stream>(get_native_stream_handle(stream));
}

} // namespace

void destroy_event(void* event, c10::DeviceIndex device_index) noexcept {
  if (event == nullptr || !try_ensure_runtime_backend_for_current_thread()) {
    return;
  }

  int original_device = 0;
  if (device_index < 0 || rt::GetDevice(&original_device) != rt::kSuccess ||
      rt::SetDevice(device_index) != rt::kSuccess) {
    return;
  }

  (void)rt::EventDestroy(event);
  (void)rt::SetDevice(original_device);
}

void record_event(
    void** event,
    const c10::Stream& stream,
    c10::DeviceIndex device_index,
    c10::EventFlag flag) {
  TORCH_CHECK(event != nullptr, "expected storage for an infini event");
  TORCH_CHECK(
      flag == c10::EventFlag::PYTORCH_DEFAULT ||
          flag == c10::EventFlag::BACKEND_DEFAULT,
      "unsupported infini event flag");
  TORCH_CHECK(
      device_index < 0 || device_index == stream.device_index(),
      "infini event already belongs to infini:",
      static_cast<int>(device_index),
      " and cannot be recorded on infini:",
      static_cast<int>(stream.device_index()));

  const c10::DeviceGuard guard{stream.device()};
  const auto stream_handle = native_stream(stream);
  bool created = false;
  if (*event == nullptr) {
    rt::Event native{};
    const auto status = rt::EventCreate(&native);
    TORCH_CHECK(
        status == rt::kSuccess,
        runtime_error_message(status, "EventCreate"),
        "; the selected InfiniRT backend may not implement event operations");
    *event = native;
    created = true;
  }

  try {
    check(rt::EventRecord(native_event(*event), stream_handle), "EventRecord");
  } catch (...) {
    if (created) {
      (void)rt::EventDestroy(*event);
      *event = nullptr;
    }
    throw;
  }
}

void block_event(void* event, const c10::Stream& stream) {
  const c10::DeviceGuard guard{stream.device()};
  check(
      rt::StreamWaitEvent(native_stream(stream), native_event(event), 0),
      "StreamWaitEvent");
}

bool query_event(void* event) {
  ensure_runtime_backend_for_current_thread();
  return rt::EventQuery(native_event(event)) == rt::kSuccess;
}

void synchronize_event(void* event) {
  ensure_runtime_backend_for_current_thread();
  check(rt::EventSynchronize(native_event(event)), "EventSynchronize");
}

double elapsed_time(
    void* start_event,
    void* end_event,
    c10::DeviceIndex device_index) {
  const c10::DeviceGuard guard{event_device(device_index)};
  float milliseconds = 0.0F;
  check(
      rt::EventElapsedTime(
          &milliseconds, native_event(start_event), native_event(end_event)),
      "EventElapsedTime");
  return milliseconds;
}

} // namespace torch_infini
