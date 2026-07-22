#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>

#include "torch_infini.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch_infini::register_backend();

  m.def("_runtime_backend_name", &torch_infini::runtime_backend_name);
  m.def("is_available", &torch_infini::is_available);
  m.def("device_count", &torch_infini::device_count);
  m.def("current_device", &torch_infini::current_device);
  m.def("set_device", &torch_infini::set_device);
  m.def("synchronize", &torch_infini::synchronize);
  m.def("get_device_name", &torch_infini::device_name);
  m.def(
      "_event_elapsed_time",
      [](std::uintptr_t start_event,
         std::uintptr_t end_event,
         int device_index) {
        return torch_infini::elapsed_time(
            reinterpret_cast<void*>(start_event),
            reinterpret_cast<void*>(end_event),
            static_cast<c10::DeviceIndex>(device_index));
      });
  m.def("_stream_native_handle", [](std::int64_t stream_id, int device_index) {
    const auto stream = c10::Stream{
        c10::Stream::UNSAFE,
        c10::Device{
            torch_infini::kDeviceType,
            static_cast<c10::DeviceIndex>(device_index)},
        stream_id};
    return reinterpret_cast<std::uintptr_t>(
        torch_infini::get_native_stream_handle(stream));
  });
}
