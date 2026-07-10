#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "infini_torch.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch_infini::InitializeBackend();

  m.def("initialize", &torch_infini::InitializeBackend);
  m.def("is_available", &torch_infini::IsAvailable);
  m.def("device_count", &torch_infini::DeviceCount);
  m.def("current_device", &torch_infini::CurrentDevice);
  m.def("set_device", &torch_infini::SetDevice);
  m.def("synchronize", &torch_infini::Synchronize);
  m.def("get_device_name", &torch_infini::DeviceName);
}
