#include <pybind11/pybind11.h>
#include <torch/extension.h>

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
}
