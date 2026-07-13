#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "torch_infini.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch_infini::initialize_backend();

  m.def("initialize", &torch_infini::initialize_backend);
  m.def("is_available", &torch_infini::is_available);
  m.def("device_count", &torch_infini::device_count);
  m.def("current_device", &torch_infini::current_device);
  m.def("set_device", &torch_infini::set_device);
  m.def("synchronize", &torch_infini::synchronize);
  m.def("get_device_name", &torch_infini::device_name);
}
