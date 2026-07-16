#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

#include "infini_ops.h"

namespace py = pybind11;

namespace {

py::dict tensor_metadata(const at::Tensor& tensor) {
  const auto view = torch_infini::infini_ops::to_tensor_view(tensor);

  py::dict metadata;
  metadata["data_ptr"] = reinterpret_cast<std::uintptr_t>(view.data());
  metadata["dtype"] = std::string{infini::rt::kDataTypeToDesc.at(view.dtype())};
  metadata["device_type"] =
      std::string{infini::rt::Device::StringFromType(view.device().type())};
  metadata["device_index"] = view.device().index();
  metadata["shape"] = view.shape();
  metadata["strides"] = view.strides();
  metadata["is_contiguous"] = view.IsContiguous();
  metadata["has_broadcast_dim"] = view.HasBroadcastDim();
  return metadata;
}

py::dict device_metadata(const std::string& device_name) {
  const auto device =
      torch_infini::infini_ops::to_device(c10::Device{device_name});

  py::dict metadata;
  metadata["device_type"] =
      std::string{infini::rt::Device::StringFromType(device.type())};
  metadata["device_index"] = device.index();
  return metadata;
}

py::dict execution_context_metadata(std::uintptr_t stream_address) {
  const auto stream =
      reinterpret_cast<infini::rt::runtime::Stream>(stream_address);
  const auto context = torch_infini::infini_ops::make_execution_context(stream);

  py::dict metadata;
  metadata["stream"] =
      reinterpret_cast<std::uintptr_t>(context.handle.stream());
  metadata["implementation_index"] = context.config.implementation_index();
  return metadata;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_metadata", &tensor_metadata);
  m.def("device_metadata", &device_metadata);
  m.def("execution_context_metadata", &execution_context_metadata);
}
