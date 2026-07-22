#include "infini_ops.h"

#include <ATen/ExpandUtils.h>
#include <c10/util/Exception.h>

#include <limits>

#include "torch_infini.h"

namespace torch_infini::infini_ops {

infini::rt::DataType to_data_type(c10::ScalarType scalar_type) {
  using DataType = infini::rt::DataType;

  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return DataType::kUInt8;
    case c10::ScalarType::Char:
      return DataType::kInt8;
    case c10::ScalarType::Short:
      return DataType::kInt16;
    case c10::ScalarType::Int:
      return DataType::kInt32;
    case c10::ScalarType::Long:
      return DataType::kInt64;
    case c10::ScalarType::UInt16:
      return DataType::kUInt16;
    case c10::ScalarType::UInt32:
      return DataType::kUInt32;
    case c10::ScalarType::UInt64:
      return DataType::kUInt64;
    case c10::ScalarType::Half:
      return DataType::kFloat16;
    case c10::ScalarType::BFloat16:
      return DataType::kBFloat16;
    case c10::ScalarType::Float:
      return DataType::kFloat32;
    case c10::ScalarType::Double:
      return DataType::kFloat64;
    default:
      TORCH_CHECK(
          false,
          "InfiniOps does not support ATen dtype ",
          c10::toString(scalar_type));
  }
}

infini::rt::Device to_device(const c10::Device& device) {
  if (device.is_cpu()) {
    const auto index = device.has_index() ? device.index() : 0;
    TORCH_CHECK(
        index == 0, "InfiniOps only supports CPU device index 0, got ", device);
    return {infini::rt::Device::Type::kCpu, 0};
  }

  TORCH_CHECK(
      device.type() == kDeviceType,
      "InfiniOps adapters only support CPU and infini devices, got ",
      device);
  ensure_runtime_backend_for_current_thread();

  const auto index = device.has_index() ? device.index() : current_device();
  TORCH_CHECK(index >= 0, "infini device index must be non-negative");
  TORCH_CHECK(
      index < device_count(),
      "infini device index ",
      index,
      " is out of range for ",
      device_count(),
      " devices");
  return {infini::rt::runtime_device_type(), index};
}

infini::rt::TensorView::Shape to_shape(c10::IntArrayRef sizes) {
  infini::rt::TensorView::Shape shape;
  shape.reserve(sizes.size());
  for (const auto size : sizes) {
    TORCH_CHECK(size >= 0, "InfiniOps tensor dimensions must be non-negative");
    shape.push_back(static_cast<infini::rt::TensorView::Size>(size));
  }
  return shape;
}

infini::rt::TensorView::Strides to_strides(c10::IntArrayRef strides) {
  infini::rt::TensorView::Strides converted;
  converted.reserve(strides.size());
  for (const auto stride : strides) {
    TORCH_CHECK(
        stride >= std::numeric_limits<infini::rt::TensorView::Stride>::min() &&
            stride <=
                std::numeric_limits<infini::rt::TensorView::Stride>::max(),
        "InfiniOps tensor stride is out of range: ",
        stride);
    converted.push_back(static_cast<infini::rt::TensorView::Stride>(stride));
  }
  return converted;
}

infini::rt::TensorView to_tensor_view(const at::Tensor& tensor) {
  const auto data_type = to_data_type(tensor.scalar_type());
  const auto device = to_device(tensor.device());
  const auto shape = to_shape(tensor.sizes());
  const auto strides = to_strides(tensor.strides());
  return {tensor.data_ptr(), shape, data_type, device, strides};
}

infini::rt::TensorView to_expanded_tensor_view(
    const at::Tensor& tensor,
    c10::IntArrayRef sizes) {
  const auto geometry = at::inferExpandGeometry_dimvector(
      tensor.sizes(), tensor.strides(), sizes);
  const auto data_type = to_data_type(tensor.scalar_type());
  const auto device = to_device(tensor.device());
  const auto shape = to_shape(geometry.sizes);
  const auto strides = to_strides(geometry.strides);
  return {tensor.data_ptr(), shape, data_type, device, strides};
}

ExecutionContext make_execution_context(infini::rt::runtime::Stream stream) {
  ExecutionContext context;
  context.handle.set_stream(reinterpret_cast<void*>(stream));
  context.config.set_implementation_index(0);
  return context;
}

} // namespace torch_infini::infini_ops
