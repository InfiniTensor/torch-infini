#include <ATen/ops/empty_like.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include "infini_ops.h"
#include "torch_infini.h"

namespace torch_infini {

namespace {

bool is_supported_relu_dtype(c10::ScalarType scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
    case c10::ScalarType::Char:
    case c10::ScalarType::Short:
    case c10::ScalarType::Int:
    case c10::ScalarType::Long:
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Float:
    case c10::ScalarType::Double:
      return true;
    default:
      return false;
  }
}

void check_relu_input(const at::Tensor& self) {
  TORCH_CHECK(
      self.device().type() == kDeviceType,
      "aten::relu expects an infini tensor, got ",
      self.device());
  TORCH_CHECK(
      is_supported_relu_dtype(self.scalar_type()),
      "aten::relu does not support dtype ",
      self.scalar_type());
}

void check_native_relu_support(infini::rt::Device::Type device_type) {
  using DeviceType = infini::rt::Device::Type;

  switch (device_type) {
    case DeviceType::kCpu:
    case DeviceType::kNvidia:
    case DeviceType::kMetax:
    case DeviceType::kMoore:
    case DeviceType::kIluvatar:
      return;
    default:
      TORCH_CHECK(
          false,
          "InfiniOps Relu implementation 0 is unavailable for runtime backend ",
          infini::rt::Device::StringFromType(device_type));
  }
}

} // namespace

at::Tensor relu(const at::Tensor& self) {
  check_relu_input(self);
  const c10::DeviceGuard guard{self.device()};

  const auto runtime_device = infini_ops::to_device(self.device());
  check_native_relu_support(runtime_device.type());
  (void)infini_ops::to_data_type(self.scalar_type());

  auto output =
      at::empty_like(self, self.options(), at::MemoryFormat::Preserve);
  if (output.numel() == 0) {
    return output;
  }

  const auto input_view = infini_ops::to_tensor_view(self);
  const auto output_view = infini_ops::to_tensor_view(output);
  const auto stream = get_current_stream(self.device());
  submit_stream_work(stream, {self, output}, [&](rt::Stream native_stream) {
    const auto context = infini_ops::make_execution_context(native_stream);
    infini::ops::Relu::Call(
        context.handle, context.config, input_view, output_view);
  });
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("relu", TORCH_FN(relu));
}

} // namespace torch_infini
