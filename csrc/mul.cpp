#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/scalar_tensor.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

#include "infini_ops.h"
#include "torch_infini.h"

namespace torch_infini {

namespace {

bool is_wrapped_number(const at::Tensor& tensor) {
  return tensor.device().is_cpu() && tensor.dim() == 0 &&
      tensor.unsafeGetTensorImpl()->is_wrapped_number();
}

void check_mul_inputs(const at::Tensor& self, const at::Tensor& other) {
  const auto other_is_wrapped_number = is_wrapped_number(other);
  TORCH_CHECK(
      self.device().type() == kDeviceType &&
          (other.device().type() == kDeviceType || other_is_wrapped_number),
      "aten::mul.Tensor expects two infini tensors, got ",
      self.device(),
      " and ",
      other.device());
  if (other_is_wrapped_number) {
    TORCH_CHECK(
        at::result_type(self, other) == self.scalar_type(),
        "aten::mul.Tensor does not support type promotion yet");
    return;
  }
  TORCH_CHECK(
      self.device() == other.device(),
      "aten::mul.Tensor requires tensors on the same infini device, got ",
      self.device(),
      " and ",
      other.device());
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "aten::mul.Tensor does not support type promotion yet, got ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
}

void check_native_mul_support(infini::rt::Device::Type device_type) {
  using DeviceType = infini::rt::Device::Type;

  switch (device_type) {
    case DeviceType::kCpu:
    case DeviceType::kNvidia:
    case DeviceType::kAscend:
    case DeviceType::kMetax:
    case DeviceType::kMoore:
    case DeviceType::kIluvatar:
      return;
    default:
      TORCH_CHECK(
          false,
          "InfiniOps Mul implementation 0 is unavailable for runtime backend ",
          infini::rt::Device::StringFromType(device_type));
  }
}

at::Tensor allocate_mul_output(
    const at::Tensor& self,
    c10::IntArrayRef output_size) {
  if (self.sizes().equals(output_size) && self.is_non_overlapping_and_dense()) {
    return at::empty_like(self, self.options(), at::MemoryFormat::Preserve);
  }
  return at::empty(output_size, self.options());
}

at::Tensor copy_wrapped_number_to_infini(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto host_scalar = at::scalar_tensor(
      other.item(),
      at::TensorOptions()
          .dtype(self.scalar_type())
          .device(c10::DeviceType::CPU));
  auto device_scalar = at::empty({}, self.options());
  copy_(device_scalar, host_scalar, false);
  return device_scalar;
}

} // namespace

at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
  const auto self_is_wrapped_number = is_wrapped_number(self);
  const auto swap_inputs =
      self_is_wrapped_number && other.device().type() == kDeviceType;
  const auto& input = swap_inputs ? other : self;
  const auto& multiplier = swap_inputs ? self : other;

  check_mul_inputs(input, multiplier);
  const auto output_size =
      at::infer_size_dimvector(input.sizes(), multiplier.sizes());
  const c10::DeviceGuard guard{input.device()};

  const auto runtime_device = infini_ops::to_device(input.device());
  check_native_mul_support(runtime_device.type());
  (void)infini_ops::to_data_type(input.scalar_type());

  auto output = allocate_mul_output(input, output_size);
  if (output.numel() == 0) {
    return output;
  }

  auto normalized_multiplier = is_wrapped_number(multiplier)
      ? copy_wrapped_number_to_infini(input, multiplier)
      : multiplier;
  const auto self_view =
      infini_ops::to_expanded_tensor_view(input, output_size);
  const auto other_view =
      infini_ops::to_expanded_tensor_view(normalized_multiplier, output_size);
  const auto output_view = infini_ops::to_tensor_view(output);
  const auto stream = get_current_stream(input.device());
  submit_stream_work(
      stream,
      {input, normalized_multiplier, output},
      [&](rt::Stream native_stream) {
        const auto context = infini_ops::make_execution_context(native_stream);
        infini::ops::Mul::Call(
            context.handle, context.config, self_view, other_view, output_view);
      });
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mul.Tensor", TORCH_FN(mul));
}

} // namespace torch_infini
