#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include "infini_ops.h"
#include "torch_infini.h"

namespace torch_infini {

namespace {

void check_add_inputs(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  TORCH_CHECK(
      self.device().type() == kDeviceType &&
          other.device().type() == kDeviceType,
      "aten::add.Tensor expects two infini tensors, got ",
      self.device(),
      " and ",
      other.device());
  TORCH_CHECK(
      self.device() == other.device(),
      "aten::add.Tensor requires tensors on the same infini device, got ",
      self.device(),
      " and ",
      other.device());
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "aten::add.Tensor does not support type promotion yet, got ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  TORCH_CHECK(
      alpha.equal(1), "aten::add.Tensor only supports alpha == 1, got ", alpha);
}

void check_native_add_support(infini::rt::Device::Type device_type) {
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
          "InfiniOps Add implementation 0 is unavailable for runtime backend ",
          infini::rt::Device::StringFromType(device_type));
  }
}

at::Tensor allocate_add_output(
    const at::Tensor& self,
    c10::IntArrayRef output_size) {
  if (self.sizes().equals(output_size) && self.is_non_overlapping_and_dense()) {
    return at::empty_like(self, self.options(), at::MemoryFormat::Preserve);
  }
  return at::empty(output_size, self.options());
}

} // namespace

at::Tensor add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  check_add_inputs(self, other, alpha);
  const auto output_size =
      at::infer_size_dimvector(self.sizes(), other.sizes());
  const c10::DeviceGuard guard{self.device()};

  const auto runtime_device = infini_ops::to_device(self.device());
  check_native_add_support(runtime_device.type());
  (void)infini_ops::to_data_type(self.scalar_type());

  auto output = allocate_add_output(self, output_size);
  if (output.numel() == 0) {
    return output;
  }

  const auto self_view = infini_ops::to_expanded_tensor_view(self, output_size);
  const auto other_view =
      infini_ops::to_expanded_tensor_view(other, output_size);
  const auto output_view = infini_ops::to_tensor_view(output);
  const auto stream = get_current_stream(self.device());
  submit_stream_work(stream, [&](rt::Stream native_stream) {
    const auto context = infini_ops::make_execution_context(native_stream);
    infini::ops::Add::Call(
        context.handle, context.config, self_view, other_view, output_view);
  });
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", TORCH_FN(add));
}

} // namespace torch_infini
