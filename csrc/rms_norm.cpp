#include <ATen/ops/empty_like.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include <limits>
#include <optional>

#include "infini_ops.h"
#include "torch_infini.h"

namespace torch_infini {

namespace {

void check_rms_norm_inputs(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight) {
  TORCH_CHECK(
      input.device().type() == kDeviceType,
      "aten::rms_norm expects an infini tensor, got ",
      input.device());
  TORCH_CHECK_NOT_IMPLEMENTED(
      input.scalar_type() == c10::ScalarType::Float,
      "aten::rms_norm currently supports only torch.float32 inputs, got ",
      input.scalar_type());
  TORCH_CHECK_NOT_IMPLEMENTED(
      input.dim() == 2 || input.dim() == 3,
      "aten::rms_norm currently supports only two-dimensional or "
      "three-dimensional inputs, got ",
      input.dim(),
      " dimensions");
  TORCH_CHECK_NOT_IMPLEMENTED(
      input.is_contiguous(),
      "aten::rms_norm currently supports only contiguous inputs");
  TORCH_CHECK_NOT_IMPLEMENTED(
      normalized_shape.size() == 1,
      "aten::rms_norm currently supports only one normalized dimension");
  TORCH_CHECK(
      normalized_shape[0] == input.sym_size(-1),
      "aten::rms_norm expected normalized_shape to match the last input "
      "dimension");
  TORCH_CHECK_NOT_IMPLEMENTED(
      input.size(-1) > 0,
      "aten::rms_norm currently requires a non-empty normalized dimension");
  TORCH_CHECK_NOT_IMPLEMENTED(
      weight.has_value() && weight->defined(),
      "aten::rms_norm currently requires an affine weight");

  const auto& affine_weight = *weight;
  TORCH_CHECK(
      affine_weight.device().type() == kDeviceType,
      "aten::rms_norm expects an infini weight, got ",
      affine_weight.device());
  TORCH_CHECK(
      affine_weight.device() == input.device(),
      "aten::rms_norm expects input and weight on the same device, got ",
      input.device(),
      " and ",
      affine_weight.device());
  TORCH_CHECK(
      affine_weight.scalar_type() == input.scalar_type(),
      "aten::rms_norm expects input and weight to have the same dtype, got ",
      input.scalar_type(),
      " and ",
      affine_weight.scalar_type());
  TORCH_CHECK(
      affine_weight.dim() == 1 &&
          affine_weight.sym_size(0) == normalized_shape[0],
      "aten::rms_norm expected weight to have shape normalized_shape");
  TORCH_CHECK_NOT_IMPLEMENTED(
      affine_weight.is_contiguous(),
      "aten::rms_norm currently supports only contiguous weights");
}

void check_native_rms_norm_support(infini::rt::Device::Type device_type) {
  using DeviceType = infini::rt::Device::Type;

  switch (device_type) {
    case DeviceType::kCpu:
    case DeviceType::kNvidia:
    case DeviceType::kCambricon:
    case DeviceType::kAscend:
    case DeviceType::kMetax:
    case DeviceType::kMoore:
    case DeviceType::kIluvatar:
      return;
    default:
      TORCH_CHECK(
          false,
          "InfiniOps RmsNorm implementation 0 is unavailable for runtime "
          "backend ",
          infini::rt::Device::StringFromType(device_type));
  }
}

} // namespace

at::Tensor rms_norm(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    std::optional<double> eps) {
  check_rms_norm_inputs(input, normalized_shape, weight);
  const c10::DeviceGuard guard{input.device()};

  const auto runtime_device = infini_ops::to_device(input.device());
  check_native_rms_norm_support(runtime_device.type());
  (void)infini_ops::to_data_type(input.scalar_type());

  auto output =
      at::empty_like(input, input.options(), at::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }

  const auto input_view = infini_ops::to_tensor_view(input);
  const auto weight_view = infini_ops::to_tensor_view(*weight);
  const auto output_view = infini_ops::to_tensor_view(output);
  const auto infini_eps =
      static_cast<float>(eps.value_or(std::numeric_limits<float>::epsilon()));
  const auto stream = get_current_stream(input.device());
  submit_stream_work(
      stream, {input, *weight, output}, [&](rt::Stream native_stream) {
        const auto context = infini_ops::make_execution_context(native_stream);
        infini::ops::RmsNorm::Call(
            context.handle,
            context.config,
            input_view,
            weight_view,
            infini_eps,
            output_view);
      });
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(rms_norm));
}

} // namespace torch_infini
