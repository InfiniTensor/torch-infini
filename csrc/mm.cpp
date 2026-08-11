#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <optional>

#include "infini_ops.h"
#include "torch_infini.h"

namespace torch_infini {

namespace {

void check_mm_inputs(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(
      self.device().type() == kDeviceType &&
          mat2.device().type() == kDeviceType,
      "aten::mm expects two infini tensors, got ",
      self.device(),
      " and ",
      mat2.device());
  TORCH_CHECK(
      self.device() == mat2.device(),
      "aten::mm requires tensors on the same infini device, got ",
      self.device(),
      " and ",
      mat2.device());
  TORCH_CHECK(
      self.scalar_type() == mat2.scalar_type(),
      "aten::mm requires matching dtypes, got ",
      self.scalar_type(),
      " and ",
      mat2.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Float,
      "aten::mm currently only supports torch.float32, got ",
      self.scalar_type());
  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2,
      "aten::mm expects 2D tensors, got ",
      self.dim(),
      "D and ",
      mat2.dim(),
      "D tensors");
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "aten::mm requires matching inner dimensions, got ",
      self.sizes(),
      " and ",
      mat2.sizes());
  TORCH_CHECK(
      self.is_non_overlapping_and_dense() &&
          mat2.is_non_overlapping_and_dense(),
      "aten::mm currently only supports non-overlapping dense inputs");
}

void check_native_gemm_support(infini::rt::Device::Type device_type) {
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
          "InfiniOps Gemm implementation 0 is unavailable for runtime backend ",
          infini::rt::Device::StringFromType(device_type));
  }
}

void zero_mm_output(const at::Tensor& output, const c10::Stream& stream) {
  const auto nbytes = static_cast<std::size_t>(output.numel()) *
      static_cast<std::size_t>(output.element_size());
  run_synchronous_stream_work(stream, [&] {
    check(rt::Memset(output.data_ptr(), 0, nbytes), "Memset");
  });
}

} // namespace

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  check_mm_inputs(self, mat2);
  const c10::DeviceGuard guard{self.device()};

  const auto runtime_device = infini_ops::to_device(self.device());
  check_native_gemm_support(runtime_device.type());
  (void)infini_ops::to_data_type(self.scalar_type());

  auto output = at::empty({self.size(0), mat2.size(1)}, self.options());
  if (output.numel() == 0) {
    return output;
  }

  const auto stream = get_current_stream(self.device());
  if (self.size(1) == 0) {
    zero_mm_output(output, stream);
    return output;
  }

  const auto self_view = infini_ops::to_tensor_view(self);
  const auto mat2_view = infini_ops::to_tensor_view(mat2);
  const auto output_view = infini_ops::to_tensor_view(output);
  submit_stream_work(
      stream, {self, mat2, output}, [&](rt::Stream native_stream) {
        const auto context = infini_ops::make_execution_context(native_stream);
        infini::ops::Gemm::Call(
            context.handle,
            context.config,
            self_view,
            mat2_view,
            std::optional<infini::ops::Tensor>{},
            std::optional<float>{1.0F},
            std::optional<float>{0.0F},
            std::optional<int>{0},
            std::optional<int>{0},
            output_view);
      });
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", TORCH_FN(mm));
}

} // namespace torch_infini
