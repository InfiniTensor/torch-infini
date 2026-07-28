#include <ATen/ExpandUtils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include <array>

#include "infini_ops.h"
#include "torch_infini.h"

namespace torch_infini {

namespace {

void check_addmm_inputs(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  TORCH_CHECK(
      self.device().type() == kDeviceType &&
          mat1.device().type() == kDeviceType &&
          mat2.device().type() == kDeviceType,
      "aten::addmm expects three infini tensors, got ",
      self.device(),
      ", ",
      mat1.device(),
      ", and ",
      mat2.device());
  TORCH_CHECK(
      self.device() == mat1.device() && self.device() == mat2.device(),
      "aten::addmm requires tensors on the same infini device, got ",
      self.device(),
      ", ",
      mat1.device(),
      ", and ",
      mat2.device());
  TORCH_CHECK(
      self.scalar_type() == mat1.scalar_type() &&
          self.scalar_type() == mat2.scalar_type(),
      "aten::addmm requires matching dtypes, got ",
      self.scalar_type(),
      ", ",
      mat1.scalar_type(),
      ", and ",
      mat2.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Float,
      "aten::addmm currently only supports torch.float32, got ",
      self.scalar_type());
  TORCH_CHECK(
      mat1.dim() == 2 && mat2.dim() == 2,
      "aten::addmm expects 2D matrix tensors, got ",
      mat1.dim(),
      "D and ",
      mat2.dim(),
      "D tensors");
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0),
      "aten::addmm requires matching matrix inner dimensions, got ",
      mat1.sizes(),
      " and ",
      mat2.sizes());

  const std::array<int64_t, 2> output_size{mat1.size(0), mat2.size(1)};
  const c10::IntArrayRef output_size_ref{output_size};
  const auto broadcast_size =
      at::infer_size_dimvector(self.sizes(), output_size_ref);
  TORCH_CHECK(
      c10::IntArrayRef(broadcast_size).equals(output_size_ref),
      "aten::addmm input shape ",
      self.sizes(),
      " cannot be broadcast to matrix result shape ",
      output_size_ref);
  TORCH_CHECK(
      beta.equal(1),
      "aten::addmm currently only supports beta == 1, got ",
      beta);
  TORCH_CHECK(
      alpha.equal(1),
      "aten::addmm currently only supports alpha == 1, got ",
      alpha);
}

void check_native_addmm_support(infini::rt::Device::Type device_type) {
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
          "The composed InfiniOps Add/Gemm path for aten::addmm is unavailable "
          "for runtime backend ",
          infini::rt::Device::StringFromType(device_type));
  }
}

} // namespace

at::Tensor addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  check_addmm_inputs(self, mat1, mat2, beta, alpha);
  const c10::DeviceGuard guard{self.device()};

  const auto runtime_device = infini_ops::to_device(self.device());
  check_native_addmm_support(runtime_device.type());

  auto product = torch_infini::mm(mat1, mat2);
  return torch_infini::add(product, self, at::Scalar{1});
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("addmm", TORCH_FN(addmm));
}

} // namespace torch_infini
