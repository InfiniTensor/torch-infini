#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include "infini_torch.h"

namespace torch_infini {

namespace {

bool IsInfini(const at::Tensor& tensor) {
  return tensor.device().type() == kDeviceType;
}

bool IsCpu(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::CPU;
}

std::size_t TensorNbytes(const at::Tensor& tensor) {
  return static_cast<std::size_t>(tensor.numel()) *
      static_cast<std::size_t>(tensor.element_size());
}

void CheckCopyShape(const at::Tensor& dst, const at::Tensor& src) {
  TORCH_CHECK(
      dst.scalar_type() == src.scalar_type(),
      "infini copy_ MVP requires matching dtype, got ",
      dst.scalar_type(),
      " and ",
      src.scalar_type());
  TORCH_CHECK(
      dst.sizes().equals(src.sizes()),
      "infini copy_ MVP requires matching sizes, got ",
      dst.sizes(),
      " and ",
      src.sizes());
  TORCH_CHECK(
      dst.is_contiguous() && src.is_contiguous(),
      "infini copy_ MVP only supports contiguous tensors");
}

} // namespace

at::Tensor& InfiniCopy(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  (void)non_blocking;
  CheckCopyShape(self, src);

  const auto nbytes = TensorNbytes(self);
  if (nbytes == 0) {
    return self;
  }

  if (IsInfini(self) && IsCpu(src)) {
    SetDevice(self.device().index());
    Check(
        rt::Memcpy(
            self.data_ptr(), src.data_ptr(), nbytes, rt::kMemcpyHostToDevice),
        "Memcpy(HostToDevice)");
    return self;
  }

  if (IsCpu(self) && IsInfini(src)) {
    SetDevice(src.device().index());
    Check(
        rt::Memcpy(
            self.data_ptr(), src.data_ptr(), nbytes, rt::kMemcpyDeviceToHost),
        "Memcpy(DeviceToHost)");
    return self;
  }

  if (IsInfini(self) && IsInfini(src)) {
    TORCH_CHECK(
        self.device().index() == src.device().index(),
        "infini copy_ MVP only supports same-device copies");
    SetDevice(self.device().index());
    Check(
        rt::Memcpy(
            self.data_ptr(), src.data_ptr(), nbytes, rt::kMemcpyDeviceToDevice),
        "Memcpy(DeviceToDevice)");
    return self;
  }

  TORCH_CHECK(false, "infini copy_ expected a CPU/infini tensor pair");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("copy_", TORCH_FN(InfiniCopy));
}

} // namespace torch_infini
