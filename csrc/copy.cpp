#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include "torch_infini.h"

namespace torch_infini {

namespace {

bool is_infini(const at::Tensor& tensor) {
  return tensor.device().type() == kDeviceType;
}

bool is_cpu(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::CPU;
}

std::size_t tensor_nbytes(const at::Tensor& tensor) {
  return static_cast<std::size_t>(tensor.numel()) *
      static_cast<std::size_t>(tensor.element_size());
}

void check_copy_shape(const at::Tensor& dst, const at::Tensor& src) {
  TORCH_CHECK(
      dst.scalar_type() == src.scalar_type(),
      "infini copy_ currently requires matching dtype, got ",
      dst.scalar_type(),
      " and ",
      src.scalar_type());
  TORCH_CHECK(
      dst.sizes().equals(src.sizes()),
      "infini copy_ currently requires matching sizes, got ",
      dst.sizes(),
      " and ",
      src.sizes());
  TORCH_CHECK(
      dst.is_contiguous() && src.is_contiguous(),
      "infini copy_ currently only supports contiguous tensors");
}

} // namespace

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  (void)non_blocking;
  check_copy_shape(self, src);

  const auto nbytes = tensor_nbytes(self);
  if (nbytes == 0) {
    return self;
  }

  if (is_infini(self) && is_cpu(src)) {
    set_device(self.device().index());
    check(
        rt::Memcpy(
            self.data_ptr(), src.data_ptr(), nbytes, rt::kMemcpyHostToDevice),
        "Memcpy(HostToDevice)");
    return self;
  }

  if (is_cpu(self) && is_infini(src)) {
    set_device(src.device().index());
    check(
        rt::Memcpy(
            self.data_ptr(), src.data_ptr(), nbytes, rt::kMemcpyDeviceToHost),
        "Memcpy(DeviceToHost)");
    return self;
  }

  if (is_infini(self) && is_infini(src)) {
    TORCH_CHECK(
        self.device().index() == src.device().index(),
        "infini copy_ currently only supports same-device copies");
    set_device(self.device().index());
    check(
        rt::Memcpy(
            self.data_ptr(), src.data_ptr(), nbytes, rt::kMemcpyDeviceToDevice),
        "Memcpy(DeviceToDevice)");
    return self;
  }

  TORCH_CHECK(false, "infini copy_ expected a CPU/infini tensor pair");
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("copy_", TORCH_FN(copy_));
}

} // namespace torch_infini
