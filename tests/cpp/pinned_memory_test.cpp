#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/from_blob.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <utility>

namespace {

bool is_pinned_ptr(std::uintptr_t data) {
  return at::detail::getPrivateUse1Hooks().isPinnedPtr(
      reinterpret_cast<const void*>(data));
}

at::Tensor with_alternative_context(const at::Tensor& tensor) {
  auto owner = tensor;
  return at::from_blob(
      tensor.data_ptr(),
      tensor.sizes(),
      tensor.strides(),
      [owner = std::move(owner)](void*) mutable { owner = at::Tensor{}; },
      tensor.options(),
      tensor.device());
}

std::uintptr_t storage_context(const at::Tensor& tensor) {
  return reinterpret_cast<std::uintptr_t>(
      tensor.storage().data_ptr().get_context());
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_pinned_ptr", &is_pinned_ptr);
  m.def("with_alternative_context", &with_alternative_context);
  m.def("storage_context", &storage_context);
}
