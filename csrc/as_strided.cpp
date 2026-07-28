#include <ATen/native/Resize.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "torch_infini.h"

namespace torch_infini {

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  auto result = at::detail::make_tensor<c10::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  at::native::setStrided(
      result, size, stride, storage_offset.value_or(self.sym_storage_offset()));
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("as_strided", TORCH_FN(as_strided));
}

} // namespace torch_infini
