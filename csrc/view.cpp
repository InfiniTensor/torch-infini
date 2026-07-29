#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>

#include "torch_infini.h"

namespace torch_infini {

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  const auto inferred_size = at::infer_size_dv(size, self.sym_numel());
  const auto stride = at::detail::computeStride(
      self.sym_sizes(), self.sym_strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is not compatible with input tensor's size and stride "
      "(at least one dimension spans across two contiguous subspaces). "
      "Use .reshape(...) instead.");
  return as_strided(self, inferred_size, *stride, std::nullopt);
}

at::Tensor reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return as_strided(self, size, stride, std::nullopt);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_reshape_alias", TORCH_FN(reshape_alias));
  m.impl("view", TORCH_FN(view));
}

} // namespace torch_infini
