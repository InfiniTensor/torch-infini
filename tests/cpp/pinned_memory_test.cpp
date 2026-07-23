#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <pybind11/pybind11.h>

#include <cstdint>

namespace {

bool is_pinned_ptr(std::uintptr_t data) {
  return at::detail::getPrivateUse1Hooks().isPinnedPtr(
      reinterpret_cast<const void*>(data));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_pinned_ptr", &is_pinned_ptr);
}
