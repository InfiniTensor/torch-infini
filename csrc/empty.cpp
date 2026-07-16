#include <ATen/EmptyTensor.h>
#include <ATen/InitialTensorOptions.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>

#include "torch_infini.h"

namespace torch_infini {

namespace {

c10::Device normalize_device(std::optional<c10::Device> device) {
  if (!device.has_value()) {
    return c10::Device{
        kDeviceType, static_cast<c10::DeviceIndex>(current_device())};
  }
  TORCH_CHECK(
      device->type() == kDeviceType,
      "expected an infini device, got ",
      *device);
  if (device->has_index()) {
    return *device;
  }
  return c10::Device{
      kDeviceType, static_cast<c10::DeviceIndex>(current_device())};
}

c10::ScalarType normalize_dtype(std::optional<c10::ScalarType> dtype) {
  return dtype.value_or(at::get_default_dtype_as_scalartype());
}

void check_options(
    std::optional<c10::Layout> layout,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(
      layout.value_or(c10::Layout::Strided) == c10::Layout::Strided,
      "infini tensors currently only support strided layout");
  TORCH_CHECK(
      !pin_memory.value_or(false),
      "infini tensors do not support pinned host memory yet");
}

const c10::DispatchKeySet& dispatch_key_set() {
  static const c10::DispatchKeySet key_set(c10::DispatchKey::PrivateUse1);
  return key_set;
}

} // namespace

at::Tensor empty(
    c10::SymIntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  check_options(layout, pin_memory);
  const auto normalized_device = normalize_device(device);
  const c10::DeviceGuard guard{normalized_device};
  return at::detail::empty_generic_symint(
      size,
      get_allocator(),
      dispatch_key_set(),
      normalize_dtype(dtype),
      memory_format);
}

at::Tensor empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  check_options(layout, pin_memory);
  const auto normalized_device = normalize_device(device);
  const c10::DeviceGuard guard{normalized_device};
  return at::detail::empty_strided_symint_generic(
      size,
      stride,
      get_allocator(),
      dispatch_key_set(),
      normalize_dtype(dtype));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
  m.impl("empty_strided", TORCH_FN(empty_strided));
}

} // namespace torch_infini
