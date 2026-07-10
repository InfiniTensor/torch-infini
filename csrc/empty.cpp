#include <ATen/EmptyTensor.h>
#include <ATen/InitialTensorOptions.h>
#include <c10/core/Allocator.h>
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>

#include "infini_torch.h"

namespace torch_infini {

namespace {

c10::Device NormalizeDevice(std::optional<c10::Device> device) {
  if (!device.has_value()) {
    return c10::Device{
        kDeviceType, static_cast<c10::DeviceIndex>(CurrentDevice())};
  }
  TORCH_CHECK(
      device->type() == kDeviceType,
      "expected an infini device, got ",
      *device);
  if (device->has_index()) {
    return *device;
  }
  return c10::Device{
      kDeviceType, static_cast<c10::DeviceIndex>(CurrentDevice())};
}

c10::ScalarType NormalizeDtype(std::optional<c10::ScalarType> dtype) {
  return dtype.value_or(at::get_default_dtype_as_scalartype());
}

void CheckOptions(
    std::optional<c10::Layout> layout,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(
      layout.value_or(c10::Layout::Strided) == c10::Layout::Strided,
      "infini tensors only support strided layout in the MVP");
  TORCH_CHECK(
      !pin_memory.value_or(false),
      "infini tensors do not support pinned host memory yet");
}

const c10::DispatchKeySet& DispatchKeySet() {
  static const c10::DispatchKeySet key_set(c10::DispatchKey::PrivateUse1);
  return key_set;
}

} // namespace

at::Tensor InfiniEmpty(
    c10::SymIntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  CheckOptions(layout, pin_memory);
  const auto normalized_device = NormalizeDevice(device);
  SetDevice(normalized_device.index());
  return at::detail::empty_generic_symint(
      size,
      GetInfiniAllocator(),
      DispatchKeySet(),
      NormalizeDtype(dtype),
      memory_format);
}

at::Tensor InfiniEmptyStrided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  CheckOptions(layout, pin_memory);
  const auto normalized_device = NormalizeDevice(device);
  SetDevice(normalized_device.index());
  return at::detail::empty_strided_symint_generic(
      size,
      stride,
      GetInfiniAllocator(),
      DispatchKeySet(),
      NormalizeDtype(dtype));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(InfiniEmpty));
  m.impl("empty_strided", TORCH_FN(InfiniEmptyStrided));
}

} // namespace torch_infini
