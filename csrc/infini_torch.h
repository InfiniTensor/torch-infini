#ifndef TORCH_INFINI_INFINI_TORCH_H_
#define TORCH_INFINI_INFINI_TORCH_H_

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <infini/rt.h>
#include <torch/library.h>

#include <cstddef>
#include <optional>
#include <string>

namespace torch_infini {

namespace rt = infini::rt::runtime;

constexpr const char* kBackendName = "infini";
constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

void Check(rt::Error status, const char* call);

std::string RuntimeErrorMessage(rt::Error status, const char* call);

int DeviceCount() noexcept;
int CurrentDevice();
void SetDevice(int device);
void Synchronize(int device);
bool IsAvailable();
std::string DeviceName(int device);

void InitializeBackend();

c10::Allocator* GetInfiniAllocator();

at::Tensor InfiniEmpty(
    c10::SymIntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);

at::Tensor InfiniEmptyStrided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory);

at::Tensor& InfiniCopy(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking);

} // namespace torch_infini

#endif // TORCH_INFINI_INFINI_TORCH_H_
