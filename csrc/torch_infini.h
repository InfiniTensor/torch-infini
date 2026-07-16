#ifndef TORCH_INFINI_TORCH_INFINI_H_
#define TORCH_INFINI_TORCH_INFINI_H_

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <infini/rt.h>
#include <torch/library.h>

#include <cstddef>
#include <optional>
#include <string>

namespace torch_infini {

namespace rt = infini::rt::runtime;

constexpr const char* kBackendName = "infini";
constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

void check(rt::Error status, const char* call);

std::string runtime_error_message(rt::Error status, const char* call);

int device_count() noexcept;
int current_device();
void set_device(int device);
void synchronize(int device);
bool is_available();
std::string device_name(int device);

void register_backend();
bool try_ensure_runtime_backend_for_current_thread() noexcept;
void ensure_runtime_backend_for_current_thread();
std::string runtime_backend_name();

c10::Stream get_current_stream(c10::Device device);
c10::Stream get_default_stream(c10::Device device);
c10::Stream create_stream(c10::Device device, int priority);
c10::Stream get_stream_from_global_pool(
    c10::Device device,
    bool is_high_priority);
c10::Stream exchange_current_stream(c10::Stream stream);
void* get_native_stream_handle(c10::Stream stream);
bool query_stream(const c10::Stream& stream);
void synchronize_stream(const c10::Stream& stream);

c10::Allocator* get_allocator();

at::Tensor empty(
    c10::SymIntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);

at::Tensor empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory);

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);

} // namespace torch_infini

#endif // TORCH_INFINI_TORCH_INFINI_H_
