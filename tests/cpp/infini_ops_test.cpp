#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <c10/core/DeviceGuard.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <string>
#include <thread>

#include "infini_ops.h"
#include "torch_infini.h"

namespace py = pybind11;

namespace {

py::dict tensor_metadata(const at::Tensor& tensor) {
  const auto view = torch_infini::infini_ops::to_tensor_view(tensor);

  py::dict metadata;
  metadata["data_ptr"] = reinterpret_cast<std::uintptr_t>(view.data());
  metadata["dtype"] = std::string{infini::rt::kDataTypeToDesc.at(view.dtype())};
  metadata["device_type"] =
      std::string{infini::rt::Device::StringFromType(view.device().type())};
  metadata["device_index"] = view.device().index();
  metadata["shape"] = view.shape();
  metadata["strides"] = view.strides();
  metadata["is_contiguous"] = view.IsContiguous();
  metadata["has_broadcast_dim"] = view.HasBroadcastDim();
  return metadata;
}

py::dict device_metadata(const std::string& device_name) {
  const auto device =
      torch_infini::infini_ops::to_device(c10::Device{device_name});

  py::dict metadata;
  metadata["device_type"] =
      std::string{infini::rt::Device::StringFromType(device.type())};
  metadata["device_index"] = device.index();
  return metadata;
}

py::dict execution_context_metadata(std::uintptr_t stream_address) {
  const auto stream =
      reinterpret_cast<infini::rt::runtime::Stream>(stream_address);
  const auto context = torch_infini::infini_ops::make_execution_context(stream);

  py::dict metadata;
  metadata["stream"] =
      reinterpret_cast<std::uintptr_t>(context.handle.stream());
  metadata["implementation_index"] = context.config.implementation_index();
  return metadata;
}

py::dict current_execution_context_metadata(const std::string& device_name) {
  const c10::Device device{device_name};
  const auto stream = torch_infini::get_current_stream(device);
  const auto native_stream = reinterpret_cast<torch_infini::rt::Stream>(
      torch_infini::get_native_stream_handle(stream));
  const auto context =
      torch_infini::infini_ops::make_execution_context(native_stream);

  py::dict metadata;
  metadata["stream"] =
      reinterpret_cast<std::uintptr_t>(context.handle.stream());
  metadata["implementation_index"] = context.config.implementation_index();
  return metadata;
}

bool stream_synchronize_waits_for_submission(const std::string& device_name) {
  const c10::Device device{device_name};
  const c10::DeviceGuard guard{device};
  const auto stream = torch_infini::get_current_stream(device);

  std::promise<void> submission_started;
  auto submission_started_future = submission_started.get_future();
  std::promise<void> release_submission;
  auto release_submission_future = release_submission.get_future().share();
  std::promise<void> synchronization_started;
  auto synchronization_started_future = synchronization_started.get_future();
  std::promise<void> synchronization_finished;
  auto synchronization_finished_future = synchronization_finished.get_future();
  std::atomic<bool> submission_started_signaled{false};
  std::exception_ptr submission_error;
  std::exception_ptr synchronization_error;
  const auto signal_submission_started = [&] {
    if (!submission_started_signaled.exchange(true)) {
      submission_started.set_value();
    }
  };

  std::thread submitter([&] {
    try {
      const c10::DeviceGuard thread_guard{device};
      torch_infini::submit_stream_work(stream, [&](torch_infini::rt::Stream) {
        signal_submission_started();
        release_submission_future.wait();
      });
    } catch (...) {
      submission_error = std::current_exception();
      signal_submission_started();
    }
  });

  submission_started_future.wait();
  std::thread synchronizer([&] {
    try {
      const c10::DeviceGuard thread_guard{device};
      synchronization_started.set_value();
      torch_infini::synchronize_stream(stream);
      synchronization_finished.set_value();
    } catch (...) {
      synchronization_error = std::current_exception();
      synchronization_finished.set_value();
    }
  });

  synchronization_started_future.wait();
  const auto synchronized_while_submission_open =
      synchronization_finished_future.wait_for(
          std::chrono::milliseconds{500}) == std::future_status::ready;
  release_submission.set_value();
  submitter.join();
  synchronizer.join();

  if (submission_error != nullptr) {
    std::rethrow_exception(submission_error);
  }
  if (synchronization_error != nullptr) {
    std::rethrow_exception(synchronization_error);
  }
  return !synchronized_while_submission_open;
}

bool stream_submission_waits_for_synchronous_work(
    const std::string& device_name) {
  const c10::Device device{device_name};
  const c10::DeviceGuard guard{device};
  const auto stream = torch_infini::get_current_stream(device);

  std::promise<void> synchronous_work_started;
  auto synchronous_work_started_future = synchronous_work_started.get_future();
  std::promise<void> release_synchronous_work;
  auto release_synchronous_work_future =
      release_synchronous_work.get_future().share();
  std::promise<void> submission_started;
  auto submission_started_future = submission_started.get_future();
  std::promise<void> submission_finished;
  auto submission_finished_future = submission_finished.get_future();
  std::atomic<bool> synchronous_work_started_signaled{false};
  std::exception_ptr synchronous_work_error;
  std::exception_ptr submission_error;
  const auto signal_synchronous_work_started = [&] {
    if (!synchronous_work_started_signaled.exchange(true)) {
      synchronous_work_started.set_value();
    }
  };

  std::thread synchronous_worker([&] {
    try {
      const c10::DeviceGuard thread_guard{device};
      torch_infini::run_synchronous_stream_work(stream, [&] {
        signal_synchronous_work_started();
        release_synchronous_work_future.wait();
      });
    } catch (...) {
      synchronous_work_error = std::current_exception();
      signal_synchronous_work_started();
    }
  });

  synchronous_work_started_future.wait();
  std::thread submitter([&] {
    try {
      const c10::DeviceGuard thread_guard{device};
      submission_started.set_value();
      torch_infini::submit_stream_work(stream, [](torch_infini::rt::Stream) {});
      submission_finished.set_value();
    } catch (...) {
      submission_error = std::current_exception();
      submission_finished.set_value();
    }
  });

  submission_started_future.wait();
  const auto submitted_while_synchronous_work_open =
      submission_finished_future.wait_for(std::chrono::milliseconds{500}) ==
      std::future_status::ready;
  release_synchronous_work.set_value();
  synchronous_worker.join();
  submitter.join();
  torch_infini::synchronize_stream(stream);

  if (synchronous_work_error != nullptr) {
    std::rethrow_exception(synchronous_work_error);
  }
  if (submission_error != nullptr) {
    std::rethrow_exception(submission_error);
  }
  return !submitted_while_synchronous_work_open;
}

void copy_storage_from_cpu(at::Tensor destination, const at::Tensor& source) {
  TORCH_CHECK(
      destination.device().type() == torch_infini::kDeviceType,
      "destination must be an infini tensor");
  TORCH_CHECK(source.device().is_cpu(), "source must be a CPU tensor");
  TORCH_CHECK(
      destination.storage().nbytes() == source.storage().nbytes(),
      "source and destination storage sizes must match");

  const c10::DeviceGuard guard{destination.device()};
  torch_infini::check(
      torch_infini::rt::Memcpy(
          destination.data_ptr(),
          source.data_ptr(),
          source.storage().nbytes(),
          torch_infini::rt::kMemcpyHostToDevice),
      "Memcpy(HostToDevice)");
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_metadata", &tensor_metadata);
  m.def("device_metadata", &device_metadata);
  m.def("execution_context_metadata", &execution_context_metadata);
  m.def(
      "current_execution_context_metadata",
      &current_execution_context_metadata);
  m.def(
      "stream_synchronize_waits_for_submission",
      &stream_synchronize_waits_for_submission);
  m.def(
      "stream_submission_waits_for_synchronous_work",
      &stream_submission_waits_for_synchronous_work);
  m.def("copy_storage_from_cpu", &copy_storage_from_cpu);
}
