#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "torch_infini.h"

namespace torch_infini {

namespace {

using NativeStream = rt::Stream;

struct StreamEntry {
  NativeStream stream{};
  rt::Event query_event{};
  bool query_event_created{false};
  bool query_event_recorded{false};
  bool known_complete{true};
  bool native_handle_exposed{false};
  bool owned{false};
};

c10::DeviceIndex checked_device_index(c10::DeviceIndex device_index) {
  const auto count = static_cast<c10::DeviceIndex>(device_count());
  TORCH_CHECK(
      device_index >= 0 && device_index < count,
      "infini device index ",
      static_cast<int>(device_index),
      " is out of range for ",
      static_cast<int>(count),
      " devices");
  return device_index;
}

c10::Device checked_stream_device(c10::Device device) {
  TORCH_CHECK(
      device.type() == kDeviceType, "expected an infini device, got ", device);
  const auto index = device.has_index()
      ? device.index()
      : static_cast<c10::DeviceIndex>(current_device());
  return c10::Device{kDeviceType, checked_device_index(index)};
}

class StreamRegistry {
 public:
  StreamRegistry() : streams_(static_cast<std::size_t>(device_count())) {
    for (auto& device_streams : streams_) {
      device_streams.emplace(0, StreamEntry{});
    }
  }

  StreamRegistry(const StreamRegistry&) = delete;
  StreamRegistry& operator=(const StreamRegistry&) = delete;

  ~StreamRegistry() {
    if (!try_ensure_runtime_backend_for_current_thread()) {
      return;
    }

    int original_device = 0;
    const bool restore_device = rt::GetDevice(&original_device) == rt::kSuccess;
    for (std::size_t device = 0; device < streams_.size(); ++device) {
      if (rt::SetDevice(static_cast<int>(device)) != rt::kSuccess) {
        continue;
      }
      for (const auto& [stream_id, entry] : streams_[device]) {
        (void)stream_id;
        (void)rt::StreamSynchronize(entry.stream);
        if (entry.query_event_created) {
          (void)rt::EventDestroy(entry.query_event);
        }
        if (entry.owned) {
          (void)rt::StreamDestroy(entry.stream);
        }
      }
    }
    if (restore_device) {
      (void)rt::SetDevice(original_device);
    }
  }

  c10::Stream create(c10::Device device, int priority) {
    TORCH_CHECK(
        priority == 0,
        "InfiniRT does not expose stream priorities; expected priority 0, got ",
        priority);
    device = checked_stream_device(device);
    const c10::DeviceGuard guard{device};

    NativeStream native_stream{};
    check(rt::StreamCreate(&native_stream), "StreamCreate");

    const auto stream_id =
        next_stream_id_.fetch_add(1, std::memory_order_relaxed);
    TORCH_CHECK(stream_id > 0, "infini stream ID space is exhausted");

    try {
      const std::lock_guard<std::mutex> lock{mutex_};
      const auto [iterator, inserted] =
          streams_[static_cast<std::size_t>(device.index())].emplace(
              stream_id,
              StreamEntry{native_stream, {}, false, false, true, false, true});
      (void)iterator;
      TORCH_CHECK(inserted, "duplicate infini stream ID ", stream_id);
    } catch (...) {
      (void)rt::StreamDestroy(native_stream);
      throw;
    }

    return c10::Stream{c10::Stream::UNSAFE, device, stream_id};
  }

  NativeStream native_stream(const c10::Stream& stream) {
    const std::lock_guard<std::mutex> lock{mutex_};
    return checked_entry(stream).stream;
  }

  NativeStream native_stream_for_work(const c10::Stream& stream) {
    const std::lock_guard<std::mutex> lock{mutex_};
    auto& entry = checked_entry(stream);
    mark_work_pending(entry);
    entry.native_handle_exposed = true;
    return entry.stream;
  }

  void submit(
      const c10::Stream& stream,
      const std::function<void(NativeStream)>& submit) {
    const std::lock_guard<std::mutex> lock{mutex_};
    auto& entry = checked_entry(stream);
    mark_work_pending(entry);
    submit(entry.stream);
  }

  void synchronize(const c10::Stream& stream) {
    const std::lock_guard<std::mutex> lock{mutex_};
    auto& entry = checked_entry(stream);
    check(rt::StreamSynchronize(entry.stream), "StreamSynchronize");
    entry.known_complete = true;
  }

  bool query(const c10::Stream& stream) {
    const std::lock_guard<std::mutex> lock{mutex_};
    auto& entry = checked_entry(stream);
    TORCH_CHECK(
        !entry.native_handle_exposed,
        "infini stream query is unavailable after its native handle has been "
        "exposed because InfiniRT does not expose StreamQuery");
    if (entry.known_complete) {
      return true;
    }
    record_query_event(entry);
    entry.known_complete = rt::EventQuery(entry.query_event) == rt::kSuccess;
    return entry.known_complete;
  }

 private:
  static void mark_work_pending(StreamEntry& entry) {
    entry.query_event_recorded = false;
    entry.known_complete = false;
  }

  void record_query_event(StreamEntry& entry) {
    if (!entry.query_event_created) {
      check(rt::EventCreate(&entry.query_event), "EventCreate");
      entry.query_event_created = true;
    }
    if (!entry.query_event_recorded) {
      check(rt::EventRecord(entry.query_event, entry.stream), "EventRecord");
      entry.query_event_recorded = true;
    }
  }

  StreamEntry& checked_entry(const c10::Stream& stream) {
    TORCH_CHECK(
        stream.device_type() == kDeviceType,
        "expected an infini stream, got ",
        stream);
    const auto device_index = checked_device_index(stream.device_index());
    auto& device_streams = streams_[static_cast<std::size_t>(device_index)];
    const auto iterator = device_streams.find(stream.id());
    TORCH_CHECK(
        iterator != device_streams.end(),
        "unknown infini stream ID ",
        stream.id(),
        " on device ",
        static_cast<int>(device_index));
    return iterator->second;
  }

  std::mutex mutex_;
  std::atomic<c10::StreamId> next_stream_id_{1};
  std::vector<std::unordered_map<c10::StreamId, StreamEntry>> streams_;
};

StreamRegistry& stream_registry() {
  static StreamRegistry registry;
  return registry;
}

std::vector<c10::StreamId>& current_stream_ids() {
  thread_local std::vector<c10::StreamId> streams;
  const auto count = static_cast<std::size_t>(device_count());
  if (streams.size() < count) {
    streams.resize(count, 0);
  }
  return streams;
}

NativeStream native_stream(const c10::Stream& stream) {
  ensure_runtime_backend_for_current_thread();
  return stream_registry().native_stream(stream);
}

} // namespace

c10::Stream get_current_stream(c10::Device device) {
  ensure_runtime_backend_for_current_thread();
  device = checked_stream_device(device);
  const auto stream_id =
      current_stream_ids()[static_cast<std::size_t>(device.index())];
  return c10::Stream{c10::Stream::UNSAFE, device, stream_id};
}

c10::Stream get_default_stream(c10::Device device) {
  ensure_runtime_backend_for_current_thread();
  return c10::Stream{c10::Stream::DEFAULT, checked_stream_device(device)};
}

c10::Stream create_stream(c10::Device device, int priority) {
  ensure_runtime_backend_for_current_thread();
  return stream_registry().create(device, priority);
}

c10::Stream get_stream_from_global_pool(
    c10::Device device,
    bool is_high_priority) {
  TORCH_CHECK(
      !is_high_priority,
      "InfiniRT does not expose high-priority stream creation");
  return create_stream(device, 0);
}

c10::Stream exchange_current_stream(c10::Stream stream) {
  ensure_runtime_backend_for_current_thread();
  (void)native_stream(stream);
  const auto device_index = checked_device_index(stream.device_index());
  auto& current = current_stream_ids()[static_cast<std::size_t>(device_index)];
  const auto previous =
      c10::Stream{c10::Stream::UNSAFE, stream.device(), current};
  current = stream.id();
  return previous;
}

void* get_native_stream_handle(c10::Stream stream) {
  ensure_runtime_backend_for_current_thread();
  return reinterpret_cast<void*>(
      stream_registry().native_stream_for_work(stream));
}

void submit_stream_work(
    const c10::Stream& stream,
    const std::function<void(rt::Stream)>& submit) {
  const c10::DeviceGuard guard{stream.device()};
  ensure_runtime_backend_for_current_thread();
  stream_registry().submit(stream, submit);
}

bool query_stream(const c10::Stream& stream) {
  const c10::DeviceGuard guard{stream.device()};
  ensure_runtime_backend_for_current_thread();
  return stream_registry().query(stream);
}

void synchronize_stream(const c10::Stream& stream) {
  const c10::DeviceGuard guard{stream.device()};
  ensure_runtime_backend_for_current_thread();
  stream_registry().synchronize(stream);
}

} // namespace torch_infini
