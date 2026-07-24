import gc
from pathlib import Path
import threading

import pytest
import torch

import torch_infini


REPO_ROOT = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.skipif(
    not torch.infini.is_available(), reason="no infini device"
)


@pytest.fixture(scope="session")
def pinned_memory_test_module(tmp_path_factory):
    from torch.utils.cpp_extension import load

    return load(
        name="torch_infini_pinned_memory_test",
        sources=[str(REPO_ROOT / "tests" / "cpp" / "pinned_memory_test.cpp")],
        build_directory=str(tmp_path_factory.mktemp("pinned-memory-extension")),
        verbose=True,
    )


def test_tensor_pin_memory_uses_infini_host_allocator(pinned_memory_test_module):
    source = torch.arange(16, dtype=torch.float32)

    assert not source.is_pinned("infini")
    assert not pinned_memory_test_module.is_pinned_ptr(source.data_ptr())

    pinned = source.pin_memory("infini")

    assert pinned.device.type == "cpu"
    assert pinned.is_pinned("infini")
    assert pinned_memory_test_module.is_pinned_ptr(pinned.data_ptr())
    torch.testing.assert_close(pinned, source)


def test_storage_pin_memory_uses_infini_host_allocator(pinned_memory_test_module):
    source = torch.arange(16, dtype=torch.uint8)

    pinned = source.untyped_storage().pin_memory("infini")
    restored = torch.empty(0, dtype=torch.uint8).set_(pinned, 0, (16,), (1,))

    assert pinned.device.type == "cpu"
    assert pinned.is_pinned("infini")
    assert pinned_memory_test_module.is_pinned_ptr(pinned.data_ptr())
    torch.testing.assert_close(restored, source)


def test_pinned_views_and_interior_pointers_are_recognized(
    pinned_memory_test_module,
):
    pinned = torch.arange(16, dtype=torch.float32).pin_memory("infini")
    view = pinned[3:]

    assert view.data_ptr() > pinned.data_ptr()
    assert view.is_pinned("infini")
    assert pinned_memory_test_module.is_pinned_ptr(view.data_ptr())


def test_pinned_memory_copies_roundtrip_synchronously():
    source = torch.arange(16, dtype=torch.float32).pin_memory("infini")
    device = torch.empty_like(source, device="infini")
    result = torch.zeros_like(source).pin_memory("infini")

    device.copy_(source, non_blocking=False)
    result.copy_(device, non_blocking=False)

    torch.testing.assert_close(result, source)


def test_nonblocking_copy_request_falls_back_without_async_memcpy():
    if torch_infini._C._runtime_backend_name() != "cpu":
        pytest.skip("requires pinned host memory without asynchronous memcpy")

    stream = torch.infini.Stream()
    source = torch.arange(16, dtype=torch.float32).pin_memory("infini")
    device = torch.empty_like(source, device="infini")
    result = torch.zeros_like(source).pin_memory("infini")

    with torch.infini.stream(stream):
        device.copy_(source, non_blocking=True)
        result.copy_(device, non_blocking=True)

    assert stream.query()
    torch.testing.assert_close(result, source)


def _requires_nonblocking_host_copies():
    if torch_infini._C._runtime_backend_name() not in {
        "nvidia",
        "metax",
        "moore",
        "iluvatar",
        "hygon",
    }:
        pytest.skip("selected InfiniRT backend has no non-blocking host copies")


def _queue_pending_work(stream, numel=16 * 1024 * 1024):
    source = torch.empty(numel, dtype=torch.float32, device="infini")
    source.copy_(torch.ones(numel, dtype=torch.float32))
    with torch.infini.stream(stream):
        result = torch.add(source, source)
    return source, result


def _with_alternative_context(pinned_memory_test_module, tensor):
    wrapped = pinned_memory_test_module.with_alternative_context(tensor)
    assert wrapped.data_ptr() == tensor.data_ptr()
    assert pinned_memory_test_module.storage_context(
        wrapped
    ) != pinned_memory_test_module.storage_context(tensor)
    return wrapped


def test_pinned_host_to_device_copy_returns_before_completion():
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    source = torch.arange(262144, dtype=torch.float32).pin_memory("infini")
    destination = torch.empty_like(source, device="infini")

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)

    assert not stream.query()
    stream.synchronize()
    actual = torch.empty_like(source)
    actual.copy_(destination)
    torch.testing.assert_close(actual, source)
    assert blocker


def test_pinned_device_to_host_copy_returns_before_completion():
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    expected = torch.arange(262144, dtype=torch.float32)
    source = torch.empty_like(expected, device="infini")
    source.copy_(expected)
    destination = torch.empty_like(expected).pin_memory("infini")

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)

    assert not stream.query()
    stream.synchronize()
    torch.testing.assert_close(destination, expected)
    assert blocker


def test_sliced_pinned_host_copy_records_storage_context():
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    allocation = torch.arange(262147, dtype=torch.float32).pin_memory("infini")
    source = allocation[3:]
    expected = source.clone()
    destination = torch.empty_like(source, device="infini")

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)
    assert not stream.query()

    del allocation, source
    gc.collect()

    assert stream.query()
    actual = torch.empty_like(expected)
    actual.copy_(destination)
    torch.testing.assert_close(actual, expected)
    assert blocker


def test_alternative_host_context_falls_back_to_pointer_range(
    pinned_memory_test_module,
):
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    allocation = torch.arange(262147, dtype=torch.float32).pin_memory("infini")
    source = _with_alternative_context(pinned_memory_test_module, allocation[3:])
    expected = source.clone()
    destination = torch.empty_like(source, device="infini")
    del allocation
    gc.collect()

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)
    assert not stream.query()

    del source
    gc.collect()

    assert stream.query()
    actual = torch.empty_like(expected)
    actual.copy_(destination)
    torch.testing.assert_close(actual, expected)
    assert blocker


def test_external_device_destination_uses_synchronous_fallback(
    pinned_memory_test_module,
):
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    source = torch.arange(262144, dtype=torch.float32).pin_memory("infini")
    allocation = torch.empty_like(source, device="infini")
    destination = _with_alternative_context(pinned_memory_test_module, allocation)
    del allocation
    gc.collect()

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)

    assert stream.query()
    actual = torch.empty_like(source)
    actual.copy_(destination)
    torch.testing.assert_close(actual, source)
    assert blocker


def test_external_device_source_uses_synchronous_fallback(
    pinned_memory_test_module,
):
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    expected = torch.arange(262144, dtype=torch.float32)
    allocation = torch.empty_like(expected, device="infini")
    allocation.copy_(expected)
    source = _with_alternative_context(pinned_memory_test_module, allocation)
    destination = torch.empty_like(expected).pin_memory("infini")
    del allocation
    gc.collect()

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)

    assert stream.query()
    torch.testing.assert_close(destination, expected)
    assert blocker


@pytest.mark.parametrize("release", ["host", "device"])
def test_host_to_device_copy_keeps_storage_alive(release):
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    source = torch.arange(262144, dtype=torch.float32).pin_memory("infini")
    expected = source.clone()
    destination = torch.empty_like(source, device="infini")

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)
    assert not stream.query()

    if release == "host":
        del source
    else:
        del destination
    gc.collect()

    assert stream.query()
    assert blocker
    if release == "host":
        actual = torch.empty_like(expected)
        actual.copy_(destination)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("release", ["host", "device"])
def test_device_to_host_copy_keeps_storage_alive(release):
    _requires_nonblocking_host_copies()
    stream = torch.infini.Stream()
    blocker = _queue_pending_work(stream)
    expected = torch.arange(262144, dtype=torch.float32)
    source = torch.empty_like(expected, device="infini")
    source.copy_(expected)
    destination = torch.empty_like(expected).pin_memory("infini")

    with torch.infini.stream(stream):
        destination.copy_(source, non_blocking=True)
    assert not stream.query()

    if release == "host":
        del destination
    else:
        del source
    gc.collect()

    assert stream.query()
    assert blocker
    if release == "device":
        torch.testing.assert_close(destination, expected)


def test_freed_pinned_allocation_is_removed_from_live_ranges(
    pinned_memory_test_module,
):
    pinned_tensors = [torch.arange(16, dtype=torch.float32).pin_memory("infini")]
    pointer = pinned_tensors[0].data_ptr()

    assert pinned_memory_test_module.is_pinned_ptr(pointer)

    worker = threading.Thread(target=pinned_tensors.clear)
    worker.start()
    worker.join()
    gc.collect()

    assert not pinned_memory_test_module.is_pinned_ptr(pointer)
