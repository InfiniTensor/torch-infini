import gc
from pathlib import Path
import threading

import pytest
import torch

import torch_infini  # noqa: F401


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
