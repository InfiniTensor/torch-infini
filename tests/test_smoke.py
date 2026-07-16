import threading

import pytest
import torch

import torch_infini  # noqa: F401


def _alternate_device():
    count = torch.infini.device_count()
    if count < 2:
        pytest.skip("requires at least two infini devices")
    initial_device = torch.infini.current_device()
    return initial_device, (initial_device + 1) % count


def test_privateuse1_backend_is_named_infini():
    assert torch.device("infini:0").type == "infini"
    assert hasattr(torch, "infini")


def test_runtime_backend_is_selected_automatically():
    assert torch.infini.is_available()
    assert torch.infini.device_count() > 0


@pytest.mark.parametrize(
    "name",
    [
        "current_stream",
        "default_stream",
        "get_backend",
        "init",
        "is_initialized",
        "manual_seed",
        "manual_seed_all",
    ],
)
def test_unimplemented_api_is_not_advertised(name):
    assert not hasattr(torch.infini, name)


@pytest.mark.skipif(not torch.infini.is_available(), reason="no infini device")
def test_device_management():
    count = torch.infini.device_count()
    initial_device = torch.infini.current_device()
    target_device = (initial_device + 1) % count

    with torch.infini.device(target_device):
        assert torch.infini.current_device() == target_device
        assert torch.infini.get_device_name() == f"infini:{target_device}"
        torch.infini.synchronize()

    assert torch.infini.current_device() == initial_device


def test_explicit_device_synchronize_preserves_current_device():
    initial_device, target_device = _alternate_device()

    try:
        torch.infini.synchronize(target_device)
        assert torch.infini.current_device() == initial_device
    finally:
        torch.infini.set_device(initial_device)


def test_explicit_device_allocation_preserves_current_device():
    initial_device, target_device = _alternate_device()
    tensor = None

    try:
        tensor = torch.empty(16, device=f"infini:{target_device}")
        assert tensor.device.index == target_device
        assert torch.infini.current_device() == initial_device
    finally:
        del tensor
        torch.infini.set_device(initial_device)


def test_explicit_device_allocation_restores_current_device_on_error():
    initial_device, target_device = _alternate_device()

    try:
        with pytest.raises(RuntimeError):
            torch.empty((-1,), device=f"infini:{target_device}")
        assert torch.infini.current_device() == initial_device
    finally:
        torch.infini.set_device(initial_device)


def test_copy_preserves_current_device():
    initial_device, target_device = _alternate_device()
    src = torch.arange(16, dtype=torch.float32)
    first = torch.empty_like(src, device=f"infini:{target_device}")
    second = torch.empty_like(first)
    out = torch.empty_like(src)

    try:
        torch.infini.set_device(initial_device)
        first.copy_(src)
        assert torch.infini.current_device() == initial_device

        out.copy_(first)
        assert torch.infini.current_device() == initial_device

        second.copy_(first)
        assert torch.infini.current_device() == initial_device
    finally:
        del first
        del second
        torch.infini.set_device(initial_device)


@pytest.mark.skipif(not torch.infini.is_available(), reason="no infini device")
def test_empty_and_cpu_roundtrip():
    src = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    dst = torch.empty(src.shape, dtype=src.dtype, device="infini:0")

    dst.copy_(src)
    out = torch.empty_like(src)
    out.copy_(dst)

    torch.testing.assert_close(out, src)


@pytest.mark.skipif(not torch.infini.is_available(), reason="no infini device")
def test_tensor_can_be_destroyed_on_worker_thread():
    tensors = [torch.empty(16, device="infini")]

    worker = threading.Thread(target=tensors.clear)
    worker.start()
    worker.join()

    assert not tensors
    torch.infini.synchronize()


def test_tensor_destruction_preserves_worker_current_device():
    allocation_device, worker_device = _alternate_device()
    tensors = [torch.empty(16, device=f"infini:{allocation_device}")]
    observed = []
    errors = []

    def destroy_tensor():
        try:
            torch.infini.set_device(worker_device)
            tensors.clear()
            observed.append(torch.infini.current_device())
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    worker = threading.Thread(target=destroy_tensor)
    worker.start()
    worker.join()

    assert not errors
    assert observed == [worker_device]
    assert not tensors


if __name__ == "__main__":
    print("torch", torch.__version__)
    print("available", torch.infini.is_available())
    print("count", torch.infini.device_count())
    test_privateuse1_backend_is_named_infini()
    test_empty_and_cpu_roundtrip()
