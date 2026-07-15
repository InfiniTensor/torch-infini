import pytest
import torch

import torch_infini  # noqa: F401


def test_privateuse1_backend_is_named_infini():
    assert torch.device("infini:0").type == "infini"
    assert hasattr(torch, "infini")


@pytest.mark.parametrize(
    "name",
    [
        "current_stream",
        "default_stream",
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


@pytest.mark.skipif(not torch.infini.is_available(), reason="no infini device")
def test_empty_and_cpu_roundtrip():
    src = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    dst = torch.empty(src.shape, dtype=src.dtype, device="infini:0")

    dst.copy_(src)
    out = torch.empty_like(src)
    out.copy_(dst)

    torch.testing.assert_close(out, src)


if __name__ == "__main__":
    print("torch", torch.__version__)
    print("available", torch.infini.is_available())
    print("count", torch.infini.device_count())
    test_privateuse1_backend_is_named_infini()
    test_empty_and_cpu_roundtrip()
