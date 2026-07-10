import pytest
import torch

import torch_infini  # noqa: F401


def test_privateuse1_backend_is_named_infini():
    assert torch.device("infini:0").type == "infini"
    assert hasattr(torch, "infini")


@pytest.mark.skipif(not torch.infini.is_available(), reason="no infini device")
def test_empty_and_cpu_roundtrip():
    src = torch.arange(8, dtype=torch.float32)
    dst = torch.empty((8,), device="infini:0")

    dst.copy_(src)
    out = torch.empty_like(src)
    out.copy_(dst)

    assert torch.equal(out, src)


if __name__ == "__main__":
    print("torch", torch.__version__)
    print("available", torch.infini.is_available())
    print("count", torch.infini.device_count())
    test_privateuse1_backend_is_named_infini()
    test_empty_and_cpu_roundtrip()
