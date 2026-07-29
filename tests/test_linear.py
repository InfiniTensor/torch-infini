import pytest
import torch

import torch_infini  # noqa: F401


# Match PyTorch's TF32 test precision for CUDA matrix multiplication.
TF32_TOLERANCE = {"rtol": 5e-3, "atol": 5e-3}


def _copy_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.empty(tensor.shape, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def test_bias_free_linear_inference_matches_cpu() -> None:
    model = torch.nn.Linear(4, 3, bias=False).eval()
    input_cpu = torch.linspace(-2.0, 3.0, steps=20, dtype=torch.float32).reshape(5, 4)
    with torch.no_grad():
        model.weight.copy_(
            torch.linspace(-1.5, 2.0, steps=12, dtype=torch.float32).reshape(3, 4)
        )
        expected = model(input_cpu)

    model = model.to("infini")
    model.eval()
    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = model(input_infini)

    assert not model.training
    assert model.bias is None
    assert model.weight.device.type == "infini"
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(
        _copy_to_cpu(result),
        expected,
        **TF32_TOLERANCE,
    )


def test_linear_inference_with_bias_matches_cpu() -> None:
    model = torch.nn.Linear(4, 3, bias=True).eval()
    input_cpu = torch.linspace(-2.0, 3.0, steps=20, dtype=torch.float32).reshape(5, 4)
    with torch.no_grad():
        model.weight.copy_(
            torch.linspace(-1.5, 2.0, steps=12, dtype=torch.float32).reshape(3, 4)
        )
        model.bias.copy_(torch.tensor([-1.25, 0.5, 2.0], dtype=torch.float32))
        expected = model(input_cpu)

    model = model.to("infini")
    model.eval()
    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = model(input_infini)

    assert not model.training
    assert model.bias is not None
    assert model.weight.device.type == "infini"
    assert model.bias.device.type == "infini"
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(
        _copy_to_cpu(result),
        expected,
        **TF32_TOLERANCE,
    )


@pytest.mark.parametrize(
    "bias",
    [False, True],
    ids=("without-bias", "with-bias"),
)
def test_batched_linear_inference_matches_cpu(bias: bool) -> None:
    model = torch.nn.Linear(4, 3, bias=bias).eval()
    input_cpu = torch.linspace(-2.0, 3.0, steps=24, dtype=torch.float32).reshape(
        2, 3, 4
    )
    with torch.no_grad():
        model.weight.copy_(
            torch.linspace(-1.5, 2.0, steps=12, dtype=torch.float32).reshape(3, 4)
        )
        if model.bias is not None:
            model.bias.copy_(torch.tensor([-1.25, 0.5, 2.0], dtype=torch.float32))
        expected = model(input_cpu)

    model = model.to("infini")
    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = model(input_infini)

    assert not model.training
    assert (model.bias is not None) is bias
    assert model.weight.device.type == "infini"
    if model.bias is not None:
        assert model.bias.device.type == "infini"
    assert result.shape == (2, 3, 3)
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(
        _copy_to_cpu(result),
        expected,
        **TF32_TOLERANCE,
    )
