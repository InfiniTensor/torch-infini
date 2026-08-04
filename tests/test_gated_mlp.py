import torch

import torch_infini  # noqa: F401


# Match PyTorch's TF32 test precision for CUDA matrix multiplication.
TF32_TOLERANCE = {"rtol": 5e-3, "atol": 5e-3}


class _GatedMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 6, bias=False)
        self.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 4, bias=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(input_tensor))
        return self.down_proj(gate * self.up_proj(input_tensor))


def _copy_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.empty(tensor.shape, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def _make_gated_mlp() -> _GatedMLP:
    model = _GatedMLP().eval()
    with torch.no_grad():
        model.gate_proj.weight.copy_(
            torch.linspace(-1.0, 1.0, steps=24, dtype=torch.float32).reshape(6, 4)
        )
        model.up_proj.weight.copy_(
            torch.linspace(1.5, -0.5, steps=24, dtype=torch.float32).reshape(6, 4)
        )
        model.down_proj.weight.copy_(
            torch.linspace(-0.75, 1.25, steps=24, dtype=torch.float32).reshape(4, 6)
        )
    return model


def test_gated_mlp_inference_matches_cpu() -> None:
    model = _make_gated_mlp()
    input_cpu = torch.linspace(-2.0, 3.0, steps=24, dtype=torch.float32).reshape(
        2, 3, 4
    )
    with torch.no_grad():
        expected = model(input_cpu)

    model = model.to("infini")
    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = model(input_infini)

    assert not model.training
    assert all(
        module.bias is None
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    )
    assert all(parameter.device.type == "infini" for parameter in model.parameters())
    assert result.shape == input_cpu.shape
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(
        _copy_to_cpu(result),
        expected,
        **TF32_TOLERANCE,
    )
