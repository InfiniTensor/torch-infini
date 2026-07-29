import torch

import torch_infini  # noqa: F401


# Match PyTorch's TF32 test precision for CUDA matrix multiplication.
TF32_TOLERANCE = {"rtol": 5e-3, "atol": 5e-3}


def _copy_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.empty(tensor.shape, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def test_tiny_mlp_inference_matches_cpu() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 3),
    ).eval()
    input_cpu = torch.tensor(
        (
            (-2.0, -1.0, 0.0, 1.0),
            (0.5, -0.5, 2.0, -3.0),
            (4.0, 1.0, -2.0, 0.25),
        ),
        dtype=torch.float32,
    )

    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                (
                    (1.0, 0.0, -0.5, 0.25),
                    (-0.5, 1.0, 0.0, -0.25),
                    (0.25, -0.5, 1.0, 0.0),
                    (0.0, 0.25, -0.5, 1.0),
                    (0.5, 0.5, 0.5, 0.5),
                ),
                dtype=torch.float32,
            )
        )
        model[0].bias.copy_(
            torch.tensor((-0.5, 0.25, -0.75, 0.5, 0.0), dtype=torch.float32)
        )
        model[2].weight.copy_(
            torch.tensor(
                (
                    (0.5, -1.0, 0.25, 0.75, -0.5),
                    (-0.25, 0.5, 1.0, -0.5, 0.25),
                    (1.0, 0.25, -0.75, 0.5, -0.25),
                ),
                dtype=torch.float32,
            )
        )
        model[2].bias.copy_(torch.tensor((0.1, -0.2, 0.3), dtype=torch.float32))

        hidden = model[0](input_cpu)
        assert torch.any(hidden < 0)
        assert torch.any(hidden > 0)
        expected = model(input_cpu)

    model = model.to("infini")
    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = model(input_infini)

    assert not model.training
    assert all(parameter.device.type == "infini" for parameter in model.parameters())
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(
        _copy_to_cpu(result),
        expected,
        **TF32_TOLERANCE,
    )
