import torch

import torch_infini  # noqa: F401


def _copy_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.empty(tensor.shape, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def test_relu_module_inference_matches_cpu() -> None:
    module = torch.nn.ReLU().eval()
    input_cpu = torch.tensor(
        ((-3.0, -0.0, 0.0, 2.0), (1.5, -2.5, 4.0, -1.0)),
        dtype=torch.float32,
    )
    expected = module(input_cpu)

    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = module(input_infini)

    assert not module.training
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(_copy_to_cpu(result), expected)
