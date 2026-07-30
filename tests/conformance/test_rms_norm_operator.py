import pytest
import torch
import torch.nn.functional as functional

from .operator_oracle import (
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    copy_cpu_tensor,
)


RMS_NORM_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


def _rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float | None,
) -> torch.Tensor:
    return functional.rms_norm(
        input_tensor,
        (input_tensor.shape[-1],),
        weight=weight,
        eps=eps,
    )


def _weight() -> torch.Tensor:
    return torch.tensor((0.5, 1.0, 1.5, 2.0), dtype=torch.float32)


def _rank2_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.linspace(-3.0, 4.0, steps=12, dtype=torch.float32).reshape(
        3, 4
    )
    return (
        copy_cpu_tensor(input_tensor, device),
        copy_cpu_tensor(_weight(), device),
    )


def _rank3_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.linspace(-3.0, 4.0, steps=24, dtype=torch.float32).reshape(
        2, 3, 4
    )
    return (
        copy_cpu_tensor(input_tensor, device),
        copy_cpu_tensor(_weight(), device),
    )


def _empty_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    return (
        copy_cpu_tensor(torch.empty((0, 3, 4), dtype=torch.float32), device),
        copy_cpu_tensor(_weight(), device),
    )


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase(
            "rank2-default-eps",
            _rms_norm,
            _rank2_inputs,
            {"eps": None},
        ),
        OperatorCase(
            "rank3-default-eps",
            _rms_norm,
            _rank3_inputs,
            {"eps": None},
        ),
        OperatorCase(
            "rank3-explicit-eps",
            _rms_norm,
            _rank3_inputs,
            {"eps": 1e-5},
        ),
        OperatorCase(
            "empty",
            _rms_norm,
            _empty_inputs,
            {"eps": 1e-5},
        ),
    ],
    ids=lambda case: case.name,
)
def test_rms_norm_matches_cpu_oracle(case: OperatorCase) -> None:
    assert_operator_matches_cpu(case, **RMS_NORM_TOLERANCE)


def test_rms_norm_module_inference_matches_cpu() -> None:
    module = torch.nn.RMSNorm(4, eps=None).eval()
    input_cpu = torch.linspace(-3.0, 4.0, steps=24, dtype=torch.float32).reshape(
        2, 3, 4
    )
    with torch.no_grad():
        module.weight.copy_(_weight())
        expected = module(input_cpu)

    module = module.to("infini")
    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = module(input_infini)

    output_cpu = torch.empty(result.shape, dtype=result.dtype)
    output_cpu.copy_(result)
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(output_cpu, expected, **RMS_NORM_TOLERANCE)


def test_rms_norm_without_weight_is_not_implemented() -> None:
    input_tensor, _ = _rank2_inputs("infini")

    with pytest.raises(NotImplementedError, match=r"requires an affine weight"):
        functional.rms_norm(input_tensor, (4,), weight=None)


def test_rms_norm_multiple_normalized_dimensions_are_not_implemented() -> None:
    input_tensor, _ = _rank3_inputs("infini")
    weight = copy_cpu_tensor(torch.ones((3, 4), dtype=torch.float32), "infini")

    with pytest.raises(NotImplementedError, match=r"one normalized dimension"):
        functional.rms_norm(input_tensor, (3, 4), weight=weight)


def test_rms_norm_noncontiguous_input_is_not_implemented() -> None:
    source = torch.linspace(-3.0, 4.0, steps=24, dtype=torch.float32).reshape(2, 4, 3)
    input_tensor = copy_cpu_tensor(source, "infini").transpose(1, 2)
    weight = copy_cpu_tensor(_weight(), "infini")

    with pytest.raises(NotImplementedError, match=r"contiguous inputs"):
        functional.rms_norm(input_tensor, (4,), weight=weight)


def test_rms_norm_non_float_input_is_not_implemented() -> None:
    input_tensor = copy_cpu_tensor(torch.ones((2, 4), dtype=torch.float16), "infini")
    weight = copy_cpu_tensor(torch.ones(4, dtype=torch.float16), "infini")

    with pytest.raises(NotImplementedError, match=r"torch.float32"):
        functional.rms_norm(input_tensor, (4,), weight=weight)


def test_rms_norm_records_storage_on_current_stream(infini_ops_test_module) -> None:
    input_tensor, weight = _rank3_inputs("infini")
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = functional.rms_norm(input_tensor, (4,), weight=weight)
        recorded = [
            infini_ops_test_module.allocation_records_current_stream(tensor)
            for tensor in (input_tensor, weight, result)
        ]

    assert recorded == [True, True, True]
    stream.synchronize()
