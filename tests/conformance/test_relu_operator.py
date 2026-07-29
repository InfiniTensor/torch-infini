from functools import partial

import pytest
import torch

from .operator_oracle import (
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    copy_cpu_tensor,
    copy_strided_cpu_tensor,
)


def _typed_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    values = (0, 1, 2, 255) if dtype == torch.uint8 else (-3, -1, 0, 2)
    source = torch.tensor(values, dtype=dtype)
    return (copy_cpu_tensor(source, device),)


def _scalar_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    return (copy_cpu_tensor(torch.tensor(-2.5, dtype=torch.float32), device),)


def _empty_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    return (copy_cpu_tensor(torch.empty((0, 3), dtype=torch.float32), device),)


def _expanded_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.tensor(((-2.0, -0.5, 0.0, 3.0),), dtype=torch.float32)
    return (copy_cpu_tensor(source, device).expand(3, 4),)


def _transposed_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.linspace(-3.0, 4.0, steps=12, dtype=torch.float32).reshape(3, 4)
    return (copy_cpu_tensor(source, device).t(),)


def _gapped_inputs(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.empty_strided((2, 3), (5, 1), dtype=torch.float32)
    source.copy_(torch.tensor(((-2.0, -1.0, 0.0), (0.5, 2.0, 4.0))))
    return (copy_strided_cpu_tensor(source, device, copy_storage_from_cpu),)


def _channels_last_inputs(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    shape = (2, 3, 4, 5)
    source = torch.empty(shape, dtype=torch.float32, memory_format=torch.channels_last)
    source.copy_(torch.linspace(-3.0, 4.0, steps=120).reshape(shape))
    return (copy_strided_cpu_tensor(source, device, copy_storage_from_cpu),)


RELU_DTYPES = (
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
)


@pytest.mark.parametrize("dtype", RELU_DTYPES, ids=str)
def test_relu_dtype_matches_cpu_oracle(dtype: torch.dtype) -> None:
    case = OperatorCase(
        f"dtype-{dtype}",
        torch.relu,
        partial(_typed_inputs, dtype=dtype),
    )

    assert_operator_matches_cpu(case)


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase("scalar", torch.relu, _scalar_inputs),
        OperatorCase("empty", torch.relu, _empty_inputs),
        OperatorCase("expanded", torch.relu, _expanded_inputs),
    ],
    ids=lambda case: case.name,
)
def test_relu_shape_matches_cpu_oracle(case: OperatorCase) -> None:
    assert_operator_matches_cpu(case)


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase("transposed", torch.relu, _transposed_inputs),
        OperatorCase("gapped", torch.relu, _gapped_inputs),
        OperatorCase("channels-last", torch.relu, _channels_last_inputs),
    ],
    ids=lambda case: case.name,
)
def test_relu_layout_matches_cpu_oracle(
    case: OperatorCase,
    infini_ops_test_module,
) -> None:
    assert_operator_matches_cpu(
        case,
        copy_storage_from_cpu=infini_ops_test_module.copy_storage_from_cpu,
        copy_storage_to_cpu=infini_ops_test_module.copy_storage_to_cpu,
    )


def _bool_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    return (copy_cpu_tensor(torch.tensor((False, True)), device),)


def test_relu_bool_error_matches_cpu_oracle() -> None:
    case = OperatorCase("bool", torch.relu, _bool_inputs)

    assert_operator_matches_cpu(case, error_match=r"(not supported|does not support)")


def test_relu_inplace_is_not_implemented() -> None:
    (input_tensor,) = _typed_inputs("infini", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match=r"aten::relu_"):
        input_tensor.relu_()


def test_relu_records_storage_on_current_stream(infini_ops_test_module) -> None:
    (input_tensor,) = _typed_inputs("infini", dtype=torch.float32)
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = torch.relu(input_tensor)
        recorded = [
            infini_ops_test_module.allocation_records_current_stream(tensor)
            for tensor in (input_tensor, result)
        ]

    assert recorded == [True, True]
    stream.synchronize()
