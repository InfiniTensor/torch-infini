import pytest
import torch

from .operator_oracle import (
    ExpectedOperatorGapError,
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    assert_tensor_matches_cpu,
    assert_tensor_values_match,
    copy_cpu_tensor,
    copy_strided_cpu_tensor,
    invoke,
    tensor_metadata,
)


def _contiguous_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    rhs = torch.linspace(-2.0, 3.5, steps=12, dtype=torch.float32).reshape(3, 4)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _broadcast_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    rhs = torch.linspace(-1.5, 1.5, steps=4, dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _strided_tensor(
    values: torch.Tensor,
    strides: tuple[int, ...],
    device: str,
    copy_storage_from_cpu: StorageCopier | None,
) -> torch.Tensor:
    source = torch.empty_strided(values.shape, strides, dtype=values.dtype)
    source.copy_(values)
    return copy_strided_cpu_tensor(source, device, copy_storage_from_cpu)


def _transposed_inputs(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = _strided_tensor(
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        (1, 4),
        device,
        copy_storage_from_cpu,
    )
    rhs = _strided_tensor(
        torch.linspace(-3.0, 2.5, steps=12, dtype=torch.float32).reshape(4, 3),
        (1, 4),
        device,
        copy_storage_from_cpu,
    )
    assert lhs.stride() == rhs.stride() == (1, 4)
    assert not lhs.is_contiguous()
    assert not rhs.is_contiguous()
    return lhs, rhs


def _gapped_strided_inputs(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = _strided_tensor(
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        (10, 2),
        device,
        copy_storage_from_cpu,
    )
    rhs = _strided_tensor(
        torch.linspace(-5.0, 0.5, steps=12, dtype=torch.float32).reshape(3, 4),
        (10, 2),
        device,
        copy_storage_from_cpu,
    )
    assert lhs.stride() == rhs.stride() == (10, 2)
    assert lhs.storage_offset() == rhs.storage_offset() == 0
    assert not lhs.is_contiguous()
    assert not rhs.is_contiguous()
    return lhs, rhs


def _int32_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(-6, 6, dtype=torch.int32).reshape(3, 4)
    rhs = torch.arange(12, 0, -1, dtype=torch.int32).reshape(3, 4)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _incompatible_broadcast_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs = torch.arange(4, dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _mixed_dtype_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.linspace(-2.0, 2.0, steps=6, dtype=torch.float32).reshape(2, 3)
    rhs = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


ADD_SUCCESS_CASES = (
    OperatorCase("contiguous-float32", torch.add, _contiguous_inputs),
    OperatorCase("broadcasting-float32", torch.add, _broadcast_inputs),
    OperatorCase("explicit-alpha-one", torch.add, _contiguous_inputs, {"alpha": 1}),
    OperatorCase("same-dtype-int32", torch.add, _int32_inputs),
)


@pytest.mark.parametrize("case", ADD_SUCCESS_CASES, ids=lambda case: case.name)
def test_add_tensor_matches_cpu_oracle(case: OperatorCase) -> None:
    assert_operator_matches_cpu(case)


def test_add_tensor_gapped_strides_match_cpu_oracle(infini_ops_test_module) -> None:
    case = OperatorCase(
        "gapped-strided-float32",
        torch.add,
        _gapped_strided_inputs,
    )

    assert_operator_matches_cpu(
        case,
        copy_storage_from_cpu=infini_ops_test_module.copy_storage_from_cpu,
    )


def test_add_tensor_broadcast_error_matches_cpu_oracle() -> None:
    case = OperatorCase(
        "incompatible-broadcast",
        torch.add,
        _incompatible_broadcast_inputs,
    )

    assert_operator_matches_cpu(case, error_match=r"must match.*size")


@pytest.mark.xfail(
    reason="aten::add.Tensor does not preserve dense transposed output strides",
    raises=ExpectedOperatorGapError,
    strict=True,
)
def test_add_tensor_expected_transposed_output_metadata_gap(
    infini_ops_test_module,
) -> None:
    case = OperatorCase("transposed-output-layout", torch.add, _transposed_inputs)
    expected = invoke(case, "cpu")
    actual = invoke(
        case,
        "infini",
        infini_ops_test_module.copy_storage_from_cpu,
    )

    assert expected.error is None
    assert expected.tensor is not None
    assert actual.error is None
    assert actual.tensor is not None
    assert actual.tensor.device.type == "infini"
    assert_tensor_values_match(case.name, expected.tensor, actual.tensor)

    expected_metadata = tensor_metadata(expected.tensor)
    actual_metadata = tensor_metadata(actual.tensor)
    differences = {
        field: (expected_value, actual_metadata[field])
        for field, expected_value in expected_metadata.items()
        if actual_metadata[field] != expected_value
    }
    if not differences:
        assert_tensor_matches_cpu(case.name, expected.tensor, actual.tensor)
        return
    assert differences == {
        "stride": ((1, 4), (3, 1)),
        "is_contiguous": (False, True),
    }
    raise ExpectedOperatorGapError(
        f"{case.name}: CPU/infini output metadata differs: {differences}"
    )


@pytest.mark.parametrize(
    ("case", "error_substring"),
    [
        pytest.param(
            OperatorCase(
                "nonunit-alpha",
                torch.add,
                _contiguous_inputs,
                {"alpha": 2},
            ),
            r"only supports alpha == 1",
            id="nonunit-alpha",
            marks=pytest.mark.xfail(
                reason="aten::add.Tensor currently supports only alpha == 1",
                raises=ExpectedOperatorGapError,
                strict=True,
            ),
        ),
        pytest.param(
            OperatorCase("mixed-dtype-promotion", torch.add, _mixed_dtype_inputs),
            r"does not support type promotion",
            id="mixed-dtype-promotion",
            marks=pytest.mark.xfail(
                reason="aten::add.Tensor does not implement type promotion yet",
                raises=ExpectedOperatorGapError,
                strict=True,
            ),
        ),
    ],
)
def test_add_tensor_expected_cpu_oracle_gap(
    case: OperatorCase,
    error_substring: str,
) -> None:
    expected = invoke(case, "cpu")
    actual = invoke(case, "infini")

    assert expected.error is None
    assert expected.tensor is not None
    if actual.error is None:
        assert actual.tensor is not None
        assert_tensor_matches_cpu(case.name, expected.tensor, actual.tensor)
        return
    assert type(actual.error) is RuntimeError
    assert error_substring in str(actual.error)
    raise ExpectedOperatorGapError(
        f"{case.name}: CPU returned a tensor while infini raised {actual.error}"
    )
