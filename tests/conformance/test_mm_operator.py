import pytest
import torch

from .operator_oracle import (
    ExpectedOperatorGapError,
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    assert_tensor_matches_cpu,
    copy_cpu_tensor,
    invoke,
)


def _contiguous_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    rhs = torch.linspace(-2.0, 3.0, steps=20, dtype=torch.float32).reshape(4, 5)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _linear_weight_layout_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.linspace(-3.0, 4.0, steps=12, dtype=torch.float32).reshape(3, 4)
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 7.0
    mat2 = copy_cpu_tensor(weight, device).t()
    assert mat2.shape == (4, 5)
    assert mat2.stride() == (1, 4)
    assert not mat2.is_contiguous()
    return copy_cpu_tensor(lhs, device), mat2


def _empty_m_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.empty((0, 4), dtype=torch.float32)
    rhs = torch.ones((4, 3), dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _empty_n_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.ones((3, 4), dtype=torch.float32)
    rhs = torch.empty((4, 0), dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _empty_k_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.empty((3, 0), dtype=torch.float32)
    rhs = torch.empty((0, 5), dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


MM_SUCCESS_CASES = (
    OperatorCase("contiguous-float32", torch.mm, _contiguous_inputs),
    OperatorCase(
        "linear-transposed-weight-float32",
        torch.mm,
        _linear_weight_layout_inputs,
    ),
    OperatorCase("empty-m", torch.mm, _empty_m_inputs),
    OperatorCase("empty-n", torch.mm, _empty_n_inputs),
    OperatorCase("empty-k", torch.mm, _empty_k_inputs),
)

# Match PyTorch's TF32 test precision for CUDA matrix multiplication.
TF32_TOLERANCE = {"rtol": 5e-3, "atol": 5e-3}


@pytest.mark.parametrize("case", MM_SUCCESS_CASES, ids=lambda case: case.name)
def test_mm_matches_cpu_oracle(case: OperatorCase) -> None:
    assert_operator_matches_cpu(case, **TF32_TOLERANCE)


def _one_dimensional_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(4, dtype=torch.float32)
    rhs = torch.ones((4, 3), dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _mismatched_k_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.ones((2, 4), dtype=torch.float32)
    rhs = torch.ones((3, 5), dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _mixed_dtype_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.ones((2, 4), dtype=torch.float32)
    rhs = torch.ones((4, 3), dtype=torch.float64)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        pytest.param(
            OperatorCase("one-dimensional-input", torch.mm, _one_dimensional_inputs),
            r"(2D|matrix)",
            id="one-dimensional-input",
        ),
        pytest.param(
            OperatorCase("mismatched-k", torch.mm, _mismatched_k_inputs),
            r"(inner dimensions|cannot be multiplied)",
            id="mismatched-k",
        ),
        pytest.param(
            OperatorCase("mixed-dtype", torch.mm, _mixed_dtype_inputs),
            r"dtypes?",
            id="mixed-dtype",
        ),
    ],
)
def test_mm_error_matches_cpu_oracle(
    case: OperatorCase,
    error_match: str,
) -> None:
    assert_operator_matches_cpu(case, error_match=error_match)


def _float64_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.ones((2, 4), dtype=torch.float64)
    rhs = torch.ones((4, 3), dtype=torch.float64)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


@pytest.mark.xfail(
    reason="aten::mm currently supports only torch.float32",
    raises=ExpectedOperatorGapError,
    strict=True,
)
def test_mm_float64_expected_cpu_oracle_gap() -> None:
    case = OperatorCase("float64", torch.mm, _float64_inputs)
    expected = invoke(case, "cpu")
    actual = invoke(case, "infini")

    assert expected.error is None
    assert expected.tensor is not None
    if actual.error is None:
        assert actual.tensor is not None
        assert_tensor_matches_cpu(case.name, expected.tensor, actual.tensor)
        return
    assert type(actual.error) is RuntimeError
    assert "only supports torch.float32" in str(actual.error)
    raise ExpectedOperatorGapError(
        f"{case.name}: CPU returned a tensor while infini raised {actual.error}"
    )


def test_mm_rejects_cpu_rhs() -> None:
    lhs = copy_cpu_tensor(torch.ones((2, 4), dtype=torch.float32), "infini")
    rhs = torch.ones((4, 3), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="expects two infini tensors"):
        torch.mm(lhs, rhs)


def test_mm_rejects_cross_device_inputs() -> None:
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")
    lhs = copy_cpu_tensor(
        torch.ones((2, 4), dtype=torch.float32),
        "infini:0",
    )
    rhs = copy_cpu_tensor(
        torch.ones((4, 3), dtype=torch.float32),
        "infini:1",
    )

    with pytest.raises(RuntimeError, match="same infini device"):
        torch.mm(lhs, rhs)


def test_mm_records_storage_on_current_stream(infini_ops_test_module) -> None:
    lhs, mat2 = _linear_weight_layout_inputs("infini")
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = torch.mm(lhs, mat2)
        recorded = [
            infini_ops_test_module.allocation_records_current_stream(tensor)
            for tensor in (lhs, mat2, result)
        ]

    assert recorded == [True, True, True]
    stream.synchronize()
