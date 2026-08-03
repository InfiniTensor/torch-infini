from functools import partial

import pytest
import torch

import torch_infini

from .operator_oracle import (
    ExpectedOperatorGapError,
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    assert_tensor_matches_cpu,
    copy_cpu_tensor,
    copy_strided_cpu_tensor,
    invoke,
)


NATIVE_MUL_BACKENDS = {
    "ascend",
    "cpu",
    "iluvatar",
    "metax",
    "moore",
    "nvidia",
}


@pytest.fixture
def native_mul_backend() -> str:
    backend = torch_infini._C._runtime_backend_name()
    if backend not in NATIVE_MUL_BACKENDS:
        pytest.skip(f"InfiniOps Mul is unavailable on the {backend} backend")
    return backend


def _typed_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.tensor((-4, -2, -1, 0, 1, 3), dtype=dtype)
    rhs = torch.tensor((3, -2, 4, 5, -6, 2), dtype=dtype)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _scalar_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.tensor(-2.5, dtype=torch.float32)
    rhs = torch.tensor(4.0, dtype=torch.float32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _empty_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.empty((0, 3), dtype=torch.float32)
    rhs = torch.empty((0, 3), dtype=torch.float32)
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
    return lhs, rhs


def _gapped_inputs(
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
    return lhs, rhs


def _expanded_inputs(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.arange(3, dtype=torch.float32).reshape(1, 3).expand(4, 3)
    rhs = torch.linspace(-1.0, 1.0, steps=3).reshape(1, 3).expand(4, 3)
    return (
        copy_strided_cpu_tensor(lhs, device, copy_storage_from_cpu),
        copy_strided_cpu_tensor(rhs, device, copy_storage_from_cpu),
    )


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
    lhs = torch.linspace(-2.0, 2.0, steps=6, dtype=torch.float32)
    rhs = torch.arange(6, dtype=torch.int32)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


def _bool_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    lhs = torch.tensor((True, False, True), dtype=torch.bool)
    rhs = torch.tensor((True, True, False), dtype=torch.bool)
    return copy_cpu_tensor(lhs, device), copy_cpu_tensor(rhs, device)


MUL_DTYPES = (
    pytest.param(torch.float64, 1e-12, 1e-12, id="float64"),
    pytest.param(torch.float32, 1e-6, 1e-6, id="float32"),
    pytest.param(torch.float16, 1e-3, 1e-3, id="float16"),
    pytest.param(torch.bfloat16, 1e-2, 1e-2, id="bfloat16"),
    pytest.param(torch.int32, 0, 0, id="int32"),
)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), MUL_DTYPES)
def test_mul_dtype_matches_cpu_oracle(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    native_mul_backend: str,
) -> None:
    case = OperatorCase(
        f"dtype-{dtype}",
        torch.mul,
        partial(_typed_inputs, dtype=dtype),
    )

    assert native_mul_backend in NATIVE_MUL_BACKENDS
    assert_operator_matches_cpu(case, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase("scalar", torch.mul, _scalar_inputs),
        OperatorCase("empty", torch.mul, _empty_inputs),
        OperatorCase("broadcast", torch.mul, _broadcast_inputs),
    ],
    ids=lambda case: case.name,
)
def test_mul_shape_matches_cpu_oracle(
    case: OperatorCase,
    native_mul_backend: str,
) -> None:
    assert native_mul_backend in NATIVE_MUL_BACKENDS
    assert_operator_matches_cpu(case, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase("transposed", torch.mul, _transposed_inputs),
        OperatorCase("gapped", torch.mul, _gapped_inputs),
        OperatorCase("expanded", torch.mul, _expanded_inputs),
    ],
    ids=lambda case: case.name,
)
def test_mul_layout_matches_cpu_oracle(
    case: OperatorCase,
    native_mul_backend: str,
    infini_ops_test_module,
) -> None:
    assert native_mul_backend in NATIVE_MUL_BACKENDS
    assert_operator_matches_cpu(
        case,
        copy_storage_from_cpu=infini_ops_test_module.copy_storage_from_cpu,
        copy_storage_to_cpu=infini_ops_test_module.copy_storage_to_cpu,
        rtol=1e-6,
        atol=1e-6,
    )


def test_mul_broadcast_error_matches_cpu_oracle(native_mul_backend: str) -> None:
    case = OperatorCase(
        "incompatible-broadcast",
        torch.mul,
        _incompatible_broadcast_inputs,
    )

    assert native_mul_backend in NATIVE_MUL_BACKENDS
    assert_operator_matches_cpu(case, error_match=r"must match.*size")


@pytest.mark.parametrize(
    ("case", "error_substring"),
    [
        pytest.param(
            OperatorCase("mixed-dtype-promotion", torch.mul, _mixed_dtype_inputs),
            "does not support type promotion",
            id="mixed-dtype-promotion",
            marks=pytest.mark.xfail(
                reason="aten::mul.Tensor does not implement type promotion yet",
                raises=ExpectedOperatorGapError,
                strict=True,
            ),
        ),
        pytest.param(
            OperatorCase("bool", torch.mul, _bool_inputs),
            "InfiniOps does not support ATen dtype Bool",
            id="bool",
            marks=pytest.mark.xfail(
                reason="InfiniOps does not expose a boolean Mul dtype",
                raises=ExpectedOperatorGapError,
                strict=True,
            ),
        ),
    ],
)
def test_mul_expected_cpu_oracle_gap(
    case: OperatorCase,
    error_substring: str,
    native_mul_backend: str,
) -> None:
    expected = invoke(case, "cpu")
    actual = invoke(case, "infini")

    assert native_mul_backend in NATIVE_MUL_BACKENDS
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


def test_mul_unsupported_backend_reports_native_gap() -> None:
    backend = torch_infini._C._runtime_backend_name()
    if backend in NATIVE_MUL_BACKENDS:
        pytest.skip(f"InfiniOps Mul is available on the {backend} backend")
    lhs, rhs = _typed_inputs("infini", dtype=torch.float32)

    with pytest.raises(
        RuntimeError,
        match=rf"Mul implementation 0 is unavailable.*{backend}",
    ):
        torch.mul(lhs, rhs)


def test_mul_scalar_argument_is_not_implemented() -> None:
    lhs, _ = _typed_inputs("infini", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="expects two infini tensors"):
        torch.mul(lhs, 2.0)


def test_mul_out_overload_is_not_implemented() -> None:
    lhs, rhs = _typed_inputs("infini", dtype=torch.float32)
    out = torch.empty_like(lhs)

    with pytest.raises(NotImplementedError, match=r"aten::mul\.out"):
        torch.mul(lhs, rhs, out=out)


def test_mul_inplace_overload_is_not_implemented() -> None:
    lhs, rhs = _typed_inputs("infini", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match=r"aten::mul\.out"):
        lhs.mul_(rhs)


def test_mul_rejects_cpu_other() -> None:
    lhs, _ = _typed_inputs("infini", dtype=torch.float32)
    _, rhs = _typed_inputs("cpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="expects two infini tensors"):
        torch.mul(lhs, rhs)


def test_mul_records_storage_on_current_stream(
    native_mul_backend: str,
    infini_ops_test_module,
) -> None:
    lhs, rhs = _typed_inputs("infini", dtype=torch.float32)
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = torch.mul(lhs, rhs)
        recorded = [
            infini_ops_test_module.allocation_records_current_stream(tensor)
            for tensor in (lhs, rhs, result)
        ]

    assert native_mul_backend in NATIVE_MUL_BACKENDS
    assert recorded == [True, True, True]
    stream.synchronize()
