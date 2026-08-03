from functools import partial

import pytest
import torch
import torch.nn.functional as functional

import torch_infini

from .operator_oracle import (
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    copy_cpu_tensor,
    copy_strided_cpu_tensor,
)


NATIVE_SILU_BACKENDS = {"cpu", "iluvatar", "metax", "moore", "nvidia"}


@pytest.fixture
def native_silu_backend() -> str:
    backend = torch_infini._C._runtime_backend_name()
    if backend not in NATIVE_SILU_BACKENDS:
        pytest.skip(f"InfiniOps SiLU is unavailable on the {backend} backend")
    return backend


def _typed_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    source = torch.tensor((-8.0, -3.0, -0.0, 0.5, 2.0, 8.0), dtype=dtype)
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


def _transposed_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.linspace(-4.0, 5.0, steps=12, dtype=torch.float32).reshape(3, 4)
    return (copy_cpu_tensor(source, device).t(),)


def _gapped_inputs(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.empty_strided((2, 3), (5, 1), dtype=torch.float32)
    source.copy_(torch.tensor(((-3.0, -1.0, 0.0), (0.5, 2.0, 6.0))))
    return (copy_strided_cpu_tensor(source, device, copy_storage_from_cpu),)


SILU_DTYPES = (
    pytest.param(torch.float64, 1e-7, 1e-8, id="float64"),
    pytest.param(torch.float32, 1e-5, 1e-6, id="float32"),
    pytest.param(torch.float16, 1e-3, 1e-3, id="float16"),
    pytest.param(torch.bfloat16, 1e-2, 1e-2, id="bfloat16"),
)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), SILU_DTYPES)
def test_silu_dtype_matches_cpu_oracle(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    native_silu_backend: str,
) -> None:
    case = OperatorCase(
        f"dtype-{dtype}",
        functional.silu,
        partial(_typed_inputs, dtype=dtype),
    )

    assert native_silu_backend in NATIVE_SILU_BACKENDS
    assert_operator_matches_cpu(case, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase("scalar", functional.silu, _scalar_inputs),
        OperatorCase("empty", functional.silu, _empty_inputs),
    ],
    ids=lambda case: case.name,
)
def test_silu_shape_matches_cpu_oracle(
    case: OperatorCase,
    native_silu_backend: str,
) -> None:
    assert native_silu_backend in NATIVE_SILU_BACKENDS
    assert_operator_matches_cpu(case, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        OperatorCase("transposed", functional.silu, _transposed_inputs),
        OperatorCase("gapped", functional.silu, _gapped_inputs),
    ],
    ids=lambda case: case.name,
)
def test_silu_layout_matches_cpu_oracle(
    case: OperatorCase,
    native_silu_backend: str,
    infini_ops_test_module,
) -> None:
    assert native_silu_backend in NATIVE_SILU_BACKENDS
    assert_operator_matches_cpu(
        case,
        copy_storage_from_cpu=infini_ops_test_module.copy_storage_from_cpu,
        copy_storage_to_cpu=infini_ops_test_module.copy_storage_to_cpu,
        rtol=1e-5,
        atol=1e-6,
    )


def test_silu_module_inference_matches_cpu(native_silu_backend: str) -> None:
    module = torch.nn.SiLU().eval()
    input_cpu = torch.linspace(-4.0, 5.0, steps=24, dtype=torch.float32).reshape(
        2, 3, 4
    )
    expected = module(input_cpu)

    input_infini = input_cpu.to("infini")
    with torch.no_grad():
        result = module(input_infini)

    output_cpu = torch.empty(result.shape, dtype=result.dtype)
    output_cpu.copy_(result)
    assert native_silu_backend in NATIVE_SILU_BACKENDS
    assert result.device.type == "infini"
    assert not result.requires_grad
    torch.testing.assert_close(output_cpu, expected, rtol=1e-5, atol=1e-6)


def test_silu_unsupported_backend_reports_native_gap() -> None:
    backend = torch_infini._C._runtime_backend_name()
    if backend in NATIVE_SILU_BACKENDS:
        pytest.skip(f"InfiniOps SiLU is available on the {backend} backend")
    (input_tensor,) = _typed_inputs("infini", dtype=torch.float32)

    with pytest.raises(
        RuntimeError,
        match=rf"Silu implementation 0 is unavailable.*{backend}",
    ):
        functional.silu(input_tensor)


def test_silu_integer_dtype_matches_cpu_error(native_silu_backend: str) -> None:
    case = OperatorCase(
        "int32",
        functional.silu,
        partial(_typed_inputs, dtype=torch.int32),
    )

    assert native_silu_backend in NATIVE_SILU_BACKENDS
    assert_operator_matches_cpu(case, error_match=r"not support|not implemented")


def test_silu_inplace_is_not_implemented() -> None:
    (input_tensor,) = _typed_inputs("infini", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match=r"aten::silu(?:_|\.out)"):
        functional.silu(input_tensor, inplace=True)


def test_silu_records_storage_on_current_stream(
    native_silu_backend: str,
    infini_ops_test_module,
) -> None:
    (input_tensor,) = _typed_inputs("infini", dtype=torch.float32)
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = functional.silu(input_tensor)
        recorded = [
            infini_ops_test_module.allocation_records_current_stream(tensor)
            for tensor in (input_tensor, result)
        ]

    assert native_silu_backend in NATIVE_SILU_BACKENDS
    assert recorded == [True, True]
    stream.synchronize()
