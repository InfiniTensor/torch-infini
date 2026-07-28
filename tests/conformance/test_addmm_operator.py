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


def _bias_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.linspace(-1.0, 2.0, steps=5, dtype=torch.float32)
    mat1 = torch.linspace(-3.0, 4.0, steps=12, dtype=torch.float32).reshape(3, 4)
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 7.0
    mat2 = copy_cpu_tensor(weight, device).t()
    return (
        copy_cpu_tensor(input_tensor, device),
        copy_cpu_tensor(mat1, device),
        mat2,
    )


def _matrix_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.linspace(-2.0, 1.0, steps=15, dtype=torch.float32).reshape(
        3, 5
    )
    mat1 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mat2 = torch.linspace(-2.0, 3.0, steps=20, dtype=torch.float32).reshape(4, 5)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


def _scalar_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.tensor(1.5, dtype=torch.float32)
    mat1 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    mat2 = torch.linspace(-1.0, 2.0, steps=12, dtype=torch.float32).reshape(3, 4)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


def _empty_m_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.ones(5, dtype=torch.float32)
    mat1 = torch.empty((0, 4), dtype=torch.float32)
    mat2 = torch.ones((4, 5), dtype=torch.float32)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


def _empty_k_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.linspace(-1.0, 2.0, steps=5, dtype=torch.float32)
    mat1 = torch.empty((3, 0), dtype=torch.float32)
    mat2 = torch.empty((0, 5), dtype=torch.float32)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


ADDMM_SUCCESS_CASES = (
    OperatorCase("broadcast-bias", torch.addmm, _bias_inputs),
    OperatorCase(
        "explicit-unit-scalars",
        torch.addmm,
        _bias_inputs,
        {"beta": 1.0, "alpha": 1.0},
    ),
    OperatorCase("matrix-input", torch.addmm, _matrix_input),
    OperatorCase("scalar-input", torch.addmm, _scalar_input),
    OperatorCase("empty-m", torch.addmm, _empty_m_inputs),
    OperatorCase("empty-k", torch.addmm, _empty_k_inputs),
)

# Match PyTorch's TF32 test precision for CUDA matrix multiplication.
TF32_TOLERANCE = {"rtol": 5e-3, "atol": 5e-3}


@pytest.mark.parametrize("case", ADDMM_SUCCESS_CASES, ids=lambda case: case.name)
def test_addmm_matches_cpu_oracle(case: OperatorCase) -> None:
    assert_operator_matches_cpu(case, **TF32_TOLERANCE)


def _one_dimensional_matrix_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.ones(3, dtype=torch.float32)
    mat1 = torch.arange(4, dtype=torch.float32)
    mat2 = torch.ones((4, 3), dtype=torch.float32)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


def _mismatched_k_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.ones(5, dtype=torch.float32)
    mat1 = torch.ones((2, 4), dtype=torch.float32)
    mat2 = torch.ones((3, 5), dtype=torch.float32)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


def _nonbroadcast_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.ones((2, 2), dtype=torch.float32)
    mat1 = torch.ones((3, 4), dtype=torch.float32)
    mat2 = torch.ones((4, 5), dtype=torch.float32)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


def _mixed_dtype_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.ones(5, dtype=torch.float32)
    mat1 = torch.ones((3, 4), dtype=torch.float32)
    mat2 = torch.ones((4, 5), dtype=torch.float64)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        pytest.param(
            OperatorCase(
                "one-dimensional-matrix",
                torch.addmm,
                _one_dimensional_matrix_inputs,
            ),
            r"(2D|matrix)",
            id="one-dimensional-matrix",
        ),
        pytest.param(
            OperatorCase("mismatched-k", torch.addmm, _mismatched_k_inputs),
            r"(inner dimensions|cannot be multiplied)",
            id="mismatched-k",
        ),
        pytest.param(
            OperatorCase("nonbroadcast-input", torch.addmm, _nonbroadcast_input),
            r"(broadcast|size)",
            id="nonbroadcast-input",
        ),
        pytest.param(
            OperatorCase("mixed-dtype", torch.addmm, _mixed_dtype_inputs),
            r"dtypes?",
            id="mixed-dtype",
        ),
    ],
)
def test_addmm_error_matches_cpu_oracle(
    case: OperatorCase,
    error_match: str,
) -> None:
    assert_operator_matches_cpu(case, error_match=error_match)


def _float64_inputs(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    input_tensor = torch.ones(5, dtype=torch.float64)
    mat1 = torch.ones((3, 4), dtype=torch.float64)
    mat2 = torch.ones((4, 5), dtype=torch.float64)
    return tuple(
        copy_cpu_tensor(tensor, device) for tensor in (input_tensor, mat1, mat2)
    )


ADDMM_EXPECTED_GAPS = (
    pytest.param(
        OperatorCase("alpha-two", torch.addmm, _bias_inputs, {"alpha": 2.0}),
        "only supports alpha == 1",
        id="alpha-two",
    ),
    pytest.param(
        OperatorCase("beta-two", torch.addmm, _bias_inputs, {"beta": 2.0}),
        "only supports beta == 1",
        id="beta-two",
    ),
    pytest.param(
        OperatorCase("float64", torch.addmm, _float64_inputs),
        "only supports torch.float32",
        id="float64",
    ),
)


@pytest.mark.parametrize(("case", "error_message"), ADDMM_EXPECTED_GAPS)
@pytest.mark.xfail(
    reason="aten::addmm has intentionally narrow scalar and dtype coverage",
    raises=ExpectedOperatorGapError,
    strict=True,
)
def test_addmm_expected_cpu_oracle_gap(
    case: OperatorCase,
    error_message: str,
) -> None:
    expected = invoke(case, "cpu")
    actual = invoke(case, "infini")

    assert expected.error is None
    assert expected.tensor is not None
    if actual.error is None:
        assert actual.tensor is not None
        tolerance = TF32_TOLERANCE if expected.tensor.dtype == torch.float32 else {}
        assert_tensor_matches_cpu(
            case.name,
            expected.tensor,
            actual.tensor,
            **tolerance,
        )
        return
    assert type(actual.error) is RuntimeError
    assert error_message in str(actual.error)
    raise ExpectedOperatorGapError(
        f"{case.name}: CPU returned a tensor while infini raised {actual.error}"
    )


def test_addmm_rejects_cpu_mat2() -> None:
    input_tensor, mat1, _ = _bias_inputs("infini")
    mat2 = torch.ones((4, 5), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="expects three infini tensors"):
        torch.addmm(input_tensor, mat1, mat2)


def test_addmm_rejects_cross_device_inputs() -> None:
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")
    input_tensor = copy_cpu_tensor(torch.ones(5, dtype=torch.float32), "infini:0")
    mat1 = copy_cpu_tensor(torch.ones((3, 4), dtype=torch.float32), "infini:0")
    mat2 = copy_cpu_tensor(torch.ones((4, 5), dtype=torch.float32), "infini:1")

    with pytest.raises(RuntimeError, match="same infini device"):
        torch.addmm(input_tensor, mat1, mat2)


def test_addmm_records_storage_on_current_stream(infini_ops_test_module) -> None:
    input_tensor, mat1, mat2 = _bias_inputs("infini")
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = torch.addmm(input_tensor, mat1, mat2)
        recorded = [
            infini_ops_test_module.allocation_records_current_stream(tensor)
            for tensor in (input_tensor, mat1, mat2, result)
        ]

    assert recorded == [True, True, True, True]
    stream.synchronize()
