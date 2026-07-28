import gc

import pytest
import torch

from .operator_oracle import (
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    copy_cpu_tensor,
)


def _matrix_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return (copy_cpu_tensor(source, device),)


def test_matrix_transpose_matches_cpu_oracle(infini_ops_test_module) -> None:
    case = OperatorCase("matrix-transpose", torch.t, _matrix_input)

    assert_operator_matches_cpu(
        case,
        copy_storage_to_cpu=infini_ops_test_module.copy_storage_to_cpu,
    )


def test_as_strided_shares_storage_and_honors_explicit_offset() -> None:
    base = torch.empty(8, dtype=torch.float32, device="infini")

    view = torch.as_strided(base, (2, 2), (3, 1), storage_offset=1)

    assert view.shape == (2, 2)
    assert view.stride() == (3, 1)
    assert view.storage_offset() == 1
    assert view.untyped_storage()._cdata == base.untyped_storage()._cdata
    assert view.data_ptr() == base.data_ptr() + base.element_size()


def test_as_strided_defaults_to_input_storage_offset() -> None:
    base = torch.empty(8, dtype=torch.float32, device="infini")
    offset_view = torch.as_strided(base, (4,), (1,), storage_offset=2)

    nested_view = torch.as_strided(offset_view, (2,), (1,))

    assert nested_view.storage_offset() == 2
    assert nested_view.untyped_storage()._cdata == base.untyped_storage()._cdata


@pytest.mark.parametrize(
    ("size", "stride", "storage_offset", "error_match"),
    [
        ((2, 3), (1,), None, "mismatch in length of strides and shape"),
        ((2,), (-1,), None, "Negative strides are not supported"),
        ((2,), (1,), -1, "invalid storage offset"),
        ((4,), (2,), None, "out of bounds for storage"),
    ],
)
def test_as_strided_errors_match_cpu(
    size: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int | None,
    error_match: str,
) -> None:
    for device in ("cpu", "infini"):
        tensor = torch.empty(6, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match=error_match):
            torch.as_strided(tensor, size, stride, storage_offset)


def test_transposed_view_keeps_storage_alive(infini_ops_test_module) -> None:
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    base = copy_cpu_tensor(expected.t(), "infini")
    view = base.t()

    del base
    gc.collect()

    torch.infini.synchronize(view.device)
    actual = torch.empty_strided(view.shape, view.stride(), dtype=view.dtype)
    infini_ops_test_module.copy_storage_to_cpu(actual, view)
    torch.testing.assert_close(actual, expected)
