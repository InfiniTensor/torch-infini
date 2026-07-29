import gc

import pytest
import torch

from .operator_oracle import (
    OperatorCase,
    StorageCopier,
    assert_operator_matches_cpu,
    copy_cpu_tensor,
    copy_strided_cpu_tensor,
)


def _contiguous_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    return (copy_cpu_tensor(source, device),)


def _transposed_input(
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
    return (copy_strided_cpu_tensor(source, device, copy_storage_from_cpu),)


def _scalar_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    return (copy_cpu_tensor(torch.tensor(3.0), device),)


def _zero_sized_input(
    device: str,
    _copy_storage_from_cpu: StorageCopier | None = None,
) -> tuple[torch.Tensor, ...]:
    return (torch.empty((0, 3), dtype=torch.float32, device=device),)


def _view_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(3, -1)


def _view_transposed_chunks(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(3, 2, 2, 2)


def _view_scalar(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(1)


def _view_zero_sized(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(3, 0)


def _reshape_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return torch.reshape(tensor, (2, 12))


def _reshape_transposed_chunks(tensor: torch.Tensor) -> torch.Tensor:
    return torch.reshape(tensor, (3, 2, 2, 2))


def _flatten_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.Flatten(start_dim=1)(tensor)


VIEW_COMPATIBLE_CASES = (
    OperatorCase("contiguous-view", _view_contiguous, _contiguous_input),
    OperatorCase(
        "transposed-chunk-view",
        _view_transposed_chunks,
        _transposed_input,
    ),
    OperatorCase("scalar-view", _view_scalar, _scalar_input),
    OperatorCase("zero-sized-view", _view_zero_sized, _zero_sized_input),
    OperatorCase("contiguous-reshape", _reshape_contiguous, _contiguous_input),
    OperatorCase(
        "transposed-chunk-reshape",
        _reshape_transposed_chunks,
        _transposed_input,
    ),
    OperatorCase("contiguous-flatten", _flatten_contiguous, _contiguous_input),
)


@pytest.mark.parametrize(
    "case",
    VIEW_COMPATIBLE_CASES,
    ids=lambda case: case.name,
)
def test_view_compatible_operations_match_cpu_oracle(
    case: OperatorCase,
    infini_ops_test_module,
) -> None:
    assert_operator_matches_cpu(
        case,
        copy_storage_from_cpu=infini_ops_test_module.copy_storage_from_cpu,
        copy_storage_to_cpu=infini_ops_test_module.copy_storage_to_cpu,
    )


def test_view_shares_storage_and_preserves_offset() -> None:
    base = torch.empty(12, dtype=torch.float32, device="infini")
    offset_base = torch.as_strided(base, (6,), (1,), storage_offset=2)

    view = offset_base.view(2, 3)

    assert view.shape == (2, 3)
    assert view.stride() == (3, 1)
    assert view.storage_offset() == 2
    assert view.untyped_storage()._cdata == base.untyped_storage()._cdata
    assert view.data_ptr() == base.data_ptr() + 2 * base.element_size()


@pytest.mark.parametrize(
    ("input_shape", "view_shape", "error_match"),
    [
        ((2, 3), (5,), r"shape '\[5\]' is invalid for input of size 6"),
        ((2, 3), (-1, -1), "only one dimension can be inferred"),
        ((0,), (-1, 0), "unspecified dimension size -1 can be any value"),
    ],
)
def test_invalid_view_shapes_match_cpu(
    input_shape: tuple[int, ...],
    view_shape: tuple[int, ...],
    error_match: str,
) -> None:
    for device in ("cpu", "infini"):
        tensor = torch.empty(input_shape, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match=error_match):
            tensor.view(view_shape)


def test_incompatible_view_stride_error_matches_cpu() -> None:
    error_match = "view size is not compatible with input tensor's size and stride"

    for device in ("cpu", "infini"):
        tensor = torch.empty((2, 3), dtype=torch.float32, device=device).t()
        with pytest.raises(RuntimeError, match=error_match):
            tensor.view(6)


def test_view_keeps_storage_alive() -> None:
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    base = copy_cpu_tensor(expected, "infini")
    view = base.view(4, 3)

    del base
    gc.collect()

    actual = torch.empty(view.shape, dtype=view.dtype)
    actual.copy_(view)
    torch.testing.assert_close(actual, expected.view(4, 3))
