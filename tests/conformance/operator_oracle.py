import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch


StorageCopier = Callable[[torch.Tensor, torch.Tensor], None]


@dataclass(frozen=True)
class OperatorCase:
    name: str
    operator: Callable[..., torch.Tensor]
    make_inputs: Callable[[str, StorageCopier | None], tuple[torch.Tensor, ...]]
    kwargs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorResult:
    tensor: torch.Tensor | None = None
    error: Exception | None = None


class ExpectedOperatorGapError(AssertionError):
    """Raised only after a known CPU/backend divergence is validated."""


def copy_cpu_tensor(source: torch.Tensor, device: str) -> torch.Tensor:
    if source.device.type != "cpu" or not source.is_contiguous():
        raise ValueError("source must be a contiguous CPU tensor")
    if torch.device(device).type == "cpu":
        return source.clone()

    result = torch.empty(source.shape, dtype=source.dtype, device=device)
    result.copy_(source)
    return result


def copy_strided_cpu_tensor(
    source: torch.Tensor,
    device: str,
    copy_storage_from_cpu: StorageCopier | None,
) -> torch.Tensor:
    if source.device.type != "cpu" or source.layout != torch.strided:
        raise ValueError("source must be a strided CPU tensor")
    if source.storage_offset() != 0:
        raise ValueError("raw storage copies require a zero storage offset")
    if torch.device(device).type == "cpu":
        return source
    if copy_storage_from_cpu is None:
        raise ValueError("a raw storage copier is required for strided Infini tensors")

    result = torch.empty_strided(
        source.shape,
        source.stride(),
        dtype=source.dtype,
        device=device,
    )
    copy_storage_from_cpu(result, source)
    return result


def invoke(
    case: OperatorCase,
    device: str,
    copy_storage_from_cpu: StorageCopier | None = None,
) -> OperatorResult:
    inputs = case.make_inputs(device, copy_storage_from_cpu)
    try:
        result = case.operator(*inputs, **case.kwargs)
    except Exception as error:
        return OperatorResult(error=error)

    if not isinstance(result, torch.Tensor):
        raise AssertionError(
            f"{case.name}: operator returned {type(result).__name__}, expected Tensor"
        )
    return OperatorResult(tensor=result)


def assert_operator_matches_cpu(
    case: OperatorCase,
    *,
    device: str = "infini",
    error_match: str | None = None,
    copy_storage_from_cpu: StorageCopier | None = None,
    copy_storage_to_cpu: StorageCopier | None = None,
) -> None:
    expected = invoke(case, "cpu")
    actual = invoke(case, device, copy_storage_from_cpu)

    if expected.error is not None or actual.error is not None:
        if expected.error is None or actual.error is None:
            raise AssertionError(
                f"{case.name}: CPU outcome {_format_result(expected)} != "
                f"{device} outcome {_format_result(actual)}"
            )
        if type(actual.error) is not type(expected.error):
            raise AssertionError(
                f"{case.name}: CPU raised {_format_error(expected.error)}, "
                f"{device} raised {_format_error(actual.error)}"
            )
        if error_match is None:
            raise AssertionError(
                f"{case.name}: error comparison requires a message pattern"
            )
        for backend, error in (("CPU", expected.error), (device, actual.error)):
            if re.search(error_match, str(error)) is None:
                raise AssertionError(
                    f"{case.name}: {backend} error did not match "
                    f"{error_match!r}: {error}"
                )
        return

    if expected.tensor is None or actual.tensor is None:
        raise AssertionError(f"{case.name}: missing successful tensor result")
    assert_tensor_matches_cpu(
        case.name,
        expected.tensor,
        actual.tensor,
        device,
        copy_storage_to_cpu,
    )


def assert_tensor_matches_cpu(
    case_name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    device: str = "infini",
    copy_storage_to_cpu: StorageCopier | None = None,
) -> None:
    expected_device_type = torch.device(device).type
    if actual.device.type != expected_device_type:
        raise AssertionError(
            f"{case_name}: result device {actual.device.type!r} != "
            f"{expected_device_type!r}"
        )

    expected_metadata = tensor_metadata(expected)
    actual_metadata = tensor_metadata(actual)
    if actual_metadata != expected_metadata:
        raise AssertionError(
            f"{case_name}: output metadata mismatch\n"
            f"CPU: {expected_metadata}\n"
            f"{device}: {actual_metadata}"
        )
    assert_tensor_values_match(case_name, expected, actual, copy_storage_to_cpu)


def assert_tensor_values_match(
    case_name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    copy_storage_to_cpu: StorageCopier | None = None,
) -> None:
    torch.testing.assert_close(
        _copy_result_to_cpu(actual, copy_storage_to_cpu),
        expected,
        msg=lambda message: f"{case_name}: output values differ\n{message}",
    )


def tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "dtype": tensor.dtype,
        "layout": tensor.layout,
        "requires_grad": tensor.requires_grad,
        "is_contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
    }


def _copy_result_to_cpu(
    result: torch.Tensor,
    copy_storage_to_cpu: StorageCopier | None,
) -> torch.Tensor:
    if result.device.type == "cpu":
        return result
    if result.device.type != "infini":
        raise AssertionError(f"cannot copy a {result.device.type} result to CPU")

    torch.infini.synchronize(result.device)
    if not result.is_contiguous():
        if copy_storage_to_cpu is None:
            raise AssertionError(
                "a raw storage copier is required for noncontiguous Infini results"
            )
        cpu_result = torch.empty_strided(
            result.shape,
            result.stride(),
            dtype=result.dtype,
        )
        copy_storage_to_cpu(cpu_result, result)
        return cpu_result

    cpu_result = torch.empty(result.shape, dtype=result.dtype)
    cpu_result.copy_(result)
    return cpu_result


def _format_result(result: OperatorResult) -> str:
    if result.error is not None:
        return _format_error(result.error)
    if result.tensor is None:
        return "no result"
    return f"Tensor(metadata={tensor_metadata(result.tensor)})"


def _format_error(error: Exception) -> str:
    return f"{type(error).__name__}({error})"
