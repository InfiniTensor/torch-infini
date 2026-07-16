"""`torch.infini` device module."""

from __future__ import annotations

from contextlib import ContextDecorator, nullcontext
from typing import Any

import torch

from . import _C


def _normalize_device_index(device: Any | None = None) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    torch_device = torch.device(device)
    if torch_device.type != "infini":
        raise ValueError(f"expected an infini device, got {torch_device}")
    if torch_device.index is None:
        return current_device()
    return torch_device.index


def is_available() -> bool:
    return _C.is_available()


def device_count() -> int:
    return _C.device_count()


def current_device() -> int:
    return _C.current_device()


def set_device(device: Any) -> None:
    _C.set_device(_normalize_device_index(device))


def synchronize(device: Any | None = None) -> None:
    _C.synchronize(_normalize_device_index(device))


def get_device_name(device: Any | None = None) -> str:
    return _C.get_device_name(_normalize_device_index(device))


class Stream(torch.Stream):
    def __new__(
        cls,
        device: Any | None = None,
        *,
        priority: int = 0,
        stream_id: int | None = None,
        device_index: int | None = None,
        device_type: int | None = None,
    ) -> Stream:
        identity = (stream_id, device_index, device_type)
        if any(value is not None for value in identity):
            if device is not None or any(value is None for value in identity):
                raise TypeError(
                    "stream_id, device_index, and device_type must be provided "
                    "together without device"
                )
            result = super().__new__(
                cls,
                stream_id=stream_id,
                device_index=device_index,
                device_type=device_type,
                priority=priority,
            )
        else:
            index = _normalize_device_index(device)
            result = super().__new__(
                cls, torch.device("infini", index), priority=priority
            )

        if result.device.type != "infini":
            raise ValueError(f"expected an infini stream, got {result.device}")
        return result

    @property
    def native_handle(self) -> int:
        return _C._stream_native_handle(self.stream_id, self.device_index)


def _wrap_stream(stream_obj: torch.Stream) -> Stream:
    return Stream(
        stream_id=stream_obj.stream_id,
        device_index=stream_obj.device_index,
        device_type=stream_obj.device_type,
    )


def _check_stream(stream_obj: torch.Stream) -> torch.Stream:
    if not isinstance(stream_obj, torch.Stream):
        raise TypeError(f"expected a torch.Stream, got {type(stream_obj).__name__}")
    if stream_obj.device.type != "infini":
        raise ValueError(f"expected an infini stream, got {stream_obj.device}")
    return stream_obj


def current_stream(device: Any | None = None) -> Stream:
    index = _normalize_device_index(device)
    return _wrap_stream(torch.accelerator.current_stream(index))


def default_stream(device: Any | None = None) -> Stream:
    index = _normalize_device_index(device)
    current = torch.accelerator.current_stream(index)
    return Stream(
        stream_id=0,
        device_index=index,
        device_type=current.device_type,
    )


def set_stream(stream_obj: torch.Stream | None) -> None:
    if stream_obj is None:
        return
    torch.accelerator.set_stream(_check_stream(stream_obj))


def stream(stream_obj: torch.Stream | None) -> Any:
    if stream_obj is None:
        return nullcontext()
    return _check_stream(stream_obj)


class device(ContextDecorator):
    def __init__(self, device_spec: Any):
        self.idx = _normalize_device_index(device_spec)
        self.prev_idx: int | None = None

    def __enter__(self) -> device:
        self.prev_idx = current_device()
        set_device(self.idx)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.prev_idx is not None:
            set_device(self.prev_idx)
        return False


__all__ = [
    "Stream",
    "current_device",
    "current_stream",
    "default_stream",
    "device",
    "device_count",
    "get_device_name",
    "is_available",
    "set_device",
    "set_stream",
    "stream",
    "synchronize",
]
