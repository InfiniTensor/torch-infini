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


class Event(torch.Event):
    def __new__(
        cls,
        *,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> Event:
        if blocking:
            raise NotImplementedError(
                "torch.infini.Event does not support blocking=True"
            )
        if interprocess:
            raise NotImplementedError(
                "torch.infini.Event does not support interprocess=True"
            )
        result = super().__new__(
            cls,
            "infini",
            enable_timing=enable_timing,
        )
        result._enable_timing = enable_timing
        return result

    def record(self, stream_obj: torch.Stream | None = None) -> None:
        if stream_obj is None:
            stream_obj = current_stream()
        super().record(_check_stream(stream_obj))

    def wait(self, stream_obj: torch.Stream | None = None) -> None:
        if stream_obj is None:
            stream_obj = current_stream()
        super().wait(_check_stream(stream_obj))

    def elapsed_time(self, end_event: Event) -> float:
        if not isinstance(end_event, Event):
            raise TypeError(f"expected an infini Event, got {type(end_event).__name__}")
        if not self._enable_timing or not end_event._enable_timing:
            raise ValueError(
                "Both events must be created with argument 'enable_timing=True'."
            )
        if self.event_id == 0 or end_event.event_id == 0:
            raise ValueError(
                "Both events must be recorded before calculating elapsed time."
            )
        if self.device != end_event.device:
            raise ValueError("Both events must be recorded on the same device.")
        if not self.query() or not end_event.query():
            raise RuntimeError(
                "Both events must be completed before calculating elapsed time."
            )
        return _C._event_elapsed_time(
            self.event_id,
            end_event.event_id,
            self.device.index,
        )


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

    def record_event(self, event: torch.Event | None = None) -> torch.Event:
        if event is None:
            event = Event()
        _check_event(event).record(self)
        return event

    def wait_event(self, event: torch.Event) -> None:
        _check_event(event).wait(self)

    def wait_stream(self, stream_obj: torch.Stream) -> None:
        self.wait_event(_check_stream(stream_obj).record_event())


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


def _check_event(event_obj: torch.Event) -> torch.Event:
    if not isinstance(event_obj, torch.Event):
        raise TypeError(f"expected a torch.Event, got {type(event_obj).__name__}")
    if event_obj.device.type != "infini":
        raise ValueError(f"expected an infini event, got {event_obj.device}")
    return event_obj


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
    "Event",
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
