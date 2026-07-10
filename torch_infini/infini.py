"""`torch.infini` device module."""

from __future__ import annotations

from contextlib import ContextDecorator
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


def is_initialized() -> bool:
    return True


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


def current_stream(device: Any | None = None) -> int:
    _normalize_device_index(device)
    return 0


def default_stream(device: Any | None = None) -> int:
    _normalize_device_index(device)
    return 0


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


def _is_in_bad_fork() -> bool:
    return False


def manual_seed_all(seed: int) -> None:
    del seed


def manual_seed(seed: int) -> None:
    manual_seed_all(seed)


__all__ = [
    "current_device",
    "current_stream",
    "default_stream",
    "device",
    "device_count",
    "get_device_name",
    "is_available",
    "is_initialized",
    "manual_seed",
    "manual_seed_all",
    "set_device",
    "synchronize",
]
