"""PyTorch integration for the Infini accelerator stack."""

from __future__ import annotations

import torch

_BACKEND_NAME = "infini"


def _rename_backend() -> None:
    rename_backend = getattr(
        torch.utils,
        "rename_privateuse1_backend",
        getattr(torch, "rename_privateuse1_backend", None),
    )
    if rename_backend is None:
        raise RuntimeError(
            "this PyTorch build does not expose rename_privateuse1_backend"
        )
    try:
        rename_backend(_BACKEND_NAME)
    except RuntimeError as exc:
        message = str(exc).lower()
        if _BACKEND_NAME not in message or "already" not in message:
            raise


_rename_backend()

from . import _C as _C
from . import infini as infini


def _install_device_module() -> None:
    registered = getattr(torch, _BACKEND_NAME, None)
    if registered is infini:
        return
    if registered is not None:
        raise RuntimeError(
            f"torch.{_BACKEND_NAME} is already registered by another module"
        )
    torch._register_device_module(_BACKEND_NAME, infini)


def _generate_privateuse1_methods() -> None:
    try:
        torch.utils.generate_methods_for_privateuse1_backend(
            for_tensor=True,
            for_module=True,
            for_storage=True,
        )
    except RuntimeError as exc:
        message = str(exc)
        if _BACKEND_NAME not in message and "already" not in message.lower():
            raise


_C.initialize()
_install_device_module()
_generate_privateuse1_methods()


__all__ = ["infini"]
