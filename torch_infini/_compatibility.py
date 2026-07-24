"""Runtime compatibility checks for the torch-infini extension."""

from __future__ import annotations

import re


_VERSION_PREFIX = re.compile(r"^([0-9]+)\.([0-9]+)")
_NORMALIZED_MAJOR_MINOR = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _major_minor(version: str) -> str:
    match = _VERSION_PREFIX.match(version)
    if match is None:
        raise ValueError
    return f"{int(match.group(1))}.{int(match.group(2))}"


def check_torch_version(
    *,
    runtime_version: object,
    build_version: object,
    build_major_minor: object,
) -> None:
    """Reject a native extension built for another PyTorch minor version."""
    if runtime_version is None:
        raise ImportError(
            "torch-infini cannot verify compatibility because the runtime PyTorch "
            "version is unavailable."
        )
    if not isinstance(runtime_version, str):
        raise ImportError(
            "torch-infini cannot verify compatibility because the runtime PyTorch "
            f"version is invalid: {runtime_version!r}."
        )
    try:
        runtime_major_minor = _major_minor(runtime_version)
    except ValueError as exc:
        raise ImportError(
            "torch-infini cannot verify compatibility because the runtime PyTorch "
            f"version is invalid: {runtime_version!r}."
        ) from exc

    if build_version is None or build_major_minor is None:
        raise ImportError(
            "torch-infini build metadata is unavailable. Reinstall torch-infini "
            "from a complete wheel or rebuild it."
        )
    if not isinstance(build_version, str) or not isinstance(build_major_minor, str):
        raise ImportError(
            "torch-infini build metadata is invalid. Reinstall torch-infini from a "
            "complete wheel or rebuild it."
        )
    try:
        parsed_build_major_minor = _major_minor(build_version)
    except ValueError as exc:
        raise ImportError(
            "torch-infini build metadata is invalid: the recorded PyTorch version "
            f"is {build_version!r}."
        ) from exc
    if _NORMALIZED_MAJOR_MINOR.fullmatch(build_major_minor) is None:
        raise ImportError(
            "torch-infini build metadata is invalid: the recorded PyTorch "
            f"major.minor is {build_major_minor!r}."
        )
    if parsed_build_major_minor != build_major_minor:
        raise ImportError(
            "torch-infini build metadata is inconsistent: the recorded PyTorch "
            f"version {build_version!r} does not have major.minor "
            f"{build_major_minor!r}."
        )

    if runtime_major_minor != build_major_minor:
        raise ImportError(
            f"torch-infini was built against PyTorch {build_version} "
            f"(major.minor {build_major_minor}), but the installed PyTorch is "
            f"{runtime_version} (major.minor {runtime_major_minor}). Install a "
            "torch-infini wheel built with the installed PyTorch minor version."
        )
