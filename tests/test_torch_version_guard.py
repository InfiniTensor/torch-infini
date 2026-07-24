import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_compatibility_module():
    module_path = REPO_ROOT / "torch_infini" / "_compatibility.py"
    spec = importlib.util.spec_from_file_location(
        "torch_infini_compatibility_under_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "runtime_version",
    [
        "2.13",
        "2.13.0+cpu",
        "2.13.0.dev20260724",
        "2.13.0a0+gitabcdef",
        "2.13.0rc1",
    ],
)
def test_matching_torch_minor_accepts_version_suffixes(runtime_version):
    compatibility = _load_compatibility_module()

    compatibility.check_torch_version(
        runtime_version=runtime_version,
        build_version="2.13.0+cpu",
        build_major_minor="2.13",
    )


def test_mismatched_torch_minor_is_rejected():
    compatibility = _load_compatibility_module()

    with pytest.raises(
        ImportError,
        match=(
            r"torch-infini was built against PyTorch 2\.12\.0\+cpu "
            r"\(major\.minor 2\.12\), but the installed PyTorch is "
            r"2\.13\.0a0\+gitabcdef \(major\.minor 2\.13\)"
        ),
    ):
        compatibility.check_torch_version(
            runtime_version="2.13.0a0+gitabcdef",
            build_version="2.12.0+cpu",
            build_major_minor="2.12",
        )


@pytest.mark.parametrize(
    ("runtime_version", "build_version", "build_major_minor", "message"),
    [
        (None, "2.13.0", "2.13", "runtime PyTorch version is unavailable"),
        ("not-a-version", "2.13.0", "2.13", "runtime PyTorch version is invalid"),
        ("2.13.0", None, None, "build metadata is unavailable"),
        ("2.13.0", "not-a-version", "2.13", "build metadata is invalid"),
        ("2.13.0", "2.13.0", "not-a-minor", "build metadata is invalid"),
        ("2.13.0", "2.13.0", "2.12", "build metadata is inconsistent"),
    ],
)
def test_unavailable_or_malformed_versions_fail_clearly(
    runtime_version, build_version, build_major_minor, message
):
    compatibility = _load_compatibility_module()

    with pytest.raises(ImportError, match=message):
        compatibility.check_torch_version(
            runtime_version=runtime_version,
            build_version=build_version,
            build_major_minor=build_major_minor,
        )


@pytest.mark.parametrize(
    ("build_info", "message"),
    [
        (None, "build metadata is unavailable"),
        (
            'BUILD_TORCH_VERSION = "invalid"\nBUILD_TORCH_MAJOR_MINOR = "2.12"\n',
            "build metadata is invalid",
        ),
        ('BUILD_TORCH_VERSION = "2.13.0"\n', "build metadata is invalid"),
        ("BUILD_TORCH_VERSION =\n", "build metadata is invalid"),
        (
            'BUILD_TORCH_VERSION = "2.12.0+cpu"\nBUILD_TORCH_MAJOR_MINOR = "2.12"\n',
            "was built against PyTorch 2.12.0\\+cpu",
        ),
    ],
)
def test_guard_fails_before_backend_rename_and_native_import(
    monkeypatch, tmp_path, build_info, message
):
    package_name = f"torch_infini_guard_under_test_{tmp_path.name}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    shutil.copy(REPO_ROOT / "torch_infini" / "__init__.py", package_dir)
    shutil.copy(REPO_ROOT / "torch_infini" / "_compatibility.py", package_dir)
    if build_info is not None:
        (package_dir / "_build_info.py").write_text(build_info, encoding="utf-8")
    (package_dir / "_C.py").write_text(
        'raise AssertionError("native extension imported")\n', encoding="utf-8"
    )

    rename_calls = []
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0+cpu"
    torch.utils = types.SimpleNamespace(rename_privateuse1_backend=rename_calls.append)
    monkeypatch.setitem(sys.modules, "torch", torch)

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)

    with pytest.raises(ImportError, match=message):
        spec.loader.exec_module(package)

    assert rename_calls == []
    assert f"{package_name}._C" not in sys.modules
