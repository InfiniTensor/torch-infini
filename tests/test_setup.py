import runpy
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_setup(monkeypatch):
    captured = {}

    def cpp_extension(**kwargs):
        captured["extension"] = kwargs
        return kwargs

    class BuildPy:
        def run(self):
            captured["build_py_run"] = True

    class BuildExtension:
        def run(self):
            captured["build_ext_run"] = True

    setuptools = types.ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.setdefault("setup", kwargs)
    setuptools_command = types.ModuleType("setuptools.command")
    build_py_module = types.ModuleType("setuptools.command.build_py")
    build_py_module.build_py = BuildPy
    cpp_extension_module = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension_module.BuildExtension = BuildExtension
    cpp_extension_module.CppExtension = cpp_extension
    cpp_extension_module.CUDA_HOME = None
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.cpp_extension = cpp_extension_module
    torch = types.ModuleType("torch")
    torch.__version__ = "2.13.0a0+gitabcdef"
    torch.compiled_with_cxx11_abi = lambda: True
    torch.utils = torch_utils

    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setitem(sys.modules, "setuptools.command", setuptools_command)
    monkeypatch.setitem(sys.modules, "setuptools.command.build_py", build_py_module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension_module)

    runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="__main__")
    return captured


def test_infiniops_and_infinirt_paths_are_only_used_for_linking(monkeypatch, tmp_path):
    infini_ops_prefix = tmp_path / "infiniops"
    infini_ops_include_dir = infini_ops_prefix / "include"
    infini_ops_library_dir = infini_ops_prefix / "lib"
    infini_ops_include_dir.mkdir(parents=True)
    infini_ops_library_dir.mkdir()
    extra_infini_ops_include_dir = tmp_path / "extra-infiniops-include"
    extra_infini_ops_library_dir = tmp_path / "extra-infiniops-lib"
    extra_infini_ops_include_dir.mkdir()
    extra_infini_ops_library_dir.mkdir()

    infini_rt_prefix = tmp_path / "infinirt"
    infini_rt_include_dir = infini_rt_prefix / "include"
    infini_rt_library_dir = infini_rt_prefix / "lib64"
    infini_rt_include_dir.mkdir(parents=True)
    infini_rt_library_dir.mkdir()
    extra_infini_rt_include_dir = tmp_path / "extra-infinirt-include"
    extra_infini_rt_library_dir = tmp_path / "extra-infinirt-lib"
    extra_infini_rt_include_dir.mkdir()
    extra_infini_rt_library_dir.mkdir()

    monkeypatch.setenv("INFINI_OPS_PREFIX", str(infini_ops_prefix))
    monkeypatch.setenv("INFINI_OPS_INCLUDE_DIRS", str(extra_infini_ops_include_dir))
    monkeypatch.setenv("INFINI_OPS_LIBRARY_DIRS", str(extra_infini_ops_library_dir))
    monkeypatch.setenv("INFINI_RT_PREFIX", str(infini_rt_prefix))
    monkeypatch.setenv("INFINI_RT_INCLUDE_DIRS", str(extra_infini_rt_include_dir))
    monkeypatch.setenv("INFINI_RT_LIBRARY_DIRS", str(extra_infini_rt_library_dir))

    extension = _run_setup(monkeypatch)["extension"]

    assert extension["libraries"] == []
    assert extension["extra_link_args"] == [
        "-Wl,--no-as-needed",
        "-linfiniops",
        "-linfinirt",
        "-Wl,--as-needed",
    ]
    assert extension["library_dirs"] == [
        str(infini_ops_library_dir),
        str(extra_infini_ops_library_dir),
        str(infini_rt_library_dir),
        str(extra_infini_rt_library_dir),
    ]
    assert extension["include_dirs"][:4] == [
        str(infini_ops_include_dir),
        str(extra_infini_ops_include_dir),
        str(infini_rt_include_dir),
        str(extra_infini_rt_include_dir),
    ]
    assert extension["include_dirs"][-1] == str(REPO_ROOT / "csrc")
    assert "csrc/add.cpp" in extension["sources"]
    assert "csrc/hooks.cpp" in extension["sources"]
    assert "csrc/host_allocator.cpp" in extension["sources"]
    assert "csrc/infini_ops.cpp" in extension["sources"]
    assert "runtime_library_dirs" not in extension


def test_infiniops_paths_are_required(monkeypatch, tmp_path):
    infini_rt_prefix = tmp_path / "infinirt"
    (infini_rt_prefix / "include").mkdir(parents=True)
    (infini_rt_prefix / "lib").mkdir()

    monkeypatch.delenv("INFINI_OPS_PREFIX", raising=False)
    monkeypatch.delenv("INFINI_OPS_INCLUDE_DIRS", raising=False)
    monkeypatch.delenv("INFINI_OPS_LIBRARY_DIRS", raising=False)
    monkeypatch.setenv("INFINI_RT_PREFIX", str(infini_rt_prefix))

    with pytest.raises(RuntimeError, match="InfiniOps headers were not found"):
        _run_setup(monkeypatch)


@pytest.mark.parametrize("command_name", ["build_py", "build_ext"])
def test_build_commands_generate_torch_build_info(monkeypatch, tmp_path, command_name):
    infini_ops_include_dir = tmp_path / "infiniops" / "include"
    infini_rt_include_dir = tmp_path / "infinirt" / "include"
    infini_ops_include_dir.mkdir(parents=True)
    infini_rt_include_dir.mkdir(parents=True)
    monkeypatch.setenv("INFINI_OPS_INCLUDE_DIRS", str(infini_ops_include_dir))
    monkeypatch.setenv("INFINI_RT_INCLUDE_DIRS", str(infini_rt_include_dir))

    captured = _run_setup(monkeypatch)
    package_root = tmp_path / "package-root"
    (package_root / "torch_infini").mkdir(parents=True)
    command = captured["setup"]["cmdclass"][command_name]
    command.run.__globals__["PACKAGE_ROOT"] = package_root

    command().run()

    build_info_path = package_root / "torch_infini" / "_build_info.py"
    build_info = runpy.run_path(str(build_info_path))
    assert build_info["BUILD_TORCH_VERSION"] == "2.13.0a0+gitabcdef"
    assert build_info["BUILD_TORCH_MAJOR_MINOR"] == "2.13"
    assert build_info["BUILD_TORCH_CXX11_ABI"] is True
    assert captured[f"{command_name}_run"] is True


@pytest.mark.parametrize(
    ("compiled_with_cxx11_abi", "message"),
    [
        (None, "torch.compiled_with_cxx11_abi is unavailable"),
        (lambda: 1, r"torch.compiled_with_cxx11_abi\(\) returned 1"),
    ],
)
def test_build_commands_require_a_boolean_torch_cxx11_abi(
    monkeypatch,
    tmp_path,
    compiled_with_cxx11_abi,
    message,
):
    infini_ops_include_dir = tmp_path / "infiniops" / "include"
    infini_rt_include_dir = tmp_path / "infinirt" / "include"
    infini_ops_include_dir.mkdir(parents=True)
    infini_rt_include_dir.mkdir(parents=True)
    monkeypatch.setenv("INFINI_OPS_INCLUDE_DIRS", str(infini_ops_include_dir))
    monkeypatch.setenv("INFINI_RT_INCLUDE_DIRS", str(infini_rt_include_dir))

    captured = _run_setup(monkeypatch)
    command = captured["setup"]["cmdclass"]["build_py"]
    command.run.__globals__["torch"].compiled_with_cxx11_abi = compiled_with_cxx11_abi
    package_root = tmp_path / "package-root"
    (package_root / "torch_infini").mkdir(parents=True)
    command.run.__globals__["PACKAGE_ROOT"] = package_root

    with pytest.raises(RuntimeError, match=message):
        command().run()

    assert "build_py_run" not in captured
