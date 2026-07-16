import runpy
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_infinirt_library_dirs_are_only_used_for_linking(monkeypatch, tmp_path):
    prefix = tmp_path / "infinirt"
    prefix_library_dir = prefix / "lib"
    prefix_library_dir.mkdir(parents=True)
    extra_library_dir = tmp_path / "extra-lib"
    extra_library_dir.mkdir()

    captured = {}

    def cpp_extension(**kwargs):
        captured["extension"] = kwargs
        return kwargs

    setuptools = types.ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.setdefault("setup", kwargs)
    cpp_extension_module = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension_module.BuildExtension = object
    cpp_extension_module.CppExtension = cpp_extension
    cpp_extension_module.CUDA_HOME = None
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.cpp_extension = cpp_extension_module
    torch = types.ModuleType("torch")
    torch.utils = torch_utils

    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension_module)
    monkeypatch.setenv("INFINI_RT_PREFIX", str(prefix))
    monkeypatch.setenv("INFINI_RT_LIBRARY_DIRS", str(extra_library_dir))

    runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="__main__")

    extension = captured["extension"]
    assert extension["libraries"] == ["infinirt"]
    assert extension["library_dirs"] == [
        str(prefix_library_dir),
        str(extra_library_dir),
    ]
    assert "runtime_library_dirs" not in extension
