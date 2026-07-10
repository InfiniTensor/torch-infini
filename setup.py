from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME


PACKAGE_ROOT = Path(__file__).resolve().parent


def _split_paths(value):
    return [path for path in value.split(os.pathsep) if path]


def _cuda_include_dirs():
    include_dirs = []
    include_dirs.extend(_split_paths(os.environ.get("CUDA_INCLUDE_DIRS", "")))

    for root in (CUDA_HOME, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if not root:
            continue
        root_path = Path(root)
        include_dirs.append(str(root_path / "include"))
        include_dirs.append(str(root_path / "targets" / "x86_64-linux" / "include"))

    for path in (
        Path("/usr/local/cuda/include"),
        Path("/usr/local/cuda/targets/x86_64-linux/include"),
    ):
        include_dirs.append(str(path))

    existing_dirs = []
    seen = set()
    for path in include_dirs:
        if path in seen or not Path(path).exists():
            continue
        seen.add(path)
        existing_dirs.append(path)
    return existing_dirs


def _infini_paths():
    include_dirs = []
    library_dirs = []
    runtime_library_dirs = []

    prefix = os.environ.get("INFINI_RT_PREFIX")
    if prefix:
        prefix_path = Path(prefix)
        include_dirs.append(str(prefix_path / "include"))
        library_dirs.extend(
            str(path)
            for path in (prefix_path / "lib", prefix_path / "lib64")
            if path.exists()
        )
        runtime_library_dirs.extend(library_dirs)

    include_dirs.extend(_split_paths(os.environ.get("INFINI_RT_INCLUDE_DIRS", "")))
    library_dirs.extend(_split_paths(os.environ.get("INFINI_RT_LIBRARY_DIRS", "")))
    runtime_library_dirs.extend(
        _split_paths(os.environ.get("INFINI_RT_RUNTIME_LIBRARY_DIRS", ""))
    )

    if not include_dirs:
        raise RuntimeError(
            "InfiniRT headers were not found. Set INFINI_RT_PREFIX to an "
            "installed InfiniRT prefix, or set INFINI_RT_INCLUDE_DIRS."
        )

    return include_dirs, library_dirs, runtime_library_dirs


include_dirs, library_dirs, runtime_library_dirs = _infini_paths()
include_dirs.extend(_cuda_include_dirs())

setup(
    packages=["torch_infini"],
    ext_modules=[
        CppExtension(
            name="torch_infini._C",
            sources=[
                "csrc/allocator.cpp",
                "csrc/copy.cpp",
                "csrc/device_guard.cpp",
                "csrc/empty.cpp",
                "csrc/init.cpp",
                "csrc/runtime.cpp",
            ],
            include_dirs=[*include_dirs, str(PACKAGE_ROOT / "csrc")],
            libraries=["infinirt"],
            library_dirs=library_dirs,
            runtime_library_dirs=runtime_library_dirs,
            extra_compile_args={"cxx": ["-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
