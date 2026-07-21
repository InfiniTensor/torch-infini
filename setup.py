from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME


PACKAGE_ROOT = Path(__file__).resolve().parent


def _split_paths(value):
    return [path for path in value.split(os.pathsep) if path]


def _cuda_include_dirs():
    # Some InfiniRT installations expose CUDA headers from their public API.
    # Keep this transitive dependency isolated until InfiniRT exports it.
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


def _dependency_paths(
    dependency_name,
    prefix_variable,
    include_dirs_variable,
    library_dirs_variable,
):
    include_dirs = []
    library_dirs = []

    prefix = os.environ.get(prefix_variable)
    if prefix:
        prefix_path = Path(prefix)
        include_dirs.append(str(prefix_path / "include"))
        library_dirs.extend(
            str(path)
            for path in (prefix_path / "lib", prefix_path / "lib64")
            if path.exists()
        )

    include_dirs.extend(_split_paths(os.environ.get(include_dirs_variable, "")))
    library_dirs.extend(_split_paths(os.environ.get(library_dirs_variable, "")))

    if not include_dirs:
        raise RuntimeError(
            f"{dependency_name} headers were not found. Set {prefix_variable} "
            f"to an installed {dependency_name} prefix, or set "
            f"{include_dirs_variable}."
        )

    return include_dirs, library_dirs


infini_ops_include_dirs, infini_ops_library_dirs = _dependency_paths(
    "InfiniOps",
    "INFINI_OPS_PREFIX",
    "INFINI_OPS_INCLUDE_DIRS",
    "INFINI_OPS_LIBRARY_DIRS",
)
infini_rt_include_dirs, infini_rt_library_dirs = _dependency_paths(
    "InfiniRT",
    "INFINI_RT_PREFIX",
    "INFINI_RT_INCLUDE_DIRS",
    "INFINI_RT_LIBRARY_DIRS",
)
include_dirs = [*infini_ops_include_dirs, *infini_rt_include_dirs]
include_dirs.extend(_cuda_include_dirs())
library_dirs = [*infini_ops_library_dirs, *infini_rt_library_dirs]

setup(
    packages=["torch_infini"],
    ext_modules=[
        CppExtension(
            name="torch_infini._C",
            sources=[
                "csrc/add.cpp",
                "csrc/allocator.cpp",
                "csrc/copy.cpp",
                "csrc/device_guard.cpp",
                "csrc/empty.cpp",
                "csrc/infini_ops.cpp",
                "csrc/init.cpp",
                "csrc/runtime.cpp",
                "csrc/stream.cpp",
            ],
            include_dirs=[*include_dirs, str(PACKAGE_ROOT / "csrc")],
            libraries=[],
            library_dirs=library_dirs,
            extra_compile_args={"cxx": ["-std=c++17"]},
            extra_link_args=[
                "-Wl,--no-as-needed",
                "-linfiniops",
                "-linfinirt",
                "-Wl,--as-needed",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
