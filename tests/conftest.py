import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dependency_paths(prefix_variable, override_variable, subdirs):
    paths = []
    prefix = os.environ.get(prefix_variable)
    if prefix:
        paths.extend(Path(prefix) / subdir for subdir in subdirs)
    paths.extend(
        Path(path)
        for path in os.environ.get(override_variable, "").split(os.pathsep)
        if path
    )
    return [str(path) for path in paths if path.is_dir()]


def _cuda_include_dirs():
    from torch.utils.cpp_extension import CUDA_HOME

    include_dirs = [
        Path(path)
        for path in os.environ.get("CUDA_INCLUDE_DIRS", "").split(os.pathsep)
        if path
    ]
    for root in (CUDA_HOME, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if root:
            root_path = Path(root)
            include_dirs.extend(
                [
                    root_path / "include",
                    root_path / "targets" / "x86_64-linux" / "include",
                ]
            )
    include_dirs.extend(
        [
            Path("/usr/local/cuda/include"),
            Path("/usr/local/cuda/targets/x86_64-linux/include"),
        ]
    )

    return list(dict.fromkeys(str(path) for path in include_dirs if path.is_dir()))


@pytest.fixture(scope="session")
def infini_ops_test_module(tmp_path_factory):
    import torch_infini

    include_dirs = [
        str(REPO_ROOT / "csrc"),
        *_dependency_paths("INFINI_OPS_PREFIX", "INFINI_OPS_INCLUDE_DIRS", ["include"]),
        *_dependency_paths("INFINI_RT_PREFIX", "INFINI_RT_INCLUDE_DIRS", ["include"]),
        *_cuda_include_dirs(),
    ]
    library_dirs = [
        *_dependency_paths(
            "INFINI_OPS_PREFIX", "INFINI_OPS_LIBRARY_DIRS", ["lib", "lib64"]
        ),
        *_dependency_paths(
            "INFINI_RT_PREFIX", "INFINI_RT_LIBRARY_DIRS", ["lib", "lib64"]
        ),
    ]
    if len(include_dirs) < 3 or len(library_dirs) < 2:
        pytest.skip("InfiniOps and InfiniRT development prefixes are required")

    from torch.utils.cpp_extension import load

    extension_path = Path(torch_infini._C.__file__).resolve()
    return load(
        name="torch_infini_infini_ops_test",
        sources=[str(REPO_ROOT / "tests" / "cpp" / "infini_ops_test.cpp")],
        extra_include_paths=include_dirs,
        extra_ldflags=[
            *(f"-L{path}" for path in library_dirs),
            str(extension_path),
            "-linfiniops",
            "-linfinirt",
        ],
        build_directory=str(tmp_path_factory.mktemp("infini-ops-extension")),
        verbose=True,
    )
