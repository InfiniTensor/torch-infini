from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NATIVE_COMMITS = {
    "INFINIRT_COMMIT": (
        "InfiniRT",
        "95c70080f9551e61241110497d163dfcdf9dc7e7",
    ),
    "INFINIOPS_COMMIT": (
        "InfiniOps",
        "296271487beb594a248fd463e5fff14f7ab74293",
    ),
}


def test_python_and_torch_compatibility_metadata():
    with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    assert pyproject["project"]["dependencies"] == ["torch>=2.12,<2.14"]
    assert [
        requirement
        for requirement in pyproject["build-system"]["requires"]
        if requirement.startswith("torch")
    ] == ["torch>=2.12,<2.14"]
    assert {
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    }.issubset(pyproject["project"]["classifiers"])
    assert (
        "tomli>=1; python_version < '3.11'"
        in pyproject["project"]["optional-dependencies"]["dev"]
    )


def test_ci_matrix_matches_the_documented_compatibility_policy():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "cpu.yml").read_text()
    )
    cpu_job = workflow["jobs"]["cpu"]
    matrix = cpu_job["strategy"]["matrix"]["include"]

    pairs = {
        (
            ".".join(row["python_version"].split(".")[:2]),
            row["pytorch_version"].split("+")[0],
        )
        for row in matrix
    }
    assert pairs == {
        (python_version, pytorch_version)
        for python_version in ("3.10", "3.11", "3.12")
        for pytorch_version in ("2.12.0", "2.13.0")
    }
    assert len(matrix) == len(pairs)
    assert len({row["artifact_name"] for row in matrix}) == len(matrix)
    assert sum(row["job_name"] == "CPU" for row in matrix) == 1

    cross_minor_pairs = {
        (
            ".".join(row["pytorch_version"].split("+")[0].split(".")[:2]),
            ".".join(row["mismatch_pytorch_version"].split("+")[0].split(".")[:2]),
        )
        for row in matrix
        if row.get("mismatch_pytorch_version")
    }
    assert cross_minor_pairs == {("2.12", "2.13"), ("2.13", "2.12")}

    step_names = [step["name"] for step in cpu_job["steps"]]
    assert step_names[-2:] == [
        "Install mismatched PyTorch",
        "Verify PyTorch minor mismatch is rejected",
    ]
    assert (
        step_names.index("Run test suite outside the source tree") < len(step_names) - 2
    )

    readme = " ".join((REPO_ROOT / "README.md").read_text().split())
    for env_name, (project, commit) in EXPECTED_NATIVE_COMMITS.items():
        assert cpu_job["env"][env_name] == commit
        assert f"{project} commit `{commit}`" in readme
