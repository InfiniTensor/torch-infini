import os
import subprocess
import sys

import pytest


def _run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_automatically_selects_a_runtime_backend():
    result = _run_python(
        "import torch, torch_infini; "
        "assert torch.infini.is_available(); "
        "assert torch.infini.device_count() > 0; "
        "assert not hasattr(torch.infini, 'init'); "
        "assert not hasattr(torch.infini, 'is_initialized'); "
        "assert not hasattr(torch.infini, 'get_backend')"
    )
    assert result.returncode == 0, result.stderr


def test_expected_runtime_backend_is_selected():
    expected = os.environ.get("TORCH_INFINI_TEST_EXPECTED_BACKEND")
    if expected is None:
        pytest.skip("expected runtime backend was not provided by the test environment")

    assertions = [f"assert torch_infini._C._runtime_backend_name() == {expected!r}"]
    if expected == "nvidia":
        assertions.append(
            "assert torch.infini.device_count() == torch.cuda.device_count() > 0"
        )
    result = _run_python("import torch, torch_infini; " + "; ".join(assertions))
    assert result.returncode == 0, result.stderr


def test_runtime_backend_is_bound_in_worker_threads():
    result = _run_python(
        "import threading\n"
        "import torch\n"
        "import torch_infini\n"
        "expected_count = torch.infini.device_count()\n"
        "observed = []\n"
        "def probe():\n"
        "    observed.append((torch.infini.device_count(), "
        "torch.infini.current_device()))\n"
        "worker = threading.Thread(target=probe)\n"
        "worker.start()\n"
        "worker.join()\n"
        "assert observed == [(expected_count, 0)]"
    )
    assert result.returncode == 0, result.stderr
