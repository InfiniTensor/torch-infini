import threading

import pytest
import torch

import torch_infini  # noqa: F401


def _to_infini(tensor, device="infini"):
    result = torch.empty_like(tensor, device=device)
    result.copy_(tensor)
    return result


def _to_cpu(tensor):
    result = torch.empty(tensor.shape, dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def _assert_matches_cpu(result, expected):
    torch.infini.synchronize(result.device)
    torch.testing.assert_close(_to_cpu(result), expected)


def test_add_tensor_matches_cpu():
    lhs_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs_cpu = torch.full((2, 3), 2.5, dtype=torch.float32)
    lhs = _to_infini(lhs_cpu)
    rhs = _to_infini(rhs_cpu)

    result = torch.add(lhs, rhs)

    _assert_matches_cpu(result, torch.add(lhs_cpu, rhs_cpu))


def test_add_tensor_broadcasts_inputs():
    lhs_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    rhs_cpu = torch.arange(4, dtype=torch.float32)
    lhs = _to_infini(lhs_cpu)
    rhs = _to_infini(rhs_cpu)

    result = torch.add(lhs, rhs)

    _assert_matches_cpu(result, torch.add(lhs_cpu, rhs_cpu))


def test_add_tensor_returns_empty_broadcasted_result():
    lhs = _to_infini(torch.empty((0, 3), dtype=torch.float32))
    rhs = _to_infini(torch.ones((1, 3), dtype=torch.float32))

    result = torch.add(lhs, rhs)

    assert result.shape == (0, 3)
    assert result.numel() == 0


def test_add_tensor_rejects_unsupported_empty_dtype():
    lhs = _to_infini(torch.empty(0, dtype=torch.bool))
    rhs = _to_infini(torch.empty(0, dtype=torch.bool))

    with pytest.raises(RuntimeError, match="InfiniOps does not support ATen dtype"):
        torch.add(lhs, rhs)


def test_add_tensor_rejects_type_promotion():
    lhs = _to_infini(torch.ones(4, dtype=torch.float32))
    rhs = _to_infini(torch.ones(4, dtype=torch.int32))

    with pytest.raises(RuntimeError, match="does not support type promotion"):
        torch.add(lhs, rhs)


def test_add_tensor_rejects_nonunit_alpha():
    lhs = _to_infini(torch.ones(4, dtype=torch.float32))
    rhs = _to_infini(torch.ones(4, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="only supports alpha == 1"):
        torch.add(lhs, rhs, alpha=2)


def test_add_tensor_rejects_cpu_other():
    lhs = _to_infini(torch.ones(4, dtype=torch.float32))
    rhs = torch.ones(4, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="expects two infini tensors"):
        torch.add(lhs, rhs)


def test_add_scalar_overload_is_not_implemented():
    lhs = _to_infini(torch.ones(4, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="expects two infini tensors"):
        torch.add(lhs, 1)


def test_add_out_overload_is_not_implemented():
    lhs = _to_infini(torch.ones(4, dtype=torch.float32))
    rhs = _to_infini(torch.ones(4, dtype=torch.float32))
    out = torch.empty_like(lhs)

    with pytest.raises(NotImplementedError, match=r"aten::add\.out"):
        torch.add(lhs, rhs, out=out)


def test_add_inplace_overload_is_not_implemented():
    lhs = _to_infini(torch.ones(4, dtype=torch.float32))
    rhs = _to_infini(torch.ones(4, dtype=torch.float32))

    with pytest.raises(NotImplementedError, match=r"aten::add\.out"):
        lhs.add_(rhs)


def test_add_tensor_runs_on_worker_thread():
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    lhs = _to_infini(expected)
    rhs = _to_infini(torch.ones_like(expected))
    observed = []
    errors = []

    def run_add():
        try:
            result = torch.add(lhs, rhs)
            torch.infini.synchronize(result.device)
            observed.append(_to_cpu(result))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    worker = threading.Thread(target=run_add)
    worker.start()
    worker.join()

    assert not errors
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected + 1)


def test_add_tensor_preserves_current_device():
    count = torch.infini.device_count()
    if count < 2:
        pytest.skip("requires at least two infini devices")
    initial_device = torch.infini.current_device()
    target_device = (initial_device + 1) % count
    lhs = _to_infini(torch.ones(4), device=f"infini:{target_device}")
    rhs = _to_infini(torch.ones(4), device=f"infini:{target_device}")

    try:
        result = torch.add(lhs, rhs)
        assert torch.infini.current_device() == initial_device
        _assert_matches_cpu(result, torch.full((4,), 2.0))
    finally:
        torch.infini.set_device(initial_device)


def test_add_tensor_rejects_cross_device_inputs():
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")
    lhs = _to_infini(torch.ones(4), device="infini:0")
    rhs = _to_infini(torch.ones(4), device="infini:1")

    with pytest.raises(RuntimeError, match="same infini device"):
        torch.add(lhs, rhs)


def test_add_tensor_uses_current_nondefault_stream():
    lhs_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs_cpu = torch.ones_like(lhs_cpu)
    lhs = _to_infini(lhs_cpu)
    rhs = _to_infini(rhs_cpu)
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        result = torch.add(lhs, rhs)

    stream.synchronize()
    torch.testing.assert_close(_to_cpu(result), lhs_cpu + rhs_cpu)
