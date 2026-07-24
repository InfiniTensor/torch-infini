import os

import pytest
import torch

ALL_OPTIONAL_CAPABILITIES = {
    "async_memcpy": True,
    "events": True,
    "pinned_host_allocation": True,
    "async_allocation": True,
    "async_free": True,
}
ASYNC_MEMCPY_ONLY_CAPABILITIES = {
    "async_memcpy": True,
    "events": False,
    "pinned_host_allocation": False,
    "async_allocation": False,
    "async_free": False,
}
NO_ASYNC_MEMORY_CAPABILITIES = {
    "async_memcpy": True,
    "events": True,
    "pinned_host_allocation": True,
    "async_allocation": False,
    "async_free": False,
}
EXPECTED_RUNTIME_CAPABILITY_POLICY = {
    "cpu": {
        "async_memcpy": False,
        "events": True,
        "pinned_host_allocation": True,
        "async_allocation": False,
        "async_free": False,
    },
    "nvidia": ALL_OPTIONAL_CAPABILITIES,
    "cambricon": ASYNC_MEMCPY_ONLY_CAPABILITIES,
    "ascend": ASYNC_MEMCPY_ONLY_CAPABILITIES,
    "metax": ALL_OPTIONAL_CAPABILITIES,
    "moore": NO_ASYNC_MEMORY_CAPABILITIES,
    "iluvatar": ALL_OPTIONAL_CAPABILITIES,
    "hygon": ALL_OPTIONAL_CAPABILITIES,
}


def test_every_infinirt_backend_has_explicit_runtime_capabilities(
    infini_ops_test_module,
):
    assert (
        infini_ops_test_module.runtime_capability_policy()
        == EXPECTED_RUNTIME_CAPABILITY_POLICY
    )


def test_copy_routing_agrees_with_runtime_capability_model(
    infini_ops_test_module,
):
    expected_routing = {
        backend: capabilities["async_memcpy"]
        for backend, capabilities in EXPECTED_RUNTIME_CAPABILITY_POLICY.items()
    }

    assert infini_ops_test_module.copy_async_memcpy_routing() == expected_routing


@pytest.mark.parametrize(
    ("tensor", "shape", "strides", "is_contiguous", "has_broadcast_dim"),
    [
        (torch.empty((2, 3, 4)), [2, 3, 4], [12, 4, 1], True, False),
        (torch.empty((1, 3)).expand(4, 3), [4, 3], [0, 1], False, True),
        (torch.empty((4, 6))[:, ::2], [4, 3], [6, 2], False, False),
    ],
)
def test_cpu_tensor_metadata(
    infini_ops_test_module,
    tensor,
    shape,
    strides,
    is_contiguous,
    has_broadcast_dim,
):
    metadata = infini_ops_test_module.tensor_metadata(tensor)

    assert metadata == {
        "data_ptr": tensor.data_ptr(),
        "dtype": "float32",
        "device_type": "cpu",
        "device_index": 0,
        "shape": shape,
        "strides": strides,
        "is_contiguous": is_contiguous,
        "has_broadcast_dim": has_broadcast_dim,
    }


@pytest.mark.parametrize(
    ("shape", "strides", "is_contiguous", "has_broadcast_dim"),
    [
        ((2, 3, 4), (12, 4, 1), True, False),
        ((4, 3), (0, 1), False, True),
        ((4, 3), (6, 2), False, False),
    ],
)
def test_infini_tensor_metadata(
    infini_ops_test_module,
    shape,
    strides,
    is_contiguous,
    has_broadcast_dim,
):
    expected_backend = os.environ.get("TORCH_INFINI_TEST_EXPECTED_BACKEND")
    if expected_backend is None:
        pytest.skip("expected runtime backend was not provided")

    tensor = torch.empty_strided(shape, strides, dtype=torch.float32, device="infini:0")
    metadata = infini_ops_test_module.tensor_metadata(tensor)

    assert metadata == {
        "data_ptr": tensor.data_ptr(),
        "dtype": "float32",
        "device_type": expected_backend,
        "device_index": 0,
        "shape": list(shape),
        "strides": list(strides),
        "is_contiguous": is_contiguous,
        "has_broadcast_dim": has_broadcast_dim,
    }


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64])
def test_unsupported_dtype_fails_clearly(infini_ops_test_module, dtype):
    tensor = torch.empty(4, dtype=dtype)

    with pytest.raises(RuntimeError, match="InfiniOps does not support ATen dtype"):
        infini_ops_test_module.tensor_metadata(tensor)


@pytest.mark.parametrize("device", ["cuda:0", "meta"])
def test_unsupported_device_fails_clearly(infini_ops_test_module, device):
    with pytest.raises(
        RuntimeError, match="InfiniOps adapters only support CPU and infini devices"
    ):
        infini_ops_test_module.device_metadata(device)


def test_execution_context_uses_stream_and_native_implementation(
    infini_ops_test_module,
):
    stream_address = 0x1234

    assert infini_ops_test_module.execution_context_metadata(stream_address) == {
        "stream": stream_address,
        "implementation_index": 0,
    }


def test_execution_context_uses_current_nondefault_stream(infini_ops_test_module):
    stream = torch.infini.Stream()

    with torch.infini.stream(stream):
        metadata = infini_ops_test_module.current_execution_context_metadata(
            str(stream.device)
        )

    assert metadata == {
        "stream": stream.native_handle,
        "implementation_index": 0,
    }


def test_stream_synchronize_waits_for_in_flight_submission(
    infini_ops_test_module,
):
    assert infini_ops_test_module.stream_synchronize_waits_for_submission("infini:0")


def test_stream_submission_waits_for_synchronous_work(infini_ops_test_module):
    assert infini_ops_test_module.stream_submission_waits_for_synchronous_work(
        "infini:0"
    )


def test_add_tensor_supports_noncontiguous_inputs(infini_ops_test_module):
    lhs_cpu = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    rhs_cpu = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    lhs_cpu.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    rhs_cpu.copy_(torch.full((2, 3), 2.5, dtype=torch.float32))
    lhs = torch.empty_strided(
        lhs_cpu.shape, lhs_cpu.stride(), dtype=lhs_cpu.dtype, device="infini"
    )
    rhs = torch.empty_strided(
        rhs_cpu.shape, rhs_cpu.stride(), dtype=rhs_cpu.dtype, device="infini"
    )
    infini_ops_test_module.copy_storage_from_cpu(lhs, lhs_cpu)
    infini_ops_test_module.copy_storage_from_cpu(rhs, rhs_cpu)

    result = torch.add(lhs, rhs)

    torch.infini.synchronize(result.device)
    result_cpu = torch.empty(result.shape, dtype=result.dtype)
    result_cpu.copy_(result)
    torch.testing.assert_close(result_cpu, torch.add(lhs_cpu, rhs_cpu))
