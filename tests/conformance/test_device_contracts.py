import pytest

from .backends import compare_observations


CONTRACT_ID = "device"
COVERED_API = frozenset(
    {
        "current_device",
        "device",
        "device_count",
        "get_device_name",
        "is_available",
        "set_device",
        "synchronize",
    }
)


def _observe_device_module(backend):
    module = backend.module
    count = module.device_count()
    current = module.current_device()
    name = module.get_device_name(current)
    return {
        "available": module.is_available(),
        "count_is_positive_int": type(count) is int and count > 0,
        "current_is_valid_int": (type(current) is int and 0 <= current < count),
        "name_is_nonempty_string": isinstance(name, str) and bool(name),
    }


def test_device_module_contract(accelerator_backend):
    assert _observe_device_module(accelerator_backend) == {
        "available": True,
        "count_is_positive_int": True,
        "current_is_valid_int": True,
        "name_is_nonempty_string": True,
    }


def test_available_device_observations_match(available_accelerator_backends):
    observations = {
        backend.name: _observe_device_module(backend)
        for backend in available_accelerator_backends
    }

    assert compare_observations(observations) == {}


def test_device_context_restores_after_normal_and_exceptional_exit(
    accelerator_backend,
):
    module = accelerator_backend.module
    initial = module.current_device()

    with module.device(initial):
        assert module.current_device() == initial

    assert module.current_device() == initial

    def fail_inside_device_context():
        with module.device(initial):
            assert module.current_device() == initial
            raise RuntimeError("forced")

    with pytest.raises(RuntimeError, match="forced"):
        fail_inside_device_context()

    assert module.current_device() == initial


def test_device_context_selects_and_restores_another_device(
    accelerator_backend,
):
    module = accelerator_backend.module
    count = module.device_count()
    if count < 2:
        pytest.skip("requires at least two devices")
    initial = module.current_device()
    target = (initial + 1) % count

    try:
        with module.device(target):
            assert module.current_device() == target
        assert module.current_device() == initial
    finally:
        module.set_device(initial)


def test_explicit_synchronize_current_device_preserves_current_device(
    accelerator_backend,
):
    module = accelerator_backend.module
    initial = module.current_device()

    module.synchronize(initial)

    assert module.current_device() == initial


def test_explicit_synchronize_another_device_preserves_current_device(
    accelerator_backend,
):
    module = accelerator_backend.module
    count = module.device_count()
    if count < 2:
        pytest.skip("requires at least two devices")
    initial = module.current_device()
    target = (initial + 1) % count

    try:
        module.synchronize(target)
        assert module.current_device() == initial
    finally:
        module.set_device(initial)
