import threading

import pytest

from .backends import compare_observations


def _observe_streams(backend):
    module = backend.module
    previous = module.current_stream()
    default = module.default_stream()
    created = module.Stream()
    try:
        with module.stream(created):
            selected_inside = module.current_stream() == created
        restored = module.current_stream() == previous
    finally:
        module.set_stream(previous)
    return {
        "context_restores_stream": restored,
        "default_is_current": default == previous,
        "new_is_distinct": created != default,
        "query_returns_bool": isinstance(created.query(), bool),
        "selected_inside_context": selected_inside,
        "stream_device_is_normalized": (
            backend.normalize_device(created.device)[0] == "accelerator"
        ),
    }


def test_stream_module_contract(accelerator_backend):
    assert _observe_streams(accelerator_backend) == {
        "context_restores_stream": True,
        "default_is_current": True,
        "new_is_distinct": True,
        "query_returns_bool": True,
        "selected_inside_context": True,
        "stream_device_is_normalized": True,
    }


def test_available_stream_observations_match(available_accelerator_backends):
    observations = {
        backend.name: _observe_streams(backend)
        for backend in available_accelerator_backends
    }

    assert compare_observations(observations) == {}


def test_stream_context_restores_after_exception(accelerator_backend):
    module = accelerator_backend.module
    previous = module.current_stream()
    created = module.Stream()

    def fail_inside_stream_context():
        with module.stream(created):
            assert module.current_stream() == created
            raise RuntimeError("forced")

    try:
        with pytest.raises(RuntimeError, match="forced"):
            fail_inside_stream_context()
        assert module.current_stream() == previous
    finally:
        module.set_stream(previous)


def test_stream_query_and_synchronize(accelerator_backend):
    stream = accelerator_backend.module.Stream()

    assert isinstance(stream.query(), bool)
    stream.synchronize()
    assert stream.query()


def test_current_stream_is_thread_local(accelerator_backend):
    module = accelerator_backend.module
    main_stream = module.current_stream()
    created = module.Stream()
    observed = []

    def use_stream():
        before = module.current_stream()
        module.set_stream(created)
        observed.append((before, module.current_stream()))

    worker = threading.Thread(target=use_stream)
    worker.start()
    worker.join()

    assert observed == [(module.default_stream(), created)]
    assert module.current_stream() == main_stream


def test_current_stream_is_device_specific(accelerator_backend):
    module = accelerator_backend.module
    count = module.device_count()
    if count < 2:
        pytest.skip("requires at least two devices")
    initial_device = module.current_device()
    previous = [module.current_stream(index) for index in range(2)]
    streams = [
        module.Stream(device=f"{accelerator_backend.device_type}:{index}")
        for index in range(2)
    ]

    try:
        module.set_stream(streams[0])
        module.set_stream(streams[1])

        assert module.current_stream(0) == streams[0]
        assert module.current_stream(1) == streams[1]
    finally:
        module.set_stream(previous[0])
        module.set_stream(previous[1])
        module.set_device(initial_device)
