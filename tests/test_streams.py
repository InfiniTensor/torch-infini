import threading

import pytest
import torch

import torch_infini


pytestmark = pytest.mark.skipif(
    not torch.infini.is_available(), reason="no infini device"
)


def test_stream_api_is_exposed():
    stream = torch.infini.Stream()

    assert isinstance(stream, torch.Stream)
    assert callable(torch.infini.current_stream)
    assert callable(torch.infini.default_stream)
    assert callable(torch.infini.set_stream)
    assert callable(torch.infini.stream)


def test_default_and_new_streams_have_distinct_ids():
    default = torch.infini.default_stream()
    current = torch.infini.current_stream()
    created = torch.infini.Stream()

    assert default == current
    assert created != default
    assert created.stream_id != default.stream_id
    assert created.device == torch.device("infini", torch.infini.current_device())

    if torch_infini._C._runtime_backend_name() == "nvidia":
        assert created.native_handle != default.native_handle


def test_stream_can_be_reconstructed_from_its_c10_identity():
    stream = torch.infini.Stream()
    reconstructed = torch.infini.Stream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )
    generic = torch.Stream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )

    assert reconstructed == stream
    assert reconstructed.native_handle == stream.native_handle
    if hasattr(generic, "native_handle"):
        assert generic.native_handle == stream.native_handle


def test_stream_context_restores_current_stream_on_error():
    previous = torch.infini.current_stream()
    created = torch.infini.Stream()

    def fail_in_stream():
        with torch.infini.stream(created):
            assert torch.infini.current_stream() == created
            raise RuntimeError("forced")

    with pytest.raises(RuntimeError, match="forced"):
        fail_in_stream()

    assert torch.infini.current_stream() == previous


def test_none_stream_helpers_are_noops():
    previous = torch.infini.current_stream()

    torch.infini.set_stream(None)
    with torch.infini.stream(None):
        assert torch.infini.current_stream() == previous

    assert torch.infini.current_stream() == previous


def test_current_stream_is_thread_local():
    main_stream = torch.infini.current_stream()
    created = torch.infini.Stream()
    observed = []

    def use_stream():
        before = torch.infini.current_stream()
        torch.infini.set_stream(created)
        observed.append((before, torch.infini.current_stream()))

    worker = threading.Thread(target=use_stream)
    worker.start()
    worker.join()

    assert observed == [(torch.infini.default_stream(), created)]
    assert torch.infini.current_stream() == main_stream


def test_current_stream_is_device_specific():
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")

    initial_device = torch.infini.current_device()
    previous = [torch.infini.current_stream(index) for index in range(2)]
    streams = [torch.infini.Stream(device=f"infini:{index}") for index in range(2)]

    try:
        torch.infini.set_stream(streams[0])
        torch.infini.set_stream(streams[1])

        assert torch.infini.current_stream(0) == streams[0]
        assert torch.infini.current_stream(1) == streams[1]
    finally:
        torch.infini.set_stream(previous[0])
        torch.infini.set_stream(previous[1])
        torch.infini.set_device(initial_device)


def test_stream_query_and_synchronize():
    stream = torch.infini.Stream()

    assert isinstance(stream.query(), bool)
    stream.synchronize()
    assert stream.query()


def test_stream_query_is_ready_after_synchronizing_wait():
    source = torch.arange(262144, dtype=torch.float32)
    device_source = torch.empty_like(source, device="infini")
    device_source.copy_(source)
    producer = torch.infini.Stream()
    consumer = torch.infini.Stream()

    with torch.infini.stream(producer):
        produced = torch.add(device_source, device_source)
    consumer.wait_stream(producer)
    with torch.infini.stream(consumer):
        result = torch.add(produced, device_source)
    consumer.synchronize()

    assert consumer.query()
    actual = torch.empty_like(source)
    actual.copy_(result)
    torch.testing.assert_close(actual, source * 3)


def test_stream_query_rejects_exposed_native_handle():
    stream = torch.infini.Stream()
    _ = stream.native_handle

    with pytest.raises(RuntimeError, match="InfiniRT does not expose StreamQuery"):
        stream.query()
