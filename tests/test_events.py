import threading

import pytest
import torch

import torch_infini  # noqa: F401


pytestmark = pytest.mark.skipif(
    not torch.infini.is_available(), reason="no infini device"
)


def test_event_api_is_exposed():
    event = torch.infini.Event()

    assert isinstance(event, torch.Event)
    assert event.query()


@pytest.mark.parametrize("option", ["blocking", "interprocess"])
def test_event_rejects_unsupported_options(option):
    with pytest.raises(NotImplementedError, match=option):
        torch.infini.Event(**{option: True})


def test_event_methods_validate_stream_type():
    event = torch.infini.Event()

    with pytest.raises(TypeError, match=r"expected a torch\.Stream"):
        event.record(object())
    with pytest.raises(TypeError, match=r"expected a torch\.Stream"):
        event.wait(object())


def test_event_records_and_synchronizes_on_nondefault_stream():
    stream = torch.infini.Stream()
    event = torch.infini.Event()

    event.record(stream)

    assert event.device == stream.device
    assert isinstance(event.query(), bool)
    event.synchronize()
    assert event.query()


def test_event_uses_current_stream_by_default():
    stream = torch.infini.Stream()
    event = torch.infini.Event()

    with torch.infini.stream(stream):
        event.record()
        event.wait()
    stream.synchronize()

    assert event.query()


def test_stream_records_and_waits_for_event():
    producer = torch.infini.Stream()
    consumer = torch.infini.Stream()

    event = producer.record_event()
    consumer.wait_event(event)
    consumer.synchronize()

    assert isinstance(event, torch.infini.Event)
    assert event.query()


def test_stream_waits_for_stream():
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

    actual = torch.empty_like(source)
    actual.copy_(result)
    torch.testing.assert_close(actual, source * 3)


def test_stream_wait_owns_pending_event_lifetime():
    source = torch.arange(262144, dtype=torch.float32)
    device_source = torch.empty_like(source, device="infini")
    device_source.copy_(source)
    producer = torch.infini.Stream()
    consumer = torch.infini.Stream()

    with torch.infini.stream(producer):
        produced = torch.add(device_source, device_source)
    consumer.wait_event(producer.record_event())
    with torch.infini.stream(consumer):
        result = torch.add(produced, device_source)
    consumer.synchronize()

    actual = torch.empty_like(source)
    actual.copy_(result)
    torch.testing.assert_close(actual, source * 3)


def test_stream_event_methods_validate_event_type():
    stream = torch.infini.Stream()

    assert "wait_event" in torch.infini.Stream.__dict__
    with pytest.raises(TypeError, match=r"expected a torch\.Event"):
        stream.record_event(object())
    with pytest.raises(TypeError, match=r"expected a torch\.Event"):
        stream.wait_event(object())


def test_stream_wait_stream_validates_stream_type():
    stream = torch.infini.Stream()

    with pytest.raises(TypeError, match=r"expected a torch\.Stream"):
        stream.wait_stream(object())


def test_stream_wait_stream_rejects_another_backend():
    stream = torch.infini.Stream()
    cpu_stream = torch.Stream(device="cpu")

    with pytest.raises(ValueError, match="expected an infini stream"):
        stream.wait_stream(cpu_stream)


def test_stream_rejects_event_from_another_backend():
    stream = torch.infini.Stream()
    cpu_event = torch.Event(device="cpu")

    with pytest.raises(ValueError, match="expected an infini event"):
        stream.record_event(cpu_event)
    with pytest.raises(ValueError, match="expected an infini event"):
        stream.wait_event(cpu_event)


def test_elapsed_time_requires_timing_events():
    stream = torch.infini.Stream()
    start = stream.record_event()
    end = stream.record_event()
    end.synchronize()

    with pytest.raises(ValueError, match="enable_timing=True"):
        start.elapsed_time(end)


def test_timing_events_measure_elapsed_time():
    stream = torch.infini.Stream()
    start = torch.infini.Event(enable_timing=True)
    end = torch.infini.Event(enable_timing=True)

    start.record(stream)
    end.record(stream)
    end.synchronize()

    assert start.elapsed_time(end) >= 0.0


def test_event_rejects_rerecording_on_another_device():
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")

    first = torch.infini.Stream(device="infini:0")
    second = torch.infini.Stream(device="infini:1")
    event = torch.infini.Event()
    event.record(first)

    with pytest.raises(RuntimeError, match="already belongs to infini:0"):
        event.record(second)


def test_stream_waits_for_event_from_another_device():
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")

    producer = torch.infini.Stream(device="infini:0")
    consumer = torch.infini.Stream(device="infini:1")
    event = producer.record_event()

    consumer.wait_event(event)
    consumer.synchronize()

    assert event.query()


def test_event_destruction_preserves_worker_current_device():
    if torch.infini.device_count() < 2:
        pytest.skip("requires at least two infini devices")

    stream = torch.infini.Stream(device="infini:0")
    events = [stream.record_event()]
    observed = []
    errors = []

    def destroy_event():
        try:
            torch.infini.set_device(1)
            events.clear()
            observed.append(torch.infini.current_device())
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    worker = threading.Thread(target=destroy_event)
    worker.start()
    worker.join()

    assert not errors
    assert observed == [1]
    assert not events
