import torch

from .backends import compare_observations
from .workloads import assert_matches_cpu, make_pending_add_input


CONTRACT_ID = "event"
COVERED_API = frozenset({"Event"})


def _observe_events(backend):
    module = backend.module
    stream = module.Stream()
    initial = module.Event()
    event = module.Event()
    source, device_source = make_pending_add_input(backend)
    with module.stream(stream):
        result = torch.add(device_source, device_source)
    event.record(stream)
    event.synchronize()
    completed_after_synchronize = event.query()
    assert_matches_cpu(result, source * 2)

    start = module.Event(enable_timing=True)
    end = module.Event(enable_timing=True)
    start.record(stream)
    end.record(stream)
    end.synchronize()
    elapsed = start.elapsed_time(end)

    return {
        "completed_after_synchronize": completed_after_synchronize,
        "initial_event_is_complete": initial.query(),
        "initial_query_returns_bool": isinstance(initial.query(), bool),
        "recorded_device_matches_stream": (
            backend.normalize_device(event.device)
            == backend.normalize_device(stream.device)
        ),
        "timing_is_nonnegative_float": (isinstance(elapsed, float) and elapsed >= 0.0),
    }


def test_event_module_contract(accelerator_backend):
    assert _observe_events(accelerator_backend) == {
        "completed_after_synchronize": True,
        "initial_event_is_complete": True,
        "initial_query_returns_bool": True,
        "recorded_device_matches_stream": True,
        "timing_is_nonnegative_float": True,
    }


def test_available_event_observations_match(available_accelerator_backends):
    observations = {
        backend.name: _observe_events(backend)
        for backend in available_accelerator_backends
    }

    assert compare_observations(observations) == {}


def test_event_uses_current_stream_by_default(accelerator_backend):
    module = accelerator_backend.module
    producer = module.Stream()
    consumer = module.Stream()
    event = module.Event()
    source, device_source = make_pending_add_input(accelerator_backend)

    with module.stream(producer):
        produced = torch.add(device_source, device_source)
        event.record()
    with module.stream(consumer):
        event.wait()
    consumer.synchronize()

    assert event.query()

    with module.stream(consumer):
        result = torch.add(produced, device_source)
    consumer.synchronize()

    assert_matches_cpu(result, source * 3)


def test_stream_records_and_waits_for_event(accelerator_backend):
    module = accelerator_backend.module
    producer = module.Stream()
    consumer = module.Stream()
    source, device_source = make_pending_add_input(accelerator_backend)

    with module.stream(producer):
        produced = torch.add(device_source, device_source)
    event = producer.record_event()
    consumer.wait_event(event)
    consumer.synchronize()

    assert event.query()

    with module.stream(consumer):
        result = torch.add(produced, device_source)
    consumer.synchronize()

    assert_matches_cpu(result, source * 3)


def test_stream_waits_for_stream(accelerator_backend):
    module = accelerator_backend.module
    producer = module.Stream()
    consumer = module.Stream()
    source, device_source = make_pending_add_input(accelerator_backend)

    with module.stream(producer):
        produced = torch.add(device_source, device_source)
    completion = producer.record_event()
    consumer.wait_stream(producer)
    consumer.synchronize()

    assert completion.query()
    assert consumer.query()

    with module.stream(consumer):
        result = torch.add(produced, device_source)
    consumer.synchronize()

    assert_matches_cpu(result, source * 3)
