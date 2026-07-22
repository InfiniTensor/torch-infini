import torch


_WORK_ITEMS = 262144
_PENDING_WORK_ITEMS = 16777216


def make_add_input(backend, *, work_items=_WORK_ITEMS):
    source = torch.arange(work_items, dtype=torch.float32)
    device_source = torch.empty_like(source, device=backend.device_type)
    device_source.copy_(source)
    return source, device_source


def make_pending_add_input(backend):
    return make_add_input(backend, work_items=_PENDING_WORK_ITEMS)


def assert_matches_cpu(result, expected):
    actual = torch.empty_like(expected)
    actual.copy_(result)
    torch.testing.assert_close(actual, expected)
