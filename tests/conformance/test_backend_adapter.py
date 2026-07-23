import pytest
import torch

from .backends import AcceleratorBackend, compare_observations


def test_backend_normalizes_accelerator_device_identity():
    backend = AcceleratorBackend("test", "cuda", torch.cuda)

    assert backend.normalize_device(torch.device("cuda:2")) == (
        "accelerator",
        2,
    )
    assert backend.normalize_device(torch.device("cuda")) == (
        "accelerator",
        None,
    )


def test_backend_rejects_another_device_type():
    backend = AcceleratorBackend("test", "cuda", torch.cuda)

    with pytest.raises(ValueError, match="expected a cuda device"):
        backend.normalize_device(torch.device("cpu"))


def test_observation_comparison_returns_only_differing_fields():
    observations = {
        "cuda": {"available": True, "name_is_string": True},
        "infini": {"available": True, "name_is_string": False},
    }

    assert compare_observations(observations) == {
        "name_is_string": {"cuda": True, "infini": False}
    }


def test_matching_observations_have_no_differences():
    observations = {
        "cuda": {"available": True},
        "infini": {"available": True},
    }

    assert compare_observations(observations) == {}


def test_observation_comparison_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one backend"):
        compare_observations({})


def test_observation_comparison_rejects_different_fields():
    observations = {
        "cuda": {"available": True},
        "infini": {"available": True, "count_is_positive": True},
    }

    with pytest.raises(ValueError, match="same fields"):
        compare_observations(observations)
