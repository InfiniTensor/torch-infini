from pathlib import Path

from . import test_device_contracts, test_event_contracts, test_stream_contracts
from .reporting import load_profile


PROFILE_PATH = Path(__file__).with_name("api_profile.json")
CONTRACT_MODULES = (
    test_device_contracts,
    test_event_contracts,
    test_stream_contracts,
)


def test_api_profile_declares_current_exports_and_advisory_gaps():
    profile = load_profile(PROFILE_PATH)
    required = {
        name
        for name, entry in profile["symbols"].items()
        if entry["status"] == "required"
    }

    assert required == {
        "Event",
        "Stream",
        "current_device",
        "current_stream",
        "default_stream",
        "device",
        "device_count",
        "get_device_name",
        "is_available",
        "set_device",
        "set_stream",
        "stream",
        "synchronize",
    }
    assert profile["symbols"]["init"]["status"] == "excluded"
    assert profile["symbols"]["manual_seed"]["status"] == "planned"


def test_advisory_reasons_are_complete_sentences():
    profile = load_profile(PROFILE_PATH)
    reasons = [
        entry["reason"]
        for entry in profile["symbols"].values()
        if entry["status"] != "required"
    ]

    assert reasons
    assert all(reason[0].isupper() and reason.endswith(".") for reason in reasons)


def test_required_capabilities_are_linked_to_executable_contracts():
    contract_ids = [module.CONTRACT_ID for module in CONTRACT_MODULES]
    assert len(contract_ids) == len(set(contract_ids))

    executable_coverage = {}
    for module in CONTRACT_MODULES:
        for name in module.COVERED_API:
            executable_coverage.setdefault(name, set()).add(module.CONTRACT_ID)

    profile = load_profile(PROFILE_PATH)
    declared_coverage = {
        name: set(entry["contracts"])
        for name, entry in profile["symbols"].items()
        if entry["status"] == "required"
    }

    assert declared_coverage == executable_coverage
