from pathlib import Path

from .reporting import load_profile


PROFILE_PATH = Path(__file__).with_name("api_profile.json")


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
