from pathlib import Path

import torch

import torch_infini  # noqa: F401

from .reporting import build_report, load_profile


PROFILE_PATH = Path(__file__).with_name("api_profile.json")


def test_required_infini_api_is_compatible_with_profile():
    report = build_report(
        torch.cuda,
        torch.infini,
        load_profile(PROFILE_PATH),
        torch_version=torch.__version__,
    )
    failures = [
        entry
        for entry in report["entries"]
        if entry["classification"] in {"required-incompatible", "target-unclassified"}
    ]

    assert failures == []


def test_every_infini_export_is_a_required_capability():
    profile = load_profile(PROFILE_PATH)
    required = {
        name
        for name, entry in profile["symbols"].items()
        if entry["status"] == "required"
    }

    assert set(torch.infini.__all__) == required
