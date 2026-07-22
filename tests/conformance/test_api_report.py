import json
import sys
from types import ModuleType

import pytest

from .reporting import (
    build_report,
    compare_signatures,
    load_profile,
    public_names,
    render_markdown,
    symbol_kind,
    validate_profile,
    write_report,
)


def _function():
    return None


def _other_function(value=None):
    return value


class _Class:
    pass


def _module(name, **members):
    module = ModuleType(name)
    for member_name, value in members.items():
        setattr(module, member_name, value)
    return module


def _entry(status, kind, signature="none", **extra):
    entry = {
        "status": status,
        "kind": kind,
        "signature": signature,
    }
    if status == "required":
        entry["contracts"] = extra.pop("contracts", ["device"])
    else:
        entry["reason"] = extra.pop("reason", "Not part of the current scope.")
    entry.update(extra)
    return entry


def _profile(symbols):
    return {
        "schema_version": 1,
        "reference_module": "reference",
        "target_module": "target",
        "symbols": symbols,
    }


def test_report_classifies_supported_and_advisory_symbols():
    reference = _module(
        "reference",
        required=_function,
        incompatible=_Class,
        planned=_function,
        excluded=_function,
        unknown=_function,
    )
    target = _module(
        "target",
        required=_function,
        incompatible=_function,
        extra=_function,
    )
    target.__all__ = ["required", "incompatible", "extra"]
    profile = _profile(
        {
            "required": _entry("required", "function", "strict"),
            "incompatible": _entry("required", "class"),
            "planned": _entry("planned", "function"),
            "excluded": _entry("excluded", "function"),
        }
    )

    report = build_report(reference, target, profile, torch_version="test")
    classifications = {
        entry["name"]: entry["classification"] for entry in report["entries"]
    }

    assert classifications == {
        "excluded": "excluded",
        "extra": "target-unclassified",
        "incompatible": "required-incompatible",
        "planned": "planned",
        "required": "required-present",
        "unknown": "missing-unclassified",
    }
    assert report["summary"] == {
        "excluded": 1,
        "missing-unclassified": 1,
        "planned": 1,
        "required-incompatible": 1,
        "required-present": 1,
        "target-unclassified": 1,
        "total": 6,
    }


def test_strict_signature_mismatch_is_incompatible():
    reference = _module("reference", operation=_function)
    target = _module("target", operation=_other_function)
    target.__all__ = ["operation"]
    profile = _profile({"operation": _entry("required", "function", "strict")})

    report = build_report(reference, target, profile, torch_version="test")

    assert report["entries"][0]["classification"] == "required-incompatible"
    assert report["entries"][0]["signature_status"] == "different"


def test_best_effort_signature_mismatch_remains_present():
    reference = _module("reference", operation=_function)
    target = _module("target", operation=_other_function)
    target.__all__ = ["operation"]
    profile = _profile({"operation": _entry("required", "function", "best_effort")})

    report = build_report(reference, target, profile, torch_version="test")

    assert report["entries"][0]["classification"] == "required-present"
    assert report["entries"][0]["signature_status"] == "different"


def test_signature_comparison_ignores_annotations():
    def reference(value: int = 1) -> int:
        return value

    def target(value: str = 1) -> str:
        return str(value)

    assert compare_signatures(reference, target)["status"] == "equal"
    assert compare_signatures(sys.getsizeof, sys.getsizeof)["status"] == "uninspectable"


def test_public_names_and_symbol_kinds_are_stable():
    module = _module("reference", function=_function, klass=_Class, value=1)
    module._private = _function

    assert public_names(module) == ["function", "klass", "value"]
    assert symbol_kind(module.function) == "function"
    assert symbol_kind(module.klass) == "class"
    assert symbol_kind(module.value) == "int"


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        ({}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        (
            {
                "schema_version": 1,
                "reference_module": "reference",
                "target_module": "target",
                "symbols": {"name": {"status": "unknown"}},
            },
            "status",
        ),
        (
            _profile({"name": _entry("required", "module")}),
            "kind",
        ),
        (
            _profile({"name": _entry("required", "function", "exact")}),
            "signature",
        ),
        (
            _profile(
                {
                    "name": {
                        "status": "required",
                        "kind": "function",
                        "signature": "strict",
                        "contracts": [],
                    }
                }
            ),
            "contracts",
        ),
        (
            _profile(
                {
                    "name": {
                        "status": "planned",
                        "kind": "function",
                        "signature": "none",
                    }
                }
            ),
            "reason",
        ),
    ],
)
def test_validate_profile_rejects_malformed_data(profile, match):
    with pytest.raises(ValueError, match=match):
        validate_profile(profile)


def test_load_profile_reads_and_validates_json(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile({"operation": _entry("required", "function")})
    path.write_text(json.dumps(profile), encoding="utf-8")

    assert load_profile(path) == profile


def test_report_writes_deterministic_markdown_and_json(tmp_path):
    reference = _module("reference", required=_function)
    target = _module("target", required=_function)
    target.__all__ = ["required"]
    profile = _profile({"required": _entry("required", "function", "strict")})
    report = build_report(reference, target, profile, torch_version="test")
    markdown_path = tmp_path / "nested" / "report.md"
    json_path = tmp_path / "nested" / "report.json"

    write_report(
        report,
        markdown_path=markdown_path,
        json_path=json_path,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(report)
    assert "# torch.infini API Gap Report" in markdown
    assert "| required | required-present |" in markdown
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
