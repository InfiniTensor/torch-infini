import argparse
import inspect
import json
from collections.abc import Mapping
from pathlib import Path


PROFILE_SCHEMA_VERSION = 1
VALID_STATUSES = {"required", "planned", "excluded"}
VALID_KINDS = {"class", "function"}
VALID_SIGNATURE_POLICIES = {"strict", "best_effort", "none"}

CLASSIFICATIONS = (
    "required-present",
    "required-incompatible",
    "planned",
    "excluded",
    "missing-unclassified",
    "target-unclassified",
)


def validate_profile(profile):
    """Validate and return a profile without mutating it."""
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a JSON object")

    schema_version = profile.get("schema_version")
    if type(schema_version) is not int or schema_version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {PROFILE_SCHEMA_VERSION}, got {schema_version!r}"
        )

    for field in ("reference_module", "target_module"):
        value = profile.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")

    symbols = profile.get("symbols")
    if not isinstance(symbols, Mapping):
        raise ValueError("symbols must be a JSON object")

    for name, entry in symbols.items():
        if not isinstance(name, str) or not name:
            raise ValueError("symbol names must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise ValueError(f"symbol {name!r} must be a JSON object")

        status = entry.get("status")
        if not isinstance(status, str) or status not in VALID_STATUSES:
            raise ValueError(
                f"symbol {name!r} status must be one of {sorted(VALID_STATUSES)!r}"
            )

        kind = entry.get("kind")
        if not isinstance(kind, str) or kind not in VALID_KINDS:
            raise ValueError(
                f"symbol {name!r} kind must be one of {sorted(VALID_KINDS)!r}"
            )

        signature = entry.get("signature")
        if not isinstance(signature, str) or signature not in VALID_SIGNATURE_POLICIES:
            raise ValueError(
                f"symbol {name!r} signature must be one of "
                f"{sorted(VALID_SIGNATURE_POLICIES)!r}"
            )

        contracts = entry.get("contracts", [])
        if not isinstance(contracts, list) or not all(
            isinstance(contract, str) and contract for contract in contracts
        ):
            raise ValueError(
                f"symbol {name!r} contracts must be a list of non-empty strings"
            )
        if status == "required" and not contracts:
            raise ValueError(
                f"symbol {name!r} contracts must contain at least one contract"
            )

        reason = entry.get("reason")
        if status != "required" and (not isinstance(reason, str) or not reason):
            raise ValueError(f"symbol {name!r} reason must be a non-empty string")

        for field in ("reason", "issue"):
            value = entry.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"symbol {name!r} {field} must be a string or null")

    return profile


def load_profile(path):
    """Read a UTF-8 JSON profile and validate its schema."""
    with Path(path).open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    return validate_profile(profile)


def public_names(module):
    """Return sorted non-private names discovered on a reference module."""
    return sorted(name for name in dir(module) if not name.startswith("_"))


def symbol_kind(value):
    """Return class, function, or the concrete Python type name."""
    if inspect.isclass(value):
        return "class"
    if inspect.isroutine(value):
        return "function"
    return type(value).__name__


def _signature_parameters(value):
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return None

    parameters = []
    for parameter in signature.parameters.values():
        has_default = parameter.default is not inspect.Parameter.empty
        required = not has_default and parameter.kind not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "required": required,
                "default": repr(parameter.default) if has_default else None,
            }
        )
    return parameters


def compare_signatures(reference_value, target_value):
    """Return equal, different, or uninspectable signature metadata."""
    reference_signature = _signature_parameters(reference_value)
    target_signature = _signature_parameters(target_value)
    if reference_signature is None or target_signature is None:
        status = "uninspectable"
    elif reference_signature == target_signature:
        status = "equal"
    else:
        status = "different"
    return {
        "status": status,
        "reference": reference_signature,
        "target": target_signature,
    }


def _target_exports(target):
    exports = getattr(target, "__all__", ())
    if isinstance(exports, str):
        raise ValueError("target __all__ must be an iterable of strings")
    try:
        exports = list(exports)
    except TypeError as error:
        raise ValueError("target __all__ must be an iterable of strings") from error
    if not all(isinstance(name, str) for name in exports):
        raise ValueError("target __all__ must contain only strings")
    return set(exports)


def _classify_required(
    *,
    reference_present,
    target_present,
    expected_kind,
    reference_kind,
    actual_kind,
    signature_policy,
    signature_status,
):
    compatible = (
        reference_present
        and target_present
        and reference_kind == expected_kind
        and actual_kind == expected_kind
    )
    if signature_policy == "strict" and signature_status == "different":
        compatible = False
    return "required-present" if compatible else "required-incompatible"


def build_report(reference, target, profile, *, torch_version):
    """Return a deterministic serializable API gap report."""
    validate_profile(profile)
    symbols = profile["symbols"]
    target_exports = _target_exports(target)
    names = sorted(set(public_names(reference)) | set(symbols) | target_exports)
    entries = []

    for name in names:
        reference_present = hasattr(reference, name)
        target_present = hasattr(target, name)
        reference_value = getattr(reference, name) if reference_present else None
        target_value = getattr(target, name) if target_present else None
        reference_kind = symbol_kind(reference_value) if reference_present else None
        actual_kind = symbol_kind(target_value) if target_present else None

        profile_entry = symbols.get(name)
        profile_status = profile_entry["status"] if profile_entry is not None else None
        expected_kind = (
            profile_entry["kind"] if profile_entry is not None else reference_kind
        )
        signature_policy = (
            profile_entry["signature"] if profile_entry is not None else "none"
        )

        signature_status = "not-checked"
        reference_signature = None
        target_signature = None
        if signature_policy != "none":
            if reference_present and target_present:
                signature = compare_signatures(reference_value, target_value)
                signature_status = signature["status"]
                reference_signature = signature["reference"]
                target_signature = signature["target"]
            else:
                signature_status = "unavailable"

        if profile_status == "required":
            classification = _classify_required(
                reference_present=reference_present,
                target_present=target_present,
                expected_kind=expected_kind,
                reference_kind=reference_kind,
                actual_kind=actual_kind,
                signature_policy=signature_policy,
                signature_status=signature_status,
            )
        elif profile_status in {"planned", "excluded"}:
            classification = profile_status
        elif name in target_exports:
            classification = "target-unclassified"
        else:
            classification = "missing-unclassified"

        entries.append(
            {
                "name": name,
                "profile_status": profile_status,
                "reference_present": reference_present,
                "target_present": target_present,
                "expected_kind": expected_kind,
                "reference_kind": reference_kind,
                "actual_kind": actual_kind,
                "signature_policy": signature_policy,
                "signature_status": signature_status,
                "reference_signature": reference_signature,
                "target_signature": target_signature,
                "classification": classification,
                "contracts": (
                    list(profile_entry.get("contracts", []))
                    if profile_entry is not None
                    else []
                ),
                "reason": (
                    profile_entry.get("reason") if profile_entry is not None else None
                ),
                "issue": (
                    profile_entry.get("issue") if profile_entry is not None else None
                ),
            }
        )

    summary = {"total": len(entries)}
    summary.update(
        {
            classification: sum(
                entry["classification"] == classification for entry in entries
            )
            for classification in CLASSIFICATIONS
        }
    )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "torch_version": torch_version,
        "reference_module": profile["reference_module"],
        "target_module": profile["target_module"],
        "summary": summary,
        "entries": entries,
    }


def _markdown_cell(value):
    if value is None or value == []:
        return "-"
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return str(value).replace("\r", "").replace("\n", "<br>").replace("|", "\\|")


def render_markdown(report):
    """Render a deterministic summary and one table row per report entry."""
    lines = [
        "# torch.infini API Gap Report",
        "",
        f"- Schema version: `{report['schema_version']}`",
        f"- Torch version: `{_markdown_cell(report['torch_version'])}`",
        f"- Reference module: `{_markdown_cell(report['reference_module'])}`",
        f"- Target module: `{_markdown_cell(report['target_module'])}`",
        "",
        "## Summary",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
        f"| Total | {report['summary']['total']} |",
    ]
    lines.extend(
        f"| {classification} | {report['summary'][classification]} |"
        for classification in CLASSIFICATIONS
    )
    lines.extend(
        [
            "",
            "## Symbols",
            "",
            "| Name | Classification | Reference | Target | Expected kind | "
            "Reference kind | Actual kind | Signature policy | Signature status | "
            "Contracts | Reason | Issue |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in report["entries"]:
        values = (
            entry["name"],
            entry["classification"],
            "yes" if entry["reference_present"] else "no",
            "yes" if entry["target_present"] else "no",
            entry["expected_kind"],
            entry["reference_kind"],
            entry["actual_kind"],
            entry["signature_policy"],
            entry["signature_status"],
            entry["contracts"],
            entry["reason"],
            entry["issue"],
        )
        lines.append(
            "| " + " | ".join(_markdown_cell(value) for value in values) + " |"
        )
    return "\n".join(lines) + "\n"


def write_report(report, *, markdown_path, json_path):
    """Create parent directories and write UTF-8 Markdown and JSON reports."""
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torch_infini  # noqa: F401

    profile = load_profile(args.profile)
    report = build_report(
        torch.cuda,
        torch.infini,
        profile,
        torch_version=torch.__version__,
    )
    write_report(report, markdown_path=args.markdown, json_path=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
