from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_REPORT_PATH = Path(__file__).parents[2] / "tools" / "report_api_gaps.py"
_REPORT_SPEC = spec_from_file_location("torch_infini_report_api_gaps", _REPORT_PATH)
if _REPORT_SPEC is None or _REPORT_SPEC.loader is None:
    raise ImportError(f"could not load API report helpers from {_REPORT_PATH}")

_REPORT_MODULE = module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(_REPORT_MODULE)

build_report = _REPORT_MODULE.build_report
compare_signatures = _REPORT_MODULE.compare_signatures
load_profile = _REPORT_MODULE.load_profile
public_names = _REPORT_MODULE.public_names
render_markdown = _REPORT_MODULE.render_markdown
symbol_kind = _REPORT_MODULE.symbol_kind
validate_profile = _REPORT_MODULE.validate_profile
write_report = _REPORT_MODULE.write_report
