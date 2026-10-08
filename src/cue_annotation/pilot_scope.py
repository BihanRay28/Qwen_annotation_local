"""Record unfinished-window exclusions without repairing or altering source crops."""

import json
from pathlib import Path

from .util import atomic_json, digest


def pilot_scope(report_path, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    issues = report.get("issues", [])
    if any(p.startswith("out_of_range") for item in issues for p in item["issues"]):
        raise ValueError(
            "Out-of-range source frames need correction; automatic pilot exclusions do not hide them"
        )
    unfinished = sorted(
        {(item["session"], item["window"]) for item in issues if "crop_root_not_finalised" in item["issues"]}
    )
    output = Path(output)
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        fingerprint = saved.pop("scope_fingerprint", None)
        if fingerprint != digest(saved):
            raise ValueError("Saved pilot exclusion scope was modified")
        saved["scope_fingerprint"] = fingerprint
        excluded = {(item["session"], item["window"]) for item in saved["exclusions"]}
        if set(unfinished) - excluded:
            raise ValueError("Additional windows became unfinished; diagnose them or use a new pilot suite")
        return saved
    scope = {
        "version": 1,
        "source_window_count": report["windows"],
        "pilot_index_window_count": report["windows"] - len(unfinished),
        "exclusions": [
            {"session": s, "window": w, "reason": "crop_root_not_finalised"} for s, w in unfinished
        ],
        "note": "Excluded only from this engineering pilot index. Source dataset is unchanged; repair unfinished preprocessing before including these windows in bulk annotation.",
    }
    scope["scope_fingerprint"] = digest(scope)
    atomic_json(output, scope)
    return scope
