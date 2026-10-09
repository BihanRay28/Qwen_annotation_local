"""Validate full-session scope; never silently present unfinished preprocessing as labelled."""

import argparse
import json
from pathlib import Path

from cue_annotation.config import load_config
from cue_annotation.dataset import DatasetIndex
from cue_annotation.labels import full_scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--index", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--dataset", type=Path, default=Path("/dataset"))
    p.add_argument("--require-ready", action="store_true")
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    excluded = full_scope(report)
    if args.require_ready:
        if report["issues"] or not report["historical_full_inventory"]:
            raise ValueError(
                "Available indexed tracks were attempted, but source coverage is incomplete. "
                "See labels.json metadata.source_issues and metadata.missing_windows; "
                "finish original preprocessing before claiming all sessions are fully labelled."
            )
    elif args.index:
        index = DatasetIndex(args.index, args.dataset)
        try:
            selection = index.manifest["selection"]
            if selection["sessions"] or selection["windows"]:
                raise ValueError("Subset index cannot be used for full annotation")
            if {tuple(pair) for pair in selection.get("excluded_windows", [])} != set(excluded):
                raise ValueError(
                    "Index exclusions differ from current source readiness; reset and rebuild index"
                )
            index.verify_metadata(load_config(args.config))
        finally:
            index.close()
    else:
        for session, window in excluded:
            print(f"{session}:{window}")


if __name__ == "__main__":
    main()
