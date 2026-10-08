"""Small offline state checks for the sequential Spark pilot runner."""

import argparse
import json
from pathlib import Path

from cue_annotation.config import load_config
from cue_annotation.persistence import event_records, inspect_run, journal_events, key
from cue_annotation.pilot_scope import pilot_scope
from cue_annotation.qwen_backend import model_identity, resolve_cached_model
from cue_annotation.util import atomic_json, digest


def check_export(run, output):
    manifest = json.loads((run / "manifest.json").read_text())
    exported = json.loads((output / "manifest.json").read_text())
    status = inspect_run(run)
    validation = json.loads((output / "validation.json").read_text())
    if not status["complete"] or exported != manifest or validation != status:
        raise ValueError("Existing export does not match the completed run")
    if not all((output / name).is_file() for name in ("frames.jsonl", "track_windows.jsonl", "review.html")):
        raise ValueError("Existing pilot export is missing files")
    records = {
        key(record): digest(record)
        for event in journal_events(run / "journal.jsonl")
        for record in event_records(event)
    }
    with (output / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if records.pop(key(record), None) != digest(record):
                raise ValueError("Exported frames differ from the committed journal")
    if records:
        raise ValueError("Exported frames are incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    model.add_argument("--config", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("check-export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    scope = commands.add_parser("scope")
    scope.add_argument("--report", type=Path, required=True)
    scope.add_argument("--output", type=Path, required=True)
    finish = commands.add_parser("finish")
    finish.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scope":
        scope = pilot_scope(args.report, args.output)
        for item in scope["exclusions"]:
            print(f"{item['session']}:{item['window']}")
    elif args.command == "model":
        config = load_config(args.config)
        if args.output.exists():
            saved = json.loads(args.output.read_text())
            if saved["config_fingerprint"] != config.fingerprint:
                raise ValueError("Model configuration changed; use a new pilot suite directory")
            path = Path(saved["model"]["local_path"])
            identity = model_identity(path, config)
            if identity != saved["model"]:
                raise ValueError("Pinned pilot model changed; use a new pilot suite directory")
        else:
            path = resolve_cached_model(config)
            identity = model_identity(path, config)
            atomic_json(args.output, {"config_fingerprint": config.fingerprint, "model": identity})
        print(path)
    elif args.command == "check-export":
        check_export(args.run_dir, args.output)
    else:
        plan = json.loads((args.root / "selection.json").read_text())
        summaries = {}
        for stage in ("one", "five", "twenty"):
            run = args.root / f"pilot-{stage}"
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["units"] != plan["stages"][stage]["units"]:
                raise ValueError("Completed pilot scope differs from the frozen selection")
            check_export(run, args.root / f"review-{stage}")
            summaries[stage] = inspect_run(run)
        environment = json.loads((args.root / "validated-environment/validated_environment.json").read_text())
        manifest = json.loads((args.root / "pilot-twenty/manifest.json").read_text())
        if (
            environment["pilot_run_fingerprint"] != summaries["twenty"]["run_fingerprint"]
            or environment["model"] != manifest["model"]
            or not (args.root / "validated-environment/requirements.spark.lock.txt").is_file()
        ):
            raise ValueError("Environment capture is incomplete or belongs to a different pilot")
        atomic_json(
            args.root / "suite-summary.json",
            {
                "engineering_complete": True,
                "quality_reviewed": False,
                "plan_fingerprint": plan["plan_fingerprint"],
                "pilot_scope": json.loads((args.root / "pilot-scope.json").read_text()),
                "stages": summaries,
            },
        )
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
